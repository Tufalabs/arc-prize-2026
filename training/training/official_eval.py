"""Continuous evaluation on a configured held-out ARC-AGI-3 game set.

Runs as a separate process next to the trainer (spawned by training.cli when
``official_eval.enabled`` is set). It owns one or more dedicated GPUs, each
with its own vLLM server, and loops forever:

1. Read ``training_state.json`` for the latest published runtime adapter and
   its iteration, or ``initial_policy.json`` before the first update.
2. Hot-load that adapter and run every configured eval game in parallel, each
   capped at ``official_eval.cycle_minutes`` of runtime.
3. Write the standard eval/* scalars to TensorBoard at that trainer iteration
   then start the next cycle.

Eval uses the same checked gateway and token continuation as training.
Context exhaustion triggers Duck's configured journal/eviction compaction;
it does not end the episode. Evaluation can override its context, compaction
target and decoding settings while keeping the same serving checks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from training.cli import (
    LOG_FORMAT,
    _INITIAL_POLICY_FILENAME,
    _configure_hf_cache_defaults,
    _write_per_game_metric_group,
)
from training.server_endpoints import collector_server_log_paths
from training.config import ExperimentConfig, evaluation_inference_config
from training.contracts import RolloutTrajectory
from training.inference import (
    RolloutCollector,
    derive_server_base_url,
    split_slots_into_balanced_chunks,
)
from harness_support.arc3.scoring import arc_game_score_for_completed_levels
from training.official_eval_policy import record_full_weight_eval_result
from training.metrics import collection_scalars, summarize_rollouts
from training.run_artifacts import (
    resolve_rollout_windows_root,
    stage_out_rollout_scratch,
    setup_logging_for_experiment,
)
from training.run_failure import record_run_failure, raise_if_run_failed


log = logging.getLogger(__name__)

_TRAINING_RUN_STATE_FILENAME = "training_state.json"
_ADAPTER_WAIT_SECONDS = 900.0
_ADAPTER_POLL_SECONDS = 15.0
_OFFICIAL_EVAL_METRICS_FILENAME = "official_eval_metrics.jsonl"


def _abort_eval_collectors(collectors: list[RolloutCollector]) -> None:
    """Stop all rollout workers while preserving the first evaluation error."""
    for collector in collectors:
        try:
            collector.abort_window()
        except Exception:
            log.exception("failed to abort official eval sibling after fatal error")


def _resolve_eval_futures(
    futures: dict[Future[Any], int],
    *,
    collectors: list[RolloutCollector],
    run_root: Path,
) -> dict[int, Any]:
    """Surface any server's failure before waiting for its siblings to finish."""
    results: dict[int, Any] = {}
    try:
        for future in as_completed(futures):
            raise_if_run_failed(run_root)
            results[futures[future]] = future.result()
    except Exception as error:
        record_run_failure(run_root, error)
        for future in futures:
            future.cancel()
        _abort_eval_collectors(collectors)
        raise
    return results


def _hand_episodes_to_learner(
    config: ExperimentConfig,
    run_root: Path,
    *,
    cycle_index: int,
    iteration: int,
    rollouts: list[RolloutTrajectory],
) -> None:
    """Leave a few of this cycle's episodes for the Flash learner's eval/
    expert load pass (training/expert_load.py). A diagnostic: a failure here
    is logged and never stops evaluation."""
    from training.expert_load import eval_handoff_episodes, write_eval_handoff

    episodes = eval_handoff_episodes(config)
    if episodes <= 0:
        return
    try:
        path = write_eval_handoff(
            run_root,
            cycle_index=cycle_index,
            policy_iteration=iteration,
            rollouts=rollouts,
            episodes=episodes,
            max_sequence_tokens=config.training.max_sequence_tokens,
        )
    except Exception:  # noqa: BLE001 - diagnostics never stop evaluation
        log.exception("expert load: could not hand eval cycle %d to the learner", cycle_index)
        return
    if path is not None:
        log.info("expert load: eval cycle %d episodes handed to the learner at %s", cycle_index, path)


def _collect_eval_chunk(
    *,
    collector: RolloutCollector,
    chunk: list[tuple[int, str, int]],
    output_root: Path,
    model_id: str,
    policy_iteration: int,
    cycle_index: int,
) -> Any | None:
    """Collect one server's explicitly bound share of an evaluation cycle."""
    if not chunk:
        return None
    return collector.collect_window(
        output_root=output_root,
        model_id=model_id,
        policy_iteration=policy_iteration,
        schedule_index=cycle_index,
        scheduled_games=[game_id for _, game_id, _ in chunk],
        scheduled_rollout_seeds=[seed for _, _, seed in chunk],
        window_kind="eval",
    )


def _merge_eval_child_windows(
    *,
    chunks: list[list[tuple[int, str, int]]],
    child_windows: dict[int, Any],
) -> list[RolloutTrajectory]:
    """Merge server results without silently losing scheduled evaluations."""
    merged: list[tuple[int, RolloutTrajectory]] = []
    for server_index, chunk in enumerate(chunks):
        if not chunk:
            continue
        child_window = child_windows.get(server_index)
        if child_window is None:
            raise RuntimeError(
                f"official eval server {server_index} returned no window for "
                f"{len(chunk)} scheduled slots"
            )
        if len(child_window.rollouts) != len(chunk):
            raise RuntimeError(
                f"official eval server {server_index} returned "
                f"{len(child_window.rollouts)} rollouts for {len(chunk)} scheduled slots"
            )
        merged.extend(
            (slot_index, rollout)
            for (slot_index, _, _), rollout in zip(
                chunk, child_window.rollouts, strict=True
            )
        )
    merged.sort(key=lambda item: item[0])
    return [rollout for _, rollout in merged]


def _rollout_seed(
    *, training_seed: int, game_id: str, copy_index: int = 0, cycle_index: int = 0
) -> int:
    """Deterministic seed per (training seed, game, copy, cycle): copies of a
    game in one cycle and the same game across cycles draw different decode
    seeds, and a resumed run reproduces the same ones."""
    payload = (
        f"arc3-official-eval:{training_seed}:{game_id}:copy{copy_index}:cycle{cycle_index}"
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**31 - 1)


def _next_cycle_index(run_root: Path, *, rollouts_root: Path | None = None) -> int:
    """First unused eval cycle index, resuming across relaunches.

    Every relaunch previously restarted the counter at 0, which re-keyed the
    TensorBoard eval scalars onto already-used cycle steps (clobbering the
    curve) and re-used the per-cycle rollout artifact directories. The
    persisted metrics ledger is the source of truth; the artifact dirs are
    scanned as well in case a cycle crashed after writing artifacts but
    before appending its ledger row.
    """
    highest = -1
    ledger = run_root / _OFFICIAL_EVAL_METRICS_FILENAME
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                highest = max(highest, int(json.loads(line).get("cycle_index", -1)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    durable_root = run_root / "official_eval_rollouts"
    artifact_roots = [durable_root]
    if rollouts_root is not None and rollouts_root != durable_root:
        artifact_roots.append(rollouts_root)
    for artifacts_root in artifact_roots:
        if not artifacts_root.is_dir():
            continue
        for path in artifacts_root.glob("cycle_*"):
            try:
                highest = max(highest, int(path.name.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    return highest + 1


def _latest_published_policy(run_root: Path) -> tuple[int, Path | None]:
    """Latest published policy, including the initialized LoRA before update 0.

    ``training_state.json`` is written atomically after every iteration and its
    ``last_completed_iteration`` equals the published policy iteration. The
    trainer also republishes this state before launching resumed eval if
    it must regenerate an export from a previous job's node-local scratch.
    Only the explicitly published path is eligible for evaluation.
    """
    state_path = run_root / _TRAINING_RUN_STATE_FILENAME
    if not state_path.exists():
        state_path = run_root / _INITIAL_POLICY_FILENAME
        if not state_path.exists():
            raise RuntimeError("initial LoRA policy has not been published")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    iteration = int(payload.get("last_completed_iteration", -1))
    raw_adapter = str(payload.get("runtime_adapter_dir") or "").strip()
    if not raw_adapter:
        return iteration, None
    adapter_dir = Path(raw_adapter)
    if not adapter_dir.is_absolute():
        adapter_dir = run_root / adapter_dir
    return iteration, adapter_dir


def _last_evaluated_iteration(run_root: Path) -> int | None:
    """Restore the last completed policy from the eval ledger across restarts."""
    ledger = run_root / _OFFICIAL_EVAL_METRICS_FILENAME
    if not ledger.exists():
        return None
    latest = None
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            iteration = int(json.loads(line)["iteration"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        latest = iteration if latest is None else max(latest, iteration)
    return latest


def _await_published_policy(run_root: Path) -> tuple[int, Path]:
    """Wait for a published export without substituting an unrelated base model."""
    raise_if_run_failed(run_root)
    iteration, adapter_dir = _latest_published_policy(run_root)
    waited = 0.0
    while adapter_dir is not None and not adapter_dir.exists():
        if waited >= _ADAPTER_WAIT_SECONDS:
            raise RuntimeError(
                f"runtime adapter {adapter_dir} for published iteration {iteration} "
                f"is still missing after {waited:.0f}s"
            )
        time.sleep(_ADAPTER_POLL_SECONDS)
        waited += _ADAPTER_POLL_SECONDS
        raise_if_run_failed(run_root)
        iteration, adapter_dir = _latest_published_policy(run_root)
        if adapter_dir is None:
            raise RuntimeError(
                "published runtime adapter disappeared while awaiting its export"
            )
    if adapter_dir is None:
        raise RuntimeError(
            f"published iteration {iteration} has no runtime adapter path"
        )
    return iteration, adapter_dir


def _eval_server_base_urls(config: ExperimentConfig) -> list[str]:
    """One pre-launched vLLM URL per official_eval.server_gpus entry.

    Explicit ``official_eval.base_urls`` wins; otherwise the eval servers
    extend the training serving contract (training.inference
    .derive_server_base_url): the training servers hold indices 0..T-1, so
    eval server j holds index T+j. Either way the launcher can compute the
    same endpoints on its own — there is no free-port discovery anywhere,
    because a discovered port is one no launcher owns.
    """
    eval_gpus = [str(gpu) for gpu in config.official_eval.server_gpus]
    explicit = [
        str(url).strip() for url in config.official_eval.base_urls if str(url).strip()
    ]
    if explicit:
        if len(explicit) != len(eval_gpus):
            raise ValueError(
                f"official_eval.base_urls lists {len(explicit)} URLs for "
                f"{len(eval_gpus)} official_eval.server_gpus entries; give "
                "exactly one pre-launched server URL per eval GPU."
            )
        return explicit
    training_server_count = max(1, len(config.inference.server_gpus))
    return [
        derive_server_base_url(config.inference.base_url, training_server_count + index)
        for index in range(len(eval_gpus))
    ]


def _collector_config(config: ExperimentConfig, *, base_url: str) -> ExperimentConfig:
    """Derive one eval collector's config (one collector per eval server GPU).

    Slot capacity is sized for this server's share of
    ``len(games) * rollouts_per_game`` scheduled rollouts.
    """
    games = list(config.official_eval.games)
    slots = len(games) * max(1, int(config.official_eval.rollouts_per_game))
    server_count = max(1, len(config.official_eval.server_gpus))
    per_server_slots = -(-slots // server_count)  # ceil division
    eval_config = copy.deepcopy(config)
    eval_config.inference = evaluation_inference_config(eval_config)
    eval_config.parallel_rollouts = per_server_slots
    eval_config.rollout_buffer_size = 1
    eval_config.inference.games = games
    if config.official_eval.dataset is not None:
        eval_config.inference.dataset = config.official_eval.dataset
    eval_config.inference.base_url = base_url
    eval_config.inference.server_gpus = []
    eval_config.inference.server_log_paths = []
    eval_config.inference.max_runtime_minutes = float(
        config.official_eval.cycle_minutes
    )
    # The train-side runtime ramp (max_runtime_minutes_schedule) never applies
    # to eval: _effective_max_runtime_minutes returns the base value for
    # window_kind="eval", which is the single eval discriminator here.
    # RolloutCollector resolves the shared ContextContract after these overrides,
    # at collection time just as for training.
    # Deployment-style decoding for eval; training keeps its own sampling.
    eval_config.inference.agent_temperature = float(config.official_eval.temperature)
    eval_config.inference.agent_top_p = float(config.official_eval.top_p)
    eval_config.inference.agent_top_k = int(config.official_eval.top_k)
    eval_config.training.device = "cpu"
    return eval_config


def _stats_at_minutes(
    rollouts: list[RolloutTrajectory], *, minutes: float
) -> dict[str, float]:
    """Cycle stats recomputed as if every rollout had stopped at `minutes`.

    Uses per-level completion times recorded by the artifact loader
    (level_completion_elapsed_seconds); levels completed after the mark are
    dropped and the official game score is recomputed from the rest.
    """
    cutoff_seconds = float(minutes) * 60.0
    arc_scores: list[float] = []
    levels_completed: list[float] = []
    for rollout in rollouts:
        metadata = rollout.metadata if isinstance(rollout.metadata, dict) else {}
        total_levels = float(metadata.get("game_total_levels") or 0.0)
        if total_levels <= 0.0:
            continue
        level_scores = metadata.get("level_scores") or {}
        elapsed_by_level = metadata.get("level_completion_elapsed_seconds") or {}
        kept: dict[int, float] = {}
        for raw_level, score in level_scores.items():
            level = int(raw_level)
            elapsed = elapsed_by_level.get(raw_level, elapsed_by_level.get(level))
            if elapsed is None and isinstance(raw_level, int):
                elapsed = elapsed_by_level.get(str(raw_level))
            if elapsed is not None and float(elapsed) <= cutoff_seconds:
                kept[level] = float(score)
        arc_scores.append(
            arc_game_score_for_completed_levels(kept, total_levels=total_levels)
        )
        levels_completed.append(float(len(kept)))
    if not arc_scores:
        return {}
    return {
        "avg_arc_score": sum(arc_scores) / len(arc_scores),
        "avg_levels_completed": sum(levels_completed) / len(levels_completed),
        "rollouts": float(len(arc_scores)),
    }


def run_official_eval(*, config_path: str, run_root: Path) -> None:
    """Run evaluation and publish startup failures to the supervising trainer."""
    try:
        raise_if_run_failed(run_root)
        _run_official_eval(config_path=config_path, run_root=run_root)
    except Exception as error:
        record_run_failure(run_root, error)
        raise


def _run_official_eval(*, config_path: str, run_root: Path) -> None:
    """Evaluate published policies until a run failure interrupts collection."""
    repo_root = Path(__file__).resolve().parents[1]
    _configure_hf_cache_defaults()
    config = ExperimentConfig.load(config_path)
    from training.harness import get_trainer_support

    get_trainer_support(config.inference.environment).resolve_datasets(config)
    if not config.official_eval.enabled:
        raise RuntimeError("official_eval.enabled must be true to run this worker.")
    setup_logging_for_experiment(
        log_file_path=run_root / "official_eval.log", fmt=LOG_FORMAT, append=True
    )
    games = sorted(
        {game.strip() for game in config.official_eval.games if game.strip()}
    )
    rollouts_per_game = max(1, int(config.official_eval.rollouts_per_game))
    # Decision 13: every game runs rollouts_per_game times per cycle, so
    # averaging the copies lowers the per-cycle noise floor. Block layout
    # ([g1..gN, g1..gN, ...]) + whole-pass chunking gives each server
    # complete passes, spreading the copies of one game across different
    # servers whenever there is more than one. Seeds are per (game, copy,
    # cycle): the seed drives the decode seed of every model call, so
    # copies that shared one sampled identically, and a cycle that reused
    # the previous cycle's seeds re-drew the same samples from an unchanged
    # policy. The official games' layouts do not depend on the seed.
    scheduled_games = games * rollouts_per_game
    eval_gpus = [str(gpu) for gpu in config.official_eval.server_gpus]
    rollouts_root = resolve_rollout_windows_root(run_root / "official_eval_rollouts")
    rollouts_root.mkdir(parents=True, exist_ok=True)
    server_log_paths = collector_server_log_paths(config, run_root, evaluation=True)
    # Pre-launched endpoints only (explicit official_eval.base_urls, or the
    # deterministic training port contract) — never freshly discovered free
    # ports, which no launcher could have been given.
    base_urls = _eval_server_base_urls(config)
    # The eval collectors keep their persistent worker (serve dir, logs)
    # under a durable per-run root, never under a cycle directory: the loop
    # below deletes the previous cycle directory after each cycle, and a
    # worker whose serve dir lived there lost it every third cycle
    # (FileNotFoundError on the next spec write, 2026-09-02).
    worker_root = run_root / "official_eval_worker"
    worker_root.mkdir(parents=True, exist_ok=True)
    collectors = [
        RolloutCollector(
            _collector_config(config, base_url=base_urls[index]),
            repo_root=repo_root,
            server_log_path=server_log_paths[index],
            run_root=worker_root,
            failure_root=run_root,
        )
        for index in range(len(eval_gpus))
    ]
    # Keep the existing balanced distribution of full-game slots across servers.
    chunk_slot_indices_per_server = split_slots_into_balanced_chunks(
        scheduled_games, len(collectors)
    )

    def _cycle_chunks(cycle: int) -> list[list[tuple[int, str, int]]]:
        seeds = [
            _rollout_seed(
                training_seed=config.training.seed,
                game_id=game,
                copy_index=slot_index // len(games),
                cycle_index=cycle,
            )
            for slot_index, game in enumerate(scheduled_games)
        ]
        return [
            [
                (slot_index, scheduled_games[slot_index], seeds[slot_index])
                for slot_index in chunk_slot_indices
            ]
            for chunk_slot_indices in chunk_slot_indices_per_server
        ]

    writer = SummaryWriter(log_dir=str(run_root / "tensorboard"))
    metrics_path = run_root / _OFFICIAL_EVAL_METRICS_FILENAME
    log.info(
        "official eval worker starting: %d games x %d rollouts/game, "
        "cycle=%.1f min, gpus=%s, servers=%s",
        len(games),
        rollouts_per_game,
        config.official_eval.cycle_minutes,
        ",".join(eval_gpus),
        ",".join(base_urls),
    )
    try:
        # Connect-only: the eval vLLM servers are pre-launched by the job
        # script; verify each is reachable before the first cycle.
        with ThreadPoolExecutor(
            max_workers=len(collectors), thread_name_prefix="eval-ensure"
        ) as executor:
            ensure_futures = {
                executor.submit(collector.ensure_server): index
                for index, collector in enumerate(collectors)
            }
            _resolve_eval_futures(
                ensure_futures, collectors=collectors, run_root=run_root
            )
        cycle_index = _next_cycle_index(run_root, rollouts_root=rollouts_root)
        last_evaluated_iteration = _last_evaluated_iteration(run_root)
        if cycle_index:
            log.info(
                "resuming official eval at cycle %d (continuing the persisted "
                "cycle counter across relaunches)",
                cycle_index,
            )
        previous_cycle_dir: Path | None = None
        full_provider = None
        if config.training.weight_mode == "full_weight":
            from training.official_eval_policy import FullWeightEvalPolicyProvider

            trainer_pid = int(os.environ.get("ARC3_SUPERVISING_TRAINER_PID", "0"))

            def require_trainer_alive() -> None:
                raise_if_run_failed(run_root)
                if trainer_pid and os.getppid() != trainer_pid:
                    raise RuntimeError("Supervising full-weight trainer exited")

            full_provider = FullWeightEvalPolicyProvider(
                run_root=run_root,
                base_model_id=config.inference.model_id or config.training.model_name,
                server_urls=base_urls,
                request_timeout_seconds=None if trainer_pid else 1800.0,
                liveness_check=require_trainer_alive,
            )
        while True:
            if full_provider is not None:
                policy = full_provider.acquire(cycle_index=cycle_index)
                iteration, model_id, adapter_dir = (
                    policy.iteration,
                    policy.model_id,
                    None,
                )
            else:
                iteration, adapter_dir = _await_published_policy(run_root)
                model_id = "policy"
            # Full-weight startup must acknowledge the trainer's handshake,
            # even if the resumed policy has already been evaluated.
            if (
                last_evaluated_iteration is not None
                and iteration <= last_evaluated_iteration
            ):
                raise_if_run_failed(run_root)
                time.sleep(_ADAPTER_POLL_SECONDS)
                continue
            if full_provider is not None:
                for collector, identity in zip(
                    collectors, policy.identities, strict=True
                ):
                    collector.pin_full_weight_identity(identity)
            else:
                for collector in collectors:
                    raise_if_run_failed(run_root)
                    collector.load_runtime_adapter(
                        adapter_name=model_id,
                        adapter_path=adapter_dir,
                    )
            # Align with train/*; policies skipped while eval runs leave gaps.
            step = int(iteration)
            log.info(
                "dedicated eval cycle %d: iteration=%d model=%s adapter=%s",
                cycle_index,
                iteration,
                model_id,
                adapter_dir,
            )
            cycle_dir = rollouts_root / f"cycle_{cycle_index:05d}"
            chunks = _cycle_chunks(cycle_index)
            with ThreadPoolExecutor(
                max_workers=len(collectors), thread_name_prefix="eval-window"
            ) as executor:
                collect_futures = {
                    executor.submit(
                        _collect_eval_chunk,
                        collector=collectors[server_index],
                        chunk=chunks[server_index],
                        output_root=cycle_dir / f"server_{server_index}",
                        model_id=model_id,
                        policy_iteration=iteration,
                        cycle_index=cycle_index,
                    ): server_index
                    for server_index in range(len(collectors))
                }
                child_windows = {
                    server_index: child_window
                    for server_index, child_window in _resolve_eval_futures(
                        collect_futures, collectors=collectors, run_root=run_root
                    ).items()
                    if child_window is not None
                }
            raise_if_run_failed(run_root)
            # Merge child windows back into original slot order.
            rollouts = _merge_eval_child_windows(
                chunks=chunks,
                child_windows=child_windows,
            )
            collection_seconds = max(
                (child.collection_duration_seconds for child in child_windows.values()),
                default=0.0,
            )
            window_id = f"cycle_{cycle_index:05d}"
            metrics = summarize_rollouts(rollouts)
            for name, value in metrics.as_scalars(scalar_prefix="eval").items():
                writer.add_scalar(name, value, step)
            for name, value in collection_scalars(
                rollouts, collection_seconds, scalar_prefix="eval"
            ).items():
                writer.add_scalar(name, value, step)
            writer.add_scalar(
                "eval/collection_seconds",
                collection_seconds,
                step,
            )
            writer.add_scalar("eval/cycle_index", float(cycle_index), step)
            writer.add_scalar("eval/policy_iteration", float(iteration), step)
            stats_by_mark: dict[str, dict[str, float]] = {}
            for mark_minutes in config.official_eval.stats_minutes:
                mark_stats = _stats_at_minutes(rollouts, minutes=mark_minutes)
                if not mark_stats:
                    continue
                mark_label = f"{mark_minutes:g}min"
                stats_by_mark[mark_label] = mark_stats
                writer.add_scalar(
                    f"eval/{mark_label}_avg_arc_score",
                    mark_stats["avg_arc_score"],
                    step,
                )
                writer.add_scalar(
                    f"eval/{mark_label}_avg_levels_completed",
                    mark_stats["avg_levels_completed"],
                    step,
                )
            per_game = {}
            if config.metrics.log_per_game:
                per_game = _write_per_game_metric_group(
                    writer,
                    prefix="eval_by_game",
                    iteration_index=step,
                    rollouts=rollouts,
                )
            writer.flush()
            record = {
                "cycle_index": cycle_index,
                "iteration": iteration,
                "model_id": model_id,
                "adapter_dir": str(adapter_dir) if adapter_dir else None,
                "window_id": window_id,
                "collection_duration_seconds": collection_seconds,
                "rollouts": len(rollouts),
                "avg_arc_score": metrics.avg_arc_score,
                "avg_levels_completed": metrics.avg_levels_completed,
                "stats_at_minutes": stats_by_mark,
                "per_game": per_game,
                "completed_at": datetime.now(UTC).isoformat(),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if full_provider is not None:
                record_full_weight_eval_result(run_root, record)
            _hand_episodes_to_learner(
                config, run_root, cycle_index=cycle_index, iteration=iteration, rollouts=rollouts
            )
            last_evaluated_iteration = iteration
            log.info(
                "dedicated eval cycle %d complete: iteration=%d rollouts=%d "
                "avg_arc_score=%.2f avg_levels_completed=%.2f duration=%.0fs",
                cycle_index,
                iteration,
                len(rollouts),
                metrics.avg_arc_score,
                metrics.avg_levels_completed,
                collection_seconds,
            )
            # The metrics JSONL persists everything we need;
            # keep only the latest cycle's raw window artifacts for debugging.
            if previous_cycle_dir is not None:
                shutil.rmtree(previous_cycle_dir, ignore_errors=True)
            previous_cycle_dir = cycle_dir
            cycle_index += 1
    except Exception as error:
        record_run_failure(run_root, error)
        _abort_eval_collectors(collectors)
        raise
    finally:
        writer.close()
        stage_out_error_is_secondary = sys.exc_info()[0] is not None
        try:
            stage_out_rollout_scratch(
                run_root=run_root,
                rollout_roots=[rollouts_root],
                server_log_paths=server_log_paths,
                cleanup_rollout_artifacts=config.training.cleanup_rollout_artifacts,
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to stage official eval artifacts or server logs")
            if not stage_out_error_is_secondary:
                raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous official-game evaluation worker."
    )
    parser.add_argument("--config", required=True, help="Training config JSON path.")
    parser.add_argument("--run-root", required=True, help="Training run directory.")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        run_official_eval(config_path=args.config, run_root=Path(args.run_root))
    except Exception:
        log.exception("official eval worker failed")
        raise


if __name__ == "__main__":
    sys.exit(main())
