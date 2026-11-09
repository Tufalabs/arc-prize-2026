from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.tensorboard import SummaryWriter

from training.server_endpoints import collector_server_log_paths
from training.segment_budget import SegmentBudget
from training.rollout_archive import RolloutArchiver
from training.shutdown import (
    SEGMENT_YIELD_EXIT_CODE,
    SegmentYield,
    TrainingShutdown,
    raise_if_shutdown_requested,
    shutdown_requested,
    training_sigterm_handler,
)
from training.config import ExperimentConfig
from training.contracts import RolloutTrajectory
from training.game_sampling import (
    configure_game_sampling,
    publish_game_nonzero_rates,
)
from training.memory_profile import build_memory_logger
from training.memory_profile.report import generate_report
from training.ppo import PPOTrainer
from training.inference import (
    ContinuousRolloutWorker,
    _rollout_status_counts,
)
from training.metrics import (
    collection_scalars,
    expert_load_scalars,
    format_expert_load,
    summarize_rollouts,
)
from training.run_artifacts import (
    default_shared_storage_root,
    read_text_tail,
    record_run_launch_time,
    resolve_rollout_windows_root,
    stage_out_rollout_scratch,
    setup_experiment_directory,
    setup_logging_for_experiment,
    write_json_atomic as _write_json_atomic,
)
from training.run_failure import (
    FatalTrainingError,
    raise_if_run_failed,
    record_run_failure,
    is_teardown_error,
)


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
log = logging.getLogger(__name__)
_TRAINING_RUN_STATE_FILENAME = "training_state.json"
_INITIAL_POLICY_FILENAME = "initial_policy.json"
# Policy-drift watch band: the trust region only rejects ratio blowouts
# (clip-fraction gated), so a slow serving-vs-training weight drift can
# sit below it indefinitely. A healthy lag-1 window sits within ~1e-3 of
# 1.0; a stepped window outside this band earns a loud warning line and
# a TB alarm scalar, not a rejection.
_POLICY_DRIFT_RATIO_BAND = 0.02
_EFFECTIVE_CONFIG_FILENAME = "effective_config.json"
_ROLLOUT_RETURN_HISTORY_KEY = "rollout_return_history"
_ROLLOUT_ARC_SCORE_HISTORY_KEY = "rollout_arc_score_history"
_ROLLOUT_LEVELS_HISTORY_KEY = "rollout_levels_history"
_SAMPLE_BUDGET_STATE_KEY = "sample_budget_state"
_ENTROPY_CONTROL_STATE_KEY = "entropy_control_state"
_TENSORBOARD_TAG_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class SynchronousRolloutContractError(RuntimeError):
    """Fatal violation of the policy-coherent rollout-window contract."""


def _check_training_health(
    *,
    run_root: Path,
    collector: ContinuousRolloutWorker,
    official_eval_process: subprocess.Popen | None,
) -> None:
    """Propagate worker failures before consuming data or doing more optimization."""
    raise_if_run_failed(run_root)
    collector.raise_if_failed()
    if official_eval_process is not None:
        return_code = official_eval_process.poll()
        if return_code is not None:
            try:
                tail = read_text_tail(run_root / "official_eval.stdout.log", lines=40)
            except OSError as error:
                tail = f"Could not read evaluation log: {error}"
            error = FatalTrainingError(
                f"official eval worker exited unexpectedly with code {return_code}; "
                f"evaluation will not be restarted. See {run_root / 'official_eval.stdout.log'}"
                + (f"\n{tail}" if tail else "")
            )
            record_run_failure(run_root, error)
            # Preserve an earlier cause if another process reported it
            # between the health check above and this exit observation.
            raise_if_run_failed(run_root)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_output_dir(repo_root: Path, output_dir: str) -> Path:
    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _configure_hf_cache_defaults() -> None:
    shared_cache_root = default_shared_storage_root()
    cache_root = (
        shared_cache_root
        if shared_cache_root.exists()
        else Path.home() / ".cache" / "huggingface"
    )
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HF_ASSETS_CACHE", str(cache_root / "assets"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_root / "xet"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
    os.environ.setdefault("HUGGINGFACE_ASSETS_CACHE", os.environ["HF_ASSETS_CACHE"])


def _configure_fast_kernel_defaults() -> None:
    os.environ.setdefault("FLA_TILELANG", "0")
    os.environ.setdefault("FLA_CONV_BACKEND", "triton")


def _resolve_training_device(config: ExperimentConfig) -> str:
    requested = str(config.training.device).strip() or "cuda"
    if not torch.cuda.is_available():
        return "cpu"
    if requested.startswith("cuda:"):
        return requested
    if requested != "cuda":
        return requested

    return "cuda:0"


def _policy_lag_windows(
    *,
    iteration_index: int,
    rollout_policy_iteration: int,
) -> int:
    """Return the actual trainer-to-rollout policy distance.

    Fixed-policy collectors remain on policy -1 while the trainer advances,
    so their lag grows after the initial same-policy window even though the
    synchronous lag cap is intentionally not enforced for that smoke mode.
    """
    latest_available_policy_iteration = iteration_index - 1
    return max(0, latest_available_policy_iteration - rollout_policy_iteration)


def _validate_synchronous_rollout_window(
    window: Any,
    *,
    policy_lag_windows: int,
    expected_policy_iteration: int | None = None,
) -> None:
    """Fail-stop unless a complete wave is policy-coherent and at most lag one."""
    if (
        expected_policy_iteration is not None
        and window.policy_iteration != expected_policy_iteration
    ):
        raise SynchronousRolloutContractError(
            f"Fatal synchronous rollout contract violation: window {window.window_id} "
            f"has policy iteration {window.policy_iteration}, expected {expected_policy_iteration}. "
            "Synchronous training requires the latest completed update before every collection."
        )
    if policy_lag_windows > 1:
        raise SynchronousRolloutContractError(
            f"Fatal synchronous rollout contract violation: window "
            f"{window.window_id} has policy lag {policy_lag_windows}. Training "
            "is stopping instead of consuming experience beyond the one-window "
            "overlap contract."
        )
    expected_policy_iteration = int(window.policy_iteration)
    inconsistent_policy_rollouts = [
        rollout.trajectory_id
        for rollout in window.rollouts
        if int((rollout.metadata or {}).get("rollout_policy_iteration", -999))
        != expected_policy_iteration
    ]
    if inconsistent_policy_rollouts:
        raise SynchronousRolloutContractError(
            f"Fatal synchronous rollout contract violation: window "
            f"{window.window_id} mixed policy "
            f"iterations; expected {expected_policy_iteration} for every "
            "trajectory, mismatches=" + ", ".join(inconsistent_policy_rollouts)
        )


def _archive_server_log_tail(
    *,
    run_root: Path,
    server_log_path: Path | None,
    iteration_index: int,
    label: str,
    lines: int = 200,
) -> Path | None:
    if server_log_path is None or not server_log_path.exists():
        return None

    snapshots_dir = run_root / "vllm_server_logs"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots_dir / f"iter_{iteration_index:06d}_{label}.log"
    try:
        tail = read_text_tail(server_log_path, lines=lines)
    except OSError as exc:
        log.warning("failed to read vLLM server log from %s: %s", server_log_path, exc)
        return None

    snapshot_path.write_text(tail + ("\n" if tail else ""), encoding="utf-8")
    log.info("archived vLLM server log tail to %s", snapshot_path)
    return snapshot_path


def _archive_server_log_tails(
    *,
    run_root: Path,
    server_log_paths: list[Path],
    iteration_index: int,
    label: str,
    lines: int = 200,
) -> None:
    """Archive a tail per managed server. With server logs on node-local
    scratch these snapshots are the run's only durable server-log record, so
    every server gets one, not just server 0. Server 0 keeps the bare label
    for continuity with existing snapshot names."""
    for server_index, server_log_path in enumerate(server_log_paths):
        _archive_server_log_tail(
            run_root=run_root,
            server_log_path=server_log_path,
            iteration_index=iteration_index,
            label=label if server_index == 0 else f"{label}_server_{server_index}",
            lines=lines,
        )


def _short_text(value: str, *, limit: int = 280) -> str:
    collapsed = " ".join(str(value or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)] + "..."


def _log_rollout_examples(rollouts: list[Any], *, count: int) -> None:
    if count <= 0:
        return

    for example_index, rollout in enumerate(rollouts[:count], start=1):
        first_step = rollout.steps[0] if rollout.steps else None
        assistant_message = (
            first_step.assistant_message if first_step is not None else {}
        )
        tool_calls = assistant_message.get("tool_calls") or []
        tool_names = [
            str(
                (item.get("function", {}) if isinstance(item, dict) else {}).get(
                    "name", ""
                )
            ).strip()
            for item in tool_calls
            if isinstance(item, dict)
        ]
        metadata = rollout.metadata if isinstance(rollout.metadata, dict) else {}
        log.info(
            "rollout example %d/%d: game=%s return=%.4f done=%s steps=%d actions=%s status=%s run_dir=%s first_response=%s tools=%s",
            example_index,
            min(count, len(rollouts)),
            rollout.task_id,
            rollout.trajectory_reward,
            rollout.done,
            len(rollout.steps),
            metadata.get("run_total_actions"),
            rollout.termination_reason,
            metadata.get("run_dir"),
            _short_text(
                assistant_message.get("content")
                or assistant_message.get("reasoning")
                or "",
            ),
            ",".join(name for name in tool_names if name) or "<none>",
        )


def _game_id_set(game_ids: list[str]) -> set[str]:
    return {str(game_id).strip() for game_id in game_ids if str(game_id).strip()}


def _mean_sampled_game_return_variance(rollouts: list[RolloutTrajectory]) -> float:
    grouped_returns: dict[str, list[float]] = {}
    for rollout in rollouts:
        game_id = str(rollout.task_id or "").strip()
        if not game_id:
            continue
        grouped_returns.setdefault(game_id, []).append(float(rollout.trajectory_reward))

    if not grouped_returns:
        return 0.0

    variances: list[float] = []
    for returns in grouped_returns.values():
        if len(returns) < 2:
            variances.append(0.0)
            continue
        mean_return = sum(returns) / len(returns)
        variances.append(
            sum((value - mean_return) ** 2 for value in returns) / len(returns)
        )
    return float(sum(variances) / len(variances))


def _new_game_return_history(
    game_ids: list[str], *, history_size: int
) -> dict[str, deque[float]]:
    return {
        game_id: deque(maxlen=max(1, int(history_size)))
        for game_id in sorted(_game_id_set(game_ids))
    }


def _update_game_return_history(
    history: dict[str, deque[float]],
    *,
    rollouts: list[RolloutTrajectory],
    history_size: int,
) -> None:
    maxlen = max(1, int(history_size))
    for rollout in rollouts:
        game_id = str(rollout.task_id or "").strip()
        if not game_id:
            continue
        game_history = history.get(game_id)
        if game_history is None:
            game_history = deque(maxlen=maxlen)
            history[game_id] = game_history
        game_history.append(float(rollout.trajectory_reward))


def _rollout_final_arc_score(rollout: RolloutTrajectory) -> float:
    metadata = rollout.metadata
    if not isinstance(metadata, dict):
        return 0.0
    try:
        return max(0.0, float(metadata.get("final_arc_score", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _history_rollouts(rollouts: list[RolloutTrajectory]) -> list[RolloutTrajectory]:
    """Rollouts eligible for the persistent per-game history buffers.

    ERROR-status episodes are infra failures (server died mid-request) and
    carry no policy signal; recording their zero scores dilutes the 64-deep
    history means for days (observed 2026-07-16: score means halved by a
    7-minute fleet death). Zero-action TIMEOUTs remain eligible -- a policy
    that never acts is genuine policy behavior -- and so do CRASHED episodes,
    scored like a TIMEOUT on what they reached before their agent raised.
    """
    return [
        rollout
        for rollout in rollouts
        if str(rollout.termination_reason or "").strip().upper() != "ERROR"
    ]


def _update_game_arc_score_history(
    history: dict[str, deque[float]],
    *,
    rollouts: list[RolloutTrajectory],
    history_size: int,
) -> None:
    maxlen = max(1, int(history_size))
    for rollout in rollouts:
        game_id = str(rollout.task_id or "").strip()
        if not game_id:
            continue
        game_history = history.get(game_id)
        if game_history is None:
            game_history = deque(maxlen=maxlen)
            history[game_id] = game_history
        game_history.append(_rollout_final_arc_score(rollout))


def _rollout_levels_completed(rollout: RolloutTrajectory) -> float:
    metadata = rollout.metadata
    if not isinstance(metadata, dict):
        return 0.0
    try:
        return max(0.0, float(metadata.get("levels_completed", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _update_game_levels_history(
    history: dict[str, deque[float]],
    *,
    rollouts: list[RolloutTrajectory],
    history_size: int,
) -> None:
    maxlen = max(1, int(history_size))
    for rollout in rollouts:
        game_id = str(rollout.task_id or "").strip()
        if not game_id:
            continue
        game_history = history.get(game_id)
        if game_history is None:
            game_history = deque(maxlen=maxlen)
            history[game_id] = game_history
        game_history.append(_rollout_levels_completed(rollout))


def _mean_game_return_history(
    history: dict[str, deque[float]],
    game_ids: list[str],
    *,
    include_unobserved: bool = True,
) -> tuple[float, int, int]:
    configured_games = sorted(_game_id_set(game_ids))
    if not configured_games:
        return 0.0, 0, 0

    per_game_means: list[float] = []
    observed_games = 0
    for game_id in configured_games:
        values = list(history.get(game_id, ()))
        if values:
            observed_games += 1
            per_game_means.append(sum(values) / len(values))
        elif include_unobserved:
            per_game_means.append(0.0)
    return (
        sum(per_game_means) / len(per_game_means) if per_game_means else 0.0,
        observed_games,
        len(configured_games),
    )


def _games_nonzero_return_history(
    history: dict[str, deque[float]], game_ids: list[str]
) -> int:
    """Count configured games whose return history holds any non-zero entry."""
    configured_games = _game_id_set(game_ids)
    if not configured_games:
        return 0
    return sum(
        1
        for game_id in configured_games
        if any(float(value) != 0.0 for value in history.get(game_id, ()))
    )


def _game_return_history_state(
    history: dict[str, deque[float]], *, history_size: int
) -> dict[str, object]:
    maxlen = max(1, int(history_size))
    return {
        "version": 1,
        "history_size": maxlen,
        "returns": {
            str(game_id): [float(value) for value in values]
            for game_id, values in sorted(history.items())
            if str(game_id).strip()
        },
    }


def _restore_game_return_history(
    game_ids: list[str], *, history_size: int, payload: Any
) -> dict[str, deque[float]]:
    maxlen = max(1, int(history_size))
    history = _new_game_return_history(game_ids, history_size=maxlen)
    if not isinstance(payload, dict):
        return history

    raw_returns = payload.get("returns")
    if not isinstance(raw_returns, dict):
        raw_returns = payload

    for raw_game_id, raw_values in raw_returns.items():
        game_id = str(raw_game_id).strip()
        if not game_id or not isinstance(raw_values, list):
            continue
        restored_values: list[float] = []
        for value in raw_values[-maxlen:]:
            try:
                restored_values.append(float(value))
            except (TypeError, ValueError):
                continue
        history[game_id] = deque(restored_values, maxlen=maxlen)
    return history


def _game_nonzero_return_rates(
    history: dict[str, deque[float]],
) -> dict[str, float]:
    """Trailing nonzero-return rate per game with any recorded returns.

    Feeds training.game_sampling's "nonzero_rate" weights; games without a
    recorded return stay absent so the sampler treats them optimistically.
    """
    rates: dict[str, float] = {}
    for game_id, values in history.items():
        recorded = [float(value) for value in values]
        if not recorded:
            continue
        rates[str(game_id)] = sum(1.0 for value in recorded if value > 0.0) / len(
            recorded
        )
    return rates


def _finish_reason_counts(rollouts: list[RolloutTrajectory]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rollout in rollouts:
        rollout_counts = rollout.metadata.get("finish_reason_counts")
        if isinstance(rollout_counts, dict):
            for raw_reason, raw_count in rollout_counts.items():
                reason = str(raw_reason or "").strip().lower()
                if not reason:
                    continue
                try:
                    count = int(raw_count)
                except (TypeError, ValueError):
                    continue
                counts[reason] = counts.get(reason, 0) + max(0, count)
            continue
        for step in rollout.steps:
            reason = str(step.metadata.get("finish_reason") or "").strip().lower()
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _tensorboard_tag_component(value: str) -> str:
    sanitized = _TENSORBOARD_TAG_COMPONENT_RE.sub("_", value.strip()).strip("_")
    return sanitized or "unknown"


def _write_per_game_metric_group(
    writer: SummaryWriter,
    *,
    prefix: str,
    iteration_index: int,
    rollouts: list[RolloutTrajectory],
) -> dict[str, dict[str, float]]:
    rollouts_by_game: dict[str, list[RolloutTrajectory]] = {}
    for rollout in rollouts:
        game_id = str(rollout.task_id or "").strip()
        if not game_id:
            continue
        rollouts_by_game.setdefault(game_id, []).append(rollout)

    written: dict[str, dict[str, float]] = {}
    for game_id in sorted(rollouts_by_game):
        game_rollouts = rollouts_by_game[game_id]
        metrics = summarize_rollouts(game_rollouts)
        tag_prefix = f"{prefix}/{_tensorboard_tag_component(game_id)}"
        writer.add_scalar(
            f"{tag_prefix}/avg_return", metrics.avg_return, iteration_index
        )
        writer.add_scalar(
            f"{tag_prefix}/avg_levels_completed",
            metrics.avg_levels_completed,
            iteration_index,
        )
        writer.add_scalar(
            f"{tag_prefix}/avg_arc_score", metrics.avg_arc_score, iteration_index
        )
        written[game_id] = {
            "avg_return": metrics.avg_return,
            "avg_levels_completed": metrics.avg_levels_completed,
            "avg_arc_score": metrics.avg_arc_score,
        }
    return written


def _write_scalar_group(
    writer: SummaryWriter,
    *,
    iteration_index: int,
    scalars: dict[str, float | int],
) -> None:
    for name, value in scalars.items():
        writer.add_scalar(name, value, iteration_index)


def _write_training_tensorboard_scalars(
    writer: SummaryWriter,
    *,
    train_result: Any,
    iteration_index: int,
) -> None:
    optimizer_cycle_seconds = getattr(train_result, "optimizer_cycle_seconds", 0.0)
    _write_scalar_group(
        writer,
        iteration_index=iteration_index,
        scalars={
            "train/mean_ratio": float(train_result.mean_ratio),
            "train/policy_drift_alarm": float(
                int(getattr(train_result, "optimizer_steps", 0)) > 0
                and abs(float(train_result.mean_ratio) - 1.0) > _POLICY_DRIFT_RATIO_BAND
            ),
            "train/sample_count": train_result.sample_count,
            "train/samples/raw_tokenized_steps": train_result.raw_sample_count,
            "train/samples/eligible_after_zero_signal_filter": (
                train_result.eligible_sample_count
            ),
            "train/samples/filtered_zero_signal": (
                train_result.filtered_zero_signal_sample_count
            ),
            "train/samples/dropped_by_sample_budget": (
                train_result.subsampled_sample_count
            ),
            "train/samples/used_for_policy_update": train_result.sample_count,
            "train/sample_budget_fraction": train_result.sample_budget_fraction,
            "train/packed_episodes": float(getattr(train_result, "packed_episodes", 0)),
            "train/packed_sequences": float(
                getattr(train_result, "packed_sequences", 0)
            ),
            "train/segment_boundaries": float(
                getattr(train_result, "segment_boundaries", 0)
            ),
            "train/sequences_per_episode": (
                getattr(train_result, "packed_sequences", 0)
                / getattr(train_result, "packed_episodes", 0)
                if getattr(train_result, "packed_episodes", 0)
                else 0.0
            ),
            "train/final_loss": train_result.final_loss,
            "train/mean_entropy": train_result.mean_entropy,
            "train/entropy_loss": train_result.entropy_loss,
            "train/entropy_coef": train_result.entropy_coef_used,
            "train/zero_signal_denominator_scale": (
                train_result.zero_signal_denominator_scale
            ),
            "train/probe_entropy": train_result.probe_entropy,
            "train/entropy_to_policy_grad_ratio": (
                train_result.entropy_to_policy_grad_ratio
            ),
            "train/final_learning_rate": train_result.final_learning_rate,
            "train/optimizer_steps": train_result.optimizer_steps,
            "train/mean_clip_fraction": train_result.mean_clip_fraction,
            "train/previous_policy_kl": getattr(
                train_result, "mean_previous_policy_kl", 0.0
            ),
            "train/reference_kl": getattr(train_result, "mean_reference_kl", 0.0),
            "train/window_update_rejected": (
                1.0 if getattr(train_result, "window_update_rejected", False) else 0.0
            ),
            "train/post_update_clip_fraction": getattr(
                train_result, "post_update_clip_fraction", 0.0
            ),
            "train/trust_region_lr_scale": getattr(
                train_result, "trust_region_lr_scale", 1.0
            ),
            "train/mean_grad_norm": train_result.mean_grad_norm,
            "train/baseline_loss": train_result.baseline_loss,
            "train/baseline_normalized_mse": train_result.baseline_loss,
            "train/baseline_mean_value": train_result.baseline_mean_value,
            "train/z_timing/baseline_feature_seconds": getattr(
                train_result, "baseline_feature_seconds", 0.0
            ),
            "train/baseline_mean_target": train_result.baseline_mean_target,
            "train/baseline_explained_variance": (
                train_result.baseline_explained_variance
            ),
            "train/baseline_post_update_mean_value": (
                train_result.baseline_post_update_mean_value
            ),
            "train/baseline_post_update_mean_absolute_error": (
                train_result.baseline_post_update_mean_absolute_error
            ),
            "train/baseline_post_update_explained_variance": (
                train_result.baseline_post_update_explained_variance
            ),
            "train/tokens_per_s": (
                getattr(train_result, "processed_tokens", 0)
                / optimizer_cycle_seconds
                if optimizer_cycle_seconds > 0
                else 0.0
            ),
            "train/forward_time_s": getattr(train_result, "forward_time_s", 0.0),
            "train/backward_time_s": getattr(train_result, "backward_time_s", 0.0),
            "train/forward_backward_time_s": (
                getattr(train_result, "forward_time_s", 0.0)
                + getattr(train_result, "backward_time_s", 0.0)
            ),
            "train/z_timing/optimization_seconds": train_result.optimization_seconds,
            "train/z_timing/optimizer_cycle_seconds": optimizer_cycle_seconds,
            "train/z_timing/policy_export_seconds": train_result.policy_export_seconds,
            "train/extra_cycles": float(train_result.extra_cycles),
            "train/extra_optimizer_steps": float(train_result.extra_optimizer_steps),
            "train/z_timing/extra_phase_seconds": float(
                train_result.extra_phase_seconds
            ),
            "train/z_timing/train_window_seconds": (
                train_result.total_train_window_seconds
            ),
        },
    )
    for name, value in getattr(train_result, "baseline_metrics", {}).items():
        writer.add_scalar(f"train/baseline/{name}", value, iteration_index)
    for name, value in getattr(train_result, "critic_metrics", {}).items():
        writer.add_scalar(f"train/critic/{name}", value, iteration_index)
    for name, value in getattr(train_result, "policy_update_metrics", {}).items():
        writer.add_scalar(f"train/policy_update/{name}", value, iteration_index)
    # Null provenance means no measured tokens; do not chart it as policy 0.
    for name in (
        "token_count",
        "mean_abs",
        "p95_abs",
        "max_abs",
        "outlier_count",
        "outlier_fraction",
        "signed_mean",
        "biased_token_count",
        "biased_mean_abs",
        "sampler_policy_iteration",
        "trainer_policy_iteration",
        "collection_lag",
        "checked_policy_lag",
        "seconds",
        "snapshot_seconds",
        "forward_seconds",
        "input_token_count",
        "sample_count",
        "batch_count",
        "startup",
        "expanded",
    ):
        value = getattr(train_result, f"rollout_logprob_parity_{name}", None)
        if value is not None:
            writer.add_scalar(f"train/logprob_parity/{name}", value, iteration_index)


def _write_expert_load(
    writer: SummaryWriter,
    *,
    summary: dict[str, dict[str, float]],
    iteration_index: int,
    rollout_step: int,
    num_iterations: int,
) -> None:
    """Flash expert load balance: rollout/ tags on the rollout axis like
    every rollout/ scalar, train/ and eval/ on the trainer iteration."""
    if not summary:
        return
    for name, value in expert_load_scalars(summary).items():
        step = rollout_step if name.startswith("rollout/") else iteration_index
        writer.add_scalar(name, value, step)
    log.info(
        "expert load iteration %s/%s %s",
        iteration_index + 1,
        num_iterations,
        format_expert_load(summary),
    )


def _publish_initial_policy(
    run_root: Path, trainer: PPOTrainer, *, resume_metadata: dict[str, Any]
) -> Path:
    """Publish the exact initial rollout/eval policy before either collector starts."""
    # Export the trainer's actual initialized weights, including identity LoRA
    # for a fresh base. Every measured window must exercise runtime LoRA.
    adapter_dir, runtime_adapter, _ = trainer._export_policy_snapshot(
        run_root / "checkpoints", -1
    )
    if not (runtime_adapter / "adapter_config.json").is_file():
        raise RuntimeError(
            f"initial runtime adapter export is incomplete: {runtime_adapter}"
        )
    trainer_state_path = trainer._save_training_state(
        run_root / "checkpoints",
        iteration_index=-1,
        adapter_dir=adapter_dir,
        runtime_adapter_dir=runtime_adapter,
    )
    # A preemption during the first window must restore the exact initialized
    # policy, critic, optimizer, and RNG state, including a warm start's state.
    resume_state = {
        **resume_metadata,
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "run_root": str(run_root),
        "last_completed_iteration": -1,
        "next_iteration": 0,
        "next_schedule_index": 0,
        "adapter_dir": _run_relative_path(run_root, adapter_dir),
        "runtime_adapter_dir": _run_relative_path(run_root, runtime_adapter),
        "trainer_state_path": _run_relative_path(run_root, trainer_state_path),
        "optimizer_step_count": trainer.optimizer_step_count,
        "latest_window_optimizer_steps": 0,
    }
    _write_json_atomic(
        run_root / _INITIAL_POLICY_FILENAME,
        {
            "last_completed_iteration": -1,
            "initial_adapter_dir": str(adapter_dir.resolve()),
            "runtime_adapter_dir": str(runtime_adapter.resolve()),
            "resume_state": resume_state,
        },
    )
    _write_json_atomic(_training_run_state_path(run_root), resume_state)
    return runtime_adapter


def _run_relative_path(run_root: Path, path: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(run_root.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return str(path)


def _resolve_run_state_path(run_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return run_root / path


def _training_run_state_path(run_root: Path) -> Path:
    return run_root / _TRAINING_RUN_STATE_FILENAME


def _load_training_run_state(run_root: Path) -> dict[str, Any] | None:
    state_path = _training_run_state_path(run_root)
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{state_path} must contain a JSON object.")
    return payload


def _clamp_sample_budget_fraction(config: ExperimentConfig, value: float) -> float:
    budget = config.training.sample_budget
    if budget.mode == "full":
        return 1.0
    return min(budget.max_fraction, max(budget.min_fraction, float(value)))


def _initial_sample_budget_state(
    config: ExperimentConfig, resume_state: dict[str, Any] | None
) -> dict[str, Any]:
    budget = config.training.sample_budget
    fraction = budget.initial_fraction
    if resume_state is not None:
        payload = resume_state.get(_SAMPLE_BUDGET_STATE_KEY)
        if isinstance(payload, dict):
            try:
                fraction = float(payload.get("next_fraction", fraction))
            except (TypeError, ValueError):
                fraction = budget.initial_fraction
    fraction = _clamp_sample_budget_fraction(config, fraction)
    return {
        "mode": budget.mode,
        "current_fraction": fraction,
        "next_fraction": fraction,
        "target_train_to_collection_ratio": budget.target_train_to_collection_ratio,
        "last_update_reason": "initial",
        "last_observed_train_to_collection_ratio": None,
    }


def _format_break_fraction(rollouts) -> float:
    """Share of the window's reported model turns the harness flagged
    malformed (metadata.format_break): the leading indicator of the
    drift-to-degeneration collapse."""
    steps = [
        step
        for rollout in rollouts
        for step in (getattr(rollout, "steps", None) or [])
        if not (isinstance(step.metadata, dict) and step.metadata.get("void_attempt"))
    ]
    if not steps:
        return 0.0
    flagged = sum(
        1
        for step in steps
        if isinstance(step.metadata, dict) and step.metadata.get("format_break")
    )
    return flagged / len(steps)


def _collapse_alarm_reasons(
    *,
    format_break_fraction: float,
    format_break_fraction_alarm: float,
    actions_per_rollout: float,
    actions_history: list[float],
    actions_drop_fraction: float,
    watch_windows: int,
) -> list[str]:
    """Why the collapse watch fires this window (empty = healthy).

    Two leading indicators of the drift the 8-GPU champion run died of: the
    malformed-turn fraction crossing its alarm, and actions per episode
    falling below ``actions_drop_fraction`` of their trailing median once
    ``watch_windows`` windows of history exist (the median, not the peak: one
    spiky window must not turn every normal window after it into an alarm)."""
    reasons: list[str] = []
    if (
        format_break_fraction_alarm > 0.0
        and format_break_fraction >= format_break_fraction_alarm
    ):
        reasons.append(
            f"format_break_fraction {format_break_fraction:.3f} >= "
            f"{format_break_fraction_alarm:.3f}"
        )
    history = [float(value) for value in actions_history]
    if len(history) >= max(1, int(watch_windows)) and actions_drop_fraction > 0.0:
        ordered = sorted(history)
        middle = len(ordered) // 2
        baseline = (
            ordered[middle]
            if len(ordered) % 2
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        if baseline > 0.0 and actions_per_rollout < actions_drop_fraction * baseline:
            reasons.append(
                f"actions_per_rollout {actions_per_rollout:.1f} < "
                f"{actions_drop_fraction:.2f} x trailing median {baseline:.1f}"
            )
    return reasons


def _rollout_status_scalars(rollouts) -> dict[str, float]:
    """A window's context-exhausted and crashed rollouts, as counts and
    fractions of the window. Crashed: games whose agent raised, contained by
    the harness and scored on what they reached (status CRASHED; the window
    goes on)."""
    counts = _rollout_status_counts(rollouts)
    total = max(1, len(rollouts))
    scalars: dict[str, float] = {}
    for status in ("context_exhausted", "crashed"):
        scalars[f"rollout/{status}_count"] = float(counts[status])
        scalars[f"rollout/{status}_fraction"] = counts[status] / total
    return scalars


def _require_no_infra_error_rollouts(rollouts, *, iteration_index):
    """A rollout the infrastructure truncated (termination_reason == ERROR)
    fails the run loudly, naming every crashed episode. A crashed rollout is
    a bug to find and fix, never data to quietly skip: training past it
    hides breakage and biases the window toward the episodes that survived
    it. The raise is not chained by auto-relaunch, so the run stays down
    until the cause is fixed.

    A game whose agent raised (termination_reason == CRASHED) is not one of
    these since 2026-10-11: the workers run the harness's own
    ``on_agent_error="crash_game"``, which ends only that game and scores
    what it reached, and it trains like any other terminal (counted in
    ``rollout/crashed_count``). A serving failure still stops the window
    before it gets here: the rollout gateway poisons it on any failed
    generation."""
    errored = [
        rollout
        for rollout in rollouts
        if str(getattr(rollout, "termination_reason", "") or "").strip().upper()
        == "ERROR"
    ]
    if not errored:
        return
    details = []
    for rollout in errored:
        meta = getattr(rollout, "metadata", None) or {}
        note = str(
            meta.get("note") or meta.get("error") or meta.get("status") or ""
        ).strip()
        details.append(
            f"  - game={getattr(rollout, 'task_id', '?')} "
            f"trajectory={getattr(rollout, 'trajectory_id', '?')}"
            + (f" note={note[:400]}" if note else "")
        )
    raise RuntimeError(
        f"FATAL: {len(errored)} of {len(rollouts)} rollout(s) crashed during "
        f"collection for iteration {iteration_index + 1} "
        "(termination_reason=ERROR). Refusing to train past a crashed "
        "rollout:\n" + "\n".join(details) + "\n  full agent tracebacks: "
        "platypus_worker_logs_port-*/worker.log in this run directory"
    )


def _initial_entropy_control_state(
    config: ExperimentConfig, resume_state: dict[str, Any] | None
) -> dict[str, Any]:
    control = config.training.entropy_control
    coef = control.coef_init
    applied_coef = 0.0
    if resume_state is not None:
        payload = resume_state.get(_ENTROPY_CONTROL_STATE_KEY)
        if isinstance(payload, dict):
            try:
                coef = float(payload.get("current_coef", coef))
                applied_coef = float(payload.get("applied_coef", 0.0))
            except (TypeError, ValueError):
                coef = control.coef_init
                applied_coef = 0.0
    coef = min(control.coef_max, max(control.coef_min, coef))
    return {
        "enabled": control.enabled,
        "current_coef": coef,
        # A fresh run waits for a low-entropy measurement. A resumed run
        # keeps the bonus selected by its last accepted window, including
        # across no-signal windows before the next accepted update.
        "applied_coef": (min(coef, max(0.0, applied_coef)) if control.enabled else 0.0),
        "last_mean_entropy": None,
        "last_action": "initial",
    }


def _update_entropy_control_state(
    *,
    config: ExperimentConfig,
    state: dict[str, Any],
    train_result: Any,
) -> dict[str, Any]:
    """Sign-based bounded integrator with a dead-band (Skywork-OR1 /
    TRL use_adaptive_entropy family): step the coefficient up below the
    entropy band, decay it toward the floor above the band, hold inside.
    Hard caps + decay-above-band are the anti-windup/anti-domination
    guarantees; iterations without an optimizer step contribute no signal
    and leave the coefficient untouched. The gate is optimizer_steps, not
    sample_count: a parity-rejected window returns sample_count=len(samples)
    with optimizer_steps=0 and mean_entropy=0.0, which a sample_count guard
    would misread as entropy collapse and step the coefficient up. A
    trust-region-rejected window reports positive optimizer_steps, but its
    update was rolled back — the entropy it measured belongs to discarded
    weights, so it is no_signal too."""
    control = config.training.entropy_control
    coef = min(
        control.coef_max,
        max(control.coef_min, float(state.get("current_coef", control.coef_init))),
    )
    mean_entropy: float | None = None
    action = "hold"
    if not control.enabled:
        action = "disabled"
    elif int(getattr(train_result, "optimizer_steps", 0)) <= 0 or bool(
        getattr(train_result, "window_update_rejected", False)
    ):
        action = "no_signal"
    else:
        mean_entropy = float(train_result.mean_entropy)
        if mean_entropy < control.target_low:
            coef = min(control.coef_max, coef + control.coef_step)
            action = "step_up"
        else:
            # One-sided floor (owner decision 2026-07-20, Skywork/TRL
            # semantics): at or above the floor the bonus is fully gated
            # off and the coefficient relaxes; entropy is never pushed
            # DOWN by this controller.
            coef = max(control.coef_min, coef - control.coef_step)
            action = "floor_ok"
        if mean_entropy > control.target_high:
            action = "alert_high"
            log.warning(
                "entropy %.4f above alert threshold %.3f with the bonus "
                "gated off -- if it keeps climbing, the cause is the task "
                "loss (clip asymmetry / low-signal batches), not this "
                "controller",
                mean_entropy,
                control.target_high,
            )
    # One-sided gating: the bonus is applied ONLY while entropy is below the
    # floor. On no-signal windows the previous applied value carries over.
    if mean_entropy is None:
        applied_coef = float(state.get("applied_coef", 0.0))
    elif mean_entropy < control.target_low:
        applied_coef = coef
    else:
        applied_coef = 0.0
    if control.enabled:
        log.info(
            "entropy control: mean_entropy=%s floor=%.3f alert=%.3f -> %s, "
            "coef=%.6f applied=%.6f",
            "n/a" if mean_entropy is None else f"{mean_entropy:.4f}",
            control.target_low,
            control.target_high,
            action,
            coef,
            applied_coef,
        )
    return {
        "enabled": control.enabled,
        "current_coef": coef,
        "applied_coef": applied_coef,
        "last_mean_entropy": mean_entropy,
        "last_action": action,
    }


def _update_sample_budget_state(
    *,
    config: ExperimentConfig,
    state: dict[str, Any],
    train_result: Any,
    rollout_iteration_seconds: float,
    trainer_iteration_seconds: float,
    policy_lag_windows: int,
    is_first_process_optimization_pass: bool,
) -> dict[str, Any]:
    """Steer the sample fraction so one full trainer iteration (window receipt
    through the adapter hot-load, minus opportunistic extra cycles) takes
    ``target_train_to_collection_ratio`` of one full rollout iteration (the
    wall time between consecutive windows becoming available)."""
    budget = config.training.sample_budget
    current_fraction = _clamp_sample_budget_fraction(
        config, float(state.get("current_fraction", budget.initial_fraction))
    )
    observed_ratio: float | None = None
    reason = "disabled"
    next_fraction = current_fraction

    if budget.mode == "auto" and is_first_process_optimization_pass:
        # The first optimization pass after every process start includes
        # one-time model/kernel compilation and allocator warm-up. Persisted
        # iteration numbers cannot identify this pass after a resume, so the
        # caller supplies process-local state. Keep the restored coverage and
        # begin timing feedback after this warm-up pass completes.
        reason = "first_process_optimization_warmup"
    elif budget.mode == "auto":
        # Gate on an actual optimization signal: a parity-rejected window
        # reports sample_count>0 but optimizer_steps=0 and a near-zero train
        # duration; feeding that into the timing loop reads as spare trainer
        # capacity and wrongly raises the coverage fraction.
        if (
            int(getattr(train_result, "optimizer_steps", 0)) > 0
            and rollout_iteration_seconds > 0.0
            and trainer_iteration_seconds > 0.0
        ):
            observed_ratio = trainer_iteration_seconds / rollout_iteration_seconds
            # Pure timing feedback: trainer finished before the next wave
            # filled -> raise the fraction; trainer slower than the wave ->
            # lower it. No policy-lag cap: in the one-wave async pipeline the
            # lag is structural (a 30-min episode spans >1 iteration by
            # construction), training less does not reduce it, and the old
            # cap ratcheted the fraction to the floor and pinned it there.
            # Off-policy safety lives in the per-segment captured logprobs
            # and PPO clipping, not the batch size.
            raw_next = current_fraction * (
                budget.target_train_to_collection_ratio / max(observed_ratio, 1e-9)
            )
            smoothed = current_fraction + budget.adjustment_rate * (
                raw_next - current_fraction
            )
            next_fraction = _clamp_sample_budget_fraction(config, smoothed)
            reason = "timing_update"
        else:
            reason = "no_optimizer_timing_signal"
    elif budget.mode == "full":
        next_fraction = 1.0

    return {
        "mode": budget.mode,
        "current_fraction": next_fraction,
        "next_fraction": next_fraction,
        "used_fraction": train_result.sample_budget_fraction,
        "target_train_to_collection_ratio": budget.target_train_to_collection_ratio,
        "adjustment_rate": budget.adjustment_rate,
        "last_update_reason": reason,
        "last_observed_train_to_collection_ratio": observed_ratio,
        "last_rollout_iteration_seconds": float(rollout_iteration_seconds),
        "last_trainer_iteration_seconds": float(trainer_iteration_seconds),
        "last_sample_count": int(train_result.sample_count),
        "last_eligible_sample_count": int(train_result.eligible_sample_count),
        "last_policy_lag_windows": int(policy_lag_windows),
    }


def _build_rollout_collector(
    config: ExperimentConfig,
    *,
    repo_root: Path,
    rollouts_root: Path,
    run_root: Path,
) -> ContinuousRolloutWorker:
    """Build the sole training rollout path: synchronous fixed-policy waves."""
    return ContinuousRolloutWorker(
        config,
        repo_root=repo_root,
        output_root=rollouts_root,
        run_root=run_root,
    )


def _has_checkpoint_artifacts_without_state(run_root: Path) -> bool:
    checkpoints_root = run_root / "checkpoints"
    if not checkpoints_root.exists():
        return False
    if (checkpoints_root / "latest_adapter").exists() or (
        checkpoints_root / "latest_runtime_adapter"
    ).exists():
        return True
    return (
        any(checkpoints_root.glob("adapter_iter_*"))
        or any(checkpoints_root.glob("full_weight_iter_*"))
        or (checkpoints_root / "latest").is_symlink()
        or (checkpoints_root / ".full_weight_in_progress").exists()
    )


def _resolve_resume_state(
    run_root: Path, *, requested_run_dir: bool
) -> dict[str, Any] | None:
    resume_state = _load_training_run_state(run_root)
    if resume_state is not None and not requested_run_dir:
        raise RuntimeError(
            f"{run_root} contains {_TRAINING_RUN_STATE_FILENAME}, but resume was not explicitly requested. "
            "Use --run-dir/ARC3_RUN_DIR to resume this run, or choose a fresh RUN_TIMESTAMP for a new experiment."
        )
    if resume_state is None and _has_checkpoint_artifacts_without_state(run_root):
        if requested_run_dir:
            raise RuntimeError(
                f"{run_root} contains checkpoint artifacts but no {_TRAINING_RUN_STATE_FILENAME}. "
                "It cannot be safely resumed because optimizer state was not saved."
            )
        raise RuntimeError(
            f"{run_root} already contains checkpoint artifacts but no {_TRAINING_RUN_STATE_FILENAME}. "
            "Refusing to start a fresh experiment in a populated run directory; choose a fresh RUN_TIMESTAMP."
        )
    return resume_state


def _save_config_snapshots(
    run_root: Path, config_path: str, config: ExperimentConfig
) -> Path:
    source_config_path = Path(config_path)
    if source_config_path.exists():
        destination = run_root / "input_config.json"
        if not destination.exists():
            shutil.copy2(source_config_path, destination)
    effective_config_path = run_root / _EFFECTIVE_CONFIG_FILENAME
    _write_json_atomic(effective_config_path, config.to_dict())
    return effective_config_path


def _write_training_run_state(
    *,
    run_root: Path,
    config_path: str,
    config: ExperimentConfig,
    iteration_metrics: dict[str, object],
    train_result: Any,
    trainer: PPOTrainer,
    rollout_return_history: dict[str, deque[float]] | None = None,
    rollout_arc_score_history: dict[str, deque[float]] | None = None,
    rollout_levels_history: dict[str, deque[float]] | None = None,
    sample_budget_state: dict[str, Any] | None = None,
    entropy_control_state: dict[str, Any] | None = None,
    full_backend: Any | None = None,
) -> None:
    iteration_index = int(iteration_metrics["iteration"])
    next_schedule_index = int(
        iteration_metrics.get("next_schedule_index", iteration_index + 1)
    )
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "run_root": str(run_root),
        "config_path": str(config_path),
        "last_completed_iteration": iteration_index,
        "next_iteration": iteration_index + 1,
        "next_schedule_index": next_schedule_index,
        "optimizer_step_count": trainer.optimizer_step_count,
        "latest_window_optimizer_steps": train_result.optimizer_steps,
        "advantage_group_mean_reward": train_result.advantage_group_mean_reward,
        "advantage_group_std_reward": train_result.advantage_group_std_reward,
        "total_configured_iterations": config.num_iterations,
        "latest_iteration_metrics": iteration_metrics,
    }
    if full_backend is None:
        payload.update(
            {
                "adapter_dir": _run_relative_path(run_root, train_result.adapter_dir),
                "runtime_adapter_dir": _run_relative_path(
                    run_root, train_result.runtime_adapter_dir
                ),
                "trainer_state_path": _run_relative_path(
                    run_root, train_result.trainer_state_path
                ),
            }
        )
    if sample_budget_state is not None:
        payload[_SAMPLE_BUDGET_STATE_KEY] = sample_budget_state
    if entropy_control_state is not None:
        payload[_ENTROPY_CONTROL_STATE_KEY] = entropy_control_state
    if rollout_return_history is not None:
        payload[_ROLLOUT_RETURN_HISTORY_KEY] = _game_return_history_state(
            rollout_return_history,
            history_size=config.sampling.return_history_size,
        )
    if rollout_arc_score_history is not None:
        payload[_ROLLOUT_ARC_SCORE_HISTORY_KEY] = _game_return_history_state(
            rollout_arc_score_history,
            history_size=config.sampling.return_history_size,
        )
    if rollout_levels_history is not None:
        payload[_ROLLOUT_LEVELS_HISTORY_KEY] = _game_return_history_state(
            rollout_levels_history,
            history_size=config.sampling.return_history_size,
        )
    if full_backend is None:
        _write_json_atomic(_training_run_state_path(run_root), payload)
    else:
        payload.pop("version")
        full_backend.finalize_iteration(
            run_root=run_root,
            iteration_index=iteration_index,
            train_result=train_result,
            common_run_state=payload,
            force_checkpoint=shutdown_requested(),
        )


def _resume_paths_from_state(
    run_root: Path, state: dict[str, Any]
) -> tuple[Path, Path, Path]:
    adapter_dir = _resolve_run_state_path(run_root, state["adapter_dir"])
    runtime_adapter_dir = _resolve_run_state_path(
        run_root, state["runtime_adapter_dir"]
    )
    trainer_state_path = _resolve_run_state_path(run_root, state["trainer_state_path"])
    # The runtime (vLLM-format) export may live on a previous job's node
    # scratch; it is regenerated from the adapter checkpoint at resume.
    missing = [path for path in (adapter_dir, trainer_state_path) if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            f"Run resume state points at missing checkpoint artifacts: {formatted}"
        )
    return adapter_dir, runtime_adapter_dir, trainer_state_path


def run_training(
    config_path: str,
    *,
    backend_factory: Callable[[ExperimentConfig], Any] | None = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _configure_hf_cache_defaults()
    _configure_fast_kernel_defaults()
    if os.environ.get("ARC3_BENCH_SYNTHETIC_WINDOWS", "0") not in ("", "0"):
        raise ValueError(
            "Synthetic benchmark logprobs cannot satisfy the mandatory policy-parity contract"
        )
    config = ExperimentConfig.load(config_path)
    if config.training.uses_torchrun and backend_factory is None:
        raise RuntimeError(
            "Full-weight/Flash training must use its dedicated entrypoint under torchrun"
        )
    if (
        config.training.weight_mode == "full_weight"
        and int(os.environ.get("ARC3_BENCH_SYNTHETIC_WINDOWS", "0") or 0) > 0
    ):
        raise ValueError(
            "ARC3_BENCH_SYNTHETIC_WINDOWS is not supported for full-weight training"
        )
    full_backend = backend_factory(config) if backend_factory is not None else None
    if full_backend is not None and not config.training.uses_torchrun:
        raise ValueError(
            "A full-weight backend requires training.weight_mode='full_weight'"
        )
    from training.harness import get_trainer_support
    from training.multimodal import build_checkpoint_identity

    harness_support = get_trainer_support(config.inference.environment)
    dataset_identity = harness_support.resolve_datasets(config)
    checkpoint_identity = build_checkpoint_identity(
        config, support=harness_support, manifest=dataset_identity
    )
    if full_backend is not None and checkpoint_identity is None:
        checkpoint_identity = {
            "environment": config.inference.environment,
            "advantage_estimator": config.training.advantage_estimator,
            "games": sorted(config.inference.games or [config.inference.game]),
        }
    grpo_puzzles = config.sampling.game_sampling_games_per_window
    trainer_identity_kwargs = (
        {
            "checkpoint_identity": checkpoint_identity,
            "grpo_group_size": config.parallel_rollouts // grpo_puzzles
            if grpo_puzzles
            else None,
            "grpo_puzzles_per_window": grpo_puzzles,
        }
        if config.training.advantage_estimator in {"grpo", "dr_grpo"}
        or full_backend is not None
        else {"checkpoint_identity": checkpoint_identity}
    )
    run_root, _log_path, tensorboard_dir = setup_experiment_directory(
        _resolve_output_dir(repo_root, config.training.output_dir)
    )
    # A failed run stays failed across scheduler continuations or manual
    # invocation; clearing its failure record is an explicit recovery step.
    raise_if_run_failed(run_root)
    # Anchor for wall-clock schedules (inference.max_runtime_minutes_schedule):
    # first trainer start, not sbatch submission; idempotent across legs.
    record_run_launch_time(run_root)
    requested_run_dir = bool(os.environ.get("ARC3_RUN_DIR", "").strip())
    resume_state = _resolve_resume_state(run_root, requested_run_dir=requested_run_dir)
    log.info(
        "policy refresh: runtime LoRA hot-loads between synchronous waves "
        "with one-window overlap."
    )
    config.training.device = _resolve_training_device(config)
    _set_seed(config.training.seed)
    setup_logging_for_experiment(
        log_file_path=_log_path, fmt=LOG_FORMAT, append=resume_state is not None
    )
    effective_config_path = _save_config_snapshots(run_root, config_path, config)
    start_iteration = 0
    last_completed_iteration = -1
    resume_adapter_dir: Path | None = None
    resume_runtime_adapter_dir: Path | None = None
    resume_trainer_state_path: Path | None = None
    next_schedule_index = 0
    full_resume = None
    if full_backend is not None:
        full_resume = full_backend.resolve_resume(run_root, resume_state)
        start_iteration = full_resume.start_iteration
        last_completed_iteration = full_resume.last_completed_iteration
        next_schedule_index = full_resume.next_schedule_index
    elif resume_state is not None:
        version = int(resume_state.get("version", 0))
        if version != 1:
            raise RuntimeError(
                f"Unsupported run resume state version {version}; expected 1 in {_training_run_state_path(run_root)}."
            )
        start_iteration = int(resume_state.get("next_iteration", 0))
        last_completed_iteration = int(
            resume_state.get("last_completed_iteration", start_iteration - 1)
        )
        next_schedule_index = int(
            resume_state.get("next_schedule_index", start_iteration)
        )
        resume_adapter_dir, resume_runtime_adapter_dir, resume_trainer_state_path = (
            _resume_paths_from_state(
                run_root,
                resume_state,
            )
        )
    log.info("starting training run %s", run_root)
    log.info("loaded config from %s", config_path)
    frozen_repo_root = str(os.environ.get("ARC3_FROZEN_REPO_ROOT") or "").strip()
    if frozen_repo_root:
        log.info("using frozen source snapshot: %s", frozen_repo_root)
    runtime_fingerprint_path = str(
        os.environ.get("ARC3_RUNTIME_FINGERPRINT_PATH") or ""
    ).strip()
    if runtime_fingerprint_path:
        log.info("runtime environment fingerprint: %s", runtime_fingerprint_path)
    log.info("training device resolved to %s", config.training.device)
    log.info("inference endpoint: %s", config.inference.base_url)
    log.info("run log file: %s", _log_path)
    log.info("tensorboard dir: %s", tensorboard_dir)
    log.info("rollout scheduling: parallel_rollouts=%s", config.parallel_rollouts)
    log.info(
        "sample budget: mode=%s initial_fraction=%.3f min=%.3f max=%.3f target_train_to_collection=%.3f adjustment_rate=%.3f",
        config.training.sample_budget.mode,
        config.training.sample_budget.initial_fraction,
        config.training.sample_budget.min_fraction,
        config.training.sample_budget.max_fraction,
        config.training.sample_budget.target_train_to_collection_ratio,
        config.training.sample_budget.adjustment_rate,
    )
    log.info(
        "sampling: game_sampling_mode=%s weight_floor=%.3f return_history_size=%d",
        config.sampling.game_sampling_mode,
        config.sampling.game_sampling_weight_floor,
        config.sampling.return_history_size,
    )
    log.info(
        "rollout buffering: synchronous fixed-policy waves with one-window overlap; "
        "adapters change only between waves and policy lag is capped at one update."
    )
    if resume_state is not None:
        log.info(
            "resuming run %s from iteration %s using adapter=%s runtime_adapter=%s trainer_state=%s next_schedule_index=%s",
            run_root,
            start_iteration,
            resume_adapter_dir,
            resume_runtime_adapter_dir,
            resume_trainer_state_path,
            next_schedule_index,
        )

    rollouts_root = resolve_rollout_windows_root(run_root / "rollouts")
    official_eval_rollouts_root = resolve_rollout_windows_root(
        run_root / "official_eval_rollouts"
    )
    official_eval_server_log_paths = collector_server_log_paths(
        config, run_root, evaluation=True
    )
    checkpoints_root = run_root / "checkpoints"
    run_root.mkdir(parents=True, exist_ok=True)
    rollouts_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    if rollouts_root != run_root / "rollouts":
        log.info("rollout windows on node-local scratch: %s", rollouts_root)

    collector = _build_rollout_collector(
        config,
        repo_root=repo_root,
        rollouts_root=rollouts_root,
        run_root=run_root,
    )
    # One window per remaining iteration; the overlapped schedule would
    # otherwise collect one more after the final iteration and drop it.
    # Test doubles for the collector may not implement the limit.
    limit_windows = getattr(collector, "limit_windows", None)
    if limit_windows is not None:
        limit_windows(max(0, config.num_iterations - start_iteration))
    published_policy_path: Path | None = resume_runtime_adapter_dir
    results: list[dict[str, object]] = []
    writer_kwargs = {"purge_step": start_iteration} if resume_state is not None else {}
    if resume_state is not None:
        # Self-heal the queue-wait gap this resume just created: rewrite the
        # prior segments' events (closed files; our writer is not open yet)
        # so TensorBoard's RELATIVE axis reads as active training time.
        try:
            from training.tb_gap_collapse import collapse_run_dir

            collapse_run_dir(tensorboard_dir)
        except Exception as exc:  # cosmetic repair must never block training
            log.warning("tensorboard gap collapse skipped: %s", exc)
    writer = SummaryWriter(log_dir=str(tensorboard_dir), **writer_kwargs)
    rollout_return_history = _restore_game_return_history(
        config.inference.games or [config.inference.game],
        history_size=config.sampling.return_history_size,
        payload=(
            resume_state.get(_ROLLOUT_RETURN_HISTORY_KEY)
            if resume_state is not None
            else None
        ),
    )
    rollout_arc_score_history = _restore_game_return_history(
        config.inference.games or [config.inference.game],
        history_size=config.sampling.return_history_size,
        payload=(
            resume_state.get(_ROLLOUT_ARC_SCORE_HISTORY_KEY)
            if resume_state is not None
            else None
        ),
    )
    rollout_levels_history = _restore_game_return_history(
        config.inference.games or [config.inference.game],
        history_size=config.sampling.return_history_size,
        payload=(
            resume_state.get(_ROLLOUT_LEVELS_HISTORY_KEY)
            if resume_state is not None
            else None
        ),
    )
    # The collector thread samples window games through process-wide state
    # (training.game_sampling): its call sites only know the window shape,
    # while the trailing return history lives here. Publish the mode/floor
    # and the (resume-restored) rates before collector.start().
    configure_game_sampling(
        mode=config.sampling.game_sampling_mode,
        weight_floor=config.sampling.game_sampling_weight_floor,
        games_per_window=config.sampling.game_sampling_games_per_window,
    )
    publish_game_nonzero_rates(_game_nonzero_return_rates(rollout_return_history))
    sample_budget_state = _initial_sample_budget_state(config, resume_state)
    process_has_completed_optimization_pass = False
    previous_window_received_at: float | None = None
    # The segment deadline, when the launcher supplies one, stops this process
    # at the last iteration boundary that leaves its state durable. A resumed
    # segment inherits the previous segment's timing until it has its own.
    segment_budget = SegmentBudget.from_environment()
    if resume_state is not None:
        previous_metrics = resume_state.get("latest_iteration_metrics") or {}
        segment_budget.seed_iteration_estimate(
            previous_metrics.get("segment_iteration_seconds")
        )
        segment_budget.record_checkpoint(
            previous_metrics.get("segment_checkpoint_seconds")
        )
    if segment_budget.enabled:
        log.info("segment budget: %s", json.dumps(segment_budget.describe()))
    entropy_control_state = _initial_entropy_control_state(config, resume_state)
    memory_logger = build_memory_logger(
        config.memory_profiler,
        run_root,
        base_metadata={"run_id": run_root.name},
        device=config.training.device,
    )
    memory_output_dir = run_root / config.memory_profiler.output_subdir

    def _write_memory_report_if_enabled() -> None:
        if not (
            config.memory_profiler.enabled
            and config.memory_profiler.report.enabled
            and config.memory_profiler.report.format == "html"
            and memory_output_dir.exists()
        ):
            return
        try:
            generate_report(
                memory_output_dir,
                memory_output_dir / "memory_report.html",
            )
        except Exception:
            log.exception("failed to generate CUDA memory report")

    if start_iteration >= config.num_iterations:
        writer.close()
        log.info(
            "run %s already has %d completed iterations and config.num_iterations=%d; nothing to do",
            run_root,
            start_iteration,
            config.num_iterations,
        )
        print(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "iterations": results,
                },
                indent=2,
            )
        )
        return

    official_eval_process: subprocess.Popen | None = None
    official_eval_log_handle = None
    rollout_archiver = (
        RolloutArchiver(run_root)
        if not config.training.cleanup_rollout_artifacts
        and rollouts_root != run_root / "rollouts"
        else None
    )

    def _check_health(*, honor_shutdown: bool = True) -> None:
        if honor_shutdown:
            raise_if_shutdown_requested()
        _check_training_health(
            run_root=run_root,
            collector=collector,
            official_eval_process=official_eval_process,
        )

    def _next_window_ready() -> bool:
        _check_health()
        if full_backend is not None and shutdown_requested():
            return True
        return collector.window_available()

    def _next_window_seconds_remaining() -> float | None:
        _check_health()
        if full_backend is not None and shutdown_requested():
            return 0.0
        return collector.seconds_until_window_ready()

    def _spawn_official_eval_worker() -> subprocess.Popen:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "training.official_eval",
                "--config",
                str(effective_config_path),
                "--run-root",
                str(run_root),
            ],
            cwd=str(repo_root),
            env={**os.environ, "ARC3_SUPERVISING_TRAINER_PID": str(os.getpid())},
            stdout=official_eval_log_handle,
            stderr=subprocess.STDOUT,
        )
        log.info(
            "launched dedicated eval worker (pid=%d, %d games x %d rollouts/game, gpus=%s, cycle=%.1f min)",
            process.pid,
            len(config.official_eval.games),
            config.official_eval.rollouts_per_game,
            ",".join(config.official_eval.server_gpus),
            config.official_eval.cycle_minutes,
        )
        return process

    try:
        _check_health()
        if full_backend is not None:
            trainer = full_backend.create_trainer(
                game_vocab=config.inference.games or [config.inference.game],
                memory_logger=memory_logger,
                resume=full_resume,
                **trainer_identity_kwargs,
            )
            collector.set_next_schedule_index(next_schedule_index)
            # Preserve the full-weight handshake; LoRA (below) and Flash
            # publish adapters and never participate in this handshake.
            if config.official_eval.enabled and config.training.weight_mode == "full_weight":
                from training.official_eval_policy import (
                    clear_full_weight_eval_handshake,
                )

                clear_full_weight_eval_handshake(run_root)
                official_eval_log_handle = (run_root / "official_eval.stdout.log").open(
                    "a"
                )
                official_eval_process = _spawn_official_eval_worker()
            full_backend.prepare_rollouts(
                collector,
                resume=full_resume,
                run_root=run_root,
                official_eval_process=official_eval_process,
            )
        elif resume_state is not None:
            trainer = PPOTrainer(
                config.training,
                **trainer_identity_kwargs,
                game_vocab=config.inference.games or [config.inference.game],
                resume_adapter_dir=resume_adapter_dir,
                resume_state_path=resume_trainer_state_path,
                memory_logger=memory_logger,
            )
            collector.set_next_schedule_index(next_schedule_index)
            if resume_runtime_adapter_dir is not None:
                if not resume_runtime_adapter_dir.exists():
                    log.info(
                        "runtime adapter export %s is gone (previous job's scratch); re-exporting from %s",
                        resume_runtime_adapter_dir,
                        resume_adapter_dir,
                    )
                    resume_runtime_adapter_dir = trainer._save_runtime_adapter(
                        resume_adapter_dir
                    )
                    resume_state = {
                        **resume_state,
                        "runtime_adapter_dir": _run_relative_path(
                            run_root, resume_runtime_adapter_dir
                        ),
                    }
                    _write_json_atomic(_training_run_state_path(run_root), resume_state)
                published_policy_path = resume_runtime_adapter_dir
                _check_health()
                collector.publish_policy(
                    resume_runtime_adapter_dir,
                    policy_iteration=last_completed_iteration,
                )
        else:
            init_adapter_dir = (
                Path(config.training.init_adapter_dir).expanduser()
                if config.training.init_adapter_dir
                else None
            )
            init_trainer_state_path = (
                Path(config.training.init_trainer_state_path).expanduser()
                if config.training.init_trainer_state_path
                else None
            )
            if init_adapter_dir is not None:
                log.info(
                    "initializing fresh run from adapter=%s trainer_state=%s",
                    init_adapter_dir,
                    init_trainer_state_path,
                )
            trainer = PPOTrainer(
                config.training,
                **trainer_identity_kwargs,
                game_vocab=config.inference.games or [config.inference.game],
                resume_adapter_dir=init_adapter_dir,
                resume_state_path=init_trainer_state_path,
                memory_logger=memory_logger,
            )
            # Export the trainer's actual initialized adapter, including its
            # zero-delta LoRA on a cold start. Every rollout uses the same
            # serving path from the outset; warm starts preserve loaded values.
            _check_health()
            init_runtime_adapter_dir = _publish_initial_policy(
                run_root, trainer, resume_metadata={}
            )
            _check_health()
            collector.publish_policy(init_runtime_adapter_dir, policy_iteration=-1)
            published_policy_path = init_runtime_adapter_dir
        if config.training.weight_mode == "flash":
            published_policy_path = full_backend.published_policy_path
        # Both fresh and resumed evaluation require the published LoRA before
        # spawning. Initial policy -1 must not suppress policy 0's first sync.
        if config.official_eval.enabled and config.training.weight_mode != "full_weight":
            official_eval_log_handle = (run_root / "official_eval.stdout.log").open("a")
            official_eval_process = _spawn_official_eval_worker()
        _check_health()
        if full_backend is None:
            collector.start()
        log.info("inference server log: %s", collector.server_log_path)
        log.info(
            "waiting for first rollout window (parallel_rollouts=%s, max_runtime_minutes=%s)",
            config.parallel_rollouts,
            config.inference.max_runtime_minutes,
        )
        for iteration_index in range(start_iteration, config.num_iterations):
            _check_health()
            if full_backend is not None:
                segment_budget.record_checkpoint(
                    getattr(full_backend, "last_checkpoint_seconds", None)
                )
            if segment_budget.should_yield():
                # Draining an older upload alone would lose off-cadence updates.
                # Commit the latest iteration before reporting a successful yield.
                if full_backend is not None:
                    full_backend.checkpoint_latest_iteration()
                    segment_budget.record_checkpoint(
                        full_backend.last_checkpoint_seconds
                    )
                description = segment_budget.describe()
                log.info(
                    "segment budget cannot fit iteration %s/%s; yielding for a "
                    "continuation: %s",
                    iteration_index + 1,
                    config.num_iterations,
                    json.dumps(description),
                )
                _write_json_atomic(
                    run_root / "segment_yield.json",
                    {
                        "next_iteration": iteration_index,
                        "yielded_at": datetime.now(UTC).isoformat(),
                        **description,
                    },
                )
                raise SegmentYield(
                    f"segment budget ends before iteration {iteration_index + 1} "
                    "could finish"
                )
            iteration_started_at = time.perf_counter()
            log.info(
                "starting iteration %s/%s", iteration_index + 1, config.num_iterations
            )
            window = collector.next_window(check_health=_check_health)
            _check_health()
            window_received_at = time.perf_counter()
            # The rollout clock: wall time between consecutive windows
            # becoming available (join included). The first window of a
            # process has no predecessor, so its own collection time stands in.
            rollout_iteration_seconds = (
                window_received_at - previous_window_received_at
                if previous_window_received_at is not None
                else float(window.collection_duration_seconds)
            )
            previous_window_received_at = window_received_at
            log.info(
                "received rollout window %s with %d retained rollouts collected by policy iteration %s (collection=%.2fs launched=%d zero_action=%d)",
                window.window_id,
                len(window.rollouts),
                window.policy_iteration,
                window.collection_duration_seconds,
                window.launched_rollouts,
                window.zero_action_rollouts,
            )
            policy_lag_windows = _policy_lag_windows(
                iteration_index=iteration_index,
                rollout_policy_iteration=window.policy_iteration,
            )
            # Deliberately unrecovered inside the iteration loop: a contract
            # violation reaches the outer finally block, stops workers and
            # servers, and exits the trainer process non-zero.
            _validate_synchronous_rollout_window(
                window,
                policy_lag_windows=policy_lag_windows,
                expected_policy_iteration=(
                    iteration_index - 1
                    if config.training.rollout_schedule == "synchronous"
                    else None
                ),
            )
            # Harness settings the rollouts report (ARC-2: the Duck's prompt
            # injection) are pinned in the checkpoint identity at the first
            # window; one that reports others, after a resume too, stops the
            # run naming the setting (training/rollout_identity.py).
            reported = harness_support.rollout_identity(window.rollouts)
            if reported is not None:
                (trainer if full_backend is None else full_backend).pin_checkpoint_identity(
                    reported
                )
            _archive_server_log_tails(
                run_root=run_root,
                server_log_paths=collector.server_log_paths,
                iteration_index=iteration_index,
                label="window_received",
            )
            # Rollout scalars are plotted on the rollout's OWN axis (the
            # monotonically increasing window/schedule index, persisted across
            # resumes) rather than the trainer iteration -- collection and
            # training run on independent clocks.
            rollout_step = int(window.schedule_index)
            writer.add_scalar(
                "rollout/collection_seconds",
                window.collection_duration_seconds,
                rollout_step,
            )
            writer.add_scalar(
                "rollout/trainer_iteration", float(iteration_index), rollout_step
            )
            _write_scalar_group(
                writer,
                iteration_index=rollout_step,
                scalars=collection_scalars(
                    window.rollouts, window.collection_duration_seconds
                ),
            )
            rollout_metrics = summarize_rollouts(window.rollouts)
            # Window-level metrics above stay raw (an all-ERROR window SHOULD
            # read as an anomaly in TB); the persistent histories exclude
            # infra-ERROR episodes.
            history_rollouts = _history_rollouts(window.rollouts)
            _update_game_return_history(
                rollout_return_history,
                rollouts=history_rollouts,
                history_size=config.sampling.return_history_size,
            )
            _update_game_arc_score_history(
                rollout_arc_score_history,
                rollouts=history_rollouts,
                history_size=config.sampling.return_history_size,
            )
            _update_game_levels_history(
                rollout_levels_history,
                rollouts=history_rollouts,
                history_size=config.sampling.return_history_size,
            )
            # Refresh the sampler's curriculum weights from the just-updated
            # trailing returns (a no-op for balanced runs).
            publish_game_nonzero_rates(
                _game_nonzero_return_rates(rollout_return_history)
            )
            finish_reason_counts = _finish_reason_counts(window.rollouts)
            if "actions_history" not in locals():
                actions_history: list[float] = []
            finish_reason_total = sum(finish_reason_counts.values())
            finish_reason_length_count = finish_reason_counts.get("length", 0)
            finish_reason_length_fraction = finish_reason_length_count / max(
                1, finish_reason_total
            )
            rollout_avg_return_sampled = rollout_metrics.avg_return
            (
                rollout_avg_return,
                rollout_return_history_observed_games,
                rollout_return_history_game_count,
            ) = _mean_game_return_history(
                rollout_return_history,
                config.inference.games or [config.inference.game],
            )
            rollout_avg_arc_score, rollout_arc_score_observed_games, _ = (
                _mean_game_return_history(
                    rollout_arc_score_history,
                    config.inference.games or [config.inference.game],
                    include_unobserved=False,
                )
            )
            rollout_games_nonzero_return_history = _games_nonzero_return_history(
                rollout_return_history,
                config.inference.games or [config.inference.game],
            )
            rollout_observed_games = len(
                _game_id_set(config.inference.games)
                & {
                    str(rollout.task_id).strip()
                    for rollout in window.rollouts
                    if str(rollout.task_id).strip()
                }
            )
            rollout_mean_return_variance = _mean_sampled_game_return_variance(
                window.rollouts
            )
            # Full summarize_rollouts scalar set, mirroring the eval side
            # (official_eval writes as_scalars("eval") wholesale), so rollout
            # and eval expose the same metrics -- incl. return_std/min/max
            # dispersion. Replace avg_return with its history-based value;
            # the history-based ARC score is only plotted for weighted runs.
            tensorboard_rollout_scalars = dict(rollout_metrics.as_scalars())
            # avg_return averages over all configured games; avg_arc_score
            # averages only games with recorded scores, with equal game weight.
            # Short histories are averaged over their actual length, never
            # zero-padded out to return_history_size. The _sampled variant
            # keeps the raw within-window mean observable.
            # rollouts_nonzero_return counts environments in the current
            # window that earned a non-zero return.
            tensorboard_rollout_scalars["rollout/avg_return"] = rollout_avg_return
            if (
                config.sampling.game_sampling_mode == "nonzero_rate"
                and rollout_arc_score_observed_games > 0
            ):
                tensorboard_rollout_scalars["rollout/avg_arc_score"] = (
                    rollout_avg_arc_score
                )
            else:
                # as_scalars() includes the raw ARC score, so remove it too:
                # balanced runs and empty histories have no history score.
                tensorboard_rollout_scalars.pop("rollout/avg_arc_score", None)
            prefix_cache_hit_rates = collector.prefix_cache_hit_rates()
            if prefix_cache_hit_rates:
                tensorboard_rollout_scalars["rollout/prefix_cache_hit_rate"] = sum(
                    prefix_cache_hit_rates.values()
                ) / len(prefix_cache_hit_rates)
                for server_index, rate in prefix_cache_hit_rates.items():
                    tensorboard_rollout_scalars[
                        f"rollout/prefix_cache_hit_rate_server_{server_index}"
                    ] = rate
            windowed_cache_rates = collector.windowed_prefix_cache_hit_rates()
            if windowed_cache_rates:
                tensorboard_rollout_scalars["rollout/prefix_cache_hit_rate_window"] = (
                    sum(windowed_cache_rates.values()) / len(windowed_cache_rates)
                )
            # Plain mean over this window's sampled rollouts. Weighted runs
            # also plot the observed games' history in rollout/avg_arc_score.
            tensorboard_rollout_scalars["rollout/avg_arc_score_sampled"] = (
                rollout_metrics.avg_arc_score
            )
            tensorboard_rollout_scalars.update(
                _rollout_status_scalars(window.rollouts)
            )
            tensorboard_rollout_scalars[
                "rollout/dropped_incomplete_tail_model_turns"
            ] = float(
                sum(
                    int(
                        (rollout.metadata or {}).get(
                            "dropped_incomplete_tail_model_turns", 0
                        )
                        or 0
                    )
                    for rollout in window.rollouts
                )
            )
            tensorboard_rollout_scalars["rollout/finish_reason_length_fraction"] = (
                finish_reason_length_fraction
            )
            format_break_fraction = _format_break_fraction(window.rollouts)
            tensorboard_rollout_scalars["rollout/format_break_fraction"] = (
                format_break_fraction
            )
            actions_history.append(rollout_metrics.avg_actions_per_rollout)
            del actions_history[: -max(1, int(config.training.collapse_watch_windows))]
            collapse_reasons = _collapse_alarm_reasons(
                format_break_fraction=format_break_fraction,
                format_break_fraction_alarm=config.training.collapse_format_break_fraction_alarm,
                actions_per_rollout=rollout_metrics.avg_actions_per_rollout,
                actions_history=actions_history,
                actions_drop_fraction=config.training.collapse_actions_drop_fraction,
                watch_windows=config.training.collapse_watch_windows,
            )
            tensorboard_rollout_scalars["train/collapse_alarm"] = (
                1.0 if collapse_reasons else 0.0
            )
            if collapse_reasons:
                message = (
                    f"collapse watch: iteration {iteration_index + 1}: "
                    + "; ".join(collapse_reasons)
                )
                if config.training.stop_on_collapse:
                    writer.flush()
                    raise RuntimeError("FATAL: " + message)
                log.warning(message)
            for name, value in tensorboard_rollout_scalars.items():
                writer.add_scalar(name, value, rollout_step)
            if config.metrics.log_per_game:
                _write_per_game_metric_group(
                    writer,
                    prefix="rollout_by_game",
                    iteration_index=rollout_step,
                    rollouts=window.rollouts,
                )
            writer.flush()
            _log_rollout_examples(
                window.rollouts,
                count=config.training.debug_rollout_examples_per_window,
            )
            # An infra-ERROR episode fails the run before the optimizer sees
            # the window: a server dying mid-episode truncates it with a
            # bogus terminal (V=0 at a healthy mid-episode state), and a
            # crashed rollout is a bug to fix, never data to skip. Loud and
            # named, and never chained by auto-relaunch. Zero-action
            # TIMEOUTs stay -- hitting the episode budget IS a legitimate
            # terminal.
            writer.flush()
            _require_no_infra_error_rollouts(
                window.rollouts, iteration_index=iteration_index
            )
            train_rollouts = list(window.rollouts)
            log.info(
                "starting optimization for iteration %s on %d rollout trajectories",
                iteration_index + 1,
                len(train_rollouts),
            )
            writer.flush()
            _check_health()
            train_result = (full_backend or trainer).train_window(
                train_rollouts,
                output_dir=checkpoints_root,
                iteration_index=iteration_index,
                sample_budget_fraction=float(
                    sample_budget_state.get("current_fraction", 1.0)
                ),
                policy_lag_windows=policy_lag_windows,
                entropy_coef=(
                    float(entropy_control_state["applied_coef"])
                    if config.training.entropy_control.enabled
                    else None
                ),
                next_window_ready=_next_window_ready,
                next_window_seconds_remaining=_next_window_seconds_remaining,
                check_health=_check_health,
            )
            # A full-weight command defers TERM until every rank finishes.
            # Commit that update (including the first one) before unwinding.
            _check_health(honor_shutdown=full_backend is None)
            entropy_control_state = _update_entropy_control_state(
                config=config,
                state=entropy_control_state,
                train_result=train_result,
            )
            published_policy_path = train_result.runtime_adapter_dir
            publication_wait_seconds = 0.0
            _check_health(honor_shutdown=full_backend is None)
            if full_backend is None:
                collector.publish_policy(
                    published_policy_path, policy_iteration=iteration_index
                )
            elif not shutdown_requested():
                full_backend.publish_after_train(
                    collector,
                    train_result=train_result,
                    iteration_index=iteration_index,
                    should_publish=iteration_index + 1 < config.num_iterations,
                )
                publication_wait_seconds = float(
                    getattr(full_backend, "publication_wait_seconds", 0.0)
                )
            # The trainer clock: everything between receiving the window and
            # the new adapter being live on the servers, minus the extra
            # cycles that only fill idle time until the next window and the
            # idle wait for a safe full-weight publication boundary.
            trainer_iteration_seconds = max(
                0.0,
                time.perf_counter()
                - window_received_at
                - publication_wait_seconds
                - float(getattr(train_result, "extra_phase_seconds", 0.0) or 0.0),
            )
            sample_budget_state = _update_sample_budget_state(
                config=config,
                state=sample_budget_state,
                train_result=train_result,
                rollout_iteration_seconds=rollout_iteration_seconds,
                trainer_iteration_seconds=trainer_iteration_seconds,
                policy_lag_windows=policy_lag_windows,
                is_first_process_optimization_pass=(
                    not process_has_completed_optimization_pass
                    and int(getattr(train_result, "optimizer_steps", 0)) > 0
                ),
            )
            if int(getattr(train_result, "optimizer_steps", 0)) > 0:
                process_has_completed_optimization_pass = True
            log.info(
                "iteration timing: rollout=%.1fs trainer=%.1fs (optimizer=%.1fs extra_cycles=%.1fs) ratio=%s target=%.2f -> sample_fraction %.3f (%s)",
                rollout_iteration_seconds,
                trainer_iteration_seconds,
                float(getattr(train_result, "total_train_window_seconds", 0.0) or 0.0),
                float(getattr(train_result, "extra_phase_seconds", 0.0) or 0.0),
                (
                    "n/a"
                    if sample_budget_state["last_observed_train_to_collection_ratio"]
                    is None
                    else f"{sample_budget_state['last_observed_train_to_collection_ratio']:.3f}"
                ),
                config.training.sample_budget.target_train_to_collection_ratio,
                sample_budget_state["next_fraction"],
                sample_budget_state["last_update_reason"],
            )
            writer.add_scalar(
                "rollout/iteration_seconds", rollout_iteration_seconds, rollout_step
            )
            writer.add_scalar(
                "train/iteration_seconds", trainer_iteration_seconds, iteration_index
            )

            writer.add_scalar(
                "train/publication_wait_seconds",
                publication_wait_seconds,
                iteration_index,
            )

            # Sample preparation can read source artifacts until PPO
            # returns; training has finished here, so the window dir is
            # safe to remove.
            if config.training.cleanup_rollout_artifacts:
                _cleanup_window_artifacts(window.window_dir)
            elif rollout_archiver is not None:
                rollout_archiver.submit(window.window_dir)
            else:
                log.info(
                    "retaining rollout artifacts for debugging at %s", window.window_dir
                )

            iteration_metrics = {
                "iteration": iteration_index,
                "window_id": window.window_id,
                "schedule_index": window.schedule_index,
                "next_schedule_index": window.schedule_index + 1,
                "rollout_policy_iteration": window.policy_iteration,
                "policy_lag_windows": policy_lag_windows,
                "retained_rollouts": len(window.rollouts),
                "collection_duration_seconds": window.collection_duration_seconds,
                "rollout_iteration_seconds": rollout_iteration_seconds,
                "trainer_iteration_seconds": trainer_iteration_seconds,
                "publication_wait_seconds": publication_wait_seconds,
                "launched_rollouts": window.launched_rollouts,
                "zero_action_rollouts": window.zero_action_rollouts,
                "failed_processes": window.failed_processes,
                "artifact_rollouts": window.artifact_rollouts,
                "crashed_rollouts": window.crashed_rollouts,
                "avg_return": rollout_avg_return,
                "avg_return_sampled": rollout_avg_return_sampled,
                "avg_levels_completed": rollout_metrics.avg_levels_completed,
                "max_levels_completed": rollout_metrics.max_levels_completed,
                "min_return": rollout_metrics.min_return,
                "max_return": rollout_metrics.max_return,
                "avg_rollout_length": rollout_metrics.avg_rollout_length,
                "avg_actions_per_rollout": rollout_metrics.avg_actions_per_rollout,
                "avg_step_reward": rollout_metrics.avg_step_reward,
                "done_fraction": rollout_metrics.done_fraction,
                "observed_game_count": rollout_observed_games,
                "games_nonzero_return": rollout_metrics.games_nonzero_return,
                "games_nonzero_return_history": rollout_games_nonzero_return_history,
                "avg_return_observed_game_count": rollout_return_history_observed_games,
                "avg_return_configured_game_count": rollout_return_history_game_count,
                "mean_return_variance": rollout_mean_return_variance,
                "avg_generation_rounds_per_rollout": (
                    rollout_metrics.avg_generation_rounds_per_rollout
                ),
                "avg_tokens_per_generation": rollout_metrics.avg_tokens_per_generation,
                "finish_reason_counts": finish_reason_counts,
                "finish_reason_length_count": finish_reason_length_count,
                "finish_reason_total_count": finish_reason_total,
                "finish_reason_length_fraction": finish_reason_length_fraction,
                "avg_void_attempts_per_rollout": (
                    rollout_metrics.avg_void_attempts_per_rollout
                ),
                "void_attempt_fraction": rollout_metrics.void_attempt_fraction,
                # ARC-2 malformed calls (empty replies among them, counted apart)
                # and the share of episodes charged the penalty cap.
                "avg_malformed_calls": rollout_metrics.avg_malformed_calls,
                "avg_empty_replies_per_rollout": (
                    rollout_metrics.avg_empty_replies_per_rollout
                ),
                "penalty_cap_fraction": rollout_metrics.penalty_cap_fraction,
                "sample_count": train_result.sample_count,
                "raw_sample_count": train_result.raw_sample_count,
                "eligible_sample_count": train_result.eligible_sample_count,
                "filtered_zero_signal_sample_count": train_result.filtered_zero_signal_sample_count,
                "subsampled_sample_count": train_result.subsampled_sample_count,
                "sample_budget_fraction": train_result.sample_budget_fraction,
                "sample_budget_step_multiple": train_result.sample_budget_step_multiple,
                "sample_budget_next_fraction": sample_budget_state["next_fraction"],
                "sample_budget_update_reason": sample_budget_state[
                    "last_update_reason"
                ],
                "train_to_collection_ratio": sample_budget_state[
                    "last_observed_train_to_collection_ratio"
                ],
                "trainable_parameter_count": train_result.trainable_parameter_count,
                "mean_advantage": train_result.mean_advantage,
                "std_advantage": train_result.std_advantage,
                "mean_return_to_go": train_result.mean_return_to_go,
                "baseline_loss": train_result.baseline_loss,
                "baseline_update_steps": train_result.baseline_update_steps,
                "baseline_mean_value": train_result.baseline_mean_value,
                "baseline_mean_target": train_result.baseline_mean_target,
                "baseline_explained_variance": train_result.baseline_explained_variance,
                "baseline_post_update_mean_value": train_result.baseline_post_update_mean_value,
                "baseline_post_update_mean_absolute_error": train_result.baseline_post_update_mean_absolute_error,
                "baseline_post_update_explained_variance": train_result.baseline_post_update_explained_variance,
                "baseline_mean_raw_advantage": train_result.baseline_mean_raw_advantage,
                "baseline_std_raw_advantage": train_result.baseline_std_raw_advantage,
                "baseline_update_seconds": train_result.baseline_update_seconds,
                "optimizer_skipped_reason": train_result.optimizer_skipped_reason,
                "mean_advantage_by_level": {
                    str(level): mean_advantage
                    for level, mean_advantage in train_result.mean_advantage_by_level.items()
                },
                "mean_old_logprob": train_result.mean_old_logprob,
                "rollout_logprob_parity_token_count": train_result.rollout_logprob_parity_token_count,
                "rollout_logprob_parity_mean_abs": train_result.rollout_logprob_parity_mean_abs,
                "rollout_logprob_parity_p95_abs": train_result.rollout_logprob_parity_p95_abs,
                "rollout_logprob_parity_max_abs": train_result.rollout_logprob_parity_max_abs,
                "rollout_logprob_parity_outlier_count": train_result.rollout_logprob_parity_outlier_count,
                "rollout_logprob_parity_outlier_fraction": train_result.rollout_logprob_parity_outlier_fraction,
                "rollout_logprob_parity_signed_mean": train_result.rollout_logprob_parity_signed_mean,
                "final_loss": train_result.final_loss,
                "mean_entropy": train_result.mean_entropy,
                "entropy_loss": train_result.entropy_loss,
                "entropy_coef": train_result.entropy_coef_used,
                "entropy_to_policy_grad_ratio": (
                    train_result.entropy_to_policy_grad_ratio
                ),
                "final_learning_rate": train_result.final_learning_rate,
                "optimizer_steps": train_result.optimizer_steps,
                "mean_ratio": train_result.mean_ratio,
                "mean_clip_fraction": train_result.mean_clip_fraction,
                "mean_grad_norm": train_result.mean_grad_norm,
                "max_grad_norm": train_result.max_grad_norm,
                "transition_reward_nonzero_fraction_raw": train_result.transition_reward_nonzero_fraction_raw,
                "transition_reward_positive_fraction_raw": train_result.transition_reward_positive_fraction_raw,
                "transition_reward_sum_raw": train_result.transition_reward_sum_raw,
                "transition_reward_nonzero_fraction_kept": train_result.transition_reward_nonzero_fraction_kept,
                "transition_reward_positive_fraction_kept": train_result.transition_reward_positive_fraction_kept,
                "transition_reward_sum_kept": train_result.transition_reward_sum_kept,
                "return_to_go_nonzero_fraction_kept": train_result.return_to_go_nonzero_fraction_kept,
                "return_to_go_positive_fraction_kept": train_result.return_to_go_positive_fraction_kept,
                "advantage_nonzero_fraction_kept": train_result.advantage_nonzero_fraction_kept,
                "advantage_positive_fraction_kept": train_result.advantage_positive_fraction_kept,
                "positive_transition_reward_sample_count_raw": train_result.positive_transition_reward_sample_count_raw,
                "positive_transition_reward_sample_count_kept": train_result.positive_transition_reward_sample_count_kept,
                "trainable_param_norm_before": train_result.trainable_param_norm_before,
                "trainable_param_norm_after": train_result.trainable_param_norm_after,
                "advantage_group_mean_reward": train_result.advantage_group_mean_reward,
                "advantage_group_std_reward": train_result.advantage_group_std_reward,
                "total_completion_tokens": train_result.total_completion_tokens,
                "total_env_steps": train_result.total_env_steps,
                "avg_actions_per_completed_level": train_result.avg_actions_per_completed_level,
                "avg_tokens_per_rollout": train_result.avg_tokens_per_rollout,
                "avg_tokens_per_step": train_result.avg_tokens_per_step,
                "avg_tokens_per_model_turn": train_result.avg_tokens_per_model_turn,
                "avg_prompt_tokens_per_sample": train_result.avg_prompt_tokens_per_sample,
                "avg_history_messages_per_sample": train_result.avg_history_messages_per_sample,
                "sample_build_seconds": train_result.sample_build_seconds,
                "optimization_seconds": train_result.optimization_seconds,
                "policy_export_seconds": train_result.policy_export_seconds,
                "train_window_seconds": train_result.total_train_window_seconds,
                "cuda_trainer_model_allocated_gb": train_result.cuda_trainer_model_allocated_gb,
                "cuda_optimization_peak_delta_gb": train_result.cuda_optimization_peak_delta_gb,
                "cuda_optimization_persistent_delta_gb": train_result.cuda_optimization_persistent_delta_gb,
                "adapter_dir": (
                    str(train_result.adapter_dir)
                    if train_result.adapter_dir is not None
                    else None
                ),
                "runtime_adapter_dir": (
                    str(train_result.runtime_adapter_dir)
                    if train_result.runtime_adapter_dir is not None
                    else None
                ),
                "trainer_state_path": str(train_result.trainer_state_path),
            }
            segment_budget.record_iteration(time.perf_counter() - iteration_started_at)
            if full_backend is not None:
                segment_budget.record_checkpoint(
                    getattr(full_backend, "last_checkpoint_seconds", None)
                )
            iteration_metrics["segment_iteration_seconds"] = (
                segment_budget.iteration_seconds[-1]
            )
            iteration_metrics["segment_checkpoint_seconds"] = (
                segment_budget.checkpoint_seconds
            )
            results.append(iteration_metrics)
            _write_training_run_state(
                run_root=run_root,
                config_path=config_path,
                config=config,
                iteration_metrics=iteration_metrics,
                train_result=train_result,
                trainer=trainer,
                rollout_return_history=rollout_return_history,
                rollout_arc_score_history=rollout_arc_score_history,
                rollout_levels_history=rollout_levels_history,
                sample_budget_state=sample_budget_state,
                entropy_control_state=entropy_control_state,
                full_backend=full_backend,
            )
            _write_memory_report_if_enabled()

            _write_training_tensorboard_scalars(
                writer,
                train_result=train_result,
                iteration_index=iteration_index,
            )
            _write_expert_load(
                writer,
                summary=getattr(train_result, "expert_load", None) or {},
                iteration_index=iteration_index,
                rollout_step=rollout_step,
                num_iterations=config.num_iterations,
            )
            writer.add_scalar(
                "train/sample_budget_next_fraction",
                float(sample_budget_state["next_fraction"]),
                iteration_index,
            )
            observed_train_to_collection = sample_budget_state[
                "last_observed_train_to_collection_ratio"
            ]
            if observed_train_to_collection is not None:
                writer.add_scalar(
                    "train/train_to_collection_ratio",
                    float(observed_train_to_collection),
                    iteration_index,
                )
            writer.flush()
            _archive_server_log_tails(
                run_root=run_root,
                server_log_paths=collector.server_log_paths,
                iteration_index=iteration_index,
                label="post_train",
            )

            if (
                int(getattr(train_result, "optimizer_steps", 0)) > 0
                and abs(float(train_result.mean_ratio) - 1.0) > _POLICY_DRIFT_RATIO_BAND
            ):
                log.warning(
                    "policy-drift watch: mean_ratio=%.4f is outside 1.0+/-%.2f on a stepped "
                    "window -- the serving and training weights may be diverging below the "
                    "trust-region gate (check adapter sync and rollout_logprob_parity_*)",
                    float(train_result.mean_ratio),
                    _POLICY_DRIFT_RATIO_BAND,
                )
            log.info(
                "completed iteration %s/%s avg_return=%.4f avg_return_sampled=%.4f avg_levels_completed=%.2f max_levels_completed=%.2f loss=%.4f lr=%.3e mean_ratio=%.4f clip_frac=%.4f mean_grad_norm=%.4f transition_reward_positive_raw=%.3f return_to_go_positive_kept=%.3f advantage_positive_kept=%.3f samples=%s eligible_samples=%s subsampled_samples=%s budget_fraction=%.3f next_budget_fraction=%.3f budget_reason=%s finish_reason_length=%s/%s avg_tokens_per_rollout=%.1f avg_tokens_per_step=%.1f avg_history_messages=%.2f rollout_policy_iteration=%s policy_lag=%s collection=%.2fs sample_build=%.2fs optimize=%.2fs export=%.2fs total_train=%.2fs optimizer_skipped_reason=%s trainer_adapter=%s runtime_adapter=%s published_policy=%s",
                iteration_index + 1,
                config.num_iterations,
                rollout_avg_return,
                rollout_avg_return_sampled,
                rollout_metrics.avg_levels_completed,
                rollout_metrics.max_levels_completed,
                train_result.final_loss,
                train_result.final_learning_rate,
                train_result.mean_ratio,
                train_result.mean_clip_fraction,
                train_result.mean_grad_norm,
                train_result.transition_reward_positive_fraction_raw,
                train_result.return_to_go_positive_fraction_kept,
                train_result.advantage_positive_fraction_kept,
                train_result.sample_count,
                train_result.eligible_sample_count,
                train_result.subsampled_sample_count,
                train_result.sample_budget_fraction,
                sample_budget_state["next_fraction"],
                sample_budget_state["last_update_reason"],
                finish_reason_length_count,
                finish_reason_total,
                train_result.avg_tokens_per_rollout,
                train_result.avg_tokens_per_step,
                train_result.avg_history_messages_per_sample,
                window.policy_iteration,
                policy_lag_windows,
                window.collection_duration_seconds,
                train_result.sample_build_seconds,
                train_result.optimization_seconds,
                train_result.policy_export_seconds,
                train_result.total_train_window_seconds,
                train_result.optimizer_skipped_reason,
                train_result.adapter_dir,
                train_result.runtime_adapter_dir,
                published_policy_path,
            )
        _check_health()
    except TrainingShutdown:
        log.info("scheduled training shutdown; staging completed artifacts")
        raise
    except BaseException as error:
        if shutdown_requested() and is_teardown_error(error):
            # The serving hosts were stopped by the same scheduled shutdown;
            # an in-flight generation failing against them is that shutdown.
            log.info(
                "scheduled training shutdown: upstream servers stopped during "
                "collection (%s); staging completed artifacts",
                error,
            )
            raise TrainingShutdown(
                "received SIGTERM; upstream servers stopped"
            ) from error
        # Log the root cause BEFORE teardown: the finally below stops servers
        # and joins workers, any of which can block (a desynced DDP worker
        # wedged job 510826 in a collective forever) and would otherwise bury
        # the actual failure without a trace.
        log.exception(
            "FATAL: training failed; automatic relaunch is disabled; tearing down"
        )
        record_run_failure(run_root, error)
        raise
    finally:
        if official_eval_process is not None and official_eval_process.poll() is None:
            official_eval_process.terminate()
            try:
                official_eval_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                official_eval_process.kill()
        if official_eval_log_handle is not None:
            official_eval_log_handle.close()
        collector.stop()
        # trainer is unbound if PPOTrainer construction itself failed.
        if full_backend is not None:
            # A scheduled shutdown or segment yield leaves the peers serving
            # and possibly a checkpoint upload in flight: drain both.
            error_type = sys.exc_info()[0]
            full_backend.shutdown(
                graceful=error_type is None or issubclass(error_type, TrainingShutdown)
            )
        elif "trainer" in locals():
            trainer.shutdown_workers()
        writer.close()
        _write_memory_report_if_enabled()
        stage_out_error_is_secondary = sys.exc_info()[0] is not None
        try:
            stage_out_rollout_scratch(
                run_root=run_root,
                rollout_roots=[rollouts_root, official_eval_rollouts_root],
                server_log_paths=[
                    *collector.server_log_paths,
                    *official_eval_server_log_paths,
                ],
                cleanup_rollout_artifacts=config.training.cleanup_rollout_artifacts,
                archiver=rollout_archiver,
            )
        except Exception:  # noqa: BLE001
            log.exception("failed to stage rollout artifacts or server logs")
            if not stage_out_error_is_secondary:
                raise
        _remove_empty_directory(rollouts_root)

    # A worker can fail during final teardown after the last loop check.
    # The eval process was intentionally terminated above, so check only
    # the shared failure record rather than treating that exit as a crash.
    raise_if_run_failed(run_root)
    log.info("run complete: %s", run_root)
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "iterations": results,
            },
            indent=2,
        )
    )


def _cleanup_window_artifacts(window_dir: Path) -> None:
    if not window_dir.exists():
        return
    shutil.rmtree(window_dir, ignore_errors=True)


def _remove_empty_directory(path: Path) -> None:
    try:
        if path.exists() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC3 training pipeline")
    parser.add_argument(
        "--config", required=True, help="Path to the training config JSON file."
    )
    parser.add_argument(
        "--run-dir",
        help=(
            "Use this run directory instead of creating a timestamped one. "
            "If it contains training_state.json, training resumes from its latest checkpoint."
        ),
    )
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.run_dir:
        os.environ["ARC3_RUN_DIR"] = args.run_dir
    with training_sigterm_handler():
        try:
            run_training(args.config)
        except SegmentYield:
            sys.exit(SEGMENT_YIELD_EXIT_CODE)
        except TrainingShutdown:
            return
        except Exception:
            log.exception("training failed")
            raise


if __name__ == "__main__":
    main()
