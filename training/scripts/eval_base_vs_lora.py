"""Paired evaluation driver: base model vs a trained LoRA on the official games.

Runs INSIDE the platypus venv on the eval job's node (invoked as
``<platypus_venv>/bin/python scripts/eval_base_vs_lora.py ...`` by
scripts/eval_base_vs_lora.sbatch) against TWO pre-launched vLLM servers:

- the **base** arm: the plain base model (served under its model id),
- the **lora** arm: base + the trained adapter, hot-loaded by THIS driver
  under a static alias via vLLM's runtime-LoRA endpoint (the same
  ``POST {base_url}/adapters/load`` contract the trainer uses in
  ``training.inference.sync_collector_runtime_adapter``).

Episodes are **per-episode benchmarks**, not rollout windows: each episode is
one platypus/TAAF benchmark of 1 game x 1 pass with
``RunConfig.rng_seed_per_game={game: seed}`` and a per-arm runtime cap,
mirroring how scripts/platypus_rollout_worker.py invokes the harness (import
paths, DuckAgent knobs, sampling override). Each episode uses a capture
gateway that enforces native sampling and exact continuation against the
trainer-owned engine. This is what
allows several seeds of the same game to run concurrently (a single TAAF
benchmark forbids duplicate game_ids) and gives continuous backfill: each
server keeps ``--concurrency-per-server`` episodes in flight and starts the
next queued (game, seed) the moment one finishes — no window synchronization.

Every episode appends one JSON line to ``<eval_dir>/episodes_<arm>.jsonl``
carrying the game/seed/arm identity, the final outcome, and a per-step trace:
cumulative generated (completion) tokens, the official ARC score after the
step, and elapsed game wallclock. Token counts come from the TAAF trace
(``traces/<run_id>/steps.jsonl`` — ``output_token_ids`` per step, the same
field scripts/platypus_rollout_worker.py distills into ``output_len``); the
score progression is reconstructed from ``benchmark.json``'s
``actions_per_level`` / ``base_actions_per_level`` / step ``action_index``
exactly the way training/platypus_rollout.py builds per-step rewards (its
helpers are imported, not re-implemented). The ARC score only changes when a
level completes, so the per-step score trace is a step function at level
completions — that is the finest score granularity the harness contract has.

The analysis (scripts/eval_base_vs_lora_analysis.py) evaluates "score at
token budget B" by walking each trace; post-hoc truncation is exact because
capping generation cannot change the history before the cap. The runtime
caps here are wall-clock SAFETY BACKSTOPS only: ``RunConfig`` at the pinned
harness commit exposes no per-episode generated-token budget (its only
token knob is the per-LEVEL ``max_tokens_per_level``, which the validated
eval protocol pins to None by owner ruling, 6aad3b0f; training inherits the
Duck preset's value instead, and each episode records what it played under),
so the generated-token budget is enforced exactly by the analysis, not at
runtime.

Re-running the driver with the same ``--eval-dir`` resumes: episodes already
recorded with status "ok" are skipped, everything else is re-run (their stale
artifact dirs are wiped first).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# The repo's live packages must shadow whatever arc3-training snapshot the
# (lock-keyed, content-reused) venv holds — the same reason the smoke sbatch
# runs the trainer as `python -m training...` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests  # noqa: E402  (after the sys.path fix, like every import below)

from harness_support.arc3.scoring import (  # noqa: E402
    _action_index_ranges,
    _cumulative_arc_scores,
    _level_boundaries,
)

_LORA_ALIAS_DEFAULT = "policy"
_RUNTIME_LORA_REQUEST_TIMEOUT_SECONDS = 180
_DEFAULT_CONCURRENCY_PER_SERVER = 50


# --------------------------------------------------------------------------
# arm configuration
# --------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class ArmConfig:
    arm: str  # "base" | "lora"
    base_url: str  # native engine origin: http://127.0.0.1:PORT
    served_model: str  # base model id, or the runtime LoRA alias
    cap_seconds: float  # per-episode runtime budget (RunConfig.max_runtime_s_per_game)
    max_completion_tokens: int
    reasoning_effort: str | None
    context_window_tokens: int | None
    compact_to_tokens: int | None
    sampling: dict[str, Any] | None
    api_key: str | None
    max_tokens_per_game: int | None = None
    tokenizer_path: str | None = None


def _sampling(arm: ArmConfig) -> dict[str, Any]:
    # Keep the deployment protocol's nucleus/temperature, while enforcing
    # the same neutral penalties as every native training/evaluation draw.
    return (
        arm.sampling
        if arm.sampling is not None
        else {"temperature": 0.6, "top_p": 0.8899009607194754, "top_k": 194}
    )


def _build_episode_benchmark(*, game: str, seed: int, arm: ArmConfig, job_dir: Path):
    """One-game one-pass benchmark, invoked exactly like the training rollout
    worker builds its window benchmark (scripts/platypus_rollout_worker.py
    ``_build_benchmark``); the episode attaches its gateway before spawning."""
    from platypus import RunConfig
    from duck.agent import DuckAgent
    from training_backend import configure_training_agent
    from ullm.backends.engine_proxy import EngineProxy
    from training_backend import (
        qwen_engine_factory,
        training_backend_factory,
        training_sampling,
    )
    from serving_contract import (
        ContractLLMAgentSolver,
        install_contract_backend,
    )
    from taaf.benchmark import Benchmark
    from taaf.game_api import GameAPI

    run_config = RunConfig(
        process_mode="subprocess",
        min_request_gap_s=0.0,
        max_concurrent=1,
        on_agent_error="raise",
        max_runtime_s_per_game=float(arm.cap_seconds),
        rng_seed=int(seed),
        # The layout seed contract: both arms play the identical (game, seed)
        # instance, so every pair in the analysis is a true paired sample.
        rng_seed_per_game={str(game): int(seed)},
    )
    overrides: dict[str, Any] = {"file_tools": "none"}
    # Text-only, always (owner decision mirrored from the training stack):
    # the servers are launched with every modality limited to 0.
    overrides["multimodal_context"] = ""
    if arm.reasoning_effort is not None:
        overrides["reasoning_effort"] = arm.reasoning_effort
    if arm.context_window_tokens is not None:
        overrides["context_window_tokens"] = int(arm.context_window_tokens)
    if arm.compact_to_tokens is not None:
        overrides["compact_to_tokens"] = int(arm.compact_to_tokens)
    if arm.max_tokens_per_game is not None:
        # The duck's own per-game generated-token budget: the episode ends
        # itself (GiveUpError) at the budget, so wall-clock is a backstop
        # only.
        overrides["max_tokens_per_game"] = int(arm.max_tokens_per_game)
    # The per-game budget is the only token limiter in eval (owner ruling,
    # 6aad3b0f): the Duck's per-level give-up is disabled here, unlike
    # training, which inherits the preset's value. Each episode records it.
    overrides["max_tokens_per_level"] = None
    overrides["malformed_retries"] = 0
    # The remote server owns image processor limits; older harness pins have no such control.
    if hasattr(DuckAgent, "derive_max_pixels"):
        overrides["derive_max_pixels"] = False

    engine = "qwen3.8-27b"
    agent = DuckAgent.build(
        engine=engine,
        environment="slurm-b200",
        backend_factory=qwen_engine_factory(
            EngineProxy(
                base_url=arm.base_url,
                model=arm.served_model,
                api_key=arm.api_key or "",
                tokenizer_path=arm.tokenizer_path,
            ),
            engine=engine,
        ),
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        reply_cap_tokens=int(arm.max_completion_tokens),
        **overrides,
    )
    agent.run_config = run_config
    configure_training_agent(
        agent,
        backend_factory=training_backend_factory(
            source_factory=agent.llm_config.backend_factory
        ),
        sampling=training_sampling(_sampling(arm)),
        max_concurrent=1,
    )
    agent.prepare_for_backend()
    install_contract_backend(agent)
    job_dir.mkdir(parents=True, exist_ok=True)
    return Benchmark(
        games=[GameAPI(env_name=str(game))],
        solver=ContractLLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=job_dir,
        trace_root=job_dir / "traces",
    )


def _episode_gateway(benchmark, arm: ArmConfig):
    """Pin one native engine and attach its contract before spawning the agent."""
    from training.context_contract import resolve_context_contract
    from training.inference import _validated_adapter_identity
    from training.rollout_gateway import CaptureGateway, GatewayError, SamplingPolicy
    from training.vllm_runtime import native_url, validate_server_runtime

    agent = benchmark.solver.agent
    try:
        validate_server_runtime(arm.base_url, headers=_auth_headers(arm.api_key))
        response = requests.get(
            native_url(arm.base_url, "/metadata"),
            params={"model": arm.served_model},
            headers=_auth_headers(arm.api_key),
            timeout=15,
        )
        response.raise_for_status()
        metadata = response.json()
        adapter = _validated_adapter_identity(metadata.get("adapter"), arm.served_model)
        context = resolve_context_contract(
            window=agent.context_window_tokens,
            reply=arm.max_completion_tokens,
            target=agent._compact_target_tokens(),
            margin=agent.safety_margin_tokens,
        )
        if (
            type(metadata.get("max_model_len")) is not int
            or metadata["max_model_len"] < context.max_served_sequence_tokens
        ):
            raise ValueError("evaluation context exceeds the native engine's context")
        sampling = _sampling(arm)
        policy = SamplingPolicy(
            model=arm.served_model,
            expected_adapter=adapter,
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling.get("top_k") or -1),
            context=context,
            max_runtime_seconds=arm.cap_seconds,
        )
        gateway = CaptureGateway(upstream_base_url=arm.base_url, port=0)
        gateway.set_policy(policy)
        try:
            gateway.start()
            factory = agent.llm_config.backend_factory
            agent.llm_config.backend_factory = dataclasses.replace(
                factory,
                engine=dataclasses.replace(factory.engine, base_url=gateway.base_url),
            )
        except BaseException:
            gateway.stop()
            raise
        return gateway
    except Exception as error:
        raise GatewayError(
            f"evaluation serving contract setup failed: {error}"
        ) from error


async def _checked_benchmark(benchmark, gateway) -> None:
    """A swallowed HTTP rejection still aborts the evaluation immediately."""
    task = asyncio.create_task(benchmark.run(soft_end_time=None))
    try:
        while not task.done():
            await asyncio.wait((task,), timeout=1.0)
            gateway.raise_if_invalid()
        await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --------------------------------------------------------------------------
# episode artifact parsing (benchmark.json + traces/<run_id>/steps.jsonl)
# --------------------------------------------------------------------------


def _read_trace_steps(trace_dir: Path) -> list[dict[str, Any]]:
    steps_path = trace_dir / "steps.jsonl"
    if not steps_path.exists():
        # A run that crashed (or gave up) before its first model call leaves
        # no trace folder; that is a legitimate zero-step episode (same
        # tolerance as scripts/platypus_rollout_worker._step_reports).
        return []
    steps: list[dict[str, Any]] = []
    with steps_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


def _parse_episode(
    job_dir: Path,
    *,
    game: str,
    seed: int,
    arm: ArmConfig,
    started_at: float,
    finished_at: float,
    max_tokens_per_level: list[int] | None,
) -> dict[str, Any]:
    benchmark = json.loads((job_dir / "benchmark.json").read_text(encoding="utf-8"))
    runs = benchmark.get("game_runs") or []
    if len(runs) != 1:
        raise RuntimeError(
            f"{job_dir}/benchmark.json holds {len(runs)} game_runs; a "
            "per-episode benchmark must hold exactly one"
        )
    run = runs[0]
    run_id = str(run["run_id"])
    history = run.get("history") or []
    total_actions = len(history)
    actions_per_level = [int(count) for count in run.get("actions_per_level") or []]
    base_actions_per_level = [
        int(count) for count in run.get("base_actions_per_level") or []
    ]
    levels_completed = int(run.get("levels_completed", 0))
    number_of_levels = int(run.get("number_of_levels", 0))
    action_wallclock = [float(entry.get("wallclock_seconds", 0.0)) for entry in history]

    # Official cumulative ARC score after each completed level, and the action
    # index (0-based, cumulative) that completed it — the exact helpers the
    # trainer uses to attribute per-step rewards (training/platypus_rollout.py).
    cumulative_scores = _cumulative_arc_scores(
        levels_completed, actions_per_level, base_actions_per_level, number_of_levels
    )
    final_score = cumulative_scores[-1] if cumulative_scores else 0.0
    boundaries = _level_boundaries(actions_per_level, levels_completed)

    raw_steps = _read_trace_steps(job_dir / "traces" / run_id)
    ranges = _action_index_ranges(
        [{"action_index": step.get("action_index")} for step in raw_steps],
        total_actions,
    )

    steps_out: list[dict[str, Any]] = []
    cum_tokens = 0
    cursor = 0  # actions executed after the steps processed so far
    for position, (raw, (range_start, range_end)) in enumerate(
        zip(raw_steps, ranges, strict=True)
    ):
        output_len = len(raw.get("output_token_ids") or [])
        cum_tokens += output_len
        if range_end > range_start:
            cursor = range_end
        levels_after = sum(1 for boundary in boundaries if boundary < cursor)
        score_after = cumulative_scores[levels_after - 1] if levels_after else 0.0
        elapsed = action_wallclock[cursor - 1] if cursor > 0 else 0.0
        steps_out.append(
            {
                "step_index": position,
                "output_tokens": output_len,
                "cum_output_tokens": cum_tokens,
                "action_index": raw.get("action_index"),
                "actions_executed_after": cursor,
                "levels_completed_after": levels_after,
                "score_after": score_after,
                "elapsed_seconds_after": elapsed,
            }
        )

    if raw_steps and cum_tokens == 0:
        raise RuntimeError(
            f"{job_dir}: trace has {len(raw_steps)} steps but zero output "
            "tokens — token capture is off, and the whole eval keys on "
            "generated-token budgets"
        )

    return {
        "arm": arm.arm,
        "game": game,
        "seed": int(seed),
        "run_id": run_id,
        "taaf_game_id": str(run.get("game_id", "")),
        "state": str(run.get("state", "")),
        "levels_completed": levels_completed,
        "number_of_levels": number_of_levels,
        "actions_per_level": actions_per_level,
        "base_actions_per_level": base_actions_per_level,
        "final_score": final_score,
        "final_wallclock_seconds": run.get("final_wallclock_seconds"),
        "total_actions": total_actions,
        "total_steps": len(steps_out),
        "total_output_tokens": cum_tokens,
        "runtime_cap_seconds": float(arm.cap_seconds),
        # The Duck's per-level completion-token budget this episode played
        # under (its preset's; null: none).
        "max_tokens_per_level": max_tokens_per_level,
        "episode_started_unix": started_at,
        "episode_finished_unix": finished_at,
        "steps": steps_out,
        # A crashed game run is an infrastructure failure, not a zero score:
        # it must fail the tally loudly, never dilute the means.
        "status": "ok" if str(run.get("state", "")) != "crashed" else "crashed",
    }


# --------------------------------------------------------------------------
# scheduling: per-server worker pool with continuous backfill
# --------------------------------------------------------------------------


def _episode_key(game: str, seed: int) -> str:
    return f"{game}_s{seed}"


async def _run_one_episode(
    *,
    game: str,
    seed: int,
    arm: ArmConfig,
    job_dir: Path,
    watchdog_seconds: float,
) -> dict[str, Any]:
    if job_dir.exists():
        # A retried/resumed episode must not mix artifacts with its failed
        # predecessor's.
        shutil.rmtree(job_dir)
    from training.rollout_gateway import GatewayError

    started_at = time.time()
    gateway = None
    try:
        benchmark = _build_episode_benchmark(
            game=game, seed=seed, arm=arm, job_dir=job_dir
        )
        gateway = await asyncio.to_thread(_episode_gateway, benchmark, arm)
        # The duck runtime cannot interrupt an in-flight LLM turn, so the
        # harness cap (max_runtime_s_per_game) can overshoot; the watchdog
        # bounds a wedged episode so it cannot hold a worker slot forever.
        await asyncio.wait_for(
            _checked_benchmark(benchmark, gateway), timeout=watchdog_seconds
        )
        await asyncio.to_thread(gateway.drain_window)
        finished_at = time.time()
        per_level = benchmark.solver.agent.max_tokens_per_level
        return _parse_episode(
            job_dir,
            game=game,
            seed=seed,
            arm=arm,
            started_at=started_at,
            finished_at=finished_at,
            max_tokens_per_level=None if per_level is None else list(per_level),
        )
    except BaseException as exc:  # noqa: BLE001 — recorded, counted, reported
        if isinstance(
            exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError, GatewayError)
        ):
            raise
        status = (
            "watchdog_timeout" if isinstance(exc, asyncio.TimeoutError) else "error"
        )
        print(
            f"[{arm.arm}] EPISODE FAILED game={game} seed={seed} ({status}): "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        return {
            "arm": arm.arm,
            "game": game,
            "seed": int(seed),
            "status": status,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_cap_seconds": float(arm.cap_seconds),
            "episode_started_unix": started_at,
            "episode_finished_unix": time.time(),
            "final_score": 0.0,
            "steps": [],
        }
    finally:
        if gateway is not None:
            try:
                # Capture an already-admitted final draw even if a watchdog
                # ended the client. Do not reuse or silently replay it.
                await asyncio.to_thread(gateway.drain_window)
            finally:
                await asyncio.to_thread(gateway.stop)


async def _arm_worker(
    *,
    worker_id: int,
    arm: ArmConfig,
    queue: asyncio.Queue,
    episodes_root: Path,
    results_path: Path,
    watchdog_seconds: float,
    counters: dict[str, int],
    startup_stagger_seconds: float,
) -> None:
    if startup_stagger_seconds > 0:
        # A cold engine drops a simultaneous first-connect blast from every
        # worker (connect timeouts surfacing as anyio ValueErrors); ramp the
        # workers in instead.
        await asyncio.sleep(worker_id * startup_stagger_seconds)
    while True:
        try:
            game, seed = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        job_dir = episodes_root / arm.arm / _episode_key(game, seed)
        record = await _run_one_episode(
            game=game,
            seed=seed,
            arm=arm,
            job_dir=job_dir,
            watchdog_seconds=watchdog_seconds,
        )
        # Single event loop + no awaits between open and close: appends are
        # serialized per line by construction.
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        counters["done"] += 1
        if record["status"] != "ok":
            counters["failed"] += 1
        print(
            f"[{arm.arm}] {counters['done']}/{counters['total']} done "
            f"(worker {worker_id}): {game} seed={seed} status={record['status']} "
            f"state={record.get('state', '-')} score={record.get('final_score', 0.0):.2f} "
            f"gen_tokens={record.get('total_output_tokens', 0)}",
            flush=True,
        )


def _load_completed_keys(results_path: Path) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    if not results_path.exists():
        return completed
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add((str(record["game"]), int(record["seed"])))
    return completed


async def _run_arm(
    *,
    arm: ArmConfig,
    games: list[str],
    seeds: list[int],
    eval_dir: Path,
    concurrency: int,
    watchdog_margin_seconds: float,
    startup_stagger_seconds: float,
) -> dict[str, int]:
    results_path = eval_dir / f"episodes_{arm.arm}.jsonl"
    completed = _load_completed_keys(results_path)
    if completed:
        print(
            f"[{arm.arm}] resuming: {len(completed)} episodes already recorded ok",
            flush=True,
        )
    queue: asyncio.Queue = asyncio.Queue()
    # Pass-major order: the first `concurrency` in-flight episodes cover the
    # games broadly instead of stacking every seed of the first games.
    scheduled = 0
    for seed in seeds:
        for game in games:
            if (game, seed) in completed:
                continue
            queue.put_nowait((game, seed))
            scheduled += 1
    counters = {"done": 0, "failed": 0, "total": scheduled}
    if scheduled == 0:
        print(f"[{arm.arm}] nothing to do (all episodes recorded)", flush=True)
        return counters
    watchdog_seconds = float(arm.cap_seconds) + float(watchdog_margin_seconds)
    workers = [
        _arm_worker(
            worker_id=index,
            arm=arm,
            queue=queue,
            episodes_root=eval_dir / "episodes",
            results_path=results_path,
            watchdog_seconds=watchdog_seconds,
            counters=counters,
            startup_stagger_seconds=startup_stagger_seconds,
        )
        for index in range(min(concurrency, scheduled))
    ]
    await asyncio.gather(*workers)
    return counters


# --------------------------------------------------------------------------
# LoRA server preparation (export + runtime load, mirroring the trainer)
# --------------------------------------------------------------------------


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _model_alias_available(base_url: str, model_id: str, api_key: str | None) -> bool:
    response = requests.get(
        base_url.rstrip("/") + "/models", headers=_auth_headers(api_key), timeout=15
    )
    response.raise_for_status()
    models = response.json().get("models") or []
    return any(
        str(item.get("name", "")).strip() == model_id
        for item in models
        if isinstance(item, dict)
    )


def _warm_alias(base_url: str, model_id: str, api_key: str | None) -> float:
    """1-token completion through the alias: proves it actually serves and
    claims the GPU LoRA slot before episodes start (see
    training.inference.RolloutCollector.warm_runtime_adapter)."""
    started = time.perf_counter()
    response = requests.post(
        base_url.rstrip("/") + "/adapters/warm",
        json={"name": model_id},
        headers=_auth_headers(api_key),
        timeout=_RUNTIME_LORA_REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"alias {model_id!r} did not serve a warm-up completion: "
            f"HTTP {response.status_code} {response.text[:300]}"
        )
    return time.perf_counter() - started


def _prepare_lora_arm(
    *,
    adapter_dir: Path,
    eval_dir: Path,
    lora_base_url: str,
    alias: str,
    api_key: str | None,
) -> Path:
    """Export the trainer-format checkpoint to the runtime-vLLM layout and
    hot-load it under ``alias`` — the exact activation path training uses
    (training/cli.py warm starts re-export with export_runtime_lora_adapter;
    training/inference.py loads via POST /adapters/load with a fresh
    adapter identity and verifies the alias serves)."""
    from training.runtime_lora import export_runtime_lora_adapter

    runtime_dir = eval_dir / "runtime_vllm_adapter"
    if (runtime_dir / "adapter_model.safetensors").exists():
        print(f"reusing runtime adapter export at {runtime_dir}", flush=True)
    else:
        export = export_runtime_lora_adapter(
            source_adapter_dir=adapter_dir,
            output_dir=runtime_dir,
            validate_load_invariants=True,
            compute_adapter_sha256=True,
        )
        print(
            f"exported runtime adapter: rank={export.runtime_rank} "
            f"alpha={export.runtime_lora_alpha} mode={export.export_mode} "
            f"sha256={export.adapter_sha256[:16]} -> {runtime_dir}",
            flush=True,
        )

    load_url = lora_base_url.rstrip("/") + "/adapters/load"
    payload = {
        "name": alias,
        "path": str(runtime_dir.resolve()),
    }
    response = requests.post(
        load_url,
        json=payload,
        headers=_auth_headers(api_key),
        timeout=_RUNTIME_LORA_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    from training.inference import _validated_adapter_identity

    identity = _validated_adapter_identity(response.json().get("adapter"), alias)
    if identity["id"] == 0 or identity["path"] != payload["path"]:
        raise RuntimeError(
            "native engine loaded a different adapter path than requested"
        )
    if not _model_alias_available(lora_base_url, alias, api_key):
        raise RuntimeError(
            f"runtime LoRA alias {alias!r} is not listed by {lora_base_url}/models "
            "after a reported load"
        )
    warm_seconds = _warm_alias(lora_base_url, alias, api_key)
    print(
        f"runtime adapter {alias!r} served its warm-up completion in "
        f"{warm_seconds:.2f}s",
        flush=True,
    )
    return runtime_dir


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        choices=("training", "deployment"),
        default="training",
        help=(
            "training: mirror the trainer's agent knobs (T=1.0, config window/"
            "cap/effort) for an on-policy A/B. deployment: the duck's own "
            "benchmark protocol - preset sampling profiles, xhigh effort, "
            "90k window - comparable to the official duck baselines."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/ppo_2gpu_arc3_games25_platypus.json",
        help="training config the game list + agent knobs are read from",
    )
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument(
        "--base-base-url",
        default=None,
        help="base-arm native engine origin (no /v1) (required for the base arm)",
    )
    parser.add_argument(
        "--arms",
        choices=("both", "base", "lora"),
        default="both",
        help="which arms to run; single-arm runs need only that arm's server",
    )
    parser.add_argument(
        "--max-tokens-per-game",
        type=int,
        default=None,
        help="runtime per-episode generated-token budget (duck max_tokens_per_game); the episode self-terminates at the budget",
    )
    parser.add_argument(
        "--startup-stagger-seconds",
        type=float,
        default=2.0,
        help="per-worker launch delay (worker i waits i*this) so cold engines never see a simultaneous connect blast",
    )
    parser.add_argument(
        "--duck-default-context",
        action="store_true",
        help="leave context_window/compact_to at the duck's own defaults instead of the config's training values",
    )
    parser.add_argument(
        "--lora-base-url", required=True, help="lora-arm native engine origin (no /v1)"
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="trainer-format LoRA checkpoint dir (adapter_config.json + safetensors)",
    )
    parser.add_argument("--lora-alias", default=_LORA_ALIAS_DEFAULT)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1000,
        help="episode seeds are seed_base + pass_index, identical for both arms",
    )
    parser.add_argument(
        "--base-cap-minutes",
        type=float,
        default=60.0,
        help="wall-clock safety backstop per base-arm episode; the analysis "
        "token budget (80k generated tokens) is expected to bind first",
    )
    parser.add_argument(
        "--lora-cap-minutes",
        type=float,
        default=60.0,
        help="wall-clock safety backstop per lora-arm episode (same backstop "
        "design as the base arm)",
    )
    parser.add_argument(
        "--concurrency-per-server",
        type=int,
        default=_DEFAULT_CONCURRENCY_PER_SERVER,
        help="episodes kept in flight on each server (continuous backfill)",
    )
    parser.add_argument(
        "--watchdog-margin-minutes",
        type=float,
        default=20.0,
        help="per-episode hard timeout margin beyond the harness runtime cap",
    )
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    inference = config["inference"]
    games = [str(game) for game in inference["games"]]
    if not games:
        raise RuntimeError(f"{args.config} lists no games")
    seeds = [int(args.seed_base) + index for index in range(int(args.passes))]

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    sampling = {
        "temperature": float(inference["agent_temperature"]),
        "top_p": float(inference["agent_top_p"]),
        "top_k": int(inference.get("agent_top_k") or 0) or None,
    }
    if args.protocol == "deployment":
        # The duck's standard benchmark protocol: preset defaults everywhere.
        # No knob overrides (window 90112 / compact 33925 / effort xhigh come
        # from the duck itself); the reply cap must still be stated (it has no
        # default anywhere) - 8192 is the existing deployment reply cap.
        common = {
            "max_completion_tokens": 8192,
            "reasoning_effort": None,
            "context_window_tokens": None,
            "compact_to_tokens": None,
            "sampling": None,  # keep deployment T/top-p/top-k, with neutral penalties
            "api_key": args.api_key,
        }
    else:
        common = {
            "max_completion_tokens": int(inference["agent_max_completion_tokens"]),
            "reasoning_effort": inference.get("agent_reasoning_effort") or None,
            "context_window_tokens": inference.get("agent_context_window"),
            "compact_to_tokens": inference.get("agent_compact_to_tokens"),
            "sampling": sampling,
            "api_key": args.api_key,
        }
    if args.duck_default_context:
        common["context_window_tokens"] = None
        common["compact_to_tokens"] = None
    common["max_tokens_per_game"] = args.max_tokens_per_game
    common["tokenizer_path"] = inference.get("tokenizer_id") or inference["model_id"]

    run_base = args.arms in ("both", "base")
    run_lora = args.arms in ("both", "lora")
    base_arm = lora_arm = None
    if run_base:
        if not args.base_base_url:
            raise RuntimeError("--base-base-url is required when the base arm runs")
        base_arm = ArmConfig(
            arm="base",
            base_url=args.base_base_url,
            served_model=str(inference["model_id"]),
            cap_seconds=float(args.base_cap_minutes) * 60.0,
            **common,
        )
    if run_lora:
        lora_arm = ArmConfig(
            arm="lora",
            base_url=args.lora_base_url,
            served_model=str(args.lora_alias),
            cap_seconds=float(args.lora_cap_minutes) * 60.0,
            **common,
        )

    # Every selected server must be up and serving the right identity before
    # any episode starts (health-waited by the sbatch already; this is the
    # identity check, not the liveness wait).
    runtime_adapter_dir = None
    if base_arm is not None and not _model_alias_available(
        base_arm.base_url, base_arm.served_model, args.api_key
    ):
        raise RuntimeError(
            f"base server {base_arm.base_url} does not list {base_arm.served_model!r}"
        )
    if lora_arm is not None:
        runtime_adapter_dir = _prepare_lora_arm(
            adapter_dir=Path(args.adapter_dir),
            eval_dir=eval_dir,
            lora_base_url=lora_arm.base_url,
            alias=lora_arm.served_model,
            api_key=args.api_key,
        )

    manifest = {
        "config": args.config,
        "games": games,
        "seeds": seeds,
        "episodes_per_arm": len(games) * len(seeds),
        "concurrency_per_server": int(args.concurrency_per_server),
        "arms": args.arms,
        "max_tokens_per_game": args.max_tokens_per_game,
        "base": None
        if base_arm is None
        else {
            "base_url": base_arm.base_url,
            "served_model": base_arm.served_model,
            "cap_seconds": base_arm.cap_seconds,
        },
        "lora": None
        if lora_arm is None
        else {
            "base_url": lora_arm.base_url,
            "served_model": lora_arm.served_model,
            "cap_seconds": lora_arm.cap_seconds,
            "adapter_dir": str(args.adapter_dir),
            "runtime_adapter_dir": str(runtime_adapter_dir),
        },
        # The EFFECTIVE sampling: None means the duck's own profiles
        # (deployment protocol); a dict is the training-mirror override.
        "sampling": common["sampling"],
        "agent": {
            key: common[key]
            for key in (
                "max_completion_tokens",
                "reasoning_effort",
                "context_window_tokens",
                "compact_to_tokens",
            )
        },
        "started_unix": time.time(),
    }
    (eval_dir / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    active_arms = [arm for arm in (base_arm, lora_arm) if arm is not None]
    print(
        f"eval [{args.arms}]: {len(games)} games x {len(seeds)} seeds per arm "
        f"({len(games) * len(seeds)} episodes/arm), "
        + ", ".join(f"{a.arm} cap={a.cap_seconds:.0f}s" for a in active_arms)
        + f", token budget {args.max_tokens_per_game or 'none'}, "
        f"{args.concurrency_per_server} in flight per server -> {eval_dir}",
        flush=True,
    )

    watchdog_margin_seconds = float(args.watchdog_margin_minutes) * 60.0
    results = await asyncio.gather(
        *(
            _run_arm(
                arm=arm,
                games=games,
                seeds=seeds,
                eval_dir=eval_dir,
                concurrency=int(args.concurrency_per_server),
                watchdog_margin_seconds=watchdog_margin_seconds,
                startup_stagger_seconds=float(args.startup_stagger_seconds),
            )
            for arm in active_arms
        )
    )
    failed = sum(counters["failed"] for counters in results)
    done = sum(counters["done"] for counters in results)
    print(
        f"eval finished: {done} episodes run this invocation, {failed} failed",
        flush=True,
    )
    if failed:
        print(
            "some episodes failed at the driver level — re-run the driver with "
            "the same --eval-dir to retry just those (ok episodes are skipped)",
            flush=True,
        )
        return 2
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
