from __future__ import annotations

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import pytest

from re_arc import EnvSampler, list_game_ids
from re_arc.dsl import is_terminal, resolve_action, unpack_step_result
from re_arc.dsl.precomputed_actions import load_precomputed_actions

RANDOM_UNSOLVABLE_STRESS_ENV = "RE_ARC_RUN_RANDOM_UNSOLVABLE_STRESS"
RANDOM_UNSOLVABLE_WORKERS_ENV = "RE_ARC_RANDOM_UNSOLVABLE_WORKERS"
RANDOM_UNSOLVABLE_SHARD_COUNT_ENV = "RE_ARC_RANDOM_UNSOLVABLE_SHARD_COUNT"
RANDOM_UNSOLVABLE_SHARD_INDEX_ENV = "RE_ARC_RANDOM_UNSOLVABLE_SHARD_INDEX"
RANDOM_UNSOLVABLE_REPORT_DIR_ENV = "RE_ARC_RANDOM_UNSOLVABLE_REPORT_DIR"
DEFAULT_RANDOM_ACTIONS_PER_LEVEL = 50_000
DEFAULT_RANDOM_SEED = 0
DISPLAY_COORD_MIN = 0
DISPLAY_COORD_MAX = 63
DEFAULT_REPORT_DIR = Path("artifacts") / "random_unsolvable_stress"


def _env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    return int(raw)


def _state_name(observation) -> str:
    state = getattr(observation, "state", None)
    return str(getattr(state, "name", state))


def _levels_completed(observation) -> int:
    return int(getattr(observation, "levels_completed", 0) or 0)


def _win_levels(observation) -> int:
    return int(getattr(observation, "win_levels", 0) or 0)


def _close_env(env) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        with suppress(Exception):
            close_fn()


def _random_action_data(rng: random.Random, action) -> dict[str, int]:
    if getattr(action, "is_complex", lambda: False)():
        return {
            "x": rng.randint(DISPLAY_COORD_MIN, DISPLAY_COORD_MAX),
            "y": rng.randint(DISPLAY_COORD_MIN, DISPLAY_COORD_MAX),
        }
    return {}


def _build_level_prefixes(game_id: str) -> tuple[list[list[tuple[int, dict[str, int]]]] | None, str | None]:
    precomputed = load_precomputed_actions(game_id)
    if precomputed is None:
        return None, f"{game_id}: missing precomputed actions required to reach every level deterministically."

    sampler = EnvSampler(augment=False, seed=DEFAULT_RANDOM_SEED)
    env = sampler.make(game_id=game_id, seed=DEFAULT_RANDOM_SEED)

    try:
        observation = env.reset()
        level_count = _win_levels(observation)
        prefixes: list[list[tuple[int, dict[str, int]]]] = [[] for _ in range(level_count)]
        actions_so_far: list[tuple[int, dict[str, int]]] = []
        current_level = _levels_completed(observation)

        if current_level >= level_count:
            return None, f"{game_id}: initial levels_completed={current_level} is outside win_levels={level_count}."

        prefixes[current_level] = list(actions_so_far)
        for action_id, action_data in precomputed.get("actions", []):
            action_id = int(action_id)
            payload = {str(key): int(value) for key, value in dict(action_data or {}).items()}
            action = resolve_action(env, action_id)
            observation, _, done, _ = unpack_step_result(env.step(action, data=payload))
            actions_so_far.append((action_id, payload))
            new_level = _levels_completed(observation)
            if new_level > current_level:
                for level_idx in range(current_level + 1, min(new_level, level_count - 1) + 1):
                    prefixes[level_idx] = list(actions_so_far)
            current_level = new_level
            if done:
                break

        missing_levels = [str(level_idx) for level_idx, prefix in enumerate(prefixes) if level_idx > 0 and not prefix]
        if missing_levels:
            return None, f"{game_id}: could not derive deterministic prefixes for levels {', '.join(missing_levels)}."

        return prefixes, None
    finally:
        _close_env(env)


def _resolve_workers(game_count: int) -> int:
    configured = _int_env(RANDOM_UNSOLVABLE_WORKERS_ENV, 0)
    if configured > 0:
        return max(1, min(configured, game_count))

    slurm_cpus = _int_env("SLURM_CPUS_PER_TASK", 0)
    if slurm_cpus > 0:
        return max(1, min(slurm_cpus, game_count))

    detected = os.cpu_count() or 1
    return max(1, min(detected, game_count))


def _resolve_shard(total_games: int) -> tuple[int, int]:
    shard_count = _int_env(RANDOM_UNSOLVABLE_SHARD_COUNT_ENV, _int_env("SLURM_ARRAY_TASK_COUNT", 1))
    shard_index = _int_env(RANDOM_UNSOLVABLE_SHARD_INDEX_ENV, _int_env("SLURM_ARRAY_TASK_ID", 0))

    if shard_count <= 0:
        raise ValueError(f"{RANDOM_UNSOLVABLE_SHARD_COUNT_ENV} must be positive; got {shard_count}.")
    if not (0 <= shard_index < shard_count):
        raise ValueError(
            f"Shard index must be in [0, {shard_count - 1}], got {shard_index}. "
            f"Configure {RANDOM_UNSOLVABLE_SHARD_INDEX_ENV} or SLURM_ARRAY_TASK_ID correctly."
        )
    if total_games <= 0:
        return shard_count, shard_index
    return shard_count, shard_index


def _resolved_report_dir() -> Path:
    configured = str(os.environ.get(RANDOM_UNSOLVABLE_REPORT_DIR_ENV, "")).strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_REPORT_DIR


def _report_stem() -> str:
    job_id = str(os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local").strip()
    task_id = str(os.environ.get("SLURM_ARRAY_TASK_ID") or "0").strip()
    return f"job-{job_id}_task-{task_id}"


def _write_report(
    *, status: str, random_steps: int, seed: int, workers: int, selected_game_ids: list[str], failures: list[str]
) -> Path:
    shard_count, shard_index = _resolve_shard(len(list(list_game_ids())))
    report_dir = _resolved_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = _report_stem()
    timestamp = datetime.now(UTC).isoformat()

    payload = {
        "status": status,
        "created_at_utc": timestamp,
        "random_steps": int(random_steps),
        "seed": int(seed),
        "workers": int(workers),
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "selected_game_ids": list(selected_game_ids),
        "failure_count": len(failures),
        "failures": list(sorted(failures)),
        "slurm_job_id": str(os.environ.get("SLURM_JOB_ID") or ""),
        "slurm_array_job_id": str(os.environ.get("SLURM_ARRAY_JOB_ID") or ""),
        "slurm_array_task_id": str(os.environ.get("SLURM_ARRAY_TASK_ID") or ""),
    }

    json_path = report_dir / f"{stem}.json"
    txt_path = report_dir / f"{stem}.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"status={status}",
        f"created_at_utc={timestamp}",
        f"random_steps={random_steps}",
        f"seed={seed}",
        f"workers={workers}",
        f"shard_index={shard_index}/{shard_count}",
        f"selected_games={len(selected_game_ids)}",
        f"failure_count={len(failures)}",
    ]
    if selected_game_ids:
        lines.append("games=" + ", ".join(selected_game_ids))
    if failures:
        lines.append("failures:")
        lines.extend(f"- {entry}" for entry in sorted(failures))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return txt_path


def _shard_game_ids(game_ids: list[str]) -> list[str]:
    shard_count, shard_index = _resolve_shard(len(game_ids))
    return [game_id for idx, game_id in enumerate(game_ids) if idx % shard_count == shard_index]


def _bootstrap_to_level(env, prefix: list[tuple[int, dict[str, int]]], target_level: int):
    observation = env.reset()
    current_level = _levels_completed(observation)
    if current_level == target_level:
        return observation, None
    if current_level > target_level:
        return None, (f"reset started beyond target level {target_level}; observed levels_completed={current_level}")

    for action_id, payload in prefix:
        action = resolve_action(env, int(action_id))
        observation, _, done, _ = unpack_step_result(env.step(action, data=dict(payload)))
        current_level = _levels_completed(observation)
        if current_level > target_level:
            return None, (
                f"overshot target level {target_level} while replaying prefix; "
                f"observed levels_completed={current_level}"
            )
        if done and current_level < target_level:
            return None, (
                f"terminated before target level {target_level}; "
                f"state={_state_name(observation)!r} levels_completed={current_level}"
            )

    current_level = _levels_completed(observation)
    if current_level != target_level:
        return None, f"expected levels_completed={target_level} after bootstrap, got {current_level}"
    return observation, None


def _exercise_level_with_random_actions(
    game_id: str, *, target_level: int, prefix: list[tuple[int, dict[str, int]]], random_steps: int, seed: int
) -> str | None:
    sampler = EnvSampler(augment=False, seed=seed)
    env = sampler.make(game_id=game_id, seed=seed)
    rng = random.Random(f"{seed}:{game_id}:{target_level}")

    try:
        observation, bootstrap_error = _bootstrap_to_level(env, prefix, target_level)
        if bootstrap_error is not None:
            return f"{game_id}: level={target_level}: bootstrap failed: {bootstrap_error}."

        for step_idx in range(1, random_steps + 1):
            selectable_actions = list(env.action_space)
            if not selectable_actions:
                return f"{game_id}: level={target_level}: no actions available at random step {step_idx}."

            action = rng.choice(selectable_actions)
            payload = _random_action_data(rng, action)
            observation, _, done, _ = unpack_step_result(env.step(action, data=payload))

            if _state_name(observation) == "WIN":
                return (
                    f"{game_id}: level={target_level}: random play reached WIN at step {step_idx} "
                    f"with action={int(getattr(action, 'value', -1))} payload={payload}."
                )

            current_level = _levels_completed(observation)
            if current_level > target_level:
                return (
                    f"{game_id}: level={target_level}: random play solved the level at step {step_idx} "
                    f"(levels_completed advanced to {current_level})."
                )

            if done or is_terminal(observation) or current_level != target_level:
                observation, bootstrap_error = _bootstrap_to_level(env, prefix, target_level)
                if bootstrap_error is not None:
                    return (
                        f"{game_id}: level={target_level}: "
                        f"re-bootstrap failed after random step {step_idx}: {bootstrap_error}."
                    )

        return None
    finally:
        _close_env(env)


def _run_game_random_unsolvable_check(game_id: str, *, random_steps: int, seed: int) -> list[str]:
    failures: list[str] = []
    prefixes, prefix_error = _build_level_prefixes(game_id)
    if prefix_error is not None:
        return [prefix_error]
    if prefixes is None:
        return [f"{game_id}: failed to derive level prefixes for an unknown reason."]

    for level_idx, prefix in enumerate(prefixes):
        failure = _exercise_level_with_random_actions(
            game_id, target_level=level_idx, prefix=prefix, random_steps=random_steps, seed=seed
        )
        if failure is not None:
            failures.append(failure)
    return failures


def test_random_actions_do_not_solve_any_level():
    if not _env_flag_enabled(RANDOM_UNSOLVABLE_STRESS_ENV):
        pytest.skip(
            f"Random unsolvable stress test is disabled by default. Set {RANDOM_UNSOLVABLE_STRESS_ENV}=1 to run it."
        )

    random_steps = _int_env("RE_ARC_RANDOM_UNSOLVABLE_STEPS", DEFAULT_RANDOM_ACTIONS_PER_LEVEL)
    seed = _int_env("RE_ARC_RANDOM_UNSOLVABLE_SEED", DEFAULT_RANDOM_SEED)
    selected_game_ids = _shard_game_ids(list(list_game_ids()))

    if not selected_game_ids:
        _ = _write_report(
            status="skipped", random_steps=random_steps, seed=seed, workers=0, selected_game_ids=[], failures=[]
        )
        pytest.skip("Random unsolvable stress shard has no assigned games.")

    failures: list[str] = []
    workers = _resolve_workers(len(selected_game_ids))
    if workers == 1:
        for game_id in selected_game_ids:
            failures.extend(_run_game_random_unsolvable_check(game_id, random_steps=random_steps, seed=seed))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_game_id = {
                pool.submit(_run_game_random_unsolvable_check, game_id, random_steps=random_steps, seed=seed): game_id
                for game_id in selected_game_ids
            }
            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    failures.extend(future.result())
                except Exception as exc:
                    failures.append(f"{game_id}: worker raised {type(exc).__name__}: {exc}")

    report_path = _write_report(
        status=("failed" if failures else "passed"),
        random_steps=random_steps,
        seed=seed,
        workers=workers,
        selected_game_ids=selected_game_ids,
        failures=failures,
    )
    if failures:
        pytest.fail(
            f"Random unsolvable stress failures (central report: {report_path}):\n- " + "\n- ".join(sorted(failures))
        )
