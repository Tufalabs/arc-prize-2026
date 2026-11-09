from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress

import pytest

from re_arc import EnvSampler, list_game_ids
from re_arc.dsl import is_terminal, unpack_step_result

RANDOM_ACTION_SMOKE_ENV = "RE_ARC_RUN_RANDOM_ACTION_SMOKE"
RANDOM_ACTION_SMOKE_WORKERS_ENV = "RE_ARC_RANDOM_ACTION_SMOKE_WORKERS"
RANDOM_ACTION_SMOKE_STEPS_ENV = "RE_ARC_RANDOM_ACTION_SMOKE_STEPS"
DEFAULT_RANDOM_ACTION_SMOKE_STEPS = 50_000
DEFAULT_RANDOM_ACTION_SMOKE_SEED = 0
DISPLAY_COORD_MIN = 0
DISPLAY_COORD_MAX = 63


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


def _resolve_workers(game_count: int) -> int:
    configured = _int_env(RANDOM_ACTION_SMOKE_WORKERS_ENV, 0)
    if configured > 0:
        return max(1, min(configured, game_count))
    detected = os.cpu_count() or 1
    return max(1, min(detected, game_count))


def _exercise_game(game_id: str, *, random_steps: int, seed: int) -> str | None:
    sampler = EnvSampler(augment=False, seed=seed)
    env = sampler.make(game_id=game_id, seed=seed)
    rng = random.Random(f"{seed}:{game_id}:random-smoke")

    try:
        observation = env.reset()
        for step_idx in range(1, random_steps + 1):
            selectable_actions = list(env.action_space)
            if not selectable_actions:
                return f"{game_id}: no actions available at random step {step_idx}."

            action = rng.choice(selectable_actions)
            payload = _random_action_data(rng, action)

            try:
                observation, _, done, _ = unpack_step_result(env.step(action, data=payload))
            except Exception as exc:
                action_id = int(getattr(action, "value", -1))
                return (
                    f"{game_id}: random step {step_idx} raised {type(exc).__name__}: {exc} "
                    f"(action={action_id}, payload={payload})"
                )

            if done or is_terminal(observation) or _state_name(observation) in {"WIN", "GAME_OVER"}:
                observation = env.reset()
                if observation is None:
                    return f"{game_id}: reset returned no observation after random step {step_idx}."

        return None
    finally:
        _close_env(env)


def test_random_actions_smoke_all_games():
    if not _env_flag_enabled(RANDOM_ACTION_SMOKE_ENV):
        pytest.skip(f"Random action smoke test is disabled by default. Set {RANDOM_ACTION_SMOKE_ENV}=1 to run it.")

    random_steps = _int_env(RANDOM_ACTION_SMOKE_STEPS_ENV, DEFAULT_RANDOM_ACTION_SMOKE_STEPS)
    seed = DEFAULT_RANDOM_ACTION_SMOKE_SEED
    game_ids = list(list_game_ids())

    failures: list[str] = []
    workers = _resolve_workers(len(game_ids))

    if workers == 1:
        for game_id in game_ids:
            failure = _exercise_game(game_id, random_steps=random_steps, seed=seed)
            if failure is not None:
                failures.append(failure)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_game_id = {
                pool.submit(_exercise_game, game_id, random_steps=random_steps, seed=seed): game_id
                for game_id in game_ids
            }
            for future in as_completed(future_to_game_id):
                game_id = future_to_game_id[future]
                try:
                    failure = future.result()
                except Exception as exc:
                    failures.append(f"{game_id}: worker raised {type(exc).__name__}: {exc}")
                    continue
                if failure is not None:
                    failures.append(failure)

    if failures:
        pytest.fail("Random action smoke failures:\n- " + "\n- ".join(sorted(failures)))
