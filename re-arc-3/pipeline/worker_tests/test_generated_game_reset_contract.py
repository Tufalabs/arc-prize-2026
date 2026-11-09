from __future__ import annotations

import random
from contextlib import suppress

import numpy as np
import pytest
from arcengine import GameAction
from tests._worker_game import discover_generated_game

from re_arc import EnvSampler
from re_arc.dsl import create_dsl_agent, is_terminal, resolve_action, unpack_step_result

MAX_BOOTSTRAP_STEPS = 1200
MOVES_PER_TRIAL = 10
RANDOM_SEED = 0
DISPLAY_COORD_MIN = 0
DISPLAY_COORD_MAX = 63
RESET_ACTION_ID = int(GameAction.RESET.value)


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


def _final_frame(observation) -> np.ndarray | None:
    frame = getattr(observation, "frame", None)
    if frame is None:
        return None
    if isinstance(frame, list):
        for layer in reversed(frame):
            if layer is None:
                continue
            arr = np.asarray(layer)
            if arr.ndim >= 2:
                return arr.copy()
        return None
    arr = np.asarray(frame)
    return arr.copy() if arr.ndim >= 2 else None


def _frame_shape(frame: np.ndarray | None) -> tuple[int, int]:
    if frame is None:
        return DISPLAY_COORD_MAX + 1, DISPLAY_COORD_MAX + 1
    return int(frame.shape[-2]), int(frame.shape[-1])


def _pick_random_action(rng: random.Random, action_space, frame_shape: tuple[int, int]):
    candidates = [action for action in action_space if int(getattr(action, "value", -1)) != RESET_ACTION_ID]
    if not candidates:
        return None, {}
    action = rng.choice(candidates)
    data: dict[str, int] = {}
    if getattr(action, "is_complex", lambda: False)():
        height, width = frame_shape
        data = {
            "x": rng.randint(DISPLAY_COORD_MIN, min(DISPLAY_COORD_MAX, max(DISPLAY_COORD_MIN, width - 1))),
            "y": rng.randint(DISPLAY_COORD_MIN, min(DISPLAY_COORD_MAX, max(DISPLAY_COORD_MIN, height - 1))),
        }
    return action, data


def _bootstrap_to_level(env, game_id: str, target_level: int):
    observation = env.reset()
    if _levels_completed(observation) == target_level:
        return observation, None

    agent = create_dsl_agent(game_id)
    agent.reset_episode()
    agent.observe(observation)

    for _ in range(MAX_BOOTSTRAP_STEPS):
        if is_terminal(observation):
            return None, f"terminated before reaching level {target_level}; state={_state_name(observation)}"

        action_id, action_data = agent.next_action(env, observation)
        action = resolve_action(env, action_id)
        agent.record_action(action_id)
        observation, _reward, _done, _info = unpack_step_result(env.step(action, data=action_data or {}))
        agent.observe(observation)

        level = _levels_completed(observation)
        if level == target_level:
            return observation, None
        if level > target_level:
            return None, f"overshot target level {target_level}; landed at {level}"

    return None, f"DSL bootstrap exceeded {MAX_BOOTSTRAP_STEPS} steps before reaching level {target_level}"


def _check_reset_restores_level(game_id: str, sampler: EnvSampler, target_level: int) -> bool | None:
    env = sampler.make(game_id=game_id, seed=RANDOM_SEED)
    try:
        initial_observation, bootstrap_error = _bootstrap_to_level(env, game_id, target_level)
        if bootstrap_error is not None:
            return None
        assert initial_observation is not None
        initial_frame = _final_frame(initial_observation)
        if initial_frame is None:
            return None
        frame_shape = _frame_shape(initial_frame)

        rng = random.Random(f"generated-reset-restores:{game_id}:{target_level}")
        actions_taken: list[tuple[object, dict[str, int]]] = []
        winning_action_idx: int | None = None

        for _ in range(MOVES_PER_TRIAL):
            action, payload = _pick_random_action(rng, env.action_space, frame_shape)
            if action is None:
                break
            observation, _reward, done, _info = unpack_step_result(env.step(action, data=dict(payload)))
            actions_taken.append((action, dict(payload)))
            if _levels_completed(observation) > target_level:
                winning_action_idx = len(actions_taken) - 1
                break
            if done or _state_name(observation) == "GAME_OVER":
                break

        if winning_action_idx is not None:
            _close_env(env)
            env = sampler.make(game_id=game_id, seed=RANDOM_SEED)
            _bootstrap_to_level(env, game_id, target_level)
            for action, payload in actions_taken[:winning_action_idx]:
                env.step(action, data=dict(payload))

        if not actions_taken:
            return None

        post_reset_observation, _reward, _done, _info = unpack_step_result(env.step(GameAction.RESET))
        post_reset_frame = _final_frame(post_reset_observation)

        assert post_reset_frame is not None, f"{game_id} level {target_level}: RESET produced no frame payload."
        assert np.array_equal(initial_frame, post_reset_frame), (
            f"{game_id} level {target_level}: RESET did not restore the initial "
            f"frame after {len(actions_taken)} random action(s)."
        )
        return True
    finally:
        _close_env(env)


def test_generated_game_reset_restores_initial_frame():
    generated_game = discover_generated_game()
    game_id = str(generated_game["game_id"])
    sampler = EnvSampler(augment=False, seed=RANDOM_SEED)
    probe = sampler.make(game_id=game_id, seed=RANDOM_SEED)
    try:
        level_count = _win_levels(probe.reset()) or 1
    finally:
        _close_env(probe)

    any_checked = False
    for target_level in range(level_count):
        if _check_reset_restores_level(game_id, sampler, target_level):
            any_checked = True

    if not any_checked:
        pytest.skip(f"{game_id}: no level could be exercised for RESET.")
