from __future__ import annotations

import random
from contextlib import suppress

import numpy as np
import pytest
from arcengine import GameAction

from re_arc import EnvSampler, list_game_ids
from re_arc.dsl import resolve_action, unpack_step_result
from re_arc.dsl.precomputed_actions import load_precomputed_actions

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
    # frame is a list of rendered frames spanning the action's animation;
    # only the last one is the steady state a client displays. Copy because
    # the engine returns a reference to its render buffer, which subsequent
    # env.step() calls overwrite in place.
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
    candidates = [a for a in action_space if int(getattr(a, "value", -1)) != RESET_ACTION_ID]
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


def _bootstrap_to_level(env, dsl_actions, target_level: int):
    # Replay the DSL trace until levels_completed == target_level. Returns the
    # observation at that moment, plus an error string, or (obs, None) on
    # success / (None, reason) when the target can't be reached.
    obs = env.reset()
    level = _levels_completed(obs)
    if level == target_level:
        return obs, None
    if level > target_level:
        return None, f"reset started beyond target level {target_level}; observed level={level}"
    for action_id, payload in dsl_actions:
        action = resolve_action(env, int(action_id))
        obs, _, done, _ = unpack_step_result(env.step(action, data=dict(payload or {})))
        level = _levels_completed(obs)
        if level == target_level:
            return obs, None
        if level > target_level:
            return None, f"overshot target level {target_level}; landed at {level}"
        if done:
            return None, f"terminated before reaching level {target_level}; state={_state_name(obs)}"
    return None, f"DSL trace exhausted before reaching level {target_level}"


def _check_reset_restores_level(
    game_id: str, sampler: EnvSampler, dsl_actions: list[tuple[int, dict[str, int]]], target_level: int
) -> bool | None:
    """Verify that a mid-play RESET on ``target_level`` restores the same
    frame a client saw when the level was first entered.

    Returns True when the check ran, None when it was skipped (unreachable
    level, trivial setup, etc.). Raises AssertionError on a real mismatch.

    The captured "initial frame" is the final rendered frame from the
    observation that a client receives on arrival at ``target_level`` —
    either env.reset() for level 0, or the observation whose step() fired
    next_level() for level N > 0. That's what the player sees when they
    land on the level, and therefore what RESET is expected to restore.
    """
    env = sampler.make(game_id=game_id, seed=RANDOM_SEED)
    try:
        initial_observation, bootstrap_error = _bootstrap_to_level(env, dsl_actions, target_level)
        if bootstrap_error is not None:
            return None
        assert initial_observation is not None
        initial_frame = _final_frame(initial_observation)
        if initial_frame is None:
            return None
        frame_shape = _frame_shape(initial_frame)

        rng = random.Random(f"reset-restores:{game_id}:{target_level}")
        actions_taken: list[tuple[object, dict[str, int]]] = []
        winning_action_idx: int | None = None

        for _ in range(MOVES_PER_TRIAL):
            action, payload = _pick_random_action(rng, env.action_space, frame_shape)
            if action is None:
                break
            observation, _, done, _ = unpack_step_result(env.step(action, data=dict(payload)))
            actions_taken.append((action, dict(payload)))
            if _levels_completed(observation) > target_level:
                winning_action_idx = len(actions_taken) - 1
                break
            # Loss states are fine to test from directly: a subsequent RESET
            # should still restore the level's pristine initial frame.
            if done or _state_name(observation) == "GAME_OVER":
                break

        if winning_action_idx is not None:
            # Random play crossed into the next level, so a RESET from here
            # would restore that level, not target_level. Rebuild the env,
            # re-bootstrap to target_level, and replay every recorded action
            # except the winning one — that leaves us in a non-trivial
            # mid-level state before the RESET.
            _close_env(env)
            env = sampler.make(game_id=game_id, seed=RANDOM_SEED)
            _, rebootstrap_error = _bootstrap_to_level(env, dsl_actions, target_level)
            if rebootstrap_error is not None:
                return None
            for action, payload in actions_taken[:winning_action_idx]:
                env.step(action, data=dict(payload))

        if not actions_taken:
            return None

        post_reset_observation, _, _, _ = unpack_step_result(env.step(GameAction.RESET))
        post_reset_frame = _final_frame(post_reset_observation)

        assert post_reset_frame is not None, f"{game_id} level {target_level}: RESET produced no frame payload."
        assert np.array_equal(initial_frame, post_reset_frame), (
            f"{game_id} level {target_level}: RESET did not restore the initial "
            f"frame after {len(actions_taken)} random action(s)."
        )
        return True
    finally:
        _close_env(env)


@pytest.mark.parametrize("game_id", list(list_game_ids()))
def test_reset_restores_initial_frame(game_id: str):
    sampler = EnvSampler(augment=False, seed=RANDOM_SEED)
    probe = sampler.make(game_id=game_id, seed=RANDOM_SEED)
    try:
        level_count = _win_levels(probe.reset()) or 1
    finally:
        _close_env(probe)

    precomputed = load_precomputed_actions(game_id)
    dsl_actions: list[tuple[int, dict[str, int]]] = []
    if precomputed is not None:
        for action_id, payload in precomputed.get("actions", []):
            dsl_actions.append((int(action_id), {str(k): int(v) for k, v in (payload or {}).items()}))

    any_checked = False
    for target_level in range(level_count):
        if target_level > 0 and not dsl_actions:
            continue
        if _check_reset_restores_level(game_id, sampler, dsl_actions, target_level):
            any_checked = True

    if not any_checked:
        pytest.skip(f"{game_id}: no level could be exercised (unreachable or trivial).")
