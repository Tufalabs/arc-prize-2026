from __future__ import annotations

import pytest
from arcengine import GameAction

from re_arc import EnvSampler

GAME_WITHOUT_NATIVE_RESET = "identify_the_agent-0002"
GAME_WITH_NATIVE_UNDO = "paint_snakes-0001"
RANDOM_SEED = 0
RESET_ACTION_ID = int(GameAction.RESET.value)
UNDO_ACTION_ID = int(GameAction.ACTION7.value)


def _make_env(game_id: str):
    sampler = EnvSampler(include=[game_id], augment=False, seed=RANDOM_SEED)
    return sampler.make(game_id=game_id, seed=RANDOM_SEED)


def _action_ids(env) -> list[int]:
    return [int(action.value) for action in env.action_space]


def test_reset_is_always_advertised_in_observations_and_action_space():
    env = _make_env(GAME_WITHOUT_NATIVE_RESET)
    initial = env.reset()

    assert RESET_ACTION_ID in initial.available_actions
    assert list(initial.available_actions).count(RESET_ACTION_ID) == 1
    # RESET is discoverable through the public action_space as well, even
    # before any action has been taken in the level.
    assert _action_ids(env).count(RESET_ACTION_ID) == 1

    moved, _, _, _ = env.step(GameAction.ACTION1)

    assert RESET_ACTION_ID in moved.available_actions
    assert list(moved.available_actions).count(RESET_ACTION_ID) == 1

    reset_obs, _, _, info = env.step(GameAction.RESET)

    assert info["level_reset"] is True
    assert RESET_ACTION_ID in reset_obs.available_actions


def test_reset_before_any_action_is_a_harmless_ignored_no_op():
    env = _make_env(GAME_WITHOUT_NATIVE_RESET)
    env.reset()

    obs, reward, done, info = env.step(GameAction.RESET)

    assert info["reset_ignored"] is True
    assert reward == pytest.approx(0.0)
    assert done is False
    assert RESET_ACTION_ID in obs.available_actions


def test_action7_is_not_synthesized_for_games_without_native_undo():
    env = _make_env(GAME_WITHOUT_NATIVE_RESET)
    initial = env.reset()

    assert UNDO_ACTION_ID not in initial.available_actions

    moved, _, _, _ = env.step(GameAction.ACTION1)

    assert UNDO_ACTION_ID not in moved.available_actions
    assert UNDO_ACTION_ID not in _action_ids(env)


def test_game_with_native_action7_keeps_its_own_advertising_and_handling():
    env = _make_env(GAME_WITH_NATIVE_UNDO)
    initial = env.reset()

    assert list(initial.available_actions).count(UNDO_ACTION_ID) == 1
    assert RESET_ACTION_ID in initial.available_actions

    game = env._game
    start_energy = int(game.steps_left)
    start_pos = tuple(game.snakes[game.active_snake]["pos"])

    env.step(GameAction.ACTION4)
    assert tuple(game.snakes[game.active_snake]["pos"]) != start_pos

    undone, _, _, _ = env.step(GameAction.ACTION7)

    # The game's own undo semantics apply: the move is reverted and both the
    # move and the undo consume energy.
    assert tuple(game.snakes[game.active_snake]["pos"]) == start_pos
    assert int(game.steps_left) == start_energy - 2
    assert list(undone.available_actions).count(UNDO_ACTION_ID) == 1
