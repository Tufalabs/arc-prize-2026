from __future__ import annotations

from types import SimpleNamespace

import pytest
from arcengine import GameAction

from re_arc.reward_env import TransitionRewardEnv


def _first_non_reset_action():
    for action in GameAction:
        if int(getattr(action, "value", -1)) != int(GameAction.RESET.value):
            return action
    raise RuntimeError("GameAction enum has no non-reset action.")


NON_RESET_ACTION = _first_non_reset_action()


def _obs(*, levels_completed: int, win_levels: int, state_name: str):
    return SimpleNamespace(
        game_id="test-game",
        levels_completed=int(levels_completed),
        win_levels=int(win_levels),
        state=SimpleNamespace(name=str(state_name)),
    )


class _FakeEnv:
    def __init__(self, *, reset_obs, step_obs):
        self._reset_obs = reset_obs
        self._step_obs = list(step_obs)
        self.action_space = (NON_RESET_ACTION, GameAction.RESET)

    def reset(self, *args, **kwargs):
        _ = (args, kwargs)
        return self._reset_obs

    def step(self, action, data=None):
        _ = (action, data)
        assert self._step_obs, "No configured observation left for step()."
        return self._step_obs.pop(0)


class _TupleStepEnv(_FakeEnv):
    def step(self, action, data=None):
        observation = super().step(action, data=data)
        return observation, 0.0, True, {"from_inner": True}


def test_terminal_win_without_level_transition_reconciles_episode_reward():
    wrapped = TransitionRewardEnv(
        _FakeEnv(
            reset_obs=_obs(levels_completed=0, win_levels=1, state_name="RUNNING"),
            step_obs=[_obs(levels_completed=0, win_levels=1, state_name="WIN")],
        )
    )
    wrapped.reset()

    _, reward, done, info = wrapped.step(NON_RESET_ACTION)

    assert done is True
    assert reward > 0.0
    assert reward == pytest.approx(1.0)
    assert wrapped.episode_reward == pytest.approx(1.0)
    assert info["episode_reward"] == pytest.approx(1.0)
    assert info["level_transition"] is False


def test_normal_transition_reward_is_unchanged():
    wrapped = TransitionRewardEnv(
        _FakeEnv(
            reset_obs=_obs(levels_completed=0, win_levels=4, state_name="RUNNING"),
            step_obs=[_obs(levels_completed=1, win_levels=4, state_name="RUNNING")],
        )
    )
    wrapped.reset()

    _, reward, done, info = wrapped.step(NON_RESET_ACTION)

    assert done is False
    assert reward == pytest.approx(0.25)
    assert wrapped.episode_reward == pytest.approx(0.25)
    assert info["episode_reward"] == pytest.approx(0.25)
    assert info["level_transition"] is True


def test_terminal_win_correction_does_not_double_count_after_unit_return():
    wrapped = TransitionRewardEnv(
        _FakeEnv(
            reset_obs=_obs(levels_completed=0, win_levels=1, state_name="RUNNING"),
            step_obs=[
                _obs(levels_completed=1, win_levels=1, state_name="RUNNING"),
                _obs(levels_completed=1, win_levels=1, state_name="WIN"),
            ],
        )
    )
    wrapped.reset()

    _, first_reward, first_done, _ = wrapped.step(NON_RESET_ACTION)
    _, second_reward, second_done, info = wrapped.step(NON_RESET_ACTION)

    assert first_done is False
    assert first_reward == pytest.approx(1.0)
    assert second_done is True
    assert second_reward == pytest.approx(0.0)
    assert wrapped.episode_reward == pytest.approx(1.0)
    assert info["episode_reward"] == pytest.approx(1.0)


def test_step_unpacks_inner_tuple_results():
    wrapped = TransitionRewardEnv(
        _TupleStepEnv(
            reset_obs=_obs(levels_completed=0, win_levels=1, state_name="RUNNING"),
            step_obs=[_obs(levels_completed=0, win_levels=1, state_name="WIN")],
        )
    )
    wrapped.reset()

    _, reward, done, info = wrapped.step(NON_RESET_ACTION)

    assert done is True
    assert reward == pytest.approx(1.0)
    assert wrapped.episode_reward == pytest.approx(1.0)
    assert info["episode_reward"] == pytest.approx(1.0)


def test_game_over_still_exposes_level_reset_action_without_counter():
    wrapped = TransitionRewardEnv(
        _FakeEnv(reset_obs=_obs(levels_completed=2, win_levels=4, state_name="GAME_OVER"), step_obs=[])
    )
    wrapped.reset()

    assert GameAction.RESET in wrapped.action_space


def test_game_over_level_reset_is_not_ignored_without_counter():
    wrapped = TransitionRewardEnv(
        _FakeEnv(reset_obs=_obs(levels_completed=2, win_levels=4, state_name="GAME_OVER"), step_obs=[])
    )
    wrapped.reset()

    obs, reward, done, info = wrapped.step(GameAction.RESET)

    assert obs.state.name == "GAME_OVER"
    assert reward == pytest.approx(0.0)
    assert done is True
    assert info["level_reset"] is True
    assert info["reset_ignored"] is False


def test_action_space_always_includes_reset_exactly_once():
    wrapped = TransitionRewardEnv(
        _FakeEnv(reset_obs=_obs(levels_completed=0, win_levels=4, state_name="RUNNING"), step_obs=[])
    )
    reset_action_id = int(GameAction.RESET.value)

    # The inner fake advertises RESET itself; the wrapper must not duplicate
    # it, and must expose it even before any action in the level.
    action_ids = [int(action.value) for action in wrapped.action_space]
    assert action_ids.count(reset_action_id) == 1

    wrapped.reset()

    action_ids = [int(action.value) for action in wrapped.action_space]
    assert action_ids.count(reset_action_id) == 1


def test_observations_always_advertise_reset():
    wrapped = TransitionRewardEnv(
        _FakeEnv(
            reset_obs=_obs(levels_completed=0, win_levels=4, state_name="RUNNING"),
            step_obs=[_obs(levels_completed=0, win_levels=4, state_name="RUNNING")],
        )
    )
    reset_action_id = int(GameAction.RESET.value)

    observation = wrapped.reset()

    assert list(observation.available_actions) == [reset_action_id]

    observation, _, _, _ = wrapped.step(NON_RESET_ACTION)

    assert list(observation.available_actions) == [reset_action_id]
