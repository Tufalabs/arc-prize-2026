from __future__ import annotations

import copy
import os
from collections.abc import Callable
from typing import cast

from arcengine import GameAction

from .dsl import is_terminal, unpack_step_result
from .types import Action, ActionData, ActionValue, Environment, IntCoercible, Observation, StepInfo, StepResult

ParsedStepResult = tuple[Observation, float, bool, dict[str, ActionValue]]


def _safe_int(value: IntCoercible | None, default: int) -> int:
    try:
        return int(cast(IntCoercible, value))
    except (TypeError, ValueError):
        return int(default)


def _action_id_value(action: Action) -> int:
    value = getattr(action, "value", action)
    if isinstance(value, tuple) and value:
        value = value[0]
    return int(cast(IntCoercible, value))


def _state_name(obs: Observation | None) -> str:
    state = getattr(obs, "state", None) if obs is not None else None
    return str(getattr(state, "name", state)).upper()


def _with_reset_advertised(observation: Observation) -> Observation:
    """
    Return the observation with RESET included in `available_actions`.

    The engine accepts RESET for every game on every turn (and the official
    docs require it at game over), so advertise it even when a game omits it.
    A copy is adjusted so the engine's own frame data is never mutated.
    """
    available = list(getattr(observation, "available_actions", None) or [])
    reset_action_id = _action_id_value(GameAction.RESET)
    if reset_action_id in available:
        return observation
    advertised = copy.copy(observation)
    advertised.available_actions = [*available, reset_action_id]
    return advertised


class TransitionRewardEnv:
    """
    ARC env wrapper with transition-only per-step rewards.

    Reward definition:
    - per-step reward is 0 unless a level transition occurs
    - transition reward is solved_levels / num_levels

    RESET is always available: every observation advertises it in
    `available_actions` and `action_space` always includes it. A RESET taken
    before any action in the current level is a harmless ignored no-op.
    """

    def __init__(self, env: Environment, *, game_id: str | None = None, seed: int | None = None) -> None:
        self._env = env
        self._requested_game_id = str(game_id or "").strip().lower()
        self._seed = seed

        self._obs: Observation | None = None
        self._game_id = self._requested_game_id
        self._num_levels = 0
        self._current_level = 0
        self._actions_in_level = 0
        self._episode_reward = 0.0

    @property
    def action_space(self) -> tuple[Action, ...]:
        reset_action_id = _action_id_value(GameAction.RESET)
        base_actions = [action for action in self._env.action_space if _action_id_value(action) != reset_action_id]
        base_actions.append(GameAction.RESET)
        return tuple(base_actions)

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def name(self) -> str:
        env_name = getattr(self._env, "name", None)
        if env_name:
            return str(env_name)
        return str(self._game_id)

    @property
    def episode_reward(self) -> float:
        return float(self._episode_reward)

    def reset(self) -> Observation:
        self._obs = self._env.reset()
        if self._obs is None:
            raise RuntimeError("Reset returned no observation.")
        self._obs = _with_reset_advertised(self._obs)

        observed_game_id = str(getattr(self._obs, "game_id", None) or "").strip().lower()
        self._game_id = observed_game_id or self._requested_game_id
        self._num_levels = _safe_int(getattr(self._obs, "win_levels", None), 0)
        self._current_level = _safe_int(getattr(self._obs, "levels_completed", None), 0)
        self._actions_in_level = 0
        self._episode_reward = 0.0
        return self._obs

    def _build_info(
        self, *, level_transition: bool, level_reset: bool = False, reset_ignored: bool = False
    ) -> StepInfo:
        return {
            "game_id": self._game_id,
            "level_transition": bool(level_transition),
            "level_reset": bool(level_reset),
            "reset_ignored": bool(reset_ignored),
            "current_level": int(self._current_level),
            "num_levels": int(max(1, self._num_levels)),
            "actions_in_level": int(self._actions_in_level),
            "episode_reward": float(self._episode_reward),
        }

    def _level_reset_observation(self) -> Observation:
        """
        Reset current level while preserving episode augmentation when available.
        """
        key = "ONLY_RESET_LEVELS"
        prev = os.environ.get(key)
        os.environ[key] = "true"
        try:
            reset_fn = cast(Callable[..., Observation | None], self._env.reset)
            try:
                return cast(Observation, reset_fn(preserve_transform=True))
            except TypeError:
                return cast(Observation, reset_fn())
        finally:
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev

    def _step_level_reset(self) -> StepResult:
        if self._actions_in_level <= 0 and _state_name(self._obs) != "GAME_OVER":
            obs = self._obs
            if obs is None:
                raise RuntimeError("Call reset() before step().")
            done = bool(is_terminal(obs))
            info = self._build_info(level_transition=False, level_reset=False, reset_ignored=True)
            return obs, 0.0, done, info

        obs = self._level_reset_observation()
        if obs is None:
            raise RuntimeError("Level reset returned no observation.")

        obs = _with_reset_advertised(obs)
        self._obs = obs
        self._current_level = _safe_int(getattr(obs, "levels_completed", None), self._current_level)
        self._actions_in_level = 0

        done = bool(is_terminal(obs))
        info = self._build_info(level_transition=False, level_reset=True, reset_ignored=False)
        return obs, 0.0, done, info

    def step(self, action: Action, data: ActionData | None = None) -> StepResult:
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")

        action_id = _action_id_value(action)
        if action_id == _action_id_value(GameAction.RESET):
            return self._step_level_reset()

        prev_level = _safe_int(getattr(self._obs, "levels_completed", None), self._current_level)
        step_result = self._env.step(action, data=data or {})
        obs, _, inner_done, _ = cast(ParsedStepResult, unpack_step_result(step_result))
        if obs is None:
            raise RuntimeError("Step returned no observation.")
        obs = _with_reset_advertised(obs)

        self._actions_in_level += 1
        new_level = _safe_int(getattr(obs, "levels_completed", None), prev_level)

        reward = 0.0
        level_transition = new_level > prev_level
        if level_transition:
            reward = self._compute_transition_reward(prev_level, new_level)
            self._actions_in_level = 0
        elif new_level < prev_level:
            self._actions_in_level = 0

        self._current_level = new_level
        self._obs = obs

        done = bool(inner_done or is_terminal(obs))
        is_win = _state_name(obs) == "WIN"

        if done and is_win:
            correction = max(0.0, 1.0 - (self._episode_reward + reward))
            reward += correction

        self._episode_reward += reward
        info = self._build_info(level_transition=bool(level_transition))
        return obs, float(reward), done, info

    def _compute_transition_reward(self, prev_level: int, new_level: int) -> float:
        num_levels = int(max(1, self._num_levels))
        solved_levels = max(0, int(new_level) - int(prev_level))
        return float(solved_levels) / float(num_levels)

    def __getattr__(self, name: str) -> object:
        # During copy/deepcopy, Python may probe attributes on a partially
        # constructed instance before __init__ has set _env.
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)
