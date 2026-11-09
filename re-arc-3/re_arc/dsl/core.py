from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import pkgutil
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import Any


def find_shortest_action_plan(
    start_state: Any,
    is_goal: Callable[[tuple[Any, ...]], bool],
    expand: Callable[[tuple[Any, ...]], Iterable[tuple[int, tuple[Any, ...] | None]]],
    dominance_key: Callable[[tuple[Any, ...]], tuple[Any, ...]],
    dominance_score: Callable[[tuple[Any, ...]], int],
) -> Any:
    """
    Breadth-first plan search with dominance pruning.

    If two states have the same dominance key, only keep the one with the higher
    dominance score.
    """
    queue = deque([start_state])
    previous = {start_state: None}
    previous_action = {}
    best_score = {dominance_key(start_state): dominance_score(start_state)}
    goal_state = None

    while queue:
        state = queue.popleft()
        if is_goal(state):
            goal_state = state
            break

        for action_id, next_state in expand(state):
            if next_state is None:
                continue

            key = dominance_key(next_state)
            score = dominance_score(next_state)
            prior = best_score.get(key)
            if prior is not None and prior >= score:
                continue
            best_score[key] = score

            if next_state not in previous:
                previous[next_state] = state
                previous_action[next_state] = action_id
                queue.append(next_state)

    if goal_state is None:
        return None

    actions = []
    cursor = goal_state
    while previous[cursor] is not None:
        actions.append(previous_action[cursor])
        cursor = previous[cursor]
    actions.reverse()
    return actions


MOVE_ACTION_BY_DELTA = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}

DELTA_BY_MOVE_ACTION = {action: delta for delta, action in MOVE_ACTION_BY_DELTA.items()}


def observation_level_index(observation: Any, total_levels: int | None = None) -> Any:
    raw = getattr(observation, "levels_completed", None)
    if raw is None:
        return None
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        return None
    if idx < 0:
        idx = 0
    if total_levels is not None and total_levels > 0:
        idx = min(idx, total_levels - 1)
    return idx


def camera_grid_to_display(camera: Any, gx: int, gy: int, display_size: int = 64) -> Any:
    scale_x = int(display_size / camera.width)
    scale_y = int(display_size / camera.height)
    scale = min(scale_x, scale_y)
    x_padding = int((display_size - (camera.width * scale)) / 2)
    y_padding = int((display_size - (camera.height * scale)) / 2)
    dx = int((gx - camera.x) * scale + x_padding)
    dy = int((gy - camera.y) * scale + y_padding)
    return dx, dy


def iter_frame_layers(observation: Any) -> Any:
    frame = getattr(observation, "frame", None)
    if frame is None:
        return []
    if isinstance(frame, list):
        return [layer for layer in frame if layer is not None]
    return [frame]


def layer_to_grid(layer: Any) -> Any:
    if hasattr(layer, "tolist"):
        return layer.tolist()
    return layer


def frame_signature(observation: Any) -> Any:
    layers = iter_frame_layers(observation)
    hasher = hashlib.sha1()
    hasher.update(f"layers={len(layers)}".encode())
    for idx, layer in enumerate(layers):
        hasher.update(f"|{idx}|".encode())
        hasher.update(json.dumps(layer_to_grid(layer), separators=(",", ":")).encode("utf-8"))
    return hasher.hexdigest()


def state_name(observation: Any) -> Any:
    state = getattr(observation, "state", None)
    return getattr(state, "name", str(state))


def is_terminal(observation: Any) -> Any:
    name = state_name(observation)
    return name in {"WIN", "GAME_OVER"}


def unpack_step_result(step_result: Any) -> Any:
    if isinstance(step_result, tuple) and len(step_result) >= 4:
        observation = step_result[0]
        reward = float(step_result[1] or 0.0)
        done = bool(step_result[2])
        info = step_result[3] if isinstance(step_result[3], dict) else {}
        return observation, reward, done, info
    observation = step_result
    return observation, 0.0, is_terminal(observation), {}


def resolve_action(env: Any, action_id: int) -> Any:
    for action in env.action_space:
        if int(getattr(action, "value", -1)) == int(action_id):
            return action
    raise ValueError(f"Action id {action_id} is not available for this environment.")


@dataclass
class ObservationHistory:
    frame_signatures: list[str] = field(default_factory=list)
    state_names: list[str] = field(default_factory=list)
    action_ids: list[int] = field(default_factory=list)

    def clear(self: Any) -> Any:
        self.frame_signatures.clear()
        self.state_names.clear()
        self.action_ids.clear()


class DslAgent:
    def __init__(self, game_id: str, total_levels: int | None = None):
        self.game_id = game_id
        self.total_levels = total_levels
        self.solved_levels = 0
        self.history = ObservationHistory()

    def reset_episode(self: Any) -> Any:
        self.solved_levels = 0
        self.history.clear()

    def observe(self: Any, observation: Any) -> Any:
        self.history.frame_signatures.append(frame_signature(observation))
        self.history.state_names.append(state_name(observation))
        levels_completed = getattr(observation, "levels_completed", None)
        if levels_completed is not None:
            with contextlib.suppress(TypeError, ValueError):
                self.mark_levels_solved(int(levels_completed))
        if state_name(observation) == "WIN" and self.total_levels is not None:
            self.solved_levels = self.total_levels

    def record_action(self: Any, action_id: int) -> Any:
        self.history.action_ids.append(int(action_id))

    def mark_levels_solved(self: Any, solved_levels: int) -> Any:
        solved = int(solved_levels)
        if self.total_levels is not None:
            solved = max(0, min(solved, self.total_levels))
        self.solved_levels = max(self.solved_levels, solved)

    def normalized_score(self: Any, observation: Any = None) -> Any:
        if observation is not None and state_name(observation) == "WIN":
            return 1.0
        if self.total_levels is None or self.total_levels <= 0:
            return 0.0
        return float(self.solved_levels) / float(self.total_levels)

    def next_action(self: Any, env: Any, observation: Any) -> Any:
        raise NotImplementedError


class CachedProgramDslAgent(DslAgent):
    """
    Base class for DSL agents that execute a cached per-level action program.

    Subclasses should only implement `_build_level_program(env)`.
    """

    def __init__(self, game_id: str, total_levels: int | None = None):
        super().__init__(game_id=game_id, total_levels=total_levels)
        self._current_level_idx: int | None = None
        self._action_idx = 0
        self._programs: dict[int, list[tuple[int, dict[str, int]]]] = {}

    @property
    def _agent_tag(self) -> str:
        return self.game_id.split("-", 1)[0].lower()

    def reset_episode(self: Any) -> Any:
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def _level_index(self: Any, observation: Any) -> int | None:
        return observation_level_index(observation, self.total_levels)

    def _on_new_level(self: Any, env: Any, level_idx: int) -> Any:
        if level_idx not in self._programs:
            self._programs[level_idx] = self._build_level_program(env)

    def _sync_level(self: Any, env: Any, observation: Any) -> Any:
        level_idx = self._level_index(observation)
        if level_idx is None:
            return

        self.mark_levels_solved(level_idx)
        reset_level = bool(getattr(observation, "full_reset", False))

        if self._current_level_idx is None or level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
            self._on_new_level(env, level_idx)
            return

        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def _missing_level_error(self) -> RuntimeError:
        return RuntimeError(f"Missing `levels_completed` in {self._agent_tag} observation.")

    def _program_exhausted_error(self: Any, level_idx: int, program: list[tuple[int, dict[str, int]]]) -> Any:
        return RuntimeError(
            f"{self._agent_tag} DSL program exhausted before level advance level={level_idx} steps={len(program)}"
        )

    def _build_level_program(self: Any, env: Any) -> list[tuple[int, dict[str, int]]]:
        raise NotImplementedError

    def next_action(self: Any, env: Any, observation: Any) -> Any:
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise self._missing_level_error()

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise self._program_exhausted_error(self._current_level_idx, program)

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data


def create_dsl_agent(game_id: str) -> Any:
    game_key = (game_id or "").strip().lower()
    if not game_key:
        raise ValueError("game_id is required to resolve a DSL agent.")
    game_base = game_key.split("-", 1)[0]
    try:
        agent_cls = _resolve_agent_class(game_base)
    except ModuleNotFoundError as exc:
        available = ", ".join(_discover_agent_modules()) or "(none)"
        raise ValueError(
            f"No DSL agent is implemented for `{game_id}` yet. Available agent modules: {available}."
        ) from exc
    except ValueError as exc:
        raise ValueError(f"Invalid DSL agent module for `{game_id}`: {exc}") from exc
    return agent_cls(game_id)


def resolve_agent_class(game_id: str) -> Any:
    """Resolve a game's DSL agent class without instantiating it.

    Raises ModuleNotFoundError if the game has no agent module, or ValueError if
    the module exists but defines no valid agent.
    """
    game_base = (game_id or "").strip().lower().split("-", 1)[0]
    return _resolve_agent_class(game_base)


@lru_cache(maxsize=1)
def _discover_agent_modules() -> tuple[str, ...]:
    agents_pkg = importlib.import_module("re_arc.dsl.agents")
    names = [module.name for module in pkgutil.iter_modules(agents_pkg.__path__) if not module.name.startswith("_")]
    return tuple(sorted(names))


@cache
def _resolve_agent_class(game_base: str) -> Any:
    module_name = f"re_arc.dsl.agents.{game_base}"
    module = importlib.import_module(module_name)

    explicit = getattr(module, "AGENT_CLASS", None)
    if explicit is not None:
        if not isinstance(explicit, type) or not issubclass(explicit, DslAgent):
            raise ValueError(f"{module_name}.AGENT_CLASS must be a DslAgent subclass.")
        return explicit

    candidates = []
    for obj in vars(module).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, DslAgent)
            and obj is not DslAgent
            and obj.__module__ == module.__name__
        ):
            candidates.append(obj)
    if not candidates:
        raise ValueError(f"{module_name} defines no DslAgent subclass.")
    if len(candidates) > 1:
        names = ", ".join(sorted(cls.__name__ for cls in candidates))
        raise ValueError(f"{module_name} is ambiguous ({names}); define AGENT_CLASS.")
    return candidates[0]


def run_dsl_episode(
    env: Any,
    agent: DslAgent,
    max_actions: int,
    initial_observation: Any = None,
    observation_callback: Callable[[object], None] | None = None,
    step_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    if max_actions <= 0:
        raise ValueError("max_actions must be > 0.")

    agent.reset_episode()
    observation = initial_observation if initial_observation is not None else env.reset()
    agent.observe(observation)
    if observation_callback is not None:
        observation_callback(observation)

    actions = 0
    episode_reward = 0.0
    while actions < max_actions and not is_terminal(observation):
        previous_observation = observation
        action_id, action_data = agent.next_action(env, observation)
        agent.record_action(action_id)
        action = resolve_action(env, action_id)
        payload = dict(action_data or {})
        observation, reward, done, info = unpack_step_result(env.step(action, data=payload))
        actions += 1
        episode_reward += reward
        agent.observe(observation)
        if observation_callback is not None:
            observation_callback(observation)
        if step_callback is not None:
            step_callback(
                {
                    "index": actions,
                    "action_id": int(action_id),
                    "action_name": getattr(action, "name", str(action)),
                    "action_data": payload,
                    "reward": float(reward),
                    "done": bool(done),
                    "info": info,
                    "episode_reward": float(episode_reward),
                    "previous_observation": previous_observation,
                    "observation": observation,
                }
            )
        if done:
            break

    return {
        "observation": observation,
        "actions": actions,
        "state": state_name(observation),
        "normalized_score": agent.normalized_score(observation),
        "episode_reward": episode_reward,
        "solved_levels": agent.solved_levels,
        "total_levels": agent.total_levels,
    }
