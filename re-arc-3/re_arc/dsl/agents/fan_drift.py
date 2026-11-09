from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module

from ..core import DslAgent, find_shortest_action_plan, observation_level_index

_env_module = import_module("re_arc.environment_files.fan_drift.0001.fandrift")
LEVEL_SPECS = _env_module.LEVEL_SPECS
LevelSpec = _env_module.LevelSpec
LevelState = _env_module.LevelState
fan_click_action_data = _env_module.fan_click_action_data
initial_level_state = _env_module.initial_level_state
simulate_level_step = _env_module.simulate_level_step


def _dominance_key(state: tuple[int, tuple[tuple[int, int], ...], int]) -> tuple[int, tuple[tuple[int, int], ...], int]:
    return state


def _dominance_score(state: tuple[int, tuple[tuple[int, int], ...], int, int]) -> int:
    return int(state[3])


def _goal(state: tuple[int, tuple[tuple[int, int], ...], int, int], level_spec: LevelSpec) -> bool:
    return int(state[2]) == (1 << len(level_spec.bins)) - 1


def _expand(
    state: tuple[int, tuple[tuple[int, int], ...], int, int], level_spec: LevelSpec
) -> Iterable[tuple[int, tuple[int, tuple[tuple[int, int], ...], int, int] | None]]:
    fan_mask, leaves, filled_mask, remaining = state
    if remaining <= 0:
        return []

    current = LevelState(fan_mask=fan_mask, leaves=leaves, filled_mask=filled_mask, remaining_budget=remaining)
    out: list[tuple[int, tuple[int, tuple[tuple[int, int], ...], int, int] | None]] = []

    waited, solved, failed = simulate_level_step(level_spec, current, "wait")
    out.append(
        (
            5,
            None
            if failed and not solved
            else (waited.fan_mask, waited.leaves, waited.filled_mask, waited.remaining_budget),
        )
    )

    for fan_index in range(len(level_spec.fans)):
        nxt, solved, failed = simulate_level_step(level_spec, current, f"fan:{fan_index}")
        out.append(
            (
                6 + fan_index,
                None if failed and not solved else (nxt.fan_mask, nxt.leaves, nxt.filled_mask, nxt.remaining_budget),
            )
        )
    return out


def _solve_level(level_spec: LevelSpec) -> list[tuple[int, dict[str, int]]]:
    start = initial_level_state(level_spec)
    start_state = (start.fan_mask, start.leaves, start.filled_mask, start.remaining_budget)
    plan = find_shortest_action_plan(
        start_state=start_state,
        is_goal=lambda state: _goal(state, level_spec),
        expand=lambda state: _expand(state, level_spec),
        dominance_key=lambda state: _dominance_key((state[0], state[1], state[2])),
        dominance_score=_dominance_score,
    )
    if plan is None:
        raise RuntimeError(f"Fan Drift DSL could not solve {level_spec.name}.")

    program: list[tuple[int, dict[str, int]]] = []
    for action in plan:
        if action == 5:
            program.append((5, {}))
            continue
        fan_index = action - 6
        fan = level_spec.fans[fan_index]
        program.append((6, fan_click_action_data(fan)))
    return program


class FanDriftDslAgent(DslAgent):
    def __init__(self, game_id: str = "fan_drift-0001"):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_SPECS))
        self._current_level_idx: int | None = None
        self._action_idx = 0
        self._programs = {index: _solve_level(level_spec) for index, level_spec in enumerate(LEVEL_SPECS)}

    def reset_episode(self) -> None:
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Fan Drift DSL could not determine the current level.")

        self.mark_levels_solved(level_idx)
        if self._current_level_idx is None or level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
        elif bool(getattr(observation, "full_reset", False)) and self._action_idx > 0:
            self._action_idx = 0

        program = self._programs[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"Fan Drift DSL exhausted its program on level {level_idx}.")

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1

        if action_id != 6:
            return action_id, action_data

        return 6, dict(action_data)


AGENT_CLASS = FanDriftDslAgent
