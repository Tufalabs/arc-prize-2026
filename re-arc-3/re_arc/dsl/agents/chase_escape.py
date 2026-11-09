from __future__ import annotations

import heapq
from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent

_game_mod = import_module("re_arc.environment_files.chase_escape.0001.chaseescape")
ACTION_TO_DELTA = _game_mod.ACTION_TO_DELTA
LEVEL_MODELS = _game_mod.LEVEL_MODELS
SPACE_ACTION = _game_mod.SPACE_ACTION
TERMINAL_LOSE = _game_mod.TERMINAL_LOSE
TERMINAL_WIN = _game_mod.TERMINAL_WIN
SimState = _game_mod.SimState
initial_state = _game_mod.initial_state
simulate_step = _game_mod.simulate_step

MAX_EXPANSIONS = 220000


def _state_search_key(state: SimState) -> tuple:
    return (
        state.player,
        state.chaser,
        state.crates,
        state.switch_open_mask,
        state.locked_open_mask,
        state.has_key,
        state.key_mask,
        state.hourglass_mask,
        state.switch_overlap_mask,
        state.terminal_mode,
    )


def _heuristic(level_model, state: SimState) -> int:
    px, py = state.player
    ex, ey = level_model.exit_rect.x, level_model.exit_rect.y
    score = abs(px - ex) + abs(py - ey)

    if level_model.locked_gates and not state.has_key:
        best = None
        for idx, key_rect in enumerate(level_model.keys):
            if not (state.key_mask & (1 << idx)):
                continue
            dist = abs(px - key_rect.x) + abs(py - key_rect.y)
            best = dist if best is None else min(best, dist)
        if best is not None:
            score += best + 3

    closed_switches = 0
    for gate_idx in range(len(level_model.switch_gates)):
        if not (state.switch_open_mask & (1 << gate_idx)):
            closed_switches += 1
    if closed_switches > 0 and level_model.switches:
        nearest = min(abs(px - sw.x) + abs(py - sw.y) for sw in level_model.switches)
        score += nearest // 2

    return int(score)


def _plan_for_level(level_model) -> list[int]:
    start = initial_state(level_model)

    frontier: list[tuple[int, int, int, SimState]] = []
    counter = 0
    heapq.heappush(frontier, (_heuristic(level_model, start), 0, counter, start))

    prev: dict[SimState, tuple[SimState, int] | None] = {start: None}
    best_cost: dict[SimState, int] = {start: 0}

    best_time_by_key: dict[tuple, int] = {_state_search_key(start): start.time_remaining}

    expansions = 0
    goal: SimState | None = None

    actions = tuple(sorted((*ACTION_TO_DELTA.keys(), SPACE_ACTION)))

    while frontier and expansions < MAX_EXPANSIONS:
        _, g, _, state = heapq.heappop(frontier)
        if best_cost.get(state, 10**9) != g:
            continue

        if state.terminal_mode == TERMINAL_WIN:
            goal = state
            break

        expansions += 1

        for action_id in actions:
            next_state = simulate_step(level_model, state, action_id).state
            if next_state.terminal_mode == TERMINAL_LOSE:
                continue

            new_g = g + 1
            existing = best_cost.get(next_state)
            if existing is not None and existing <= new_g:
                continue

            key = _state_search_key(next_state)
            prev_best_time = best_time_by_key.get(key)
            if prev_best_time is not None and prev_best_time >= next_state.time_remaining:
                continue
            best_time_by_key[key] = next_state.time_remaining

            best_cost[next_state] = new_g
            prev[next_state] = (state, int(action_id))
            counter += 1
            heapq.heappush(frontier, (new_g + _heuristic(level_model, next_state), new_g, counter, next_state))

    if goal is None:
        raise RuntimeError(
            f"Failed to find chase_escape plan for level '{level_model.name}' after {expansions} expansions."
        )

    plan: list[int] = []
    cursor = goal
    while prev[cursor] is not None:
        parent, action_id = prev[cursor]
        plan.append(int(action_id))
        cursor = parent
    plan.reverse()

    # WIN freezes until space (or 20 steps); append space to advance immediately.
    plan.append(int(SPACE_ACTION))
    return plan


class ChaseEscapeDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_MODELS))
        self._planning_level_idx = 0

    def _on_new_level(self, env, level_idx: int):
        self._planning_level_idx = int(level_idx)
        super()._on_new_level(env, level_idx)

    def _build_level_program(self, env):
        _ = env
        idx = max(0, min(int(self._planning_level_idx), len(LEVEL_MODELS) - 1))
        actions = _plan_for_level(LEVEL_MODELS[idx])
        return [(int(action_id), {}) for action_id in actions]


AGENT_CLASS = ChaseEscapeDslAgent
