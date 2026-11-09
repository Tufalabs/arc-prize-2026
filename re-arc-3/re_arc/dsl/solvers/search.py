from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from itertools import count

from .protocol import ExpandFn, GoalFn, HeuristicFn


def _reconstruct_actions[StateT, ActionT](
    goal_state: StateT, previous: dict[StateT, StateT | None], previous_action: dict[StateT, ActionT]
) -> list[ActionT]:
    actions: list[ActionT] = []
    cursor = goal_state
    while previous[cursor] is not None:
        actions.append(previous_action[cursor])
        cursor = previous[cursor]  # type: ignore[assignment]
    actions.reverse()
    return actions


def bfs_plan[StateT, ActionT](
    start_state: StateT,
    is_goal: GoalFn[StateT],
    expand: ExpandFn[ActionT, StateT],
    *,
    dominance_key: Callable[[StateT], tuple] | None = None,
    dominance_score: Callable[[StateT], float] | None = None,
) -> list[ActionT] | None:
    """Shortest-action BFS.

    Ignores transition costs and treats each edge as unit cost.
    """
    queue = deque([start_state])
    previous: dict[StateT, StateT | None] = {start_state: None}
    previous_action: dict[StateT, ActionT] = {}

    best_score: dict[tuple, float] | None = None
    if dominance_key is not None and dominance_score is not None:
        best_score = {dominance_key(start_state): dominance_score(start_state)}

    while queue:
        state = queue.popleft()
        if is_goal(state):
            return _reconstruct_actions(state, previous, previous_action)

        for action, next_state, _ in expand(state):
            if best_score is not None and dominance_key is not None and dominance_score is not None:
                key = dominance_key(next_state)
                score = dominance_score(next_state)
                prior = best_score.get(key)
                if prior is not None and prior >= score:
                    continue
                best_score[key] = score

            if next_state in previous:
                continue
            previous[next_state] = state
            previous_action[next_state] = action
            queue.append(next_state)

    return None


def dijkstra_plan[StateT, ActionT](
    start_state: StateT, is_goal: GoalFn[StateT], expand: ExpandFn[ActionT, StateT]
) -> list[ActionT] | None:
    """Lowest-total-cost plan for non-negative edge costs."""
    seq = count()
    frontier: list[tuple[float, int, StateT]] = [(0.0, next(seq), start_state)]
    distance: dict[StateT, float] = {start_state: 0.0}
    previous: dict[StateT, StateT | None] = {start_state: None}
    previous_action: dict[StateT, ActionT] = {}

    while frontier:
        cost_so_far, _ord, state = heapq.heappop(frontier)
        if cost_so_far > distance.get(state, float("inf")):
            continue
        if is_goal(state):
            return _reconstruct_actions(state, previous, previous_action)

        for action, next_state, step_cost in expand(state):
            if step_cost < 0:
                raise ValueError("dijkstra_plan requires non-negative costs.")
            new_cost = cost_so_far + float(step_cost)
            if new_cost >= distance.get(next_state, float("inf")):
                continue
            distance[next_state] = new_cost
            previous[next_state] = state
            previous_action[next_state] = action
            heapq.heappush(frontier, (new_cost, next(seq), next_state))

    return None


def astar_plan[StateT, ActionT](
    start_state: StateT, is_goal: GoalFn[StateT], expand: ExpandFn[ActionT, StateT], heuristic: HeuristicFn[StateT]
) -> list[ActionT] | None:
    """A* search with non-negative step costs."""
    seq = count()
    g_score: dict[StateT, float] = {start_state: 0.0}
    previous: dict[StateT, StateT | None] = {start_state: None}
    previous_action: dict[StateT, ActionT] = {}

    start_f = float(heuristic(start_state))
    frontier: list[tuple[float, int, StateT]] = [(start_f, next(seq), start_state)]

    while frontier:
        _f_score, _ord, state = heapq.heappop(frontier)
        if is_goal(state):
            return _reconstruct_actions(state, previous, previous_action)

        current_g = g_score.get(state, float("inf"))
        for action, next_state, step_cost in expand(state):
            if step_cost < 0:
                raise ValueError("astar_plan requires non-negative costs.")
            tentative_g = current_g + float(step_cost)
            if tentative_g >= g_score.get(next_state, float("inf")):
                continue
            g_score[next_state] = tentative_g
            previous[next_state] = state
            previous_action[next_state] = action
            next_f = tentative_g + float(heuristic(next_state))
            heapq.heappush(frontier, (next_f, next(seq), next_state))

    return None


def beam_search[StateT, ActionT](
    start_state: StateT,
    is_goal: GoalFn[StateT],
    expand: ExpandFn[ActionT, StateT],
    heuristic: HeuristicFn[StateT],
    *,
    width: int,
    max_depth: int,
) -> list[ActionT] | None:
    """Heuristic beam search for wide spaces where exact search is too expensive."""
    if width <= 0:
        raise ValueError("beam_search width must be > 0.")
    if max_depth < 0:
        raise ValueError("beam_search max_depth must be >= 0.")

    if is_goal(start_state):
        return []

    beam: list[StateT] = [start_state]
    previous: dict[StateT, StateT | None] = {start_state: None}
    previous_action: dict[StateT, ActionT] = {}

    for _ in range(max_depth):
        candidates: list[tuple[float, StateT, StateT, ActionT]] = []
        for state in beam:
            for action, next_state, _ in expand(state):
                if next_state not in previous:
                    previous[next_state] = state
                    previous_action[next_state] = action
                candidates.append((float(heuristic(next_state)), next_state, state, action))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        beam = []
        seen: set[StateT] = set()
        for _score, next_state, _parent, _action in candidates:
            if next_state in seen:
                continue
            if is_goal(next_state):
                return _reconstruct_actions(next_state, previous, previous_action)
            seen.add(next_state)
            beam.append(next_state)
            if len(beam) >= width:
                break

    return None
