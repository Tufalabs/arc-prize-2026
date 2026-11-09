from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from importlib import import_module

from ..core import CachedProgramDslAgent


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


def _bfs_plan[StateT, ActionT](
    start_state: StateT,
    is_goal: Callable[[StateT], bool],
    expand: Callable[[StateT], Iterable[tuple[ActionT, StateT, float]]],
    *,
    dominance_key: Callable[[StateT], tuple] | None = None,
    dominance_score: Callable[[StateT], float] | None = None,
) -> list[ActionT] | None:
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


_ENV_MOD = import_module("re_arc.environment_files.maze_runner_reach_the_goal.0001.mazerunnerreachthegoal")

_deserialize_model = _ENV_MOD._deserialize_model
_rects_overlap = _ENV_MOD._rects_overlap
ACTION_SPACE = _ENV_MOD.ACTION_SPACE
WIN_ANIM_STEPS = _ENV_MOD.WIN_ANIM_STEPS
apply_action_transition = _ENV_MOD.apply_action_transition
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model


class MazeRunnerReachTheGoalDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)
        goal_x, goal_y = model["goal"]

        start_state = initial_search_state_from_model(model)

        def is_goal(state: tuple[int, int, int, int, int, int]) -> bool:
            px, py = int(state[0]), int(state[1])
            return _rects_overlap((px, py, 2, 2), (goal_x, goal_y, 2, 2))

        def expand(state: tuple[int, int, int, int, int, int]):
            for action_id in (1, 2, 3, 4, 5):
                next_state, won = apply_action_transition(model, state, action_id)
                if next_state is None:
                    continue
                yield action_id, next_state, 1.0
                if won:
                    continue

        def dominance_key(state: tuple[int, int, int, int, int, int]) -> tuple[int, int, int]:
            px, py, _time, tick, _walk, _bump = state
            return int(px), int(py), int(tick) % 2

        def dominance_score(state: tuple[int, int, int, int, int, int]) -> float:
            return float(state[2])

        move_plan = _bfs_plan(
            start_state, is_goal, expand, dominance_key=dominance_key, dominance_score=dominance_score
        )
        if move_plan is None:
            raise RuntimeError("maze_runner_reach_the_goal DSL could not solve the current level.")

        program = [(int(action_id), {}) for action_id in move_plan]
        program.extend((int(ACTION_SPACE), {}) for _ in range(int(WIN_ANIM_STEPS)))
        return program


AGENT_CLASS = MazeRunnerReachTheGoalDslAgent
