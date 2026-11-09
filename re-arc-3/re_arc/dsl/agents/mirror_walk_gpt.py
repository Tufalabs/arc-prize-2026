from __future__ import annotations

from collections import deque

from ..core import DslAgent, observation_level_index

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

MOVE_DELTAS = {UP: ((-1, 0), (-1, 0)), DOWN: ((1, 0), (1, 0)), LEFT: ((0, -1), (0, 1)), RIGHT: ((0, 1), (0, -1))}

LEVELS = [
    {
        "left_start": (12, 2),
        "right_start": (12, 13),
        "exit_cell": (5, 5),
        "button_cell": None,
        "budget": 30,
        "walls": frozenset(
            {
                *((row, col) for row in range(15) for col in range(16) if row in {0, 14} or col in {0, 15}),
                *((row, col) for row in range(15) for col in (7, 8)),
                (9, 12),
                (10, 12),
                (11, 12),
                (12, 12),
                (13, 12),
            }
        ),
    },
    {
        "left_start": (12, 2),
        "right_start": (12, 13),
        "exit_cell": (6, 6),
        "button_cell": (10, 11),
        "budget": 30,
        "walls": frozenset(
            {
                *((row, col) for row in range(15) for col in range(16) if row in {0, 14} or col in {0, 15}),
                *((row, col) for row in range(15) for col in (7, 8)),
                (10, 10),
                (9, 11),
            }
        ),
    },
    {
        "left_start": (12, 3),
        "right_start": (12, 12),
        "exit_cell": (4, 4),
        "button_cell": (8, 10),
        "budget": 30,
        "walls": frozenset(
            {
                *((row, col) for row in range(15) for col in range(16) if row in {0, 14} or col in {0, 15}),
                *((row, col) for row in range(15) for col in (7, 8)),
                (9, 5),
                (10, 5),
                (11, 5),
                (12, 5),
                (8, 9),
                (7, 10),
            }
        ),
    },
]


def _button_pressed(level: dict, right_pos: tuple[int, int]) -> bool:
    return level["button_cell"] is not None and right_pos == level["button_cell"]


def _exit_open(level: dict, right_pos: tuple[int, int]) -> bool:
    return level["button_cell"] is None or _button_pressed(level, right_pos)


def _passable(level: dict, cell: tuple[int, int], *, exit_open: bool) -> bool:
    if cell in level["walls"]:
        return False
    if cell == level["exit_cell"] and not exit_open:
        return False
    row, col = cell
    return 0 <= row < 15 and 0 <= col < 16


def _advance(
    level: dict, state: tuple[int, int, int, int, int], action_id: int
) -> tuple[int, int, int, int, int] | None:
    left_r, left_c, right_r, right_c, budget = state
    if budget <= 0:
        return None

    left_delta, right_delta = MOVE_DELTAS[action_id]
    left_pos = (left_r, left_c)
    right_pos = (right_r, right_c)
    prior_exit_open = _exit_open(level, right_pos)

    left_target = (left_r + left_delta[0], left_c + left_delta[1])
    right_target = (right_r + right_delta[0], right_c + right_delta[1])
    if _passable(level, left_target, exit_open=prior_exit_open):
        left_pos = left_target
    if _passable(level, right_target, exit_open=prior_exit_open):
        right_pos = right_target

    return (left_pos[0], left_pos[1], right_pos[0], right_pos[1], budget - 1)


def _is_goal(level: dict, state: tuple[int, int, int, int, int]) -> bool:
    left_pos = (state[0], state[1])
    right_pos = (state[2], state[3])
    return left_pos == level["exit_cell"] and _exit_open(level, right_pos)


def _shortest_plan(level: dict) -> list[int]:
    start = (*level["left_start"], *level["right_start"], int(level["budget"]))
    queue = deque([start])
    prev: dict[tuple[int, int, int, int, int], tuple[int, int, int, int, int] | None] = {start: None}
    prev_action: dict[tuple[int, int, int, int, int], int] = {}

    goal = None
    while queue:
        state = queue.popleft()
        if _is_goal(level, state):
            goal = state
            break
        for action_id in (UP, DOWN, LEFT, RIGHT):
            next_state = _advance(level, state, action_id)
            if next_state is None or next_state in prev:
                continue
            prev[next_state] = state
            prev_action[next_state] = action_id
            queue.append(next_state)

    if goal is None:
        raise RuntimeError("Mirror Walk GPT BFS failed to find a winning plan.")

    actions: list[int] = []
    cursor = goal
    while prev[cursor] is not None:
        actions.append(prev_action[cursor])
        cursor = prev[cursor]
    actions.reverse()
    return actions


LEVEL_PROGRAMS = [_shortest_plan(level) for level in LEVELS]
EXPECTED_LENGTHS = [10, 10, 10]
for level_idx, expected in enumerate(EXPECTED_LENGTHS):
    actual = len(LEVEL_PROGRAMS[level_idx])
    if actual != expected:
        raise RuntimeError(
            f"Mirror Walk GPT regression: level {level_idx + 1} expected shortest length {expected}, found {actual}."
        )


class MirrorWalkGptDslAgent(DslAgent):
    def __init__(self, game_id: str = "mirror_walk_gpt-0001"):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Mirror Walk GPT observation is missing `levels_completed`.")

        self.mark_levels_solved(level_idx)
        reset_level = bool(getattr(observation, "full_reset", False))

        if self._current_level_idx is None or level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
        elif reset_level and self._action_idx > 0:
            self._action_idx = 0

        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"Mirror Walk GPT DSL program exhausted before level advance. level={self._current_level_idx}"
            )

        action_id = int(program[self._action_idx])
        self._action_idx += 1
        return action_id, {}


AGENT_CLASS = MirrorWalkGptDslAgent
