from __future__ import annotations

from collections import deque

from ..core import DslAgent, observation_level_index

LEVEL_MAPS = (
    (
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
        "..S.....E.....",
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
    ),
    (
        "E....#########",
        ".....#########",
        ".....#########",
        ".....#########",
        ".........#####",
        "####......####",
        "#####.....####",
        "#####.....####",
        "#####........#",
        "########......",
        "#########.....",
        "#########.....",
        "#########....S",
        "#########.....",
    ),
    (
        "#########....E",
        "#########.....",
        "#########.....",
        "#########.....",
        "#########.....",
        "#########..###",
        "#########..###",
        "#####.....####",
        "#####..#..####",
        "####......####",
        ".....#########",
        "....##########",
        "....##########",
        "S...##########",
    ),
)

MOVE_DELTAS = ((0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4))


def _shortest_program(rows: tuple[str, ...]) -> list[tuple[int, dict[str, int]]]:
    start: tuple[int, int] | None = None
    exit_pos: tuple[int, int] | None = None
    open_cells: set[tuple[int, int]] = set()

    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            if tile != "#":
                open_cells.add((x, y))
            if tile == "S":
                start = (x, y)
            elif tile == "E":
                exit_pos = (x, y)

    if start is None or exit_pos is None:
        raise ValueError("Each level must have exactly one start and one exit.")

    frontier = deque([start])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    move_taken: dict[tuple[int, int], int] = {}

    while frontier:
        current = frontier.popleft()
        if current == exit_pos:
            break
        current_x, current_y = current
        for delta_x, delta_y, action_id in MOVE_DELTAS:
            nxt = (current_x + delta_x, current_y + delta_y)
            if nxt not in open_cells or nxt in previous:
                continue
            previous[nxt] = current
            move_taken[nxt] = action_id
            frontier.append(nxt)

    if exit_pos not in previous:
        raise RuntimeError("Shadow Step DSL could not find a valid route to the exit.")

    actions: list[int] = []
    cursor = exit_pos
    while cursor != start:
        actions.append(move_taken[cursor])
        cursor = previous[cursor]
        if cursor is None:
            raise RuntimeError("Shadow Step DSL route reconstruction failed.")
    actions.reverse()
    return [(action_id, {}) for action_id in actions]


PROGRAMS = {level_idx: _shortest_program(rows) for level_idx, rows in enumerate(LEVEL_MAPS)}


class ShadowStepDslAgent(DslAgent):
    def __init__(self, game_id: str = "shadow_step-0001"):
        super().__init__(game_id=game_id, total_levels=3)
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, env, observation):
        del env
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Shadow Step DSL requires `levels_completed` in the observation.")

        self.mark_levels_solved(level_idx)
        if level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0

        program = PROGRAMS[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"Shadow Step DSL program exhausted on level {level_idx}.")

        action = program[self._action_idx]
        self._action_idx += 1
        return action


AGENT_CLASS = ShadowStepDslAgent
