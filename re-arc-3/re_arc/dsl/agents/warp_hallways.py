from __future__ import annotations

from collections import deque

from ..core import CachedProgramDslAgent

MOVE_ACTIONS = ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0)))

LEVEL_ROWS = (
    (
        "############",
        "#S...###...#",
        "###.####A###",
        "#...####.###",
        "#.#####..###",
        "#...A##..G##",
        "#######...##",
        "############",
        "############",
        "############",
        "############",
        "############",
    ),
    (
        "############",
        "#S...###B..#",
        "#.######.###",
        "#...####...#",
        "###.########",
        "#...A###A..#",
        "###.####.###",
        "#...B###...#",
        "##########.#",
        "########..G#",
        "########..##",
        "############",
    ),
    (
        "############",
        "############",
        "#######A..##",
        "#########.##",
        "#...A####.G#",
        "#.####....##",
        "#...##.##.##",
        "#.#.##....##",
        "#.#B##.#####",
        "#S####...B##",
        "############",
        "############",
    ),
)


def _build_program(rows: tuple[str, ...]) -> list[tuple[int, dict[str, int]]]:
    height = len(rows)
    width = len(rows[0])
    start = (-1, -1)
    goal = (-1, -1)
    portal_cells: dict[str, list[tuple[int, int]]] = {}

    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            if tile == "S":
                start = (x, y)
            elif tile == "G":
                goal = (x, y)
            elif tile in {"A", "B"}:
                portal_cells.setdefault(tile, []).append((x, y))

    portal_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for cells in portal_cells.values():
        first, second = cells
        portal_lookup[first] = second
        portal_lookup[second] = first

    queue: deque[tuple[int, int]] = deque([start])
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    previous_action: dict[tuple[int, int], int] = {}

    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for action_id, (dx, dy) in MOVE_ACTIONS:
            next_x = cell[0] + dx
            next_y = cell[1] + dy
            if 0 <= next_x < width and 0 <= next_y < height and rows[next_y][next_x] != "#":
                next_cell = (next_x, next_y)
                if next_cell in portal_lookup:
                    next_cell = portal_lookup[next_cell]
            else:
                next_cell = cell

            if next_cell in previous:
                continue
            previous[next_cell] = cell
            previous_action[next_cell] = action_id
            queue.append(next_cell)

    if goal not in previous:
        raise RuntimeError("Warp Hallways level is unsolvable.")

    actions: list[int] = []
    cursor = goal
    while previous[cursor] is not None:
        actions.append(previous_action[cursor])
        cursor = previous[cursor]
    actions.reverse()
    return [(action_id, {}) for action_id in actions]


class WarpHallwaysDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str) -> None:
        super().__init__(game_id=game_id, total_levels=3)

    def _on_new_level(self, _env, level_idx: int):
        if level_idx not in self._programs:
            self._programs[level_idx] = _build_program(LEVEL_ROWS[level_idx])

    def _build_level_program(self, _env):
        raise NotImplementedError


AGENT_CLASS = WarpHallwaysDslAgent
