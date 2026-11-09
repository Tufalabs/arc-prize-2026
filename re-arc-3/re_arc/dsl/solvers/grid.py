from __future__ import annotations

from collections.abc import Iterable

from ..core import MOVE_ACTION_BY_DELTA, camera_grid_to_display
from .search import bfs_plan

GridPos = tuple[int, int]


def in_bounds(pos: GridPos, width: int, height: int) -> bool:
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def cardinal_neighbors(pos: GridPos) -> list[GridPos]:
    x, y = pos
    return [(x + dx, y + dy) for dx, dy in MOVE_ACTION_BY_DELTA]


def passable_neighbors(pos: GridPos, *, width: int, height: int, blocked: set[GridPos] | None = None) -> list[GridPos]:
    blocked = blocked or set()
    out: list[GridPos] = []
    for npos in cardinal_neighbors(pos):
        if not in_bounds(npos, width, height):
            continue
        if npos in blocked:
            continue
        out.append(npos)
    return out


def shortest_path(
    start: GridPos, goal: GridPos, *, width: int, height: int, blocked: set[GridPos] | None = None
) -> list[GridPos] | None:
    blocked = blocked or set()

    def is_goal(pos: GridPos) -> bool:
        return pos == goal

    def expand(pos: GridPos):
        for npos in passable_neighbors(pos, width=width, height=height, blocked=blocked):
            # action encodes the next coordinate for path reconstruction.
            yield npos, npos, 1.0

    actions = bfs_plan(start, is_goal, expand)
    if actions is None:
        return None
    if not actions:
        return [start]

    path = [start]
    path.extend(actions)
    return path


def shortest_path_actions(
    start: GridPos,
    goal: GridPos,
    *,
    width: int,
    height: int,
    blocked: set[GridPos] | None = None,
    forbidden: set[GridPos] | None = None,
) -> list[int] | None:
    blocked_cells = set(blocked or set())
    forbidden_cells = set(forbidden or set())

    def is_goal(pos: GridPos) -> bool:
        return pos == goal

    def expand(pos: GridPos):
        x, y = pos
        for (dx, dy), action_id in MOVE_ACTION_BY_DELTA.items():
            npos = (x + dx, y + dy)
            if not in_bounds(npos, width, height):
                continue
            if npos in blocked_cells:
                continue
            if npos in forbidden_cells and npos != goal:
                continue
            yield action_id, npos, 1.0

    return bfs_plan(start, is_goal, expand)


def grid_to_display_click(camera, pos: GridPos, display_size: int = 64) -> dict[str, int]:
    x, y = camera_grid_to_display(camera, pos[0], pos[1], display_size=display_size)
    return {"x": int(x), "y": int(y)}


def points_to_set(points: Iterable[GridPos]) -> set[GridPos]:
    return {tuple(map(int, point)) for point in points}
