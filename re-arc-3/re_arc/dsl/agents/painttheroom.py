from __future__ import annotations

from collections import deque

from re_arc.dsl.core import DslAgent, camera_grid_to_display

MOVE_ACTIONS = ((1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0))


def _line_clear(game, src: tuple[int, int], dst: tuple[int, int]) -> bool:
    x0, y0 = src
    x1, y1 = dst
    if x0 != x1 and y0 != y1:
        return False

    if x0 == x1:
        step = 1 if y1 > y0 else -1
        return all(not game._is_wall_like(x0, y) for y in range(y0 + step, y1, step))

    step = 1 if x1 > x0 else -1
    return all(not game._is_wall_like(x, y0) for x in range(x0 + step, x1, step))


def _bfs(game, start: tuple[int, int]):
    queue = deque([start])
    dist = {start: 0}
    prev: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}

    while queue:
        x, y = queue.popleft()
        for action_id, dx, dy in MOVE_ACTIONS:
            nxt = (x + dx, y + dy)
            if nxt in dist:
                continue
            if not game._cursor_walkable(nxt[0], nxt[1]):
                continue
            dist[nxt] = dist[(x, y)] + 1
            prev[nxt] = ((x, y), action_id)
            queue.append(nxt)

    return dist, prev


def _path_first_action(
    prev: dict[tuple[int, int], tuple[tuple[int, int], int] | None], target: tuple[int, int]
) -> int | None:
    cursor = target
    first = None
    while prev.get(cursor) is not None:
        parent, action_id = prev[cursor]  # type: ignore[index]
        first = action_id
        cursor = parent
    return first


class PainttheroomDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=4)

    def next_action(self, env, observation):
        _ = observation
        game = env._game
        cursor = tuple(int(v) for v in game._cursor)

        unfinished = [cell for cell in game._required_cells if int(game._cell_state.get(cell, 0)) != 4]

        if not unfinished:
            return 6, {"x": -1, "y": -1}

        dist, prev = _bfs(game, cursor)

        if (
            game._lever_cell is not None
            and cursor == game._lever_cell
            and game._gate_pending_target is None
            and not game._gate_open
        ):
            unreachable = [cell for cell in unfinished if cell not in dist]
            if unreachable:
                return 5, {}

        if (game._lever_cell is None or cursor != game._lever_cell) and (cursor in unfinished):
            return 5, {}

        visible_from_cursor = [cell for cell in unfinished if _line_clear(game, cursor, cell)]
        if visible_from_cursor:
            visible_from_cursor.sort(key=lambda cell: abs(cell[0] - cursor[0]) + abs(cell[1] - cursor[1]), reverse=True)
            tx, ty = visible_from_cursor[0]
            dx, dy = camera_grid_to_display(game.camera, int(tx), int(ty))
            return 6, {"x": int(dx), "y": int(dy)}

        best_pos: tuple[int, int] | None = None
        best_score: int | None = None
        best_visible: list[tuple[int, int]] = []

        for pos, steps in dist.items():
            visible = [cell for cell in unfinished if _line_clear(game, pos, cell)]
            if not visible:
                continue
            score = int(len(visible) * 100 - steps * 8)
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
                best_visible = visible

        if best_pos is not None:
            if best_pos == cursor and best_visible:
                best_visible.sort(key=lambda cell: abs(cell[0] - cursor[0]) + abs(cell[1] - cursor[1]), reverse=True)
                tx, ty = best_visible[0]
                dx, dy = camera_grid_to_display(game.camera, int(tx), int(ty))
                return 6, {"x": int(dx), "y": int(dy)}

            first_action = _path_first_action(prev, best_pos)
            if first_action is not None:
                return int(first_action), {}

        reachable_unfinished = [cell for cell in unfinished if cell in dist]
        if reachable_unfinished:
            target = min(reachable_unfinished, key=lambda cell: dist[cell])
            first_action = _path_first_action(prev, target)
            if first_action is not None:
                return int(first_action), {}

        if game._lever_cell is not None and game._lever_cell in dist:
            first_action = _path_first_action(prev, game._lever_cell)
            if first_action is not None:
                return int(first_action), {}

        return 6, {"x": -1, "y": -1}


AGENT_CLASS = PainttheroomDslAgent
