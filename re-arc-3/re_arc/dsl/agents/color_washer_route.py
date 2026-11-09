from __future__ import annotations

from collections import deque

from ..core import CachedProgramDslAgent

ACTION_BY_DELTA = {(0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}
COLOR_INDEX = {"neutral": 0, "red": 1, "blue": 2}
COLOR_NAME = {value: key for key, value in COLOR_INDEX.items()}
BOARD_SIZE = 12


class ColorWasherRouteDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "color_washer_route-0001") -> None:
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        rows = [str(row) for row in game.current_level.get_data("rows")]
        start_raw = game.current_level.get_data("start")
        start = (int(start_raw[0]), int(start_raw[1]))
        level_idx = int(game.level_index)
        path = self._find_path(rows, start)
        program = [(action_id, {}) for action_id in path]
        if level_idx < self.total_levels - 1:
            program.append((5, {}))
        return program

    def _find_path(self, rows: list[str], start: tuple[int, int]) -> list[int]:
        start_state = (start[0], start[1], COLOR_INDEX["neutral"])
        queue = deque([start_state])
        previous = {start_state: None}
        previous_action: dict[tuple[int, int, int], int] = {}
        goal_state: tuple[int, int, int] | None = None

        while queue:
            state = queue.popleft()
            if self._is_goal(rows, state):
                goal_state = state
                break
            for action_id, next_state in self._expand(rows, state):
                if next_state in previous:
                    continue
                previous[next_state] = state
                previous_action[next_state] = action_id
                queue.append(next_state)

        if goal_state is None:
            raise RuntimeError("Color Washer Route DSL could not find a winning path.")

        actions: list[int] = []
        cursor = goal_state
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]
        actions.reverse()
        return actions

    def _tile_at(self, rows: list[str], x: int, y: int) -> str:
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            return "#"
        return rows[y][x]

    def _expand(self, rows: list[str], state: tuple[int, int, int]) -> list[tuple[int, tuple[int, int, int]]]:
        x, y, color_idx = state
        color_name = COLOR_NAME[color_idx]
        out: list[tuple[int, tuple[int, int, int]]] = []
        for dx, dy in ACTION_BY_DELTA:
            action_id = ACTION_BY_DELTA[(dx, dy)]
            nx = x + dx
            ny = y + dy
            tile = self._tile_at(rows, nx, ny)
            next_color = color_name
            moved = False

            if tile == "#":
                next_state = (x, y, color_idx)
                out.append((action_id, next_state))
                continue
            if tile in {".", "S"}:
                moved = True
            elif tile == "r":
                moved = True
                next_color = "red"
            elif tile == "b":
                moved = True
                next_color = "blue"
            elif tile == "n":
                moved = True
                next_color = "neutral"
            elif tile == "R":
                moved = color_name == "red"
            elif tile == "B":
                moved = color_name == "blue"
            elif tile == "N":
                moved = color_name == "neutral"
            elif tile == "d":
                moved = color_name == "blue"

            if not moved:
                out.append((action_id, (x, y, color_idx)))
                continue

            out.append((action_id, (nx, ny, COLOR_INDEX[next_color])))
        return out

    def _is_goal(self, rows: list[str], state: tuple[int, int, int]) -> bool:
        x, y, color_idx = state
        return rows[y][x] == "d" and color_idx == COLOR_INDEX["blue"]


AGENT_CLASS = ColorWasherRouteDslAgent
