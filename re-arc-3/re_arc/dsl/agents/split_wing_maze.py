from __future__ import annotations

from collections import deque
from typing import Any

from ..core import CachedProgramDslAgent, camera_grid_to_display

SELECT_BLUE = "blue"
SELECT_MAGENTA = "magenta"
MOVE_ACTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _unwrap_game(env: Any) -> Any:
    outer = getattr(env, "_env", env)
    direct = getattr(outer, "_game", None)
    if direct is not None:
        return direct
    inner = getattr(outer, "_env", None)
    if inner is not None:
        nested = getattr(inner, "_game", None)
        if nested is not None:
            return nested
        return inner
    return outer


def _click_payload(camera: Any, cell: tuple[int, int]) -> dict[str, int]:
    px = 2 + (int(cell[0]) * 6) + 2
    py = 4 + (int(cell[1]) * 6) + 2
    dx, dy = camera_grid_to_display(camera, px, py)
    return {"x": int(dx), "y": int(dy)}


def _state_key(
    blue_pos: tuple[int, int], magenta_pos: tuple[int, int], selected: str
) -> tuple[tuple[int, int], tuple[int, int], str]:
    return blue_pos, magenta_pos, selected


class SplitWingMazeDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env: Any) -> list[tuple[int, dict[str, int]]]:
        game = _unwrap_game(env)
        level = game.current_level
        walls = frozenset(tuple(cell) for cell in level.get_data("walls"))
        blue_start = tuple(level.get_data("blue_start"))
        magenta_start = tuple(level.get_data("magenta_start"))
        magenta_home = tuple(level.get_data("magenta_home"))

        start = _state_key(blue_start, magenta_start, SELECT_BLUE)
        queue = deque([start])
        previous: dict[
            tuple[tuple[int, int], tuple[int, int], str], tuple[tuple[int, int], tuple[int, int], str] | None
        ] = {start: None}
        previous_action: dict[tuple[tuple[int, int], tuple[int, int], str], tuple[int, dict[str, int]]] = {}
        goal_state: tuple[tuple[int, int], tuple[int, int], str] | None = None

        def is_open(cell: tuple[int, int]) -> bool:
            x, y = cell
            return 0 <= x < 10 and 0 <= y < 10 and cell not in walls

        while queue:
            state = queue.popleft()
            blue_pos, magenta_pos, selected = state
            if blue_pos == magenta_home:
                goal_state = state
                break

            other_selected = SELECT_MAGENTA if selected == SELECT_BLUE else SELECT_BLUE
            clicked_cell = magenta_pos if other_selected == SELECT_MAGENTA else blue_pos
            switch_state = _state_key(blue_pos, magenta_pos, other_selected)
            if switch_state not in previous:
                previous[switch_state] = state
                previous_action[switch_state] = (6, _click_payload(game.camera, clicked_cell))
                queue.append(switch_state)

            for action_id, (dx, dy) in MOVE_ACTIONS.items():
                if selected == SELECT_BLUE:
                    current = blue_pos
                    blocker = magenta_pos
                else:
                    current = magenta_pos
                    blocker = blue_pos
                destination = (current[0] + dx, current[1] + dy)
                if not is_open(destination) or destination == blocker:
                    destination = current
                next_state = _state_key(
                    destination if selected == SELECT_BLUE else blue_pos,
                    destination if selected == SELECT_MAGENTA else magenta_pos,
                    selected,
                )
                if next_state in previous:
                    continue
                previous[next_state] = state
                previous_action[next_state] = (action_id, {})
                queue.append(next_state)

        if goal_state is None:
            raise RuntimeError("split_wing_maze DSL could not find a winning state.")

        program: list[tuple[int, dict[str, int]]] = []
        cursor = goal_state
        while previous[cursor] is not None:
            program.append(previous_action[cursor])
            cursor = previous[cursor]
        program.reverse()

        program.append((6, _click_payload(game.camera, magenta_home)))
        return program


AGENT_CLASS = SplitWingMazeDslAgent
