from __future__ import annotations

from collections import deque

from arcengine import GameAction

from ..core import CachedProgramDslAgent

PINK_HOME = (1, 4)
BLUE_HOME = (8, 4)
LEVEL_SAFE_CELLS = {
    0: {
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 2),
        (6, 2),
        (7, 2),
        (8, 2),
        (1, 3),
        (8, 3),
        (1, 4),
        (8, 4),
        (1, 5),
        (2, 5),
        (7, 5),
        (8, 5),
    },
    1: {
        (1, 4),
        (8, 4),
        (2, 4),
        (3, 4),
        (4, 4),
        (4, 3),
        (4, 2),
        (5, 2),
        (6, 2),
        (6, 3),
        (6, 4),
        (4, 5),
        (4, 6),
        (5, 6),
        (6, 6),
        (6, 5),
        (7, 4),
        (1, 5),
        (2, 5),
    },
    2: {
        (1, 4),
        (8, 4),
        (2, 4),
        (3, 4),
        (1, 5),
        (2, 5),
        (2, 6),
        (3, 3),
        (3, 2),
        (4, 2),
        (5, 2),
        (5, 3),
        (5, 4),
        (3, 5),
        (3, 6),
        (4, 6),
        (5, 6),
        (5, 5),
        (6, 3),
        (7, 3),
        (7, 4),
        (6, 5),
        (7, 5),
    },
}
MOVE_ACTIONS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _cell_center(cell: tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    return 7 + (5 * x) + 2, 12 + (5 * y) + 2


class HazardBatonDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(getattr(env, "_current_level", 0))
        safe_cells = LEVEL_SAFE_CELLS[level_idx]
        start_state = (PINK_HOME, BLUE_HOME, "blue")
        plan = self._search(start_state, safe_cells)
        if plan is None:
            raise RuntimeError(f"hazard_baton: no plan found for level {level_idx}")
        return plan

    def _search(
        self, start_state: tuple[tuple[int, int], tuple[int, int], str], safe_cells: set[tuple[int, int]]
    ) -> list[tuple[int, dict[str, int]]] | None:
        queue = deque([start_state])
        prev: dict[
            tuple[tuple[int, int], tuple[int, int], str],
            tuple[tuple[tuple[int, int], tuple[int, int], str], tuple[int, dict[str, int]]] | None,
        ] = {start_state: None}
        goal_state: tuple[tuple[int, int], tuple[int, int], str] | None = None

        while queue:
            state = queue.popleft()
            _, blue, _ = state
            if blue == PINK_HOME:
                goal_state = state
                break

            for next_state, action in self._expand(state, safe_cells):
                if next_state in prev:
                    continue
                prev[next_state] = (state, action)
                queue.append(next_state)

        if goal_state is None:
            return None

        actions: list[tuple[int, dict[str, int]]] = []
        cursor = goal_state
        while prev[cursor] is not None:
            parent, action = prev[cursor]
            actions.append(action)
            cursor = parent
        actions.reverse()

        advance_action = (int(GameAction.ACTION4.value), {})
        return [*actions, advance_action]

    def _expand(
        self, state: tuple[tuple[int, int], tuple[int, int], str], safe_cells: set[tuple[int, int]]
    ) -> list[tuple[tuple[tuple[int, int], tuple[int, int], str], tuple[int, dict[str, int]]]]:
        pink, blue, selected = state
        out: list[tuple[tuple[tuple[int, int], tuple[int, int], str], tuple[int, dict[str, int]]]] = []

        if selected == "blue":
            click_x, click_y = _cell_center(pink)
            out.append(((pink, blue, "pink"), (int(GameAction.ACTION6.value), {"x": click_x, "y": click_y})))
        else:
            click_x, click_y = _cell_center(blue)
            out.append(((pink, blue, "blue"), (int(GameAction.ACTION6.value), {"x": click_x, "y": click_y})))

        current = blue if selected == "blue" else pink
        other = pink if selected == "blue" else blue
        for action_id, delta in MOVE_ACTIONS.items():
            target = (current[0] + delta[0], current[1] + delta[1])
            if target == other or target not in safe_cells:
                continue
            next_state = (pink, target, selected) if selected == "blue" else (target, blue, selected)
            out.append((next_state, (action_id, {})))
        return out


AGENT_CLASS = HazardBatonDslAgent
