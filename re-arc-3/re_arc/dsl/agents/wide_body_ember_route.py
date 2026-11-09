from __future__ import annotations

from collections.abc import Iterable

from ..core import CachedProgramDslAgent, find_shortest_action_plan

MOVE_DELTAS: dict[int, tuple[int, int]] = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

LEVELS: tuple[dict[str, object], ...] = (
    {"start": (2, 5), "dock": (12, 5), "budget": 18, "embers": ((5, 4, 13, 4), (5, 7, 13, 7))},
    {"start": (2, 4), "dock": (12, 3), "budget": 24, "embers": ((8, 2, 8, 4), (8, 6, 8, 7), (8, 10, 8, 11))},
    {
        "start": (2, 10),
        "dock": (12, 5),
        "budget": 28,
        "embers": ((3, 2, 5, 4), (7, 3, 9, 5), (4, 8, 6, 10), (10, 8, 11, 10), (12, 4, 14, 4), (12, 7, 14, 7)),
    },
)


def _expand_embers(rectangles: Iterable[tuple[int, int, int, int]]) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for x1, y1, x2, y2 in rectangles:
        for cell_x in range(x1, x2 + 1):
            for cell_y in range(y1, y2 + 1):
                cells.add((cell_x, cell_y))
    return frozenset(cells)


class WideBodyEmberRouteDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _on_new_level(self, _env, level_idx: int):
        if level_idx not in self._programs:
            self._programs[level_idx] = self._build_level_program(level_idx)

    def _build_level_program(self, level_idx: int) -> list[tuple[int, dict[str, int]]]:
        spec = LEVELS[level_idx]
        start = tuple(spec["start"])
        dock = tuple(spec["dock"])
        budget = int(spec["budget"])
        ember_cells = _expand_embers(spec["embers"])
        initial_state = (int(start[0]), int(start[1]), budget)

        def occupied(top_left_x: int, top_left_y: int) -> tuple[tuple[int, int], ...]:
            return (
                (top_left_x, top_left_y),
                (top_left_x + 1, top_left_y),
                (top_left_x, top_left_y + 1),
                (top_left_x + 1, top_left_y + 1),
            )

        def blocked(top_left_x: int, top_left_y: int) -> bool:
            for cell_x, cell_y in occupied(top_left_x, top_left_y):
                if cell_x < 0 or cell_x >= 16 or cell_y < 0 or cell_y >= 14:
                    return True
                if cell_x == 0 or cell_x == 15 or cell_y == 0 or cell_y == 13:
                    return True
            return False

        def hits_ember(top_left_x: int, top_left_y: int) -> bool:
            return any(cell in ember_cells for cell in occupied(top_left_x, top_left_y))

        def is_goal(state: tuple[int, int, int]) -> bool:
            return (state[0], state[1]) == dock

        def expand(state: tuple[int, int, int]):
            x, y, remaining = state
            if remaining <= 0:
                return
            for action_id, (dx, dy) in MOVE_DELTAS.items():
                attempt_x = x + dx
                attempt_y = y + dy
                next_remaining = remaining - 1
                if blocked(attempt_x, attempt_y):
                    yield action_id, (x, y, next_remaining) if next_remaining > 0 else None
                    continue
                if hits_ember(attempt_x, attempt_y):
                    yield action_id, None
                    continue
                yield action_id, (attempt_x, attempt_y, next_remaining)
            for action_id in (5, 6):
                next_remaining = remaining - 1
                yield action_id, (x, y, next_remaining) if next_remaining > 0 else None

        plan = find_shortest_action_plan(
            start_state=initial_state,
            is_goal=is_goal,
            expand=expand,
            dominance_key=lambda state: (state[0], state[1]),
            dominance_score=lambda state: int(state[2]),
        )
        if not plan:
            raise RuntimeError(f"Unable to solve level {level_idx} for {self.game_id}.")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = WideBodyEmberRouteDslAgent
