from __future__ import annotations

from collections import deque

from ..core import CachedProgramDslAgent

ACTION_TO_DELTA = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

LEVEL_SPECS = (
    {
        "rows": ("........", "........", "........", ".S.XX.G.", "...XX...", "........", "........", "........"),
        "start": (3, 1),
        "goal": (3, 6),
        "budget": 13,
    },
    {
        "rows": ("........", "..XX..G.", "..XX....", "....XX..", "....XX..", "..XX....", ".SXX....", "........"),
        "start": (6, 1),
        "goal": (1, 6),
        "budget": 16,
    },
    {
        "rows": ("XXXXXXXX", "XXXXX.XX", "X...X..X", "X.X.X..X", "X.X....X", "X.X.XXGX", "XSX.XXXX", "XXXXXXXX"),
        "start": (6, 1),
        "goal": (5, 6),
        "budget": 20,
    },
)


class SafeTileTapsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _on_new_level(self, env, level_idx: int):
        _ = env
        self._programs[level_idx] = self._build_program_for_level(level_idx)

    def _build_level_program(self, env):
        _ = env
        return self._build_program_for_level(0)

    def _build_program_for_level(self, level_idx: int):
        spec = LEVEL_SPECS[level_idx]
        rows = tuple(str(row) for row in spec["rows"])
        start = tuple(int(value) for value in spec["start"])
        goal = tuple(int(value) for value in spec["goal"])
        budget = int(spec["budget"])

        queue = deque([start])
        prev: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}

        while queue:
            row, col = queue.popleft()
            if (row, col) == goal:
                break
            for action_id, (d_row, d_col) in ACTION_TO_DELTA.items():
                next_row = row + d_row
                next_col = col + d_col
                next_pos = (next_row, next_col)
                if not (0 <= next_row < 8 and 0 <= next_col < 8):
                    continue
                if rows[next_row][next_col] == "X":
                    continue
                if next_pos in prev:
                    continue
                prev[next_pos] = ((row, col), action_id)
                queue.append(next_pos)

        if goal not in prev:
            raise RuntimeError("safe_tile_taps DSL could not find a path to the goal.")

        actions: list[int] = []
        cursor = goal
        while prev[cursor] is not None:
            parent, action_id = prev[cursor]
            actions.append(action_id)
            cursor = parent
        actions.reverse()

        if len(actions) > budget:
            raise RuntimeError(f"safe_tile_taps DSL path exceeds budget: path_len={len(actions)} budget={budget}")

        # Each successful level requires one extra ignored input to clear the win splash.
        actions.append(1)
        return [(action_id, {}) for action_id in actions]


AGENT_CLASS = SafeTileTapsDslAgent
