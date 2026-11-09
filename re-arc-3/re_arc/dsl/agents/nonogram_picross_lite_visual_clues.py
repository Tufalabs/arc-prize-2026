from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent, camera_grid_to_display

_ENV_MOD = import_module(
    "re_arc.environment_files.nonogram_picross_lite_visual_clues.0001.nonogrampicrosslitevisualclues"
)

_deserialize_model = _ENV_MOD._deserialize_model


def _line_patterns(length: int, clues: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    if not clues:
        return [tuple([0] * length)]

    suffix_min: list[int] = [0] * (len(clues) + 1)
    for idx in range(len(clues) - 1, -1, -1):
        run_color = int(clues[idx][0])
        run_len = int(clues[idx][1])
        suffix_min[idx] = run_len + suffix_min[idx + 1]
        if idx + 1 < len(clues) and int(clues[idx + 1][0]) == run_color:
            # Same-color runs must be separated by at least one empty cell.
            suffix_min[idx] += 1

    patterns: list[tuple[int, ...]] = []

    def place(idx: int, pos: int, cells: list[int]):
        if idx >= len(clues):
            if len(cells) < length:
                cells.extend([0] * (length - len(cells)))
            patterns.append(tuple(cells[:length]))
            return

        color, run_len = clues[idx]
        run_len = int(run_len)
        gap_next = 0
        if idx + 1 < len(clues) and int(clues[idx + 1][0]) == int(color):
            gap_next = 1
        min_tail = suffix_min[idx + 1] if idx + 1 < len(clues) else 0
        max_start = length - (run_len + gap_next + min_tail)
        for start in range(pos, max_start + 1):
            next_cells = list(cells)
            if start > len(next_cells):
                next_cells.extend([0] * (start - len(next_cells)))
            next_cells.extend([int(color)] * run_len)
            next_pos = start + run_len
            if gap_next:
                if next_pos >= length:
                    continue
                next_cells.append(0)
                next_pos += 1
            place(idx + 1, next_pos, next_cells)

    place(0, 0, [])
    return patterns


def _propagate(
    row_domains: list[list[tuple[int, ...]]], col_domains: list[list[tuple[int, ...]]]
) -> tuple[list[list[tuple[int, ...]]] | None, list[list[tuple[int, ...]]] | None]:
    n = len(row_domains)

    while True:
        changed = False

        row_allowed: list[list[set[int]]] = []
        for y in range(n):
            if not row_domains[y]:
                return None, None
            allowed = [set() for _ in range(n)]
            for pat in row_domains[y]:
                for x, v in enumerate(pat):
                    allowed[x].add(int(v))
            row_allowed.append(allowed)

        col_allowed: list[list[set[int]]] = []
        for x in range(n):
            if not col_domains[x]:
                return None, None
            allowed = [set() for _ in range(n)]
            for pat in col_domains[x]:
                for y, v in enumerate(pat):
                    allowed[y].add(int(v))
            col_allowed.append(allowed)

        for y in range(n):
            new_domain = []
            for pat in row_domains[y]:
                ok = True
                for x in range(n):
                    if int(pat[x]) not in col_allowed[x][y]:
                        ok = False
                        break
                if ok:
                    new_domain.append(pat)
            if not new_domain:
                return None, None
            if len(new_domain) != len(row_domains[y]):
                row_domains[y] = new_domain
                changed = True

        for x in range(n):
            new_domain = []
            for pat in col_domains[x]:
                ok = True
                for y in range(n):
                    if int(pat[y]) not in row_allowed[y][x]:
                        ok = False
                        break
                if ok:
                    new_domain.append(pat)
            if not new_domain:
                return None, None
            if len(new_domain) != len(col_domains[x]):
                col_domains[x] = new_domain
                changed = True

        if not changed:
            return row_domains, col_domains


def _solve_from_clues(
    row_clues: list[list[tuple[int, int]]], col_clues: list[list[tuple[int, int]]]
) -> list[list[int]]:
    n = len(row_clues)
    row_domains = [_line_patterns(n, list(clues)) for clues in row_clues]
    col_domains = [_line_patterns(n, list(clues)) for clues in col_clues]

    def search(rows: list[list[tuple[int, ...]]], cols: list[list[tuple[int, ...]]]) -> list[list[int]] | None:
        rows_p, cols_p = _propagate(rows, cols)
        if rows_p is None or cols_p is None:
            return None

        solved = all(len(domain) == 1 for domain in rows_p)
        if solved:
            return [list(rows_p[y][0]) for y in range(n)]

        best_is_row = True
        best_idx = -1
        best_size = 10**9
        for idx, domain in enumerate(rows_p):
            size = len(domain)
            if 1 < size < best_size:
                best_size = size
                best_idx = idx
                best_is_row = True
        for idx, domain in enumerate(cols_p):
            size = len(domain)
            if 1 < size < best_size:
                best_size = size
                best_idx = idx
                best_is_row = False

        if best_idx < 0:
            return None

        if best_is_row:
            for candidate in rows_p[best_idx]:
                next_rows = [list(domain) for domain in rows_p]
                next_cols = [list(domain) for domain in cols_p]
                next_rows[best_idx] = [candidate]
                solved_grid = search(next_rows, next_cols)
                if solved_grid is not None:
                    return solved_grid
            return None

        for candidate in cols_p[best_idx]:
            next_rows = [list(domain) for domain in rows_p]
            next_cols = [list(domain) for domain in cols_p]
            next_cols[best_idx] = [candidate]
            solved_grid = search(next_rows, next_cols)
            if solved_grid is not None:
                return solved_grid
        return None

    solved = search(row_domains, col_domains)
    if solved is None:
        raise RuntimeError("nonogram DSL could not find a solution from clues")
    return solved


class NonogramPicrossLiteVisualCluesDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)
        n = int(model["n"])
        w = int(model["w"])
        h = int(model["h"])
        two_color = bool(model["two_color"])

        solved = _solve_from_clues(model["row_clues"], model["col_clues"])

        puzzle_x0 = 2 + w
        puzzle_y0 = 2 + h

        a_cells: list[tuple[int, int]] = []
        b_cells: list[tuple[int, int]] = []
        for y in range(n):
            for x in range(n):
                value = int(solved[y][x])
                if value == 1:
                    a_cells.append((x, y))
                elif value == 2:
                    b_cells.append((x, y))

        program: list[tuple[int, dict[str, int]]] = []

        def append_click(gx: int, gy: int):
            dx, dy = camera_grid_to_display(env._game.camera, int(gx), int(gy))
            program.append((6, {"x": int(dx), "y": int(dy)}))

        for px, py in a_cells:
            append_click(puzzle_x0 + px, puzzle_y0 + py)

        if two_color and b_cells:
            program.append((5, {}))
            for px, py in b_cells:
                append_click(puzzle_x0 + px, puzzle_y0 + py)

        return program


AGENT_CLASS = NonogramPicrossLiteVisualCluesDslAgent
