from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent

_GAME_MOD = import_module("re_arc.environment_files.flood_fill.0001.floodfill")

BOARD_W = _GAME_MOD.BOARD_W
BOARD_H = _GAME_MOD.BOARD_H
PUZZLE_X = _GAME_MOD.PUZZLE_X
PUZZLE_Y = _GAME_MOD.PUZZLE_Y
CELL = _GAME_MOD.CELL
SWATCH_Y = _GAME_MOD.SWATCH_Y
SWATCH_XS = _GAME_MOD.SWATCH_XS
SWATCH_SIZE = _GAME_MOD.SWATCH_SIZE
SWATCH_COLORS = _GAME_MOD.SWATCH_COLORS
LEVEL_SPECS = _GAME_MOD.LEVEL_SPECS
find_component = _GAME_MOD.find_component
iter_components = _GAME_MOD.iter_components
is_uniform = _GAME_MOD.is_uniform
recolor_component = _GAME_MOD.recolor_component
rows_to_board = _GAME_MOD.rows_to_board


def _distinct_color_count(board: tuple[tuple[int, ...], ...]) -> int:
    return len({cell for row in board for cell in row})


def _candidate_moves(
    board: tuple[tuple[int, ...], ...], enabled_colors: tuple[int, ...]
) -> list[tuple[tuple[int, int, int], tuple[tuple[int, ...], ...], int, int]]:
    current_components = iter_components(board)
    current_count = len(current_components)
    candidates: list[tuple[tuple[int, int, int], tuple[tuple[int, ...], ...], int, int]] = []

    for component in current_components:
        cell_x, cell_y = component[0]
        current_color = board[cell_y][cell_x]
        neighbor_colors: set[int] = set()
        for x, y in component:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < BOARD_W and 0 <= ny < BOARD_H:
                    color = board[ny][nx]
                    if color != current_color:
                        neighbor_colors.add(color)

        for target_color in enabled_colors:
            if target_color == current_color or target_color not in neighbor_colors:
                continue
            next_board = recolor_component(board, component, target_color)
            next_components = iter_components(next_board)
            next_count = len(next_components)
            if next_count >= current_count:
                continue
            largest_after = max(len(next_component) for next_component in next_components)
            reduction = current_count - next_count
            candidates.append(((cell_x, cell_y, target_color), next_board, reduction, largest_after))

    candidates.sort(key=lambda item: (-item[2], -item[3], item[0][2], item[0][1], item[0][0]))
    return candidates


def _greedy_upper_bound(
    start_board: tuple[tuple[int, ...], ...], enabled_colors: tuple[int, ...]
) -> list[tuple[int, int, int]]:
    board = start_board
    path: list[tuple[int, int, int]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    while not is_uniform(board):
        seen.add(board)
        candidates = _candidate_moves(board, enabled_colors)
        if not candidates:
            raise RuntimeError("Flood Fill greedy search got stuck.")
        move, next_board, _reduction, _largest_after = candidates[0]
        if next_board in seen:
            raise RuntimeError("Flood Fill greedy search looped.")
        path.append(move)
        board = next_board
    return path


def solve_board(
    start_board: tuple[tuple[int, ...], ...], enabled_colors: tuple[int, ...]
) -> list[tuple[int, int, int]]:
    if is_uniform(start_board):
        return []

    greedy_path = _greedy_upper_bound(start_board, enabled_colors)
    lower_bound = max(1, _distinct_color_count(start_board) - 1)

    def search(
        board: tuple[tuple[int, ...], ...], depth_left: int, seen: dict[tuple[tuple[int, ...], ...], int]
    ) -> list[tuple[int, int, int]] | None:
        if is_uniform(board):
            return []
        if depth_left <= 0:
            return None
        if _distinct_color_count(board) - 1 > depth_left:
            return None
        previous_depth = seen.get(board)
        if previous_depth is not None and previous_depth >= depth_left:
            return None
        seen[board] = depth_left

        for move, next_board, _reduction, _largest_after in _candidate_moves(board, enabled_colors):
            suffix = search(next_board, depth_left - 1, seen)
            if suffix is not None:
                return [move, *suffix]
        return None

    for depth in range(lower_bound, len(greedy_path) + 1):
        result = search(start_board, depth, {})
        if result is not None:
            return result

    raise RuntimeError("Flood Fill solver could not find a solution.")


def swatch_click(color: int) -> tuple[int, dict[str, int]]:
    index = SWATCH_COLORS.index(color)
    left = SWATCH_XS[index]
    return 6, {"x": left + SWATCH_SIZE // 2, "y": SWATCH_Y + SWATCH_SIZE // 2}


def puzzle_click(cx: int, cy: int) -> tuple[int, dict[str, int]]:
    return 6, {"x": PUZZLE_X + cx * CELL + 2, "y": PUZZLE_Y + cy * CELL + 2}


class FloodFillDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "flood_fill-0001"):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_SPECS))

    def _build_level_program(self, _env) -> list[tuple[int, dict[str, int]]]:
        level_idx = self._current_level_idx
        if level_idx is None:
            raise RuntimeError("Flood Fill DSL missing current level index.")

        spec = LEVEL_SPECS[level_idx]
        board = rows_to_board(spec.rows)
        actions: list[tuple[int, dict[str, int]]] = []
        active_color = spec.default_active_color
        for cx, cy, target_color in solve_board(board, spec.enabled_colors):
            component = find_component(board, cx, cy)
            if target_color != active_color:
                actions.append(swatch_click(target_color))
                active_color = target_color
            actions.append(puzzle_click(component[0][0], component[0][1]))
            board = recolor_component(board, component, target_color)

        if not is_uniform(board):
            raise RuntimeError(f"Flood Fill DSL failed to solve level {level_idx}.")
        actions.append((5, {}))
        return actions


AGENT_CLASS = FloodFillDslAgent
