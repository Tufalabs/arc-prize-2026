from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
CELL_SIZE = 5
CELL_PITCH = 6

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

COLOR_TIPS = 0
COLOR_OPEN_CLAMP = 1
COLOR_CAP = 2
COLOR_DETAIL = 3
COLOR_DARK = 4
COLOR_BACKGROUND = 5
COLOR_HANDLE = 6
COLOR_INVALID = 8
COLOR_SELECTOR = 9
COLOR_PREVIEW = 10
COLOR_GLOW_RIM = 11
COLOR_GLOW_CORE = 12
COLOR_CLOSED_CLAMP = 13
COLOR_COMPLETE = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    lever_rows: tuple[str, ...]
    lever_ids: dict[tuple[int, int], str]
    initial_selection: str
    baseline_actions: int


@dataclass(frozen=True)
class Lever:
    object_id: str
    row: int
    col: int
    orientation: str


LEVEL_SPECS = (
    LevelSpec("One Line", ("000",), (".H.",), {(0, 1): "A"}, "A", 1),
    LevelSpec("It Stays Gone", ("000", "000"), (".H.", ".H."), {(0, 1): "A", (1, 1): "B"}, "A", 3),
    LevelSpec("Open the Clamp", ("10", "01"), ("VH", ".."), {(0, 0): "A", (0, 1): "B"}, "A", 5),
    LevelSpec(
        "Choose a Chain",
        ("1010", "1010", "0101", "0101"),
        ("VH..", "H.V.", "HV..", ".H.V"),
        {(0, 1): "A", (0, 0): "B", (1, 0): "C", (1, 2): "D", (2, 0): "E", (2, 1): "F", (3, 1): "G", (3, 3): "H"},
        "A",
        7,
    ),
    LevelSpec(
        "Leave a Final Line",
        ("111000", "111000", "111000", "000111", "000111", "000111"),
        ("V..H..", ".V...H", ".HV...", "H..VV.", "....HV", "...H.."),
        {
            (0, 3): "A",
            (1, 5): "B",
            (0, 0): "C",
            (1, 1): "D",
            (2, 1): "E",
            (2, 2): "F",
            (3, 0): "G",
            (3, 3): "H",
            (3, 4): "I",
            (4, 4): "J",
            (5, 3): "K",
            (4, 5): "L",
        },
        "C",
        14,
    ),
    LevelSpec(
        "Old Debts",
        ("1110000", "1110000", "1110000", "1110000", "0001111", "0001111", "0001111"),
        ("V..H...", "HV.....", "..H....", "..VH...", "hH.V...", ".v.HV..", "H.h..VV"),
        {
            (0, 3): "A",
            (0, 0): "B",
            (1, 0): "C",
            (1, 1): "D",
            (3, 3): "E",
            (3, 2): "F",
            (2, 2): "G",
            (4, 1): "H",
            (4, 3): "I",
            (5, 3): "J",
            (5, 4): "K",
            (6, 0): "L",
            (6, 5): "M",
            (6, 6): "N",
            (4, 0): "S1",
            (5, 1): "S2",
            (6, 2): "S3",
        },
        "S3",
        18,
    ),
    LevelSpec(
        "Alternating Commitments",
        (
            "1010101010",
            "0101010101",
            "1010101010",
            "0101010101",
            "1010101010",
            "0101010101",
            "1010101010",
            "0101010101",
            "1010101010",
            "0101010101",
        ),
        (
            "V.......VH",
            "HVv......V",
            "..H.......",
            "...H..h...",
            "....VhH...",
            "....vV.H..",
            "......V.H.",
            ".......VhH",
            "H.V.......",
            ".H.V......",
        ),
        {
            (0, 9): "A",
            (0, 0): "B",
            (8, 0): "C",
            (8, 2): "F",
            (2, 2): "E",
            (0, 8): "D",
            (6, 8): "G",
            (6, 6): "H",
            (4, 6): "I",
            (4, 4): "J",
            (1, 0): "K",
            (1, 1): "L",
            (9, 1): "M",
            (9, 3): "P",
            (3, 3): "O",
            (1, 9): "N",
            (7, 9): "Q",
            (7, 7): "R",
            (5, 7): "S",
            (5, 5): "T",
            (4, 5): "X0",
            (1, 2): "X1",
            (3, 6): "X2",
            (5, 4): "X3",
            (7, 8): "X4",
        },
        "X0",
        28,
    ),
)


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


def _parse_levers(spec: LevelSpec) -> tuple[list[Lever], set[int]]:
    levers: list[Lever] = []
    initially_spent: set[int] = set()
    for row, symbols in enumerate(spec.lever_rows):
        for col, symbol in enumerate(symbols):
            if symbol not in "HVhv":
                continue
            object_id = spec.lever_ids[(row, col)]
            levers.append(Lever(object_id, row, col, symbol.upper()))
            if symbol.islower():
                initially_spent.add(len(levers) - 1)
    return levers, initially_spent


def _build_level(spec: LevelSpec) -> Level:
    if len(spec.rows) != len(spec.lever_rows) or any(
        len(board_row) != len(lever_row) for board_row, lever_row in zip(spec.rows, spec.lever_rows, strict=True)
    ):
        raise ValueError(f"Malformed layout for {spec.name}.")
    levers, initially_spent = _parse_levers(spec)
    return Level(
        sprites=[],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        data={
            "rows": spec.rows,
            "levers": levers,
            "initially_spent": initially_spent,
            "initial_selection": spec.initial_selection,
            "step_budget": spec.baseline_actions * 6,
        },
        name=spec.name,
    )


class VanishingLeversView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: VanishingLevers | None = None

    @staticmethod
    def _line(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        x0 = max(0, min(VIEW_SIZE - 1, x0))
        x1 = max(0, min(VIEW_SIZE - 1, x1))
        y0 = max(0, min(VIEW_SIZE - 1, y0))
        y1 = max(0, min(VIEW_SIZE - 1, y1))
        if x0 > x1 or y0 > y1:
            return
        if y0 == y1:
            frame[y0, x0 : x1 + 1] = color
        elif x0 == x1:
            frame[y0 : y1 + 1, x0] = color

    def _rect(self, frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        self._line(frame, x0, y0, x1, y0, color)
        self._line(frame, x0, y1, x1, y1, color)
        self._line(frame, x0, y0, x0, y1, color)
        self._line(frame, x1, y0, x1, y1, color)

    def _draw_cell(self, frame: np.ndarray, game: VanishingLevers, row: int, col: int) -> None:
        x = game.origin_x + col * CELL_PITCH
        y = game.origin_y + row * CELL_PITCH
        glowing = game.board[row][col]
        frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = COLOR_GLOW_RIM if glowing else COLOR_DARK
        frame[y + 1 : y + 4, x + 1 : x + 4] = COLOR_GLOW_CORE if glowing else COLOR_BACKGROUND

        if game.animation_kind == "pull" and game.animation_phase == 2 and (row, col) in game.affected_cells:
            frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = COLOR_PREVIEW

    def _lever_pixels(self, game: VanishingLevers, index: int) -> np.ndarray:
        lever = game.levers[index]
        folding_selected = game.animation_kind == "pull" and index == game.selected and game.animation_phase < 3
        spent = index in game.spent and not folding_selected
        if spent:
            pixels = np.full((3, 3), COLOR_CAP, dtype=np.int8)
            pixels[0, 0] = COLOR_DETAIL
            pixels[2, 2] = COLOR_DETAIL
            if lever.orientation == "H":
                pixels[1, :] = COLOR_TIPS
            else:
                pixels[:, 1] = COLOR_TIPS
            if game.animation_kind == "spent" and game.animation_phase == 1 and index == game.selected:
                pixels[pixels == COLOR_CAP] = COLOR_TIPS
            return pixels

        clamped = game.board[lever.row][lever.col]
        if game.animation_kind == "pull" and game.animation_phase < 4:
            clamped = game.pre_board[lever.row][lever.col]
        pixels = np.full((3, 3), -1, dtype=np.int8)
        if lever.orientation == "H":
            pixels[0, 1] = COLOR_HANDLE
            pixels[1, :] = (COLOR_TIPS, COLOR_HANDLE, COLOR_TIPS)
            pixels[2, :] = (
                (COLOR_CLOSED_CLAMP, COLOR_HANDLE, COLOR_CLOSED_CLAMP)
                if clamped
                else (COLOR_OPEN_CLAMP, -1, COLOR_OPEN_CLAMP)
            )
        else:
            pixels[1, 0] = COLOR_HANDLE
            pixels[:, 1] = (COLOR_TIPS, COLOR_HANDLE, COLOR_TIPS)
            pixels[:, 2] = (
                (COLOR_CLOSED_CLAMP, COLOR_HANDLE, COLOR_CLOSED_CLAMP)
                if clamped
                else (COLOR_OPEN_CLAMP, -1, COLOR_OPEN_CLAMP)
            )
        if game.animation_kind == "clamped" and game.animation_phase == 1 and index == game.selected:
            pixels[pixels == COLOR_CLOSED_CLAMP] = COLOR_INVALID
        if game.animation_kind == "pull" and game.animation_phase == 1 and index == game.selected:
            pixels[pixels == COLOR_HANDLE] = COLOR_DETAIL
        return pixels

    def _draw_lever(self, frame: np.ndarray, game: VanishingLevers, index: int) -> None:
        lever = game.levers[index]
        x = game.origin_x + lever.col * CELL_PITCH + 1
        y = game.origin_y + lever.row * CELL_PITCH + 1
        pixels = self._lever_pixels(game, index)
        mask = pixels >= 0
        region = frame[y : y + 3, x : x + 3]
        region[mask] = pixels[mask]

    def _draw_preview(self, frame: np.ndarray, game: VanishingLevers) -> None:
        lever = game.levers[game.selected]
        if lever.orientation == "H":
            y = game.origin_y + lever.row * CELL_PITCH
            self._rect(frame, game.origin_x - 1, y - 1, game.origin_x + game.board_width, y + CELL_SIZE, COLOR_PREVIEW)
        else:
            x = game.origin_x + lever.col * CELL_PITCH
            self._rect(frame, x - 1, game.origin_y - 1, x + CELL_SIZE, game.origin_y + game.board_height, COLOR_PREVIEW)

    def _draw_selector(self, frame: np.ndarray, game: VanishingLevers) -> None:
        lever = game.levers[game.selected]
        x = game.origin_x + lever.col * CELL_PITCH - 1
        y = game.origin_y + lever.row * CELL_PITCH - 1
        if game.animation_kind == "nudge" and game.animation_phase == 1:
            dx, dy = game.nudge_delta
            x += dx
            y += dy
        self._rect(frame, x, y, x + CELL_SIZE + 1, y + CELL_SIZE + 1, COLOR_SELECTOR)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BACKGROUND
        for row in range(len(game.board)):
            for col in range(len(game.board[0])):
                self._draw_cell(frame, game, row, col)

        self._draw_preview(frame, game)
        for index in range(len(game.levers)):
            self._draw_lever(frame, game, index)
        self._draw_selector(frame, game)

        frame_color = COLOR_COMPLETE if game.solved else COLOR_DETAIL
        if not game.solved and not any(
            index not in game.spent and not game.board[item.row][item.col] for index, item in enumerate(game.levers)
        ):
            frame_color = COLOR_CLOSED_CLAMP
        self._rect(
            frame,
            game.origin_x - 2,
            game.origin_y - 2,
            game.origin_x + game.board_width + 1,
            game.origin_y + game.board_height + 1,
            frame_color,
        )
        if game.solved and game.animation_kind == "pull" and game.animation_phase == 3:
            self._rect(
                frame,
                game.origin_x - 1,
                game.origin_y - 1,
                game.origin_x + game.board_width,
                game.origin_y + game.board_height,
                COLOR_COMPLETE,
            )

        filled = round(VIEW_SIZE * game.remaining_steps / max(1, game.step_budget))
        frame[VIEW_SIZE - 1, :] = COLOR_DETAIL
        if filled:
            frame[VIEW_SIZE - 1, :filled] = COLOR_COMPLETE
        return frame


class VanishingLevers(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = VanishingLeversView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "vanishing_levers-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._view]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK, ACTION_UNDO],
            seed,
        )
        self._view.game = weakref.proxy(self)

    def on_set_level(self, _level: Level) -> None:
        rows = self.current_level.get_data("rows")
        self.board = [[cell == "1" for cell in row] for row in rows]
        self.levers = list(self.current_level.get_data("levers"))
        self.spent = set(self.current_level.get_data("initially_spent"))
        selected_id = self.current_level.get_data("initial_selection")
        self.selected = next(index for index, lever in enumerate(self.levers) if lever.object_id == selected_id)
        self.step_budget = int(self.current_level.get_data("step_budget"))
        self.remaining_steps = self.step_budget
        self.step_count = 0
        self.solved = False
        self.terminal = False
        self.undo_history: list[tuple[tuple[tuple[bool, ...], ...], frozenset[int], int]] = []
        self.animation_kind = ""
        self.animation_phase = 0
        self.animation_length = 0
        self.affected_cells: set[tuple[int, int]] = set()
        self.pre_board = [row[:] for row in self.board]
        self.nudge_delta = (0, 0)
        self.pending_loss = False
        width = len(self.board[0])
        height = len(self.board)
        self.board_width = CELL_PITCH * width - 1
        self.board_height = CELL_PITCH * height - 1
        self.origin_x = (VIEW_SIZE - self.board_width) // 2
        self.origin_y = (VIEW_SIZE - self.board_height) // 2

    def _snapshot(self) -> tuple[tuple[tuple[bool, ...], ...], frozenset[int], int]:
        return tuple(tuple(row) for row in self.board), frozenset(self.spent), self.selected

    def _restore(self, snapshot: tuple[tuple[tuple[bool, ...], ...], frozenset[int], int]) -> None:
        board, spent, selected = snapshot
        self.board = [list(row) for row in board]
        self.spent = set(spent)
        self.selected = selected
        self.solved = False
        self.terminal = False

    def _spend_action(self) -> None:
        self.step_count += 1
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.remaining_steps == 0 and not self.solved:
            self.pending_loss = True

    def _finish_or_lose(self) -> None:
        if self.solved:
            self.next_level()
            self.complete_action()
        elif self.pending_loss:
            self.terminal = True
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _start_animation(self, kind: str, length: int) -> None:
        self.animation_kind = kind
        self.animation_phase = 1
        self.animation_length = length

    def _advance_animation(self) -> None:
        if self.animation_phase < self.animation_length:
            self.animation_phase += 1
            return
        self.animation_kind = ""
        self.animation_phase = 0
        self.affected_cells.clear()
        self._finish_or_lose()

    def _selection_target(self, action: int) -> int | None:
        current = self.levers[self.selected]
        candidates: list[tuple[tuple[int, int, int, int, str], int]] = []
        for index, lever in enumerate(self.levers):
            if action == ACTION_RIGHT and lever.col > current.col:
                key = (abs(lever.row - current.row), lever.col - current.col, lever.row, lever.col, lever.object_id)
            elif action == ACTION_LEFT and lever.col < current.col:
                key = (abs(lever.row - current.row), current.col - lever.col, lever.row, lever.col, lever.object_id)
            elif action == ACTION_DOWN and lever.row > current.row:
                key = (abs(lever.col - current.col), lever.row - current.row, lever.row, lever.col, lever.object_id)
            elif action == ACTION_UP and lever.row < current.row:
                key = (abs(lever.col - current.col), current.row - lever.row, lever.row, lever.col, lever.object_id)
            else:
                continue
            candidates.append((key, index))
        return min(candidates)[1] if candidates else None

    def _move_selector(self, action: int) -> None:
        target = self._selection_target(action)
        if target is not None:
            self.selected = target
            self._finish_or_lose()
            return
        self.nudge_delta = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}[action]
        self._start_animation("nudge", 2)

    def _pull_selected(self) -> None:
        lever = self.levers[self.selected]
        if self.selected in self.spent:
            self._start_animation("spent", 2)
            return
        if self.board[lever.row][lever.col]:
            self._start_animation("clamped", 2)
            return

        self.undo_history.append(self._snapshot())
        self.pre_board = [row[:] for row in self.board]
        if lever.orientation == "H":
            self.affected_cells = {(lever.row, col) for col in range(len(self.board[0]))}
        else:
            self.affected_cells = {(row, lever.col) for row in range(len(self.board))}
        for row, col in self.affected_cells:
            self.board[row][col] = not self.board[row][col]
        self.spent.add(self.selected)
        self.solved = all(all(row) for row in self.board)
        self.terminal = self.solved
        if self.solved:
            self.pending_loss = False
        self._start_animation("pull", 4)

    def _undo(self) -> None:
        if self.undo_history:
            self._restore(self.undo_history.pop())
        self.animation_kind = ""
        self.animation_phase = 0
        self.affected_cells.clear()
        self.pending_loss = self.remaining_steps == 0 and not self.solved
        self._finish_or_lose()

    def step(self) -> None:
        if self.animation_kind:
            self._advance_animation()
            return

        action = _action_id(self.action.id)
        if action == _action_id(GameAction.RESET):
            self.complete_action()
            return
        if self.terminal:
            self.complete_action()
            return

        self._spend_action()
        if action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            self._move_selector(action)
        elif action == ACTION_SPACE:
            self._pull_selected()
        elif action == ACTION_UNDO:
            self._undo()
        else:
            self._finish_or_lose()
