from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
CELL_SIZE = 5
BODY_SIZE = 4

COLOR_CANVAS = 0
COLOR_EMPTY = 1
COLOR_FLOOR = 4
COLOR_INSET_LIGHT = 5
COLOR_OUTLINE = 11
COLOR_COMPLETE = 14
COLOR_ERROR = 8

LETTER_TO_COLOR = {"M": 6, "R": 8, "B": 9, "C": 10, "Y": 11, "O": 12, "G": 14, "P": 15}
INSETS = {
    6: ((1, 1),),
    8: ((2, 1),),
    9: ((1, 2),),
    10: ((2, 2),),
    11: ((1, 1), (2, 1)),
    12: ((1, 2), (2, 2)),
    14: ((1, 1), (2, 2)),
    15: ((2, 1), (1, 2)),
}
INSET_INK = {6: 0, 8: 0, 9: 0, 10: 5, 11: 5, 12: 5, 14: 5, 15: 0}


class Feedback(Enum):
    NONE = auto()
    VALID_PAIR = auto()
    INVALID_BLOCK = auto()
    EMPTY_CLICK = auto()
    COMPLETE = auto()


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    baseline: int


LEVEL_SPECS = (
    LevelSpec("Two Orientations", ("...Y", "BB.Y"), 2),
    LevelSpec("The Drop", ("P.", "G.", "OP", "YG", "BO", "BY"), 5),
    LevelSpec("One Row or Two", (".GC", ".RR", ".PC", ".BM", "BPR", "GMR"), 7),
    LevelSpec(
        "Expose Before Aligning",
        ("Y.O.Y.B.", "P.P.ORG.", "M.G.MCP.", "PRYRCPR.", "CMCCBCG.", "BGYMBPO.", "BYRPCOP.", "CMMORYOB"),
        25,
    ),
    LevelSpec(
        "Preserve the Bridge",
        (
            "MB........Y.",
            "BC.....R.PG.",
            "MP..G..C.BC.",
            "YR..CP.R.OG.",
            "BY..RY.G.RO.",
            "PM..MBYR.GB.",
            "YMCRYGRY.GY.",
            "PGOYPYPYORB.",
            "BYBOMPBRPCY.",
            "RGPBCYCGRYBO",
        ),
        40,
    ),
    LevelSpec(
        "Coupled Wave",
        (
            "MB..G..R..Y.",
            "BC..CP.C.PG.",
            "MP..RY.R.BC.",
            "YR.OMPMG.OG.",
            "BY.RCBPM.RO.",
            "PM.YOCYR.GB.",
            "YMCOYGRY.GY.",
            "PGOBPYPYORB.",
            "BYBRMPBRPCY.",
            "RGPRCYCGRYBO",
        ),
        45,
    ),
    LevelSpec(
        "Whole-Board Orchestration",
        (
            "MB..G....PY.",
            "BC..C..R.BG.",
            "MP..RP.C.OC.",
            "YR.OMY.R.RG.",
            "BY.RCPMG.GO.",
            "PM.YOBPMOMB.",
            "YM.OYCYRPGM.",
            "PGCBPGRYBRY.",
            "BYORMYPYPCB.",
            "CCBGGPBRPBY.",
            "RGPRCYCGRYBO",
        ),
        50,
    ),
)


def _parse_columns(rows: tuple[str, ...]) -> list[list[int]]:
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Drop & Pair level rows must have equal width")
    return [
        [LETTER_TO_COLOR[rows[row][x]] for row in range(len(rows) - 1, -1, -1) if rows[row][x] != "."]
        for x in range(width)
    ]


class DropPairView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: DropPair | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_CANVAS
        ox, oy = game.board_origin

        for x in range(game.board_width):
            for y in range(game.board_height):
                px, py = game.cell_top_left(x, y)
                frame[py : py + BODY_SIZE, px : px + BODY_SIZE] = COLOR_EMPTY

        if game.animation_kind == "fall":
            for x, old_y, color, final_y in game.falling_blocks:
                distance = old_y - final_y
                displacement = (5 * distance * game.fall_phase + 1) // 3
                px = ox + CELL_SIZE * x
                py = oy + CELL_SIZE * (game.board_height - 1 - old_y) + displacement
                self._draw_block(frame, px, py, color)
            for x, y, color in game.stationary_blocks:
                px, py = game.cell_top_left(x, y)
                self._draw_block(frame, px, py, color)
        else:
            depress = 1 if game.cue_depressed else 0
            shake = game.shake_offset if game.animation_kind == "invalid" else 0
            for x, column in enumerate(game.columns):
                for y, color in enumerate(column):
                    px, py = game.cell_top_left(x, y)
                    self._draw_block(frame, px + shake if (x, y) == game.invalid_cell else px, py + depress, color)

        if game.animation_kind == "outline":
            for x, y in game.pair_cells:
                px, py = game.cell_top_left(x, y)
                frame[py, px : px + BODY_SIZE] = COLOR_OUTLINE
                frame[py + BODY_SIZE - 1, px : px + BODY_SIZE] = COLOR_OUTLINE
                frame[py : py + BODY_SIZE, px] = COLOR_OUTLINE
                frame[py : py + BODY_SIZE, px + BODY_SIZE - 1] = COLOR_OUTLINE

        if game.animation_kind == "invalid" and game.invalid_cell is not None:
            x, y = game.invalid_cell
            color = game.columns[x][y]
            px, py = game.cell_top_left(x, y)
            px += game.shake_offset
            ink = COLOR_CANVAS if color == COLOR_ERROR else COLOR_ERROR
            for i in range(BODY_SIZE):
                if 0 <= px + i < VIEW_SIZE:
                    frame[py + i, px + i] = ink
                    frame[py + i, px + BODY_SIZE - 1 - i] = ink

        floor_color = COLOR_COMPLETE if game.floor_flash else COLOR_FLOOR
        floor_y = oy + CELL_SIZE * game.board_height
        frame[floor_y : floor_y + 2, ox : ox + CELL_SIZE * game.board_width] = floor_color
        self._draw_budget(frame, game.remaining_steps, game.step_budget)
        return frame

    @staticmethod
    def _draw_block(frame: np.ndarray, px: int, py: int, color: int) -> None:
        y0, y1 = max(0, py), min(VIEW_SIZE, py + BODY_SIZE)
        x0, x1 = max(0, px), min(VIEW_SIZE, px + BODY_SIZE)
        if y0 < y1 and x0 < x1:
            frame[y0:y1, x0:x1] = color
        for ix, iy in INSETS[color]:
            xx, yy = px + ix, py + iy
            if 0 <= xx < VIEW_SIZE and 0 <= yy < VIEW_SIZE:
                frame[yy, xx] = INSET_INK[color]

    @staticmethod
    def _draw_budget(frame: np.ndarray, remaining: int, budget: int) -> None:
        frame[62:64, 2:62] = COLOR_EMPTY
        filled = max(0, min(60, (60 * remaining) // max(1, budget)))
        if filled:
            frame[62:64, 2 : 2 + filled] = COLOR_OUTLINE


class DropPair(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.view = DropPairView()
        self.columns: list[list[int]] = []
        self.board_width = 0
        self.board_height = 0
        self.step_budget = 1
        self.remaining_steps = 1
        self.feedback = Feedback.NONE
        self.won = False
        self.undo_history: list[list[list[int]]] = []
        self.animation_kind = ""
        self.animation_step = 0
        self.pair_cells: tuple[tuple[int, int], ...] = ()
        self.invalid_cell: tuple[int, int] | None = None
        self.shake_offset = 0
        self.fall_phase = 0
        self.falling_blocks: list[tuple[int, int, int, int]] = []
        self.stationary_blocks: list[tuple[int, int, int]] = []
        self.pending_columns: list[list[int]] | None = None
        self.floor_flash = False
        self.cue_depressed = False
        self._first_cue_done = False

        levels = [Level(sprites=[], grid_size=(64, 64), data={"spec": spec}, name=spec.name) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, 64, 64, COLOR_CANVAS, COLOR_CANVAS, [self.view])
        super().__init__("drop_pair-0001", levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6, 7], seed=seed)
        self.view.game = self

    @property
    def board_origin(self) -> tuple[int, int]:
        return (VIEW_SIZE - CELL_SIZE * self.board_width) // 2, (62 - CELL_SIZE * self.board_height) // 2

    def cell_top_left(self, x: int, y: int) -> tuple[int, int]:
        ox, oy = self.board_origin
        return ox + CELL_SIZE * x, oy + CELL_SIZE * (self.board_height - 1 - y)

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.columns = _parse_columns(spec.rows)
        self.board_width = len(spec.rows[0])
        self.board_height = len(spec.rows)
        self.step_budget = spec.baseline * 6
        self.remaining_steps = self.step_budget
        self.feedback = Feedback.NONE
        self.won = False
        self.undo_history = []
        self._clear_animation()
        self.cue_depressed = self.level_index == 0 and not self._first_cue_done

    def _clear_animation(self) -> None:
        self.animation_kind = ""
        self.animation_step = 0
        self.pair_cells = ()
        self.invalid_cell = None
        self.shake_offset = 0
        self.fall_phase = 0
        self.falling_blocks = []
        self.stationary_blocks = []
        self.pending_columns = None
        self.floor_flash = False

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            if self.cue_depressed:
                if self.animation_step == 0:
                    self.animation_step = 1
                    return
                self.cue_depressed = False
                self._first_cue_done = True
            self.complete_action()
            return

        if self.animation_kind:
            self._advance_animation()
            return

        action_id = self.action.id.value
        if action_id not in (1, 2, 3, 4, 5, 6, 7):
            self.complete_action()
            return

        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.remaining_steps == 0:
            self.feedback = Feedback.EMPTY_CLICK
            self.lose()
            self.complete_action()
            return

        if action_id in (1, 2, 3, 4, 5):
            self.feedback = Feedback.NONE
            self.complete_action()
            return
        if action_id == 7:
            self._undo()
            return
        self._handle_click()

    def _undo(self) -> None:
        if self.undo_history:
            self.columns = self.undo_history.pop()
            self.won = False
        self.feedback = Feedback.NONE
        self._clear_animation()
        self.complete_action()

    def _handle_click(self) -> None:
        px = int(self.action.data.get("x", 0))
        py = int(self.action.data.get("y", 0))
        ox, oy = self.board_origin
        if not (ox <= px < ox + CELL_SIZE * self.board_width and oy <= py < oy + CELL_SIZE * self.board_height):
            self.feedback = Feedback.EMPTY_CLICK
            self.complete_action()
            return
        x = (px - ox) // CELL_SIZE
        render_row = (py - oy) // CELL_SIZE
        y = self.board_height - 1 - render_row
        if y < 0 or y >= len(self.columns[x]):
            self.feedback = Feedback.EMPTY_CLICK
            self.complete_action()
            return

        color = self.columns[x][y]
        neighbors = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_width and 0 <= ny < len(self.columns[nx]) and self.columns[nx][ny] == color:
                neighbors.append((nx, ny))
        if len(neighbors) != 1:
            if len(neighbors) > 1 and self._debug:
                raise AssertionError("Authored invariant violated: block has multiple equal neighbors")
            self.feedback = Feedback.INVALID_BLOCK
            self.animation_kind = "invalid"
            self.animation_step = 0
            self.invalid_cell = (x, y)
            self.shake_offset = -1
            return

        pair = ((x, y), neighbors[0])
        self.undo_history.append([column.copy() for column in self.columns])
        self.pair_cells = pair
        self.pending_columns, self.falling_blocks, self.stationary_blocks = self._removal_result(pair)
        self.feedback = Feedback.VALID_PAIR
        self.animation_kind = "outline"
        self.animation_step = 0

    def _removal_result(
        self, pair: tuple[tuple[int, int], tuple[int, int]]
    ) -> tuple[list[list[int]], list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
        removed = set(pair)
        pending: list[list[int]] = []
        falling: list[tuple[int, int, int, int]] = []
        stationary: list[tuple[int, int, int]] = []
        for x, column in enumerate(self.columns):
            new_column = [color for y, color in enumerate(column) if (x, y) not in removed]
            pending.append(new_column)
            for old_y, color in enumerate(column):
                if (x, old_y) in removed:
                    continue
                final_y = old_y - sum(1 for rx, ry in removed if rx == x and ry < old_y)
                item = (x, old_y, color, final_y)
                if final_y != old_y:
                    falling.append(item)
                else:
                    stationary.append((x, old_y, color))
        return pending, falling, stationary

    def _advance_animation(self) -> None:
        if self.animation_kind == "invalid":
            self.animation_step += 1
            if self.animation_step == 1:
                self.shake_offset = 1
            else:
                self._clear_animation()
                self.complete_action()
            return

        if self.animation_kind == "outline":
            self.animation_kind = "fall"
            self.fall_phase = 1
            return
        if self.animation_kind == "fall" and self.fall_phase == 1:
            self.fall_phase = 2
            return
        if self.animation_kind == "fall":
            if self.pending_columns is None:
                raise RuntimeError("Missing pending Drop & Pair settlement")
            self.columns = self.pending_columns
            self.pending_columns = None
            self.won = all(not column for column in self.columns)
            if self.won:
                self.feedback = Feedback.COMPLETE
                self.animation_kind = "complete"
                self.floor_flash = True
                self.animation_step = 0
            else:
                self._clear_animation()
                self.complete_action()
            return
        if self.animation_kind == "complete":
            self.animation_step += 1
            if self.animation_step < 2:
                return
            self._clear_animation()
            self.complete_action()
            self.next_level()
