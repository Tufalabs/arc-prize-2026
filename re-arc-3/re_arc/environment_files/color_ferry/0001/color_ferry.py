from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

VIEW_SIZE = 64
BOARD_ORIGIN = 6
CELL_SIZE = 3

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7
MOVE_DELTAS = {ACTION_UP: (-1, 0), ACTION_DOWN: (1, 0), ACTION_LEFT: (0, -1), ACTION_RIGHT: (0, 1)}

BACKGROUND = 5
WALL = 4
WALL_HIGHLIGHT = 3
FLOOR = 1
WHITE = 0
BLUE = 9
BLUE_ACCENT = 10
YELLOW = 11
YELLOW_ACCENT = 12
MAGENTA = 6
MAGENTA_ACCENT = 7
DOCK = 14
INVALID = 8

COLOR_BY_SYMBOL = {"B": "blue", "Y": "yellow", "M": "magenta"}
BASIN_COLOR_BY_SYMBOL = {"b": "blue", "y": "yellow", "m": "magenta"}
BASE_COLOR = {"blue": BLUE, "yellow": YELLOW, "magenta": MAGENTA}
ACCENT_COLOR = {"blue": BLUE_ACCENT, "yellow": YELLOW_ACCENT, "magenta": MAGENTA_ACCENT}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    start_color: str
    baseline_actions: int


LEVEL_SPECS = (
    LevelSpec(
        "Two Shutters",
        (
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "########DD#######",
            "#########.#######",
            "#####.YSB.#######",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
        ),
        "blue",
        4,
    ),
    LevelSpec(
        "The Basin Repaints the Stone",
        (
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#########DD######",
            "#########.#######",
            "#########.#######",
            "####.YS.B.#######",
            "######b##########",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
        ),
        "yellow",
        8,
    ),
    LevelSpec(
        "Blue, Yellow, Blue",
        (
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "############D####",
            "#########b.BD####",
            "#########.#######",
            "######y.Y.#######",
            "######.##########",
            "####SB.##########",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
        ),
        "blue",
        12,
    ),
    LevelSpec(
        "The Basin You Avoid",
        (
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#####......######",
            "#####.####.Y.####",
            "#####.##...#.####",
            "#####.##.###..###",
            "#####y..b####.###",
            "####..#######.###",
            "####.#####B..b###",
            "####S##D...######",
            "#######D#########",
            "#################",
            "#################",
            "#################",
        ),
        "blue",
        29,
    ),
    LevelSpec(
        "The Magenta Transfer",
        (
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "####...Y#########",
            "####.##.#.Y.m####",
            "###y.##.#.##.####",
            "###.##....##.####",
            "###.##.#..##.####",
            "###.##.##.M..####",
            "###B##M##########",
            "###.##...b.B.####",
            "###S########DD###",
            "#################",
            "#################",
            "#################",
        ),
        "blue",
        45,
    ),
    LevelSpec(
        "The Harmless Basin Becomes a Trap",
        (
            "#################",
            "#############D###",
            "#############D###",
            "#######...###D###",
            "#######.#.###.###",
            "#######.#b..#.###",
            "#######Y###.#B###",
            "###...Y.B...#b..#",
            "###.###B#####Y#.#",
            "###y###...###.#.#",
            "###.#####.###.#.#",
            "#...#####mM..m#.#",
            "#.###########.#.#",
            "#.###########yY.#",
            "#M###############",
            "#S###############",
            "#################",
        ),
        "magenta",
        56,
    ),
    LevelSpec(
        "Braided Harbors",
        (
            "#################",
            "#################",
            "#################",
            "#####....m###mM.#",
            "#####.###M###.#.#",
            "#####.###...#.#.#",
            "#####Y#####.#.#.#",
            "###.Y.B..yY.Y.#.#",
            "###.#.#####M###.#",
            "###.#...###.###.#",
            "###.###B###.###.#",
            "#..y###b....###b#",
            "#.#############B#",
            "#.#############.#",
            "#B#############.#",
            "#S#####DDD......#",
            "#################",
        ),
        "blue",
        72,
    ),
)


def _pixel_origin(cell: tuple[int, int]) -> tuple[int, int]:
    row, col = cell
    return BOARD_ORIGIN + CELL_SIZE * col, BOARD_ORIGIN + CELL_SIZE * row


def _motif(color: str) -> np.ndarray:
    base = BASE_COLOR[color]
    accent = ACCENT_COLOR[color]
    pixels = np.full((3, 3), base, dtype=np.int8)
    if color == "blue":
        pixels[1, :] = accent
    elif color == "yellow":
        pixels[:, 1] = accent
    else:
        np.fill_diagonal(pixels, accent)
    return pixels


def _build_level(spec: LevelSpec) -> Level:
    if len(spec.rows) != 17 or any(len(row) != 17 for row in spec.rows):
        raise ValueError(f"{spec.name} must be a 17x17 ASCII map")
    starts = [(r, c) for r, row in enumerate(spec.rows) for c, char in enumerate(row) if char == "S"]
    if len(starts) != 1:
        raise ValueError(f"{spec.name} must contain exactly one start")
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), BACKGROUND, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level(
        sprites=[board],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={
            "rows": spec.rows,
            "start": starts[0],
            "start_color": spec.start_color,
            "baseline_actions": spec.baseline_actions,
            "step_budget": spec.baseline_actions * 6,
        },
    )


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


class ColorFerry(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "color_ferry-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BACKGROUND, BACKGROUND),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UNDO],
            seed,
        )

    def on_set_level(self, _level: Level) -> None:
        self.board_sprite = self.current_level.get_sprites_by_tag("board")[0]
        self.rows = tuple(self.current_level.get_data("rows"))
        self.agent_position = tuple(self.current_level.get_data("start"))
        self.carried_color = str(self.current_level.get_data("start_color"))
        self.step_budget = int(self.current_level.get_data("step_budget"))
        self.remaining_steps = self.step_budget
        self.complete = False
        self.undo_history: list[tuple[tuple[int, int], str, bool]] = []
        self.animation_phases: list[str] = []
        self.animation_context: dict[str, object] = {}
        self._terminal_after_animation = False
        self._sync_visuals("final")

    def _char_at(self, cell: tuple[int, int]) -> str:
        row, col = cell
        if not (0 <= row < 17 and 0 <= col < 17):
            return "#"
        return self.rows[row][col]

    def _spend_step(self) -> bool:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.remaining_steps > 0:
            return False
        self.animation_phases.clear()
        self._sync_visuals("final")
        self.lose()
        self.complete_action()
        return True

    def _queue_animation(self, phases: list[str], **context: object) -> None:
        self.animation_phases = phases
        self.animation_context = context

    def _finish_success(self) -> None:
        self.next_level()
        self.complete_action()

    def _play_animation_frame(self) -> None:
        phase = self.animation_phases.pop(0)
        self._sync_visuals(phase)
        if self.animation_phases:
            return
        if self._terminal_after_animation:
            self._terminal_after_animation = False
            self._finish_success()
        else:
            self.complete_action()

    def _resolve_move(self, dr: int, dc: int) -> None:
        if self.complete:
            self._sync_visuals("complete_final")
            self.complete_action()
            return

        target = (self.agent_position[0] + dr, self.agent_position[1] + dc)
        char = self._char_at(target)
        if char == "#":
            if self._spend_step():
                return
            self._queue_animation(["wall_bump", "final"])
            self._play_animation_frame()
            return
        if char in COLOR_BY_SYMBOL and COLOR_BY_SYMBOL[char] != self.carried_color:
            if self._spend_step():
                return
            self._queue_animation(["gate_flash_1", "gate_flash_2", "final"], struck_gate=target)
            self._play_animation_frame()
            return

        self.undo_history.append((self.agent_position, self.carried_color, self.complete))
        self.agent_position = target
        old_color = self.carried_color
        if char in BASIN_COLOR_BY_SYMBOL:
            self.carried_color = BASIN_COLOR_BY_SYMBOL[char]
        if char == "D":
            self.complete = True

        if self._spend_step():
            return
        if self.complete:
            self._terminal_after_animation = True
            self._queue_animation(["complete_flash", "complete_contract", "complete_final"])
        elif char in BASIN_COLOR_BY_SYMBOL and self.carried_color != old_color:
            self._queue_animation(
                ["recolor_pulse", "recolor_half", "final"],
                basin=target,
                old_color=old_color,
                new_color=self.carried_color,
            )
        elif char in BASIN_COLOR_BY_SYMBOL:
            self._queue_animation(["same_pulse", "final"], basin=target)
        else:
            self._sync_visuals("final")
            self.complete_action()
            return
        self._play_animation_frame()

    def _undo(self) -> None:
        if self.undo_history and not self.complete:
            self.agent_position, self.carried_color, self.complete = self.undo_history.pop()
        self.animation_phases.clear()
        self.animation_context = {}
        self._terminal_after_animation = False
        self._sync_visuals("final")
        self.complete_action()

    def step(self) -> None:
        if self.animation_phases:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._sync_visuals("final")
            self.complete_action()
            return
        if action == ACTION_UNDO:
            self._undo()
            return
        if action in MOVE_DELTAS:
            self._resolve_move(*MOVE_DELTAS[action])
            return
        # SPACE, CLICK, and unknown generic host input are exact no-ops.
        self.complete_action()

    def _draw_static_board(self, frame: np.ndarray, phase: str) -> None:
        for row, line in enumerate(self.rows):
            for col, char in enumerate(line):
                x0, y0 = _pixel_origin((row, col))
                frame[y0 : y0 + 3, x0 : x0 + 3] = WALL if char == "#" else FLOOR

        for row, line in enumerate(self.rows):
            for col, char in enumerate(line):
                cell = (row, col)
                x0, y0 = _pixel_origin(cell)
                if char in BASIN_COLOR_BY_SYMBOL:
                    color = BASIN_COLOR_BY_SYMBOL[char]
                    pixels = _motif(color)
                    pixels[1, 1] = FLOOR
                    frame[y0 : y0 + 3, x0 : x0 + 3] = pixels
                elif char in COLOR_BY_SYMBOL:
                    self._draw_gate(frame, cell, COLOR_BY_SYMBOL[char], phase)
                elif char == "D":
                    frame[y0 : y0 + 3, x0 : x0 + 3] = DOCK
                    frame[y0 + 1, x0 + 1] = WHITE

        # A color-neutral connected cart emblem spans every dock platform.
        dock_cells = [(r, c) for r, line in enumerate(self.rows) for c, char in enumerate(line) if char == "D"]
        for row, col in dock_cells:
            x0, y0 = _pixel_origin((row, col))
            frame[y0 + 2, x0 : x0 + 3] = WHITE

    def _draw_gate(self, frame: np.ndarray, cell: tuple[int, int], color: str, phase: str) -> None:
        x0, y0 = _pixel_origin(cell)
        struck = self.animation_context.get("struck_gate") == cell
        if struck and phase == "gate_flash_1":
            frame[y0 : y0 + 3, x0 : x0 + 3] = INVALID
            frame[y0 + 1, x0 + 1] = WALL
            return
        if struck and phase == "gate_flash_2":
            frame[y0 : y0 + 3, x0 : x0 + 3] = _motif(color)
            frame[y0, x0] = frame[y0, x0 + 2] = INVALID
            frame[y0 + 2, x0] = frame[y0 + 2, x0 + 2] = INVALID
            return

        open_now = color == self.carried_color
        old_color = self.animation_context.get("old_color")
        new_color = self.animation_context.get("new_color")
        affected = color in {old_color, new_color}
        if phase == "recolor_pulse" and affected:
            motif = _motif(color)
            if color == old_color:
                # The formerly open shutter has just started extending inward.
                frame[y0 : y0 + 3, x0 : x0 + 3] = FLOOR
                frame[y0 : y0 + 3, x0] = motif[:, 0]
                frame[y0 : y0 + 3, x0 + 2] = motif[:, 2]
                frame[y0, x0 + 1] = frame[y0 + 2, x0 + 1] = BASE_COLOR[color]
            else:
                # The newly matching shutter has just started retracting.
                frame[y0 : y0 + 3, x0 : x0 + 3] = motif
                frame[y0 + 1, x0 : x0 + 3] = WALL
                frame[y0 : y0 + 3, x0 + 1] = WALL
                frame[y0 + 1, x0 + 1] = FLOOR
            return
        if phase == "recolor_half" and affected:
            frame[y0 : y0 + 3, x0 : x0 + 3] = FLOOR
            motif = _motif(color)
            frame[y0, x0 : x0 + 3] = motif[0]
            frame[y0 + 2, x0 : x0 + 3] = motif[2]
            frame[y0 + 1, x0] = frame[y0 + 1, x0 + 2] = WALL
            return
        if open_now:
            frame[y0 : y0 + 3, x0 : x0 + 3] = FLOOR
            motif = _motif(color)
            frame[y0 : y0 + 3, x0] = motif[:, 0]
            frame[y0 : y0 + 3, x0 + 2] = motif[:, 2]
            frame[y0 + 1, x0] = frame[y0 + 1, x0 + 2] = WALL_HIGHLIGHT
        else:
            frame[y0 : y0 + 3, x0 : x0 + 3] = _motif(color)
            frame[y0 + 1, x0 : x0 + 3] = WALL
            frame[y0 : y0 + 3, x0 + 1] = WALL

    def _draw_basin_halo(self, frame: np.ndarray, cell: tuple[int, int], color: str, bright: bool) -> None:
        x0, y0 = _pixel_origin(cell)
        halo_color = ACCENT_COLOR[color] if bright else BASE_COLOR[color]
        points = ((x0 - 2, y0 + 1), (x0 + 4, y0 + 1), (x0 + 1, y0 - 2), (x0 + 1, y0 + 4))
        for x, y in points:
            if 0 <= x < VIEW_SIZE and 0 <= y < VIEW_SIZE:
                frame[y, x] = halo_color

    def _draw_player(self, frame: np.ndarray, phase: str) -> None:
        if phase in {"complete_contract", "complete_final"}:
            return
        row, col = self.agent_position
        x0, y0 = _pixel_origin((row, col))
        left, top = x0 - 1, y0 - 1
        pixels = np.full((5, 5), -1, dtype=np.int8)
        pixels[0:3, 1:4] = _motif(self.carried_color)
        pixels[1:3, 0] = WHITE
        pixels[1:3, 4] = WHITE
        pixels[3, 1:4] = WHITE
        pixels[4, 1] = pixels[4, 3] = WALL_HIGHLIGHT
        if phase == "wall_bump":
            pixels[1:4, :] = np.where(pixels[1:4, :] >= 0, WALL_HIGHLIGHT, -1)
        y1, y2 = max(0, top), min(VIEW_SIZE, top + 5)
        x1, x2 = max(0, left), min(VIEW_SIZE, left + 5)
        source = pixels[y1 - top : y2 - top, x1 - left : x2 - left]
        frame[y1:y2, x1:x2] = np.where(source >= 0, source, frame[y1:y2, x1:x2])

    def _draw_completion(self, frame: np.ndarray, phase: str) -> None:
        if phase == "complete_flash":
            for row, line in enumerate(self.rows):
                for col, char in enumerate(line):
                    if char == "D":
                        x0, y0 = _pixel_origin((row, col))
                        frame[y0 : y0 + 3, x0 : x0 + 3] = WHITE
                        frame[y0 + 1, x0 + 1] = DOCK
        elif phase in {"complete_contract", "complete_final"}:
            x0, y0 = _pixel_origin(self.agent_position)
            frame[y0 : y0 + 3, x0 : x0 + 3] = DOCK
            frame[y0, x0 + 1] = WHITE
            frame[y0 + 1, x0 : x0 + 3] = WHITE
            frame[y0 + 2, x0] = frame[y0 + 2, x0 + 2] = WALL_HIGHLIGHT

    def _draw_budget(self, frame: np.ndarray) -> None:
        frame[60:63, 6:57] = WALL
        width = int(51 * self.remaining_steps / max(1, self.step_budget))
        if width:
            frame[60:63, 6 : 6 + width] = ACCENT_COLOR[self.carried_color]
        frame[59, 6] = frame[59, 56] = WHITE

    def _sync_visuals(self, phase: str) -> None:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), BACKGROUND, dtype=np.int8)
        self._draw_static_board(frame, phase)
        char = self._char_at(self.agent_position)
        if char in BASIN_COLOR_BY_SYMBOL:
            bright = phase in {"recolor_pulse", "same_pulse"}
            self._draw_basin_halo(frame, self.agent_position, BASIN_COLOR_BY_SYMBOL[char], bright)
        if phase == "recolor_pulse":
            basin = self.animation_context.get("basin")
            if isinstance(basin, tuple):
                self._draw_basin_halo(frame, basin, self.carried_color, True)
        self._draw_player(frame, phase)
        self._draw_completion(frame, phase)
        self._draw_budget(frame)
        self.board_sprite.pixels = frame
