from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "tumbling_blocks-0001"
CELL = 7
BOARD = 63

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = range(1, 8)
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
SIDE_FOR_MOVE = {UP: "north", DOWN: "south", LEFT: "west", RIGHT: "east"}

FLOOR = 0
CHASSIS = 1
GRAY = 2
SHADOW = 3
WALL = 4
BLACK = 5
MAGENTA = 6
PINK = 7
RED = 8
BLUE = 9
LIGHT_BLUE = 10
YELLOW = 11
ORANGE = 12
MAROON = 13
GREEN = 14
PURPLE = 15

YD, BP, GB, OR, MX, RD = "YD", "BP", "GB", "OR", "MX", "RD"
O0 = (YD, BP, GB, OR, MX, RD)
FACE_COLORS = {YD: YELLOW, BP: LIGHT_BLUE, GB: GREEN, OR: ORANGE, MX: MAGENTA, RD: RED}
GLYPHS = {
    YD: (".#.", "#.#", ".#."),
    BP: (".#.", "###", ".#."),
    GB: (".#.", ".#.", ".#."),
    OR: ("###", "#.#", "###"),
    MX: ("#.#", ".#.", "#.#"),
    RD: ("...", ".#.", "..."),
}
IDENTITY_COLOR = {"blue": BLUE, "pink": PINK}


@dataclass(frozen=True)
class CubeSpec:
    identity: str
    cell: tuple[int, int]


@dataclass(frozen=True)
class DockSpec:
    identity: str
    cell: tuple[int, int]
    orientation: tuple[str, str, str, str, str, str]


@dataclass(frozen=True)
class TerminalSpec:
    identity: str
    wall: tuple[int, int]
    contact: tuple[int, int]
    side: str
    face: str


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    cubes: tuple[CubeSpec, ...]
    docks: tuple[DockSpec, ...]
    terminals: tuple[TerminalSpec, ...]
    baseline: int


LEVEL_SPECS = (
    LevelSpec(
        "First Tumble",
        (
            "#########",
            "#########",
            "#....####",
            "#PC..####",
            "#.#d.####",
            "#....####",
            "#########",
            "#########",
            "#########",
        ),
        (CubeSpec("blue", (2, 3)),),
        (DockSpec("blue", (3, 4), (BP, GB, YD, MX, RD, OR)),),
        (),
        4,
    ),
    LevelSpec(
        "Same Top, Wrong Sides",
        (
            "#########",
            "#......##",
            "#.PC.#.##",
            "#....d.##",
            "#......##",
            "#......##",
            "#########",
            "#########",
            "#########",
        ),
        (CubeSpec("blue", (3, 2)),),
        (DockSpec("blue", (5, 3), (BP, YD, MX, RD, GB, OR)),),
        (),
        11,
    ),
    LevelSpec(
        "The Face Plate",
        (
            "#########",
            "#########",
            "###T#####",
            "#....d.##",
            "#..C...##",
            "#..P...##",
            "#......##",
            "#########",
            "#########",
        ),
        (CubeSpec("blue", (3, 4)),),
        (DockSpec("blue", (5, 3), (BP, YD, MX, RD, GB, OR)),),
        (TerminalSpec("blue", (3, 2), (3, 3), "north", YD),),
        5,
    ),
    LevelSpec(
        "Five Is One",
        (
            "........#",
            "PC.d....#",
            "........#",
            "...##...#",
            "...##...#",
            "........#",
            ".......T#",
            "........#",
            "#########",
        ),
        (CubeSpec("blue", (1, 1)),),
        (DockSpec("blue", (3, 1), (OR, MX, YD, GB, RD, BP)),),
        (TerminalSpec("blue", (7, 6), (6, 6), "east", BP),),
        26,
    ),
    LevelSpec(
        "Return Changed",
        (
            "....T....",
            "PC.......",
            ".........",
            "...##....",
            "...##.#..",
            "...##...T",
            ".d.......",
            ".........",
            ".........",
        ),
        (CubeSpec("blue", (1, 1)),),
        (DockSpec("blue", (1, 6), (OR, RD, MX, YD, GB, BP)),),
        (TerminalSpec("blue", (4, 0), (4, 1), "north", YD), TerminalSpec("blue", (8, 5), (7, 5), "east", GB)),
        49,
    ),
    LevelSpec(
        "Shared Turning Bay",
        (
            "###...###",
            "###...###",
            "###.....#",
            "........T",
            "..PBA...1",
            ".........",
            "###...T##",
            "###....##",
            "###.2..##",
        ),
        (CubeSpec("blue", (4, 4)), CubeSpec("pink", (3, 4))),
        (DockSpec("blue", (8, 4), O0), DockSpec("pink", (4, 8), (RD, GB, BP, MX, OR, YD))),
        (TerminalSpec("blue", (8, 3), (7, 3), "east", RD), TerminalSpec("pink", (6, 6), (5, 6), "east", MX)),
        51,
    ),
    LevelSpec(
        "Interleaved Circuit",
        (
            "####..###",
            "###...###",
            "###.....#",
            "##......T",
            "##PBA...1",
            "##.......",
            "###...T##",
            "###....##",
            "####2..##",
        ),
        (CubeSpec("blue", (4, 4)), CubeSpec("pink", (3, 4))),
        (DockSpec("blue", (8, 4), (GB, YD, BP, RD, OR, MX)), DockSpec("pink", (4, 8), (MX, BP, YD, OR, RD, GB))),
        (TerminalSpec("pink", (8, 3), (7, 3), "east", GB), TerminalSpec("blue", (6, 6), (5, 6), "east", RD)),
        85,
    ),
)


def _action_id(action: Any) -> int:
    return int(getattr(action, "value", action))


def _roll(orientation: tuple[str, ...], action: int) -> tuple[str, ...]:
    top, north, east, south, west, bottom = orientation
    if action == UP:
        return south, top, east, bottom, west, north
    if action == DOWN:
        return north, bottom, east, top, west, south
    if action == RIGHT:
        return west, north, top, south, bottom, east
    if action == LEFT:
        return east, north, bottom, south, top, west
    raise ValueError(f"Not a roll action: {action}")


def _put_face(pixels: np.ndarray, x: int, y: int, face: str) -> None:
    pixels[y : y + 3, x : x + 3] = FACE_COLORS[face]
    for gy, row in enumerate(GLYPHS[face]):
        for gx, mark in enumerate(row):
            if mark == "#":
                pixels[y + gy, x + gx] = BLACK


class TumblingBlocks(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = []
        for spec in LEVEL_SPECS:
            board = Sprite(
                np.full((64, 64), BLACK, dtype=np.int8),
                name="board",
                x=0,
                y=0,
                layer=0,
                collidable=False,
                tags=["board"],
            )
            levels.append(Level([board], grid_size=(64, 64), data={"spec": spec}, name=spec.name))
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, 64, 64, BLACK, BLACK),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.board_sprite = level.get_sprites_by_tag("board")[0]
        self.walls = {
            (x, y) for y, row in enumerate(self.spec.rows) for x, symbol in enumerate(row) if symbol in {"#", "T"}
        }
        players = [(x, y) for y, row in enumerate(self.spec.rows) for x, symbol in enumerate(row) if symbol == "P"]
        if len(players) != 1:
            raise ValueError(f"{self.spec.name} must contain exactly one pusher")
        self.pusher = players[0]
        self.cubes = [{"identity": cube.identity, "cell": cube.cell, "orientation": O0} for cube in self.spec.cubes]
        self.terminals = [
            {
                "identity": terminal.identity,
                "wall": terminal.wall,
                "contact": terminal.contact,
                "side": terminal.side,
                "face": terminal.face,
                "lit": False,
            }
            for terminal in self.spec.terminals
        ]
        self.step_budget = self.spec.baseline * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[dict[str, Any]] = []
        self.animation: dict[str, Any] | None = None
        self.feedback: dict[str, Any] | None = None
        self._render_board()

    def _snapshot(self) -> dict[str, Any]:
        return {
            "pusher": self.pusher,
            "cubes": [(cube["cell"], cube["orientation"]) for cube in self.cubes],
            "lit": [terminal["lit"] for terminal in self.terminals],
        }

    def _restore(self, snapshot: dict[str, Any]) -> None:
        self.pusher = snapshot["pusher"]
        for cube, (cell, orientation) in zip(self.cubes, snapshot["cubes"], strict=True):
            cube["cell"] = cell
            cube["orientation"] = orientation
        for terminal, lit in zip(self.terminals, snapshot["lit"], strict=True):
            terminal["lit"] = lit

    def _spend_step(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)

    def _is_open(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < 9 and 0 <= y < 9 and cell not in self.walls

    def _cube_at(self, cell: tuple[int, int]) -> int | None:
        for index, cube in enumerate(self.cubes):
            if cube["cell"] == cell:
                return index
        return None

    def _terminal_checks(self, cube_index: int) -> None:
        cube = self.cubes[cube_index]
        side_index = {"north": 1, "east": 2, "south": 3, "west": 4}
        for index, terminal in enumerate(self.terminals):
            if terminal["lit"] or terminal["contact"] != cube["cell"]:
                continue
            if terminal["identity"] != cube["identity"]:
                self.feedback = {"kind": "wrong_identity", "terminal": index, "cube": cube_index}
            elif cube["orientation"][side_index[terminal["side"]]] == terminal["face"]:
                terminal["lit"] = True
            else:
                self.feedback = {"kind": "wrong_face", "terminal": index, "cube": cube_index, "side": terminal["side"]}

    def _dock_status(self, dock: DockSpec) -> str:
        cube_index = self._cube_at(dock.cell)
        if cube_index is None:
            return "empty"
        cube = self.cubes[cube_index]
        if cube["identity"] == dock.identity and cube["orientation"] == dock.orientation:
            return "correct"
        return "wrong"

    def _is_solved(self) -> bool:
        if not all(terminal["lit"] for terminal in self.terminals):
            return False
        if len(self.cubes) != len(self.spec.docks):
            return False
        return all(self._dock_status(dock) == "correct" for dock in self.spec.docks)

    def _finish_or_continue(self) -> None:
        self._render_board()
        if self._is_solved():
            self.animation = {"kind": "complete", "phase": 0}
            return
        if self.remaining_steps == 0:
            self.lose()
        self.complete_action()

    def _start_feedback(self, kind: str, **values: Any) -> None:
        self.animation = {"kind": kind, "phase": 0, **values}
        self._render_board()

    def _resolve_move(self, action: int) -> None:
        self._spend_step()
        dx, dy = DELTAS[action]
        neighbor = (self.pusher[0] + dx, self.pusher[1] + dy)
        if not self._is_open(neighbor):
            self.feedback = {"kind": "wall", "side": SIDE_FOR_MOVE[action]}
            self._start_feedback("wall")
            return
        cube_index = self._cube_at(neighbor)
        if cube_index is None:
            self.undo_history.append(self._snapshot())
            self.pusher = neighbor
            self._finish_or_continue()
            return
        destination = (neighbor[0] + dx, neighbor[1] + dy)
        if not self._is_open(destination) or self._cube_at(destination) is not None:
            self.feedback = {"kind": "blocked", "cube": cube_index, "action": action}
            self._start_feedback("blocked", cube=cube_index, action=action)
            return
        self.undo_history.append(self._snapshot())
        self.animation = {
            "kind": "roll",
            "phase": 0,
            "cube": cube_index,
            "action": action,
            "old_cell": neighbor,
            "new_cell": destination,
            "old_orientation": self.cubes[cube_index]["orientation"],
            "new_orientation": _roll(self.cubes[cube_index]["orientation"], action),
        }
        self._render_board()

    def _advance_animation(self) -> None:
        assert self.animation is not None
        kind = self.animation["kind"]
        if kind == "roll":
            self.animation["phase"] += 1
            if self.animation["phase"] < 3:
                self._render_board()
                return
            cube_index = self.animation["cube"]
            self.cubes[cube_index]["cell"] = self.animation["new_cell"]
            self.cubes[cube_index]["orientation"] = self.animation["new_orientation"]
            self.pusher = self.animation["old_cell"]
            self.animation = None
            self._terminal_checks(cube_index)
            self._finish_or_continue()
            return
        if kind == "complete":
            self.animation["phase"] += 1
            self._render_board(completion_flash=True)
            if self.animation["phase"] >= 3:
                self.next_level()
                self.complete_action()
            return
        self.animation["phase"] += 1
        if self.animation["phase"] >= 1:
            self.animation = None
            self.feedback = None
            self._finish_or_continue()

    def step(self) -> None:
        if self.animation is not None:
            self._advance_animation()
            return
        if self.action.id == GameAction.RESET:
            self._render_board()
            self.complete_action()
            return
        action = _action_id(self.action.id)
        self.feedback = None
        if action in DELTAS:
            self._resolve_move(action)
            return
        if action in {SPACE, CLICK}:
            self.feedback = {"kind": "neutral"}
            self._start_feedback("neutral")
            return
        if action == UNDO:
            self._spend_step()
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self.feedback = None
            self._finish_or_continue()
            return
        self.complete_action()

    def _draw_glyph(self, frame: np.ndarray, x: int, y: int, face: str) -> None:
        _put_face(frame, x, y, face)

    def _draw_dock(self, frame: np.ndarray, dock: DockSpec, completion_flash: bool) -> None:
        x0, y0 = dock.cell[0] * CELL, dock.cell[1] * CELL
        top, north, east, south, west, _bottom = dock.orientation
        frame[y0, x0 + 2 : x0 + 5] = FACE_COLORS[north]
        frame[y0 + 2 : y0 + 5, x0 + 6] = FACE_COLORS[east]
        frame[y0 + 6, x0 + 2 : x0 + 5] = FACE_COLORS[south]
        frame[y0 + 2 : y0 + 5, x0] = FACE_COLORS[west]
        self._draw_glyph(frame, x0 + 2, y0 + 2, top)
        identity_color = IDENTITY_COLOR[dock.identity]
        frame[y0, x0] = identity_color
        frame[y0 + 6, x0 + 6] = identity_color
        status = self._dock_status(dock)
        tick_color = {"empty": GRAY, "wrong": MAROON, "correct": GREEN}[status]
        if completion_flash and status == "correct":
            tick_color = GREEN
        frame[y0 + 1, x0] = tick_color
        frame[y0 + 5, x0 + 6] = tick_color
        if completion_flash and status == "correct":
            frame[y0, x0 + 1] = GREEN
            frame[y0 + 1, x0 + 6] = GREEN
            frame[y0 + 5, x0] = GREEN
            frame[y0 + 6, x0 + 5] = GREEN

    def _draw_terminal(self, frame: np.ndarray, index: int, terminal: dict[str, Any], flash: bool) -> None:
        x0, y0 = terminal["wall"][0] * CELL, terminal["wall"][1] * CELL
        rim = GREEN if terminal["lit"] or flash else GRAY
        feedback = self.feedback
        if feedback and feedback.get("terminal") == index and feedback["kind"] == "wrong_face":
            rim = MAROON
        plate_x, plate_y = x0 + 1, y0 + 1
        frame[plate_y : plate_y + 5, plate_x : plate_x + 5] = rim
        self._draw_glyph(frame, plate_x + 1, plate_y + 1, terminal["face"])
        pin = IDENTITY_COLOR[terminal["identity"]]
        if feedback and feedback.get("terminal") == index and feedback["kind"] == "wrong_identity":
            pin = GRAY
        frame[plate_y, plate_x] = pin
        frame[plate_y + 4, plate_x + 4] = pin
        if flash:
            frame[y0, x0 + 1 : x0 + 6] = GREEN
            frame[y0 + 6, x0 + 1 : x0 + 6] = GREEN
            frame[y0 + 1 : y0 + 6, x0] = GREEN
            frame[y0 + 1 : y0 + 6, x0 + 6] = GREEN
        side = terminal["side"]
        if side == "north":
            frame[y0 + 6, x0 + 2 : x0 + 5] = rim
        elif side == "south":
            frame[y0, x0 + 2 : x0 + 5] = rim
        elif side == "west":
            frame[y0 + 2 : y0 + 5, x0 + 6] = rim
        else:
            frame[y0 + 2 : y0 + 5, x0] = rim

    def _draw_cube_pixels(
        self,
        frame: np.ndarray,
        pixel_x: int,
        pixel_y: int,
        identity: str,
        orientation: tuple[str, ...],
        *,
        compress: bool = False,
        dominant_side: str | None = None,
        flash_side: str | None = None,
    ) -> None:
        top, north, east, south, west, bottom = orientation
        for yy in range(1, 7):
            for xx in range(1, 7):
                px, py = pixel_x + xx, pixel_y + yy
                if 0 <= px < BOARD and 0 <= py < BOARD:
                    frame[py, px] = SHADOW
        body = np.full((5, 5), CHASSIS, dtype=np.int8)
        body[0, 1:4] = FACE_COLORS[north]
        body[1:4, 4] = FACE_COLORS[east]
        body[4, 1:4] = FACE_COLORS[south]
        body[1:4, 0] = FACE_COLORS[west]
        _put_face(body, 1, 1, top)
        if compress:
            body[1, 1:4] = CHASSIS
            body[3, 1:4] = CHASSIS
        if dominant_side is not None:
            color = FACE_COLORS[{"north": north, "east": east, "south": south, "west": west}[dominant_side]]
            if dominant_side in {"north", "south"}:
                row = 0 if dominant_side == "north" else 4
                body[row, :] = color
            else:
                col = 0 if dominant_side == "west" else 4
                body[:, col] = color
        body[0, 0] = IDENTITY_COLOR[identity]
        body[0, 4] = IDENTITY_COLOR[identity]
        if flash_side is not None:
            if flash_side == "north":
                body[0, :] = MAROON
            elif flash_side == "south":
                body[4, :] = MAROON
            elif flash_side == "west":
                body[:, 0] = MAROON
            else:
                body[:, 4] = MAROON
        self._blit(frame, body, pixel_x, pixel_y)
        chip = np.array([[BLACK, FACE_COLORS[bottom]], [FACE_COLORS[bottom], FACE_COLORS[bottom]]], dtype=np.int8)
        self._blit(frame, chip, pixel_x + 5, pixel_y + 5)

    @staticmethod
    def _blit(frame: np.ndarray, pixels: np.ndarray, x: int, y: int) -> None:
        height, width = pixels.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(BOARD, x + width), min(BOARD, y + height)
        if x1 < x2 and y1 < y2:
            frame[y1:y2, x1:x2] = pixels[y1 - y : y2 - y, x1 - x : x2 - x]

    def _draw_pusher(self, frame: np.ndarray) -> None:
        x0, y0 = self.pusher[0] * CELL, self.pusher[1] * CELL
        pixels = np.array(
            [
                [-1, PURPLE, PURPLE, PURPLE, -1],
                [PURPLE, PURPLE, BLACK, PURPLE, PURPLE],
                [PURPLE, BLACK, PURPLE, BLACK, PURPLE],
                [PURPLE, PURPLE, PURPLE, PURPLE, PURPLE],
                [-1, PURPLE, -1, PURPLE, -1],
            ],
            dtype=np.int8,
        )
        if self.feedback and self.feedback["kind"] == "neutral":
            pixels[pixels == PURPLE] = GRAY
        if self.feedback and self.feedback["kind"] == "wall":
            side = self.feedback["side"]
            if side == "north":
                pixels[0, :] = MAROON
            elif side == "south":
                pixels[4, :] = MAROON
            elif side == "west":
                pixels[:, 0] = MAROON
            else:
                pixels[:, 4] = MAROON
        for yy in range(5):
            for xx in range(5):
                if pixels[yy, xx] >= 0:
                    frame[y0 + 1 + yy, x0 + 1 + xx] = pixels[yy, xx]

    def _render_board(self, *, completion_flash: bool = False) -> None:
        frame = np.full((64, 64), BLACK, dtype=np.int8)
        for y, row in enumerate(self.spec.rows):
            for x, symbol in enumerate(row):
                color = WALL if symbol in {"#", "T"} else FLOOR
                frame[y * CELL : (y + 1) * CELL, x * CELL : (x + 1) * CELL] = color
        for dock in self.spec.docks:
            self._draw_dock(frame, dock, completion_flash)
        for index, terminal in enumerate(self.terminals):
            self._draw_terminal(frame, index, terminal, completion_flash)

        roll = self.animation if self.animation and self.animation["kind"] == "roll" else None
        blocked = self.animation if self.animation and self.animation["kind"] == "blocked" else None
        for index, cube in enumerate(self.cubes):
            cell = cube["cell"]
            orientation = cube["orientation"]
            offset_x = offset_y = 0
            compress = False
            dominant_side = None
            flash_side = None
            if roll and roll["cube"] == index:
                phase = roll["phase"]
                distance = (2, 4, 6)[phase]
                dx, dy = DELTAS[roll["action"]]
                offset_x, offset_y = dx * distance, dy * distance
                compress = phase == 0
                dominant_side = SIDE_FOR_MOVE[roll["action"]] if phase == 1 else None
                orientation = roll["new_orientation"] if phase == 2 else roll["old_orientation"]
            elif blocked and blocked["cube"] == index and blocked["phase"] == 0:
                dx, dy = DELTAS[blocked["action"]]
                offset_x, offset_y = dx, dy
                flash_side = SIDE_FOR_MOVE[blocked["action"]]
            if self.feedback and self.feedback.get("cube") == index and self.feedback["kind"] == "wrong_face":
                flash_side = self.feedback["side"]
            self._draw_cube_pixels(
                frame,
                cell[0] * CELL + offset_x,
                cell[1] * CELL + offset_y,
                cube["identity"],
                orientation,
                compress=compress,
                dominant_side=dominant_side,
                flash_side=flash_side,
            )
        self._draw_pusher(frame)
        filled = int(62 * self.remaining_steps / max(1, self.step_budget))
        frame[63, 0:63] = SHADOW
        if filled:
            frame[63, 0:filled] = GREEN
        frame[:, 63] = BLACK
        self.board_sprite.pixels = frame


def _validate_orientation_algebra() -> None:
    orientation = O0
    for _ in range(4):
        orientation = _roll(orientation, RIGHT)
    assert orientation == O0
    orientation = O0
    for _ in range(4):
        orientation = _roll(orientation, DOWN)
    assert orientation == O0
    assert _roll(_roll(O0, RIGHT), LEFT) == O0
    assert _roll(_roll(O0, UP), DOWN) == O0
    orientation = O0
    for action in (RIGHT, DOWN, LEFT, UP):
        orientation = _roll(orientation, action)
    assert orientation == (MX, YD, OR, RD, BP, GB)
    orientation = O0
    for action in (UP, RIGHT, DOWN, DOWN, RIGHT):
        orientation = _roll(orientation, action)
    assert orientation == (BP, RD, GB, YD, MX, OR)
    orientation = O0
    for action in (DOWN, DOWN, RIGHT, RIGHT, UP):
        orientation = _roll(orientation, action)
    assert orientation == (BP, YD, MX, RD, GB, OR)


_validate_orientation_algebra()
