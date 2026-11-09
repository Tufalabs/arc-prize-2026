from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "leap_frog-0001"
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

VIEW_SIZE = 64
BOARD_ORIGIN = 5
CELL_PITCH = 6
TILE_SIZE = 5

COLOR_WHITE = 0
COLOR_GRAY = 2
COLOR_DARK_GRAY = 3
COLOR_BACKGROUND = 4
COLOR_PINK = 7
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_GREEN = 14
COLOR_PURPLE = 15

MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
KIND_COLOR = {"G": COLOR_GREEN, "R": COLOR_RED, "B": COLOR_BLUE, "P": COLOR_PURPLE}
JUMP_DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))


@dataclass(frozen=True)
class PlatformSpec:
    pid: str
    footprint: tuple[tuple[int, int], ...]
    rail: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PieceSpec:
    piece_id: str
    kind: str
    cell: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    floor: tuple[tuple[int, int], ...]
    platforms: tuple[PlatformSpec, ...]
    pieces: tuple[PieceSpec, ...]
    step_budget: int


LEVEL_SPECS = (
    LevelSpec(
        "First reduction chain",
        ((2, 4), (3, 4), (4, 4), (5, 4)),
        (),
        (PieceSpec("G1", "G", (2, 4)), PieceSpec("G2", "G", (3, 4)), PieceSpec("G3", "G", (5, 4))),
        24,
    ),
    LevelSpec(
        "Bridge carrier",
        ((2, 4), (3, 4)),
        (PlatformSpec("Y1", ((5, 4), (6, 4)), ((4, 4), (5, 4), (6, 4))),),
        (PieceSpec("G1", "G", (2, 4)), PieceSpec("G2", "G", (3, 4)), PieceSpec("G3", "G", (6, 4))),
        30,
    ),
    LevelSpec(
        "Synchronized carts",
        ((2, 4), (3, 4)),
        (
            PlatformSpec("Y1", ((5, 4), (5, 5)), ((4, 4), (5, 4), (4, 5), (5, 5))),
            PlatformSpec("Y2", ((6, 3), (6, 4)), ((5, 3), (6, 3), (5, 4), (6, 4))),
        ),
        (PieceSpec("G1", "G", (2, 4)), PieceSpec("G2", "G", (3, 4)), PieceSpec("G3", "G", (6, 4))),
        30,
    ),
    LevelSpec(
        "Permanent hurdle",
        ((4, 2), (4, 3), (4, 4), (3, 4), (5, 4)),
        (),
        (PieceSpec("G1", "G", (4, 2)), PieceSpec("G2", "G", (3, 4)), PieceSpec("P1", "P", (4, 3))),
        24,
    ),
    LevelSpec(
        "Riding post",
        ((4, 2), (4, 4), (3, 4), (5, 4)),
        (PlatformSpec("Y1", ((5, 3), (6, 3)), ((4, 3), (5, 3), (6, 3))),),
        (PieceSpec("G1", "G", (4, 2)), PieceSpec("G2", "G", (3, 4)), PieceSpec("P1", "P", (5, 3))),
        30,
    ),
    LevelSpec(
        "Red clearance",
        ((3, 4), (4, 4), (5, 4), (5, 3), (5, 2), (5, 5)),
        (),
        (
            PieceSpec("G1", "G", (3, 4)),
            PieceSpec("G2", "G", (4, 4)),
            PieceSpec("G3", "G", (5, 3)),
            PieceSpec("R1", "R", (5, 4)),
        ),
        36,
    ),
    LevelSpec(
        "Spend blue",
        ((4, 2), (4, 3), (4, 4), (3, 4), (5, 4)),
        (),
        (PieceSpec("G1", "G", (4, 2)), PieceSpec("G2", "G", (3, 4)), PieceSpec("B1", "B", (4, 3))),
        24,
    ),
    LevelSpec(
        "Move and spend blue",
        ((2, 2), (3, 2), (4, 1), (4, 3), (3, 3), (5, 3), (5, 4)),
        (PlatformSpec("Y1", ((5, 2), (6, 2)), ((4, 2), (5, 2), (6, 2))),),
        (
            PieceSpec("G1", "G", (4, 1)),
            PieceSpec("G2", "G", (3, 3)),
            PieceSpec("G3", "G", (5, 4)),
            PieceSpec("B1", "B", (2, 2)),
            PieceSpec("P1", "P", (3, 2)),
        ),
        54,
    ),
    LevelSpec(
        "Greedy trap",
        ((3, 2), (5, 2), (5, 1), (5, 3), (4, 3), (6, 2), (6, 1), (6, 0)),
        (
            PlatformSpec("Ypost", ((5, 2),), ((4, 2), (5, 2))),
            PlatformSpec("Y2", ((7, 3), (7, 4)), ((6, 3), (6, 4), (7, 3), (7, 4))),
            PlatformSpec("Y3", ((8, 3),), ((7, 3), (8, 3))),
        ),
        (
            PieceSpec("G1", "G", (5, 1)),
            PieceSpec("G2", "G", (4, 3)),
            PieceSpec("G3", "G", (7, 4)),
            PieceSpec("G4", "G", (6, 1)),
            PieceSpec("G5", "G", (8, 3)),
            PieceSpec("R1", "R", (6, 2)),
            PieceSpec("B1", "B", (3, 2)),
            PieceSpec("P1", "P", (5, 2)),
        ),
        90,
    ),
    LevelSpec(
        "Final network",
        ((2, 3), (3, 3), (3, 4), (5, 4), (5, 5), (5, 3), (6, 3), (6, 2), (6, 1), (6, 0), (5, 2), (5, 1)),
        (
            PlatformSpec("Cpost", ((6, 4),), ((4, 4), (5, 4), (6, 4))),
            PlatformSpec("Cland", ((6, 3),), ((4, 3), (5, 3), (6, 3))),
            PlatformSpec("C3", ((8, 4),), ((6, 4), (7, 4), (8, 4))),
            PlatformSpec("C5", ((8, 3),), ((7, 3), (8, 3))),
            PlatformSpec("Lift", ((5, 7), (5, 8)), ((5, 4), (5, 5), (5, 6), (5, 7), (5, 8))),
        ),
        (
            PieceSpec("G2source", "G", (2, 3)),
            PieceSpec("G1", "G", (5, 5)),
            PieceSpec("G3", "G", (8, 4)),
            PieceSpec("G4", "G", (6, 1)),
            PieceSpec("G5", "G", (8, 3)),
            PieceSpec("G6", "G", (5, 7)),
            PieceSpec("G7", "G", (5, 1)),
            PieceSpec("R1", "R", (6, 2)),
            PieceSpec("Bleft", "B", (3, 3)),
            PieceSpec("Bpost", "B", (3, 4)),
            PieceSpec("P1", "P", (6, 4)),
        ),
        150,
    ),
)


def _cell_rect(cell: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y = cell
    px = BOARD_ORIGIN + x * CELL_PITCH
    py = BOARD_ORIGIN + y * CELL_PITCH
    return px, py, px + TILE_SIZE, py + TILE_SIZE


def _cell_center(cell: tuple[int, int]) -> tuple[int, int]:
    x0, y0, _, _ = _cell_rect(cell)
    return x0 + 2, y0 + 2


def _screen_to_cell(x: int, y: int) -> tuple[int, int] | None:
    gx = (x - BOARD_ORIGIN) // CELL_PITCH
    gy = (y - BOARD_ORIGIN) // CELL_PITCH
    if not (0 <= gx <= 8 and 0 <= gy <= 8):
        return None
    x0, y0, _, _ = _cell_rect((gx, gy))
    if x < x0 or y < y0 or x >= x0 + CELL_PITCH or y >= y0 + CELL_PITCH:
        return None
    return int(gx), int(gy)


class LeapFrogView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: LeapFrog | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BACKGROUND
        self._draw_rails(frame, game)
        self._draw_floor(frame, game)
        self._draw_platforms(frame, game)
        self._draw_highlights(frame, game)
        self._draw_pieces(frame, game)
        self._draw_step_bar(frame, game)
        self._draw_flash(frame, game)
        return frame

    def _draw_floor(self, frame: np.ndarray, game: LeapFrog) -> None:
        for cell in game.floor:
            x0, y0, x1, y1 = _cell_rect(cell)
            frame[y0:y1, x0:x1] = COLOR_WHITE

    def _draw_rails(self, frame: np.ndarray, game: LeapFrog) -> None:
        rail_cells = set()
        for platform in game.platforms.values():
            rail_cells.update(platform["rail"])
        for cell in rail_cells:
            cx, cy = _cell_center(cell)
            frame[cy, cx - 2 : cx + 3] = COLOR_GRAY
            frame[cy - 2 : cy + 3, cx] = COLOR_GRAY

    def _draw_platforms(self, frame: np.ndarray, game: LeapFrog) -> None:
        for platform in game.platforms.values():
            cells = set(platform["footprint"])
            for cell in cells:
                x0, y0, x1, y1 = _cell_rect(cell)
                frame[y0:y1, x0:x1] = COLOR_YELLOW
                if (cell[0] + 1, cell[1]) in cells:
                    frame[y0:y1, x1 : x1 + 1] = COLOR_YELLOW
                if (cell[0], cell[1] + 1) in cells:
                    frame[y1 : y1 + 1, x0:x1] = COLOR_YELLOW

    def _draw_highlights(self, frame: np.ndarray, game: LeapFrog) -> None:
        for cell in game.legal_landings:
            cx, cy = _cell_center(cell)
            frame[cy - 2, cx - 1 : cx + 2] = COLOR_LIGHT_BLUE
            frame[cy + 2, cx - 1 : cx + 2] = COLOR_LIGHT_BLUE
            frame[cy - 1 : cy + 2, cx - 2] = COLOR_LIGHT_BLUE
            frame[cy - 1 : cy + 2, cx + 2] = COLOR_LIGHT_BLUE
        if game.selected_piece is not None and game.selected_piece in game.pieces:
            cell = game.pieces[game.selected_piece]["cell"]
            x0, y0, x1, y1 = _cell_rect(cell)
            frame[y0 - 1 : y1 + 1, x0 - 1] = COLOR_PINK
            frame[y0 - 1 : y1 + 1, x1] = COLOR_PINK
            frame[y0 - 1, x0 - 1 : x1 + 1] = COLOR_PINK
            frame[y1, x0 - 1 : x1 + 1] = COLOR_PINK

    def _draw_pieces(self, frame: np.ndarray, game: LeapFrog) -> None:
        for piece in game.pieces.values():
            cell = piece["cell"]
            kind = piece["kind"]
            cx, cy = _cell_center(cell)
            color = KIND_COLOR[kind]
            if kind == "P":
                frame[cy - 2, cx] = color
                frame[cy - 1, cx - 1 : cx + 2] = color
                frame[cy, cx - 2 : cx + 3] = color
                frame[cy + 1, cx - 1 : cx + 2] = color
                frame[cy + 2, cx] = color
            else:
                frame[cy - 2 : cy + 3, cx - 1 : cx + 2] = color
                frame[cy - 1 : cy + 2, cx - 2 : cx + 3] = color

    def _draw_step_bar(self, frame: np.ndarray, game: LeapFrog) -> None:
        x0 = 5
        y0 = 61
        width = 54
        frame[y0:63, x0 : x0 + width] = COLOR_DARK_GRAY
        fill = max(0, min(width, int(width * game.remaining_steps / max(1, game.step_budget))))
        if fill:
            frame[y0:63, x0 : x0 + fill] = COLOR_LIGHT_BLUE

    def _draw_flash(self, frame: np.ndarray, game: LeapFrog) -> None:
        if game.flash <= 0:
            return
        frame[0:2, :] = COLOR_DARK_GRAY
        frame[62:64, :] = COLOR_DARK_GRAY
        frame[:, 0:2] = COLOR_DARK_GRAY
        frame[:, 62:64] = COLOR_DARK_GRAY


def _build_level(spec: LevelSpec) -> Level:
    return Level(
        sprites=[Sprite(np.full((1, 1), -1, dtype=np.int8), name="anchor", collidable=False)],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={"spec": spec},
    )


def _action_id(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    return int(value)


class LeapFrog(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = LeapFrogView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._view]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed,
        )
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        self.floor = set(spec.floor)
        self.platforms = {
            platform.pid: {"footprint": set(platform.footprint), "rail": set(platform.rail)}
            for platform in spec.platforms
        }
        self.pieces = {piece.piece_id: {"kind": piece.kind, "cell": tuple(piece.cell)} for piece in spec.pieces}
        self.selected_piece: str | None = None
        self.legal_landings: set[tuple[int, int]] = set()
        self.step_budget = int(spec.step_budget)
        self.remaining_steps = self.step_budget
        self.flash = 0

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self._refresh_selection()
            self.complete_action()
            return

        self.flash = max(0, self.flash - 1)
        solved_before = self._green_count() == 1
        acted = self._resolve_action()
        if acted and not solved_before:
            self._spend_step()
        if self._green_count() == 1:
            self._clear_selection()
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _resolve_action(self) -> bool:
        action = _action_id(self.action.id)
        if action in MOVE_DELTAS:
            moved = self._shove_platforms(MOVE_DELTAS[action])
            self._clear_selection()
            if not moved:
                self.flash = 1
            return True
        if action == ACTION_SPACE:
            self._clear_selection()
            return True
        if action == ACTION_CLICK:
            x = int(self.action.data.get("x", 0))
            y = int(self.action.data.get("y", 0))
            self._handle_click(x, y)
            return True
        return False

    def _handle_click(self, x: int, y: int) -> None:
        cell = _screen_to_cell(x, y)
        if cell is None:
            self._clear_selection()
            self.flash = 1
            return
        if self.selected_piece is not None and cell in self.legal_landings:
            self._jump_selected(cell)
            return
        piece_id = self._piece_at(cell)
        if piece_id is not None and self.pieces[piece_id]["kind"] in {"G", "R", "B"}:
            self.selected_piece = piece_id
            self._refresh_selection()
            return
        self._clear_selection()
        self.flash = 1

    def _jump_selected(self, landing: tuple[int, int]) -> None:
        mover_id = self.selected_piece
        if mover_id is None or mover_id not in self.pieces:
            self._clear_selection()
            return
        start = self.pieces[mover_id]["cell"]
        dx = (landing[0] - start[0]) // 2
        dy = (landing[1] - start[1]) // 2
        middle = (start[0] + dx, start[1] + dy)
        jumped_id = self._piece_at(middle)
        self.pieces[mover_id]["cell"] = landing
        if jumped_id is not None:
            mover_kind = self.pieces[mover_id]["kind"]
            jumped_kind = self.pieces[jumped_id]["kind"]
            if mover_kind == "G" and jumped_kind in {"G", "B"}:
                del self.pieces[jumped_id]
        self._clear_selection()

    def _shove_platforms(self, delta: tuple[int, int]) -> bool:
        dx, dy = delta
        proposed = {}
        eligible = set()
        for pid, platform in self.platforms.items():
            footprint = platform["footprint"]
            next_footprint = {(x + dx, y + dy) for x, y in footprint}
            proposed[pid] = next_footprint
            if next_footprint <= platform["rail"]:
                eligible.add(pid)

        moving = set(eligible)
        while True:
            blocked = self._blocked_platforms(moving, proposed, delta)
            if not blocked:
                break
            moving -= blocked
            if not moving:
                return False

        for pid in moving:
            self.platforms[pid]["footprint"] = proposed[pid]
        for piece in self.pieces.values():
            for pid in moving:
                old_cells = {(x - dx, y - dy) for x, y in self.platforms[pid]["footprint"]}
                if piece["cell"] in old_cells:
                    px, py = piece["cell"]
                    piece["cell"] = (px + dx, py + dy)
                    break
        return bool(moving)

    def _blocked_platforms(
        self, moving: set[str], proposed: dict[str, set[tuple[int, int]]], delta: tuple[int, int]
    ) -> set[str]:
        blocked: set[str] = set()
        dx, dy = delta
        stationary_footprints = [platform["footprint"] for pid, platform in self.platforms.items() if pid not in moving]
        moving_piece_targets: list[tuple[int, int]] = []

        for pid in moving:
            old = self.platforms[pid]["footprint"]
            new = proposed[pid]
            if any(new & stationary for stationary in stationary_footprints):
                blocked.add(pid)
                continue
            for piece in self.pieces.values():
                cell = piece["cell"]
                if cell in old:
                    moving_piece_targets.append((cell[0] + dx, cell[1] + dy))
                    continue
                if cell in new:
                    blocked.add(pid)
                    break

        final_cells: dict[tuple[int, int], str] = {}
        for pid, platform in self.platforms.items():
            cells = proposed[pid] if pid in moving else platform["footprint"]
            for cell in cells:
                owner = final_cells.get(cell)
                if owner is not None and (owner in moving or pid in moving):
                    blocked.add(pid)
                    blocked.add(owner)
                final_cells[cell] = pid

        seen_riders: set[tuple[int, int]] = set()
        for cell in moving_piece_targets:
            if cell in seen_riders:
                blocked.update(moving)
            seen_riders.add(cell)
        return blocked

    def _green_count(self) -> int:
        return sum(1 for piece in self.pieces.values() if piece["kind"] == "G")

    def _piece_at(self, cell: tuple[int, int]) -> str | None:
        for piece_id, piece in self.pieces.items():
            if piece["cell"] == cell:
                return piece_id
        return None

    def _is_supported(self, cell: tuple[int, int]) -> bool:
        if cell in self.floor:
            return True
        return any(cell in platform["footprint"] for platform in self.platforms.values())

    def _refresh_selection(self) -> None:
        self.legal_landings = set()
        if self.selected_piece is None or self.selected_piece not in self.pieces:
            self.selected_piece = None
            return
        piece = self.pieces[self.selected_piece]
        if piece["kind"] not in {"G", "R", "B"}:
            self.selected_piece = None
            return
        sx, sy = piece["cell"]
        for dx, dy in JUMP_DIRS:
            middle = (sx + dx, sy + dy)
            landing = (sx + 2 * dx, sy + 2 * dy)
            jumped_id = self._piece_at(middle)
            if jumped_id is None:
                continue
            if self._piece_at(landing) is not None:
                continue
            if not self._is_supported(landing):
                continue
            self.legal_landings.add(landing)

    def _clear_selection(self) -> None:
        self.selected_piece = None
        self.legal_landings = set()

    def _spend_step(self) -> None:
        self.remaining_steps -= 1
