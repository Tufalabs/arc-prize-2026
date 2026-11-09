from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

BOARD_WIDTH = 14
BOARD_HEIGHT = 12
CELL_SIZE = 4
BOARD_LEFT = 4
BOARD_TOP = 8
VIEW_SIZE = 64

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

COLOR_OUTSIDE = 0
COLOR_AIR = 1
COLOR_TERRAIN_CAP = 3
COLOR_TERRAIN = 4
COLOR_DARK = 5
COLOR_LINE = 6
COLOR_INVALID = 8
COLOR_TUG = 9
COLOR_TUG_LIGHT = 10
COLOR_FRAME = 11
COLOR_CARGO = 12
COLOR_LOADED = 13
COLOR_COMPLETE = 14

CARGO_PIXELS = np.array(((12, 12, 12, 12), (11, 5, 5, 11), (11, 5, 5, 11), (12, 12, 12, 12)), dtype=np.int8)
TUG_PIXELS = np.array(((-1, 10, 10, -1), (9, 9, 9, 9), (10, 5, 5, 10), (-1, 9, 9, -1)), dtype=np.int8)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    baseline_actions: int


LEVEL_SPECS = (
    LevelSpec(
        "First Tow",
        (
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "...d.TC.......",
            ".#########....",
            "..............",
            "..............",
            "..............",
        ),
        4,
    ),
    LevelSpec(
        "Missing Ledge",
        (
            "..............",
            "..............",
            "..............",
            ".....TC.......",
            ".....#####....",
            "..............",
            "..............",
            "..............",
            "...d..........",
            "..#####.......",
            "..............",
            "..............",
        ),
        10,
    ),
    LevelSpec(
        "Living Landing",
        (
            "..............",
            "..............",
            ".....TC.......",
            "......####....",
            "..............",
            "..............",
            "....dCd.......",
            "...#####......",
            "..............",
            "..............",
            "..............",
            "..............",
        ),
        13,
    ),
    LevelSpec(
        "Borrowed Berth",
        (
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "........C##...",
            ".......TO.....",
            "..#########...",
            "..............",
            "............d.",
            "..........####",
            "..............",
        ),
        32,
    ),
    LevelSpec(
        "Relay Crates",
        (
            "..............",
            "..............",
            "......CT......",
            "...####.......",
            "....C.........",
            "..###....d....",
            ".......C##....",
            "...d..##......",
            "...##..d......",
            "...######.....",
            "..............",
            "..............",
        ),
        45,
    ),
    LevelSpec(
        "Last Door",
        (
            ".......TC.....",
            "........####..",
            "......C.......",
            "...####.......",
            "..##....C.....",
            "..#.d...###...",
            "..#####...d...",
            "....d...###...",
            "....###C..d...",
            ".......####...",
            "..............",
            "..............",
        ),
        91,
    ),
    LevelSpec(
        "Vacated Berth",
        (
            ".......TC.....",
            "........####..",
            "......C.......",
            "...####..C....",
            "......C..####.",
            "...####.......",
            "..............",
            "...d..........",
            "...####.......",
            "..........d...",
            "..d..d.O.##...",
            ".##########...",
        ),
        102,
    ),
)


@dataclass(frozen=True)
class RenderFrame:
    cargo_pixels: tuple[tuple[int, int], ...]
    tug_pixel: tuple[int, int] | None
    attached_id: int | None
    line_color: int = COLOR_LINE
    line_fraction: float = 1.0
    port_flash: bool = False
    wreck_cell: tuple[int, int] | None = None
    completion_pulse: bool = False


class TowlineDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: Towline | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_OUTSIDE
        frame[BOARD_TOP : BOARD_TOP + BOARD_HEIGHT * CELL_SIZE, BOARD_LEFT : BOARD_LEFT + BOARD_WIDTH * CELL_SIZE] = (
            COLOR_AIR
        )
        frame[BOARD_TOP + BOARD_HEIGHT * CELL_SIZE, BOARD_LEFT : BOARD_LEFT + BOARD_WIDTH * CELL_SIZE] = COLOR_DARK
        visual = game.visual_frame or game._settled_frame()
        self._draw_hud(frame, game)
        self._draw_docks(frame, game, visual)
        self._draw_terrain(frame, game)
        self._draw_cargo(frame, game, visual)
        self._draw_occupied_dock_corners(frame, game, visual)
        self._draw_tug(frame, visual)
        self._draw_line(frame, visual)
        self._draw_wreck(frame, visual)
        if visual.completion_pulse:
            frame[7, 4:60] = COLOR_COMPLETE
            frame[57, 4:60] = COLOR_COMPLETE
        return frame

    def _draw_hud(self, frame: np.ndarray, game: Towline) -> None:
        frame[2:4, 4:60] = COLOR_DARK
        width = int(56 * max(0, game.remaining_steps) / game.step_budget)
        if width:
            frame[2:4, 4 : 4 + width] = COLOR_TUG_LIGHT
        if game.remaining_steps <= 0:
            frame[2:4, 4:60] = COLOR_INVALID
        for index in range(len(LEVEL_SPECS)):
            x = 16 + index * 5
            color = COLOR_COMPLETE if index < game.level_index else COLOR_DARK
            if index == game.level_index:
                color = COLOR_TUG_LIGHT
            frame[5:7, x : x + 3] = color

    def _draw_docks(self, frame: np.ndarray, game: Towline, visual: RenderFrame) -> None:
        occupied = {game._pixel_to_cell(pixel) for pixel in visual.cargo_pixels if game._pixel_aligned(pixel)}
        all_complete = occupied == game.docks
        for cell in game.docks:
            x, y = _cell_pixel(cell)
            color = COLOR_COMPLETE if cell in occupied or all_complete else COLOR_FRAME
            frame[y : y + 4, x] = color
            frame[y : y + 4, x + 3] = color
            frame[y + 3, x : x + 4] = color

    def _draw_terrain(self, frame: np.ndarray, game: Towline) -> None:
        for cell_x, cell_y in game.terrain:
            x, y = _cell_pixel((cell_x, cell_y))
            frame[y : y + 4, x : x + 4] = COLOR_TERRAIN
            if (cell_x, cell_y - 1) not in game.terrain:
                frame[y, x : x + 4] = COLOR_TERRAIN_CAP

    def _draw_cargo(self, frame: np.ndarray, game: Towline, visual: RenderFrame) -> None:
        logical_positions = {
            cargo_id: game._pixel_to_cell(pixel)
            for cargo_id, pixel in enumerate(visual.cargo_pixels)
            if game._pixel_aligned(pixel)
        }
        occupied = set(logical_positions.values())
        for cargo_id, (x, y) in enumerate(visual.cargo_pixels):
            pixels = CARGO_PIXELS.copy()
            cell = logical_positions.get(cargo_id)
            if cell is not None and (cell[0], cell[1] - 1) in occupied:
                pixels[1:3, (0, 3)] = COLOR_LOADED
            _blit(frame, pixels, x, y)

    def _draw_occupied_dock_corners(self, frame: np.ndarray, game: Towline, visual: RenderFrame) -> None:
        for pixel in visual.cargo_pixels:
            if not game._pixel_aligned(pixel) or game._pixel_to_cell(pixel) not in game.docks:
                continue
            x, y = pixel
            frame[y, x] = COLOR_COMPLETE
            frame[y, x + 3] = COLOR_COMPLETE
            frame[y + 3, x] = COLOR_COMPLETE
            frame[y + 3, x + 3] = COLOR_COMPLETE

    def _draw_tug(self, frame: np.ndarray, visual: RenderFrame) -> None:
        if visual.tug_pixel is None:
            return
        pixels = TUG_PIXELS.copy()
        if visual.port_flash:
            pixels[2, 0] = COLOR_INVALID
            pixels[2, 3] = COLOR_INVALID
        _blit(frame, pixels, *visual.tug_pixel)

    def _draw_line(self, frame: np.ndarray, visual: RenderFrame) -> None:
        if visual.attached_id is None or visual.tug_pixel is None or visual.line_fraction <= 0:
            return
        cargo_x, _cargo_y = visual.cargo_pixels[visual.attached_id]
        tug_x, tug_y = visual.tug_pixel
        direction = 1 if cargo_x > tug_x else -1
        start_x = tug_x + (3 if direction > 0 else 0)
        end_x = cargo_x + (0 if direction > 0 else 3)
        target = round(start_x + (end_x - start_x) * visual.line_fraction)
        left, right = sorted((start_x, target))
        line_y = tug_y + 2
        if 0 <= line_y < VIEW_SIZE:
            frame[line_y, max(0, left) : min(VIEW_SIZE, right + 1)] = visual.line_color

    def _draw_wreck(self, frame: np.ndarray, visual: RenderFrame) -> None:
        if visual.wreck_cell is None:
            return
        x, y = _cell_pixel(visual.wreck_cell)
        frame[y + 2 : y + 4, x + 1 : x + 3] = COLOR_TUG
        if y + 4 < VIEW_SIZE:
            frame[y + 4, x : x + 4] = COLOR_INVALID


def _blit(frame: np.ndarray, pixels: np.ndarray, x: int, y: int) -> None:
    height, width = pixels.shape
    if x >= VIEW_SIZE or y >= VIEW_SIZE or x + width <= 0 or y + height <= 0:
        return
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(VIEW_SIZE, x + width), min(VIEW_SIZE, y + height)
    source = pixels[y0 - y : y1 - y, x0 - x : x1 - x]
    mask = source >= 0
    frame[y0:y1, x0:x1][mask] = source[mask]


def _cell_pixel(cell: tuple[int, int]) -> tuple[int, int]:
    return BOARD_LEFT + cell[0] * CELL_SIZE, BOARD_TOP + cell[1] * CELL_SIZE


def _action_id(action: GameAction) -> int:
    return int(action.value)


class Towline(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.display = TowlineDisplay()
        levels = [Level([], (64, 64), {"spec": spec}, spec.name) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, 64, 64, COLOR_OUTSIDE, COLOR_OUTSIDE, [self.display])
        super().__init__(
            "towline-0001",
            levels,
            camera,
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_UNDO],
            seed=seed,
        )
        self.display.game = self

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.spec = spec
        self.terrain: frozenset[tuple[int, int]] = frozenset()
        self.docks: frozenset[tuple[int, int]] = frozenset()
        cargo: list[tuple[int, int]] = []
        terrain: set[tuple[int, int]] = set()
        docks: set[tuple[int, int]] = set()
        tug: tuple[int, int] | None = None
        for y, row in enumerate(spec.rows):
            if len(row) != BOARD_WIDTH:
                raise ValueError(f"{spec.name}: row {y} must have {BOARD_WIDTH} cells")
            for x, symbol in enumerate(row):
                if symbol == "#":
                    terrain.add((x, y))
                elif symbol == "d":
                    docks.add((x, y))
                elif symbol == "C":
                    cargo.append((x, y))
                elif symbol == "O":
                    docks.add((x, y))
                    cargo.append((x, y))
                elif symbol == "T":
                    tug = (x, y)
        if tug is None:
            raise ValueError(f"{spec.name}: missing tug")
        self.terrain = frozenset(terrain)
        self.docks = frozenset(docks)
        self.cargo_positions = cargo
        self.tug_position: tuple[int, int] | None = tug
        self.attached_id: int | None = None
        self.mode: Literal["PLAYING", "COMPLETE", "CRUSHED"] = "PLAYING"
        self.wreck_cell: tuple[int, int] | None = None
        self.step_budget = spec.baseline_actions * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[
            tuple[tuple[tuple[int, int], ...], tuple[int, int] | None, int | None, str, tuple[int, int] | None]
        ] = []
        self.animation_frames: list[RenderFrame] = []
        self.visual_frame: RenderFrame | None = None
        self._pending_terminal: Literal["complete", "crushed", "budget"] | None = None

    def _settled_frame(self, **changes: object) -> RenderFrame:
        values: dict[str, object] = {
            "cargo_pixels": tuple(_cell_pixel(cell) for cell in self.cargo_positions),
            "tug_pixel": _cell_pixel(self.tug_position) if self.tug_position is not None else None,
            "attached_id": self.attached_id,
            "wreck_cell": self.wreck_cell,
        }
        values.update(changes)
        return RenderFrame(**values)  # type: ignore[arg-type]

    @staticmethod
    def _pixel_aligned(pixel: tuple[int, int]) -> bool:
        return (pixel[0] - BOARD_LEFT) % CELL_SIZE == 0 and (pixel[1] - BOARD_TOP) % CELL_SIZE == 0

    @staticmethod
    def _pixel_to_cell(pixel: tuple[int, int]) -> tuple[int, int]:
        return (pixel[0] - BOARD_LEFT) // CELL_SIZE, (pixel[1] - BOARD_TOP) // CELL_SIZE

    def _snapshot(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], tuple[int, int] | None, int | None, str, tuple[int, int] | None]:
        return tuple(self.cargo_positions), self.tug_position, self.attached_id, self.mode, self.wreck_cell

    def _restore(
        self,
        snapshot: tuple[tuple[tuple[int, int], ...], tuple[int, int] | None, int | None, str, tuple[int, int] | None],
    ) -> None:
        cargo, self.tug_position, self.attached_id, mode, self.wreck_cell = snapshot
        self.cargo_positions = list(cargo)
        self.mode = mode  # type: ignore[assignment]

    def _inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < BOARD_WIDTH and 0 <= cell[1] < BOARD_HEIGHT

    def _blocked(self, cell: tuple[int, int]) -> bool:
        return not self._inside(cell) or cell in self.terrain or cell in self.cargo_positions

    def _falling_set(self) -> set[int]:
        by_position = {position: cargo_id for cargo_id, position in enumerate(self.cargo_positions)}
        falling: set[int] = set()
        changed = True
        while changed:
            changed = False
            for cargo_id, (x, y) in enumerate(self.cargo_positions):
                if cargo_id in falling or y == BOARD_HEIGHT - 1 or (x, y + 1) in self.terrain:
                    continue
                below = by_position.get((x, y + 1))
                if below is None or below in falling:
                    falling.add(cargo_id)
                    changed = True
        return falling

    def _resolve_gravity(self, frames: list[RenderFrame]) -> None:
        first_tick = True
        while True:
            falling = self._falling_set()
            if not falling:
                return
            if first_tick and self.attached_id in falling:
                breaking_id = self.attached_id
                frames.append(self._settled_frame(line_color=COLOR_INVALID))
                self.attached_id = None
                frames.append(self._settled_frame(attached_id=breaking_id, line_fraction=0.5))
            first_tick = False
            start_pixels = tuple(_cell_pixel(cell) for cell in self.cargo_positions)
            half_pixels = tuple(
                (x, y + 2) if cargo_id in falling else (x, y) for cargo_id, (x, y) in enumerate(start_pixels)
            )
            frames.append(
                RenderFrame(
                    half_pixels,
                    _cell_pixel(self.tug_position) if self.tug_position is not None else None,
                    self.attached_id,
                    wreck_cell=self.wreck_cell,
                )
            )
            next_positions = [
                (x, y + 1) if cargo_id in falling else (x, y) for cargo_id, (x, y) in enumerate(self.cargo_positions)
            ]
            collision = self.tug_position is not None and self.tug_position in {
                next_positions[cargo_id] for cargo_id in falling
            }
            self.cargo_positions = next_positions
            if collision:
                self.wreck_cell = self.tug_position
                self.tug_position = None
                self.attached_id = None
                self.mode = "CRUSHED"
                frames.append(self._settled_frame())
                self._pending_terminal = "crushed"
                return
            frames.append(self._settled_frame())

    def _is_complete(self) -> bool:
        return self.mode == "PLAYING" and set(self.cargo_positions) == self.docks

    def _queue_completion(self, frames: list[RenderFrame]) -> None:
        self.mode = "COMPLETE"
        for index in range(6):
            frames.append(self._settled_frame(completion_pulse=index % 2 == 0))
        self._pending_terminal = "complete"

    def _finish_or_animate(self, frames: list[RenderFrame]) -> None:
        if self.remaining_steps <= 0 and self._pending_terminal != "crushed":
            self.mode = "CRUSHED"
            self._pending_terminal = "budget"
            frames.append(self._settled_frame())
        elif self._pending_terminal is None and self._is_complete():
            self._queue_completion(frames)
        if not frames:
            frames.append(self._settled_frame())
        self.animation_frames = frames
        self._play_animation_frame()

    def _play_animation_frame(self) -> None:
        self.visual_frame = self.animation_frames.pop(0)
        if self.animation_frames:
            return
        terminal = self._pending_terminal
        self._pending_terminal = None
        self.complete_action()
        if terminal == "complete":
            self.next_level()
        elif terminal in {"crushed", "budget"}:
            self.lose()

    def _invalid_move_frames(self, dx: int, dy: int) -> list[RenderFrame]:
        tug_pixel = _cell_pixel(self.tug_position) if self.tug_position is not None else None
        if tug_pixel is None:
            return [self._settled_frame()]
        return [self._settled_frame(tug_pixel=(tug_pixel[0] + dx, tug_pixel[1] + dy)), self._settled_frame()]

    def _invalid_line_frames(self) -> list[RenderFrame]:
        return [self._settled_frame(line_color=COLOR_INVALID, line_fraction=2.0), self._settled_frame()]

    def _resolve_move(self, action: int, frames: list[RenderFrame]) -> bool:
        if self.tug_position is None:
            return False
        dx, dy = MOVE_DELTAS[action]
        if self.attached_id is None:
            destination = (self.tug_position[0] + dx, self.tug_position[1] + dy)
            if self._blocked(destination):
                frames.extend(self._invalid_move_frames(dx, dy))
                return False
            self.tug_position = destination
            frames.append(self._settled_frame())
            return True

        cargo_id = self.attached_id
        cargo = self.cargo_positions[cargo_id]
        away = (self.tug_position[0] - cargo[0], self.tug_position[1] - cargo[1])
        destination = (self.tug_position[0] + away[0], self.tug_position[1] + away[1])
        loaded = (cargo[0], cargo[1] - 1) in self.cargo_positions
        if (dx, dy) != away or self._blocked(destination) or loaded:
            frames.extend(self._invalid_line_frames())
            return False

        old_tug = self.tug_position
        old_pixels = tuple(_cell_pixel(cell) for cell in self.cargo_positions)
        old_tug_pixel = _cell_pixel(old_tug)
        frames.append(RenderFrame(old_pixels, old_tug_pixel, cargo_id))
        middle = list(old_pixels)
        middle[cargo_id] = (old_pixels[cargo_id][0] + away[0] * 2, old_pixels[cargo_id][1])
        middle_tug = (old_tug_pixel[0] + away[0] * 2, old_tug_pixel[1])
        frames.append(RenderFrame(tuple(middle), middle_tug, cargo_id))
        self.cargo_positions[cargo_id] = old_tug
        self.tug_position = destination
        frames.append(self._settled_frame())
        return True

    def _resolve_space(self, frames: list[RenderFrame]) -> bool:
        if self.tug_position is None:
            return False
        if self.attached_id is not None:
            frames.append(self._settled_frame(line_fraction=0.5))
            self.attached_id = None
            frames.append(self._settled_frame(line_fraction=0.0))
            return True
        adjacent = [
            cargo_id
            for cargo_id, (x, y) in enumerate(self.cargo_positions)
            if y == self.tug_position[1] and abs(x - self.tug_position[0]) == 1
        ]
        if len(adjacent) != 1:
            frames.extend([self._settled_frame(port_flash=True), self._settled_frame()])
            return False
        self.attached_id = adjacent[0]
        frames.extend([self._settled_frame(line_fraction=0.5), self._settled_frame()])
        return True

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action == int(GameAction.RESET.value):
            self.visual_frame = self._settled_frame()
            self.complete_action()
            return
        if action == ACTION_CLICK:
            self.visual_frame = self._settled_frame()
            self.complete_action()
            return
        if self.mode != "PLAYING":
            self.complete_action()
            return

        self.remaining_steps -= 1
        frames: list[RenderFrame] = []
        if action == ACTION_UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self._finish_or_animate([self._settled_frame()])
            return

        snapshot = self._snapshot()
        successful = False
        if action in MOVE_DELTAS:
            successful = self._resolve_move(action, frames)
        elif action == ACTION_SPACE:
            successful = self._resolve_space(frames)
        if successful:
            self.undo_history.append(snapshot)
            self._resolve_gravity(frames)
        self._finish_or_animate(frames)
