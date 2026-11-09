from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay
from arcengine.enums import GameAction

GAME_ID = "tumbling_colors-0001"
VIEW_SIZE = 64
CELL_SIZE = 9

UP, DOWN, LEFT, RIGHT, SPACE, CLICK = 1, 2, 3, 4, 5, 6
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

BACKGROUND = 0
FLOOR = 1
FUTURE = 2
DOCK_INACTIVE = 3
WALL = 4
INK = 5
MAGENTA = 6
RED = 8
BLUE = 9
READY = 10
YELLOW = 11
REJECT = 12
GREEN = 14
PURPLE = 15

INITIAL_ORIENTATION = (YELLOW, PURPLE, RED, GREEN, BLUE, MAGENTA)


@dataclass(frozen=True)
class CheckpointSpec:
    cell: tuple[int, int]
    required_top: int


@dataclass(frozen=True)
class GateSpec:
    a: tuple[int, int]
    b: tuple[int, int]
    color: int


@dataclass(frozen=True)
class LevelSpec:
    name: str
    xmin: int
    ymin: int
    rows: tuple[str, ...]
    dock_requirement: tuple[int, int, int]
    gates: tuple[GateSpec, ...] = ()
    baseline_actions: int = 1


LEVEL_SPECS = (
    LevelSpec("First Tumble", 1, 1, ("S.D",), (PURPLE, GREEN, MAGENTA), baseline_actions=2),
    LevelSpec("The Longer Way", 1, 1, ("...", "S.D"), (YELLOW, RED, MAGENTA), baseline_actions=4),
    LevelSpec("Come Back Green", 1, 1, ("...", "S1D"), (MAGENTA, PURPLE, GREEN), baseline_actions=6),
    LevelSpec(
        "Two-Sided Threshold",
        2,
        1,
        ("D.#..", "S...1"),
        (RED, PURPLE, MAGENTA),
        gates=(GateSpec((3, 2), (4, 2), MAGENTA),),
        baseline_actions=15,
    ),
    LevelSpec(
        "Not Yet",
        1,
        1,
        ("1..###", "....S.", "#D.#.2"),
        (GREEN, BLUE, YELLOW),
        gates=(GateSpec((3, 2), (4, 2), RED),),
        baseline_actions=34,
    ),
    LevelSpec(
        "Borrowed Rooms",
        1,
        1,
        ("#####..", "2.#....", "...X#.1", "...####"),
        (YELLOW, RED, MAGENTA),
        gates=(GateSpec((3, 3), (4, 3), MAGENTA), GateSpec((4, 2), (5, 2), GREEN)),
        baseline_actions=46,
    ),
    LevelSpec(
        "Crooked Circuit",
        1,
        1,
        ("#####.3", "2.###..", "..#....", "...X#.1", "...####"),
        (BLUE, YELLOW, GREEN),
        gates=(GateSpec((3, 4), (4, 4), MAGENTA), GateSpec((4, 3), (5, 3), GREEN)),
        baseline_actions=64,
    ),
)

CHECKPOINT_COLORS = {1: GREEN, 2: MAGENTA, 3: YELLOW}
# Level-specific checkpoint colors that differ from the general visual sequence.
CHECKPOINT_OVERRIDES = {3: (MAGENTA,), 4: (PURPLE, MAGENTA), 5: (GREEN, PURPLE), 6: (YELLOW, MAGENTA, YELLOW)}


def _parse_level(spec: LevelSpec, index: int) -> dict[str, Any]:
    floor: set[tuple[int, int]] = set()
    start: tuple[int, int] | None = None
    dock: tuple[int, int] | None = None
    checkpoint_cells: list[tuple[int, tuple[int, int]]] = []
    for row_index, row in enumerate(spec.rows):
        for col_index, symbol in enumerate(row):
            cell = (spec.xmin + col_index, spec.ymin + row_index)
            if symbol != "#":
                floor.add(cell)
            if symbol in ("S", "X"):
                start = cell
            if symbol in ("D", "X"):
                dock = cell
            if symbol.isdigit():
                checkpoint_cells.append((int(symbol), cell))
    if start is None or dock is None:
        raise ValueError(f"{spec.name} must contain a start and dock")
    checkpoint_cells.sort()
    override = CHECKPOINT_OVERRIDES.get(index)
    checkpoints = []
    for order, (_, cell) in enumerate(checkpoint_cells):
        color = override[order] if override is not None else CHECKPOINT_COLORS[order + 1]
        checkpoints.append(CheckpointSpec(cell, color))
    return {
        "spec": spec,
        "floor": frozenset(floor),
        "start": start,
        "dock": dock,
        "checkpoints": tuple(checkpoints),
        "budget": spec.baseline_actions * 6,
    }


def _make_level(spec: LevelSpec, index: int) -> Level:
    return Level(sprites=[], grid_size=(64, 64), data=_parse_level(spec, index), name=spec.name)


class TumblingColorsView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: TumblingColors | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = BACKGROUND
        self._draw_board(frame, game)
        self._draw_dock(frame, game)
        self._draw_checkpoints(frame, game)
        self._draw_gates(frame, game)
        self._draw_cube(frame, game)
        self._draw_contact_feedback(frame, game)
        self._draw_dock_rails(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _cell_top_left(self, game: TumblingColors, cell: tuple[int, int]) -> tuple[int, int]:
        spec = game.level_spec
        width = len(spec.rows[0])
        height = len(spec.rows)
        origin_x = (VIEW_SIZE - CELL_SIZE * width) // 2
        origin_y = (VIEW_SIZE - CELL_SIZE * height) // 2
        return origin_x + CELL_SIZE * (cell[0] - spec.xmin), origin_y + CELL_SIZE * (cell[1] - spec.ymin)

    def _draw_board(self, frame: np.ndarray, game: TumblingColors) -> None:
        spec = game.level_spec
        for row_index, row in enumerate(spec.rows):
            for col_index, _ in enumerate(row):
                cell = (spec.xmin + col_index, spec.ymin + row_index)
                x, y = self._cell_top_left(game, cell)
                frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = FLOOR if cell in game.floor else WALL

    def _dock_frame_color(self, game: TumblingColors) -> int:
        visual = game.visual
        if visual.get("dock_color") is not None:
            return int(visual["dock_color"])
        if game.terminal:
            return GREEN
        return READY if game.checkpoint_index == len(game.checkpoints) else DOCK_INACTIVE

    def _draw_dock(self, frame: np.ndarray, game: TumblingColors) -> None:
        x, y = self._cell_top_left(game, game.dock)
        frame[y : y + 9, x : x + 9] = DOCK_INACTIVE
        color = self._dock_frame_color(game)
        frame[y, x : x + 9] = color
        frame[y + 8, x : x + 9] = color
        frame[y : y + 9, x] = color
        frame[y : y + 9, x + 8] = color
        pulse = int(game.visual.get("dock_pulse", 0))
        if pulse:
            for distance in range(1, pulse + 1):
                x0, x1 = max(0, x - distance), min(63, x + 8 + distance)
                y0, y1 = max(0, y - distance), min(63, y + 8 + distance)
                frame[y0, x0 : x1 + 1] = GREEN
                frame[y1, x0 : x1 + 1] = GREEN
                frame[y0 : y1 + 1, x0] = GREEN
                frame[y0 : y1 + 1, x1] = GREEN

    def _checkpoint_color(self, game: TumblingColors, index: int) -> int:
        overrides = game.visual.get("checkpoint_colors", {})
        if index in overrides:
            return int(overrides[index])
        if index < game.checkpoint_index:
            return GREEN
        if index == game.checkpoint_index:
            return READY
        return FUTURE

    def _draw_checkpoints(self, frame: np.ndarray, game: TumblingColors) -> None:
        mask = (
            "....A....",
            "..AAAAA..",
            ".AAAAAAA.",
            ".AACCCAA.",
            "AAACCCAAA",
            ".AACCCAA.",
            ".AAAAAAA.",
            "..AAAAA..",
            "....A....",
        )
        for index, checkpoint in enumerate(game.checkpoints):
            x, y = self._cell_top_left(game, checkpoint.cell)
            ring = self._checkpoint_color(game, index)
            for row, pattern in enumerate(mask):
                for col, symbol in enumerate(pattern):
                    if symbol == "A":
                        frame[y + row, x + col] = ring
                    elif symbol == "C":
                        frame[y + row, x + col] = checkpoint.required_top

    def _draw_gate(self, frame: np.ndarray, game: TumblingColors, gate: GateSpec, color: int) -> None:
        ax, ay = gate.a
        bx, by = gate.b
        a_px, a_py = self._cell_top_left(game, gate.a)
        if ay == by:
            edge_x = a_px + 8 if ax < bx else a_px - 1
            top = a_py + 1
            frame[top, edge_x : edge_x + 2] = INK
            frame[top + 1 : top + 6, edge_x : edge_x + 2] = color
            frame[top + 6, edge_x : edge_x + 2] = INK
        else:
            edge_y = a_py + 8 if ay < by else a_py - 1
            left = a_px + 1
            frame[edge_y : edge_y + 2, left] = INK
            frame[edge_y : edge_y + 2, left + 1 : left + 6] = color
            frame[edge_y : edge_y + 2, left + 6] = INK

    def _draw_gates(self, frame: np.ndarray, game: TumblingColors) -> None:
        flash_gate = game.visual.get("gate_flash")
        for index, gate in enumerate(game.level_spec.gates):
            color = int(game.visual.get("gate_color", gate.color)) if flash_gate == index else gate.color
            self._draw_gate(frame, game, gate, color)

    def _cube_pixels(self, orientation: tuple[int, ...]) -> np.ndarray:
        up, _, _, south, east, _ = orientation
        patterns = (
            ".........",
            "...kk....",
            "..kTTk...",
            ".kTTTTk..",
            "kTTTTTTk.",
            "kFFFkRRRk",
            "kFFFkRRRk",
            ".kFFkRRk.",
            "..kk.kk..",
        )
        colors = {"k": INK, "T": up, "F": south, "R": east}
        pixels = np.full((9, 9), -1, dtype=np.int8)
        for row, pattern in enumerate(patterns):
            for col, symbol in enumerate(pattern):
                if symbol in colors:
                    pixels[row, col] = colors[symbol]
        return pixels

    def _blit(self, frame: np.ndarray, pixels: np.ndarray, x: int, y: int) -> None:
        height, width = pixels.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(64, x + width), min(64, y + height)
        if x0 >= x1 or y0 >= y1:
            return
        source = pixels[y0 - y : y1 - y, x0 - x : x1 - x]
        target = frame[y0:y1, x0:x1]
        frame[y0:y1, x0:x1] = np.where(source >= 0, source, target)

    def _draw_cube(self, frame: np.ndarray, game: TumblingColors) -> None:
        visual = game.visual
        cell = visual.get("cube_cell", game.cube_position)
        orientation = visual.get("cube_orientation", game.orientation)
        x, y = self._cell_top_left(game, cell)
        offset = visual.get("cube_offset", (0, 0))
        x += int(offset[0])
        y += int(offset[1])
        pixels = self._cube_pixels(orientation)
        lean = visual.get("lean")
        if lean is None:
            self._blit(frame, pixels, x, y)
            return
        # A one-pixel row shear makes the translated cube visibly tip over the edge.
        dx, dy = lean
        if dx:
            sheared = np.full_like(pixels, -1)
            for row in range(9):
                shift = (row // 3) * (1 if dx > 0 else -1)
                for col in range(9):
                    new_col = col + shift
                    if 0 <= new_col < 9:
                        sheared[row, new_col] = pixels[row, col]
            pixels = sheared
        elif dy:
            sheared = np.full_like(pixels, -1)
            for row in range(9):
                shift = (row // 3) * (1 if dy > 0 else -1)
                new_row = row + shift
                if 0 <= new_row < 9:
                    sheared[new_row] = pixels[row]
            pixels = sheared
        self._blit(frame, pixels, x, y)

    def _draw_contact_feedback(self, frame: np.ndarray, game: TumblingColors) -> None:
        edge = game.visual.get("wall_flash")
        if edge is None:
            return
        cell, direction = edge
        x, y = self._cell_top_left(game, cell)
        dx, dy = direction
        if dx > 0:
            frame[y + 1 : y + 8, min(63, x + 8)] = REJECT
        elif dx < 0:
            frame[y + 1 : y + 8, max(0, x)] = REJECT
        elif dy > 0:
            frame[min(63, y + 8), x + 1 : x + 8] = REJECT
        else:
            frame[max(0, y), x + 1 : x + 8] = REJECT

    def _draw_dock_rails(self, frame: np.ndarray, game: TumblingColors) -> None:
        x, y = self._cell_top_left(game, game.dock)
        req_u, req_s, req_e = game.level_spec.dock_requirement
        frame[y, x + 2 : x + 7] = req_u
        frame[y + 8, x : x + 5] = req_s
        frame[y + 4 : y + 9, x + 8] = req_e

    def _draw_budget(self, frame: np.ndarray, game: TumblingColors) -> None:
        frame[61:63, 5:59] = INK
        width = round(54 * game.remaining_steps / max(1, game.step_budget))
        if width > 0:
            color = READY if game.remaining_steps > game.step_budget // 4 else REJECT
            frame[61:63, 5 : 5 + width] = color
        elif game.feedback == "budget_expired":
            frame[0, :] = REJECT
            frame[63, :] = REJECT
            frame[:, 0] = REJECT
            frame[:, 63] = REJECT


class TumblingColors(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.view = TumblingColorsView()
        self.level_spec = LEVEL_SPECS[0]
        self.floor: frozenset[tuple[int, int]] = frozenset()
        self.checkpoints: tuple[CheckpointSpec, ...] = ()
        self.cube_position = (0, 0)
        self.dock = (0, 0)
        self.orientation = INITIAL_ORIENTATION
        self.checkpoint_index = 0
        self.step_count = 0
        self.step_budget = 1
        self.remaining_steps = 1
        self.terminal = False
        self.selected_object = None
        self.cursor_state = None
        self.feedback: str | None = None
        self.visual: dict[str, Any] = {}
        self._frames: list[dict[str, Any]] = []
        self._finish_kind: str | None = None
        levels = [_make_level(spec, index) for index, spec in enumerate(LEVEL_SPECS)]
        camera = Camera(0, 0, 64, 64, BACKGROUND, BACKGROUND, [self.view])
        super().__init__(GAME_ID, levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6], seed=seed)
        self.view.game = self

    def on_set_level(self, level: Level) -> None:
        data = level.get_data("spec")
        # Level.get_data returns individual keys; retain the immutable parsed data.
        self.level_spec = data
        self.floor = level.get_data("floor")
        self.cube_position = level.get_data("start")
        self.dock = level.get_data("dock")
        self.checkpoints = level.get_data("checkpoints")
        self.orientation = INITIAL_ORIENTATION
        self.checkpoint_index = 0
        self.step_count = 0
        self.step_budget = int(level.get_data("budget"))
        self.remaining_steps = self.step_budget
        self.terminal = False
        self.selected_object = None
        self.cursor_state = None
        self.feedback = None
        self.visual = {}
        self._frames = []
        self._finish_kind = None

    def _reset_attempt(self) -> None:
        self.cube_position = self.current_level.get_data("start")
        self.orientation = INITIAL_ORIENTATION
        self.checkpoint_index = 0
        self.step_count = 0
        self.remaining_steps = self.step_budget
        self.terminal = False
        self.feedback = None
        self.visual = {}
        self._frames = []
        self._finish_kind = None

    def _rolled(self, action: int) -> tuple[int, int, int, int, int, int]:
        u, d, n, s, e, w = self.orientation
        if action == RIGHT:
            return w, e, n, s, u, d
        if action == LEFT:
            return e, w, n, s, d, u
        if action == DOWN:
            return n, s, d, u, e, w
        return s, n, u, d, e, w

    def _directional_face(self, action: int) -> int:
        _, _, north, south, east, west = self.orientation
        return {UP: north, DOWN: south, LEFT: west, RIGHT: east}[action]

    def _crossed_gate(self, old: tuple[int, int], new: tuple[int, int]) -> tuple[int, GateSpec] | None:
        edge = frozenset((old, new))
        for index, gate in enumerate(self.level_spec.gates):
            if frozenset((gate.a, gate.b)) == edge:
                return index, gate
        return None

    def _queue_blocked(self, direction: tuple[int, int], gate_index: int | None) -> None:
        self.feedback = "gate_rejection" if gate_index is not None else "wall_bump"
        still = {"cube_cell": self.cube_position, "cube_orientation": self.orientation}
        shifted = dict(still)
        shifted["cube_offset"] = (direction[0] * 2, direction[1] * 2)
        shifted["lean"] = direction
        if gate_index is None:
            shifted["wall_flash"] = (self.cube_position, direction)
        else:
            shifted["gate_flash"] = gate_index
            shifted["gate_color"] = REJECT
        self._frames = [still, shifted, still]
        self._finish_kind = "normal"

    def _post_roll_frames(self, destination: tuple[int, int], new_orientation: tuple[int, ...]) -> list[dict[str, Any]]:
        base = {"cube_cell": destination, "cube_orientation": new_orientation}
        frames: list[dict[str, Any]] = []
        if destination in [checkpoint.cell for checkpoint in self.checkpoints]:
            index = next(i for i, checkpoint in enumerate(self.checkpoints) if checkpoint.cell == destination)
            if index == self.checkpoint_index:
                checkpoint = self.checkpoints[index]
                if new_orientation[0] == checkpoint.required_top:
                    self.checkpoint_index += 1
                    self.feedback = "checkpoint_complete"
                    frames.extend([base, base, base])
                else:
                    self.feedback = "checkpoint_mismatch"
                    flash = dict(base)
                    flash["checkpoint_colors"] = {index: REJECT}
                    frames.extend([flash, flash, base])
            elif index > self.checkpoint_index:
                self.feedback = "future_checkpoint"
                flash = dict(base)
                flash["checkpoint_colors"] = {index: FLOOR, self.checkpoint_index: READY}
                frames.extend([flash, base])

        if destination == self.dock:
            matches = (
                self.checkpoint_index == len(self.checkpoints)
                and (new_orientation[0], new_orientation[3], new_orientation[4]) == self.level_spec.dock_requirement
            )
            if matches:
                self.terminal = True
                self.feedback = "dock_success"
                frames = []
                for pulse in (0, 1, 2, 1):
                    item = dict(base)
                    item["dock_color"] = GREEN
                    item["dock_pulse"] = pulse
                    frames.append(item)
                self._finish_kind = "win"
            else:
                self.feedback = "dock_rejection"
                reject = dict(base)
                reject["dock_color"] = REJECT
                frames.extend([reject, reject, base])
        return frames

    def _queue_roll(self, action: int, destination: tuple[int, int], gate_index: int | None) -> None:
        direction = DELTAS[action]
        old_position = self.cube_position
        old_orientation = self.orientation
        new_orientation = self._rolled(action)
        gate_flash = {"gate_flash": gate_index, "gate_color": READY} if gate_index is not None else {}
        self._frames = [
            {"cube_cell": old_position, "cube_orientation": old_orientation},
            {
                "cube_cell": old_position,
                "cube_orientation": old_orientation,
                "cube_offset": (direction[0] * 3, direction[1] * 3),
                "lean": direction,
                **gate_flash,
            },
            {
                "cube_cell": old_position,
                "cube_orientation": new_orientation,
                "cube_offset": (direction[0] * 6, direction[1] * 6),
                "lean": direction,
                **gate_flash,
            },
        ]
        final = {"cube_cell": destination, "cube_orientation": new_orientation}
        final["_commit_roll"] = (destination, new_orientation)
        self._frames.append(final)

    def _finish_action(self) -> None:
        kind = self._finish_kind
        self._finish_kind = None
        if kind == "win":
            self.next_level()
            # Keep the solving input atomic across ArcEngine's deferred level switch.
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.feedback = "budget_expired"
            self.lose()
        self.complete_action()

    def _play_frame(self) -> None:
        self.visual = self._frames.pop(0)
        commit = self.visual.pop("_commit_roll", None)
        if commit is not None:
            destination, new_orientation = commit
            self.cube_position = destination
            self.orientation = new_orientation
            feedback_frames = self._post_roll_frames(destination, new_orientation)
            if feedback_frames:
                self.visual = feedback_frames.pop(0)
                self._frames = feedback_frames + self._frames
            if self._finish_kind is None:
                self._finish_kind = "normal"
        if not self._frames:
            self._finish_action()

    def step(self) -> None:
        if self._frames:
            self._play_frame()
            return
        self.visual = {}
        self.feedback = None
        action = int(self.action.id.value)
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return
        if action == SPACE:
            self._reset_attempt()
            self.complete_action()
            return
        self.step_count += 1
        self.remaining_steps = max(0, self.step_budget - self.step_count)
        if action == CLICK:
            if self.remaining_steps <= 0:
                self.lose()
            self.complete_action()
            return
        if action not in DELTAS:
            self.complete_action()
            return
        dx, dy = DELTAS[action]
        destination = (self.cube_position[0] + dx, self.cube_position[1] + dy)
        if destination not in self.floor:
            self._queue_blocked((dx, dy), None)
            self._play_frame()
            return
        crossed = self._crossed_gate(self.cube_position, destination)
        if crossed is not None and self._directional_face(action) != crossed[1].color:
            self._queue_blocked((dx, dy), crossed[0])
            self._play_frame()
            return
        self._queue_roll(action, destination, crossed[0] if crossed else None)
        self._play_frame()
