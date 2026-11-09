from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

VIEW_SIZE = 64
CELL_SIZE = 4

OUTER = 4
VOID = 5
FLOOR = 1
FRAME = 2
WHITE = 0
BEAM = 10
PLAYER = 6
PLAYER_TIP = 7
REJECT = 8
LAMP = 11
NOZZLE = 12
UNSAFE = 13
EXIT = 14

UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
SPACE, CLICK = 5, 6
UNDO = 7
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
ORIENTATIONS = ("NE", "SE", "SW", "NW")
REFLECTIONS = {
    ("NE", DOWN): RIGHT,
    ("NE", LEFT): UP,
    ("SE", LEFT): DOWN,
    ("SE", UP): RIGHT,
    ("SW", UP): LEFT,
    ("SW", RIGHT): DOWN,
    ("NW", RIGHT): UP,
    ("NW", DOWN): LEFT,
}

MIRROR_NE = np.array(
    [[FRAME, WHITE, BEAM, BEAM], [FRAME, OUTER, WHITE, BEAM], [FRAME, OUTER, OUTER, WHITE], [FRAME] * 4], dtype=np.int8
)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    player_facing: int
    mirror_orientations: tuple[tuple[str, str], ...]
    baseline_actions: int

    @property
    def budget(self) -> int:
        return self.baseline_actions * 6


LEVEL_SPECS = (
    LevelSpec(
        "First Refraction",
        ("         ", "      P  ", "      .  ", " >  A..  ", "     .   ", "         ", "    E    ", "         "),
        DOWN,
        (("A", "SE"),),
        8,
    ),
    LevelSpec(
        "The Capped Bridge",
        (
            "          ",
            "     ..P  ",
            "   ...    ",
            " >  A.    ",
            "    .     ",
            "          ",
            "          ",
            "     ..E  ",
            "          ",
        ),
        LEFT,
        (("A", "SE"),),
        15,
    ),
    LevelSpec(
        "Before You Cross",
        (
            "  ...P..  ",
            "  . B  C  ",
            "  .       ",
            " >. A.    ",
            "  .. .    ",
            "     ..   ",
            "       E  ",
            "          ",
        ),
        RIGHT,
        (("A", "SW"), ("B", "NE"), ("C", "SE")),
        26,
    ),
    LevelSpec(
        "Overshoot",
        (
            "             ",
            "      P      ",
            " > A...      ",
            "    .        ",
            "             ",
            "             ",
            "      C   E  ",
            " > ..B...    ",
            "      ...    ",
            "             ",
            "             ",
        ),
        DOWN,
        (("A", "SE"), ("B", "SE"), ("C", "NE")),
        26,
    ),
    LevelSpec(
        "Farthest First",
        (
            "              ",
            "              ",
            "              ",
            "     ...      ",
            "   ..B......C ",
            "   .  .       ",
            "   .> .A      ",
            "   ..  P      ",
            "    ..    ..  ",
            "     ......   ",
            "            E ",
        ),
        UP,
        (("A", "SE"), ("B", "NE"), ("C", "SE")),
        59,
    ),
    LevelSpec(
        "Borrowed Footing",
        (
            "              ",
            "              ",
            "     .......  ",
            "   ...C    D  ",
            "   .          ",
            "   .          ",
            "   . >A       ",
            "   . # ...N   ",
            "   . P   .    ",
            "   ..    ..   ",
            "    ...   ^E  ",
            "              ",
        ),
        RIGHT,
        (("A", "SW"), ("N", "SE"), ("C", "NE"), ("D", "SE")),
        64,
    ),
    LevelSpec(
        "Prism Paths",
        (
            "               ",
            "               ",
            "               ",
            "               ",
            "     ... ..... ",
            "    .B....   .C",
            "    .          ",
            "    .          ",
            "    . >A       ",
            "    . # ..N    ",
            "    . P  .     ",
            "    ..   ..... ",
            "     ...  ^   E",
            "               ",
            "               ",
        ),
        RIGHT,
        (("A", "SW"), ("N", "SE"), ("B", "NE"), ("C", "SE")),
        74,
    ),
)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _parse_level(spec: LevelSpec) -> dict[str, object]:
    height = len(spec.rows)
    width = len(spec.rows[0])
    if any(len(row) != width for row in spec.rows):
        raise ValueError(f"Ragged layout in {spec.name}")

    floor: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    lamps: dict[tuple[int, int], int] = {}
    mirrors: dict[str, tuple[int, int]] = {}
    player = exit_cell = None
    lamp_symbols = {"^": UP, "v": DOWN, "<": LEFT, ">": RIGHT}
    for row, line in enumerate(spec.rows):
        for col, symbol in enumerate(line):
            cell = (row, col)
            if symbol == " ":
                continue
            if symbol == "#":
                walls.add(cell)
                continue
            floor.add(cell)
            if symbol in lamp_symbols:
                lamps[cell] = lamp_symbols[symbol]
            elif symbol == "P":
                player = cell
            elif symbol == "E":
                exit_cell = cell
            elif symbol.isalpha() and symbol not in {"P", "E"}:
                mirrors[symbol] = cell
    if player is None or exit_cell is None:
        raise ValueError(f"Missing player or exit in {spec.name}")
    orientations = dict(spec.mirror_orientations)
    if set(mirrors) != set(orientations):
        raise ValueError(f"Mirror metadata mismatch in {spec.name}")
    return {
        "height": height,
        "width": width,
        "floor": floor,
        "walls": walls,
        "lamps": lamps,
        "mirrors": mirrors,
        "orientations": orientations,
        "player": player,
        "exit": exit_cell,
        "facing": spec.player_facing,
        "budget": spec.budget,
    }


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), OUTER, dtype=np.int8), name="board", layer=0, collidable=False, tags=["board"]
    )
    return Level(sprites=[board], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": _parse_level(spec)}, name=spec.name)


class PrismPaths(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "prism_paths",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, OUTER, OUTER),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, UNDO],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        data = level.get_data("spec")
        self.height = int(data["height"])
        self.width = int(data["width"])
        self.floor = set(data["floor"])
        self.walls = set(data["walls"])
        self.lamps = dict(data["lamps"])
        self.mirror_positions = dict(data["mirrors"])
        self.mirror_orientations = dict(data["orientations"])
        self.player_position = tuple(data["player"])
        self.player_facing = int(data["facing"])
        self.exit_position = tuple(data["exit"])
        self.step_budget = int(data["budget"])
        self.remaining_steps = self.step_budget
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.animation_frames: list[np.ndarray] = []
        self.animation_finish: str | None = None
        self.undo_history: list[tuple[tuple[int, int], int, dict[str, tuple[int, int]], dict[str, str]]] = []
        self.beams, self.beam_steps = self._trace_beams(self.mirror_positions, self.mirror_orientations)
        self._show(self._render())

    def _offsets(self) -> tuple[int, int]:
        return (VIEW_SIZE - CELL_SIZE * self.width) // 2, (VIEW_SIZE - CELL_SIZE * self.height) // 2

    def _pixel(self, cell: tuple[int, int]) -> tuple[int, int]:
        x_offset, y_offset = self._offsets()
        return x_offset + CELL_SIZE * cell[1], y_offset + CELL_SIZE * cell[0]

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.height and 0 <= cell[1] < self.width

    def _trace_beams(
        self, positions: dict[str, tuple[int, int]], orientations: dict[str, str]
    ) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[tuple[int, int], int], int]]:
        mirror_at = {cell: mirror_id for mirror_id, cell in positions.items()}
        reached: dict[tuple[int, int], list[int]] = {}
        steps: dict[tuple[tuple[int, int], int], int] = {}
        for lamp_cell, initial_direction in self.lamps.items():
            cell = lamp_cell
            direction = initial_direction
            seen: set[tuple[tuple[int, int], int]] = set()
            distance = 0
            while True:
                dr, dc = DELTAS[direction]
                cell = (cell[0] + dr, cell[1] + dc)
                distance += 1
                state = (cell, direction)
                if not self._in_bounds(cell) or state in seen:
                    break
                seen.add(state)
                reached.setdefault(cell, []).append(direction)
                steps[(cell, len(reached[cell]) - 1)] = distance
                if cell in self.walls or cell in self.lamps:
                    break
                mirror_id = mirror_at.get(cell)
                if mirror_id is not None:
                    outgoing = REFLECTIONS.get((orientations[mirror_id], direction))
                    if outgoing is None:
                        break
                    direction = outgoing
        return reached, steps

    def _draw_beams(
        self,
        frame: np.ndarray,
        beams: dict[tuple[int, int], list[int]],
        beam_steps: dict[tuple[tuple[int, int], int], int],
        *,
        color: int = BEAM,
        max_distance: int | None = None,
    ) -> None:
        for cell, directions in beams.items():
            visible = [
                direction
                for index, direction in enumerate(directions)
                if max_distance is None or beam_steps[(cell, index)] <= max_distance
            ]
            if not visible:
                continue
            x, y = self._pixel(cell)
            if len(visible) >= 2:
                frame[y : y + 4, x : x + 4] = color if color == REJECT else WHITE
                continue
            direction = visible[0]
            if cell not in self.floor:
                frame[y : y + 4, x : x + 4] = color
            if direction in (LEFT, RIGHT):
                frame[y + 1 : y + 3, x : x + 4] = WHITE if color == BEAM else color
            else:
                frame[y : y + 4, x + 1 : x + 3] = WHITE if color == BEAM else color

    def _mirror_pixels(self, orientation: str, *, color: int | None = None, flash: bool = False) -> np.ndarray:
        pixels = np.rot90(MIRROR_NE, -ORIENTATIONS.index(orientation)).copy()
        if color is not None:
            pixels[:, :] = color
            return pixels
        if flash:
            pixels[(pixels == BEAM) | (pixels == WHITE)] = WHITE

        # Persistent color-keyed routing cue: amber marks the canonical input
        # edge and orange marks the reflected output edge. Rotating the prism
        # rotates both markers with it.
        input_edge = orientation[0]
        output_edge = orientation[1]
        if input_edge == "N":
            pixels[0, 1:3] = LAMP
        else:
            pixels[3, 1:3] = LAMP
        if output_edge == "E":
            pixels[1:3, 3] = NOZZLE
        else:
            pixels[1:3, 0] = NOZZLE
        return pixels

    @staticmethod
    def _blit(frame: np.ndarray, pixels: np.ndarray, x: int, y: int) -> None:
        height, width = pixels.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(VIEW_SIZE, x + width), min(VIEW_SIZE, y + height)
        if x0 >= x1 or y0 >= y1:
            return
        frame[y0:y1, x0:x1] = pixels[y0 - y : y1 - y, x0 - x : x1 - x]

    def _draw_lamp(self, frame: np.ndarray, cell: tuple[int, int], direction: int) -> None:
        x, y = self._pixel(cell)
        frame[y + 1 : y + 3, x + 1 : x + 3] = LAMP
        if direction == UP:
            frame[y, x + 1 : x + 3] = NOZZLE
        elif direction == DOWN:
            frame[y + 3, x + 1 : x + 3] = NOZZLE
        elif direction == LEFT:
            frame[y + 1 : y + 3, x] = NOZZLE
        else:
            frame[y + 1 : y + 3, x + 3] = NOZZLE

    def _draw_player(
        self, frame: np.ndarray, position: tuple[int, int], facing: int, pixel: tuple[int, int] | None
    ) -> None:
        x, y = pixel if pixel is not None else self._pixel(position)
        pixels = np.full((4, 4), PLAYER, dtype=np.int8)
        if facing == UP:
            pixels[0, 1:3] = PLAYER_TIP
        elif facing == DOWN:
            pixels[3, 1:3] = PLAYER_TIP
        elif facing == LEFT:
            pixels[1:3, 0] = PLAYER_TIP
        else:
            pixels[1:3, 3] = PLAYER_TIP
        self._blit(frame, pixels, x, y)

    def _draw_rotation_preview(
        self,
        frame: np.ndarray,
        positions: dict[str, tuple[int, int]],
        orientations: dict[str, str],
        player_position: tuple[int, int],
    ) -> None:
        dr, dc = DELTAS[self.player_facing]
        adjacent = (player_position[0] + dr, player_position[1] + dc)
        mirror_id = next((mid for mid, cell in positions.items() if cell == adjacent), None)
        if mirror_id is None:
            return

        candidate_orientations = dict(orientations)
        old_index = ORIENTATIONS.index(candidate_orientations[mirror_id])
        candidate_orientations[mirror_id] = ORIENTATIONS[(old_index + 1) % 4]
        candidate_beams, _ = self._trace_beams(positions, candidate_orientations)
        preview_color = NOZZLE if self._is_safe(player_position, candidate_beams) else REJECT
        for cell in set(candidate_beams) - set(self.beams):
            if not self._in_bounds(cell):
                continue
            x, y = self._pixel(cell)
            frame[y, x] = frame[y + 3, x + 3] = preview_color

        destination = (adjacent[0] + dr, adjacent[1] + dc)
        if self._in_bounds(destination):
            x, y = self._pixel(destination)
            frame[y, x : x + 4] = preview_color
            frame[y + 3, x : x + 4] = preview_color
            frame[y : y + 4, x] = preview_color
            frame[y : y + 4, x + 3] = preview_color

    def _render(
        self,
        *,
        positions: dict[str, tuple[int, int]] | None = None,
        orientations: dict[str, str] | None = None,
        player_position: tuple[int, int] | None = None,
        beams: dict[tuple[int, int], list[int]] | None = None,
        beam_steps: dict[tuple[tuple[int, int], int], int] | None = None,
        beam_color: int = BEAM,
        max_distance: int | None = None,
        mirror_pixel: dict[str, tuple[int, int]] | None = None,
        player_pixel: tuple[int, int] | None = None,
        reject_mirror: str | None = None,
        flash_mirror: str | None = None,
        blocked_cell: tuple[int, int] | None = None,
        unsafe_afterimage: bool = False,
        exit_flash: bool = False,
    ) -> np.ndarray:
        positions = self.mirror_positions if positions is None else positions
        orientations = self.mirror_orientations if orientations is None else orientations
        player_position = self.player_position if player_position is None else player_position
        beams = self.beams if beams is None else beams
        beam_steps = self.beam_steps if beam_steps is None else beam_steps
        frame = np.full((VIEW_SIZE, VIEW_SIZE), OUTER, dtype=np.int8)
        x_offset, y_offset = self._offsets()
        frame[y_offset : y_offset + 4 * self.height, x_offset : x_offset + 4 * self.width] = VOID
        for cell in self.floor:
            x, y = self._pixel(cell)
            frame[y : y + 4, x : x + 4] = FLOOR
        self._draw_beams(frame, beams, beam_steps, color=beam_color, max_distance=max_distance)
        x, y = self._pixel(self.exit_position)
        frame[y : y + 4, x : x + 4] = WHITE if exit_flash else EXIT
        frame[y + 1 : y + 3, x + 1 : x + 3] = EXIT if exit_flash else WHITE
        for cell in self.walls:
            x, y = self._pixel(cell)
            frame[y : y + 4, x : x + 4] = OUTER
        for cell, direction in self.lamps.items():
            self._draw_lamp(frame, cell, direction)
        mirror_pixel = mirror_pixel or {}
        for mirror_id, cell in positions.items():
            mx, my = mirror_pixel.get(mirror_id, self._pixel(cell))
            pixels = self._mirror_pixels(
                orientations[mirror_id],
                color=REJECT if mirror_id == reject_mirror else None,
                flash=mirror_id == flash_mirror,
            )
            self._blit(frame, pixels, mx, my)
        self._draw_player(frame, player_position, self.player_facing, player_pixel)
        if not mirror_pixel and player_pixel is None and reject_mirror is None and blocked_cell is None:
            self._draw_rotation_preview(frame, positions, orientations, player_position)
        if blocked_cell is not None and self._in_bounds(blocked_cell):
            x, y = self._pixel(blocked_cell)
            frame[y : y + 4, x : x + 4] = REJECT
        if unsafe_afterimage:
            x, y = self._pixel(self.player_position)
            frame[y, x] = frame[y, x + 3] = frame[y + 3, x] = frame[y + 3, x + 3] = UNSAFE
        filled = int(60 * max(0, self.remaining_steps) / max(1, self.step_budget))
        frame[63, 2:62] = OUTER
        if filled:
            frame[63, 2 : 2 + filled] = NOZZLE
        return frame

    def _show(self, frame: np.ndarray) -> None:
        self.board.pixels = frame.astype(np.int8, copy=False)

    def _queue(self, frames: list[np.ndarray], finish: str = "complete") -> None:
        self.animation_frames = frames
        self.animation_finish = finish

    def _finish_kind(self) -> str:
        if self.player_position == self.exit_position:
            return "win"
        if self.remaining_steps <= 0:
            return "lose"
        return "complete"

    def _completion_frames(self) -> list[np.ndarray]:
        return [self._render(exit_flash=index % 2 == 0) for index in range(6)] + [self._render()]

    def _spend_action(self) -> None:
        self.remaining_steps -= 1

    def _capture_state(self) -> tuple[tuple[int, int], int, dict[str, tuple[int, int]], dict[str, str]]:
        return (self.player_position, self.player_facing, dict(self.mirror_positions), dict(self.mirror_orientations))

    @staticmethod
    def _state_key(
        state: tuple[tuple[int, int], int, dict[str, tuple[int, int]], dict[str, str]],
    ) -> tuple[object, ...]:
        player, facing, positions, orientations = state
        return player, facing, tuple(sorted(positions.items())), tuple(sorted(orientations.items()))

    def _undo(self) -> None:
        if self.undo_history:
            player, facing, positions, orientations = self.undo_history.pop()
            self.player_position = player
            self.player_facing = facing
            self.mirror_positions = dict(positions)
            self.mirror_orientations = dict(orientations)
            self.beams, self.beam_steps = self._trace_beams(self.mirror_positions, self.mirror_orientations)
        self._queue([self._render()], self._finish_kind())

    def _is_safe(self, position: tuple[int, int], candidate_beams: dict[tuple[int, int], list[int]]) -> bool:
        return position in self.floor or bool(candidate_beams.get(position))

    def _ordinary_traversable(self, cell: tuple[int, int]) -> bool:
        if (
            not self._in_bounds(cell)
            or cell in self.walls
            or cell in self.lamps
            or cell in self.mirror_positions.values()
        ):
            return False
        return cell in self.floor or bool(self.beams.get(cell))

    def _resolve_direction(self, action: int) -> None:
        self.player_facing = action
        dr, dc = DELTAS[action]
        adjacent = (self.player_position[0] + dr, self.player_position[1] + dc)
        mirror_at = {cell: mirror_id for mirror_id, cell in self.mirror_positions.items()}
        mirror_id = mirror_at.get(adjacent)
        if mirror_id is None:
            if self._ordinary_traversable(adjacent):
                self.player_position = adjacent
            if self.player_position == self.exit_position:
                self._queue(self._completion_frames(), "win")
            else:
                self._queue([self._render()], self._finish_kind())
            return

        destination = (adjacent[0] + dr, adjacent[1] + dc)
        occupied = (
            destination in self.mirror_positions.values() or destination in self.walls or destination in self.lamps
        )
        geometric = self._in_bounds(destination) and destination in self.floor and not occupied
        if not geometric:
            frames = [self._render(blocked_cell=destination) for _ in range(2)] + [self._render()]
            self._queue(frames, self._finish_kind())
            return

        candidate_positions = dict(self.mirror_positions)
        candidate_positions[mirror_id] = destination
        candidate_player = adjacent
        candidate_beams, candidate_steps = self._trace_beams(candidate_positions, self.mirror_orientations)
        if not self._is_safe(candidate_player, candidate_beams):
            frame = self._render(
                positions=candidate_positions,
                player_position=candidate_player,
                beams=candidate_beams,
                beam_steps=candidate_steps,
                beam_color=REJECT,
                reject_mirror=mirror_id,
                unsafe_afterimage=True,
            )
            self._queue([frame.copy() for _ in range(3)] + [self._render()], self._finish_kind())
            return

        old_positions = dict(self.mirror_positions)
        old_player = self.player_position
        old_x, old_y = self._pixel(adjacent)
        player_x, player_y = self._pixel(old_player)
        tween_frames = []
        for amount in (1, 2, 3):
            tween_frames.append(
                self._render(
                    positions=old_positions,
                    player_position=old_player,
                    mirror_pixel={mirror_id: (old_x + dc * amount, old_y + dr * amount)},
                    player_pixel=(player_x + dc * amount, player_y + dr * amount),
                )
            )
        self.mirror_positions = candidate_positions
        self.player_position = candidate_player
        self.beams, self.beam_steps = candidate_beams, candidate_steps
        reveal = [
            self._render(max_distance=distance)
            for distance in range(4, max(4, max(candidate_steps.values(), default=0)) + 4, 4)
        ]
        frames = [*tween_frames, self._render(max_distance=0), *reveal]
        if self.player_position == self.exit_position:
            frames.extend(self._completion_frames())
            self._queue(frames, "win")
        else:
            self._queue(frames, self._finish_kind())

    def _resolve_space(self) -> None:
        dr, dc = DELTAS[self.player_facing]
        adjacent = (self.player_position[0] + dr, self.player_position[1] + dc)
        mirror_id = next((mid for mid, cell in self.mirror_positions.items() if cell == adjacent), None)
        if mirror_id is None:
            self._queue([self._render()], self._finish_kind())
            return
        old_orientations = dict(self.mirror_orientations)
        candidate_orientations = dict(old_orientations)
        old_index = ORIENTATIONS.index(candidate_orientations[mirror_id])
        candidate_orientations[mirror_id] = ORIENTATIONS[(old_index + 1) % 4]
        candidate_beams, candidate_steps = self._trace_beams(self.mirror_positions, candidate_orientations)
        if not self._is_safe(self.player_position, candidate_beams):
            frame = self._render(
                orientations=candidate_orientations,
                beams=candidate_beams,
                beam_steps=candidate_steps,
                beam_color=REJECT,
                reject_mirror=mirror_id,
                unsafe_afterimage=True,
            )
            self._queue([frame.copy() for _ in range(3)] + [self._render()], self._finish_kind())
            return
        flash = self._render(orientations=old_orientations, flash_mirror=mirror_id)
        self.mirror_orientations = candidate_orientations
        self.beams, self.beam_steps = candidate_beams, candidate_steps
        reveal = [
            self._render(max_distance=distance)
            for distance in range(4, max(4, max(candidate_steps.values(), default=0)) + 4, 4)
        ]
        self._queue([flash, self._render(max_distance=0), *reveal], self._finish_kind())

    def _play_queued_frame(self) -> None:
        self._show(self.animation_frames.pop(0))
        if self.animation_frames:
            return
        finish = self.animation_finish
        self.animation_finish = None
        if finish == "win":
            self.next_level()
            self.complete_action()
        elif finish == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def step(self) -> None:
        if self.animation_frames:
            self._play_queued_frame()
            return

        action = _action_id(self.action.id)
        if action == int(GameAction.RESET.value):
            self._show(self._render())
            self.complete_action()
            return
        self._spend_action()
        if action == UNDO:
            self._undo()
        elif action in DELTAS:
            before = self._capture_state()
            self._resolve_direction(action)
            if self._state_key(self._capture_state()) != self._state_key(before):
                self.undo_history.append(before)
        elif action == SPACE:
            before = self._capture_state()
            self._resolve_space()
            if self._state_key(self._capture_state()) != self._state_key(before):
                self.undo_history.append(before)
        else:
            self._queue([self._render()], self._finish_kind())
        self._play_queued_frame()
