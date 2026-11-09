from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

VIEW_SIZE = 64
BOARD_WIDTH = 20
BOARD_HEIGHT = 17
CELL_SIZE = 3
BOARD_LEFT = 2
BOARD_TOP = 10

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_USED = 3
COLOR_WALL = 4
COLOR_RECESS = 5
COLOR_RED = 8
COLOR_VOID = 9
COLOR_GOAL = 14
BEAM_COLORS = (11, 6, 15, 10)


@dataclass(frozen=True)
class BeamSpec:
    name: str
    orientation: str
    fixed_axis: int
    length: int
    start: int
    sleeve: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    beams: tuple[BeamSpec, ...]
    baseline_actions: int

    @property
    def max_steps(self) -> int:
        return self.baseline_actions * 6


LEVEL_SPECS = (
    LevelSpec(
        "Borrowed Cell",
        (
            "____________________",
            "____________________",
            "____________________",
            "______##____________",
            "______##____________",
            "______##____________",
            "______##____________",
            "__....##__.GG.______",
            "__.P._____.GG.______",
            "__..._##__....______",
            "______##____________",
            "______##____________",
            "______##____________",
            "______##____________",
            "____________________",
            "____________________",
            "____________________",
        ),
        (BeamSpec("A", "H", 8, 5, 4, (6, 7)),),
        10,
    ),
    LevelSpec(
        "Don't Take the Third",
        (
            "____________________",
            "____________________",
            "____________________",
            "_______##___________",
            "_______##___________",
            "_______##___________",
            "_______##___________",
            "_GG____##.....______",
            "_GG__.___.....______",
            "_..__P.##.....______",
            "_____..##.....______",
            "_____..##___________",
            "_______##___________",
            "_______##___________",
            "____________________",
            "____________________",
            "____________________",
        ),
        (BeamSpec("A", "H", 8, 7, 5, (7, 8)),),
        20,
    ),
    LevelSpec(
        "Borrowed Footing",
        (
            "____________________",
            "____________________",
            "____________________",
            "____##____##________",
            "____##____##________",
            "____##____##________",
            "GG__##____##________",
            "GG__##____##....____",
            ".._.__..____....____",
            "___P##....##....____",
            "___.##....##....____",
            "___.##....##....____",
            "____##____##________",
            "____##____##________",
            "____________________",
            "____________________",
            "____________________",
        ),
        (BeamSpec("A", "H", 8, 5, 3, (4, 5)), BeamSpec("B", "H", 8, 5, 9, (10, 11))),
        32,
    ),
    LevelSpec(
        "The Extra Cell",
        (
            "____________________",
            "________##_____##___",
            "________##_____##___",
            "________##_____##___",
            "GG______##.....##...",
            "GG_P..____.....##...",
            "..___..___.....__...",
            "________##.....##...",
            "________##.....##...",
            "________##_____##___",
            "________##_____##___",
            "________##_____##___",
            "________##_____##___",
            "________##_____##___",
            "____________________",
            "____________________",
            "____________________",
        ),
        (BeamSpec("A", "H", 5, 8, 4, (8, 9)), BeamSpec("B", "H", 6, 9, 10, (15, 16))),
        42,
    ),
    LevelSpec(
        "Corner Relay",
        (
            "____________________",
            "_______________##___",
            "_____##________##...",
            "_____##_____...##...",
            "_____##___#_...__...",
            "_____##____....##...",
            "_____##_____...##...",
            "_____######_#####___",
            "_____######_#####___",
            "_____##________##___",
            "_____##_____________",
            "GG.__##__.._..______",
            "GG._.____.._..______",
            "..._P##__.._..______",
            "..._.##__.....______",
            "____.##__.....______",
            "____________________",
        ),
        (
            BeamSpec("A", "H", 12, 7, 4, (5, 6)),
            BeamSpec("B", "V", 11, 7, 5, (7, 8)),
            BeamSpec("C", "H", 4, 7, 12, (15, 16)),
        ),
        56,
    ),
    LevelSpec(
        "Restore Past the Start",
        (
            "____________________",
            "_______________##___",
            "_______________##...",
            "__________.....##...",
            "________#_.....__...",
            "_________......##...",
            "__________.....##...",
            "_____####_####_##___",
            "_____####_####_##___",
            "_______##______##___",
            "_______##_....._____",
            "_______##_....._____",
            "GG_____##_....._____",
            "GG_P..____....._____",
            "..___..##_....._____",
            "_______##_....._____",
            "_______##___________",
        ),
        (
            BeamSpec("A", "H", 13, 8, 4, (7, 8)),
            BeamSpec("B", "V", 9, 7, 5, (7, 8)),
            BeamSpec("C", "H", 4, 9, 10, (15, 16)),
        ),
        68,
    ),
    LevelSpec(
        "Two Preparations",
        (
            "____________________",
            "_____________##_____",
            "_____________##.....",
            "__________...##.....",
            "________#_...__.....",
            "_________....##.....",
            "__________...##.....",
            "_____####_#####_____",
            "_____####_#####_____",
            "_______##____####___",
            "_______##_.....##___",
            "_______##_.....##___",
            "GG_____##_.....##...",
            "GG_P..____....___...",
            "..___..##_......#...",
            "_______##_.....##...",
            "_______##______##___",
        ),
        (
            BeamSpec("A", "H", 13, 8, 4, (7, 8)),
            BeamSpec("B", "V", 9, 7, 5, (7, 8)),
            BeamSpec("C", "H", 4, 7, 10, (13, 14)),
            BeamSpec("D", "H", 13, 4, 15, (15, 16)),
        ),
        78,
    ),
)


class CausewayDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: TelescopicCauseway | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        x0, x1 = BOARD_LEFT, BOARD_LEFT + BOARD_WIDTH * CELL_SIZE
        frame[3:6, x0:x1] = COLOR_FLOOR
        used = min(BOARD_WIDTH * CELL_SIZE, (BOARD_WIDTH * CELL_SIZE * game.step_count) // game.max_steps)
        if used:
            frame[3:6, x0 : x0 + used] = COLOR_USED
        return frame


def _cell_rect(frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
    x, y = cell
    px = BOARD_LEFT + x * CELL_SIZE
    py = BOARD_TOP + y * CELL_SIZE
    frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = color


def _parse_rows(
    rows: tuple[str, ...],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    if len(rows) != BOARD_HEIGHT or any(len(row) != BOARD_WIDTH for row in rows):
        raise ValueError("Causeway layouts must be exactly 20x17.")
    floor: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    player: tuple[int, int] | None = None
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            if symbol in ".GP":
                floor.add((x, y))
            if symbol == "#":
                walls.add((x, y))
            elif symbol == "G":
                goals.add((x, y))
            elif symbol == "P":
                player = (x, y)
            elif symbol not in "_.":
                raise ValueError(f"Unknown causeway layout symbol: {symbol!r}")
    if player is None or not goals:
        raise ValueError("Each causeway level needs a player and goal.")
    return floor, walls, goals, player


def _build_level(spec: LevelSpec) -> Level:
    floor, walls, goals, player = _parse_rows(spec.rows)
    canvas = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_VOID, dtype=np.int8),
        name="causeway_canvas",
        layer=0,
        collidable=False,
        tags=["canvas"],
    )
    return Level(
        sprites=[canvas],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={
            "floor": floor,
            "walls": walls,
            "goals": goals,
            "player": player,
            "beams": spec.beams,
            "max_steps": spec.max_steps,
        },
    )


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


class TelescopicCauseway(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = CausewayDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "telescopic_causeway-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_VOID, COLOR_VOID, [self._display]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed,
        )
        self._display.game = self
        self._render_state()

    def on_set_level(self, _level: Level) -> None:
        self.canvas = self.current_level.get_sprites_by_tag("canvas")[0]
        self.floor_cells = set(self.current_level.get_data("floor"))
        self.wall_cells = set(self.current_level.get_data("walls"))
        self.goal_cells = set(self.current_level.get_data("goals"))
        self.player = tuple(self.current_level.get_data("player"))
        self.beam_specs = tuple(self.current_level.get_data("beams"))
        self.beam_starts = [beam.start for beam in self.beam_specs]
        self.max_steps = int(self.current_level.get_data("max_steps"))
        self.step_count = 0
        self._visual_mode: tuple[object, ...] | None = None
        self._animation_phase = 0
        self._animation_length = 0
        self._pending_win = False
        self._render_state()

    def _beam_cells(self, index: int, start: int | None = None) -> set[tuple[int, int]]:
        beam = self.beam_specs[index]
        active_start = self.beam_starts[index] if start is None else start
        if beam.orientation == "H":
            return {(active_start + offset, beam.fixed_axis) for offset in range(beam.length)}
        return {(beam.fixed_axis, active_start + offset) for offset in range(beam.length)}

    def _all_beam_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for index in range(len(self.beam_specs)):
            cells.update(self._beam_cells(index))
        return cells

    def _supported(self, cell: tuple[int, int]) -> bool:
        return cell in self.floor_cells or cell in self._all_beam_cells()

    def _push_candidate(self, action: int) -> tuple[int, int, tuple[int, int], tuple[int, int]] | None:
        for index, beam in enumerate(self.beam_specs):
            start = self.beam_starts[index]
            if beam.orientation == "H":
                if action == ACTION_RIGHT and self.player == (start - 1, beam.fixed_axis):
                    return index, start + 1, (start, beam.fixed_axis), (start + beam.length, beam.fixed_axis)
                if action == ACTION_LEFT and self.player == (start + beam.length, beam.fixed_axis):
                    return (index, start - 1, (start + beam.length - 1, beam.fixed_axis), (start - 1, beam.fixed_axis))
            elif action == ACTION_DOWN and self.player == (beam.fixed_axis, start - 1):
                return index, start + 1, (beam.fixed_axis, start), (beam.fixed_axis, start + beam.length)
            elif action == ACTION_UP and self.player == (beam.fixed_axis, start + beam.length):
                return (index, start - 1, (beam.fixed_axis, start + beam.length - 1), (beam.fixed_axis, start - 1))
        return None

    def _push_is_valid(self, index: int, proposed_start: int, player_destination: tuple[int, int]) -> bool:
        beam = self.beam_specs[index]
        proposed = self._beam_cells(index, proposed_start)
        if any(not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT) for x, y in proposed):
            return False
        if proposed & self.wall_cells:
            return False
        for other_index in range(len(self.beam_specs)):
            if other_index != index and proposed & self._beam_cells(other_index):
                return False
        covered_axis = {x if beam.orientation == "H" else y for x, y in proposed}
        if not all(value in covered_axis for value in range(beam.sleeve[0], beam.sleeve[1] + 1)):
            return False
        return player_destination in self.floor_cells

    def _start_feedback(self, mode: tuple[object, ...], frames: int) -> None:
        self._visual_mode = mode
        self._animation_phase = 0
        self._animation_length = frames
        self._render_state()

    def _advance_feedback(self) -> None:
        self._animation_phase += 1
        self._render_state()
        if self._animation_phase < self._animation_length - 1:
            return
        mode = self._visual_mode[0] if self._visual_mode else None
        self._visual_mode = None
        if mode == "win":
            self.next_level()
        self.complete_action()

    def _resolve_direction(self, action: int) -> None:
        push = self._push_candidate(action)
        if push is not None:
            index, proposed_start, retract_cell, extension_cell = push
            old_start = self.beam_starts[index]
            old_player = self.player
            if self._push_is_valid(index, proposed_start, retract_cell):
                self.beam_starts[index] = proposed_start
                self.player = retract_cell
                self._start_feedback(("push", index, old_start, proposed_start, old_player, retract_cell), 4)
            else:
                self._start_feedback(("invalid", retract_cell, extension_cell), 4)
            return

        dx, dy = MOVE_DELTAS[action]
        target = (self.player[0] + dx, self.player[1] + dy)
        if (
            0 <= target[0] < BOARD_WIDTH
            and 0 <= target[1] < BOARD_HEIGHT
            and target not in self.wall_cells
            and self._supported(target)
        ):
            self.player = target
            self._render_state()
        else:
            self._start_feedback(("blocked",), 2)

    def _check_terminal(self) -> bool:
        if self.player in self.goal_cells:
            self._pending_win = True
            self._start_feedback(("win",), 6)
            return True
        if self.step_count >= self.max_steps:
            self._visual_mode = None
            self._render_state()
            self.lose()
            self.complete_action()
            return True
        return False

    def _draw_beam(self, frame: np.ndarray, index: int, start: int, pixel_offset: int = 0) -> None:
        beam = self.beam_specs[index]
        color = BEAM_COLORS[index]
        if beam.orientation == "H":
            px = BOARD_LEFT + start * CELL_SIZE + pixel_offset
            py = BOARD_TOP + beam.fixed_axis * CELL_SIZE
            frame[py : py + CELL_SIZE, max(0, px) : min(VIEW_SIZE, px + beam.length * CELL_SIZE)] = color
            centers = ((px + 1, py + 1), (px + (beam.length - 1) * CELL_SIZE + 1, py + 1))
        else:
            px = BOARD_LEFT + beam.fixed_axis * CELL_SIZE
            py = BOARD_TOP + start * CELL_SIZE + pixel_offset
            frame[max(0, py) : min(VIEW_SIZE, py + beam.length * CELL_SIZE), px : px + CELL_SIZE] = color
            centers = ((px + 1, py + 1), (px + 1, py + (beam.length - 1) * CELL_SIZE + 1))
        for cx, cy in centers:
            if 0 <= cx < VIEW_SIZE and 0 <= cy < VIEW_SIZE:
                frame[cy, cx] = COLOR_WHITE

    def _render_state(self) -> None:
        if not hasattr(self, "canvas"):
            return
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_VOID, dtype=np.int8)
        for cell in self.floor_cells:
            _cell_rect(frame, cell, COLOR_FLOOR)
        for beam in self.beam_specs:
            if beam.orientation == "H":
                sleeve_cells = {(value, beam.fixed_axis) for value in range(beam.sleeve[0], beam.sleeve[1] + 1)}
            else:
                sleeve_cells = {(beam.fixed_axis, value) for value in range(beam.sleeve[0], beam.sleeve[1] + 1)}
            for cell in sleeve_cells - self.floor_cells:
                _cell_rect(frame, cell, COLOR_RECESS)
        goal_color = COLOR_GOAL
        if self._visual_mode and self._visual_mode[0] == "win" and self._animation_phase % 2 == 1:
            goal_color = COLOR_WHITE
        for cell in self.goal_cells:
            _cell_rect(frame, cell, goal_color)
            x, y = cell
            frame[BOARD_TOP + y * CELL_SIZE + 1, BOARD_LEFT + x * CELL_SIZE + 1] = COLOR_WHITE
        for cell in self.wall_cells:
            _cell_rect(frame, cell, COLOR_WALL)

        push_mode = self._visual_mode and self._visual_mode[0] == "push"
        animated_index = int(self._visual_mode[1]) if push_mode else -1
        for index, start in enumerate(self.beam_starts):
            if index == animated_index:
                old_start = int(self._visual_mode[2])
                new_start = int(self._visual_mode[3])
                direction = 1 if new_start > old_start else -1
                self._draw_beam(frame, index, old_start, direction * self._animation_phase)
            else:
                self._draw_beam(frame, index, start)

        player = self.player
        player_color = COLOR_WHITE
        if push_mode and self._animation_phase < 3:
            player = tuple(self._visual_mode[4])
        if self._visual_mode and self._visual_mode[0] == "blocked":
            player_color = COLOR_USED
        px = BOARD_LEFT + player[0] * CELL_SIZE
        py = BOARD_TOP + player[1] * CELL_SIZE
        frame[py + 1, px : px + CELL_SIZE] = player_color
        frame[py : py + CELL_SIZE, px + 1] = player_color
        frame[py + 1, px + 1] = COLOR_RED

        if self._visual_mode and self._visual_mode[0] == "invalid":
            for cell in (self._visual_mode[1], self._visual_mode[2]):
                x, y = cell
                if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT:
                    _cell_rect(frame, cell, COLOR_RED)
        self.canvas.pixels = frame

    def step(self) -> None:
        if self._visual_mode is not None:
            self._advance_feedback()
            return
        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._render_state()
            self.complete_action()
            return
        self.step_count += 1
        if action in MOVE_DELTAS:
            self._resolve_direction(action)
        else:
            self._render_state()
        if self._check_terminal():
            return
        if self._visual_mode is None:
            self.complete_action()
