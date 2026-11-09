from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

VIEW_SIZE = 64
PLANK_LENGTH = 5

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

NORTH = (-1, 0)
SOUTH = (1, 0)
WEST = (0, -1)
EAST = (0, 1)
DIRECTIONS = {ACTION_UP: NORTH, ACTION_DOWN: SOUTH, ACTION_LEFT: WEST, ACTION_RIGHT: EAST}

COLOR_GRIP = 0
COLOR_GROUND = 1
COLOR_WALL = 4
COLOR_OUTSIDE = 5
COLOR_PLAYER = 6
COLOR_FACING = 7
COLOR_ERROR = 8
COLOR_VOID = 9
COLOR_GHOST = 10
COLOR_CAP = 11
COLOR_PLANK = 12
COLOR_GOAL = 14
COLOR_COMPLETE = 15

BASELINES = (10, 19, 30, 36, 52, 59, 71)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    player: tuple[int, int]
    facing: tuple[int, int]
    carried: bool
    plank_cells: tuple[tuple[int, int], ...] = ()


LEVEL_SPECS = (
    LevelSpec(
        "Lift, Turn, Span",
        (
            "~~~~~~~~~~~~",
            "~~~~~~~~~~~~",
            "~~~~~~~~~~~~",
            "~~~.~~~~~~~~",
            "~~~.~~~~~~~~",
            "~~~.~~~~~~~~",
            "~~~.~~~~~~~~",
            "~....~~~.G.~",
            "~~~.~~~~~~~~",
            "~~~~~~~~~~~~",
            "~~~~~~~~~~~~",
        ),
        (8, 3),
        NORTH,
        False,
        ((3, 3), (4, 3), (5, 3), (6, 3), (7, 3)),
    ),
    LevelSpec(
        "The Stranded Cap",
        (
            "~~~~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~~~~",
            "....~~~.#~~~~~~~~~",
            "....~~~...~~~G.~~~",
            "~~~~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~~~~",
        ),
        (4, 2),
        EAST,
        True,
    ),
    LevelSpec(
        "The Corner the Plank Cannot Take",
        (
            "###########G######",
            "###########~######",
            "###########~######",
            "###########~######",
            "###########.######",
            "#...........~~~~~#",
            "###.~~~~###.######",
            "###.~~~~###.~~~###",
            "###.~~~~###.~~~~##",
            "###.~~~####.~~~~##",
            "###.#######.~~~~##",
            "###.........~~~~~#",
            "###~~~~~##########",
            "###~~~~~##########",
            "###~~~~~##########",
            "###~~~~###########",
            "###~##############",
            "##################",
        ),
        (5, 3),
        EAST,
        False,
        ((5, 4), (5, 5), (5, 6), (5, 7), (5, 8)),
    ),
    LevelSpec(
        "Turnaround Spur",
        (
            "#####################",
            "#####################",
            "#####################",
            "#####################",
            "##########..~~...####",
            "##########.~~~~######",
            "##########.~~~~######",
            "#~~~~~.......~~######",
            "##~~~~.~~~~~.~#######",
            "##~~~~.~~~~#.########",
            "##~~~~.~~~~#.########",
            "###~~~.~~~~#.########",
            "######.###...########",
            "######......~~~G#####",
            "######~~~~~##########",
            "######~~~~~##########",
            "######~~~~~##########",
            "######~~~~###########",
            "######~##############",
        ),
        (4, 15),
        WEST,
        True,
    ),
    LevelSpec(
        "False Finish",
        (
            "#####################",
            "#####################",
            "#####################",
            "#####################",
            "#######..~~~......###",
            "#############.~~~.###",
            "#############.~~~.###",
            "#############.~~~.###",
            "###......~~~......###",
            "###.~~~.#####~~~~~###",
            "###.~~~.#####~~~~~###",
            "###.~~~.#####~~~~~###",
            "###.~~~.######~~~~###",
            "###.###.#########~###",
            "###......~~~G.....###",
            "###~~~~~#############",
            "###~~~~~#############",
            "###~~~~~#############",
            "###~~~~##############",
            "###~#################",
            "#####################",
        ),
        (4, 7),
        EAST,
        True,
    ),
    LevelSpec(
        "Offset Circuit",
        (
            "#######################",
            "#######################",
            "#######################",
            "#######################",
            "######...~~~.#~~~~~####",
            "######...~~~.......####",
            "#######~~~~~#.~~~~.####",
            "#######~~~~~#.~~~~.####",
            "#######~~~~~#.~~~~.####",
            "#######~~~~##.#~~~.####",
            "#######~#####.####.####",
            "###......~~~.......####",
            "###.~~~.######~~~~~####",
            "###.~~~.######~~~~~####",
            "###.~~~.######~~~~~####",
            "###.~~~.#######~~~~####",
            "###.###.##########~####",
            "###......~~~G......####",
            "###~~~~~###############",
            "###~~~~~###############",
            "###~~~~~###############",
            "###~~~~################",
            "###~###################",
        ),
        (4, 7),
        EAST,
        True,
    ),
    LevelSpec(
        "Full Relay",
        (
            "##########################",
            "##########################",
            "##########################",
            "##########################",
            "##########################",
            "########...~~~.......#####",
            "###############.~~~~.#####",
            "###############.~~~~.#####",
            "###############.~~~~.#####",
            "###############.#~~~.#####",
            "###############.####.#####",
            "#####......~~~.......#####",
            "#####.~~~~.#####~~~~~#####",
            "#####.......~~~G~~~~~#####",
            "#####.~~~~.~~~~#~~~~~#####",
            "#####.~~~#~~~~~##~~~~#####",
            "#####~~~~~~~~~~#####~#####",
            "#####~~~~~~~~~~###########",
            "#####.~~~~.~~~~###########",
            "#####......~~~~~##########",
            "##########################",
            "##########################",
            "##########################",
        ),
        (5, 9),
        EAST,
        True,
    ),
)


class PlankRelayHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: PlankRelay | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.game is None:
            return frame
        x0, width = 6, 52
        frame[60:63, x0 : x0 + width] = COLOR_WALL
        remaining = max(0, self.game.step_budget - self.game.step_count)
        filled = int((width - 2) * remaining / max(1, self.game.step_budget))
        frame[61, x0 + 1 : x0 + 1 + filled] = COLOR_GOAL
        return frame


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _make_level(spec: LevelSpec, budget: int) -> Level:
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_OUTSIDE, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level([board], (64, 64), {"spec": spec, "budget": budget}, spec.name)


class PlankRelay(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = PlankRelayHud()
        levels = [_make_level(spec, BASELINES[index] * 6) for index, spec in enumerate(LEVEL_SPECS)]
        super().__init__(
            "plank_relay-0001",
            levels,
            Camera(0, 0, 64, 64, COLOR_OUTSIDE, COLOR_OUTSIDE, [self._hud]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, _level: Level) -> None:
        self.spec: LevelSpec = self.current_level.get_data("spec")
        self.rows = self.spec.rows
        self.height = len(self.rows)
        self.width = len(self.rows[0])
        self.tile_size = min(4, 60 // self.height, 60 // self.width)
        self.x_origin = (64 - self.tile_size * self.width) // 2
        self.y_origin = (64 - self.tile_size * self.height) // 2
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.player_position = self.spec.player
        self.player_facing = self.spec.facing
        self.plank_mode = "CARRIED" if self.spec.carried else "DROPPED"
        self.dropped_plank_cells = tuple(self.spec.plank_cells)
        self.goal_cells = {
            (row, col) for row, line in enumerate(self.rows) for col, value in enumerate(line) if value == "G"
        }
        self.step_budget = int(self.current_level.get_data("budget"))
        self.step_count = 0
        self.error_preview: dict[str, object] | None = None
        self._visual_override: dict[str, object] | None = None
        self._animation: list[dict[str, object] | None] = []
        self._animation_finish = "complete"
        self.terminal_success = False
        self.terminal_failure = False
        self.undo_history: list[tuple[tuple[int, int], tuple[int, int], str, tuple[tuple[int, int], ...]]] = []
        self._sync_board()

    def _snapshot(self) -> tuple[tuple[int, int], tuple[int, int], str, tuple[tuple[int, int], ...]]:
        return self.player_position, self.player_facing, self.plank_mode, tuple(self.dropped_plank_cells)

    def _restore(self, state: tuple[tuple[int, int], tuple[int, int], str, tuple[tuple[int, int], ...]]) -> None:
        self.player_position, self.player_facing, self.plank_mode, self.dropped_plank_cells = state
        self.error_preview = None
        self._visual_override = None
        self._animation = []
        self.terminal_success = False
        self.terminal_failure = False

    def _terrain(self, cell: tuple[int, int]) -> str | None:
        row, col = cell
        if row < 0 or col < 0 or row >= self.height or col >= self.width:
            return None
        return self.rows[row][col]

    @staticmethod
    def _add(cell: tuple[int, int], direction: tuple[int, int], amount: int = 1) -> tuple[int, int]:
        return cell[0] + direction[0] * amount, cell[1] + direction[1] * amount

    def _held_cells(
        self, player: tuple[int, int] | None = None, facing: tuple[int, int] | None = None
    ) -> tuple[tuple[int, int], ...]:
        pivot = self.player_position if player is None else player
        direction = self.player_facing if facing is None else facing
        return tuple(self._add(pivot, direction, distance) for distance in range(1, PLANK_LENGTH + 1))

    def _plank_cells(self) -> tuple[tuple[int, int], ...]:
        return self._held_cells() if self.plank_mode == "CARRIED" else self.dropped_plank_cells

    def _is_ground(self, cell: tuple[int, int]) -> bool:
        return self._terrain(cell) in {".", "G"}

    def _carried_footprint_valid(self, player: tuple[int, int], facing: tuple[int, int]) -> bool:
        if not self._is_ground(player):
            return False
        return all(self._terrain(cell) not in {None, "#"} for cell in self._held_cells(player, facing))

    def _rotation_sweep(self, old_facing: tuple[int, int], new_facing: tuple[int, int]) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for a in range(6):
            for b in range(6):
                if not 1 <= a * a + b * b <= 25:
                    continue
                row = self.player_position[0] + a * old_facing[0] + b * new_facing[0]
                col = self.player_position[1] + a * old_facing[1] + b * new_facing[1]
                cells.add((row, col))
        return cells

    def _queue_animation(self, frames: list[dict[str, object] | None], finish: str = "complete") -> None:
        self._animation = frames
        self._animation_finish = finish
        self._play_animation_frame()

    def _play_animation_frame(self) -> None:
        self._visual_override = self._animation.pop(0)
        self._sync_board()
        if self._animation:
            return
        if self._animation_finish == "next":
            self.next_level()
            self.complete_action()
        elif self._animation_finish == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _finish_resolved_action(self, *, animate: list[dict[str, object] | None] | None = None) -> None:
        if self.player_position in self.goal_cells:
            self.terminal_success = True
            pulses = [
                {"kind": "completion", "color": color}
                for color in (COLOR_GOAL, COLOR_GRIP, COLOR_COMPLETE, COLOR_GOAL, COLOR_GRIP, COLOR_COMPLETE)
            ]
            pulses.append(None)
            self._queue_animation(pulses, "next")
            return
        if self.step_count >= self.step_budget:
            self.terminal_failure = True
            self._queue_animation([{"kind": "failure"}, None], "lose")
            return
        if animate:
            self._queue_animation(animate)
            return
        self._visual_override = None
        self._sync_board()
        self.complete_action()

    def _invalid(self, preview: dict[str, object]) -> None:
        self.error_preview = preview
        frames: list[dict[str, object] | None] = [
            {"kind": "error", "phase": 0, "preview": preview},
            {"kind": "error", "phase": 1, "preview": preview},
        ]
        if self.step_count >= self.step_budget:
            self.terminal_failure = True
            frames.extend(({"kind": "failure"}, None))
            self._queue_animation(frames, "lose")
            return
        self._queue_animation(frames)

    def _move_without_plank(self, direction: tuple[int, int]) -> None:
        target = self._add(self.player_position, direction)
        if self._is_ground(target) or target in self.dropped_plank_cells:
            self.player_position = target
            self.player_facing = direction
            self._finish_resolved_action()
            return
        self._invalid({"type": "foot", "cells": (target,)})

    def _move_with_plank(self, direction: tuple[int, int]) -> None:
        facing = self.player_facing
        if direction == facing or direction == (-facing[0], -facing[1]):
            target = self._add(self.player_position, direction)
            attempted = self._held_cells(target, facing)
            if self._carried_footprint_valid(target, facing):
                self.player_position = target
                self._finish_resolved_action()
                return
            collisions = tuple(cell for cell in (target, *attempted) if self._terrain(cell) in {None, "#"})
            self._invalid({"type": "carry", "player": target, "cells": attempted, "collisions": collisions})
            return

        sweep = self._rotation_sweep(facing, direction)
        blocked = tuple(cell for cell in sweep if self._terrain(cell) in {None, "#"})
        if blocked:
            self._invalid({"type": "rotation", "cells": tuple(sweep), "collisions": blocked})
            return

        old_angle = degrees(atan2(facing[0], facing[1]))
        new_angle = degrees(atan2(direction[0], direction[1]))
        delta = (new_angle - old_angle + 180) % 360 - 180
        old_cells = self._held_cells()
        new_cells = self._held_cells(self.player_position, direction)
        self.player_facing = direction
        frames: list[dict[str, object] | None] = [
            {"kind": "plank_snapshot", "cells": old_cells, "carried": True, "ghost": False},
            {"kind": "rotation", "angle": old_angle + delta * 0.25},
            {"kind": "rotation", "angle": old_angle + delta * 0.5},
            {"kind": "rotation", "angle": old_angle + delta * 0.75},
            {"kind": "plank_snapshot", "cells": new_cells, "carried": True, "ghost": False},
        ]
        self._finish_resolved_action(animate=frames)

    def _drop(self) -> None:
        cells = self._held_cells()
        unsupported = tuple(cell for cell in (cells[0], cells[-1]) if not self._is_ground(cell))
        blocked = tuple(cell for cell in cells if self._terrain(cell) in {None, "#"})
        if unsupported or blocked:
            self._invalid({"type": "drop", "cells": cells, "collisions": blocked, "unsupported": unsupported})
            return
        source = {"kind": "plank_snapshot", "cells": cells, "carried": True, "ghost": False}
        ghost = {"kind": "plank_snapshot", "cells": cells, "carried": False, "ghost": True}
        destination = {"kind": "plank_snapshot", "cells": cells, "carried": False, "ghost": False}
        self.plank_mode = "DROPPED"
        self.dropped_plank_cells = cells
        self._finish_resolved_action(animate=[source, ghost, destination])

    def _lift(self) -> None:
        cells = self.dropped_plank_cells
        unit = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
        stance0 = self._add(cells[0], (-unit[0], -unit[1]))
        stance4 = self._add(cells[-1], unit)
        facing: tuple[int, int] | None = None
        if self.player_position == stance0 and self._is_ground(stance0):
            facing = unit
        elif self.player_position == stance4 and self._is_ground(stance4):
            facing = (-unit[0], -unit[1])
        if facing is None:
            inaccessible = []
            if not self._is_ground(stance0):
                inaccessible.append(cells[0])
            if not self._is_ground(stance4):
                inaccessible.append(cells[-1])
            if not inaccessible:
                inaccessible = [cells[0], cells[-1]]
            self._invalid({"type": "lift", "cells": tuple(inaccessible)})
            return
        source = {"kind": "plank_snapshot", "cells": cells, "carried": False, "ghost": False}
        ghost = {"kind": "plank_snapshot", "cells": cells, "carried": True, "ghost": True}
        destination = {"kind": "plank_snapshot", "cells": cells, "carried": True, "ghost": False}
        self.plank_mode = "CARRIED"
        self.player_facing = facing
        self._finish_resolved_action(animate=[source, ghost, destination])

    def step(self) -> None:
        if self._animation:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action == _action_id(GameAction.RESET):
            self._sync_board()
            self.complete_action()
            return

        self.error_preview = None
        self._visual_override = None
        self.step_count += 1
        if action == ACTION_UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self._finish_resolved_action()
            return
        before = self._snapshot()
        if action in DIRECTIONS:
            if self.plank_mode == "CARRIED":
                self._move_with_plank(DIRECTIONS[action])
            else:
                self._move_without_plank(DIRECTIONS[action])
            if self._snapshot() != before:
                self.undo_history.append(before)
            return
        if action == ACTION_SPACE:
            if self.plank_mode == "CARRIED":
                self._drop()
            else:
                self._lift()
            if self._snapshot() != before:
                self.undo_history.append(before)
            return
        self._finish_resolved_action()

    def _cell_rect(self, cell: tuple[int, int]) -> tuple[slice, slice] | None:
        row, col = cell
        if not (0 <= row < self.height and 0 <= col < self.width):
            return None
        y = self.y_origin + row * self.tile_size
        x = self.x_origin + col * self.tile_size
        return slice(y, y + self.tile_size), slice(x, x + self.tile_size)

    def _paint_cell(self, pixels: np.ndarray, cell: tuple[int, int], color: int) -> None:
        rect = self._cell_rect(cell)
        if rect is not None:
            pixels[rect] = color

    def _draw_plank_cells(
        self, pixels: np.ndarray, cells: tuple[tuple[int, int], ...], *, carried: bool, ghost: bool = False
    ) -> None:
        for index, cell in enumerate(cells):
            if ghost:
                color = COLOR_GHOST
            else:
                color = COLOR_CAP if index in {0, len(cells) - 1} else COLOR_PLANK
            self._paint_cell(pixels, cell, color)
        if carried and not ghost:
            rect = self._cell_rect(cells[0])
            if rect is not None:
                y = rect[0].start + self.tile_size // 2
                x = rect[1].start + self.tile_size // 2
                pixels[min(y, rect[0].stop - 1), min(x, rect[1].stop - 1)] = COLOR_GRIP

    def _draw_rotating_plank(self, pixels: np.ndarray, angle: float) -> None:
        tile = self.tile_size
        pivot_y = self.y_origin + self.player_position[0] * tile + (tile - 1) / 2
        pivot_x = self.x_origin + self.player_position[1] * tile + (tile - 1) / 2
        angle_rad = radians(angle)
        dy, dx = sin(angle_rad), cos(angle_rad)
        half = max(0, (tile - 1) // 2)
        for distance in np.linspace(tile * 0.65, tile * 5.45, tile * 7):
            y = round(pivot_y + dy * distance)
            x = round(pivot_x + dx * distance)
            along = distance / tile
            color = COLOR_CAP if along < 1.35 or along > 4.65 else COLOR_PLANK
            for offset in range(-half, half + 1):
                yy = round(y + dx * offset)
                xx = round(x - dy * offset)
                if 0 <= yy < 64 and 0 <= xx < 64:
                    pixels[yy, xx] = color
        grip_y = round(pivot_y + dy * tile)
        grip_x = round(pivot_x + dx * tile)
        if 0 <= grip_y < 64 and 0 <= grip_x < 64:
            pixels[grip_y, grip_x] = COLOR_GRIP

    def _draw_player(self, pixels: np.ndarray) -> None:
        rect = self._cell_rect(self.player_position)
        if rect is None:
            return
        pixels[rect] = COLOR_PLAYER
        y0, y1 = rect[0].start, rect[0].stop
        x0, x1 = rect[1].start, rect[1].stop
        dr, dc = self.player_facing
        if dr < 0:
            pixels[y0, x0:x1] = COLOR_FACING
        elif dr > 0:
            pixels[y1 - 1, x0:x1] = COLOR_FACING
        elif dc < 0:
            pixels[y0:y1, x0] = COLOR_FACING
        else:
            pixels[y0:y1, x1 - 1] = COLOR_FACING

    def _checker(self, pixels: np.ndarray, cell: tuple[int, int], phase: int, *, solid: bool = False) -> None:
        rect = self._cell_rect(cell)
        if rect is None:
            return
        for y in range(rect[0].start, rect[0].stop):
            for x in range(rect[1].start, rect[1].stop):
                if solid or (x + y + phase) % 2 == 0:
                    pixels[y, x] = COLOR_ERROR

    def _draw_error(self, pixels: np.ndarray, override: dict[str, object]) -> None:
        preview = override["preview"]
        assert isinstance(preview, dict)
        phase = int(override["phase"])
        collision_cells = set(preview.get("collisions", ()))
        unsupported = set(preview.get("unsupported", ()))
        for cell in preview.get("cells", ()):
            self._checker(pixels, cell, phase, solid=cell in collision_cells or cell in unsupported)
        player = preview.get("player")
        if player is not None:
            self._checker(pixels, player, phase, solid=player in collision_cells)

    def _sync_board(self) -> None:
        pixels = np.full((64, 64), COLOR_OUTSIDE, dtype=np.int8)
        terrain_colors = {".": COLOR_GROUND, "~": COLOR_VOID, "#": COLOR_WALL, "G": COLOR_GOAL}
        for row, line in enumerate(self.rows):
            for col, symbol in enumerate(line):
                self._paint_cell(pixels, (row, col), terrain_colors[symbol])

        override = self._visual_override
        if override and override.get("kind") == "rotation":
            self._draw_rotating_plank(pixels, float(override["angle"]))
        elif override and override.get("kind") == "plank_snapshot":
            self._draw_plank_cells(
                pixels, tuple(override["cells"]), carried=bool(override["carried"]), ghost=bool(override["ghost"])
            )
        else:
            self._draw_plank_cells(pixels, self._plank_cells(), carried=self.plank_mode == "CARRIED")

        self._draw_player(pixels)
        if override and override.get("kind") == "error":
            self._draw_error(pixels, override)
        elif override and override.get("kind") == "completion":
            self._paint_cell(pixels, self.player_position, int(override["color"]))
        elif override and override.get("kind") == "failure":
            for cell in self._plank_cells():
                self._checker(pixels, cell, 0)
            self._checker(pixels, self.player_position, 0, solid=True)
        self.board.pixels = pixels
