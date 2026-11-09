from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
BOARD_LEFT = 1
BOARD_TOP = 7
CELL_SIZE = 3

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
OPPOSITE = {ACTION_UP: ACTION_DOWN, ACTION_DOWN: ACTION_UP, ACTION_LEFT: ACTION_RIGHT, ACTION_RIGHT: ACTION_LEFT}

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_SPENT = 3
COLOR_PANEL = 4
COLOR_VOID = 5
COLOR_SEAL = 6
COLOR_ACTIVE_SEAL = 7
COLOR_INVALID = 8
COLOR_PLAYER = 9
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_EXHAUSTED = 13
COLOR_GREEN = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    baseline_actions: int

    @property
    def step_cap(self) -> int:
        return self.baseline_actions * 6


LEVEL_SPECS = (
    LevelSpec(
        "The Shortened Return",
        (
            "#################",
            "#################",
            "#################",
            "########P########",
            "########.########",
            "#####...1########",
            "######.#.########",
            "####X#.##########",
            "###....##########",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
        ),
        8,
    ),
    LevelSpec(
        "One Cell Behind",
        (
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "##########2######",
            "##P.......1.#####",
            "##########.######",
            "##########.######",
            "##########.######",
            "##########X######",
            "##########.######",
            "#################",
            "#################",
            "#################",
        ),
        14,
    ),
    LevelSpec(
        "The Non-Inverse Tour",
        (
            "#################",
            "#################",
            "#################",
            "#####...X...#####",
            "#####.##.##3#####",
            "#####.##.##.#####",
            "#####.##.##.#####",
            "#####.##.##.#####",
            "###.....P2#.#####",
            "#####.#####.#####",
            "#####.#####.#####",
            "#####.#####.#####",
            "#####.#####.#####",
            "#####1......#####",
            "#################",
            "#################",
            "#################",
        ),
        23,
    ),
    LevelSpec(
        "Pocket Circuit",
        (
            "#################",
            "#################",
            "#################",
            "#####...3..######",
            "#####.####.######",
            "#####.####.######",
            "#####.####.######",
            "#####.####.######",
            "##..4..P.......##",
            "#######.##.##.###",
            "#######.#####.###",
            "#######1###2..###",
            "#######.#########",
            "#######X#########",
            "#################",
            "#################",
            "#################",
        ),
        32,
    ),
    LevelSpec(
        "The Junction You Cross",
        (
            "#################",
            "#################",
            "#################",
            "####.3.1.########",
            "####2###.########",
            "####.###.X...####",
            "####.###.###.####",
            "####.###5###.####",
            "#.......P.......#",
            "####.###.########",
            "########.########",
            "########.########",
            "########.########",
            "########.########",
            "########...4#####",
            "########.########",
            "#################",
        ),
        43,
    ),
    LevelSpec(
        "Four-Petal Coupling",
        (
            "#################",
            "#################",
            "#################",
            "#####...5########",
            "########.########",
            "###6####...X##.##",
            "###.####.#####1##",
            "###.####.#####.##",
            "#.......P.......#",
            "##.#####.########",
            "##4#####.########",
            "##.#####.########",
            "########.########",
            "########...3#####",
            "########2########",
            "########.########",
            "#################",
        ),
        48,
    ),
    LevelSpec(
        "Keep Three North",
        (
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "#################",
            "##..........#####",
            "##.##.....1X.####",
            "##4##3.##..P..5##",
            "##........#.#####",
            "##..#..##.....6##",
            "##..##.##########",
            "##..##.##########",
            "##2.##.##########",
            "###.##.##########",
            "###.##.##########",
            "#################",
        ),
        53,
    ),
    LevelSpec(
        "Reconstruct the Approach",
        (
            "#################",
            "#################",
            "#######.#########",
            "#######X#########",
            "####.##.#########",
            "####5##.#########",
            "####.##.#######.#",
            "####............#",
            "####.##..2.3....#",
            "#...P..#.######.#",
            "###..#..6.....4.#",
            "###..##########7#",
            "###.###########.#",
            "###.#############",
            "###1#############",
            "#################",
            "#################",
        ),
        61,
    ),
)


def _parse_rows(rows: tuple[str, ...]) -> dict[str, object]:
    if len(rows) != 17 or any(len(row) != 17 for row in rows):
        raise ValueError("Compass Stretch levels must be 17x17.")
    floor: set[tuple[int, int]] = set()
    start: tuple[int, int] | None = None
    exit_cell: tuple[int, int] | None = None
    numbered_seals: list[tuple[int, tuple[int, int]]] = []
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            if symbol == "#":
                continue
            floor.add((x, y))
            if symbol == "P":
                start = (x, y)
            elif symbol == "X":
                exit_cell = (x, y)
            elif symbol.isdigit():
                numbered_seals.append((int(symbol), (x, y)))
    if start is None or exit_cell is None or not numbered_seals:
        raise ValueError("Every level needs P, X, and at least one numbered seal.")
    seals = tuple(cell for _, cell in sorted(numbered_seals))
    return {"floor": frozenset(floor), "start": start, "exit": exit_cell, "seals": seals}


def _build_level(spec: LevelSpec) -> Level:
    return Level(
        sprites=[], grid_size=(VIEW_SIZE, VIEW_SIZE), name=spec.name, data={"spec": spec, **_parse_rows(spec.rows)}
    )


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


def _cell_pixels(cell: tuple[int, int]) -> tuple[int, int]:
    return BOARD_LEFT + CELL_SIZE * cell[0], BOARD_TOP + CELL_SIZE * cell[1]


class CompassStretchDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: CompassStretch | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_VOID
        self._draw_board(frame, game)
        self._draw_compass(frame, game)
        self._draw_budget(frame, game)
        self._draw_win_pulse(frame, game)
        return frame

    def _fill_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x, y = _cell_pixels(cell)
        frame[y : y + 3, x : x + 3] = color

    def _draw_board(self, frame: np.ndarray, game: CompassStretch) -> None:
        for cell in game.floor:
            self._fill_cell(frame, cell, COLOR_FLOOR)

        phase = game.visual_phase
        for index, cell in enumerate(game.seals):
            active = bool(game.activated_seals & (1 << index))
            x, y = _cell_pixels(cell)
            ring = COLOR_ACTIVE_SEAL if active else COLOR_SEAL
            center = COLOR_YELLOW if active else COLOR_FLOOR
            if phase.get("seal_cell") == cell:
                seal_stage = int(phase.get("seal_stage", 2))
                ring = COLOR_SEAL if seal_stage == 0 else COLOR_ACTIVE_SEAL
                center = COLOR_YELLOW if seal_stage >= 2 else COLOR_FLOOR
            frame[y : y + 3, x : x + 3] = ring
            frame[y + 1, x + 1] = center

        x, y = _cell_pixels(game.exit_cell)
        exit_stage = int(phase.get("exit_stage", 3)) if phase.get("opening_exit") else 3
        if game.exit_open and exit_stage >= 2:
            exit_ring = COLOR_GREEN
            exit_center = COLOR_WHITE if exit_stage >= 3 else COLOR_PANEL
        elif game.exit_open:
            exit_ring = COLOR_ORANGE if exit_stage == 0 else COLOR_GREEN
            exit_center = COLOR_PANEL
        else:
            exit_ring = COLOR_ORANGE
            exit_center = COLOR_PANEL
        frame[y : y + 3, x : x + 3] = exit_ring
        frame[y + 1, x + 1] = exit_center

        for cell in phase.get("centerline", ()):
            if cell in game.floor:
                cx, cy = _cell_pixels(cell)
                frame[cy + 1, cx + 1] = COLOR_LIGHT_BLUE

        for cell in phase.get("invalid_cells", ()):
            if 0 <= cell[0] < 17 and 0 <= cell[1] < 17:
                cx, cy = _cell_pixels(cell)
                frame[cy + 1, cx + 1] = COLOR_INVALID
        missing = phase.get("invalid_missing")
        if phase.get("invalid_strong") and missing is not None and 0 <= missing[0] < 17 and 0 <= missing[1] < 17:
            mx, my = _cell_pixels(missing)
            frame[my + 1, mx : mx + 3] = COLOR_INVALID
            frame[my : my + 3, mx + 1] = COLOR_INVALID
        landing = phase.get("invalid_landing")
        if landing is not None and 0 <= landing[0] < 17 and 0 <= landing[1] < 17:
            lx, ly = _cell_pixels(landing)
            frame[ly, lx : lx + 3] = COLOR_INVALID
            frame[ly + 2, lx : lx + 3] = COLOR_INVALID
            frame[ly : ly + 3, lx] = COLOR_INVALID
            frame[ly : ly + 3, lx + 2] = COLOR_INVALID

        if not phase:
            special = set(game.seals) | {game.exit_cell, game.player}
            for action, reach in game.reach.items():
                dx, dy = DELTAS[action]
                path = [(game.player[0] + dx * step, game.player[1] + dy * step) for step in range(1, reach + 1)]
                if path and all(cell in game.floor for cell in path) and path[-1] not in special:
                    lx, ly = _cell_pixels(path[-1])
                    frame[ly + 1, lx + 1] = COLOR_LIGHT_BLUE

        player_cell = phase.get("player_cell", game.player)
        px, py = _cell_pixels(player_cell)
        frame[py, px + 1] = COLOR_PLAYER
        frame[py + 1, px] = COLOR_PLAYER
        frame[py + 1, px + 1] = COLOR_LIGHT_BLUE
        frame[py + 1, px + 2] = COLOR_PLAYER
        frame[py + 2, px + 1] = COLOR_PLAYER

    def _draw_compass(self, frame: np.ndarray, game: CompassStretch) -> None:
        frame[7:18, 53:64] = COLOR_PANEL
        cx, cy = 58, 12
        frame[cy - 1, cx] = COLOR_PLAYER
        frame[cy, cx - 1 : cx + 2] = COLOR_PLAYER
        frame[cy, cx] = COLOR_LIGHT_BLUE
        frame[cy + 1, cx] = COLOR_PLAYER
        phase = game.visual_phase
        flash_action = phase.get("flash_action")
        invalid_action = phase.get("invalid_action")
        for action, reach in game.reach.items():
            dx, dy = DELTAS[action]
            color = COLOR_LIGHT_BLUE
            if action == flash_action:
                color = COLOR_WHITE
            if action == invalid_action:
                color = COLOR_INVALID
            for distance in range(1, reach + 1):
                frame[cy + dy * (distance + 1), cx + dx * (distance + 1)] = color

        transfer = phase.get("transfer")
        if transfer is not None:
            used, stage = transfer
            opposite = OPPOSITE[used]
            odx, ody = DELTAS[opposite]
            udx, udy = DELTAS[used]
            if stage == 0:
                old_opposite_tip = game.reach[opposite] + 2
                tx = cx + odx * old_opposite_tip
                ty = cy + ody * old_opposite_tip
            elif stage == 1:
                tx, ty = cx, cy
            else:
                new_used_tip = game.reach[used] + 1
                tx = cx + udx * new_used_tip
                ty = cy + udy * new_used_tip
            frame[ty, tx] = COLOR_YELLOW
        for action in phase.get("saturated_tips", ()):
            dx, dy = DELTAS[action]
            reach = game.reach[action]
            frame[cy + dy * (reach + 1), cx + dx * (reach + 1)] = COLOR_YELLOW

    def _draw_budget(self, frame: np.ndarray, game: CompassStretch) -> None:
        frame[61:63, 2:62] = COLOR_SPENT
        remaining = max(0, game.step_cap - game.step_count)
        available = math.ceil(60 * remaining / game.step_cap)
        if available:
            color = COLOR_ORANGE if remaining / game.step_cap < 0.1 else COLOR_LIGHT_BLUE
            frame[61:63, 2 : 2 + available] = color
        if game.attempt_locked:
            frame[61:63, 2:62] = COLOR_EXHAUSTED

    def _draw_win_pulse(self, frame: np.ndarray, game: CompassStretch) -> None:
        stage = game.visual_phase.get("win_pulse")
        if stage is None:
            return
        px, py = _cell_pixels(game.player)
        cx, cy = px + 1, py + 1
        radius = 2 + int(stage) % 4
        x0, x1 = max(0, cx - radius), min(63, cx + radius)
        y0, y1 = max(0, cy - radius), min(63, cy + radius)
        frame[y0, x0 : x1 + 1] = COLOR_GREEN
        frame[y1, x0 : x1 + 1] = COLOR_GREEN
        frame[y0 : y1 + 1, x0] = COLOR_GREEN
        frame[y0 : y1 + 1, x1] = COLOR_GREEN


class CompassStretch(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = CompassStretchDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "compass_stretch-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_VOID, COLOR_VOID, [self._display]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed,
        )
        self._display.game = self

    def on_set_level(self, _level: Level) -> None:
        self.floor = set(self.current_level.get_data("floor"))
        self.start = tuple(self.current_level.get_data("start"))
        self.exit_cell = tuple(self.current_level.get_data("exit"))
        self.seals = tuple(self.current_level.get_data("seals"))
        self.step_cap = int(self.current_level.get_data("spec").step_cap)
        self._reset_attempt()

    def _reset_attempt(self) -> None:
        self.player = self.start
        self.reach = {ACTION_UP: 2, ACTION_DOWN: 3, ACTION_RIGHT: 2, ACTION_LEFT: 3}
        self.activated_seals = 0
        self.exit_open = False
        self.selection = None
        self.step_count = 0
        self.attempt_locked = False
        self.done = False
        self.animation_frames: list[dict[str, object]] = []
        self.visual_phase: dict[str, object] = {}
        self.pending_move: dict[str, object] | None = None

    def _finish_if_needed(self) -> None:
        self.visual_phase = {}
        if self.done:
            self.next_level()
            # next_level() defers non-final transitions until the next engine tick.
            # Mark this input complete so it is not replayed on the fresh level.
            self.complete_action()
            return
        if self.step_count >= self.step_cap:
            self.attempt_locked = True
        self.complete_action()

    def _commit_pending_move(self) -> None:
        pending = self.pending_move
        if pending is None:
            return
        action = int(pending["action"])
        landing = tuple(pending["landing"])
        self.player = landing
        seal_index = pending.get("seal_index")
        if seal_index is not None:
            self.activated_seals |= 1 << int(seal_index)
        opposite = OPPOSITE[action]
        if self.reach[opposite] > 1:
            self.reach[action] += 1
            self.reach[opposite] -= 1
        self.exit_open = self.activated_seals == (1 << len(self.seals)) - 1
        self.done = self.player == self.exit_cell and self.exit_open
        self.pending_move = None

    def _play_animation_frame(self) -> None:
        phase = self.animation_frames.pop(0)
        if phase.pop("commit", False):
            self._commit_pending_move()
        self.visual_phase = phase
        if not self.animation_frames:
            self._finish_if_needed()

    def _queue_valid_move(self, action: int, path: list[tuple[int, int]]) -> None:
        landing = path[-1]
        seal_index = self.seals.index(landing) if landing in self.seals else None
        newly_activated = seal_index is not None and not (self.activated_seals & (1 << seal_index))
        predicted_mask = self.activated_seals | ((1 << seal_index) if newly_activated else 0)
        opens_exit = not self.exit_open and predicted_mask == (1 << len(self.seals)) - 1
        will_win = landing == self.exit_cell and (self.exit_open or opens_exit)
        transfers = self.reach[OPPOSITE[action]] > 1
        self.pending_move = {
            "action": action,
            "landing": landing,
            "seal_index": seal_index if newly_activated else None,
        }
        self.animation_frames = [{"centerline": tuple(path), "flash_action": action}]
        for index, cell in enumerate(path):
            self.animation_frames.append(
                {"player_cell": cell, "flash_action": action, "commit": index == len(path) - 1}
            )
        if transfers:
            self.animation_frames.extend({"flash_action": action, "transfer": (action, stage)} for stage in range(3))
        else:
            tips = (action, OPPOSITE[action])
            self.animation_frames.extend({"saturated_tips": tips} for _ in range(2))
        if newly_activated:
            self.animation_frames.extend({"seal_cell": landing, "seal_stage": stage} for stage in range(3))
        if opens_exit:
            self.animation_frames.extend({"opening_exit": True, "exit_stage": stage} for stage in range(4))
        if will_win:
            self.animation_frames.extend({"win_pulse": stage} for stage in range(8))

    def _queue_invalid_move(self, action: int, path: list[tuple[int, int]]) -> None:
        in_bounds = tuple(cell for cell in path if 0 <= cell[0] < 17 and 0 <= cell[1] < 17)
        landing = path[-1]
        first_missing = next((cell for cell in path if cell not in self.floor), None)
        self.animation_frames = []
        for stage in range(6):
            self.animation_frames.append(
                {
                    "invalid_cells": in_bounds,
                    "invalid_missing": first_missing,
                    "invalid_strong": stage % 2 == 0,
                    "invalid_landing": landing,
                    "invalid_action": action,
                }
            )

    def _start_directional_action(self, action: int) -> None:
        self.step_count += 1
        dx, dy = DELTAS[action]
        path = [
            (self.player[0] + dx * distance, self.player[1] + dy * distance)
            for distance in range(1, self.reach[action] + 1)
        ]
        if all(cell in self.floor for cell in path):
            self._queue_valid_move(action, path)
        else:
            self._queue_invalid_move(action, path)
        self._play_animation_frame()

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        self.visual_phase = {}
        if action == _action_id(GameAction.RESET):
            self.complete_action()
            return
        if action == ACTION_SPACE:
            self._reset_attempt()
            self.complete_action()
            return
        if self.attempt_locked:
            self.complete_action()
            return
        if action in DELTAS:
            self._start_directional_action(action)
            return
        if action == ACTION_CLICK:
            self.step_count += 1
            if self.step_count >= self.step_cap:
                self.attempt_locked = True
            self.complete_action()
            return
        self.complete_action()
