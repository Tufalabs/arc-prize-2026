from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "vector_wake-0001"
SIZE = 64
CELL = 4
BOARD_X = 4
BOARD_Y = 8

BG = 5
VOID = 4
FLOOR = 1
RED = 2
HUD_EMPTY = 3
BLUE = 9
WAKE = 10
CHECK_INACTIVE = 11
CHECK_CENTER = 12
CHECK_ACTIVE_CENTER = 14
DOCK_FRAME = 14
DOCK_READY = 0

UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
UNDO = 7
DIRECTIONS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
BASELINES = (9, 12, 19, 28, 43, 54, 63)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    initial_heading: int | None
    initial_speed: int
    budget: int


LEVEL_SPECS = (
    LevelSpec(
        "Runway Cup",
        (
            "##############",
            "##############",
            "##############",
            "##############",
            "##############",
            "##########D###",
            "##########.###",
            "#S.....a...###",
            "##############",
            "##############",
            "##############",
            "##############",
            "##############",
            "##############",
        ),
        None,
        0,
        6 * BASELINES[0],
    ),
    LevelSpec(
        "Short Corner",
        (
            "##############",
            "##############",
            "##############",
            "#########..D##",
            "#########.####",
            "#########.####",
            "#########a####",
            "#########.####",
            "#########.####",
            "#########.####",
            "#########.####",
            "#S.........###",
            "##############",
            "##############",
        ),
        None,
        0,
        6 * BASELINES[1],
    ),
    LevelSpec(
        "Three Faces of a Junction",
        (
            "##############",
            "##############",
            "##############",
            "##############",
            "#######a######",
            "#######.######",
            "#######.######",
            "####.S....####",
            "####D####.####",
            "#########.####",
            "#########b####",
            "##############",
            "##############",
            "##############",
        ),
        None,
        0,
        6 * BASELINES[2],
    ),
    LevelSpec(
        "The Returning Circuit",
        (
            "##############",
            "##############",
            "###D##########",
            "###.##########",
            "###.##########",
            "###.#a########",
            "###.#.########",
            "###..S......##",
            "#####.#####.##",
            "#####.#####.##",
            "#####.#####b##",
            "#####.#####.##",
            "#####.......##",
            "##############",
        ),
        RIGHT,
        3,
        6 * BASELINES[3],
    ),
    LevelSpec(
        "Farther First",
        (
            "##############",
            "##############",
            "##############",
            "##.a.........#",
            "##.####.####.#",
            "##.####.####.#",
            "##.####.####.#",
            "##S......b##.#",
            "##.####.####.#",
            "##..D##...d#.#",
            "#######.####.#",
            "#######.####.#",
            "#######...c..#",
            "##############",
        ),
        RIGHT,
        3,
        6 * BASELINES[4],
    ),
    LevelSpec(
        "Braided Handoff",
        (
            "##############",
            "##########e###",
            "##########.###",
            "##.a.........#",
            "##.####.##.#.#",
            "##.####S...#.#",
            "##.####.####.#",
            "##.......b##.#",
            "##.####.####.#",
            "##..D##...d#.#",
            "#######.####.#",
            "#######.####.#",
            "#######...c..#",
            "##############",
        ),
        RIGHT,
        2,
        6 * BASELINES[5],
    ),
    LevelSpec(
        "Reserved Berth",
        (
            "##############",
            "#...a.......f.",
            "#.#######.###.",
            "#.###b###.###.",
            "#.###.###.###.",
            "#.........###.",
            "#S###.###.###.",
            "#..D#.###.###.",
            "#.###.###.###.",
            "#.#e..........",
            "#####.###.###.",
            "#####.###c###.",
            "#####.#######.",
            "#####...d.....",
        ),
        DOWN,
        2,
        6 * BASELINES[6],
    ),
)


def _parse(spec: LevelSpec) -> dict[str, object]:
    floor: set[tuple[int, int]] = set()
    checkpoints: list[tuple[int, int]] = []
    start = (-1, -1)
    dock = (-1, -1)
    for row, line in enumerate(spec.rows):
        if len(line) != 14:
            raise ValueError(f"{spec.name}: row {row} is not 14 cells")
        for col, char in enumerate(line):
            if char == "#":
                continue
            floor.add((row, col))
            if char == "S":
                start = (row, col)
            elif char == "D":
                dock = (row, col)
            elif "a" <= char <= "f":
                checkpoints.append((row, col))
    checkpoints.sort(key=lambda cell: spec.rows[cell[0]][cell[1]])
    return {"floor": floor, "start": start, "dock": dock, "checkpoints": tuple(checkpoints)}


def _level(spec: LevelSpec) -> Level:
    canvas = Sprite(
        np.full((SIZE, SIZE), BG, dtype=np.int8), name="canvas", x=0, y=0, layer=0, collidable=False, tags=["canvas"]
    )
    return Level([canvas], grid_size=(SIZE, SIZE), data={"spec": spec, **_parse(spec)}, name=spec.name)


class VectorWake(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._canvas: Sprite | None = None
        self._frames: list[dict[str, object]] = []
        self._finish_after_frames = False
        self._lose_after_frames = False
        super().__init__(
            GAME_ID,
            [_level(spec) for spec in LEVEL_SPECS],
            Camera(0, 0, SIZE, SIZE, BG, BG),
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.floor: set[tuple[int, int]] = set(level.get_data("floor"))
        self.position: tuple[int, int] = tuple(level.get_data("start"))
        self.dock_position: tuple[int, int] = tuple(level.get_data("dock"))
        self.checkpoints: tuple[tuple[int, int], ...] = tuple(level.get_data("checkpoints"))
        self.checkpoint_active = [False] * len(self.checkpoints)
        self.heading = self.spec.initial_heading
        self.speed = self.spec.initial_speed
        self.last_sweep: list[tuple[int, int]] = []
        self.wake_cells = self._initial_wake()
        self.done = False
        self.selected_object = None
        self.cursor_cell = None
        self.cursor_visible = False
        self.remaining_steps = self.spec.budget
        self.step_budget = self.spec.budget
        self._submitted_calls = 0
        self._frames = []
        self._finish_after_frames = False
        self._lose_after_frames = False
        self.undo_history: list[
            tuple[
                tuple[int, int],
                tuple[bool, ...],
                int | None,
                int,
                tuple[tuple[int, int], ...],
                tuple[tuple[int, int], ...],
            ]
        ] = []
        self._canvas = level.get_sprites_by_tag("canvas")[0]
        self._set_frame({})

    def _snapshot(
        self,
    ) -> tuple[
        tuple[int, int], tuple[bool, ...], int | None, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]
    ]:
        return (
            self.position,
            tuple(self.checkpoint_active),
            self.heading,
            self.speed,
            tuple(self.last_sweep),
            tuple(self.wake_cells),
        )

    def _restore(
        self,
        state: tuple[
            tuple[int, int], tuple[bool, ...], int | None, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]
        ],
    ) -> None:
        position, checkpoints, heading, speed, last_sweep, wake_cells = state
        self.position = position
        self.checkpoint_active = list(checkpoints)
        self.heading = heading
        self.speed = speed
        self.last_sweep = list(last_sweep)
        self.wake_cells = list(wake_cells)
        self.done = False
        self._finish_after_frames = False
        self._lose_after_frames = False

    def _initial_wake(self) -> list[tuple[int, int]]:
        if self.speed == 0 or self.heading is None:
            return []
        dr, dc = DIRECTIONS[self.heading]
        cells = []
        for distance in range(self.speed, 0, -1):
            cell = (self.position[0] - dr * distance, self.position[1] - dc * distance)
            if cell in self.floor:
                cells.append(cell)
        return cells

    def _candidate(self, action: int) -> tuple[int | None, int, list[tuple[int, int]]]:
        if self.speed == 0:
            heading, speed = action, 1
        elif action == self.heading:
            heading, speed = self.heading, self.speed + 1
        elif action == OPPOSITE[self.heading]:
            heading, speed = self.heading, self.speed - 1
        else:
            heading, speed = action, self.speed
        if speed == 0:
            heading = None
        path: list[tuple[int, int]] = []
        if heading is not None:
            dr, dc = DIRECTIONS[heading]
            path = [(self.position[0] + dr * i, self.position[1] + dc * i) for i in range(1, speed + 1)]
        return heading, speed, path

    def _spend_step(self) -> None:
        self._submitted_calls += 1
        self.remaining_steps = max(0, self.remaining_steps - 1)

    def _resolve_direction(self, action: int) -> None:
        heading, speed, path = self._candidate(action)
        blocked_index = next((i for i, cell in enumerate(path) if cell not in self.floor), None)
        if speed > 3:
            self._queue_frames(
                [
                    {"invalid_path": path, "invalid_hud": True},
                    {"invalid_path": path, "dim": True, "invalid_hud": True},
                    {"invalid_path": path, "invalid_settled": True},
                ]
            )
            return
        if blocked_index is not None:
            self._queue_frames(
                [
                    {"invalid_path": path[: blocked_index + 1]},
                    {"invalid_path": path[: blocked_index + 1], "dim": True},
                    {"invalid_path": path[: blocked_index + 1], "invalid_settled": True},
                ]
            )
            return

        old_position = self.position
        old_heading = self.heading
        newly_active: dict[tuple[int, int], int] = {}
        for cell in path:
            if cell in self.checkpoints:
                idx = self.checkpoints.index(cell)
                if not self.checkpoint_active[idx]:
                    self.checkpoint_active[idx] = True
                    newly_active[cell] = idx
        if path:
            self.position = path[-1]
        self.speed = speed
        self.heading = heading
        self.last_sweep = list(path)
        self.wake_cells = [] if speed == 0 else [old_position, *path[:-1]]
        won = all(self.checkpoint_active) and self.position == self.dock_position and self.speed == 0

        if path:
            visible_active = [active for active in self.checkpoint_active]
            for idx in newly_active.values():
                visible_active[idx] = False
            frames: list[dict[str, object]] = []
            for index, cell in enumerate(path):
                if cell in newly_active:
                    visible_active[newly_active[cell]] = True
                frames.append(
                    {
                        "token_position": cell,
                        "wake_cells": [old_position, *path[:index]],
                        "checkpoint_active": tuple(visible_active),
                        "dock_cross": cell == self.dock_position,
                    }
                )
            self._queue_frames(frames)
        elif won:
            self._queue_frames([{"token_heading": old_heading}, {}])
        else:
            self._queue_frames([{"token_heading": old_heading}, {}])

        if won:
            self.done = True
            completion = [{"completion_color": color} for color in (DOCK_READY, DOCK_FRAME, DOCK_READY, DOCK_FRAME)]
            self._frames.extend(completion)
            self._finish_after_frames = True

    def _queue_frames(self, frames: list[dict[str, object]]) -> None:
        self._frames = frames

    def _play_frame(self) -> None:
        frame_state = self._frames.pop(0)
        self._set_frame(frame_state)
        if not self._frames:
            if self._finish_after_frames:
                self._finish_after_frames = False
                # Mark the submitted action complete before requesting the deferred
                # level transition, so it is not replayed against the next level.
                self.complete_action()
                self.next_level()
            elif self._lose_after_frames:
                self._lose_after_frames = False
                self.lose()
                self.complete_action()
            else:
                self.complete_action()

    def step(self) -> None:
        if self._frames:
            self._play_frame()
            return
        if self.action.id == GameAction.RESET:
            self._set_frame({})
            self.complete_action()
            return
        if self.done:
            self.complete_action()
            return
        self._spend_step()
        action = int(self.action.id.value)
        if action == UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self._queue_frames([{}])
        elif action in DIRECTIONS:
            before = self._snapshot()
            self._resolve_direction(action)
            if self._snapshot() != before:
                self.undo_history.append(before)
        else:
            self._queue_frames([{"unsupported": True}, {}])
        if not self.done and self.remaining_steps <= 0:
            self._frames.append({"budget_flash": True})
            self._lose_after_frames = True
        self._play_frame()

    def _set_frame(self, transient: dict[str, object]) -> None:
        if self._canvas is not None:
            self._canvas.pixels = self._render(transient)

    def _render(self, transient: dict[str, object]) -> np.ndarray:
        frame = np.full((SIZE, SIZE), BG, dtype=np.int8)
        frame[BOARD_Y:SIZE, BOARD_X : BOARD_X + 14 * CELL] = VOID
        for row, col in self.floor:
            self._fill_cell(frame, (row, col), FLOOR)

        active_bits = list(transient.get("checkpoint_active", self.checkpoint_active))
        for index, cell in enumerate(self.checkpoints):
            self._draw_checkpoint(frame, cell, active_bits[index])
        self._draw_dock(frame, transient)

        wake = transient.get("wake_cells", self.wake_cells)
        for cell in wake:
            x, y = self._cell_xy(cell)
            frame[y + 1 : y + 3, x + 1 : x + 3] = WAKE

        token_position = transient.get("token_position", self.position)
        self._draw_token(frame, token_position, transient)
        self._draw_hud(frame)
        self._draw_budget(frame)

        invalid_path = transient.get("invalid_path")
        if invalid_path:
            for cell in invalid_path[:-1]:
                self._outline_cell(frame, cell, RED)
            self._fill_cell(frame, invalid_path[-1], RED)
        if transient.get("invalid_hud"):
            frame[0:7, 0:5] = RED
        if transient.get("invalid_settled"):
            frame[1, 23:26] = RED
            frame[2, 24] = RED
            frame[3, 23:26] = RED
        if transient.get("unsupported"):
            frame[0:7, 0:SIZE] = HUD_EMPTY
        if transient.get("budget_flash"):
            frame[6:8, :] = RED
        if transient.get("dock_cross"):
            x, y = self._cell_xy(self.dock_position)
            frame[y - 1, x - 1 : x + CELL + 1] = CHECK_ACTIVE_CENTER
            frame[y + CELL, x - 1 : x + CELL + 1] = CHECK_ACTIVE_CENTER
            frame[y - 1 : y + CELL + 1, x - 1] = CHECK_ACTIVE_CENTER
            frame[y - 1 : y + CELL + 1, x + CELL] = CHECK_ACTIVE_CENTER
        if transient.get("dim"):
            mask = frame != RED
            frame[mask] = np.where(frame[mask] == BG, BG, HUD_EMPTY)
        return frame

    def _draw_hud(self, frame: np.ndarray) -> None:
        icon = self._token_mask(self.heading if self.speed else None)
        self._blit_mask(frame, icon, 1, 1, BLUE)
        for index in range(3):
            x = 9 + index * 5
            frame[2:5, x : x + 3] = CHECK_ACTIVE_CENTER if index < self.speed else HUD_EMPTY

    def _draw_budget(self, frame: np.ndarray) -> None:
        frame[6:8, 27:63] = HUD_EMPTY
        width = int(36 * self.remaining_steps / max(1, self.step_budget))
        if width:
            frame[6:8, 27 : 27 + width] = CHECK_ACTIVE_CENTER

    def _draw_checkpoint(self, frame: np.ndarray, cell: tuple[int, int], active: bool) -> None:
        outer = RED if active else CHECK_INACTIVE
        center = CHECK_ACTIVE_CENTER if active else CHECK_CENTER
        pixels = np.array(
            [
                [-1, outer, outer, -1],
                [outer, center, center, outer],
                [outer, center, center, outer],
                [-1, outer, outer, -1],
            ],
            dtype=np.int8,
        )
        x, y = self._cell_xy(cell)
        self._blit(frame, pixels, x, y)

    def _draw_dock(self, frame: np.ndarray, transient: dict[str, object]) -> None:
        color = DOCK_READY if all(self.checkpoint_active) else WAKE
        frame_color = DOCK_FRAME
        if "completion_color" in transient:
            color = int(transient["completion_color"])
            frame_color = color
        pixels = np.array(
            [
                [frame_color] * 4,
                [frame_color, color, color, frame_color],
                [frame_color, color, color, frame_color],
                [frame_color] * 4,
            ],
            dtype=np.int8,
        )
        x, y = self._cell_xy(self.dock_position)
        frame[y : y + 4, x : x + 4] = pixels

    def _draw_token(self, frame: np.ndarray, cell: tuple[int, int], transient: dict[str, object]) -> None:
        color = BLUE
        if "completion_color" in transient:
            color = int(transient["completion_color"])
        token_heading = transient.get("token_heading", self.heading if self.speed else None)
        mask = self._token_mask(token_heading)
        x, y = self._cell_xy(cell)
        self._blit_mask(frame, mask, x, y, color)

    @staticmethod
    def _token_mask(heading: int | None) -> np.ndarray:
        right = np.array([[0, 0, 1, 0], [1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int8)
        if heading is None:
            return np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
        turns = {RIGHT: 0, DOWN: 1, LEFT: 2, UP: 3}[heading]
        return np.rot90(right, -turns)

    @staticmethod
    def _blit_mask(frame: np.ndarray, mask: np.ndarray, x: int, y: int, color: int) -> None:
        area = frame[y : y + mask.shape[0], x : x + mask.shape[1]]
        area[mask == 1] = color

    @staticmethod
    def _blit(frame: np.ndarray, pixels: np.ndarray, x: int, y: int) -> None:
        area = frame[y : y + pixels.shape[0], x : x + pixels.shape[1]]
        area[pixels >= 0] = pixels[pixels >= 0]

    @staticmethod
    def _cell_xy(cell: tuple[int, int]) -> tuple[int, int]:
        row, col = cell
        return BOARD_X + col * CELL, BOARD_Y + row * CELL

    def _fill_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        row, col = cell
        if 0 <= row < 14 and 0 <= col < 14:
            x, y = self._cell_xy(cell)
            frame[y : y + CELL, x : x + CELL] = color

    def _outline_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        row, col = cell
        if not (0 <= row < 14 and 0 <= col < 14):
            return
        x, y = self._cell_xy(cell)
        frame[y, x : x + CELL] = color
        frame[y + CELL - 1, x : x + CELL] = color
        frame[y : y + CELL, x] = color
        frame[y : y + CELL, x + CELL - 1] = color
