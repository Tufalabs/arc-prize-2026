from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
DIRECTION_ANGLE = {(0, -1): -90.0, (1, 0): 0.0, (0, 1): 90.0, (-1, 0): 180.0}

VIEW_SIZE = 64
CELL_SIZE = 4

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_LEAF = 2
COLOR_WALL = 3
COLOR_BACKGROUND = 4
COLOR_FAILURE = 8
COLOR_PLAYER = 9
COLOR_GUIDE = 10
COLOR_BRASS = 11
COLOR_ORANGE = 12
COLOR_GOAL = 14


@dataclass(frozen=True)
class LeafSpec:
    leaf_id: str
    hinge: tuple[int, int]
    length: int
    orientation: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    leaves: tuple[LeafSpec, ...]
    baseline_actions: int


LEVEL_SPECS = (
    LevelSpec(
        "The wall does not disappear",
        (
            "#########",
            "#...#...#",
            "#...#...#",
            "#....S..#",
            "#.......#",
            "#.......#",
            "#G..#...#",
            "#...#...#",
            "#########",
        ),
        (LeafSpec("A", (4, 5), 2, (0, -1)),),
        10,
    ),
    LevelSpec(
        "Keep control of the handle",
        (
            "#############",
            "##S##########",
            "##.##########",
            "##......#####",
            "#####...#####",
            "#####.....G##",
            "#####..######",
            "#####.#######",
            "#############",
            "#############",
        ),
        (LeafSpec("A", (5, 5), 2, (0, -1)),),
        12,
    ),
    LevelSpec(
        "The empty destination is not enough",
        (
            "#############",
            "#############",
            "#############",
            "######......#",
            "######S....##",
            "#######...###",
            "#######G..###",
            "#######....##",
            "#########...#",
            "#############",
            "#############",
        ),
        (LeafSpec("A", (9, 3), 2, (0, 1)), LeafSpec("B", (9, 8), 2, (0, -1)), LeafSpec("C", (7, 7), 3, (0, -1))),
        19,
    ),
    LevelSpec(
        "Cross before you close it",
        (
            "#############",
            "#############",
            "###S.....####",
            "#####....####",
            "######.....##",
            "#####......##",
            "#####....G.##",
            "#####......##",
            "########...##",
            "#############",
            "#############",
        ),
        (LeafSpec("A", (7, 3), 2, (0, 1)), LeafSpec("B", (6, 6), 2, (1, 0)), LeafSpec("C", (9, 7), 3, (0, -1))),
        28,
    ),
    LevelSpec(
        "Move away, then restore",
        (
            "###############",
            "###############",
            "###############",
            "###############",
            "#####...#######",
            "#####.......###",
            "###......##...#",
            "###.#.G.....#.#",
            "###....#....#.#",
            "#####..#....#.#",
            "#####...##....#",
            "##########.####",
            "##########...S#",
            "###############",
        ),
        (
            LeafSpec("A", (8, 8), 2, (1, 0)),
            LeafSpec("B", (11, 9), 3, (0, -1)),
            LeafSpec("C", (6, 5), 3, (0, 1)),
            LeafSpec("D", (5, 10), 2, (0, -1)),
        ),
        40,
    ),
    LevelSpec(
        "Open one corridor, seal another",
        (
            "###############",
            "###############",
            "###############",
            "########...####",
            "#####......####",
            "#####.......###",
            "###...........#",
            "###.#.G.....#.#",
            "###....#....#.#",
            "#####..#....#.#",
            "#####...##....#",
            "##########.####",
            "##########...S#",
            "###############",
        ),
        (
            LeafSpec("A", (8, 8), 2, (1, 0)),
            LeafSpec("B", (11, 9), 3, (0, -1)),
            LeafSpec("C", (6, 5), 3, (0, 1)),
            LeafSpec("D", (5, 10), 2, (0, -1)),
            LeafSpec("E", (9, 6), 2, (0, -1)),
        ),
        46,
    ),
    LevelSpec(
        "The cellar hinge",
        (
            "###############",
            "###############",
            "###############",
            "########...####",
            "#####......####",
            "#####.......###",
            "###...........#",
            "###.#.G.....#.#",
            "###....#....#.#",
            "####...#....#.#",
            "####....##....#",
            "####.....#.####",
            "#######......S#",
            "###############",
        ),
        (
            LeafSpec("A", (8, 8), 2, (1, 0)),
            LeafSpec("B", (11, 9), 3, (0, -1)),
            LeafSpec("C", (6, 5), 3, (0, 1)),
            LeafSpec("D", (5, 10), 2, (0, -1)),
            LeafSpec("E", (9, 6), 2, (0, -1)),
            LeafSpec("F", (7, 12), 2, (0, -1)),
        ),
        60,
    ),
)


def _parse_rows(
    rows: tuple[str, ...],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    floor: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    start = (-1, -1)
    goal = (-1, -1)
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char == "#":
                walls.add((x, y))
            else:
                floor.add((x, y))
                if char == "S":
                    start = (x, y)
                elif char == "G":
                    goal = (x, y)
    return floor, walls, start, goal


def _build_level(spec: LevelSpec) -> Level:
    floor, walls, start, goal = _parse_rows(spec.rows)
    return Level(
        sprites=[],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={
            "rows": spec.rows,
            "floor": floor,
            "walls": walls,
            "start": start,
            "goal": goal,
            "leaves": spec.leaves,
            "budget": spec.baseline_actions * 6,
        },
    )


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


class HingeHallDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: HingeHall | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BACKGROUND
        self._draw_board(frame, game)
        self._draw_step_bar(frame, game)
        return frame

    def _draw_board(self, frame: np.ndarray, game: HingeHall) -> None:
        for x, y in game.wall_cells:
            self._fill_cell(frame, game, (x, y), COLOR_WALL)
        for x, y in game.floor_cells:
            self._fill_cell(frame, game, (x, y), COLOR_FLOOR)

        self._draw_goal(frame, game)
        for leaf in game.leaf_specs:
            for direction in game._guide_directions(leaf):
                self._draw_union_outline(frame, game, game._sweep_mask(leaf, direction), COLOR_GUIDE)
                self._draw_push_arrow(frame, game, leaf, direction)

        animation = game.animation_pose
        animated_id = animation.get("leaf_id") if animation else None
        for leaf in game.leaf_specs:
            if leaf.leaf_id == animated_id:
                self._draw_rotating_leaf(frame, game, leaf, float(animation["angle"]))
            else:
                self._draw_leaf(frame, game, leaf, game.leaf_orientations[leaf.leaf_id])

        # Goal corners remain legible even under a movable leaf.
        self._draw_goal_corners(frame, game)

        player_cell = game.player_position
        if animation and float(animation.get("angle", 90.0)) < 90.0:
            player_cell = tuple(animation["old_player"])
        self._draw_player(frame, game, player_cell)
        self._draw_feedback(frame, game)

    def _fill_cell(self, frame: np.ndarray, game: HingeHall, cell: tuple[int, int], color: int) -> None:
        x0, y0 = game._cell_pixel(cell)
        frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = color

    def _draw_goal(self, frame: np.ndarray, game: HingeHall) -> None:
        x0, y0 = game._cell_pixel(game.goal_position)
        frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = COLOR_GOAL
        frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = COLOR_FLOOR

    def _draw_goal_corners(self, frame: np.ndarray, game: HingeHall) -> None:
        x0, y0 = game._cell_pixel(game.goal_position)
        for dx, dy in ((0, 0), (3, 0), (0, 3), (3, 3)):
            frame[y0 + dy, x0 + dx] = COLOR_GOAL

    def _draw_union_outline(self, frame: np.ndarray, game: HingeHall, cells: set[tuple[int, int]], color: int) -> None:
        pixels: set[tuple[int, int]] = set()
        for cell in cells:
            x0, y0 = game._cell_pixel(cell)
            for py in range(y0, y0 + CELL_SIZE):
                for px in range(x0, x0 + CELL_SIZE):
                    pixels.add((px, py))
        for px, py in pixels:
            if any((px + dx, py + dy) not in pixels for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                if 0 <= px < VIEW_SIZE and 0 <= py < VIEW_SIZE:
                    frame[py, px] = color

    def _draw_leaf(self, frame: np.ndarray, game: HingeHall, leaf: LeafSpec, direction: tuple[int, int]) -> None:
        for cell in game._leaf_cells(leaf, direction):
            self._fill_cell(frame, game, cell, COLOR_LEAF)
        self._draw_hinge(frame, game, leaf.hinge)
        self._draw_handle(frame, game, game._free_end(leaf, direction), direction)

    def _draw_hinge(self, frame: np.ndarray, game: HingeHall, cell: tuple[int, int]) -> None:
        x0, y0 = game._cell_pixel(cell)
        frame[y0 : y0 + 4, x0 : x0 + 4] = np.array(
            [
                [-1, COLOR_BRASS, COLOR_BRASS, -1],
                [COLOR_BRASS, COLOR_ORANGE, COLOR_ORANGE, COLOR_BRASS],
                [COLOR_BRASS, COLOR_ORANGE, COLOR_ORANGE, COLOR_BRASS],
                [-1, COLOR_BRASS, COLOR_BRASS, -1],
            ],
            dtype=np.int8,
        )
        for dx, dy in ((0, 0), (3, 0), (0, 3), (3, 3)):
            if frame[y0 + dy, x0 + dx] < 0:
                frame[y0 + dy, x0 + dx] = COLOR_FLOOR

    def _draw_handle(
        self, frame: np.ndarray, game: HingeHall, cell: tuple[int, int], direction: tuple[int, int]
    ) -> None:
        x0, y0 = game._cell_pixel(cell)
        frame[y0 : y0 + 4, x0 : x0 + 4] = COLOR_BRASS
        frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = COLOR_ORANGE
        dx, dy = direction
        frame[y0 + 1 + dy, x0 + 1 + dx] = COLOR_WHITE
        frame[y0 + 2 + dy, x0 + 2 + dx] = COLOR_WHITE

    def _draw_push_arrow(self, frame: np.ndarray, game: HingeHall, leaf: LeafSpec, direction: tuple[int, int]) -> None:
        x0, y0 = game._cell_pixel(game._free_end(leaf))
        dx, dy = direction
        cx, cy = x0 + 1, y0 + 1
        if dx:
            tip_x = x0 + (3 if dx > 0 else 0)
            frame[cy : cy + 2, tip_x] = COLOR_GUIDE
            frame[cy + dy, max(x0, min(x0 + 3, tip_x - dx))] = COLOR_GUIDE
        else:
            tip_y = y0 + (3 if dy > 0 else 0)
            frame[tip_y, cx : cx + 2] = COLOR_GUIDE
            frame[max(y0, min(y0 + 3, tip_y - dy)), cx + dx] = COLOR_GUIDE

    def _draw_rotating_leaf(self, frame: np.ndarray, game: HingeHall, leaf: LeafSpec, angle: float) -> None:
        old_direction = tuple(game.animation_pose["old_direction"])
        new_direction = tuple(game.animation_pose["new_direction"])
        start = DIRECTION_ANGLE[old_direction]
        end = DIRECTION_ANGLE[new_direction]
        delta = ((end - start + 180.0) % 360.0) - 180.0
        theta = (start + delta * angle / 90.0) * pi / 180.0
        vx, vy = cos(theta), sin(theta)
        nx, ny = -vy, vx
        hx, hy = game._cell_pixel(leaf.hinge)
        cx, cy = hx + 1.5, hy + 1.5
        length_px = leaf.length * CELL_SIZE
        for py in range(max(0, int(cy - length_px - 3)), min(VIEW_SIZE, int(cy + length_px + 4))):
            for px in range(max(0, int(cx - length_px - 3)), min(VIEW_SIZE, int(cx + length_px + 4))):
                rx, ry = px + 0.5 - cx, py + 0.5 - cy
                along = rx * vx + ry * vy
                across = rx * nx + ry * ny
                if 2.0 <= along <= length_px + 2.0 and abs(across) <= 1.95:
                    frame[py, px] = COLOR_LEAF
        self._draw_hinge(frame, game, leaf.hinge)
        handle_x = round(cx + vx * (length_px + 1.0))
        handle_y = round(cy + vy * (length_px + 1.0))
        for py in range(handle_y - 1, handle_y + 2):
            for px in range(handle_x - 1, handle_x + 2):
                if 0 <= px < VIEW_SIZE and 0 <= py < VIEW_SIZE:
                    frame[py, px] = COLOR_BRASS
        if 0 <= handle_x < VIEW_SIZE and 0 <= handle_y < VIEW_SIZE:
            frame[handle_y, handle_x] = COLOR_ORANGE

    def _draw_player(self, frame: np.ndarray, game: HingeHall, cell: tuple[int, int]) -> None:
        x0, y0 = game._cell_pixel(cell)
        frame[y0 : y0 + 4, x0 : x0 + 4] = np.array(
            [
                [-1, COLOR_PLAYER, COLOR_PLAYER, -1],
                [COLOR_PLAYER, COLOR_PLAYER, COLOR_WHITE, COLOR_PLAYER],
                [COLOR_PLAYER, COLOR_WHITE, COLOR_PLAYER, COLOR_PLAYER],
                [-1, COLOR_PLAYER, COLOR_PLAYER, -1],
            ],
            dtype=np.int8,
        )
        for dx, dy in ((0, 0), (3, 0), (0, 3), (3, 3)):
            if frame[y0 + dy, x0 + dx] < 0:
                frame[y0 + dy, x0 + dx] = COLOR_FLOOR

    def _draw_feedback(self, frame: np.ndarray, game: HingeHall) -> None:
        feedback = game.feedback
        if not feedback:
            return
        kind = feedback["kind"]
        phase = int(feedback.get("phase", 0))
        if kind == "push_blocked":
            leaf = game.leaf_by_id[str(feedback["leaf_id"])]
            target = tuple(feedback["attempted_orientation"])
            self._draw_dashed_outline(frame, game, game._leaf_cells(leaf, target), COLOR_FAILURE, phase)
            for cell in feedback["blocking_coordinates"]:
                self._outline_cell(frame, game, tuple(cell), COLOR_ORANGE)
        elif kind == "collision":
            self._outline_cell(frame, game, tuple(feedback["cell"]), COLOR_FAILURE)
        elif kind == "space":
            self._outline_cell(frame, game, game.player_position, COLOR_LEAF)
        elif kind == "undo":
            self._outline_cell(frame, game, game.player_position, COLOR_GUIDE)
        elif kind == "complete":
            self._outline_cell(frame, game, game.goal_position, COLOR_GOAL)
            if phase % 2 == 0:
                x0, y0 = game._cell_pixel(game.goal_position)
                frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = COLOR_WHITE

    def _outline_cell(self, frame: np.ndarray, game: HingeHall, cell: tuple[int, int], color: int) -> None:
        x0, y0 = game._cell_pixel(cell)
        if x0 < 0 or y0 < 0 or x0 + 3 >= VIEW_SIZE or y0 + 3 >= VIEW_SIZE:
            return
        frame[y0, x0 : x0 + 4] = color
        frame[y0 + 3, x0 : x0 + 4] = color
        frame[y0 : y0 + 4, x0] = color
        frame[y0 : y0 + 4, x0 + 3] = color

    def _draw_dashed_outline(
        self, frame: np.ndarray, game: HingeHall, cells: tuple[tuple[int, int], ...], color: int, phase: int
    ) -> None:
        pixels: set[tuple[int, int]] = set()
        for cell in cells:
            x0, y0 = game._cell_pixel(cell)
            for py in range(y0, y0 + 4):
                for px in range(x0, x0 + 4):
                    pixels.add((px, py))
        for px, py in pixels:
            boundary = any((px + dx, py + dy) not in pixels for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if boundary and (px + py + phase) % 2 == 0 and 0 <= px < VIEW_SIZE and 0 <= py < VIEW_SIZE:
                frame[py, px] = color

    def _draw_step_bar(self, frame: np.ndarray, game: HingeHall) -> None:
        width = 52
        x0, y0 = 6, 1
        frame[y0 : y0 + 2, x0 : x0 + width] = COLOR_WALL
        filled = int(width * max(0, game.remaining_steps) / max(1, game.step_budget))
        if filled:
            frame[y0 : y0 + 2, x0 : x0 + filled] = COLOR_ORANGE


class HingeHall(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = HingeHallDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "hinge_hall-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._view]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UNDO],
            seed,
        )
        self._view.game = self

    def on_set_level(self, _level: Level) -> None:
        rows = tuple(self.current_level.get_data("rows"))
        self.board_width = len(rows[0])
        self.board_height = len(rows)
        self.board_origin = (
            (VIEW_SIZE - CELL_SIZE * self.board_width) // 2,
            (VIEW_SIZE - CELL_SIZE * self.board_height) // 2,
        )
        self.floor_cells = set(self.current_level.get_data("floor"))
        self.wall_cells = set(self.current_level.get_data("walls"))
        self.player_position = tuple(self.current_level.get_data("start"))
        self.goal_position = tuple(self.current_level.get_data("goal"))
        self.leaf_specs = tuple(self.current_level.get_data("leaves"))
        self.leaf_by_id = {leaf.leaf_id: leaf for leaf in self.leaf_specs}
        self.leaf_orientations = {leaf.leaf_id: leaf.orientation for leaf in self.leaf_specs}
        self.step_budget = int(self.current_level.get_data("budget"))
        self.remaining_steps = self.step_budget
        self.complete = False
        self.feedback: dict[str, object] | None = None
        self.animation_pose: dict[str, object] | None = None
        self._render_queue: list[dict[str, object]] = []
        self._finish_after_queue = False
        self.undo_history: list[tuple[tuple[int, int], dict[str, tuple[int, int]]]] = []

    def _cell_pixel(self, cell: tuple[int, int]) -> tuple[int, int]:
        return self.board_origin[0] + cell[0] * CELL_SIZE, self.board_origin[1] + cell[1] * CELL_SIZE

    def _leaf_cells(self, leaf: LeafSpec, direction: tuple[int, int] | None = None) -> tuple[tuple[int, int], ...]:
        dx, dy = direction if direction is not None else self.leaf_orientations[leaf.leaf_id]
        hx, hy = leaf.hinge
        return tuple((hx + step * dx, hy + step * dy) for step in range(1, leaf.length + 1))

    def _free_end(self, leaf: LeafSpec, direction: tuple[int, int] | None = None) -> tuple[int, int]:
        return self._leaf_cells(leaf, direction)[-1]

    def _sweep_mask(self, leaf: LeafSpec, direction: tuple[int, int]) -> set[tuple[int, int]]:
        old_dx, old_dy = self.leaf_orientations[leaf.leaf_id]
        new_dx, new_dy = direction
        hx, hy = leaf.hinge
        destination = {(hx + k * new_dx, hy + k * new_dy) for k in range(1, leaf.length + 1)}
        interior = {
            (hx + a * old_dx + b * new_dx, hy + a * old_dy + b * new_dy)
            for a in range(1, leaf.length)
            for b in range(1, leaf.length)
        }
        return destination | interior

    def _occupied_leaf_cells(self, *, exclude_id: str | None = None) -> dict[tuple[int, int], str]:
        occupied: dict[tuple[int, int], str] = {}
        for leaf in self.leaf_specs:
            if leaf.leaf_id == exclude_id:
                continue
            for cell in self._leaf_cells(leaf):
                occupied[cell] = leaf.leaf_id
        return occupied

    def _guide_directions(self, leaf: LeafSpec) -> tuple[tuple[int, int], ...]:
        dx, dy = self.leaf_orientations[leaf.leaf_id]
        candidates = ((-dy, dx), (dy, -dx))
        hinges = {other.hinge for other in self.leaf_specs if other.leaf_id != leaf.leaf_id}
        return tuple(
            direction
            for direction in candidates
            if all(cell in self.floor_cells and cell not in hinges for cell in self._sweep_mask(leaf, direction))
        )

    def _rotation_blockers(self, leaf: LeafSpec, direction: tuple[int, int]) -> set[tuple[int, int]]:
        mask = self._sweep_mask(leaf, direction)
        hinges = {other.hinge for other in self.leaf_specs if other.leaf_id != leaf.leaf_id}
        occupied = self._occupied_leaf_cells(exclude_id=leaf.leaf_id)
        return {cell for cell in mask if cell not in self.floor_cells or cell in hinges or cell in occupied}

    def _leaf_at_free_end(self, cell: tuple[int, int]) -> LeafSpec | None:
        for leaf in self.leaf_specs:
            if self._free_end(leaf) == cell:
                return leaf
        return None

    def _queue_rotation(
        self,
        leaf: LeafSpec,
        old_direction: tuple[int, int],
        new_direction: tuple[int, int],
        old_player: tuple[int, int],
    ) -> None:
        self._render_queue = [
            {
                "kind": "rotation",
                "leaf_id": leaf.leaf_id,
                "old_direction": old_direction,
                "new_direction": new_direction,
                "old_player": old_player,
                "angle": angle,
            }
            for angle in (0.0, 22.5, 45.0, 67.5, 90.0)
        ]

    def _queue_feedback(self, feedback: dict[str, object], frames: int) -> None:
        self._render_queue = [{**feedback, "phase": phase} for phase in range(frames)]

    def _queue_completion(self) -> None:
        completion = [{"kind": "complete", "phase": phase} for phase in range(8)]
        self._render_queue.extend(completion)
        self._finish_after_queue = True

    def _play_render_frame(self) -> None:
        pose = self._render_queue.pop(0)
        if pose["kind"] == "rotation":
            self.animation_pose = pose
            self.feedback = None
        else:
            self.animation_pose = None
            self.feedback = pose
        if self._render_queue:
            return
        if self._finish_after_queue:
            # Mark the submitted action complete before the deferred level swap so
            # its direction is never replayed against the new level.
            self.complete_action()
            self.next_level()
        else:
            self.complete_action()

    def _spend_action(self) -> bool:
        self.remaining_steps -= 1
        if self.remaining_steps > 0:
            return True
        self.feedback = None
        self.animation_pose = None
        self._render_queue.clear()
        self.lose()
        self.complete_action()
        return False

    def _capture_state(self) -> tuple[tuple[int, int], dict[str, tuple[int, int]]]:
        return self.player_position, dict(self.leaf_orientations)

    def _undo(self) -> None:
        if not self.undo_history:
            self._queue_feedback({"kind": "undo"}, 1)
            return
        self.player_position, orientations = self.undo_history.pop()
        self.leaf_orientations = dict(orientations)
        self.complete = False
        self.animation_pose = None
        self._finish_after_queue = False
        self._queue_feedback({"kind": "undo"}, 2)

    def _resolve_direction(self, direction: tuple[int, int]) -> bool:
        px, py = self.player_position
        dx, dy = direction
        target = (px + dx, py + dy)
        occupied = self._occupied_leaf_cells()
        hinges = {leaf.hinge for leaf in self.leaf_specs}

        if target in self.floor_cells and target not in occupied and target not in hinges:
            self.player_position = target
            if target == self.goal_position:
                self.complete = True
                self._queue_completion()
            return True

        leaf = self._leaf_at_free_end(target)
        if leaf is not None:
            old_direction = self.leaf_orientations[leaf.leaf_id]
            perpendicular = old_direction[0] * dx + old_direction[1] * dy == 0
            if perpendicular:
                blockers = self._rotation_blockers(leaf, direction)
                if not blockers:
                    old_player = self.player_position
                    self.leaf_orientations[leaf.leaf_id] = direction
                    self.player_position = target
                    self._queue_rotation(leaf, old_direction, direction, old_player)
                    if target == self.goal_position:
                        self.complete = True
                        self._queue_completion()
                    return True
                self._queue_feedback(
                    {
                        "kind": "push_blocked",
                        "leaf_id": leaf.leaf_id,
                        "attempted_orientation": direction,
                        "blocking_coordinates": tuple(sorted(blockers)),
                    },
                    4,
                )
                return False

        self._queue_feedback({"kind": "collision", "cell": target}, 2)
        return False

    def step(self) -> None:
        if self._render_queue:
            self._play_render_frame()
            return
        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self.complete_action()
            return
        if self.complete:
            self.complete_action()
            return
        if not self._spend_action():
            return

        self.feedback = None
        self.animation_pose = None
        if action == ACTION_UNDO:
            self._undo()
        elif action in MOVE_DELTAS:
            before = self._capture_state()
            if self._resolve_direction(MOVE_DELTAS[action]):
                self.undo_history.append(before)
        else:
            self.complete_action()
            return

        # Activate the first queued pose during the resolving tick. ArcEngine
        # renders after step(), so this keeps a five-pose rotation at exactly
        # five frames instead of briefly flashing the committed final state.
        if self._render_queue:
            self._play_render_frame()
        else:
            self.complete_action()
