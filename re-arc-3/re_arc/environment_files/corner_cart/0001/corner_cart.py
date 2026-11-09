from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

VIEW_SIZE = 64

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
CLOCKWISE = 5
CLICK = 6
UNDO = 7
MOVE_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

OUTSIDE = 0
FLOOR = 1
SPENT = 2
UNCHARGED = 3
WALL = 4
SHADOW = 5
MAGENTA = 6
INVALID = 8
ARM = 9
CYAN = 10
YELLOW = 11
EARLY = 13
GREEN = 14
POST = 15

FRONT = ((-1, 0), (0, 1), (1, 0), (0, -1))
REAR = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    start_orientation: int
    dock_orientation: int
    tile_size: int
    origin: tuple[int, int]
    baseline: int

    @property
    def step_budget(self) -> int:
        return self.baseline * 6


LEVEL_SPECS = (
    LevelSpec(
        "The Protruding Corner",
        (
            "..........",
            "..........",
            "..........",
            "..........",
            "...D......",
            "..#.......",
            "..#.......",
            "..S.......",
            "....##....",
            "..........",
        ),
        1,
        3,
        5,
        (7, 7),
        8,
    ),
    LevelSpec(
        "The Brass Button",
        (
            "..........",
            "..........",
            "....###...",
            "..P..##...",
            "...S.##D..",
            "....##....",
            "....##....",
            "..........",
            "..........",
            "..........",
        ),
        0,
        0,
        5,
        (7, 7),
        14,
    ),
    LevelSpec(
        "Front Makes Contact",
        (
            "..........",
            ".####.....",
            ".####..D..",
            ".####b....",
            "..........",
            "..........",
            "...##.S...",
            "...##.....",
            "..a##.....",
            "..........",
        ),
        3,
        2,
        5,
        (7, 7),
        20,
    ),
    LevelSpec(
        "Wrong Side of the Post",
        (
            "...........",
            ".........P.",
            "...........",
            "....D.##...",
            ".....S##.b.",
            "....###..P.",
            "....###....",
            "....###....",
            "...........",
            "....a......",
            "...........",
        ),
        3,
        3,
        5,
        (4, 4),
        34,
    ),
    LevelSpec(
        "Preserve the Bay",
        (
            ".............",
            ".............",
            ".........##..",
            ".........##..",
            ".......a.....",
            ".....###.....",
            "...X.###.....",
            "P....###.....",
            ".............",
            "..##b........",
            "..##.........",
            ".............",
            ".............",
        ),
        0,
        1,
        4,
        (6, 6),
        45,
    ),
    LevelSpec(
        "Three-Way Yard",
        (
            ".............",
            "..##.........",
            "..##.....##..",
            ".........##..",
            "....c...a....",
            ".....###.....",
            "...X.###.....",
            "P....###.....",
            "......b......",
            "..##.....##..",
            "..##.....##..",
            ".............",
            ".............",
        ),
        0,
        1,
        4,
        (6, 6),
        51,
    ),
    LevelSpec(
        "Full Circuit",
        (
            "..............",
            "..###.....c...",
            "..###.....##..",
            "...b......##..",
            ".....###..##..",
            ".....###......",
            ".....###.....P",
            "....X.........",
            "........###...",
            "........###...",
            "..##....###...",
            "..##a.........",
            "..##..........",
            "..............",
        ),
        2,
        1,
        4,
        (4, 4),
        63,
    ),
)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _pose_cells(elbow: tuple[int, int], orientation: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    er, ec = elbow
    fr, fc = FRONT[orientation]
    rr, rc = REAR[orientation]
    body = {(er, ec), (er + fr, ec + fc), (er + rr, ec + rc)}
    return body, (er + fr + rr, ec + fc + rc)


def _parse_level(spec: LevelSpec) -> dict[str, object]:
    height = len(spec.rows)
    width = len(spec.rows[0])
    if any(len(row) != width for row in spec.rows):
        raise ValueError(f"{spec.name}: rows are not rectangular")

    walls: set[tuple[int, int]] = set()
    posts: set[tuple[int, int]] = set()
    waypoints: list[tuple[int, int]] = []
    start: tuple[int, int] | None = None
    dock: tuple[int, int] | None = None
    waypoint_symbols: dict[str, tuple[int, int]] = {}
    for r, row in enumerate(spec.rows, 1):
        for c, symbol in enumerate(row, 1):
            cell = (r, c)
            if symbol == "#":
                walls.add(cell)
            elif symbol == "P":
                posts.add(cell)
            elif symbol == "S":
                start = cell
            elif symbol == "D":
                dock = cell
            elif symbol == "X":
                start = dock = cell
            elif symbol in "abc":
                waypoint_symbols[symbol] = cell
            elif symbol != ".":
                raise ValueError(f"{spec.name}: unknown layout symbol {symbol!r}")
    waypoints.extend(waypoint_symbols[key] for key in sorted(waypoint_symbols))
    if start is None or dock is None:
        raise ValueError(f"{spec.name}: layout needs a start and dock")
    if walls & posts or any(cell in walls | posts for cell in waypoints):
        raise ValueError(f"{spec.name}: overlapping authored structures")
    if len(waypoints) != len(set(waypoints)):
        raise ValueError(f"{spec.name}: duplicate waypoint")

    data = {
        "spec": spec,
        "height": height,
        "width": width,
        "walls": walls,
        "posts": posts,
        "waypoints": tuple(waypoints),
        "start": start,
        "dock": dock,
    }
    for pose, orientation, label in ((start, spec.start_orientation, "start"), (dock, spec.dock_orientation, "dock")):
        body, notch = _pose_cells(pose, orientation)
        envelope = body | {notch}
        if not all(1 <= r <= height and 1 <= c <= width for r, c in envelope):
            raise ValueError(f"{spec.name}: {label} envelope leaves board")
        if envelope & walls or body & posts:
            raise ValueError(f"{spec.name}: illegal {label} pose")
    return data


class CornerCartView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: CornerCart | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = OUTSIDE
        self._draw_board(frame, game)
        self._draw_dock_fill(frame, game)
        self._draw_plates(frame, game)
        self._draw_posts(frame, game)
        self._draw_cart_or_animation(frame, game)
        self._draw_dock_brackets(frame, game)
        self._draw_feedback(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _rect(self, game: CornerCart, cell: tuple[int, int]) -> tuple[int, int, int]:
        r, c = cell
        return (game.origin_x + (c - 1) * game.tile_size, game.origin_y + (r - 1) * game.tile_size, game.tile_size)

    def _draw_board(self, frame: np.ndarray, game: CornerCart) -> None:
        x0, y0 = game.origin_x, game.origin_y
        frame[y0 : y0 + game.height * game.tile_size, x0 : x0 + game.width * game.tile_size] = FLOOR
        for cell in game.walls:
            x, y, size = self._rect(game, cell)
            frame[y : y + size, x : x + size] = WALL

    def _dock_colors(self, game: CornerCart) -> dict[tuple[int, int], int]:
        body, _ = _pose_cells(game.dock_elbow, game.dock_orientation)
        fr, fc = FRONT[game.dock_orientation]
        rr, rc = REAR[game.dock_orientation]
        elbow = game.dock_elbow
        front = (elbow[0] + fr, elbow[1] + fc)
        rear = (elbow[0] + rr, elbow[1] + rc)
        count = game.lit_mask.bit_count()
        if not game.waypoints:
            return {cell: GREEN for cell in body}
        if len(game.waypoints) == 2:
            return {elbow: GREEN, front: GREEN if count >= 1 else UNCHARGED, rear: GREEN if count >= 2 else UNCHARGED}
        return {
            elbow: GREEN if count >= 1 else UNCHARGED,
            front: GREEN if count >= 2 else UNCHARGED,
            rear: GREEN if count >= 3 else UNCHARGED,
        }

    def _draw_dock_fill(self, frame: np.ndarray, game: CornerCart) -> None:
        for cell, color in self._dock_colors(game).items():
            x, y, size = self._rect(game, cell)
            frame[y + 1 : y + size - 1, x + 1 : x + size - 1] = color

    def _draw_plates(self, frame: np.ndarray, game: CornerCart) -> None:
        for index, cell in enumerate(game.waypoints):
            x, y, size = self._rect(game, cell)
            active = bool(game.lit_mask & (1 << index))
            color = GREEN if active else CYAN
            center = YELLOW if active else MAGENTA
            inset = 1
            frame[y + inset : y + size - inset, x + inset : x + size - inset] = color
            if size == 5:
                frame[y + 2, x + 2] = center
            else:
                frame[y + 1, x + 1] = center

    def _draw_posts(self, frame: np.ndarray, game: CornerCart) -> None:
        for cell in game.posts:
            x, y, size = self._rect(game, cell)
            if size == 5:
                frame[y + 1 : y + 4, x + 1 : x + 4] = POST
                frame[y, x + 2] = frame[y + 4, x + 2] = POST
                frame[y + 2, x] = frame[y + 2, x + 4] = POST
            else:
                frame[y + 1 : y + 3, x : x + 4] = POST
                frame[y : y + 4, x + 1 : x + 3] = POST

    def _fill_cart_cell(self, frame: np.ndarray, game: CornerCart, cell: tuple[int, int], color: int) -> None:
        x, y, size = self._rect(game, cell)
        frame[y : y + size, x : x + size] = color

    def _draw_notch(self, frame: np.ndarray, game: CornerCart, cell: tuple[int, int], color: int = CYAN) -> None:
        x, y, size = self._rect(game, cell)
        frame[y, x : x + 2] = color
        frame[y, x + size - 2 : x + size] = color
        frame[y + size - 1, x : x + 2] = color
        frame[y + size - 1, x + size - 2 : x + size] = color
        frame[y : y + 2, x] = color
        frame[y + size - 2 : y + size, x] = color
        frame[y : y + 2, x + size - 1] = color
        frame[y + size - 2 : y + size, x + size - 1] = color

    def _draw_cart(self, frame: np.ndarray, game: CornerCart, elbow: tuple[int, int], orientation: int) -> None:
        er, ec = elbow
        fr, fc = FRONT[orientation]
        rr, rc = REAR[orientation]
        front = (er + fr, ec + fc)
        rear = (er + rr, ec + rc)
        _, notch = _pose_cells(elbow, orientation)
        self._fill_cart_cell(frame, game, front, ARM)
        self._fill_cart_cell(frame, game, rear, ARM)
        self._fill_cart_cell(frame, game, elbow, YELLOW)
        x, y, size = self._rect(game, elbow)
        if size == 5:
            frame[y + 2, x + 2] = SHADOW
        else:
            frame[y + 1 : y + 3, x + 1 : x + 3] = SHADOW
        fx, fy, size = self._rect(game, front)
        if fr < 0:
            frame[fy, fx : fx + size] = MAGENTA
        elif fr > 0:
            frame[fy + size - 1, fx : fx + size] = MAGENTA
        elif fc < 0:
            frame[fy : fy + size, fx] = MAGENTA
        else:
            frame[fy : fy + size, fx + size - 1] = MAGENTA
        self._draw_notch(frame, game, notch)

    def _draw_pivot(self, frame: np.ndarray, game: CornerCart) -> None:
        elbow = game.elbow
        orientations = (game.orientation, game.pending_orientation)
        self._fill_cart_cell(frame, game, elbow, YELLOW)
        x, y, size = self._rect(game, elbow)
        frame[y + size // 2 - 1 : y + size // 2 + 1, x + size // 2 - 1 : x + size // 2 + 1] = SHADOW
        for orientation in orientations:
            for dr, dc in (FRONT[orientation], REAR[orientation]):
                cell = (elbow[0] + dr, elbow[1] + dc)
                cx, cy, tile = self._rect(game, cell)
                half = max(2, tile // 2)
                if dr < 0:
                    frame[cy + tile - half : cy + tile, cx : cx + tile] = ARM
                elif dr > 0:
                    frame[cy : cy + half, cx : cx + tile] = ARM
                elif dc < 0:
                    frame[cy : cy + tile, cx + tile - half : cx + tile] = ARM
                else:
                    frame[cy : cy + tile, cx : cx + half] = ARM

    def _outline_cell(self, frame: np.ndarray, game: CornerCart, cell: tuple[int, int], color: int) -> None:
        x, y, size = self._rect(game, cell)
        xa, xb = max(0, x), min(VIEW_SIZE, x + size)
        ya, yb = max(0, y), min(VIEW_SIZE, y + size)
        if xa >= xb or ya >= yb:
            return
        frame[ya, xa:xb] = frame[yb - 1, xa:xb] = color
        frame[ya:yb, xa] = frame[ya:yb, xb - 1] = color

    def _draw_cart_or_animation(self, frame: np.ndarray, game: CornerCart) -> None:
        if game.visual_mode == "pivot":
            self._draw_pivot(frame, game)
            return
        self._draw_cart(frame, game, game.elbow, game.orientation)
        if game.visual_mode in {"preview_legal", "preview_invalid"}:
            color = CYAN if game.visual_mode == "preview_legal" else INVALID
            body, notch = _pose_cells(game.pending_elbow, game.pending_orientation)
            for cell in body | {notch}:
                self._outline_cell(frame, game, cell, color)

    def _draw_dock_brackets(self, frame: np.ndarray, game: CornerCart) -> None:
        colors = self._dock_colors(game)
        if game.visual_mode == "early":
            colors = {cell: (GREEN if color == GREEN else EARLY) for cell, color in colors.items()}
        for cell, color in colors.items():
            x, y, size = self._rect(game, cell)
            bracket = GREEN if color == GREEN else color
            frame[y, x : x + 2] = frame[y, x + size - 2 : x + size] = bracket
            frame[y + size - 1, x : x + 2] = frame[y + size - 1, x + size - 2 : x + size] = bracket
        er, ec = game.dock_elbow
        fr, fc = FRONT[game.dock_orientation]
        front = (er + fr, ec + fc)
        x, y, size = self._rect(game, front)
        if fr < 0:
            frame[y, x : x + size] = MAGENTA
        elif fr > 0:
            frame[y + size - 1, x : x + size] = MAGENTA
        elif fc < 0:
            frame[y : y + size, x] = MAGENTA
        else:
            frame[y : y + size, x + size - 1] = MAGENTA
        x, y, size = self._rect(game, game.dock_elbow)
        frame[y + size // 2, x + size // 2] = YELLOW

    def _draw_feedback(self, frame: np.ndarray, game: CornerCart) -> None:
        if game.visual_mode in {"preview_invalid", "invalid_move"}:
            for cell in game.pending_collisions:
                self._outline_cell(frame, game, cell, INVALID)
                x, y, size = self._rect(game, cell)
                if 0 <= x + 1 < VIEW_SIZE and 0 <= y + 1 < VIEW_SIZE:
                    frame[
                        max(0, y + 1) : min(VIEW_SIZE, y + size - 1), max(0, x + 1) : min(VIEW_SIZE, x + size - 1)
                    ] = INVALID
        if game.visual_mode == "invalid_click" and game.invalid_click_cell is not None:
            self._outline_cell(frame, game, game.invalid_click_cell, INVALID)
        if game.visual_mode == "post_fit":
            _, notch = _pose_cells(game.elbow, game.orientation)
            self._draw_notch(frame, game, notch, YELLOW)
        if game.visual_mode == "plate":
            for index, cell in enumerate(game.waypoints):
                if game.just_lit_mask & (1 << index):
                    self._outline_cell(frame, game, cell, GREEN)
        if game.visual_mode == "success_on":
            body, notch = _pose_cells(game.elbow, game.orientation)
            for cell in body | {notch}:
                self._outline_cell(frame, game, cell, GREEN)
        if game.visual_mode == "failure_wash":
            x, y = game.origin_x, game.origin_y
            board = frame[y : y + game.height * game.tile_size, x : x + game.width * game.tile_size]
            board[board == FLOOR] = SPENT

    def _draw_budget(self, frame: np.ndarray, game: CornerCart) -> None:
        top, bottom = 4, 60
        frame[top:bottom, 62] = SPENT
        remaining = max(0, game.step_budget - game.step_count)
        filled = round((bottom - top) * remaining / game.step_budget)
        if filled:
            frame[bottom - filled : bottom, 62] = CYAN


def _build_level(spec: LevelSpec) -> Level:
    data = _parse_level(spec)
    canvas = Sprite(np.full((VIEW_SIZE, VIEW_SIZE), OUTSIDE, dtype=np.int8), layer=-50, collidable=False)
    return Level([canvas], (VIEW_SIZE, VIEW_SIZE), data=data, name=spec.name)


class CornerCart(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = CornerCartView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "corner_cart",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, OUTSIDE, OUTSIDE, [self._view]),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, CLOCKWISE, CLICK, UNDO],
            seed,
        )
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.height = int(level.get_data("height"))
        self.width = int(level.get_data("width"))
        self.walls = set(level.get_data("walls"))
        self.posts = set(level.get_data("posts"))
        self.waypoints = tuple(level.get_data("waypoints"))
        self.elbow = tuple(level.get_data("start"))
        self.orientation = spec.start_orientation
        self.dock_elbow = tuple(level.get_data("dock"))
        self.dock_orientation = spec.dock_orientation
        self.tile_size = spec.tile_size
        self.origin_x, self.origin_y = spec.origin
        self.step_budget = spec.step_budget
        self.step_count = 0
        self.lit_mask = 0
        self._terminal_state = "RUNNING"
        self.visual_mode = "normal"
        self.pending_elbow = self.elbow
        self.pending_orientation = self.orientation
        self.pending_collisions: set[tuple[int, int]] = set()
        self.pending_legal = False
        self.invalid_click_cell: tuple[int, int] | None = None
        self.just_lit_mask = 0
        self._animation: list[str] = []
        self.undo_history: list[tuple[tuple[int, int], int, int]] = []

    def _snapshot(self) -> tuple[tuple[int, int], int, int]:
        return self.elbow, self.orientation, self.lit_mask

    def _restore(self, state: tuple[tuple[int, int], int, int]) -> None:
        self.elbow, self.orientation, self.lit_mask = state
        self.pending_elbow = self.elbow
        self.pending_orientation = self.orientation
        self.pending_collisions = set()
        self.pending_legal = False
        self.invalid_click_cell = None
        self.just_lit_mask = 0
        self.visual_mode = "normal"

    def _collision_cells(self, elbow: tuple[int, int], orientation: int) -> set[tuple[int, int]]:
        body, notch = _pose_cells(elbow, orientation)
        envelope = body | {notch}
        outside = {(r, c) for r, c in envelope if not (1 <= r <= self.height and 1 <= c <= self.width)}
        return outside | (envelope & self.walls) | (body & self.posts)

    def _click_candidate(self) -> tuple[tuple[int, int], int] | None:
        display_x = int(self.action.data.get("x", 0))
        display_y = int(self.action.data.get("y", 0))
        grid = self.camera.display_to_grid(display_x, display_y)
        if grid is None:
            return None
        px, py = grid
        col = (px - self.origin_x) // self.tile_size + 1
        row = (py - self.origin_y) // self.tile_size + 1
        self.invalid_click_cell = (row, col)
        if (row, col) != self.elbow:
            return None
        return self.elbow, (self.orientation + 3) % 4

    def _candidate(self, action: int) -> tuple[tuple[int, int], int] | None:
        if action in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[action]
            return (self.elbow[0] + dr, self.elbow[1] + dc), self.orientation
        if action == CLOCKWISE:
            return self.elbow, (self.orientation + 1) % 4
        if action == CLICK:
            return self._click_candidate()
        return None

    def _commit_pending(self) -> None:
        if not self.pending_legal:
            return
        self.elbow = self.pending_elbow
        self.orientation = self.pending_orientation
        front_delta = FRONT[self.orientation]
        front = (self.elbow[0] + front_delta[0], self.elbow[1] + front_delta[1])
        old_mask = self.lit_mask
        for index, waypoint in enumerate(self.waypoints):
            if front == waypoint:
                self.lit_mask |= 1 << index
        self.just_lit_mask = self.lit_mask ^ old_mask

    def _is_success(self) -> bool:
        all_mask = (1 << len(self.waypoints)) - 1
        return self.elbow == self.dock_elbow and self.orientation == self.dock_orientation and self.lit_mask == all_mask

    def _post_action_feedback(self, *, committed: bool = True) -> None:
        if self._is_success():
            self._terminal_state = "SUCCESS"
            self.visual_mode = "success_on"
            self._animation = ["success_off", "success_on", "success_off", "success_on", "finish_success"]
            return
        if self.step_count >= self.step_budget:
            self._terminal_state = "FAILURE"
            self.visual_mode = "failure_wash"
            self._animation = ["finish_failure"]
            return
        if not committed:
            self.visual_mode = "normal"
            self.complete_action()
            return
        exact_dock = self.elbow == self.dock_elbow and self.orientation == self.dock_orientation
        _, notch = _pose_cells(self.elbow, self.orientation)
        if exact_dock:
            self.visual_mode = "early"
            self._animation = ["finish_normal"]
        elif self.just_lit_mask:
            self.visual_mode = "plate"
            self._animation = ["finish_normal"]
        elif notch in self.posts:
            self.visual_mode = "post_fit"
            self._animation = ["finish_normal"]
        else:
            self.visual_mode = "normal"
            self.complete_action()

    def _play_animation(self) -> None:
        stage = self._animation.pop(0)
        if stage == "pivot":
            self.visual_mode = "pivot"
        elif stage == "resolution":
            self._commit_pending()
            self.visual_mode = "normal"
            self._post_action_feedback(committed=self.pending_legal)
        elif stage in {"success_on", "success_off"}:
            self.visual_mode = stage
        elif stage == "finish_success":
            self.visual_mode = "normal"
            self.next_level()
            self.complete_action()
        elif stage == "finish_failure":
            self.visual_mode = "normal"
            self.lose()
            self.complete_action()
        elif stage == "finish_normal":
            self.visual_mode = "normal"
            self.complete_action()

    def step(self) -> None:
        if self._animation:
            self._play_animation()
            return
        if _action_id(self.action.id) == int(GameAction.RESET.value):
            self.visual_mode = "normal"
            self.complete_action()
            return
        if self._terminal_state != "RUNNING":
            self.complete_action()
            return

        self.visual_mode = "normal"
        self.invalid_click_cell = None
        self.just_lit_mask = 0
        self.step_count += 1
        action = _action_id(self.action.id)
        if action == UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self._post_action_feedback(committed=False)
            return
        candidate = self._candidate(action)
        if candidate is None:
            self.visual_mode = "invalid_click" if action == CLICK else "normal"
            if self.step_count >= self.step_budget:
                self._terminal_state = "FAILURE"
                self.visual_mode = "failure_wash"
                self._animation = ["finish_failure"]
            else:
                self.complete_action()
            return

        self.pending_elbow, self.pending_orientation = candidate
        self.pending_collisions = self._collision_cells(*candidate)
        self.pending_legal = not self.pending_collisions
        if self.pending_legal:
            self.undo_history.append(self._snapshot())
        is_rotation = action in {CLOCKWISE, CLICK}
        if is_rotation:
            self.visual_mode = "preview_legal" if self.pending_legal else "preview_invalid"
            self._animation = ["pivot", "resolution"]
            return
        if self.pending_legal:
            self._commit_pending()
            self._post_action_feedback()
        else:
            self.visual_mode = "invalid_move"
            if self.step_count >= self.step_budget:
                self._terminal_state = "FAILURE"
                self.visual_mode = "failure_wash"
                self._animation = ["finish_failure"]
            else:
                self.complete_action()

    def _get_hidden_state(self) -> np.ndarray:
        return np.asarray(
            [self.elbow[0], self.elbow[1], self.orientation, self.lit_mask, self.step_count], dtype=np.int16
        )
