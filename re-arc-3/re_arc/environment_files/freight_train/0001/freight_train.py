from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "freight_train-0001"
BOARD_SIZE = 12
CELL_SIZE = 5
BOARD_ORIGIN = 2

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

MOVE_DELTAS = {ACTION_UP: (-1, 0), ACTION_DOWN: (1, 0), ACTION_LEFT: (0, -1), ACTION_RIGHT: (0, 1)}

COLOR_HALO = 0
COLOR_FLOOR = 1
COLOR_CART = 2
COLOR_UNDERCARRIAGE = 3
COLOR_WALL = 4
COLOR_OUTSIDE = 5
COLOR_WORKER = 6
COLOR_WALL_BUMP = 7
COLOR_IMPACT = 8
COLOR_BLUE = 9
COLOR_COMPLETION = 10
COLOR_COUPLER = 11
COLOR_ORANGE = 12
COLOR_MAROON = 13
COLOR_GREEN = 14
COLOR_PURPLE = 15

CARGO_COLORS = {"B": COLOR_BLUE, "O": COLOR_ORANGE, "M": COLOR_MAROON, "G": COLOR_GREEN, "P": COLOR_PURPLE}


@dataclass(frozen=True)
class CartSpec:
    cargo: str
    axis: str
    hook: str
    position: tuple[int, int]


@dataclass(frozen=True)
class FreightLevelSpec:
    name: str
    rows: tuple[str, ...]
    worker: tuple[int, int]
    carts: tuple[CartSpec, ...]
    bays: tuple[tuple[str, tuple[int, int]], ...]
    initial_locks: tuple[tuple[int, int], ...]
    baseline_actions: int


LEVEL_SPECS = (
    FreightLevelSpec(
        "First Delivery",
        (
            "############",
            "############",
            "############",
            "############",
            "###......###",
            "###......###",
            "###......###",
            "############",
            "############",
            "############",
            "############",
            "############",
        ),
        (5, 3),
        (CartSpec("B", "H", "R", (5, 5)),),
        (("B", (5, 6)),),
        (),
        2,
    ),
    FreightLevelSpec(
        "Loose Line",
        (
            "############",
            "############",
            "############",
            "############",
            "##.......###",
            "##.......###",
            "##.......###",
            "##.......###",
            "##.......###",
            "############",
            "############",
            "############",
        ),
        (6, 2),
        (CartSpec("B", "H", "R", (6, 3)), CartSpec("O", "H", "L", (6, 4))),
        (("B", (6, 5)), ("O", (5, 6))),
        (),
        6,
    ),
    FreightLevelSpec(
        "The Clasp",
        (
            "############",
            "############",
            "############",
            "############",
            "############",
            "#####..#####",
            "##.....#####",
            "##....######",
            "##....######",
            "############",
            "############",
            "############",
        ),
        (8, 2),
        (CartSpec("B", "H", "R", (6, 4)), CartSpec("O", "H", "R", (6, 6))),
        (("B", (5, 5)), ("O", (5, 6))),
        (),
        7,
    ),
    FreightLevelSpec(
        "Far-End Lift",
        (
            "############",
            "############",
            "#####....###",
            "#####.....##",
            "##........##",
            "##......####",
            "##......####",
            "##...#..####",
            "##......####",
            "############",
            "############",
            "############",
        ),
        (8, 2),
        (CartSpec("B", "H", "R", (7, 3)), CartSpec("O", "H", "R", (4, 8)), CartSpec("G", "V", "U", (3, 8))),
        (("B", (8, 6)), ("O", (8, 7)), ("G", (2, 8))),
        (),
        28,
    ),
    FreightLevelSpec(
        "Three-Wide Gate",
        (
            "############",
            "############",
            "###........#",
            "###........#",
            "###........#",
            "###........#",
            "######...###",
            "#..###....##",
            "#.........##",
            "#...#.....##",
            "############",
            "############",
        ),
        (9, 2),
        (
            CartSpec("B", "H", "R", (8, 2)),
            CartSpec("O", "H", "R", (8, 3)),
            CartSpec("G", "H", "R", (8, 4)),
            CartSpec("P", "V", "U", (5, 8)),
        ),
        (("B", (4, 3)), ("O", (4, 4)), ("G", (4, 5)), ("P", (9, 1))),
        ((0, 1), (1, 2)),
        40,
    ),
    FreightLevelSpec(
        "Extension Route",
        (
            "############",
            "#####......#",
            "#####......#",
            "#####.......",
            "#...........",
            "##..##..#...",
            "###.##..#...",
            "###.#...#...",
            "###.#...#...",
            "#.......#...",
            "#########...",
            "############",
        ),
        (9, 4),
        (
            CartSpec("B", "H", "R", (5, 6)),
            CartSpec("O", "H", "R", (5, 7)),
            CartSpec("G", "H", "R", (8, 3)),
            CartSpec("P", "V", "U", (3, 6)),
            CartSpec("M", "V", "U", (3, 8)),
        ),
        (("B", (10, 9)), ("O", (10, 10)), ("G", (10, 11)), ("P", (2, 6)), ("M", (2, 8))),
        ((0, 1),),
        47,
    ),
    FreightLevelSpec(
        "Two Departures",
        (
            "############",
            "#.........##",
            "#...#......#",
            "#...........",
            "............",
            "..###.###...",
            "..###.###...",
            "..###.###...",
            "..###.###...",
            "..###.###...",
            "..#######...",
            "############",
        ),
        (9, 5),
        (
            CartSpec("G", "H", "R", (8, 5)),
            CartSpec("B", "H", "R", (4, 6)),
            CartSpec("O", "H", "R", (4, 7)),
            CartSpec("P", "H", "R", (3, 5)),
            CartSpec("M", "H", "R", (3, 7)),
        ),
        (("G", (10, 9)), ("B", (10, 10)), ("O", (10, 11)), ("P", (10, 0)), ("M", (10, 1))),
        ((1, 2),),
        50,
    ),
)


def _action_id(action: GameAction | int) -> int:
    return int(action.value if isinstance(action, GameAction) else action)


def _cell_pixel(cell: tuple[int, int]) -> tuple[int, int]:
    row, col = cell
    return BOARD_ORIGIN + col * CELL_SIZE, BOARD_ORIGIN + row * CELL_SIZE


def _cart_pixels(spec: CartSpec) -> np.ndarray:
    pixels = np.full((CELL_SIZE, CELL_SIZE), -1, dtype=np.int8)
    pixels[1:4, 1:4] = COLOR_CART
    cargo = CARGO_COLORS[spec.cargo]
    for row, col in ((1, 2), (2, 1), (2, 2), (2, 3), (3, 2)):
        pixels[row, col] = cargo
    pixels[4, 1:4] = COLOR_UNDERCARRIAGE
    pixels[1:4, 4] = COLOR_COUPLER
    pixels[2, 3] = COLOR_COUPLER
    pixels[1, 0] = COLOR_COUPLER
    pixels[2, 0] = COLOR_OUTSIDE
    pixels[3, 0] = COLOR_COUPLER
    if spec.hook == "L":
        return np.fliplr(pixels).copy()
    if spec.hook == "D":
        return np.rot90(pixels, -1).copy()
    if spec.hook == "U":
        return np.rot90(pixels, 1).copy()
    return pixels


class FreightTrain(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = []
        for spec in LEVEL_SPECS:
            canvas = Sprite(
                np.full((64, 64), COLOR_OUTSIDE, dtype=np.int8),
                name="freight_canvas",
                x=0,
                y=0,
                layer=0,
                collidable=False,
                tags=["canvas"],
            )
            levels.append(Level([canvas], grid_size=(64, 64), data={"spec": spec}, name=spec.name))
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, 64, 64, COLOR_OUTSIDE, COLOR_OUTSIDE),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: FreightLevelSpec = level.get_data("spec")
        self.canvas = level.get_sprites_by_tag("canvas")[0]
        self.floor = {
            (row, col) for row, line in enumerate(self.spec.rows) for col, char in enumerate(line) if char == "."
        }
        self.bays = {cargo: cell for cargo, cell in self.spec.bays}
        self.worker_position = self.spec.worker
        self.cart_positions = [cart.position for cart in self.spec.carts]
        self.locks = {frozenset(pair) for pair in self.spec.initial_locks}
        self._normalize_initial_locks()
        self.matched_bays: set[int] = set()
        self.won = False
        self.selected_id = None
        self.cursor_state = None
        self.step_budget = self.spec.baseline_actions * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[
            tuple[tuple[int, int], tuple[tuple[int, int], ...], frozenset[frozenset[int]], frozenset[int], bool]
        ] = []
        self.animation_frames: list[tuple[str, int]] = []
        self._pending_push: dict[str, object] | None = None
        self._wall_bump = False
        self._impact_carts: set[int] = set()
        self._impact_direction: tuple[int, int] | None = None
        self._perimeter_pulse = False
        self._halo_pulse = False
        self._base_frame = self._build_base_frame()
        self._sync_canvas()

    def _build_base_frame(self) -> np.ndarray:
        frame = np.full((64, 64), COLOR_OUTSIDE, dtype=np.int8)
        for row, line in enumerate(self.spec.rows):
            for col, char in enumerate(line):
                x, y = _cell_pixel((row, col))
                frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = COLOR_FLOOR if char == "." else COLOR_WALL
        return frame

    def _normalize_initial_locks(self) -> None:
        self.locks.update(self._compatible_adjacent_pairs(self.cart_positions))

    def _faces(self, cart_id: int) -> tuple[str, str]:
        hook = self.spec.carts[cart_id].hook
        opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}[hook]
        return hook, opposite

    def _compatible_adjacent_pairs(self, positions: list[tuple[int, int]]) -> set[frozenset[int]]:
        result: set[frozenset[int]] = set()
        directions = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}
        for first in range(len(positions)):
            for second in range(first + 1, len(positions)):
                dr = positions[second][0] - positions[first][0]
                dc = positions[second][1] - positions[first][1]
                face = directions.get((dr, dc))
                if face is None:
                    continue
                opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}[face]
                first_hook, first_socket = self._faces(first)
                second_hook, second_socket = self._faces(second)
                if (face == first_hook and opposite == second_socket) or (
                    face == first_socket and opposite == second_hook
                ):
                    result.add(frozenset((first, second)))
        return result

    def _components(self) -> list[set[int]]:
        components = [{cart_id} for cart_id in range(len(self.cart_positions))]
        for seam in self.locks:
            joined = [component for component in components if component & set(seam)]
            merged = set().union(*joined)
            components = [component for component in components if component not in joined]
            components.append(merged)
        return components

    def _component_for(self, cart_id: int) -> set[int]:
        return next(component for component in self._components() if cart_id in component)

    def _snapshot(self) -> None:
        self.undo_history.append(
            (
                self.worker_position,
                tuple(self.cart_positions),
                frozenset(self.locks),
                frozenset(self.matched_bays),
                self.won,
            )
        )

    def _spend_step(self) -> bool:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.remaining_steps:
            return False
        self._sync_canvas()
        self.lose()
        self.complete_action()
        return True

    def _restore_initial_state(self) -> None:
        self.worker_position = self.spec.worker
        self.cart_positions = [cart.position for cart in self.spec.carts]
        self.locks = {frozenset(pair) for pair in self.spec.initial_locks}
        self._normalize_initial_locks()
        self.matched_bays.clear()
        self.won = False
        self.remaining_steps = self.step_budget
        self.undo_history.clear()
        self.animation_frames.clear()
        self._pending_push = None
        self._wall_bump = False
        self._impact_carts.clear()
        self._impact_direction = None

    def _attempt_push(self, contacted: int, delta: tuple[int, int]) -> tuple[set[int], set[int]]:
        affected = set(self._component_for(contacted))
        position_to_cart = {position: cart_id for cart_id, position in enumerate(self.cart_positions)}
        changed = True
        while changed:
            changed = False
            for cart_id in tuple(affected):
                row, col = self.cart_positions[cart_id]
                destination = (row + delta[0], col + delta[1])
                other = position_to_cart.get(destination)
                if other is not None and other not in affected:
                    affected.update(self._component_for(other))
                    changed = True
        obstructed = set()
        for cart_id in affected:
            row, col = self.cart_positions[cart_id]
            destination = (row + delta[0], col + delta[1])
            if destination not in self.floor:
                obstructed.add(cart_id)
        return affected, obstructed

    def _begin_push(self, affected: set[int], delta: tuple[int, int]) -> None:
        self._snapshot()
        old_positions = tuple(self.cart_positions)
        new_positions = tuple(
            (row + delta[0], col + delta[1]) if cart_id in affected else (row, col)
            for cart_id, (row, col) in enumerate(self.cart_positions)
        )
        contacted_cell = next(
            position
            for cart_id, position in enumerate(old_positions)
            if cart_id in affected
            and position == (self.worker_position[0] + delta[0], self.worker_position[1] + delta[1])
        )
        self._pending_push = {
            "affected": set(affected),
            "old_positions": old_positions,
            "new_positions": new_positions,
            "old_worker": self.worker_position,
            "new_worker": contacted_cell,
            "delta": delta,
            "fraction": 0,
        }
        self.animation_frames = [("move", 2), ("move", 4), ("move", 5), ("settle", 0)]
        self._play_animation_frame()

    def _play_animation_frame(self) -> None:
        kind, value = self.animation_frames.pop(0)
        if kind == "impact":
            self._sync_canvas()
            self._impact_carts.clear()
            self._impact_direction = None
            self.complete_action()
            return
        if kind == "move":
            assert self._pending_push is not None
            self._pending_push["fraction"] = value
            if value == 5:
                self.cart_positions = list(self._pending_push["new_positions"])
                self.worker_position = self._pending_push["new_worker"]
            self._sync_canvas()
            return
        if kind == "settle":
            self.locks.update(self._compatible_adjacent_pairs(self.cart_positions))
            self._pending_push = None
            self.matched_bays = {
                cart_id
                for cart_id, position in enumerate(self.cart_positions)
                if position == self.bays[self.spec.carts[cart_id].cargo]
            }
            self.won = len(self.matched_bays) == len(self.cart_positions)
            if self.won:
                self._halo_pulse = True
                self.animation_frames = [("finish", 0)]
                self._sync_canvas()
                return
            self._sync_canvas()
            self.complete_action()
            return
        self._halo_pulse = False
        self._sync_canvas()
        self.next_level()
        self.complete_action()

    def _draw_bay(self, frame: np.ndarray, cargo: str, cell: tuple[int, int]) -> None:
        x, y = _cell_pixel(cell)
        matching_cart = next(cart_id for cart_id, cart in enumerate(self.spec.carts) if cart.cargo == cargo)
        color = CARGO_COLORS[cargo]
        if matching_cart in self.matched_bays:
            color = COLOR_COMPLETION if self._halo_pulse else COLOR_HALO
        for row, col in (
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 4),
            (3, 0),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
        ):
            frame[y + row, x + col] = color

    def _draw_cart(self, frame: np.ndarray, cart_id: int, pixel_x: int, pixel_y: int) -> None:
        pixels = _cart_pixels(self.spec.carts[cart_id])
        for seam in self.locks:
            if cart_id not in seam:
                continue
            other = next(iter(set(seam) - {cart_id}))
            row, col = self.cart_positions[cart_id]
            other_row, other_col = self.cart_positions[other]
            dr, dc = other_row - row, other_col - col
            if dc == 1:
                pixels[1:4, 4] = COLOR_COUPLER
                pixels[4, 4] = COLOR_UNDERCARRIAGE
            elif dc == -1:
                pixels[1:4, 0] = COLOR_COUPLER
                pixels[4, 0] = COLOR_UNDERCARRIAGE
            elif dr == 1:
                pixels[4, 1:4] = COLOR_COUPLER
                pixels[4, 0] = COLOR_UNDERCARRIAGE
            elif dr == -1:
                pixels[0, 1:4] = COLOR_COUPLER
                pixels[0, 0] = COLOR_UNDERCARRIAGE
        visible = pixels >= 0
        region = frame[pixel_y : pixel_y + CELL_SIZE, pixel_x : pixel_x + CELL_SIZE]
        region[visible] = pixels[visible]

    def _sync_canvas(self) -> None:
        frame = self._base_frame.copy()
        for cargo, cell in self.spec.bays:
            self._draw_bay(frame, cargo, cell)

        pending = self._pending_push
        for cart_id, position in enumerate(self.cart_positions):
            x, y = _cell_pixel(position)
            if pending is not None and cart_id in pending["affected"] and pending["fraction"] < 5:
                old_row, old_col = pending["old_positions"][cart_id]
                dx = pending["delta"][1] * pending["fraction"]
                dy = pending["delta"][0] * pending["fraction"]
                x, y = _cell_pixel((old_row, old_col))
                x += dx
                y += dy
            self._draw_cart(frame, cart_id, x, y)

        worker_x, worker_y = _cell_pixel(self.worker_position)
        if pending is not None and pending["fraction"] < 5:
            old_row, old_col = pending["old_worker"]
            worker_x, worker_y = _cell_pixel((old_row, old_col))
            worker_x += pending["delta"][1] * pending["fraction"]
            worker_y += pending["delta"][0] * pending["fraction"]
        worker_color = COLOR_WALL_BUMP if self._wall_bump else COLOR_WORKER
        frame[worker_y + 1 : worker_y + 4, worker_x + 1 : worker_x + 4] = worker_color
        frame[worker_y + 2, worker_x + 2] = COLOR_HALO

        if self._impact_direction is not None:
            dr, dc = self._impact_direction
            for cart_id in self._impact_carts:
                x, y = _cell_pixel(self.cart_positions[cart_id])
                if dr < 0:
                    frame[y, x : x + CELL_SIZE] = COLOR_IMPACT
                elif dr > 0:
                    frame[y + 4, x : x + CELL_SIZE] = COLOR_IMPACT
                elif dc < 0:
                    frame[y : y + CELL_SIZE, x] = COLOR_IMPACT
                else:
                    frame[y : y + CELL_SIZE, x + 4] = COLOR_IMPACT

        filled = round(60 * self.remaining_steps / max(1, self.step_budget))
        frame[61, 2:62] = COLOR_IMPACT
        frame[61, 2 : 2 + filled] = COLOR_COMPLETION
        if self._perimeter_pulse:
            frame[:2, :] = COLOR_COMPLETION
            frame[-2:, :] = COLOR_COMPLETION
            frame[:, :2] = COLOR_COMPLETION
            frame[:, -2:] = COLOR_COMPLETION
        self.canvas.pixels = frame

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action == 0:
            self._sync_canvas()
            self.complete_action()
            return
        if action == ACTION_SPACE:
            if self._perimeter_pulse:
                self._perimeter_pulse = False
                self._sync_canvas()
                self.complete_action()
                return
            self._restore_initial_state()
            self._perimeter_pulse = True
            self._sync_canvas()
            return
        if self.won:
            self.complete_action()
            return
        if self._spend_step():
            return
        if action == ACTION_UNDO:
            if self.undo_history:
                worker, positions, locks, matched, won = self.undo_history.pop()
                self.worker_position = worker
                self.cart_positions = list(positions)
                self.locks = set(locks)
                self.matched_bays = set(matched)
                self.won = won
            self._sync_canvas()
            self.complete_action()
            return
        if action in (ACTION_CLICK,):
            self._sync_canvas()
            self.complete_action()
            return
        if action not in MOVE_DELTAS:
            self.complete_action()
            return

        delta = MOVE_DELTAS[action]
        target = (self.worker_position[0] + delta[0], self.worker_position[1] + delta[1])
        if target not in self.floor:
            self._wall_bump = True
            self._sync_canvas()
            self._wall_bump = False
            self.complete_action()
            return
        cart_at = {position: cart_id for cart_id, position in enumerate(self.cart_positions)}
        contacted = cart_at.get(target)
        if contacted is None:
            self._snapshot()
            self.worker_position = target
            self._sync_canvas()
            self.complete_action()
            return
        affected, obstructed = self._attempt_push(contacted, delta)
        if obstructed:
            self._impact_carts = obstructed
            self._impact_direction = delta
            self.animation_frames = [("impact", 0)]
            self._sync_canvas()
            return
        self._begin_push(affected, delta)
