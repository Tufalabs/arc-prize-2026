from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_CLICK = 6

BOARD_LEFT = 2
BOARD_TOP = 2
CELL = 6
GRID = 10
VIEW = 64

FLOOR = 1
WALL = 4
WALL_HI = 3
WHITE = 0
BLACK = 5
RED = 8
BLUE = 9
GLASS = 10
ORANGE = 12
GREEN = 14
PURPLE = 15

DIRS = {"N": (0, -1), "S": (0, 1), "W": (-1, 0), "E": (1, 0)}
ACTION_DIR = {ACTION_UP: "N", ACTION_DOWN: "S", ACTION_LEFT: "W", ACTION_RIGHT: "E"}
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}
ORIENTATIONS = ("NE", "ES", "SW", "WN")
RAMP_SIDES = {"NE": ("N", "E"), "ES": ("E", "S"), "SW": ("S", "W"), "WN": ("W", "N")}
SPLITTER_ORIENTATIONS = ("NE", "ES", "SW", "WN", "NS", "EW")
SPLITTER_SIDES = {
    "NE": ("N", "E"),
    "ES": ("E", "S"),
    "SW": ("S", "W"),
    "WN": ("W", "N"),
    "NS": ("N", "S"),
    "EW": ("E", "W"),
}
COLOR_MAP = {"red": RED, "blue": BLUE, "green": GREEN}


@dataclass(frozen=True)
class FruitSpec:
    color: str
    pos: tuple[int, int]


@dataclass(frozen=True)
class BasketSpec:
    color: str
    pos: tuple[int, int]


@dataclass(frozen=True)
class GlassSpec:
    pos: tuple[int, int]
    orientation: str
    marked: bool = False


@dataclass(frozen=True)
class SplitterSpec:
    pos: tuple[int, int]
    orientation: str = "NE"
    color: str | None = None


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    farmer: tuple[int, int]
    facing: str
    fruits: tuple[FruitSpec, ...]
    baskets: tuple[BasketSpec, ...]
    glass: tuple[GlassSpec, ...] = ()
    splitters: tuple[SplitterSpec, ...] = ()
    step_budget: int = 20


LEVELS = (
    LevelSpec(
        "First Roll",
        (
            "##########",
            "#........#",
            "#........#",
            "#........#",
            "#...R..r.#",
            "#........#",
            "#.P......#",
            "#........#",
            "#........#",
            "##########",
        ),
        (2, 6),
        "N",
        (FruitSpec("red", (4, 4)),),
        (BasketSpec("red", (7, 4)),),
        step_budget=24,
    ),
    LevelSpec(
        "First Turn",
        (
            "##########",
            "#........#",
            "#.....r..#",
            "#........#",
            "#........#",
            "#........#",
            "#..R..a..#",
            "#.....P..#",
            "#........#",
            "##########",
        ),
        (6, 7),
        "N",
        (FruitSpec("red", (3, 6)),),
        (BasketSpec("red", (6, 2)),),
        (GlassSpec((6, 6), "WN"),),
        step_budget=42,
    ),
    LevelSpec(
        "Blocker Before Turn",
        (
            "##########",
            "#........#",
            "#....r...#",
            "#........#",
            "#..b..m..#",
            "#........#",
            "#.R...B..#",
            "#P.......#",
            "#........#",
            "##########",
        ),
        (1, 7),
        "N",
        (FruitSpec("red", (2, 6)), FruitSpec("blue", (6, 6))),
        (BasketSpec("red", (5, 2)), BasketSpec("blue", (3, 4))),
        (GlassSpec((6, 4), "ES", True),),
        step_budget=102,
    ),
    LevelSpec(
        "Split the Orchard",
        (
            "##########",
            "#..r.a...#",
            "#.#....#.#",
            "#.#......#",
            "#....s...#",
            "#.R......#",
            "#..#..#.##",
            "#...P....#",
            "#....a..r#",
            "##########",
        ),
        (4, 7),
        "N",
        (FruitSpec("red", (2, 5)),),
        (BasketSpec("red", (3, 1)), BasketSpec("red", (8, 8))),
        (GlassSpec((5, 1), "SW", True), GlassSpec((5, 8), "NE", True)),
        (SplitterSpec((5, 4), "EW", color="red"),),
        step_budget=96,
    ),
    LevelSpec(
        "Glass Orchard",
        (
            "##########",
            "#........#",
            "#........#",
            "#.G..B...#",
            "#........#",
            "#........#",
            "#bR..mP..#",
            "#........#",
            "#...gr...#",
            "##########",
        ),
        (6, 6),
        "W",
        (FruitSpec("red", (2, 6)), FruitSpec("blue", (5, 3)), FruitSpec("green", (2, 3))),
        (BasketSpec("red", (5, 8)), BasketSpec("blue", (1, 6)), BasketSpec("green", (4, 8))),
        (GlassSpec((5, 6), "SW", True),),
        step_budget=204,
    ),
    LevelSpec(
        "Splitter Stopper",
        (
            "##########",
            "#........#",
            "#....r...#",
            "#..b....m#",
            "#..#..#..#",
            "#.R.....S#",
            "#.......b#",
            "#P....B..#",
            "#........#",
            "##########",
        ),
        (1, 7),
        "N",
        (FruitSpec("red", (2, 5)), FruitSpec("blue", (6, 7))),
        (BasketSpec("red", (5, 2)), BasketSpec("blue", (3, 3)), BasketSpec("blue", (8, 6))),
        (GlassSpec((8, 3), "NE", True),),
        (SplitterSpec((8, 5), color="blue"),),
        step_budget=120,
    ),
    LevelSpec(
        "Calibrated Orchard",
        (
            "##########",
            "#.m....m.#",
            "#..r.....#",
            "#......B.#",
            "#........#",
            "#....b...#",
            "##.sR....#",
            "#P.......#",
            "#........#",
            "##########",
        ),
        (1, 7),
        "N",
        (FruitSpec("red", (4, 6)), FruitSpec("blue", (7, 3))),
        (BasketSpec("red", (3, 2)), BasketSpec("blue", (5, 5))),
        (GlassSpec((2, 1), "NE", True), GlassSpec((7, 1), "WN", True)),
        (SplitterSpec((3, 6)),),
        step_budget=160,
    ),
)

LEVELS = (LEVELS[0], LEVELS[1], LEVELS[2], LEVELS[4], LEVELS[3], LEVELS[5], LEVELS[6])


class OrchardHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: GlassOrchard | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        x0 = BOARD_LEFT
        y0 = 62
        width = 60
        frame[y0:64, x0 : x0 + width] = BLACK
        filled = max(0, min(width, int(width * game.remaining_steps / max(1, game.step_budget))))
        if filled:
            frame[y0:64, x0 : x0 + filled] = GREEN
        if game.invalid_cell is not None:
            x, y = game.invalid_cell
            px, py = cell_px(x, y)
            frame[py : py + CELL, px : px + CELL] = np.where(frame[py : py + CELL, px : px + CELL] == WALL, WALL, RED)
        return frame


class GlassOrchard(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = OrchardHud()
        self._board = Sprite(
            np.full((VIEW, VIEW), FLOOR, dtype=np.int8), name="board", x=0, y=0, layer=0, collidable=False
        )
        levels = [
            Level(
                sprites=[
                    self._board.clone(),
                    Sprite(
                        np.full((CELL, CELL), GLASS, dtype=np.int8),
                        name=f"click_{idx}",
                        tags=["sys_click", "sys_every_pixel"],
                        visible=False,
                        collidable=False,
                    ),
                ],
                grid_size=(VIEW, VIEW),
                data={"spec": spec},
                name=spec.name,
            )
            for idx, spec in enumerate(LEVELS)
        ]
        camera = Camera(0, 0, VIEW, VIEW, FLOOR, BLACK, [self._hud])
        super().__init__("glass_orchard-0001", levels, camera, False, len(levels), [1, 2, 3, 4, 6], seed=seed)
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.step_budget = self.spec.step_budget
        self.remaining_steps = self.step_budget
        self.invalid_cell: tuple[int, int] | None = None
        self.farmer = self.spec.farmer
        self.facing = self.spec.facing
        self.fruits = [{"color": fruit.color, "pos": fruit.pos} for fruit in self.spec.fruits]
        self.baskets = [{"color": basket.color, "pos": basket.pos, "filled": False} for basket in self.spec.baskets]
        self.terrain: dict[tuple[int, int], dict[str, Any]] = {}
        for y, row in enumerate(self.spec.rows):
            for x, char in enumerate(row):
                self.terrain[(x, y)] = {"kind": "wall" if char == "#" else "floor"}
        for glass in self.spec.glass:
            self.terrain[glass.pos] = {"kind": "ramp", "orientation": glass.orientation, "marked": glass.marked}
        for splitter in self.spec.splitters:
            self.terrain[splitter.pos] = {
                "kind": "splitter",
                "orientation": splitter.orientation,
                "active": True,
                "color": splitter.color,
            }
        self._push_animation_frames: list[dict[str, Any]] = []
        self._push_animation_pending_finish = False
        self._board = level.get_sprites_by_name("board")[0]
        self._click_sprite = level.get_sprites_by_tag("sys_click")[0]
        self._sync_click_sprite()
        self._sync_board()

    def step(self) -> None:
        if self._push_animation_pending_finish:
            self._push_animation_pending_finish = False
            self._sync_click_sprite()
            self._sync_board()
            self._finish_resolved_action()
            return
        if self._push_animation_frames:
            self._apply_next_push_frame()
            self._sync_click_sprite()
            self._sync_board()
            if not self._push_animation_frames:
                self._push_animation_pending_finish = True
            return

        if self.action.id == GameAction.RESET:
            self.invalid_cell = None
            self._push_animation_frames = []
            self._push_animation_pending_finish = False
            self._sync_board()
            self.complete_action()
            return

        self.invalid_cell = None
        action_id = int(self.action.id.value)
        if action_id in ACTION_DIR:
            self._move_farmer(ACTION_DIR[action_id])
        elif action_id == ACTION_CLICK:
            self._click_rotate()
        else:
            self.invalid_cell = self.farmer

        self.remaining_steps -= 1
        self._sync_click_sprite()
        self._sync_board()
        if self._push_animation_frames or self._push_animation_pending_finish:
            return
        self._finish_resolved_action()

    def _finish_resolved_action(self) -> None:
        if self._is_solved():
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _move_farmer(self, direction: str) -> bool:
        self.facing = direction
        dx, dy = DIRS[direction]
        target = (self.farmer[0] + dx, self.farmer[1] + dy)
        fruit = self._fruit_at(target)
        if fruit is not None:
            return self._push_fruit(fruit, direction)
        if self._farmer_passable(target):
            self.farmer = target
            return True
        return False

    def _push_fruit(self, fruit: dict[str, Any], direction: str) -> bool:
        start = fruit["pos"]
        dx, dy = DIRS[direction]
        reserved = {self.farmer, start}
        pos = start
        moved = False
        frames: list[dict[str, Any]] = []
        while True:
            nxt = (pos[0] + dx, pos[1] + dy)
            outcome = self._fruit_enter_outcome(fruit["color"], nxt, direction, reserved)
            if outcome["kind"] == "move":
                frame_consumed = tuple(outcome.get("consume", ()))
                pos = outcome["pos"]
                frames.append({"fruit": fruit, "pos": pos, "consume": frame_consumed, "deliver": False})
                moved = True
                direction = outcome["direction"]
                dx, dy = DIRS[direction]
                continue
            if outcome["kind"] == "deliver":
                frame_consumed = tuple(outcome.get("consume", ()))
                frames.append({"fruit": fruit, "pos": outcome["pos"], "consume": frame_consumed, "deliver": True})
                self.farmer = start
                self._start_push_animation(frames)
                return True
            if outcome["kind"] == "split":
                frame_consumed = tuple(outcome.get("consume", ()))
                frames.append(
                    {
                        "fruit": fruit,
                        "pos": outcome["pos"],
                        "consume": frame_consumed,
                        "deliver": False,
                        "split": outcome["pieces"],
                    }
                )
                self.farmer = start
                self._start_push_animation(frames)
                return True
            if moved:
                self.farmer = start
                self._start_push_animation(frames)
                return True
            self.invalid_cell = nxt if in_bounds(nxt) else start
            return False

    def _start_push_animation(self, frames: list[dict[str, Any]]) -> None:
        self._push_animation_frames = frames
        self._push_animation_pending_finish = False
        self._apply_next_push_frame()
        if not self._push_animation_frames:
            self._push_animation_pending_finish = True

    def _apply_next_push_frame(self) -> None:
        if not self._push_animation_frames:
            return
        frame = self._push_animation_frames.pop(0)
        fruit = frame["fruit"]
        if fruit in self.fruits:
            fruit["pos"] = frame["pos"]
        if frame["deliver"]:
            if fruit in self.fruits:
                self.fruits.remove(fruit)
            self._fill_basket(frame["pos"])
        if "split" in frame:
            if fruit in self.fruits:
                self.fruits.remove(fruit)
            for piece in frame["split"]:
                basket = self._basket_at(piece["pos"])
                if basket is not None and basket["color"] == piece["color"] and not basket["filled"]:
                    self._fill_basket(piece["pos"])
                else:
                    self.fruits.append({"color": piece["color"], "pos": piece["pos"]})

    def _fruit_enter_outcome(
        self, color: str, cell: tuple[int, int], direction: str, reserved: set[tuple[int, int]]
    ) -> dict[str, Any]:
        if not in_bounds(cell) or cell in reserved or self._fruit_at(cell) is not None:
            return {"kind": "block"}
        basket = self._basket_at(cell)
        if basket is not None:
            if basket["color"] == color and not basket["filled"]:
                return {"kind": "deliver", "pos": cell}
            return {"kind": "block"}
        terrain = self.terrain[cell]
        kind = terrain["kind"]
        if kind == "floor":
            return {"kind": "move", "pos": cell, "direction": direction}
        if kind == "wall" or kind == "glass":
            return {"kind": "block"}
        if kind == "splitter":
            return self._splitter_outcome(color, cell, reserved)
        if kind != "ramp":
            return {"kind": "block"}
        return self._ramp_outcome(color, cell, direction, reserved)

    def _ramp_outcome(
        self, color: str, cell: tuple[int, int], direction: str, reserved: set[tuple[int, int]]
    ) -> dict[str, Any]:
        consume: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        while True:
            if cell in seen:
                return {"kind": "block"}
            seen.add(cell)

            terrain = self.terrain[cell]
            entry = OPPOSITE[direction]
            sides = RAMP_SIDES[terrain["orientation"]]
            if entry not in sides:
                return {"kind": "block"}
            consume.append(cell)

            direction = sides[1] if sides[0] == entry else sides[0]
            dx, dy = DIRS[direction]
            exit_cell = (cell[0] + dx, cell[1] + dy)
            if not in_bounds(exit_cell) or exit_cell in reserved or self._fruit_at(exit_cell) is not None:
                return {"kind": "block"}

            exit_basket = self._basket_at(exit_cell)
            if exit_basket is not None:
                if exit_basket["color"] == color and not exit_basket["filled"]:
                    return {"kind": "deliver", "pos": exit_cell, "consume": tuple(consume)}
                return {"kind": "block"}

            exit_terrain = self.terrain[exit_cell]
            if exit_terrain["kind"] == "floor":
                return {"kind": "move", "pos": exit_cell, "direction": direction, "consume": tuple(consume)}
            if exit_terrain["kind"] == "ramp":
                cell = exit_cell
                continue
            return {"kind": "block"}

    def _splitter_outcome(self, color: str, cell: tuple[int, int], reserved: set[tuple[int, int]]) -> dict[str, Any]:
        terrain = self.terrain[cell]
        if not terrain.get("active"):
            return {"kind": "block"}
        splitter_color = terrain.get("color")
        if splitter_color is not None and splitter_color != color:
            return {"kind": "block"}
        pieces = []
        consumed = []
        occupied = set(reserved)
        pass_through = set(reserved)
        for side in SPLITTER_SIDES[terrain["orientation"]]:
            dx, dy = DIRS[side]
            branch = self._splitter_branch(color, (cell[0] + dx, cell[1] + dy), side, occupied, pass_through)
            if branch is None:
                return {"kind": "block"}
            consumed.extend(branch["consume"])
            if branch["deliver"]:
                pieces.append({"color": color, "pos": branch["pos"], "deliver": True})
            else:
                pieces.append({"color": color, "pos": branch["pos"], "deliver": False})
                occupied.add(branch["pos"])
        return {"kind": "split", "pos": cell, "consume": tuple(consumed), "pieces": tuple(pieces)}

    def _splitter_branch(
        self,
        color: str,
        cell: tuple[int, int],
        direction: str,
        occupied: set[tuple[int, int]],
        pass_through: set[tuple[int, int]],
    ) -> dict[str, Any] | None:
        pos: tuple[int, int] | None = None
        consume: list[tuple[int, int]] = []
        while True:
            if not in_bounds(cell):
                return None if pos is None else {"pos": pos, "consume": tuple(consume), "deliver": False}
            if cell in occupied and cell not in pass_through:
                return None if pos is None else {"pos": pos, "consume": tuple(consume), "deliver": False}
            if self._fruit_at(cell) is not None and cell not in pass_through:
                return None if pos is None else {"pos": pos, "consume": tuple(consume), "deliver": False}
            if cell in pass_through:
                dx, dy = DIRS[direction]
                cell = (cell[0] + dx, cell[1] + dy)
                continue
            basket = self._basket_at(cell)
            if basket is not None:
                if basket["color"] == color and not basket["filled"]:
                    return {"pos": cell, "consume": tuple(consume), "deliver": True}
                return None if pos is None else {"pos": pos, "consume": tuple(consume), "deliver": False}
            terrain = self.terrain[cell]
            kind = terrain["kind"]
            if kind == "floor":
                pos = cell
            elif kind == "ramp":
                entry = OPPOSITE[direction]
                sides = RAMP_SIDES[terrain["orientation"]]
                if entry not in sides:
                    return None if pos is None else {"pos": pos, "consume": tuple(consume), "deliver": False}
                consume.append(cell)
                direction = sides[1] if sides[0] == entry else sides[0]
                dx, dy = DIRS[direction]
                cell = (cell[0] + dx, cell[1] + dy)
                continue
            else:
                return None if pos is None else {"pos": pos, "consume": tuple(consume), "deliver": False}
            dx, dy = DIRS[direction]
            cell = (cell[0] + dx, cell[1] + dy)

    def _fruit_exit_passable(self, color: str, cell: tuple[int, int], reserved: set[tuple[int, int]]) -> bool:
        if not in_bounds(cell) or cell in reserved or self._fruit_at(cell) is not None:
            return False
        basket = self._basket_at(cell)
        if basket is not None:
            return basket["color"] == color and not basket["filled"]
        return self.terrain[cell]["kind"] == "floor"

    def _crack_glass(self) -> bool:
        dx, dy = DIRS[self.facing]
        cell = (self.farmer[0] + dx, self.farmer[1] + dy)
        if in_bounds(cell) and self.terrain[cell]["kind"] == "glass":
            self.terrain[cell]["kind"] = "ramp"
            return True
        if in_bounds(cell) and self.terrain[cell]["kind"] == "splitter":
            return self._rotate_splitter(cell)
        self.invalid_cell = cell if in_bounds(cell) else self.farmer
        return False

    def _rotate_splitter(self, cell: tuple[int, int]) -> bool:
        terrain = self.terrain[cell]
        idx = SPLITTER_ORIENTATIONS.index(terrain["orientation"])
        terrain["orientation"] = SPLITTER_ORIENTATIONS[(idx + 1) % len(SPLITTER_ORIENTATIONS)]
        terrain["active"] = True
        return True

    def _click_rotate(self) -> bool:
        data = self.action.data or {}
        cell = pixel_to_cell(int(data.get("x", 0)), int(data.get("y", 0)))
        if cell is not None:
            terrain = self.terrain[cell]
            if terrain["kind"] == "splitter":
                return self._rotate_splitter(cell)
            if terrain["kind"] in {"glass", "ramp"} and terrain.get("marked"):
                idx = ORIENTATIONS.index(terrain["orientation"])
                terrain["orientation"] = ORIENTATIONS[(idx + 1) % len(ORIENTATIONS)]
                return True
        return False

    def _farmer_passable(self, cell: tuple[int, int]) -> bool:
        return (
            in_bounds(cell)
            and self._fruit_at(cell) is None
            and self.terrain[cell]["kind"] in {"floor", "ramp", "splitter"}
        )

    def _fruit_at(self, cell: tuple[int, int]) -> dict[str, Any] | None:
        for fruit in self.fruits:
            if fruit["pos"] == cell:
                return fruit
        return None

    def _basket_at(self, cell: tuple[int, int]) -> dict[str, Any] | None:
        for basket in self.baskets:
            if basket["pos"] == cell:
                return basket
        return None

    def _fill_basket(self, cell: tuple[int, int]) -> None:
        basket = self._basket_at(cell)
        if basket is not None:
            basket["filled"] = True

    def _is_solved(self) -> bool:
        return all(basket["filled"] for basket in self.baskets)

    def _sync_click_sprite(self) -> None:
        marked = None
        for cell, terrain in self.terrain.items():
            if terrain["kind"] in {"glass", "ramp"} and terrain.get("marked"):
                marked = cell
                break
        if marked is None:
            self._click_sprite.set_visible(False)
            return
        px, py = cell_px(*marked)
        self._click_sprite.set_position(px, py)
        self._click_sprite.set_visible(False)

    def _sync_board(self) -> None:
        frame = np.full((VIEW, VIEW), FLOOR, dtype=np.int8)
        for y in range(GRID):
            for x in range(GRID):
                self._draw_cell(frame, x, y)
        for basket in self.baskets:
            self._draw_basket(frame, basket)
        for fruit in self.fruits:
            self._draw_fruit(frame, fruit["pos"], COLOR_MAP[fruit["color"]])
        self._draw_farmer(frame)
        self._board.pixels = frame

    def _draw_cell(self, frame: np.ndarray, x: int, y: int) -> None:
        px, py = cell_px(x, y)
        terrain = self.terrain[(x, y)]
        kind = terrain["kind"]
        if kind == "wall":
            frame[py : py + CELL, px : px + CELL] = WALL
            frame[py, px : px + CELL] = WALL_HI
            frame[py : py + CELL, px] = WALL_HI
        else:
            frame[py : py + CELL, px : px + CELL] = FLOOR
        if kind == "glass":
            frame[py : py + CELL, px : px + CELL] = GLASS
            frame[py, px : px + 3] = WHITE
            self._draw_ramp_shape(frame, x, y, terrain["orientation"], faint=True)
        elif kind == "ramp":
            self._draw_ramp_shape(frame, x, y, terrain["orientation"], faint=False)
        elif kind == "splitter":
            self._draw_splitter_shape(frame, x, y, terrain["orientation"], terrain.get("color"))
        if terrain.get("marked"):
            frame[py + 2 : py + 4, px + 2 : px + 4] = PURPLE

    def _draw_ramp_shape(self, frame: np.ndarray, x: int, y: int, orientation: str, *, faint: bool) -> None:
        px, py = cell_px(x, y)
        color = WHITE if faint else BLUE
        coords = {
            "N": [(px + 2, py), (px + 3, py), (px + 2, py + 1), (px + 3, py + 1), (px + 2, py + 2), (px + 3, py + 2)],
            "E": [
                (px + 3, py + 2),
                (px + 4, py + 2),
                (px + 5, py + 2),
                (px + 3, py + 3),
                (px + 4, py + 3),
                (px + 5, py + 3),
            ],
            "S": [
                (px + 2, py + 3),
                (px + 3, py + 3),
                (px + 2, py + 4),
                (px + 3, py + 4),
                (px + 2, py + 5),
                (px + 3, py + 5),
            ],
            "W": [(px, py + 2), (px + 1, py + 2), (px + 2, py + 2), (px, py + 3), (px + 1, py + 3), (px + 2, py + 3)],
        }
        for side in RAMP_SIDES[orientation]:
            for cx, cy in coords[side]:
                frame[cy, cx] = color
        frame[py + 2 : py + 4, px + 2 : px + 4] = color

    def _draw_splitter_shape(self, frame: np.ndarray, x: int, y: int, orientation: str, color_name: str | None) -> None:
        px, py = cell_px(x, y)
        color = COLOR_MAP.get(str(color_name), WHITE) if color_name is not None else WHITE

        frame[py : py + CELL, px : px + CELL] = BLACK
        for ox, oy in ((2, 2), (3, 2), (2, 3), (3, 3)):
            frame[py + oy, px + ox] = color

        rays = {
            "N": (((2, 0), (2, 1)), ((3, 0), (3, 1))),
            "E": (((4, 2), (5, 2)), ((4, 3), (5, 3))),
            "S": (((2, 4), (2, 5)), ((3, 4), (3, 5))),
            "W": (((0, 2), (1, 2)), ((0, 3), (1, 3))),
        }
        for side in SPLITTER_SIDES[orientation]:
            for start, end in rays[side]:
                frame[py + start[1], px + start[0]] = color
                frame[py + end[1], px + end[0]] = color

    def _draw_basket(self, frame: np.ndarray, basket: dict[str, Any]) -> None:
        px, py = cell_px(*basket["pos"])
        color = COLOR_MAP[basket["color"]]
        frame[py + 1 : py + 5, px + 1] = color
        frame[py + 4, px + 1 : px + 5] = color
        frame[py + 1 : py + 5, px + 4] = color
        if basket["filled"]:
            frame[py + 2 : py + 4, px + 2 : px + 4] = color
            frame[py + 2, px + 2] = WHITE

    def _draw_fruit(self, frame: np.ndarray, pos: tuple[int, int], color: int) -> None:
        px, py = cell_px(*pos)
        for ox, oy in ((2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (4, 2), (1, 3), (2, 3), (3, 3), (4, 3), (2, 4), (3, 4)):
            frame[py + oy, px + ox] = color
        frame[py + 1, px + 2] = WHITE

    def _draw_farmer(self, frame: np.ndarray) -> None:
        px, py = cell_px(*self.farmer)
        frame[py + 1 : py + 5, px + 2 : px + 4] = ORANGE
        frame[py + 2 : py + 5, px + 1 : px + 5] = ORANGE
        frame[py + 1, px + 1 : px + 5] = BLACK
        notch = {"N": (px + 2, py), "S": (px + 3, py + 5), "W": (px, py + 3), "E": (px + 5, py + 2)}[self.facing]
        frame[notch[1], notch[0]] = WHITE


def cell_px(x: int, y: int) -> tuple[int, int]:
    return BOARD_LEFT + x * CELL, BOARD_TOP + y * CELL


def pixel_to_cell(x: int, y: int) -> tuple[int, int] | None:
    gx = (x - BOARD_LEFT) // CELL
    gy = (y - BOARD_TOP) // CELL
    if 0 <= gx < GRID and 0 <= gy < GRID:
        return int(gx), int(gy)
    return None


def in_bounds(cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= x < GRID and 0 <= y < GRID
