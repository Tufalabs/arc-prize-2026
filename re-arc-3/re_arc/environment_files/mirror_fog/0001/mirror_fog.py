from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "mirror_fog-0001"
VIEW_SIZE = 64
BOARD_ORIGIN = 2
CELL = 5
GRID = 12

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SPACE = 5
CLICK = 6

DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
ACTION_DIR = {UP: "U", DOWN: "D", LEFT: "L", RIGHT: "R"}
CW = {"U": "R", "R": "D", "D": "L", "L": "U"}

CLEAR = 0
FLOOR = 1
FOG = 2
METAL = 3
WALL = 4
BLACK = 5
RED = 8
BLUE = 9
GLASS = 10
YELLOW = 11
ORANGE = 12
GREEN = 14
PURPLE = 15


@dataclass(frozen=True)
class LevelSpec:
    name: str
    source: tuple[tuple[int, int], str]
    player: tuple[int, int, str]
    mirrors: tuple[tuple[int, int, str], ...]
    stands: tuple[tuple[int, int, bool], ...]
    prisms: tuple[tuple[int, int, str], ...]
    fog: tuple[tuple[int, int, int], ...]
    walls: tuple[tuple[int, int], ...] = ()
    baseline: int = 1


PROGRAMS: tuple[tuple[int | tuple[int, int], ...], ...] = (
    (SPACE, LEFT, UP, RIGHT, RIGHT, SPACE, RIGHT, SPACE),
    (SPACE, DOWN, SPACE, UP, LEFT, UP, RIGHT, RIGHT, RIGHT, SPACE, UP, RIGHT, RIGHT, DOWN, LEFT, LEFT, LEFT),
    (UP, UP, UP, LEFT, LEFT, LEFT, LEFT, LEFT, DOWN, DOWN, RIGHT, RIGHT, RIGHT, RIGHT, SPACE, UP, UP, UP, RIGHT, SPACE),
    (
        UP,
        RIGHT,
        DOWN,
        RIGHT,
        UP,
        LEFT,
        UP,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        DOWN,
        DOWN,
        DOWN,
        RIGHT,
        SPACE,
        UP,
        UP,
        UP,
        LEFT,
        LEFT,
        DOWN,
        LEFT,
        LEFT,
        LEFT,
        UP,
    ),
    (
        (7, 5),
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        DOWN,
        DOWN,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        SPACE,
        UP,
        RIGHT,
        RIGHT,
        RIGHT,
        DOWN,
        RIGHT,
        SPACE,
        DOWN,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        DOWN,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        SPACE,
        UP,
        LEFT,
        LEFT,
        LEFT,
        UP,
        UP,
        UP,
        UP,
        RIGHT,
        (7, 5),
        SPACE,
    ),
    (
        (7, 5),
        (8, 9),
        DOWN,
        DOWN,
        DOWN,
        RIGHT,
        DOWN,
        RIGHT,
        UP,
        UP,
        UP,
        SPACE,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        DOWN,
        DOWN,
        LEFT,
        UP,
        LEFT,
        LEFT,
        DOWN,
        DOWN,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        SPACE,
        UP,
        UP,
        UP,
        LEFT,
        UP,
        UP,
        RIGHT,
        (7, 5),
        SPACE,
    ),
    (
        (7, 5),
        UP,
        (8, 9),
        RIGHT,
        RIGHT,
        RIGHT,
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        RIGHT,
        UP,
        UP,
        UP,
        SPACE,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        DOWN,
        DOWN,
        DOWN,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        SPACE,
        UP,
        UP,
        UP,
        LEFT,
        UP,
        UP,
        RIGHT,
        (7, 5),
        SPACE,
        DOWN,
        LEFT,
        LEFT,
        UP,
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        RIGHT,
        RIGHT,
        UP,
    ),
    (
        (7, 5),
        UP,
        (8, 9),
        RIGHT,
        RIGHT,
        RIGHT,
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        RIGHT,
        UP,
        UP,
        UP,
        SPACE,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        LEFT,
        DOWN,
        DOWN,
        DOWN,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        RIGHT,
        SPACE,
        UP,
        LEFT,
        LEFT,
        UP,
        LEFT,
        DOWN,
        SPACE,
        RIGHT,
        RIGHT,
        RIGHT,
        UP,
        UP,
        LEFT,
        UP,
        UP,
        RIGHT,
        (7, 5),
        SPACE,
        DOWN,
        LEFT,
        LEFT,
        UP,
        DOWN,
        DOWN,
        DOWN,
        DOWN,
        RIGHT,
        RIGHT,
        UP,
    ),
)


def _divider_level_2() -> tuple[tuple[int, int], ...]:
    return tuple((x, 6) for x in (1, 2, 3, 4, 6, 7, 8, 9, 10))


LEVEL_SPECS = (
    LevelSpec(
        "First reflected clearing",
        ((0, 5), "R"),
        (6, 6, "U"),
        ((6, 5, "\\"),),
        (),
        (),
        ((6, 4, 0), (8, 4, 0), (8, 3, 0), (8, 6, 0), (9, 6, 0), (9, 4, 0)),
        baseline=len(PROGRAMS[0]),
    ),
    LevelSpec(
        "Bridge that must not close early",
        ((0, 4), "R"),
        (5, 5, "U"),
        ((5, 4, "\\"), (5, 7, "/")),
        (),
        (),
        ((5, 5, 1), (5, 3, 0), (5, 2, 0), (8, 3, 0), (8, 5, 0), (5, 6, 0), (6, 7, 0), (7, 7, 0)),
        _divider_level_2(),
        len(PROGRAMS[1]),
    ),
    LevelSpec(
        "Making a stopper",
        ((0, 5), "R"),
        (7, 9, "U"),
        ((7, 6, "/"), (3, 8, "/")),
        (),
        (),
        ((5, 5, 2), (8, 5, 0), (7, 4, 0), (7, 3, 0), (7, 7, 0), (8, 8, 0), (9, 8, 0), (10, 8, 0)),
        baseline=len(PROGRAMS[2]),
    ),
    LevelSpec(
        "Movable shade",
        ((0, 5), "R"),
        (4, 7, "U"),
        ((4, 6, "\\"), (6, 6, "\\"), (10, 8, "\\")),
        (),
        (),
        ((8, 5, 0), (9, 5, 0), (10, 6, 0), (10, 7, 0), (9, 8, 0), (8, 8, 0), (7, 8, 0)),
        baseline=len(PROGRAMS[3]),
    ),
    LevelSpec(
        "Lock route then unlock",
        ((0, 5), "R"),
        (6, 6, "U"),
        ((7, 5, "/"), (3, 8, "/"), (10, 8, "/"), (6, 10, "\\")),
        ((7, 5, False),),
        (),
        (
            (5, 5, 2),
            (7, 4, 0),
            (7, 3, 0),
            (7, 7, 0),
            (8, 8, 0),
            (9, 8, 0),
            (10, 9, 0),
            (9, 10, 0),
            (8, 10, 0),
            (7, 10, 0),
        ),
        baseline=len(PROGRAMS[4]),
    ),
    LevelSpec(
        "Rolling prism and glass turn",
        ((0, 5), "R"),
        (6, 6, "U"),
        ((7, 5, "/"), (8, 9, "/"), (2, 10, "\\")),
        ((7, 5, False), (8, 9, True)),
        ((3, 8, "U"),),
        (
            (3, 5, 0),
            (7, 4, 0),
            (7, 3, 0),
            (8, 7, 0),
            (8, 8, 0),
            (7, 10, 0),
            (6, 10, 0),
            (5, 10, 0),
            (4, 10, 0),
            (3, 10, 0),
        ),
        baseline=len(PROGRAMS[5]),
    ),
    LevelSpec(
        "Open first close later",
        ((0, 5), "R"),
        (4, 7, "U"),
        ((7, 5, "/"), (4, 6, "/"), (8, 9, "/"), (2, 10, "\\")),
        ((7, 5, False), (8, 9, True)),
        ((6, 8, "U"),),
        (
            (6, 5, 0),
            (7, 4, 0),
            (7, 3, 0),
            (8, 7, 0),
            (8, 8, 0),
            (7, 10, 0),
            (6, 10, 0),
            (5, 10, 0),
            (4, 10, 0),
            (3, 10, 0),
        ),
        baseline=len(PROGRAMS[6]),
    ),
    LevelSpec(
        "Final observatory loop",
        ((0, 5), "R"),
        (4, 7, "U"),
        ((7, 5, "/"), (4, 6, "/"), (8, 9, "/"), (2, 10, "\\"), (4, 9, "/")),
        ((7, 5, False), (8, 9, True)),
        ((6, 8, "U"),),
        (
            (6, 5, 0),
            (7, 4, 0),
            (7, 3, 0),
            (8, 7, 0),
            (8, 8, 0),
            (7, 10, 0),
            (6, 10, 0),
            (5, 10, 0),
            (4, 8, 0),
            (4, 7, 0),
        ),
        baseline=len(PROGRAMS[7]),
    ),
)


def _cell_top_left(x: int, y: int) -> tuple[int, int]:
    return BOARD_ORIGIN + x * CELL, BOARD_ORIGIN + y * CELL


def _cell_center(x: int, y: int) -> tuple[int, int]:
    px, py = _cell_top_left(x, y)
    return px + 2, py + 2


def _blank_pixels() -> np.ndarray:
    return np.full((VIEW_SIZE, VIEW_SIZE), BLACK, dtype=np.int8)


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(_blank_pixels(), name="board", x=0, y=0, layer=0, collidable=False, tags=["board"])
    click_sprites = [
        Sprite(
            np.full((CELL, CELL), -1, dtype=np.int8),
            name=f"stand_click_{x}_{y}",
            x=_cell_top_left(x, y)[0],
            y=_cell_top_left(x, y)[1],
            layer=50,
            collidable=False,
            tags=["sys_click", "sys_every_pixel"],
        )
        for x, y, _locked in spec.stands
    ]
    return Level(sprites=[board, *click_sprites], grid_size=(VIEW_SIZE, VIEW_SIZE), name=spec.name, data={"spec": spec})


def _action_id(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    return int(value)


class MirrorFog(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BLACK, BLACK),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.spec: LevelSpec = level.get_data("spec")
        px, py, facing = self.spec.player
        self.player = (px, py)
        self.facing = facing
        self.fog = {(x, y): state for x, y, state in self.spec.fog}
        self.mirrors = {(x, y): orient for x, y, orient in self.spec.mirrors}
        self.stands = {(x, y): locked for x, y, locked in self.spec.stands}
        self.prisms = {(x, y): orient for x, y, orient in self.spec.prisms}
        self.walls = set(self.spec.walls)
        self.remaining_steps = self.spec.baseline * 6
        self.step_budget = self.spec.baseline * 6
        self.last_beam: list[tuple[int, int]] = []
        self.flash_cells: dict[tuple[int, int], int] = {}
        self.animation_frames: list[np.ndarray] = []
        self._finish_after_animation = "complete"
        self._sync_board()

    def step(self) -> None:
        if self.animation_frames:
            self.board.pixels = self.animation_frames.pop(0)
            if not self.animation_frames:
                finish = self._finish_after_animation
                self._finish_after_animation = "complete"
                if finish == "next":
                    self.next_level()
                    self.complete_action()
                elif finish == "lose":
                    self.lose()
                    self.complete_action()
                else:
                    self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self._sync_board()
            self.complete_action()
            return

        accepted, prism_frames = self._apply_action()
        if not accepted:
            self.flash_cells = {self.player: RED}
            self.animation_frames = [self._render()]
            self.flash_cells = {}
            self._finish_after_animation = "complete"
            return

        self.remaining_steps -= 1
        old_fog = dict(self.fog)
        path, hits = self._trace_beam()
        self.last_beam = path
        for cell in hits:
            if self.fog[cell] < 2:
                self.fog[cell] += 1

        won = all(state != 0 for state in self.fog.values())
        lost = self.remaining_steps <= 0 and not won
        self._queue_action_frames(prism_frames, path, old_fog)
        if won:
            self._finish_after_animation = "next"
        elif lost:
            self._finish_after_animation = "lose"
        else:
            self._finish_after_animation = "complete"

    def _apply_action(self) -> tuple[bool, list[tuple[tuple[int, int], str]]]:
        action = _action_id(self.action.id)
        if action in ACTION_DIR:
            direction = ACTION_DIR[action]
            self.facing = direction
            dx, dy = DIRS[direction]
            target = (self.player[0] + dx, self.player[1] + dy)
            if target in self.mirrors:
                return self._push_or_brace_mirror(target, direction), []
            if target in self.prisms:
                return self._roll_prism(target, direction)
            if self._walkable_for_player(target):
                self.player = target
                return True, []
            return False, []

        if action == SPACE:
            dx, dy = DIRS[self.facing]
            target = (self.player[0] + dx, self.player[1] + dy)
            if target not in self.mirrors or self.stands.get(target, False):
                return False, []
            self.mirrors[target] = "/" if self.mirrors[target] == "\\" else "\\"
            return True, []

        if action == CLICK:
            data = self.action.data or {}
            cell = self._click_cell(int(data.get("x", -99)), int(data.get("y", -99)))
            if cell is None or cell not in self.stands or cell not in self.mirrors:
                return False, []
            self.stands[cell] = not self.stands[cell]
            self.flash_cells = {cell: PURPLE if self.stands[cell] else BLUE}
            return True, []

        return False, []

    def _click_cell(self, px: int, py: int) -> tuple[int, int] | None:
        x = (px - BOARD_ORIGIN) // CELL
        y = (py - BOARD_ORIGIN) // CELL
        if 0 <= x < GRID and 0 <= y < GRID:
            return x, y
        return None

    def _push_or_brace_mirror(self, cell: tuple[int, int], direction: str) -> bool:
        if self.stands.get(cell, False):
            return True
        dx, dy = DIRS[direction]
        beyond = (cell[0] + dx, cell[1] + dy)
        if self._object_passable(beyond):
            self.mirrors[beyond] = self.mirrors.pop(cell)
            self.player = cell
        return True

    def _roll_prism(self, cell: tuple[int, int], direction: str) -> tuple[bool, list[tuple[tuple[int, int], str]]]:
        del self.prisms[cell]
        pos = cell
        roll_dir = direction
        moved = False
        turned = False
        seen: set[tuple[tuple[int, int], str]] = set()
        frames: list[tuple[tuple[int, int], str]] = [(pos, roll_dir)]

        while (pos, roll_dir) not in seen:
            seen.add((pos, roll_dir))
            dx, dy = DIRS[roll_dir]
            nxt = (pos[0] + dx, pos[1] + dy)
            if self._cell_is_glass(nxt):
                roll_dir = CW[roll_dir]
                turned = True
                frames.append((pos, roll_dir))
                continue
            if self._object_passable(nxt):
                pos = nxt
                moved = True
                frames.append((pos, roll_dir))
                continue
            break

        self.prisms[pos] = roll_dir
        if moved or turned:
            self.player = cell
            return True, frames
        return False, []

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < GRID and 0 <= y < GRID

    def _is_wall(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return not self._in_bounds(cell) or x == 0 or x == GRID - 1 or y == 0 or y == GRID - 1 or cell in self.walls

    def _cell_is_glass(self, cell: tuple[int, int]) -> bool:
        return self.fog.get(cell) == 2

    def _walkable_for_player(self, cell: tuple[int, int]) -> bool:
        return (
            not self._is_wall(cell)
            and cell not in self.mirrors
            and cell not in self.prisms
            and not self._cell_is_glass(cell)
        )

    def _object_passable(self, cell: tuple[int, int]) -> bool:
        return self._walkable_for_player(cell) and cell != self.player

    def _trace_beam(self) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
        (sx, sy), direction = self.spec.source
        dx, dy = DIRS[direction]
        cell = (sx + dx, sy + dy)
        path: list[tuple[int, int]] = []
        hits: set[tuple[int, int]] = set()
        seen: set[tuple[tuple[int, int], str]] = set()

        while True:
            if (cell, direction) in seen:
                break
            seen.add((cell, direction))
            if self._is_wall(cell):
                break
            path.append(cell)
            if cell in self.fog:
                hits.add(cell)
            if cell in self.mirrors:
                direction = self._reflect(direction, self.mirrors[cell])
            elif cell in self.prisms:
                direction = self.prisms[cell]
            dx, dy = DIRS[direction]
            cell = (cell[0] + dx, cell[1] + dy)

        return path, hits

    def _reflect(self, direction: str, orient: str) -> str:
        if orient == "/":
            return {"R": "U", "U": "R", "L": "D", "D": "L"}[direction]
        return {"R": "D", "D": "R", "L": "U", "U": "L"}[direction]

    def _queue_action_frames(
        self,
        prism_frames: list[tuple[tuple[int, int], str]],
        path: list[tuple[int, int]],
        old_fog: dict[tuple[int, int], int],
    ) -> None:
        frames: list[np.ndarray] = []
        if prism_frames:
            final_prisms = dict(self.prisms)
            for pos, orient in prism_frames[1:]:
                self.prisms = {pos: orient}
                frames.append(self._render(fog_override=old_fog, beam_override=[]))
            self.prisms = final_prisms

        sweep_count = min(5, max(3, len(path)))
        for idx in range(sweep_count):
            upto = max(1, round((idx + 1) * len(path) / sweep_count))
            frames.append(self._render(fog_override=old_fog, beam_override=path[:upto], beam_core=True))
        frames.append(self._render())
        self.animation_frames = frames
        self.flash_cells = {}

    def _sync_board(self) -> None:
        self.board.pixels = self._render()

    def _render(
        self,
        *,
        fog_override: dict[tuple[int, int], int] | None = None,
        beam_override: list[tuple[int, int]] | None = None,
        beam_core: bool = False,
    ) -> np.ndarray:
        fog = self.fog if fog_override is None else fog_override
        beam = self.last_beam if beam_override is None else beam_override
        img = _blank_pixels()
        img[BOARD_ORIGIN : BOARD_ORIGIN + GRID * CELL, BOARD_ORIGIN : BOARD_ORIGIN + GRID * CELL] = FLOOR

        for y in range(GRID):
            for x in range(GRID):
                if self._is_wall((x, y)):
                    self._draw_cell_fill(img, x, y, WALL)

        for cell, state in fog.items():
            if state == 0:
                self._draw_fog(img, *cell)
            elif state == 1:
                self._draw_cell_fill(img, *cell, CLEAR)
                self._draw_clear(img, *cell)
            else:
                self._draw_glass(img, *cell)

        for x, y in self.stands:
            self._draw_stand(img, x, y, self.stands[(x, y)])

        self._draw_source(img)

        for cell in beam:
            self._draw_beam_cell(img, *cell, core=beam_core)

        for (x, y), orient in self.mirrors.items():
            self._draw_mirror(img, x, y, orient)

        for (x, y), orient in self.prisms.items():
            self._draw_prism(img, x, y, orient)

        self._draw_player(img)
        for (x, y), color in self.flash_cells.items():
            px, py = _cell_top_left(x, y)
            img[py, px : px + CELL] = color
            img[py + CELL - 1, px : px + CELL] = color
            img[py : py + CELL, px] = color
            img[py : py + CELL, px + CELL - 1] = color
        self._draw_hud(img)
        return img

    def _draw_cell_fill(self, img: np.ndarray, x: int, y: int, color: int) -> None:
        px, py = _cell_top_left(x, y)
        img[py : py + CELL, px : px + CELL] = color

    def _draw_clear(self, img: np.ndarray, x: int, y: int) -> None:
        px, py = _cell_top_left(x, y)
        img[py + 2, px + 1 : px + 4] = FLOOR
        img[py + 1 : py + 4, px + 2] = FLOOR

    def _draw_fog(self, img: np.ndarray, x: int, y: int) -> None:
        px, py = _cell_top_left(x, y)
        img[py : py + CELL, px : px + CELL] = FLOOR
        mask = ((1, 2, 2, 2, 1), (2, 2, 2, 2, 2), (2, 2, 2, 2, 1), (1, 2, 2, 2, 2), (1, 1, 2, 1, 1))
        for yy, row in enumerate(mask):
            for xx, value in enumerate(row):
                if value == 2:
                    img[py + yy, px + xx] = FOG

    def _draw_glass(self, img: np.ndarray, x: int, y: int) -> None:
        px, py = _cell_top_left(x, y)
        img[py : py + CELL, px : px + CELL] = GLASS
        img[py + 1 : py + 3, px + 1 : px + 3] = CLEAR
        img[py + 4, px + 1 : px + 5] = METAL
        img[py + 1 : py + 5, px + 4] = METAL

    def _draw_stand(self, img: np.ndarray, x: int, y: int, locked: bool) -> None:
        px, py = _cell_top_left(x, y)
        img[py, px : px + CELL] = BLUE
        img[py + CELL - 1, px : px + CELL] = BLUE
        img[py : py + CELL, px] = BLUE
        img[py : py + CELL, px + CELL - 1] = BLUE
        if locked:
            img[py + 1 : py + 4, px] = RED
            img[py + 1 : py + 4, px + 4] = RED

    def _draw_beam_cell(self, img: np.ndarray, x: int, y: int, *, core: bool) -> None:
        px, py = _cell_top_left(x, y)
        color = ORANGE if core else YELLOW
        img[py + 2, px : px + CELL] = YELLOW
        img[py + 2, px + 1 : px + 4] = color

    def _draw_source(self, img: np.ndarray) -> None:
        (x, y), direction = self.spec.source
        px, py = _cell_top_left(x, y)
        if direction == "R":
            img[py + 1 : py + 4, px + 3 : px + 5] = ORANGE
            img[py + 2, px + 4] = YELLOW
        else:
            img[py + 1 : py + 4, px + 1 : px + 4] = ORANGE

    def _draw_mirror(self, img: np.ndarray, x: int, y: int, orient: str) -> None:
        px, py = _cell_top_left(x, y)
        img[py + 1 : py + 4, px + 1 : px + 4] = METAL
        if orient == "/":
            points = ((1, 3), (2, 2), (3, 1))
        else:
            points = ((1, 1), (2, 2), (3, 3))
        for xx, yy in points:
            img[py + yy, px + xx] = YELLOW

    def _draw_prism(self, img: np.ndarray, x: int, y: int, orient: str) -> None:
        px, py = _cell_top_left(x, y)
        masks = {
            "U": ((2, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)),
            "D": ((1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (2, 3)),
            "L": ((3, 1), (2, 2), (3, 2), (1, 2), (3, 3)),
            "R": ((1, 1), (1, 2), (2, 2), (3, 2), (1, 3)),
        }
        for xx, yy in masks[orient]:
            img[py + yy, px + xx] = PURPLE
        img[py + 2, px + 2] = GLASS

    def _draw_player(self, img: np.ndarray) -> None:
        x, y = self.player
        px, py = _cell_top_left(x, y)
        img[py + 1 : py + 4, px + 1 : px + 4] = GREEN
        img[py + 4, px + 1 : px + 4] = BLACK
        if self.facing == "U":
            img[py, px + 2] = GREEN
        elif self.facing == "D":
            img[py + 4, px + 2] = GREEN
        elif self.facing == "L":
            img[py + 2, px] = GREEN
        else:
            img[py + 2, px + 4] = GREEN

    def _draw_hud(self, img: np.ndarray) -> None:
        img[62:64, 2:62] = BLACK
        filled = max(0, min(60, round(60 * self.remaining_steps / max(1, self.step_budget))))
        if filled:
            img[62:64, 2 : 2 + filled] = ORANGE
