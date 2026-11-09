from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

VIEW_SIZE = 64
CELL_SIZE = 6
COLOR_BACKGROUND = 4
COLOR_FLOOR = 1
COLOR_WALL = 3
COLOR_BLUE = 9
COLOR_BLUE_HI = 10
COLOR_MAGENTA = 6
COLOR_MAGENTA_HI = 7
COLOR_HAZARD = 8
COLOR_HAZARD_HI = 13
COLOR_YELLOW = 11
COLOR_GREEN = 14
COLOR_PLUG = 12
COLOR_BAR_EMPTY = 0
COLOR_BAR_FULL = 10

MOVE_VECTORS = {
    ACTION_UP: ((0, -1), (0, -1)),
    ACTION_DOWN: ((0, 1), (0, 1)),
    ACTION_LEFT: ((-1, 0), (1, 0)),
    ACTION_RIGHT: ((1, 0), (-1, 0)),
}


@dataclass(frozen=True)
class BlockerSpec:
    socket_a: tuple[int, int]
    socket_b: tuple[int, int]
    initial: str


@dataclass(frozen=True)
class GateFamilySpec:
    pads: tuple[tuple[int, int], ...]
    gates: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    size: tuple[int, int]
    left_start: tuple[int, int]
    right_start: tuple[int, int]
    step_budget: int
    walls: tuple[tuple[int, int], ...] = ()
    hazards: tuple[tuple[int, int], ...] = ()
    yellow: GateFamilySpec = GateFamilySpec((), ())
    green: GateFamilySpec = GateFamilySpec((), ())
    blocker: BlockerSpec | None = None


LEVEL_SPECS = (
    LevelSpec("The crossing rule", (9, 7), (1, 3), (6, 3), 18),
    LevelSpec("One block hits", (9, 7), (2, 4), (6, 5), 18, walls=((2, 3), (2, 5), (6, 6))),
    LevelSpec("Red cells reset", (9, 6), (1, 5), (7, 5), 30, walls=((1, 4), (2, 4), (5, 3), (7, 4)), hazards=((5, 5),)),
    LevelSpec(
        "Plug obstacle tool",
        (9, 7),
        (4, 3),
        (6, 6),
        24,
        walls=tuple(
            (x, y)
            for y in range(7)
            for x in range(9)
            if (x, y)
            not in {(2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (2, 4), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (6, 6)}
        ),
        blocker=BlockerSpec((4, 5), (2, 4), "A"),
    ),
    LevelSpec(
        "Gate wall pad",
        (9, 6),
        (2, 5),
        (8, 5),
        42,
        walls=((1, 5), (7, 5), (8, 3), (5, 3)),
        yellow=GateFamilySpec(pads=((2, 5),), gates=((7, 4), (6, 4))),
        green=GateFamilySpec(pads=((5, 4),), gates=((2, 4), (3, 5))),
    ),
    LevelSpec(
        "Hold cover release",
        (9, 9),
        (2, 6),
        (8, 8),
        84,
        walls=((1, 6), (2, 7), (6, 7), (5, 5)),
        hazards=((8, 6), (3, 4)),
        yellow=GateFamilySpec(pads=((2, 6),), gates=((7, 6), (6, 4))),
        green=GateFamilySpec(pads=((4, 4),), gates=((2, 5), (3, 6))),
        blocker=BlockerSpec((3, 4), (4, 3), "B"),
    ),
)


def _solid_pixels(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), color, dtype=np.int8)


def _action_id(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    return int(value)


class MirroredView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: Mirrored | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BACKGROUND
        self._draw_board(frame, game)
        self._draw_step_bar(frame, game.remaining_steps, game.step_budget)
        return frame

    def _draw_board(self, frame: np.ndarray, game: Mirrored) -> None:
        spec = game.spec
        for y in range(spec.size[1]):
            for x in range(spec.size[0]):
                self._fill_cell(frame, game, (x, y), COLOR_FLOOR)

        for cell in spec.hazards:
            self._fill_cell(frame, game, cell, COLOR_HAZARD)
            x0, y0 = game.cell_top_left(cell)
            frame[y0 + 2 : y0 + 4, x0 + 2 : x0 + 4] = COLOR_HAZARD_HI

        self._draw_gate_family(frame, game, spec.yellow, COLOR_YELLOW, game.is_family_open("Y"))
        self._draw_gate_family(frame, game, spec.green, COLOR_GREEN, game.is_family_open("G"))

        for cell in spec.walls:
            self._fill_cell(frame, game, cell, COLOR_WALL)
            x0, y0 = game.cell_top_left(cell)
            frame[y0 + 1 : y0 + CELL_SIZE, x0 + CELL_SIZE - 1] = COLOR_BACKGROUND
            frame[y0 + CELL_SIZE - 1, x0 + 1 : x0 + CELL_SIZE] = COLOR_BACKGROUND

        if spec.blocker is not None:
            for socket in (spec.blocker.socket_a, spec.blocker.socket_b):
                x0, y0 = game.cell_top_left(socket)
                frame[y0 + 1 : y0 + 5, x0 + 1] = COLOR_PLUG
                frame[y0 + 1 : y0 + 5, x0 + 4] = COLOR_PLUG
                frame[y0 + 1, x0 + 1 : x0 + 5] = COLOR_PLUG
                frame[y0 + 4, x0 + 1 : x0 + 5] = COLOR_PLUG
            self._fill_cell(frame, game, game.active_blocker_cell(), COLOR_PLUG, inset=1)
            x0, y0 = game.cell_top_left(game.active_blocker_cell())
            frame[y0 + 2 : y0 + 4, x0 + 2 : x0 + 4] = COLOR_HAZARD_HI

        if game.left_cell == game.right_cell:
            self._draw_merged_blocks(frame, game, game.left_cell)
        else:
            self._draw_block(frame, game, game.left_cell, COLOR_BLUE, COLOR_BLUE_HI)
            self._draw_block(frame, game, game.right_cell, COLOR_MAGENTA, COLOR_MAGENTA_HI)

    def _draw_gate_family(
        self, frame: np.ndarray, game: Mirrored, family: GateFamilySpec, color: int, is_open: bool
    ) -> None:
        for pad in family.pads:
            x0, y0 = game.cell_top_left(pad)
            frame[y0 + 1 : y0 + 5, x0 + 1 : x0 + 5] = color
            frame[y0 + 2 : y0 + 4, x0 + 2 : x0 + 4] = COLOR_FLOOR
        for gate in family.gates:
            x0, y0 = game.cell_top_left(gate)
            if is_open:
                self._fill_cell(frame, game, gate, COLOR_FLOOR)
                frame[y0 + 2 : y0 + 4, x0 + 1 : x0 + 5] = color
            else:
                self._fill_cell(frame, game, gate, color)
                frame[y0 + 1 : y0 + 5, x0 + 2 : x0 + 4] = COLOR_BACKGROUND

    def _draw_block(self, frame: np.ndarray, game: Mirrored, cell: tuple[int, int], color: int, hi: int) -> None:
        x0, y0 = game.cell_top_left(cell)
        frame[y0 + 1 : y0 + 6, x0 + 1 : x0 + 6] = color
        frame[y0 + 1, x0 + 1 : x0 + 5] = hi
        frame[y0 + 2, x0 + 1] = hi

    def _draw_merged_blocks(self, frame: np.ndarray, game: Mirrored, cell: tuple[int, int]) -> None:
        x0, y0 = game.cell_top_left(cell)
        frame[y0 + 1 : y0 + 6, x0 + 1 : x0 + 4] = COLOR_BLUE
        frame[y0 + 1 : y0 + 6, x0 + 4 : x0 + 6] = COLOR_MAGENTA
        frame[y0 + 1, x0 + 1 : x0 + 4] = COLOR_BLUE_HI
        frame[y0 + 1, x0 + 4 : x0 + 6] = COLOR_MAGENTA_HI

    def _fill_cell(
        self, frame: np.ndarray, game: Mirrored, cell: tuple[int, int], color: int, *, inset: int = 0
    ) -> None:
        x0, y0 = game.cell_top_left(cell)
        frame[y0 + inset : y0 + CELL_SIZE - inset, x0 + inset : x0 + CELL_SIZE - inset] = color

    def _draw_step_bar(self, frame: np.ndarray, remaining: int, budget: int) -> None:
        frame[61:63, 8:56] = COLOR_BAR_EMPTY
        filled = max(0, min(48, round(48 * remaining / max(1, budget))))
        if filled:
            frame[61:63, 8 : 8 + filled] = COLOR_BAR_FULL


class Mirrored(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = MirroredView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "mirrored-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._view]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed,
        )
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec = level.get_data("spec")
        self.board_left = (VIEW_SIZE - self.spec.size[0] * CELL_SIZE) // 2
        self.board_top = 4 if self.spec.size[1] == 9 else max(3, (60 - self.spec.size[1] * CELL_SIZE) // 2)
        self.left_cell = tuple(self.spec.left_start)
        self.right_cell = tuple(self.spec.right_start)
        self.blocker_on = self.spec.blocker.initial if self.spec.blocker is not None else ""
        self.step_budget = int(self.spec.step_budget)
        self.remaining_steps = self.step_budget

    def cell_top_left(self, cell: tuple[int, int]) -> tuple[int, int]:
        return self.board_left + cell[0] * CELL_SIZE, self.board_top + cell[1] * CELL_SIZE

    def is_family_open(self, family_name: str) -> bool:
        family = self.spec.yellow if family_name == "Y" else self.spec.green
        pads = set(family.pads)
        return self.left_cell in pads or self.right_cell in pads

    def active_blocker_cell(self) -> tuple[int, int]:
        blocker = self.spec.blocker
        if blocker is None:
            raise RuntimeError("No blocker is active for this level.")
        return blocker.socket_a if self.blocker_on == "A" else blocker.socket_b

    def _screen_to_cell(self, x: int, y: int) -> tuple[int, int] | None:
        cell_x = (x - self.board_left) // CELL_SIZE
        cell_y = (y - self.board_top) // CELL_SIZE
        if 0 <= cell_x < self.spec.size[0] and 0 <= cell_y < self.spec.size[1]:
            return int(cell_x), int(cell_y)
        return None

    def _reset_attempt(self) -> None:
        self.left_cell = tuple(self.spec.left_start)
        self.right_cell = tuple(self.spec.right_start)
        self.blocker_on = self.spec.blocker.initial if self.spec.blocker is not None else ""
        self.remaining_steps = self.step_budget

    def _solid_for_step(self, cell: tuple[int, int], open_families: set[str]) -> bool:
        x, y = cell
        if x < 0 or y < 0 or x >= self.spec.size[0] or y >= self.spec.size[1]:
            return True
        if cell in self.spec.walls:
            return True
        if self.spec.blocker is not None and cell == self.active_blocker_cell():
            return True
        if "Y" not in open_families and cell in self.spec.yellow.gates:
            return True
        return "G" not in open_families and cell in self.spec.green.gates

    def _open_families(self) -> set[str]:
        out: set[str] = set()
        if self.is_family_open("Y"):
            out.add("Y")
        if self.is_family_open("G"):
            out.add("G")
        return out

    def _move(self, action: int) -> None:
        open_families = self._open_families()
        old_left = self.left_cell
        old_right = self.right_cell
        left_vector, right_vector = MOVE_VECTORS[action]
        left_target = (old_left[0] + left_vector[0], old_left[1] + left_vector[1])
        right_target = (old_right[0] + right_vector[0], old_right[1] + right_vector[1])

        left_moved = not self._solid_for_step(left_target, open_families)
        right_moved = not self._solid_for_step(right_target, open_families)
        self.left_cell = left_target if left_moved else old_left
        self.right_cell = right_target if right_moved else old_right

        if self.left_cell in self.spec.hazards or self.right_cell in self.spec.hazards:
            self._reset_attempt()
            return

        same_cell = self.left_cell == self.right_cell
        crossed = (
            action in {ACTION_LEFT, ACTION_RIGHT}
            and left_moved
            and right_moved
            and self.left_cell == old_right
            and self.right_cell == old_left
            and old_left[1] == old_right[1]
        )
        if same_cell or crossed:
            self.next_level()

    def _click(self) -> None:
        blocker = self.spec.blocker
        if blocker is None:
            return
        data = self.action.data or {}
        cell = self._screen_to_cell(int(data.get("x", 0)), int(data.get("y", 0)))
        if cell not in {blocker.socket_a, blocker.socket_b}:
            return
        target = "B" if self.blocker_on == "A" else "A"
        target_cell = blocker.socket_b if target == "B" else blocker.socket_a
        if target_cell in {self.left_cell, self.right_cell}:
            return
        self.blocker_on = target

    def _spend_step_or_fail(self) -> bool:
        self.remaining_steps -= 1
        if self.remaining_steps > 0:
            return False
        self.lose()
        return True

    def step(self) -> None:
        action = _action_id(self.action.id)
        if action == _action_id(GameAction.RESET):
            self.complete_action()
            return

        if action == ACTION_SPACE:
            self._reset_attempt()
            self.complete_action()
            return

        if self._spend_step_or_fail():
            self.complete_action()
            return

        if action in MOVE_VECTORS:
            self._move(action)
        elif action == ACTION_CLICK:
            self._click()
        self.complete_action()


def _click_sprite(spec: LevelSpec, cell: tuple[int, int]) -> Sprite:
    left = (VIEW_SIZE - spec.size[0] * CELL_SIZE) // 2
    top = 4 if spec.size[1] == 9 else max(3, (60 - spec.size[1] * CELL_SIZE) // 2)
    pixels = np.full((CELL_SIZE, CELL_SIZE), -1, dtype=np.int8)
    return Sprite(
        pixels,
        name=f"click_{cell[0]}_{cell[1]}",
        x=left + cell[0] * CELL_SIZE,
        y=top + cell[1] * CELL_SIZE,
        layer=50,
        collidable=False,
        tags=["sys_click", "sys_every_pixel"],
    )


def _build_level(spec: LevelSpec) -> Level:
    sprites = [
        Sprite(_solid_pixels(VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND), name="background", layer=-100, collidable=False)
    ]
    if spec.blocker is not None:
        sprites.append(_click_sprite(spec, spec.blocker.socket_a))
        sprites.append(_click_sprite(spec, spec.blocker.socket_b))
    return Level(grid_size=(VIEW_SIZE, VIEW_SIZE), sprites=sprites, name=spec.name, data={"spec": spec})
