from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

VIEW_SIZE = 64

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
MOVE_DELTAS = {ACTION_UP: (0, -3), ACTION_DOWN: (0, 3), ACTION_LEFT: (-3, 0), ACTION_RIGHT: (3, 0)}

COLOR_WHITE = 0
COLOR_BACKGROUND = 1
COLOR_DARK = 4
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_GREEN = 14
COLOR_PURPLE = 15

SHAPE_PLUS = "PLUS11"
SHAPE_X = "X11"
SHAPE_RING = "RING11"
ANCHOR_MIN = -9
ANCHOR_MAX = 72
PAD_ACTIVATION_RADIUS = 1


@dataclass
class ReticleState:
    shape: str
    color: int
    x: int
    y: int


@dataclass(frozen=True)
class ReticleSpec:
    shape: str
    color: int
    anchor: tuple[int, int]


@dataclass(frozen=True)
class PadSpec:
    kind: str
    center: tuple[int, int]
    payload: int | str


@dataclass(frozen=True)
class LevelSpec:
    name: str
    reticles: tuple[ReticleSpec, ...]
    targets: tuple[tuple[int, int, int], ...]
    pads: tuple[PadSpec, ...]
    step_budget: int


LEVEL_SPECS = (
    LevelSpec(
        "Single Alignment",
        (ReticleSpec(SHAPE_PLUS, COLOR_BLUE, (20, 32)),),
        ((27, 32, COLOR_BLUE), (37, 32, COLOR_BLUE), (32, 37, COLOR_BLUE)),
        (),
        24,
    ),
    LevelSpec(
        "Two Reticles",
        (ReticleSpec(SHAPE_PLUS, COLOR_RED, (17, 22)), ReticleSpec(SHAPE_PLUS, COLOR_BLUE, (44, 37))),
        (
            (18, 25, COLOR_RED),
            (28, 25, COLOR_RED),
            (23, 30, COLOR_RED),
            (38, 29, COLOR_BLUE),
            (33, 34, COLOR_BLUE),
            (43, 34, COLOR_BLUE),
        ),
        (),
        42,
    ),
    LevelSpec(
        "X And Ring",
        (ReticleSpec(SHAPE_X, COLOR_GREEN, (24, 18)), ReticleSpec(SHAPE_RING, COLOR_YELLOW, (51, 21))),
        (
            (31, 25, COLOR_GREEN),
            (35, 25, COLOR_GREEN),
            (31, 29, COLOR_GREEN),
            (35, 29, COLOR_GREEN),
            (34, 22, COLOR_YELLOW),
            (44, 22, COLOR_YELLOW),
            (34, 32, COLOR_YELLOW),
            (44, 32, COLOR_YELLOW),
            (39, 22, COLOR_YELLOW),
        ),
        (),
        78,
    ),
    LevelSpec(
        "Left Edge Clip",
        (
            ReticleSpec(SHAPE_PLUS, COLOR_RED, (20, 20)),
            ReticleSpec(SHAPE_PLUS, COLOR_BLUE, (25, 44)),
            ReticleSpec(SHAPE_X, COLOR_GREEN, (50, 14)),
            ReticleSpec(SHAPE_RING, COLOR_YELLOW, (51, 50)),
        ),
        (
            (3, 32, COLOR_RED),
            (8, 27, COLOR_RED),
            (13, 32, COLOR_RED),
            (29, 35, COLOR_BLUE),
            (34, 30, COLOR_BLUE),
            (39, 35, COLOR_BLUE),
            (36, 18, COLOR_GREEN),
            (46, 18, COLOR_GREEN),
            (36, 28, COLOR_GREEN),
            (46, 28, COLOR_GREEN),
            (46, 30, COLOR_YELLOW),
            (56, 30, COLOR_YELLOW),
            (46, 40, COLOR_YELLOW),
            (56, 40, COLOR_YELLOW),
            (51, 30, COLOR_YELLOW),
        ),
        (),
        126,
    ),
    LevelSpec(
        "Recolor Ring",
        (
            ReticleSpec(SHAPE_X, COLOR_BLUE, (23, 29)),
            ReticleSpec(SHAPE_RING, COLOR_YELLOW, (44, 20)),
            ReticleSpec(SHAPE_PLUS, COLOR_ORANGE, (14, 14)),
        ),
        (
            (30, 30, COLOR_BLUE),
            (34, 30, COLOR_BLUE),
            (30, 34, COLOR_BLUE),
            (34, 34, COLOR_BLUE),
            (27, 27, COLOR_GREEN),
            (37, 27, COLOR_GREEN),
            (27, 37, COLOR_GREEN),
            (37, 37, COLOR_GREEN),
            (32, 27, COLOR_GREEN),
            (12, 47, COLOR_GREEN),
            (17, 42, COLOR_GREEN),
            (22, 47, COLOR_GREEN),
        ),
        (PadSpec("recolor", (50, 35), COLOR_GREEN),),
        132,
    ),
    LevelSpec(
        "Transform Ring",
        (ReticleSpec(SHAPE_PLUS, COLOR_RED, (20, 26)), ReticleSpec(SHAPE_PLUS, COLOR_GREEN, (50, 26))),
        (
            (29, 32, COLOR_RED),
            (35, 32, COLOR_RED),
            (32, 29, COLOR_RED),
            (27, 27, COLOR_GREEN),
            (37, 27, COLOR_GREEN),
            (27, 37, COLOR_GREEN),
            (37, 37, COLOR_GREEN),
        ),
        (PadSpec("transform", (50, 38), SHAPE_RING),),
        90,
    ),
    LevelSpec(
        "Dependency Network",
        (
            ReticleSpec(SHAPE_X, COLOR_BLUE, (20, 35)),
            ReticleSpec(SHAPE_PLUS, COLOR_RED, (20, 17)),
            ReticleSpec(SHAPE_PLUS, COLOR_YELLOW, (47, 47)),
            ReticleSpec(SHAPE_PLUS, COLOR_ORANGE, (10, 29)),
        ),
        (
            (46, 46, COLOR_BLUE),
            (50, 46, COLOR_BLUE),
            (46, 50, COLOR_BLUE),
            (50, 50, COLOR_BLUE),
            (58, 50, COLOR_BLUE),
            (27, 27, COLOR_GREEN),
            (37, 27, COLOR_GREEN),
            (27, 37, COLOR_GREEN),
            (37, 37, COLOR_GREEN),
            (32, 27, COLOR_GREEN),
            (37, 32, COLOR_GREEN),
            (3, 20, COLOR_RED),
            (8, 15, COLOR_RED),
            (13, 20, COLOR_RED),
        ),
        (
            PadSpec("transform", (50, 38), SHAPE_RING),
            PadSpec("recolor", (56, 8), COLOR_GREEN),
            PadSpec("recolor", (8, 56), COLOR_BLUE),
        ),
        222,
    ),
)


def _shape_offsets(shape: str) -> tuple[tuple[int, int], ...]:
    if shape == SHAPE_PLUS:
        return tuple((d, 0) for d in range(-5, 6)) + tuple((0, d) for d in range(-5, 6) if d != 0)
    if shape == SHAPE_X:
        coords = {(d, d) for d in range(-5, 6)}
        coords.update((d, -d) for d in range(-5, 6))
        return tuple(sorted(coords))
    if shape == SHAPE_RING:
        return tuple((dx, dy) for dy in range(-5, 6) for dx in range(-5, 6) if max(abs(dx), abs(dy)) == 5)
    raise ValueError(f"Unknown reticle shape: {shape}")


SHAPE_OFFSETS = {shape: _shape_offsets(shape) for shape in (SHAPE_PLUS, SHAPE_X, SHAPE_RING)}


def _action_id(action_id: object) -> int:
    return int(getattr(action_id, "value", action_id))


def _visible(x: int, y: int) -> bool:
    return 0 <= x < VIEW_SIZE and 0 <= y < VIEW_SIZE


def _make_level(spec: LevelSpec) -> Level:
    canvas = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8),
        name="canvas",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["canvas"],
    )
    return Level(sprites=[canvas], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name)


class Reticle(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_make_level(spec) for spec in LEVEL_SPECS]
        self.canvas: Sprite | None = None
        self.spec = LEVEL_SPECS[0]
        self.reticles: list[ReticleState] = []
        self.selected_index = 0
        self.remaining_steps = 1
        self.step_budget = 1
        self._invalid_flash = False
        super().__init__(
            "reticle",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.canvas = self.current_level.get_sprites_by_tag("canvas")[0]
        self.spec = level.get_data("spec")
        self.reticles = [
            ReticleState(item.shape, item.color, item.anchor[0], item.anchor[1]) for item in self.spec.reticles
        ]
        self.selected_index = 0
        self.step_budget = int(self.spec.step_budget)
        self.remaining_steps = self.step_budget
        self._invalid_flash = False
        self._sync_canvas()

    def _sync_canvas(self) -> None:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8)
        self._draw_step_bar(frame)
        self._draw_pads(frame)
        target_colors = {(x, y): color for x, y, color in self.spec.targets if _visible(x, y)}
        for x, y, color in self.spec.targets:
            if _visible(x, y):
                frame[y, x] = color
        for reticle in self.reticles:
            for dx, dy in SHAPE_OFFSETS[reticle.shape]:
                x = reticle.x + dx
                y = reticle.y + dy
                target_color = target_colors.get((x, y))
                if _visible(x, y) and (target_color is None or target_color == reticle.color):
                    frame[y, x] = reticle.color
        self._draw_target_frames(frame)
        self._draw_selection(frame)
        assert self.canvas is not None
        self.canvas.pixels = frame

    def _draw_step_bar(self, frame: np.ndarray) -> None:
        frame[0:2, 0:VIEW_SIZE] = COLOR_DARK
        filled = max(0, min(VIEW_SIZE, int(VIEW_SIZE * self.remaining_steps / max(1, self.step_budget))))
        if filled:
            frame[0:2, 0:filled] = COLOR_GREEN

    def _draw_pads(self, frame: np.ndarray) -> None:
        for pad in self.spec.pads:
            x, y = pad.center
            if pad.kind == "recolor":
                color = int(pad.payload)
                for dx, dy in (
                    (0, -2),
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                    (-2, 0),
                    (-1, 0),
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (0, 1),
                    (-1, 1),
                    (1, 1),
                    (0, 2),
                ):
                    px, py = x + dx, y + dy
                    if _visible(px, py):
                        frame[py, px] = color
                for dx, dy in ((0, -3), (-3, 0), (3, 0), (0, 3), (0, 0)):
                    px, py = x + dx, y + dy
                    if _visible(px, py):
                        frame[py, px] = COLOR_DARK
            elif pad.kind == "transform":
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if max(abs(dx), abs(dy)) != 3:
                            continue
                        px, py = x + dx, y + dy
                        if _visible(px, py):
                            frame[py, px] = COLOR_PURPLE
                if _visible(x, y):
                    frame[y, x] = COLOR_DARK

    def _draw_target_frames(self, frame: np.ndarray) -> None:
        for x, y, _color in self.spec.targets:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    px, py = x + dx, y + dy
                    if _visible(px, py):
                        frame[py, px] = COLOR_DARK

    def _draw_selection(self, frame: np.ndarray) -> None:
        selected = self.reticles[self.selected_index]
        if _visible(selected.x, selected.y):
            frame[selected.y, selected.x] = COLOR_WHITE
            if self._invalid_flash:
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    px, py = selected.x + dx, selected.y + dy
                    if _visible(px, py):
                        frame[py, px] = COLOR_WHITE

    def _spend_step(self) -> None:
        self.remaining_steps -= 1

    def _move_selected(self, dx: int, dy: int) -> bool:
        selected = self.reticles[self.selected_index]
        next_x = selected.x + dx
        next_y = selected.y + dy
        if next_x < ANCHOR_MIN or next_x > ANCHOR_MAX or next_y < ANCHOR_MIN or next_y > ANCHOR_MAX:
            self._invalid_flash = True
            return False
        selected.x = next_x
        selected.y = next_y
        self._invalid_flash = False
        self._apply_pad(selected)
        return True

    def _apply_pad(self, selected: ReticleState) -> None:
        for pad in self.spec.pads:
            if not self._selected_on_pad(selected, pad):
                continue
            if pad.kind == "transform":
                selected.shape = str(pad.payload)
        for pad in self.spec.pads:
            if self._selected_on_pad(selected, pad) and pad.kind == "recolor":
                selected.color = int(pad.payload)

    def _selected_on_pad(self, selected: ReticleState, pad: PadSpec) -> bool:
        px, py = pad.center
        return abs(selected.x - px) <= PAD_ACTIVATION_RADIUS and abs(selected.y - py) <= PAD_ACTIVATION_RADIUS

    def _target_passes(self, target: tuple[int, int, int]) -> bool:
        tx, ty, required_color = target
        for reticle in self.reticles:
            if reticle.color != required_color:
                continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (tx + dx - reticle.x, ty + dy - reticle.y) in SHAPE_OFFSETS[reticle.shape]:
                        return True
        return False

    def _is_solved(self) -> bool:
        return all(self._target_passes(target) for target in self.spec.targets)

    def step(self) -> None:
        action = _action_id(self.action.id)
        if action == int(GameAction.RESET.value):
            self._sync_canvas()
            self.complete_action()
            return

        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._move_selected(dx, dy)
        elif action == ACTION_SPACE:
            self.selected_index = (self.selected_index + 1) % len(self.reticles)
            self._invalid_flash = False
        else:
            self._invalid_flash = False

        self._spend_step()
        self._sync_canvas()
        if self._is_solved():
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.lose()
        self.complete_action()
