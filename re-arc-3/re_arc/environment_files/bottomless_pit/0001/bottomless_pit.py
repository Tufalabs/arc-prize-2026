from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, Sprite
from arcengine.enums import BlockingMode

GAME_ID = "bottomless_pit-0001"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

BOARD_W = 11
VIEW_H = 11
TILE = 5
ORIGIN_X = 4
ORIGIN_Y = 4
CANVAS = 64

VOID = 0
PALE = 1
OUTLINE = 3
STONE = 4
DETAIL = 5
RED_INACTIVE = 7
RED_ACTIVE = 8
PLAYER = 9
GOAL = 10
ORANGE = 12
TRAP = 13
GREEN = 14
PURPLE = 15

DIR_DOWN = 1
DIR_UP = -1

BASELINE_ACTIONS = [8, 5, 7, 6, 13, 10, 17, 17, 27]
STEP_BUDGETS = [count * 6 for count in BASELINE_ACTIONS]


@dataclass(frozen=True)
class ComponentSpec:
    kind: str
    cells: tuple[tuple[int, int], ...]
    active: bool = True
    direction: int = 1


@dataclass(frozen=True)
class LevelSpec:
    name: str
    height: int
    start: tuple[int, int]
    goal: tuple[tuple[int, int], ...]
    stone: tuple[tuple[int, int], ...] = ()
    traps: tuple[tuple[int, int], ...] = ()
    components: tuple[ComponentSpec, ...] = ()


@dataclass
class ComponentState:
    cid: str
    kind: str
    cells: set[tuple[int, int]]
    active: bool = True
    consumed: bool = False
    direction: int = 1


def _cells(xs: range, y: int) -> tuple[tuple[int, int], ...]:
    return tuple((x, y) for x in xs)


LEVEL_SPECS = (
    LevelSpec(
        name="First sideways commitment",
        height=12,
        start=(1, 1),
        goal=((9, 9),),
        stone=(*_cells(range(3), 2), *_cells(range(3, 6), 6), *_cells(range(6, 9), 10)),
    ),
    LevelSpec(
        name="Breaking the floor",
        height=12,
        start=(5, 1),
        goal=((8, 9),),
        stone=((4, 1), (6, 1), (4, 5), (6, 5), *_cells(range(5, 8), 10)),
        components=(ComponentSpec("green", _cells(range(4, 7), 2)), ComponentSpec("green", _cells(range(4, 7), 6))),
    ),
    LevelSpec(
        name="Green is not always safe",
        height=8,
        start=(1, 1),
        goal=((7, 4),),
        stone=(*_cells(range(3), 2), *_cells(range(5, 7), 5)),
        traps=((3, 6), (4, 6)),
        components=(ComponentSpec("green", ((3, 5), (4, 5))), ComponentSpec("green", ((4, 4), (5, 4)))),
    ),
    LevelSpec(
        name="Orange catches",
        height=13,
        start=(5, 1),
        goal=((9, 8),),
        stone=((4, 1), (6, 1), *_cells(range(5, 7), 6)),
        traps=((7, 10), (8, 10)),
        components=(
            ComponentSpec("orange", _cells(range(4, 7), 2), active=True),
            ComponentSpec("orange", ((7, 9), (8, 9)), active=False),
        ),
    ),
    LevelSpec(
        name="Editing below the camera",
        height=19,
        start=(1, 1),
        goal=((9, 17),),
        stone=(*_cells(range(4), 2), *_cells(range(6, 9), 18)),
        traps=((4, 16), (5, 16)),
        components=(
            ComponentSpec("orange", ((4, 14), (5, 14)), active=False),
            ComponentSpec("green", ((6, 13), (6, 14))),
        ),
    ),
    LevelSpec(
        name="First gravity reversal",
        height=13,
        start=(1, 1),
        goal=((10, 10),),
        stone=(*_cells(range(4), 2), *_cells(range(3, 5), 9), *_cells(range(5, 9), 6), (9, 11)),
        traps=((5, 10),),
        components=(
            ComponentSpec("red", ((5, 7), (5, 8)), active=False),
            ComponentSpec("red", ((9, 7), (9, 8)), active=True),
        ),
    ),
    LevelSpec(
        name="Red can help or hurt",
        height=18,
        start=(1, 1),
        goal=((9, 14),),
        stone=(*_cells(range(4), 2), *_cells(range(4, 6), 10), *_cells(range(8, 10), 4)),
        traps=((6, 5), (8, 12), (10, 17)),
        components=(
            ComponentSpec("orange", ((6, 7), (7, 7)), active=False),
            ComponentSpec("orange", ((9, 15), (10, 15)), active=False),
            ComponentSpec("red", ((6, 8), (6, 9)), active=True),
            ComponentSpec("red", ((8, 6), (9, 6)), active=True),
            ComponentSpec("red", ((10, 5), (10, 6)), active=False),
        ),
    ),
    LevelSpec(
        name="Builders grow supports",
        height=19,
        start=(1, 1),
        goal=((9, 14),),
        stone=_cells(range(4), 2),
        traps=((6, 12), (7, 12), (8, 5), (10, 17)),
        components=(
            ComponentSpec("builder", ((4, 10), (5, 10)), direction=1),
            ComponentSpec("builder", ((9, 7), (10, 7)), direction=-1),
            ComponentSpec("builder", ((8, 15), (9, 15)), direction=1),
            ComponentSpec("red", ((8, 8), (8, 9)), active=True),
            ComponentSpec("red", ((10, 8), (10, 9)), active=False),
        ),
    ),
    LevelSpec(
        name="Final dependency network",
        height=25,
        start=(1, 1),
        goal=((5, 19), (5, 20)),
        stone=_cells(range(4), 2),
        traps=((4, 12), (5, 12), (8, 12), (9, 5), (10, 24), (6, 23), (7, 23)),
        components=(
            ComponentSpec("red", ((4, 5), (4, 6)), active=True),
            ComponentSpec("red", ((9, 8), (9, 9)), active=True),
            ComponentSpec("red", ((10, 7), (10, 8)), active=False),
            ComponentSpec("orange", ((4, 9), (5, 9)), active=False),
            ComponentSpec("orange", ((9, 6), (10, 6)), active=False),
            ComponentSpec("orange", ((6, 21), (7, 21)), active=False),
            ComponentSpec("green", ((6, 7), (6, 8))),
            ComponentSpec("green", ((6, 19), (6, 20))),
            ComponentSpec("builder", ((6, 9), (7, 9)), direction=1),
            ComponentSpec("builder", ((8, 21), (9, 21)), direction=1),
        ),
    ),
)


class BottomlessPit(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [
            Level(
                sprites=[_screen_sprite()],
                grid_size=(CANVAS, CANVAS),
                data={"spec": spec, "budget": STEP_BUDGETS[idx]},
                name=f"{idx + 1}. {spec.name}",
            )
            for idx, spec in enumerate(LEVEL_SPECS)
        ]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, CANVAS, CANVAS, background=VOID, letter_box=VOID),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.step_budget: int = int(level.get_data("budget"))
        self.remaining_steps: int = self.step_budget
        self.player = tuple(self.spec.start)
        self.gravity = DIR_DOWN
        self.camera_top = 0
        self._flash_cells: set[tuple[int, int]] = set()
        self._falling = False
        self._pending_budget_check = False
        self._screen = self.current_level.get_sprites_by_tag("screen")[0]
        self.components: list[ComponentState] = []
        for idx, component in enumerate(self.spec.components):
            self.components.append(
                ComponentState(
                    cid=f"c{idx}",
                    kind=component.kind,
                    cells=set(component.cells),
                    active=component.active,
                    direction=component.direction,
                )
            )
        self._sync_screen()

    def step(self) -> None:
        if self._falling:
            self._advance_fall_animation()
            return

        if self.action.id == GameAction.RESET:
            self._falling = False
            self._pending_budget_check = False
            self._sync_screen()
            self.complete_action()
            return

        self._flash_cells = set()
        accepted = self._apply_action(self.action)
        if accepted:
            self.remaining_steps -= 1

        self._sync_screen()
        if self._state_is_terminal():
            return
        if self._falling:
            self._pending_budget_check = accepted
            return
        self._finish_resolved_action(accepted)

    def _finish_resolved_action(self, accepted: bool) -> None:
        if accepted and self.remaining_steps <= 0:
            self.lose()
            self.complete_action()
            return
        self.complete_action()

    def _state_is_terminal(self) -> bool:
        return self._state.name in {"WIN", "GAME_OVER"}

    def _apply_action(self, action: ActionInput) -> bool:
        aid = action.id.value if isinstance(action.id, GameAction) else int(action.id)
        if aid == ACTION_UP:
            self.camera_top = max(0, self.camera_top - 3)
            return True
        if aid == ACTION_DOWN:
            self.camera_top = min(max(0, self.spec.height - VIEW_H), self.camera_top + 3)
            return True
        if aid == ACTION_SPACE:
            self._center_camera()
            return True
        if aid in (ACTION_LEFT, ACTION_RIGHT):
            return self._move_horizontal(-1 if aid == ACTION_LEFT else 1)
        if aid == ACTION_CLICK:
            return self._click(action.data)
        return False

    def _move_horizontal(self, dx: int) -> bool:
        x, y = self.player
        target = (x + dx, y)
        if not self._in_bounds(target) or self._is_blocking(target):
            self._flash_cells = {self.player}
            return False
        self.player = target
        if self._handle_event_cell():
            self._center_camera()
            self._sync_screen()
            return True
        self._start_fall_if_unsupported()
        self._center_camera()
        return True

    def _click(self, data: dict[str, Any]) -> bool:
        display_x = int(data.get("x", -1))
        display_y = int(data.get("y", -1))
        cell = self._display_to_cell(display_x, display_y)
        if cell is None:
            return False
        component = self._component_at(cell)
        if component is None:
            self._flash_cells = {cell}
            return False
        if component.kind == "green":
            component.cells.clear()
            if self._start_fall_if_unsupported():
                self._center_camera()
            return True
        if component.kind == "orange":
            if not component.active and self.player in component.cells:
                self._flash_cells = set(component.cells)
                return False
            component.active = not component.active
            if self._start_fall_if_unsupported():
                self._center_camera()
            return True
        if component.kind == "red":
            if self.player in component.cells:
                self._flash_cells = set(component.cells)
                return False
            component.active = not component.active
            return True
        if component.kind == "builder":
            return self._grow_builder(component)
        return False

    def _grow_builder(self, component: ComponentState) -> bool:
        front_x = (
            max(x for x, _y in component.cells) if component.direction > 0 else min(x for x, _y in component.cells)
        )
        front_cells = [(x, y) for x, y in component.cells if x == front_x]
        _, y = sorted(front_cells)[0]
        candidate = (front_x + component.direction, y)
        if (
            not self._in_bounds(candidate)
            or not self._is_visible(candidate)
            or self._cell_occupied(candidate)
            or candidate == self.player
        ):
            self._flash_cells = set(component.cells)
            return False
        component.cells.add(candidate)
        return True

    def _start_fall_if_unsupported(self) -> bool:
        below = (self.player[0], self.player[1] + self.gravity)
        if not self._in_bounds(below) or not self._is_blocking(below):
            self._falling = True
            return True
        return False

    def _advance_fall_animation(self) -> None:
        self._flash_cells = set()
        x, y = self.player
        nxt = (x, y + self.gravity)
        if not self._in_bounds(nxt):
            self._falling = False
            self._pending_budget_check = False
            self._sync_screen()
            self.lose()
            self.complete_action()
            return
        if self._is_blocking(nxt):
            self._falling = False
            pending_budget_check = self._pending_budget_check
            self._pending_budget_check = False
            self._center_camera()
            self._sync_screen()
            self._finish_resolved_action(pending_budget_check)
            return

        self.player = nxt
        self._handle_event_cell()
        self._center_camera()
        self._sync_screen()
        if self._state_is_terminal():
            self._falling = False
            self._pending_budget_check = False

    def _handle_event_cell(self) -> bool:
        if self.player in self.spec.goal:
            self.next_level()
            self.complete_action()
            return True
        if self.player in self.spec.traps:
            self.lose()
            self.complete_action()
            return True
        component = self._component_at(self.player)
        if component is not None and component.kind == "red" and component.active:
            component.cells.clear()
            component.consumed = True
            self.gravity *= -1
            return False
        return False

    def _is_blocking(self, cell: tuple[int, int]) -> bool:
        if cell in self.spec.stone:
            return True
        for component in self.components:
            if cell not in component.cells:
                continue
            if component.kind in {"green", "builder"}:
                return True
            if component.kind == "orange" and component.active:
                return True
        return False

    def _cell_occupied(self, cell: tuple[int, int]) -> bool:
        if cell in self.spec.stone or cell in self.spec.goal or cell in self.spec.traps:
            return True
        return any(cell in component.cells for component in self.components)

    def _component_at(self, cell: tuple[int, int]) -> ComponentState | None:
        for component in self.components:
            if cell in component.cells:
                return component
        return None

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < BOARD_W and 0 <= y < self.spec.height

    def _is_visible(self, cell: tuple[int, int]) -> bool:
        _x, y = cell
        return self.camera_top <= y < self.camera_top + VIEW_H

    def _center_camera(self) -> None:
        max_top = max(0, self.spec.height - VIEW_H)
        self.camera_top = max(0, min(max_top, self.player[1] - 5))

    def _display_to_cell(self, x: int, y: int) -> tuple[int, int] | None:
        if x < ORIGIN_X or y < ORIGIN_Y:
            return None
        cell_x = (x - ORIGIN_X) // TILE
        cell_y = (y - ORIGIN_Y) // TILE
        if not (0 <= cell_x < BOARD_W and 0 <= cell_y < VIEW_H):
            return None
        world = (cell_x, cell_y + self.camera_top)
        return world if self._in_bounds(world) else None

    def _sync_screen(self) -> None:
        self._screen.pixels = self._render().astype(np.int8)

    def _render(self) -> np.ndarray:
        frame = np.full((CANVAS, CANVAS), VOID, dtype=np.int8)
        frame[ORIGIN_Y - 1 : ORIGIN_Y + VIEW_H * TILE + 1, ORIGIN_X - 1] = OUTLINE
        frame[ORIGIN_Y - 1 : ORIGIN_Y + VIEW_H * TILE + 1, ORIGIN_X + VIEW_H * TILE] = OUTLINE
        frame[ORIGIN_Y - 1, ORIGIN_X - 1 : ORIGIN_X + VIEW_H * TILE + 1] = OUTLINE
        frame[ORIGIN_Y + VIEW_H * TILE, ORIGIN_X - 1 : ORIGIN_X + VIEW_H * TILE + 1] = OUTLINE

        for cell in self.spec.stone:
            self._draw_cell(frame, cell, STONE)
        for cell in self.spec.goal:
            self._draw_cell(frame, cell, GOAL)
        for cell in self.spec.traps:
            self._draw_trap(frame, cell)
        for component in self.components:
            self._draw_component(frame, component)
        self._draw_player(frame)
        self._draw_scroll(frame)
        self._draw_step_bar(frame)
        for cell in self._flash_cells:
            self._draw_cell_outline(frame, cell, DETAIL)
        return frame

    def _draw_component(self, frame: np.ndarray, component: ComponentState) -> None:
        if component.kind == "green":
            for cell in component.cells:
                self._draw_cell(frame, cell, GREEN)
                self._draw_crack(frame, cell)
        elif component.kind == "orange":
            for cell in component.cells:
                self._draw_cell(frame, cell, ORANGE if component.active else PALE)
                if not component.active:
                    self._draw_cell_outline(frame, cell, ORANGE)
        elif component.kind == "red":
            color = RED_ACTIVE if component.active else RED_INACTIVE
            for cell in component.cells:
                self._draw_cell(frame, cell, color)
                self._draw_red_mark(frame, cell, component.active)
        elif component.kind == "builder":
            for cell in component.cells:
                self._draw_cell(frame, cell, PURPLE)
            if component.cells:
                front_x = (
                    max(x for x, _y in component.cells)
                    if component.direction > 0
                    else min(x for x, _y in component.cells)
                )
                for cell in [c for c in component.cells if c[0] == front_x]:
                    self._draw_builder_cap(frame, cell, component.direction)

    def _draw_player(self, frame: np.ndarray) -> None:
        self._draw_cell(frame, self.player, PLAYER)
        px, py = self._cell_pixel(self.player)
        if px is None:
            return
        marker_y = py + TILE - 1 if self.gravity == DIR_DOWN else py
        frame[marker_y, px + 1 : px + TILE - 1] = PALE

    def _draw_trap(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        self._draw_cell(frame, cell, TRAP)
        px, py = self._cell_pixel(cell)
        if px is None:
            return
        frame[py + 1, px + 2] = DETAIL
        frame[py + 2, px + 1 : px + 4] = DETAIL

    def _draw_crack(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        px, py = self._cell_pixel(cell)
        if px is None:
            return
        frame[py + 1, px + 2] = DETAIL
        frame[py + 2, px + 1] = DETAIL
        frame[py + 3, px + 3] = DETAIL

    def _draw_red_mark(self, frame: np.ndarray, cell: tuple[int, int], active: bool) -> None:
        px, py = self._cell_pixel(cell)
        if px is None:
            return
        color = PALE if active else DETAIL
        frame[py + 1 : py + 4, px + 2] = color
        frame[py + 1, px + 1 : px + 4] = color
        frame[py + 3, px + 1 : px + 4] = color

    def _draw_builder_cap(self, frame: np.ndarray, cell: tuple[int, int], direction: int) -> None:
        px, py = self._cell_pixel(cell)
        if px is None:
            return
        cap_x = px + TILE - 1 if direction > 0 else px
        frame[py + 1 : py + TILE - 1, cap_x] = PALE

    def _draw_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        px, py = self._cell_pixel(cell)
        if px is None:
            return
        frame[py : py + TILE, px : px + TILE] = color

    def _draw_cell_outline(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        px, py = self._cell_pixel(cell)
        if px is None:
            return
        frame[py, px : px + TILE] = color
        frame[py + TILE - 1, px : px + TILE] = color
        frame[py : py + TILE, px] = color
        frame[py : py + TILE, px + TILE - 1] = color

    def _cell_pixel(self, cell: tuple[int, int]) -> tuple[int, int] | tuple[None, None]:
        x, y = cell
        view_y = y - self.camera_top
        if not (0 <= x < BOARD_W and 0 <= view_y < VIEW_H):
            return None, None
        return ORIGIN_X + x * TILE, ORIGIN_Y + view_y * TILE

    def _draw_scroll(self, frame: np.ndarray) -> None:
        if self.spec.height <= VIEW_H:
            return
        x = 61
        frame[4:59, x] = OUTLINE
        span = 55
        thumb_h = max(4, int(span * VIEW_H / self.spec.height))
        max_top = self.spec.height - VIEW_H
        y = 4 + int((span - thumb_h) * self.camera_top / max(1, max_top))
        frame[y : y + thumb_h, x : x + 2] = STONE

    def _draw_step_bar(self, frame: np.ndarray) -> None:
        y = 61
        x = 4
        width = 55
        frame[y : y + 2, x : x + width] = OUTLINE
        filled = max(0, min(width, int(width * self.remaining_steps / max(1, self.step_budget))))
        if filled:
            frame[y : y + 2, x : x + filled] = ORANGE


def _screen_sprite() -> Sprite:
    return Sprite(
        np.full((CANVAS, CANVAS), VOID, dtype=np.int8),
        name="screen",
        x=0,
        y=0,
        layer=0,
        blocking=BlockingMode.NOT_BLOCKED,
        collidable=False,
        tags=["screen"],
    )
