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
BOARD_WIDTH = 13
BOARD_HEIGHT = 12
CELL_SIZE = 4
BOARD_LEFT = (VIEW_SIZE - BOARD_WIDTH * CELL_SIZE) // 2
BOARD_TOP = (VIEW_SIZE - BOARD_HEIGHT * CELL_SIZE) // 2

COLOR_FLOOR = 1
COLOR_STONE = 3
COLOR_WATER = 4
COLOR_OUTLINE = 5
COLOR_INVALID = 8
COLOR_HIKER = 9
COLOR_RAIL = 10
COLOR_LATCH_CENTER = 11
COLOR_LATCH = 12
COLOR_GOAL = 14
COLOR_TURNSTILE = 15
COLOR_TURNSTILE_CENTER = 7
COLOR_BUTTON = 2
COLOR_ACTIVE_ORANGE = 12

MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    center: tuple[int, int]
    orientation: str
    rotation: str = "center"


@dataclass(frozen=True)
class StoneSpec:
    name: str
    path: tuple[tuple[int, int], ...]
    index: int


@dataclass(frozen=True)
class ButtonSpec:
    name: str
    pos: tuple[int, int]
    unlocks: tuple[str, ...]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    start: tuple[int, int]
    goal: tuple[int, int]
    floor: frozenset[tuple[int, int]]
    latches: tuple[PlatformSpec, ...]
    turnstiles: tuple[PlatformSpec, ...] = ()
    stones: tuple[StoneSpec, ...] = ()
    buttons: tuple[ButtonSpec, ...] = ()
    locked_stones: frozenset[str] = frozenset()
    budget: int = 20


BASELINE_ACTIONS = [6, 15, 7, 17, 7, 36]


LEVEL_SPECS = (
    LevelSpec(
        "First latch",
        (1, 5),
        (7, 5),
        frozenset({(1, 5), (4, 5), (5, 5), (6, 5), (7, 5)}),
        (PlatformSpec("A", (3, 5), "H"),),
        budget=24,
    ),
    LevelSpec(
        "Flip ladder",
        (2, 5),
        (8, 4),
        frozenset({(2, 5), (8, 5), (5, 6), (8, 6), (4, 7), (4, 8), (5, 8), (6, 8), (7, 8), (8, 9)}),
        (PlatformSpec("A", (4, 5), "H"), PlatformSpec("B", (8, 8), "H")),
        budget=36,
    ),
    LevelSpec(
        "Wait for stone",
        (2, 5),
        (2, 2),
        frozenset(
            {
                (2, 2),
                (3, 2),
                (6, 2),
                (7, 2),
                (8, 2),
                (9, 2),
                (9, 3),
                (9, 4),
                (2, 5),
                (3, 5),
                (4, 5),
                (5, 5),
                (9, 5),
                (9, 6),
                (2, 7),
                (5, 8),
                (4, 9),
                (5, 9),
                (6, 9),
                (7, 9),
                (8, 9),
                (5, 10),
                (6, 10),
                (7, 10),
                (8, 10),
                (9, 10),
            }
        ),
        (PlatformSpec("A", (5, 2), "V"), PlatformSpec("B", (4, 7), "H"), PlatformSpec("C", (9, 8), "H")),
        stones=(StoneSpec("S", ((4, 2), (4, 1), (4, 2), (4, 3)), 0),),
        budget=42,
    ),
    LevelSpec(
        "Fixed turnstile",
        (2, 1),
        (8, 9),
        frozenset(
            {
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 1),
                (6, 1),
                (5, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
                (5, 6),
                (2, 7),
                (2, 8),
                (2, 9),
                (6, 9),
                (7, 9),
                (8, 9),
            }
        ),
        (PlatformSpec("P", (5, 3), "E", "edge"), PlatformSpec("B", (4, 9), "V")),
        stones=(StoneSpec("M", ((3, 9), (3, 8), (3, 9), (3, 10)), 0),),
        budget=84,
    ),
    LevelSpec(
        "Stone as latch",
        (3, 9),
        (4, 1),
        frozenset(
            {
                (4, 1),
                (4, 2),
                (5, 2),
                (5, 3),
                (5, 7),
                (4, 8),
                (5, 8),
                (6, 8),
                (10, 8),
                (3, 9),
                (4, 9),
                (5, 9),
                (6, 9),
                (10, 9),
            }
        ),
        (PlatformSpec("P", (5, 6), "E", "edge_up"), PlatformSpec("C", (8, 9), "V")),
        stones=(StoneSpec("T", ((5, 5), (6, 5), (5, 5), (4, 5)), 0),),
        buttons=(ButtonSpec("B", (10, 7), ("T",)),),
        locked_stones=frozenset({"T"}),
        budget=102,
    ),
    LevelSpec(
        "Exit-side detour",
        (1, 6),
        (7, 0),
        frozenset(
            {
                (8, 0),
                (9, 0),
                (10, 0),
                (11, 0),
                (1, 6),
                (4, 5),
                (5, 5),
                (8, 5),
                (9, 5),
                (11, 5),
                (5, 6),
                (7, 6),
                (8, 6),
                (11, 6),
                (3, 8),
                (4, 8),
                (5, 8),
                (6, 9),
                (11, 3),
                (11, 4),
            }
        ),
        (PlatformSpec("A", (3, 6), "H"), PlatformSpec("B", (7, 8), "H"), PlatformSpec("C", (11, 1), "H")),
        (PlatformSpec("T", (3, 4), "V"), PlatformSpec("U", (10, 6), "V")),
        (StoneSpec("S1", ((4, 2), (4, 3), (4, 4), (4, 3)), 0), StoneSpec("S2", ((7, 9), (7, 10), (7, 11)), 2)),
        budget=216,
    ),
)


class LatchHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: LatchBridges | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        x0 = BOARD_LEFT
        y0 = 62
        width = BOARD_WIDTH * CELL_SIZE
        frame[y0:64, x0 : x0 + width] = COLOR_OUTLINE
        filled = max(0, min(width, int(width * game.remaining_steps / max(1, game.step_budget))))
        if filled:
            frame[y0:64, x0 : x0 + filled] = COLOR_LATCH_CENTER
        return frame


def _solid_pixels(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), color, dtype=np.int8)


def _cell_rect(cell: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y = cell
    return BOARD_LEFT + x * CELL_SIZE, BOARD_TOP + y * CELL_SIZE, CELL_SIZE, CELL_SIZE


def _cell_center(cell: tuple[int, int]) -> tuple[int, int]:
    x0, y0, _, _ = _cell_rect(cell)
    return x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2


def _cell_from_pixel(pixel_x: int, pixel_y: int) -> tuple[int, int] | None:
    if not (BOARD_LEFT <= pixel_x < BOARD_LEFT + BOARD_WIDTH * CELL_SIZE):
        return None
    if not (BOARD_TOP <= pixel_y < BOARD_TOP + BOARD_HEIGHT * CELL_SIZE):
        return None
    return (pixel_x - BOARD_LEFT) // CELL_SIZE, (pixel_y - BOARD_TOP) // CELL_SIZE


def _footprint(center: tuple[int, int], orientation: str) -> frozenset[tuple[int, int]]:
    cx, cy = center
    if orientation == "H":
        return frozenset({(cx - 1, cy), (cx, cy), (cx + 1, cy)})
    return frozenset({(cx, cy - 1), (cx, cy), (cx, cy + 1)})


def _edge_footprint(pivot: tuple[int, int], orientation: str) -> frozenset[tuple[int, int]]:
    px, py = pivot
    if orientation in {"H", "E"}:
        return frozenset((px + offset, py) for offset in range(4))
    if orientation == "W":
        return frozenset((px - offset, py) for offset in range(4))
    if orientation == "N":
        return frozenset((px, py - offset) for offset in range(4))
    return frozenset((px, py + offset) for offset in range(4))


def _action_id(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    return int(value)


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(
        _solid_pixels(VIEW_SIZE, VIEW_SIZE, COLOR_WATER),
        name="board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level(sprites=[board], grid_size=(VIEW_SIZE, VIEW_SIZE), name=spec.name, data={"spec": spec})


class LatchBridges(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = LatchHud()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "latch_bridges-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_WATER, COLOR_WATER, [self._hud]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.board = level.get_sprites_by_tag("board")[0]
        self.spec: LevelSpec = level.get_data("spec")
        self.hiker = self.spec.start
        self.latches = {
            platform.name: {
                "center": platform.center,
                "orientation": platform.orientation,
                "rotation": platform.rotation,
            }
            for platform in self.spec.latches
        }
        self.turnstiles = {
            platform.name: {
                "center": platform.center,
                "orientation": platform.orientation,
                "rotation": platform.rotation,
            }
            for platform in self.spec.turnstiles
        }
        self.stones = {stone.name: {"path": stone.path, "index": stone.index} for stone in self.spec.stones}
        self.buttons = {button.name: {"pos": button.pos, "unlocks": button.unlocks} for button in self.spec.buttons}
        self.locked_stones = set(self.spec.locked_stones)
        self.step_budget = int(self.spec.budget)
        self.remaining_steps = self.step_budget
        self.flash_cells: set[tuple[int, int]] = set()
        self._sync_visuals()

    def step(self) -> None:
        if _action_id(self.action.id) == GameAction.RESET.value:
            self.flash_cells.clear()
            self._sync_visuals()
            self.complete_action()
            return

        self.flash_cells.clear()
        valid = False
        action = _action_id(self.action.id)
        if action in MOVE_DELTAS:
            valid = self._move(MOVE_DELTAS[action])
        elif action == ACTION_SPACE:
            self._advance_stones()
            valid = True
        elif action == ACTION_CLICK:
            valid = self._click()

        if not valid:
            self._sync_visuals()
            self.complete_action()
            return

        self.remaining_steps -= 1
        self._sync_visuals()
        if self.hiker == self.spec.goal:
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _move(self, delta: tuple[int, int]) -> bool:
        ox, oy = self.hiker
        dx, dy = delta
        destination = (ox + dx, oy + dy)
        if not self._is_traversable(destination):
            self.flash_cells.add(destination)
            return False

        self.hiker = destination
        self._activate_button_at(destination)
        self._advance_stones()
        return True

    def _click(self) -> bool:
        data = self.action.data or {}
        cell = _cell_from_pixel(int(data.get("x", 0)), int(data.get("y", 0)))
        if cell is None:
            return False
        latch_name = self._latch_at(cell)
        if latch_name is not None:
            before = str(self.latches[latch_name]["orientation"])
            self._try_rotate_latch(latch_name)
            if self.latches[latch_name]["orientation"] == before:
                return False
            self._advance_stones()
            return True
        button_name = self._button_at(cell)
        if button_name is not None:
            if not self._activate_button(button_name):
                return False
            self._advance_stones()
            return True
        name = self._turnstile_at(cell)
        if name is None or not self._hiker_can_operate_turnstile(name):
            self.flash_cells.add(cell)
            return False
        center = self.turnstiles[name]["center"]
        if self.hiker in self._platform_cells("turnstiles", name) and self.hiker != center:
            self.flash_cells.add(self.hiker)
            return False
        target = self._rotated_footprint(self.turnstiles[name])
        if not self._turnstile_target_clear(name, target):
            self.flash_cells.update(target)
            return False
        self.turnstiles[name]["orientation"] = self._rotated_orientation(self.turnstiles[name]["orientation"])
        self._advance_stones()
        return True

    def _try_rotate_latch(self, name: str) -> None:
        target = self._rotated_footprint(self.latches[name])
        if self._latch_target_clear(name, target):
            self.latches[name]["orientation"] = self._rotated_orientation(self.latches[name])
        else:
            self.flash_cells.update(target)

    def _advance_stones(self) -> None:
        desired: dict[str, tuple[int, tuple[int, int]]] = {}
        target_counts: dict[tuple[int, int], int] = {}
        current_stones = self._stone_cells()
        platform_cells = self._all_platform_cells()

        for name, stone in self.stones.items():
            if name in self.locked_stones:
                continue
            path = stone["path"]
            next_index = (int(stone["index"]) + 1) % len(path)
            target = path[next_index]
            desired[name] = (next_index, target)
            target_counts[target] = target_counts.get(target, 0) + 1

        for name, (next_index, target) in desired.items():
            blocked = target == self.hiker or target in platform_cells
            blocked = blocked or any(other != name and target == cell for other, cell in current_stones.items())
            blocked = blocked or target_counts[target] > 1
            if not blocked:
                self.stones[name]["index"] = next_index

    def _is_traversable(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
            return False
        if cell in self._stone_cells().values():
            return False
        return (
            cell in self.spec.floor
            or cell == self.spec.goal
            or cell in self._button_cells()
            or cell in self._all_platform_cells()
        )

    def _latch_at(self, cell: tuple[int, int]) -> str | None:
        for name in self.latches:
            if cell in self._platform_cells("latches", name):
                return name
        return None

    def _turnstile_at(self, cell: tuple[int, int]) -> str | None:
        for name in self.turnstiles:
            if cell in self._platform_cells("turnstiles", name):
                return name
        return None

    def _button_at(self, cell: tuple[int, int]) -> str | None:
        for name, button in self.buttons.items():
            if button["pos"] == cell:
                return name
        return None

    def _button_cells(self) -> set[tuple[int, int]]:
        return {button["pos"] for button in self.buttons.values()}

    def _activate_button_at(self, cell: tuple[int, int]) -> bool:
        name = self._button_at(cell)
        return name is not None and self._activate_button(name)

    def _activate_button(self, name: str) -> bool:
        before = set(self.locked_stones)
        for stone_name in self.buttons[name]["unlocks"]:
            self.locked_stones.discard(stone_name)
        return self.locked_stones != before

    def _hiker_can_operate_turnstile(self, name: str) -> bool:
        center = self.turnstiles[name]["center"]
        if self.hiker == center:
            return True
        hx, hy = self.hiker
        for cell in self._platform_cells("turnstiles", name):
            cx, cy = cell
            if abs(hx - cx) <= 1 and abs(hy - cy) <= 1:
                return True
        return False

    def _rotated_footprint(self, platform: dict[str, object]) -> frozenset[tuple[int, int]]:
        return self._footprint_for(platform, self._rotated_orientation(platform))

    def _rotated_orientation(self, platform: object) -> str:
        orientation = platform["orientation"] if isinstance(platform, dict) else platform
        if isinstance(platform, dict) and str(platform.get("rotation", "center")) == "edge_up":
            if str(orientation) == "E":
                return "N"
            if str(orientation) == "N":
                return "E"
        if str(orientation) == "E":
            return "W"
        if str(orientation) == "W":
            return "E"
        if str(orientation) == "N":
            return "S"
        if str(orientation) == "S":
            return "N"
        return "V" if str(orientation) == "H" else "H"

    def _latch_target_clear(self, name: str, target: frozenset[tuple[int, int]]) -> bool:
        blockers = self._occupied_by_platforms(except_kind="latches", except_name=name)
        stones = set(self._stone_cells().values())
        return all(
            self._inside(cell) and cell != self.hiker and cell not in stones and cell not in blockers for cell in target
        )

    def _turnstile_target_clear(self, name: str, target: frozenset[tuple[int, int]]) -> bool:
        center = self.turnstiles[name]["center"]
        blockers = self._occupied_by_platforms(except_kind="turnstiles", except_name=name)
        stones = set(self._stone_cells().values())
        for cell in target:
            if not self._inside(cell) or cell in stones or cell in blockers:
                return False
            if cell == self.hiker and cell != center:
                return False
        return True

    def _inside(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT

    def _platform_cells(self, kind: str, name: str) -> frozenset[tuple[int, int]]:
        collection = self.latches if kind == "latches" else self.turnstiles
        return self._footprint_for(collection[name], str(collection[name]["orientation"]))

    def _footprint_for(self, platform: dict[str, object], orientation: str) -> frozenset[tuple[int, int]]:
        center = platform["center"]
        if str(platform.get("rotation", "center")).startswith("edge"):
            return _edge_footprint(center, orientation)
        return _footprint(center, orientation)

    def _all_platform_cells(self) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for name in self.latches:
            out.update(self._platform_cells("latches", name))
        for name in self.turnstiles:
            out.update(self._platform_cells("turnstiles", name))
        return out

    def _occupied_by_platforms(self, *, except_kind: str, except_name: str) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for name in self.latches:
            if except_kind == "latches" and name == except_name:
                continue
            out.update(self._platform_cells("latches", name))
        for name in self.turnstiles:
            if except_kind == "turnstiles" and name == except_name:
                continue
            out.update(self._platform_cells("turnstiles", name))
        return out

    def _stone_cells(self) -> dict[str, tuple[int, int]]:
        return {name: stone["path"][int(stone["index"])] for name, stone in self.stones.items()}

    def _sync_visuals(self) -> None:
        frame = _solid_pixels(VIEW_SIZE, VIEW_SIZE, COLOR_WATER)
        self._draw_floor(frame)
        self._draw_goal(frame)
        self._draw_rails(frame)
        self._draw_buttons(frame)
        for name in self.latches:
            self._draw_platform(
                frame,
                self._platform_cells("latches", name),
                self.latches[name]["center"],
                COLOR_LATCH,
                COLOR_LATCH_CENTER,
            )
        for name in self.turnstiles:
            is_active = self._turnstile_is_active(name)
            body_color = COLOR_ACTIVE_ORANGE if is_active else COLOR_TURNSTILE
            center_color = COLOR_LATCH_CENTER if is_active else COLOR_TURNSTILE_CENTER
            self._draw_platform(
                frame,
                self._platform_cells("turnstiles", name),
                self.turnstiles[name]["center"],
                body_color,
                center_color,
            )
        self._draw_stones(frame)
        self._draw_hiker(frame)
        self._draw_flash(frame)
        self.board.pixels = frame

    def _draw_floor(self, frame: np.ndarray) -> None:
        for cell in self.spec.floor:
            x0, y0, w, h = _cell_rect(cell)
            frame[y0 : y0 + h, x0 : x0 + w] = COLOR_FLOOR

    def _draw_goal(self, frame: np.ndarray) -> None:
        x0, y0, _, _ = _cell_rect(self.spec.goal)
        frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = COLOR_GOAL
        frame[y0 + 1 : y0 + 4, x0 + 1 : x0 + 4] = COLOR_FLOOR
        frame[y0 + 2, x0 + 2] = COLOR_GOAL

    def _draw_rails(self, frame: np.ndarray) -> None:
        for stone in self.stones.values():
            for cell in stone["path"]:
                x0, y0, _, _ = _cell_rect(cell)
                frame[y0 + 2, x0 : x0 + CELL_SIZE] = COLOR_RAIL
                frame[y0 : y0 + CELL_SIZE, x0 + 2] = COLOR_RAIL

    def _draw_buttons(self, frame: np.ndarray) -> None:
        for button in self.buttons.values():
            x0, y0, _, _ = _cell_rect(button["pos"])
            frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = COLOR_OUTLINE
            frame[y0 + 1 : y0 + 4, x0 + 1 : x0 + 4] = COLOR_BUTTON

    def _turnstile_is_active(self, name: str) -> bool:
        if not self._hiker_can_operate_turnstile(name):
            return False
        center = self.turnstiles[name]["center"]
        if self.hiker in self._platform_cells("turnstiles", name) and self.hiker != center:
            return False
        return self._turnstile_target_clear(name, self._rotated_footprint(self.turnstiles[name]))

    def _draw_platform(
        self,
        frame: np.ndarray,
        cells: frozenset[tuple[int, int]],
        center: tuple[int, int],
        body_color: int,
        center_color: int,
    ) -> None:
        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        x0 = BOARD_LEFT + min(xs) * CELL_SIZE
        y0 = BOARD_TOP + min(ys) * CELL_SIZE
        x1 = BOARD_LEFT + (max(xs) + 1) * CELL_SIZE
        y1 = BOARD_TOP + (max(ys) + 1) * CELL_SIZE
        frame[y0:y1, x0:x1] = COLOR_OUTLINE
        frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = body_color
        cx, cy, _, _ = _cell_rect(center)
        frame[cy + 1 : cy + 4, cx + 1 : cx + 4] = center_color

    def _draw_stones(self, frame: np.ndarray) -> None:
        for cell in self._stone_cells().values():
            x0, y0, _, _ = _cell_rect(cell)
            frame[y0 + 1 : y0 + 4, x0 + 1 : x0 + 4] = COLOR_STONE
            frame[y0 + 2, x0] = COLOR_STONE
            frame[y0 + 2, x0 + 4] = COLOR_STONE

    def _draw_hiker(self, frame: np.ndarray) -> None:
        x0, y0, _, _ = _cell_rect(self.hiker)
        frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = COLOR_OUTLINE
        frame[y0 + 1 : y0 + 4, x0 + 1 : x0 + 4] = COLOR_HIKER
        frame[y0, x0 + 2] = COLOR_HIKER

    def _draw_flash(self, frame: np.ndarray) -> None:
        for cell in self.flash_cells:
            if not self._inside(cell):
                continue
            x0, y0, _, _ = _cell_rect(cell)
            frame[y0, x0 : x0 + CELL_SIZE] = COLOR_INVALID
            frame[y0 + CELL_SIZE - 1, x0 : x0 + CELL_SIZE] = COLOR_INVALID
            frame[y0 : y0 + CELL_SIZE, x0] = COLOR_INVALID
            frame[y0 : y0 + CELL_SIZE, x0 + CELL_SIZE - 1] = COLOR_INVALID


def turnstile_click(name: str, level_index: int) -> tuple[int, int]:
    spec = LEVEL_SPECS[level_index]
    for turnstile in spec.turnstiles:
        if turnstile.name == name:
            return _cell_center(turnstile.center)
    raise ValueError(f"No turnstile {name!r} on level {level_index}.")
