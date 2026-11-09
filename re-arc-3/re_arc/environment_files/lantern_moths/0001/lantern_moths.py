from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "lantern_moths-0001"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

VIEW_SIZE = 64
COLOR_VOID = 5
COLOR_WALL = 4
COLOR_FLOOR = 1
COLOR_SCREEN = 3
COLOR_LANTERN = 11
COLOR_LANTERN_CORE = 12
COLOR_RAY = 11
COLOR_BLUE = 9
COLOR_BLUE_FLOWER = 10
COLOR_PINK = 6
COLOR_PINK_FLOWER = 7
COLOR_RED_FLOWER = 13
COLOR_RED = COLOR_RED_FLOWER
COLOR_REPELLER = 14
COLOR_REPELLER_CORE = 3
COLOR_INVALID = 8
COLOR_SELECT = 15
COLOR_BAR_EMPTY = 4
COLOR_BAR_FULL = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    size: tuple[int, int]
    floor: frozenset[tuple[int, int]]
    movables: tuple[dict[str, object], ...]
    moths: tuple[dict[str, object], ...]
    flowers: tuple[dict[str, object], ...]
    budget: int


class LanternHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: LanternMoths | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[0:3, 2:62] = COLOR_BAR_EMPTY
        filled = max(0, min(60, int(60 * game.remaining_steps / max(1, game.step_budget))))
        if filled:
            frame[0:3, 2 : 2 + filled] = COLOR_BAR_FULL
        return frame


def _range_cells(x1: int, x2: int, y1: int, y2: int | None = None) -> set[tuple[int, int]]:
    if y2 is None:
        y2 = y1
    return {(x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)}


def _all_floor(width: int, height: int) -> frozenset[tuple[int, int]]:
    return frozenset((x, y) for x in range(width) for y in range(height))


def _level_specs() -> tuple[LevelSpec, ...]:
    specs: list[LevelSpec] = []
    specs.append(
        LevelSpec(
            "First Alignment",
            (5, 2),
            _all_floor(5, 2),
            ({"id": "L", "kind": "lantern", "pos": (4, 1)},),
            ({"id": "B", "color": "blue", "pos": (1, 0), "awake": True},),
            ({"color": "blue", "pos": (3, 0)},),
            24,
        )
    )

    floor = _range_cells(1, 5, 2) | {(3, 1)}
    specs.append(
        LevelSpec(
            "The Dark Screen",
            (6, 5),
            frozenset(floor),
            ({"id": "L", "kind": "lantern", "pos": (5, 2)}, {"id": "S", "kind": "screen", "pos": (3, 2)}),
            ({"id": "B", "color": "blue", "pos": (1, 2), "awake": True},),
            ({"color": "blue", "pos": (4, 2)},),
            30,
        )
    )

    floor = _range_cells(2, 2, 1, 4) | _range_cells(2, 6, 3) | _range_cells(2, 6, 4)
    specs.append(
        LevelSpec(
            "Lantern Stopper",
            (7, 6),
            frozenset(floor),
            ({"id": "L", "kind": "lantern", "pos": (2, 4)}, {"id": "S", "kind": "screen", "pos": (4, 3)}),
            ({"id": "B", "color": "blue", "pos": (2, 1), "awake": True},),
            ({"color": "blue", "pos": (5, 3)},),
            78,
        )
    )

    floor = (
        _range_cells(3, 6, 2) | _range_cells(3, 3, 2, 5) | _range_cells(3, 6, 4) | _range_cells(6, 6, 2, 4) | {(2, 5)}
    )
    specs.append(
        LevelSpec(
            "Competing Lanterns",
            (7, 6),
            frozenset(floor),
            ({"id": "D", "kind": "lantern", "pos": (3, 5)}, {"id": "E", "kind": "lantern", "pos": (6, 2)}),
            ({"id": "B", "color": "blue", "pos": (3, 2), "awake": True},),
            ({"color": "blue", "pos": (5, 4)},),
            54,
        )
    )

    floor = {
        (2, 0),
        (3, 0),
        (5, 0),
        (3, 1),
        (5, 1),
        (3, 2),
        (4, 2),
        (5, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),
        (3, 4),
        (5, 4),
        (7, 4),
        (3, 5),
        (4, 5),
        (5, 5),
    }
    specs.append(
        LevelSpec(
            "Waking a Blocking Sleeper",
            (8, 6),
            frozenset(floor),
            (
                {"id": "K", "kind": "lantern", "pos": (3, 5)},
                {"id": "T", "kind": "lantern", "pos": (2, 0)},
                {"id": "G", "kind": "repeller", "pos": (5, 0)},
                {"id": "R", "kind": "lantern", "pos": (7, 4)},
                {"id": "S", "kind": "screen", "pos": (3, 2)},
            ),
            (
                {"id": "B", "color": "blue", "pos": (3, 3), "awake": True},
                {"id": "P", "color": "pink", "pos": (1, 3), "awake": True},
                {"id": "R", "color": "red", "pos": (5, 1), "awake": True},
            ),
            ({"color": "blue", "pos": (3, 1)}, {"color": "pink", "pos": (6, 3)}, {"color": "red", "pos": (5, 5)}),
            90,
        )
    )

    floor = {
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (8, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),
        (8, 3),
        (9, 3),
        (10, 3),
        (2, 4),
        (3, 4),
        (4, 4),
        (8, 4),
        (10, 4),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
        (6, 5),
        (7, 5),
        (8, 5),
        (9, 5),
        (10, 5),
        (1, 6),
        (2, 6),
        (3, 6),
    }
    specs.append(
        LevelSpec(
            "Keeping One Asleep",
            (11, 7),
            frozenset(floor),
            (
                {"id": "A", "kind": "lantern", "pos": (4, 1)},
                {"id": "C", "kind": "lantern", "pos": (2, 1)},
                {"id": "H", "kind": "repeller", "pos": (10, 3)},
                {"id": "S", "kind": "screen", "pos": (3, 4)},
            ),
            (
                {"id": "B", "color": "blue", "pos": (4, 5), "awake": True},
                {"id": "P", "color": "pink", "pos": (1, 5), "awake": True},
            ),
            ({"color": "blue", "pos": (4, 2)}, {"color": "pink", "pos": (7, 1)}),
            144,
        )
    )

    floor = (
        _range_cells(2, 6, 0)
        | _range_cells(2, 2, 0, 4)
        | _range_cells(2, 6, 1)
        | _range_cells(6, 6, 0, 6)
        | _range_cells(3, 6, 6)
        | _range_cells(3, 3, 3, 6)
        | _range_cells(3, 6, 5)
        | _range_cells(5, 5, 1, 5)
    )
    specs.append(
        LevelSpec(
            "Split, Then Reunite",
            (8, 7),
            frozenset(floor),
            (
                {"id": "M", "kind": "lantern", "pos": (2, 0)},
                {"id": "U", "kind": "screen", "pos": (4, 1)},
                {"id": "D", "kind": "screen", "pos": (4, 5)},
            ),
            (
                {"id": "B", "color": "blue", "pos": (2, 4), "awake": True},
                {"id": "P", "color": "pink", "pos": (3, 3), "awake": True},
            ),
            ({"color": "blue", "pos": (5, 2)}, {"color": "pink", "pos": (5, 4)}),
            228,
        )
    )
    return tuple([*specs[:4], specs[6], *specs[4:6]])


class LanternMoths(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._canvas = Sprite(
            np.full((VIEW_SIZE, VIEW_SIZE), COLOR_VOID, dtype=np.int8),
            name="canvas",
            layer=0,
            visible=True,
            collidable=False,
            tags=["canvas"],
        )
        self._hud = LanternHud()
        levels = [
            Level(sprites=[self._canvas.clone()], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name)
            for spec in _level_specs()
        ]
        camera = Camera(
            0, 0, VIEW_SIZE, VIEW_SIZE, background=COLOR_VOID, letter_box=COLOR_VOID, interfaces=[self._hud]
        )
        self._hud.game = self
        self.spec = _level_specs()[0]
        self.movables: dict[str, dict[str, object]] = {}
        self.moths: dict[str, dict[str, object]] = {}
        self.flowers: list[dict[str, object]] = []
        self.selected_id: str | None = None
        self.remaining_steps = 1
        self.step_budget = 1
        self.cell_size = 6
        self.board_left = 2
        self.board_top = 8
        self._flash_cell: tuple[int, int] | None = None
        self._flutter_cells: set[tuple[int, int]] = set()
        super().__init__(GAME_ID, levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6], seed=seed)

    def on_set_level(self, level: Level) -> None:
        self.spec = level.get_data("spec")
        self.movables = {str(item["id"]): deepcopy(item) for item in self.spec.movables}
        self.moths = {str(item["id"]): deepcopy(item) for item in self.spec.moths}
        self.flowers = [deepcopy(item) for item in self.spec.flowers]
        self.selected_id = "L" if self.spec.name == "First Alignment" and "L" in self.movables else None
        self.remaining_steps = int(self.spec.budget)
        self.step_budget = int(self.spec.budget)
        width, height = self.spec.size
        self.cell_size = min(9, (VIEW_SIZE - 6) // width, (VIEW_SIZE - 13) // height)
        self.board_left = (VIEW_SIZE - self.cell_size * width) // 2
        self.board_top = 6 + (VIEW_SIZE - 6 - self.cell_size * height) // 2
        self._flash_cell = None
        self._flutter_cells = set()
        self._sync_visuals()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self._sync_visuals()
            self.complete_action()
            return

        self._flash_cell = None
        self._flutter_cells = set()
        self.remaining_steps -= 1

        if self.action.id == GameAction.ACTION6:
            self._handle_click()
        elif self.action.id.value in MOVE_DELTAS:
            self._handle_move(self.action.id.value)
            self._advance_moths()
        elif self.action.id == GameAction.ACTION5:
            self._advance_moths()

        self._wake_dark_sleepers()
        self._sync_visuals()
        if self._is_solved():
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _handle_click(self) -> None:
        data = self.action.data or {}
        point = self.camera.display_to_grid(int(data.get("x", 0)), int(data.get("y", 0)))
        if point is None:
            self.selected_id = None
            return
        cell = self._pixel_to_cell(*point)
        if cell is None:
            self.selected_id = None
            return
        for object_id, item in self.movables.items():
            if tuple(item["pos"]) == cell:
                self.selected_id = object_id
                return
        self.selected_id = None

    def _handle_move(self, action_id: int) -> None:
        if self.selected_id is None or self.selected_id not in self.movables:
            return
        dx, dy = MOVE_DELTAS[action_id]
        item = self.movables[self.selected_id]
        x, y = item["pos"]
        dest = (int(x) + dx, int(y) + dy)
        if not self._can_move_object_to(dest):
            self._flash_cell = tuple(item["pos"])
            return
        item["pos"] = dest

    def _advance_moths(self) -> None:
        intents: dict[str, tuple[int, int] | None] = {}
        occupied = {tuple(moth["pos"]) for moth in self.moths.values()}
        for moth_id, moth in self.moths.items():
            if self._matching_flower_at(tuple(moth["pos"]), str(moth["color"])) is not None:
                moth["awake"] = False
                intents[moth_id] = None
                continue
            if not moth["awake"]:
                intents[moth_id] = None
                continue
            visible = self._visible_lights_from(tuple(moth["pos"]))
            if not visible:
                intents[moth_id] = None
                continue
            visible.sort(key=self._light_priority_key)
            best_key = self._light_priority_key(visible[0])
            tied = [entry for entry in visible if self._light_priority_key(entry) == best_key]
            if len(tied) > 1:
                self._flutter_cells.add(tuple(moth["pos"]))
                intents[moth_id] = None
                continue
            _distance, lx, ly, kind = tied[0]
            x, y = moth["pos"]
            dx = 0 if lx == x else (1 if lx > x else -1)
            dy = 0 if ly == y else (1 if ly > y else -1)
            if kind == "repeller":
                dx = -dx
                dy = -dy
            dest = (int(x) + dx, int(y) + dy)
            if self._is_solid(dest) or dest in occupied:
                intents[moth_id] = None
            else:
                intents[moth_id] = dest

        target_counts: dict[tuple[int, int], int] = {}
        for dest in intents.values():
            if dest is not None:
                target_counts[dest] = target_counts.get(dest, 0) + 1

        for moth_id, dest in intents.items():
            if dest is None or target_counts[dest] > 1:
                continue
            moth = self.moths[moth_id]
            moth["pos"] = dest
            flower = self._flower_at(dest)
            if flower is not None and flower["color"] == moth["color"]:
                moth["awake"] = False

    def _wake_dark_sleepers(self) -> None:
        for moth in self.moths.values():
            if moth["awake"]:
                continue
            if self._matching_flower_at(tuple(moth["pos"]), str(moth["color"])) is not None:
                continue
            if not self._visible_lights_from(tuple(moth["pos"])):
                moth["awake"] = True

    def _is_solved(self) -> bool:
        for moth in self.moths.values():
            if self._matching_flower_at(tuple(moth["pos"]), str(moth["color"])) is None:
                return False
        return True

    def _can_move_object_to(self, cell: tuple[int, int]) -> bool:
        if cell not in self.spec.floor:
            return False
        if any(tuple(item["pos"]) == cell for item in self.movables.values()):
            return False
        if any(tuple(moth["pos"]) == cell for moth in self.moths.values()):
            return False
        return self._flower_at(cell) is None

    def _is_solid(self, cell: tuple[int, int]) -> bool:
        return cell not in self.spec.floor or any(tuple(item["pos"]) == cell for item in self.movables.values())

    def _visible_lights_from(self, cell: tuple[int, int]) -> list[tuple[int, int, int, str]]:
        out: list[tuple[int, int, int, str]] = []
        x, y = cell
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            cx, cy = x + dx, y + dy
            distance = 1
            while True:
                probe = (cx, cy)
                if probe not in self.spec.floor:
                    break
                blocker = self._movable_at(probe)
                if blocker is not None:
                    if blocker["kind"] in {"lantern", "repeller"}:
                        out.append((distance, cx, cy, str(blocker["kind"])))
                    break
                cx += dx
                cy += dy
                distance += 1
        return out

    def _light_priority_key(self, light: tuple[int, int, int, str]) -> tuple[int, int]:
        distance, _x, _y, kind = light
        priority = 0 if kind == "repeller" else 1
        return priority, distance

    def _movable_at(self, cell: tuple[int, int]) -> dict[str, object] | None:
        for item in self.movables.values():
            if tuple(item["pos"]) == cell:
                return item
        return None

    def _flower_at(self, cell: tuple[int, int]) -> dict[str, object] | None:
        for flower in self.flowers:
            if tuple(flower["pos"]) == cell:
                return flower
        return None

    def _matching_flower_at(self, cell: tuple[int, int], color: str) -> dict[str, object] | None:
        flower = self._flower_at(cell)
        if flower is None or flower["color"] != color:
            return None
        return flower

    def _pixel_to_cell(self, px: int, py: int) -> tuple[int, int] | None:
        x = (px - self.board_left) // self.cell_size
        y = (py - self.board_top) // self.cell_size
        width, height = self.spec.size
        if x < 0 or y < 0 or x >= width or y >= height:
            return None
        return int(x), int(y)

    def cell_center_display(self, cell: tuple[int, int]) -> tuple[int, int]:
        x, y = cell
        return (
            self.board_left + x * self.cell_size + self.cell_size // 2,
            self.board_top + y * self.cell_size + self.cell_size // 2,
        )

    def _sync_visuals(self) -> None:
        canvas = self.current_level.get_sprites_by_name("canvas")[0]
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_VOID, dtype=np.int8)
        self._draw_floor(frame)
        self._draw_rays(frame)
        for flower in self.flowers:
            self._draw_flower(frame, tuple(flower["pos"]), str(flower["color"]))
        for object_id, item in self.movables.items():
            self._draw_movable(frame, object_id, item)
        for moth in self.moths.values():
            self._draw_moth(frame, tuple(moth["pos"]), str(moth["color"]), bool(moth["awake"]))
        if self._flash_cell is not None:
            self._draw_cell_outline(frame, self._flash_cell, COLOR_INVALID)
        canvas.pixels = frame

    def _draw_floor(self, frame: np.ndarray) -> None:
        for x, y in self.spec.floor:
            x0, y0 = self._cell_rect(x, y)
            frame[y0 : y0 + self.cell_size, x0 : x0 + self.cell_size] = COLOR_FLOOR

    def _draw_rays(self, frame: np.ndarray) -> None:
        for item in self.movables.values():
            if item["kind"] not in {"lantern", "repeller"}:
                continue
            lx, ly = item["pos"]
            ray_color = COLOR_REPELLER if item["kind"] == "repeller" else COLOR_RAY
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                cx, cy = int(lx) + dx, int(ly) + dy
                while (cx, cy) in self.spec.floor and self._movable_at((cx, cy)) is None:
                    x0, y0 = self._cell_rect(cx, cy)
                    mid = self.cell_size // 2
                    if dx:
                        frame[y0 + mid : y0 + mid + 1, x0 : x0 + self.cell_size] = ray_color
                    else:
                        frame[y0 : y0 + self.cell_size, x0 + mid : x0 + mid + 1] = ray_color
                    cx += dx
                    cy += dy

    def _draw_flower(self, frame: np.ndarray, cell: tuple[int, int], color: str) -> None:
        x0, y0 = self._cell_rect(*cell)
        c = self._flower_color(color)
        s = self.cell_size
        frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = c
        frame[y0 + 2 : y0 + s - 2, x0 + 2 : x0 + s - 2] = COLOR_FLOOR

    def _draw_movable(self, frame: np.ndarray, object_id: str, item: dict[str, object]) -> None:
        x0, y0 = self._cell_rect(*tuple(item["pos"]))
        s = self.cell_size
        if item["kind"] == "lantern":
            frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_LANTERN
            frame[y0 + 2 : y0 + s - 2, x0 + 2 : x0 + s - 2] = COLOR_LANTERN_CORE
            frame[y0, x0 + s // 2] = COLOR_LANTERN_CORE
        elif item["kind"] == "repeller":
            frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_REPELLER
            frame[y0 + 2 : y0 + s - 2, x0 + 2 : x0 + s - 2] = COLOR_REPELLER_CORE
            frame[y0 + s - 1, x0 + s // 2] = COLOR_REPELLER_CORE
        else:
            frame[y0 + 1 : y0 + s - 1, x0 + 1 : x0 + s - 1] = COLOR_SCREEN
            frame[y0 + 1 : y0 + s - 1, x0 + s // 2] = COLOR_WALL
        if object_id == self.selected_id:
            self._draw_cell_outline(frame, tuple(item["pos"]), COLOR_SELECT)

    def _draw_moth(self, frame: np.ndarray, cell: tuple[int, int], color: str, awake: bool) -> None:
        x0, y0 = self._cell_rect(*cell)
        c = self._moth_color(color)
        s = self.cell_size
        cx = x0 + s // 2
        cy = y0 + s // 2
        if awake:
            widen = cell in self._flutter_cells
            left = max(x0 + 1, cx - (3 if widen and s >= 7 else 2))
            right = min(x0 + s - 1, cx + (3 if widen and s >= 7 else 2))
            frame[cy, left : right + 1] = c
            frame[cy - 1, left:cx] = c
            frame[cy - 1, cx + 1 : right + 1] = c
            frame[cy + 1, left:cx] = c
            frame[cy + 1, cx + 1 : right + 1] = c
        else:
            frame[cy - 2 : cy + 3, cx - 1 : cx + 2] = c
            frame[cy, cx - 2 : cx + 3] = c

    def _moth_color(self, color: str) -> int:
        if color == "blue":
            return COLOR_BLUE
        if color == "red":
            return COLOR_RED
        return COLOR_PINK

    def _flower_color(self, color: str) -> int:
        if color == "blue":
            return COLOR_BLUE_FLOWER
        if color == "red":
            return COLOR_RED_FLOWER
        return COLOR_PINK_FLOWER

    def _draw_cell_outline(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x0, y0 = self._cell_rect(*cell)
        s = self.cell_size
        frame[y0, x0 : x0 + s] = color
        frame[y0 + s - 1, x0 : x0 + s] = color
        frame[y0 : y0 + s, x0] = color
        frame[y0 : y0 + s, x0 + s - 1] = color

    def _cell_rect(self, x: int, y: int) -> tuple[int, int]:
        return self.board_left + x * self.cell_size, self.board_top + y * self.cell_size
