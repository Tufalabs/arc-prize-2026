from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "wind_sails-0001"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
DIR_ORDER = ("N", "E", "S", "W")
MOVE_ACTIONS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_WALL = 4
COLOR_OUTLINE = 5
COLOR_WHITE = 0
COLOR_RED = 8
COLOR_WORKER = 11
COLOR_PAD_A = 11
COLOR_PAD_B = 7
COLOR_FAN = 10
COLOR_BLUE = 9
COLOR_ORANGE = 12
COLOR_GREEN = 14
COLOR_PURPLE = 15
COLOR_MAROON = 13
COLOR_MAGENTA = 6


@dataclass
class CrateState:
    cid: str
    color: int
    pos: tuple[int, int]
    sail_dir: str
    harbor: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    size: tuple[int, int]
    wind_dir: str
    worker: tuple[int, int]
    crates: tuple[tuple[str, int, tuple[int, int], str, tuple[int, int]], ...]
    walls: tuple[tuple[int, int], ...] = ()
    fans: tuple[tuple[int, int, str], ...] = ()
    pads: tuple[tuple[int, int, str], ...] = ()
    gates: tuple[tuple[int, int, str], ...] = ()
    baseline: int = 1


LEVEL_SPECS = (
    LevelSpec("Catch the wind", (7, 5), "E", (3, 4), (("B", COLOR_BLUE, (1, 2), "W", (5, 2)),), baseline=10),
    LevelSpec(
        "Stop the short crate",
        (7, 6),
        "E",
        (2, 2),
        (("B", COLOR_BLUE, (1, 1), "N", (6, 1)), ("O", COLOR_ORANGE, (1, 3), "E", (3, 3))),
        baseline=14,
    ),
    LevelSpec(
        "Windbreak timing",
        (10, 7),
        "E",
        (2, 6),
        (
            ("B", COLOR_BLUE, (1, 0), "E", (8, 0)),
            ("G", COLOR_GREEN, (2, 3), "E", (3, 3)),
            ("O", COLOR_ORANGE, (7, 3), "E", (7, 3)),
            ("P", COLOR_PURPLE, (1, 5), "W", (4, 5)),
        ),
        walls=tuple((x, 1) for x in range(10)),
        baseline=20,
    ),
    LevelSpec(
        "Hold the gate",
        (10, 7),
        "E",
        (2, 5),
        (("G", COLOR_GREEN, (1, 4), "E", (4, 4)), ("B", COLOR_BLUE, (7, 3), "N", (8, 3))),
        walls=((5, 0), (5, 1), (5, 2), (5, 4), (5, 5), (5, 6)),
        pads=((3, 4, "A"),),
        gates=((5, 3, "A"),),
        baseline=26,
    ),
    LevelSpec(
        "Turn the corner fan",
        (10, 10),
        "E",
        (1, 8),
        (
            ("G", COLOR_GREEN, (1, 5), "E", (4, 5)),
            ("B", COLOR_BLUE, (2, 7), "E", (6, 3)),
            ("O", COLOR_ORANGE, (7, 6), "N", (7, 4)),
        ),
        walls=((9, 7), (9, 9)),
        fans=((1, 8, "E"), (9, 8, "N")),
        pads=((3, 5, "A"),),
        gates=((8, 8, "A"),),
        baseline=54,
    ),
    LevelSpec(
        "North first, west later",
        (10, 10),
        "E",
        (1, 8),
        (
            ("G", COLOR_GREEN, (1, 5), "E", (4, 5)),
            ("O", COLOR_ORANGE, (7, 5), "E", (7, 5)),
            ("B", COLOR_BLUE, (2, 7), "E", (6, 3)),
            ("P", COLOR_PURPLE, (8, 6), "N", (5, 4)),
        ),
        walls=((9, 7), (9, 9)),
        fans=((1, 8, "E"), (1, 1, "W"), (9, 8, "N")),
        pads=((3, 5, "A"),),
        gates=((8, 8, "A"),),
        baseline=66,
    ),
    LevelSpec(
        "Leave before the lock closes",
        (10, 10),
        "E",
        (1, 8),
        (
            ("G", COLOR_GREEN, (1, 5), "E", (4, 5)),
            ("O", COLOR_ORANGE, (7, 5), "E", (7, 5)),
            ("B", COLOR_BLUE, (2, 7), "E", (6, 2)),
            ("P", COLOR_PURPLE, (8, 6), "N", (5, 4)),
            ("R", COLOR_MAROON, (9, 3), "W", (7, 3)),
        ),
        walls=((9, 7), (9, 9), (9, 0), (8, 0), (9, 2)),
        fans=((1, 8, "E"), (9, 8, "N"), (9, 1, "W")),
        pads=((3, 5, "A"), (8, 4, "B")),
        gates=((8, 8, "A"), (8, 1, "B")),
        baseline=77,
    ),
    LevelSpec(
        "Blocker, passenger, pad-holder",
        (11, 10),
        "E",
        (1, 8),
        (
            ("G", COLOR_GREEN, (1, 5), "E", (4, 5)),
            ("O", COLOR_ORANGE, (8, 5), "E", (8, 5)),
            ("B", COLOR_BLUE, (2, 7), "E", (7, 2)),
            ("M", COLOR_MAGENTA, (2, 3), "E", (5, 3)),
            ("P", COLOR_PURPLE, (9, 6), "N", (6, 4)),
            ("R", COLOR_MAROON, (10, 3), "W", (8, 3)),
        ),
        walls=((10, 7), (10, 9), (10, 0), (10, 2)),
        fans=((1, 8, "E"), (10, 8, "N"), (10, 1, "W")),
        pads=((3, 5, "A"), (9, 4, "B")),
        gates=((9, 8, "A"), (9, 1, "B")),
        baseline=88,
    ),
)


def _make_levels() -> list[Level]:
    levels = []
    for spec in LEVEL_SPECS:
        sprite = Sprite(np.full((64, 64), COLOR_BG, dtype=np.int8), name="frame", layer=0, collidable=False)
        levels.append(Level([sprite], grid_size=(64, 64), data={"spec": spec}, name=spec.name))
    return levels


class WindSails(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            GAME_ID,
            _make_levels(),
            Camera(0, 0, 64, 64, background=COLOR_BG, letter_box=COLOR_BG),
            False,
            8,
            [1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.width, self.height = self.spec.size
        self.worker_pos = self.spec.worker
        self.wind_dir = self.spec.wind_dir
        self.selected_crate_id: str | None = None
        self.crates = {
            cid: CrateState(cid, color, pos, sail_dir, harbor) for cid, color, pos, sail_dir, harbor in self.spec.crates
        }
        self.walls = set(self.spec.walls)
        self.fans = {(x, y): direction for x, y, direction in self.spec.fans}
        self.pads = {(x, y): controller for x, y, controller in self.spec.pads}
        self.gates = {(x, y): controller for x, y, controller in self.spec.gates}
        self.open_controllers: set[str] = set()
        self.step_budget = self.spec.baseline * 6
        self.remaining_steps = self.step_budget
        self._animation_frames: list[np.ndarray] = []
        self._pending_solve = False
        self._recompute_gates()
        self._sync_frame()

    def step(self) -> None:
        if self._animation_frames:
            self._set_frame(self._animation_frames.pop(0))
            if not self._animation_frames:
                if self._pending_solve:
                    self._pending_solve = False
                    self.next_level()
                    self.complete_action()
                else:
                    self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self._sync_frame()
            self.complete_action()
            return

        accepted = self._resolve_action()
        if accepted:
            self.remaining_steps -= 1
        if self._animation_frames:
            if accepted and self.remaining_steps <= 0 and not self._pending_solve:
                self.lose()
                self.complete_action()
            return
        self._sync_frame()

        if accepted and self._is_solved():
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.lose()
            self.complete_action()
            return
        self.complete_action()

    def _resolve_action(self) -> bool:
        action_id = int(self.action.id.value)
        if action_id in MOVE_ACTIONS:
            return self._move_worker(*MOVE_ACTIONS[action_id])
        if action_id == ACTION_SPACE:
            return self._gust()
        if action_id == ACTION_CLICK:
            data = self.action.data or {}
            return self._click(int(data.get("x", 0)), int(data.get("y", 0)))
        self._flash_invalid()
        return False

    def _snapshot(self):
        return (
            self.worker_pos,
            self.wind_dir,
            self.selected_crate_id,
            deepcopy(self.crates),
            set(self.open_controllers),
            self.remaining_steps,
        )

    def _restore(self, snapshot) -> None:
        (
            self.worker_pos,
            self.wind_dir,
            self.selected_crate_id,
            self.crates,
            self.open_controllers,
            self.remaining_steps,
        ) = snapshot

    def _move_worker(self, dx: int, dy: int) -> bool:
        snapshot = self._snapshot()
        target = (self.worker_pos[0] + dx, self.worker_pos[1] + dy)
        if not self._worker_passable(target):
            self._flash_invalid()
            return False
        self.worker_pos = target
        if target in self.fans:
            self.wind_dir = self.fans[target]
        if not self._recompute_gates():
            self._restore(snapshot)
            self._flash_invalid()
            return False
        return True

    def _click(self, px: int, py: int) -> bool:
        cell = self._pixel_to_cell(px, py)
        if cell is None:
            return False
        crate = self._crate_at(cell)
        if crate is None:
            return False
        if abs(crate.pos[0] - self.worker_pos[0]) + abs(crate.pos[1] - self.worker_pos[1]) != 1:
            return False
        if self.selected_crate_id != crate.cid:
            self.selected_crate_id = crate.cid
        else:
            idx = DIR_ORDER.index(crate.sail_dir)
            crate.sail_dir = DIR_ORDER[(idx + 1) % len(DIR_ORDER)]
        self._recompute_gates()
        return True

    def _gust(self) -> bool:
        snapshot = self._snapshot()
        moving: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        dx, dy = DIRS[self.wind_dir]

        for lane in self._lanes():
            blocker = self._first_blocker(lane)
            if blocker is None or blocker[0] != "crate":
                continue
            crate = self.crates[blocker[1]]
            if crate.sail_dir != self.wind_dir:
                continue
            target = (crate.pos[0] + dx, crate.pos[1] + dy)
            if target == self.worker_pos:
                self._restore(snapshot)
                self._flash_invalid()
                return False
            if self._crate_passable(target, ignore=crate.cid):
                moving[crate.cid] = (crate.pos, target)

        if not moving:
            return True

        for cid, (_old, new) in moving.items():
            self.crates[cid].pos = new
        if not self._recompute_gates():
            self._restore(snapshot)
            self._flash_invalid()
            return False

        self._queue_gust_frames(moving)
        if self._is_solved():
            self._pending_solve = True
        return True

    def _lanes(self) -> list[list[tuple[int, int]]]:
        lanes = []
        if self.wind_dir == "E":
            for y in range(self.height):
                lanes.append([(x, y) for x in range(self.width)])
        elif self.wind_dir == "W":
            for y in range(self.height):
                lanes.append([(x, y) for x in range(self.width - 1, -1, -1)])
        elif self.wind_dir == "S":
            for x in range(self.width):
                lanes.append([(x, y) for y in range(self.height)])
        else:
            for x in range(self.width):
                lanes.append([(x, y) for y in range(self.height - 1, -1, -1)])
        return lanes

    def _first_blocker(self, lane: list[tuple[int, int]]) -> tuple[str, str] | None:
        for cell in lane:
            if cell in self.walls or (cell in self.gates and self.gates[cell] not in self.open_controllers):
                return ("wall", "")
            crate = self._crate_at(cell)
            if crate is not None:
                return ("crate", crate.cid)
        return None

    def _worker_passable(self, cell: tuple[int, int]) -> bool:
        if not self._in_bounds(cell) or cell in self.walls or self._crate_at(cell) is not None:
            return False
        if cell in self.gates and self.gates[cell] not in self.open_controllers:
            return False
        return True

    def _crate_passable(self, cell: tuple[int, int], ignore: str | None = None) -> bool:
        if not self._in_bounds(cell) or cell in self.walls or cell in self.fans or cell == self.worker_pos:
            return False
        if cell in self.gates and self.gates[cell] not in self.open_controllers:
            return False
        crate = self._crate_at(cell)
        return crate is None or crate.cid == ignore

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def _crate_at(self, cell: tuple[int, int]) -> CrateState | None:
        for crate in self.crates.values():
            if crate.pos == cell:
                return crate
        return None

    def _recompute_gates(self) -> bool:
        pressed_cells = {cell for cell in self.pads if self._crate_at(cell) is not None}
        controllers = set(self.gates.values())
        open_controllers = {
            controller
            for controller in controllers
            if all((cell in pressed_cells) for cell, c in self.pads.items() if c == controller)
        }
        self.open_controllers = open_controllers
        for cell, controller in self.gates.items():
            if controller in self.open_controllers:
                continue
            if cell == self.worker_pos or self._crate_at(cell) is not None:
                return False
        return True

    def _is_solved(self) -> bool:
        return all(crate.pos == crate.harbor for crate in self.crates.values())

    def _layout(self) -> tuple[int, int, int]:
        cell = min(6, 58 // self.width, 52 // self.height)
        left = (64 - self.width * cell) // 2
        top = 7 + max(0, (52 - self.height * cell) // 2)
        return left, top, cell

    def _pixel_to_cell(self, px: int, py: int) -> tuple[int, int] | None:
        left, top, cell = self._layout()
        x = (px - left) // cell
        y = (py - top) // cell
        if 0 <= x < self.width and 0 <= y < self.height:
            return (int(x), int(y))
        return None

    def _sync_frame(self) -> None:
        self._set_frame(self._render())

    def _set_frame(self, frame: np.ndarray) -> None:
        self.current_level.get_sprites_by_name("frame")[0].pixels = frame.astype(np.int8, copy=False)

    def _queue_gust_frames(self, moving: dict[str, tuple[tuple[int, int], tuple[int, int]]]) -> None:
        frames = []
        for step in (1 / 3, 2 / 3, 1.0):
            frames.append(self._render(slide={cid: (old, new, step) for cid, (old, new) in moving.items()}))
        self._animation_frames = frames

    def _flash_invalid(self) -> None:
        frame = self._render()
        frame[0:3, :] = COLOR_RED
        frame[-3:, :] = COLOR_RED
        frame[:, 0:3] = COLOR_RED
        frame[:, -3:] = COLOR_RED
        self._animation_frames = [frame, self._render()]

    def _render(self, slide: dict[str, tuple[tuple[int, int], tuple[int, int], float]] | None = None) -> np.ndarray:
        frame = np.full((64, 64), COLOR_BG, dtype=np.int8)
        left, top, cell = self._layout()
        board_w = self.width * cell
        board_h = self.height * cell
        frame[top : top + board_h, left : left + board_w] = COLOR_FLOOR
        frame[max(0, top - 1) : top + board_h + 1, max(0, left - 1)] = COLOR_OUTLINE
        frame[max(0, top - 1) : top + board_h + 1, min(63, left + board_w)] = COLOR_OUTLINE
        frame[max(0, top - 1), max(0, left - 1) : left + board_w + 1] = COLOR_OUTLINE
        frame[min(63, top + board_h), max(0, left - 1) : left + board_w + 1] = COLOR_OUTLINE

        for crate in self.crates.values():
            self._draw_harbor(frame, crate.harbor, crate.color)
        for cell_pos, controller in self.pads.items():
            self._draw_pad(frame, cell_pos, COLOR_PAD_A if controller == "A" else COLOR_PAD_B)
        for cell_pos, direction in self.fans.items():
            self._draw_fan(frame, cell_pos, direction)
        for wall in self.walls:
            self._fill_cell(frame, wall, COLOR_WALL, inset=0)
        for cell_pos, controller in self.gates.items():
            open_gate = controller in self.open_controllers
            self._draw_gate(frame, cell_pos, COLOR_PAD_A if controller == "A" else COLOR_PAD_B, open_gate)

        for crate in self.crates.values():
            offset = (0.0, 0.0)
            if slide and crate.cid in slide:
                old, new, t = slide[crate.cid]
                offset = ((old[0] - new[0]) * (1.0 - t), (old[1] - new[1]) * (1.0 - t))
            self._draw_crate(frame, crate, offset)
        self._draw_worker(frame)
        self._draw_hud(frame)
        return frame

    def _cell_rect(self, cell_pos: tuple[int, int], offset: tuple[float, float] = (0, 0)) -> tuple[int, int, int]:
        left, top, cell = self._layout()
        x = round(left + (cell_pos[0] + offset[0]) * cell)
        y = round(top + (cell_pos[1] + offset[1]) * cell)
        return x, y, cell

    def _fill_cell(self, frame: np.ndarray, cell_pos: tuple[int, int], color: int, inset: int = 1) -> None:
        x, y, cell = self._cell_rect(cell_pos)
        frame[y + inset : y + cell - inset, x + inset : x + cell - inset] = color

    def _draw_harbor(self, frame: np.ndarray, cell_pos: tuple[int, int], color: int) -> None:
        x, y, cell = self._cell_rect(cell_pos)
        frame[y + 1 : y + cell - 1, x + 1 : x + cell - 1] = COLOR_WHITE
        frame[y + 2 : y + cell - 2, x + 2 : x + cell - 2] = color

    def _draw_pad(self, frame: np.ndarray, cell_pos: tuple[int, int], color: int) -> None:
        x, y, cell = self._cell_rect(cell_pos)
        frame[y + cell // 2 : y + cell - 1, x + 1 : x + cell - 1] = color
        frame[y + cell - 2 : y + cell - 1, x + 2 : x + cell - 2] = COLOR_OUTLINE

    def _draw_gate(self, frame: np.ndarray, cell_pos: tuple[int, int], color: int, open_gate: bool) -> None:
        x, y, cell = self._cell_rect(cell_pos)
        if open_gate:
            frame[y + 1 : y + cell - 1, x + 1] = color
            frame[y + 1 : y + cell - 1, x + cell - 2] = color
        else:
            frame[y : y + cell, x : x + cell] = COLOR_WALL
            frame[y + 1 : y + cell - 1, x + cell // 2] = color

    def _draw_fan(self, frame: np.ndarray, cell_pos: tuple[int, int], direction: str) -> None:
        x, y, cell = self._cell_rect(cell_pos)
        frame[y + 1 : y + cell - 1, x + 1 : x + cell - 1] = COLOR_FAN
        self._draw_arrow(frame, x, y, cell, direction, COLOR_WHITE)

    def _draw_crate(self, frame: np.ndarray, crate: CrateState, offset: tuple[float, float]) -> None:
        x, y, cell = self._cell_rect(crate.pos, offset)
        if self.selected_crate_id == crate.cid:
            frame[y : y + cell, x : x + cell] = COLOR_WORKER
        frame[y + 1 : y + cell - 1, x + 1 : x + cell - 1] = COLOR_OUTLINE
        frame[y + 2 : y + cell - 2, x + 2 : x + cell - 2] = crate.color
        self._draw_arrow(frame, x, y, cell, crate.sail_dir, COLOR_WHITE)

    def _draw_worker(self, frame: np.ndarray) -> None:
        x, y, cell = self._cell_rect(self.worker_pos)
        frame[y + 1 : y + cell - 1, x + 2 : x + cell - 2] = COLOR_OUTLINE
        frame[y + 2 : y + cell - 2, x + 2 : x + cell - 2] = COLOR_WORKER
        if cell >= 5:
            frame[y + 2, x + 2 : x + cell - 2] = COLOR_WHITE

    def _draw_arrow(self, frame: np.ndarray, x: int, y: int, cell: int, direction: str, color: int) -> None:
        cx = x + cell // 2
        cy = y + cell // 2
        if direction == "N":
            frame[y + 1 : cy + 1, cx] = color
            frame[y + 2, max(x + 1, cx - 1) : min(x + cell - 1, cx + 2)] = color
        elif direction == "S":
            frame[cy : y + cell - 1, cx] = color
            frame[y + cell - 3, max(x + 1, cx - 1) : min(x + cell - 1, cx + 2)] = color
        elif direction == "E":
            frame[cy, cx : x + cell - 1] = color
            frame[max(y + 1, cy - 1) : min(y + cell - 1, cy + 2), x + cell - 3] = color
        else:
            frame[cy, x + 1 : cx + 1] = color
            frame[max(y + 1, cy - 1) : min(y + cell - 1, cy + 2), x + 2] = color

    def _draw_hud(self, frame: np.ndarray) -> None:
        frame[0:5, 0:64] = COLOR_BG
        self._draw_arrow(frame, 1, 0, 5, self.wind_dir, COLOR_FAN)
        bar_x = 8
        bar_w = 54
        filled = max(0, min(bar_w, int(bar_w * self.remaining_steps / max(1, self.step_budget))))
        frame[2:4, bar_x : bar_x + bar_w] = COLOR_WALL
        if filled:
            frame[2:4, bar_x : bar_x + filled] = COLOR_FAN
