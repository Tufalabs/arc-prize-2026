from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_SOLID = 2
COLOR_PLAYER = 3
COLOR_PLAYER_FLASH = 4
COLOR_ARROW = 5
COLOR_TARGET_FRAME = 6
COLOR_TARGET_CENTER = 7
COLOR_TARGET_HIT = 8
COLOR_EXIT_CLOSED = 9
COLOR_EXIT_OPEN = 10
COLOR_SPIKES_OFF = 11
COLOR_SPIKES_ON = 12
COLOR_MIRROR = 13
COLOR_BREAKABLE = 14
COLOR_TIME = 15

MOVE_BY_ACTION = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

FACING_TO_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

ACTION_TO_FACING = {
    GameAction.ACTION1: "up",
    GameAction.ACTION2: "down",
    GameAction.ACTION3: "left",
    GameAction.ACTION4: "right",
}

SLASH_REFLECT = {"up": "right", "right": "up", "down": "left", "left": "down"}

BACKSLASH_REFLECT = {"up": "left", "left": "up", "down": "right", "right": "down"}


@dataclass
class MovingTargetSpec:
    centers: list[tuple[int, int]]


LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1 - First shot opens the way",
        "time_rows": 2,
        "rows": [
            "................................",
            "................................",
            "################################",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "################################",
        ],
        "player": (4, 15, "right"),
        "targets": [(15, 8)],
        "moving_targets": [],
        "exit_center": (25, 8),
        "mirrors": {},
        "breakables": [],
        "spikes": [],
    },
    {
        "name": "Level 2 - Shoot through what you cannot walk through",
        "time_rows": 2,
        "rows": [
            "................................",
            "................................",
            "################################",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#||||||||||||||||||||||||||||||#",
            "#..............................#",
            "#..............................#",
            "################.###############",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "################################",
        ],
        "player": (6, 15, "right"),
        "targets": [(16, 5)],
        "moving_targets": [],
        "exit_center": (25, 14),
        "mirrors": {},
        "breakables": [],
        "spikes": [],
    },
    {
        "name": "Level 3 - Two targets and breakable gate",
        "time_rows": 2,
        "rows": [
            "................................",
            "................................",
            "################################",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#||||||||||||||||||||||||||||||#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............%..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "################################",
        ],
        "player": (4, 17, "right"),
        "targets": [(7, 5), (23, 13)],
        "moving_targets": [],
        "exit_center": (26, 17),
        "mirrors": {},
        "breakables": [(16, 14)],
        "spikes": [],
    },
    {
        "name": "Level 4 - Mirrors",
        "time_rows": 3,
        "rows": [
            "................................",
            "................................",
            "................................",
            "################################",
            "#..............................#",
            "#.........#####................#",
            "#.........#...#................#",
            "#.........#...#................#",
            "#.........#...#................#",
            "#.........##.##................#",
            "#..........#.#.................#",
            "#..........#.#.................#",
            "#..........#.#.................#",
            "#..........#.#.................#",
            "#..............................#",
            "#...........#..................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "################################",
        ],
        "player": (6, 14, "right"),
        "targets": [(12, 7)],
        "moving_targets": [],
        "exit_center": (25, 17),
        "mirrors": {(12, 14): "\\"},
        "breakables": [],
        "spikes": [],
    },
    {
        "name": "Level 5 - Spikes and moving target timing",
        "time_rows": 3,
        "rows": [
            "................................",
            "................................",
            "................................",
            "################################",
            "#...............#..............#",
            "#...............#..###########.#",
            "#...............#..#.........#.#",
            "#...............#............#.#",
            "#...............#..#.........#.#",
            "#...............#..###########.#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#..............................#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "################################",
        ],
        "player": (4, 17, "up"),
        "targets": [(7, 7)],
        "moving_targets": [MovingTargetSpec(centers=[(22, 7), (26, 7)])],
        "exit_center": (26, 17),
        "mirrors": {},
        "breakables": [],
        "spikes": [(16, 13), (24, 15), (25, 15), (26, 15)],
    },
    {
        "name": "Level 6 - All together",
        "time_rows": 3,
        "rows": [
            "................................",
            "................................",
            "................................",
            "################################",
            "#...............#..............#",
            "#...............#..###########.#",
            "#...............#..#.........#.#",
            "#|||||||||||||||#............#.#",
            "#...............#..#.........#.#",
            "#...............#..###########.#",
            "#...............#.....#####....#",
            "#..............%......#...#....#",
            "#...............#.....#...#....#",
            "#...............#.....##.##....#",
            "#...............#..............#",
            "#...............#.......#......#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "################################",
        ],
        "player": (4, 16, "right"),
        "targets": [(7, 5), (24, 11)],
        "moving_targets": [MovingTargetSpec(centers=[(22, 7), (26, 7)])],
        "exit_center": (26, 17),
        "mirrors": {(24, 14): "\\"},
        "breakables": [(15, 11)],
        "spikes": [(16, 11), (25, 15), (26, 15), (27, 15)],
    },
]


def _build_level(spec: dict) -> Level:
    rows = [str(row) for row in spec["rows"]]
    if not rows:
        raise ValueError("Level rows must not be empty.")
    width = len(rows[0])
    height = len(rows)
    for row in rows:
        if len(row) != width:
            raise ValueError("All level rows must have equal width.")

    pixels = np.full((height, width), int(COLOR_BG), dtype=np.int8)
    canvas = Sprite(
        pixels=pixels, name="canvas", x=0, y=0, layer=0, collidable=False, tags=["sys_click", "sys_every_pixel"]
    )

    moving_centers = [list(mt.centers) for mt in spec.get("moving_targets") or []]

    return Level(
        name=str(spec["name"]),
        grid_size=(width, height),
        sprites=[canvas],
        data={
            "rows": rows,
            "time_rows": int(spec["time_rows"]),
            "player": tuple(spec["player"]),
            "targets": [tuple(center) for center in spec.get("targets") or []],
            "moving_targets": moving_centers,
            "exit_center": tuple(spec["exit_center"]),
            "mirrors": [
                (int(pos[0]), int(pos[1]), str(orient)) for pos, orient in dict(spec.get("mirrors") or {}).items()
            ],
            "breakables": [tuple(pos) for pos in spec.get("breakables") or []],
            "spikes": [tuple(pos) for pos in spec.get("spikes") or []],
        },
    )


class GridArchery(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        first_width, first_height = levels[0].grid_size or (32, 20)
        camera = Camera(0, 0, first_width, first_height, COLOR_BG, COLOR_BG, [])
        super().__init__(
            game_id="grid_archery-0001",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._rows = [str(row) for row in (level.get_data("rows") or [])]
        self._height = len(self._rows)
        self._width = len(self._rows[0]) if self._rows else 0
        self._time_rows = int(level.get_data("time_rows") or 2)
        self._time_total = self._time_rows * self._width
        self._time_remaining = self._time_total
        self._time_slots = self._build_time_slots()

        player_x, player_y, facing = level.get_data("player")
        self._player_x = int(player_x)
        self._player_y = int(player_y)
        self._player_facing = str(facing)
        self._fire_flash = 0

        self._arrow_pos: tuple[int, int] | None = None
        self._arrow_facing: str | None = None
        self._tail_pos: tuple[int, int] | None = None
        self._tail_life = 0

        self._walls: set[tuple[int, int]] = set()
        self._fences: set[tuple[int, int]] = set()
        for y, row in enumerate(self._rows):
            for x, ch in enumerate(row):
                if ch == "#":
                    self._walls.add((x, y))
                elif ch == "|":
                    self._fences.add((x, y))

        self._intact_breakables: set[tuple[int, int]] = {
            tuple(map(int, pos)) for pos in (level.get_data("breakables") or [])
        }
        self._crumbles: dict[tuple[int, int], int] = {}

        self._mirrors: dict[tuple[int, int], str] = {}
        for mx, my, orient in level.get_data("mirrors") or []:
            self._mirrors[(int(mx), int(my))] = str(orient)

        self._spikes: set[tuple[int, int]] = {tuple(map(int, pos)) for pos in (level.get_data("spikes") or [])}
        self._tick = 0
        self._spikes_on = False

        self._targets: list[dict] = []
        for center in level.get_data("targets") or []:
            self._targets.append({"center": tuple(map(int, center)), "path": None, "path_idx": 0, "hit": False})
        for path in level.get_data("moving_targets") or []:
            centers = [tuple(map(int, center)) for center in path]
            if not centers:
                continue
            self._targets.append({"center": centers[0], "path": centers, "path_idx": 0, "hit": False})

        self._exit_center = tuple(map(int, level.get_data("exit_center") or (0, 0)))
        self._exit_open = False
        self._exit_blink = 0

        self._render()

    def _build_time_slots(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for y in range(self._time_rows - 1, -1, -1):
            for x in range(self._width - 1, -1, -1):
                out.append((x, y))
        return out

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 0 <= y < self._height

    def _target_cells(self, center: tuple[int, int]) -> set[tuple[int, int]]:
        cx, cy = center
        return {(cx + dx, cy + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}

    def _find_target_index_at(self, x: int, y: int, *, only_unhit: bool = False) -> int | None:
        for idx, target in enumerate(self._targets):
            if only_unhit and target["hit"]:
                continue
            if (x, y) in self._target_cells(target["center"]):
                return idx
        return None

    def _all_targets_hit(self) -> bool:
        return all(bool(target["hit"]) for target in self._targets)

    def _exit_cells(self) -> set[tuple[int, int]]:
        cx, cy = self._exit_center
        return {(cx + dx, cy + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}

    def _is_closed_exit_tile(self, x: int, y: int) -> bool:
        return (not self._exit_open) and (x, y) in self._exit_cells()

    def _is_player_blocked(self, x: int, y: int) -> bool:
        if not self._in_bounds(x, y):
            return True
        if (x, y) in self._walls:
            return True
        if (x, y) in self._fences:
            return True
        if (x, y) in self._intact_breakables:
            return True
        if self._is_closed_exit_tile(x, y):
            return True
        return self._find_target_index_at(x, y, only_unhit=False) is not None

    def _is_wall_for_arrow_spawn(self, x: int, y: int) -> bool:
        if not self._in_bounds(x, y):
            return True
        return (x, y) in self._walls

    def _toggle_click_mirror(self) -> None:
        data = self.action.data or {}
        display_x = int(data.get("x", -1))
        display_y = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(display_x, display_y)
        if grid_pos is None:
            return
        gx, gy = int(grid_pos[0]), int(grid_pos[1])
        orient = self._mirrors.get((gx, gy))
        if orient is None:
            return
        self._mirrors[(gx, gy)] = "/" if orient == "\\" else "\\"

    def _apply_move_action(self, action) -> None:
        facing = ACTION_TO_FACING.get(action)
        if facing is None:
            return
        self._player_facing = facing
        dx, dy = FACING_TO_DELTA[facing]
        nx = self._player_x + dx
        ny = self._player_y + dy
        if self._is_player_blocked(nx, ny):
            return
        self._player_x = nx
        self._player_y = ny

    def _try_fire(self, action) -> None:
        if action != GameAction.ACTION5:
            return
        if self._arrow_pos is not None:
            return
        dx, dy = FACING_TO_DELTA[self._player_facing]
        sx = self._player_x + dx
        sy = self._player_y + dy
        if self._is_wall_for_arrow_spawn(sx, sy):
            return
        self._arrow_pos = (sx, sy)
        self._arrow_facing = self._player_facing
        self._fire_flash = 1

    def _reflect(self, mirror: str, facing: str) -> str:
        if mirror == "/":
            return SLASH_REFLECT[facing]
        return BACKSLASH_REFLECT[facing]

    def _move_arrow(self) -> bool:
        if self._arrow_pos is None or self._arrow_facing is None:
            return False

        px, py = self._arrow_pos
        dx, dy = FACING_TO_DELTA[self._arrow_facing]
        nx = px + dx
        ny = py + dy

        self._tail_pos = (px, py)
        self._tail_life = 1

        if not self._in_bounds(nx, ny):
            self._arrow_pos = None
            self._arrow_facing = None
            return False

        if (nx, ny) == (self._player_x, self._player_y):
            return True

        if (nx, ny) in self._walls or self._is_closed_exit_tile(nx, ny):
            self._arrow_pos = None
            self._arrow_facing = None
            return False

        if (nx, ny) in self._intact_breakables:
            self._intact_breakables.remove((nx, ny))
            self._crumbles[(nx, ny)] = 2
            self._arrow_pos = None
            self._arrow_facing = None
            return False

        mirror = self._mirrors.get((nx, ny))
        if mirror is not None:
            self._arrow_pos = (nx, ny)
            self._arrow_facing = self._reflect(mirror, self._arrow_facing)
            return False

        target_idx = self._find_target_index_at(nx, ny, only_unhit=False)
        if target_idx is not None:
            if not self._targets[target_idx]["hit"]:
                self._targets[target_idx]["hit"] = True
            self._arrow_pos = None
            self._arrow_facing = None
            return False

        self._arrow_pos = (nx, ny)
        return False

    def _animate_environment(self) -> None:
        next_crumbles: dict[tuple[int, int], int] = {}
        for pos, ttl in self._crumbles.items():
            next_ttl = int(ttl) - 1
            if next_ttl > 0:
                next_crumbles[pos] = next_ttl
        self._crumbles = next_crumbles

        self._tick += 1
        self._spikes_on = ((self._tick // 2) % 2) == 1

        for target in self._targets:
            if target["hit"]:
                continue
            path = target["path"]
            if not path:
                continue
            idx = (int(target["path_idx"]) + 1) % len(path)
            target["path_idx"] = idx
            target["center"] = path[idx]

    def _resolve_hazards(self) -> bool:
        return bool((self._player_x, self._player_y) in self._spikes and self._spikes_on)

    def _update_exit(self) -> None:
        if self._all_targets_hit() and not self._exit_open:
            self._exit_open = True
            self._exit_blink = 3
        if self._exit_open and self._exit_blink > 0:
            self._exit_blink -= 1

    def _decrement_time(self) -> bool:
        self._time_remaining -= 1
        return self._time_remaining <= 0

    def _has_won(self) -> bool:
        return self._exit_open and (self._player_x, self._player_y) == self._exit_center

    def _render(self) -> None:
        grid = [[COLOR_FLOOR for _ in range(self._width)] for _ in range(self._height)]

        for y, row in enumerate(self._rows):
            for x, ch in enumerate(row):
                if ch in {"#", "|"}:
                    grid[y][x] = COLOR_SOLID
                else:
                    grid[y][x] = COLOR_FLOOR

        for x, y in self._intact_breakables:
            if self._in_bounds(x, y):
                grid[y][x] = COLOR_BREAKABLE
        for x, y in self._crumbles:
            if self._in_bounds(x, y):
                grid[y][x] = COLOR_BREAKABLE

        for x, y in self._spikes:
            if self._in_bounds(x, y):
                grid[y][x] = COLOR_SPIKES_ON if self._spikes_on else COLOR_SPIKES_OFF

        for target in self._targets:
            cx, cy = target["center"]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    tx, ty = cx + dx, cy + dy
                    if not self._in_bounds(tx, ty):
                        continue
                    if target["hit"]:
                        grid[ty][tx] = COLOR_TARGET_HIT
                    elif dx == 0 and dy == 0:
                        grid[ty][tx] = COLOR_TARGET_CENTER
                    else:
                        grid[ty][tx] = COLOR_TARGET_FRAME

        cx, cy = self._exit_center
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ex, ey = cx + dx, cy + dy
                if not self._in_bounds(ex, ey):
                    continue
                if self._exit_open:
                    grid[ey][ex] = COLOR_EXIT_OPEN
                else:
                    grid[ey][ex] = COLOR_EXIT_CLOSED

        for (mx, my), _orient in self._mirrors.items():
            if self._in_bounds(mx, my):
                grid[my][mx] = COLOR_MIRROR

        if self._tail_pos is not None:
            tx, ty = self._tail_pos
            if self._in_bounds(tx, ty):
                grid[ty][tx] = COLOR_ARROW
        if self._arrow_pos is not None:
            ax, ay = self._arrow_pos
            if self._in_bounds(ax, ay):
                grid[ay][ax] = COLOR_ARROW

        if self._in_bounds(self._player_x, self._player_y):
            grid[self._player_y][self._player_x] = COLOR_PLAYER_FLASH if self._fire_flash > 0 else COLOR_PLAYER

        filled = max(0, min(self._time_total, self._time_remaining))
        for idx, (x, y) in enumerate(self._time_slots):
            grid[y][x] = COLOR_TIME if idx < filled else COLOR_BG

        canvas = self.current_level.get_sprites_by_name("canvas")[0]
        canvas.pixels = np.array(grid, dtype=np.int8)

    def step(self) -> None:
        if getattr(self.action.id, "value", self.action.id) == 0 and self.level_index > 0:
            # arcengine's perform_action loop runs step() once after
            # handle_reset; without this guard the sim would advance a
            # tick, so mid-play RESET on any level entered via
            # next_level() would land one tick past the frame the
            # client saw on arrival. Level 0 keeps the legacy tick to
            # preserve env.reset()'s observation and the DSL trace.
            self.complete_action()
            return

        if self._tail_life > 0:
            self._tail_life -= 1
            if self._tail_life <= 0:
                self._tail_pos = None
        if self._fire_flash > 0:
            self._fire_flash -= 1

        action = self.action.id

        if action == GameAction.ACTION6:
            self._toggle_click_mirror()

        self._apply_move_action(action)
        self._try_fire(action)

        if self._move_arrow():
            self.lose()
            self.complete_action()
            return

        self._animate_environment()

        if self._resolve_hazards():
            self.lose()
            self.complete_action()
            return

        self._update_exit()

        if self._decrement_time():
            self.lose()
            self.complete_action()
            return

        if self._has_won():
            self.next_level()
            self.complete_action()
            return

        self._render()
        self.complete_action()
