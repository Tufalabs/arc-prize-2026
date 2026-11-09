from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

# Palette IDs required by the spec.
COLOR_UI_EMPTY = 0
COLOR_UNCAPTURED = 1
COLOR_CAPTURED = 2
COLOR_SAFE = 3
COLOR_SOLID = 4
COLOR_TRAIL_A = 5
COLOR_TRAIL_B = 6
COLOR_PLAYER_A = 7
COLOR_PLAYER_B = 8
COLOR_WANDERER_A = 9
COLOR_WANDERER_B = 10
COLOR_HUNTER_A = 11
COLOR_HUNTER_B = 12
COLOR_SPARK_A = 13
COLOR_SPARK_B = 14
COLOR_GOAL = 15

TILE_SAFE = "safe"
TILE_UNCAPTURED = "uncaptured"
TILE_OBSTACLE = "obstacle"
TILE_GOAL = "goal"
TILE_PULSE_A = "pulse_a"
TILE_PULSE_B = "pulse_b"

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
WAIT_ACTION_ID = int(GameAction.ACTION5.value)


@dataclass(frozen=True)
class LevelConfig:
    width: int
    height: int
    capture_pct: int
    max_time: int
    fail_penalty: int
    fill_rate: int
    wanderer_speeds: tuple[int, ...]
    hunter_speeds: tuple[int, ...]
    layout: tuple[str, ...]


@dataclass
class EnemyState:
    kind: str
    x: int
    y: int
    start_x: int
    start_y: int
    w: int
    h: int
    speed: int
    dx: int = 0
    dy: int = 0
    spark_path_idx: int = 0
    spark_dir: int = 1


LEVEL_CONFIGS: tuple[LevelConfig, ...] = (
    LevelConfig(
        width=24,
        height=19,
        capture_pct=20,
        max_time=800,
        fail_penalty=120,
        fill_rate=40,
        wanderer_speeds=(2,),
        hunter_speeds=(),
        layout=(
            "========================",
            "########################",
            "########################",
            "########################",
            "###..................###",
            "###..................###",
            "###..................###",
            "###..................###",
            "###..................###",
            "###........!!........###",
            "###........!!........###",
            "###..................###",
            "###..................###",
            "###..................###",
            "###..................###",
            "###..................###",
            "#####################***",
            "#@@##################***",
            "#@@##################***",
        ),
    ),
    LevelConfig(
        width=28,
        height=21,
        capture_pct=35,
        max_time=900,
        fail_penalty=140,
        fill_rate=50,
        wanderer_speeds=(1, 1),
        hunter_speeds=(),
        layout=(
            "============================",
            "############################",
            "############################",
            "############################",
            "###......................###",
            "###......................###",
            "###.....!!...............###",
            "###.....!!...............###",
            "###......................###",
            "###.........XXXX.........###",
            "###.........XXXX.........###",
            "###.........XXXX.........###",
            "###.........XXXX.........###",
            "###...............!!.....###",
            "###...............!!.....###",
            "###......................###",
            "###......................###",
            "###......................###",
            "#########################***",
            "#@@######################***",
            "#@@######################***",
        ),
    ),
    LevelConfig(
        width=32,
        height=23,
        capture_pct=45,
        max_time=1000,
        fail_penalty=160,
        fill_rate=55,
        wanderer_speeds=(1,),
        hunter_speeds=(1,),
        layout=(
            "================================",
            "#############################***",
            "#############################***",
            "#############################***",
            "###.............XX...........###",
            "###.............XX...........###",
            "###...!!........XX...........###",
            "###...!!........XX...........###",
            "###.............XX...........###",
            "###..........................###",
            "###..........................###",
            "###.............XX...........###",
            "###.......#####.XX...........###",
            "###.......#####.XX.....$$....###",
            "###.......#####.XX.....$$....###",
            "###.......#####.XX...........###",
            "###.......#####.XX...........###",
            "###..........................###",
            "###.............XX...........###",
            "###.............XX...........###",
            "################################",
            "#@@#############################",
            "#@@#############################",
        ),
    ),
    LevelConfig(
        width=36,
        height=25,
        capture_pct=55,
        max_time=1100,
        fail_penalty=180,
        fill_rate=60,
        wanderer_speeds=(1, 1),
        hunter_speeds=(1,),
        layout=(
            "====================================",
            "####################################",
            "####################################",
            "####################^###############",
            "###..............................###",
            "###..............................###",
            "###..............!!..............###",
            "###..............!!..............###",
            "###.....XXXXXX...................###",
            "###.....XXXXXX...................###",
            "###.....XXXXXX...................###",
            "###.....XXXXXX............$$.....###",
            "###.......................$$.....###",
            "###..............................###",
            "###...................XXXXXXXX...###",
            "###.......!!..........XXXXXXXX...###",
            "###.......!!..........XXXXXXXX...###",
            "###..............................###",
            "###..............................###",
            "###..............................###",
            "###..............................###",
            "###..............................###",
            "#################################***",
            "#@@##############################***",
            "#@@##############################***",
        ),
    ),
    LevelConfig(
        width=40,
        height=27,
        capture_pct=60,
        max_time=1200,
        fail_penalty=200,
        fill_rate=65,
        wanderer_speeds=(1,),
        hunter_speeds=(1, 1),
        layout=(
            "========================================",
            "####################|###################",
            "####################|###################",
            "######^#############|###################",
            "###.................|................###",
            "###.................|................###",
            "###.................|................###",
            "###.................|................###",
            "###.......!!........|................###",
            "###.......!!........|...###..........###",
            "###.................|...###..........###",
            "###.................|...###..........###",
            "###.................|.......$$.......###",
            "###.................|.......$$.......###",
            "###.................|................###",
            "###...XXXXXXXX......|................###",
            "###...XXXXXXXX......|................###",
            "###...XXXXXXXX......|................###",
            "###.................|.........$$.....###",
            "###.................|.........$$.....###",
            "###.................|................###",
            "###.................|................###",
            "###.................|................###",
            "###.................|................###",
            "####################|################***",
            "#@@#################|################***",
            "#@@#################|################***",
        ),
    ),
    LevelConfig(
        width=48,
        height=31,
        capture_pct=70,
        max_time=1400,
        fail_penalty=220,
        fill_rate=70,
        wanderer_speeds=(1, 1, 1),
        hunter_speeds=(1, 1),
        layout=(
            "================================================",
            "################|###############/###############",
            "################|###############/#######^#######",
            "######^#########|###############/###############",
            "###.............|.............../............###",
            "###.............|.............../............###",
            "###.......!!....|.............../............###",
            "###.......!!....|.............../............###",
            "###.............|.............../.....###....###",
            "###.............|.......!!....../.....###....###",
            "###....###......|.......!!....../.....###....###",
            "###....###......|...........$$../............###",
            "###....###......|...........$$../............###",
            "###.............|.............../............###",
            "###.............|.......XXXXXX../............###",
            "###.............|.......XXXXXX../............###",
            "###.............|.......XXXXXX../............###",
            "###.............|.$$....XXXXXX../............###",
            "###.............|.$$.###......../............###",
            "###.............|....###......../............###",
            "###.............|....###......../.....!!.....###",
            "###.............|.............../.....!!.....###",
            "###.............|.............../............###",
            "###.............|.............../............###",
            "###.............|.............../............###",
            "###.............|.............../............###",
            "###.............|.............../............###",
            "###.............|.............../............###",
            "################|###############/############***",
            "#@@#############|###############/############***",
            "#@@#############|###############/############***",
        ),
    ),
)


def _rect_cells(x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    return [(cx, cy) for cy in range(y, y + h) for cx in range(x, x + w)]


def _parse_level(config: LevelConfig) -> dict:
    rows = list(config.layout)
    if len(rows) != config.height:
        raise ValueError("Layout height does not match config.")
    if any(len(row) != config.width for row in rows):
        raise ValueError("Layout width does not match config.")

    grid = [[TILE_UNCAPTURED for _ in range(config.width)] for _ in range(config.height)]
    player_start: tuple[int, int] | None = None
    goal_cells: set[tuple[int, int]] = set()
    wanderer_cells: list[tuple[int, int]] = []
    hunter_cells: list[tuple[int, int]] = []
    spark_cells: list[tuple[int, int]] = []

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if y == 0:
                grid[y][x] = TILE_OBSTACLE
                continue
            if ch == "#":
                grid[y][x] = TILE_SAFE
            elif ch == ".":
                grid[y][x] = TILE_UNCAPTURED
            elif ch == "X":
                grid[y][x] = TILE_OBSTACLE
            elif ch == "|":
                grid[y][x] = TILE_PULSE_A
            elif ch == "/":
                grid[y][x] = TILE_PULSE_B
            elif ch == "*":
                grid[y][x] = TILE_GOAL
                goal_cells.add((x, y))
            elif ch == "@":
                grid[y][x] = TILE_SAFE
            elif ch == "!":
                grid[y][x] = TILE_UNCAPTURED
                wanderer_cells.append((x, y))
            elif ch == "$":
                grid[y][x] = TILE_UNCAPTURED
                hunter_cells.append((x, y))
            elif ch == "^":
                grid[y][x] = TILE_SAFE
                spark_cells.append((x, y))
            else:
                raise ValueError(f"Unsupported layout cell {ch!r} at {(x, y)}")

    for y in range(config.height - 1):
        for x in range(config.width - 1):
            if rows[y][x] == "@" and rows[y][x + 1] == "@" and rows[y + 1][x] == "@" and rows[y + 1][x + 1] == "@":
                player_start = (x, y)

    if player_start is None:
        raise ValueError("Could not infer 2x2 player start block")

    goal_xs = [x for x, _ in goal_cells]
    goal_ys = [y for _, y in goal_cells]
    if not goal_cells or (max(goal_xs) - min(goal_xs) != 2) or (max(goal_ys) - min(goal_ys) != 2):
        raise ValueError("Goal must be a 3x3 block")

    def _find_blocks(cells: list[tuple[int, int]], mark: str) -> list[tuple[int, int]]:
        tops: list[tuple[int, int]] = []
        cell_set = set(cells)
        for x, y in sorted(cell_set):
            if (x - 1, y) in cell_set or (x, y - 1) in cell_set:
                continue
            block = {(x + dx, y + dy) for dy in range(2) for dx in range(2)}
            if block.issubset(cell_set):
                tops.append((x, y))
        if len(tops) != len(cells) // 4:
            raise ValueError(f"Malformed {mark} 2x2 placement")
        return tops

    wanderer_starts = _find_blocks(wanderer_cells, "wanderer")
    hunter_starts = _find_blocks(hunter_cells, "hunter")
    if len(wanderer_starts) != len(config.wanderer_speeds):
        raise ValueError("Wanderer count mismatch")
    if len(hunter_starts) != len(config.hunter_speeds):
        raise ValueError("Hunter count mismatch")

    uncaptured_cells = {
        (x, y) for y in range(1, config.height) for x in range(config.width) if grid[y][x] == TILE_UNCAPTURED
    }

    if not uncaptured_cells:
        raise ValueError("Level has no fillable cells")

    min_dot_x = min(x for x, _ in uncaptured_cells)
    max_dot_x = max(x for x, _ in uncaptured_cells)
    min_dot_y = min(y for _, y in uncaptured_cells)
    max_dot_y = max(y for _, y in uncaptured_cells)

    ring: list[tuple[int, int]] = []
    left = max(0, min_dot_x - 1)
    right = min(config.width - 1, max_dot_x + 1)
    top = max(1, min_dot_y - 1)
    bottom = min(config.height - 1, max_dot_y + 1)

    for x in range(left, right + 1):
        ring.append((x, top))
    for y in range(top + 1, bottom + 1):
        ring.append((right, y))
    for x in range(right - 1, left - 1, -1):
        ring.append((x, bottom))
    for y in range(bottom - 1, top, -1):
        ring.append((left, y))

    sparks: list[EnemyState] = []
    ring_index_by_pos = {pos: idx for idx, pos in enumerate(ring)}
    for idx, (sx, sy) in enumerate(sorted(spark_cells)):
        if (sx, sy) in ring_index_by_pos:
            start_idx = ring_index_by_pos[(sx, sy)]
        else:
            start_idx = min(range(len(ring)), key=lambda ridx: abs(ring[ridx][0] - sx) + abs(ring[ridx][1] - sy))
            sx, sy = ring[start_idx]
        sparks.append(
            EnemyState(
                kind="spark",
                x=sx,
                y=sy,
                start_x=sx,
                start_y=sy,
                w=1,
                h=1,
                speed=1,
                spark_path_idx=start_idx,
                spark_dir=1 if idx % 2 == 0 else -1,
            )
        )

    enemy_states: list[EnemyState] = []
    wanderer_dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for idx, ((x, y), speed) in enumerate(zip(sorted(wanderer_starts), config.wanderer_speeds, strict=False)):
        dx, dy = wanderer_dirs[idx % len(wanderer_dirs)]
        enemy_states.append(
            EnemyState(kind="wanderer", x=x, y=y, start_x=x, start_y=y, w=2, h=2, speed=int(speed), dx=dx, dy=dy)
        )
    hunter_dirs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    for idx, ((x, y), speed) in enumerate(zip(sorted(hunter_starts), config.hunter_speeds, strict=False)):
        dx, dy = hunter_dirs[idx % len(hunter_dirs)]
        enemy_states.append(
            EnemyState(kind="hunter", x=x, y=y, start_x=x, start_y=y, w=2, h=2, speed=int(speed), dx=dx, dy=dy)
        )
    enemy_states.extend(sparks)

    return {
        "grid": grid,
        "player_start": player_start,
        "goal_cells": goal_cells,
        "goal_top_left": (min(goal_xs), min(goal_ys)),
        "uncaptured_cells": uncaptured_cells,
        "fillable_total": len(uncaptured_cells),
        "ring": ring,
        "capture_required": max(1, int((config.capture_pct / 100.0) * len(uncaptured_cells))),
    }


PARSED_LEVELS = tuple(_parse_level(cfg) for cfg in LEVEL_CONFIGS)
ACTIVE_LEVEL_COUNT = 1


class TerritoryCapture(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [
            Level(
                name=f"Territory Capture {idx + 1}",
                grid_size=(cfg.width, cfg.height),
                sprites=[
                    Sprite(
                        pixels=np.zeros((cfg.height, cfg.width), dtype=np.int8),
                        name="board",
                        x=0,
                        y=0,
                        layer=0,
                        tags=["board", "sys_static"],
                        collidable=False,
                    )
                ],
                data={"level_idx": idx},
            )
            for idx, cfg in enumerate(LEVEL_CONFIGS[:ACTIVE_LEVEL_COUNT])
        ]
        camera = Camera(width=LEVEL_CONFIGS[0].width, height=LEVEL_CONFIGS[0].height, background=COLOR_UI_EMPTY)
        super().__init__(
            game_id="territory_capture-0001",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

        self._level_idx = 0
        self._board_sprite: Sprite | None = None
        self._player_x = 0
        self._player_y = 0
        self._trail: set[tuple[int, int]] = set()
        self._captured: set[tuple[int, int]] = set()
        self._fill_queue: deque[tuple[int, int]] = deque()
        self._fill_pending: set[tuple[int, int]] = set()
        self._enemies: list[EnemyState] = []
        self._time = 0
        self._tick = 0
        self._goal_open = False

    def on_set_level(self, level: Level) -> None:
        level_idx = int(level.get_data("level_idx") or 0)
        self._load_level(level_idx)

    def _load_level(self, level_idx: int) -> None:
        self._level_idx = int(level_idx)
        cfg = LEVEL_CONFIGS[self._level_idx]
        parsed = PARSED_LEVELS[self._level_idx]

        self.camera.width = cfg.width
        self.camera.height = cfg.height

        board = self.current_level.get_sprites_by_name("board")
        self._board_sprite = board[0] if board else None
        self._player_x, self._player_y = parsed["player_start"]
        self._trail = set()
        self._captured = set()
        self._fill_queue = deque()
        self._fill_pending = set()
        self._goal_open = False
        self._time = cfg.max_time
        self._tick = 0

        self._enemies = [
            EnemyState(
                kind=e.kind,
                x=e.start_x,
                y=e.start_y,
                start_x=e.start_x,
                start_y=e.start_y,
                w=e.w,
                h=e.h,
                speed=e.speed,
                dx=e.dx,
                dy=e.dy,
                spark_path_idx=e.spark_path_idx,
                spark_dir=e.spark_dir,
            )
            for e in _initial_enemy_states(self._level_idx)
        ]
        self._render()

    def _current_cfg(self) -> LevelConfig:
        return LEVEL_CONFIGS[self._level_idx]

    def _current_parsed(self) -> dict:
        return PARSED_LEVELS[self._level_idx]

    def _is_pulse_on(self, tile_type: str, tick: int | None = None) -> bool:
        t = self._tick if tick is None else int(tick)
        phase = (t // 2) % 2
        if tile_type == TILE_PULSE_A:
            return phase == 0
        if tile_type == TILE_PULSE_B:
            return phase == 1
        return False

    def _tile_for_player(self, x: int, y: int) -> str:
        tile = self._current_parsed()["grid"][y][x]
        if tile in {TILE_UNCAPTURED, TILE_SAFE, TILE_OBSTACLE, TILE_GOAL}:
            return tile
        if tile in {TILE_PULSE_A, TILE_PULSE_B}:
            return tile
        return TILE_OBSTACLE

    def _tile_blocks_player(self, x: int, y: int) -> bool:
        tile = self._tile_for_player(x, y)
        if tile == TILE_OBSTACLE:
            return True
        if tile == TILE_GOAL and not self._goal_open:
            return True
        if tile in {TILE_PULSE_A, TILE_PULSE_B}:
            return self._is_pulse_on(tile)
        return False

    def _tile_blocks_interior_enemy(self, x: int, y: int) -> bool:
        tile = self._current_parsed()["grid"][y][x]
        if tile in (TILE_OBSTACLE, TILE_GOAL):
            return True
        if tile == TILE_SAFE:
            return True
        if (x, y) in self._captured:
            return True
        if tile in {TILE_PULSE_A, TILE_PULSE_B}:
            return self._is_pulse_on(tile)
        return False

    def _tile_blocks_spark(self, x: int, y: int) -> bool:
        tile = self._current_parsed()["grid"][y][x]
        if tile in {TILE_PULSE_A, TILE_PULSE_B}:
            return self._is_pulse_on(tile)
        return False

    def _footprint_fits(self, x: int, y: int, w: int, h: int, checker) -> bool:
        cfg = self._current_cfg()
        if x < 0 or y < 1 or x + w > cfg.width or y + h > cfg.height:
            return False
        return all(not checker(cx, cy) for cx, cy in _rect_cells(x, y, w, h))

    def _player_safe_mode(self) -> bool:
        parsed = self._current_parsed()
        grid = parsed["grid"]
        for cx, cy in _rect_cells(self._player_x, self._player_y, 2, 2):
            tile = grid[cy][cx]
            if tile == TILE_SAFE:
                continue
            if (cx, cy) in self._captured:
                continue
            return False
        return True

    def _apply_intended_move(self, action_id: int) -> None:
        delta = MOVE_DELTAS.get(int(action_id))
        if delta is None:
            return
        nx = self._player_x + delta[0]
        ny = self._player_y + delta[1]
        if self._footprint_fits(nx, ny, 2, 2, self._tile_blocks_player):
            self._player_x = nx
            self._player_y = ny

    def _update_trail_and_capture(self) -> None:
        parsed = self._current_parsed()
        grid = parsed["grid"]

        was_drawing = bool(self._trail)
        safe_mode = self._player_safe_mode()

        if not safe_mode:
            for cx, cy in _rect_cells(self._player_x, self._player_y, 2, 2):
                if grid[cy][cx] == TILE_UNCAPTURED and (cx, cy) not in self._captured:
                    self._trail.add((cx, cy))
            return

        if was_drawing and safe_mode:
            self._close_loop()

    def _close_loop(self) -> None:
        if not self._trail:
            return
        parsed = self._current_parsed()
        grid = parsed["grid"]

        blocked = set(self._trail)
        blocked.update(self._captured)
        blocked.update({(x, y) for (x, y) in parsed["goal_cells"]})

        enemy_reachable: set[tuple[int, int]] = set()
        q: deque[tuple[int, int]] = deque()

        for enemy in self._enemies:
            if enemy.kind == "spark":
                continue
            for cx, cy in _rect_cells(enemy.x, enemy.y, enemy.w, enemy.h):
                if grid[cy][cx] != TILE_UNCAPTURED:
                    continue
                if (cx, cy) in blocked:
                    continue
                if (cx, cy) in enemy_reachable:
                    continue
                enemy_reachable.add((cx, cy))
                q.append((cx, cy))

        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 1 or nx >= self._current_cfg().width or ny >= self._current_cfg().height:
                    continue
                if (nx, ny) in enemy_reachable or (nx, ny) in blocked:
                    continue
                if grid[ny][nx] != TILE_UNCAPTURED:
                    continue
                enemy_reachable.add((nx, ny))
                q.append((nx, ny))

        to_capture: set[tuple[int, int]] = set(self._trail)
        for x, y in parsed["uncaptured_cells"]:
            if (x, y) in self._captured or (x, y) in self._trail:
                continue
            if (x, y) not in enemy_reachable:
                to_capture.add((x, y))

        self._trail = set()
        self._fill_queue = deque(sorted(to_capture, key=lambda p: (p[1], p[0])))
        self._fill_pending = set(to_capture)

    def _overlaps_player(self, enemy: EnemyState) -> bool:
        p = (self._player_x, self._player_y, 2, 2)
        e = (enemy.x, enemy.y, enemy.w, enemy.h)
        return _rect_overlap(p, e)

    def _enemy_touches_trail(self, enemy: EnemyState) -> bool:
        for cell in _rect_cells(enemy.x, enemy.y, enemy.w, enemy.h):
            if cell in self._trail:
                return True
        return False

    def _resolve_collisions_pre_enemy_move(self) -> bool:
        drawing = bool(self._trail)
        for enemy in self._enemies:
            if enemy.kind != "spark":
                if self._enemy_touches_trail(enemy):
                    return True
                if drawing and self._overlaps_player(enemy):
                    return True
            else:
                if self._overlaps_player(enemy):
                    return True
        return False

    def _advance_fill(self) -> int:
        cfg = self._current_cfg()
        converted = 0
        for _ in range(min(cfg.fill_rate, len(self._fill_queue))):
            cell = self._fill_queue.popleft()
            self._fill_pending.discard(cell)
            if cell in self._captured:
                continue
            parsed = self._current_parsed()
            if cell not in parsed["uncaptured_cells"]:
                continue
            self._captured.add(cell)
            converted += 1
        return converted

    def _move_enemy_bouncer(self, enemy: EnemyState) -> None:
        if enemy.dx == 0 and enemy.dy == 0:
            enemy.dx, enemy.dy = 1, 0

        nx = enemy.x + enemy.dx
        ny = enemy.y + enemy.dy
        if self._footprint_fits(nx, ny, enemy.w, enemy.h, self._tile_blocks_interior_enemy):
            enemy.x, enemy.y = nx, ny
            return

        enemy.dx *= -1
        enemy.dy *= -1
        nx = enemy.x + enemy.dx
        ny = enemy.y + enemy.dy
        if self._footprint_fits(nx, ny, enemy.w, enemy.h, self._tile_blocks_interior_enemy):
            enemy.x, enemy.y = nx, ny

    def _nearest_trail_cell(self, enemy: EnemyState) -> tuple[int, int] | None:
        if not self._trail:
            return None
        ex = enemy.x + enemy.w // 2
        ey = enemy.y + enemy.h // 2
        best = None
        best_d = 10**9
        for tx, ty in self._trail:
            d = abs(tx - ex) + abs(ty - ey)
            if d < best_d:
                best_d = d
                best = (tx, ty)
        return best

    def _move_hunter(self, enemy: EnemyState) -> None:
        target = self._nearest_trail_cell(enemy)
        if target is None:
            self._move_enemy_bouncer(enemy)
            return

        ex, ey = enemy.x, enemy.y
        tx, ty = target
        candidates: list[tuple[int, int]] = []
        if tx > ex:
            candidates.append((1, 0))
        elif tx < ex:
            candidates.append((-1, 0))
        if ty > ey:
            candidates.append((0, 1))
        elif ty < ey:
            candidates.append((0, -1))

        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if delta not in candidates:
                candidates.append(delta)

        for dx, dy in candidates:
            nx = enemy.x + dx
            ny = enemy.y + dy
            if self._footprint_fits(nx, ny, enemy.w, enemy.h, self._tile_blocks_interior_enemy):
                enemy.x, enemy.y = nx, ny
                enemy.dx, enemy.dy = dx, dy
                return

        self._move_enemy_bouncer(enemy)

    def _move_spark(self, enemy: EnemyState) -> None:
        ring = self._current_parsed()["ring"]
        n = len(ring)
        if n == 0:
            return
        next_idx = (enemy.spark_path_idx + enemy.spark_dir) % n
        nx, ny = ring[next_idx]
        if self._tile_blocks_spark(nx, ny):
            enemy.spark_dir *= -1
            next_idx = (enemy.spark_path_idx + enemy.spark_dir) % n
            nx, ny = ring[next_idx]
            if self._tile_blocks_spark(nx, ny):
                return
        enemy.spark_path_idx = next_idx
        enemy.x, enemy.y = nx, ny

    def _move_enemies(self) -> None:
        for enemy in self._enemies:
            if (self._tick % max(1, enemy.speed)) != 0:
                continue
            if enemy.kind == "wanderer":
                self._move_enemy_bouncer(enemy)
            elif enemy.kind == "hunter":
                self._move_hunter(enemy)
            else:
                self._move_spark(enemy)

    def _check_goal_completion(self) -> bool:
        if not self._goal_open:
            return False
        goal_cells = self._current_parsed()["goal_cells"]
        for cell in _rect_cells(self._player_x, self._player_y, 2, 2):
            if cell in goal_cells:
                return True
        return False

    def _capture_count(self) -> int:
        parsed = self._current_parsed()
        return sum(1 for cell in self._captured if cell in parsed["uncaptured_cells"])

    def _capture_ratio(self) -> float:
        parsed = self._current_parsed()
        if parsed["fillable_total"] <= 0:
            return 0.0
        return float(self._capture_count()) / float(parsed["fillable_total"])

    def _fail_penalty(self) -> int:
        return min(self._current_cfg().fail_penalty, 45)

    def _fail_reset(self) -> None:
        parsed = self._current_parsed()
        self._time = max(0, self._time - self._fail_penalty())
        self._player_x, self._player_y = parsed["player_start"]
        self._trail = set()
        self._fill_queue.clear()
        self._fill_pending.clear()

        self._enemies = [
            EnemyState(
                kind=e.kind,
                x=e.start_x,
                y=e.start_y,
                start_x=e.start_x,
                start_y=e.start_y,
                w=e.w,
                h=e.h,
                speed=e.speed,
                dx=e.dx,
                dy=e.dy,
                spark_path_idx=e.spark_path_idx,
                spark_dir=e.spark_dir,
            )
            for e in _initial_enemy_states(self._level_idx)
        ]

    def _render(self) -> None:
        cfg = self._current_cfg()
        parsed = self._current_parsed()
        blink = (self._tick % 2) == 0

        frame = np.zeros((cfg.height, cfg.width), dtype=np.int8)

        # Base playfield (rows 1..H-1).
        for y in range(1, cfg.height):
            for x in range(cfg.width):
                tile = parsed["grid"][y][x]
                if tile == TILE_UNCAPTURED:
                    color = COLOR_CAPTURED if (x, y) in self._captured else COLOR_UNCAPTURED
                elif tile == TILE_SAFE:
                    color = COLOR_SAFE
                elif tile == TILE_OBSTACLE:
                    color = COLOR_SOLID
                elif tile == TILE_GOAL:
                    if self._goal_open:
                        color = COLOR_GOAL if blink else COLOR_TRAIL_B
                    else:
                        color = COLOR_SOLID
                elif tile in {TILE_PULSE_A, TILE_PULSE_B}:
                    color = COLOR_TRAIL_B if self._is_pulse_on(tile) else COLOR_UNCAPTURED
                else:
                    color = COLOR_SOLID

                if (x, y) in self._fill_pending:
                    color = COLOR_TRAIL_B
                if (x, y) in self._trail:
                    color = COLOR_TRAIL_A if blink else COLOR_TRAIL_B
                frame[y, x] = int(color)

        # Timebar row.
        ratio = float(self._time) / float(max(1, cfg.max_time))
        fill = int(np.floor(ratio * cfg.width))
        fill = max(0, min(cfg.width, fill))
        low_flash = ratio < 0.15 and not blink
        for x in range(cfg.width):
            if x < fill:
                frame[0, x] = COLOR_TRAIL_B if low_flash else COLOR_CAPTURED
            else:
                frame[0, x] = COLOR_UI_EMPTY
        pointer_x = max(0, min(cfg.width - 1, fill - 1 if fill > 0 else 0))
        frame[0, pointer_x] = COLOR_GOAL

        # Enemies.
        for enemy in self._enemies:
            if enemy.kind == "wanderer":
                color = COLOR_WANDERER_A if blink else COLOR_WANDERER_B
            elif enemy.kind == "hunter":
                color = COLOR_HUNTER_A if blink else COLOR_HUNTER_B
            else:
                color = COLOR_SPARK_A if blink else COLOR_SPARK_B
            for cx, cy in _rect_cells(enemy.x, enemy.y, enemy.w, enemy.h):
                frame[cy, cx] = int(color)

        # Player on top.
        safe_mode = self._player_safe_mode()
        for cx, cy in _rect_cells(self._player_x, self._player_y, 2, 2):
            if safe_mode:
                frame[cy, cx] = COLOR_PLAYER_A if blink else COLOR_PLAYER_B
            else:
                frame[cy, cx] = COLOR_PLAYER_A if ((cx + cy + self._tick) % 2 == 0) else COLOR_GOAL

        if self._board_sprite is not None:
            self._board_sprite.pixels = frame

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id not in {1, 2, 3, 4, 5}:
            action_id = WAIT_ACTION_ID

        # 1) apply intended move
        self._apply_intended_move(action_id)

        # 2) update trail/capture state
        self._update_trail_and_capture()

        # 3) resolve collisions before enemy movement
        failed = self._resolve_collisions_pre_enemy_move()

        # 4) advance fill animation
        newly_captured = self._advance_fill()

        # 5) move enemies
        self._move_enemies()
        failed = failed or self._resolve_collisions_pre_enemy_move()

        # Capture refund and objective checks.
        if newly_captured > 0:
            self._time = min(self._current_cfg().max_time, self._time + (newly_captured // 6))
        if not self._goal_open and self._capture_count() >= self._current_parsed()["capture_required"]:
            self._goal_open = True
        if self._goal_open:
            self.next_level()
            self.complete_action()
            return

        if self._check_goal_completion():
            self.next_level()
            self.complete_action()
            return

        # 6) update time and animation frames.
        self._time -= 1
        if self._time <= 0:
            self.lose()
            self.complete_action()
            return

        if failed:
            if self._time - self._fail_penalty() <= 0:
                self.lose()
                self.complete_action()
                return
            self._fail_reset()

        self._tick += 1
        self._render()
        self.complete_action()


def _initial_enemy_states(level_idx: int) -> list[EnemyState]:
    cfg = LEVEL_CONFIGS[level_idx]
    parsed = PARSED_LEVELS[level_idx]

    row_chars = cfg.layout
    wanderer_cells: list[tuple[int, int]] = []
    hunter_cells: list[tuple[int, int]] = []
    spark_cells: list[tuple[int, int]] = []
    for y, row in enumerate(row_chars):
        for x, ch in enumerate(row):
            if ch == "!":
                wanderer_cells.append((x, y))
            elif ch == "$":
                hunter_cells.append((x, y))
            elif ch == "^":
                spark_cells.append((x, y))

    def _find_blocks(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        tops: list[tuple[int, int]] = []
        cell_set = set(cells)
        for x, y in sorted(cell_set):
            if (x - 1, y) in cell_set or (x, y - 1) in cell_set:
                continue
            block = {(x + dx, y + dy) for dy in range(2) for dx in range(2)}
            if block.issubset(cell_set):
                tops.append((x, y))
        return sorted(tops)

    enemies: list[EnemyState] = []
    wanderers = _find_blocks(wanderer_cells)
    for idx, ((x, y), speed) in enumerate(zip(wanderers, cfg.wanderer_speeds, strict=False)):
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = dirs[idx % len(dirs)]
        enemies.append(
            EnemyState(kind="wanderer", x=x, y=y, start_x=x, start_y=y, w=2, h=2, speed=int(speed), dx=dx, dy=dy)
        )

    hunters = _find_blocks(hunter_cells)
    for idx, ((x, y), speed) in enumerate(zip(hunters, cfg.hunter_speeds, strict=False)):
        dirs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        dx, dy = dirs[idx % len(dirs)]
        enemies.append(
            EnemyState(kind="hunter", x=x, y=y, start_x=x, start_y=y, w=2, h=2, speed=int(speed), dx=dx, dy=dy)
        )

    ring = parsed["ring"]
    ring_index = {pos: idx for idx, pos in enumerate(ring)}
    for idx, (x, y) in enumerate(sorted(spark_cells)):
        if (x, y) in ring_index:
            start_idx = ring_index[(x, y)]
            sx, sy = x, y
        else:
            start_idx = min(range(len(ring)), key=lambda ridx: abs(ring[ridx][0] - x) + abs(ring[ridx][1] - y))
            sx, sy = ring[start_idx]
        enemies.append(
            EnemyState(
                kind="spark",
                x=sx,
                y=sy,
                start_x=sx,
                start_y=sy,
                w=1,
                h=1,
                speed=1,
                spark_path_idx=start_idx,
                spark_dir=1 if idx % 2 == 0 else -1,
            )
        )
    return enemies


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (ax < bx + bw) and (bx < ax + aw) and (ay < by + bh) and (by < ay + ah)


# Solver helpers ---------------------------------------------------------------


def make_level_macro_plan(level_idx: int) -> list[int]:
    """Generate a deterministic macro-plan based on repeated vertical cuts."""
    cfg = LEVEL_CONFIGS[level_idx]
    parsed = PARSED_LEVELS[level_idx]

    min_dot_x = min(x for x, _ in parsed["uncaptured_cells"])
    max_dot_x = max(x for x, _ in parsed["uncaptured_cells"])
    min_dot_y = min(y for _, y in parsed["uncaptured_cells"])
    max_dot_y = max(y for _, y in parsed["uncaptured_cells"])

    start_x, start_y = parsed["player_start"]
    top_safe_y = max(1, min_dot_y - 2)
    bottom_safe_y = min(cfg.height - 2, max_dot_y + 1)

    action_ids: list[int] = []
    px, py = start_x, start_y

    def _move_to(tx: int, ty: int) -> None:
        nonlocal px, py
        while px < tx:
            action_ids.append(4)
            px += 1
        while px > tx:
            action_ids.append(3)
            px -= 1
        while py < ty:
            action_ids.append(2)
            py += 1
        while py > ty:
            action_ids.append(1)
            py -= 1

    # Use alternating columns so loops close against safe/captured boundaries.
    cut_columns = list(range(min_dot_x - 1, max_dot_x - 1, 2))
    for idx, col in enumerate(cut_columns):
        col = max(0, min(cfg.width - 2, col))
        _move_to(col, bottom_safe_y)
        action_ids.extend([5, 5])
        _move_to(col, top_safe_y)
        action_ids.extend([5, 5])
        _move_to(col + 1, top_safe_y)
        _move_to(col + 1, bottom_safe_y)
        action_ids.extend([5, 5])

        if idx % 3 == 2:
            action_ids.extend([5, 5, 5])

    goal_x, goal_y = parsed["goal_top_left"]
    _move_to(max(0, goal_x - 1), max(1, goal_y - 1))
    action_ids.extend([5] * 8)
    return action_ids


def simulate_plan_until_win(level_idx: int, actions: list[int], max_steps: int = 5000) -> bool:
    game = TerritoryCapture(seed=0)
    # Jump to desired level by loading directly.
    game._load_level(level_idx)
    steps = 0
    cursor = 0

    class _Action:
        def __init__(self, aid: int):
            self.id = aid
            self.data = {}

    while steps < max_steps:
        if game._goal_open:
            return True
        if cursor < len(actions):
            aid = int(actions[cursor])
            cursor += 1
        else:
            aid = WAIT_ACTION_ID
        game._set_action(_Action(aid))
        game.step()
        if game._time <= 0:
            return False
        if game._check_goal_completion():
            return True
        if game._capture_count() >= game._current_parsed()["capture_required"]:
            return True
        steps += 1
    return False


__all__ = ["LEVEL_CONFIGS", "PARSED_LEVELS", "TerritoryCapture", "make_level_macro_plan", "simulate_plan_until_win"]
