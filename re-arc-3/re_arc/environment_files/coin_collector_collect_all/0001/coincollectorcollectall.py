from __future__ import annotations

import math

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "coin_collector_collect_all-0001"
VARIANT = "0001"

SPEC = {
    "spec_id": "spec-custom-coin-collector-collect-all-0001",
    "objective": "obj.collect_all",
    "mechanics": ["mech.avatar_navigation", "mech.collect_all", "mech.timer_bar"],
    "layout_template": "fixed_six_level_series",
    "action_set": ["up", "down", "left", "right", "space"],
    "target_optimal_steps": 150,
    "grid_size": [32, 18],
    "level_count": 6,
}

COLOR_FLOOR = 0
COLOR_WALL = 1
COLOR_PLAYER_BODY = 2
COLOR_PLAYER_DIR = 3
COLOR_COIN_A = 4
COLOR_COIN_B = 5
COLOR_TIMEBAR_FILL = 6
COLOR_TIMEBAR_EMPTY = 7
COLOR_FAIL_FLASH = 8
COLOR_WIN_FLASH = 9

FLASH_STEPS = 8

MOVE = {
    int(GameAction.ACTION1.value): (0, -1, "up"),
    int(GameAction.ACTION2.value): (0, 1, "down"),
    int(GameAction.ACTION3.value): (-1, 0, "left"),
    int(GameAction.ACTION4.value): (1, 0, "right"),
}

RAW_LEVELS: list[dict[str, object]] = [
    {
        "steps_per_cell": 8,
        "layout": [
            "========================",
            "########################",
            "#......................#",
            "#..........o...........#",
            "#......................#",
            "#.....o................#",
            "#......................#",
            "#..................o...#",
            "#......................#",
            "#......................#",
            "#..>@..................#",
            "#..@@..................#",
            "#......................#",
            "########################",
        ],
    },
    {
        "steps_per_cell": 7,
        "layout": [
            "========================",
            "########################",
            "#......................#",
            "#..######..............#",
            "#..######....o.........#",
            "#..######..............#",
            "#..######........#######",
            "#..........o.....#######",
            "#................#######",
            "#..o.............#######",
            "#..######....o...#######",
            "#>@######..............#",
            "#@@.................o..#",
            "########################",
        ],
    },
    {
        "steps_per_cell": 7,
        "layout": [
            "================================",
            "################################",
            "#..............##..............#",
            "#..o.......#...##...#.......o..#",
            "#..............##......####....#",
            "#.#..o.........##....#....o..#.#",
            "#....#.........................#",
            "#....####...o......o...........#",
            "#..............................#",
            "#..o......................o....#",
            "#......####....##....####......#",
            "#..o...........##...........o..#",
            "#........#.....##.....#........#",
            "#....o..###....##....###..o....#",
            "#..............##..............#",
            "#..>@..........##.........o....#",
            "#..@@..........##..............#",
            "################################",
        ],
    },
    {
        "steps_per_cell": 6,
        "layout": [
            "================================",
            "################################",
            "#..............................#",
            "#..o..............####....o....#",
            "#.................####.........#",
            "#....######....................#",
            "#....######....o...............#",
            "#......................###.....#",
            "#....o.................###.....#",
            "###.....#######..####.......####",
            "#.....................o........#",
            "#.....#####....................#",
            "#.....#####........######......#",
            "#...............o..######..o...#",
            "#..>@..........................#",
            "#..@@..........................#",
            "#..............................#",
            "################################",
        ],
    },
    {
        "steps_per_cell": 6,
        "layout": [
            "================================",
            "################################",
            "#..o.................#....o....#",
            "#.............o.........#..#...#",
            "#....##....#...............###.#",
            "#..........#....##.............#",
            "###..########...#####..##..#####",
            "#..........###..#..............#",
            "#..o........#....#......o......#",
            "#..........#....#..............#",
            "#......o....##..##....o........#",
            "#..............o...............#",
            "###....#######............######",
            "#..............................#",
            "#..>@......###.........o.......#",
            "#..@@......###..#..............#",
            "#..............o.........o.....#",
            "################################",
        ],
    },
    {
        "steps_per_cell": 5,
        "layout": [
            "================================",
            "################################",
            "#..o..............##....o......#",
            "#................##............#",
            "#....#########..########..#..###",
            "#...........##........##....o..#",
            "#..o..............##...........#",
            "###.......####..########..###..#",
            "#......#.....o.............o...#",
            "#..o...#...............o.......#",
            "#.........###...#####.......####",
            "#........o.....................#",
            "#......##........##...o........#",
            "###..########...#######.....#..#",
            "#..>@....##....o...##..........#",
            "#..@@....##........##....o.....#",
            "#............o.............o...#",
            "################################",
        ],
    },
]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _parse_level(index: int, raw: dict[str, object]) -> Level:
    rows = [str(row) for row in list(raw["layout"])]
    height = len(rows)
    width = len(rows[0])
    if not rows:
        raise ValueError(f"level {index} has empty layout")
    if any(len(row) != width for row in rows):
        raise ValueError(f"level {index} has non-rectangular layout")

    steps_per_cell = int(raw["steps_per_cell"])
    walls: set[tuple[int, int]] = set()
    coins: list[tuple[int, int]] = []
    player_top_left: tuple[int, int] | None = None
    facing = "right"

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if y == 0:
                continue
            if ch == "#":
                walls.add((x, y))
            elif ch in {"o", "O"}:
                coins.append((x, y))
            elif ch in {"^", "v", "<", ">"}:
                player_top_left = (x, y)
                facing = {"^": "up", "v": "down", "<": "left", ">": "right"}[ch]

    if player_top_left is None:
        raise ValueError(f"level {index} missing player marker")

    sprites: list[Sprite] = [
        Sprite(_solid(width, height, COLOR_FLOOR), name="floor", x=0, y=0, layer=0, tags=["floor"], collidable=False),
        Sprite(
            _solid(width, 1, COLOR_TIMEBAR_FILL),
            name="timebar",
            x=0,
            y=0,
            layer=1,
            tags=["hud", "timer"],
            collidable=False,
        ),
    ]

    for idx_wall, (wx, wy) in enumerate(sorted(walls)):
        sprites.append(
            Sprite(
                [[COLOR_WALL]], name=f"wall_{idx_wall}", x=wx, y=wy, layer=2, tags=["wall", "blocker"], collidable=True
            )
        )

    for idx_coin, (cx, cy) in enumerate(coins):
        sprites.append(
            Sprite(
                [[COLOR_COIN_A]],
                name=f"coin_{idx_coin}",
                x=cx,
                y=cy,
                layer=3,
                tags=["coin", f"coin_{idx_coin}"],
                collidable=False,
            )
        )

    sprites.append(
        Sprite(
            np.array([[COLOR_PLAYER_DIR, COLOR_PLAYER_BODY], [COLOR_PLAYER_BODY, COLOR_PLAYER_BODY]], dtype=np.int8),
            name="player",
            x=player_top_left[0],
            y=player_top_left[1],
            layer=4,
            tags=["player"],
            collidable=True,
        )
    )

    return Level(
        name=f"CoinCollectorCollectAll L{index + 1}",
        grid_size=(width, height),
        sprites=sprites,
        data={
            "steps_per_cell": steps_per_cell,
            "time_total": width * steps_per_cell,
            "player_start": [player_top_left[0], player_top_left[1]],
            "player_facing": facing,
            "walls": [[x, y] for x, y in sorted(walls)],
            "coins": [[x, y] for x, y in coins],
            "spec": SPEC,
        },
    )


class CoinCollectorCollectAll(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_parse_level(i, raw) for i, raw in enumerate(RAW_LEVELS)]
        camera = Camera(width=32, height=18, background=COLOR_FLOOR)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

        self._width = 0
        self._height = 0
        self._steps_per_cell = 1
        self._time_total = 1
        self._time_remaining_steps = 1

        self._walls: set[tuple[int, int]] = set()
        self._coins: dict[tuple[int, int], Sprite] = {}

        self._player: Sprite | None = None
        self._timebar: Sprite | None = None
        self._facing = "right"
        self._coin_frame_is_a = True

        self._phase = "PLAY"
        self._flash_steps_left = 0

    def on_set_level(self, level: Level) -> None:
        grid_size = tuple(level.grid_size)
        self._width = int(grid_size[0])
        self._height = int(grid_size[1])
        self._steps_per_cell = int(level.get_data("steps_per_cell") or 1)
        self._time_total = int(level.get_data("time_total") or (self._width * self._steps_per_cell))
        self._time_remaining_steps = self._time_total

        self._walls = {
            (int(cell[0]), int(cell[1]))
            for cell in (level.get_data("walls") or [])
            if isinstance(cell, (list, tuple)) and len(cell) == 2
        }

        self._coins = {}
        for coin in level.get_sprites_by_tag("coin"):
            self._coins[(int(coin.x), int(coin.y))] = coin

        players = level.get_sprites_by_name("player")
        timers = level.get_sprites_by_name("timebar")
        self._player = players[0] if players else None
        self._timebar = timers[0] if timers else None

        self._facing = str(level.get_data("player_facing") or "right")
        self._coin_frame_is_a = True
        self._phase = "PLAY"
        self._flash_steps_left = 0

        self._sync_player_pixels()
        self._sync_coins()
        self._sync_timebar()

    def _sync_player_pixels(self) -> None:
        if self._player is None:
            return
        self._player.pixels = np.array(
            [[COLOR_PLAYER_DIR, COLOR_PLAYER_BODY], [COLOR_PLAYER_BODY, COLOR_PLAYER_BODY]], dtype=np.int8
        )

    def _sync_coins(self) -> None:
        color = COLOR_COIN_A if self._coin_frame_is_a else COLOR_COIN_B
        for sprite in self._coins.values():
            sprite.pixels = np.array([[color]], dtype=np.int8)

    def _filled_cells(self) -> int:
        if self._time_remaining_steps <= 0:
            return 0
        return math.ceil(self._time_remaining_steps / max(1, self._steps_per_cell))

    def _sync_timebar(self) -> None:
        if self._timebar is None:
            return

        if self._phase == "FAIL_FLASH":
            row = np.full((1, self._width), COLOR_FAIL_FLASH, dtype=np.int8)
            self._timebar.pixels = row
            return

        if self._phase == "WIN_FLASH":
            flash_color = COLOR_WIN_FLASH if (self._flash_steps_left % 2 == 0) else COLOR_TIMEBAR_FILL
            row = np.full((1, self._width), flash_color, dtype=np.int8)
            self._timebar.pixels = row
            return

        filled = self._filled_cells()
        row = np.full((1, self._width), COLOR_TIMEBAR_EMPTY, dtype=np.int8)
        if filled > 0:
            row[:, :filled] = COLOR_TIMEBAR_FILL
        self._timebar.pixels = row

    def _player_pos(self) -> tuple[int, int]:
        if self._player is None:
            return (0, 1)
        return int(self._player.x), int(self._player.y)

    def _can_place_player(self, top_left_x: int, top_left_y: int) -> bool:
        if top_left_x < 0 or top_left_x + 1 >= self._width:
            return False
        if top_left_y < 1 or top_left_y + 1 >= self._height:
            return False
        for dy in (0, 1):
            for dx in (0, 1):
                if (top_left_x + dx, top_left_y + dy) in self._walls:
                    return False
        return True

    def _try_move(self, dx: int, dy: int) -> None:
        if self._player is None:
            return
        px, py = self._player_pos()
        nx = px + dx
        ny = py + dy
        if self._can_place_player(nx, ny):
            self._player.set_position(nx, ny)

    def _collect_overlapped_coins(self) -> None:
        px, py = self._player_pos()
        touched: list[tuple[int, int]] = []
        for dy in (0, 1):
            for dx in (0, 1):
                pos = (px + dx, py + dy)
                if pos in self._coins:
                    touched.append(pos)
        for pos in touched:
            sprite = self._coins.pop(pos)
            sprite.set_position(-10, -10)

    def _begin_flash(self, phase: str) -> None:
        self._phase = phase
        self._flash_steps_left = FLASH_STEPS

    def _tick_flash(self) -> None:
        if self._phase == "PLAY":
            return

        self._flash_steps_left -= 1
        if self._flash_steps_left > 0:
            self._sync_timebar()
            return

        if self._phase == "WIN_FLASH":
            self._phase = "PLAY"
            self.next_level()
            return

        self._phase = "PLAY"
        self.lose()

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

        if self._phase != "PLAY":
            self._tick_flash()
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id in MOVE:
            dx, dy, facing = MOVE[action_id]
            self._facing = facing
            self._try_move(dx, dy)
        elif action_id == int(GameAction.ACTION5.value):
            pass

        self._collect_overlapped_coins()

        self._time_remaining_steps = max(0, self._time_remaining_steps - 1)

        self._coin_frame_is_a = not self._coin_frame_is_a
        self._sync_coins()

        if self._time_remaining_steps == 0:
            self._begin_flash("FAIL_FLASH")
        elif not self._coins:
            self._begin_flash("WIN_FLASH")

        self._sync_player_pixels()
        self._sync_timebar()
        self.complete_action()


def _novel_signature_coin_collector_collect_all(seed: int) -> int:
    acc = int(seed)
    for idx in range(1, 25):
        acc = (acc * 37 + 19 * idx + 7) % 104729
        if idx % 3 == 0:
            acc = (acc ^ (idx * 97)) % 104729
    return acc


__all__ = ["GAME_ID", "SPEC", "CoinCollectorCollectAll"]
