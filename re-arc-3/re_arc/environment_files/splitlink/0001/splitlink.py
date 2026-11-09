from __future__ import annotations

import itertools

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_TILES = 13
TILE_SIZE = 3
PLAYFIELD_SIZE = GRID_TILES * TILE_SIZE
GRID_WIDTH = PLAYFIELD_SIZE
GRID_HEIGHT = PLAYFIELD_SIZE + TILE_SIZE
PLAYFIELD_Y_OFFSET = TILE_SIZE

COLORS = {
    "floor": 1,
    "wall": 5,
    "player": 9,
    "goal": 11,
    "terminal_1": 12,
    "terminal_2": 13,
    "terminal_3": 14,
    "terminal_on": 10,
    "time_bg": 3,
    "time_fill": 8,
}

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -TILE_SIZE),
    GameAction.ACTION2: (0, TILE_SIZE),
    GameAction.ACTION3: (-TILE_SIZE, 0),
    GameAction.ACTION4: (TILE_SIZE, 0),
}

START_TILE = (1, 1)
GOAL_TILE = (11, 11)
TERMINAL_TILES = [(3, 3), (9, 2), (6, 10)]
DOOR_TILES = [(5, 3), (8, 6), (10, 11)]
DOOR_TAGS = ["wall_tag_a", "wall_tag_b", "wall_tag_c"]

ROOM1_TILES = [(x, y) for x in range(1, 5) for y in range(1, 5)]
ROOM2_TILES = [(x, y) for x in range(8, 12) for y in range(1, 5)]
ROOM3_TILES = [(x, y) for x in range(5, 9) for y in range(8, 12)]
CORRIDOR_TILES = (
    [(x, 3) for x in range(4, 9)]
    + [(8, y) for y in range(4, 9)]
    + [(x, 8) for x in range(6, 8)]
    + [(6, y) for y in range(9, 12)]
    + [(x, 11) for x in range(9, 12)]
)

WALKABLE_TILES = set(
    ROOM1_TILES + ROOM2_TILES + ROOM3_TILES + CORRIDOR_TILES + [START_TILE, GOAL_TILE, *TERMINAL_TILES, *DOOR_TILES]
)

WAYPOINTS = [(1, 1), (3, 1), (3, 3), (8, 3), (8, 2), (9, 2), (8, 2), (8, 8), (6, 8), (6, 10), (6, 11), (11, 11)]
SPACE_STEPS = [1, 4, 7]


def _tile_to_xy(tile: tuple[int, int]) -> tuple[int, int]:
    return tile[0] * TILE_SIZE, tile[1] * TILE_SIZE + PLAYFIELD_Y_OFFSET


def _square(color: int, *, name: str, tile: tuple[int, int], layer: int, tags: list[str], collidable: bool) -> Sprite:
    x, y = _tile_to_xy(tile)
    return Sprite(
        pixels=[[color for _ in range(TILE_SIZE)] for _ in range(TILE_SIZE)],
        name=name,
        x=x,
        y=y,
        layer=layer,
        tags=tags,
        collidable=collidable,
    )


def _expand_path(waypoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for (x0, y0), (x1, y1) in itertools.pairwise(waypoints):
        dx = 0 if x1 == x0 else (1 if x1 > x0 else -1)
        dy = 0 if y1 == y0 else (1 if y1 > y0 else -1)
        if dx != 0 and dy != 0:
            raise ValueError("waypoints must be orthogonally connected")
        if not path:
            path.append((x0, y0))
        x, y = x0, y0
        while (x, y) != (x1, y1):
            x += dx
            y += dy
            path.append((x, y))
    return path


def _build_action_plan(path: list[tuple[int, int]], space_steps: list[int]) -> list[int]:
    out: list[int] = []
    for segment_idx, ((x0, y0), (x1, y1)) in enumerate(itertools.pairwise(path)):
        dx = x1 - x0
        dy = y1 - y0
        if dx == 1 and dy == 0:
            out.append(int(GameAction.ACTION4.value))
        elif dx == -1 and dy == 0:
            out.append(int(GameAction.ACTION3.value))
        elif dx == 0 and dy == 1:
            out.append(int(GameAction.ACTION2.value))
        elif dx == 0 and dy == -1:
            out.append(int(GameAction.ACTION1.value))
        else:
            raise ValueError("invalid step between path tiles")
        if segment_idx in space_steps:
            out.append(int(GameAction.ACTION5.value))
    return out


def _build_level() -> Level:
    sprites: list[Sprite] = [
        Sprite(
            pixels=[[COLORS["floor"] for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)],
            name="floor",
            x=0,
            y=0,
            layer=0,
            tags=["floor", "sys_static"],
            collidable=False,
        ),
        Sprite(
            pixels=[[COLORS["time_bg"] for _ in range(GRID_WIDTH)] for _ in range(TILE_SIZE)],
            name="time-bg",
            x=0,
            y=0,
            layer=1,
            tags=["time_bar", "sys_static"],
            collidable=False,
        ),
        Sprite(
            pixels=[[COLORS["time_fill"] for _ in range(GRID_WIDTH)] for _ in range(TILE_SIZE)],
            name="time-fill",
            x=0,
            y=0,
            layer=2,
            tags=["time_bar_fill"],
            collidable=False,
        ),
    ]

    for gy in range(GRID_TILES):
        for gx in range(GRID_TILES):
            tile = (gx, gy)
            if tile in WALKABLE_TILES:
                continue
            sprites.append(
                _square(
                    COLORS["wall"],
                    name=f"wall-{gx}-{gy}",
                    tile=tile,
                    layer=3,
                    tags=["wall", "blocker", "sys_static"],
                    collidable=True,
                )
            )

    for idx, tile in enumerate(DOOR_TILES):
        sprites.append(
            _square(
                COLORS["wall"],
                name=f"door-{idx + 1}",
                tile=tile,
                layer=4,
                tags=["wall", "blocker", DOOR_TAGS[idx]],
                collidable=True,
            )
        )

    for idx, tile in enumerate(TERMINAL_TILES):
        sprites.append(
            _square(
                COLORS[f"terminal_{idx + 1}"],
                name=f"terminal-{idx + 1}",
                tile=tile,
                layer=5,
                tags=["terminal", f"seq_{idx + 1}"],
                collidable=False,
            )
        )

    sprites.append(_square(COLORS["goal"], name="goal", tile=GOAL_TILE, layer=5, tags=["goal"], collidable=False))
    sprites.append(_square(COLORS["player"], name="player", tile=START_TILE, layer=6, tags=["player"], collidable=True))

    path = _expand_path(WAYPOINTS)
    action_plan = _build_action_plan(path, SPACE_STEPS)
    time_limit = max(120, len(action_plan) * 4)
    return Level(
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=sprites,
        data={"time_limit": time_limit, "action_plan": action_plan, "sequence_length": len(TERMINAL_TILES)},
    )


class Splitlink(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level()]
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLORS["floor"])
        super().__init__(
            game_id="splitlink", levels=levels, camera=camera, win_score=1, available_actions=[1, 2, 3, 4, 5], seed=seed
        )
        self._player = None
        self._goal = None
        self._terminals: list[Sprite] = []
        self._doors: list[Sprite] = []
        self._time_fill = None
        self._time_limit = 0
        self._time_left = 0
        self._sequence_progress = 0

    def on_set_level(self, level: Level) -> None:
        players = level.get_sprites_by_name("player")
        goals = level.get_sprites_by_name("goal")
        terminals = sorted(level.get_sprites_by_tag("terminal"), key=lambda s: s.name)
        doors = sorted(
            [s for s in level.get_sprites_by_tag("blocker") if any(tag in s.tags for tag in DOOR_TAGS)],
            key=lambda s: s.name,
        )
        time_fill = level.get_sprites_by_name("time-fill")
        if (
            not players
            or not goals
            or len(terminals) != len(TERMINAL_TILES)
            or len(doors) != len(DOOR_TILES)
            or not time_fill
        ):
            raise RuntimeError("splitlink level is missing required sprites")

        self._player = players[0]
        self._goal = goals[0]
        self._terminals = terminals
        self._doors = doors
        self._time_fill = time_fill[0]
        self._time_limit = int(level.get_data("time_limit") or 120)
        self._time_left = self._time_limit
        self._sequence_progress = 0
        self._refresh_timer_bar()

    def _is_blocked(self, x: int, y: int) -> bool:
        for sprite in self.current_level.get_sprites():
            if "blocker" not in sprite.tags:
                continue
            if not sprite.is_visible:
                continue
            if x < sprite.x or y < sprite.y or x >= sprite.x + sprite.width or y >= sprite.y + sprite.height:
                continue
            rendered = sprite.render()
            local_x = x - sprite.x
            local_y = y - sprite.y
            if rendered[local_y][local_x] < 0:
                continue
            return True
        return False

    def _try_move_player(self, dx: int, dy: int) -> None:
        if self._player is None:
            return
        new_x = self._player.x + dx
        new_y = self._player.y + dy
        min_y = PLAYFIELD_Y_OFFSET
        max_x = GRID_WIDTH - TILE_SIZE
        max_y = GRID_HEIGHT - TILE_SIZE
        if new_x < 0 or new_x > max_x or new_y < min_y or new_y > max_y:
            return
        if self._is_blocked(new_x, new_y):
            return
        self._player.set_position(new_x, new_y)

    def _refresh_timer_bar(self) -> None:
        if self._time_fill is None:
            return
        ratio = max(0.0, min(1.0, self._time_left / max(1, self._time_limit)))
        fill_width = round(ratio * GRID_WIDTH)
        pixels = []
        for _ in range(TILE_SIZE):
            row = [COLORS["time_fill"] for _ in range(fill_width)] + [-1 for _ in range(GRID_WIDTH - fill_width)]
            pixels.append(row)
        self._time_fill.pixels = np.array(pixels, dtype=np.int8)

    def _on_space(self) -> None:
        if self._player is None or self._sequence_progress >= len(self._terminals):
            return
        expected_idx = self._sequence_progress
        expected_terminal = self._terminals[expected_idx]
        if self._player.x != expected_terminal.x or self._player.y != expected_terminal.y:
            return

        expected_terminal.pixels = np.array(
            [[COLORS["terminal_on"] for _ in range(TILE_SIZE)] for _ in range(TILE_SIZE)], dtype=np.int8
        )
        door = self._doors[expected_idx]
        door.set_collidable(False)
        door.set_visible(False)
        self._sequence_progress += 1

    def step(self) -> None:
        if self._player is None or self._goal is None:
            self.complete_action()
            return

        action = self.action.id
        delta = MOVE_DELTAS.get(action)
        if delta is not None:
            self._try_move_player(delta[0], delta[1])
        elif action == GameAction.ACTION5:
            self._on_space()

        if (
            self._sequence_progress >= len(self._terminals)
            and self._player.x == self._goal.x
            and self._player.y == self._goal.y
        ):
            self.next_level()
            self.complete_action()
            return

        self._time_left -= 1
        self._refresh_timer_bar()
        if self._time_left <= 0:
            self.lose()

        self.complete_action()
