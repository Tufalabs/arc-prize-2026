from __future__ import annotations

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_TILES = 15
TILE_SIZE = 3
GRID_SIZE = GRID_TILES * TILE_SIZE

COLORS = {
    "floor": 1,
    "wall": 5,
    "player": 9,
    "collectible": 11,
    "key": 12,
    "door_locked": 8,
    "door_open": 14,
    "timer": 10,
}

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

ROOM_A = {(gx, gy) for gx in range(1, 6) for gy in range(1, 6)}
ROOM_B = {(gx, gy) for gx in range(9, 14) for gy in range(1, 6)}
ROOM_C = {(gx, gy) for gx in range(1, 14) for gy in range(9, 14)}
CORRIDOR_AB = {(6, 3), (7, 3), (8, 3)}
CORRIDOR_AC = {(3, 6), (3, 7), (3, 8)}
WALKABLE_TILES = ROOM_A | ROOM_B | ROOM_C | CORRIDOR_AB | CORRIDOR_AC

START_TILE = (1, 1)
KEY_TILE = (11, 11)
DOOR_TILE = (7, 3)
COLLECTIBLE_TILES = {(4, 4), (10, 2), (11, 11)}


def _tile_square(color: int, *, name: str, gx: int, gy: int, layer: int, tags: list[str], collidable: bool) -> Sprite:
    return Sprite(
        pixels=[[color for _ in range(TILE_SIZE)] for _ in range(TILE_SIZE)],
        name=name,
        x=gx * TILE_SIZE,
        y=gy * TILE_SIZE,
        layer=layer,
        tags=tags,
        collidable=collidable,
    )


def _timer_pixel(name: str, x: int) -> Sprite:
    return Sprite(pixels=[[COLORS["timer"]]], name=name, x=x, y=0, layer=6, tags=["timer"], collidable=False)


def _build_level() -> Level:
    sprites: list[Sprite] = [
        Sprite(
            pixels=[[COLORS["floor"] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)],
            name="floor",
            x=0,
            y=0,
            layer=0,
            tags=["floor", "sys_static"],
            collidable=False,
        )
    ]

    for gy in range(GRID_TILES):
        for gx in range(GRID_TILES):
            if (gx, gy) in WALKABLE_TILES:
                continue
            sprites.append(
                _tile_square(
                    COLORS["wall"],
                    name=f"wall-{gx}-{gy}",
                    gx=gx,
                    gy=gy,
                    layer=1,
                    tags=["wall", "blocker", "sys_static"],
                    collidable=True,
                )
            )

    sprites.append(
        _tile_square(
            COLORS["door_locked"],
            name="door",
            gx=DOOR_TILE[0],
            gy=DOOR_TILE[1],
            layer=3,
            tags=["door", "blocker", "locked"],
            collidable=True,
        )
    )

    sprites.append(
        _tile_square(
            COLORS["key"], name="key", gx=KEY_TILE[0], gy=KEY_TILE[1], layer=3, tags=["key", "pickup"], collidable=False
        )
    )

    for index, (gx, gy) in enumerate(sorted(COLLECTIBLE_TILES)):
        sprites.append(
            _tile_square(
                COLORS["collectible"],
                name=f"collectible-{index}",
                gx=gx,
                gy=gy,
                layer=2,
                tags=["collectible"],
                collidable=False,
            )
        )

    sprites.append(
        _tile_square(
            COLORS["player"],
            name="player",
            gx=START_TILE[0],
            gy=START_TILE[1],
            layer=4,
            tags=["player"],
            collidable=True,
        )
    )

    # Keep timer changes in the first pixel row so negative checks can safely ignore it.
    min_solution_steps = 52
    time_limit = max(120, min_solution_steps * 4)
    for x in range(time_limit):
        sprites.append(_timer_pixel(name=f"timer-{x}", x=x))

    return Level(
        grid_size=(GRID_SIZE, GRID_SIZE),
        sprites=sprites,
        data={
            "time_limit": time_limit,
            "collectible_total": len(COLLECTIBLE_TILES),
            "door_tile": DOOR_TILE,
            "key_tile": KEY_TILE,
            "start_tile": START_TILE,
        },
    )


class Roomswap(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level()]
        camera = Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLORS["floor"])
        super().__init__(
            game_id="roomswap", levels=levels, camera=camera, win_score=1, available_actions=[1, 2, 3, 4, 5], seed=seed
        )
        self._player: Sprite | None = None
        self._door: Sprite | None = None
        self._key: Sprite | None = None
        self._has_key = False
        self._remaining_collectibles = 0
        self._time_left = 0

    def on_set_level(self, level: Level) -> None:
        players = level.get_sprites_by_name("player")
        doors = level.get_sprites_by_name("door")
        keys = level.get_sprites_by_name("key")
        if not players or not doors:
            raise RuntimeError("missing required sprites")
        self._player = players[0]
        self._door = doors[0]
        self._key = keys[0] if keys else None
        self._has_key = False
        self._remaining_collectibles = len(level.get_sprites_by_tag("collectible"))
        self._time_left = int(level.get_data("time_limit") or 180)

    def _player_tile(self) -> tuple[int, int]:
        if self._player is None:
            return (0, 0)
        return (self._player.x // TILE_SIZE, self._player.y // TILE_SIZE)

    def _tagged_sprite_at_tile(self, gx: int, gy: int, tag: str) -> Sprite | None:
        px = gx * TILE_SIZE
        py = gy * TILE_SIZE
        for sprite in self.current_level.get_sprites_by_tag(tag):
            if sprite.x == px and sprite.y == py:
                return sprite
        return None

    def _is_blocked_tile(self, gx: int, gy: int) -> bool:
        wall = self._tagged_sprite_at_tile(gx, gy, "wall")
        if wall is not None:
            return True
        door = self._tagged_sprite_at_tile(gx, gy, "door")
        return bool(door is not None and "locked" in door.tags)

    def _try_move_player(self, dx: int, dy: int) -> None:
        if self._player is None:
            return
        gx, gy = self._player_tile()
        nx = gx + dx
        ny = gy + dy
        if self._is_blocked_tile(nx, ny):
            return
        self._player.set_position(nx * TILE_SIZE, ny * TILE_SIZE)

    def _unlock_adjacent_door(self) -> None:
        if not self._has_key or self._door is None:
            return
        if "locked" not in self._door.tags:
            return

        px, py = self._player_tile()
        dx = self._door.x // TILE_SIZE
        dy = self._door.y // TILE_SIZE
        if abs(px - dx) + abs(py - dy) != 1:
            return

        self._door.tags[:] = [tag for tag in self._door.tags if tag != "locked"]
        self._door.tags[:] = [tag for tag in self._door.tags if tag != "blocker"]
        self._door.tags.append("open")
        self._door.set_collidable(False)
        self._door.pixels[:, :] = COLORS["door_open"]

    def _pickup_key(self) -> None:
        if self._has_key:
            return
        gx, gy = self._player_tile()
        key = self._tagged_sprite_at_tile(gx, gy, "key")
        if key is None:
            return
        self.current_level.remove_sprite(key)
        self._key = None
        self._has_key = True

    def _collect_here(self) -> None:
        gx, gy = self._player_tile()
        item = self._tagged_sprite_at_tile(gx, gy, "collectible")
        if item is None:
            return
        self.current_level.remove_sprite(item)
        self._remaining_collectibles = max(0, self._remaining_collectibles - 1)

    def _update_timer(self) -> None:
        self._time_left -= 1
        remove_x = self._time_left
        timer = self.current_level.get_sprites_by_name(f"timer-{remove_x}")
        if timer:
            self.current_level.remove_sprite(timer[0])
        if self._time_left <= 0:
            self.lose()

    def step(self) -> None:
        if self._player is None:
            self.complete_action()
            return

        action = self.action.id
        move = MOVE_DELTAS.get(action)
        if move is not None:
            self._try_move_player(move[0], move[1])
        elif action == GameAction.ACTION5:
            self._pickup_key()
            self._unlock_adjacent_door()
            self._collect_here()

        if self._remaining_collectibles == 0:
            self.next_level()
            self.complete_action()
            return

        self._update_timer()
        self.complete_action()
