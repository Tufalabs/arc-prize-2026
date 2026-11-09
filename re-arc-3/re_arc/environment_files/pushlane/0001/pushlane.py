import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_WALL = 5
COLOR_PLAYER_A = 9
COLOR_PLAYER_B = 10
COLOR_CRATE = 12
COLOR_SWITCH = 8
COLOR_GATE = 13
COLOR_COLLECT = 14
COLOR_TIMER_FULL = 14
COLOR_TIMER_EMPTY = 3

CELL = 2
GRID_SIZE = (30, 10)
TIME_LIMIT = 120

ACTION_TO_DELTA = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

LAYOUT = ["###############", "#P.C..a.......#", "###.#########.#", "#S..g......bc.#", "###############"]


def _solid(w: int, h: int, color: int) -> list[list[int]]:
    return [[color for _ in range(w)] for _ in range(h)]


sprites = {
    "background": Sprite(
        pixels=_solid(GRID_SIZE[0], GRID_SIZE[1], COLOR_BG), name="background", collidable=False, visible=True, layer=-6
    ),
    "floor": Sprite(pixels=_solid(CELL, CELL, COLOR_FLOOR), name="floor", collidable=False, visible=True, layer=-4),
    "wall": Sprite(
        pixels=_solid(CELL, CELL, COLOR_WALL), name="wall", collidable=True, visible=True, layer=-1, tags=["wall"]
    ),
    "player": Sprite(
        pixels=[[COLOR_PLAYER_A, COLOR_PLAYER_B], [COLOR_PLAYER_B, COLOR_PLAYER_A]],
        name="player",
        collidable=True,
        visible=True,
        layer=4,
        tags=["player"],
    ),
    "crate": Sprite(
        pixels=[[COLOR_CRATE, COLOR_CRATE], [COLOR_CRATE, COLOR_PLAYER_B]],
        name="crate",
        collidable=True,
        visible=True,
        layer=3,
        tags=["crate"],
    ),
    "switch": Sprite(
        pixels=[[COLOR_SWITCH, COLOR_SWITCH], [COLOR_SWITCH, COLOR_COLLECT]],
        name="switch",
        collidable=False,
        visible=True,
        layer=2,
        tags=["switch"],
    ),
    "gate": Sprite(
        pixels=_solid(CELL, CELL, COLOR_GATE),
        name="gate",
        collidable=True,
        visible=True,
        layer=3,
        tags=["gate", "wall"],
    ),
    "collect": Sprite(
        pixels=[[COLOR_COLLECT, COLOR_COLLECT], [COLOR_COLLECT, COLOR_FLOOR]],
        name="collect",
        collidable=False,
        visible=True,
        layer=1,
        tags=["collect"],
    ),
}


class _TimeBar(RenderableUserDisplay):
    def __init__(self, total: int):
        self.total = max(1, int(total))
        self.remaining = self.total

    def reset(self, total: int) -> None:
        self.total = max(1, int(total))
        self.remaining = self.total

    def set_remaining(self, remaining: int) -> None:
        self.remaining = max(0, min(int(remaining), self.total))

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        ratio = float(self.remaining) / float(self.total)
        fill = round(64 * ratio)
        for x in range(64):
            frame[0, x] = COLOR_TIMER_FULL if x < fill else COLOR_TIMER_EMPTY
        return frame


def _build_level() -> Level:
    out = [sprites["background"].clone()]
    switch_xy = None
    gate_xy = None

    for gy, row in enumerate(LAYOUT):
        for gx, cell in enumerate(row):
            x = gx * CELL
            y = gy * CELL
            if cell == "#":
                out.append(sprites["wall"].clone().set_position(x, y))
                continue

            out.append(sprites["floor"].clone().set_position(x, y))
            if cell == "P":
                out.append(sprites["player"].clone().set_position(x, y))
            elif cell == "C":
                out.append(sprites["crate"].clone().set_position(x, y))
            elif cell == "S":
                out.append(sprites["switch"].clone().set_position(x, y))
                switch_xy = (x, y)
            elif cell == "g":
                out.append(sprites["gate"].clone().set_position(x, y))
                gate_xy = (x, y)
            elif cell in {"a", "b", "c"}:
                out.append(sprites["collect"].clone().set_position(x, y))

    return Level(
        sprites=out,
        grid_size=GRID_SIZE,
        name="pushlane-main",
        data={"time_limit": TIME_LIMIT, "switch_xy": switch_xy, "gate_xy": gate_xy},
    )


class Pushlane(ARCBaseGame):
    def __init__(self):
        level = _build_level()
        self._levels = [level]
        self._timer = _TimeBar(level.get_data("time_limit"))
        camera = Camera(0, 0, GRID_SIZE[0], GRID_SIZE[1], COLOR_BG, COLOR_BG, [self._timer])
        super().__init__("pushlane", self._levels, camera, False, 1, [1, 2, 3, 4, 5])

    def on_set_level(self, level: Level) -> None:
        self.player = level.get_sprites_by_tag("player")[0]
        self.switch_xy = tuple(level.get_data("switch_xy"))
        self.gate_xy = tuple(level.get_data("gate_xy"))
        self.gate_open = False
        self.remaining_time = int(level.get_data("time_limit"))
        self._timer.reset(self.remaining_time)

    def _sprite_at(self, x: int, y: int, tag: str) -> Sprite | None:
        for sprite in self.current_level.get_sprites_by_tag(tag):
            if int(sprite.x) == x and int(sprite.y) == y:
                return sprite
        return None

    def _within_bounds(self, x: int, y: int, width: int, height: int) -> bool:
        return (
            x >= 0
            and y >= 0
            and x + width <= self.current_level.grid_size[0]
            and y + height <= self.current_level.grid_size[1]
        )

    def _overlaps(self, x: int, y: int, w: int, h: int, other: Sprite) -> bool:
        ox = int(other.x)
        oy = int(other.y)
        ow = int(other.width)
        oh = int(other.height)
        return x < ox + ow and x + w > ox and y < oy + oh and y + h > oy

    def _blocked(self, x: int, y: int, moving: Sprite, ignore: Sprite | None = None) -> bool:
        if not self._within_bounds(x, y, moving.width, moving.height):
            return True
        for sprite in self.current_level.get_sprites():
            if sprite is moving or sprite is ignore:
                continue
            tags = set(getattr(sprite, "tags", []) or [])
            if not tags.intersection({"wall", "crate", "gate"}):
                continue
            if self._overlaps(x, y, moving.width, moving.height, sprite):
                return True
        return False

    def _set_gate_open(self, is_open: bool) -> None:
        gate = self._sprite_at(self.gate_xy[0], self.gate_xy[1], "gate")
        if is_open and gate is not None:
            self.current_level.remove_sprite(gate)
            self.gate_open = True
            return
        if (not is_open) and gate is None:
            self.current_level.add_sprite(sprites["gate"].clone().set_position(self.gate_xy[0], self.gate_xy[1]))
            self.gate_open = False

    def _move_player(self, dx: int, dy: int) -> None:
        nx = int(self.player.x + dx * CELL)
        ny = int(self.player.y + dy * CELL)

        crate = self._sprite_at(nx, ny, "crate")
        if crate is not None:
            cx = int(crate.x + dx * CELL)
            cy = int(crate.y + dy * CELL)
            if self._blocked(cx, cy, crate, ignore=self.player):
                return
            crate.set_position(cx, cy)
        elif self._blocked(nx, ny, self.player):
            return

        self.player.set_position(nx, ny)

    def _consume_collect(self) -> None:
        px = int(self.player.x)
        py = int(self.player.y)
        collect = self._sprite_at(px, py, "collect")
        if collect is not None:
            self.current_level.remove_sprite(collect)

    def step(self) -> None:
        self.remaining_time -= 1
        self._timer.set_remaining(self.remaining_time)
        if self.remaining_time <= 0:
            self.lose()
            self.complete_action()
            return

        action_id = int(self.action.id.value)
        if action_id in ACTION_TO_DELTA:
            dx, dy = ACTION_TO_DELTA[action_id]
            self._move_player(dx, dy)
        elif action_id == 5 and (int(self.player.x), int(self.player.y)) == self.switch_xy:
            self._set_gate_open(True)

        self._consume_collect()
        if not self.current_level.get_sprites_by_tag("collect"):
            self.next_level()
            self.complete_action()
            return

        self.complete_action()
