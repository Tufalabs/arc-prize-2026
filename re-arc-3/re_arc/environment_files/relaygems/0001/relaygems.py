from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_WIDTH = 15
GRID_HEIGHT = 11

COLORS = {
    "floor": 1,
    "wall": 5,
    "player": 9,
    "goal": 14,
    "gem": 11,
    "switch_off": 12,
    "switch_on": 6,
    "gate_closed": 8,
    "gate_open": 10,
    "timer_bg": 2,
    "timer_fill": 11,
}

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

BLOCKER_TAGS = {"wall", "blocker", "gate_closed"}

PLAYER_START = (1, 1)
GOAL_POS = (13, 2)
SWITCH_POS = (3, 8)
GATE_POS = (9, 4)
GEM_POSITIONS = [(4, 2), (7, 7), (12, 7)]

INTERNAL_WALLS = [(9, y) for y in range(1, 10) if y != GATE_POS[1]]

# fmt: off
ACTION_PLAN = [
    4, 4, 4, 2, 4, 4, 4, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 5, 4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 2, 2, 2, 4, 1, 1,
    1, 1, 1,
]
# fmt: on


def _pixel(color: int):
    return np.array([[color]], dtype=np.int8)


def _sprite(
    *,
    name: str,
    color: int,
    x: int,
    y: int,
    layer: int,
    tags: list[str],
    collidable: bool,
    width: int = 1,
    height: int = 1,
) -> Sprite:
    return Sprite(
        pixels=[[_ for _ in [color] * width] for _ in range(height)],
        name=name,
        x=x,
        y=y,
        layer=layer,
        tags=tags,
        collidable=collidable,
    )


def _build_level() -> Level:
    sprites: list[Sprite] = []

    sprites.append(
        _sprite(
            name="floor",
            color=COLORS["floor"],
            x=0,
            y=0,
            layer=0,
            tags=["floor", "sys_static"],
            collidable=False,
            width=GRID_WIDTH,
            height=GRID_HEIGHT,
        )
    )

    for x in range(GRID_WIDTH):
        sprites.append(
            _sprite(
                name=f"wall-top-{x}", color=COLORS["wall"], x=x, y=0, layer=1, tags=["wall", "blocker"], collidable=True
            )
        )
        sprites.append(
            _sprite(
                name=f"wall-bottom-{x}",
                color=COLORS["wall"],
                x=x,
                y=GRID_HEIGHT - 1,
                layer=1,
                tags=["wall", "blocker"],
                collidable=True,
            )
        )
    for y in range(1, GRID_HEIGHT - 1):
        sprites.append(
            _sprite(
                name=f"wall-left-{y}",
                color=COLORS["wall"],
                x=0,
                y=y,
                layer=1,
                tags=["wall", "blocker"],
                collidable=True,
            )
        )
        sprites.append(
            _sprite(
                name=f"wall-right-{y}",
                color=COLORS["wall"],
                x=GRID_WIDTH - 1,
                y=y,
                layer=1,
                tags=["wall", "blocker"],
                collidable=True,
            )
        )

    for wx, wy in INTERNAL_WALLS:
        sprites.append(
            _sprite(
                name=f"wall-inner-{wx}-{wy}",
                color=COLORS["wall"],
                x=wx,
                y=wy,
                layer=1,
                tags=["wall", "blocker"],
                collidable=True,
            )
        )

    sprites.append(
        _sprite(
            name="remote_gate",
            color=COLORS["gate_closed"],
            x=GATE_POS[0],
            y=GATE_POS[1],
            layer=2,
            tags=["gate", "gate_closed", "blocker"],
            collidable=True,
        )
    )
    sprites.append(
        _sprite(
            name="switch",
            color=COLORS["switch_off"],
            x=SWITCH_POS[0],
            y=SWITCH_POS[1],
            layer=2,
            tags=["switch", "switch_off"],
            collidable=False,
        )
    )
    sprites.append(
        _sprite(
            name="goal", color=COLORS["goal"], x=GOAL_POS[0], y=GOAL_POS[1], layer=2, tags=["goal"], collidable=False
        )
    )
    for idx, (gx, gy) in enumerate(GEM_POSITIONS):
        sprites.append(
            _sprite(
                name=f"gem-{idx}",
                color=COLORS["gem"],
                x=gx,
                y=gy,
                layer=3,
                tags=["gem", "collectible"],
                collidable=False,
            )
        )
    sprites.append(
        _sprite(
            name="player",
            color=COLORS["player"],
            x=PLAYER_START[0],
            y=PLAYER_START[1],
            layer=4,
            tags=["player"],
            collidable=True,
        )
    )
    sprites.append(
        _sprite(
            name="timer_bg",
            color=COLORS["timer_bg"],
            x=0,
            y=0,
            layer=5,
            tags=["timer", "ui"],
            collidable=False,
            width=GRID_WIDTH,
        )
    )
    sprites.append(
        _sprite(
            name="timer_fill",
            color=COLORS["timer_fill"],
            x=0,
            y=0,
            layer=6,
            tags=["timer", "ui"],
            collidable=False,
            width=GRID_WIDTH,
        )
    )

    time_limit = max(90, len(ACTION_PLAN) * 4)
    return Level(
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=sprites,
        data={"time_limit": time_limit, "gem_count": len(GEM_POSITIONS), "action_plan": ACTION_PLAN},
    )


class Relaygems(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level()]
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLORS["floor"])
        super().__init__(
            game_id="relaygems", levels=levels, camera=camera, win_score=1, available_actions=[1, 2, 3, 4, 5], seed=seed
        )
        self.player: Sprite | None = None
        self.goal: Sprite | None = None
        self.switch: Sprite | None = None
        self.gate: Sprite | None = None
        self.time_limit = 0
        self.time_left = 0
        self.gems_total = 0
        self.gems_collected = 0
        self.gate_open = False

    def on_set_level(self, level: Level) -> None:
        players = level.get_sprites_by_name("player")
        goals = level.get_sprites_by_name("goal")
        switches = level.get_sprites_by_name("switch")
        gates = level.get_sprites_by_name("remote_gate")
        if not players or not goals or not switches or not gates:
            raise RuntimeError("relaygems level missing required sprites")
        self.player = players[0]
        self.goal = goals[0]
        self.switch = switches[0]
        self.gate = gates[0]
        self.time_limit = int(level.get_data("time_limit") or 120)
        self.time_left = self.time_limit
        self.gems_total = int(level.get_data("gem_count") or 0)
        self.gems_collected = 0
        self.gate_open = False
        self._apply_gate_state(open_gate=False)
        self._set_switch_state(active=False)
        self._update_timer_bar()

    def _sprites_at(self, x: int, y: int) -> list[Sprite]:
        out: list[Sprite] = []
        for sprite in self.current_level.get_sprites():
            if x >= sprite.x and y >= sprite.y and x < sprite.x + sprite.width and y < sprite.y + sprite.height:
                out.append(sprite)
        return out

    def _is_blocked(self, x: int, y: int) -> bool:
        for sprite in self._sprites_at(x, y):
            if BLOCKER_TAGS.intersection(sprite.tags):
                return True
        return False

    def _try_move_player(self, dx: int, dy: int) -> None:
        if self.player is None:
            return
        nx = int(self.player.x + dx)
        ny = int(self.player.y + dy)
        if nx < 0 or ny < 0 or nx >= GRID_WIDTH or ny >= GRID_HEIGHT:
            return
        if self._is_blocked(nx, ny):
            return
        self.player.set_position(nx, ny)

    def _set_switch_state(self, active: bool) -> None:
        if self.switch is None:
            return
        tags = self.switch.tags
        if active:
            self.switch.pixels = _pixel(COLORS["switch_on"])
            tags[:] = [tag for tag in tags if tag != "switch_off"]
            if "switch_on" not in tags:
                tags.append("switch_on")
        else:
            self.switch.pixels = _pixel(COLORS["switch_off"])
            tags[:] = [tag for tag in tags if tag != "switch_on"]
            if "switch_off" not in tags:
                tags.append("switch_off")

    def _apply_gate_state(self, open_gate: bool) -> None:
        if self.gate is None:
            return
        self.gate_open = bool(open_gate)
        tags = self.gate.tags
        if self.gate_open:
            self.gate.pixels = _pixel(COLORS["gate_open"])
            tags[:] = [tag for tag in tags if tag not in {"gate_closed", "blocker"}]
            if "gate_open" not in tags:
                tags.append("gate_open")
        else:
            self.gate.pixels = _pixel(COLORS["gate_closed"])
            tags[:] = [tag for tag in tags if tag != "gate_open"]
            if "gate_closed" not in tags:
                tags.append("gate_closed")
            if "blocker" not in tags:
                tags.append("blocker")

    def _handle_collectibles(self) -> None:
        if self.player is None:
            return
        gems = self.current_level.get_sprites_by_tag("gem")
        px = int(self.player.x)
        py = int(self.player.y)
        for gem in gems:
            if int(gem.x) == px and int(gem.y) == py:
                self.current_level.remove_sprite(gem)
                self.gems_collected += 1
                break

    def _update_timer_bar(self) -> None:
        timer_fill = self.current_level.get_sprites_by_name("timer_fill")
        if timer_fill:
            self.current_level.remove_sprite(timer_fill[0])
        ratio = max(0.0, float(self.time_left) / float(max(1, self.time_limit)))
        width = max(0, min(GRID_WIDTH, round(ratio * GRID_WIDTH)))
        if width <= 0:
            return
        self.current_level.add_sprite(
            _sprite(
                name="timer_fill",
                color=COLORS["timer_fill"],
                x=0,
                y=0,
                layer=6,
                tags=["timer", "ui"],
                collidable=False,
                width=width,
            )
        )

    def step(self) -> None:
        if self.player is None or self.goal is None or self.switch is None:
            self.complete_action()
            return

        action = self.action.id
        move = MOVE_DELTAS.get(action)
        if move is not None:
            self._try_move_player(move[0], move[1])
        elif action == GameAction.ACTION5:
            on_switch = int(self.player.x) == int(self.switch.x) and int(self.player.y) == int(self.switch.y)
            if on_switch:
                self._apply_gate_state(open_gate=not self.gate_open)
                self._set_switch_state(active=self.gate_open)

        self._handle_collectibles()

        if (
            int(self.player.x) == int(self.goal.x)
            and int(self.player.y) == int(self.goal.y)
            and self.gems_collected >= self.gems_total
        ):
            self.next_level()
            self.complete_action()
            return

        self.time_left -= 1
        self._update_timer_bar()
        if self.time_left <= 0:
            self.lose()
            self.complete_action()
            return

        self.complete_action()
