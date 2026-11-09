from __future__ import annotations

import random

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "identify_the_agent-0002"
VARIANT = "0002"
PIXEL_GRID = 64  # full 64x64 rendering grid
TIME_LIMIT = 200

COLOR_BG = 0
TOKEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Cross collectible pattern (4x4 pixels)
# fmt: off
CROSS_PATTERN = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
], dtype=np.int8)
# fmt: on


LEVEL_CONFIG = [
    (4, (2, 3), (12, 10), None),  # L1: open
    (2, (25, 10), (12, 22), None),  # L2: open, higher res
    (4, (3, 12), (4, 5), (8, 7)),  # L3: open, collect cross
]

ACTION_TO_DELTA = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}


def _token(color: int, size: int) -> np.ndarray:
    return np.full((size, size), int(color), dtype=np.int8)


def _build_level(include_cross: bool = False) -> Level:
    sprites = [
        Sprite(
            pixels=np.full((PIXEL_GRID, PIXEL_GRID), COLOR_BG, dtype=np.int8),
            name="floor",
            x=0,
            y=0,
            layer=0,
            tags=["floor", "sys_static"],
            collidable=False,
        ),
        Sprite(pixels=_token(1, 4), name="agent", x=0, y=0, layer=3, tags=["agent"], collidable=True),
        Sprite(pixels=_token(2, 4), name="target", x=0, y=0, layer=2, tags=["target"], collidable=False),
    ]
    if include_cross:
        sprites.append(
            Sprite(pixels=CROSS_PATTERN.copy(), name="cross", x=0, y=0, layer=2, tags=["cross"], collidable=False)
        )
    return Level(grid_size=(PIXEL_GRID, PIXEL_GRID), sprites=sprites, data={"time_limit": TIME_LIMIT})


class IdentifyTheAgent(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._seed = int(seed)
        rng = random.Random(self._seed)
        self._level_colors: list[tuple[int, int, int | None]] = []
        for cfg in LEVEL_CONFIG:
            agent_color = rng.choice(TOKEN_COLORS)
            target_color = rng.choice(TOKEN_COLORS)
            cross_color = rng.choice(TOKEN_COLORS) if cfg[3] is not None else None
            self._level_colors.append((agent_color, target_color, cross_color))
        self._time_left = TIME_LIMIT
        self._agent: Sprite | None = None
        self._target: Sprite | None = None
        self._agent_cell = (0, 0)
        self._target_cell = (0, 0)
        self._cell_size = 4
        self._grid_cells = PIXEL_GRID // 4
        self._identify_score = 0
        self._cross: Sprite | None = None
        self._cross_cell = (0, 0)
        self._cross_collected = False
        camera = Camera(width=PIXEL_GRID, height=PIXEL_GRID, background=COLOR_BG)
        super().__init__(
            game_id=GAME_ID,
            levels=[_build_level(include_cross=cfg[3] is not None) for cfg in LEVEL_CONFIG],
            camera=camera,
            win_score=len(LEVEL_CONFIG),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def _place_sprite(self, sprite: Sprite, cx: int, cy: int) -> None:
        sprite.set_position(cx * self._cell_size, cy * self._cell_size)

    def on_set_level(self, level: Level) -> None:
        self._time_left = int(level.get_data("time_limit") or TIME_LIMIT)
        self._identify_score = 0

        token_size, agent_pos, target_pos, cross_cell = LEVEL_CONFIG[self.level_index]
        agent_color, target_color, cross_color = self._level_colors[self.level_index]
        self._cell_size = token_size
        self._grid_cells = PIXEL_GRID // token_size

        agents = level.get_sprites_by_name("agent")
        targets = level.get_sprites_by_name("target")
        if not agents or not targets:
            raise RuntimeError("identify_the_agent level is missing agent or target sprites.")

        self._agent = agents[0]
        self._target = targets[0]

        self._agent.pixels = _token(agent_color, token_size)
        self._target.pixels = _token(target_color, token_size)

        self._agent_cell = agent_pos
        self._target_cell = target_pos
        self._place_sprite(self._agent, *agent_pos)
        self._place_sprite(self._target, *target_pos)

        # cross collectible
        self._cross_collected = False
        self._cross = None
        if cross_cell is not None:
            crosses = level.get_sprites_by_name("cross")
            if crosses:
                self._cross = crosses[0]
                self._cross_cell = cross_cell
                self._cross.pixels = np.where(CROSS_PATTERN, cross_color, COLOR_BG).astype(np.int8)
                self._place_sprite(self._cross, *self._cross_cell)

    def _try_move(self, dx: int, dy: int) -> None:
        if self._agent is None:
            return
        cx, cy = self._agent_cell
        nx, ny = cx + dx, cy + dy
        if not (0 <= nx < self._grid_cells and 0 <= ny < self._grid_cells):
            return
        self._agent_cell = (nx, ny)
        self._place_sprite(self._agent, nx, ny)
        self._identify_score += 1

    def _agent_on_target(self) -> bool:
        return self._agent_cell == self._target_cell

    def _check_cross_pickup(self) -> None:
        if self._cross is not None and not self._cross_collected:
            if self._agent_cell == self._cross_cell:
                self._cross_collected = True
                # hide the cross by making it fully transparent/bg
                self._cross.pixels = np.full_like(self._cross.pixels, COLOR_BG)

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        move = ACTION_TO_DELTA.get(action_id)
        if move is not None:
            self._try_move(move[0], move[1])

        self._check_cross_pickup()

        self._time_left -= 1
        # must collect cross (if present) before goal counts
        can_finish = self._cross is None or self._cross_collected
        if can_finish and self._agent_on_target():
            self.next_level()
            self.complete_action()
            return
        if self._time_left <= 0:
            self.lose()
        self.complete_action()
