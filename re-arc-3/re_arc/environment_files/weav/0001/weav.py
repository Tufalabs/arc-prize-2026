from __future__ import annotations

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 1
WALL_COLOR = 4
PHASE0_COLOR = 10
PHASE1_COLOR = 15
GOAL_COLOR = 11
PLAYER_COLORS = [9, 14]

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

LEVEL_SPECS: list[tuple[str, int, int, int]] = [
    ("Level 1", 9, 9, 6),
    ("Level 2", 11, 9, 8),
    ("Level 3", 11, 11, 10),
    ("Level 4", 13, 11, 12),
    ("Level 5", 13, 13, 14),
    ("Level 6", 15, 13, 16),
    ("Level 7", 15, 15, 18),
]


def _snake_path(width: int, height: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    y = 1
    direction = 1
    while y <= height - 2:
        x_range = range(1, width - 1) if direction == 1 else range(width - 2, 0, -1)
        for x in x_range:
            path.append((x, y))

        y += 1
        if y > height - 2:
            break

        connector_x = width - 2 if direction == 1 else 1
        path.append((connector_x, y))
        y += 1
        direction *= -1

    return path


def _build_level(spec: tuple[str, int, int, int], seed: int) -> Level:
    name, width, height, lock_count = spec
    path = _snake_path(width, height)
    path_set = set(path)

    start = path[0]
    goal = path[-1]
    start_mode = seed % 2

    usable = path[2:-2]
    step = max(1, len(usable) // (lock_count + 1))
    idx = step
    phase0_tiles: set[tuple[int, int]] = set()
    phase1_tiles: set[tuple[int, int]] = set()

    placed = 0
    while placed < lock_count and idx < len(usable):
        cell = usable[idx]
        if ((cell[0] + cell[1] + seed + placed) % 2) == 0:
            phase0_tiles.add(cell)
        else:
            phase1_tiles.add(cell)
        placed += 1
        idx += step

    for cell in usable:
        if placed >= lock_count:
            break
        if cell in phase0_tiles or cell in phase1_tiles:
            continue
        if ((cell[0] * 3 + cell[1] + seed + placed) % 2) == 0:
            phase0_tiles.add(cell)
        else:
            phase1_tiles.add(cell)
        placed += 1

    phase0_tiles.discard(start)
    phase0_tiles.discard(goal)
    phase1_tiles.discard(start)
    phase1_tiles.discard(goal)

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]
    wall_pixels = [[-1] * width for _ in range(height)]
    phase0_pixels = [[-1] * width for _ in range(height)]
    phase1_pixels = [[-1] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                wall_pixels[y][x] = WALL_COLOR
                continue
            if (x, y) not in path_set:
                wall_pixels[y][x] = WALL_COLOR

    for x, y in phase0_tiles:
        phase0_pixels[y][x] = PHASE0_COLOR
    for x, y in phase1_tiles:
        phase1_pixels[y][x] = PHASE1_COLOR

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(pixels=phase0_pixels, name="phase0_tiles", collidable=False, layer=-8, tags=["phase0"]),
        Sprite(pixels=phase1_pixels, name="phase1_tiles", collidable=False, layer=-7, tags=["phase1"]),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=1, tags=["wall"]),
        Sprite(pixels=[[GOAL_COLOR]], name="goal", x=goal[0], y=goal[1], collidable=False, layer=3, tags=["goal"]),
        Sprite(
            pixels=[[PLAYER_COLORS[start_mode]]],
            name="player",
            x=start[0],
            y=start[1],
            collidable=False,
            layer=5,
            tags=["player"],
        ),
    ]

    return Level(
        name=name,
        sprites=sprites,
        grid_size=(width, height),
        data={
            "width": width,
            "height": height,
            "start": start,
            "goal": goal,
            "start_mode": start_mode,
            "walls": [(x, y) for y in range(height) for x in range(width) if wall_pixels[y][x] != -1],
            "phase0_tiles": sorted((int(x), int(y)) for x, y in phase0_tiles),
            "phase1_tiles": sorted((int(x), int(y)) for x, y in phase1_tiles),
        },
    )


class Weav(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec, seed=idx + 67) for idx, spec in enumerate(LEVEL_SPECS)]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__(
            "weav", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4, 5]
        )

    def on_set_level(self, level: Level) -> None:
        self._width = int(level.get_data("width"))
        self._height = int(level.get_data("height"))
        self._goal = tuple(int(v) for v in level.get_data("goal"))
        self._mode = int(level.get_data("start_mode") or 0) % 2

        self._walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        self._phase_tiles = {
            0: {tuple(int(v) for v in item) for item in (level.get_data("phase0_tiles") or [])},
            1: {tuple(int(v) for v in item) for item in (level.get_data("phase1_tiles") or [])},
        }

        self._player = self.current_level.get_sprites_by_name("player")[0]
        self._sync_visuals()

    def _sync_visuals(self) -> None:
        self._player.pixels[0][0] = PLAYER_COLORS[self._mode]

    def _blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._width or y >= self._height:
            return True
        if (x, y) in self._walls:
            return True
        if (x, y) in self._phase_tiles[0] and self._mode != 0:
            return True
        return bool((x, y) in self._phase_tiles[1] and self._mode != 1)

    def _toggle_mode(self) -> None:
        self._mode ^= 1
        self._sync_visuals()

    def _try_move(self, dx: int, dy: int) -> None:
        nx = int(self._player.x + dx)
        ny = int(self._player.y + dy)
        if self._blocked(nx, ny):
            return
        self._player.set_position(nx, ny)
        if (nx, ny) == self._goal:
            self.next_level()

    def step(self) -> None:
        action = self.action.id
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._try_move(dx, dy)
        elif action == GameAction.ACTION5:
            self._toggle_mode()
        self.complete_action()
