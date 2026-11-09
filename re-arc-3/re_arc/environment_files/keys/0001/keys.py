from __future__ import annotations

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 1
WALL_COLOR = 5
PLAYER_COLOR = 9
GOAL_COLOR = 10
LOCK_COLOR = 8
KEY_COLORS = [11, 12, 14, 15, 6, 7]

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),  # W
    GameAction.ACTION2: (0, 1),  # S
    GameAction.ACTION3: (-1, 0),  # A
    GameAction.ACTION4: (1, 0),  # D
}

LEVEL_SPECS: list[tuple[str, int, int, int]] = [
    ("Level 1", 9, 9, 1),
    ("Level 2", 11, 9, 2),
    ("Level 3", 11, 11, 3),
    ("Level 4", 13, 11, 3),
    ("Level 5", 13, 13, 4),
    ("Level 6", 15, 13, 4),
    ("Level 7", 15, 15, 5),
    ("Level 8", 17, 15, 6),
]


def _build_corridor_grid(width: int, height: int) -> list[list[str]]:
    grid = [["."] * width for _ in range(height)]
    for x in range(width):
        grid[0][x] = "#"
        grid[height - 1][x] = "#"
    for y in range(height):
        grid[y][0] = "#"
        grid[y][width - 1] = "#"

    for col in range(2, width - 2, 3):
        gap_row = 1 if (col // 3) % 2 == 0 else height - 2
        for row in range(1, height - 1):
            if row == gap_row:
                continue
            grid[row][col] = "#"
    return grid


def _reachable(grid: list[list[str]], start: tuple[int, int]) -> set[tuple[int, int]]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    stack = [start]
    seen = {start}
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if grid[ny][nx] == "#":
                continue
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                stack.append((nx, ny))
    return seen


def _place_keys(
    grid: list[list[str]], key_count: int, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    reachable = _reachable(grid, start)
    floor_cells = [
        (x, y)
        for y, row in enumerate(grid)
        for x, cell in enumerate(row)
        if cell == "." and (x, y) in reachable and (x, y) not in (start, goal)
    ]
    if not floor_cells:
        return []
    step = max(1, len(floor_cells) // (key_count + 1))
    positions: list[tuple[int, int]] = []
    idx = step
    while len(positions) < key_count and idx < len(floor_cells):
        positions.append(floor_cells[idx])
        idx += step
    if len(positions) < key_count:
        for pos in floor_cells:
            if pos in positions:
                continue
            positions.append(pos)
            if len(positions) >= key_count:
                break
    for x, y in positions:
        grid[y][x] = "K"
    return positions


def _build_level(spec: tuple[str, int, int, int]) -> Level:
    name, width, height, keys = spec
    grid = _build_corridor_grid(width, height)
    start = (1, 1)
    goal = (width - 2, height - 2)
    grid[start[1]][start[0]] = "P"
    grid[goal[1]][goal[0]] = "G"
    key_order = _place_keys(grid, keys, start, goal)
    return _level_from_grid(name, grid, key_order, start)


def _level_from_grid(
    name: str, grid: list[list[str]], key_order: list[tuple[int, int]], start: tuple[int, int]
) -> Level:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if width == 0 or height == 0:
        raise ValueError("Grid must be non-empty.")

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]
    wall_pixels = [[-1] * width for _ in range(height)]

    player_pos: tuple[int, int] | None = None
    goal_pos: tuple[int, int] | None = None
    key_positions: list[tuple[int, int]] = []

    for y, row in enumerate(grid):
        if len(row) != width:
            raise ValueError("All grid rows must have the same width.")
        for x, cell in enumerate(row):
            if cell == "#":
                wall_pixels[y][x] = WALL_COLOR
            elif cell == "P":
                player_pos = (x, y)
            elif cell == "G":
                goal_pos = (x, y)
            elif cell == "K":
                key_positions.append((x, y))

    if player_pos is None or goal_pos is None:
        raise ValueError("Each level must have a player and a goal.")

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=1, tags=["wall"]),
        Sprite(
            pixels=[[GOAL_COLOR]], name="goal", x=goal_pos[0], y=goal_pos[1], collidable=False, layer=3, tags=["goal"]
        ),
        Sprite(
            pixels=[[PLAYER_COLOR]],
            name="player",
            x=player_pos[0],
            y=player_pos[1],
            collidable=True,
            layer=5,
            tags=["player"],
        ),
    ]

    if key_order:
        key_positions = key_order
    for idx, (x, y) in enumerate(key_positions):
        color = KEY_COLORS[idx % len(KEY_COLORS)]
        sprites.append(Sprite(pixels=[[color]], name=f"key_{idx}", x=x, y=y, collidable=False, layer=4, tags=["key"]))

    sprites.append(
        Sprite(
            pixels=[[LOCK_COLOR]], name="lock", x=goal_pos[0], y=goal_pos[1], collidable=True, layer=6, tags=["lock"]
        )
    )

    return Level(
        name=name,
        sprites=sprites,
        grid_size=(width, height),
        data={"goal": goal_pos, "start": start, "key_order": key_order},
    )


class Keys(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__(
            "keys", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4, 5]
        )

    def on_set_level(self, _level: Level) -> None:
        self._key_index = 0
        self._update_lock()

    def _update_lock(self) -> None:
        lock = self.current_level.get_sprites_by_tag("lock")
        if not lock:
            return
        unlocked = self._key_index >= len(self.current_level.get_data("key_order") or [])
        for sprite in lock:
            sprite.set_collidable(not unlocked)
            sprite.set_visible(not unlocked)

    def _collect_keys(self, player: Sprite) -> bool:
        keys = self.current_level.get_sprites_by_tag("key")
        if not keys:
            return True
        expected = self._key_index
        for key in keys:
            if key.x == player.x and key.y == player.y:
                index = int(key.name.split("_")[-1])
                if index != expected:
                    self.lose()
                    return False
                self.current_level.remove_sprite(key)
                self._key_index += 1
                self._update_lock()
                return True
        return True

    def _at_goal(self, player: Sprite) -> bool:
        goal = self.current_level.get_data("goal")
        return bool(goal) and player.x == goal[0] and player.y == goal[1]

    def step(self) -> None:
        action = self.action.id
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            player = self.current_level.get_sprites_by_name("player")[0]
            self.try_move_sprite(player, dx, dy)
            if not self._collect_keys(player):
                self.complete_action()
                return
            if self._key_index >= len(self.current_level.get_data("key_order") or []) and self._at_goal(player):
                self.next_level()

        self.complete_action()
