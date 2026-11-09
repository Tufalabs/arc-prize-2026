from __future__ import annotations

from collections import deque

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 2
WALL_COLOR = 6
PLAYER_COLOR = 14
CRATE_COLOR = 12
GOAL_COLOR = 11
ICE_COLOR = 10

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),  # W
    GameAction.ACTION2: (0, 1),  # S
    GameAction.ACTION3: (-1, 0),  # A
    GameAction.ACTION4: (1, 0),  # D
}

LEVEL_SPECS: list[tuple[str, int, int, int]] = [
    ("Level 1", 11, 11, 1),
    ("Level 2", 13, 11, 2),
    ("Level 3", 13, 13, 2),
    ("Level 4", 15, 13, 3),
    ("Level 5", 15, 15, 3),
    ("Level 6", 17, 15, 4),
    ("Level 7", 17, 17, 5),
    ("Level 8", 19, 17, 6),
]

ACTIVE_LEVEL_COUNT = 7


def _build_room_grid(width: int, height: int, pillars: int) -> list[list[str]]:
    grid = [["."] * width for _ in range(height)]
    for x in range(width):
        grid[0][x] = "#"
        grid[height - 1][x] = "#"
    for y in range(height):
        grid[y][0] = "#"
        grid[y][width - 1] = "#"

    inner_w = max(1, width - 6)
    inner_h = max(1, height - 6)
    for idx in range(pillars):
        x = 3 + (idx * 3) % inner_w
        y = 3 + (idx * 2) % inner_h
        grid[y][x] = "#"

    return grid


def _place_ice(grid: list[list[str]], width: int, height: int, bands: int) -> None:
    for idx in range(bands):
        row = 2 + (idx * 3) % (height - 4)
        for col in range(2, width - 2):
            if grid[row][col] == ".":
                grid[row][col] = "I"
        col = 2 + (idx * 4) % (width - 4)
        for r in range(2, height - 2):
            if grid[r][col] == ".":
                grid[r][col] = "I"


def _build_level(spec: tuple[str, int, int, int]) -> Level:
    name, width, height, pillars = spec
    grid = _build_room_grid(width, height, pillars)
    _place_ice(grid, width, height, max(1, pillars // 2))

    start = (2, 2)
    goal = (width - 3, height - 3)
    crate = (width // 2, height // 2)

    for pos in (start, goal, crate):
        grid[pos[1]][pos[0]] = "."

    grid[start[1]][start[0]] = "P"
    grid[goal[1]][goal[0]] = "G"
    grid[crate[1]][crate[0]] = "C"

    return _level_from_grid(name, grid)


def _level_from_grid(name: str, grid: list[list[str]]) -> Level:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if width == 0 or height == 0:
        raise ValueError("Grid must be non-empty.")

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]
    wall_pixels = [[-1] * width for _ in range(height)]
    ice_pixels = [[-1] * width for _ in range(height)]

    player_pos: tuple[int, int] | None = None
    goal_pos: tuple[int, int] | None = None
    crate_pos: tuple[int, int] | None = None

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
            elif cell == "C":
                crate_pos = (x, y)
            elif cell == "I":
                ice_pixels[y][x] = ICE_COLOR

    if player_pos is None or goal_pos is None or crate_pos is None:
        raise ValueError("Each level must have a player, crate, and goal.")

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=1, tags=["wall"]),
        Sprite(pixels=ice_pixels, name="ice", collidable=False, layer=-5, tags=["ice"]),
        Sprite(
            pixels=[[GOAL_COLOR]], name="goal", x=goal_pos[0], y=goal_pos[1], collidable=False, layer=3, tags=["goal"]
        ),
        Sprite(
            pixels=[[CRATE_COLOR]],
            name="crate",
            x=crate_pos[0],
            y=crate_pos[1],
            collidable=True,
            layer=4,
            tags=["crate"],
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

    return Level(
        name=name, sprites=sprites, grid_size=(width, height), data={"goal": goal_pos, "width": width, "height": height}
    )


class Push(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS[:ACTIVE_LEVEL_COUNT]]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        self._solvable_cache: dict[int, dict[tuple[tuple[int, int], tuple[int, int]], bool]] = {}
        super().__init__(
            "push", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4, 5]
        )

    def on_set_level(self, level: Level) -> None:
        self._level_width = int(level.get_data("width"))
        self._level_height = int(level.get_data("height"))
        self._goal = tuple(level.get_data("goal"))
        self._wall_positions = {
            (x, y)
            for y, row in enumerate(level.get_sprites_by_name("walls")[0].render())
            for x, value in enumerate(row)
            if value == WALL_COLOR
        }
        self._ice_positions = {
            (x, y)
            for y, row in enumerate(level.get_sprites_by_name("ice")[0].render())
            for x, value in enumerate(row)
            if value == ICE_COLOR
        }
        self._solvable_cache.setdefault(self.level_index, {})

    def _is_wall(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._level_width or y >= self._level_height:
            return True
        return (x, y) in self._wall_positions

    def _crate_at(self, x: int, y: int) -> Sprite | None:
        crate = self.current_level.get_sprites_by_name("crate")[0]
        if crate.x == x and crate.y == y:
            return crate
        return None

    def _is_ice(self, x: int, y: int) -> bool:
        return (x, y) in self._ice_positions

    def _state_blocked(self, x: int, y: int, crate_pos: tuple[int, int]) -> bool:
        return self._is_wall(x, y) or (x, y) == crate_pos

    def _slide_state_to_stop(
        self, x: int, y: int, dx: int, dy: int, crate_pos: tuple[int, int], ignore_crate: bool = False
    ) -> tuple[int, int]:
        while self._is_ice(x, y):
            nx, ny = x + dx, y + dy
            if self._is_wall(nx, ny) or ((nx, ny) == crate_pos and not ignore_crate):
                break
            x, y = nx, ny
        return x, y

    def _simulate_move(
        self, player_pos: tuple[int, int], crate_pos: tuple[int, int], dx: int, dy: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        next_x = player_pos[0] + dx
        next_y = player_pos[1] + dy

        if self._is_wall(next_x, next_y):
            return player_pos, crate_pos

        next_crate_pos = crate_pos
        if (next_x, next_y) == crate_pos:
            beyond_x = next_x + dx
            beyond_y = next_y + dy
            if self._state_blocked(beyond_x, beyond_y, crate_pos):
                return player_pos, crate_pos
            next_crate_pos = self._slide_state_to_stop(beyond_x, beyond_y, dx, dy, crate_pos, ignore_crate=True)

        next_player_pos = self._slide_state_to_stop(next_x, next_y, dx, dy, next_crate_pos)
        return next_player_pos, next_crate_pos

    def _has_solution_from_current_state(self) -> bool:
        player = self.current_level.get_sprites_by_name("player")[0]
        crate = self.current_level.get_sprites_by_name("crate")[0]
        start_state = ((player.x, player.y), (crate.x, crate.y))

        if start_state[1] == self._goal:
            return True

        cache = self._solvable_cache[self.level_index]
        cached = cache.get(start_state)
        if cached is not None:
            return cached

        queue = deque([start_state])
        visited = {start_state}

        while queue:
            player_pos, crate_pos = queue.popleft()
            for dx, dy in MOVE_DELTAS.values():
                next_state = self._simulate_move(player_pos, crate_pos, dx, dy)
                if next_state[1] == self._goal:
                    cache[start_state] = True
                    return True
                if next_state in visited:
                    continue
                visited.add(next_state)
                queue.append(next_state)

        for state in visited:
            cache[state] = False
        return False

    def _slide_to_stop(self, x: int, y: int, dx: int, dy: int, blocked) -> tuple[int, int]:
        while self._is_ice(x, y):
            nx, ny = x + dx, y + dy
            if blocked(nx, ny):
                break
            x, y = nx, ny
        return x, y

    def _try_move(self, dx: int, dy: int) -> None:
        player = self.current_level.get_sprites_by_name("player")[0]
        next_x = player.x + dx
        next_y = player.y + dy

        if self._is_wall(next_x, next_y):
            return

        crate = self._crate_at(next_x, next_y)
        if crate is not None:
            beyond_x = next_x + dx
            beyond_y = next_y + dy
            if self._is_wall(beyond_x, beyond_y) or self._crate_at(beyond_x, beyond_y):
                return
            crate_x, crate_y = beyond_x, beyond_y
            crate_x, crate_y = self._slide_to_stop(
                crate_x, crate_y, dx, dy, lambda x, y: self._is_wall(x, y) or self._crate_at(x, y)
            )
            crate.set_position(crate_x, crate_y)

        player_x, player_y = next_x, next_y
        player_x, player_y = self._slide_to_stop(
            player_x, player_y, dx, dy, lambda x, y: self._is_wall(x, y) or self._crate_at(x, y)
        )
        player.set_position(player_x, player_y)

        if crate is not None and (crate.x, crate.y) == self._goal:
            self.next_level()
            return

        if not self._has_solution_from_current_state():
            self.lose()

    def step(self) -> None:
        action = self.action.id
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._try_move(dx, dy)

        self.complete_action()
