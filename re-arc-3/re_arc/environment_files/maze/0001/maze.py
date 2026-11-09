from __future__ import annotations

from collections.abc import Iterable

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 4
WALL_COLOR = 3
PLAYER_COLOR = 9
GOAL_COLOR = 11
DOOR_COLOR = 12

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),  # W
    GameAction.ACTION2: (0, 1),  # S
    GameAction.ACTION3: (-1, 0),  # A
    GameAction.ACTION4: (1, 0),  # D
}


LEVEL_SPECS: list[tuple[str, int, int, int]] = [
    ("Level 1", 9, 9, 1),
    ("Level 2", 11, 9, 1),
    ("Level 3", 11, 11, 2),
    ("Level 4", 13, 11, 2),
    ("Level 5", 13, 13, 3),
    ("Level 6", 15, 13, 3),
    ("Level 7", 15, 15, 4),
    ("Level 8", 17, 15, 5),
]


def _build_snake_maze(width: int, height: int) -> tuple[list[list[str]], list[tuple[int, int]]]:
    if width < 5 or height < 5:
        raise ValueError("Maze size must be at least 5x5.")

    grid = [["#"] * width for _ in range(height)]
    path: list[tuple[int, int]] = []

    y = 1
    direction = 1
    while y <= height - 2:
        if direction == 1:
            x_range: Iterable[int] = range(1, width - 1)
        else:
            x_range = range(width - 2, 0, -1)

        for x in x_range:
            grid[y][x] = "."
            path.append((x, y))

        y += 1
        if y > height - 2:
            break

        connect_x = width - 2 if direction == 1 else 1
        grid[y][connect_x] = "."
        path.append((connect_x, y))

        y += 1
        direction *= -1

    return grid, path


def _select_doors(path: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if count <= 0 or len(path) < 3:
        return []

    usable = path[1:-1]
    step = max(1, len(usable) // (count + 1))
    positions: list[tuple[int, int]] = []
    index = step

    while len(positions) < count and index < len(usable):
        positions.append(usable[index])
        index += step

    if len(positions) < count:
        for pos in usable:
            if pos in positions:
                continue
            positions.append(pos)
            if len(positions) >= count:
                break

    return positions


def _build_level(spec: tuple[str, int, int, int]) -> Level:
    name, width, height, doors = spec
    grid, path = _build_snake_maze(width, height)

    start = path[0]
    goal = path[-1]
    grid[start[1]][start[0]] = "P"
    grid[goal[1]][goal[0]] = "G"

    for door_pos in _select_doors(path, doors):
        if door_pos in (start, goal):
            continue
        grid[door_pos[1]][door_pos[0]] = "D"

    return _level_from_grid(name, grid)


def _level_from_grid(name: str, grid: list[list[str]]) -> Level:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if width == 0 or height == 0:
        raise ValueError("Grid must be non-empty.")

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]
    wall_pixels = [[-1] * width for _ in range(height)]

    player_pos: tuple[int, int] | None = None
    goal_pos: tuple[int, int] | None = None
    door_positions: list[tuple[int, int]] = []

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
            elif cell == "D":
                door_positions.append((x, y))

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

    for idx, (x, y) in enumerate(door_positions):
        sprites.append(
            Sprite(pixels=[[DOOR_COLOR]], name=f"door_{idx}", x=x, y=y, collidable=True, layer=4, tags=["door"])
        )

    return Level(
        name=name, sprites=sprites, grid_size=(width, height), data={"goal": goal_pos, "width": width, "height": height}
    )


class Maze(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        self._player: Sprite | None = None
        self._goal_pos: tuple[int, int] | None = None
        super().__init__(
            "maze", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4, 5, 6]
        )

    def on_set_level(self, level: Level) -> None:
        players = level.get_sprites_by_name("player")
        self._player = players[0] if players else None

        goal = level.get_data("goal")
        self._goal_pos = (int(goal[0]), int(goal[1])) if goal is not None else None

    def _open_door_at(self, x: int, y: int) -> None:
        door = self.current_level.get_sprite_at(x, y, tag="door")
        if not door:
            return
        door.set_collidable(False)
        door.set_visible(False)

    def _try_click_door(self) -> None:
        data = self.action.data or {}
        display_x = int(data.get("x", -1))
        display_y = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(display_x, display_y)
        if grid_pos is None:
            return
        self._open_door_at(grid_pos[0], grid_pos[1])

    def _try_move_player(self, dx: int, dy: int) -> None:
        player = self._player
        if player is None:
            return
        self.try_move_sprite(player, dx, dy)
        goal_pos = self._goal_pos
        if goal_pos and player.x == goal_pos[0] and player.y == goal_pos[1]:
            self.next_level()

    def step(self) -> None:
        action = self.action.id

        if action == GameAction.ACTION6:
            self._try_click_door()
        elif action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._try_move_player(dx, dy)

        self.complete_action()
