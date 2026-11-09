from __future__ import annotations

from collections import deque

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "pathfinder-0001"

COLOR_FLOOR = 0
COLOR_WALL = 5
COLOR_GOAL = 8
COLOR_PLAYER = 14

ACTION_UP = int(GameAction.ACTION1.value)
ACTION_DOWN = int(GameAction.ACTION2.value)
ACTION_LEFT = int(GameAction.ACTION3.value)
ACTION_RIGHT = int(GameAction.ACTION4.value)

MOVE_BY_ACTION = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

# Keep maze sizes aligned to factors of 64 so the fixed 64x64 renderer never needs letterbox padding.
LEVEL_BLUEPRINTS = (
    ("####", "#S.#", "##G#", "####"),
    ("########", "#S....##", "#####.##", "#...#.##", "#.#.#.##", "#G#...##", "########", "########"),
    (
        "################",
        "#S......#.....##",
        "#######.#.###.##",
        "#.....#.#.#G..##",
        "#.#####.#.######",
        "#.#.....#.....##",
        "#.#.#####.###.##",
        "#...#...#.#.#.##",
        "#.#####.#.#.#.##",
        "#.....#.....#.##",
        "#####.#######.##",
        "#...#...#.....##",
        "#.#.###.#.###.##",
        "#.#.......#...##",
        "################",
        "################",
    ),
    (
        "################",
        "#S....#.......##",
        "#####.###.###.##",
        "#...#.....#.#.##",
        "#.#.#######.#.##",
        "#.#.....#.....##",
        "#.#####.#.######",
        "#.....#.#.#...##",
        "#####.#.#.#.#.##",
        "#.....#...#G#.##",
        "#.###########.##",
        "#.#.......#...##",
        "#.###.###.#.#.##",
        "#.....#.....#.##",
        "################",
        "################",
    ),
    (
        "################################",
        "#S..#.......#.........#.......##",
        "###.###.###.#.#######.#.#####.##",
        "#.#.....#...#.#.....#...#...#.##",
        "#.#######.###.#####.#####.#.#.##",
        "#...#...#.#.#...#.......#.#...##",
        "#.###.#.#.#.###.#.###.###.###.##",
        "#.#...#...#...#...#G..#...#.#.##",
        "#.#.#######.#.#.#######.###.#.##",
        "#...#.....#.#...#...#...#.....##",
        "#.###.###.#######.#.#.###.######",
        "#.#...#.#.......#.#.#.#...#...##",
        "#.#.###.#######.#.#.#.#####.#.##",
        "#.#.#.....#...#...#.#...#...#.##",
        "#.#.#.#.###.#.#####.#.#.#.###.##",
        "#.#.#.#...#.#.....#.#.#.#.#...##",
        "#.#.#.###.#.#####.#.###.#.#.#.##",
        "#.#.#...#...#...#...#...#.#.#.##",
        "#.#.#####.#####.#####.###.#.####",
        "#.#.....#.......#.#.......#...##",
        "#.#####.#######.#.#.#########.##",
        "#.....#.....#...#.#.#.......#.##",
        "#####.#####.#.###.#.#.#.###.#.##",
        "#.#...#.....#.#...#.#.#.#...#.##",
        "#.#.###.#####.#.###.###.#.###.##",
        "#.#...#...#...#.........#.#.#.##",
        "#.###.###.#.#############.#.#.##",
        "#...#.....#...#.......#.....#.##",
        "#.#.#########.#.#####.#######.##",
        "#.#.............#.............##",
        "################################",
        "################################",
    ),
)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _action_id(action_obj: object) -> int:
    return int(getattr(action_obj, "value", action_obj))


def _neighbors(model: dict[str, object], state: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = int(state[0]), int(state[1])
    width = int(model["width"])
    height = int(model["height"])
    walls = set(tuple(item) for item in model["walls"])

    out: list[tuple[int, int]] = []
    for dx, dy in MOVE_BY_ACTION.values():
        nx = x + dx
        ny = y + dy
        if nx < 0 or ny < 0 or nx >= width or ny >= height:
            continue
        if (nx, ny) in walls:
            continue
        out.append((nx, ny))
    return out


def _shortest_distance(model: dict[str, object], start: tuple[int, int], goal: tuple[int, int]) -> int:
    queue: deque[tuple[int, int]] = deque([start])
    distances = {start: 0}

    while queue:
        current = queue.popleft()
        if current == goal:
            return int(distances[current])
        for nxt in _neighbors(model, current):
            if nxt in distances:
                continue
            distances[nxt] = int(distances[current]) + 1
            queue.append(nxt)

    raise ValueError("Maze blueprint is unsolvable.")


def _parse_blueprint(rows: tuple[str, ...], name: str) -> dict[str, object]:
    if not rows:
        raise ValueError(f"{name}: blueprint cannot be empty.")

    width = len(rows[0])
    height = len(rows)
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name}: rows must have consistent width.")

    walls: list[tuple[int, int]] = []
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None

    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell == "#":
                walls.append((x, y))
            elif cell == "S":
                if start is not None:
                    raise ValueError(f"{name}: multiple starts.")
                start = (x, y)
            elif cell == "G":
                if goal is not None:
                    raise ValueError(f"{name}: multiple goals.")
                goal = (x, y)
            elif cell != ".":
                raise ValueError(f"{name}: unsupported cell `{cell}`.")

    if start is None or goal is None:
        raise ValueError(f"{name}: expected exactly one start and one goal.")

    model = {
        "name": name,
        "width": width,
        "height": height,
        "walls": sorted((int(x), int(y)) for x, y in walls),
        "start": [int(start[0]), int(start[1])],
        "goal": [int(goal[0]), int(goal[1])],
    }
    model["shortest_steps"] = _shortest_distance(model, start, goal)
    return model


LEVEL_MODELS = tuple(_parse_blueprint(rows, name=f"Level {index + 1}") for index, rows in enumerate(LEVEL_BLUEPRINTS))


def _deserialize_model(raw: dict[str, object]) -> dict[str, object]:
    walls = [tuple(int(v) for v in item) for item in (raw.get("walls") or [])]
    start = tuple(int(v) for v in (raw.get("start") or (0, 0)))
    goal = tuple(int(v) for v in (raw.get("goal") or (0, 0)))
    return {
        "name": str(raw.get("name") or ""),
        "width": int(raw.get("width") or 0),
        "height": int(raw.get("height") or 0),
        "walls": walls,
        "start": start,
        "goal": goal,
        "shortest_steps": int(raw.get("shortest_steps") or 0),
    }


def current_goal_from_model(model: dict[str, object]) -> tuple[int, int]:
    goal = tuple(int(v) for v in model["goal"])
    return goal[0], goal[1]


def initial_search_state_from_model(model: dict[str, object]) -> tuple[int, int]:
    start = tuple(int(v) for v in model["start"])
    return int(start[0]), int(start[1])


def apply_action_transition(
    model: dict[str, object], state: tuple[int, int], action_id: int
) -> tuple[tuple[int, int] | None, bool]:
    normalized_action = _action_id(action_id)
    if normalized_action not in MOVE_BY_ACTION:
        return None, False

    x, y = int(state[0]), int(state[1])
    dx, dy = MOVE_BY_ACTION[normalized_action]
    nx = x + dx
    ny = y + dy
    width = int(model["width"])
    height = int(model["height"])
    walls = set(tuple(item) for item in model["walls"])

    if nx < 0 or ny < 0 or nx >= width or ny >= height:
        return None, False
    if (nx, ny) in walls:
        return None, False

    next_state = (int(nx), int(ny))
    return next_state, next_state == current_goal_from_model(model)


def _build_level(model: dict[str, object], level_index: int) -> Level:
    width = int(model["width"])
    height = int(model["height"])

    floor = Sprite(
        pixels=_solid(width, height, COLOR_FLOOR), name="floor", x=0, y=0, layer=0, tags=["floor"], collidable=False
    )
    walls = Sprite(
        pixels=np.full((height, width), -1, dtype=np.int8),
        name="walls",
        x=0,
        y=0,
        layer=1,
        tags=["wall"],
        collidable=True,
    )
    goal = Sprite(
        pixels=np.array([[COLOR_GOAL]], dtype=np.int8),
        name="goal",
        x=int(model["goal"][0]),
        y=int(model["goal"][1]),
        layer=2,
        tags=["goal"],
        collidable=False,
    )
    player = Sprite(
        pixels=np.array([[COLOR_PLAYER]], dtype=np.int8),
        name="player",
        x=int(model["start"][0]),
        y=int(model["start"][1]),
        layer=3,
        tags=["player"],
        collidable=True,
    )
    return Level(
        name=str(model["name"]),
        grid_size=(width, height),
        sprites=[floor, walls, goal, player],
        data={"level_index": int(level_index), "model": model},
    )


class Pathfinder(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(model, index) for index, model in enumerate(LEVEL_MODELS)]
        first_width = int(LEVEL_MODELS[0]["width"])
        first_height = int(LEVEL_MODELS[0]["height"])
        camera = Camera(x=0, y=0, width=first_width, height=first_height, background=COLOR_WALL, letter_box=COLOR_WALL)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT],
            seed=seed,
        )

        self._model: dict[str, object] | None = None
        self._route_score = 0
        self._player: Sprite | None = None
        self._goal_sprite: Sprite | None = None
        self._walls_sprite: Sprite | None = None

    def current_level_model(self) -> dict[str, object]:
        if self._model is None:
            raise RuntimeError("Pathfinder level model is not initialized.")
        return _deserialize_model(self._model)

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level.get_data("model"))
        width = int(self._model["width"])
        height = int(self._model["height"])
        walls = set(tuple(item) for item in self._model["walls"])
        start = tuple(int(v) for v in self._model["start"])
        goal = tuple(int(v) for v in self._model["goal"])

        self.camera.width = width
        self.camera.height = height
        self.camera.x = 0
        self.camera.y = 0
        self.camera.background = COLOR_WALL
        self.camera.letter_box = COLOR_WALL

        floors = level.get_sprites_by_name("floor")
        goals = level.get_sprites_by_name("goal")
        players = level.get_sprites_by_name("player")
        walls_sprites = level.get_sprites_by_name("walls")

        self._walls_sprite = walls_sprites[0] if walls_sprites else None
        self._goal_sprite = goals[0] if goals else None
        self._player = players[0] if players else None

        if floors:
            floors[0].pixels = _solid(width, height, COLOR_FLOOR)

        if self._walls_sprite is None or self._goal_sprite is None or self._player is None:
            raise RuntimeError("Pathfinder level is missing required sprites.")

        wall_pixels = np.full((height, width), -1, dtype=np.int8)
        for x, y in walls:
            wall_pixels[int(y), int(x)] = COLOR_WALL
        self._walls_sprite.pixels = wall_pixels

        self._goal_sprite.set_position(int(goal[0]), int(goal[1]))
        self._goal_sprite.pixels = np.array([[COLOR_GOAL]], dtype=np.int8)

        self._player.set_position(int(start[0]), int(start[1]))
        self._player.pixels = np.array([[COLOR_PLAYER]], dtype=np.int8)

    def _try_move_player(self, action_id: int) -> None:
        if self._player is None or self._model is None:
            return

        next_state, won = apply_action_transition(
            self._model, (int(self._player.x), int(self._player.y)), int(action_id)
        )
        if next_state is None:
            return

        self._player.set_position(int(next_state[0]), int(next_state[1]))
        if won:
            self._route_score += 1
            self.next_level()

    def step(self) -> None:
        action_id = _action_id(self.action.id)
        if action_id in MOVE_BY_ACTION:
            self._try_move_player(action_id)
        self.complete_action()
