from __future__ import annotations

import random
from collections import deque

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 1
WALL_COLOR = 4
PILOT_COLOR = 9
ORB_COLOR = 14
PILOT_GOAL_COLOR = 11
ORB_GOAL_COLOR = 12

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

LEVEL_SPECS: list[tuple[str, int, int, int]] = [
    ("Level 1", 9, 9, 16),
    ("Level 2", 11, 9, 20),
    ("Level 3", 11, 11, 24),
    ("Level 4", 13, 11, 28),
    ("Level 5", 13, 13, 32),
    ("Level 6", 15, 13, 36),
    ("Level 7", 15, 15, 40),
    ("Level 8", 17, 15, 44),
]


def _neighbors(state: tuple[int, int, int, int], walls: set[tuple[int, int]], width: int, height: int):
    px, py, ox, oy = state

    def blocked(x: int, y: int):
        if x < 0 or y < 0 or x >= width or y >= height:
            return True
        return (x, y) in walls

    for action_id, (dx, dy) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))):
        npx, npy = px + dx, py + dy
        if blocked(npx, npy) or (npx, npy) == (ox, oy):
            npx, npy = px, py

        nox, noy = ox - dx, oy - dy
        if blocked(nox, noy) or (nox, noy) == (npx, npy):
            nox, noy = ox, oy

        nxt = (npx, npy, nox, noy)
        if nxt != state:
            yield action_id, nxt


def _find_solution(
    *,
    walls: set[tuple[int, int]],
    width: int,
    height: int,
    pilot_start: tuple[int, int],
    orb_start: tuple[int, int],
    pilot_goal: tuple[int, int],
    orb_goal: tuple[int, int],
):
    start = (pilot_start[0], pilot_start[1], orb_start[0], orb_start[1])
    goal = (pilot_goal[0], pilot_goal[1], orb_goal[0], orb_goal[1])

    queue = deque([start])
    previous: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None] = {start: None}
    previous_action: dict[tuple[int, int, int, int], int] = {}

    while queue:
        state = queue.popleft()
        if state == goal:
            break
        for action_id, nxt in _neighbors(state, walls, width, height):
            if nxt in previous:
                continue
            previous[nxt] = state
            previous_action[nxt] = action_id
            queue.append(nxt)

    if goal not in previous:
        return None

    actions: list[int] = []
    cursor = goal
    while previous[cursor] is not None:
        actions.append(previous_action[cursor])
        cursor = previous[cursor]  # type: ignore[index]
    actions.reverse()
    return actions


def _build_walls(
    *,
    width: int,
    height: int,
    seed: int,
    target_count: int,
    pilot_start: tuple[int, int],
    orb_start: tuple[int, int],
    pilot_goal: tuple[int, int],
    orb_goal: tuple[int, int],
):
    border = set()
    for x in range(width):
        border.add((x, 0))
        border.add((x, height - 1))
    for y in range(height):
        border.add((0, y))
        border.add((width - 1, y))

    protected = {pilot_start, orb_start, pilot_goal, orb_goal}
    for x, y in list(protected):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                protected.add((nx, ny))

    rng = random.Random(seed)
    interior = [(x, y) for y in range(1, height - 1) for x in range(1, width - 1) if (x, y) not in protected]
    rng.shuffle(interior)

    walls = set(border)
    for x, y in interior:
        if len(walls) - len(border) >= target_count:
            break
        mx, my = width - 1 - x, height - 1 - y
        if (x, y) in protected or (mx, my) in protected:
            continue
        walls.add((x, y))
        walls.add((mx, my))

    def solvable(test_walls: set[tuple[int, int]]):
        return (
            _find_solution(
                walls=test_walls,
                width=width,
                height=height,
                pilot_start=pilot_start,
                orb_start=orb_start,
                pilot_goal=pilot_goal,
                orb_goal=orb_goal,
            )
            is not None
        )

    if solvable(walls):
        return walls

    interior_walls = sorted([cell for cell in walls if cell not in border])
    keep = interior_walls
    while keep:
        keep = keep[::2]
        candidate = set(border)
        candidate.update(keep)
        if solvable(candidate):
            return candidate

    fallback = set(border)
    if solvable(fallback):
        return fallback

    raise RuntimeError("orbx level generator could not build a solvable level.")


def _build_level(spec: tuple[str, int, int, int], seed: int) -> Level:
    name, width, height, walls_target = spec
    pilot_start = (1, 1)
    orb_start = (width - 2, height - 2)
    pilot_goal = (width - 2, 1)
    orb_goal = (1, height - 2)

    walls = _build_walls(
        width=width,
        height=height,
        seed=seed,
        target_count=walls_target,
        pilot_start=pilot_start,
        orb_start=orb_start,
        pilot_goal=pilot_goal,
        orb_goal=orb_goal,
    )

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]
    wall_pixels = [[-1] * width for _ in range(height)]
    for x, y in walls:
        wall_pixels[y][x] = WALL_COLOR

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=1, tags=["wall"]),
        Sprite(
            pixels=[[PILOT_GOAL_COLOR]],
            name="pilot_goal",
            x=pilot_goal[0],
            y=pilot_goal[1],
            collidable=False,
            layer=2,
            tags=["pilot_goal"],
        ),
        Sprite(
            pixels=[[ORB_GOAL_COLOR]],
            name="orb_goal",
            x=orb_goal[0],
            y=orb_goal[1],
            collidable=False,
            layer=2,
            tags=["orb_goal"],
        ),
        Sprite(
            pixels=[[ORB_COLOR]], name="orb", x=orb_start[0], y=orb_start[1], collidable=False, layer=4, tags=["orb"]
        ),
        Sprite(
            pixels=[[PILOT_COLOR]],
            name="pilot",
            x=pilot_start[0],
            y=pilot_start[1],
            collidable=False,
            layer=5,
            tags=["pilot"],
        ),
    ]

    return Level(
        name=name,
        sprites=sprites,
        grid_size=(width, height),
        data={
            "width": width,
            "height": height,
            "walls": sorted((int(x), int(y)) for x, y in walls),
            "pilot_start": pilot_start,
            "orb_start": orb_start,
            "pilot_goal": pilot_goal,
            "orb_goal": orb_goal,
        },
    )


class Orbx(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec, seed=idx + 17) for idx, spec in enumerate(LEVEL_SPECS)]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__(
            "orbx", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4]
        )

    def on_set_level(self, level: Level) -> None:
        self._width = int(level.get_data("width"))
        self._height = int(level.get_data("height"))
        self._walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}

        self._pilot_goal = tuple(int(v) for v in level.get_data("pilot_goal"))
        self._orb_goal = tuple(int(v) for v in level.get_data("orb_goal"))

        self._pilot_sprite = self.current_level.get_sprites_by_name("pilot")[0]
        self._orb_sprite = self.current_level.get_sprites_by_name("orb")[0]

        self._pilot_cell = (int(self._pilot_sprite.x), int(self._pilot_sprite.y))
        self._orb_cell = (int(self._orb_sprite.x), int(self._orb_sprite.y))

    def _blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._width or y >= self._height:
            return True
        return (x, y) in self._walls

    def _apply_move(self, dx: int, dy: int) -> None:
        px, py = self._pilot_cell
        ox, oy = self._orb_cell

        npx, npy = px + dx, py + dy
        if self._blocked(npx, npy) or (npx, npy) == (ox, oy):
            npx, npy = px, py

        nox, noy = ox - dx, oy - dy
        if self._blocked(nox, noy) or (nox, noy) == (npx, npy):
            nox, noy = ox, oy

        self._pilot_cell = (int(npx), int(npy))
        self._orb_cell = (int(nox), int(noy))

        self._pilot_sprite.set_position(self._pilot_cell[0], self._pilot_cell[1])
        self._orb_sprite.set_position(self._orb_cell[0], self._orb_cell[1])

        if self._pilot_cell == self._pilot_goal and self._orb_cell == self._orb_goal:
            self.next_level()

    def step(self) -> None:
        action = self.action.id
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._apply_move(dx, dy)
        self.complete_action()
