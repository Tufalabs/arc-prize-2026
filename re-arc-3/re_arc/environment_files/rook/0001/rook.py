from __future__ import annotations

from collections import deque

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 1
WALL_COLOR = 5
PLAYER_COLOR = 9
GOAL_COLOR = 11
LOCK_COLOR = 12
BEACON_COLOR = 14

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "map": ["#########", "#S..#..G#", "#...#...#", "#..B....#", "#...#...#", "#.......#", "#########"],
    },
    {
        "name": "Level 2",
        "map": [
            "###########",
            "#S...#...G#",
            "#.#..#..#.#",
            "#...B...#.#",
            "#.#..#..#.#",
            "#...B.....#",
            "###########",
        ],
    },
    {
        "name": "Level 3",
        "map": [
            "###########",
            "#S....#..G#",
            "#.#.#.#.#.#",
            "#..B...B..#",
            "###.#.#.###",
            "#.........#",
            "###########",
        ],
    },
    {
        "name": "Level 4",
        "map": [
            "#############",
            "#S..#....#G.#",
            "#.#.#.##.#..#",
            "#...#B...#..#",
            "###.###.#.###",
            "#...B...#...#",
            "#.#.###.###.#",
            "#.....B.....#",
            "#############",
        ],
    },
    {
        "name": "Level 5",
        "map": [
            "#############",
            "#S....#...G.#",
            "#.#.#.#.#.#.#",
            "#..B...#...B#",
            "###.###.###.#",
            "#...#...#...#",
            "#.#.#.###.#.#",
            "#B..........#",
            "#############",
        ],
    },
    {
        "name": "Level 6",
        "map": [
            "###############",
            "#S...#....#..G#",
            "#.#.#.##.#.#..#",
            "#..B...#...#..#",
            "###.###.#.###.#",
            "#...#..B#...#.#",
            "#.#.#.###.#.#.#",
            "#...B.....#...#",
            "###############",
        ],
    },
    {
        "name": "Level 7",
        "map": [
            "###############",
            "#S..#....#...G#",
            "#.#.#.##.#.#.##",
            "#..B...#...#..#",
            "###.###.#.###.#",
            "#...#..B#...#.#",
            "#.#.#.###.#.#.#",
            "#..B....B.#...#",
            "###############",
        ],
    },
]


def _parse_grid(lines: list[str]):
    height = len(lines)
    width = len(lines[0]) if height else 0
    if width == 0 or height == 0:
        raise ValueError("rook map must be non-empty")
    for row in lines:
        if len(row) != width:
            raise ValueError("rook map rows must have equal width")

    walls: set[tuple[int, int]] = set()
    start = None
    goal = None
    beacons: list[tuple[int, int]] = []

    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "S":
                start = (x, y)
            elif ch == "G":
                goal = (x, y)
            elif ch == "B":
                beacons.append((x, y))

    if start is None or goal is None:
        raise ValueError("rook map requires S and G")

    return width, height, walls, start, goal, beacons


def _validate_solvable(level_data: dict) -> None:
    width = int(level_data["width"])
    height = int(level_data["height"])
    walls = {tuple(int(v) for v in item) for item in (level_data["walls"] or [])}
    start = tuple(int(v) for v in level_data["start"])
    goal = tuple(int(v) for v in level_data["goal"])
    beacons = [tuple(int(v) for v in item) for item in (level_data["beacons"] or [])]
    beacon_index = {pos: idx for idx, pos in enumerate(beacons)}
    target_mask = (1 << len(beacons)) - 1

    start_mask = 0
    bit = beacon_index.get(start)
    if bit is not None:
        start_mask |= 1 << bit

    start_state = (start[0], start[1], start_mask)
    queue = deque([start_state])
    seen = {start_state}

    while queue:
        x, y, mask = queue.popleft()
        if (x, y) == goal and mask == target_mask:
            return

        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x, y
            nmask = int(mask)
            moved = False
            won = False
            while True:
                tx = nx + dx
                ty = ny + dy
                if tx < 0 or ty < 0 or tx >= width or ty >= height or (tx, ty) in walls:
                    break
                nx, ny = tx, ty
                moved = True
                bit = beacon_index.get((nx, ny))
                if bit is not None:
                    nmask |= 1 << bit
                if (nx, ny) == goal and nmask == target_mask:
                    won = True
                    break
            if won:
                return
            if not moved:
                continue
            nxt = (nx, ny, nmask)
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)

    raise ValueError(f"rook level `{level_data['name']}` is unsolvable")


def _build_level(spec: dict) -> Level:
    name = str(spec.get("name", "Level"))
    lines = list(spec.get("map") or [])

    width, height, walls, start, goal, beacons = _parse_grid(lines)
    candidate_walls = set(walls)
    candidate_beacons = list(beacons)
    while True:
        level_data = {
            "name": name,
            "width": width,
            "height": height,
            "walls": sorted((int(x), int(y)) for x, y in candidate_walls),
            "start": (int(start[0]), int(start[1])),
            "goal": (int(goal[0]), int(goal[1])),
            "beacons": [(int(x), int(y)) for x, y in candidate_beacons],
        }
        try:
            _validate_solvable(level_data)
            break
        except ValueError:
            if candidate_beacons:
                candidate_beacons = candidate_beacons[:-1]
                continue
            interior = [
                cell for cell in sorted(candidate_walls) if 0 < cell[0] < width - 1 and 0 < cell[1] < height - 1
            ]
            if not interior:
                raise
            remove = set(interior[::2])
            candidate_walls = {cell for cell in candidate_walls if cell not in remove}

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]
    wall_pixels = [[-1] * width for _ in range(height)]
    for x, y in candidate_walls:
        wall_pixels[y][x] = WALL_COLOR

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=1, tags=["wall"]),
        Sprite(pixels=[[GOAL_COLOR]], name="goal", x=goal[0], y=goal[1], collidable=False, layer=2, tags=["goal"]),
        Sprite(
            pixels=[[LOCK_COLOR]], name="goal_lock", x=goal[0], y=goal[1], collidable=True, layer=5, tags=["goal_lock"]
        ),
        Sprite(
            pixels=[[PLAYER_COLOR]], name="player", x=start[0], y=start[1], collidable=False, layer=6, tags=["player"]
        ),
    ]

    for idx, (bx, by) in enumerate(candidate_beacons):
        sprites.append(
            Sprite(
                pixels=[[BEACON_COLOR]], name=f"beacon_{idx}", x=bx, y=by, collidable=False, layer=4, tags=["beacon"]
            )
        )

    return Level(name=name, sprites=sprites, grid_size=(width, height), data=level_data)


class Rook(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__(
            "rook", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4]
        )

    def on_set_level(self, level: Level) -> None:
        self._width = int(level.get_data("width"))
        self._height = int(level.get_data("height"))
        self._walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        self._goal = tuple(int(v) for v in level.get_data("goal"))
        self._beacons = [tuple(int(v) for v in item) for item in (level.get_data("beacons") or [])]
        self._beacon_index = {pos: idx for idx, pos in enumerate(self._beacons)}
        self._target_mask = (1 << len(self._beacons)) - 1
        self._mask = 0

        self._player = self.current_level.get_sprites_by_name("player")[0]
        self._goal_lock = self.current_level.get_sprites_by_name("goal_lock")[0]

        self._collect_at(int(self._player.x), int(self._player.y))
        self._sync_lock()

    def _blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._width or y >= self._height:
            return True
        if (x, y) in self._walls:
            return True
        return bool((x, y) == self._goal and self._mask != self._target_mask)

    def _collect_at(self, x: int, y: int) -> None:
        bit = self._beacon_index.get((x, y))
        if bit is None:
            return
        if ((self._mask >> bit) & 1) == 1:
            return
        self._mask |= 1 << bit
        sprite = self.current_level.get_sprites_by_name(f"beacon_{bit}")
        if sprite:
            self.current_level.remove_sprite(sprite[0])

    def _sync_lock(self) -> None:
        unlocked = self._mask == self._target_mask
        self._goal_lock.set_collidable(not unlocked)
        self._goal_lock.set_visible(not unlocked)

    def _slide_player(self, dx: int, dy: int) -> None:
        moved = False
        while True:
            nx = int(self._player.x + dx)
            ny = int(self._player.y + dy)
            if self._blocked(nx, ny):
                break
            self._player.set_position(nx, ny)
            moved = True
            self._collect_at(nx, ny)
            self._sync_lock()
            if (nx, ny) == self._goal and self._mask == self._target_mask:
                self.next_level()
                return

        if moved and (int(self._player.x), int(self._player.y)) == self._goal and self._mask == self._target_mask:
            self.next_level()

    def step(self) -> None:
        action = self.action.id
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._slide_player(dx, dy)
        self.complete_action()
