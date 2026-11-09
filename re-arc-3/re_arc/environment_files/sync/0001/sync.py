from __future__ import annotations

from collections import deque

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 0
WALL_COLOR = 5
ALPHA_COLOR = 9
BETA_COLOR = 14
GOAL_ALPHA_COLOR = 10
GOAL_BETA_COLOR = 11
ACTIVE_COLORS = [6, 12]  # alpha active, beta active

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "active": 0,
        "map": ["#########", "#A....X.#", "#..###..#", "#..#..#.#", "#..###..#", "#.Y....B#", "#########"],
    },
    {
        "name": "Level 2",
        "active": 1,
        "map": [
            "###########",
            "#A...#...X#",
            "#.##.#.##.#",
            "#....#....#",
            "#.##.#.##.#",
            "#Y...#...B#",
            "###########",
        ],
    },
    {
        "name": "Level 3",
        "active": 0,
        "map": [
            "###########",
            "#A....#..X#",
            "#.###.#.#.#",
            "#...#...#.#",
            "###.#.###.#",
            "#Y..#....B#",
            "###########",
        ],
    },
    {
        "name": "Level 4",
        "active": 1,
        "map": [
            "#############",
            "#A....#....X#",
            "#.##.#.#.##.#",
            "#....#.#....#",
            "###.###.###.#",
            "#....#.#....#",
            "#.##.#.#.##.#",
            "#Y....#....B#",
            "#############",
        ],
    },
    {
        "name": "Level 5",
        "active": 0,
        "map": [
            "#############",
            "#A...#....X.#",
            "#.#.#.#.###.#",
            "#.#...#...#.#",
            "#.###.###.#.#",
            "#...#...#...#",
            "#.#.###.#.#.#",
            "#.Y.....#..B#",
            "#############",
        ],
    },
    {
        "name": "Level 6",
        "active": 1,
        "map": [
            "###############",
            "#A....#....#X.#",
            "#.##.#.##.#.#.#",
            "#....#....#...#",
            "###.###.###.###",
            "#...#....#....#",
            "#.#.#.##.#.##.#",
            "#.Y.#....#..B.#",
            "###############",
        ],
    },
]


def _parse_grid(lines: list[str]):
    height = len(lines)
    width = len(lines[0]) if height else 0
    if width == 0 or height == 0:
        raise ValueError("sync map must be non-empty")
    for row in lines:
        if len(row) != width:
            raise ValueError("sync map rows must have equal width")

    walls: set[tuple[int, int]] = set()
    alpha_start = None
    beta_start = None
    alpha_goal = None
    beta_goal = None

    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "A":
                alpha_start = (x, y)
            elif ch == "B":
                beta_start = (x, y)
            elif ch == "X":
                alpha_goal = (x, y)
            elif ch == "Y":
                beta_goal = (x, y)

    if alpha_start is None or beta_start is None or alpha_goal is None or beta_goal is None:
        raise ValueError("sync map requires A,B,X,Y")

    return width, height, walls, alpha_start, beta_start, alpha_goal, beta_goal


def _validate_solvable(level_data: dict) -> None:
    width = int(level_data["width"])
    height = int(level_data["height"])
    walls = {tuple(int(v) for v in item) for item in (level_data["walls"] or [])}
    alpha_start = tuple(int(v) for v in level_data["alpha_start"])
    beta_start = tuple(int(v) for v in level_data["beta_start"])
    alpha_goal = tuple(int(v) for v in level_data["alpha_goal"])
    beta_goal = tuple(int(v) for v in level_data["beta_goal"])
    active = int(level_data["start_active"]) % 2

    start_state = (alpha_start[0], alpha_start[1], beta_start[0], beta_start[1], active)
    queue = deque([start_state])
    seen = {start_state}

    while queue:
        ax, ay, bx, by, active_agent = queue.popleft()
        if (ax, ay) == alpha_goal and (bx, by) == beta_goal:
            return

        toggled = (ax, ay, bx, by, active_agent ^ 1)
        if toggled not in seen:
            seen.add(toggled)
            queue.append(toggled)

        for action_id in (1, 2, 3, 4):
            dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action_id - 1]

            if active_agent == 0:
                nax, nay = ax + dx, ay + dy
                nbx, nby = bx, by
                if nax < 0 or nay < 0 or nax >= width or nay >= height or (nax, nay) in walls:
                    nax, nay = ax, ay
                if (nax, nay) == (nbx, nby):
                    nax, nay = ax, ay
            else:
                nax, nay = ax, ay
                nbx, nby = bx + dx, by + dy
                if nbx < 0 or nby < 0 or nbx >= width or nby >= height or (nbx, nby) in walls:
                    nbx, nby = bx, by
                if (nbx, nby) == (nax, nay):
                    nbx, nby = bx, by

            nxt = (nax, nay, nbx, nby, active_agent)
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)

    raise ValueError(f"sync level `{level_data['name']}` is unsolvable")


def _build_level(spec: dict) -> Level:
    name = str(spec.get("name", "Level"))
    lines = list(spec.get("map") or [])
    start_active = int(spec.get("active", 0)) % 2

    width, height, walls, a_start, b_start, a_goal, b_goal = _parse_grid(lines)
    candidate_walls = set(walls)
    while True:
        level_data = {
            "name": name,
            "width": width,
            "height": height,
            "walls": sorted((int(x), int(y)) for x, y in candidate_walls),
            "alpha_start": (int(a_start[0]), int(a_start[1])),
            "beta_start": (int(b_start[0]), int(b_start[1])),
            "alpha_goal": (int(a_goal[0]), int(a_goal[1])),
            "beta_goal": (int(b_goal[0]), int(b_goal[1])),
            "start_active": int(start_active),
        }
        try:
            _validate_solvable(level_data)
            break
        except ValueError:
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

    sprites = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=1, tags=["wall"]),
        Sprite(
            pixels=[[GOAL_ALPHA_COLOR]],
            name="goal_alpha",
            x=a_goal[0],
            y=a_goal[1],
            collidable=False,
            layer=2,
            tags=["goal_alpha"],
        ),
        Sprite(
            pixels=[[GOAL_BETA_COLOR]],
            name="goal_beta",
            x=b_goal[0],
            y=b_goal[1],
            collidable=False,
            layer=2,
            tags=["goal_beta"],
        ),
        Sprite(
            pixels=[[ALPHA_COLOR]], name="alpha", x=a_start[0], y=a_start[1], collidable=False, layer=6, tags=["alpha"]
        ),
        Sprite(
            pixels=[[BETA_COLOR]], name="beta", x=b_start[0], y=b_start[1], collidable=False, layer=6, tags=["beta"]
        ),
        Sprite(
            pixels=[[ACTIVE_COLORS[start_active]]],
            name="active_indicator",
            x=0,
            y=0,
            collidable=False,
            layer=8,
            tags=["active_indicator"],
        ),
    ]

    return Level(name=name, sprites=sprites, grid_size=(width, height), data=level_data)


class Sync(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__(
            "sync", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4, 5]
        )

    def on_set_level(self, level: Level) -> None:
        self._width = int(level.get_data("width"))
        self._height = int(level.get_data("height"))
        self._walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        self._alpha_goal = tuple(int(v) for v in level.get_data("alpha_goal"))
        self._beta_goal = tuple(int(v) for v in level.get_data("beta_goal"))
        self._active = int(level.get_data("start_active") or 0) % 2

        self._alpha = self.current_level.get_sprites_by_name("alpha")[0]
        self._beta = self.current_level.get_sprites_by_name("beta")[0]
        self._indicator = self.current_level.get_sprites_by_name("active_indicator")[0]
        self._sync_indicator()

    def _sync_indicator(self) -> None:
        self._indicator.pixels[0][0] = ACTIVE_COLORS[self._active]

    def _blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._width or y >= self._height:
            return True
        return (x, y) in self._walls

    def _try_move_active(self, dx: int, dy: int) -> None:
        if self._active == 0:
            mover = self._alpha
            other = self._beta
        else:
            mover = self._beta
            other = self._alpha

        nx = int(mover.x + dx)
        ny = int(mover.y + dy)
        if self._blocked(nx, ny):
            return
        if (nx, ny) == (int(other.x), int(other.y)):
            return

        mover.set_position(nx, ny)
        if (int(self._alpha.x), int(self._alpha.y)) == self._alpha_goal and (
            int(self._beta.x),
            int(self._beta.y),
        ) == self._beta_goal:
            self.next_level()

    def step(self) -> None:
        action = self.action.id
        if action == GameAction.ACTION5:
            self._active ^= 1
            self._sync_indicator()
        elif action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._try_move_active(dx, dy)
        self.complete_action()
