from __future__ import annotations

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 0
WALL_COLOR = 5
PLAYER_COLOR = 9
GOAL_COLOR = 11
PHASE_COLORS = [6, 8, 10, 12]

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

LEVEL_SPECS: list[tuple[str, int, int, int]] = [
    ("Level 1", 9, 9, 3),
    ("Level 2", 11, 9, 4),
    ("Level 3", 11, 11, 5),
    ("Level 4", 13, 11, 6),
    ("Level 5", 13, 13, 7),
    ("Level 6", 15, 13, 8),
]


def _build_snake_path(width: int, height: int) -> list[tuple[int, int]]:
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


def _select_gates(path: list[tuple[int, int]], count: int, seed: int):
    usable = path[3:-3]
    if count <= 0 or not usable:
        return []

    step = max(1, len(usable) // (count + 1))
    picks: list[tuple[int, int]] = []
    idx = step
    while len(picks) < count and idx < len(usable):
        picks.append(usable[idx])
        idx += step
    if len(picks) < count:
        for pos in usable:
            if pos in picks:
                continue
            picks.append(pos)
            if len(picks) >= count:
                break

    out: list[tuple[int, int, int]] = []
    for index, (gx, gy) in enumerate(picks[:count]):
        required_phase = (seed + index + gx + gy) % 4
        out.append((gx, gy, required_phase))
    return out


def _build_level(spec: tuple[str, int, int, int], seed: int) -> Level:
    name, width, height, gate_count = spec
    path = _build_snake_path(width, height)
    path_set = set(path)

    start = path[0]
    goal = path[-1]

    gates = _select_gates(path, gate_count, seed)
    {(x, y): phase for x, y, phase in gates if (x, y) not in (start, goal)}

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]
    wall_pixels = [[-1] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                wall_pixels[y][x] = WALL_COLOR
                continue
            if (x, y) not in path_set:
                wall_pixels[y][x] = WALL_COLOR

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=1, tags=["wall"]),
        Sprite(pixels=[[GOAL_COLOR]], name="goal", x=goal[0], y=goal[1], collidable=False, layer=3, tags=["goal"]),
        Sprite(
            pixels=[[PLAYER_COLOR]], name="player", x=start[0], y=start[1], collidable=False, layer=6, tags=["player"]
        ),
    ]

    for idx, (gx, gy, phase) in enumerate(gates):
        sprites.append(
            Sprite(
                pixels=[[PHASE_COLORS[phase]]], name=f"gate_{idx}", x=gx, y=gy, collidable=False, layer=4, tags=["gate"]
            )
        )

    sprites.append(
        Sprite(
            pixels=[[PHASE_COLORS[0]]],
            name="phase_indicator",
            x=0,
            y=0,
            collidable=False,
            layer=8,
            tags=["phase_indicator"],
        )
    )

    return Level(
        name=name,
        sprites=sprites,
        grid_size=(width, height),
        data={
            "width": width,
            "height": height,
            "walls": [(x, y) for y in range(height) for x in range(width) if wall_pixels[y][x] != -1],
            "start": start,
            "goal": goal,
            "start_phase": 0,
            "gate_phases": [{"x": x, "y": y, "phase": phase} for x, y, phase in gates],
        },
    )


class Turn(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec, seed=idx + 31) for idx, spec in enumerate(LEVEL_SPECS)]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__(
            "turn", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4]
        )

    def on_set_level(self, level: Level) -> None:
        self._width = int(level.get_data("width"))
        self._height = int(level.get_data("height"))
        self._walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        self._goal = tuple(int(v) for v in level.get_data("goal"))
        self._phase = int(level.get_data("start_phase") or 0) % 4

        self._gate_phase: dict[tuple[int, int], int] = {}
        for entry in level.get_data("gate_phases") or []:
            gx = int(entry["x"])
            gy = int(entry["y"])
            phase = int(entry["phase"]) % 4
            self._gate_phase[(gx, gy)] = phase

        self._player = self.current_level.get_sprites_by_name("player")[0]
        self._indicator = self.current_level.get_sprites_by_name("phase_indicator")[0]
        self._update_indicator()

    def _update_indicator(self) -> None:
        self._indicator.pixels[0][0] = PHASE_COLORS[self._phase]

    def _blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._width or y >= self._height:
            return True
        if (x, y) in self._walls:
            return True
        required = self._gate_phase.get((x, y))
        return bool(required is not None and required != self._phase)

    def _advance_phase(self) -> None:
        self._phase = (self._phase + 1) % 4
        self._update_indicator()

    def _try_move(self, dx: int, dy: int) -> None:
        nx = int(self._player.x + dx)
        ny = int(self._player.y + dy)
        if not self._blocked(nx, ny):
            self._player.set_position(nx, ny)
            if (nx, ny) == self._goal:
                self.next_level()
                return
        self._advance_phase()

    def step(self) -> None:
        action = self.action.id
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._try_move(dx, dy)
        self.complete_action()
