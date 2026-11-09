from __future__ import annotations

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 1
WALL_COLOR = 4
PLAYER_COLORS = [9, 14]
GOAL_COLOR = 11
LOCK_COLOR = 12
RUNE_MODE_COLORS = {0: 6, 1: 15}

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
    ("Level 7", 15, 15, 9),
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
    name, width, height, rune_count = spec
    path = _snake_path(width, height)
    path_set = set(path)

    start = path[0]
    goal = path[-1]
    start_mode = seed % 2

    usable = path[3:-3]
    step = max(1, len(usable) // (rune_count + 1))

    runes: list[dict[str, int]] = []
    idx = step
    while len(runes) < rune_count and idx < len(usable):
        x, y = usable[idx]
        runes.append({"x": int(x), "y": int(y), "mode": int((x + y + seed + len(runes)) % 2)})
        idx += step

    for x, y in usable:
        if len(runes) >= rune_count:
            break
        if any(r["x"] == x and r["y"] == y for r in runes):
            continue
        runes.append({"x": int(x), "y": int(y), "mode": int((x * 3 + y + seed + len(runes)) % 2)})

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
        Sprite(pixels=[[GOAL_COLOR]], name="goal", x=goal[0], y=goal[1], collidable=False, layer=2, tags=["goal"]),
        Sprite(
            pixels=[[LOCK_COLOR]], name="goal_lock", x=goal[0], y=goal[1], collidable=True, layer=5, tags=["goal_lock"]
        ),
        Sprite(
            pixels=[[PLAYER_COLORS[start_mode]]],
            name="player",
            x=start[0],
            y=start[1],
            collidable=True,
            layer=6,
            tags=["player"],
        ),
    ]

    for idx, rune in enumerate(runes):
        sprites.append(
            Sprite(
                pixels=[[RUNE_MODE_COLORS[int(rune["mode"])]]],
                name=f"rune_{idx}",
                x=int(rune["x"]),
                y=int(rune["y"]),
                collidable=False,
                layer=4,
                tags=["rune"],
            )
        )

    return Level(
        name=name,
        sprites=sprites,
        grid_size=(width, height),
        data={"width": width, "height": height, "start": start, "goal": goal, "start_mode": start_mode, "runes": runes},
    )


class Glyph(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec, seed=idx + 97) for idx, spec in enumerate(LEVEL_SPECS)]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__(
            "glyph", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4, 5, 6]
        )

    def on_set_level(self, level: Level) -> None:
        self._goal = tuple(int(v) for v in level.get_data("goal"))
        self._start_mode = int(level.get_data("start_mode") or 0) % 2
        self._mode = int(self._start_mode)
        self._progress = 0
        self._runes = [
            {"x": int(entry["x"]), "y": int(entry["y"]), "mode": int(entry["mode"]) % 2}
            for entry in (level.get_data("runes") or [])
        ]

        self._player = self.current_level.get_sprites_by_name("player")[0]
        self._goal_lock = self.current_level.get_sprites_by_name("goal_lock")[0]

        self._sync_player()
        self._reset_runes()
        self._sync_goal_lock()

    def _sync_player(self) -> None:
        self._player.pixels[0][0] = PLAYER_COLORS[self._mode]

    def _reset_runes(self) -> None:
        for sprite in list(self.current_level.get_sprites_by_tag("rune")):
            self.current_level.remove_sprite(sprite)

        for idx, rune in enumerate(self._runes):
            self.current_level.add_sprite(
                Sprite(
                    pixels=[[RUNE_MODE_COLORS[int(rune["mode"])]]],
                    name=f"rune_{idx}",
                    x=int(rune["x"]),
                    y=int(rune["y"]),
                    collidable=False,
                    layer=4,
                    tags=["rune"],
                )
            )

    def _sync_goal_lock(self) -> None:
        unlocked = self._progress >= len(self._runes)
        self._goal_lock.set_collidable(not unlocked)
        self._goal_lock.set_visible(not unlocked)

    def _toggle_mode(self) -> None:
        self._mode ^= 1
        self._sync_player()

    def _distance_to_player(self, x: int, y: int) -> int:
        return abs(int(self._player.x) - x) + abs(int(self._player.y) - y)

    def _expected_rune(self):
        if self._progress < 0 or self._progress >= len(self._runes):
            return None
        return self._runes[self._progress]

    def _click_rune(self) -> bool:
        data = self.action.data or {}
        dx = int(data.get("x", -1))
        dy = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(dx, dy)
        if grid_pos is None:
            return False

        expected = self._expected_rune()
        if expected is None:
            return False

        gx = int(grid_pos[0])
        gy = int(grid_pos[1])

        if gx != int(expected["x"]) or gy != int(expected["y"]):
            return False
        if self._mode != int(expected["mode"]):
            return False
        if self._distance_to_player(gx, gy) > 1:
            return False

        rune_name = f"rune_{self._progress}"
        rune_sprites = self.current_level.get_sprites_by_name(rune_name)
        if rune_sprites:
            self.current_level.remove_sprite(rune_sprites[0])

        self._progress += 1
        self._sync_goal_lock()
        return True

    def _handle_wrong_rune(self) -> None:
        self.lose()

    def _try_move(self, dx: int, dy: int) -> None:
        self.try_move_sprite(self._player, dx, dy)
        if self._progress >= len(self._runes) and (int(self._player.x), int(self._player.y)) == self._goal:
            self.next_level()

    def step(self) -> None:
        action = self.action.id

        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._try_move(dx, dy)
        elif action == GameAction.ACTION5:
            self._toggle_mode()
        elif action == GameAction.ACTION6:
            if not self._click_rune() and self._expected_rune() is not None:
                self._handle_wrong_rune()
                self.complete_action()
                return

        self.complete_action()
