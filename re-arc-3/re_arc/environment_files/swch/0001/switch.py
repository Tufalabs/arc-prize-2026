from __future__ import annotations

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 0
WALL_COLOR = 13
PLAYER_COLOR = 8
GOAL_COLOR = 11
GATE_COLOR = 15
SWITCH_COLOR = 6

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),  # W
    GameAction.ACTION2: (0, 1),  # S
    GameAction.ACTION3: (-1, 0),  # A
    GameAction.ACTION4: (1, 0),  # D
}

LEVEL_SPECS: list[tuple[str, int, int, int]] = [
    ("Level 1", 9, 9, 2),
    ("Level 2", 11, 9, 2),
    ("Level 3", 11, 11, 3),
    ("Level 4", 13, 11, 3),
    ("Level 5", 13, 13, 4),
    ("Level 6", 15, 13, 4),
    ("Level 7", 15, 15, 5),
    ("Level 8", 17, 15, 5),
]


def _build_level(spec: tuple[str, int, int, int]) -> Level:
    name, width, height, gates = spec
    grid = [["."] * width for _ in range(height)]
    for x in range(width):
        grid[0][x] = "#"
        grid[height - 1][x] = "#"
    for y in range(height):
        grid[y][0] = "#"
        grid[y][width - 1] = "#"

    gate_positions: list[tuple[int, int]] = []
    for idx in range(gates):
        col = 2 + idx * 3
        if col >= width - 1:
            break
        for row in range(1, height - 1):
            grid[row][col] = "#"
        gate_row = 1 + (idx * 2) % (height - 2)
        grid[gate_row][col] = "D"
        gate_positions.append((col, gate_row))

    player = (1, 1)
    goal = (width - 2, height - 2)
    switch_pos = (1, height - 2)

    grid[player[1]][player[0]] = "P"
    grid[goal[1]][goal[0]] = "G"
    grid[switch_pos[1]][switch_pos[0]] = "S"

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
    gate_positions: list[tuple[int, int]] = []
    switch_positions: list[tuple[int, int]] = []

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
                gate_positions.append((x, y))
            elif cell == "S":
                switch_positions.append((x, y))

    if player_pos is None or goal_pos is None or not switch_positions:
        raise ValueError("Each level must have a player, goal, and switch.")

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

    for idx, (x, y) in enumerate(gate_positions):
        sprites.append(
            Sprite(pixels=[[GATE_COLOR]], name=f"gate_{idx}", x=x, y=y, collidable=True, layer=4, tags=["gate"])
        )

    for idx, (x, y) in enumerate(switch_positions):
        sprites.append(
            Sprite(pixels=[[SWITCH_COLOR]], name=f"switch_{idx}", x=x, y=y, collidable=False, layer=2, tags=["switch"])
        )

    return Level(name=name, sprites=sprites, grid_size=(width, height), data={"goal": goal_pos})


class Switch(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__(
            "swch", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4, 6]
        )

    def on_set_level(self, _level: Level) -> None:
        self._gates_open = False
        self._apply_gate_state()

    def _apply_gate_state(self) -> None:
        for gate in self.current_level.get_sprites_by_tag("gate"):
            gate.set_collidable(not self._gates_open)
            gate.set_visible(not self._gates_open)

    def _try_click_switch(self) -> None:
        data = self.action.data or {}
        display_x = int(data.get("x", -1))
        display_y = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(display_x, display_y)
        if grid_pos is None:
            return
        for switch in self.current_level.get_sprites_by_tag("switch"):
            if switch.x == grid_pos[0] and switch.y == grid_pos[1]:
                self._gates_open = not self._gates_open
                self._apply_gate_state()
                break

    def _try_move_player(self, dx: int, dy: int) -> None:
        player = self.current_level.get_sprites_by_name("player")[0]
        self.try_move_sprite(player, dx, dy)
        goal = self.current_level.get_data("goal")
        if goal and player.x == goal[0] and player.y == goal[1]:
            self.next_level()

    def step(self) -> None:
        action = self.action.id
        if action == GameAction.ACTION6:
            self._try_click_switch()
        elif action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._try_move_player(dx, dy)

        self.complete_action()
