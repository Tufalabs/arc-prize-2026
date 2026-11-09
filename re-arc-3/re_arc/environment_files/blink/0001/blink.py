from __future__ import annotations

from collections import deque

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 1
WALL_COLOR = 5
PLAYER_COLOR = 9
GOAL_COLOR = 11
ANCHOR_IDLE_COLOR = 6
ANCHOR_SELECTED_COLOR = 14
INDICATOR_COLORS = [3, 6, 8, 10, 12, 14, 15]

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

LEVEL_SPECS: list[tuple[str, int, int, int]] = [
    ("Level 1", 9, 7, 1),
    ("Level 2", 11, 7, 2),
    ("Level 3", 15, 7, 3),
    ("Level 4", 15, 9, 3),
    ("Level 5", 15, 9, 3),
    ("Level 6", 17, 11, 4),
    ("Level 7", 17, 11, 4),
    ("Level 8", 19, 13, 4),
]


def _build_char_grid(width: int, height: int, pairs: int, seed: int):
    if width < (pairs * 3) + 5:
        raise ValueError("blink width is too small for requested pair count")

    grid = [["."] * width for _ in range(height)]
    for x in range(width):
        grid[0][x] = "#"
        grid[height - 1][x] = "#"
    for y in range(height):
        grid[y][0] = "#"
        grid[y][width - 1] = "#"

    separators = [3 + (i * 3) for i in range(pairs)]
    for col in separators:
        if col >= width - 1:
            raise ValueError("blink separator exceeds map width")
        for row in range(1, height - 1):
            grid[row][col] = "#"

    start = (1, 1)
    goal = (width - 2, height - 2)
    grid[start[1]][start[0]] = "S"
    grid[goal[1]][goal[0]] = "G"

    anchors: dict[str, list[tuple[int, int]]] = {}
    for idx, col in enumerate(separators):
        anchor_id = chr(ord("a") + idx)

        left_x = col - 1
        right_x = col + 1

        left_y = 1 + ((seed + (idx * 3)) % (height - 2))
        right_y = 1 + ((seed * 2 + (idx * 5)) % (height - 2))

        if (left_x, left_y) == start:
            left_y = min(height - 2, left_y + 1)
        if (right_x, right_y) == goal:
            right_y = max(1, right_y - 1)

        grid[left_y][left_x] = anchor_id
        grid[right_y][right_x] = anchor_id
        anchors[anchor_id] = [(left_x, left_y), (right_x, right_y)]

    return grid, start, goal, anchors


def _parse_grid(grid: list[list[str]]):
    height = len(grid)
    width = len(grid[0]) if height else 0

    walls: set[tuple[int, int]] = set()
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None
    anchors: dict[str, list[tuple[int, int]]] = {}

    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == "#":
                walls.add((x, y))
            elif cell == "S":
                start = (x, y)
            elif cell == "G":
                goal = (x, y)
            elif "a" <= cell <= "z":
                anchors.setdefault(cell, []).append((x, y))

    if start is None or goal is None:
        raise ValueError("blink level requires S and G")

    for anchor_id, pts in anchors.items():
        if len(pts) != 2:
            raise ValueError(f"blink anchor `{anchor_id}` must appear exactly twice")

    return {"width": width, "height": height, "walls": walls, "start": start, "goal": goal, "anchors": anchors}


def _solve(level_data: dict):
    width = int(level_data["width"])
    height = int(level_data["height"])
    walls = set(level_data["walls"])
    start = tuple(level_data["start"])
    goal = tuple(level_data["goal"])

    anchor_positions: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    anchor_ids: list[str] = []
    for entry in level_data["anchors"]:
        aid = str(entry["id"])
        p0 = (int(entry["points"][0][0]), int(entry["points"][0][1]))
        p1 = (int(entry["points"][1][0]), int(entry["points"][1][1]))
        anchor_positions[aid] = (p0, p1)
        anchor_ids.append(aid)

    selected_values = [-1, *list(range(len(anchor_ids)))]
    start_state = (start[0], start[1], -1)

    def blocked(x: int, y: int):
        if x < 0 or y < 0 or x >= width or y >= height:
            return True
        return (x, y) in walls

    queue = deque([start_state])
    previous = {start_state: None}
    previous_action: dict[tuple[int, int, int], tuple[str, int | None]] = {}

    while queue:
        state = queue.popleft()
        x, y, selected = state
        if (x, y) == goal:
            break

        for action_id, (dx, dy) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))):
            nx, ny = x + dx, y + dy
            if blocked(nx, ny):
                continue
            nxt = (nx, ny, selected)
            if nxt in previous:
                continue
            previous[nxt] = state
            previous_action[nxt] = ("move", action_id)
            queue.append(nxt)

        for sel in selected_values:
            if sel == selected:
                continue
            nxt = (x, y, sel)
            if nxt in previous:
                continue
            previous[nxt] = state
            previous_action[nxt] = ("select", sel)
            queue.append(nxt)

        if selected >= 0:
            aid = anchor_ids[selected]
            p0, p1 = anchor_positions[aid]
            if (x, y) == p0:
                nxt = (p1[0], p1[1], selected)
            elif (x, y) == p1:
                nxt = (p0[0], p0[1], selected)
            else:
                nxt = None
            if nxt is not None and nxt not in previous:
                previous[nxt] = state
                previous_action[nxt] = ("blink", None)
                queue.append(nxt)

    goal_state = None
    for state in previous:
        if (state[0], state[1]) == goal:
            goal_state = state
            break

    if goal_state is None:
        return None

    plan = []
    cursor = goal_state
    while previous[cursor] is not None:
        plan.append(previous_action[cursor])
        cursor = previous[cursor]  # type: ignore[index]
    plan.reverse()
    return plan


def _build_level(spec: tuple[str, int, int, int], seed: int) -> Level:
    name, width, height, pairs = spec
    grid, _, _, _ = _build_char_grid(width, height, pairs, seed)
    parsed = _parse_grid(grid)

    walls = parsed["walls"]
    start = parsed["start"]
    goal = parsed["goal"]
    anchors = parsed["anchors"]

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]
    wall_pixels = [[-1] * width for _ in range(height)]
    for x, y in walls:
        wall_pixels[y][x] = WALL_COLOR

    anchor_entries = []
    for aid in sorted(anchors.keys()):
        pts = anchors[aid]
        anchor_entries.append(
            {"id": aid, "points": [(int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1]))]}
        )

    level_data = {
        "width": width,
        "height": height,
        "start": (int(start[0]), int(start[1])),
        "goal": (int(goal[0]), int(goal[1])),
        "walls": sorted((int(x), int(y)) for x, y in walls),
        "anchors": anchor_entries,
    }

    if _solve(level_data) is None:
        raise ValueError(f"blink level `{name}` is unsolvable")

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=1, tags=["wall"]),
        Sprite(pixels=[[GOAL_COLOR]], name="goal", x=goal[0], y=goal[1], collidable=False, layer=3, tags=["goal"]),
        Sprite(
            pixels=[[PLAYER_COLOR]], name="player", x=start[0], y=start[1], collidable=False, layer=6, tags=["player"]
        ),
        Sprite(
            pixels=[[INDICATOR_COLORS[0]]],
            name="selection_indicator",
            x=0,
            y=0,
            collidable=False,
            layer=7,
            tags=["selection_indicator"],
        ),
    ]

    for aid, pts in sorted(anchors.items()):
        for idx, (ax, ay) in enumerate(pts):
            sprites.append(
                Sprite(
                    pixels=[[ANCHOR_IDLE_COLOR]],
                    name=f"anchor_{aid}_{idx}",
                    x=ax,
                    y=ay,
                    collidable=False,
                    layer=5,
                    tags=["anchor", f"anchor_id_{aid}"],
                )
            )

    return Level(name=name, sprites=sprites, grid_size=(width, height), data=level_data)


class Blink(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec, seed=idx + 79) for idx, spec in enumerate(LEVEL_SPECS)]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__(
            "blink", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4, 5, 6]
        )

    def on_set_level(self, level: Level) -> None:
        self._width = int(level.get_data("width"))
        self._height = int(level.get_data("height"))
        self._goal = tuple(int(v) for v in level.get_data("goal"))
        self._walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}

        self._anchor_positions: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        self._anchor_at: dict[tuple[int, int], str] = {}
        self._anchor_ids: list[str] = []

        for entry in level.get_data("anchors") or []:
            aid = str(entry["id"])
            p0 = (int(entry["points"][0][0]), int(entry["points"][0][1]))
            p1 = (int(entry["points"][1][0]), int(entry["points"][1][1]))
            self._anchor_positions[aid] = (p0, p1)
            self._anchor_at[p0] = aid
            self._anchor_at[p1] = aid
            self._anchor_ids.append(aid)

        self._anchor_ids.sort()
        self._selected_id: str | None = None

        self._player = self.current_level.get_sprites_by_name("player")[0]
        self._indicator = self.current_level.get_sprites_by_name("selection_indicator")[0]
        self._sync_anchor_visuals()

    def _blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._width or y >= self._height:
            return True
        return (x, y) in self._walls

    def _sync_anchor_visuals(self) -> None:
        for aid in self._anchor_ids:
            selected = aid == self._selected_id
            color = ANCHOR_SELECTED_COLOR if selected else ANCHOR_IDLE_COLOR
            for sprite in self.current_level.get_sprites_by_tag(f"anchor_id_{aid}"):
                sprite.pixels[0][0] = color

        if self._selected_id is None:
            self._indicator.pixels[0][0] = INDICATOR_COLORS[0]
            return

        idx = self._anchor_ids.index(self._selected_id)
        self._indicator.pixels[0][0] = INDICATOR_COLORS[(idx + 1) % len(INDICATOR_COLORS)]

    def _try_move(self, dx: int, dy: int) -> None:
        nx = int(self._player.x + dx)
        ny = int(self._player.y + dy)
        if self._blocked(nx, ny):
            return
        self._player.set_position(nx, ny)
        if (nx, ny) == self._goal:
            self.next_level()

    def _select_anchor(self) -> None:
        data = self.action.data or {}
        dx = int(data.get("x", -1))
        dy = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(dx, dy)
        if grid_pos is None:
            return
        aid = self._anchor_at.get((int(grid_pos[0]), int(grid_pos[1])))
        if aid is None:
            return
        self._selected_id = aid
        self._sync_anchor_visuals()

    def _blink(self) -> None:
        if self._selected_id is None:
            return
        pair = self._anchor_positions.get(self._selected_id)
        if pair is None:
            return

        current = (int(self._player.x), int(self._player.y))
        if current == pair[0]:
            target = pair[1]
        elif current == pair[1]:
            target = pair[0]
        else:
            return

        self._player.set_position(target[0], target[1])
        if target == self._goal:
            self.next_level()

    def step(self) -> None:
        action = self.action.id
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._try_move(dx, dy)
        elif action == GameAction.ACTION5:
            self._blink()
        elif action == GameAction.ACTION6:
            self._select_anchor()
        self.complete_action()
