from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 1
WALL_COLOR = 4
PHASE0_COLOR = 10
PHASE1_COLOR = 15
GOAL_COLOR = 11
BRIDGE_CLOSED_COLOR = 12
BRIDGE_OPEN_COLOR = 2
NODE_COLOR = 6

PLAYER_PHASE0_PIXELS = [[-1, 10, -1], [10, 0, 10], [-1, 10, -1]]
PLAYER_PHASE1_PIXELS = [[-1, 15, -1], [15, 6, 15], [-1, 15, -1]]

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),  # W
    GameAction.ACTION2: (0, 1),  # S
    GameAction.ACTION3: (-1, 0),  # A
    GameAction.ACTION4: (1, 0),  # D
}

LEVEL_SPECS: list[dict] = [
    {
        "name": "Phase Warmup",
        "map": ["###########", "#S..1111.G#", "###.###.###", "#..0000...#", "###########"],
        "node_effects": {},
        "start_phase": 0,
    },
    {
        "name": "Relay Gate",
        "map": ["###########", "#S...a...G#", "#.#######.#", "#..A.....##", "###########"],
        "node_effects": {"A": "a"},
        "start_phase": 0,
    },
    {
        "name": "Phase Relay",
        "map": ["#############", "#S..111a11G##", "###.###.#.###", "#..000A.#...#", "#############"],
        "node_effects": {"A": "a"},
        "start_phase": 0,
    },
    {
        "name": "Dual Locks",
        "map": ["#############", "#S..a..111G##", "#.###.#.#.###", "#.A..b0..B..#", "#############"],
        "node_effects": {"A": "b", "B": "a"},
        "start_phase": 0,
    },
    {
        "name": "Cross Coupling",
        "map": ["#############", "#S..a..111G##", "#.###.#.#.###", "#.A..b0c.B..#", "#############"],
        "node_effects": {"A": "bc", "B": "ab"},
        "start_phase": 1,
    },
]


def _parse_level_map(lines: list[str]):
    if not lines:
        raise ValueError("Level map must not be empty.")
    width = len(lines[0])
    height = len(lines)
    if width == 0:
        raise ValueError("Level map width must be > 0.")
    for row in lines:
        if len(row) != width:
            raise ValueError("All level rows must have equal width.")

    walls: set[tuple[int, int]] = set()
    phase_tiles = {0: set(), 1: set()}
    bridges: dict[str, tuple[int, int]] = {}
    nodes: dict[str, tuple[int, int]] = {}
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None

    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "S":
                start = (x, y)
            elif ch == "G":
                goal = (x, y)
            elif ch == "0":
                phase_tiles[0].add((x, y))
            elif ch == "1":
                phase_tiles[1].add((x, y))
            elif "a" <= ch <= "z":
                bridges[ch] = (x, y)
            elif "A" <= ch <= "Z" and ch not in {"S", "G"}:
                nodes[ch] = (x, y)

    if start is None or goal is None:
        raise ValueError("Each level requires one S start and one G goal.")

    if start[0] <= 0 or start[1] <= 0 or start[0] >= width - 1 or start[1] >= height - 1:
        raise ValueError("Start must not be on the outer border (needed for 3x3 player sprite).")

    return {
        "width": width,
        "height": height,
        "walls": walls,
        "phase_tiles": phase_tiles,
        "bridges": bridges,
        "nodes": nodes,
        "start": start,
        "goal": goal,
    }


def _build_level(spec: dict) -> Level:
    parsed = _parse_level_map(list(spec.get("map") or []))
    width = int(parsed["width"])
    height = int(parsed["height"])
    walls = parsed["walls"]
    phase_tiles = parsed["phase_tiles"]
    bridges = parsed["bridges"]
    nodes = parsed["nodes"]
    start = tuple(parsed["start"])
    goal = tuple(parsed["goal"])

    node_effects_raw = dict(spec.get("node_effects") or {})
    node_effects: dict[str, list[str]] = {}
    for node_id, targets in node_effects_raw.items():
        node_effects[str(node_id)] = [ch for ch in str(targets)]

    for node_id in node_effects:
        if node_id not in nodes:
            raise ValueError(f"Unknown node `{node_id}` in node_effects.")
    for node_id in nodes:
        node_effects.setdefault(node_id, [])
    for node_id, targets in node_effects.items():
        for bridge_id in targets:
            if bridge_id not in bridges:
                raise ValueError(f"Node `{node_id}` references unknown bridge `{bridge_id}`.")

    start_phase = int(spec.get("start_phase", 0)) & 1
    initial_open = {ch for ch in str(spec.get("initial_open", ""))}
    for bridge_id in initial_open:
        if bridge_id not in bridges:
            raise ValueError(f"initial_open references unknown bridge `{bridge_id}`.")

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]
    wall_pixels = [[-1] * width for _ in range(height)]
    phase0_pixels = [[-1] * width for _ in range(height)]
    phase1_pixels = [[-1] * width for _ in range(height)]

    for x, y in walls:
        wall_pixels[y][x] = WALL_COLOR
    for x, y in phase_tiles[0]:
        phase0_pixels[y][x] = PHASE0_COLOR
    for x, y in phase_tiles[1]:
        phase1_pixels[y][x] = PHASE1_COLOR

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10),
        Sprite(pixels=phase0_pixels, name="phase0_tiles", collidable=False, layer=-8, tags=["phase0"]),
        Sprite(pixels=phase1_pixels, name="phase1_tiles", collidable=False, layer=-7, tags=["phase1"]),
        Sprite(pixels=wall_pixels, name="walls", collidable=True, layer=2, tags=["wall"]),
        Sprite(pixels=[[GOAL_COLOR]], name="goal", x=goal[0], y=goal[1], collidable=False, layer=4, tags=["goal"]),
        Sprite(
            pixels=np.array(PLAYER_PHASE0_PIXELS if start_phase == 0 else PLAYER_PHASE1_PIXELS, dtype=int),
            name="player",
            x=start[0] - 1,
            y=start[1] - 1,
            collidable=False,
            layer=7,
            tags=["player"],
        ),
    ]

    for bridge_id in sorted(bridges.keys()):
        bx, by = bridges[bridge_id]
        is_open = bridge_id in initial_open
        sprites.append(
            Sprite(
                pixels=[[BRIDGE_OPEN_COLOR if is_open else BRIDGE_CLOSED_COLOR]],
                name=f"bridge_{bridge_id}",
                x=bx,
                y=by,
                collidable=not is_open,
                layer=5,
                tags=["bridge"],
            )
        )

    for node_id in sorted(nodes.keys()):
        nx, ny = nodes[node_id]
        sprites.append(
            Sprite(pixels=[[NODE_COLOR]], name=f"node_{node_id}", x=nx, y=ny, collidable=False, layer=6, tags=["node"])
        )

    level_data = {
        "width": width,
        "height": height,
        "walls": sorted((int(x), int(y)) for x, y in walls),
        "phase0_tiles": sorted((int(x), int(y)) for x, y in phase_tiles[0]),
        "phase1_tiles": sorted((int(x), int(y)) for x, y in phase_tiles[1]),
        "start": (int(start[0]), int(start[1])),
        "goal": (int(goal[0]), int(goal[1])),
        "start_phase": int(start_phase),
        "bridges": [
            {
                "id": bridge_id,
                "x": int(bridges[bridge_id][0]),
                "y": int(bridges[bridge_id][1]),
                "open": bool(bridge_id in initial_open),
            }
            for bridge_id in sorted(bridges.keys())
        ],
        "nodes": [
            {
                "id": node_id,
                "x": int(nodes[node_id][0]),
                "y": int(nodes[node_id][1]),
                "toggles": [str(ch) for ch in node_effects[node_id]],
            }
            for node_id in sorted(nodes.keys())
        ],
    }

    return Level(name=str(spec.get("name", "Level")), sprites=sprites, grid_size=(width, height), data=level_data)


class Flux(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 4, 4, [])
        super().__init__(
            "flux", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[1, 2, 3, 4, 5, 6]
        )

    def on_set_level(self, level: Level) -> None:
        self._width = int(level.get_data("width"))
        self._height = int(level.get_data("height"))
        self._player_cell = tuple(int(v) for v in level.get_data("start"))
        self._goal_cell = tuple(int(v) for v in level.get_data("goal"))
        self._phase = int(level.get_data("start_phase") or 0) & 1

        self._wall_cells = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        self._phase_tiles = {
            0: {tuple(int(v) for v in item) for item in (level.get_data("phase0_tiles") or [])},
            1: {tuple(int(v) for v in item) for item in (level.get_data("phase1_tiles") or [])},
        }

        self._bridge_index: dict[str, int] = {}
        self._bridge_positions: dict[str, tuple[int, int]] = {}
        self._bridge_cell_to_index: dict[tuple[int, int], int] = {}
        self._bridge_mask = 0

        for idx, bridge in enumerate(level.get_data("bridges") or []):
            bridge_id = str(bridge["id"])
            bx = int(bridge["x"])
            by = int(bridge["y"])
            self._bridge_index[bridge_id] = idx
            self._bridge_positions[bridge_id] = (bx, by)
            self._bridge_cell_to_index[(bx, by)] = idx
            if bool(bridge.get("open", False)):
                self._bridge_mask |= 1 << idx

        self._node_at: dict[tuple[int, int], str] = {}
        self._node_positions: dict[str, tuple[int, int]] = {}
        self._node_masks: dict[str, int] = {}
        for node in level.get_data("nodes") or []:
            node_id = str(node["id"])
            nx = int(node["x"])
            ny = int(node["y"])
            toggles = [str(ch) for ch in (node.get("toggles") or [])]

            mask = 0
            for bridge_id in toggles:
                bit = self._bridge_index.get(bridge_id)
                if bit is not None:
                    mask ^= 1 << bit

            self._node_at[(nx, ny)] = node_id
            self._node_positions[node_id] = (nx, ny)
            self._node_masks[node_id] = mask

        self._player_sprite = self.current_level.get_sprites_by_name("player")[0]
        self._sync_visuals()

    def _bridge_is_open(self, bridge_id: str) -> bool:
        bit = self._bridge_index.get(bridge_id)
        if bit is None:
            return False
        return bool((self._bridge_mask >> bit) & 1)

    def _sync_visuals(self) -> None:
        pattern = PLAYER_PHASE0_PIXELS if self._phase == 0 else PLAYER_PHASE1_PIXELS
        self._player_sprite.pixels = np.array(pattern, dtype=int)
        self._player_sprite.set_position(self._player_cell[0] - 1, self._player_cell[1] - 1)

        for bridge_id, _ in self._bridge_positions.items():
            bridge_sprite = self.current_level.get_sprites_by_name(f"bridge_{bridge_id}")[0]
            is_open = self._bridge_is_open(bridge_id)
            bridge_sprite.pixels[0][0] = BRIDGE_OPEN_COLOR if is_open else BRIDGE_CLOSED_COLOR
            bridge_sprite.set_collidable(not is_open)

    def _cell_blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._width or y >= self._height:
            return True
        if (x, y) in self._wall_cells:
            return True

        bridge_bit = self._bridge_cell_to_index.get((x, y))
        if bridge_bit is not None and ((self._bridge_mask >> bridge_bit) & 1) == 0:
            return True

        if (x, y) in self._phase_tiles[0] and self._phase != 0:
            return True
        return bool((x, y) in self._phase_tiles[1] and self._phase != 1)

    def _try_move_player(self, dx: int, dy: int) -> None:
        nx = int(self._player_cell[0] + dx)
        ny = int(self._player_cell[1] + dy)
        if self._cell_blocked(nx, ny):
            return
        self._player_cell = (nx, ny)
        if self._player_cell == self._goal_cell:
            self.next_level()
            return
        self._sync_visuals()

    def _toggle_phase(self) -> None:
        self._phase ^= 1
        self._sync_visuals()

    def _try_click_node(self) -> None:
        data = self.action.data or {}
        display_x = int(data.get("x", -1))
        display_y = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(display_x, display_y)
        if grid_pos is None:
            return

        node_id = self._node_at.get((int(grid_pos[0]), int(grid_pos[1])))
        if node_id is None:
            return

        mask = int(self._node_masks.get(node_id, 0))
        if mask == 0:
            return

        self._bridge_mask ^= mask
        self._sync_visuals()

    def step(self) -> None:
        action = self.action.id
        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            self._try_move_player(dx, dy)
        elif action == GameAction.ACTION5:
            self._toggle_phase()
        elif action == GameAction.ACTION6:
            self._try_click_node()

        self.complete_action()
