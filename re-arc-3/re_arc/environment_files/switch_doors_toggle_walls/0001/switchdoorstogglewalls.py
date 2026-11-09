from __future__ import annotations

import math
from collections import deque

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "switch_doors_toggle_walls-0001"

COLOR_TIMEBAR_EMPTY = 0
COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_PLAYER = 3
COLOR_SWITCH_A_IDLE = 4
COLOR_SWITCH_A_PRESSED = 5
COLOR_DOOR_A_CLOSED = 6
COLOR_DOOR_A_OPEN = 7
COLOR_SWITCH_B_IDLE = 8
COLOR_SWITCH_B_PRESSED = 9
COLOR_DOOR_B_CLOSED = 10
COLOR_DOOR_B_OPEN = 11
COLOR_EXIT_BASE = 12
COLOR_EXIT_PULSE = 13
COLOR_HAZARD = 14
COLOR_TIMEBAR_FILLED = 15

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

FAIL_ANIMATION_STEPS = 4
WIN_ANIMATION_STEPS = 3


LEVEL_SPECS: list[dict[str, object]] = [
    {
        "name": "Level 1",
        "size": (24, 12),
        "time_limit": 120,
        "layout": [
            "#======================#",
            "########################",
            "#@......()|............#",
            "#.........#............#",
            "#.........#............#",
            "#.........#............#",
            "#.........#............#",
            "#.........#......../\\..#",
            "#.........#........\\/..#",
            "#.........#............#",
            "#.........#............#",
            "########################",
        ],
    },
    {
        "name": "Level 2",
        "size": (24, 12),
        "time_limit": 150,
        "layout": [
            "#======================#",
            "########################",
            "#@......#......#.......#",
            "#.......#......#.......#",
            "#.......:..()..#.......#",
            "#.......#......#.......#",
            "#.......#......#.......#",
            "#.......#......#.......#",
            "#.......#......|..../\\.#",
            "#.......#......|....\\/.#",
            "#.......#......#.......#",
            "########################",
        ],
    },
    {
        "name": "Level 3",
        "size": (24, 14),
        "time_limit": 170,
        "layout": [
            "#======================#",
            "########################",
            "#@.....................#",
            "#.................()...#",
            "#......................#",
            "#..........######......#",
            "#......................#",
            "#......................#",
            "###..#####|||###########",
            "#..^.^.................#",
            "##########.............#",
            "#................../\\..#",
            "#..................\\/..#",
            "########################",
        ],
    },
    {
        "name": "Level 4",
        "size": (24, 12),
        "time_limit": 190,
        "layout": [
            "#======================#",
            "########################",
            "#@..()..#......#.......#",
            "#.......#.[]...#.......#",
            "#.......|......#.......#",
            "#.......#..^...#.......#",
            "#.......#......+.......#",
            "#.......#......+.......#",
            "#.......#......#.../\\..#",
            "#.......#......#...\\/..#",
            "#.......#......#.......#",
            "########################",
        ],
    },
    {
        "name": "Level 5",
        "size": (28, 18),
        "time_limit": 260,
        "layout": [
            "#==========================#",
            "############################",
            "#.@..()......#.............#",
            "#............#....[].......#",
            "#............#.............#",
            "#............|.............#",
            "#............#.............#",
            "#....^.......#.............#",
            "#............#.............#",
            "####### +####################",
            "#............#.............#",
            "#..[]..^^....#.....^.......#",
            "#............|.............#",
            "#............#.............#",
            "#....()......#.............#",
            "#............#......./\\....#",
            "#............#.......\\/....#",
            "############################",
        ],
    },
    {
        "name": "Level 6",
        "size": (28, 20),
        "time_limit": 320,
        "layout": [
            "#==========================#",
            "############################",
            "#@..()....#........#........#",
            "#........#..[]....#........#",
            "#....^....#........#........#",
            "#........|....()..#........#",
            "#........#..^^....#........#",
            "#........#........+....[]..#",
            "#........#...^....#........#",
            "#....#####........#........#",
            "#....#####..^^....#........#",
            "#...^.....#........#..^^....#",
            "#........#........-........#",
            "#........#....^....#........#",
            "#........:..[]....#........#",
            "#........#........#..()....#",
            "#........#........#.../\\...#",
            "#........#........#...\\/...#",
            "#........#..^.....#........#",
            "############################",
        ],
    },
]


def _normalize_layout_rows(rows: list[str], width: int) -> list[str]:
    out: list[str] = []
    for raw in rows:
        row = raw.replace(" ", "")
        if len(row) < width:
            row = row + ("#" * (width - len(row)))
        if len(row) > width:
            if row.startswith("#") and row.endswith("#"):
                body = list(row[1:-1])
                while len(body) > width - 2:
                    trim_at = -1
                    for idx in range(len(body) - 1, -1, -1):
                        if body[idx] == ".":
                            trim_at = idx
                            break
                    if trim_at < 0:
                        body.pop()
                    else:
                        body.pop(trim_at)
                row = "#" + "".join(body) + "#"
            else:
                row = row[:width]
        out.append(row)
    return out


def _build_level(spec: dict[str, object]) -> Level:
    name = str(spec["name"])
    width, height = (int(v) for v in spec["size"])
    time_limit = int(spec["time_limit"])
    rows = _normalize_layout_rows(list(spec["layout"]), width)
    if len(rows) != height:
        raise ValueError(f"{name}: expected {height} rows, got {len(rows)}")

    floor_pixels = [[COLOR_FLOOR] * width for _ in range(height)]
    for x in range(width):
        floor_pixels[0][x] = COLOR_TIMEBAR_EMPTY

    wall_pixels = [[-1] * width for _ in range(height)]
    hazard_pixels = [[-1] * width for _ in range(height)]

    start: tuple[int, int] | None = None
    walls: set[tuple[int, int]] = set()
    hazards: set[tuple[int, int]] = set()
    exit_cells: set[tuple[int, int]] = set()

    switches: list[dict[str, object]] = []
    door_a_initial_closed: set[tuple[int, int]] = set()
    door_b_initial_closed: set[tuple[int, int]] = set()
    door_a_positions: set[tuple[int, int]] = set()
    door_b_positions: set[tuple[int, int]] = set()

    switch_a_idx = 0
    switch_b_idx = 0

    for y, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"{name}: row {y} has width {len(row)} expected {width}")
        x = 0
        while x < width:
            ch = row[x]
            cell = (x, y)

            if y == 0:
                if ch == "#":
                    walls.add(cell)
                    wall_pixels[y][x] = COLOR_WALL
                x += 1
                continue

            if ch == "#":
                walls.add(cell)
                wall_pixels[y][x] = COLOR_WALL
            elif ch == ".":
                pass
            elif ch == "@":
                if start is not None:
                    raise ValueError(f"{name}: multiple player starts")
                start = cell
            elif ch == "^":
                hazards.add(cell)
                hazard_pixels[y][x] = COLOR_HAZARD
            elif ch == "(":
                if x + 1 >= width or row[x + 1] != ")":
                    raise ValueError(f"{name}: invalid Switch A at {(x, y)}")
                switches.append({"name": f"switch_a_{switch_a_idx}", "kind": "A", "cells": [(x, y), (x + 1, y)]})
                switch_a_idx += 1
                x += 1
            elif ch == "[":
                if x + 1 >= width or row[x + 1] != "]":
                    raise ValueError(f"{name}: invalid Switch B at {(x, y)}")
                switches.append({"name": f"switch_b_{switch_b_idx}", "kind": "B", "cells": [(x, y), (x + 1, y)]})
                switch_b_idx += 1
                x += 1
            elif ch == "|":
                door_a_positions.add(cell)
                door_a_initial_closed.add(cell)
            elif ch == ":":
                door_a_positions.add(cell)
            elif ch == "+":
                door_b_positions.add(cell)
                door_b_initial_closed.add(cell)
            elif ch == "-":
                door_b_positions.add(cell)
            elif ch in {"/", "\\"}:
                exit_cells.add(cell)
            elif ch in {"=", "_"}:
                pass
            else:
                raise ValueError(f"{name}: unsupported glyph {ch!r} at {(x, y)}")
            x += 1

    if start is None:
        raise ValueError(f"{name}: missing player start")
    if not exit_cells:
        raise ValueError(f"{name}: missing exit portal")

    sprites: list[Sprite] = [
        Sprite(pixels=floor_pixels, name="floor", layer=-10, collidable=False, tags=["floor"]),
        Sprite(pixels=wall_pixels, name="walls", layer=1, collidable=True, tags=["wall"]),
        Sprite(pixels=hazard_pixels, name="hazards", layer=2, collidable=False, tags=["hazard"]),
        Sprite(
            pixels=[[COLOR_TIMEBAR_EMPTY] * width], name="timebar", x=0, y=0, layer=6, collidable=False, tags=["hud"]
        ),
    ]

    for switch in switches:
        sx, sy = switch["cells"][0]
        idle_color = COLOR_SWITCH_A_IDLE if switch["kind"] == "A" else COLOR_SWITCH_B_IDLE
        sprites.append(
            Sprite(
                pixels=[[idle_color, idle_color]],
                name=str(switch["name"]),
                x=int(sx),
                y=int(sy),
                layer=4,
                collidable=False,
                tags=["switch", f"switch_{str(switch['kind']).lower()}"],
            )
        )

    for idx, (x, y) in enumerate(sorted(door_a_positions)):
        sprites.append(
            Sprite(
                pixels=[[COLOR_DOOR_A_CLOSED if (x, y) in door_a_initial_closed else COLOR_DOOR_A_OPEN]],
                name=f"door_a_{idx}",
                x=int(x),
                y=int(y),
                layer=5,
                collidable=(x, y) in door_a_initial_closed,
                tags=["door_a", "door"],
            )
        )

    for idx, (x, y) in enumerate(sorted(door_b_positions)):
        sprites.append(
            Sprite(
                pixels=[[COLOR_DOOR_B_CLOSED if (x, y) in door_b_initial_closed else COLOR_DOOR_B_OPEN]],
                name=f"door_b_{idx}",
                x=int(x),
                y=int(y),
                layer=5,
                collidable=(x, y) in door_b_initial_closed,
                tags=["door_b", "door"],
            )
        )

    for idx, (x, y) in enumerate(sorted(exit_cells)):
        sprites.append(
            Sprite(
                pixels=[[COLOR_EXIT_BASE]],
                name=f"exit_{idx}",
                x=int(x),
                y=int(y),
                layer=3,
                collidable=False,
                tags=["exit"],
            )
        )

    sprites.append(
        Sprite(
            pixels=[[COLOR_PLAYER]],
            name="player",
            x=int(start[0]),
            y=int(start[1]),
            layer=7,
            collidable=True,
            tags=["player"],
        )
    )

    return Level(
        name=name,
        grid_size=(width, height),
        sprites=sprites,
        data={
            "width": width,
            "height": height,
            "time_limit": time_limit,
            "start": [int(start[0]), int(start[1])],
            "walls": [[int(x), int(y)] for x, y in sorted(walls)],
            "hazards": [[int(x), int(y)] for x, y in sorted(hazards)],
            "exit_cells": [[int(x), int(y)] for x, y in sorted(exit_cells)],
            "switches": [
                {"name": str(s["name"]), "kind": str(s["kind"]), "cells": [[int(px), int(py)] for px, py in s["cells"]]}
                for s in switches
            ],
            "door_a_positions": [[int(x), int(y)] for x, y in sorted(door_a_positions)],
            "door_b_positions": [[int(x), int(y)] for x, y in sorted(door_b_positions)],
            "door_a_initial_closed": [[int(x), int(y)] for x, y in sorted(door_a_initial_closed)],
            "door_b_initial_closed": [[int(x), int(y)] for x, y in sorted(door_b_initial_closed)],
        },
    )


def _switch_name_by_cell(switches: list[dict[str, object]]) -> dict[tuple[int, int], str]:
    lookup: dict[tuple[int, int], str] = {}
    for switch in switches:
        name = str(switch["name"])
        for x, y in switch["cells"]:
            lookup[(int(x), int(y))] = name
    return lookup


def _door_closed(cell: tuple[int, int], *, initial_closed: set[tuple[int, int]], parity: int) -> bool:
    closed = cell in initial_closed
    if parity % 2 == 1:
        closed = not closed
    return closed


def _is_level_solvable(level: Level) -> bool:
    width = int(level.get_data("width"))
    height = int(level.get_data("height"))
    start = tuple(int(v) for v in level.get_data("start"))
    time_limit = int(level.get_data("time_limit"))

    walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
    hazards = {tuple(int(v) for v in item) for item in (level.get_data("hazards") or [])}
    exits = {tuple(int(v) for v in item) for item in (level.get_data("exit_cells") or [])}

    door_a_positions = {tuple(int(v) for v in item) for item in (level.get_data("door_a_positions") or [])}
    door_b_positions = {tuple(int(v) for v in item) for item in (level.get_data("door_b_positions") or [])}
    door_a_initial_closed = {tuple(int(v) for v in item) for item in (level.get_data("door_a_initial_closed") or [])}
    door_b_initial_closed = {tuple(int(v) for v in item) for item in (level.get_data("door_b_initial_closed") or [])}

    switch_specs = list(level.get_data("switches") or [])
    switch_cells: dict[tuple[int, int], tuple[int, str]] = {}
    for idx, switch in enumerate(switch_specs):
        kind = str(switch.get("kind", "")).upper()
        if kind not in {"A", "B"}:
            continue
        for x, y in switch.get("cells") or []:
            switch_cells[(int(x), int(y))] = (idx, kind)

    last_switch = -1
    if start in switch_cells:
        last_switch = int(switch_cells[start][0])

    start_state = (int(start[0]), int(start[1]), 0, 0, last_switch, 0)
    queue = deque([start_state])
    seen = {start_state}

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    while queue:
        x, y, parity_a, parity_b, prev_switch, steps = queue.popleft()
        if (x, y) in exits and steps <= time_limit:
            return True
        if steps >= time_limit:
            continue

        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            tx, ty = x, y
            if in_bounds(nx, ny) and ny > 0 and (nx, ny) not in walls:
                blocked = False
                if (nx, ny) in door_a_positions and _door_closed(
                    (nx, ny), initial_closed=door_a_initial_closed, parity=parity_a
                ):
                    blocked = True
                if (nx, ny) in door_b_positions and _door_closed(
                    (nx, ny), initial_closed=door_b_initial_closed, parity=parity_b
                ):
                    blocked = True
                if not blocked:
                    tx, ty = nx, ny

            n_parity_a = parity_a
            n_parity_b = parity_b
            switch_entry = switch_cells.get((tx, ty))
            switch_id = -1
            if switch_entry is not None:
                switch_id, kind = switch_entry
                if switch_id != prev_switch:
                    if kind == "A":
                        n_parity_a ^= 1
                    else:
                        n_parity_b ^= 1

            if (tx, ty) in hazards:
                continue

            next_state = (tx, ty, n_parity_a, n_parity_b, switch_id, steps + 1)
            if next_state in seen:
                continue
            seen.add(next_state)
            queue.append(next_state)

    return False


class SwitchDoorsToggleWalls(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        for level in levels:
            if not _is_level_solvable(level):
                raise ValueError(f"{level.name} is not solvable with current mechanics")

        first_w, first_h = levels[0].grid_size
        camera = Camera(width=first_w, height=first_h, background=COLOR_TIMEBAR_EMPTY)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

        self._time_limit = 0
        self._time_remaining = 0
        self._phase = "play"
        self._phase_ticks = 0
        self._portal_pulse_on = False

        self._width = 0
        self._height = 0
        self._start = (0, 0)

        self._walls: set[tuple[int, int]] = set()
        self._hazards: set[tuple[int, int]] = set()
        self._exit_cells: set[tuple[int, int]] = set()

        self._door_a_positions: set[tuple[int, int]] = set()
        self._door_b_positions: set[tuple[int, int]] = set()
        self._door_a_initial_closed: set[tuple[int, int]] = set()
        self._door_b_initial_closed: set[tuple[int, int]] = set()
        self._door_a_parity = 0
        self._door_b_parity = 0

        self._switches: list[dict[str, object]] = []
        self._switch_by_cell: dict[tuple[int, int], str] = {}
        self._last_switch_name: str | None = None

        self._player: Sprite | None = None
        self._timebar: Sprite | None = None
        self._door_a_sprites: dict[tuple[int, int], Sprite] = {}
        self._door_b_sprites: dict[tuple[int, int], Sprite] = {}
        self._switch_sprites: dict[str, Sprite] = {}
        self._exit_sprites: list[Sprite] = []

    def on_set_level(self, level: Level) -> None:
        self._width = int(level.get_data("width"))
        self._height = int(level.get_data("height"))
        self._time_limit = int(level.get_data("time_limit"))
        self._time_remaining = self._time_limit
        self._start = tuple(int(v) for v in level.get_data("start"))

        self._walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        self._hazards = {tuple(int(v) for v in item) for item in (level.get_data("hazards") or [])}
        self._exit_cells = {tuple(int(v) for v in item) for item in (level.get_data("exit_cells") or [])}

        self._door_a_positions = {tuple(int(v) for v in item) for item in (level.get_data("door_a_positions") or [])}
        self._door_b_positions = {tuple(int(v) for v in item) for item in (level.get_data("door_b_positions") or [])}
        self._door_a_initial_closed = {
            tuple(int(v) for v in item) for item in (level.get_data("door_a_initial_closed") or [])
        }
        self._door_b_initial_closed = {
            tuple(int(v) for v in item) for item in (level.get_data("door_b_initial_closed") or [])
        }

        self._door_a_parity = 0
        self._door_b_parity = 0

        raw_switches = list(level.get_data("switches") or [])
        self._switches = [
            {
                "name": str(s.get("name")),
                "kind": str(s.get("kind", "")).upper(),
                "cells": [(int(px), int(py)) for px, py in (s.get("cells") or [])],
            }
            for s in raw_switches
        ]
        self._switch_by_cell = _switch_name_by_cell(self._switches)

        self._phase = "play"
        self._phase_ticks = 0
        self._portal_pulse_on = False

        self._player = next(iter(level.get_sprites_by_name("player")), None)
        self._timebar = next(iter(level.get_sprites_by_name("timebar")), None)
        self._door_a_sprites = {(int(sprite.x), int(sprite.y)): sprite for sprite in level.get_sprites_by_tag("door_a")}
        self._door_b_sprites = {(int(sprite.x), int(sprite.y)): sprite for sprite in level.get_sprites_by_tag("door_b")}
        self._switch_sprites = {str(sprite.name): sprite for sprite in level.get_sprites_by_tag("switch")}
        self._exit_sprites = list(level.get_sprites_by_tag("exit"))

        self._last_switch_name = None
        if self._player is not None:
            start_cell = (int(self._player.x), int(self._player.y))
            self._last_switch_name = self._switch_by_cell.get(start_cell)

        self._sync_doors()
        self._sync_switch_visuals()
        self._sync_player_visual()
        self._sync_exit_visuals()
        self._sync_timebar()

    def _door_a_closed(self, cell: tuple[int, int]) -> bool:
        return _door_closed(cell, initial_closed=self._door_a_initial_closed, parity=self._door_a_parity)

    def _door_b_closed(self, cell: tuple[int, int]) -> bool:
        return _door_closed(cell, initial_closed=self._door_b_initial_closed, parity=self._door_b_parity)

    def _sync_doors(self) -> None:
        for cell, sprite in self._door_a_sprites.items():
            closed = self._door_a_closed(cell)
            sprite.set_collidable(closed)
            sprite.pixels = np.array([[COLOR_DOOR_A_CLOSED if closed else COLOR_DOOR_A_OPEN]], dtype=np.int8)

        for cell, sprite in self._door_b_sprites.items():
            closed = self._door_b_closed(cell)
            sprite.set_collidable(closed)
            sprite.pixels = np.array([[COLOR_DOOR_B_CLOSED if closed else COLOR_DOOR_B_OPEN]], dtype=np.int8)

    def _sync_switch_visuals(self) -> None:
        current = None
        if self._player is not None:
            current = self._switch_by_cell.get((int(self._player.x), int(self._player.y)))

        for switch in self._switches:
            name = str(switch["name"])
            kind = str(switch["kind"]).upper()
            sprite = self._switch_sprites.get(name)
            if sprite is None:
                continue
            is_pressed = current == name
            if kind == "A":
                color = COLOR_SWITCH_A_PRESSED if is_pressed else COLOR_SWITCH_A_IDLE
            else:
                color = COLOR_SWITCH_B_PRESSED if is_pressed else COLOR_SWITCH_B_IDLE
            sprite.pixels = np.array([[color, color]], dtype=np.int8)

    def _sync_player_visual(self) -> None:
        if self._player is None:
            return
        color = COLOR_PLAYER
        if self._phase == "fail":
            color = COLOR_HAZARD if self._phase_ticks % 2 == 0 else COLOR_PLAYER
        self._player.pixels = np.array([[color]], dtype=np.int8)

    def _sync_exit_visuals(self) -> None:
        pulse = self._portal_pulse_on
        if self._phase == "win" and (self._phase_ticks % 2 == 1):
            pulse = True
        color = COLOR_EXIT_PULSE if pulse else COLOR_EXIT_BASE
        for sprite in self._exit_sprites:
            sprite.pixels = np.array([[color]], dtype=np.int8)

    def _sync_timebar(self) -> None:
        if self._timebar is None:
            return
        width = self._width
        segments = max(0, width - 2)
        if self._time_limit <= 0:
            fill = 0
        else:
            fill = math.ceil((self._time_remaining * segments) / float(self._time_limit))
        fill = max(0, min(segments, fill))

        row = [COLOR_TIMEBAR_EMPTY] * width
        if width > 0:
            row[0] = COLOR_WALL
        if width > 1:
            row[-1] = COLOR_WALL
        for i in range(segments):
            row[1 + i] = COLOR_TIMEBAR_FILLED if i < fill else COLOR_TIMEBAR_EMPTY

        self._timebar.pixels = np.array([row], dtype=np.int8)

    def _cell_blocked(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self._width or y >= self._height:
            return True
        if y == 0:
            return True
        cell = (x, y)
        if cell in self._walls:
            return True
        if cell in self._door_a_positions and self._door_a_closed(cell):
            return True
        return bool(cell in self._door_b_positions and self._door_b_closed(cell))

    def _try_move_player(self, action: GameAction) -> None:
        if self._player is None:
            return
        delta = MOVE_DELTAS.get(action)
        if delta is None:
            return
        px, py = int(self._player.x), int(self._player.y)
        nx, ny = px + delta[0], py + delta[1]
        if self._cell_blocked(nx, ny):
            return
        self._player.set_position(nx, ny)

    def _toggle_switch(self, switch_name: str) -> None:
        switch = next((s for s in self._switches if str(s["name"]) == switch_name), None)
        if switch is None:
            return
        kind = str(switch["kind"]).upper()
        if kind == "A":
            self._door_a_parity ^= 1
        elif kind == "B":
            self._door_b_parity ^= 1
        self._sync_doors()

    def _resolve_tile_effects(self) -> None:
        if self._player is None:
            self._enter_fail_state()
            return
        pos = (int(self._player.x), int(self._player.y))

        if pos in self._hazards:
            self._enter_fail_state()
            return

        current_switch = self._switch_by_cell.get(pos)
        if current_switch is not None and current_switch != self._last_switch_name:
            self._toggle_switch(current_switch)
        self._last_switch_name = current_switch

        if pos in self._exit_cells:
            self._enter_win_state()

    def _enter_fail_state(self) -> None:
        if self._phase != "play":
            return
        self._phase = "fail"
        self._phase_ticks = FAIL_ANIMATION_STEPS

    def _enter_win_state(self) -> None:
        if self._phase != "play":
            return
        self._phase = "win"
        self._phase_ticks = WIN_ANIMATION_STEPS

    def _advance_animations(self) -> bool:
        self._portal_pulse_on = not self._portal_pulse_on

        if self._phase == "fail":
            self._phase_ticks -= 1
            if self._phase_ticks <= 0:
                self.lose()
                return True
        elif self._phase == "win":
            self._phase_ticks -= 1
            if self._phase_ticks <= 0:
                self.next_level()
                return True

        self._sync_switch_visuals()
        self._sync_player_visual()
        self._sync_exit_visuals()
        return False

    def step(self) -> None:
        if getattr(self.action.id, "value", self.action.id) == 0 and self.level_index > 0:
            # arcengine's perform_action loop runs step() once after
            # handle_reset; without this guard the sim would advance a
            # tick, so mid-play RESET on any level entered via
            # next_level() would land one tick past the frame the
            # client saw on arrival. Level 0 keeps the legacy tick to
            # preserve env.reset()'s observation and the DSL trace.
            self.complete_action()
            return

        action = self.action.id

        if action == GameAction.ACTION5:
            self.level_reset()
            self.complete_action()
            return

        if self._phase == "play" and action in MOVE_DELTAS:
            self._try_move_player(action)

        if self._phase == "play":
            self._resolve_tile_effects()

        transitioned = self._advance_animations()
        if transitioned:
            self.complete_action()
            return

        if self._phase == "play":
            self._time_remaining -= 1
            if self._time_remaining <= 0:
                self._enter_fail_state()

        self._sync_timebar()
        self.complete_action()
