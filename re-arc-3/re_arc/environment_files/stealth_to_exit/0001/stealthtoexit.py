from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "stealth_to_exit-0001"
VARIANT = "0001"

# Color ids from the spec.
C_VOID = 0
C_WALL = 1
C_FLOOR = 2
C_PLAYER = 3
C_GUARD_BODY = 4
C_GUARD_HEAD = 5
C_VISION = 6
C_EXIT = 7
C_TIME_FILL = 8
C_TIME_EMPTY = 9
C_HIDE = 10
C_DOOR = 11
C_KEY = 12
C_NOISE = 13
C_ALERT = 14
C_BLINK = 15

MOVE_ACTIONS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

NOISE_FRAMES = 3
DOOR_OPEN_ANIM_STEPS = 3
CAUGHT_ANIM_STEPS = 4


@dataclass
class GuardState:
    head: tuple[int, int]
    facing: tuple[int, int]
    behavior: str
    mode: str
    anchor_head: tuple[int, int]
    anchor_facing: tuple[int, int]
    patrol_min: int | None = None
    patrol_max: int | None = None
    patrol_y: int | None = None
    patrol_pause: int = 0
    patrol_dir: int = 1
    pause_left: int = 0
    reverse_after_pause: bool = False
    investigate_target: tuple[int, int] | None = None
    investigate_steps_left: int = 0


LEVEL_SPECS = [
    {
        "name": "Stealth To Exit L1",
        "layout": [
            "========================",
            "########################",
            "#@.....................#",
            "#...........Gg.....EE..#",
            "#####.#########.###EE###",
            "#####.#########.########",
            "#####.#########.########",
            "#####.#########.########",
            "#####.#########.########",
            "#####.#########.########",
            "#####...........########",
            "########################",
            "########################",
            "########################",
        ],
        "vision_range": 6,
        "time_per_cell": 3,
        "noise_max_dist": 8,
        "hearing_radius": 7,
        "investigate_timeout": 10,
        "guards": [{"behavior": "sentry"}],
    },
    {
        "name": "Stealth To Exit L2",
        "layout": [
            "========================",
            "########################",
            "##################.EE###",
            "##################.EE###",
            "##################.#####",
            "##################.#####",
            "##################.#####",
            "###...........gG...#####",
            "###.####################",
            "###.####################",
            "###.####################",
            "###@####################",
            "########################",
            "########################",
        ],
        "vision_range": 6,
        "time_per_cell": 3,
        "noise_max_dist": 8,
        "hearing_radius": 7,
        "investigate_timeout": 10,
        "guards": [{"behavior": "patrol", "patrol_min": 15, "patrol_max": 18, "patrol_y": 7, "patrol_pause": 1}],
    },
    {
        "name": "Stealth To Exit L3",
        "layout": [
            "========================",
            "########################",
            "########################",
            "########################",
            "########################",
            "########################",
            "###################EE###",
            "#@..........Gg.....EE..#",
            "#####.##################",
            "#####HH#################",
            "#####HH#################",
            "########################",
            "########################",
            "########################",
        ],
        "vision_range": 6,
        "time_per_cell": 3,
        "noise_max_dist": 8,
        "hearing_radius": 7,
        "investigate_timeout": 10,
        "guards": [{"behavior": "patrol", "patrol_min": 13, "patrol_max": 17, "patrol_y": 7, "patrol_pause": 0}],
    },
    {
        "name": "Stealth To Exit L4",
        "layout": [
            "========================",
            "########################",
            "########################",
            "########################",
            "########################",
            "########################",
            "####################EE##",
            "###.........Gg......EE.#",
            "#.....##################",
            "#..@..##################",
            "#.....##################",
            "########################",
            "########################",
            "########################",
        ],
        "vision_range": 7,
        "time_per_cell": 3,
        "noise_max_dist": 10,
        "hearing_radius": 8,
        "investigate_timeout": 12,
        "guards": [{"behavior": "sentry"}],
    },
    {
        "name": "Stealth To Exit L5",
        "layout": [
            "============================",
            "############################",
            "#.........##################",
            "#....gG...##################",
            "#....*....##################",
            "#.........##################",
            "#########.##################",
            "#####.###.##################",
            "#####............Gg+....EE##",
            "#####..............+....EE##",
            "#####.###.##HH##############",
            "#####.###.##HH##############",
            "#.........##################",
            "#.@.......##################",
            "#.........##################",
            "############################",
        ],
        "vision_range": 7,
        "time_per_cell": 3,
        "noise_max_dist": 10,
        "hearing_radius": 8,
        "investigate_timeout": 14,
        "guards": [
            {"behavior": "patrol", "patrol_min": 4, "patrol_max": 9, "patrol_y": 3, "patrol_pause": 1},
            {"behavior": "sentry"},
        ],
    },
    {
        "name": "Stealth To Exit L6",
        "layout": [
            "================================",
            "################################",
            "###########################.EE##",
            "###########################.EE##",
            "############HH......#######.####",
            "############HH.gG.*.#######.####",
            "############........#######.####",
            "###############.###########.####",
            "###############.###########.####",
            "###############.###########.####",
            "#.......................Gg.+####",
            "#............gG............+####",
            "#######.#############HH#########",
            "#######.#############HH#########",
            "#.........######################",
            "#.@.......######################",
            "#.........######################",
            "################################",
        ],
        "vision_range": 8,
        "time_per_cell": 3,
        "noise_max_dist": 12,
        "hearing_radius": 9,
        "investigate_timeout": 16,
        "guards": [
            {"behavior": "patrol", "patrol_min": 16, "patrol_max": 20, "patrol_y": 5, "patrol_pause": 1},
            {"behavior": "patrol", "patrol_min": 10, "patrol_max": 22, "patrol_y": 11, "patrol_pause": 0},
            {"behavior": "sentry"},
        ],
    },
]

ACTIVE_LEVEL_COUNT = 1


def _blank(w: int, h: int, color: int) -> np.ndarray:
    return np.full((h, w), int(color), dtype=np.int8)


def _build_levels() -> list[Level]:
    levels: list[Level] = []
    for idx, spec in enumerate(LEVEL_SPECS[:ACTIVE_LEVEL_COUNT]):
        layout = spec["layout"]
        width = len(layout[0])
        height = len(layout)
        board = Sprite(
            _blank(width, height, C_FLOOR),
            name="board",
            x=0,
            y=0,
            layer=0,
            tags=["board", "sys_click", "sys_every_pixel"],
            collidable=False,
        )
        levels.append(
            Level(
                name=str(spec["name"]),
                grid_size=(width, height),
                sprites=[board],
                data={"spec_index": idx, "spec": spec},
            )
        )
    return levels


class StealthToExit(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._board: Sprite | None = None

        self._w = 0
        self._h = 0
        self._step_count = 0

        self._vision_range = 6
        self._time_per_cell = 3
        self._time_max = 1
        self._time_remaining = 1

        self._noise_max_dist = 8
        self._hearing_radius = 7
        self._investigate_timeout = 10

        self._walls: set[tuple[int, int]] = set()
        self._floors: set[tuple[int, int]] = set()
        self._exits: set[tuple[int, int]] = set()
        self._hiding: set[tuple[int, int]] = set()
        self._hiding_groups: list[set[tuple[int, int]]] = []
        self._keys: set[tuple[int, int]] = set()
        self._door_tiles: set[tuple[int, int]] = set()

        self._player = (0, 0)
        self._has_key = False
        self._door_open = False
        self._door_opening_steps = 0

        self._guards: list[GuardState] = []

        self._noise_event: dict[str, object] | None = None
        self._caught_anim_steps = 0

        self._vision_tiles: set[tuple[int, int]] = set()
        self._vision_overlap: set[tuple[int, int]] = set()

        levels = _build_levels()
        camera = Camera(width=levels[0].grid_size[0], height=levels[0].grid_size[1], background=C_VOID)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        board = level.get_sprites_by_name("board")
        if not board:
            raise RuntimeError("stealth_to_exit: missing board sprite")
        self._board = board[0]

        spec = level.get_data("spec")
        if not isinstance(spec, dict):
            raise RuntimeError("stealth_to_exit: missing level spec")
        layout = spec["layout"]
        self._w = len(layout[0])
        self._h = len(layout)
        self._vision_range = int(spec["vision_range"])
        self._time_per_cell = int(spec["time_per_cell"])
        self._time_max = self._w * self._time_per_cell
        self._time_remaining = self._time_max
        self._noise_max_dist = int(spec["noise_max_dist"])
        self._hearing_radius = int(spec["hearing_radius"])
        self._investigate_timeout = int(spec["investigate_timeout"])

        self._step_count = 0
        self._has_key = False
        self._door_open = False
        self._door_opening_steps = 0
        self._noise_event = None
        self._caught_anim_steps = 0

        (
            self._walls,
            self._floors,
            self._exits,
            self._hiding,
            self._hiding_groups,
            self._keys,
            self._door_tiles,
            self._player,
            guard_starts,
        ) = self._parse_layout(layout)
        self._guards = self._build_guards(guard_starts, spec["guards"])

        self._recompute_vision()
        self._render()

    def _parse_layout(
        self, layout: list[str]
    ) -> tuple[
        set[tuple[int, int]],
        set[tuple[int, int]],
        set[tuple[int, int]],
        set[tuple[int, int]],
        list[set[tuple[int, int]]],
        set[tuple[int, int]],
        set[tuple[int, int]],
        tuple[int, int],
        list[tuple[tuple[int, int], tuple[int, int]]],
    ]:
        walls: set[tuple[int, int]] = set()
        floors: set[tuple[int, int]] = set()
        exits: set[tuple[int, int]] = set()
        hiding: set[tuple[int, int]] = set()
        keys: set[tuple[int, int]] = set()
        doors: set[tuple[int, int]] = set()
        player = (1, 1)

        grid = [list(row) for row in layout]
        used_guard_tiles: set[tuple[int, int]] = set()
        guards: list[tuple[tuple[int, int], tuple[int, int]]] = []

        for y, row in enumerate(grid):
            for x, ch in enumerate(row):
                if y == 0:
                    continue
                if ch == "#":
                    walls.add((x, y))
                else:
                    floors.add((x, y))

                if ch == "@":
                    player = (x, y)
                elif ch == "E":
                    exits.add((x, y))
                elif ch == "H":
                    hiding.add((x, y))
                elif ch == "*":
                    keys.add((x, y))
                elif ch == "+":
                    doors.add((x, y))
                elif ch == "G":
                    if (x, y) in used_guard_tiles:
                        continue
                    face = None
                    body = None
                    if x > 0 and grid[y][x - 1] == "g" and (x - 1, y) not in used_guard_tiles:
                        face = (1, 0)
                        body = (x - 1, y)
                    elif x + 1 < self._w and grid[y][x + 1] == "g" and (x + 1, y) not in used_guard_tiles:
                        face = (-1, 0)
                        body = (x + 1, y)
                    elif y > 0 and grid[y - 1][x] == "g" and (x, y - 1) not in used_guard_tiles:
                        face = (0, 1)
                        body = (x, y - 1)
                    elif y + 1 < self._h and grid[y + 1][x] == "g" and (x, y + 1) not in used_guard_tiles:
                        face = (0, -1)
                        body = (x, y + 1)
                    if face is not None and body is not None:
                        used_guard_tiles.add((x, y))
                        used_guard_tiles.add(body)
                        guards.append(((x, y), face))

        hiding_groups = self._connected_groups(hiding)
        return walls, floors, exits, hiding, hiding_groups, keys, doors, player, guards

    def _connected_groups(self, points: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
        groups: list[set[tuple[int, int]]] = []
        unseen = set(points)
        while unseen:
            start = unseen.pop()
            q = deque([start])
            group = {start}
            while q:
                x, y = q.popleft()
                for dx, dy in DIRS:
                    nxt = (x + dx, y + dy)
                    if nxt in unseen:
                        unseen.remove(nxt)
                        group.add(nxt)
                        q.append(nxt)
            groups.append(group)
        return groups

    def _build_guards(
        self, starts: list[tuple[tuple[int, int], tuple[int, int]]], behavior_specs: list[dict[str, object]]
    ) -> list[GuardState]:
        if len(starts) != len(behavior_specs):
            raise RuntimeError("stealth_to_exit: guard count mismatch")
        guards: list[GuardState] = []
        for idx, (head, facing) in enumerate(starts):
            conf = behavior_specs[idx]
            behavior = str(conf["behavior"])
            patrol_dir = 1 if facing[0] >= 0 else -1
            guards.append(
                GuardState(
                    head=head,
                    facing=facing,
                    behavior=behavior,
                    mode=behavior,
                    anchor_head=head,
                    anchor_facing=facing,
                    patrol_min=int(conf["patrol_min"]) if "patrol_min" in conf else None,
                    patrol_max=int(conf["patrol_max"]) if "patrol_max" in conf else None,
                    patrol_y=int(conf["patrol_y"]) if "patrol_y" in conf else None,
                    patrol_pause=int(conf.get("patrol_pause", 0)),
                    patrol_dir=patrol_dir,
                )
            )
        return guards

    def _in_bounds_play(self, x: int, y: int) -> bool:
        return 0 <= x < self._w and 1 <= y < self._h

    def _door_blocks(self, pos: tuple[int, int]) -> bool:
        if self._door_open:
            return False
        return pos in self._door_tiles

    def _is_wall_or_door(self, pos: tuple[int, int]) -> bool:
        if pos in self._walls:
            return True
        return bool(self._door_blocks(pos))

    def _is_player_passable(self, pos: tuple[int, int]) -> bool:
        if not self._in_bounds_play(pos[0], pos[1]):
            return False
        return not self._is_wall_or_door(pos)

    def _is_guard_passable(self, pos: tuple[int, int]) -> bool:
        if not self._in_bounds_play(pos[0], pos[1]):
            return False
        return not self._is_wall_or_door(pos)

    def _is_hidden(self) -> bool:
        return self._player in self._hiding

    def _decode_click(self) -> tuple[int, int] | None:
        data = self.action.data if isinstance(self.action.data, dict) else {}
        try:
            raw_x = int(data.get("x", -9999))
            raw_y = int(data.get("y", -9999))
        except (TypeError, ValueError):
            return None

        scale, x_off, y_off = self.camera._calculate_scale_and_offset()
        if scale > 0:
            gx = int((raw_x - x_off) // scale + self.camera.x)
            gy = int((raw_y - y_off) // scale + self.camera.y)
            if self._in_bounds_play(gx, gy):
                return gx, gy

        if self._in_bounds_play(raw_x, raw_y):
            return raw_x, raw_y
        return None

    def _player_action(self) -> tuple[int, int] | None:
        action_id = int(getattr(self.action.id, "value", self.action.id))

        if action_id in MOVE_ACTIONS:
            dx, dy = MOVE_ACTIONS[action_id]
            nx = self._player[0] + dx
            ny = self._player[1] + dy
            if self._is_player_passable((nx, ny)):
                self._player = (nx, ny)
                if self._player in self._keys:
                    self._keys.remove(self._player)
                    self._has_key = True
            return None

        if action_id == int(GameAction.ACTION5.value):
            if self._has_key and not self._door_open and self._door_opening_steps <= 0:
                px, py = self._player
                for dx, dy in DIRS:
                    if (px + dx, py + dy) in self._door_tiles:
                        self._door_opening_steps = DOOR_OPEN_ANIM_STEPS
                        break
            return None

        if action_id == int(GameAction.ACTION6.value):
            click = self._decode_click()
            if click is None:
                return None
            if abs(click[0] - self._player[0]) + abs(click[1] - self._player[1]) > self._noise_max_dist:
                return None
            if not self._is_guard_passable(click):
                return None
            return click

        return None

    def _advance_timebar(self) -> None:
        self._time_remaining = max(0, self._time_remaining - 1)

    def _advance_dynamic_effects(self, pending_noise_center: tuple[int, int] | None) -> None:
        if pending_noise_center is not None:
            self._noise_event = {"center": pending_noise_center, "age": -1}

        if self._noise_event is not None:
            age = int(self._noise_event["age"]) + 1
            if age >= NOISE_FRAMES:
                self._noise_event = None
            else:
                self._noise_event["age"] = age

        if self._door_opening_steps > 0:
            self._door_opening_steps -= 1
            if self._door_opening_steps <= 0:
                self._door_open = True

    def _bfs_next_step(self, start: tuple[int, int], goal: tuple[int, int]) -> tuple[int, int] | None:
        if start == goal:
            return None
        q = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for dx, dy in DIRS:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in parent:
                    continue
                if not self._is_guard_passable(nxt):
                    continue
                parent[nxt] = cur
                q.append(nxt)

        if goal not in parent:
            return None

        cur = goal
        while parent[cur] != start:
            prev = parent[cur]
            if prev is None:
                return None
            cur = prev
        return cur

    def _start_investigate(self, guard: GuardState, center: tuple[int, int]) -> None:
        guard.mode = "investigate"
        guard.investigate_target = center
        guard.investigate_steps_left = self._investigate_timeout

    def _finish_investigate(self, guard: GuardState) -> None:
        if guard.behavior == "sentry":
            guard.mode = "return"
        else:
            guard.mode = guard.behavior
            guard.investigate_target = None
            guard.investigate_steps_left = 0

    def _update_guard_patrol(self, guard: GuardState) -> None:
        if guard.patrol_min is None or guard.patrol_max is None or guard.patrol_y is None:
            return
        if guard.pause_left > 0:
            guard.pause_left -= 1
            if guard.pause_left <= 0 and guard.reverse_after_pause:
                guard.patrol_dir *= -1
                guard.facing = (guard.patrol_dir, 0)
                guard.reverse_after_pause = False
            return

        hx, hy = guard.head
        if hy != guard.patrol_y:
            guard.head = (hx, guard.patrol_y)
            hy = guard.patrol_y

        if (guard.patrol_dir > 0 and hx >= guard.patrol_max) or (guard.patrol_dir < 0 and hx <= guard.patrol_min):
            if guard.patrol_pause > 0:
                guard.pause_left = guard.patrol_pause
                guard.reverse_after_pause = True
                return
            guard.patrol_dir *= -1

        nx = hx + guard.patrol_dir
        if nx < guard.patrol_min:
            nx = guard.patrol_min
        if nx > guard.patrol_max:
            nx = guard.patrol_max

        if self._is_guard_passable((nx, hy)):
            guard.head = (nx, hy)
            guard.facing = (guard.patrol_dir, 0)

    def _update_guard_investigate(self, guard: GuardState) -> None:
        target = guard.investigate_target
        if target is None:
            self._finish_investigate(guard)
            return

        if guard.head == target:
            self._finish_investigate(guard)
            return

        if guard.investigate_steps_left <= 0:
            self._finish_investigate(guard)
            return

        nxt = self._bfs_next_step(guard.head, target)
        guard.investigate_steps_left -= 1
        if nxt is None:
            if guard.investigate_steps_left <= 0:
                self._finish_investigate(guard)
            return

        dx = nxt[0] - guard.head[0]
        dy = nxt[1] - guard.head[1]
        if (dx, dy) in DIRS:
            guard.facing = (dx, dy)
            guard.head = nxt

    def _update_guard_return(self, guard: GuardState) -> None:
        if guard.head == guard.anchor_head:
            guard.mode = "sentry"
            guard.facing = guard.anchor_facing
            guard.investigate_target = None
            guard.investigate_steps_left = 0
            return

        nxt = self._bfs_next_step(guard.head, guard.anchor_head)
        if nxt is None:
            guard.mode = "sentry"
            guard.facing = guard.anchor_facing
            guard.head = guard.anchor_head
            return

        dx = nxt[0] - guard.head[0]
        dy = nxt[1] - guard.head[1]
        if (dx, dy) in DIRS:
            guard.facing = (dx, dy)
            guard.head = nxt

    def _update_guards(self) -> None:
        noise_center = None
        if self._noise_event is not None:
            noise_center = self._noise_event.get("center")
            if not isinstance(noise_center, tuple):
                noise_center = None

        for guard in self._guards:
            if noise_center is not None:
                hx, hy = guard.head
                if abs(hx - noise_center[0]) + abs(hy - noise_center[1]) <= self._hearing_radius:
                    self._start_investigate(guard, noise_center)

            if guard.mode == "patrol":
                self._update_guard_patrol(guard)
            elif guard.mode == "investigate":
                self._update_guard_investigate(guard)
            elif guard.mode == "return":
                self._update_guard_return(guard)

    def _guard_tiles(self, guard: GuardState) -> set[tuple[int, int]]:
        hx, hy = guard.head
        fx, fy = guard.facing
        bx, by = hx - fx, hy - fy
        out = {(hx, hy)}
        if self._in_bounds_play(bx, by):
            out.add((bx, by))
        return out

    def _line_clear(self, start: tuple[int, int], target: tuple[int, int]) -> bool:
        x0, y0 = start
        x1, y1 = target

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while (x, y) != (x1, y1):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            if (x, y) == (x1, y1):
                break
            if self._is_wall_or_door((x, y)):
                return False

        return not self._is_wall_or_door((x1, y1))

    def _guard_vision_tiles(self, guard: GuardState) -> set[tuple[int, int]]:
        hx, hy = guard.head
        fx, fy = guard.facing
        out: set[tuple[int, int]] = set()
        for dist in range(1, self._vision_range + 1):
            spread = dist // 2
            for lateral in range(-spread, spread + 1):
                if fx != 0:
                    tx = hx + fx * dist
                    ty = hy + lateral
                else:
                    tx = hx + lateral
                    ty = hy + fy * dist
                if not self._in_bounds_play(tx, ty):
                    continue
                if not self._line_clear((hx, hy), (tx, ty)):
                    continue
                out.add((tx, ty))
        return out

    def _recompute_vision(self) -> None:
        counts: dict[tuple[int, int], int] = {}
        for guard in self._guards:
            for tile in self._guard_vision_tiles(guard):
                counts[tile] = counts.get(tile, 0) + 1
        self._vision_tiles = set(counts.keys())
        self._vision_overlap = {tile for tile, c in counts.items() if c > 1}

    def _noise_tiles(self) -> set[tuple[int, int]]:
        if self._noise_event is None:
            return set()
        center = self._noise_event.get("center")
        age = int(self._noise_event.get("age", 99))
        if not isinstance(center, tuple) or age < 0 or age >= NOISE_FRAMES:
            return set()

        cx, cy = center
        tiles: set[tuple[int, int]] = set()
        radius = age
        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                if not self._in_bounds_play(x, y):
                    continue
                if abs(x - cx) + abs(y - cy) == radius:
                    tiles.add((x, y))
        return tiles

    def _occupied_hiding_corner(self) -> tuple[int, int] | None:
        if self._player not in self._hiding:
            return None
        for group in self._hiding_groups:
            if self._player in group:
                return min(group, key=lambda p: (p[1], p[0]))
        return None

    def _timebar_fill_cells(self) -> int:
        if self._time_per_cell <= 0:
            return 0
        return int(np.ceil(float(self._time_remaining) / float(self._time_per_cell)))

    def _resolve_outcome(self) -> str | None:
        hidden = self._is_hidden()

        if (not hidden) and (self._player in self._vision_tiles):
            self._caught_anim_steps = CAUGHT_ANIM_STEPS
            return "caught"

        if self._player in self._exits:
            return "win"

        if self._time_remaining <= 0:
            return "timeout"

        return None

    def _render(self) -> None:
        if self._board is None:
            return

        frame = np.full((self._h, self._w), C_FLOOR, dtype=np.int8)

        for x in range(self._w):
            frame[0, x] = C_TIME_EMPTY

        for x, y in self._walls:
            frame[y, x] = C_WALL

        exit_color = C_BLINK if (self._step_count % 2 == 0) else C_EXIT
        for x, y in self._exits:
            frame[y, x] = exit_color

        for x, y in self._hiding:
            frame[y, x] = C_HIDE

        occupied_corner = self._occupied_hiding_corner()
        if occupied_corner is not None and (self._step_count % 2 == 0):
            frame[occupied_corner[1], occupied_corner[0]] = C_BLINK

        for x, y in self._keys:
            frame[y, x] = C_KEY

        if not self._door_open:
            door_color = C_DOOR
            if self._door_opening_steps > 0 and (self._step_count % 2 == 0):
                door_color = C_BLINK
            for x, y in self._door_tiles:
                frame[y, x] = door_color

        for x, y in self._vision_tiles:
            frame[y, x] = C_ALERT if (x, y) in self._vision_overlap else C_VISION

        for x, y in self._noise_tiles():
            frame[y, x] = C_NOISE

        alert_phase = self._caught_anim_steps > 0 and (self._step_count % 2 == 0)
        for guard in self._guards:
            gx, gy = guard.head
            bx, by = gx - guard.facing[0], gy - guard.facing[1]
            if self._in_bounds_play(bx, by):
                frame[by, bx] = C_ALERT if alert_phase else C_GUARD_BODY
            if self._in_bounds_play(gx, gy):
                frame[gy, gx] = C_ALERT if alert_phase else C_GUARD_HEAD

        px, py = self._player
        frame[py, px] = C_ALERT if alert_phase else C_PLAYER

        fill = max(0, min(self._w, self._timebar_fill_cells()))
        for x in range(self._w):
            frame[0, x] = C_TIME_FILL if x < fill else C_TIME_EMPTY

        self._board.pixels = frame

    def step(self) -> None:
        if self._caught_anim_steps > 0:
            self._caught_anim_steps -= 1
            self._step_count += 1
            if self._caught_anim_steps <= 0:
                self.lose()
                self.complete_action()
                return
            self._render()
            self.complete_action()
            return

        pending_noise_center = self._player_action()
        self._advance_timebar()
        self._advance_dynamic_effects(pending_noise_center)
        self._update_guards()
        self._recompute_vision()

        outcome = self._resolve_outcome()
        self._step_count += 1
        self._render()

        if outcome == "win":
            self.next_level()
        elif outcome == "timeout":
            self.lose()

        self.complete_action()
