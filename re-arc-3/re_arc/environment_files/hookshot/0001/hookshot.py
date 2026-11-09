from __future__ import annotations

from collections import deque

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "hookshot"

COLOR_FLOOR = 0
COLOR_WALL = 1
COLOR_TIME_EMPTY = 2
COLOR_HIGHLIGHT = 3
COLOR_DOOR_CLOSED = 4
COLOR_SWITCH = 5
COLOR_HOOK = 6
COLOR_TETHER = 7
COLOR_LAVA = 8
COLOR_SPIKES = 9
COLOR_TIME_FILL = 10
COLOR_EXIT = 11
COLOR_PLAYER_UP = 12
COLOR_PLAYER_RIGHT = 13
COLOR_PLAYER_DOWN = 14
COLOR_PLAYER_LEFT = 15

FACING_TO_DELTA = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

ACTION_TO_FACING = {
    int(GameAction.ACTION1.value): 0,
    int(GameAction.ACTION4.value): 1,
    int(GameAction.ACTION2.value): 2,
    int(GameAction.ACTION3.value): 3,
}

FACING_TO_COLOR = {0: COLOR_PLAYER_UP, 1: COLOR_PLAYER_RIGHT, 2: COLOR_PLAYER_DOWN, 3: COLOR_PLAYER_LEFT}


LEVEL_LAYOUTS: list[list[str]] = [
    [
        "============================",
        "############################",
        "#.>.......~~~~~~*..........#",
        "#.........~~~~~~...........#",
        "#.........~~~~~~...........#",
        "#.........~~~~~~...........#",
        "#.........~~~~~~...........#",
        "#.........~~~~~~...........#",
        "#.........~~~~~~...........#",
        "#.........~~~~~~.......$$..#",
        "#.........~~~~~~.......$$..#",
        "#.........~~~~~~...........#",
        "#.........~~~~~~...........#",
        "############################",
    ],
    [
        "================================",
        "################################",
        "#...........#~~~~..............#",
        "#...........#~~~~..........$$..#",
        "#...........#~~~~..........$$..#",
        "#...........#~~~~..............#",
        "#...........#~~~~..............#",
        "#...........#~~~~..............#",
        "#............~*.~.*............#",
        "#...........#~..~..............#",
        "#...........#~~~~..............#",
        "#...........#~~~~..............#",
        "#...........#~~~~..............#",
        "#..>........#~~~~..............#",
        "#...........#~~~~..............#",
        "################################",
    ],
    [
        "================================",
        "################################",
        "#.........!!!!!!!!!!!!!!.......#",
        "#.........!!!!!!!!!!!!!!....$$.#",
        "#.........!!!!!!!!!!!!!!....$$.#",
        "#.........!!!!!!!!!!!!!!.......#",
        "#..>......!!!!!*!!!!!!!!.......#",
        "#.........!!!!!!!!!!!!!!.......#",
        "#.........!!!!!!!!!!!!!!.......#",
        "#.........!!!!!!!!!!!!!!.......#",
        "#.........!!!!!!!!!!!!!!.......#",
        "#.........!!!!!!!!!!!!!!.......#",
        "#.........!!!!!!!!!!!!!!*......#",
        "#.........!!!!!!!!!!!!!!.......#",
        "#.........!!!!!!!!!!!!!!.......#",
        "################################",
    ],
    [
        "================================",
        "################################",
        "#.....!!!!!!!!!!!!!!!!!!.......#",
        "#.....!!!!!!!!!!!!!!!!!!....$$.#",
        "#.....!!!!!!!!!!!!!!!!!!....$$.#",
        "#.....!!!!!!!!!!!!!!!!!!.......#",
        "#.....!!!!!!!!!!%!!!!!!!.......#",
        "#.....!!!...!!!!.!!!!..........#",
        "#..>..!!!.*.!!!!*!!!!.*........#",
        "#.....!!!...!!!!.!!!!..........#",
        "#.....!!!!!!!!!!.!!!!!!!.......#",
        "#.....!!!!!!!!!!!!!!!!!!.......#",
        "#.....!!!!!!!!!!!!!!!!!!.......#",
        "#.....!!!!!!!!!!!!!!!!!!.......#",
        "#.....!!!!!!!!!!!!!!!!!!.......#",
        "################################",
    ],
    [
        "================================",
        "################################",
        "#...............#..............#",
        "#...............#...........$$.#",
        "#...~~~~~~~~~~..#...........$$.#",
        "#...~~~~~~~~~~..#..............#",
        "#...~~~~...~~~..#..............#",
        "#...~~~~.*+~~~..#......*.......#",
        "#...~~~~...~~~..#!!!!!!!!!!!!!!#",
        "#...............#!!!!!!!!!!!!!!#",
        "#...............#!!!!!!!!!!!!!!#",
        "#...............&&.............#",
        "#...............&&.............#",
        "#...............#..............#",
        "#...>...........#..............#",
        "#...............#..............#",
        "#...............#..............#",
        "################################",
    ],
    [
        "====================================",
        "####################################",
        "#................##...........##...#",
        "#................##...........##.$$#",
        "#...~~~~~~~~~~~..##...........&&.$$#",
        "#...~~~~~~%~~~~..##...........&&...#",
        "#..*~~~~~~*..~~..##...........##...#",
        "#...~~~~~~.+.~~..##...........##...#",
        "#...~~~~~~...~~..##.!!!!!!!!!!##...#",
        "#................##.!!!!!!!!!!##...#",
        "#................##*!!!*..!!!!##...#",
        "#................##.!!!.+.!!!!##...#",
        "#................##.!!!...!!!!##...#",
        "#................##.!!!!!!!!!!##...#",
        "#................##.!!!!!!!!!!##...#",
        "#................&&.!!!!!!!!!!##...#",
        "#................&&.!!!!!!!!!!##...#",
        "#..>.............##...........##...#",
        "#................##...........##...#",
        "####################################",
    ],
]


LEVEL_CONFIGS = [
    {"saw_ranges": [], "switch_links": []},
    {"saw_ranges": [], "switch_links": []},
    {"saw_ranges": [], "switch_links": []},
    {"saw_ranges": [((16, 6), 6, 10)], "switch_links": []},
    {"saw_ranges": [], "switch_links": [[0]]},
    {"saw_ranges": [((10, 5), 5, 8)], "switch_links": [[0], [1]]},
]


def _door_components(doors: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    components: list[list[tuple[int, int]]] = []
    remaining = set(doors)
    while remaining:
        start = min(remaining)
        queue = deque([start])
        component: list[tuple[int, int]] = []
        remaining.remove(start)
        while queue:
            x, y = queue.popleft()
            component.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
        components.append(sorted(component))
    components.sort(key=lambda cells: (min(x for x, _ in cells), min(y for _, y in cells)))
    return components


def _sanitize_row(row: str, width: int) -> str:
    cleaned = row.replace(" ", ".")
    if len(cleaned) != width:
        raise ValueError(f"layout width mismatch expected={width} got={len(cleaned)} row={cleaned!r}")
    return cleaned


def _parse_level(layout: list[str], config: dict, index: int) -> dict:
    if not layout:
        raise ValueError("hookshot level layout is empty")
    width = len(layout[0])
    height = len(layout)
    for row in layout:
        if len(row) != width:
            raise ValueError("hookshot rows must have equal width")

    walls: set[tuple[int, int]] = set()
    hooks: set[tuple[int, int]] = set()
    lava: set[tuple[int, int]] = set()
    spikes: set[tuple[int, int]] = set()
    exits: set[tuple[int, int]] = set()
    switches: list[tuple[int, int]] = []
    saw_starts: list[tuple[int, int]] = []
    door_cells: set[tuple[int, int]] = set()
    start_pos: tuple[int, int] | None = None
    start_facing = 1

    for y, raw_row in enumerate(layout):
        row = _sanitize_row(raw_row, width)
        for x, ch in enumerate(row):
            if y == 0:
                continue
            if ch == "#":
                walls.add((x, y))
            elif ch == "*":
                hooks.add((x, y))
            elif ch == "~":
                lava.add((x, y))
            elif ch == "!":
                spikes.add((x, y))
            elif ch == "$":
                exits.add((x, y))
            elif ch == "+":
                switches.append((x, y))
            elif ch == "%":
                saw_starts.append((x, y))
            elif ch == "&":
                door_cells.add((x, y))
            elif ch in "^>v<":
                if start_pos is not None:
                    raise ValueError("hookshot level has multiple starts")
                start_pos = (x, y)
                start_facing = {"^": 0, ">": 1, "v": 2, "<": 3}[ch]

    if start_pos is None:
        raise ValueError("hookshot level missing player start")
    if not exits:
        raise ValueError("hookshot level missing exit")

    doors = _door_components(door_cells)
    saw_specs: list[dict[str, int]] = []
    if len(config["saw_ranges"]) != len(saw_starts):
        raise ValueError(f"hookshot level {index + 1} saw config mismatch")
    for i, saw_start in enumerate(sorted(saw_starts)):
        expected_start, y_min, y_max = config["saw_ranges"][i]
        if tuple(expected_start) != saw_start:
            raise ValueError(f"hookshot level {index + 1} saw start mismatch expected={expected_start} got={saw_start}")
        saw_specs.append(
            {"x": int(saw_start[0]), "y": int(saw_start[1]), "dir": 1, "min_y": int(y_min), "max_y": int(y_max)}
        )

    links: list[list[int]] = []
    if config["switch_links"]:
        if len(config["switch_links"]) != len(switches):
            raise ValueError(f"hookshot level {index + 1} switch config mismatch")
        links = [list(link) for link in config["switch_links"]]
    else:
        links = [[] for _ in switches]

    return {
        "level_index": int(index),
        "width": int(width),
        "height": int(height),
        "walls": sorted((int(x), int(y)) for x, y in walls),
        "hooks": sorted((int(x), int(y)) for x, y in hooks),
        "lava": sorted((int(x), int(y)) for x, y in lava),
        "spikes": sorted((int(x), int(y)) for x, y in spikes),
        "exits": sorted((int(x), int(y)) for x, y in exits),
        "switches": sorted((int(x), int(y)) for x, y in switches),
        "switch_links": links,
        "doors": [[(int(x), int(y)) for x, y in cells] for cells in doors],
        "saws": saw_specs,
        "start": (int(start_pos[0]), int(start_pos[1])),
        "start_facing": int(start_facing),
        "time_per_segment": 3,
        "misfire_penalty": 6,
        "death_penalty": 15,
        "time_max_steps": int(width * 3),
    }


def _build_levels() -> list[Level]:
    levels: list[Level] = []
    for idx, layout in enumerate(LEVEL_LAYOUTS):
        model = _parse_level(layout, LEVEL_CONFIGS[idx], idx)
        width = int(model["width"])
        height = int(model["height"])
        board = Sprite(
            pixels=np.full((height, width), COLOR_FLOOR, dtype=np.int8),
            name="board",
            x=0,
            y=0,
            layer=0,
            tags=["board", "sys_static"],
            collidable=False,
        )
        levels.append(
            Level(name=f"Hookshot {idx + 1}", grid_size=(width, height), sprites=[board], data={"model": model})
        )
    return levels


def _deserialize_model(level: Level) -> dict:
    raw = dict(level.get_data("model") or {})
    return {
        "level_index": int(raw["level_index"]),
        "width": int(raw["width"]),
        "height": int(raw["height"]),
        "walls": {tuple(int(v) for v in cell) for cell in raw.get("walls", [])},
        "hooks": {tuple(int(v) for v in cell) for cell in raw.get("hooks", [])},
        "lava": {tuple(int(v) for v in cell) for cell in raw.get("lava", [])},
        "spikes": {tuple(int(v) for v in cell) for cell in raw.get("spikes", [])},
        "exits": {tuple(int(v) for v in cell) for cell in raw.get("exits", [])},
        "switches": [tuple(int(v) for v in cell) for cell in raw.get("switches", [])],
        "switch_links": [tuple(int(door_id) for door_id in links) for links in raw.get("switch_links", [])],
        "doors": [tuple(tuple(int(v) for v in cell) for cell in door) for door in raw.get("doors", [])],
        "saws": [
            {
                "x": int(spec["x"]),
                "y": int(spec["y"]),
                "dir": int(spec["dir"]),
                "min_y": int(spec["min_y"]),
                "max_y": int(spec["max_y"]),
            }
            for spec in raw.get("saws", [])
        ],
        "start": tuple(int(v) for v in raw["start"]),
        "start_facing": int(raw["start_facing"]),
        "time_per_segment": int(raw["time_per_segment"]),
        "misfire_penalty": int(raw["misfire_penalty"]),
        "death_penalty": int(raw["death_penalty"]),
        "time_max_steps": int(raw["time_max_steps"]),
    }


def initial_search_state_from_model(model: dict) -> tuple:
    start_x, start_y = model["start"]
    start_facing = int(model["start_facing"])
    time_left = int(model["time_max_steps"])
    saw_state = tuple((int(spec["y"]), int(spec["dir"])) for spec in model["saws"])
    door_state = tuple(0 for _ in model["doors"])
    switch_mask = 0
    stun = 0
    return (int(start_x), int(start_y), start_facing, stun, saw_state, door_state, switch_mask, time_left)


def _door_blocked_cells(model: dict, door_state: tuple[int, ...]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for door_id, state in enumerate(door_state):
        if int(state) == -1:
            continue
        blocked.update(model["doors"][door_id])
    return blocked


def _saw_positions(model: dict, saw_state: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    positions: set[tuple[int, int]] = set()
    for idx, (sy, _sdir) in enumerate(saw_state):
        positions.add((int(model["saws"][idx]["x"]), int(sy)))
    return positions


def _is_lethal(model: dict, px: int, py: int, saw_state: tuple[tuple[int, int], ...]) -> bool:
    pos = (int(px), int(py))
    return pos in model["lava"] or pos in model["spikes"] or pos in _saw_positions(model, saw_state)


def _cast_hook(
    model: dict, px: int, py: int, facing: int, blocked_doors: set[tuple[int, int]]
) -> tuple[tuple[int, int] | None, list[tuple[int, int]], bool]:
    width = int(model["width"])
    height = int(model["height"])
    dx, dy = FACING_TO_DELTA[int(facing)]

    nearest_hook: tuple[int, int] | None = None
    ray_cells: list[tuple[int, int]] = []

    x = int(px)
    y = int(py)
    while True:
        x += dx
        y += dy
        if x < 0 or y < 1 or x >= width or y >= height:
            break
        pos = (x, y)
        ray_cells.append(pos)
        if pos in model["walls"] or pos in blocked_doors:
            break
        if nearest_hook is None and pos in model["hooks"]:
            nearest_hook = pos
            break

    if nearest_hook is not None:
        try:
            hook_idx = ray_cells.index(nearest_hook)
        except ValueError:
            hook_idx = len(ray_cells) - 1
        return nearest_hook, ray_cells[: hook_idx + 1], False

    return None, ray_cells, True


def apply_action_transition(model: dict, state: tuple, action_id: int) -> tuple[tuple, bool, bool]:
    next_state, outcome = _simulate_step(model, state, int(action_id), with_effects=False)
    return next_state, bool(outcome["won"]), bool(outcome["restarted"] or outcome.get("lost", False))


def _simulate_step(model: dict, state: tuple, action_id: int, *, with_effects: bool) -> tuple[tuple, dict]:
    initial_state = initial_search_state_from_model(model)
    if int(action_id) == int(GameAction.ACTION6.value):
        return initial_state, {"won": False, "restarted": True, "lost": False, "flash": False, "tether": []}

    (px, py, facing, stun, saw_state, door_state, switch_mask, time_left) = state
    px = int(px)
    py = int(py)
    facing = int(facing)
    stun = int(stun)
    time_left = int(time_left)
    saw_state_list = [[int(item[0]), int(item[1])] for item in saw_state]
    door_state_list = [int(value) for value in door_state]
    switch_mask = int(switch_mask)

    misfire_penalty = int(model["misfire_penalty"])
    death_penalty = int(model["death_penalty"])

    blocked_doors = _door_blocked_cells(model, tuple(door_state_list))
    flash = False
    tether: list[tuple[int, int]] = []
    extra_penalty = 0

    if stun > 0:
        stun -= 1
    else:
        move_facing = ACTION_TO_FACING.get(int(action_id))
        if move_facing is not None:
            facing = int(move_facing)
            dx, dy = FACING_TO_DELTA[facing]
            tx = px + dx
            ty = py + dy
            blocked = (
                tx < 0
                or ty < 1
                or tx >= int(model["width"])
                or ty >= int(model["height"])
                or (tx, ty) in model["walls"]
                or (tx, ty) in blocked_doors
            )
            if not blocked:
                px = tx
                py = ty
        elif int(action_id) == int(GameAction.ACTION5.value):
            hook_pos, ray_cells, misfire = _cast_hook(model, px, py, facing, blocked_doors)
            tether = list(ray_cells)
            if hook_pos is not None:
                px, py = hook_pos
            elif misfire:
                stun = 1
                extra_penalty += misfire_penalty
                flash = True

    if _is_lethal(model, px, py, tuple((v[0], v[1]) for v in saw_state_list)):
        px, py = model["start"]
        facing = int(model["start_facing"])
        stun = 0
        extra_penalty += death_penalty
        flash = True

    for switch_idx, switch_pos in enumerate(model["switches"]):
        if (px, py) != switch_pos:
            continue
        if ((switch_mask >> switch_idx) & 1) == 0:
            switch_mask |= 1 << switch_idx
            for door_id in model["switch_links"][switch_idx]:
                if 0 <= int(door_id) < len(door_state_list) and door_state_list[int(door_id)] == 0:
                    door_state_list[int(door_id)] = 2

    for idx, door_timer in enumerate(door_state_list):
        if int(door_timer) <= 0:
            continue
        next_timer = int(door_timer) - 1
        if next_timer <= 0:
            door_state_list[idx] = -1
        else:
            door_state_list[idx] = next_timer

    for idx, saw in enumerate(model["saws"]):
        y = int(saw_state_list[idx][0])
        direction = int(saw_state_list[idx][1])
        ny = y + direction
        if ny < int(saw["min_y"]) or ny > int(saw["max_y"]):
            direction *= -1
            ny = y + direction
        saw_state_list[idx][0] = ny
        saw_state_list[idx][1] = direction

    if _is_lethal(model, px, py, tuple((v[0], v[1]) for v in saw_state_list)):
        px, py = model["start"]
        facing = int(model["start_facing"])
        stun = 0
        extra_penalty += death_penalty
        flash = True

    time_left -= 1 + extra_penalty

    won = (px, py) in model["exits"]
    lost = bool(time_left <= 0 and not won)
    restarted = False
    next_state = (
        int(px),
        int(py),
        int(facing),
        int(stun),
        tuple((int(item[0]), int(item[1])) for item in saw_state_list),
        tuple(int(value) for value in door_state_list),
        int(switch_mask),
        int(time_left),
    )

    if not with_effects:
        return next_state, {"won": won, "restarted": restarted, "lost": lost}

    return next_state, {
        "won": bool(won),
        "restarted": bool(restarted),
        "lost": bool(lost),
        "flash": bool(flash),
        "tether": list(tether),
    }


class Hookshot(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = _build_levels()
        width = int(levels[0].grid_size[0])
        height = int(levels[0].grid_size[1])
        camera = Camera(width=width, height=height, background=COLOR_FLOOR)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )
        self._model: dict | None = None
        self._hook_state: tuple | None = None
        self._board: Sprite | None = None
        self._phase = 0
        self._tether_cells: list[tuple[int, int]] = []
        self._flash_player = False

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level)
        self._hook_state = initial_search_state_from_model(self._model)
        boards = level.get_sprites_by_name("board")
        if not boards:
            raise RuntimeError("hookshot level is missing board sprite")
        self._board = boards[0]
        self._phase = 0
        self._tether_cells = []
        self._flash_player = False

        if self.camera.width != int(self._model["width"]) or self.camera.height != int(self._model["height"]):
            self.camera.width = int(self._model["width"])
            self.camera.height = int(self._model["height"])
        self._render()

    def _render(self) -> None:
        if self._model is None or self._hook_state is None or self._board is None:
            return

        (px, py, facing, _stun, saw_state, door_state, switch_mask, time_left) = self._hook_state

        width = int(self._model["width"])
        height = int(self._model["height"])
        pixels = np.full((height, width), COLOR_FLOOR, dtype=np.int8)

        for x, y in self._model["walls"]:
            pixels[int(y), int(x)] = COLOR_WALL

        for x, y in self._model["lava"]:
            pixels[int(y), int(x)] = COLOR_LAVA

        for x, y in self._model["spikes"]:
            pixels[int(y), int(x)] = COLOR_SPIKES

        for x, y in self._model["hooks"]:
            pixels[int(y), int(x)] = COLOR_HOOK

        exit_color = COLOR_EXIT if (self._phase % 2 == 0) else COLOR_HIGHLIGHT
        for x, y in self._model["exits"]:
            pixels[int(y), int(x)] = exit_color

        for idx, door_cells in enumerate(self._model["doors"]):
            timer = int(door_state[idx])
            if timer == -1:
                continue
            color = COLOR_DOOR_CLOSED if timer == 0 else COLOR_HIGHLIGHT
            for x, y in door_cells:
                pixels[int(y), int(x)] = color

        for switch_idx, (sx, sy) in enumerate(self._model["switches"]):
            pressed = ((int(switch_mask) >> switch_idx) & 1) != 0 or (int(px), int(py)) == (int(sx), int(sy))
            pixels[int(sy), int(sx)] = COLOR_HIGHLIGHT if pressed else COLOR_SWITCH

        saw_color = COLOR_SPIKES if (self._phase % 2 == 0) else COLOR_HIGHLIGHT
        for idx, (saw_y, _saw_dir) in enumerate(saw_state):
            saw_x = int(self._model["saws"][idx]["x"])
            pixels[int(saw_y), saw_x] = saw_color

        for tx, ty in self._tether_cells:
            if 0 <= int(tx) < width and 1 <= int(ty) < height:
                pixels[int(ty), int(tx)] = COLOR_TETHER

        player_color = COLOR_HIGHLIGHT if self._flash_player else FACING_TO_COLOR.get(int(facing), COLOR_PLAYER_RIGHT)
        pixels[int(py), int(px)] = player_color

        time_left = max(0, int(time_left))
        segments = width
        time_per_segment = int(self._model["time_per_segment"])
        filled = (time_left + time_per_segment - 1) // max(1, time_per_segment)
        filled = max(0, min(segments, filled))
        for x in range(width):
            pixels[0, x] = COLOR_TIME_FILL if x < filled else COLOR_TIME_EMPTY

        self._board.pixels = pixels

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

        if self._model is None or self._hook_state is None:
            self.lose()
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id == int(GameAction.ACTION6.value):
            self.level_reset()
            self.complete_action()
            return

        self._tether_cells = []
        self._flash_player = False

        next_state, outcome = _simulate_step(self._model, self._hook_state, action_id, with_effects=True)
        self._hook_state = next_state
        self._tether_cells = list(outcome.get("tether", []))
        self._flash_player = bool(outcome.get("flash", False))
        self._phase ^= 1

        if bool(outcome.get("won", False)):
            self.next_level()
            self.complete_action()
            return

        if bool(outcome.get("lost", False)):
            self.lose()
            self.complete_action()
            return

        if bool(outcome.get("restarted", False)):
            self.level_reset()
            self.complete_action()
            return

        self._render()
        self.complete_action()
