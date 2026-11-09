from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GAME_ID = "magnet-0001"

COLOR_BG = 0
COLOR_WALL = 1
COLOR_FLOOR = 2
COLOR_PLAYER_OFF = 3
COLOR_PLAYER_ON = 4
COLOR_METAL = 5
COLOR_PAD_UNLIT = 6
COLOR_PAD_LIT = 7
COLOR_SPIKE = 8
COLOR_PIT = 9
COLOR_TIME_REMAIN = 10
COLOR_TIME_EMPTY = 11
COLOR_PLATE = 12
COLOR_DOOR_CLOSED = 13
COLOR_DOOR_OPEN = 14
COLOR_METAL_MOVED = 15

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

PULL_RANGE = 6
MODE_PLAY = 0
MODE_WIN_ANIM = 1
MODE_FAIL_ANIM = 2

WIN_ANIM_STEPS = 8
FAIL_ANIM_STEPS = 10

LEVEL_SPECS = [
    {
        "name": "Level 1",
        "max_time": 120,
        "grid": [
            "▌■■■■■■■■■■■■■■■■■■■■▐",
            "######################",
            "#····················#",
            "#····················#",
            "#····················#",
            "#····················#",
            "#····················#",
            "#·······▛▜····┌┐·····#",
            "#·······▙▟····└┘·····#",
            "#····················#",
            "#····················#",
            "#····················#",
            "#··◇·················#",
            "#····················#",
            "#····················#",
            "#····················#",
            "#····················#",
            "######################",
        ],
    },
    {
        "name": "Level 2",
        "max_time": 160,
        "grid": [
            "▌■■■■■■■■■■■■■■■■■■■■▐",
            "######################",
            "#·········##·········#",
            "#·········##···┌┐····#",
            "#·········##···└┘····#",
            "#·········##·········#",
            "#··▛▜·····##·········#",
            "#··▙▟·····##·········#",
            "#·········##·········#",
            "#·········##·········#",
            "#·········##·········#",
            "#·········##·········#",
            "#····················#",
            "#····················#",
            "#··◇······##·········#",
            "#·········##·········#",
            "#·········##·········#",
            "######################",
        ],
    },
    {
        "name": "Level 3",
        "max_time": 190,
        "grid": [
            "▌■■■■■■■■■■■■■■■■■■■■▐",
            "######################",
            "#····················#",
            "#··············┌┐····#",
            "#··············└┘····#",
            "#··▛▜················#",
            "#··▙▟················#",
            "#····················#",
            "#········◇···········#",
            "#·········^^^^·······#",
            "#·········^^^^·······#",
            "#··▛▜················#",
            "#··▙▟················#",
            "#··············┌┐····#",
            "#··············└┘····#",
            "#····················#",
            "#····················#",
            "######################",
        ],
    },
    {
        "name": "Level 4",
        "max_time": 220,
        "grid": [
            "▌■■■■■■■■■■■■■■■■■■■■▐",
            "######################",
            "#···········#········#",
            "#·◇······#··#········#",
            "#···········#········#",
            "#··▛▜····#··#········#",
            "#··▙▟····#··#········#",
            "#···········#········#",
            "#···········#········#",
            "#·▒▒▒▒▒▒▒▒··▒▒▒▒▒▒▒▒·#",
            "#·▒▒▒▒▒▒▒▒··▒▒▒▒▒▒▒▒·#",
            "#····················#",
            "#····················#",
            "#··············┌┐····#",
            "#··············└┘····#",
            "#····················#",
            "#····················#",
            "######################",
        ],
    },
    {
        "name": "Level 5",
        "max_time": 260,
        "grid": [
            "▌■■■■■■■■■■■■■■■■■■■■▐",
            "######################",
            "#·········##·········#",
            "#·········##···┌┐····#",
            "#·········##···└┘····#",
            "#··▛▜·····##·········#",
            "#··▙▟·····##·········#",
            "#·········╫╫·········#",
            "#·········╫╫·········#",
            "#····▛▜···╫╫·········#",
            "#····▙▟···##·········#",
            "#·········##·········#",
            "#·········##···┌┐····#",
            "#·········##···└┘····#",
            "#···○·····##·········#",
            "#·◇·······##·········#",
            "#·········##·········#",
            "######################",
        ],
    },
    {
        "name": "Level 6",
        "max_time": 320,
        "grid": [
            "▌■■■■■■■■■■■■■■■■■■■■▐",
            "######################",
            "#····················#",
            "#········▛▜····┌┐····#",
            "#········▙▟····└┘····#",
            "#····○···╫╫··········#",
            "#········╫╫··········#",
            "#········╫╫····┌┐····#",
            "#··············└┘····#",
            "#········▛▜··········#",
            "#········▙▟···^^^^···#",
            "#····················#",
            "#··········○··╫╫·····#",
            "#·············╫╫·····#",
            "#········▛▜···╫╫┌┐···#",
            "#········▙▟······└┘··#",
            "#·◇··············▒▒··#",
            "######################",
        ],
    },
]


def _norm_lines(lines: list[str]) -> list[str]:
    return [line.replace(" ", "") for line in lines]


def _door_components(cells: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    remaining = set(cells)
    groups: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        comp = set()
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in comp or (cx, cy) not in remaining:
                continue
            comp.add((cx, cy))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cx + dx, cy + dy)
                if nxt in remaining and nxt not in comp:
                    stack.append(nxt)
        remaining -= comp
        groups.append(tuple(sorted(comp)))
    groups.sort(key=lambda g: (min(x for x, _ in g), min(y for _, y in g)))
    return groups


def _pair_plates_to_doors(plates: list[tuple[int, int]], doors: list[tuple[tuple[int, int], ...]]) -> tuple[int, ...]:
    if not plates or not doors:
        return tuple()
    door_centers = []
    for cells in doors:
        sx = sum(x for x, _ in cells)
        sy = sum(y for _, y in cells)
        door_centers.append((sx / len(cells), sy / len(cells)))

    result = []
    for px, py in plates:
        best_idx = 0
        best_dist = None
        for idx, (cx, cy) in enumerate(door_centers):
            dist = abs(px - cx) + abs(py - cy)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        result.append(best_idx)
    return tuple(result)


def _parse_level(spec: dict) -> dict:
    lines = _norm_lines(list(spec["grid"]))
    if len(lines) != 18:
        raise ValueError("Each level must have exactly 18 rows.")

    width = len(lines[0])
    height = len(lines)
    if width != 22:
        raise ValueError("Each level must have exactly 22 columns.")
    if any(len(row) != width for row in lines):
        raise ValueError("All rows must have equal width.")

    walls: set[tuple[int, int]] = set()
    spikes: set[tuple[int, int]] = set()
    pits: set[tuple[int, int]] = set()
    plates: list[tuple[int, int]] = []
    door_cells: set[tuple[int, int]] = set()
    pads: list[tuple[int, int]] = []
    metals: list[tuple[int, int]] = []
    player: tuple[int, int] | None = None

    for y, row in enumerate(lines):
        if y == 0:
            continue
        for x, ch in enumerate(row):
            if ch == "#":
                walls.add((x, y))
            elif ch == "^":
                spikes.add((x, y))
            elif ch == "▒":
                pits.add((x, y))
            elif ch == "○":
                plates.append((x, y))
            elif ch == "╫":
                door_cells.add((x, y))
            elif ch == "┌":
                pads.append((x, y))
            elif ch == "▛":
                metals.append((x, y))
            elif ch == "◇":
                player = (x, y)

    if player is None:
        raise ValueError("Missing player in level.")

    doors = _door_components(door_cells)
    plates = sorted(plates)
    pads = sorted(pads)
    metals = sorted(metals)

    if len(pads) > len(metals):
        raise ValueError("Each level needs at least as many metals as pads.")

    return {
        "name": str(spec["name"]),
        "width": int(width),
        "height": int(height),
        "max_time": int(spec["max_time"]),
        "walls": tuple(sorted(walls)),
        "spikes": tuple(sorted(spikes)),
        "pits": tuple(sorted(pits)),
        "plates": tuple(plates),
        "doors": tuple(doors),
        "plate_to_door": _pair_plates_to_doors(plates, doors),
        "pads": tuple(pads),
        "metals": tuple(metals),
        "player": tuple(player),
    }


LEVEL_MODELS = tuple(_parse_level(spec) for spec in LEVEL_SPECS)


def _plate_pressed_mask(model: dict, metals: tuple[tuple[int, int], ...]) -> int:
    pressed_mask = 0
    for plate_idx, (px, py) in enumerate(model["plates"]):
        for mx, my in metals:
            if mx < 0:
                continue
            if mx <= px <= mx + 1 and my <= py <= my + 1:
                pressed_mask |= 1 << plate_idx
                break
    return int(pressed_mask)


def _toggle_doors_from_plate_edges(model: dict, prev_pressed_mask: int, pressed_mask: int, door_open_mask: int) -> int:
    next_mask = int(door_open_mask)
    rising = int(pressed_mask) & ~int(prev_pressed_mask)
    for plate_idx, door_idx in enumerate(model["plate_to_door"]):
        if not (rising & (1 << plate_idx)):
            continue
        if 0 <= int(door_idx) < len(model["doors"]):
            next_mask ^= 1 << int(door_idx)
    return int(next_mask)


def _closed_door_cells(model: dict, door_open_mask: int) -> set[tuple[int, int]]:
    closed: set[tuple[int, int]] = set()
    for idx, group in enumerate(model["doors"]):
        if int(door_open_mask) & (1 << idx):
            continue
        closed.update(group)
    return closed


def _metal_cells(mx: int, my: int) -> tuple[tuple[int, int], ...]:
    return ((mx, my), (mx + 1, my), (mx, my + 1), (mx + 1, my + 1))


def _all_metal_cells(metals: tuple[tuple[int, int], ...], *, ignore_idx: int | None = None) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for idx, (mx, my) in enumerate(metals):
        if ignore_idx is not None and idx == ignore_idx:
            continue
        if mx < 0:
            continue
        out.update(_metal_cells(mx, my))
    return out


def _line_clear(player: tuple[int, int], target: tuple[int, int], blocked_cells: set[tuple[int, int]]) -> bool:
    px, py = player
    tx, ty = target
    if px == tx:
        y0, y1 = sorted((py, ty))
        return all((px, y) not in blocked_cells for y in range(y0 + 1, y1))
    if py == ty:
        x0, x1 = sorted((px, tx))
        return all((x, py) not in blocked_cells for x in range(x0 + 1, x1))
    return False


def _pull_direction(
    player: tuple[int, int], metal: tuple[int, int], blocked_cells: set[tuple[int, int]]
) -> tuple[int, int] | None:
    px, py = player
    mx, my = metal

    candidates: list[tuple[int, int, int]] = []
    for cx, cy in _metal_cells(mx, my):
        if px == cx:
            dist = abs(py - cy)
            if dist > 0 and _line_clear(player, (cx, cy), blocked_cells):
                candidates.append((0, -1 if py < cy else 1, dist))
        elif py == cy:
            dist = abs(px - cx)
            if dist > 0 and _line_clear(player, (cx, cy), blocked_cells):
                candidates.append((-1 if px < cx else 1, 0, dist))

    if not candidates:
        return None
    candidates.sort(key=lambda v: v[2])
    return candidates[0][0], candidates[0][1]


def _nearest_cell_distance(player: tuple[int, int], metal: tuple[int, int]) -> int:
    px, py = player
    mx, my = metal
    return min(abs(px - cx) + abs(py - cy) for cx, cy in _metal_cells(mx, my))


def _blocked_for_player(model: dict, x: int, y: int, metals: tuple[tuple[int, int], ...], door_open_mask: int) -> bool:
    if x < 0 or y < 0 or x >= model["width"] or y >= model["height"]:
        return True
    if (x, y) in model["walls"]:
        return True
    if (x, y) in _closed_door_cells(model, door_open_mask):
        return True
    return (x, y) in _all_metal_cells(metals)


def _apply_pull(
    model: dict,
    player: tuple[int, int],
    metals: tuple[tuple[int, int], ...],
    door_open_mask: int,
    clamped_mask: int,
    pad_mask: int,
) -> tuple[tuple[tuple[int, int], ...] | None, int, int, int]:
    blocked_los = set(model["walls"]) | _closed_door_cells(model, door_open_mask)

    affected: list[tuple[int, int, tuple[int, int]]] = []
    for idx, metal in enumerate(metals):
        if metal[0] < 0:
            continue
        if clamped_mask & (1 << idx):
            continue
        distance = _nearest_cell_distance(player, metal)
        if distance > PULL_RANGE:
            continue
        direction = _pull_direction(player, metal, blocked_los)
        if direction is None:
            continue
        affected.append((distance, idx, direction))

    affected.sort(key=lambda item: (item[0], item[1]))

    next_metals = list(metals)
    moved_mask = 0

    for _, idx, (dx, dy) in affected:
        mx, my = next_metals[idx]
        if mx < 0:
            continue

        destination = tuple((cx + dx, cy + dy) for cx, cy in _metal_cells(mx, my))
        blocked = False
        for cx, cy in destination:
            if cx < 0 or cy < 0 or cx >= model["width"] or cy >= model["height"]:
                blocked = True
                break
            if (cx, cy) in model["walls"]:
                blocked = True
                break
            if (cx, cy) in _closed_door_cells(model, door_open_mask):
                blocked = True
                break

        if blocked:
            continue

        occupied = _all_metal_cells(tuple(next_metals), ignore_idx=idx)
        if any(cell in occupied for cell in destination):
            continue

        if any(cell in model["pits"] for cell in destination):
            next_metals[idx] = (-1, -1)
            continue

        next_metals[idx] = (mx + dx, my + dy)
        moved_mask |= 1 << idx

    updated_metals = tuple(next_metals)
    next_pad_mask = int(pad_mask)
    next_clamped_mask = int(clamped_mask)

    for pad_idx, (px, py) in enumerate(model["pads"]):
        if next_pad_mask & (1 << pad_idx):
            continue
        for metal_idx, (mx, my) in enumerate(updated_metals):
            if mx < 0:
                continue
            if mx == px and my == py:
                next_pad_mask |= 1 << pad_idx
                next_clamped_mask |= 1 << metal_idx
                break

    unlit = len(model["pads"]) - int(next_pad_mask.bit_count())
    free_metals = 0
    for idx, (mx, _) in enumerate(updated_metals):
        if mx < 0:
            continue
        if next_clamped_mask & (1 << idx):
            continue
        free_metals += 1

    if free_metals < unlit:
        return None, next_clamped_mask, next_pad_mask, moved_mask

    return updated_metals, next_clamped_mask, next_pad_mask, moved_mask


def _apply_movement(
    model: dict, player: tuple[int, int], action_id: int, metals: tuple[tuple[int, int], ...], door_open_mask: int
) -> tuple[int, int]:
    dx, dy = MOVE_DELTAS.get(action_id, (0, 0))
    if dx == 0 and dy == 0:
        return player

    nx = player[0] + dx
    ny = player[1] + dy
    if _blocked_for_player(model, nx, ny, metals, door_open_mask):
        return player
    return nx, ny


def _state_after_action(model: dict, state: tuple, action_id: int) -> tuple | None:
    (px, py, magnet_on, time_left, tick, metals, door_open_mask, plate_pressed_mask, clamped_mask, pad_mask) = state

    player = (int(px), int(py))
    time_left = int(time_left)
    tick = int(tick)
    metals = tuple((int(mx), int(my)) for mx, my in metals)
    door_open_mask = int(door_open_mask)
    plate_pressed_mask = int(plate_pressed_mask)
    clamped_mask = int(clamped_mask)
    pad_mask = int(pad_mask)

    if action_id in MOVE_DELTAS:
        player = _apply_movement(model, player, int(action_id), metals, door_open_mask)
    elif int(action_id) == 5:
        magnet_on = 0 if int(magnet_on) else 1

    moved_mask = 0
    if int(magnet_on):
        pulled = _apply_pull(model, player, metals, door_open_mask, clamped_mask, pad_mask)
        updated_metals, clamped_mask, pad_mask, moved_mask = pulled
        if updated_metals is None:
            return None
        metals = updated_metals
    else:
        for pad_idx, (tx, ty) in enumerate(model["pads"]):
            if pad_mask & (1 << pad_idx):
                continue
            for metal_idx, (mx, my) in enumerate(metals):
                if mx == tx and my == ty:
                    pad_mask |= 1 << pad_idx
                    clamped_mask |= 1 << metal_idx
                    break

    pressed_now = _plate_pressed_mask(model, metals)
    door_open_mask = _toggle_doors_from_plate_edges(model, plate_pressed_mask, pressed_now, door_open_mask)
    plate_pressed_mask = pressed_now

    if player in model["spikes"] or player in model["pits"]:
        return None

    time_left -= 1
    if time_left <= 0:
        return None

    return (
        int(player[0]),
        int(player[1]),
        int(magnet_on),
        int(time_left),
        int(tick + 1),
        tuple((int(mx), int(my)) for mx, my in metals),
        int(door_open_mask),
        int(plate_pressed_mask),
        int(clamped_mask),
        int(pad_mask),
        int(moved_mask),
    )


def _serialize_model(model: dict) -> dict:
    return {
        "name": str(model["name"]),
        "width": int(model["width"]),
        "height": int(model["height"]),
        "max_time": int(model["max_time"]),
        "walls": [list(v) for v in model["walls"]],
        "spikes": [list(v) for v in model["spikes"]],
        "pits": [list(v) for v in model["pits"]],
        "plates": [list(v) for v in model["plates"]],
        "doors": [[list(v) for v in group] for group in model["doors"]],
        "plate_to_door": list(model["plate_to_door"]),
        "pads": [list(v) for v in model["pads"]],
        "metals": [list(v) for v in model["metals"]],
        "player": list(model["player"]),
    }


def _deserialize_model(level: Level | dict) -> dict:
    data = level.get_data("model") if isinstance(level, Level) else level
    if data is None:
        raise ValueError("Missing serialized level model.")

    return {
        "name": str(data["name"]),
        "width": int(data["width"]),
        "height": int(data["height"]),
        "max_time": int(data["max_time"]),
        "walls": {tuple(int(v) for v in cell) for cell in data["walls"]},
        "spikes": {tuple(int(v) for v in cell) for cell in data["spikes"]},
        "pits": {tuple(int(v) for v in cell) for cell in data["pits"]},
        "plates": tuple(tuple(int(v) for v in cell) for cell in data["plates"]),
        "doors": tuple(tuple(tuple(int(v) for v in cell) for cell in group) for group in data["doors"]),
        "plate_to_door": tuple(int(v) for v in data["plate_to_door"]),
        "pads": tuple(tuple(int(v) for v in cell) for cell in data["pads"]),
        "metals": tuple(tuple(int(v) for v in cell) for cell in data["metals"]),
        "player": tuple(int(v) for v in data["player"]),
    }


def initial_search_state_from_model(model: dict) -> tuple:
    return (
        int(model["player"][0]),
        int(model["player"][1]),
        0,
        int(model["max_time"]),
        0,
        tuple((int(mx), int(my)) for mx, my in model["metals"]),
        0,
        0,
        0,
        0,
    )


def apply_action_transition(model: dict, state: tuple, action_id: int) -> tuple[tuple | None, bool]:
    nxt = _state_after_action(model, state, int(action_id))
    if nxt is None:
        return None, False
    all_lit_mask = (1 << len(model["pads"])) - 1
    won = int(nxt[9]) == all_lit_mask
    return nxt[:10], won


def _build_level(level_idx: int, model: dict) -> Level:
    pixels = np.full((int(model["height"]), int(model["width"])), COLOR_BG, dtype=np.int8)
    frame = Sprite(pixels=pixels, name="frame", x=0, y=0, layer=0, tags=["frame"], collidable=False)
    return Level(
        name=str(model["name"]),
        sprites=[frame],
        grid_size=(int(model["width"]), int(model["height"])),
        data={"level_idx": int(level_idx), "model": _serialize_model(model)},
    )


class Magnet(ARCBaseGame):
    def __init__(self):
        self._level_models = tuple(LEVEL_MODELS)
        levels = [_build_level(idx, model) for idx, model in enumerate(self._level_models)]
        camera = Camera(
            0, 0, int(self._level_models[0]["width"]), int(self._level_models[0]["height"]), COLOR_BG, COLOR_BG, []
        )
        super().__init__(
            GAME_ID.split("-", 1)[0],
            levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
        )

        self._model: dict | None = None
        self._frame: Sprite | None = None
        self._level_idx = 0

        self._player = (0, 0)
        self._metals: tuple[tuple[int, int], ...] = tuple()
        self._clamped_mask = 0
        self._pad_mask = 0
        self._magnet_on = False
        self._time_left = 0
        self._tick = 0
        self._last_moved_mask = 0
        self._door_open_mask = 0
        self._plate_pressed_mask = 0

        self._mode = MODE_PLAY
        self._mode_steps_left = 0

    def on_set_level(self, level: Level) -> None:
        self._level_idx = int(level.get_data("level_idx") or 0)
        self._model = _deserialize_model(level)
        frame = self.current_level.get_sprites_by_name("frame")
        self._frame = frame[0] if frame else None
        self._restart_level_runtime()

    def _restart_level_runtime(self) -> None:
        if self._model is None:
            return
        self._player = tuple(self._model["player"])
        self._metals = tuple(self._model["metals"])
        self._clamped_mask = 0
        self._pad_mask = 0
        self._magnet_on = False
        self._time_left = int(self._model["max_time"])
        self._tick = 0
        self._last_moved_mask = 0
        self._door_open_mask = 0
        self._plate_pressed_mask = 0
        self._mode = MODE_PLAY
        self._mode_steps_left = 0
        self._render()

    def _trigger_fail(self) -> None:
        if self._mode != MODE_PLAY:
            return
        self._mode = MODE_FAIL_ANIM
        self._mode_steps_left = FAIL_ANIM_STEPS

    def _trigger_win(self) -> None:
        if self._mode != MODE_PLAY:
            return
        self._mode = MODE_WIN_ANIM
        self._mode_steps_left = WIN_ANIM_STEPS

    def _render(self) -> None:
        if self._frame is None or self._model is None:
            return

        width = int(self._model["width"])
        height = int(self._model["height"])
        frame = np.full((height, width), COLOR_FLOOR, dtype=np.int8)

        for x, y in self._model["walls"]:
            frame[y, x] = COLOR_WALL

        for x, y in self._model["pits"]:
            frame[y, x] = COLOR_PIT
        for x, y in self._model["spikes"]:
            frame[y, x] = COLOR_SPIKE

        for door_idx, group in enumerate(self._model["doors"]):
            color = COLOR_DOOR_OPEN if (self._door_open_mask & (1 << door_idx)) else COLOR_DOOR_CLOSED
            for x, y in group:
                frame[y, x] = color

        for idx, (x, y) in enumerate(self._model["pads"]):
            is_lit = bool(self._pad_mask & (1 << idx))
            if is_lit:
                if self._mode == MODE_WIN_ANIM and (self._mode_steps_left % 2 == 0):
                    color = COLOR_TIME_REMAIN
                else:
                    color = COLOR_PAD_LIT
            else:
                color = COLOR_PAD_UNLIT
            frame[y : y + 2, x : x + 2] = color

        for x, y in self._model["plates"]:
            frame[y, x] = COLOR_PLATE

        for idx, (mx, my) in enumerate(self._metals):
            if mx < 0:
                continue
            color = COLOR_METAL_MOVED if (self._last_moved_mask & (1 << idx)) else COLOR_METAL
            frame[my : my + 2, mx : mx + 2] = color

        px, py = self._player
        frame[py, px] = COLOR_PLAYER_ON if self._magnet_on else COLOR_PLAYER_OFF

        if self._mode == MODE_FAIL_ANIM:
            flash_color = COLOR_SPIKE if (self._mode_steps_left % 2 == 0) else COLOR_FLOOR
            for y in range(1, height):
                for x in range(width):
                    if frame[y, x] != COLOR_WALL:
                        frame[y, x] = flash_color

        frame[0, :] = COLOR_TIME_EMPTY
        frame[0, 0] = COLOR_TIME_REMAIN
        frame[0, width - 1] = COLOR_TIME_REMAIN

        max_time = max(1, int(self._model["max_time"]))
        filled = int((int(self._time_left) * 20) // max_time)
        filled = max(0, min(20, filled))
        blink = (self._time_left * 10 <= max_time) and (self._tick % 2 == 0)

        for idx in range(20):
            x = idx + 1
            if idx < filled:
                frame[0, x] = COLOR_SPIKE if blink else COLOR_TIME_REMAIN
            else:
                frame[0, x] = COLOR_TIME_EMPTY

        self._frame.pixels = frame

    def _advance_win_anim(self) -> None:
        self._mode_steps_left -= 1
        if self._mode_steps_left <= 0:
            self.next_level()
            return
        self._tick += 1
        self._last_moved_mask = 0
        self._render()

    def _advance_fail_anim(self) -> None:
        self._mode_steps_left -= 1
        if self._mode_steps_left <= 0:
            self.lose()
            return
        self._tick += 1
        self._last_moved_mask = 0
        self._render()

    def _play_step(self, action_id: int) -> None:
        if self._model is None:
            return

        player = self._player
        metals = self._metals
        clamped_mask = self._clamped_mask
        pad_mask = self._pad_mask
        magnet_on = self._magnet_on
        door_open_mask = self._door_open_mask
        plate_pressed_mask = self._plate_pressed_mask

        if action_id in MOVE_DELTAS:
            player = _apply_movement(self._model, player, action_id, metals, door_open_mask)
        elif action_id == 5:
            magnet_on = not magnet_on

        moved_mask = 0
        if magnet_on:
            pulled = _apply_pull(self._model, player, metals, door_open_mask, clamped_mask, pad_mask)
            updated_metals, clamped_mask, pad_mask, moved_mask = pulled
            if updated_metals is None:
                self._player = player
                self._metals = metals
                self._clamped_mask = clamped_mask
                self._pad_mask = pad_mask
                self._magnet_on = magnet_on
                self._door_open_mask = door_open_mask
                self._plate_pressed_mask = plate_pressed_mask
                self._last_moved_mask = moved_mask
                self._tick += 1
                self._trigger_fail()
                self._render()
                return
            metals = updated_metals
        else:
            for pad_idx, (tx, ty) in enumerate(self._model["pads"]):
                if pad_mask & (1 << pad_idx):
                    continue
                for metal_idx, (mx, my) in enumerate(metals):
                    if mx == tx and my == ty:
                        pad_mask |= 1 << pad_idx
                        clamped_mask |= 1 << metal_idx
                        break

        pressed_now = _plate_pressed_mask(self._model, metals)
        door_open_mask = _toggle_doors_from_plate_edges(self._model, plate_pressed_mask, pressed_now, door_open_mask)
        plate_pressed_mask = pressed_now

        self._player = player
        self._metals = metals
        self._clamped_mask = clamped_mask
        self._pad_mask = pad_mask
        self._magnet_on = magnet_on
        self._door_open_mask = door_open_mask
        self._plate_pressed_mask = plate_pressed_mask
        self._last_moved_mask = moved_mask

        self._tick += 1

        if player in self._model["spikes"] or player in self._model["pits"]:
            self._trigger_fail()
            self._render()
            return

        self._time_left -= 1
        if self._time_left <= 0:
            self._trigger_fail()
            self._render()
            return

        all_lit = self._pad_mask == ((1 << len(self._model["pads"])) - 1)
        if all_lit:
            self._trigger_win()

        self._render()

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

        action_id = int(getattr(self.action.id, "value", self.action.id))

        if self._mode == MODE_WIN_ANIM:
            self._advance_win_anim()
            self.complete_action()
            return

        if self._mode == MODE_FAIL_ANIM:
            self._advance_fail_anim()
            self.complete_action()
            return

        self._play_step(action_id)
        self.complete_action()
