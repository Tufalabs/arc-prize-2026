from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GAME_ID = "boulder_push_roll-0001"

COLOR_PIT = 0
COLOR_WALL = 1
COLOR_FLOOR = 2
COLOR_PLAYER = 3
COLOR_BOULDER_RESTING = 4
COLOR_BOULDER_FALLING = 5
COLOR_PLATE = 6
COLOR_PLATE_PRESSED = 7
COLOR_DOOR_CLOSED = 8
COLOR_DOOR_OPEN = 9
COLOR_SPIKES_SAFE = 10
COLOR_SPIKES_DANGER = 11
COLOR_GOAL = 12
COLOR_TIMEBAR_FULL = 13
COLOR_TIMEBAR_EMPTY = 14
COLOR_WARNING = 15

GRID_W = 24
GRID_H = 16
PLAY_MIN_Y = 1
PLAY_MAX_Y = GRID_H - 1

ACTION_TO_DELTA = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


LEVEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "Level 1",
        "time_limit": 120,
        "grid": [
            "========================",
            "########################",
            "#@.........#...........#",
            "#.....O....#...........#",
            "#..........#...........#",
            "#......p...#...........#",
            "#..........#...........#",
            "#..........#...........#",
            "#..........D...........#",
            "#..........D...........#",
            "#..........#...........#",
            "#..........#......XX...#",
            "#..........#......XX...#",
            "#..........#...........#",
            "#..........#...........#",
            "########################",
        ],
        "links": [((7, 5), (11, 8))],
    },
    {
        "name": "Level 2",
        "time_limit": 140,
        "grid": [
            "========================",
            "########################",
            "#@.........#...........#",
            "#..........#...........#",
            "#....O~....#...........#",
            "#.....~....#...........#",
            "#.....~....#...........#",
            "#.....~....D...........#",
            "#.....~....D...........#",
            "#.....~....#...........#",
            "#.....p....#...........#",
            "#..........#.......XX..#",
            "#..........#.......XX..#",
            "#..........#...........#",
            "#..........#...........#",
            "########################",
        ],
        "links": [((6, 10), (11, 7))],
    },
    {
        "name": "Level 3",
        "time_limit": 160,
        "grid": [
            "========================",
            "########################",
            "#@.........#############",
            "#..........#############",
            "#....O~....#############",
            "#.....~....#############",
            "#.....~....#############",
            "#.....~....#############",
            "#.....~....D.v.v.v.v.v.#",
            "#.....~....D.v.v.v.v.v.#",
            "#.....p....###########.#",
            "#..........#...........#",
            "#..........#.......XX..#",
            "#..........#.......XX..#",
            "#..........#...........#",
            "########################",
        ],
        "links": [((6, 10), (11, 8))],
    },
    {
        "name": "Level 4",
        "time_limit": 180,
        "grid": [
            "========================",
            "########################",
            "#........#.....#.......#",
            "#........#.....#.......#",
            "#........#.....#..XX...#",
            "#........#.....#..XX...#",
            "#........#..#..#.......#",
            "#........#..#..D.......#",
            "#........#..#..D.......#",
            "#........#.p#..#.......#",
            "#........#.....#.......#",
            "#........D.....#.......#",
            "#..O.O...D.....#.......#",
            "#@.....p.#.....#.......#",
            "#........#.....#.......#",
            "########################",
        ],
        "links": [((11, 9), (15, 7)), ((7, 13), (9, 11))],
    },
    {
        "name": "Level 5",
        "time_limit": 160,
        "grid": [
            "========================",
            "########################",
            "#...............##.....#",
            "#.........~.....##..XX.#",
            "#....O....~.....##..XX.#",
            "#....O....~.....##.....#",
            "#.........~.....##.....#",
            "#.........~.....DD.....#",
            "#.........~.....DD.....#",
            "#.........~.....##.....#",
            "#.........~.....##.....#",
            "#.........~.....##.....#",
            "#.........~.....##.....#",
            "#.@.......p.....##.....#",
            "#.........p.....##.....#",
            "########################",
        ],
        "links": [((10, 13), (16, 7))],
    },
    {
        "name": "Level 6",
        "time_limit": 140,
        "grid": [
            "========================",
            "########################",
            "#........###...####....#",
            "#........###...####.XX.#",
            "#........###.O~####.XX.#",
            "#........###.O~####....#",
            "#........###..~..DD....#",
            "#........###..~..DD....#",
            "#........###vv~..##....#",
            "#........###..~..##....#",
            "#........###vv~..##....#",
            "#........###..~..##....#",
            "#..O.....D....p..##....#",
            "#.@..p...D....p..##....#",
            "#........#.......##....#",
            "########################",
        ],
        "links": [((5, 13), (9, 12)), ((14, 12), (17, 6))],
    },
]


def _connected_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    pending = set(cells)
    out: list[set[tuple[int, int]]] = []
    while pending:
        start = pending.pop()
        comp = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ACTION_TO_DELTA.values():
                nxt = (x + dx, y + dy)
                if nxt in pending:
                    pending.remove(nxt)
                    comp.add(nxt)
                    queue.append(nxt)
        out.append(comp)
    return out


def _serialize_model(model: dict[str, Any]) -> dict[str, Any]:
    def ser_coords(values: set[tuple[int, int]]) -> list[list[int]]:
        return [[int(x), int(y)] for x, y in sorted(values)]

    links: list[dict[str, list[list[int]]]] = []
    for link in model["door_links"]:
        links.append(
            {
                "plate_tiles": [[int(x), int(y)] for x, y in sorted(link["plate_tiles"])],
                "door_tiles": [[int(x), int(y)] for x, y in sorted(link["door_tiles"])],
            }
        )

    return {
        "name": str(model["name"]),
        "width": int(model["width"]),
        "height": int(model["height"]),
        "time_limit": int(model["time_limit"]),
        "segment_steps": int(model["segment_steps"]),
        "player_start": [int(model["player_start"][0]), int(model["player_start"][1])],
        "boulders": ser_coords(model["boulders"]),
        "walls": ser_coords(model["walls"]),
        "pits": ser_coords(model["pits"]),
        "plates": ser_coords(model["plates"]),
        "doors": ser_coords(model["doors"]),
        "spikes": ser_coords(model["spikes"]),
        "goals": ser_coords(model["goals"]),
        "door_links": links,
    }


def _deserialize_model(level: Level | dict[str, Any]) -> dict[str, Any]:
    raw = level if isinstance(level, dict) else (level.get_data("model") or {})

    def de_coords(values: Any) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for item in values or []:
            if item is None or len(item) != 2:
                continue
            out.add((int(item[0]), int(item[1])))
        return out

    links: list[dict[str, set[tuple[int, int]]]] = []
    for link in raw.get("door_links") or []:
        links.append(
            {"plate_tiles": de_coords(link.get("plate_tiles")), "door_tiles": de_coords(link.get("door_tiles"))}
        )

    return {
        "name": str(raw.get("name") or "Level"),
        "width": int(raw.get("width") or GRID_W),
        "height": int(raw.get("height") or GRID_H),
        "time_limit": int(raw.get("time_limit") or 1),
        "segment_steps": int(raw.get("segment_steps") or 1),
        "player_start": (int((raw.get("player_start") or [1, 1])[0]), int((raw.get("player_start") or [1, 1])[1])),
        "boulders": de_coords(raw.get("boulders")),
        "walls": de_coords(raw.get("walls")),
        "pits": de_coords(raw.get("pits")),
        "plates": de_coords(raw.get("plates")),
        "doors": de_coords(raw.get("doors")),
        "spikes": de_coords(raw.get("spikes")),
        "goals": de_coords(raw.get("goals")),
        "door_links": links,
    }


def _parse_level(spec: dict[str, Any]) -> dict[str, Any]:
    rows = list(spec["grid"])
    if len(rows) != GRID_H or any(len(row) != GRID_W for row in rows):
        raise ValueError(f"{spec['name']}: level must be {GRID_W}x{GRID_H}")

    player_start: tuple[int, int] | None = None
    boulders: set[tuple[int, int]] = set()
    walls: set[tuple[int, int]] = set()
    pits: set[tuple[int, int]] = set()
    plates: set[tuple[int, int]] = set()
    doors: set[tuple[int, int]] = set()
    spikes: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if y == 0:
                continue
            if ch == "#":
                walls.add((x, y))
            elif ch == "~":
                pits.add((x, y))
            elif ch == "@":
                player_start = (x, y)
            elif ch in {"O", "o"}:
                boulders.add((x, y))
            elif ch in {"p", "P"}:
                plates.add((x, y))
            elif ch in {"D", "d"}:
                doors.add((x, y))
            elif ch in {"v", "V"}:
                spikes.add((x, y))
            elif ch == "X":
                goals.add((x, y))
            elif ch == ".":
                pass
            else:
                raise ValueError(f"{spec['name']}: unsupported char `{ch}` at {(x, y)}")

    if player_start is None:
        raise ValueError(f"{spec['name']}: missing player")
    if not goals:
        raise ValueError(f"{spec['name']}: missing goal tiles")

    plate_components = _connected_components(plates)
    door_components = _connected_components(doors)

    def find_plate_component(coord: tuple[int, int]) -> set[tuple[int, int]]:
        for comp in plate_components:
            if coord in comp:
                return comp
        raise ValueError(f"{spec['name']}: plate link coord {coord} is not a plate")

    def find_door_component(coord: tuple[int, int]) -> set[tuple[int, int]]:
        for comp in door_components:
            if coord in comp:
                return comp
        raise ValueError(f"{spec['name']}: door link coord {coord} is not a door")

    links: list[dict[str, set[tuple[int, int]]]] = []
    for plate_coord, door_coord in spec.get("links") or []:
        links.append(
            {
                "plate_tiles": set(find_plate_component((int(plate_coord[0]), int(plate_coord[1])))),
                "door_tiles": set(find_door_component((int(door_coord[0]), int(door_coord[1])))),
            }
        )

    if len(links) == 0 and doors:
        raise ValueError(f"{spec['name']}: door tiles exist but no link mapping provided")

    time_limit = int(spec["time_limit"])
    segment_steps = max(1, int(np.ceil(float(time_limit) / float(GRID_W))))

    return {
        "name": str(spec["name"]),
        "width": GRID_W,
        "height": GRID_H,
        "time_limit": time_limit,
        "segment_steps": segment_steps,
        "player_start": player_start,
        "boulders": boulders,
        "walls": walls,
        "pits": pits,
        "plates": plates,
        "doors": doors,
        "spikes": spikes,
        "goals": goals,
        "door_links": links,
    }


def _coords_from_state(boulders_state: tuple[tuple[int, int, int], ...]):
    resting: set[tuple[int, int]] = set()
    falling: set[tuple[int, int]] = set()
    for x, y, falling_flag in boulders_state:
        if int(falling_flag):
            falling.add((int(x), int(y)))
        else:
            resting.add((int(x), int(y)))
    return resting, falling


def _state_from_coords(
    resting: set[tuple[int, int]], falling: set[tuple[int, int]]
) -> tuple[tuple[int, int, int], ...]:
    merged = [(int(x), int(y), 0) for x, y in resting]
    merged.extend((int(x), int(y), 1) for x, y in falling)
    merged.sort()
    return tuple(merged)


def _in_play_bounds(x: int, y: int, model: dict[str, Any]) -> bool:
    return 0 <= int(x) < int(model["width"]) and PLAY_MIN_Y <= int(y) < int(model["height"])


def _pressed_tiles(
    model: dict[str, Any], player: tuple[int, int], resting_boulders: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    pressed: set[tuple[int, int]] = set()
    for tile in model["plates"]:
        if tile == player or tile in resting_boulders:
            pressed.add(tile)
    return pressed


def _open_doors_from_pressed(model: dict[str, Any], pressed_tiles: set[tuple[int, int]]) -> set[tuple[int, int]]:
    opened: set[tuple[int, int]] = set()
    for link in model["door_links"]:
        if any(tile in pressed_tiles for tile in link["plate_tiles"]):
            opened |= link["door_tiles"]
    return opened


def _is_spike_extended(spike_phase: int) -> bool:
    return int(spike_phase) in {2, 3}


def _cell_is_blocking_for_player(
    model: dict[str, Any],
    x: int,
    y: int,
    resting: set[tuple[int, int]],
    falling: set[tuple[int, int]],
    open_doors: set[tuple[int, int]],
) -> bool:
    pos = (int(x), int(y))
    if not _in_play_bounds(pos[0], pos[1], model):
        return True
    if pos in model["walls"]:
        return True
    if pos in model["doors"] and pos not in open_doors:
        return True
    return bool(pos in resting or pos in falling)


def _cell_is_push_destination(
    model: dict[str, Any],
    x: int,
    y: int,
    resting: set[tuple[int, int]],
    falling: set[tuple[int, int]],
    open_doors: set[tuple[int, int]],
) -> bool:
    pos = (int(x), int(y))
    if not _in_play_bounds(pos[0], pos[1], model):
        return False
    if pos in model["walls"]:
        return False
    if pos in model["doors"] and pos not in open_doors:
        return False
    return not (pos in resting or pos in falling)


def _legal_move_exists(
    model: dict[str, Any],
    player: tuple[int, int],
    resting: set[tuple[int, int]],
    falling: set[tuple[int, int]],
    open_doors: set[tuple[int, int]],
    spike_phase: int,
) -> bool:
    deadly_spikes = _is_spike_extended(spike_phase)
    px, py = player
    for dx, dy in ACTION_TO_DELTA.values():
        tx, ty = px + dx, py + dy
        target = (tx, ty)
        if not _in_play_bounds(tx, ty, model):
            continue

        if target in resting:
            bx, by = tx + dx, ty + dy
            if not _cell_is_push_destination(model, bx, by, resting, falling, open_doors):
                continue
            if (bx, by) in model["spikes"] and deadly_spikes:
                # Pushing into extended spikes is allowed.
                return True
            return True

        if _cell_is_blocking_for_player(model, tx, ty, resting, falling, open_doors):
            continue
        if target in model["pits"]:
            continue
        if target in model["spikes"] and deadly_spikes:
            continue
        return True

    return False


def initial_search_state_from_model(model: dict[str, Any]):
    return (
        int(model["player_start"][0]),
        int(model["player_start"][1]),
        int(model["time_limit"]),
        0,  # spike phase
        0,  # low-time flash parity
        _state_from_coords(set(model["boulders"]), set()),
    )


def apply_action_transition(
    model: dict[str, Any], state: tuple[int, int, int, int, int, tuple[tuple[int, int, int], ...]], action_id: int
):
    px, py, time_left, spike_phase, flash_parity, boulders_state = state
    time_left = int(time_left)
    if time_left <= 0:
        return None, False

    resting, falling = _coords_from_state(boulders_state)
    player = (int(px), int(py))

    pressed_before = _pressed_tiles(model, player, resting)
    open_doors_before = _open_doors_from_pressed(model, pressed_before)

    move = ACTION_TO_DELTA.get(int(action_id))
    if move is not None:
        dx, dy = move
        tx, ty = player[0] + dx, player[1] + dy
        target = (tx, ty)

        if _in_play_bounds(tx, ty, model) and target not in model["pits"]:
            if target in resting:
                bx, by = tx + dx, ty + dy
                if _cell_is_push_destination(model, bx, by, resting, falling, open_doors_before):
                    resting.remove(target)
                    if (bx, by) in model["pits"]:
                        falling.add((bx, by))
                    else:
                        resting.add((bx, by))
                    player = target
            elif not _cell_is_blocking_for_player(model, tx, ty, resting, falling, open_doors_before):
                player = target

    occupied = set(resting) | set(falling)
    moved_falling: set[tuple[int, int]] = set()
    new_resting: set[tuple[int, int]] = set()

    for x, y in sorted(occupied, key=lambda pos: (-int(pos[1]), int(pos[0]))):
        is_falling = (x, y) in falling
        should_fall = is_falling or ((x, y + 1) in model["pits"])

        if should_fall:
            nx, ny = int(x), int(y) + 1
            dst = (nx, ny)
            can_move = _in_play_bounds(nx, ny, model)
            if can_move and dst in model["walls"]:
                can_move = False
            if can_move and dst in model["doors"] and dst not in open_doors_before:
                can_move = False
            if can_move and dst in occupied:
                can_move = False
            if can_move and dst in moved_falling:
                can_move = False

            if can_move:
                if dst == player:
                    return None, False
                if dst in model["pits"]:
                    moved_falling.add(dst)
                else:
                    new_resting.add(dst)
            else:
                if (x, y) in model["pits"]:
                    moved_falling.add((x, y))
                else:
                    new_resting.add((x, y))
        else:
            new_resting.add((x, y))

    resting = new_resting
    falling = moved_falling

    pressed_after = _pressed_tiles(model, player, resting)
    open_doors_after = _open_doors_from_pressed(model, pressed_after)
    closed_after = set(model["doors"]) - open_doors_after
    if player in closed_after:
        return None, False

    next_phase = (int(spike_phase) + 1) % 4
    if player in model["spikes"] and _is_spike_extended(next_phase):
        return None, False

    if not _legal_move_exists(model, player, resting, falling, open_doors_after, next_phase):
        return None, False

    next_time = time_left - 1
    next_flash = (int(flash_parity) + 1) % 2

    won = player in model["goals"]
    if (not won) and next_time <= 0:
        return None, False

    next_state = (
        int(player[0]),
        int(player[1]),
        int(next_time),
        int(next_phase),
        int(next_flash),
        _state_from_coords(resting, falling),
    )
    return next_state, bool(won)


def _validate_level_solvable(model: dict[str, Any]) -> None:
    start = initial_search_state_from_model(model)
    queue = deque([start])
    seen: set[tuple[int, int, int, int, tuple[tuple[int, int, int], ...]]] = {
        (start[0], start[1], start[3], start[4], start[5])
    }

    while queue:
        state = queue.popleft()
        for action_id in (1, 2, 3, 4, 5):
            nxt, won = apply_action_transition(model, state, action_id)
            if nxt is None:
                continue
            if won:
                return
            key = (nxt[0], nxt[1], nxt[3], nxt[4], nxt[5])
            if key in seen:
                continue
            seen.add(key)
            queue.append(nxt)

    raise ValueError(f"{model['name']}: unsolvable under full transition rules")


def _build_level(spec: dict[str, Any]) -> Level:
    model = _parse_level(spec)
    serialized = _serialize_model(model)

    board = np.full((GRID_H, GRID_W), COLOR_FLOOR, dtype=np.int8)
    board[0, :] = COLOR_TIMEBAR_FULL

    sprite = Sprite(pixels=board, name="board", x=0, y=0, layer=0, tags=["board", "sys_static"], collidable=False)

    return Level(name=str(spec["name"]), grid_size=(GRID_W, GRID_H), sprites=[sprite], data={"model": serialized})


class BoulderPushRoll(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        camera = Camera(width=GRID_W, height=GRID_H, background=COLOR_PIT)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

        self._board: Sprite | None = None
        self._model: dict[str, Any] | None = None

        self._player = (1, 1)
        self._resting_boulders: set[tuple[int, int]] = set()
        self._falling_boulders: set[tuple[int, int]] = set()
        self._time_left = 1
        self._spike_phase = 0
        self._low_time_flash_parity = 0

        self._death_marker: tuple[int, int] | None = None
        self._death_timer = 0

    def on_set_level(self, level: Level) -> None:
        self._board = next(iter(level.get_sprites_by_name("board")), None)
        if self._board is None:
            raise RuntimeError("boulder_push_roll level missing board sprite")

        self._model = _deserialize_model(level)
        self._load_level_runtime()

    def _load_level_runtime(self) -> None:
        if self._model is None:
            return
        self._player = tuple(self._model["player_start"])
        self._resting_boulders = set(self._model["boulders"])
        self._falling_boulders = set()
        self._time_left = int(self._model["time_limit"])
        self._spike_phase = 0
        self._low_time_flash_parity = 0
        self._death_marker = None
        self._death_timer = 0
        self._render_board()

    def _pressed_tiles(self) -> set[tuple[int, int]]:
        if self._model is None:
            return set()
        return _pressed_tiles(self._model, self._player, self._resting_boulders)

    def _open_doors(self, pressed_tiles: set[tuple[int, int]]) -> set[tuple[int, int]]:
        if self._model is None:
            return set()
        return _open_doors_from_pressed(self._model, pressed_tiles)

    def _trigger_failure(self) -> None:
        self._death_marker = tuple(self._player)
        self._death_timer = 3

    def _tick_death_screen(self) -> None:
        if self._death_timer <= 0:
            return
        self._death_timer -= 1
        if self._death_timer == 0:
            self.lose()
        else:
            self._render_board()

    def _render_board(self) -> None:
        if self._board is None or self._model is None:
            return

        frame = np.full((GRID_H, GRID_W), COLOR_FLOOR, dtype=np.int8)

        for x in range(GRID_W):
            frame[0, x] = COLOR_TIMEBAR_EMPTY

        for x, y in self._model["pits"]:
            frame[y, x] = COLOR_PIT
        for x, y in self._model["walls"]:
            frame[y, x] = COLOR_WALL

        pressed = self._pressed_tiles()
        opened_doors = self._open_doors(pressed)

        for x, y in self._model["plates"]:
            frame[y, x] = COLOR_PLATE_PRESSED if (x, y) in pressed else COLOR_PLATE

        for x, y in self._model["doors"]:
            frame[y, x] = COLOR_DOOR_OPEN if (x, y) in opened_doors else COLOR_DOOR_CLOSED

        spike_color = COLOR_SPIKES_DANGER if _is_spike_extended(self._spike_phase) else COLOR_SPIKES_SAFE
        for x, y in self._model["spikes"]:
            frame[y, x] = spike_color

        for x, y in self._model["goals"]:
            frame[y, x] = COLOR_GOAL

        for x, y in self._resting_boulders:
            frame[y, x] = COLOR_BOULDER_RESTING
        for x, y in self._falling_boulders:
            frame[y, x] = COLOR_BOULDER_FALLING

        if self._death_timer > 0 and self._death_marker is not None:
            mx, my = self._death_marker
            frame[my, mx] = COLOR_WARNING
        else:
            px, py = self._player
            frame[py, px] = COLOR_PLAYER

        seg_steps = max(1, int(self._model["segment_steps"]))
        filled = int(np.ceil(float(max(0, self._time_left)) / float(seg_steps)))
        filled = max(0, min(GRID_W, filled))

        low_time = self._time_left <= int(np.floor(0.2 * int(self._model["time_limit"])))
        fill_color = COLOR_WARNING if (low_time and self._low_time_flash_parity == 1) else COLOR_TIMEBAR_FULL
        for x in range(GRID_W):
            frame[0, x] = fill_color if x < filled else COLOR_TIMEBAR_EMPTY

        self._board.pixels = frame

    def step(self) -> None:
        if self._model is None:
            self.complete_action()
            return

        if self._death_timer > 0:
            self._tick_death_screen()
            self.complete_action()
            return

        state = (
            int(self._player[0]),
            int(self._player[1]),
            int(self._time_left),
            int(self._spike_phase),
            int(self._low_time_flash_parity),
            _state_from_coords(set(self._resting_boulders), set(self._falling_boulders)),
        )

        action_id = int(self.action.id.value)
        next_state, won = apply_action_transition(self._model, state, action_id)

        if next_state is None:
            self._trigger_failure()
            self._render_board()
            self.complete_action()
            return

        self._player = (int(next_state[0]), int(next_state[1]))
        self._time_left = int(next_state[2])
        self._spike_phase = int(next_state[3])
        self._low_time_flash_parity = int(next_state[4])
        self._resting_boulders, self._falling_boulders = _coords_from_state(next_state[5])

        self._render_board()

        if won:
            self.next_level()

        self.complete_action()


__all__ = [
    "GAME_ID",
    "BoulderPushRoll",
    "_deserialize_model",
    "apply_action_transition",
    "initial_search_state_from_model",
]
