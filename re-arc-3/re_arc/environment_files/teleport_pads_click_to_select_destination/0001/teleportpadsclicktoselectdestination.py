from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "teleport_pads_click_to_select_destination-0001"

GRID_WIDTH = 32
GRID_HEIGHT = 18
TIME_MAX = 160

COLOR_TIME_EMPTY = 0
COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_PLAYER = 3
COLOR_EXIT = 4
COLOR_GOOD = 5
COLOR_KEY = 6
COLOR_GATE = 7
COLOR_ENTRY_BASE = 8
COLOR_ENTRY_GLOW = 9
COLOR_PAD_O = 10
COLOR_PAD_P = 11
COLOR_PAD_Q = 12
COLOR_PAD_R = 13
COLOR_HAZARD_ON = 14
COLOR_DIM = 15

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

NETWORK_CODE = {"O": 1, "P": 2, "Q": 3}
CODE_NETWORK = {value: key for key, value in NETWORK_CODE.items()}

LEVEL_LAYOUTS: list[tuple[str, list[str]]] = [
    (
        "Level 1 - First Teleport",
        [
            "================================",
            "################################",
            "#..S............#..............#",
            "#.......OO......#....XX........#",
            "#.......OO......#....XX........#",
            "#...............#..............#",
            "#...............#..oo..........#",
            "#...............#..oo..........#",
            "#...............#......####....#",
            "#...............#......####....#",
            "#...............#......####....#",
            "#...............#......####....#",
            "#...............#..........oo..#",
            "#...............#..........oo..#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "################################",
        ],
    ),
    (
        "Level 2 - Chained Teleports",
        [
            "================================",
            "################################",
            "#..S......#.........#..........#",
            "#....OO...#.........#.......XX.#",
            "#....OO...#...oo....#.......XX.#",
            "#.........#...oo....#..pp......#",
            "#.........#.........#..pp......#",
            "#.........#....#PP..#....###...#",
            "#.........#....#PP..#....###...#",
            "#.........#....#....#....###...#",
            "#.........#.oo.#....#....###...#",
            "#.........#.oo.#....#....###...#",
            "#.........#....#....#....###pp.#",
            "#.........#....#....#....###pp.#",
            "#.........#.........#..........#",
            "#.........#.........#..........#",
            "#.........#.........#..........#",
            "################################",
        ],
    ),
    (
        "Level 3 - Key Gate Return",
        [
            "================================",
            "################################",
            "#..XX...........#..............#",
            "#..XX...........#..............#",
            "#...............#..oo..........#",
            "#...............#..oo..........#",
            "#...............#......KK......#",
            "#...............#......KK......#",
            "######GG###########............#",
            "######GG###########............#",
            "#.........oo....#..OO..........#",
            "#.........oo....#..OO..........#",
            "#..S............#..............#",
            "#...............#..............#",
            "#....OO.........#..............#",
            "#....OO.........#..............#",
            "#...............#..............#",
            "################################",
        ],
    ),
    (
        "Level 4 - Spike Timing",
        [
            "================================",
            "################################",
            "#..S............#..............#",
            "#.....OO........#..............#",
            "#.....OO........#..oo..........#",
            "#...............#..oo..........#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............######^^######.#",
            "#...............#..............#",
            "#...............####^^##########",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..............#",
            "#...............#..........XX..#",
            "#...............#..........XX..#",
            "#...............#..............#",
            "################################",
        ],
    ),
    (
        "Level 5 - Laser Barrier",
        [
            "================================",
            "################################",
            "#..S......#.........############",
            "#....OO...#.........#####XX#####",
            "#....OO...#...oo....#####XX#####",
            "#.........#...oo....#####..#####",
            "#.........#....PP...#####..#####",
            "#.........#....PP...#####..#####",
            "#.........#.........#LL##..##LL#",
            "#.........#.........#LL##::##LL#",
            "#.........#.........#LL##..##LL#",
            "#.........#.........#####..#####",
            "#.........#.........#..pp......#",
            "#.........#.........#..pp......#",
            "#.........#.........#......pp..#",
            "#.........#.........#......pp..#",
            "#.........#.........#..........#",
            "################################",
        ],
    ),
    (
        "Level 6 - Router Final",
        [
            "================================",
            "################################",
            "#........#........#.rr.QQ..XX..#",
            "#..S.....#........#.rr.QQ..XX..#",
            "#........#.oo.....#........GG..#",
            "#....OO..#.oo.....#........GG..#",
            "#....OO..#........##############",
            "#........#.....PP.#.pp.........#",
            "#........#.....PP.#.pp....:QQ..#",
            "#........#........#.....!.:QQ..#",
            "#........#........#.qq......rr.#",
            "#........#........#.qq......rr.#",
            "#........#........##############",
            "#........#........#.qq.........#",
            "#........#.....oo.#.qq...^^KK..#",
            "#........#.....oo.#...QQ.^^KK..#",
            "#........#........#...QQ.......#",
            "################################",
        ],
    ),
]


def _sorted_cells(cells: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((int(x), int(y)) for x, y in cells)


def _neighbors4(x: int, y: int) -> list[tuple[int, int]]:
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _connected_components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        component = [start]
        while queue:
            x, y = queue.popleft()
            for nx, ny in _neighbors4(x, y):
                if (nx, ny) not in remaining:
                    continue
                remaining.remove((nx, ny))
                component.append((nx, ny))
                queue.append((nx, ny))
        components.append(_sorted_cells(component))
    components.sort(key=lambda group: (group[0][1], group[0][0]))
    return components


def _parse_layout(name: str, lines: list[str]) -> dict:
    if len(lines) != GRID_HEIGHT:
        raise ValueError(f"{name}: expected {GRID_HEIGHT} rows, got {len(lines)}")

    walls: set[tuple[int, int]] = set()
    exits: set[tuple[int, int]] = set()
    keys: set[tuple[int, int]] = set()
    gates: set[tuple[int, int]] = set()
    spikes: set[tuple[int, int]] = set()
    laser_paths: set[tuple[int, int]] = set()
    emitters: set[tuple[int, int]] = set()
    switches: set[tuple[int, int]] = set()

    entries = {"O": set(), "P": set(), "Q": set()}
    pads = {"o": set(), "p": set(), "q": set(), "r": set()}

    spawn: tuple[int, int] | None = None

    valid_chars = {"=", ".", "#", "S", "X", "O", "o", "P", "p", "Q", "q", "r", "K", "G", "^", ":", "L", "!"}

    for y, row in enumerate(lines):
        if len(row) != GRID_WIDTH:
            raise ValueError(f"{name}: expected width {GRID_WIDTH}, got {len(row)} at row {y}")
        if y == 0:
            if row != "=" * GRID_WIDTH:
                raise ValueError(f"{name}: row 0 must be full '=' timebar")
            continue

        for x, ch in enumerate(row):
            if ch not in valid_chars:
                raise ValueError(f"{name}: unsupported tile '{ch}' at {(x, y)}")
            if ch == "#":
                walls.add((x, y))
            elif ch == "S":
                if spawn is not None:
                    raise ValueError(f"{name}: expected exactly one spawn")
                spawn = (x, y)
            elif ch == "X":
                exits.add((x, y))
            elif ch == "K":
                keys.add((x, y))
            elif ch == "G":
                gates.add((x, y))
            elif ch == "^":
                spikes.add((x, y))
            elif ch == ":":
                laser_paths.add((x, y))
            elif ch == "L":
                emitters.add((x, y))
            elif ch == "!":
                switches.add((x, y))
            elif ch in entries:
                entries[ch].add((x, y))
            elif ch in pads:
                pads[ch].add((x, y))

    if spawn is None:
        raise ValueError(f"{name}: missing spawn")
    if not exits:
        raise ValueError(f"{name}: missing exit tiles")

    entry_components = {network: _connected_components(cells) for network, cells in entries.items()}
    entry_component_lookup: dict[str, dict[tuple[int, int], int]] = {}
    for network, components in entry_components.items():
        lookup: dict[tuple[int, int], int] = {}
        for idx, cells in enumerate(components):
            for cell in cells:
                lookup[cell] = idx
        entry_component_lookup[network] = lookup

    return {
        "name": name,
        "width": GRID_WIDTH,
        "height": GRID_HEIGHT,
        "time_max": TIME_MAX,
        "start": (int(spawn[0]), int(spawn[1])),
        "walls": _sorted_cells(walls),
        "exits": _sorted_cells(exits),
        "keys": _sorted_cells(keys),
        "gates": _sorted_cells(gates),
        "spikes": _sorted_cells(spikes),
        "laser_paths": _sorted_cells(laser_paths),
        "emitters": _sorted_cells(emitters),
        "switches": _sorted_cells(switches),
        "entries": {network: _sorted_cells(cells) for network, cells in entries.items()},
        "pads": {kind: _sorted_cells(cells) for kind, cells in pads.items()},
        "entry_components": {
            network: [[(int(x), int(y)) for x, y in group] for group in groups]
            for network, groups in entry_components.items()
        },
        "entry_component_lookup": {
            network: {f"{x},{y}": int(idx) for (x, y), idx in lookup.items()}
            for network, lookup in entry_component_lookup.items()
        },
    }


def _runtime_model_from_serialized(raw: dict) -> dict:
    temp_level = Level(name=str(raw.get("name") or "Level"), grid_size=(GRID_WIDTH, GRID_HEIGHT), sprites=[], data=raw)
    return _deserialize_model(temp_level)


def _deserialize_model(level: Level) -> dict:
    raw = {}
    for key in (
        "name",
        "width",
        "height",
        "time_max",
        "start",
        "walls",
        "exits",
        "keys",
        "gates",
        "spikes",
        "laser_paths",
        "emitters",
        "switches",
        "entries",
        "pads",
        "entry_components",
        "entry_component_lookup",
    ):
        raw[key] = level.get_data(key)

    def _to_set(values) -> set[tuple[int, int]]:
        return {tuple(int(v) for v in item) for item in (values or [])}

    entries = {network: _to_set((raw.get("entries") or {}).get(network)) for network in ("O", "P", "Q")}
    pads = {kind: _to_set((raw.get("pads") or {}).get(kind)) for kind in ("o", "p", "q", "r")}

    entry_components: dict[str, list[list[tuple[int, int]]]] = {}
    for network in ("O", "P", "Q"):
        groups = (raw.get("entry_components") or {}).get(network) or []
        entry_components[network] = [[tuple(int(v) for v in cell) for cell in group] for group in groups]

    entry_component_lookup: dict[str, dict[tuple[int, int], int]] = {}
    for network in ("O", "P", "Q"):
        lookup_raw = (raw.get("entry_component_lookup") or {}).get(network) or {}
        lookup: dict[tuple[int, int], int] = {}
        for key, idx in lookup_raw.items():
            sx, sy = str(key).split(",", 1)
            lookup[(int(sx), int(sy))] = int(idx)
        entry_component_lookup[network] = lookup

    return {
        "name": str(raw.get("name") or "Level"),
        "width": int(raw.get("width") or GRID_WIDTH),
        "height": int(raw.get("height") or GRID_HEIGHT),
        "time_max": int(raw.get("time_max") or TIME_MAX),
        "start": tuple(int(v) for v in (raw.get("start") or (1, 1))),
        "walls": _to_set(raw.get("walls")),
        "exits": _to_set(raw.get("exits")),
        "keys": _to_set(raw.get("keys")),
        "gates": _to_set(raw.get("gates")),
        "spikes": _to_set(raw.get("spikes")),
        "laser_paths": _to_set(raw.get("laser_paths")),
        "emitters": _to_set(raw.get("emitters")),
        "switches": _to_set(raw.get("switches")),
        "entries": entries,
        "pads": pads,
        "entry_components": entry_components,
        "entry_component_lookup": entry_component_lookup,
    }


def _entry_network_at(model: dict, pos: tuple[int, int]) -> str | None:
    for network in ("O", "P", "Q"):
        if pos in model["entries"][network]:
            return network
    return None


def _valid_pads_for_network(model: dict, network: str, q_mode_a: bool) -> set[tuple[int, int]]:
    if network == "O":
        return set(model["pads"]["o"])
    if network == "P":
        return set(model["pads"]["p"])
    if network == "Q":
        return set(model["pads"]["q"] if q_mode_a else model["pads"]["r"])
    return set()


def _hazards_active(tick_mod: int) -> bool:
    return int(tick_mod) % 4 in {2, 3}


def _is_blocked(model: dict, pos: tuple[int, int], gate_open_now: bool) -> bool:
    x, y = pos
    if x < 0 or y < 1 or x >= int(model["width"]) or y >= int(model["height"]):
        return True
    if pos in model["walls"]:
        return True
    if pos in model["emitters"]:
        return True
    return bool(not gate_open_now and pos in model["gates"])


def initial_search_state_from_model(model: dict) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
    sx, sy = model["start"]
    return (int(sx), int(sy), int(model["time_max"]), 0, 0, 1, 0, -1, -1, -1, 0)


def apply_action_transition(
    model: dict,
    state: tuple[int, int, int, int, int, int, int, int, int, int, int],
    action: tuple[str, int | tuple[int, int] | None],
) -> tuple[tuple[int, int, int, int, int, int, int, int, int, int, int] | None, bool]:
    (
        px,
        py,
        time_remaining,
        tick_mod,
        has_key,
        q_mode_a,
        pending_net,
        pending_dx,
        pending_dy,
        pending_comp,
        _gate_open_now,
    ) = state

    tick_mod = (int(tick_mod) + 1) % 4
    time_remaining = int(time_remaining) - 1
    gate_open_now = 1 if int(has_key) else 0

    pos = (int(px), int(py))
    has_key_now = int(has_key)
    q_mode = 1 if int(q_mode_a) else 0

    pending_old = {"net": int(pending_net), "dx": int(pending_dx), "dy": int(pending_dy), "comp": int(pending_comp)}
    pending_new = {"net": 0, "dx": -1, "dy": -1, "comp": -1}

    kind, payload = action

    if kind == "move":
        action_id = int(payload or 0)
        dx, dy = MOVE_DELTAS.get(action_id, (0, 0))
        candidate = (pos[0] + dx, pos[1] + dy)
        if not _is_blocked(model, candidate, gate_open_now=bool(gate_open_now)):
            pos = candidate
    elif kind == "space":
        if pos in model["switches"]:
            q_mode = 0 if q_mode else 1
    elif kind == "click":
        target = tuple(int(v) for v in (payload or (-1, -1)))
        network = _entry_network_at(model, pos)
        if network is not None:
            valid_targets = _valid_pads_for_network(model, network, q_mode_a=bool(q_mode))
            if target in valid_targets:
                component_idx = int(model["entry_component_lookup"][network].get(pos, -1))
                if component_idx >= 0:
                    pending_new = {
                        "net": int(NETWORK_CODE[network]),
                        "dx": int(target[0]),
                        "dy": int(target[1]),
                        "comp": int(component_idx),
                    }

    if pending_old["net"] > 0:
        old_network = CODE_NETWORK.get(int(pending_old["net"]))
        if old_network is not None:
            comp_index = int(pending_old["comp"])
            components = model["entry_components"].get(old_network) or []
            if 0 <= comp_index < len(components) and (pos in set(tuple(cell) for cell in components[comp_index])):
                pos = (int(pending_old["dx"]), int(pending_old["dy"]))

    if has_key_now == 0 and pos in model["keys"]:
        has_key_now = 1

    if time_remaining == 0:
        return None, False

    hazard_cells = model["spikes"] | model["laser_paths"]
    if _hazards_active(tick_mod) and pos in hazard_cells:
        return None, False

    has_gates = bool(model["gates"])
    won = pos in model["exits"] and ((not has_gates) or bool(gate_open_now))

    next_state = (
        int(pos[0]),
        int(pos[1]),
        int(time_remaining),
        int(tick_mod),
        int(has_key_now),
        int(q_mode),
        int(pending_new["net"]),
        int(pending_new["dx"]),
        int(pending_new["dy"]),
        int(pending_new["comp"]),
        int(gate_open_now),
    )
    return next_state, bool(won)


def find_solution_actions(model: dict):
    start = initial_search_state_from_model(model)
    queue = deque([start])
    prev = {start: None}
    prev_action: dict[
        tuple[int, int, int, int, int, int, int, int, int, int, int], tuple[str, int | tuple[int, int] | None]
    ] = {}
    best_time: dict[tuple[int, int, int, int, int, int, int, int, int, int], int] = {}

    def _dominance_key(s):
        return (
            int(s[0]),
            int(s[1]),
            int(s[3]),
            int(s[4]),
            int(s[5]),
            int(s[6]),
            int(s[7]),
            int(s[8]),
            int(s[9]),
            int(s[10]),
        )

    best_time[_dominance_key(start)] = int(start[2])

    while queue:
        state = queue.popleft()

        candidate_actions: list[tuple[str, int | tuple[int, int] | None]] = [
            ("move", 1),
            ("move", 2),
            ("move", 3),
            ("move", 4),
            ("space", None),
        ]

        on_network = _entry_network_at(model, (int(state[0]), int(state[1])))
        if on_network is not None:
            q_mode = bool(int(state[5]))
            for cell in sorted(_valid_pads_for_network(model, on_network, q_mode_a=q_mode)):
                candidate_actions.append(("click", (int(cell[0]), int(cell[1]))))

        for action in candidate_actions:
            next_state, won = apply_action_transition(model, state, action)
            if next_state is None:
                continue

            key = _dominance_key(next_state)
            next_time = int(next_state[2])
            prior_best = best_time.get(key)
            if prior_best is not None and prior_best >= next_time:
                continue
            best_time[key] = next_time

            if next_state not in prev:
                prev[next_state] = state
                prev_action[next_state] = action
                queue.append(next_state)

            if won:
                plan = []
                cursor = next_state
                while prev[cursor] is not None:
                    plan.append(prev_action[cursor])
                    cursor = prev[cursor]  # type: ignore[index]
                plan.reverse()
                return plan

    return None


def _build_level(name: str, lines: list[str]) -> Level:
    model = _parse_layout(name, lines)
    if find_solution_actions(_runtime_model_from_serialized(model)) is None:
        raise ValueError(f"{name}: level is unsolvable under full mechanics")

    board = np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_FLOOR, dtype=np.int8)
    player = np.array([[COLOR_PLAYER]], dtype=np.int8)

    sprites = [
        Sprite(pixels=board, name="board", x=0, y=0, collidable=False, layer=0, tags=["board"]),
        Sprite(
            pixels=player,
            name="player",
            x=int(model["start"][0]),
            y=int(model["start"][1]),
            collidable=False,
            layer=10,
            tags=["player"],
        ),
    ]

    return Level(name=name, grid_size=(GRID_WIDTH, GRID_HEIGHT), sprites=sprites, data=model)


class TeleportPadsClickToSelectDestination(ARCBaseGame):
    def __init__(self):
        levels = [_build_level(name, lines) for name, lines in LEVEL_LAYOUTS]
        camera = Camera(0, 0, GRID_WIDTH, GRID_HEIGHT, 4, 4, [])
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
        )

        self._model: dict | None = None
        self._board_sprite: Sprite | None = None
        self._player_sprite: Sprite | None = None

        self._tick = 0
        self._time_remaining = TIME_MAX
        self._player = (1, 1)
        self._has_key = False
        self._q_mode_a = True
        self._gate_open_now = False

        self._pending: dict | None = None
        self._warp_fx_cells: set[tuple[int, int]] = set()
        self._warp_fx_life = 0

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level)

        self._board_sprite = next(iter(self.current_level.get_sprites_by_name("board")), None)
        self._player_sprite = next(iter(self.current_level.get_sprites_by_name("player")), None)

        sx, sy = self._model["start"]
        self._tick = 0
        self._time_remaining = int(self._model["time_max"])
        self._player = (int(sx), int(sy))
        self._has_key = False
        self._q_mode_a = True
        self._gate_open_now = False
        self._pending = None
        self._warp_fx_cells = set()
        self._warp_fx_life = 0

        self._render()

    def _model_required(self) -> dict:
        if self._model is None:
            raise RuntimeError("level model is not initialized")
        return self._model

    def _parse_click_cell(self) -> tuple[int, int] | None:
        payload = self.action.data if isinstance(self.action.data, dict) else None
        if not payload:
            return None
        try:
            display_x = int(payload.get("x", -1))
            display_y = int(payload.get("y", -1))
        except (TypeError, ValueError):
            return None
        grid = self.camera.display_to_grid(display_x, display_y)
        if grid is None:
            return None
        gx, gy = int(grid[0]), int(grid[1])
        model = self._model_required()
        if gx < 0 or gy < 0 or gx >= int(model["width"]) or gy >= int(model["height"]):
            return None
        return gx, gy

    def _entry_network_at_player(self) -> str | None:
        return _entry_network_at(self._model_required(), self._player)

    def _entry_component_index(self, network: str, pos: tuple[int, int]) -> int:
        model = self._model_required()
        return int(model["entry_component_lookup"].get(network, {}).get(pos, -1))

    def _valid_pads_for_network(self, network: str) -> set[tuple[int, int]]:
        return _valid_pads_for_network(self._model_required(), network, q_mode_a=self._q_mode_a)

    def _blocked(self, candidate: tuple[int, int], gate_open_now: bool) -> bool:
        return _is_blocked(self._model_required(), candidate, gate_open_now=gate_open_now)

    def _apply_move(self, dx: int, dy: int, gate_open_now: bool) -> None:
        target = (int(self._player[0] + dx), int(self._player[1] + dy))
        if self._blocked(target, gate_open_now=gate_open_now):
            return
        self._player = target

    def _apply_click(self) -> dict | None:
        click_cell = self._parse_click_cell()
        if click_cell is None:
            return None
        network = self._entry_network_at_player()
        if network is None:
            return None

        valid_pads = self._valid_pads_for_network(network)
        if click_cell not in valid_pads:
            return None

        comp_index = self._entry_component_index(network, self._player)
        if comp_index < 0:
            return None

        return {
            "network": str(network),
            "dest": (int(click_cell[0]), int(click_cell[1])),
            "entry_comp": int(comp_index),
        }

    def _resolve_pending_old(self, pending_old: dict | None) -> None:
        if pending_old is None:
            return
        network = str(pending_old.get("network") or "")
        dest = tuple(int(v) for v in (pending_old.get("dest") or (-1, -1)))
        comp_index = int(pending_old.get("entry_comp", -1))

        model = self._model_required()
        components = model["entry_components"].get(network) or []
        if not (0 <= comp_index < len(components)):
            return

        entry_component = {tuple(cell) for cell in components[comp_index]}
        if self._player not in entry_component:
            return

        self._player = (int(dest[0]), int(dest[1]))
        self._warp_fx_cells = set(entry_component)
        self._warp_fx_cells.add(self._player)
        self._warp_fx_life = 1

    def _collect_key_if_present(self) -> None:
        model = self._model_required()
        if self._has_key:
            return
        if self._player in model["keys"]:
            self._has_key = True

    def _is_on_active_hazard(self) -> bool:
        model = self._model_required()
        if not _hazards_active(self._tick):
            return False
        return self._player in (model["spikes"] | model["laser_paths"])

    def _check_win(self) -> bool:
        model = self._model_required()
        if self._player not in model["exits"]:
            return False
        if not model["gates"]:
            return True
        return bool(self._gate_open_now)

    def _timebar_filled(self) -> int:
        if self._time_remaining <= 0:
            return 0
        return max(0, min(GRID_WIDTH, int((self._time_remaining + 4) // 5)))

    def _render(self) -> None:
        model = self._model_required()
        if self._board_sprite is None or self._player_sprite is None:
            return

        board = np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_FLOOR, dtype=np.int8)

        filled = self._timebar_filled()
        if filled > 0:
            board[0, :filled] = COLOR_GOOD
        if filled < GRID_WIDTH:
            board[0, filled:] = COLOR_TIME_EMPTY

        hazards_on = _hazards_active(self._tick)
        odd_tick = bool(self._tick % 2)

        for x, y in model["walls"]:
            board[y, x] = COLOR_WALL

        for x, y in model["gates"]:
            board[y, x] = COLOR_FLOOR if self._gate_open_now else COLOR_GATE

        for x, y in model["exits"]:
            board[y, x] = COLOR_EXIT

        if not self._has_key:
            for x, y in model["keys"]:
                board[y, x] = COLOR_KEY

        for x, y in model["switches"]:
            board[y, x] = COLOR_DIM

        for x, y in model["spikes"]:
            board[y, x] = COLOR_HAZARD_ON if hazards_on else COLOR_DIM

        for x, y in model["laser_paths"]:
            board[y, x] = COLOR_HAZARD_ON if hazards_on else COLOR_DIM

        for x, y in model["emitters"]:
            board[y, x] = COLOR_HAZARD_ON if hazards_on else COLOR_DIM

        entry_pulse = COLOR_ENTRY_GLOW if odd_tick else COLOR_ENTRY_BASE
        for network in ("O", "P", "Q"):
            for x, y in model["entries"][network]:
                board[y, x] = entry_pulse

        pad_base_colors = {"o": COLOR_PAD_O, "p": COLOR_PAD_P, "q": COLOR_PAD_Q, "r": COLOR_PAD_R}
        for pad_kind, base_color in pad_base_colors.items():
            for x, y in model["pads"][pad_kind]:
                board[y, x] = int(base_color)

        active_network = self._entry_network_at_player()
        if active_network is not None:
            for x, y in self._valid_pads_for_network(active_network):
                board[y, x] = COLOR_DIM if odd_tick else int(board[y, x])

        if self._pending is not None:
            dx, dy = self._pending["dest"]
            board[dy, dx] = COLOR_DIM if odd_tick else COLOR_GOOD

        if self._warp_fx_life > 0:
            for x, y in self._warp_fx_cells:
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                    board[y, x] = COLOR_ENTRY_GLOW

        self._board_sprite.pixels = board
        self._player_sprite.set_position(int(self._player[0]), int(self._player[1]))
        self._player_sprite.pixels = np.array([[COLOR_PLAYER]], dtype=np.int8)

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

        model = self._model_required()

        self._tick += 1
        self._time_remaining = max(0, self._time_remaining - 1)
        self._gate_open_now = bool(self._has_key)

        if self._warp_fx_life > 0:
            self._warp_fx_life -= 1
            if self._warp_fx_life <= 0:
                self._warp_fx_cells = set()

        pending_old = self._pending
        pending_new = None

        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action_id]
            self._apply_move(dx, dy, gate_open_now=self._gate_open_now)
        elif action_id == int(GameAction.ACTION5.value):
            if self._player in model["switches"]:
                self._q_mode_a = not self._q_mode_a
        elif action_id == int(GameAction.ACTION6.value):
            pending_new = self._apply_click()

        self._resolve_pending_old(pending_old)
        self._collect_key_if_present()
        self._pending = pending_new

        if self._time_remaining == 0 or self._is_on_active_hazard():
            self.lose()
            self.complete_action()
            return

        if self._check_win():
            self.next_level()
            self.complete_action()
            return

        self._render()
        self.complete_action()


__all__ = [
    "TeleportPadsClickToSelectDestination",
    "_deserialize_model",
    "apply_action_transition",
    "find_solution_actions",
    "initial_search_state_from_model",
]
