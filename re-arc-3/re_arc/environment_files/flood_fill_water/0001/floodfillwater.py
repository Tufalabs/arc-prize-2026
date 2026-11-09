from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "flood_fill_water-0001"

ACTION_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
ACTION_SPACE = int(GameAction.ACTION5.value)

COLOR_FLOOR = 0
COLOR_WALL = 1
COLOR_PLAYER = 2
COLOR_EXIT = 3
COLOR_WATER = 4
COLOR_FOAM = 5
COLOR_SOURCE_OFF = 6
COLOR_SOURCE_ON = 7
COLOR_GATE_CLOSED = 8
COLOR_GATE_OPEN = 9
COLOR_VALVE = 10
COLOR_DRAIN = 11
COLOR_SENSOR = 12
COLOR_CONDUIT = 13
COLOR_CRATE = 14
COLOR_PROTECTED = 15

PLAY_CHARS = {"#", ".", "@", "X", "s", "g", "v", "d", "p", ":", "c", "!"}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    width: int
    height: int
    time_max: int
    layout: tuple[str, ...]
    crate_mode: str
    valve_links: tuple[tuple[str, ...], ...]
    sensor_links_open: tuple[tuple[str, ...], ...]
    initial_sources_on: tuple[str, ...]
    initial_gates_open: tuple[str, ...]


@dataclass(frozen=True)
class LevelModel:
    name: str
    width: int
    height: int
    time_max: int
    player_start: int
    walls: int
    conduit: int
    protected: int
    exits: int
    floors_for_crates: int
    passable_player_base: int
    flood_passable_base: int
    sources: tuple[int, ...]
    gates: tuple[int, ...]
    valves: tuple[int, ...]
    drains: tuple[int, ...]
    sensors: tuple[int, ...]
    crates: tuple[int, ...]
    valve_source_toggle_masks: tuple[int, ...]
    valve_gate_toggle_masks: tuple[int, ...]
    sensor_gate_open_masks: tuple[int, ...]
    initial_source_on: int
    initial_gate_open: int
    neighbors: tuple[int, ...]
    drain_adjacent: tuple[int, ...]
    exit_cells: tuple[int, ...]


@dataclass(frozen=True)
class GameState:
    player: int
    crates: tuple[int, ...]
    source_on: int
    gate_open: int
    sensor_wet: int
    pending_gate_open: int
    foam: int
    settled: int
    time_left: int
    tick: int


@dataclass(frozen=True)
class StepInfo:
    gate_flash: int
    drain_flash: int
    status: str


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        name="Level 1 — Start it, then race",
        width=28,
        height=18,
        time_max=80,
        crate_mode="domino",
        valve_links=(("S1", "G1"),),
        sensor_links_open=(),
        initial_sources_on=(),
        initial_gates_open=(),
        layout=(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "############################",
            "#..ss:.....######..........#",
            "#..ss:.....######..........#",
            "#....:.....######..........#",
            "#....:.....######..........#",
            "#....:.....######..........#",
            "#....:.....######..........#",
            "#....:.....######..........#",
            "#....:.....######..........#",
            "#....:.....######..........#",
            "#....:.............#########",
            "#....:::::::::::::gg..XX..##",
            "#....:::::::::::::gg..XX..##",
            "#..vv:.............#########",
            "#.@vv:.............#########",
            "#..........................#",
            "############################",
        ),
    ),
    LevelSpec(
        name="Level 2 — Close the gate behind you",
        width=28,
        height=18,
        time_max=90,
        crate_mode="domino",
        valve_links=(("S1", "G1"), ("G1",)),
        sensor_links_open=(),
        initial_sources_on=(),
        initial_gates_open=(),
        layout=(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "############################",
            "#..ss....:...##............#",
            "#..ss....:...##............#",
            "#........:...##............#",
            "#........:...##............#",
            "#........:...##............#",
            "#........:...##..vv........#",
            "#........:...gg..vv........#",
            "#........:...gg............#",
            "#........:...##............#",
            "#........:...##............#",
            "#........:...##......XX....#",
            "#..vv....:...##......XX....#",
            "#.@vv....:...##............#",
            "#............##............#",
            "#............##............#",
            "############################",
        ),
    ),
    LevelSpec(
        name="Level 3 — Turn it off, let the drain clear it",
        width=28,
        height=18,
        time_max=110,
        crate_mode="domino",
        valve_links=(("S1",), ("G1",)),
        sensor_links_open=(),
        initial_sources_on=("S1",),
        initial_gates_open=(),
        layout=(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "############################",
            "#................##........#",
            "#..ss:...........##........#",
            "#..ss:...........##........#",
            "#....:...........##........#",
            "#....:...........##........#",
            "#....:...........##........#",
            "#....:...dd......##........#",
            "#....:...dd......##........#",
            "#....:...........##........#",
            "#....:...........##....XX..#",
            "#....:...........gg....XX..#",
            "#..vv:.vv::::::::gg........#",
            "#.@vv:.vv::::::::##........#",
            "#................##........#",
            "#................##........#",
            "############################",
        ),
    ),
    LevelSpec(
        name="Level 4 — Wet the sensor",
        width=28,
        height=18,
        time_max=120,
        crate_mode="domino",
        valve_links=(("S1",), ("G1",)),
        sensor_links_open=(("G2",),),
        initial_sources_on=(),
        initial_gates_open=(),
        layout=(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "############################",
            "#..ss......................#",
            "#..ss......................#",
            "#..........................#",
            "#.........########.........#",
            "#.........gg..pp..#........#",
            "#.........gg..pp..#........#",
            "#.........########.........#",
            "#..........................#",
            "#............dd....#########",
            "#..................gg..XX.##",
            "#..................gg..XX.##",
            "#..................##.....##",
            "#..vv:vv:::::::::::#########",
            "#.@vv:vv:::::::::::........#",
            "#..........................#",
            "############################",
        ),
    ),
    LevelSpec(
        name="Level 5 — Crates as a dam",
        width=28,
        height=18,
        time_max=140,
        crate_mode="domino",
        valve_links=(("S1",),),
        sensor_links_open=(("G1",),),
        initial_sources_on=(),
        initial_gates_open=(),
        layout=(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "############################",
            "#..ss:...........##........#",
            "#..ss:...........##........#",
            "#....:...........##........#",
            "#....:...pp......##........#",
            "#....:...pp......##....XX..#",
            "#....:...........##....XX..#",
            "#....:.........cc...........#",
            "#....:...........##........#",
            "#....:...........##........#",
            "#....:...........gg........#",
            "#..vv:...........gg........#",
            "#.@vv:...........##........#",
            "#................##........#",
            "#................##........#",
            "#................##........#",
            "############################",
        ),
    ),
    LevelSpec(
        name="Level 6 — Full system",
        width=32,
        height=22,
        time_max=180,
        crate_mode="single",
        valve_links=(("S1",), ("S2",), ("G1",)),
        sensor_links_open=(("G2",),),
        initial_sources_on=(),
        initial_gates_open=(),
        layout=(
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
            "################################",
            "#..ss...............########### #",
            "#..ss...............########### #",
            "#..................c..!!!....## #",
            "#...................#.!!!....## #",
            "#...................########### #",
            "#...................########### #",
            "#............pp.....#........## #",
            "#............pp.....#........## #",
            "#..........gg.......#........## #",
            "#..........gg.......#........## #",
            "#...........dd......#........## #",
            "#...........dd......#........## #",
            "#...................gg........## #",
            "#...................gg........## #",
            "#..ss...............#........## #",
            "#..ss...............#....XX..## #",
            "#..vv..vv..vv...... .#....XX..## #",
            "#.@vv..vv..vv...... .#........## #",
            "#...................########### #",
            "################################",
        ),
    ),
)


def _bit(idx: int) -> int:
    return 1 << idx


def _iter_bits(mask: int):
    value = int(mask)
    while value:
        lsb = value & -value
        idx = lsb.bit_length() - 1
        yield idx
        value ^= lsb


def _to_idx(x: int, y: int, width: int) -> int:
    return y * width + x


def _to_xy(idx: int, width: int) -> tuple[int, int]:
    return idx % width, idx // width


def _checker(x: int, y: int, a: int, b: int, tick: int = 0) -> int:
    return a if ((x + y + tick) % 2 == 0) else b


def _shift_mask(mask: int, dx: int, dy: int, width: int, height: int) -> int | None:
    out = 0
    for idx in _iter_bits(mask):
        x, y = _to_xy(idx, width)
        nx = x + dx
        ny = y + dy
        if nx < 0 or nx >= width or ny < 1 or ny >= height:
            return None
        out |= _bit(_to_idx(nx, ny, width))
    return out


def _normalize_layout(spec: LevelSpec) -> list[str]:
    rows: list[str] = []
    for raw in spec.layout:
        row = raw.replace(" ", ".")
        row = "".join(ch if ch in PLAY_CHARS or ch == "^" else "." for ch in row)
        if len(row) < spec.width:
            row = row + ("." * (spec.width - len(row)))
        elif len(row) > spec.width:
            row = row[: spec.width]
        rows.append(row)
    if len(rows) != spec.height:
        raise ValueError(f"{spec.name}: expected {spec.height} rows, got {len(rows)}")
    return rows


def _extract_blocks(rows: list[str], width: int, height: int, token: str) -> list[int]:
    visited: set[tuple[int, int]] = set()
    masks: list[int] = []
    for y in range(height):
        for x in range(width):
            if rows[y][x] != token or (x, y) in visited:
                continue
            stack = [(x, y)]
            mask = 0
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                if cx < 0 or cx >= width or cy < 0 or cy >= height:
                    continue
                if rows[cy][cx] != token:
                    continue
                visited.add((cx, cy))
                mask |= _bit(_to_idx(cx, cy, width))
                stack.extend(((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)))
            masks.append(mask)
    return masks


def _extract_crates(rows: list[str], width: int, height: int, mode: str) -> list[int]:
    visited: set[tuple[int, int]] = set()
    out: list[int] = []
    for y in range(height):
        for x in range(width):
            if rows[y][x] != "c" or (x, y) in visited:
                continue
            if mode == "single":
                visited.add((x, y))
                out.append(_bit(_to_idx(x, y, width)))
                continue
            if x + 1 < width and rows[y][x + 1] == "c" and (x + 1, y) not in visited:
                visited.add((x, y))
                visited.add((x + 1, y))
                out.append(_bit(_to_idx(x, y, width)) | _bit(_to_idx(x + 1, y, width)))
            else:
                raise ValueError(f"domino crate malformed at {(x, y)}")
    return out


def _cells_from_mask(mask: int, width: int) -> tuple[int, ...]:
    return tuple(sorted(_iter_bits(mask), key=lambda idx: (_to_xy(idx, width)[1], _to_xy(idx, width)[0])))


def _build_model(spec: LevelSpec) -> LevelModel:
    rows = _normalize_layout(spec)
    width = spec.width
    height = spec.height

    if not all(ch == "^" for ch in rows[0]):
        raise ValueError(f"{spec.name}: row 0 must be the timebar")

    player_start: int | None = None
    walls = 0
    conduit = 0
    protected = 0
    floors_for_crates = 0

    for y in range(1, height):
        row = rows[y]
        for x, cell in enumerate(row):
            idx = _to_idx(x, y, width)
            if cell == "#":
                walls |= _bit(idx)
            elif cell == ":":
                conduit |= _bit(idx)
            elif cell == "!":
                protected |= _bit(idx)
            elif cell == "@":
                player_start = idx
                floors_for_crates |= _bit(idx)
            elif cell == ".":
                floors_for_crates |= _bit(idx)

    if player_start is None:
        raise ValueError(f"{spec.name}: missing player start")

    sources = tuple(_extract_blocks(rows, width, height, "s"))
    gates = tuple(_extract_blocks(rows, width, height, "g"))
    valves = tuple(_extract_blocks(rows, width, height, "v"))
    drains = tuple(_extract_blocks(rows, width, height, "d"))
    sensors = tuple(_extract_blocks(rows, width, height, "p"))
    exits = tuple(_extract_blocks(rows, width, height, "X"))
    crates = tuple(_extract_crates(rows, width, height, spec.crate_mode))

    exits_mask = 0
    for mask in exits:
        exits_mask |= mask

    source_name_to_idx = {f"S{i + 1}": i for i in range(len(sources))}
    gate_name_to_idx = {f"G{i + 1}": i for i in range(len(gates))}

    valve_source_masks: list[int] = []
    valve_gate_masks: list[int] = []
    for links in spec.valve_links:
        source_mask = 0
        gate_mask = 0
        for label in links:
            if label.startswith("S"):
                source_mask |= _bit(source_name_to_idx[label])
            elif label.startswith("G"):
                gate_mask |= _bit(gate_name_to_idx[label])
            else:
                raise ValueError(f"{spec.name}: unknown valve target {label}")
        valve_source_masks.append(source_mask)
        valve_gate_masks.append(gate_mask)

    if len(valve_source_masks) != len(valves):
        raise ValueError(f"{spec.name}: valve links mismatch")

    sensor_gate_open_masks: list[int] = []
    for links in spec.sensor_links_open:
        gate_mask = 0
        for label in links:
            gate_mask |= _bit(gate_name_to_idx[label])
        sensor_gate_open_masks.append(gate_mask)

    if len(sensor_gate_open_masks) != len(sensors):
        raise ValueError(f"{spec.name}: sensor links mismatch")

    initial_source_on = 0
    for label in spec.initial_sources_on:
        initial_source_on |= _bit(source_name_to_idx[label])

    initial_gate_open = 0
    for label in spec.initial_gates_open:
        initial_gate_open |= _bit(gate_name_to_idx[label])

    source_cells = 0
    valve_cells = 0
    sensor_cells = 0
    drain_cells = 0
    for m in sources:
        source_cells |= m
    for m in valves:
        valve_cells |= m
    for m in sensors:
        sensor_cells |= m
    for m in drains:
        drain_cells |= m

    passable_player_base = floors_for_crates | conduit | exits_mask | protected | sensor_cells | drain_cells
    flood_passable_base = passable_player_base

    neighbors: list[int] = [0] * (width * height)
    for y in range(height):
        for x in range(width):
            idx = _to_idx(x, y, width)
            if y == 0:
                neighbors[idx] = 0
                continue
            mask = 0
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= width or ny < 1 or ny >= height:
                    continue
                mask |= _bit(_to_idx(nx, ny, width))
            neighbors[idx] = mask

    drain_adjacent: list[int] = []
    for dmask in drains:
        adj = 0
        for idx in _iter_bits(dmask):
            adj |= neighbors[idx]
        adj &= ~dmask
        drain_adjacent.append(adj)

    exit_cells = tuple(sorted(_iter_bits(exits_mask)))

    return LevelModel(
        name=spec.name,
        width=width,
        height=height,
        time_max=spec.time_max,
        player_start=player_start,
        walls=walls,
        conduit=conduit,
        protected=protected,
        exits=exits_mask,
        floors_for_crates=floors_for_crates,
        passable_player_base=passable_player_base,
        flood_passable_base=flood_passable_base,
        sources=sources,
        gates=gates,
        valves=valves,
        drains=drains,
        sensors=sensors,
        crates=crates,
        valve_source_toggle_masks=tuple(valve_source_masks),
        valve_gate_toggle_masks=tuple(valve_gate_masks),
        sensor_gate_open_masks=tuple(sensor_gate_open_masks),
        initial_source_on=initial_source_on,
        initial_gate_open=initial_gate_open,
        neighbors=tuple(neighbors),
        drain_adjacent=tuple(drain_adjacent),
        exit_cells=exit_cells,
    )


def build_level_models() -> tuple[LevelModel, ...]:
    return tuple(_build_model(spec) for spec in LEVEL_SPECS)


LEVEL_MODELS = build_level_models()


def initial_state(model: LevelModel) -> GameState:
    return GameState(
        player=model.player_start,
        crates=model.crates,
        source_on=model.initial_source_on,
        gate_open=model.initial_gate_open,
        sensor_wet=0,
        pending_gate_open=0,
        foam=0,
        settled=0,
        time_left=model.time_max,
        tick=0,
    )


def _active_gate_cells(model: LevelModel, gate_open_bits: int) -> int:
    cells = 0
    for i, gmask in enumerate(model.gates):
        if gate_open_bits & _bit(i):
            cells |= gmask
    return cells


def _source_cells_on(model: LevelModel, source_on_bits: int) -> int:
    cells = 0
    for i, smask in enumerate(model.sources):
        if source_on_bits & _bit(i):
            cells |= smask
    return cells


def _find_adjacent_valve(model: LevelModel, player_idx: int) -> int | None:
    for i, vmask in enumerate(model.valves):
        for vidx in _iter_bits(vmask):
            if model.neighbors[player_idx] & _bit(vidx):
                return i
    return None


def _find_crate_index(crates: tuple[int, ...], idx: int) -> int | None:
    mask = _bit(idx)
    for i, cmask in enumerate(crates):
        if cmask & mask:
            return i
    return None


def _status_for_state(model: LevelModel, player: int, wet_now: int, time_left: int) -> str:
    if wet_now & _bit(player):
        return "LOSE"
    if wet_now & model.exits:
        return "LOSE"
    if wet_now & model.protected:
        return "LOSE"
    if time_left <= 0:
        return "LOSE"
    if model.exits & _bit(player):
        return "WIN"
    return "RUNNING"


def _advance_state(model: LevelModel, state: GameState, action_id: int) -> tuple[GameState, StepInfo]:
    gate_flash = 0
    drain_flash = 0

    source_on = int(state.source_on)
    gate_open = int(state.gate_open)
    sensor_wet = int(state.sensor_wet)
    pending_gate_open = int(state.pending_gate_open)
    foam = int(state.foam)
    settled = int(state.settled)
    time_left = int(state.time_left)
    tick = int(state.tick)
    player = int(state.player)
    crates = list(state.crates)

    if pending_gate_open:
        before = gate_open
        gate_open |= pending_gate_open
        gate_flash |= before ^ gate_open
        pending_gate_open = 0

    move = ACTION_DELTAS.get(int(action_id))
    if move is not None:
        dx, dy = move
        px, py = _to_xy(player, model.width)
        nx = px + dx
        ny = py + dy
        if 0 <= nx < model.width and 1 <= ny < model.height:
            nidx = _to_idx(nx, ny, model.width)
            wet = foam | settled
            if not (wet & _bit(nidx)):
                crate_idx = _find_crate_index(tuple(crates), nidx)
                closed_gate_cells = 0
                for i, gmask in enumerate(model.gates):
                    if not (gate_open & _bit(i)):
                        closed_gate_cells |= gmask
                static_blockers = model.walls | closed_gate_cells
                source_cells = 0
                valve_cells = 0
                for smask in model.sources:
                    source_cells |= smask
                for vmask in model.valves:
                    valve_cells |= vmask
                static_blockers |= source_cells | valve_cells

                occupied_by_crates = 0
                for cmask in crates:
                    occupied_by_crates |= cmask

                if crate_idx is not None:
                    shifted = _shift_mask(crates[crate_idx], dx, dy, model.width, model.height)
                    if shifted is not None:
                        other_crates = occupied_by_crates & ~crates[crate_idx]
                        blocked = static_blockers | other_crates | wet
                        if (shifted & blocked) == 0 and (shifted & ~model.floors_for_crates) == 0:
                            crates[crate_idx] = shifted
                            player = nidx
                else:
                    if (static_blockers | occupied_by_crates) & _bit(nidx) == 0:
                        passable = model.passable_player_base | _active_gate_cells(model, gate_open)
                        if passable & _bit(nidx):
                            player = nidx

    elif int(action_id) == ACTION_SPACE:
        valve_idx = _find_adjacent_valve(model, player)
        if valve_idx is not None:
            source_toggle = model.valve_source_toggle_masks[valve_idx]
            gate_toggle = model.valve_gate_toggle_masks[valve_idx]
            source_on ^= source_toggle
            before = gate_open
            gate_open ^= gate_toggle
            gate_flash |= before ^ gate_open

    time_left -= 1

    prev_foam = foam
    settled |= prev_foam
    foam = 0

    frontier = prev_foam | _source_cells_on(model, source_on)
    wet_now = settled
    flood_passable = model.flood_passable_base | _active_gate_cells(model, gate_open)

    spread = 0
    for idx in _iter_bits(frontier):
        spread |= model.neighbors[idx]
    new_foam = spread & flood_passable & ~wet_now
    foam |= new_foam
    wet_now |= new_foam

    drain_cells = 0
    for dmask in model.drains:
        drain_cells |= dmask

    removed = wet_now & drain_cells
    if removed:
        foam &= ~removed
        settled &= ~removed
        wet_now &= ~removed

    for i, _dmask in enumerate(model.drains):
        adj = model.drain_adjacent[i] & wet_now
        if not adj:
            continue
        removed_here = 0
        foam_cells = sorted(_iter_bits(adj & foam))
        for idx in foam_cells[:2]:
            removed_here |= _bit(idx)
        if removed_here.bit_count() < 2:
            settled_cells = sorted(_iter_bits(adj & settled & ~removed_here))
            for idx in settled_cells[: 2 - removed_here.bit_count()]:
                removed_here |= _bit(idx)
        if removed_here:
            drain_flash |= _bit(i)
            foam &= ~removed_here
            settled &= ~removed_here
            wet_now &= ~removed_here

    pending_next = 0
    new_sensor_wet = sensor_wet
    for i, smask in enumerate(model.sensors):
        now_wet = bool(wet_now & smask)
        was_wet = bool(sensor_wet & _bit(i))
        if now_wet:
            new_sensor_wet |= _bit(i)
        else:
            new_sensor_wet &= ~_bit(i)
        if now_wet and not was_wet:
            pending_next |= model.sensor_gate_open_masks[i]
    sensor_wet = new_sensor_wet

    status = _status_for_state(model, player, wet_now, time_left)

    next_state = GameState(
        player=player,
        crates=tuple(crates),
        source_on=source_on,
        gate_open=gate_open,
        sensor_wet=sensor_wet,
        pending_gate_open=pending_next,
        foam=foam,
        settled=settled,
        time_left=time_left,
        tick=tick + 1,
    )
    return next_state, StepInfo(gate_flash=gate_flash, drain_flash=drain_flash, status=status)


class FloodFillWater(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [
            Level(
                name=model.name,
                grid_size=(model.width, model.height),
                sprites=[
                    Sprite(
                        pixels=np.full((model.height, model.width), COLOR_FLOOR, dtype=np.int8),
                        name="canvas",
                        x=0,
                        y=0,
                        layer=0,
                        tags=["canvas"],
                        collidable=False,
                    )
                ],
                data={"model_index": i},
            )
            for i, model in enumerate(LEVEL_MODELS)
        ]
        camera = Camera(
            width=max(m.width for m in LEVEL_MODELS), height=max(m.height for m in LEVEL_MODELS), background=COLOR_FLOOR
        )
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._model: LevelModel | None = None
        self._sim_state: GameState | None = None
        self._canvas: Sprite | None = None
        self._gate_flash = 0
        self._drain_flash = 0

    def on_set_level(self, level: Level) -> None:
        idx = int(level.get_data("model_index") or 0)
        self._model = LEVEL_MODELS[idx]
        self._sim_state = initial_state(self._model)
        self._canvas = level.get_sprites_by_name("canvas")[0]
        self._gate_flash = 0
        self._drain_flash = 0
        self._render()

    def _render(self) -> None:
        if self._model is None or self._sim_state is None or self._canvas is None:
            return
        model = self._model
        state = self._sim_state
        board = np.full((model.height, model.width), COLOR_FLOOR, dtype=np.int8)

        fill = round((max(0, state.time_left) / max(1, model.time_max)) * model.width)
        fill = max(0, min(model.width, fill))
        board[0, :fill] = COLOR_CONDUIT
        board[0, fill:] = COLOR_FLOOR

        for idx in _iter_bits(model.walls):
            x, y = _to_xy(idx, model.width)
            board[y, x] = COLOR_WALL
        for idx in _iter_bits(model.conduit):
            x, y = _to_xy(idx, model.width)
            board[y, x] = COLOR_CONDUIT
        for idx in _iter_bits(model.protected):
            x, y = _to_xy(idx, model.width)
            board[y, x] = COLOR_PROTECTED
        for idx in _iter_bits(model.exits):
            x, y = _to_xy(idx, model.width)
            board[y, x] = COLOR_EXIT

        for i, smask in enumerate(model.sources):
            on = bool(state.source_on & _bit(i))
            for idx in _iter_bits(smask):
                x, y = _to_xy(idx, model.width)
                if on:
                    board[y, x] = _checker(x, y, COLOR_SOURCE_ON, COLOR_SOURCE_OFF, state.tick)
                else:
                    board[y, x] = COLOR_SOURCE_OFF

        for i, gmask in enumerate(model.gates):
            open_now = bool(state.gate_open & _bit(i))
            flash = bool(self._gate_flash & _bit(i))
            for idx in _iter_bits(gmask):
                x, y = _to_xy(idx, model.width)
                if flash:
                    board[y, x] = _checker(x, y, COLOR_GATE_OPEN, COLOR_GATE_CLOSED, state.tick)
                else:
                    board[y, x] = COLOR_GATE_OPEN if open_now else COLOR_GATE_CLOSED

        for i, vmask in enumerate(model.valves):
            linked_on = False
            if model.valve_source_toggle_masks[i]:
                linked_on |= bool(state.source_on & model.valve_source_toggle_masks[i])
            if model.valve_gate_toggle_masks[i]:
                linked_on |= bool(state.gate_open & model.valve_gate_toggle_masks[i])
            cells = _cells_from_mask(vmask, model.width)
            for idx in cells:
                x, y = _to_xy(idx, model.width)
                board[y, x] = COLOR_VALVE
            if cells:
                corner = max(cells)
                cx, cy = _to_xy(corner, model.width)
                board[cy, cx] = COLOR_GATE_OPEN if linked_on else COLOR_GATE_CLOSED

        for i, dmask in enumerate(model.drains):
            flash = bool(self._drain_flash & _bit(i))
            for idx in _iter_bits(dmask):
                x, y = _to_xy(idx, model.width)
                board[y, x] = _checker(x, y, COLOR_DRAIN, COLOR_FOAM, state.tick) if flash else COLOR_DRAIN

        for i, smask in enumerate(model.sensors):
            wet = bool(state.sensor_wet & _bit(i))
            for idx in _iter_bits(smask):
                x, y = _to_xy(idx, model.width)
                if wet and ((state.tick % 2) == 1):
                    board[y, x] = _checker(x, y, COLOR_SENSOR, COLOR_FOAM, state.tick)
                else:
                    board[y, x] = COLOR_SENSOR

        for cmask in state.crates:
            for idx in _iter_bits(cmask):
                x, y = _to_xy(idx, model.width)
                board[y, x] = COLOR_CRATE

        for idx in _iter_bits(state.settled):
            x, y = _to_xy(idx, model.width)
            board[y, x] = COLOR_WATER
        for idx in _iter_bits(state.foam):
            x, y = _to_xy(idx, model.width)
            board[y, x] = COLOR_FOAM

        px, py = _to_xy(state.player, model.width)
        board[py, px] = COLOR_PLAYER

        self._canvas.pixels = board

    def step(self) -> None:
        if self._model is None or self._sim_state is None:
            self.complete_action()
            return

        self._gate_flash = 0
        self._drain_flash = 0

        raw_action = getattr(self.action, "id", 0)
        action_id = int(getattr(raw_action, "value", raw_action) or 0)
        if action_id not in (1, 2, 3, 4, 5):
            self._render()
            self.complete_action()
            return

        next_state, info = _advance_state(self._model, self._sim_state, action_id)
        self._sim_state = next_state
        self._gate_flash = info.gate_flash
        self._drain_flash = info.drain_flash

        if info.status == "WIN":
            self.next_level()
        elif info.status == "LOSE":
            self.lose()

        self._render()
        self.complete_action()


__all__ = ["GAME_ID", "LEVEL_MODELS", "FloodFillWater", "GameState", "LevelModel", "_advance_state", "initial_state"]
