from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "one_screen_precision_jumps-0001"
WIDTH = 32
HEIGHT = 18
TIMEBAR_Y = 0

# Physics tuning chosen to match the authored layouts.
JUMP_VELOCITY = -3
GRAVITY = 1
MAX_FALL_SPEED = 3

# Color palette (0..15) from the spec.
COLOR_EMPTY = 0
COLOR_SOLID = 1
COLOR_ONE_WAY = 2
COLOR_SPIKE_EXTENDED = 3
COLOR_PLAYER_HEAD = 4
COLOR_PLAYER_BODY = 5
COLOR_EXIT_BASE = 6
COLOR_EXIT_PULSE = 7
COLOR_CHECKPOINT_INACTIVE = 8
COLOR_CHECKPOINT_ACTIVE = 9
COLOR_TIME_SAFE = 10
COLOR_TIME_WARNING = 11
COLOR_TIME_DANGER = 12
COLOR_MOVING_PLATFORM = 13
COLOR_CRUMBLE_INTACT = 14
COLOR_SPIKE_RETRACTED = 15

MOVE_LEFT = int(GameAction.ACTION3.value)
MOVE_RIGHT = int(GameAction.ACTION4.value)
JUMP = int(GameAction.ACTION5.value)
CLICK = int(GameAction.ACTION6.value)


@dataclass(frozen=True)
class ToggleSpec:
    positions: tuple[tuple[int, int], ...]
    period: int
    extended_steps: int


@dataclass(frozen=True)
class PlatformSpec:
    row: int
    min_x: int
    max_x: int
    start_x: int
    length: int = 3


@dataclass(frozen=True)
class CrumbleSpec:
    positions: tuple[tuple[int, int], ...]
    crack_delay: int
    fall_delay: int


@dataclass(frozen=True)
class CheckpointSpec:
    cells: tuple[tuple[int, int], ...]
    top_y: int


@dataclass(frozen=True)
class LevelModel:
    name: str
    width: int
    height: int
    time_limit: int
    solids: tuple[tuple[int, int], ...]
    one_way: tuple[tuple[int, int], ...]
    static_spikes: tuple[tuple[int, int], ...]
    exit_cells: tuple[tuple[int, int], ...]
    checkpoints: tuple[CheckpointSpec, ...]
    checkpoint_cells: tuple[tuple[int, int], ...]
    checkpoint_respawns: tuple[tuple[int, int], ...]  # head positions
    start_head: tuple[int, int]
    toggle: ToggleSpec | None
    platform: PlatformSpec | None
    crumble: CrumbleSpec | None
    crumble_index: tuple[tuple[int, int], ...]


# Search/runtime state tuple.
# (px, py_head, vy, time_left, checkpoint_mask, respawn_checkpoint_idx,
#  toggle_tick, platform_x, platform_dir, pulse_tick, crumble_ages...)
SearchState = tuple[int, ...]


LEVEL_LAYOUTS: tuple[dict, ...] = (
    {
        "name": "Level 1",
        "time_limit": 240,
        "rows": (
            "|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#...........###................#",
            "#..........................XX..#",
            "#..a.......................XX..#",
            "#..A.......................XX..#",
            "######..####..####..####..######",
        ),
    },
    {
        "name": "Level 2",
        "time_limit": 260,
        "rows": (
            "|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#...........###................#",
            "#..................###.........#",
            "#..............................#",
            "#..........................XX..#",
            "#..a.......................XX..#",
            "#..A......^.......^^^......XX..#",
            "######..################..######",
        ),
    },
    {
        "name": "Level 3",
        "time_limit": 300,
        "rows": (
            "|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..........................XX..#",
            "#..........................XX..#",
            "#..........................XX..#",
            "#..............................#",
            "#....................###########",
            "#.....................----.....#",
            "#..............................#",
            "#...................----.......#",
            "#..............................#",
            "#.................----.........#",
            "#..a...........CC..............#",
            "#..A.......^^..CC...^..........#",
            "########..######..######..######",
        ),
    },
    {
        "name": "Level 4",
        "time_limit": 320,
        "rows": (
            "|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..........................XX..#",
            "#..........................XX..#",
            "#..a..CC.....MMM...........XX..#",
            "#..A..CC.....^^^^^^^^^^.^^.XX..#",
            "############..........##########",
        ),
        "platform": {"row": 15, "min_x": 13, "max_x": 20, "start_x": 13, "length": 3},
    },
    {
        "name": "Level 5",
        "time_limit": 360,
        "rows": (
            "|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#......................####XX..#",
            "#..a..CC...................XX..#",
            "#..A..CC.++++++++++++###^^^XX..#",
            "########^^^^^^^^^^^^############",
        ),
        "toggle": {"positions": tuple((x, 15) for x in range(24, 27)), "period": 16, "extended_steps": 8},
        "crumble": {"crack_delay": 6, "fall_delay": 12},
    },
    {
        "name": "Level 6",
        "time_limit": 420,
        "rows": (
            "|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..........................XX..#",
            "#..........................XX..#",
            "#..........................XX..#",
            "#.................#######++#####",
            "#..............................#",
            "#..................^^^^^.......#",
            "#.................#######......#",
            "#.................----.........#",
            "#..........MMM.................#",
            "#..a..CC..............CC.......#",
            "#..A..CC..............CC..^^...#",
            "##########^^^^^^^^^^############",
        ),
        "platform": {"row": 14, "min_x": 11, "max_x": 18, "start_x": 11, "length": 3},
        "toggle": {"positions": tuple((x, 11) for x in range(18, 23)), "period": 20, "extended_steps": 10},
        "crumble": {"crack_delay": 5, "fall_delay": 10},
    },
)


def _inside_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _player_cells(px: int, py_head: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((px, py_head), (px, py_head + 1))


def _pulse_on(tick: int) -> bool:
    return (tick % 8) < 4


def _toggle_extended(spec: ToggleSpec | None, tick: int) -> bool:
    if spec is None or spec.period <= 0:
        return False
    return (tick % spec.period) < spec.extended_steps


def _parse_rows(raw_rows: Iterable[str]) -> tuple[str, ...]:
    rows = tuple(str(row).rstrip() for row in raw_rows)
    if len(rows) != HEIGHT:
        raise ValueError(f"Expected {HEIGHT} rows, found {len(rows)}")
    for row in rows:
        if len(row) != WIDTH:
            raise ValueError(f"Each row must be width {WIDTH}; got {len(row)}")
    return rows


def _collect_checkpoints(rows: tuple[str, ...]) -> tuple[CheckpointSpec, ...]:
    cells = {(x, y) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch == "C"}
    checkpoints: list[CheckpointSpec] = []
    seen: set[tuple[int, int]] = set()

    for x, y in sorted(cells, key=lambda p: (p[1], p[0])):
        if (x, y) in seen:
            continue
        block = {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}
        if not block.issubset(cells):
            continue
        seen.update(block)
        checkpoints.append(CheckpointSpec(cells=tuple(sorted(block)), top_y=y))

    return tuple(checkpoints)


def _candidate_respawn_head_positions(_model_dict: dict, checkpoint: CheckpointSpec) -> list[tuple[int, int]]:
    xs = sorted({x for x, _ in checkpoint.cells})
    top_y = checkpoint.top_y
    candidates = [(x, top_y - 2) for x in xs]
    left = min(xs) - 1
    right = max(xs) + 1
    candidates.append((left, top_y - 2))
    candidates.append((right, top_y - 2))
    return candidates


def _build_level_model(spec: dict) -> LevelModel:
    rows = _parse_rows(spec["rows"])

    solids: set[tuple[int, int]] = set()
    one_way: set[tuple[int, int]] = set()
    static_spikes: set[tuple[int, int]] = set()
    exit_cells: set[tuple[int, int]] = set()
    crumble_cells: set[tuple[int, int]] = set()
    start_head: tuple[int, int] | None = None

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if y == TIMEBAR_Y:
                continue
            if ch == "#":
                solids.add((x, y))
            elif ch == "-":
                one_way.add((x, y))
            elif ch == "^":
                static_spikes.add((x, y))
            elif ch == "X":
                exit_cells.add((x, y))
            elif ch == "+":
                crumble_cells.add((x, y))
            elif ch == "a":
                start_head = (x, y)

    if start_head is None:
        raise ValueError(f"{spec['name']}: missing player head `a`")

    checkpoints = _collect_checkpoints(rows)
    checkpoint_cells = tuple(sorted(cell for cp in checkpoints for cell in cp.cells))

    toggle_raw = spec.get("toggle")
    toggle_spec = None
    toggle_positions: set[tuple[int, int]] = set()
    if toggle_raw:
        toggle_positions = {tuple(p) for p in toggle_raw["positions"]}
        static_spikes -= toggle_positions
        toggle_spec = ToggleSpec(
            positions=tuple(sorted(toggle_positions)),
            period=int(toggle_raw["period"]),
            extended_steps=int(toggle_raw["extended_steps"]),
        )

    platform_raw = spec.get("platform")
    platform_spec = None
    if platform_raw:
        platform_spec = PlatformSpec(
            row=int(platform_raw["row"]),
            min_x=int(platform_raw["min_x"]),
            max_x=int(platform_raw["max_x"]),
            start_x=int(platform_raw["start_x"]),
            length=int(platform_raw.get("length", 3)),
        )
        for x in range(platform_spec.start_x, platform_spec.start_x + platform_spec.length):
            solids.discard((x, platform_spec.row))

    crumble_raw = spec.get("crumble")
    crumble_spec = None
    if crumble_cells and crumble_raw:
        crumble_spec = CrumbleSpec(
            positions=tuple(sorted(crumble_cells)),
            crack_delay=int(crumble_raw["crack_delay"]),
            fall_delay=int(crumble_raw["fall_delay"]),
        )
    elif crumble_cells:
        raise ValueError(f"{spec['name']}: crumble tiles present without crumble config")

    model_dict = {
        "width": WIDTH,
        "height": HEIGHT,
        "solids": tuple(sorted(solids)),
        "one_way": tuple(sorted(one_way)),
        "static_spikes": tuple(sorted(static_spikes)),
        "toggle_positions": tuple(sorted(toggle_positions)),
        "platform": platform_spec,
    }

    checkpoint_respawns: list[tuple[int, int]] = []
    for cp in checkpoints:
        respawn = start_head
        for cand_x, cand_head_y in _candidate_respawn_head_positions(model_dict, cp):
            head = (cand_x, cand_head_y)
            body = (cand_x, cand_head_y + 1)
            support = (cand_x, cand_head_y + 2)
            if not _inside_bounds(head[0], head[1], WIDTH, HEIGHT):
                continue
            if not _inside_bounds(body[0], body[1], WIDTH, HEIGHT):
                continue
            if head in solids or body in solids:
                continue
            if head in static_spikes or body in static_spikes:
                continue
            if support in solids or support in one_way or support in set(cp.cells):
                respawn = (cand_x, cand_head_y)
                break
        checkpoint_respawns.append(respawn)

    crumble_index = tuple(sorted(crumble_cells))

    return LevelModel(
        name=str(spec["name"]),
        width=WIDTH,
        height=HEIGHT,
        time_limit=int(spec["time_limit"]),
        solids=tuple(sorted(solids)),
        one_way=tuple(sorted(one_way)),
        static_spikes=tuple(sorted(static_spikes)),
        exit_cells=tuple(sorted(exit_cells)),
        checkpoints=checkpoints,
        checkpoint_cells=checkpoint_cells,
        checkpoint_respawns=tuple(checkpoint_respawns),
        start_head=start_head,
        toggle=toggle_spec,
        platform=platform_spec,
        crumble=crumble_spec,
        crumble_index=crumble_index,
    )


def _serialize_model(model: LevelModel) -> dict:
    out = {
        "name": model.name,
        "width": int(model.width),
        "height": int(model.height),
        "time_limit": int(model.time_limit),
        "solids": [list(v) for v in model.solids],
        "one_way": [list(v) for v in model.one_way],
        "static_spikes": [list(v) for v in model.static_spikes],
        "exit_cells": [list(v) for v in model.exit_cells],
        "checkpoint_cells": [list(v) for v in model.checkpoint_cells],
        "checkpoints": [
            {"cells": [list(v) for v in checkpoint.cells], "top_y": int(checkpoint.top_y)}
            for checkpoint in model.checkpoints
        ],
        "checkpoint_respawns": [list(v) for v in model.checkpoint_respawns],
        "start_head": list(model.start_head),
        "crumble_index": [list(v) for v in model.crumble_index],
    }
    if model.toggle is not None:
        out["toggle"] = {
            "positions": [list(v) for v in model.toggle.positions],
            "period": int(model.toggle.period),
            "extended_steps": int(model.toggle.extended_steps),
        }
    if model.platform is not None:
        out["platform"] = {
            "row": int(model.platform.row),
            "min_x": int(model.platform.min_x),
            "max_x": int(model.platform.max_x),
            "start_x": int(model.platform.start_x),
            "length": int(model.platform.length),
        }
    if model.crumble is not None:
        out["crumble"] = {
            "positions": [list(v) for v in model.crumble.positions],
            "crack_delay": int(model.crumble.crack_delay),
            "fall_delay": int(model.crumble.fall_delay),
        }
    return out


def _deserialize_model(raw: dict) -> LevelModel:
    checkpoints = tuple(
        CheckpointSpec(
            cells=tuple((int(v[0]), int(v[1])) for v in item.get("cells") or []), top_y=int(item.get("top_y") or 0)
        )
        for item in (raw.get("checkpoints") or [])
    )

    toggle_raw = raw.get("toggle")
    toggle = None
    if toggle_raw:
        toggle = ToggleSpec(
            positions=tuple((int(v[0]), int(v[1])) for v in (toggle_raw.get("positions") or [])),
            period=int(toggle_raw.get("period") or 0),
            extended_steps=int(toggle_raw.get("extended_steps") or 0),
        )

    platform_raw = raw.get("platform")
    platform = None
    if platform_raw:
        platform = PlatformSpec(
            row=int(platform_raw.get("row") or 0),
            min_x=int(platform_raw.get("min_x") or 0),
            max_x=int(platform_raw.get("max_x") or 0),
            start_x=int(platform_raw.get("start_x") or 0),
            length=int(platform_raw.get("length") or 3),
        )

    crumble_raw = raw.get("crumble")
    crumble = None
    if crumble_raw:
        crumble = CrumbleSpec(
            positions=tuple((int(v[0]), int(v[1])) for v in (crumble_raw.get("positions") or [])),
            crack_delay=int(crumble_raw.get("crack_delay") or 0),
            fall_delay=int(crumble_raw.get("fall_delay") or 0),
        )

    return LevelModel(
        name=str(raw.get("name") or "Level"),
        width=int(raw.get("width") or WIDTH),
        height=int(raw.get("height") or HEIGHT),
        time_limit=int(raw.get("time_limit") or 240),
        solids=tuple((int(v[0]), int(v[1])) for v in (raw.get("solids") or [])),
        one_way=tuple((int(v[0]), int(v[1])) for v in (raw.get("one_way") or [])),
        static_spikes=tuple((int(v[0]), int(v[1])) for v in (raw.get("static_spikes") or [])),
        exit_cells=tuple((int(v[0]), int(v[1])) for v in (raw.get("exit_cells") or [])),
        checkpoints=checkpoints,
        checkpoint_cells=tuple((int(v[0]), int(v[1])) for v in (raw.get("checkpoint_cells") or [])),
        checkpoint_respawns=tuple((int(v[0]), int(v[1])) for v in (raw.get("checkpoint_respawns") or [])),
        start_head=(int(raw["start_head"][0]), int(raw["start_head"][1])),
        toggle=toggle,
        platform=platform,
        crumble=crumble,
        crumble_index=tuple((int(v[0]), int(v[1])) for v in (raw.get("crumble_index") or [])),
    )


def _initial_crumble_ages(model: LevelModel) -> tuple[int, ...]:
    return tuple(-1 for _ in model.crumble_index)


def initial_search_state_from_model(model: LevelModel | dict) -> SearchState:
    parsed = _deserialize_model(model) if isinstance(model, dict) else model
    platform_x = parsed.platform.start_x if parsed.platform is not None else 0
    return (
        int(parsed.start_head[0]),
        int(parsed.start_head[1]),
        0,
        int(parsed.time_limit),
        0,
        -1,
        0,
        int(platform_x),
        1,
        0,
        *_initial_crumble_ages(parsed),
    )


def _state_prefix_len() -> int:
    return 10


def _split_state(model: LevelModel, state: SearchState):
    prefix_len = _state_prefix_len()
    if len(state) < prefix_len:
        raise ValueError("Invalid state length")
    (px, py_head, vy, time_left, checkpoint_mask, respawn_cp, toggle_tick, platform_x, platform_dir, pulse_tick) = (
        int(v) for v in state[:prefix_len]
    )
    crumble_ages = tuple(int(v) for v in state[prefix_len:])
    if len(crumble_ages) != len(model.crumble_index):
        raise ValueError("Crumble state length mismatch")
    return (
        px,
        py_head,
        vy,
        time_left,
        checkpoint_mask,
        respawn_cp,
        toggle_tick,
        platform_x,
        platform_dir,
        pulse_tick,
        crumble_ages,
    )


def _compose_state(
    px: int,
    py_head: int,
    vy: int,
    time_left: int,
    checkpoint_mask: int,
    respawn_cp: int,
    toggle_tick: int,
    platform_x: int,
    platform_dir: int,
    pulse_tick: int,
    crumble_ages: tuple[int, ...],
) -> SearchState:
    return (
        int(px),
        int(py_head),
        int(vy),
        int(time_left),
        int(checkpoint_mask),
        int(respawn_cp),
        int(toggle_tick),
        int(platform_x),
        int(platform_dir),
        int(pulse_tick),
        *(int(v) for v in crumble_ages),
    )


def _platform_cells(model: LevelModel, platform_x: int) -> set[tuple[int, int]]:
    if model.platform is None:
        return set()
    return {(platform_x + i, model.platform.row) for i in range(model.platform.length)}


def _active_toggle_spikes(model: LevelModel, toggle_tick: int) -> set[tuple[int, int]]:
    if model.toggle is None:
        return set()
    return set(model.toggle.positions) if _toggle_extended(model.toggle, toggle_tick) else set()


def _crumble_cell_state(model: LevelModel, crumble_ages: tuple[int, ...]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    if model.crumble is None:
        return out
    for idx, cell in enumerate(model.crumble_index):
        age = crumble_ages[idx]
        if age < 0:
            out[cell] = COLOR_CRUMBLE_INTACT
            continue
        if age >= model.crumble.fall_delay:
            continue
        if age >= model.crumble.crack_delay:
            out[cell] = COLOR_TIME_WARNING
        else:
            out[cell] = COLOR_CRUMBLE_INTACT
    return out


def _solid_side_cells(model: LevelModel, platform_x: int, crumble_ages: tuple[int, ...]) -> set[tuple[int, int]]:
    solids = set(model.solids)
    solids.update(model.checkpoint_cells)
    solids.update(_platform_cells(model, platform_x))
    if model.crumble is not None:
        for idx, cell in enumerate(model.crumble_index):
            age = crumble_ages[idx]
            if age < 0 or age < model.crumble.fall_delay:
                solids.add(cell)
    return solids


def _solid_down_cells(
    model: LevelModel, platform_x: int, crumble_ages: tuple[int, ...]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    solids_side = _solid_side_cells(model, platform_x, crumble_ages)
    return solids_side, set(model.one_way)


def _collides_side(model: LevelModel, x: int, y: int, platform_x: int, crumble_ages: tuple[int, ...]) -> bool:
    return (x, y) in _solid_side_cells(model, platform_x, crumble_ages)


def _is_grounded(model: LevelModel, px: int, py_head: int, platform_x: int, crumble_ages: tuple[int, ...]) -> bool:
    foot = (px, py_head + 2)
    solids_side, one_way = _solid_down_cells(model, platform_x, crumble_ages)
    return foot in solids_side or foot in one_way


def _out_of_bounds_fail(model: LevelModel, px: int, py_head: int) -> bool:
    for x, y in _player_cells(px, py_head):
        if x < 0 or x >= model.width:
            return True
        if y >= model.height:
            return True
    return False


def _respawn_head(model: LevelModel, respawn_checkpoint_idx: int) -> tuple[int, int]:
    if respawn_checkpoint_idx < 0 or respawn_checkpoint_idx >= len(model.checkpoint_respawns):
        return model.start_head
    return model.checkpoint_respawns[respawn_checkpoint_idx]


def _is_crushed_against_solid(
    model: LevelModel, px: int, py_head: int, platform_x: int, crumble_ages: tuple[int, ...]
) -> bool:
    platform_cells = _platform_cells(model, platform_x)
    side_solids = _solid_side_cells(model, platform_x, crumble_ages)
    for x, y in _player_cells(px, py_head):
        if (x, y) in platform_cells:
            return True
        if (x, y) in side_solids:
            return True
    return False


def apply_action_transition(model: LevelModel | dict, state: SearchState, action_id: int) -> tuple[SearchState, bool]:
    parsed = _deserialize_model(model) if isinstance(model, dict) else model

    (
        px,
        py_head,
        vy,
        time_left,
        checkpoint_mask,
        respawn_cp,
        toggle_tick,
        platform_x,
        platform_dir,
        pulse_tick,
        crumble_ages,
    ) = _split_state(parsed, state)

    action = int(action_id)
    respawn_pending = False

    # 1-2: read input, click sets respawn_pending only.
    if action == CLICK:
        respawn_pending = True

    # 3: horizontal intent.
    if action in (MOVE_LEFT, MOVE_RIGHT):
        dx = -1 if action == MOVE_LEFT else 1
        nx = px + dx
        blocked = False
        for cx, cy in _player_cells(nx, py_head):
            if _collides_side(parsed, cx, cy, platform_x, crumble_ages):
                blocked = True
                break
        if not blocked:
            px = nx

    # 4: jump only if grounded.
    if action == JUMP and _is_grounded(parsed, px, py_head, platform_x, crumble_ages):
        vy = JUMP_VELOCITY

    # 5: gravity + vertical movement with cellwise collision.
    if vy != 0:
        step = -1 if vy < 0 else 1
        for _ in range(abs(vy)):
            nhead = py_head + step
            blocked = False
            for cx, cy in _player_cells(px, nhead):
                if step > 0:
                    solids_side, one_way = _solid_down_cells(parsed, platform_x, crumble_ages)
                    if (cx, cy) in solids_side or (cx, cy) in one_way:
                        blocked = True
                        break
                else:
                    if _collides_side(parsed, cx, cy, platform_x, crumble_ages):
                        blocked = True
                        break
            if blocked:
                vy = 0
                break
            py_head = nhead

    vy = min(vy + GRAVITY, MAX_FALL_SPEED)

    # 6: interactions in specified order.
    active_spikes = set(parsed.static_spikes)
    active_spikes.update(_active_toggle_spikes(parsed, toggle_tick))

    if _out_of_bounds_fail(parsed, px, py_head):
        respawn_pending = True

    if not respawn_pending:
        for cx, cy in _player_cells(px, py_head):
            if (cx, cy) in active_spikes:
                respawn_pending = True
                break

    if not respawn_pending and _is_crushed_against_solid(parsed, px, py_head, platform_x, crumble_ages):
        respawn_pending = True

    if not respawn_pending:
        player_cells = set(_player_cells(px, py_head))
        for idx, checkpoint in enumerate(parsed.checkpoints):
            if player_cells.intersection(set(checkpoint.cells)):
                checkpoint_mask |= 1 << idx
                respawn_cp = idx

    if not respawn_pending:
        player_cells = set(_player_cells(px, py_head))
        if player_cells.intersection(set(parsed.exit_cells)):
            won_state = _compose_state(
                px,
                py_head,
                vy,
                time_left,
                checkpoint_mask,
                respawn_cp,
                toggle_tick,
                platform_x,
                platform_dir,
                pulse_tick,
                crumble_ages,
            )
            return won_state, True

    # 7: world state advance.
    time_left -= 1
    if time_left <= 0:
        respawn_pending = True

    if parsed.toggle is not None and parsed.toggle.period > 0:
        toggle_tick = (toggle_tick + 1) % parsed.toggle.period

    # Move platform and carry rider.
    if parsed.platform is not None:
        old_cells = _platform_cells(parsed, platform_x)

        next_platform_x = platform_x + platform_dir
        if next_platform_x < parsed.platform.min_x or next_platform_x > parsed.platform.max_x:
            platform_dir *= -1
            next_platform_x = platform_x + platform_dir
        delta_x = next_platform_x - platform_x

        rider = (px, py_head + 2) in old_cells

        platform_x = next_platform_x

        if rider and delta_x != 0:
            nx = px + delta_x
            blocked = False
            for cx, cy in _player_cells(nx, py_head):
                if _collides_side(parsed, cx, cy, platform_x, crumble_ages):
                    blocked = True
                    break
            if blocked:
                respawn_pending = True
            else:
                px = nx

        if _is_crushed_against_solid(parsed, px, py_head, platform_x, crumble_ages):
            respawn_pending = True

    # Crumble update.
    if parsed.crumble is not None and parsed.crumble_index:
        age_list = list(crumble_ages)

        support_cell = (px, py_head + 2)
        support_idx = {cell: i for i, cell in enumerate(parsed.crumble_index)}.get(support_cell)
        if support_idx is not None and age_list[support_idx] < 0:
            age_list[support_idx] = 0

        for idx, age in enumerate(age_list):
            if age >= 0:
                age_list[idx] = age + 1
        crumble_ages = tuple(age_list)

    pulse_tick += 1

    # 8: respawn and dynamic reset.
    if respawn_pending:
        respawn_x, respawn_y = _respawn_head(parsed, respawn_cp)
        px = respawn_x
        py_head = respawn_y
        vy = 0
        time_left = parsed.time_limit
        toggle_tick = 0
        pulse_tick = 0
        if parsed.platform is not None:
            platform_x = parsed.platform.start_x
            platform_dir = 1
        crumble_ages = _initial_crumble_ages(parsed)

    next_state = _compose_state(
        px,
        py_head,
        vy,
        time_left,
        checkpoint_mask,
        respawn_cp,
        toggle_tick,
        platform_x,
        platform_dir,
        pulse_tick,
        crumble_ages,
    )
    return next_state, False


def _draw_state(model: LevelModel, state: SearchState) -> np.ndarray:
    (
        px,
        py_head,
        _vy,
        time_left,
        checkpoint_mask,
        _respawn_cp,
        toggle_tick,
        platform_x,
        _platform_dir,
        pulse_tick,
        crumble_ages,
    ) = _split_state(model, state)

    grid = [[COLOR_EMPTY for _ in range(model.width)] for _ in range(model.height)]

    for x, y in model.solids:
        if _inside_bounds(x, y, model.width, model.height):
            grid[y][x] = COLOR_SOLID

    for x, y in model.one_way:
        if _inside_bounds(x, y, model.width, model.height):
            grid[y][x] = COLOR_ONE_WAY

    for x, y in model.static_spikes:
        if _inside_bounds(x, y, model.width, model.height):
            grid[y][x] = COLOR_SPIKE_EXTENDED

    for x, y in _active_toggle_spikes(model, toggle_tick):
        if _inside_bounds(x, y, model.width, model.height):
            grid[y][x] = COLOR_SPIKE_EXTENDED

    if model.toggle is not None and not _toggle_extended(model.toggle, toggle_tick):
        for x, y in model.toggle.positions:
            if _inside_bounds(x, y, model.width, model.height):
                grid[y][x] = COLOR_SPIKE_RETRACTED

    for cell, color in _crumble_cell_state(model, crumble_ages).items():
        x, y = cell
        if _inside_bounds(x, y, model.width, model.height):
            grid[y][x] = color

    checkpoint_pulse_on = _pulse_on(pulse_tick)
    for idx, checkpoint in enumerate(model.checkpoints):
        active = (checkpoint_mask & (1 << idx)) != 0
        color = (
            COLOR_CHECKPOINT_ACTIVE
            if active and checkpoint_pulse_on
            else (COLOR_CHECKPOINT_ACTIVE if active else COLOR_CHECKPOINT_INACTIVE)
        )
        for x, y in checkpoint.cells:
            if _inside_bounds(x, y, model.width, model.height):
                grid[y][x] = color

    exit_color = COLOR_EXIT_PULSE if _pulse_on(pulse_tick) else COLOR_EXIT_BASE
    for x, y in model.exit_cells:
        if _inside_bounds(x, y, model.width, model.height):
            grid[y][x] = exit_color

    for x, y in _platform_cells(model, platform_x):
        if _inside_bounds(x, y, model.width, model.height):
            grid[y][x] = COLOR_MOVING_PLATFORM

    for x, y in _player_cells(px, py_head):
        if _inside_bounds(x, y, model.width, model.height):
            grid[y][x] = COLOR_PLAYER_HEAD if y == py_head else COLOR_PLAYER_BODY

    # Timebar at row 0.
    fill_cells = max(0, min(model.width - 2, int((time_left * (model.width - 2)) / max(1, model.time_limit))))
    frac = float(time_left) / float(max(1, model.time_limit))
    if frac > 0.5:
        fill_color = COLOR_TIME_SAFE
    elif frac > 0.1:
        fill_color = COLOR_TIME_WARNING
    else:
        fill_color = COLOR_TIME_DANGER

    grid[TIMEBAR_Y][0] = COLOR_SOLID
    grid[TIMEBAR_Y][model.width - 1] = COLOR_SOLID
    for x in range(1, model.width - 1):
        grid[TIMEBAR_Y][x] = fill_color if x <= fill_cells else COLOR_EMPTY

    return np.array(grid, dtype=np.int8)


def _solve_level(model: LevelModel) -> list[int] | None:
    start = initial_search_state_from_model(model)
    queue = deque([start])
    previous: dict[SearchState, SearchState | None] = {start: None}
    previous_action: dict[SearchState, int] = {}

    best_time_by_key: dict[tuple[int, ...], int] = {}

    def dominance_key(state: SearchState) -> tuple[int, ...]:
        (
            px,
            py_head,
            vy,
            _time_left,
            checkpoint_mask,
            respawn_cp,
            toggle_tick,
            platform_x,
            platform_dir,
            _pulse_tick,
            *crumble,
        ) = state
        return (
            int(px),
            int(py_head),
            int(vy),
            int(checkpoint_mask),
            int(respawn_cp),
            int(toggle_tick),
            int(platform_x),
            int(platform_dir),
            *(int(v) for v in crumble),
        )

    while queue:
        state = queue.popleft()
        key = dominance_key(state)
        time_left = int(state[3])
        prev_best = best_time_by_key.get(key)
        if prev_best is not None and prev_best >= time_left:
            continue
        best_time_by_key[key] = time_left

        for action_id in (MOVE_LEFT, MOVE_RIGHT, JUMP):
            next_state, won = apply_action_transition(model, state, action_id)
            if won:
                actions: list[int] = [action_id]
                cursor = state
                while previous[cursor] is not None:
                    actions.append(previous_action[cursor])
                    cursor = previous[cursor]  # type: ignore[assignment]
                actions.reverse()
                return actions
            if next_state in previous:
                continue
            previous[next_state] = state
            previous_action[next_state] = action_id
            queue.append(next_state)

    return None


def _build_level(spec: dict) -> Level:
    model = _build_level_model(spec)
    initial_state = initial_search_state_from_model(model)
    pixels = _draw_state(model, initial_state)
    board = Sprite(pixels=pixels, name="board", x=0, y=0, layer=1, tags=["board"], collidable=False)
    return Level(
        name=model.name, grid_size=(model.width, model.height), sprites=[board], data={"model": _serialize_model(model)}
    )


class OneScreenPrecisionJumps(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(spec) for spec in LEVEL_LAYOUTS]
        camera = Camera(width=WIDTH, height=HEIGHT, background=COLOR_EMPTY)
        super().__init__(
            game_id="one_screen_precision_jumps",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[MOVE_LEFT, MOVE_RIGHT, JUMP, CLICK],
            seed=seed,
        )
        self._model: LevelModel | None = None
        self._search_state: SearchState | None = None
        self._board: Sprite | None = None

    def on_set_level(self, level: Level) -> None:
        model = _deserialize_model(level.get_data("model") or {})
        state = initial_search_state_from_model(model)

        boards = level.get_sprites_by_name("board")
        if not boards:
            raise RuntimeError("one_screen_precision_jumps: missing board sprite")

        self._model = model
        self._search_state = state
        self._board = boards[0]
        self._render()

    def _render(self) -> None:
        if self._model is None or self._search_state is None or self._board is None:
            return
        self._board.pixels = _draw_state(self._model, self._search_state)

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

        if self._model is None or self._search_state is None:
            self.complete_action()
            return

        raw_action = self.action.id
        action_id = int(getattr(raw_action, "value", raw_action))
        if action_id not in {MOVE_LEFT, MOVE_RIGHT, JUMP, CLICK}:
            action_id = 0

        next_state, won = apply_action_transition(self._model, self._search_state, action_id)
        self._search_state = next_state

        if won:
            self.next_level()
            self.complete_action()
            return

        self._render()
        self.complete_action()
