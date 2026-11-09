from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "dodger_falling_blocks-0001"
GRID_W = 26
GRID_H = 22
TIMEBAR_STEPS = 48

COLOR_EMPTY = 0
COLOR_WALL = 1
COLOR_PLAYER_A = 2
COLOR_PLAYER_B = 3
COLOR_HAZARD_SMALL = 4
COLOR_HAZARD_SLAB = 5
COLOR_HAZARD_FAST = 6
COLOR_HAZARD_BIG = 7
COLOR_WARNING = 8
COLOR_BARRIER_INTACT = 9
COLOR_BARRIER_CRACKED = 10
COLOR_TRAIL = 11
COLOR_RUBBLE = 12
COLOR_TIMEBAR_REMAINING = 13
COLOR_TIMEBAR_SPENT = 14

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
    int(GameAction.ACTION5.value): (0, 0),
}


def INSIDE_X_TO_GRID(inside_x):
    return int(inside_x) + 1


@dataclass(frozen=True)
class WarningGroup:
    kind: str
    x: int


@dataclass(frozen=True)
class Hazard:
    kind: str
    x: int
    y: int


@dataclass
class RuntimeState:
    level_index: int
    tick: int
    time_left: int
    blink_phase: int
    breathe_phase: int
    player_x: int
    player_y: int
    hazards: list[Hazard]
    warnings: list[WarningGroup]
    barriers: dict[tuple[int, int], int]  # 1=intact, 2=cracked
    rubble: dict[tuple[int, int], int]
    trails: dict[tuple[int, int], int]
    small_seq_idx: int
    slab_seq_idx: int
    fast_seq_idx: int
    big_seq_idx: int
    type_seq_idx: int
    type_seq2_idx: int


LEVEL_LAYOUTS = [
    [
        "##########################",
        "#========================#",
        "#========================#",
        "##########################",
        "#........................#",
        "#........................#",
        "#.....*..................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#...........{}...........#",
        "#...........[]...........#",
        "##########################",
    ],
    [
        "##########################",
        "#========================#",
        "#========================#",
        "##########################",
        "#.....^............^.....#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#.......+........+.......#",
        "#.......+........+.......#",
        "#.......+................#",
        "#.......+........+.......#",
        "#................+.......#",
        "#.......+........+.......#",
        "#.......+........+.......#",
        "#.......+........+.......#",
        "#.......+........+.......#",
        "#.......+........+.......#",
        "#...........{}...........#",
        "#...........[]...........#",
        "##########################",
    ],
    [
        "##########################",
        "#========================#",
        "#========================#",
        "##########################",
        "#..^^^..............^....#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#..................+.....#",
        "#.....+............+.....#",
        "#.....+........+...+.....#",
        "#.....+............+.....#",
        "#.....+..................#",
        "#...........+..+.........#",
        "#...........+..+.........#",
        "#........+..+............#",
        "#........+..+............#",
        "#........+...............#",
        "#........+...............#",
        "#...........{}...........#",
        "#...........[]...........#",
        "##########################",
    ],
    [
        "##########################",
        "#========================#",
        "#========================#",
        "##########################",
        "#....!....^^^............#",
        "#........................#",
        "#....................:...#",
        "#....................!...#",
        "#........................#",
        "#...........+............#",
        "#...........+............#",
        "#......+..........+......#",
        "#......+....+.....+......#",
        "#......+....+.....+......#",
        "#......+....+.....+......#",
        "#......+..+.+.....+......#",
        "#......+..+.+.....+......#",
        "#......+..+.......+......#",
        "#......+..........+......#",
        "#...........{}...........#",
        "#...........[]...........#",
        "##########################",
    ],
    [
        "##########################",
        "#========================#",
        "#========================#",
        "##########################",
        "#.....??...........^^^...#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#....+..........+........#",
        "#....+..........+........#",
        "#....+...+......+........#",
        "#....+...+......+...+....#",
        "#....+...+......+...+....#",
        "#....+...+......+...+....#",
        "#........+..........+....#",
        "#........+..........+....#",
        "#........+..........+....#",
        "#........+..........+....#",
        "#...........{}...........#",
        "#...........[]...........#",
        "##########################",
    ],
    [
        "##########################",
        "#========================#",
        "#========================#",
        "##########################",
        "#<..!....^..??..^^^..^...#",
        "#........................#",
        "#........................#",
        "#........................#",
        "#......+....+....+...+...#",
        "#......+....+....+...+...#",
        "#......+....+........+...#",
        "#......+....+....+...+...#",
        "#...........+....+...+...#",
        "#......+....+....+...+...#",
        "#......+....+....+...+...#",
        "#......+.........+...+...#",
        "#......+....+....+...+...#",
        "#......+....+....+.......#",
        "#......+....+....+...+...#",
        "#...........{}...........#",
        "#...........[]...........#",
        "##########################",
    ],
]


LEVEL_CONFIGS = [
    {
        "small_cols": [2, 6, 10, 14, 18, 22],
        "type_cycle": ["small"],
        "use_warnings": False,
        "rubble": False,
        "wind": False,
    },
    {
        "small_cols": [5, 18, 9, 14, 3, 20, 7, 16],
        "type_cycle": ["small"],
        "use_warnings": True,
        "rubble": False,
        "wind": False,
    },
    {
        "small_cols": [4, 19, 6, 16, 9, 12, 1, 21],
        "slab_cols": [2, 8, 14, 18],
        "type_cycle": ["small", "small", "slab", "small"],
        "use_warnings": True,
        "rubble": False,
        "wind": False,
    },
    {
        "small_cols": [3, 20, 6, 17, 9, 14, 4, 21],
        "fast_cols": [3, 20, 6, 17, 9, 14, 4, 21],
        "slab_cols": [5, 12, 17],
        "type_cycle": ["small", "small", "fast", "small", "slab", "small", "fast", "small"],
        "use_warnings": True,
        "rubble": False,
        "wind": False,
    },
    {
        "small_cols": [2, 21, 6, 18, 9, 14, 4, 19],
        "fast_cols": [2, 21, 6, 18, 9, 14, 4, 19],
        "slab_cols": [3, 10, 16],
        "big_cols": [4, 11, 17],
        "type_cycle": ["small", "fast", "small", "slab", "small"],
        "use_warnings": True,
        "rubble": True,
        "wind": False,
    },
    {
        "small_cols": [4, 19, 7, 16, 10, 13, 2, 21],
        "fast_cols": [4, 19, 7, 16, 10, 13, 2, 21],
        "slab_cols": [5, 9, 14, 18],
        "big_cols": [6, 12, 17],
        "type_cycle": ["small", "fast", "slab", "small", "fast"],
        "type_cycle2": ["small", "big", "small", "big"],
        "use_warnings": True,
        "rubble": True,
        "wind": True,
    },
]


def _empty_grid(fill: int = -1) -> np.ndarray:
    return np.full((GRID_H, GRID_W), int(fill), dtype=np.int8)


def _hazard_size(kind: str) -> tuple[int, int]:
    if kind == "small" or kind == "fast":
        return (1, 1)
    if kind == "slab":
        return (3, 1)
    if kind == "big":
        return (2, 2)
    raise ValueError(f"unknown hazard kind: {kind}")


def _hazard_cells(kind: str, x: int, y: int) -> list[tuple[int, int]]:
    w, h = _hazard_size(kind)
    return [(x + dx, y + dy) for dy in range(h) for dx in range(w)]


def _warning_width(kind: str) -> int:
    if kind == "small" or kind == "fast":
        return 1
    if kind == "slab":
        return 3
    if kind == "big":
        return 2
    raise ValueError(f"unknown warning kind: {kind}")


def _player_cells(x: int, y: int) -> list[tuple[int, int]]:
    return [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]


def _wall_cells() -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for x in range(GRID_W):
        out.add((x, 0))
        out.add((x, GRID_H - 1))
        out.add((x, 3))
    for y in range(GRID_H):
        out.add((0, y))
        out.add((GRID_W - 1, y))
    return out


WALL_CELLS = _wall_cells()


def _level_layout_to_state(level_index: int) -> RuntimeState:
    rows = LEVEL_LAYOUTS[level_index]
    if len(rows) != GRID_H:
        raise RuntimeError("invalid level layout height")

    player_x = 11
    player_y = 19
    hazards: list[Hazard] = []
    warnings: list[WarningGroup] = []
    barriers: dict[tuple[int, int], int] = {}
    rubble: dict[tuple[int, int], int] = {}
    trails: dict[tuple[int, int], int] = {}
    seen_big: set[tuple[int, int]] = set()

    for y, row in enumerate(rows):
        if len(row) != GRID_W:
            raise RuntimeError("invalid level layout width")
        x = 0
        while x < GRID_W:
            ch = row[x]
            if ch == "{" and x + 1 < GRID_W and row[x + 1] == "}":
                player_x = x
                player_y = y
                x += 2
                continue
            if ch == "[" and x + 1 < GRID_W and row[x + 1] == "]":
                x += 2
                continue
            if ch == "+":
                barriers[(x, y)] = 1
            elif ch == ";":
                barriers[(x, y)] = 2
            elif ch == ",":
                rubble[(x, y)] = 6
            elif ch == ":":
                trails[(x, y)] = 2
            elif ch == "*":
                hazards.append(Hazard("small", x, y))
            elif ch == "!":
                if y == 4:
                    warnings.append(WarningGroup("fast", x))
                else:
                    hazards.append(Hazard("fast", x, y))
            elif ch == "~":
                run = 1
                while x + run < GRID_W and row[x + run] == "~":
                    run += 1
                for offset in range(0, run, 3):
                    hazards.append(Hazard("slab", x + offset, y))
                x += run
                continue
            elif ch == "%" and (x, y) not in seen_big:
                if (
                    x + 1 < GRID_W
                    and y + 1 < GRID_H
                    and (
                        row[x + 1] == "%"
                        and LEVEL_LAYOUTS[level_index][y + 1][x] == "%"
                        and LEVEL_LAYOUTS[level_index][y + 1][x + 1] == "%"
                    )
                ):
                    hazards.append(Hazard("big", x, y))
                    seen_big.update({(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)})
            elif ch == "^":
                run = 1
                while x + run < GRID_W and row[x + run] == "^":
                    run += 1
                idx = 0
                while idx + 3 <= run:
                    warnings.append(WarningGroup("slab", x + idx))
                    idx += 3
                while idx < run:
                    warnings.append(WarningGroup("small", x + idx))
                    idx += 1
                x += run
                continue
            elif ch == "?":
                run = 1
                while x + run < GRID_W and row[x + run] == "?":
                    run += 1
                idx = 0
                while idx + 2 <= run:
                    warnings.append(WarningGroup("big", x + idx))
                    idx += 2
                x += run
                continue
            x += 1

    return RuntimeState(
        level_index=level_index,
        tick=0,
        time_left=TIMEBAR_STEPS,
        blink_phase=0,
        breathe_phase=0,
        player_x=player_x,
        player_y=player_y,
        hazards=hazards,
        warnings=warnings,
        barriers=barriers,
        rubble=rubble,
        trails=trails,
        small_seq_idx=0,
        slab_seq_idx=0,
        fast_seq_idx=0,
        big_seq_idx=0,
        type_seq_idx=0,
        type_seq2_idx=0,
    )


def _copy_state(state: RuntimeState) -> RuntimeState:
    return RuntimeState(
        level_index=state.level_index,
        tick=state.tick,
        time_left=state.time_left,
        blink_phase=state.blink_phase,
        breathe_phase=state.breathe_phase,
        player_x=state.player_x,
        player_y=state.player_y,
        hazards=list(state.hazards),
        warnings=list(state.warnings),
        barriers=dict(state.barriers),
        rubble=dict(state.rubble),
        trails=dict(state.trails),
        small_seq_idx=state.small_seq_idx,
        slab_seq_idx=state.slab_seq_idx,
        fast_seq_idx=state.fast_seq_idx,
        big_seq_idx=state.big_seq_idx,
        type_seq_idx=state.type_seq_idx,
        type_seq2_idx=state.type_seq2_idx,
    )


def _hazard_occupancy(hazards: Iterable[Hazard], *, exclude: Hazard | None = None) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for hazard in hazards:
        if exclude is not None and hazard == exclude:
            continue
        out.update(_hazard_cells(hazard.kind, hazard.x, hazard.y))
    return out


def _advance_warning_sequences(state: RuntimeState) -> list[WarningGroup]:
    cfg = LEVEL_CONFIGS[state.level_index]
    out: list[WarningGroup] = []

    def place(kind: str, inside_col: int):
        gx = INSIDE_X_TO_GRID(inside_col)
        width = _warning_width(kind)
        if gx < 1 or gx + width - 1 > 24:
            return
        out.append(WarningGroup(kind, gx))

    if state.level_index == 0:
        return []

    if state.level_index == 1:
        cols = cfg["small_cols"]
        inside = cols[state.small_seq_idx % len(cols)]
        state.small_seq_idx += 1
        place("small", inside)
        return out

    if state.level_index in (2, 3, 4):
        cycle = cfg["type_cycle"]
        kind = cycle[state.type_seq_idx % len(cycle)]
        state.type_seq_idx += 1
        if kind == "small":
            cols = cfg["small_cols"]
            inside = cols[state.small_seq_idx % len(cols)]
            state.small_seq_idx += 1
            place("small", inside)
        elif kind == "fast":
            cols = cfg["fast_cols"]
            inside = cols[state.small_seq_idx % len(cols)]
            state.small_seq_idx += 1
            place("fast", inside)
        elif kind == "slab":
            cols = cfg["slab_cols"]
            inside = cols[state.slab_seq_idx % len(cols)]
            state.slab_seq_idx += 1
            place("slab", inside)

        if state.level_index == 4 and (state.tick + 1) % 6 == 0:
            cols = cfg["big_cols"]
            inside = cols[state.big_seq_idx % len(cols)]
            state.big_seq_idx += 1
            place("big", inside)

        return out

    # Level 6: two groups per step, placed left->right when they overlap.
    if state.level_index == 5:
        cycle_a = cfg["type_cycle"]
        cycle_b = cfg["type_cycle2"]
        type_a = cycle_a[state.type_seq_idx % len(cycle_a)]
        type_b = cycle_b[state.type_seq2_idx % len(cycle_b)]
        state.type_seq_idx += 1
        state.type_seq2_idx += 1

        def next_inside(kind: str) -> int:
            if kind == "small":
                cols = cfg["small_cols"]
                val = cols[state.small_seq_idx % len(cols)]
                state.small_seq_idx += 1
                return val
            if kind == "fast":
                cols = cfg["fast_cols"]
                val = cols[state.small_seq_idx % len(cols)]
                state.small_seq_idx += 1
                return val
            if kind == "slab":
                cols = cfg["slab_cols"]
                val = cols[state.slab_seq_idx % len(cols)]
                state.slab_seq_idx += 1
                return val
            if kind == "big":
                cols = cfg["big_cols"]
                val = cols[state.big_seq_idx % len(cols)]
                state.big_seq_idx += 1
                return val
            raise ValueError(f"unknown warning type: {kind}")

        pending = [
            WarningGroup(type_a, INSIDE_X_TO_GRID(next_inside(type_a))),
            WarningGroup(type_b, INSIDE_X_TO_GRID(next_inside(type_b))),
        ]

        occupied: set[tuple[int, int]] = set()
        for group in sorted(pending, key=lambda item: item.x):
            width = _warning_width(group.kind)
            cells = {(group.x + offset, 4) for offset in range(width)}
            if any(x < 1 or x > 24 for x, _ in cells):
                continue
            if cells & occupied:
                continue
            occupied |= cells
            out.append(group)
        return out

    return out


def _cells_overlap_player(cells: Iterable[tuple[int, int]], player_x: int, player_y: int) -> bool:
    pset = set(_player_cells(player_x, player_y))
    return any(cell in pset for cell in cells)


def _apply_barrier_hits(state: RuntimeState, target_cells: list[tuple[int, int]]) -> tuple[bool, bool]:
    # Track whether the sweep touched intact or already-cracked barriers.
    hit_intact = False
    hit_cracked = False
    for cell in target_cells:
        barrier_state = state.barriers.get(cell)
        if barrier_state == 1:
            hit_intact = True
        elif barrier_state == 2:
            hit_cracked = True
    if hit_intact:
        for cell in target_cells:
            if state.barriers.get(cell) == 1:
                state.barriers[cell] = 2
        return True, hit_cracked
    return False, hit_cracked


def _attempt_hazard_move(
    *, state: RuntimeState, hazard: Hazard, dx: int, dy: int, occupancy: set[tuple[int, int]], drift_phase: bool
) -> tuple[str, Hazard]:
    _hazard_cells(hazard.kind, hazard.x, hazard.y)
    dst_hazard = Hazard(hazard.kind, hazard.x + dx, hazard.y + dy)
    dst_cells = _hazard_cells(dst_hazard.kind, dst_hazard.x, dst_hazard.y)

    hit_intact, hit_cracked = _apply_barrier_hits(state, dst_cells)
    if hit_intact:
        return "stall", hazard

    for cell in dst_cells:
        if cell in WALL_CELLS:
            return ("blocked_drift" if drift_phase else "blocked_drop"), hazard
        if cell in state.rubble:
            return ("blocked_drift" if drift_phase else "blocked_drop"), hazard
        if cell in occupancy:
            return ("blocked_drift" if drift_phase else "blocked_drop"), hazard

    if hit_cracked:
        for cell in dst_cells:
            if state.barriers.get(cell) == 2:
                state.barriers.pop(cell, None)

    return "moved", dst_hazard


def _spawn_hazard_from_warning(warning: WarningGroup) -> Hazard:
    spawn_y = 5
    return Hazard(warning.kind, warning.x, spawn_y)


def _collect_hazard_lethal_cells(hazards: Iterable[Hazard]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for hazard in hazards:
        out.update(_hazard_cells(hazard.kind, hazard.x, hazard.y))
    return out


def _wind_dx_for_tick(tick: int) -> int:
    phase = (tick // 4) % 2
    return -1 if phase == 0 else 1


@dataclass(frozen=True)
class StepHazardTrace:
    start_cells: frozenset[tuple[int, int]]
    lethal_cells: frozenset[tuple[int, int]]


def _advance_world(
    state: RuntimeState, *, player_x: int, player_y: int, check_player_collision: bool
) -> tuple[bool, StepHazardTrace]:
    cfg = LEVEL_CONFIGS[state.level_index]
    lethal_cells = _collect_hazard_lethal_cells(state.hazards)
    start_cells = set(lethal_cells)

    def mark_and_check(hazard_cells: Iterable[tuple[int, int]]) -> bool:
        cells = list(hazard_cells)
        lethal_cells.update(cells)
        return check_player_collision and _cells_overlap_player(cells, player_x, player_y)

    # Phase 2: wind drift on level 6.
    if cfg["wind"]:
        dx = _wind_dx_for_tick(state.tick)
        ordered = sorted(state.hazards, key=lambda item: (item.y, item.x))
        occupancy = _hazard_occupancy(ordered)
        moved: list[Hazard] = []
        for hazard in ordered:
            for cell in _hazard_cells(hazard.kind, hazard.x, hazard.y):
                occupancy.discard(cell)
            status, candidate = _attempt_hazard_move(
                state=state, hazard=hazard, dx=dx, dy=0, occupancy=occupancy, drift_phase=True
            )
            kept = hazard if status in {"stall", "blocked_drift", "blocked_drop"} else candidate
            moved.append(kept)
            for cell in _hazard_cells(kept.kind, kept.x, kept.y):
                occupancy.add(cell)
            if mark_and_check(_hazard_cells(kept.kind, kept.x, kept.y)):
                return True, StepHazardTrace(frozenset(start_cells), frozenset(lethal_cells))
        state.hazards = moved

    # Phase 3: hazards fall.
    ordered = sorted(state.hazards, key=lambda item: (item.y, item.x))
    occupancy = _hazard_occupancy(ordered)
    fallen: list[Hazard] = []
    for hazard in ordered:
        for cell in _hazard_cells(hazard.kind, hazard.x, hazard.y):
            occupancy.discard(cell)

        current = hazard
        destroyed = False
        substeps = 2 if hazard.kind == "fast" else 1
        for sub in range(substeps):
            status, candidate = _attempt_hazard_move(
                state=state, hazard=current, dx=0, dy=1, occupancy=occupancy, drift_phase=False
            )
            if status == "moved":
                if hazard.kind == "fast" and sub == 0:
                    for trail_cell in _hazard_cells(candidate.kind, candidate.x, candidate.y):
                        state.trails[trail_cell] = 2
                current = candidate
                if mark_and_check(_hazard_cells(current.kind, current.x, current.y)):
                    return True, StepHazardTrace(frozenset(start_cells), frozenset(lethal_cells))
                continue
            if status == "stall":
                break
            if status == "blocked_drop":
                destroyed = True
                if cfg["rubble"]:
                    for cell in _hazard_cells(current.kind, current.x, current.y):
                        if cell not in WALL_CELLS:
                            state.rubble[cell] = 6
                break
            # drift-block should never appear in drop phase.
            break

        if not destroyed:
            fallen.append(current)
            for cell in _hazard_cells(current.kind, current.x, current.y):
                occupancy.add(cell)

    state.hazards = fallen

    # Phase 4: warnings become hazards at spawn row.
    spawned: list[Hazard] = []
    occupancy = _hazard_occupancy(state.hazards)
    for warning in sorted(state.warnings, key=lambda item: item.x):
        hazard = _spawn_hazard_from_warning(warning)
        cells = _hazard_cells(hazard.kind, hazard.x, hazard.y)
        if any(cell in WALL_CELLS or cell in state.rubble for cell in cells):
            continue
        if any(cell in occupancy for cell in cells):
            continue
        spawned.append(hazard)
        occupancy.update(cells)
        if mark_and_check(cells):
            return True, StepHazardTrace(frozenset(start_cells), frozenset(lethal_cells))
    state.hazards.extend(spawned)
    state.warnings = []

    # Level 1 special: direct spawn without warning phase.
    if state.level_index == 0:
        cols = LEVEL_CONFIGS[0]["small_cols"]
        inside = cols[state.small_seq_idx % len(cols)]
        state.small_seq_idx += 1
        direct = Hazard("small", INSIDE_X_TO_GRID(inside), 5)
        cells = _hazard_cells(direct.kind, direct.x, direct.y)
        if not any(cell in WALL_CELLS or cell in state.rubble or cell in occupancy for cell in cells):
            state.hazards.append(direct)
            if mark_and_check(cells):
                return True, StepHazardTrace(frozenset(start_cells), frozenset(lethal_cells))

    # Phase 5: generate warnings for next step.
    if state.level_index != 0:
        new_groups = _advance_warning_sequences(state)
        occupied = {(group.x + offset, 4) for group in state.warnings for offset in range(_warning_width(group.kind))}
        for group in new_groups:
            cells = {(group.x + offset, 4) for offset in range(_warning_width(group.kind))}
            if any(x < 1 or x > 24 for x, _ in cells):
                continue
            if cells & occupied:
                continue
            occupied |= cells
            state.warnings.append(group)

    # Phase 6: animation updates.
    state.blink_phase ^= 1
    state.breathe_phase ^= 1

    new_trails: dict[tuple[int, int], int] = {}
    for cell, ttl in state.trails.items():
        nxt = int(ttl) - 1
        if nxt > 0:
            new_trails[cell] = nxt
    state.trails = new_trails

    new_rubble: dict[tuple[int, int], int] = {}
    for cell, ttl in state.rubble.items():
        nxt = int(ttl) - 1
        if nxt > 0:
            new_rubble[cell] = nxt
    state.rubble = new_rubble

    # Phase 7: timebar decrements.
    state.time_left -= 1
    state.tick += 1

    return False, StepHazardTrace(frozenset(start_cells), frozenset(lethal_cells))


def build_survival_plan(state: RuntimeState) -> list[int] | None:
    sim_state = _copy_state(state)
    steps = sim_state.time_left

    solids_per_step: list[set[tuple[int, int]]] = []
    hazard_start_per_step: list[set[tuple[int, int]]] = []
    lethal_per_step: list[set[tuple[int, int]]] = []

    for _ in range(steps):
        hazard_start = _collect_hazard_lethal_cells(sim_state.hazards)
        solids = set(WALL_CELLS) | set(sim_state.barriers.keys()) | set(sim_state.rubble.keys())
        solids_per_step.append(solids)
        hazard_start_per_step.append(hazard_start)
        _, trace = _advance_world(sim_state, player_x=-999, player_y=-999, check_player_collision=False)
        lethal_per_step.append(set(trace.lethal_cells))

    start = (state.player_x, state.player_y)
    start_key = (0, start[0], start[1])
    queue = deque([start_key])
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_key: None}
    parent_action: dict[tuple[int, int, int], int] = {}

    def can_stand(px: int, py: int, blocked: set[tuple[int, int]]) -> bool:
        if px < 1 or py < 4 or px + 1 > 24 or py + 1 > 20:
            return False
        return all(cell not in blocked for cell in _player_cells(px, py))

    goal_key: tuple[int, int, int] | None = None

    while queue:
        t, px, py = queue.popleft()
        if t >= steps:
            goal_key = (t, px, py)
            break

        blocked_for_move = solids_per_step[t] | hazard_start_per_step[t]
        lethal = lethal_per_step[t]

        for action_id, (dx, dy) in MOVE_DELTAS.items():
            nx = px + dx
            ny = py + dy
            if not can_stand(nx, ny, blocked_for_move):
                continue
            if any(cell in lethal for cell in _player_cells(nx, ny)):
                continue
            nxt_key = (t + 1, nx, ny)
            if nxt_key in parent:
                continue
            parent[nxt_key] = (t, px, py)
            parent_action[nxt_key] = action_id
            queue.append(nxt_key)

    if goal_key is None:
        return None

    actions: list[int] = []
    cursor = goal_key
    while parent[cursor] is not None:
        actions.append(parent_action[cursor])
        cursor = parent[cursor]  # type: ignore[assignment]
    actions.reverse()
    return actions


class DodgerFallingBlocks(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [self._build_level(i) for i in range(len(LEVEL_LAYOUTS))]
        camera = Camera(width=GRID_W, height=GRID_H, background=COLOR_EMPTY)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._runtime: RuntimeState | None = None
        self._floor: Sprite | None = None
        self._walls: Sprite | None = None
        self._timebar: Sprite | None = None
        self._barrier_layer: Sprite | None = None
        self._rubble_layer: Sprite | None = None
        self._trail_layer: Sprite | None = None
        self._warning_layer: Sprite | None = None
        self._small_layer: Sprite | None = None
        self._slab_layer: Sprite | None = None
        self._fast_layer: Sprite | None = None
        self._big_layer: Sprite | None = None
        self._player_layer: Sprite | None = None

    def _build_level(self, level_index: int) -> Level:
        sprites: list[Sprite] = [
            Sprite(
                pixels=np.full((GRID_H, GRID_W), COLOR_EMPTY, dtype=np.int8),
                name="floor",
                x=0,
                y=0,
                layer=0,
                tags=["floor", "sys_static"],
                collidable=False,
            ),
            Sprite(
                pixels=_empty_grid(), name="walls", x=0, y=0, layer=1, tags=["walls", "sys_static"], collidable=False
            ),
            Sprite(pixels=_empty_grid(), name="timebar", x=0, y=0, layer=2, tags=["ui", "timebar"], collidable=False),
            Sprite(pixels=_empty_grid(), name="barriers", x=0, y=0, layer=3, tags=["barriers"], collidable=False),
            Sprite(pixels=_empty_grid(), name="rubble", x=0, y=0, layer=4, tags=["rubble"], collidable=False),
            Sprite(pixels=_empty_grid(), name="trail", x=0, y=0, layer=5, tags=["trail"], collidable=False),
            Sprite(pixels=_empty_grid(), name="warnings", x=0, y=0, layer=6, tags=["warnings"], collidable=False),
            Sprite(
                pixels=_empty_grid(), name="hazard-small", x=0, y=0, layer=7, tags=["hazard_small"], collidable=False
            ),
            Sprite(pixels=_empty_grid(), name="hazard-slab", x=0, y=0, layer=8, tags=["hazard_slab"], collidable=False),
            Sprite(pixels=_empty_grid(), name="hazard-fast", x=0, y=0, layer=9, tags=["hazard_fast"], collidable=False),
            Sprite(pixels=_empty_grid(), name="hazard-big", x=0, y=0, layer=10, tags=["hazard_big"], collidable=False),
            Sprite(pixels=_empty_grid(), name="player", x=0, y=0, layer=11, tags=["player"], collidable=False),
        ]
        return Level(
            name=f"Level {level_index + 1}",
            grid_size=(GRID_W, GRID_H),
            sprites=sprites,
            data={"level_index": level_index},
        )

    def on_set_level(self, level: Level) -> None:
        level_index = int(level.get_data("level_index") or 0)
        self._runtime = _level_layout_to_state(level_index)

        self._floor = level.get_sprites_by_name("floor")[0]
        self._walls = level.get_sprites_by_name("walls")[0]
        self._timebar = level.get_sprites_by_name("timebar")[0]
        self._barrier_layer = level.get_sprites_by_name("barriers")[0]
        self._rubble_layer = level.get_sprites_by_name("rubble")[0]
        self._trail_layer = level.get_sprites_by_name("trail")[0]
        self._warning_layer = level.get_sprites_by_name("warnings")[0]
        self._small_layer = level.get_sprites_by_name("hazard-small")[0]
        self._slab_layer = level.get_sprites_by_name("hazard-slab")[0]
        self._fast_layer = level.get_sprites_by_name("hazard-fast")[0]
        self._big_layer = level.get_sprites_by_name("hazard-big")[0]
        self._player_layer = level.get_sprites_by_name("player")[0]

        self._redraw()

    def _action_to_delta(self, action_id: int) -> tuple[int, int]:
        return MOVE_DELTAS.get(action_id, (0, 0))

    def _can_place_player(self, x: int, y: int) -> bool:
        if x < 1 or y < 4 or x + 1 > 24 or y + 1 > 20:
            return False
        if self._runtime is None:
            return False
        hazard_cells = _collect_hazard_lethal_cells(self._runtime.hazards)
        blocked = set(WALL_CELLS) | set(self._runtime.barriers.keys()) | set(self._runtime.rubble.keys()) | hazard_cells
        return all(cell not in blocked for cell in _player_cells(x, y))

    def _player_hit_now(self) -> bool:
        if self._runtime is None:
            return False
        return _cells_overlap_player(
            _collect_hazard_lethal_cells(self._runtime.hazards), self._runtime.player_x, self._runtime.player_y
        )

    def _wind_indicator(self) -> tuple[int, int] | None:
        if self._runtime is None or self._runtime.level_index != 5:
            return None
        dx = _wind_dx_for_tick(self._runtime.tick)
        return (1, 4) if dx < 0 else (24, 4)

    def _redraw(self) -> None:
        if self._runtime is None:
            return

        wall_grid = _empty_grid()
        for x, y in WALL_CELLS:
            wall_grid[y, x] = COLOR_WALL
        self._walls.pixels = wall_grid

        time_grid = _empty_grid()
        remaining = max(0, int(self._runtime.time_left))
        idx = 0
        for y in (1, 2):
            for x in range(1, 25):
                time_grid[y, x] = COLOR_TIMEBAR_REMAINING if idx < remaining else COLOR_TIMEBAR_SPENT
                idx += 1
        self._timebar.pixels = time_grid

        barrier_grid = _empty_grid()
        for (x, y), kind in self._runtime.barriers.items():
            barrier_grid[y, x] = COLOR_BARRIER_INTACT if kind == 1 else COLOR_BARRIER_CRACKED
        self._barrier_layer.pixels = barrier_grid

        rubble_grid = _empty_grid()
        for (x, y), ttl in self._runtime.rubble.items():
            if ttl <= 2 and (self._runtime.tick % 2 == 1):
                continue
            rubble_grid[y, x] = COLOR_RUBBLE
        self._rubble_layer.pixels = rubble_grid

        trail_grid = _empty_grid()
        for x, y in self._runtime.trails:
            trail_grid[y, x] = COLOR_TRAIL
        self._trail_layer.pixels = trail_grid

        warning_grid = _empty_grid()
        if self._runtime.blink_phase == 0:
            for warning in self._runtime.warnings:
                for offset in range(_warning_width(warning.kind)):
                    wx = warning.x + offset
                    if 1 <= wx <= 24:
                        warning_grid[4, wx] = COLOR_WARNING
            wind_cell = self._wind_indicator()
            if wind_cell is not None:
                warning_grid[wind_cell[1], wind_cell[0]] = COLOR_WARNING
        self._warning_layer.pixels = warning_grid

        small_grid = _empty_grid()
        slab_grid = _empty_grid()
        fast_grid = _empty_grid()
        big_grid = _empty_grid()
        for hazard in self._runtime.hazards:
            cells = _hazard_cells(hazard.kind, hazard.x, hazard.y)
            target = {"small": small_grid, "slab": slab_grid, "fast": fast_grid, "big": big_grid}[hazard.kind]
            color = {
                "small": COLOR_HAZARD_SMALL,
                "slab": COLOR_HAZARD_SLAB,
                "fast": COLOR_HAZARD_FAST,
                "big": COLOR_HAZARD_BIG,
            }[hazard.kind]
            for x, y in cells:
                target[y, x] = color

        self._small_layer.pixels = small_grid
        self._slab_layer.pixels = slab_grid
        self._fast_layer.pixels = fast_grid
        self._big_layer.pixels = big_grid

        player_grid = _empty_grid()
        pcolor = COLOR_PLAYER_A if self._runtime.breathe_phase == 0 else COLOR_PLAYER_B
        for x, y in _player_cells(self._runtime.player_x, self._runtime.player_y):
            player_grid[y, x] = pcolor
        self._player_layer.pixels = player_grid

    def export_solver_state(self) -> RuntimeState:
        if self._runtime is None:
            raise RuntimeError("dodger_falling_blocks state is not initialized")
        return _copy_state(self._runtime)

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

        if self._runtime is None:
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        dx, dy = self._action_to_delta(action_id)

        # 1) Player action.
        next_x = self._runtime.player_x + dx
        next_y = self._runtime.player_y + dy
        if self._can_place_player(next_x, next_y):
            self._runtime.player_x = next_x
            self._runtime.player_y = next_y

        if self._player_hit_now():
            self.lose()
            self._redraw()
            self.complete_action()
            return

        # 2..7) World update.
        dead, _ = _advance_world(
            self._runtime, player_x=self._runtime.player_x, player_y=self._runtime.player_y, check_player_collision=True
        )
        if dead:
            self.lose()
            self._redraw()
            self.complete_action()
            return

        if self._runtime.time_left <= 0:
            self.next_level()
            self.complete_action()
            return

        self._redraw()
        self.complete_action()
