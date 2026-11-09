from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "chase_escape-0001"

GRID_W = 28
GRID_H = 16
PLAY_MIN_Y = 1
PLAY_MAX_Y = 15

COLOR_VOID = 0
COLOR_WALL = 1
COLOR_FLOOR = 2
COLOR_PLAYER_MAIN = 3
COLOR_PLAYER_ACCENT = 4
COLOR_CHASER_MAIN = 5
COLOR_CHASER_ACCENT = 6
COLOR_EXIT_FRAME = 7
COLOR_GLOW = 8
COLOR_GATE_CLOSED = 9
COLOR_SWITCH = 10
COLOR_KEY = 11
COLOR_CRATE = 12
COLOR_HOURGLASS = 13
COLOR_TIME_DANGER = 14
COLOR_TIME_OK = 15

ACTION_TO_DELTA = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
SPACE_ACTION = int(GameAction.ACTION5.value)
VALID_ACTIONS = tuple(sorted((*ACTION_TO_DELTA.keys(), SPACE_ACTION)))

TERMINAL_NONE = 0
TERMINAL_WIN = 1
TERMINAL_LOSE = 2
TERMINAL_ANIM_STEPS = 20


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class LevelModel:
    name: str
    time_max: int
    walls: frozenset[tuple[int, int]]
    player_start: tuple[int, int]
    chaser_start: tuple[int, int]
    exit_rect: Rect
    switches: tuple[Rect, ...]
    switch_gates: tuple[Rect, ...]
    switch_links: tuple[tuple[int, ...], ...]
    locked_gates: tuple[Rect, ...]
    keys: tuple[Rect, ...]
    hourglasses: tuple[Rect, ...]
    crates: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SimState:
    player: tuple[int, int]
    chaser: tuple[int, int]
    crates: tuple[tuple[int, int], ...]
    switch_open_mask: int
    locked_open_mask: int
    has_key: bool
    key_mask: int
    hourglass_mask: int
    switch_overlap_mask: int
    time_remaining: int
    anim_phase: int
    terminal_mode: int
    terminal_ticks: int
    switch_flash_mask: int


@dataclass(frozen=True)
class StepResult:
    state: SimState
    advance_level: bool
    restart_level: bool


LEVEL_SPECS = [
    {
        "name": "Level 1 - First Run",
        "time_max": 120,
        "links": [],
        "rows": [
            "============================",
            "############################",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "#..CC..PP.............EEE..#",
            "#..CC..PP.............EEE..#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "############################",
        ],
    },
    {
        "name": "Level 2 - Detour for Time",
        "time_max": 140,
        "links": [],
        "rows": [
            "============================",
            "############################",
            "#.............#............#",
            "#..CC.........#............#",
            "#..CC...............HH.....#",
            "#...................HH.....#",
            "#.............#............#",
            "#.............#............#",
            "#.............#............#",
            "#.............#............#",
            "#.............#............#",
            "#.............#............#",
            "#...PP.................EEE.#",
            "#...PP.................EEE.#",
            "#.............#............#",
            "############################",
        ],
    },
    {
        "name": "Level 3 - Switch Opens the Way",
        "time_max": 160,
        "links": [(0, 0)],
        "rows": [
            "============================",
            "############################",
            "#.CC.........##............#",
            "#.CC.........##............#",
            "#............##............#",
            "#............##............#",
            "#............##............#",
            "#............##............#",
            "#............GG............#",
            "#............GG............#",
            "#...SS.......##............#",
            "#...SS.......##............#",
            "#......PP....##.......EEE..#",
            "#......PP....##.......EEE..#",
            "#............##............#",
            "############################",
        ],
    },
    {
        "name": "Level 4 - Key and Lock",
        "time_max": 170,
        "links": [],
        "rows": [
            "============================",
            "############################",
            "#.........CC.##............#",
            "#.........CC.##............#",
            "#..KK........##............#",
            "#..KK........##............#",
            "#.......##...##............#",
            "#.......##...##............#",
            "#............LL............#",
            "#............LL............#",
            "#............##............#",
            "#............##............#",
            "#.....PP.....##.......EEE..#",
            "#.....PP.....##.......EEE..#",
            "#............##............#",
            "############################",
        ],
    },
    {
        "name": "Level 5 - Barricade",
        "time_max": 130,
        "links": [],
        "rows": [
            "============================",
            "############################",
            "#..........................#",
            "#.....................EEE..#",
            "#............PP.......EEE..#",
            "#............PP............#",
            "#............BB............#",
            "#............BB............#",
            "#############..########..###",
            "#..........................#",
            "#............CC............#",
            "#............CC............#",
            "#..........................#",
            "#..........................#",
            "#..........................#",
            "############################",
        ],
    },
    {
        "name": "Level 6 - Full Escape",
        "time_max": 200,
        "links": [(0, 0)],
        "rows": [
            "============================",
            "############################",
            "#............##CC.....EEE..#",
            "#............##CC.....EEE..#",
            "#...KK.......##............#",
            "#...KK.......##########LL###",
            "#............##########LL###",
            "#........HH..##............#",
            "#........HH..##............#",
            "#............GG....BB......#",
            "#............GG....BB......#",
            "#......SS....##............#",
            "#......SS....##............#",
            "#....PP......##............#",
            "#....PP......##............#",
            "############################",
        ],
    },
]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _in_bounds_rect(x: int, y: int, w: int, h: int) -> bool:
    return x >= 0 and y >= PLAY_MIN_Y and (x + w) <= GRID_W and (y + h) <= GRID_H


def _rect_cells(rect: Rect) -> tuple[tuple[int, int], ...]:
    return tuple((rect.x + dx, rect.y + dy) for dy in range(rect.h) for dx in range(rect.w))


def _rect_cells_at(x: int, y: int, w: int, h: int) -> tuple[tuple[int, int], ...]:
    return tuple((x + dx, y + dy) for dy in range(h) for dx in range(w))


def _rect_overlap(a: Rect, b: Rect) -> bool:
    return not (a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y)


def _component_rects(rows: list[str], symbol: str) -> list[Rect]:
    h = len(rows)
    w = len(rows[0]) if rows else 0
    seen: set[tuple[int, int]] = set()
    rects: list[Rect] = []
    for y in range(h):
        for x in range(w):
            if rows[y][x] != symbol or (x, y) in seen:
                continue
            stack = [(x, y)]
            cells: list[tuple[int, int]] = []
            seen.add((x, y))
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and rows[ny][nx] == symbol and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            rects.append(Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
    rects.sort(key=lambda r: (r.y, r.x))
    return rects


def _parse_level(spec: dict) -> LevelModel:
    rows = [str(r) for r in spec["rows"]]
    if len(rows) != GRID_H:
        raise ValueError(f"{spec['name']}: expected {GRID_H} rows, got {len(rows)}")
    if any(len(row) != GRID_W for row in rows):
        raise ValueError(f"{spec['name']}: all rows must have width {GRID_W}")
    if rows[0] != "=" * GRID_W:
        raise ValueError(f"{spec['name']}: row 0 must be timebar placeholder")

    walls = frozenset((x, y) for y in range(1, GRID_H) for x in range(GRID_W) if rows[y][x] == "#")

    players = _component_rects(rows, "P")
    chasers = _component_rects(rows, "C")
    exits = _component_rects(rows, "E")
    switches = tuple(_component_rects(rows, "S"))
    switch_gates = tuple(_component_rects(rows, "G"))
    locked = tuple(_component_rects(rows, "L"))
    keys = tuple(_component_rects(rows, "K"))
    hourglasses = tuple(_component_rects(rows, "H"))
    crates = tuple((rect.x, rect.y) for rect in _component_rects(rows, "B"))

    if len(players) != 1 or players[0].w != 2 or players[0].h != 2:
        raise ValueError(f"{spec['name']}: expected one 2x2 player")
    if len(chasers) != 1 or chasers[0].w != 2 or chasers[0].h != 2:
        raise ValueError(f"{spec['name']}: expected one 2x2 chaser")
    if len(exits) != 1 or exits[0].w != 3 or exits[0].h != 2:
        raise ValueError(f"{spec['name']}: expected one 3x2 exit")

    for coll, tag in (
        (switches, "switch"),
        (switch_gates, "switch-gate"),
        (locked, "locked"),
        (keys, "key"),
        (hourglasses, "hourglass"),
    ):
        for rect in coll:
            if rect.w != 2 or rect.h != 2:
                raise ValueError(f"{spec['name']}: {tag} must be 2x2")
    for cx, cy in crates:
        if not _in_bounds_rect(cx, cy, 2, 2):
            raise ValueError(f"{spec['name']}: crate out of bounds")

    links = [tuple(int(x) for x in pair) for pair in spec.get("links", [])]
    switch_links: list[tuple[int, ...]] = [tuple() for _ in switches]
    for s_idx, g_idx in links:
        if s_idx < 0 or s_idx >= len(switches) or g_idx < 0 or g_idx >= len(switch_gates):
            raise ValueError(f"{spec['name']}: invalid switch link ({s_idx}, {g_idx})")
        row_links = list(switch_links[s_idx])
        if g_idx not in row_links:
            row_links.append(g_idx)
        switch_links[s_idx] = tuple(row_links)

    return LevelModel(
        name=str(spec["name"]),
        time_max=int(spec["time_max"]),
        walls=walls,
        player_start=(players[0].x, players[0].y),
        chaser_start=(chasers[0].x, chasers[0].y),
        exit_rect=exits[0],
        switches=switches,
        switch_gates=switch_gates,
        switch_links=tuple(switch_links),
        locked_gates=locked,
        keys=keys,
        hourglasses=hourglasses,
        crates=crates,
    )


LEVEL_MODELS: tuple[LevelModel, ...] = tuple(_parse_level(spec) for spec in LEVEL_SPECS)


def initial_state(model: LevelModel) -> SimState:
    key_mask = 0
    for idx in range(len(model.keys)):
        key_mask |= 1 << idx
    hg_mask = 0
    for idx in range(len(model.hourglasses)):
        hg_mask |= 1 << idx
    return SimState(
        player=model.player_start,
        chaser=model.chaser_start,
        crates=tuple(model.crates),
        switch_open_mask=0,
        locked_open_mask=0,
        has_key=False,
        key_mask=key_mask,
        hourglass_mask=hg_mask,
        switch_overlap_mask=0,
        time_remaining=int(model.time_max),
        anim_phase=0,
        terminal_mode=TERMINAL_NONE,
        terminal_ticks=0,
        switch_flash_mask=0,
    )


def _mask_has(mask: int, idx: int) -> bool:
    return bool(mask & (1 << idx))


def _set_mask(mask: int, idx: int, value: bool) -> int:
    if value:
        return mask | (1 << idx)
    return mask & ~(1 << idx)


def _gate_closed_cells(model: LevelModel, switch_open_mask: int, locked_open_mask: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set(model.walls)
    for idx, gate in enumerate(model.switch_gates):
        if not _mask_has(switch_open_mask, idx):
            blocked.update(_rect_cells(gate))
    for idx, gate in enumerate(model.locked_gates):
        if not _mask_has(locked_open_mask, idx):
            blocked.update(_rect_cells(gate))
    return blocked


def _crate_cells(crates: tuple[tuple[int, int], ...], skip_idx: int | None = None) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for idx, (x, y) in enumerate(crates):
        if skip_idx is not None and idx == skip_idx:
            continue
        cells.update(_rect_cells_at(x, y, 2, 2))
    return cells


def _actor_rect(pos: tuple[int, int]) -> Rect:
    return Rect(pos[0], pos[1], 2, 2)


def _player_switch_overlap_mask(model: LevelModel, player_pos: tuple[int, int]) -> int:
    player_rect = _actor_rect(player_pos)
    mask = 0
    for idx, switch in enumerate(model.switches):
        if _rect_overlap(player_rect, switch):
            mask |= 1 << idx
    return mask


def _has_exit_overlap(model: LevelModel, player_pos: tuple[int, int]) -> bool:
    return _rect_overlap(_actor_rect(player_pos), model.exit_rect)


def _has_chaser_overlap(player_pos: tuple[int, int], chaser_pos: tuple[int, int]) -> bool:
    return _rect_overlap(_actor_rect(player_pos), _actor_rect(chaser_pos))


def _is_adjacent(player_pos: tuple[int, int], gate: Rect) -> bool:
    for px, py in _rect_cells_at(player_pos[0], player_pos[1], 2, 2):
        for gx, gy in _rect_cells(gate):
            if abs(px - gx) + abs(py - gy) == 1:
                return True
    return False


def _simulate_player_move(
    model: LevelModel, state: SimState, action_id: int
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    delta = ACTION_TO_DELTA.get(int(action_id))
    if delta is None:
        return state.player, state.crates

    dx, dy = delta
    nx = state.player[0] + dx
    ny = state.player[1] + dy
    if not _in_bounds_rect(nx, ny, 2, 2):
        return state.player, state.crates

    dest_cells = set(_rect_cells_at(nx, ny, 2, 2))
    if dest_cells & set(_rect_cells_at(state.chaser[0], state.chaser[1], 2, 2)):
        return state.player, state.crates

    closed_cells = _gate_closed_cells(model, state.switch_open_mask, state.locked_open_mask)
    if dest_cells & closed_cells:
        return state.player, state.crates

    overlaps: list[int] = []
    for idx, (cx, cy) in enumerate(state.crates):
        if dest_cells & set(_rect_cells_at(cx, cy, 2, 2)):
            overlaps.append(idx)

    if not overlaps:
        return (nx, ny), state.crates

    if len(overlaps) > 1:
        return state.player, state.crates

    crate_idx = overlaps[0]
    crate_x, crate_y = state.crates[crate_idx]
    target_cx = crate_x + dx
    target_cy = crate_y + dy
    if not _in_bounds_rect(target_cx, target_cy, 2, 2):
        return state.player, state.crates

    crate_dest_cells = set(_rect_cells_at(target_cx, target_cy, 2, 2))
    if crate_dest_cells & closed_cells:
        return state.player, state.crates

    other_crates = _crate_cells(state.crates, skip_idx=crate_idx)
    if crate_dest_cells & other_crates:
        return state.player, state.crates

    if crate_dest_cells & set(_rect_cells_at(state.chaser[0], state.chaser[1], 2, 2)):
        return state.player, state.crates

    crates = list(state.crates)
    crates[crate_idx] = (target_cx, target_cy)
    return (nx, ny), tuple(crates)


def _collect_items(model: LevelModel, state: SimState) -> tuple[bool, int, int, int]:
    player_rect = _actor_rect(state.player)
    has_key = state.has_key
    key_mask = state.key_mask
    hourglass_mask = state.hourglass_mask
    time_remaining = state.time_remaining

    for idx, key in enumerate(model.keys):
        if not _mask_has(key_mask, idx):
            continue
        if _rect_overlap(player_rect, key):
            key_mask = _set_mask(key_mask, idx, False)
            has_key = True

    bonus = max(1, int(model.time_max // 4))
    for idx, hg in enumerate(model.hourglasses):
        if not _mask_has(hourglass_mask, idx):
            continue
        if _rect_overlap(player_rect, hg):
            hourglass_mask = _set_mask(hourglass_mask, idx, False)
            time_remaining = min(int(model.time_max), time_remaining + bonus)

    return has_key, key_mask, hourglass_mask, time_remaining


def _resolve_switches(model: LevelModel, state: SimState, old_overlap_mask: int) -> tuple[int, int, int]:
    overlap_mask = _player_switch_overlap_mask(model, state.player)
    entered = overlap_mask & (~old_overlap_mask)
    switch_open_mask = state.switch_open_mask
    flash_mask = 0

    for switch_idx in range(len(model.switches)):
        if not _mask_has(entered, switch_idx):
            continue
        flash_mask = _set_mask(flash_mask, switch_idx, True)
        for gate_idx in model.switch_links[switch_idx]:
            gate_open = _mask_has(switch_open_mask, gate_idx)
            switch_open_mask = _set_mask(switch_open_mask, gate_idx, not gate_open)

    return overlap_mask, switch_open_mask, flash_mask


def _resolve_locked_open(model: LevelModel, state: SimState) -> int:
    if not state.has_key:
        return state.locked_open_mask

    mask = state.locked_open_mask
    for idx, gate in enumerate(model.locked_gates):
        if _mask_has(mask, idx):
            continue
        if _is_adjacent(state.player, gate):
            mask = _set_mask(mask, idx, True)
    return mask


def _simulate_chaser_move(model: LevelModel, state: SimState) -> tuple[int, int]:
    cx, cy = state.chaser
    px, py = state.player
    dx = px - cx
    dy = py - cy

    if abs(dx) >= abs(dy):
        axis_order = ((1 if dx > 0 else -1 if dx < 0 else 0, 0), (0, 1 if dy > 0 else -1 if dy < 0 else 0))
    else:
        axis_order = ((0, 1 if dy > 0 else -1 if dy < 0 else 0), (1 if dx > 0 else -1 if dx < 0 else 0, 0))

    closed_cells = _gate_closed_cells(model, state.switch_open_mask, state.locked_open_mask)
    crate_cells = _crate_cells(state.crates)

    for mx, my in axis_order:
        if mx == 0 and my == 0:
            continue
        nx, ny = cx + mx, cy + my
        if not _in_bounds_rect(nx, ny, 2, 2):
            continue
        cells = set(_rect_cells_at(nx, ny, 2, 2))
        if cells & closed_cells:
            continue
        if cells & crate_cells:
            continue
        return nx, ny

    return cx, cy


def simulate_step(model: LevelModel, state: SimState, action_id: int) -> StepResult:
    action_id = int(action_id)
    if action_id not in VALID_ACTIONS:
        action_id = SPACE_ACTION

    anim_phase = 1 - int(state.anim_phase)

    if state.terminal_mode == TERMINAL_WIN:
        ticks = int(state.terminal_ticks) + 1
        if action_id == SPACE_ACTION or ticks >= TERMINAL_ANIM_STEPS:
            return StepResult(
                state=SimState(
                    player=state.player,
                    chaser=state.chaser,
                    crates=state.crates,
                    switch_open_mask=state.switch_open_mask,
                    locked_open_mask=state.locked_open_mask,
                    has_key=state.has_key,
                    key_mask=state.key_mask,
                    hourglass_mask=state.hourglass_mask,
                    switch_overlap_mask=state.switch_overlap_mask,
                    time_remaining=state.time_remaining,
                    anim_phase=anim_phase,
                    terminal_mode=TERMINAL_WIN,
                    terminal_ticks=ticks,
                    switch_flash_mask=0,
                ),
                advance_level=True,
                restart_level=False,
            )
        return StepResult(
            state=SimState(
                player=state.player,
                chaser=state.chaser,
                crates=state.crates,
                switch_open_mask=state.switch_open_mask,
                locked_open_mask=state.locked_open_mask,
                has_key=state.has_key,
                key_mask=state.key_mask,
                hourglass_mask=state.hourglass_mask,
                switch_overlap_mask=state.switch_overlap_mask,
                time_remaining=state.time_remaining,
                anim_phase=anim_phase,
                terminal_mode=TERMINAL_WIN,
                terminal_ticks=ticks,
                switch_flash_mask=0,
            ),
            advance_level=False,
            restart_level=False,
        )

    if state.terminal_mode == TERMINAL_LOSE:
        ticks = int(state.terminal_ticks) + 1
        if action_id == SPACE_ACTION or ticks >= TERMINAL_ANIM_STEPS:
            return StepResult(
                state=SimState(
                    player=state.player,
                    chaser=state.chaser,
                    crates=state.crates,
                    switch_open_mask=state.switch_open_mask,
                    locked_open_mask=state.locked_open_mask,
                    has_key=state.has_key,
                    key_mask=state.key_mask,
                    hourglass_mask=state.hourglass_mask,
                    switch_overlap_mask=state.switch_overlap_mask,
                    time_remaining=state.time_remaining,
                    anim_phase=anim_phase,
                    terminal_mode=TERMINAL_LOSE,
                    terminal_ticks=ticks,
                    switch_flash_mask=0,
                ),
                advance_level=False,
                restart_level=True,
            )
        return StepResult(
            state=SimState(
                player=state.player,
                chaser=state.chaser,
                crates=state.crates,
                switch_open_mask=state.switch_open_mask,
                locked_open_mask=state.locked_open_mask,
                has_key=state.has_key,
                key_mask=state.key_mask,
                hourglass_mask=state.hourglass_mask,
                switch_overlap_mask=state.switch_overlap_mask,
                time_remaining=state.time_remaining,
                anim_phase=anim_phase,
                terminal_mode=TERMINAL_LOSE,
                terminal_ticks=ticks,
                switch_flash_mask=0,
            ),
            advance_level=False,
            restart_level=False,
        )

    old_overlap = state.switch_overlap_mask
    player_pos, crates = _simulate_player_move(model, state, action_id)

    state_after_move = SimState(
        player=player_pos,
        chaser=state.chaser,
        crates=crates,
        switch_open_mask=state.switch_open_mask,
        locked_open_mask=state.locked_open_mask,
        has_key=state.has_key,
        key_mask=state.key_mask,
        hourglass_mask=state.hourglass_mask,
        switch_overlap_mask=old_overlap,
        time_remaining=state.time_remaining,
        anim_phase=anim_phase,
        terminal_mode=TERMINAL_NONE,
        terminal_ticks=0,
        switch_flash_mask=0,
    )

    has_key, key_mask, hourglass_mask, time_remaining = _collect_items(model, state_after_move)

    interim = SimState(
        player=player_pos,
        chaser=state_after_move.chaser,
        crates=crates,
        switch_open_mask=state_after_move.switch_open_mask,
        locked_open_mask=state_after_move.locked_open_mask,
        has_key=has_key,
        key_mask=key_mask,
        hourglass_mask=hourglass_mask,
        switch_overlap_mask=old_overlap,
        time_remaining=time_remaining,
        anim_phase=anim_phase,
        terminal_mode=TERMINAL_NONE,
        terminal_ticks=0,
        switch_flash_mask=0,
    )

    overlap_mask, switch_open_mask, flash_mask = _resolve_switches(model, interim, old_overlap)
    locked_open_mask = _resolve_locked_open(
        model,
        SimState(
            player=interim.player,
            chaser=interim.chaser,
            crates=interim.crates,
            switch_open_mask=switch_open_mask,
            locked_open_mask=interim.locked_open_mask,
            has_key=interim.has_key,
            key_mask=interim.key_mask,
            hourglass_mask=interim.hourglass_mask,
            switch_overlap_mask=overlap_mask,
            time_remaining=interim.time_remaining,
            anim_phase=interim.anim_phase,
            terminal_mode=TERMINAL_NONE,
            terminal_ticks=0,
            switch_flash_mask=flash_mask,
        ),
    )

    if _has_exit_overlap(model, interim.player):
        return StepResult(
            state=SimState(
                player=interim.player,
                chaser=interim.chaser,
                crates=interim.crates,
                switch_open_mask=switch_open_mask,
                locked_open_mask=locked_open_mask,
                has_key=interim.has_key,
                key_mask=interim.key_mask,
                hourglass_mask=interim.hourglass_mask,
                switch_overlap_mask=overlap_mask,
                time_remaining=interim.time_remaining,
                anim_phase=anim_phase,
                terminal_mode=TERMINAL_WIN,
                terminal_ticks=0,
                switch_flash_mask=flash_mask,
            ),
            advance_level=False,
            restart_level=False,
        )

    move_state = SimState(
        player=interim.player,
        chaser=interim.chaser,
        crates=interim.crates,
        switch_open_mask=switch_open_mask,
        locked_open_mask=locked_open_mask,
        has_key=interim.has_key,
        key_mask=interim.key_mask,
        hourglass_mask=interim.hourglass_mask,
        switch_overlap_mask=overlap_mask,
        time_remaining=interim.time_remaining,
        anim_phase=anim_phase,
        terminal_mode=TERMINAL_NONE,
        terminal_ticks=0,
        switch_flash_mask=flash_mask,
    )
    new_chaser = _simulate_chaser_move(model, move_state)
    new_time = max(0, move_state.time_remaining - 1)

    lost = _has_chaser_overlap(move_state.player, new_chaser) or new_time <= 0
    return StepResult(
        state=SimState(
            player=move_state.player,
            chaser=new_chaser,
            crates=move_state.crates,
            switch_open_mask=move_state.switch_open_mask,
            locked_open_mask=move_state.locked_open_mask,
            has_key=move_state.has_key,
            key_mask=move_state.key_mask,
            hourglass_mask=move_state.hourglass_mask,
            switch_overlap_mask=move_state.switch_overlap_mask,
            time_remaining=new_time,
            anim_phase=anim_phase,
            terminal_mode=TERMINAL_LOSE if lost else TERMINAL_NONE,
            terminal_ticks=0,
            switch_flash_mask=move_state.switch_flash_mask,
        ),
        advance_level=False,
        restart_level=False,
    )


def _paint_block(grid: np.ndarray, rect: Rect, pattern: np.ndarray) -> None:
    grid[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w] = pattern


def _draw_frame(model: LevelModel, state: SimState) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), int(COLOR_FLOOR), dtype=np.int8)

    ratio = 0.0 if model.time_max <= 0 else (float(state.time_remaining) / float(model.time_max))
    fill = int(GRID_W * max(0.0, min(1.0, ratio)))
    bar_color = COLOR_TIME_OK if ratio > 0.25 else COLOR_TIME_DANGER
    grid[0, :] = COLOR_VOID
    if fill > 0:
        grid[0, :fill] = int(bar_color)

    for x, y in model.walls:
        grid[y, x] = COLOR_WALL

    gate_open_a = np.array([[COLOR_GLOW, COLOR_FLOOR], [COLOR_FLOOR, COLOR_FLOOR]], dtype=np.int8)
    gate_open_b = np.array([[COLOR_FLOOR, COLOR_FLOOR], [COLOR_FLOOR, COLOR_GLOW]], dtype=np.int8)
    gate_open = gate_open_b if state.anim_phase else gate_open_a

    switch_gate_closed = np.array(
        [[COLOR_SWITCH, COLOR_GATE_CLOSED], [COLOR_GATE_CLOSED, COLOR_GATE_CLOSED]], dtype=np.int8
    )
    lock_gate_closed = np.array([[COLOR_KEY, COLOR_GATE_CLOSED], [COLOR_GATE_CLOSED, COLOR_GATE_CLOSED]], dtype=np.int8)

    for idx, gate in enumerate(model.switch_gates):
        pattern = gate_open if _mask_has(state.switch_open_mask, idx) else switch_gate_closed
        _paint_block(grid, gate, pattern)

    for idx, gate in enumerate(model.locked_gates):
        pattern = gate_open if _mask_has(state.locked_open_mask, idx) else lock_gate_closed
        _paint_block(grid, gate, pattern)

    switch_color = COLOR_GLOW
    default_switch = np.full((2, 2), int(COLOR_SWITCH), dtype=np.int8)
    flash_switch = np.full((2, 2), int(switch_color), dtype=np.int8)
    for idx, switch in enumerate(model.switches):
        _paint_block(grid, switch, flash_switch if _mask_has(state.switch_flash_mask, idx) else default_switch)

    key_color = COLOR_GLOW if state.anim_phase else COLOR_KEY
    for idx, key in enumerate(model.keys):
        if _mask_has(state.key_mask, idx):
            _paint_block(grid, key, np.full((2, 2), int(key_color), dtype=np.int8))

    hg_color = COLOR_GLOW if state.anim_phase else COLOR_HOURGLASS
    for idx, hourglass in enumerate(model.hourglasses):
        if _mask_has(state.hourglass_mask, idx):
            _paint_block(grid, hourglass, np.full((2, 2), int(hg_color), dtype=np.int8))

    crate_a = np.array([[COLOR_GLOW, COLOR_CRATE], [COLOR_CRATE, COLOR_GLOW]], dtype=np.int8)
    crate_b = np.array([[COLOR_CRATE, COLOR_GLOW], [COLOR_GLOW, COLOR_CRATE]], dtype=np.int8)
    crate_pattern = crate_b if state.anim_phase else crate_a
    for cx, cy in state.crates:
        _paint_block(grid, Rect(cx, cy, 2, 2), crate_pattern)

    exit_a = np.array(
        [[COLOR_EXIT_FRAME, COLOR_GLOW, COLOR_EXIT_FRAME], [COLOR_GLOW, COLOR_EXIT_FRAME, COLOR_GLOW]], dtype=np.int8
    )
    exit_b = np.array(
        [[COLOR_EXIT_FRAME, COLOR_EXIT_FRAME, COLOR_EXIT_FRAME], [COLOR_GLOW, COLOR_GLOW, COLOR_GLOW]], dtype=np.int8
    )
    exit_win = np.array(
        [[COLOR_GLOW, COLOR_GLOW, COLOR_GLOW], [COLOR_GLOW, COLOR_EXIT_FRAME, COLOR_GLOW]], dtype=np.int8
    )

    if state.terminal_mode == TERMINAL_WIN and state.terminal_ticks < TERMINAL_ANIM_STEPS:
        _paint_block(grid, model.exit_rect, exit_win if state.anim_phase else exit_b)
    else:
        _paint_block(grid, model.exit_rect, exit_b if state.anim_phase else exit_a)

    player_a = np.array(
        [[COLOR_PLAYER_MAIN, COLOR_PLAYER_ACCENT], [COLOR_PLAYER_ACCENT, COLOR_PLAYER_MAIN]], dtype=np.int8
    )
    player_b = np.array(
        [[COLOR_PLAYER_ACCENT, COLOR_PLAYER_MAIN], [COLOR_PLAYER_MAIN, COLOR_PLAYER_ACCENT]], dtype=np.int8
    )
    if state.has_key:
        player_a = np.where(player_a == COLOR_PLAYER_ACCENT, COLOR_KEY, player_a)
        player_b = np.where(player_b == COLOR_PLAYER_ACCENT, COLOR_KEY, player_b)

    chaser_a = np.array(
        [[COLOR_CHASER_MAIN, COLOR_CHASER_ACCENT], [COLOR_CHASER_ACCENT, COLOR_CHASER_MAIN]], dtype=np.int8
    )
    chaser_b = np.array(
        [[COLOR_CHASER_ACCENT, COLOR_CHASER_MAIN], [COLOR_CHASER_MAIN, COLOR_CHASER_ACCENT]], dtype=np.int8
    )

    if state.terminal_mode == TERMINAL_LOSE and (state.terminal_ticks % 2 == 0):
        player_a = np.where(player_a == COLOR_PLAYER_MAIN, COLOR_TIME_DANGER, player_a)
        player_b = np.where(player_b == COLOR_PLAYER_MAIN, COLOR_TIME_DANGER, player_b)
        chaser_a = np.where(chaser_a == COLOR_CHASER_MAIN, COLOR_TIME_DANGER, chaser_a)
        chaser_b = np.where(chaser_b == COLOR_CHASER_MAIN, COLOR_TIME_DANGER, chaser_b)

    _paint_block(grid, Rect(state.chaser[0], state.chaser[1], 2, 2), chaser_b if state.anim_phase else chaser_a)
    _paint_block(grid, Rect(state.player[0], state.player[1], 2, 2), player_b if state.anim_phase else player_a)

    return grid


def _build_level(model: LevelModel, idx: int) -> Level:
    board = Sprite(
        _solid(GRID_W, GRID_H, COLOR_VOID), name="board", x=0, y=0, layer=0, tags=["board"], collidable=False
    )
    return Level(name=model.name, grid_size=(GRID_W, GRID_H), sprites=[board], data={"model_index": int(idx)})


class ChaseEscape(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(model, idx) for idx, model in enumerate(LEVEL_MODELS)]
        camera = Camera(width=GRID_W, height=GRID_H, background=COLOR_VOID)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._route_score = 0
        self._sim_state = initial_state(LEVEL_MODELS[0])
        self._model = LEVEL_MODELS[0]
        self._board: Sprite | None = None

    def on_set_level(self, level: Level) -> None:
        model_idx = int(level.get_data("model_index") or 0)
        model_idx = max(0, min(model_idx, len(LEVEL_MODELS) - 1))
        self._model = LEVEL_MODELS[model_idx]
        self._sim_state = initial_state(self._model)
        self._route_score = 0
        self._board = next(iter(level.get_sprites_by_name("board")), None)
        self._sync_board()

    def _sync_board(self) -> None:
        if self._board is None:
            return
        self._board.pixels = _draw_frame(self._model, self._sim_state)

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

        raw_action = getattr(self.action, "id", SPACE_ACTION)
        if hasattr(raw_action, "value"):
            raw_action = raw_action.value
        action_id = int(raw_action)
        result = simulate_step(self._model, self._sim_state, action_id)
        self._sim_state = result.state

        if result.advance_level:
            self.next_level()
            self.complete_action()
            return
        if result.restart_level:
            self.lose()
            self.complete_action()
            return

        self._sync_board()
        self.complete_action()
