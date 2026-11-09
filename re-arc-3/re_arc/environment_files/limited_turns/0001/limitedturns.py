from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "limited_turns-0001"

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_PLAYER = 3
COLOR_NOSE = 4
COLOR_EXIT_BASE = 5
COLOR_EXIT_PULSE = 6
COLOR_HAZARD = 7
COLOR_WARNING = 8
COLOR_PUSHBLOCK = 9
COLOR_SWITCH_OFF = 10
COLOR_SWITCH_ON = 11
COLOR_GATE_CLOSED = 12
COLOR_TIME_FILLED = 13
COLOR_TIME_EMPTY = 14
COLOR_TURN_PIP = 15

DIR_UP = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_RIGHT = 3
DIR_NONE = -1

DIR_DELTAS = {DIR_UP: (0, -1), DIR_DOWN: (0, 1), DIR_LEFT: (-1, 0), DIR_RIGHT: (1, 0)}

ACTION_TO_DIR = {
    int(GameAction.ACTION1.value): DIR_UP,
    int(GameAction.ACTION2.value): DIR_DOWN,
    int(GameAction.ACTION3.value): DIR_LEFT,
    int(GameAction.ACTION4.value): DIR_RIGHT,
}

SPACE_ACTION_ID = int(GameAction.ACTION5.value)

FREEZE_STEPS = 10

LEVEL_SPECS = [
    {
        "name": "Level 1",
        "turn_budget": 6,
        "time_max_steps": 140,
        "layout": [
            "============================",
            "############################",
            "#@.........................#",
            "#..........................#",
            "#########################..#",
            "#########################..#",
            "#########################..#",
            "#########################..#",
            "#########################..#",
            "#########################..#",
            "#########################..#",
            "#########################..#",
            "#########################XX#",
            "#########################XX#",
            "############################",
            "############################",
            "############################",
            "############################",
        ],
    },
    {
        "name": "Level 2",
        "turn_budget": 4,
        "time_max_steps": 180,
        "layout": [
            "============================",
            "############################",
            "#.@....###.....###.....XX..#",
            "#..###.###.###.###.####XX..#",
            "#..###.....###.....######..#",
            "#..######################..#",
            "#..######################..#",
            "#..######################..#",
            "#..######################..#",
            "#..######################..#",
            "#..######################..#",
            "#..######################..#",
            "#..######################..#",
            "#..######################..#",
            "#..........................#",
            "#..........................#",
            "############################",
            "############################",
        ],
    },
    {
        "name": "Level 3",
        "turn_budget": 4,
        "time_max_steps": 200,
        "layout": [
            "============================",
            "############################",
            "#..........................#",
            "#..........................#",
            "#............####..........#",
            "#............####..........#",
            "#..........................#",
            "#########################..#",
            "#.@..%%......^^^^^^....XX..#",
            "#....%%......^^^^^^....XX..#",
            "#..........................#",
            "#..........................#",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
        ],
    },
    {
        "name": "Level 4",
        "turn_budget": 3,
        "time_max_steps": 160,
        "layout": [
            "============================",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
            "#.@........!!....+.....XX..#",
            "#................+.....XX..#",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
        ],
    },
    {
        "name": "Level 5",
        "turn_budget": 3,
        "time_max_steps": 220,
        "layout": [
            "============================",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
            "#############..#############",
            "#############&&#############",
            "####...######&&#############",
            "#.@....................XX..#",
            "#......................XX..#",
            "#############..#############",
            "#############..#############",
            "############################",
            "############################",
            "############################",
            "############################",
            "############################",
        ],
    },
    {
        "name": "Level 6",
        "turn_budget": 6,
        "time_max_steps": 240,
        "layout": [
            "============================",
            "############################",
            "############################",
            "############################",
            "#@......%%...+..^^^^...XX..#",
            "#.......%%...+..^^^^...XX..#",
            "#####.######################",
            "#####.######################",
            "#####.######################",
            "#####.######################",
            "#####.######################",
            "#####.######################",
            "#####.######################",
            "#####...&&....!!....########",
            "#####...&&..........########",
            "#####.######################",
            "############################",
            "############################",
        ],
    },
]


@dataclass(frozen=True)
class SentrySpec:
    x: int
    y: int
    axis: str
    direction: int


@dataclass(frozen=True)
class LimitedTurnsLevelModel:
    name: str
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    spawn: tuple[int, int]
    exit_cells: frozenset[tuple[int, int]]
    spikes_initial: frozenset[tuple[int, int]]
    pushblocks_initial: tuple[tuple[int, int], ...]
    switch_cells: tuple[frozenset[tuple[int, int]], ...]
    gate_tops: tuple[tuple[int, int], ...]
    gate_cells: tuple[frozenset[tuple[int, int]], ...]
    sentries_initial: tuple[SentrySpec, ...]
    turn_budget: int
    time_max_steps: int


@dataclass(frozen=True)
class LimitedTurnsState:
    player: tuple[int, int]
    facing: int
    turns_remaining: int
    time_remaining: int
    spikes: frozenset[tuple[int, int]]
    pushblocks: tuple[tuple[int, int], ...]
    switch_mask: int
    gate_timers: tuple[int, ...]  # 0 closed, >0 opening steps left, -1 open
    sentries: tuple[tuple[int, int, int], ...]  # x, y, dir


def _rect2_cells(x: int, y: int) -> tuple[tuple[int, int], ...]:
    return ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))


def _in_play_bounds(model: LimitedTurnsLevelModel, x: int, y: int) -> bool:
    return 0 <= x < model.width and 1 <= y < model.height


def _gate_blocked_cells(state: LimitedTurnsState, model: LimitedTurnsLevelModel) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for idx, cells in enumerate(model.gate_cells):
        timer = state.gate_timers[idx]
        if timer >= 0:
            blocked.update(cells)
    return blocked


def _pushblock_cells(pushblocks: tuple[tuple[int, int], ...]) -> dict[tuple[int, int], int]:
    occupied: dict[tuple[int, int], int] = {}
    for idx, (bx, by) in enumerate(pushblocks):
        for cell in _rect2_cells(bx, by):
            occupied[cell] = idx
    return occupied


def _player_can_step(model: LimitedTurnsLevelModel, state: LimitedTurnsState, direction: int) -> bool:
    if direction not in DIR_DELTAS:
        return False

    px, py = state.player
    dx, dy = DIR_DELTAS[direction]
    tx, ty = px + dx, py + dy

    if not _in_play_bounds(model, tx, ty):
        return False
    if (tx, ty) in model.walls:
        return False
    if (tx, ty) in _gate_blocked_cells(state, model):
        return False

    occupancy = _pushblock_cells(state.pushblocks)
    block_idx = occupancy.get((tx, ty))
    if block_idx is None:
        return True

    bx, by = state.pushblocks[block_idx]
    new_bx = bx + dx
    new_by = by + dy

    new_cells = set(_rect2_cells(new_bx, new_by))
    old_cells = set(_rect2_cells(bx, by))

    for cx, cy in new_cells:
        if not _in_play_bounds(model, cx, cy):
            return False
        if (cx, cy) in model.walls:
            return False
        if (cx, cy) in _gate_blocked_cells(state, model):
            return False
        if (cx, cy) in occupancy and (cx, cy) not in old_cells:
            return False

    return True


def _try_move_player_forward(
    model: LimitedTurnsLevelModel, state: LimitedTurnsState
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...], frozenset[tuple[int, int]]]:
    if state.facing not in DIR_DELTAS:
        return state.player, state.pushblocks, state.spikes

    px, py = state.player
    dx, dy = DIR_DELTAS[state.facing]
    tx, ty = px + dx, py + dy

    if not _in_play_bounds(model, tx, ty):
        return state.player, state.pushblocks, state.spikes
    if (tx, ty) in model.walls:
        return state.player, state.pushblocks, state.spikes
    if (tx, ty) in _gate_blocked_cells(state, model):
        return state.player, state.pushblocks, state.spikes

    occupancy = _pushblock_cells(state.pushblocks)
    block_idx = occupancy.get((tx, ty))
    if block_idx is None:
        return (tx, ty), state.pushblocks, state.spikes

    bx, by = state.pushblocks[block_idx]
    new_bx = bx + dx
    new_by = by + dy

    old_cells = set(_rect2_cells(bx, by))
    new_cells = set(_rect2_cells(new_bx, new_by))

    for cx, cy in new_cells:
        if not _in_play_bounds(model, cx, cy):
            return state.player, state.pushblocks, state.spikes
        if (cx, cy) in model.walls:
            return state.player, state.pushblocks, state.spikes
        if (cx, cy) in _gate_blocked_cells(state, model):
            return state.player, state.pushblocks, state.spikes
        if (cx, cy) in occupancy and (cx, cy) not in old_cells:
            return state.player, state.pushblocks, state.spikes

    moved_blocks = list(state.pushblocks)
    moved_blocks[block_idx] = (new_bx, new_by)
    moved_blocks.sort()

    new_spikes = set(state.spikes)
    for vacated in old_cells - new_cells:
        if vacated in new_spikes:
            new_spikes.remove(vacated)

    return (tx, ty), tuple(moved_blocks), frozenset(new_spikes)


def _sentry_hits_wall(model: LimitedTurnsLevelModel, x: int, y: int) -> bool:
    for cx, cy in _rect2_cells(x, y):
        if not _in_play_bounds(model, cx, cy):
            return True
        if (cx, cy) in model.walls:
            return True
    return False


def _move_sentries(model: LimitedTurnsLevelModel, state: LimitedTurnsState) -> tuple[tuple[int, int, int], ...]:
    moved: list[tuple[int, int, int]] = []
    for idx, sentry in enumerate(state.sentries):
        sx, sy, direction = sentry
        spec = model.sentries_initial[idx]

        if spec.axis == "vertical":
            nx, ny = sx, sy + direction
        else:
            nx, ny = sx + direction, sy

        if _sentry_hits_wall(model, nx, ny):
            direction *= -1
            if spec.axis == "vertical":
                nx, ny = sx, sy + direction
            else:
                nx, ny = sx + direction, sy
            if _sentry_hits_wall(model, nx, ny):
                nx, ny = sx, sy

        moved.append((nx, ny, direction))

    return tuple(moved)


def _player_overlaps_sentry(player: tuple[int, int], sentries: tuple[tuple[int, int, int], ...]) -> bool:
    px, py = player
    return any((px, py) in _rect2_cells(sx, sy) for sx, sy, _ in sentries)


def _tick_gate_timers(gate_timers: tuple[int, ...]) -> tuple[int, ...]:
    out: list[int] = []
    for timer in gate_timers:
        if timer > 0:
            timer -= 1
            if timer == 0:
                timer = -1
        out.append(timer)
    return tuple(out)


def limited_turns_step_transition(
    model: LimitedTurnsLevelModel, state: LimitedTurnsState, action_id: int
) -> tuple[LimitedTurnsState, bool, bool, bool, bool]:
    """Return (next_state, failed, won, spent_turn_flash, switch_toggled_flash)."""

    requested_dir = ACTION_TO_DIR.get(int(action_id))
    interact = int(action_id) == SPACE_ACTION_ID

    player = state.player
    facing = state.facing
    turns = state.turns_remaining
    spikes = state.spikes
    pushblocks = state.pushblocks
    switch_mask = state.switch_mask
    gate_timers = list(state.gate_timers)
    sentries = state.sentries

    switch_toggled = False

    # 2) Interact
    if interact:
        for idx, cells in enumerate(model.switch_cells):
            if player in cells:
                switch_mask ^= 1 << idx
                switch_toggled = True
                if (switch_mask & (1 << idx)) and gate_timers:
                    for gidx, timer in enumerate(gate_timers):
                        if timer == 0:
                            gate_timers[gidx] = 2
                break

    # 3) Resolve direction change + turn spend
    spent_turn = False
    candidate_state = LimitedTurnsState(
        player=player,
        facing=facing,
        turns_remaining=turns,
        time_remaining=state.time_remaining,
        spikes=spikes,
        pushblocks=pushblocks,
        switch_mask=switch_mask,
        gate_timers=tuple(gate_timers),
        sentries=sentries,
    )
    if requested_dir is not None and _player_can_step(model, candidate_state, requested_dir):
        if facing == DIR_NONE:
            facing = requested_dir
        elif facing != requested_dir:
            facing = requested_dir
            turns -= 1
            spent_turn = True

    # 4 + 5) Move player / pushblocks
    moved_state = LimitedTurnsState(
        player=player,
        facing=facing,
        turns_remaining=turns,
        time_remaining=state.time_remaining,
        spikes=spikes,
        pushblocks=pushblocks,
        switch_mask=switch_mask,
        gate_timers=tuple(gate_timers),
        sentries=sentries,
    )
    player, pushblocks, spikes = _try_move_player_forward(model, moved_state)

    # 6) Move sentries
    sentries = _move_sentries(
        model,
        LimitedTurnsState(
            player=player,
            facing=facing,
            turns_remaining=turns,
            time_remaining=state.time_remaining,
            spikes=spikes,
            pushblocks=pushblocks,
            switch_mask=switch_mask,
            gate_timers=tuple(gate_timers),
            sentries=sentries,
        ),
    )

    # 7) Fail checks
    failed = False
    if player in spikes:
        failed = True
    if _player_overlaps_sentry(player, sentries):
        failed = True
    if spent_turn and turns <= 0:
        failed = True

    # 8) Win check
    won = player in model.exit_cells

    # 9) Time decrement
    time_remaining = state.time_remaining - 1
    if time_remaining <= 0:
        failed = True

    # 10) Anim tick side effects relevant to mechanics
    gate_timers = list(_tick_gate_timers(tuple(gate_timers)))

    next_state = LimitedTurnsState(
        player=player,
        facing=facing,
        turns_remaining=turns,
        time_remaining=time_remaining,
        spikes=spikes,
        pushblocks=pushblocks,
        switch_mask=switch_mask,
        gate_timers=tuple(gate_timers),
        sentries=sentries,
    )
    return next_state, failed, won, spent_turn, switch_toggled


def initial_state_for_model(model: LimitedTurnsLevelModel) -> LimitedTurnsState:
    return LimitedTurnsState(
        player=model.spawn,
        facing=DIR_NONE,
        turns_remaining=model.turn_budget,
        time_remaining=model.time_max_steps,
        spikes=model.spikes_initial,
        pushblocks=model.pushblocks_initial,
        switch_mask=0,
        gate_timers=tuple(0 for _ in model.gate_tops),
        sentries=tuple((spec.x, spec.y, spec.direction) for spec in model.sentries_initial),
    )


def _extract_blocks(char_grid: list[str], marker: str) -> list[tuple[int, int]]:
    h = len(char_grid)
    w = len(char_grid[0])
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []

    for y in range(1, h - 1):
        for x in range(w - 1):
            if (x, y) in seen:
                continue
            if (
                char_grid[y][x] == marker
                and char_grid[y][x + 1] == marker
                and char_grid[y + 1][x] == marker
                and char_grid[y + 1][x + 1] == marker
            ):
                out.append((x, y))
                for cell in _rect2_cells(x, y):
                    seen.add(cell)

    return out


def _infer_sentry_axis(char_grid: list[str], walls: set[tuple[int, int]], x: int, y: int) -> str:
    h = len(char_grid)
    w = len(char_grid[0])

    def passable(tx: int, ty: int) -> bool:
        for cx, cy in _rect2_cells(tx, ty):
            if cx < 0 or cy < 1 or cx >= w or cy >= h:
                return False
            if (cx, cy) in walls:
                return False
        return True

    if passable(x, y - 1) or passable(x, y + 1):
        return "vertical"
    return "horizontal"


def build_limited_turns_level_model(
    *, name: str, layout: list[str], turn_budget: int, time_max_steps: int
) -> LimitedTurnsLevelModel:
    if not layout:
        raise ValueError("limited_turns layout cannot be empty")

    width = len(layout[0])
    height = len(layout)
    if width > 64 or height > 64:
        raise ValueError("limited_turns layout must be <= 64x64")
    for row in layout:
        if len(row) != width:
            raise ValueError("limited_turns layout rows must have equal length")

    walls: set[tuple[int, int]] = set()
    spikes: set[tuple[int, int]] = set()
    exits: set[tuple[int, int]] = set()
    spawn = None

    switch_cells: list[frozenset[tuple[int, int]]] = []
    gate_tops: list[tuple[int, int]] = []

    for y, row in enumerate(layout):
        for x, ch in enumerate(row):
            if y == 0:
                continue
            if ch == "#":
                walls.add((x, y))
            elif ch == "@":
                spawn = (x, y)
            elif ch == "X":
                exits.add((x, y))
            elif ch == "^":
                spikes.add((x, y))

    if spawn is None:
        raise ValueError("limited_turns level missing @ spawn")
    if not exits:
        raise ValueError("limited_turns level missing X exit")

    for y in range(1, height):
        for x in range(width - 1):
            if layout[y][x] == "!" and layout[y][x + 1] == "!":
                switch_cells.append(frozenset({(x, y), (x + 1, y)}))

    for y in range(1, height - 1):
        for x in range(width):
            if layout[y][x] == "+" and layout[y + 1][x] == "+":
                gate_tops.append((x, y))

    pushblocks = sorted(_extract_blocks(layout, "%"))
    sentry_tops = sorted(_extract_blocks(layout, "&"))

    sentries: list[SentrySpec] = []
    for sx, sy in sentry_tops:
        axis = _infer_sentry_axis(layout, walls, sx, sy)
        sentries.append(SentrySpec(x=sx, y=sy, axis=axis, direction=1))

    gate_cells = tuple(frozenset({(x, y), (x, y + 1)}) for x, y in gate_tops)

    return LimitedTurnsLevelModel(
        name=name,
        width=width,
        height=height,
        walls=frozenset(walls),
        spawn=spawn,
        exit_cells=frozenset(exits),
        spikes_initial=frozenset(spikes),
        pushblocks_initial=tuple(pushblocks),
        switch_cells=tuple(switch_cells),
        gate_tops=tuple(gate_tops),
        gate_cells=gate_cells,
        sentries_initial=tuple(sentries),
        turn_budget=int(turn_budget),
        time_max_steps=int(time_max_steps),
    )


def plan_limited_turns_actions(model: LimitedTurnsLevelModel) -> list[int]:
    start = initial_state_for_model(model)
    queue = deque([start])
    previous: dict[LimitedTurnsState, LimitedTurnsState | None] = {start: None}
    previous_action: dict[LimitedTurnsState, int] = {}

    best_time_by_key: dict[tuple, int] = {}

    while queue:
        state = queue.popleft()

        for action_id in (1, 2, 3, 4, 5):
            nxt, failed, won, _, _ = limited_turns_step_transition(model, state, action_id)
            if failed:
                continue

            if won:
                previous[nxt] = state
                previous_action[nxt] = action_id
                actions: list[int] = []
                cursor = nxt
                while previous[cursor] is not None:
                    actions.append(previous_action[cursor])
                    cursor = previous[cursor]  # type: ignore[index]
                actions.reverse()
                return actions

            state_key = (
                nxt.player,
                nxt.facing,
                nxt.turns_remaining,
                nxt.spikes,
                nxt.pushblocks,
                nxt.switch_mask,
                nxt.gate_timers,
                nxt.sentries,
            )
            prior_best_time = best_time_by_key.get(state_key)
            if prior_best_time is not None and prior_best_time >= nxt.time_remaining:
                continue
            best_time_by_key[state_key] = nxt.time_remaining

            if nxt in previous:
                continue

            previous[nxt] = state
            previous_action[nxt] = action_id
            queue.append(nxt)

    raise RuntimeError(f"No solution found for limited_turns level: {model.name}")


def _build_level_from_spec(spec: dict) -> Level:
    layout = list(spec["layout"])
    height = len(layout)
    width = len(layout[0])

    board = np.full((height, width), COLOR_BG, dtype=np.int8)

    return Level(
        name=str(spec["name"]),
        grid_size=(width, height),
        sprites=[Sprite(pixels=board, name="board", x=0, y=0, layer=0, collidable=False, tags=["board"])],
        data={
            "name": str(spec["name"]),
            "layout": layout,
            "turn_budget": int(spec["turn_budget"]),
            "time_max_steps": int(spec["time_max_steps"]),
        },
    )


class LimitedTurns(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level_from_spec(spec) for spec in LEVEL_SPECS]
        first_w, first_h = levels[0].grid_size
        camera = Camera(0, 0, first_w, first_h, COLOR_BG, COLOR_BG, [])
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

        self._model: LimitedTurnsLevelModel | None = None
        self._lt_state: LimitedTurnsState | None = None
        self._board: Sprite | None = None

        self._anim_phase = 0
        self._turn_spend_flash = 0
        self._switch_flash = 0

        self._freeze_mode: str | None = None
        self._freeze_steps = 0

    def on_set_level(self, level: Level) -> None:
        self._model = build_limited_turns_level_model(
            name=str(level.get_data("name") or "Level"),
            layout=list(level.get_data("layout") or []),
            turn_budget=int(level.get_data("turn_budget") or 1),
            time_max_steps=int(level.get_data("time_max_steps") or 1),
        )
        self._lt_state = initial_state_for_model(self._model)
        self._board = level.get_sprites_by_name("board")[0]

        self._anim_phase = 0
        self._turn_spend_flash = 0
        self._switch_flash = 0

        self._freeze_mode = None
        self._freeze_steps = 0

        self._render_board()

    def _start_freeze(self, mode: str) -> None:
        self._freeze_mode = mode
        self._freeze_steps = FREEZE_STEPS

    def _turn_pip_slots(self, player: tuple[int, int], count: int) -> list[tuple[int, int]]:
        if self._model is None:
            return []
        px, py = player
        candidates = [
            [(px + i, py - 1) for i in range(count)],
            [(px + i, py + 1) for i in range(count)],
            [(px - 1, py + i) for i in range(count)],
            [(px + 1, py + i) for i in range(count)],
        ]

        for line in candidates:
            ok = True
            for x, y in line:
                if not (0 <= x < self._model.width and 1 <= y < self._model.height):
                    ok = False
                    break
                if (x, y) in self._model.walls:
                    ok = False
                    break
            if ok:
                return line

        return []

    def _timebar_filled(self, state: LimitedTurnsState) -> int:
        assert self._model is not None
        segment = max(1, ceil(self._model.time_max_steps / float(self._model.width)))
        return ceil(max(0, state.time_remaining) / float(segment))

    def _render_board(self) -> None:
        if self._model is None or self._lt_state is None or self._board is None:
            return

        model = self._model
        state = self._lt_state

        floor_color = COLOR_FLOOR
        if self._freeze_mode == "fail" and self._freeze_steps % 2 == 0:
            floor_color = COLOR_WARNING

        canvas = np.full((model.height, model.width), floor_color, dtype=np.int8)

        for x, y in model.walls:
            canvas[y, x] = COLOR_WALL

        hazard_color = COLOR_HAZARD if self._anim_phase == 0 else COLOR_WARNING
        for x, y in state.spikes:
            canvas[y, x] = hazard_color

        for idx, cells in enumerate(model.gate_cells):
            timer = state.gate_timers[idx]
            if timer == -1:
                continue
            if timer > 0:
                gate_color = COLOR_GATE_CLOSED if self._anim_phase == 0 else COLOR_EXIT_PULSE
            else:
                gate_color = COLOR_GATE_CLOSED
            for x, y in cells:
                canvas[y, x] = gate_color

        for idx, cells in enumerate(model.switch_cells):
            is_on = bool(state.switch_mask & (1 << idx))
            switch_color = COLOR_SWITCH_ON if is_on else COLOR_SWITCH_OFF
            if self._switch_flash > 0:
                switch_color = COLOR_EXIT_PULSE
            for x, y in cells:
                canvas[y, x] = switch_color

        for bx, by in state.pushblocks:
            for x, y in _rect2_cells(bx, by):
                canvas[y, x] = COLOR_PUSHBLOCK
            if self._anim_phase == 1:
                canvas[by, bx] = COLOR_NOSE

        for _idx, sentry in enumerate(state.sentries):
            sx, sy, _ = sentry
            for x, y in _rect2_cells(sx, sy):
                canvas[y, x] = COLOR_HAZARD
            if self._anim_phase == 1:
                canvas[sy, sx] = COLOR_WARNING

        for x, y in model.exit_cells:
            exit_color = COLOR_EXIT_BASE if ((x + y + self._anim_phase) % 2 == 0) else COLOR_EXIT_PULSE
            canvas[y, x] = exit_color

        pip_slots = self._turn_pip_slots(state.player, max(0, model.turn_budget))
        visible = max(0, min(model.turn_budget, state.turns_remaining))
        for idx, (x, y) in enumerate(pip_slots):
            if idx < visible:
                canvas[y, x] = COLOR_TURN_PIP
        if self._turn_spend_flash > 0 and visible < len(pip_slots):
            fx, fy = pip_slots[visible]
            canvas[fy, fx] = COLOR_WARNING

        px, py = state.player
        if 0 <= px < model.width and 1 <= py < model.height:
            canvas[py, px] = COLOR_PLAYER

        if state.facing in DIR_DELTAS:
            dx, dy = DIR_DELTAS[state.facing]
            nx, ny = px + dx, py + dy
            if 0 <= nx < model.width and 0 <= ny < model.height:
                canvas[ny, nx] = COLOR_NOSE

        segment = max(1, ceil(model.time_max_steps / float(model.width)))
        filled = self._timebar_filled(state)
        for x in range(model.width):
            canvas[0, x] = COLOR_TIME_FILLED if x < filled else COLOR_TIME_EMPTY

        if filled > 0 and state.time_remaining <= 5 * segment:
            warn_x = filled - 1
            canvas[0, warn_x] = COLOR_TIME_FILLED if self._anim_phase == 0 else COLOR_WARNING

        self._board.pixels = canvas

    def _update_for_freeze_step(self) -> None:
        self._freeze_steps -= 1
        self._anim_phase = 1 - self._anim_phase
        self._render_board()

        if self._freeze_steps > 0:
            return

        mode = self._freeze_mode
        self._freeze_mode = None
        self._freeze_steps = 0

        if mode == "fail":
            self.lose()
        elif mode == "win":
            self.next_level()

    def _advance_nonterminal_state(self) -> None:
        if self._turn_spend_flash > 0:
            self._turn_spend_flash -= 1
        if self._switch_flash > 0:
            self._switch_flash -= 1

        self._anim_phase = 1 - self._anim_phase
        self._render_board()

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

        if self._model is None or self._lt_state is None:
            self.complete_action()
            return

        if self._freeze_mode is not None:
            self._update_for_freeze_step()
            self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))

        next_state, failed, won, spent_turn, switch_toggled = limited_turns_step_transition(
            self._model, self._lt_state, action_id
        )
        self._lt_state = next_state

        if spent_turn:
            self._turn_spend_flash = 1
        if switch_toggled:
            self._switch_flash = 1

        # Time reaching zero overrides win because time decrement happens
        # after win check.
        if failed:
            won = False

        if won:
            self._start_freeze("win")
        elif failed:
            self._start_freeze("fail")

        self._advance_nonterminal_state()
        self.complete_action()
