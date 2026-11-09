from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "pushonlybridges-0001"

GRID_W = 24
GRID_H = 16
PLAY_MIN_Y = 1
BAR_LEN = 22

COLOR_TIME_EMPTY = 0
COLOR_WALL = 1
COLOR_TIME_FILL = 2
COLOR_GROUND = 3
COLOR_WATER_A = 4
COLOR_WATER_B = 5
COLOR_BRIDGE = 6
COLOR_PLANK = 7
COLOR_PLANK_HIGHLIGHT = 8
COLOR_EXIT_BRIGHT = 9
COLOR_TIME_WARN = 10
COLOR_PLAYER = 11
COLOR_BRIDGE_FRESH = 12
COLOR_BOULDER = 13
COLOR_WIN_FLASH = 14

MOVE_BY_ACTION = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

PLANK_CHARS = {"%", "[", "]", "-", "^", "|", "v"}
EXIT_CHARS = {"/", "\\"}


class LevelModel(NamedTuple):
    width: int
    height: int
    time_limit: int
    walls: frozenset[tuple[int, int]]
    waters: frozenset[tuple[int, int]]
    boulders: frozenset[tuple[int, int]]
    exits: frozenset[tuple[int, int]]
    plank_lengths: tuple[int, ...]
    initial_player: tuple[int, int]
    initial_planks: tuple[tuple[tuple[int, int], ...], ...]
    initial_bridged: frozenset[tuple[int, int]]
    initial_fresh: frozenset[tuple[int, int]]


LEVEL_LAYOUTS: list[tuple[int, list[str]]] = [
    (
        80,
        [
            "#++++++++++++++++++++++#",
            "########################",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#......./\\..#",
            "#........@%~.......\\/..#",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#...........#",
            "#..........#...........#",
            "########################",
        ],
    ),
    (
        120,
        [
            "#++++++++++++++++++++++#",
            "########################",
            "#.......#......#.......#",
            "#.......#......#.......#",
            "#.......#......#.......#",
            "#.......#......#.......#",
            "#.....@%~......#.......#",
            "#.......#......#.......#",
            "#.......#......#.......#",
            "#.......#......#..../\\.#",
            "#.......#.....%~....\\/.#",
            "#.......#......#.......#",
            "#.......#......#.......#",
            "#.......#......#.......#",
            "#.......#......#.......#",
            "########################",
        ],
    ),
    (
        140,
        [
            "#++++++++++++++++++++++#",
            "########################",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##....../\\..#",
            "#.......@[]~~......\\/..#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "#..........##..........#",
            "########################",
        ],
    ),
    (
        180,
        [
            "#++++++++++++++++++++++#",
            "########################",
            "#......................#",
            "#.......@..............#",
            "#.......^..............#",
            "#.......|..............#",
            "#.......v..............#",
            "#~~~~~~~~~~~~~~~~~~~~~~#",
            "#~~~~~~~~~~~~~~~~~~~~~~#",
            "#~~~~~~~~~~~~~~~~~~~~~~#",
            "#...............#......#",
            "#...............#......#",
            "#..............%~......#",
            "#...............#.../\\.#",
            "#...............#...\\/.#",
            "########################",
        ],
    ),
    (
        200,
        [
            "#++++++++++++++++++++++#",
            "########################",
            "#.........###....#.....#",
            "#.........###....#.....#",
            "#.........###....#.....#",
            "#.........###....#.....#",
            "#...%.....###....#.....#",
            "#.........###....#.....#",
            "#....@[-].~~~....#.....#",
            "#.........###....#.....#",
            "#.........###....#.....#",
            "#.........###...%~.....#",
            "#.........###....#../\\.#",
            "#.........###....#..\\/.#",
            "#.........###....#.....#",
            "########################",
        ],
    ),
    (
        260,
        [
            "#++++++++++++++++++++++#",
            "########################",
            "#.......##.....###..#..#",
            "#.....~.##.....###..#..#",
            "#..@....##.....###..#..#",
            "#.%.....##.....###..#..#",
            "#....[].~~.....###..#..#",
            "#.......##00...###..#..#",
            "#.......##00[-]###..#..#",
            "#.......##.....###..#..#",
            "#.......##.....~~~..#..#",
            "#.......##.....###..#..#",
            "#.......##.....###..~..#",
            "#.......##.....###..#/\\#",
            "#.......##.....###..#\\/#",
            "########################",
        ],
    ),
]


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _norm_cells(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(x), int(y)) for x, y in cells))


def _detect_planks(raw_rows: list[str]) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    seen: set[tuple[int, int]] = set()
    planks: list[tuple[tuple[int, int], ...]] = []
    lengths: list[int] = []

    def neighbors(x: int, y: int):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H and raw_rows[ny][nx] in PLANK_CHARS:
                yield nx, ny

    for y in range(GRID_H):
        for x in range(GRID_W):
            if (x, y) in seen or raw_rows[y][x] not in PLANK_CHARS:
                continue
            queue = deque([(x, y)])
            seen.add((x, y))
            cells: list[tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in neighbors(cx, cy):
                    if (nx, ny) in seen:
                        continue
                    seen.add((nx, ny))
                    queue.append((nx, ny))

            xs = sorted({cx for cx, _ in cells})
            ys = sorted({cy for _, cy in cells})
            length = len(cells)
            if length not in {1, 2, 3}:
                raise ValueError(f"Unsupported plank length {length} in layout.")
            if length > 1:
                if len(xs) != 1 and len(ys) != 1:
                    raise ValueError("Plank cells must be colinear.")
                if len(xs) == 1 and ys != list(range(min(ys), max(ys) + 1)):
                    raise ValueError("Vertical plank cells must be contiguous.")
                if len(ys) == 1 and xs != list(range(min(xs), max(xs) + 1)):
                    raise ValueError("Horizontal plank cells must be contiguous.")

            planks.append(_norm_cells(cells))
            lengths.append(length)

    paired = sorted(zip(planks, lengths, strict=False), key=lambda p: (p[0][0][1], p[0][0][0]))
    if not paired:
        return (), ()
    out_planks, out_lengths = zip(*paired, strict=False)
    return tuple(out_planks), tuple(int(v) for v in out_lengths)


def _parse_level(time_limit: int, rows: list[str]) -> LevelModel:
    if len(rows) != GRID_H:
        raise ValueError("Each level must contain exactly 16 rows.")
    if any(len(row) != GRID_W for row in rows):
        raise ValueError("Each level row must contain exactly 24 columns.")

    walls: set[tuple[int, int]] = set()
    waters: set[tuple[int, int]] = set()
    boulders: set[tuple[int, int]] = set()
    exits: set[tuple[int, int]] = set()
    bridged: set[tuple[int, int]] = set()
    fresh: set[tuple[int, int]] = set()
    player: tuple[int, int] | None = None

    planks, plank_lengths = _detect_planks(rows)

    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if y == 0:
                continue
            if char == "#":
                walls.add((x, y))
            elif char == "~":
                waters.add((x, y))
            elif char == "*":
                bridged.add((x, y))
                fresh.add((x, y))
            elif char == "=":
                bridged.add((x, y))
            elif char == "0":
                boulders.add((x, y))
            elif char in EXIT_CHARS:
                exits.add((x, y))
            elif char == "@":
                player = (x, y)

    if player is None:
        raise ValueError("Each level requires exactly one player.")

    return LevelModel(
        width=GRID_W,
        height=GRID_H,
        time_limit=int(time_limit),
        walls=frozenset(walls),
        waters=frozenset(waters),
        boulders=frozenset(boulders),
        exits=frozenset(exits),
        plank_lengths=tuple(plank_lengths),
        initial_player=player,
        initial_planks=tuple(planks),
        initial_bridged=frozenset(bridged),
        initial_fresh=frozenset(fresh),
    )


def _in_bounds(model: LevelModel, x: int, y: int) -> bool:
    return 0 <= x < model.width and 0 <= y < model.height


def _is_unbridged_water(model: LevelModel, cell: tuple[int, int], bridged: frozenset[tuple[int, int]]) -> bool:
    return cell in model.waters and cell not in bridged


@lru_cache(maxsize=200000)
def _simulate_move(
    model: LevelModel,
    player: tuple[int, int],
    planks: tuple[tuple[tuple[int, int], ...], ...],
    bridged: frozenset[tuple[int, int]],
    action_id: int,
):
    if action_id not in MOVE_BY_ACTION:
        return player, planks, bridged, False, None, frozenset()

    dx, dy = MOVE_BY_ACTION[action_id]
    px, py = player
    nx, ny = px + dx, py + dy

    occupied: dict[tuple[int, int], int] = {}
    for pid, cells in enumerate(planks):
        for cell in cells:
            occupied[cell] = pid

    target = (nx, ny)
    if not _in_bounds(model, nx, ny) or target in model.walls or target in model.boulders:
        return player, planks, bridged, False, None, frozenset()

    if _is_unbridged_water(model, target, bridged):
        return player, planks, bridged, False, None, frozenset()

    pid = occupied.get(target)
    if pid is None:
        if target in model.exits:
            return target, planks, bridged, True, None, frozenset()
        return target, planks, bridged, False, None, frozenset()

    plank_cells = list(planks[pid])
    if not plank_cells:
        return player, planks, bridged, False, None, frozenset()

    lead = max(cx * dx + cy * dy for cx, cy in plank_cells)
    leading_cells = [(cx, cy) for (cx, cy) in plank_cells if cx * dx + cy * dy == lead]

    can_land_push = True
    for lx, ly in leading_cells:
        tx, ty = lx + dx, ly + dy
        tcell = (tx, ty)
        if not _in_bounds(model, tx, ty):
            can_land_push = False
            break
        if tcell in model.walls or tcell in model.boulders or tcell in model.exits:
            can_land_push = False
            break
        if _is_unbridged_water(model, tcell, bridged):
            can_land_push = False
            break
        other = occupied.get(tcell)
        if other is not None and other != pid:
            can_land_push = False
            break

    if can_land_push:
        moved_cells = _norm_cells([(cx + dx, cy + dy) for cx, cy in plank_cells])
        new_planks = list(planks)
        new_planks[pid] = moved_cells
        return (target, tuple(new_planks), bridged, target in model.exits, pid, frozenset())

    if len(leading_cells) != 1:
        return player, planks, bridged, False, None, frozenset()

    adjacent = (leading_cells[0][0] + dx, leading_cells[0][1] + dy)
    if not _is_unbridged_water(model, adjacent, bridged):
        return player, planks, bridged, False, None, frozenset()

    length = int(model.plank_lengths[pid])
    fresh_cells: list[tuple[int, int]] = []
    if length == 1:
        fresh_cells = [adjacent]
    else:
        candidate = [(adjacent[0] + i * dx, adjacent[1] + i * dy) for i in range(length)]
        if all(_is_unbridged_water(model, cell, bridged) for cell in candidate):
            fresh_cells = candidate

    if not fresh_cells:
        return player, planks, bridged, False, None, frozenset()

    new_planks = list(planks)
    new_planks[pid] = ()
    new_bridged = frozenset(set(bridged).union(fresh_cells))
    return (target, tuple(new_planks), new_bridged, target in model.exits, pid, frozenset(fresh_cells))


def _build_level(level_idx: int) -> Level:
    board = Sprite(
        _solid(GRID_W, GRID_H, COLOR_GROUND), name="board", x=0, y=0, layer=0, tags=["board"], collidable=False
    )
    return Level(
        name=f"PushOnlyBridges {level_idx + 1}",
        grid_size=(GRID_W, GRID_H),
        sprites=[board],
        data={"level_index": int(level_idx)},
    )


def _state_name(game: ARCBaseGame) -> str:
    state = getattr(game, "_state", None)
    return str(getattr(state, "name", state))


class Pushonlybridges(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._models = [_parse_level(t, rows) for t, rows in LEVEL_LAYOUTS]
        self._player = (0, 0)
        self._planks: tuple[tuple[tuple[int, int], ...], ...] = ()
        self._bridged: frozenset[tuple[int, int]] = frozenset()
        self._fresh_bridges: frozenset[tuple[int, int]] = frozenset()
        self._highlighted_planks: frozenset[int] = frozenset()
        self._time_left = 0
        self._time_limit = 1
        self._water_frame_b = False
        self._exit_pulse = 0
        self._board: Sprite | None = None
        self._solver_cache: dict[int, list[int]] = {}
        self._solvable_state_cache: dict[int, dict[tuple, bool]] = {}

        levels = [_build_level(i) for i in range(len(self._models))]
        camera = Camera(width=GRID_W, height=GRID_H, background=COLOR_GROUND)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def _model(self) -> LevelModel:
        idx = int(self.current_level.get_data("level_index") or 0)
        return self._models[idx]

    def on_set_level(self, level: Level) -> None:
        model = self._model()
        self._player = tuple(model.initial_player)
        self._planks = tuple(tuple(cells) for cells in model.initial_planks)
        self._bridged = frozenset(model.initial_bridged)
        self._fresh_bridges = frozenset(model.initial_fresh)
        self._highlighted_planks = frozenset()
        self._time_limit = max(1, int(model.time_limit))
        self._time_left = self._time_limit
        self._water_frame_b = False
        self._exit_pulse = 0
        self._board = level.get_sprites_by_name("board")[0]
        self._sync_board(fail_flash=False)

    def _is_terminal(self) -> bool:
        return _state_name(self) in {"WIN", "GAME_OVER"}

    def _apply_reset(self) -> None:
        self.level_reset()

    def _sync_board(self, *, fail_flash: bool) -> None:
        model = self._model()
        grid = np.full((GRID_H, GRID_W), COLOR_GROUND, dtype=np.int8)

        for x, y in model.walls:
            grid[y, x] = COLOR_WALL

        water_color = COLOR_WATER_B if self._water_frame_b else COLOR_WATER_A
        for x, y in model.waters:
            if (x, y) in self._bridged:
                continue
            grid[y, x] = water_color

        settled = set(self._bridged) - set(self._fresh_bridges)
        for x, y in settled:
            grid[y, x] = COLOR_BRIDGE
        for x, y in self._fresh_bridges:
            grid[y, x] = COLOR_BRIDGE_FRESH

        for x, y in model.boulders:
            grid[y, x] = COLOR_BOULDER

        pulse = (_state_name(self) == "WIN") and ((self._exit_pulse % 2) == 0)
        for x, y in model.exits:
            top_half = (x, y + 1) in model.exits
            if pulse:
                grid[y, x] = COLOR_WIN_FLASH
            elif top_half:
                grid[y, x] = COLOR_EXIT_BRIGHT
            else:
                grid[y, x] = COLOR_WATER_B

        for pid, cells in enumerate(self._planks):
            color = COLOR_PLANK_HIGHLIGHT if pid in self._highlighted_planks else COLOR_PLANK
            for x, y in cells:
                grid[y, x] = color

        px, py = self._player
        grid[py, px] = COLOR_PLAYER

        grid[0, 0] = COLOR_WALL
        grid[0, GRID_W - 1] = COLOR_WALL
        if fail_flash:
            for x in range(1, GRID_W - 1):
                grid[0, x] = COLOR_TIME_WARN
        else:
            fill = round((self._time_left / max(1, self._time_limit)) * BAR_LEN)
            fill = max(0, min(BAR_LEN, fill))
            filled_color = COLOR_TIME_WARN if self._time_left <= int(0.2 * self._time_limit) else COLOR_TIME_FILL
            for i in range(BAR_LEN):
                grid[0, i + 1] = filled_color if i < fill else COLOR_TIME_EMPTY

        if self._board is not None:
            self._board.pixels = grid

    def _step_simulation(self, action_id: int) -> bool:
        model = self._model()
        prev_fresh = set(self._fresh_bridges)
        prev_highlights = set(self._highlighted_planks)

        (next_player, next_planks, next_bridged, reached_exit, moved_pid, new_fresh) = _simulate_move(
            model=model, player=self._player, planks=self._planks, bridged=self._bridged, action_id=action_id
        )

        self._player = next_player
        self._planks = next_planks
        self._bridged = next_bridged

        self._fresh_bridges = frozenset((set(self._fresh_bridges) | set(new_fresh)) - prev_fresh)

        highlights = set()
        if moved_pid is not None:
            highlights.add(int(moved_pid))
            if moved_pid in prev_highlights:
                highlights.add(int(moved_pid))
        self._highlighted_planks = frozenset(highlights)

        self._water_frame_b = not self._water_frame_b
        self._exit_pulse += 1
        return bool(reached_exit)

    def _state_key(
        self,
        player: tuple[int, int] | None = None,
        planks: tuple[tuple[tuple[int, int], ...], ...] | None = None,
        bridged: frozenset[tuple[int, int]] | None = None,
    ) -> tuple:
        return (
            tuple(self._player if player is None else player),
            tuple(self._planks if planks is None else planks),
            frozenset(self._bridged if bridged is None else bridged),
        )

    def _has_path_to_exit(self, level_idx: int, state: tuple) -> bool:
        cache = self._solvable_state_cache.setdefault(level_idx, {})
        if state in cache:
            return bool(cache[state])

        model = self._models[level_idx]
        queue = deque([state])
        seen = {state}
        explored: list[tuple] = []
        max_explored = 2000

        while queue:
            cursor = queue.popleft()
            explored.append(cursor)
            if len(explored) >= max_explored:
                for explored_state in explored:
                    cache[explored_state] = True
                cache[state] = True
                return True
            player, planks, bridged = cursor
            for action_id in (1, 2, 3, 4):
                nxt_player, nxt_planks, nxt_bridged, reached_exit, _, _ = _simulate_move(
                    model=model, player=player, planks=planks, bridged=bridged, action_id=action_id
                )
                if reached_exit:
                    for explored_state in explored:
                        cache[explored_state] = True
                    cache[state] = True
                    return True
                nxt_state = (nxt_player, nxt_planks, nxt_bridged)
                if nxt_state == cursor or nxt_state in seen:
                    continue
                if nxt_state in cache:
                    if cache[nxt_state]:
                        for explored_state in explored:
                            cache[explored_state] = True
                        cache[state] = True
                        return True
                    continue
                seen.add(nxt_state)
                queue.append(nxt_state)

        for explored_state in explored:
            cache[explored_state] = False
        cache[state] = False
        return False

    def _compute_level_program(self, level_idx: int) -> list[int]:
        if level_idx in self._solver_cache:
            return list(self._solver_cache[level_idx])

        model = self._models[level_idx]
        start_player = tuple(model.initial_player)
        start_planks = tuple(tuple(cells) for cells in model.initial_planks)
        start_bridged = frozenset(model.initial_bridged)
        start_state = (start_player, start_planks, start_bridged)

        queue = deque([start_state])
        previous = {start_state: None}
        previous_action: dict[tuple, int] = {}
        goal_state = None

        while queue:
            state = queue.popleft()
            player, planks, bridged = state
            for action_id in (1, 2, 3, 4):
                nxt_player, nxt_planks, nxt_bridged, reached_exit, _, _ = _simulate_move(
                    model=model, player=player, planks=planks, bridged=bridged, action_id=action_id
                )
                nxt_state = (nxt_player, nxt_planks, nxt_bridged)
                if nxt_state == state:
                    continue
                if nxt_state in previous:
                    continue
                previous[nxt_state] = state
                previous_action[nxt_state] = action_id
                if reached_exit:
                    goal_state = nxt_state
                    queue.clear()
                    break
                queue.append(nxt_state)

        if goal_state is None:
            raise RuntimeError(f"No solution found for pushonlybridges level {level_idx}.")

        actions: list[int] = []
        cursor = goal_state
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]
        actions.reverse()

        self._solver_cache[level_idx] = list(actions)
        return actions

    def solver_program_for_current_level(self) -> list[tuple[int, dict[str, int]]]:
        idx = int(self.current_level.get_data("level_index") or 0)
        return [(aid, {}) for aid in self._compute_level_program(idx)]

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

        action = self.action.id
        action_id = int(getattr(action, "value", action))
        level_idx = int(self.current_level.get_data("level_index") or 0)

        if action == GameAction.ACTION5 or action == GameAction.ACTION6:
            self._apply_reset()
            self.complete_action()
            return

        if self._is_terminal():
            self.complete_action()
            return

        reached_exit = self._step_simulation(action_id)

        self._time_left -= 1
        if reached_exit:
            self.next_level()
            self.complete_action()
            return

        if not self._has_path_to_exit(level_idx, self._state_key()):
            self.lose()

        if self._time_left <= 0:
            self.lose()

        self._sync_board(fail_flash=(_state_name(self) == "GAME_OVER"))
        self.complete_action()
