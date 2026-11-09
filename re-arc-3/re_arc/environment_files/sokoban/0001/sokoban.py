from __future__ import annotations

from math import floor

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "sokoban-0001"

COLOR_WALL = 1
COLOR_FLOOR = 2
COLOR_TARGET_AURA_A = 3
COLOR_TARGET_AURA_B = 4
COLOR_TARGET_CENTER_A = 5
COLOR_TARGET_CENTER_B = 6
COLOR_CRATE = 7
COLOR_CRATE_MOVED = 8
COLOR_CRATE_DEADLOCK = 9
COLOR_CRATE_ON_TARGET = 10
COLOR_PLAYER_IDLE = 11
COLOR_PLAYER_STEP = 12
COLOR_TIMEBAR_FILLED = 13
COLOR_TIMEBAR_LOW = 14
COLOR_TIMEBAR_EMPTY = 15

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}
UNDO_ACTION_ID = int(GameAction.ACTION5.value)
RESTART_ACTION_ID = int(GameAction.ACTION6.value)

LEVEL_SPECS: list[dict[str, object]] = [
    {
        "name": "Sokoban 1",
        "time_max": 80,
        "rows": [
            "=============",
            "#############",
            "#...........#",
            "#..@.$..+...#",
            "#...........#",
            "#...........#",
            "#############",
        ],
    },
    {
        "name": "Sokoban 2",
        "time_max": 140,
        "rows": [
            "=============",
            "#############",
            "#...........#",
            "#..@........#",
            "#.....#.....#",
            "#.....#..+..#",
            "#..$..#.....#",
            "#.....#.....#",
            "#.....#.....#",
            "#...........#",
            "#############",
        ],
    },
    {
        "name": "Sokoban 3",
        "time_max": 190,
        "rows": [
            "===============",
            "###############",
            "#.............#",
            "#..@....#.....#",
            "#.......#..+..#",
            "#..$....#.....#",
            "#..$..........#",
            "#.......#.....#",
            "#.......#..+..#",
            "#.............#",
            "###############",
        ],
    },
    {
        "name": "Sokoban 4",
        "time_max": 240,
        "rows": [
            "=================",
            "#################",
            "#...............#",
            "#..@............#",
            "#....###...###..#",
            "#....#.#...#.#..#",
            "#....###...###..#",
            "#..$.$.$........#",
            "#...............#",
            "#....+.+.+......#",
            "#################",
        ],
    },
    {
        "name": "Sokoban 5",
        "time_max": 300,
        "rows": [
            "===================",
            "###################",
            "#...............###",
            "#..@............###",
            "#...............###",
            "#............$..###",
            "#............$..###",
            "#............$..###",
            "#..............+###",
            "#..............+###",
            "#..............+###",
            "#...............###",
            "###################",
        ],
    },
    {
        "name": "Sokoban 6",
        "time_max": 420,
        "rows": [
            "=======================",
            "#######################",
            "#.....................#",
            "#..@..................#",
            "#.....................#",
            "#..........**.........#",
            "#.........$$++........#",
            "#.....................#",
            "#.....................#",
            "#.....................#",
            "#....###.......###....#",
            "#....#.#.......#.#....#",
            "#....###.......###....#",
            "#.....................#",
            "#######################",
        ],
    },
]


Snapshot = tuple[
    tuple[int, int], tuple[tuple[int, int], ...], int, bool, bool, tuple[tuple[int, int], ...], tuple[int, int] | None
]


def _build_level(spec: dict[str, object]) -> Level:
    rows = [str(row) for row in spec["rows"]]
    if not rows:
        raise ValueError("Sokoban level rows must be non-empty.")
    width = len(rows[0])
    if width == 0:
        raise ValueError("Sokoban level width must be positive.")
    for row in rows:
        if len(row) != width:
            raise ValueError("Sokoban level rows must all have equal width.")
    if any(ch != "=" for ch in rows[0]):
        raise ValueError("Sokoban row 0 must be a full timebar row of '=' characters.")

    board = np.full((len(rows), width), COLOR_FLOOR, dtype=np.int8)
    sprite = Sprite(pixels=board, name="board", x=0, y=0, layer=0, collidable=False, tags=["board"])
    return Level(
        name=str(spec["name"]),
        grid_size=(width, len(rows)),
        sprites=[sprite],
        data={"rows": rows, "time_max": int(spec["time_max"])},
    )


class Sokoban(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._width = 0
        self._height = 0
        self._walls: set[tuple[int, int]] = set()
        self._targets: set[tuple[int, int]] = set()
        self._target_aura: set[tuple[int, int]] = set()
        self._player = (0, 0)
        self._crates: set[tuple[int, int]] = set()

        self._time_max = 1
        self._time_left = 1
        self._pulse_phase = False
        self._player_step_frame = False
        self._just_moved: tuple[int, int] | None = None
        self._deadlock_marked: set[tuple[int, int]] = set()

        self._history: list[Snapshot] = []
        self._initial_snapshot: Snapshot | None = None
        self._pending_action: tuple[int, dict[str, int]] | None = None

        self._board_sprite: Sprite | None = None

        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        camera_w = max(int(level.grid_size[0]) for level in levels)
        camera_h = max(int(level.grid_size[1]) for level in levels)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=Camera(width=camera_w, height=camera_h, background=0),
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    @property
    def solver_layout(self):
        return {
            "width": self._width,
            "height": self._height,
            "walls": tuple(sorted(self._walls)),
            "targets": tuple(sorted(self._targets)),
            "player": tuple(self._player),
            "crates": tuple(sorted(self._crates)),
        }

    def _snapshot(self) -> Snapshot:
        return (
            tuple(self._player),
            tuple(sorted(self._crates)),
            int(self._time_left),
            bool(self._pulse_phase),
            bool(self._player_step_frame),
            tuple(sorted(self._deadlock_marked)),
            self._just_moved,
        )

    def _restore_snapshot(self, snapshot: Snapshot, clear_history: bool = False) -> None:
        self._player = tuple(snapshot[0])
        self._crates = set(snapshot[1])
        self._time_left = int(snapshot[2])
        self._pulse_phase = bool(snapshot[3])
        self._player_step_frame = bool(snapshot[4])
        self._deadlock_marked = set(snapshot[5])
        self._just_moved = snapshot[6]
        if clear_history:
            self._history.clear()

    def _parse_level(self, level: Level) -> None:
        rows = level.get_data("rows") or []
        rows = [str(row) for row in rows]
        self._height = int(level.grid_size[1])
        self._width = int(level.grid_size[0])

        self._walls.clear()
        self._targets.clear()
        self._target_aura.clear()
        self._crates.clear()
        self._player = (0, 0)

        player_count = 0
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if y == 0:
                    continue
                if ch == "#":
                    self._walls.add((x, y))
                elif ch == "+":
                    self._targets.add((x, y))
                elif ch == "@":
                    self._player = (x, y)
                    player_count += 1
                elif ch == "$":
                    self._crates.add((x, y))
                elif ch == "*":
                    self._targets.add((x, y))
                    self._crates.add((x, y))
                elif ch == ".":
                    pass
                else:
                    raise ValueError(f"Unsupported Sokoban cell '{ch}' in level {level.name}.")

        if player_count != 1:
            raise ValueError(f"Sokoban level {level.name} must contain exactly one player.")
        if not self._targets:
            raise ValueError(f"Sokoban level {level.name} must contain at least one target.")
        if len(self._targets) != len(self._crates):
            raise ValueError(
                f"Sokoban level {level.name} must have equal crates and targets "
                f"(crates={len(self._crates)} targets={len(self._targets)})."
            )

        for tx, ty in self._targets:
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    ax, ay = tx + ox, ty + oy
                    if 0 <= ax < self._width and 1 <= ay < self._height:
                        self._target_aura.add((ax, ay))

    def on_set_level(self, level: Level) -> None:
        self._board_sprite = level.get_sprites_by_name("board")[0]
        self._parse_level(level)

        self._time_max = int(level.get_data("time_max") or 1)
        self._time_left = self._time_max
        self._pulse_phase = False
        self._player_step_frame = False
        self._just_moved = None
        self._deadlock_marked = set()
        self._history = []
        self._pending_action = None

        self._initial_snapshot = self._snapshot()
        self._render()

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self._width and 1 <= y < self._height

    def _is_wall(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or y < 1 or x >= self._width or y >= self._height:
            return True
        return pos in self._walls

    def _is_corner_deadlock(self, pos: tuple[int, int]) -> bool:
        if pos in self._targets:
            return False
        x, y = pos
        blocked_up = self._is_wall((x, y - 1))
        blocked_down = self._is_wall((x, y + 1))
        blocked_left = self._is_wall((x - 1, y))
        blocked_right = self._is_wall((x + 1, y))
        return (
            (blocked_up and blocked_left)
            or (blocked_up and blocked_right)
            or (blocked_down and blocked_left)
            or (blocked_down and blocked_right)
        )

    def _crate_deadlocks(self) -> set[tuple[int, int]]:
        return {crate for crate in self._crates if crate not in self._targets and self._is_corner_deadlock(crate)}

    def _all_targets_covered(self) -> bool:
        return all(target in self._crates for target in self._targets)

    def _apply_move(self, dx: int, dy: int) -> None:
        self._just_moved = None

        px, py = self._player
        nx, ny = px + dx, py + dy
        next_pos = (nx, ny)
        if self._is_wall(next_pos):
            return

        moved = False
        if next_pos in self._crates:
            bx, by = nx + dx, ny + dy
            beyond = (bx, by)
            if self._is_wall(beyond) or beyond in self._crates:
                return
            self._crates.remove(next_pos)
            self._crates.add(beyond)
            self._just_moved = beyond
            moved = True

        self._player = next_pos
        moved = True
        if moved:
            self._player_step_frame = not self._player_step_frame

    def _advance_normal_step(self, apply_action: tuple[int, dict[str, int]] | None) -> None:
        self._history.append(self._snapshot())

        if apply_action is not None:
            action_id = int(apply_action[0])
            delta = MOVE_DELTAS.get(action_id)
            if delta is not None:
                self._apply_move(delta[0], delta[1])

        current_deadlocks = self._crate_deadlocks()
        persisted = current_deadlocks & self._deadlock_marked

        self._pulse_phase = not self._pulse_phase
        self._time_left = max(0, self._time_left - 1)

        state_name = getattr(getattr(self, "_state", None), "name", "")
        if self._all_targets_covered() and state_name != "GAME_OVER":
            self.next_level()
            return

        if self._time_left == 0 and state_name != "WIN":
            self.lose()
            return

        if persisted and state_name != "WIN":
            self.lose()
            return

        self._deadlock_marked = current_deadlocks

    def _undo(self) -> None:
        if not self._history:
            return
        snapshot = self._history.pop()
        self._restore_snapshot(snapshot)

    def _restart(self) -> None:
        if self._initial_snapshot is None:
            return
        self._restore_snapshot(self._initial_snapshot, clear_history=True)

    def _render(self) -> None:
        if self._board_sprite is None:
            return

        grid = np.full((self._height, self._width), COLOR_FLOOR, dtype=np.int8)

        ratio = self._time_left / float(max(1, self._time_max))
        filled_cells = floor(ratio * self._width)
        filled_cells = max(0, min(self._width, filled_cells))
        low_time = ratio <= 0.2
        time_fill = COLOR_TIMEBAR_LOW if (low_time and self._pulse_phase) else COLOR_TIMEBAR_FILLED

        for x in range(self._width):
            grid[0, x] = np.int8(time_fill if x < filled_cells else COLOR_TIMEBAR_EMPTY)

        crate_lookup = set(self._crates)
        px, py = self._player

        for y in range(1, self._height):
            for x in range(self._width):
                pos = (x, y)
                if pos in self._walls:
                    color = COLOR_WALL
                elif pos in self._targets:
                    color = COLOR_TARGET_CENTER_B if self._pulse_phase else COLOR_TARGET_CENTER_A
                elif pos in self._target_aura:
                    color = COLOR_TARGET_AURA_B if self._pulse_phase else COLOR_TARGET_AURA_A
                else:
                    color = COLOR_FLOOR

                if pos in crate_lookup:
                    if pos in self._targets:
                        color = COLOR_CRATE_ON_TARGET
                    elif pos in self._deadlock_marked:
                        color = COLOR_CRATE_DEADLOCK
                    elif self._just_moved is not None and pos == self._just_moved:
                        color = COLOR_CRATE_MOVED
                    else:
                        color = COLOR_CRATE
                elif (x, y) == (px, py):
                    color = COLOR_PLAYER_STEP if self._player_step_frame else COLOR_PLAYER_IDLE

                grid[y, x] = np.int8(color)

        self._board_sprite.pixels = grid

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
        action_data = self.action.data if isinstance(self.action.data, dict) else {}
        current_input = (action_id, {str(k): int(v) for k, v in action_data.items() if isinstance(v, (int, float))})

        apply_action = self._pending_action
        self._pending_action = current_input

        state_name = getattr(getattr(self, "_state", None), "name", "")
        if state_name in {"WIN", "GAME_OVER"}:
            self.complete_action()
            return

        self._just_moved = None

        if apply_action is not None:
            apply_action_id = int(apply_action[0])
            if apply_action_id == UNDO_ACTION_ID:
                self._undo()
                self._render()
                self.complete_action()
                return
            if apply_action_id == RESTART_ACTION_ID:
                self._restart()
                self._render()
                self.complete_action()
                return

        self._advance_normal_step(apply_action)
        self._render()
        self.complete_action()
