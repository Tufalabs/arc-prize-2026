from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

VIEW_SIZE = 64
BOARD_LEFT = 4
BOARD_TOP = 6
CELL_SIZE = 5

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_RAIL = 3
COLOR_WALL = 4
COLOR_BLACK = 5
COLOR_RED = 8
COLOR_PLAYER = 9
COLOR_EXIT_OPEN = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_EXIT_CLOSED = 13
COLOR_GREEN = 14

BEACON_MASK = np.array(
    [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=bool
)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    baseline_actions: int

    @property
    def budget(self) -> int:
        return self.baseline_actions * 6


LEVEL_SPECS = (
    LevelSpec(
        "First Hand-Off",
        (
            "###########",
            "###########",
            "#######X###",
            "#######.###",
            "#######.###",
            "#S.A...B###",
            "###########",
            "###########",
            "###########",
            "###########",
            "###########",
        ),
        9,
    ),
    LevelSpec(
        "The Side Is the Instruction",
        (
            "###########",
            "###########",
            "##..B...C##",
            "##.#####.##",
            "##...###.##",
            "###SA###X##",
            "###..######",
            "###########",
            "###########",
            "###########",
            "###########",
        ),
        17,
    ),
    LevelSpec(
        "Branch Before Route",
        (
            "###########",
            "###X#######",
            "###.#######",
            "##.C...D.##",
            "##.#####.##",
            "##.#####.##",
            "##.#####.##",
            "##.AS##B.##",
            "##......###",
            "###########",
            "###########",
        ),
        23,
    ),
    LevelSpec(
        "Signals Pass Through History",
        (
            "###########",
            "######...##",
            "######C#.##",
            "######.#.##",
            "###X##.#.##",
            "###.##.#.##",
            "#.S.##.#.##",
            "#.A.##B..##",
            "##......###",
            "###E..D.###",
            "###########",
        ),
        37,
    ),
    LevelSpec(
        "Save the Awkward Beacon",
        (
            "###....####",
            "###.##.####",
            "###E##..D..",
            "#X..######.",
            "###F##..C..",
            "######.#...",
            "######.###.",
            "######.###.",
            "##.A##..B#.",
            "##.S.###...",
            "####.....##",
        ),
        42,
    ),
    LevelSpec(
        "Return Beam",
        (
            "###########",
            "#..X#######",
            "#.####....#",
            "#G##..C..D.",
            "#.##.#.##..",
            "#.##.#.##..",
            "#..#...##..",
            "##..S#.##..",
            "#F.A..B..E.",
            "#..#...####",
            "###########",
        ),
        51,
    ),
    LevelSpec(
        "Crossed Relay",
        (
            "##.....X###",
            "##.###..###",
            "#.E..D..H##",
            "..###.##.##",
            ".#.####..##",
            "..F##C..G##",
            "....#....##",
            "###..##..##",
            "#.A..B...##",
            "#..S#.#####",
            "###...#####",
        ),
        61,
    ),
)


def _cell_center(cell: tuple[int, int]) -> tuple[int, int]:
    return BOARD_LEFT + cell[0] * CELL_SIZE + 2, BOARD_TOP + cell[1] * CELL_SIZE + 2


def _cell_top_left(cell: tuple[int, int]) -> tuple[int, int]:
    return BOARD_LEFT + cell[0] * CELL_SIZE, BOARD_TOP + cell[1] * CELL_SIZE


def _solid_sprite() -> Sprite:
    return Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BLACK, dtype=np.int8), name="canvas", layer=-20, collidable=False
    )


def _build_level(spec: LevelSpec) -> Level:
    if len(spec.rows) != 11 or any(len(row) != 11 for row in spec.rows):
        raise ValueError(f"{spec.name} must be an 11x11 logical board.")
    starts = [(x, y) for y, row in enumerate(spec.rows) for x, char in enumerate(row) if char == "S"]
    exits = [(x, y) for y, row in enumerate(spec.rows) for x, char in enumerate(row) if char == "X"]
    beacons = {char: (x, y) for y, row in enumerate(spec.rows) for x, char in enumerate(row) if char in "ABCDEFGH"}
    if len(starts) != 1 or len(exits) != 1 or "A" not in beacons:
        raise ValueError(f"{spec.name} must have one S, one X, and an initial A beacon.")
    labels = tuple(sorted(beacons))
    return Level(
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        sprites=[_solid_sprite()],
        name=spec.name,
        data={
            "rows": spec.rows,
            "start": starts[0],
            "exit": exits[0],
            "beacon_labels": labels,
            "beacons": tuple(beacons[label] for label in labels),
            "budget": spec.budget,
        },
    )


class BeaconRelayView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: BeaconRelay | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BLACK
        self._draw_budget(frame, game)
        self._draw_board(frame, game)
        self._draw_rails(frame, game)
        self._draw_exit(frame, game)
        self._draw_beacons(frame, game)
        self._draw_active_ports(frame, game)
        self._draw_transient(frame, game)
        self._draw_player(frame, game.player_pos)
        if game.completed:
            frame[0, :] = COLOR_GREEN
            frame[-1, :] = COLOR_GREEN
            frame[:, 0] = COLOR_GREEN
            frame[:, -1] = COLOR_GREEN
        elif game.failed:
            frame[0, :] = COLOR_RED
            frame[-1, :] = COLOR_RED
            frame[:, 0] = COLOR_RED
            frame[:, -1] = COLOR_RED
        return frame

    def _draw_budget(self, frame: np.ndarray, game: BeaconRelay) -> None:
        frame[2:4, 4:59] = COLOR_RAIL
        filled = round(55 * game.remaining_steps / max(1, game.step_budget))
        if filled > 0:
            color = COLOR_GREEN if filled > 18 else COLOR_ORANGE if filled > 7 else COLOR_RED
            frame[2:4, 4 : 4 + filled] = color

    def _draw_board(self, frame: np.ndarray, game: BeaconRelay) -> None:
        for y, row in enumerate(game.rows):
            for x, char in enumerate(row):
                x0, y0 = _cell_top_left((x, y))
                frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = COLOR_WALL if char == "#" else COLOR_FLOOR

    def _draw_rails(self, frame: np.ndarray, game: BeaconRelay) -> None:
        by_row: dict[int, list[int]] = {}
        by_col: dict[int, list[int]] = {}
        for x, y in game.beacons:
            by_row.setdefault(y, []).append(x)
            by_col.setdefault(x, []).append(y)
        for y, xs in by_row.items():
            ordered = sorted(xs)
            for left, right in pairwise(ordered):
                x1, py = _cell_center((left, y))
                x2, _ = _cell_center((right, y))
                frame[py, x1 : x2 + 1] = COLOR_RAIL
        for x, ys in by_col.items():
            ordered = sorted(ys)
            for top, bottom in pairwise(ordered):
                px, y1 = _cell_center((x, top))
                _, y2 = _cell_center((x, bottom))
                frame[y1 : y2 + 1, px] = COLOR_RAIL

    def _draw_exit(self, frame: np.ndarray, game: BeaconRelay) -> None:
        x0, y0 = _cell_top_left(game.exit_cell)
        phase = 3 if game.exit_open else 0
        if game.transient and game.transient.get("kind") == "exit":
            phase = int(game.transient["phase"])
        frame[y0 : y0 + 5, x0 : x0 + 5] = COLOR_FLOOR
        if phase == 0:
            frame[y0 : y0 + 5, x0 : x0 + 5] = COLOR_EXIT_CLOSED
            frame[y0 + 1 : y0 + 5 : 2, x0 : x0 + 5] = COLOR_BLACK
        elif phase == 1:
            frame[y0 : y0 + 5, x0 : x0 + 2] = COLOR_EXIT_CLOSED
            frame[y0 : y0 + 5, x0 + 3 : x0 + 5] = COLOR_EXIT_CLOSED
        elif phase == 2:
            frame[y0 : y0 + 5, x0] = COLOR_EXIT_OPEN
            frame[y0 : y0 + 5, x0 + 4] = COLOR_EXIT_OPEN
            frame[y0, x0 + 1 : x0 + 4] = COLOR_EXIT_OPEN
        else:
            frame[y0 : y0 + 5, x0] = COLOR_EXIT_OPEN
            frame[y0 : y0 + 5, x0 + 4] = COLOR_EXIT_OPEN

    def _draw_beacons(self, frame: np.ndarray, game: BeaconRelay) -> None:
        hidden_active = bool(game.transient and game.transient.get("hide_active"))
        for index, cell in enumerate(game.beacons):
            if game.lit_mask[index]:
                self._paint_beacon(frame, cell, COLOR_GREEN, filled=True)
            elif index == game.active_beacon and not hidden_active:
                self._paint_active(frame, cell, game.pulse_phase)
            else:
                self._paint_beacon(frame, cell, COLOR_YELLOW, filled=False)

    def _paint_beacon(self, frame: np.ndarray, cell: tuple[int, int], color: int, *, filled: bool) -> None:
        x0, y0 = _cell_top_left(cell)
        mask = BEACON_MASK.copy()
        if not filled:
            mask[2, 2] = False
        patch = frame[y0 : y0 + 5, x0 : x0 + 5]
        patch[mask] = color

    def _paint_active(self, frame: np.ndarray, cell: tuple[int, int], phase: int) -> None:
        primary, accent = (COLOR_ORANGE, COLOR_YELLOW) if phase == 0 else (COLOR_YELLOW, COLOR_ORANGE)
        self._paint_beacon(frame, cell, primary, filled=True)
        x0, y0 = _cell_top_left(cell)
        frame[y0 + 2, x0 + 2] = accent
        frame[y0 + 1, x0 + 2] = accent

    def _draw_active_ports(self, frame: np.ndarray, game: BeaconRelay) -> None:
        if game.active_beacon is None:
            return
        source_index = game.active_beacon
        cx, cy = _cell_center(game.beacons[source_index])
        final_beacon = sum(not lit for lit in game.lit_mask) == 1
        for direction in MOVE_DELTAS.values():
            dx, dy = direction
            valid = final_beacon or game._candidate(source_index, direction) is not None
            color = COLOR_GREEN if valid else COLOR_RED
            frame[cy + 3 * dy, cx + 3 * dx] = color
            frame[cy + 2 * dy, cx + 2 * dx] = color

    def _draw_transient(self, frame: np.ndarray, game: BeaconRelay) -> None:
        event = game.transient
        if not event:
            return
        kind = event.get("kind")
        if kind in {"beam", "reject"}:
            source = _cell_center(event["source"])
            endpoint = event.get("endpoint")
            if endpoint is not None:
                self._draw_line(frame, source, endpoint, COLOR_YELLOW if kind == "beam" else COLOR_RED)
            flash = event.get("flash")
            if flash is not None:
                self._paint_beacon(frame, game.beacons[int(flash)], COLOR_WHITE, filled=True)
        elif kind == "stub":
            cx, cy = _cell_center(event["source"])
            dx, dy = event["direction"]
            frame[cy + dy, cx + dx] = COLOR_RED
            frame[cy + 2 * dy, cx + 2 * dx] = COLOR_RED
        elif kind == "wall":
            x0, y0 = _cell_top_left(event["cell"])
            frame[y0 : y0 + 5, x0 : x0 + 5] = COLOR_WHITE

    def _draw_line(self, frame: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        frame[y1 : y2 + 1, x1 : x2 + 1] = color

    def _draw_player(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        cx, cy = _cell_center(cell)
        frame[cy - 1, cx] = COLOR_PLAYER
        frame[cy, cx - 1 : cx + 2] = COLOR_PLAYER
        frame[cy + 1, cx] = COLOR_PLAYER
        frame[cy, cx] = COLOR_WHITE


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


class BeaconRelay(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = BeaconRelayView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "beacon_relay-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BLACK, COLOR_BLACK, [self._view]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
            seed,
        )
        self._view.game = self

    def on_set_level(self, _level: Level) -> None:
        self.rows = tuple(self.current_level.get_data("rows"))
        self.start_pos = tuple(self.current_level.get_data("start"))
        self.exit_cell = tuple(self.current_level.get_data("exit"))
        self.beacons = tuple(tuple(cell) for cell in self.current_level.get_data("beacons"))
        self.beacon_labels = tuple(self.current_level.get_data("beacon_labels"))
        self.beacon_at = {cell: index for index, cell in enumerate(self.beacons)}
        self.step_budget = int(self.current_level.get_data("budget"))
        self._restore_authored_state()

    def _restore_authored_state(self) -> None:
        self.player_pos = self.start_pos
        self.active_beacon: int | None = 0
        self.lit_mask = [False] * len(self.beacons)
        self.exit_open = False
        self.completed = False
        self.failed = False
        self.pulse_phase = 0
        self.remaining_steps = self.step_budget
        self.transient: dict[str, object] | None = None
        self.animation_frames: list[dict[str, object] | None] = []

    def _candidate(self, source_index: int, direction: tuple[int, int]) -> int | None:
        sx, sy = self.beacons[source_index]
        dx, dy = direction
        candidates: list[tuple[int, int]] = []
        for index, (x, y) in enumerate(self.beacons):
            if index == source_index or self.lit_mask[index]:
                continue
            distance = (x - sx) * dx + (y - sy) * dy
            if distance > 0 and ((dx == 0 and x == sx) or (dy == 0 and y == sy)):
                candidates.append((distance, index))
        return min(candidates)[1] if candidates else None

    def _rail_end(self, source_index: int, direction: tuple[int, int]) -> tuple[int, int] | None:
        sx, sy = self.beacons[source_index]
        dx, dy = direction
        cells = []
        for x, y in self.beacons:
            distance = (x - sx) * dx + (y - sy) * dy
            if distance > 0 and ((dx == 0 and x == sx) or (dy == 0 and y == sy)):
                cells.append((distance, (x, y)))
        return max(cells)[1] if cells else None

    def _beam_event(
        self,
        kind: str,
        source: tuple[int, int],
        direction: tuple[int, int],
        distance: int,
        *,
        hide_active: bool = False,
    ) -> dict[str, object]:
        dx, dy = direction
        endpoint_cell = (source[0] + dx * distance, source[1] + dy * distance)
        flash = self.beacon_at.get(endpoint_cell) if kind == "beam" else None
        if flash is not None and not self.lit_mask[flash]:
            flash = None
        return {
            "kind": kind,
            "source": source,
            "endpoint": _cell_center(endpoint_cell),
            "flash": flash,
            "hide_active": hide_active,
        }

    def _queue_rejection(self, source_index: int, direction: tuple[int, int]) -> None:
        source = self.beacons[source_index]
        rail_end = self._rail_end(source_index, direction)
        if rail_end is None:
            self.animation_frames = [{"kind": "stub", "source": source, "direction": direction}, None]
            return
        distance = abs(rail_end[0] - source[0]) + abs(rail_end[1] - source[1])
        outward = [self._beam_event("reject", source, direction, step) for step in range(1, distance + 1)]
        inward = [self._beam_event("reject", source, direction, step) for step in range(distance - 1, 0, -1)]
        self.animation_frames = [*outward, *inward, None]

    def _queue_accepted(self, source_index: int, target_index: int, direction: tuple[int, int]) -> None:
        source = self.beacons[source_index]
        target = self.beacons[target_index]
        distance = abs(target[0] - source[0]) + abs(target[1] - source[1])
        self.animation_frames = [
            {"kind": "beam", "source": source, "endpoint": None, "hide_active": True},
            *[self._beam_event("beam", source, direction, step, hide_active=True) for step in range(1, distance + 1)],
            None,
        ]

    def _queue_final(self, source: tuple[int, int]) -> None:
        self.animation_frames = [
            {"kind": "exit", "source": source, "phase": 0},
            {"kind": "exit", "source": source, "phase": 1},
            {"kind": "exit", "source": source, "phase": 2},
            None,
        ]

    def _resolve_move(self, direction: tuple[int, int]) -> None:
        dx, dy = direction
        destination = (self.player_pos[0] + dx, self.player_pos[1] + dy)
        char = "#"
        if 0 <= destination[0] < 11 and 0 <= destination[1] < 11:
            char = self.rows[destination[1]][destination[0]]
        if char == "#" or (destination == self.exit_cell and not self.exit_open):
            self.animation_frames = [{"kind": "wall", "cell": destination}, None]
            return

        beacon_index = self.beacon_at.get(destination)
        if self.active_beacon is None or beacon_index != self.active_beacon:
            self.player_pos = destination
            if destination == self.exit_cell and self.exit_open:
                self.completed = True
                self.next_level()
            return

        assert beacon_index is not None
        remaining = sum(not lit for lit in self.lit_mask)
        if remaining == 1:
            self.player_pos = destination
            self.lit_mask[beacon_index] = True
            self.active_beacon = None
            self.exit_open = True
            self._queue_final(destination)
            return

        target = self._candidate(beacon_index, direction)
        if target is None:
            self._queue_rejection(beacon_index, direction)
            return
        self.player_pos = destination
        self.lit_mask[beacon_index] = True
        self.active_beacon = target
        self._queue_accepted(beacon_index, target, direction)

    def _spend_step(self) -> None:
        self.remaining_steps -= 1
        if self.completed:
            return
        if self.remaining_steps <= 0:
            self.remaining_steps = 0
            self.animation_frames.clear()
            self.transient = None
            self.failed = True
            self.lose()

    def _play_animation_frame(self) -> None:
        self.transient = self.animation_frames.pop(0)
        if not self.animation_frames:
            self.complete_action()

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        self.transient = None
        if action == 0:
            self.complete_action()
            return
        if action == ACTION_SPACE:
            self._restore_authored_state()
            self.complete_action()
            return
        if action == ACTION_CLICK:
            self.complete_action()
            return
        if action in MOVE_DELTAS:
            self.pulse_phase = 1 - self.pulse_phase
            self._resolve_move(MOVE_DELTAS[action])
            self._spend_step()
            if self.animation_frames:
                self._play_animation_frame()
                return
        if not self.animation_frames:
            self.complete_action()
