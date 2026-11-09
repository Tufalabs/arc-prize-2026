from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

VIEW_SIZE = 64
CELL_SIZE = 5

COLOR_OUTSIDE = 0
COLOR_WALL = 4
COLOR_AGENT = 6
COLOR_BUMP = 7
COLOR_FAILURE = 8
COLOR_SCRAPE = 9
COLOR_ICE = 10
COLOR_CRATE_HIGHLIGHT = 11
COLOR_CRATE = 12
COLOR_SOCKET = 13
COLOR_EXIT = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    baseline_actions: int

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def budget(self) -> int:
        return self.baseline_actions * 6


LEVEL_SPECS = (
    LevelSpec("Cross, Then Stop", ("#######", "#A.X..#", "#.....#", "#.....#", "#.....#", "#.#...#", "#######"), 4),
    LevelSpec("The Vacated Square", ("#######", "#A.C..#", "##....#", "#...#.#", "#..#..#", "#...X##", "#######"), 5),
    LevelSpec("Clamp and Return", ("########", "#A.CSX.#", "#......#", "#......#", "#......#", "########"), 5),
    LevelSpec(
        "Two Brakes",
        (
            "#########",
            "#.A..X..#",
            "#..C###.#",
            "##......#",
            "##C.....#",
            "#...#...#",
            "#.#.....#",
            "#......##",
            "#########",
        ),
        25,
    ),
    LevelSpec(
        "Borrowed Backstop",
        (
            "##########",
            "#....#...#",
            "#.C....#.#",
            "#..#..X.A#",
            "#...#....#",
            "#....C#..#",
            "##.......#",
            "##...C.#.#",
            "#..#..#..#",
            "##########",
        ),
        44,
    ),
    LevelSpec(
        "Spend It Last",
        (
            "###########",
            "#..S......#",
            "#.........#",
            "#...C.....#",
            "##.#.##...#",
            "#.......###",
            "#.#.#.....#",
            "#......X..#",
            "#.C.C.#...#",
            "##.A#...#.#",
            "###########",
        ),
        52,
    ),
    LevelSpec(
        "Build the Rail",
        (
            "###########",
            "#..S..X...#",
            "#.....#...#",
            "#..#C#....#",
            "##.#.##...#",
            "#.......###",
            "#.#C#.....#",
            "#.........#",
            "#...C.#...#",
            "##.A#...#.#",
            "###########",
        ),
        62,
    ),
)


def _parse_rows(rows: tuple[str, ...]) -> dict[str, object]:
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Anchor Ice level rows must be rectangular")

    walls: set[tuple[int, int]] = set()
    crates: set[tuple[int, int]] = set()
    sockets: set[tuple[int, int]] = set()
    agent: tuple[int, int] | None = None
    exit_cell: tuple[int, int] | None = None
    allowed = {"#", ".", "A", "C", "S", "X"}
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            if symbol not in allowed:
                raise ValueError(f"Unknown Anchor Ice layout symbol: {symbol!r}")
            cell = (x, y)
            if symbol == "#":
                walls.add(cell)
            elif symbol == "A":
                agent = cell
            elif symbol == "C":
                crates.add(cell)
            elif symbol == "S":
                sockets.add(cell)
            elif symbol == "X":
                exit_cell = cell

    if agent is None or exit_cell is None:
        raise ValueError("Anchor Ice levels require exactly one agent and exit")
    height = len(rows)
    perimeter = {
        *((x, 0) for x in range(width)),
        *((x, height - 1) for x in range(width)),
        *((0, y) for y in range(height)),
        *((width - 1, y) for y in range(height)),
    }
    if not perimeter <= walls:
        raise ValueError("Anchor Ice levels require a complete wall perimeter")
    return {
        "width": width,
        "height": height,
        "walls": walls,
        "initial_agent": agent,
        "initial_crates": crates,
        "sockets": sockets,
        "exit": exit_cell,
    }


def _build_level(spec: LevelSpec) -> Level:
    data = _parse_rows(spec.rows)
    data.update({"rows": spec.rows, "budget": spec.budget})
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_OUTSIDE, dtype=np.int8),
        name="anchor_ice_board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level([board], grid_size=(VIEW_SIZE, VIEW_SIZE), data=data, name=spec.name)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


class AnchorIce(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "anchor_ice",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_OUTSIDE, COLOR_OUTSIDE, []),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK, ACTION_UNDO],
            seed,
        )

    def on_set_level(self, _level: Level) -> None:
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.width = int(self.current_level.get_data("width"))
        self.height = int(self.current_level.get_data("height"))
        self.walls = set(self.current_level.get_data("walls"))
        self.sockets = set(self.current_level.get_data("sockets"))
        self.exit = tuple(self.current_level.get_data("exit"))
        self.initial_agent = tuple(self.current_level.get_data("initial_agent"))
        self.initial_crates = set(self.current_level.get_data("initial_crates"))
        self.step_budget = int(self.current_level.get_data("budget"))
        self._reset_runtime()

    def _reset_runtime(self) -> None:
        self.agent = self.initial_agent
        self.loose_crates = set(self.initial_crates)
        self.anchored_crates: set[tuple[int, int]] = set()
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[tuple[int, int], frozenset[tuple[int, int]], frozenset[tuple[int, int]]]] = []
        self.animation_frames: list[dict[str, object]] = []
        self._pending_result: str | None = None
        self._visual_state = self._settled_visual()
        self._sync_board()

    def _settled_visual(self, **extras: object) -> dict[str, object]:
        visual: dict[str, object] = {
            "agent": self.agent,
            "loose": frozenset(self.loose_crates),
            "anchored": frozenset(self.anchored_crates),
        }
        visual.update(extras)
        return visual

    def _snapshot(self) -> tuple[tuple[int, int], frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
        return self.agent, frozenset(self.loose_crates), frozenset(self.anchored_crates)

    def _restore_snapshot(
        self, snapshot: tuple[tuple[int, int], frozenset[tuple[int, int]], frozenset[tuple[int, int]]]
    ) -> None:
        self.agent = snapshot[0]
        self.loose_crates = set(snapshot[1])
        self.anchored_crates = set(snapshot[2])

    def _origin(self) -> tuple[int, int]:
        return (VIEW_SIZE - CELL_SIZE * self.width) // 2, (VIEW_SIZE - CELL_SIZE * self.height) // 2

    def _cell_top_left(self, cell: tuple[int, int]) -> tuple[int, int]:
        origin_x, origin_y = self._origin()
        return origin_x + cell[0] * CELL_SIZE, origin_y + cell[1] * CELL_SIZE

    def _stamp(self, frame: np.ndarray, cell: tuple[int, int], pixels: np.ndarray) -> None:
        x, y = self._cell_top_left(cell)
        target = frame[y : y + CELL_SIZE, x : x + CELL_SIZE]
        mask = pixels >= 0
        target[mask] = pixels[mask]

    def _draw_exit(self, frame: np.ndarray, *, inverted: bool = False) -> None:
        green = COLOR_OUTSIDE if inverted else COLOR_EXIT
        white = COLOR_EXIT if inverted else COLOR_OUTSIDE
        pixels = np.array(
            [
                [-1, green, green, green, -1],
                [green, -1, -1, -1, green],
                [green, -1, white, -1, green],
                [green, -1, -1, -1, green],
                [-1, green, green, green, -1],
            ],
            dtype=np.int8,
        )
        self._stamp(frame, self.exit, pixels)

    def _draw_socket(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        self._stamp(
            frame,
            cell,
            np.array(
                [
                    [-1, COLOR_SOCKET, -1, COLOR_SOCKET, -1],
                    [COLOR_SOCKET, COLOR_SOCKET, -1, COLOR_SOCKET, COLOR_SOCKET],
                    [-1, -1, -1, -1, -1],
                    [COLOR_SOCKET, COLOR_SOCKET, -1, COLOR_SOCKET, COLOR_SOCKET],
                    [-1, COLOR_SOCKET, -1, COLOR_SOCKET, -1],
                ],
                dtype=np.int8,
            ),
        )

    def _draw_loose_crate(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        self._stamp(
            frame,
            cell,
            np.array(
                [
                    [-1, -1, -1, -1, -1],
                    [-1, COLOR_CRATE, COLOR_CRATE, COLOR_CRATE, -1],
                    [-1, COLOR_CRATE, COLOR_CRATE_HIGHLIGHT, COLOR_CRATE, COLOR_WALL],
                    [-1, COLOR_CRATE, COLOR_CRATE, COLOR_CRATE, COLOR_WALL],
                    [-1, -1, COLOR_WALL, COLOR_WALL, -1],
                ],
                dtype=np.int8,
            ),
        )

    def _draw_anchored_crate(self, frame: np.ndarray, cell: tuple[int, int], phase: int = 2) -> None:
        pixels = np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, COLOR_CRATE, COLOR_CRATE, COLOR_CRATE, -1],
                [-1, COLOR_CRATE, COLOR_CRATE_HIGHLIGHT, COLOR_CRATE, -1],
                [-1, COLOR_CRATE, COLOR_CRATE, COLOR_CRATE, -1],
                [-1, -1, -1, -1, -1],
            ],
            dtype=np.int8,
        )
        if phase >= 1:
            pixels[0, 1] = pixels[0, 3] = COLOR_SOCKET
            pixels[4, 1] = pixels[4, 3] = COLOR_SOCKET
        if phase >= 2:
            pixels[1, 0] = pixels[1, 4] = COLOR_SOCKET
            pixels[3, 0] = pixels[3, 4] = COLOR_SOCKET
        self._stamp(frame, cell, pixels)

    def _draw_agent(self, frame: np.ndarray, cell: tuple[int, int], pixel_offset: tuple[int, int]) -> None:
        x, y = self._cell_top_left(cell)
        dx, dy = pixel_offset
        pixels = np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1, -1, COLOR_AGENT, -1, -1],
                [-1, COLOR_AGENT, COLOR_AGENT, COLOR_AGENT, -1],
                [-1, -1, COLOR_AGENT, -1, -1],
                [-1, -1, -1, -1, -1],
            ],
            dtype=np.int8,
        )
        for py in range(CELL_SIZE):
            for px in range(CELL_SIZE):
                color = int(pixels[py, px])
                tx, ty = x + px + dx, y + py + dy
                if color >= 0 and 0 <= tx < VIEW_SIZE and 0 <= ty < VIEW_SIZE:
                    frame[ty, tx] = COLOR_BUMP if pixel_offset != (0, 0) else color

    def _render_visual(self, visual: dict[str, object]) -> np.ndarray:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_OUTSIDE, dtype=np.int8)
        origin_x, origin_y = self._origin()
        frame[origin_y : origin_y + self.height * CELL_SIZE, origin_x : origin_x + self.width * CELL_SIZE] = COLOR_ICE
        for cell in self.walls:
            x, y = self._cell_top_left(cell)
            frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = COLOR_WALL

        self._draw_exit(frame, inverted=bool(visual.get("exit_inverted", False)))
        anchored = set(visual["anchored"])
        capture_cell = visual.get("capture_cell")
        for cell in sorted(self.sockets - anchored):
            self._draw_socket(frame, cell)
        for cell in sorted(anchored):
            self._draw_anchored_crate(frame, cell)
        for cell in sorted(set(visual["loose"])):
            self._draw_loose_crate(frame, cell)
        if capture_cell is not None:
            self._draw_anchored_crate(frame, tuple(capture_cell), int(visual.get("capture_phase", 0)))

        offset = tuple(visual.get("agent_offset", (0, 0)))
        self._draw_agent(frame, tuple(visual["agent"]), offset)
        if offset != (0, 0):
            ax, ay = self._cell_top_left(tuple(visual["agent"]))
            dx, dy = offset
            behind_x = ax + 2 - dx * 2
            behind_y = ay + 2 - dy * 2
            for perpendicular in (-1, 1):
                sx = behind_x + (perpendicular if dy else 0)
                sy = behind_y + (perpendicular if dx else 0)
                if 0 <= sx < VIEW_SIZE and 0 <= sy < VIEW_SIZE:
                    frame[sy, sx] = COLOR_SCRAPE

        # Mandatory action budget bar. It sits outside every authored board.
        bar_x, bar_y, bar_width = 5, 61, 54
        frame[bar_y : bar_y + 2, bar_x : bar_x + bar_width] = COLOR_WALL
        filled = int(bar_width * max(0, self.remaining_steps) / max(1, self.step_budget))
        if filled:
            frame[bar_y : bar_y + 2, bar_x : bar_x + filled] = COLOR_EXIT
        if visual.get("failure_flash"):
            frame[0:2, :] = COLOR_FAILURE
            frame[-2:, :] = COLOR_FAILURE
            frame[:, 0:2] = COLOR_FAILURE
            frame[:, -2:] = COLOR_FAILURE
        return frame

    def _sync_board(self) -> None:
        self.board.pixels = self._render_visual(self._visual_state)

    def _is_blocked_for_crate(self, cell: tuple[int, int]) -> bool:
        return cell in self.walls or cell in self.loose_crates or cell in self.anchored_crates

    def _resolve_move(self, dx: int, dy: int) -> tuple[list[dict[str, object]], bool]:
        old_agent = self.agent
        old_loose = set(self.loose_crates)
        old_anchored = set(self.anchored_crates)
        path: list[tuple[int, int]] = []
        q = self.agent
        while True:
            next_cell = (q[0] + dx, q[1] + dy)
            if next_cell in self.walls or next_cell in self.loose_crates or next_cell in self.anchored_crates:
                break
            q = next_cell
            path.append(q)

        blocker = (q[0] + dx, q[1] + dy)
        frames: list[dict[str, object]] = []
        if blocker not in self.loose_crates:
            self.agent = q
            if q == old_agent:
                frames.append(
                    {
                        "agent": old_agent,
                        "loose": frozenset(old_loose),
                        "anchored": frozenset(old_anchored),
                        "agent_offset": (dx, dy),
                    }
                )
                frames.append(self._settled_visual())
            else:
                for cell in path:
                    frames.append({"agent": cell, "loose": frozenset(old_loose), "anchored": frozenset(old_anchored)})
            return frames, self.agent != old_agent

        behind = (blocker[0] + dx, blocker[1] + dy)
        if self._is_blocked_for_crate(behind):
            self.agent = q
            if q == old_agent:
                frames.append(
                    {
                        "agent": old_agent,
                        "loose": frozenset(old_loose),
                        "anchored": frozenset(old_anchored),
                        "agent_offset": (dx, dy),
                    }
                )
                frames.append(self._settled_visual())
            else:
                for cell in path:
                    frames.append({"agent": cell, "loose": frozenset(old_loose), "anchored": frozenset(old_anchored)})
            return frames, self.agent != old_agent

        self.loose_crates.remove(blocker)
        self.agent = blocker
        crate_path: list[tuple[int, int]] = []
        crate = blocker
        captured = False
        while True:
            next_crate = (crate[0] + dx, crate[1] + dy)
            if self._is_blocked_for_crate(next_crate):
                break
            crate = next_crate
            crate_path.append(crate)
            if crate in self.sockets:
                captured = True
                break

        if captured:
            self.anchored_crates.add(crate)
        else:
            self.loose_crates.add(crate)

        for cell in path:
            frames.append({"agent": cell, "loose": frozenset(old_loose), "anchored": frozenset(old_anchored)})
        for crate_cell in crate_path:
            moving_loose = (old_loose - {blocker}) | {crate_cell}
            if captured and crate_cell == crate:
                moving_loose.remove(crate_cell)
                frames.append(
                    {
                        "agent": blocker,
                        "loose": frozenset(moving_loose),
                        "anchored": frozenset(old_anchored),
                        "capture_cell": crate,
                        "capture_phase": 0,
                    }
                )
            else:
                frames.append({"agent": blocker, "loose": frozenset(moving_loose), "anchored": frozenset(old_anchored)})
        if captured:
            for phase in (1, 2):
                frames.append(
                    {
                        "agent": blocker,
                        "loose": frozenset(self.loose_crates),
                        "anchored": frozenset(old_anchored),
                        "capture_cell": crate,
                        "capture_phase": phase,
                    }
                )
        frames.append(self._settled_visual())
        return frames, True

    def _append_terminal_frames(self) -> None:
        if self.remaining_steps <= 0:
            self._pending_result = "lose"
            self.animation_frames.extend(
                [self._settled_visual(failure_flash=True), self._settled_visual(failure_flash=False)]
            )
            return
        if self.agent == self.exit:
            self._pending_result = "complete"
            self.animation_frames.extend(
                [
                    self._settled_visual(exit_inverted=True),
                    self._settled_visual(exit_inverted=False),
                    self._settled_visual(exit_inverted=True),
                    self._settled_visual(exit_inverted=False),
                ]
            )

    def _play_animation_frame(self) -> None:
        self._visual_state = self.animation_frames.pop(0)
        self._sync_board()
        if self.animation_frames:
            return
        if self._pending_result == "complete":
            # Mark the submitted launch complete before asking ArcEngine to
            # render the deferred next-level arrival frame. Otherwise the same
            # direction would be applied once more on the fresh board.
            self.complete_action()
            self.next_level()
        elif self._pending_result == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _spend_action(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return

        action = _action_id(self.action.id)
        self._pending_result = None
        if action == 0:
            self._visual_state = self._settled_visual()
            self._sync_board()
            self.complete_action()
            return
        if action == ACTION_SPACE:
            self._reset_runtime()
            self.complete_action()
            return

        if action == ACTION_UNDO:
            self._spend_action()
            if self.undo_history:
                self._restore_snapshot(self.undo_history.pop())
            self._visual_state = self._settled_visual()
            self.animation_frames = [self._visual_state]
            self._append_terminal_frames()
            self._play_animation_frame()
            return

        if action == ACTION_CLICK:
            # Click has no hit regions or gameplay effect; even the settled
            # rendered frame remains identical, as promised by the rules.
            self._visual_state = self._settled_visual()
            self.animation_frames = [self._visual_state]
            self._play_animation_frame()
            return

        if action in MOVE_DELTAS:
            before = self._snapshot()
            dx, dy = MOVE_DELTAS[action]
            frames, changed = self._resolve_move(dx, dy)
            if changed:
                self.undo_history.append(before)
            self._spend_action()
            self.animation_frames = frames
            self._append_terminal_frames()
            self._play_animation_frame()
            return

        self.complete_action()
