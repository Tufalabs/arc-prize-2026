from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
CELL_SIZE = 5

UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
SPACE, CLICK, UNDO = 5, 6, 7
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
ARROW_SYMBOLS = {"^": UP, "v": DOWN, "<": LEFT, ">": RIGHT}

COLOR_BACKGROUND = 0
COLOR_FLOOR = 1
COLOR_PLATE = 2
COLOR_SHADOW = 3
COLOR_WALL = 4
COLOR_DOOR = 5
COLOR_AGENT = 6
COLOR_FACING = 7
COLOR_INVALID = 8
COLOR_ARROW = 9
COLOR_RIM = 10
COLOR_BALL = 11
COLOR_BAY = 12
COLOR_EXIT_FRAME = 13
COLOR_SUCCESS = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    baseline: int


LEVEL_SPECS = (
    LevelSpec("Follow the Arrow", ("#####", "#####", "##>X#", "##.##", "#A.##", "#####"), 3),
    LevelSpec("Turn Before Entering", ("#####", "#####", "##vX#", "##.##", "##A##", "#####"), 5),
    LevelSpec("One Push, One Delivery", ("#########", "#########", "#X^Ao..B#", "#########", "#########"), 6),
    LevelSpec(
        "Downstream First",
        (
            "###########",
            "#.....#####",
            "#Ao..>#####",
            "#.###.#####",
            "#.###.#####",
            "#.###.#####",
            "#....^..B##",
            "#X#########",
            "###########",
        ),
        33,
    ),
    LevelSpec(
        "Stage at the Wall",
        (
            "############",
            "#.........##",
            "#.#####.#.##",
            "#Ao.....#.##",
            "#.#####.#.##",
            "#.....>v#.##",
            "#.####.##.##",
            "#.####.##X##",
            "######B#####",
            "############",
        ),
        58,
    ),
    LevelSpec(
        "Borrowed Route",
        (
            "############",
            "######B##.X#",
            "######.##.##",
            "######.##.##",
            "#Ao...v.v.B#",
            "#.###..#####",
            "#.###..#####",
            "#.###.o#####",
            "#......#####",
            "############",
        ),
        59,
    ),
    LevelSpec(
        "The Occupied Handle",
        (
            "############",
            "####X#####B#",
            "#....#####.#",
            "#A..v.....>#",
            "#.##>B######",
            "#.o.o#######",
            "#....#######",
            "############",
        ),
        59,
    ),
)


def _rotate(mask: np.ndarray, direction: int) -> np.ndarray:
    turns = {UP: 0, RIGHT: -1, DOWN: 2, LEFT: 1}[direction]
    return np.rot90(mask, turns)


def _agent_mask(direction: int, *, invalid_tip: bool = False) -> np.ndarray:
    mask = np.full((5, 5), -1, dtype=np.int8)
    for x, y in ((1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (2, 3)):
        mask[y, x] = COLOR_AGENT
    mask[0, 2] = COLOR_INVALID if invalid_tip else COLOR_FACING
    return _rotate(mask, direction)


def _ball_mask(size: int = 5) -> np.ndarray:
    mask = np.full((5, 5), -1, dtype=np.int8)
    if size <= 1:
        mask[2, 2] = COLOR_BALL
    elif size <= 3:
        mask[1:4, 1:4] = COLOR_BALL
    else:
        for y in range(5):
            for x in range(5):
                if abs(x - 2) + abs(y - 2) <= 2:
                    mask[y, x] = COLOR_BALL
    return mask


def _arrow_mask(direction: int, rim_color: int = COLOR_RIM) -> np.ndarray:
    mask = np.full((5, 5), COLOR_PLATE, dtype=np.int8)
    mask[0, :] = rim_color
    mask[4, :] = rim_color
    mask[:, 0] = rim_color
    mask[:, 4] = rim_color
    arrow = np.zeros((5, 5), dtype=bool)
    for x, y in ((2, 0), (1, 1), (2, 1), (3, 1), (2, 2), (2, 3)):
        arrow[y, x] = True
    mask[_rotate(arrow, direction)] = COLOR_ARROW
    return mask


class RotatingMazeDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: RotatingMaze | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None or not hasattr(game, "rows"):
            return frame
        frame[:, :] = COLOR_BACKGROUND
        self._draw_budget(frame, game)
        self._draw_board(frame, game)
        self._draw_feedback(frame, game)
        return frame

    def _draw_budget(self, frame: np.ndarray, game: RotatingMaze) -> None:
        frame[1:5, 4:60] = COLOR_WALL
        frame[2:4, 5:59] = COLOR_INVALID
        width = int(54 * game.remaining_steps / max(1, game.step_budget))
        if width:
            frame[2:4, 5 : 5 + width] = COLOR_SUCCESS
        for index in range(len(LEVEL_SPECS)):
            x = 25 + index * 2
            frame[6, x] = COLOR_SUCCESS if index <= game.level_index else COLOR_WALL

    def _draw_board(self, frame: np.ndarray, game: RotatingMaze) -> None:
        transient = game.transient
        for y, row in enumerate(game.rows):
            for x, char in enumerate(row):
                px, py = game.cell_pixel((x, y))
                frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = COLOR_WALL if char == "#" else COLOR_FLOOR

        flash_arrows = set(transient.get("flash_arrows", ()))
        arrow_override = transient.get("arrow_override", {})
        for cell, direction in game.arrows.items():
            rim = COLOR_INVALID if cell in flash_arrows else transient.get("rim_color", COLOR_RIM)
            self._draw_mask(frame, game, cell, _arrow_mask(arrow_override.get(cell, direction), rim))

        for bay in game.bays:
            if bay in game.filled_bays and bay != transient.get("force_empty_bay"):
                mask = np.full((5, 5), -1, dtype=np.int8)
                mask[0, 1:4] = COLOR_SUCCESS
                mask[4, 1:4] = COLOR_SUCCESS
                mask[1:4, 0] = COLOR_SUCCESS
                mask[1:4, 4] = COLOR_SUCCESS
                mask[1:4, 1:4] = COLOR_BALL
            else:
                mask = np.full((5, 5), -1, dtype=np.int8)
                mask[0, 1:4] = COLOR_BAY
                mask[4, 1:4] = COLOR_BAY
                mask[1:4, 0] = COLOR_BAY
                mask[1:4, 4] = COLOR_BAY
            self._draw_mask(frame, game, bay, mask)

        self._draw_exit(frame, game, transient.get("exit_progress"))

        hidden_ball = transient.get("hide_ball")
        for ball in game.balls:
            if ball != hidden_ball:
                self._draw_mask(frame, game, ball, _ball_mask())

        if not transient.get("hide_agent", False):
            offset = transient.get("agent_offset", (0, 0))
            self._draw_mask(
                frame,
                game,
                game.agent_position,
                _agent_mask(game.agent_facing, invalid_tip=transient.get("invalid_tip", False)),
                offset=offset,
            )
        if "moving_agent" in transient:
            self._draw_mask_at_pixel(frame, _agent_mask(game.agent_facing), transient["moving_agent"])
        if "moving_ball" in transient:
            self._draw_mask_at_pixel(frame, _ball_mask(transient.get("ball_size", 5)), transient["moving_ball"])

    def _draw_exit(self, frame: np.ndarray, game: RotatingMaze, progress: int | None) -> None:
        cell = game.exit_position
        px, py = game.cell_pixel(cell)
        if game.exit_open and not game.transient.get("force_exit_closed", False):
            frame[py : py + 5, px : px + 5] = COLOR_SUCCESS
            frame[py + 1 : py + 4, px + 1 : px + 4] = COLOR_DOOR
            if progress is not None:
                gate_rows = max(0, 4 - progress)
                if gate_rows:
                    frame[py + 1 : py + 1 + gate_rows, px + 1 : px + 4] = COLOR_WALL
        else:
            frame[py : py + 5, px : px + 5] = COLOR_EXIT_FRAME
            frame[py + 1 : py + 4, px + 1 : px + 4] = COLOR_WALL
            frame[py + 2, px + 2] = COLOR_BALL

    def _draw_feedback(self, frame: np.ndarray, game: RotatingMaze) -> None:
        transient = game.transient
        if "invalid_cell" in transient:
            x, y = transient["invalid_cell"]
            px, py = game.cell_pixel((x, y))
            dx, dy = DELTAS[game.agent_facing]
            if dx < 0:
                frame[py : py + 5, px + 4] = COLOR_INVALID
            elif dx > 0:
                frame[py : py + 5, px] = COLOR_INVALID
            elif dy < 0:
                frame[py + 4, px : px + 5] = COLOR_INVALID
            else:
                frame[py, px : px + 5] = COLOR_INVALID
        if "ripple" in transient:
            x, y = transient["ripple"]
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= x + dx < 64 and 0 <= y + dy < 64:
                    frame[y + dy, x + dx] = COLOR_PLATE
        if transient.get("completion"):
            frame[0, :] = COLOR_SUCCESS
            frame[-1, :] = COLOR_SUCCESS
            frame[:, 0] = COLOR_SUCCESS
            frame[:, -1] = COLOR_SUCCESS

    def _draw_mask(
        self,
        frame: np.ndarray,
        game: RotatingMaze,
        cell: tuple[int, int],
        mask: np.ndarray,
        *,
        offset: tuple[int, int] = (0, 0),
    ) -> None:
        px, py = game.cell_pixel(cell)
        self._draw_mask_at_pixel(frame, mask, (px + offset[0], py + offset[1]))

    def _draw_mask_at_pixel(self, frame: np.ndarray, mask: np.ndarray, pixel: tuple[int, int]) -> None:
        px, py = pixel
        y0, y1 = max(0, py), min(64, py + 5)
        x0, x1 = max(0, px), min(64, px + 5)
        if x0 >= x1 or y0 >= y1:
            return
        source = mask[y0 - py : y1 - py, x0 - px : x1 - px]
        target = frame[y0:y1, x0:x1]
        frame[y0:y1, x0:x1] = np.where(source >= 0, source, target)


class RotatingMaze(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.display = RotatingMazeDisplay()
        levels = [Level(grid_size=(64, 64), data={"spec": spec}, name=spec.name) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self.display])
        super().__init__(
            "rotating_maze-0001",
            levels,
            camera,
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed=seed,
        )
        self.display.game = self

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.rows = spec.rows
        self.board_height = len(self.rows)
        self.board_width = len(self.rows[0])
        self.origin_x = (64 - CELL_SIZE * self.board_width) // 2
        self.origin_y = 8 + (50 - CELL_SIZE * self.board_height) // 2
        self.walls: set[tuple[int, int]] = set()
        self.arrows: dict[tuple[int, int], int] = {}
        self.balls: set[tuple[int, int]] = set()
        self.bays: set[tuple[int, int]] = set()
        self.filled_bays: set[tuple[int, int]] = set()
        self.exit_position = (-1, -1)
        self.agent_position = (-1, -1)
        self.agent_facing = RIGHT
        for y, row in enumerate(self.rows):
            if len(row) != self.board_width:
                raise ValueError(f"Ragged level row in {spec.name}")
            for x, char in enumerate(row):
                cell = (x, y)
                if char == "#":
                    self.walls.add(cell)
                elif char in ARROW_SYMBOLS:
                    self.arrows[cell] = ARROW_SYMBOLS[char]
                elif char == "A":
                    self.agent_position = cell
                elif char == "o":
                    self.balls.add(cell)
                elif char == "B":
                    self.bays.add(cell)
                elif char == "X":
                    self.exit_position = cell
        if self.level_index == 1:
            self.agent_facing = UP
        self.exit_open = not self.balls and not self.bays
        self.won = False
        self.step_budget = 6 * spec.baseline
        self.remaining_steps = self.step_budget
        self.undo_history: list[dict[str, Any]] = []
        self.animation_frames: list[dict[str, Any]] = []
        self.transient: dict[str, Any] = {}
        self.pending_terminal: str | None = None
        self._assert_invariants()

    def cell_pixel(self, cell: tuple[int, int]) -> tuple[int, int]:
        return self.origin_x + cell[0] * CELL_SIZE, self.origin_y + cell[1] * CELL_SIZE

    def _snapshot(self) -> dict[str, Any]:
        return {
            "agent_position": self.agent_position,
            "agent_facing": self.agent_facing,
            "arrows": self.arrows.copy(),
            "balls": self.balls.copy(),
            "filled_bays": self.filled_bays.copy(),
            "exit_open": self.exit_open,
            "won": self.won,
        }

    def _restore(self, snapshot: dict[str, Any]) -> None:
        self.agent_position = snapshot["agent_position"]
        self.agent_facing = snapshot["agent_facing"]
        self.arrows = snapshot["arrows"].copy()
        self.balls = snapshot["balls"].copy()
        self.filled_bays = snapshot["filled_bays"].copy()
        self.exit_open = snapshot["exit_open"]
        self.won = snapshot["won"]

    def _inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.board_width and 0 <= cell[1] < self.board_height

    def _add(self, cell: tuple[int, int], direction: int) -> tuple[int, int]:
        dx, dy = DELTAS[direction]
        return cell[0] + dx, cell[1] + dy

    def _agent_solid(self, cell: tuple[int, int]) -> bool:
        return (
            not self._inside(cell)
            or cell in self.walls
            or cell in self.balls
            or cell in self.bays
            or (cell == self.exit_position and not self.exit_open)
        )

    def _ball_solid(self, cell: tuple[int, int], reserved: tuple[int, int]) -> bool:
        return (
            not self._inside(cell)
            or cell in self.walls
            or cell in self.balls
            or cell in self.filled_bays
            or cell == self.exit_position
            or cell == reserved
        )

    def _resolve_agent_chain(self, first: tuple[int, int]) -> tuple[list[tuple[int, int]] | None, set[tuple[int, int]]]:
        path: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        current = first
        while current in self.arrows:
            if current in seen:
                return None, set(path)
            seen.add(current)
            path.append(current)
            current = self._add(current, self.arrows[current])
        if self._agent_solid(current):
            return None, set(path)
        path.append(current)
        return path, set()

    def _trace_ball(
        self, start: tuple[int, int], direction: int
    ) -> tuple[tuple[int, int] | None, bool, list[tuple[int, int]], set[tuple[int, int]]]:
        self.balls.remove(start)
        current = start
        path: list[tuple[int, int]] = []
        entries: list[tuple[tuple[int, int], int]] = []
        seen_entries: set[tuple[tuple[int, int], int]] = set()
        flash: set[tuple[int, int]] = set()
        captured = False
        while True:
            nxt = self._add(current, direction)
            if self._ball_solid(nxt, start):
                break
            if nxt in self.bays and nxt not in self.filled_bays:
                path.append(nxt)
                self.filled_bays.add(nxt)
                captured = True
                current = nxt
                break
            if nxt not in self.arrows:
                current = nxt
                path.append(current)
                continue

            first_arrow = nxt
            run_path: list[tuple[int, int]] = []
            run_valid = True
            run_target: tuple[int, int] | None = None
            while nxt in self.arrows:
                entry = (nxt, direction)
                if entry in seen_entries:
                    repeated_at = entries.index(entry)
                    flash.update(cell for cell, _incoming in entries[repeated_at:])
                    flash.update(run_path)
                    flash.add(nxt)
                    run_valid = False
                    break
                seen_entries.add(entry)
                entries.append(entry)
                run_path.append(nxt)
                direction = self.arrows[nxt]
                nxt = self._add(nxt, direction)
            if run_valid:
                if self._ball_solid(nxt, start):
                    flash.update(run_path)
                    run_valid = False
                elif nxt in self.bays and nxt not in self.filled_bays:
                    run_target = nxt
                else:
                    run_target = nxt
            if not run_valid:
                if not flash:
                    flash.add(first_arrow)
                break
            path.extend(run_path)
            if run_target is None:
                break
            path.append(run_target)
            current = run_target
            if current in self.bays and current not in self.filled_bays:
                self.filled_bays.add(current)
                captured = True
                break

        if current == start and not captured:
            self.balls.add(start)
            return None, False, path, flash
        if not captured:
            self.balls.add(current)
            return current, False, path, flash
        return None, True, path, flash

    def _pixel_between(self, a: tuple[int, int], b: tuple[int, int], fraction: float) -> tuple[int, int]:
        ax, ay = self.cell_pixel(a)
        bx, by = self.cell_pixel(b)
        return round(ax + (bx - ax) * fraction), round(ay + (by - ay) * fraction)

    def _queue_ball_animation(
        self,
        start: tuple[int, int],
        path: list[tuple[int, int]],
        final_ball: tuple[int, int] | None,
        captured: bool,
        flash: set[tuple[int, int]],
        exit_was_open: bool,
    ) -> None:
        opening = captured and not exit_was_open and self.exit_open
        capture_bay = path[-1] if captured else None
        previous = start
        for cell in path:
            self.animation_frames.append(
                {
                    "moving_ball": self._pixel_between(previous, cell, 0.5),
                    "hide_ball": final_ball,
                    "force_empty_bay": capture_bay,
                    "force_exit_closed": opening,
                }
            )
            self.animation_frames.append(
                {
                    "moving_ball": self.cell_pixel(cell),
                    "hide_ball": final_ball,
                    "force_empty_bay": capture_bay,
                    "force_exit_closed": opening,
                }
            )
            previous = cell
        if captured:
            bay = path[-1]
            self.animation_frames.extend(
                [
                    {
                        "moving_ball": self.cell_pixel(bay),
                        "ball_size": 3,
                        "force_empty_bay": bay,
                        "force_exit_closed": opening,
                    },
                    {
                        "moving_ball": self.cell_pixel(bay),
                        "ball_size": 1,
                        "force_empty_bay": bay,
                        "force_exit_closed": opening,
                    },
                    {"force_exit_closed": opening},
                ]
            )
            if opening:
                self.animation_frames.extend({"exit_progress": value} for value in (1, 2, 3))
        if flash:
            self.animation_frames.append({"flash_arrows": tuple(flash)})
        self.animation_frames.append({})

    def _resolve_direction(self, direction: int) -> None:
        self.undo_history.append(self._snapshot())
        self.agent_facing = direction
        target = self._add(self.agent_position, direction)
        if target in self.balls:
            exit_was_open = self.exit_open
            final_ball, captured, path, flash = self._trace_ball(target, direction)
            if final_ball is None and not captured:
                self._queue_bump(direction)
                if flash:
                    self.animation_frames.insert(-1, {"flash_arrows": tuple(flash)})
                return
            self.agent_position = target
            self.exit_open = not self.balls and self.filled_bays == self.bays
            self._queue_ball_animation(target, path, final_ball, captured, flash, exit_was_open)
            return
        if target in self.arrows:
            path, flash = self._resolve_agent_chain(target)
            if path is None:
                self.animation_frames.extend([{"flash_arrows": tuple(flash)}, {}])
                return
            self.agent_position = path[-1]
            for index, cell in enumerate(path[:-1]):
                next_cell = path[index + 1]
                self.animation_frames.append({"hide_agent": True, "moving_agent": self.cell_pixel(cell)})
                self.animation_frames.append({"hide_agent": True, "moving_agent": self.cell_pixel(next_cell)})
            if self.agent_position == self.exit_position and self.exit_open:
                self.won = True
                self.animation_frames.append({"completion": True})
                self.pending_terminal = "solve"
            else:
                self.animation_frames.append({})
            return
        if self._agent_solid(target):
            self._queue_bump(direction)
            return
        self.agent_position = target
        if target == self.exit_position and self.exit_open:
            self.won = True
            self.animation_frames.append({"completion": True})
            self.pending_terminal = "solve"
        else:
            self.animation_frames.append({})

    def _queue_bump(self, direction: int) -> None:
        dx, dy = DELTAS[direction]
        self.animation_frames.extend([{"agent_offset": (dx, dy)}, {"agent_offset": (dx, dy), "invalid_tip": True}, {}])

    def _resolve_space(self) -> None:
        target = self._add(self.agent_position, self.agent_facing)
        if target not in self.arrows:
            self.animation_frames.extend([{"invalid_cell": target, "invalid_tip": True}, {}])
            return
        self.undo_history.append(self._snapshot())
        old = self.arrows[target]
        new = RIGHT if old == UP else DOWN if old == RIGHT else LEFT if old == DOWN else UP
        self.arrows[target] = new
        self.animation_frames.extend(
            [
                {"arrow_override": {target: old}, "rim_color": COLOR_SUCCESS},
                {"arrow_override": {target: old}, "rim_color": COLOR_PLATE},
                {},
            ]
        )

    def _spend_step(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.remaining_steps == 0 and self.pending_terminal != "solve":
            self.pending_terminal = "lose"
            if not self.animation_frames:
                self.animation_frames.append({"invalid_tip": True})

    def _finish_action(self) -> None:
        terminal = self.pending_terminal
        self.pending_terminal = None
        if terminal == "solve":
            self.complete_action()
            self.next_level()
        elif terminal == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _play_animation_frame(self) -> None:
        self.transient = self.animation_frames.pop(0)
        if not self.animation_frames:
            self._finish_action()

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        if self.action.id == GameAction.RESET:
            self.transient = {}
            self.complete_action()
            return

        action = int(self.action.id.value)
        self.transient = {}
        if action == UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self.animation_frames.append({})
        elif action == CLICK:
            data = self.action.data or {}
            x = max(0, min(63, int(data.get("x", 0))))
            y = max(0, min(63, int(data.get("y", 0))))
            self.animation_frames.extend([{"ripple": (x, y)}, {}])
        elif action in DELTAS:
            self._resolve_direction(action)
        elif action == SPACE:
            self._resolve_space()
        else:
            self.animation_frames.append({})

        self._spend_step()
        self._assert_invariants()
        self._play_animation_frame()

    def _assert_invariants(self) -> None:
        assert self.agent_position not in self.arrows
        assert not (self.balls & self.arrows.keys())
        assert self.agent_position not in self.bays
        assert not (self.balls & self.filled_bays)
        assert self.filled_bays <= self.bays
        assert self.exit_open == (not self.balls and self.filled_bays == self.bays)
        if self.won:
            assert self.agent_position == self.exit_position
            assert self.exit_open
