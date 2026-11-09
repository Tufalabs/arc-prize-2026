from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

VIEW_SIZE = 64
CELL_SIZE = 4

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

# Logical coordinates are (row, column).
MOVE_DELTAS = {ACTION_UP: (-1, 0), ACTION_DOWN: (1, 0), ACTION_LEFT: (0, -1), ACTION_RIGHT: (0, 1)}

COLOR_COMPLETE = 0
COLOR_FLOOR = 1
COLOR_WALL = 3
COLOR_WALL_SHADOW = 4
COLOR_OUTSIDE = 5
COLOR_AGENT = 6
COLOR_AGENT_LIGHT = 7
COLOR_FLASH = 8
COLOR_A_BODY = 9
COLOR_A_MARK = 10
COLOR_B_MARK = 11
COLOR_B_BODY = 12
COLOR_CLAMP = 14

LOOSE_MASK = ((0, 1, 1, 0), (1, 1, 1, 1), (1, 1, 1, 1), (0, 1, 1, 0))
AGENT_MASK = ((0, 1, 1, 0), (1, 1, 1, 1), (0, 1, 1, 0), (0, 1, 1, 0))


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
        "First Glide",
        (
            "##########",
            "#........#",
            "#........#",
            "#........#",
            "#.A...a..#",
            "#........#",
            "#@.......#",
            "#........#",
            "#........#",
            "##########",
        ),
        3,
    ),
    LevelSpec(
        "Bank Turn",
        (
            "##########",
            "#........#",
            "#........#",
            "#..#.....#",
            "#.....a..#",
            "#........#",
            "#..A.....#",
            "#..@.....#",
            "#........#",
            "##########",
        ),
        5,
    ),
    LevelSpec(
        "Borrowed Brake",
        (
            "##########",
            "#........#",
            "#....aa..#",
            "#........#",
            "#.#......#",
            "#@A...A..#",
            "#.#......#",
            "#........#",
            "#........#",
            "##########",
        ),
        9,
    ),
    LevelSpec(
        "Build the Brake",
        (
            "##########",
            "#....a...#",
            "#......a.#",
            "#.#......#",
            "#.A...b..#",
            "#.#......#",
            "#........#",
            "#@.B...A.#",
            "#........#",
            "##########",
        ),
        25,
    ),
    LevelSpec(
        "Through the Wrong Ring",
        (
            "##########",
            "#........#",
            "#.B...a..#",
            "#.#b.....#",
            "#@B.a..b.#",
            "#.#B.....#",
            "#....b...#",
            "#.A....A.#",
            "#........#",
            "##########",
        ),
        51,
    ),
    LevelSpec(
        "Clear the Launch Side",
        (
            "############",
            "#..........#",
            "#....B.....#",
            "#..A.....a.#",
            "#..........#",
            "#.......b..#",
            "#b@B.a...A.#",
            "#..........#",
            "#.......a..#",
            "#....A.....#",
            "#..........#",
            "############",
        ),
        54,
    ),
    LevelSpec(
        "Interlocked Rink",
        (
            "############",
            "#..........#",
            "#...B...b..#",
            "#..........#",
            "#..A.....a.#",
            "#......a...#",
            "#..........#",
            "#bB......A.#",
            "#.......a..#",
            "#.B.Q......#",
            "#...@......#",
            "############",
        ),
        85,
    ),
)


def _build_level(spec: LevelSpec) -> Level:
    height = len(spec.rows)
    width = len(spec.rows[0])
    if any(len(row) != width for row in spec.rows):
        raise ValueError(f"{spec.name} has a ragged map")

    walls: set[tuple[int, int]] = set()
    goals: list[tuple[str, tuple[int, int]]] = []
    stones: list[tuple[str, tuple[int, int]]] = []
    agent: tuple[int, int] | None = None
    for row_index, row in enumerate(spec.rows, start=1):
        for column_index, symbol in enumerate(row, start=1):
            cell = (row_index, column_index)
            if symbol == "#":
                walls.add(cell)
            elif symbol == "@":
                agent = cell
            elif symbol in {"A", "B"}:
                stones.append((symbol, cell))
            elif symbol in {"a", "b"}:
                goals.append((symbol.upper(), cell))
            elif symbol == "Q":
                stones.append(("A", cell))
                goals.append(("B", cell))
            elif symbol != ".":
                raise ValueError(f"Unknown map symbol {symbol!r}")
    if agent is None:
        raise ValueError(f"{spec.name} has no agent")
    if len(stones) != len(goals):
        raise ValueError(f"{spec.name} must have one stone per socket")

    scene = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_OUTSIDE, dtype=np.int8),
        name="scene",
        x=0,
        y=0,
        layer=0,
        visible=True,
        collidable=False,
        tags=["scene"],
    )
    return Level(
        sprites=[scene],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={
            "width": width,
            "height": height,
            "walls": tuple(sorted(walls)),
            "goals": tuple(goals),
            "stones": tuple(stones),
            "agent": agent,
            "budget": spec.budget,
        },
    )


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _add(cell: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    return cell[0] + delta[0], cell[1] + delta[1]


class StoneSkaters(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "stone_skaters-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_OUTSIDE, COLOR_OUTSIDE),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK, ACTION_UNDO],
            seed,
        )

    def on_set_level(self, _level: Level) -> None:
        self.scene = self.current_level.get_sprites_by_tag("scene")[0]
        self.width = int(self.current_level.get_data("width"))
        self.height = int(self.current_level.get_data("height"))
        self.walls = set(self.current_level.get_data("walls"))
        self.goals = [
            {"type": stone_type, "position": tuple(position)}
            for stone_type, position in self.current_level.get_data("goals")
        ]
        self.stones = [
            {"type": stone_type, "position": tuple(position), "captured": False}
            for stone_type, position in self.current_level.get_data("stones")
        ]
        self.agent_position = tuple(self.current_level.get_data("agent"))
        self.step_budget = int(self.current_level.get_data("budget"))
        self.steps_taken = 0
        self.remaining_steps = self.step_budget
        self.truncated = False
        self.selected_object = None
        self.cursor_cell = None
        self.undo_history: list[tuple[tuple[int, int], tuple[tuple[tuple[int, int], bool], ...]]] = []
        self.animation_frames: list[np.ndarray] = []
        self._finish_as_win = False
        self._finish_as_loss = False
        self._sync_scene()

    @property
    def origin(self) -> tuple[int, int]:
        return (64 - self.width * CELL_SIZE) // 2, 8 + (48 - self.height * CELL_SIZE) // 2

    def _cell_pixel(self, cell: tuple[int, int]) -> tuple[int, int]:
        origin_x, origin_y = self.origin
        return origin_x + (cell[1] - 1) * CELL_SIZE, origin_y + (cell[0] - 1) * CELL_SIZE

    def _snapshot(self) -> tuple[tuple[int, int], tuple[tuple[tuple[int, int], bool], ...]]:
        return self.agent_position, tuple((tuple(stone["position"]), bool(stone["captured"])) for stone in self.stones)

    def _restore(self, snapshot: tuple[tuple[int, int], tuple[tuple[tuple[int, int], bool], ...]]) -> None:
        self.agent_position = snapshot[0]
        for stone, (position, captured) in zip(self.stones, snapshot[1], strict=True):
            stone["position"] = position
            stone["captured"] = captured

    def _occupied(self, cell: tuple[int, int], *, except_index: int | None = None) -> int | None:
        for index, stone in enumerate(self.stones):
            if index != except_index and stone["position"] == cell:
                return index
        return None

    def _goal_type(self, cell: tuple[int, int]) -> str | None:
        for goal in self.goals:
            if goal["position"] == cell:
                return str(goal["type"])
        return None

    def _goal_filled(self, goal: dict[str, object]) -> bool:
        return any(
            stone["captured"] and stone["position"] == goal["position"] and stone["type"] == goal["type"]
            for stone in self.stones
        )

    def _is_solved(self) -> bool:
        return all(self._goal_filled(goal) for goal in self.goals)

    def _draw_mask(self, frame: np.ndarray, x: int, y: int, mask: tuple[tuple[int, ...], ...], color: int) -> None:
        for local_y, row in enumerate(mask):
            for local_x, enabled in enumerate(row):
                if enabled and 0 <= x + local_x < VIEW_SIZE and 0 <= y + local_y < VIEW_SIZE:
                    frame[y + local_y, x + local_x] = color

    def _draw_socket(self, frame: np.ndarray, stone_type: str, cell: tuple[int, int]) -> None:
        x, y = self._cell_pixel(cell)
        accent = COLOR_A_MARK if stone_type == "A" else COLOR_B_MARK
        body = COLOR_A_BODY if stone_type == "A" else COLOR_B_BODY
        frame[y, x : x + 4] = accent
        frame[y + 3, x : x + 4] = accent
        frame[y : y + 4, x] = accent
        frame[y : y + 4, x + 3] = accent
        if stone_type == "A":
            frame[y + 1, x + 1 : x + 3] = body
        else:
            frame[y + 1 : y + 3, x + 1] = body

    def _draw_loose_at_pixel(
        self, frame: np.ndarray, stone_type: str, x: int, y: int, *, flash_delta: tuple[int, int] | None = None
    ) -> None:
        body = COLOR_A_BODY if stone_type == "A" else COLOR_B_BODY
        mark = COLOR_A_MARK if stone_type == "A" else COLOR_B_MARK
        self._draw_mask(frame, x, y, LOOSE_MASK, body)
        if stone_type == "A":
            frame[y + 1, x + 1 : x + 3] = mark
        else:
            frame[y + 1 : y + 3, x + 1] = mark
        if flash_delta == (-1, 0):
            frame[y, x + 1 : x + 3] = COLOR_FLASH
        elif flash_delta == (1, 0):
            frame[y + 3, x + 1 : x + 3] = COLOR_FLASH
        elif flash_delta == (0, -1):
            frame[y + 1 : y + 3, x] = COLOR_FLASH
        elif flash_delta == (0, 1):
            frame[y + 1 : y + 3, x + 3] = COLOR_FLASH

    def _draw_captured(
        self, frame: np.ndarray, stone_type: str, cell: tuple[int, int], clamp: int, mark: int | None
    ) -> None:
        x, y = self._cell_pixel(cell)
        frame[y : y + 4, x : x + 4] = COLOR_WALL_SHADOW
        frame[y, x] = clamp
        frame[y, x + 3] = clamp
        frame[y + 3, x] = clamp
        frame[y + 3, x + 3] = clamp
        body = COLOR_A_BODY if stone_type == "A" else COLOR_B_BODY
        frame[y + 1 : y + 3, x + 1 : x + 3] = body
        actual_mark = mark if mark is not None else (COLOR_A_MARK if stone_type == "A" else COLOR_B_MARK)
        if stone_type == "A":
            frame[y + 1, x + 1 : x + 3] = actual_mark
        else:
            frame[y + 1 : y + 3, x + 1] = actual_mark

    def _draw_agent_at_pixel(self, frame: np.ndarray, x: int, y: int) -> None:
        self._draw_mask(frame, x, y, AGENT_MASK, COLOR_AGENT)
        frame[y + 1, x + 1 : x + 3] = COLOR_AGENT_LIGHT

    def _render(
        self,
        *,
        agent: tuple[int, int] | None = None,
        stones: list[dict[str, object]] | None = None,
        moving: tuple[int, int, tuple[int, int] | None] | None = None,
        omit_stone: int | None = None,
        contact_flash: tuple[tuple[int, int], tuple[int, int]] | None = None,
        clamp_color: int = COLOR_CLAMP,
        highlighted_stone: int | None = None,
        loss_flash: bool = False,
    ) -> np.ndarray:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_OUTSIDE, dtype=np.int8)
        origin_x, origin_y = self.origin
        frame[origin_y : origin_y + self.height * 4, origin_x : origin_x + self.width * 4] = COLOR_FLOOR
        for wall in self.walls:
            x, y = self._cell_pixel(wall)
            frame[y : y + 4, x : x + 4] = COLOR_WALL
            frame[y + 3, x : x + 4] = COLOR_WALL_SHADOW
            frame[y : y + 4, x + 3] = COLOR_WALL_SHADOW
        active_stones = self.stones if stones is None else stones
        for goal in self.goals:
            if not any(stone["captured"] and stone["position"] == goal["position"] for stone in active_stones):
                self._draw_socket(frame, str(goal["type"]), tuple(goal["position"]))
        for index, stone in enumerate(active_stones):
            if index == omit_stone:
                continue
            if stone["captured"]:
                mark = COLOR_COMPLETE if index == highlighted_stone else None
                self._draw_captured(frame, str(stone["type"]), tuple(stone["position"]), clamp_color, mark)
            else:
                x, y = self._cell_pixel(tuple(stone["position"]))
                self._draw_loose_at_pixel(frame, str(stone["type"]), x, y)
        if moving is not None and omit_stone is not None:
            self._draw_loose_at_pixel(
                frame, str(active_stones[omit_stone]["type"]), moving[0], moving[1], flash_delta=moving[2]
            )
        agent_cell = self.agent_position if agent is None else agent
        agent_x, agent_y = self._cell_pixel(agent_cell)
        if contact_flash is not None:
            delta = contact_flash[1]
            agent_x += delta[1]
            agent_y += delta[0]
        self._draw_agent_at_pixel(frame, agent_x, agent_y)
        if contact_flash is not None:
            cell, delta = contact_flash
            x, y = self._cell_pixel(cell)
            if delta == (-1, 0):
                frame[y + 3, x : x + 4] = COLOR_FLASH
            elif delta == (1, 0):
                frame[y, x : x + 4] = COLOR_FLASH
            elif delta == (0, -1):
                frame[y : y + 4, x + 3] = COLOR_FLASH
            else:
                frame[y : y + 4, x] = COLOR_FLASH
        self._draw_hud(frame, loss_flash=loss_flash)
        return frame

    def _draw_hud(self, frame: np.ndarray, *, loss_flash: bool = False) -> None:
        bar_x, bar_y, bar_width = 8, 3, 48
        frame[bar_y : bar_y + 2, bar_x : bar_x + bar_width] = COLOR_WALL_SHADOW
        filled = int(bar_width * self.remaining_steps / max(1, self.step_budget))
        if filled:
            frame[bar_y : bar_y + 2, bar_x : bar_x + filled] = COLOR_FLASH if loss_flash else COLOR_CLAMP
        for index in range(len(LEVEL_SPECS)):
            color = COLOR_CLAMP if index <= self.level_index else COLOR_WALL_SHADOW
            frame[0:2, 25 + index * 2] = color
        if loss_flash:
            frame[2:6, bar_x : bar_x + bar_width] = COLOR_FLASH

    def _sync_scene(self, *, loss_flash: bool = False) -> None:
        self.scene.pixels = self._render(loss_flash=loss_flash)

    def _spend_step(self) -> None:
        self.steps_taken += 1
        self.remaining_steps = max(0, self.step_budget - self.steps_taken)

    def _queue_agent_bump(self, target: tuple[int, int], delta: tuple[int, int]) -> None:
        self.animation_frames = [self._render(contact_flash=(target, delta)), self._render()]

    def _queue_failed_push(self, stone_index: int, delta: tuple[int, int]) -> None:
        x, y = self._cell_pixel(tuple(self.stones[stone_index]["position"]))
        self.animation_frames = [
            self._render(moving=(x + delta[1], y + delta[0], delta), omit_stone=stone_index),
            self._render(),
        ]

    def _queue_slide(
        self,
        stone_index: int,
        path: list[tuple[int, int]],
        delta: tuple[int, int],
        before_agent: tuple[int, int],
        before_stones: list[dict[str, object]],
        captured: bool,
    ) -> None:
        frames: list[np.ndarray] = []
        previous = tuple(before_stones[stone_index]["position"])
        for path_index, cell in enumerate(path):
            x, y = self._cell_pixel(previous)
            frames.append(
                self._render(
                    agent=before_agent,
                    stones=before_stones,
                    moving=(x + delta[1] * 2, y + delta[0] * 2, None),
                    omit_stone=stone_index,
                )
            )
            x, y = self._cell_pixel(cell)
            shown_agent = self.agent_position if path_index == len(path) - 1 else before_agent
            frames.append(
                self._render(agent=shown_agent, stones=before_stones, moving=(x, y, None), omit_stone=stone_index)
            )
            previous = cell
        if captured:
            frames.append(self._render())
            frames.append(self._render(highlighted_stone=stone_index))
            frames.append(self._render())
            if self._is_solved():
                for pulse in range(6):
                    frames.append(self._render(clamp_color=COLOR_COMPLETE if pulse % 2 else COLOR_CLAMP))
                self._finish_as_win = True
        self.animation_frames = frames

    def _resolve_move(self, delta: tuple[int, int]) -> None:
        adjacent = _add(self.agent_position, delta)
        if adjacent in self.walls:
            self._queue_agent_bump(adjacent, delta)
            return
        stone_index = self._occupied(adjacent)
        if stone_index is None:
            self.undo_history.append(self._snapshot())
            self.agent_position = adjacent
            self._sync_scene()
            return
        stone = self.stones[stone_index]
        if stone["captured"]:
            self._queue_agent_bump(adjacent, delta)
            return
        first = _add(adjacent, delta)
        if first in self.walls or self._occupied(first, except_index=stone_index) is not None:
            self._queue_failed_push(stone_index, delta)
            return

        snapshot = self._snapshot()
        before_agent = self.agent_position
        before_stones = [dict(item) for item in self.stones]
        path: list[tuple[int, int]] = []
        current = adjacent
        captured = False
        while True:
            candidate = _add(current, delta)
            if candidate in self.walls or self._occupied(candidate, except_index=stone_index) is not None:
                break
            current = candidate
            path.append(current)
            if self._goal_type(current) == stone["type"]:
                captured = True
                break
        self.undo_history.append(snapshot)
        stone["position"] = current
        stone["captured"] = captured
        self.agent_position = adjacent
        self._queue_slide(stone_index, path, delta, before_agent, before_stones, captured)

    def _play_animation_frame(self) -> None:
        self.scene.pixels = self.animation_frames.pop(0)
        if self.animation_frames:
            return
        if self._finish_as_win:
            self._finish_as_win = False
            self.next_level()
        elif self._finish_as_loss:
            self._finish_as_loss = False
            self.lose()
        self.complete_action()

    def _finish_or_fail(self) -> None:
        if self.remaining_steps <= 0 and not self._is_solved():
            self.truncated = True
            self.animation_frames.append(self._render(loss_flash=True))
            self._finish_as_loss = True
        if self.animation_frames:
            self._play_animation_frame()
        else:
            self.complete_action()

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return

        action = _action_id(self.action.id)
        if action == int(GameAction.RESET.value):
            self._sync_scene()
            self.complete_action()
            return

        self._spend_step()
        if action == ACTION_UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self._sync_scene()
        elif action in MOVE_DELTAS:
            self._resolve_move(MOVE_DELTAS[action])
        else:
            # SPACE and every valid click are deliberately inert.
            self._sync_scene()
        self._finish_or_fail()


__all__ = ["StoneSkaters"]
