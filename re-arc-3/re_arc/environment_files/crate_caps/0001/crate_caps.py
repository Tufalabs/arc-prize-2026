from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = range(1, 8)
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
FACING = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

SIZE = 64
CELL = 5
BOARD_LEFT = 4
TOP = 3
VOID = 5
FLOOR = 1
PIT_RIM = 3
DEPLETED = 4
ERROR = 8
PLAYER = 9
ACCENT = 10
DISC_HIGHLIGHT = 11
DISC = 12
EXIT = 14
WHITE = 0

PIT_MASKS = {
    0: np.array([[1, 3, 3, 3, 1], [3, 5, 5, 5, 3], [3, 5, 5, 5, 3], [3, 5, 5, 5, 3], [1, 3, 3, 3, 1]], dtype=np.int8),
    1: np.array(
        [[1, 3, 3, 3, 1], [3, 5, 5, 5, 3], [3, 5, 5, 5, 3], [3, 12, 11, 12, 3], [1, 3, 12, 3, 1]], dtype=np.int8
    ),
    2: np.array(
        [[1, 12, 12, 12, 1], [12, 11, 11, 11, 12], [12, 11, 0, 11, 12], [12, 11, 11, 11, 12], [1, 12, 12, 12, 1]],
        dtype=np.int8,
    ),
}
LOOSE_MASK = np.array(
    [[1, 1, 12, 1, 1], [1, 12, 11, 12, 1], [12, 11, 11, 11, 12], [1, 12, 11, 12, 1], [1, 1, 12, 1, 1]], dtype=np.int8
)
EXIT_MASK = np.array(
    [[1, 1, 14, 1, 1], [1, 14, 14, 14, 1], [14, 14, 0, 14, 14], [1, 14, 14, 14, 1], [1, 1, 14, 1, 1]], dtype=np.int8
)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    budget: int


LEVEL_SPECS = (
    LevelSpec("Clear Hands", ("###########",) * 5 + ("###So.E####",) + ("###########",) * 5, 24),
    LevelSpec("Two Layers", ("###########",) * 5 + ("##So0.E####", "###o#######") + ("###########",) * 4, 60),
    LevelSpec("Take the Road With You", ("###########",) * 5 + ("#S.2..1.E##",) + ("###########",) * 5, 66),
    LevelSpec(
        "One Bridge, Then Another",
        (
            "###########",
            "####E######",
            "####1######",
            "####.######",
            "####0######",
            "####.######",
            "###oS.0.o##",
            "####o######",
            "###########",
            "###########",
            "###########",
        ),
        192,
    ),
    LevelSpec(
        "Set It Down Now",
        (
            "###########",
            "###########",
            "#####o.####",
            "######.##E#",
            "####o1.##1#",
            "####2#0.0.#",
            "####.o.####",
            "######S####",
            "###########",
            "###########",
            "###########",
        ),
        246,
    ),
    LevelSpec(
        "Keep the Middle",
        (
            "###########",
            "##########E",
            "########o#0",
            "#######o.1.",
            "########0##",
            "#######o.##",
            "#####..0.o#",
            "#####0#####",
            "#S.22..####",
            "###########",
            "###########",
        ),
        360,
    ),
    LevelSpec(
        "The Far-Side Cap",
        (
            "##########E",
            "########o.0",
            "#####o.o#0#",
            "####.#o##.o",
            "####.0.##1o",
            "####2#1.0.#",
            "####...####",
            "######S####",
            "###########",
            "###########",
            "###########",
        ),
        414,
    ),
)


def _cell_origin(cell: tuple[int, int]) -> tuple[int, int]:
    row, col = cell
    return BOARD_LEFT + CELL * (col - 1), TOP + CELL * (row - 1)


def _parse(spec: LevelSpec) -> dict[str, Any]:
    floor: set[tuple[int, int]] = set()
    loose: set[tuple[int, int]] = set()
    pits: dict[tuple[int, int], int] = {}
    start = (1, 1)
    exit_cell = (1, 1)
    for row, line in enumerate(spec.rows, 1):
        if len(line) != 11:
            raise ValueError(f"{spec.name}: row {row} is not 11 cells")
        for col, char in enumerate(line, 1):
            cell = (row, col)
            if char in ".SoE":
                floor.add(cell)
            if char == "S":
                start = cell
            elif char == "E":
                exit_cell = cell
            elif char == "o":
                loose.add(cell)
            elif char in "012":
                pits[cell] = int(char)
    return {
        "floor": floor,
        "loose": loose,
        "pits": pits,
        "start": start,
        "exit": exit_cell,
        "facing": "LEFT"
        if spec.name == "One Bridge, Then Another"
        else ("UP" if spec.name in {"Set It Down Now", "The Far-Side Cap"} else "RIGHT"),
        "budget": spec.budget,
    }


class CrateCapsView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: CrateCaps | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = VOID
        state = game._render_snapshot or game._snapshot()
        position, facing, carrying, loose_tuple, pit_tuple = state
        loose = set(loose_tuple)
        pits = dict(pit_tuple)

        for cell in game.floor:
            x, y = _cell_origin(cell)
            frame[y : y + CELL, x : x + CELL] = FLOOR
        for cell, count in pits.items():
            x, y = _cell_origin(cell)
            frame[y : y + CELL, x : x + CELL] = PIT_MASKS[count]
        for cell in loose:
            x, y = _cell_origin(cell)
            frame[y : y + CELL, x : x + CELL] = LOOSE_MASK

        exit_mask = EXIT_MASK.copy()
        if game._effect.get("exit_color") is not None:
            exit_mask[exit_mask == EXIT] = int(game._effect["exit_color"])
        x, y = _cell_origin(game.exit_cell)
        frame[y : y + CELL, x : x + CELL] = exit_mask

        self._draw_player(frame, position, facing, carrying, game._effect)
        self._draw_effect(frame, game._effect, carrying)
        self._draw_bar(frame, game.steps, game.step_budget, game._failed)
        return frame

    def _draw_player(
        self, frame: np.ndarray, cell: tuple[int, int], facing: str, carrying: bool, effect: dict[str, Any]
    ) -> None:
        x, y = _cell_origin(cell)
        dr, dc = effect.get("bump", (0, 0))
        x += dc
        y += dr
        if carrying:
            frame[y, x + 1 : x + 4] = [DISC, DISC_HIGHLIGHT, DISC]
            frame[y + 2 : y + 4, x + 1 : x + 4] = PLAYER
            noses = {"UP": (1, 2), "DOWN": (4, 2), "LEFT": (3, 0), "RIGHT": (3, 4)}
        else:
            frame[y + 1 : y + 4, x + 1 : x + 4] = PLAYER
            noses = {"UP": (0, 2), "DOWN": (4, 2), "LEFT": (2, 0), "RIGHT": (2, 4)}
        nr, nc = noses[facing]
        frame[y + nr, x + nc] = ERROR if effect.get("nose_error") else ACCENT
        if carrying and effect.get("carry_error"):
            frame[y, x + 1 : x + 4] = ERROR

    def _draw_effect(self, frame: np.ndarray, effect: dict[str, Any], carrying: bool) -> None:
        del carrying
        cell = effect.get("pulse_cell")
        if cell is not None:
            x, y = _cell_origin(cell)
            color = int(effect.get("pulse_color", ACCENT))
            frame[y, x : x + CELL] = color
            frame[y + CELL - 1, x : x + CELL] = color
            frame[y : y + CELL, x] = color
            frame[y : y + CELL, x + CELL - 1] = color
        transfer = effect.get("transfer")
        if transfer is not None:
            row, col = transfer
            cx = BOARD_LEFT + round(CELL * (col - 1)) + 2
            cy = TOP + round(CELL * (row - 1)) + 2
            frame[max(0, cy - 1) : min(SIZE, cy + 2), max(0, cx - 1) : min(SIZE, cx + 2)] = DISC
            if 0 <= cy < SIZE and 0 <= cx < SIZE:
                frame[cy, cx] = DISC_HIGHLIGHT

    def _draw_bar(self, frame: np.ndarray, steps: int, budget: int, failed: bool) -> None:
        if failed:
            frame[60:62, 4:60] = ERROR
            return
        frame[60:62, 4:60] = DEPLETED
        remaining = max(0, budget - steps)
        width = int(56 * remaining / budget)
        if width:
            frame[60:62, 4 : 4 + width] = ACCENT


class CrateCaps(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = CrateCapsView()
        levels = [Level([], (64, 64), {"spec": spec, **_parse(spec)}, spec.name) for spec in LEVEL_SPECS]
        super().__init__(
            "crate_caps-0001",
            levels,
            Camera(0, 0, SIZE, SIZE, VOID, VOID, [self._view]),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed,
        )
        self._view.game = self

    def on_set_level(self, _level: Level) -> None:
        self.floor = set(self.current_level.get_data("floor"))
        self.loose_discs = set(self.current_level.get_data("loose"))
        self.pit_counts = dict(self.current_level.get_data("pits"))
        self.player_position = tuple(self.current_level.get_data("start"))
        self.exit_cell = tuple(self.current_level.get_data("exit"))
        self.facing = str(self.current_level.get_data("facing"))
        self.carrying = False
        self.steps = 0
        self.step_budget = int(self.current_level.get_data("budget"))
        self.selected = None
        self.cursor = None
        self._failed = False
        self._undo_history: list[tuple[Any, ...]] = []
        self._frames: list[tuple[tuple[Any, ...] | None, dict[str, Any]]] = []
        self._render_snapshot: tuple[Any, ...] | None = None
        self._effect: dict[str, Any] = {}
        self._finish_after_animation = False

    def _snapshot(self) -> tuple[Any, ...]:
        return (
            self.player_position,
            self.facing,
            self.carrying,
            tuple(sorted(self.loose_discs)),
            tuple(sorted(self.pit_counts.items())),
        )

    def _restore(self, snapshot: tuple[Any, ...]) -> None:
        position, facing, carrying, loose, pits = snapshot
        self.player_position = tuple(position)
        self.facing = str(facing)
        self.carrying = bool(carrying)
        self.loose_discs = set(loose)
        self.pit_counts = dict(pits)

    def _queue(self, frames: list[tuple[tuple[Any, ...] | None, dict[str, Any]]], *, finish: bool = False) -> None:
        self._frames = frames
        self._finish_after_animation = finish
        self._advance_frame()

    def _advance_frame(self) -> None:
        self._render_snapshot, self._effect = self._frames.pop(0)
        if self._frames:
            return
        if self._finish_after_animation:
            self.next_level()
        self.complete_action()

    def _spend_step(self) -> None:
        self.steps += 1

    def _check_failure(self, won: bool = False) -> None:
        if not won and self.steps >= self.step_budget:
            self._failed = True
            self.lose()

    def _target(self) -> tuple[int, int]:
        delta = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[self.facing]
        return self.player_position[0] + delta[0], self.player_position[1] + delta[1]

    def _passable(self, cell: tuple[int, int]) -> bool:
        if cell in self.loose_discs:
            return False
        return cell in self.floor or self.pit_counts.get(cell) == 2

    def _resolve_move(self, action: int) -> None:
        old = self._snapshot()
        self._undo_history.append(old)
        self.facing = FACING[action]
        dr, dc = DELTAS[action]
        target = (self.player_position[0] + dr, self.player_position[1] + dc)
        moved = self._passable(target)
        if moved:
            self.player_position = target
        self._spend_step()
        won = moved and target == self.exit_cell
        self._check_failure(won)
        new = self._snapshot()
        if won:
            frames = [(new, {"exit_color": color}) for color in (EXIT, WHITE, EXIT, WHITE, EXIT)]
            self._queue(frames, finish=True)
        elif not moved:
            frames = [(new, {}), (new, {"bump": (dr, dc), "pulse_cell": target, "pulse_color": ACCENT}), (new, {})]
            self._queue(frames)
        else:
            self._render_snapshot = None
            self._effect = {}
            self.complete_action()

    def _resolve_space(self) -> None:
        old = self._snapshot()
        target = self._target()
        valid = False
        if not self.carrying:
            if target in self.loose_discs:
                self.loose_discs.remove(target)
                self.carrying = True
                valid = True
            elif self.pit_counts.get(target, 0) in (1, 2) and target in self.pit_counts:
                self.pit_counts[target] -= 1
                self.carrying = True
                valid = True
        elif target in self.pit_counts and self.pit_counts[target] in (0, 1):
            self.pit_counts[target] += 1
            self.carrying = False
            valid = True

        self._spend_step()
        self._check_failure()
        if valid:
            self._undo_history.append(old)
            new = self._snapshot()
            pr, pc = self.player_position
            tr, tc = target
            if old[2]:
                edge = ((tr + 2 * pr) / 3, (tc + 2 * pc) / 3)
                middle = ((2 * tr + pr) / 3, (2 * tc + pc) / 3)
            else:
                edge = ((2 * tr + pr) / 3, (2 * tc + pc) / 3)
                middle = ((tr + 2 * pr) / 3, (tc + 2 * pc) / 3)
            frames = [(old, {}), (old, {"transfer": edge}), (new, {"transfer": middle}), (new, {})]
            self._queue(frames)
            return

        effect: dict[str, Any]
        if target in self.loose_discs and self.carrying:
            effect = {"pulse_cell": target, "pulse_color": ERROR, "carry_error": True}
        elif target in self.pit_counts or target in self.loose_discs:
            effect = {"pulse_cell": target, "pulse_color": ERROR}
        else:
            effect = {"nose_error": True}
        current = self._snapshot()
        self._queue([(current, effect), (current, {})])

    def _resolve_undo(self) -> None:
        self._spend_step()
        if self._undo_history:
            self._restore(self._undo_history.pop())
        self._check_failure()
        self._render_snapshot = None
        self._effect = {}
        self.complete_action()

    def step(self) -> None:
        if self._frames:
            self._advance_frame()
            return
        action = int(getattr(self.action.id, "value", self.action.id))
        if action == 0:
            self._render_snapshot = None
            self._effect = {}
            self.complete_action()
        elif action in DELTAS:
            self._resolve_move(action)
        elif action == SPACE:
            self._resolve_space()
        elif action == CLICK:
            self._spend_step()
            self._check_failure()
            self._render_snapshot = None
            self._effect = {}
            self.complete_action()
        elif action == UNDO:
            self._resolve_undo()
        else:
            self.complete_action()
