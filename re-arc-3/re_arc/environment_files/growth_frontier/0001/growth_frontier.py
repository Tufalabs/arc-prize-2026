from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GAME_ID = "growth_frontier-0001"
VIEW_SIZE = 64
CELL_SIZE = 7

COLOR_WHITE = 0
COLOR_OPEN = 1
COLOR_CONNECTOR = 2
COLOR_INACTIVE = 4
COLOR_BACKGROUND = 5
COLOR_FOUR = 6
COLOR_INVALID = 8
COLOR_ONE = 9
COLOR_SOURCE = 11
COLOR_THREE = 12
COLOR_TWO = 14

RESULT_COLORS = {1: COLOR_ONE, 2: COLOR_TWO, 3: COLOR_THREE, 4: COLOR_FOUR}
PIP_INK = {1: COLOR_WHITE, 2: COLOR_BACKGROUND, 3: COLOR_BACKGROUND, 4: COLOR_WHITE}
RING_PIPS = {1: ((3, 1),), 2: ((1, 3), (5, 3)), 3: ((3, 1), (1, 5), (5, 5)), 4: ((3, 1), (5, 3), (3, 5), (1, 3))}
CENTER_PIPS = {1: ((3, 3),), 2: ((2, 3), (4, 3)), 3: ((3, 2), (2, 4), (4, 4)), 4: ((2, 2), (4, 2), (2, 4), (4, 4))}
BRACKETS = ((0, 1), (1, 0), (5, 0), (6, 1), (0, 5), (1, 6), (5, 6), (6, 5))
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    solution: tuple[tuple[int, int], ...]

    @property
    def budget(self) -> int:
        return 6 * len(self.solution)


LEVEL_SPECS = (
    LevelSpec("First Contact", ("..1", "S11"), ((1, 1), (2, 1), (2, 0))),
    LevelSpec("Waiting Makes Two", ("S1..", "1211", "..21"), ((1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (2, 2))),
    LevelSpec(
        "Three-Sided Bay",
        ("1121", "S312", "1122"),
        ((0, 0), (1, 0), (0, 2), (1, 2), (1, 1), (2, 1), (2, 0), (2, 2), (3, 0), (3, 1), (3, 2)),
    ),
    LevelSpec(
        "Open Commitments",
        ("..1111.", ".11O121", "SO33222", ".111.O1", ".122231", "...11.."),
        (
            (1, 2),
            (1, 1),
            (2, 1),
            (2, 0),
            (3, 0),
            (4, 0),
            (4, 1),
            (1, 3),
            (2, 3),
            (3, 3),
            (2, 2),
            (3, 1),
            (3, 2),
            (5, 0),
            (5, 1),
            (4, 2),
            (6, 1),
            (5, 2),
            (6, 2),
            (1, 4),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
            (4, 4),
            (6, 3),
            (6, 4),
            (5, 3),
            (5, 4),
        ),
    ),
    LevelSpec(
        "Four Around",
        ("..1111..", ".112O21.", "11342121", "S211.O41", ".1223111", "..123221", "...1122."),
        (
            (0, 2),
            (1, 2),
            (1, 1),
            (2, 1),
            (2, 0),
            (3, 0),
            (3, 1),
            (1, 3),
            (1, 4),
            (2, 3),
            (3, 3),
            (2, 2),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (5, 2),
            (4, 2),
            (3, 2),
            (6, 1),
            (6, 2),
            (7, 2),
            (7, 3),
            (7, 4),
            (6, 4),
            (5, 4),
            (5, 3),
            (6, 3),
            (7, 5),
            (6, 5),
            (5, 5),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 6),
            (4, 5),
            (4, 4),
            (5, 6),
            (6, 6),
        ),
    ),
    LevelSpec(
        "Offset Crosses",
        (".112111.", "11212221", "122.O412", "S34O1131", "1111.241", ".12311O1", "..112222"),
        (
            (0, 2),
            (0, 1),
            (1, 1),
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 1),
            (3, 0),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (6, 2),
            (1, 2),
            (2, 2),
            (0, 4),
            (1, 4),
            (2, 4),
            (1, 3),
            (3, 4),
            (3, 3),
            (2, 3),
            (5, 3),
            (5, 2),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (6, 3),
            (1, 5),
            (2, 5),
            (2, 6),
            (3, 6),
            (3, 5),
            (4, 5),
            (5, 5),
            (5, 4),
            (7, 5),
            (6, 5),
            (6, 4),
            (4, 6),
            (5, 6),
            (6, 6),
            (7, 6),
        ),
    ),
    LevelSpec(
        "Whole Frontier",
        (".112111.", ".1212221", "11314O22", "122.1141", "S24O2.O1", "12112112", ".1234O32", "..111112"),
        (
            (0, 3),
            (0, 2),
            (1, 2),
            (1, 1),
            (1, 0),
            (2, 0),
            (2, 1),
            (3, 1),
            (3, 0),
            (4, 0),
            (4, 1),
            (3, 2),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (6, 0),
            (6, 1),
            (6, 2),
            (7, 1),
            (7, 2),
            (7, 3),
            (2, 2),
            (1, 3),
            (2, 3),
            (1, 4),
            (0, 5),
            (1, 5),
            (2, 5),
            (3, 5),
            (3, 4),
            (2, 4),
            (4, 3),
            (4, 2),
            (7, 4),
            (4, 4),
            (4, 5),
            (5, 5),
            (6, 5),
            (7, 5),
            (6, 4),
            (6, 3),
            (1, 6),
            (2, 6),
            (2, 7),
            (3, 7),
            (3, 6),
            (4, 7),
            (5, 7),
            (5, 6),
            (4, 6),
            (6, 7),
            (6, 6),
            (7, 6),
            (7, 7),
        ),
    ),
)


def _level(spec: LevelSpec) -> Level:
    canvas = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level(sprites=[canvas], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name)


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


class GrowthFrontier(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.board_sprite: Sprite | None = None
        self.spec = LEVEL_SPECS[0]
        self.playable: set[tuple[int, int]] = set()
        self.targets: dict[tuple[int, int], int] = {}
        self.open_cells: set[tuple[int, int]] = set()
        self.source = (0, 0)
        self.active: set[tuple[int, int]] = set()
        self.result_count: dict[tuple[int, int], int] = {}
        self.remaining_steps = 1
        self.step_budget = 1
        self.undo_history: list[tuple[set[tuple[int, int]], dict[tuple[int, int], int]]] = []
        self.animation_phases: list[str] = []
        self.animation_cell: tuple[int, int] | None = None
        self.changed_previews: set[tuple[int, int]] = set()
        self.feedback_point: tuple[int, int] | None = None
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND)
        super().__init__(
            GAME_ID,
            [_level(spec) for spec in LEVEL_SPECS],
            camera,
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )

    def on_set_level(self, _level: Level) -> None:
        self.spec = self.current_level.get_data("spec")
        self.board_sprite = self.current_level.get_sprites_by_tag("board")[0]
        self.playable = set()
        self.targets = {}
        self.open_cells = set()
        for y, row in enumerate(self.spec.rows):
            for x, symbol in enumerate(row):
                cell = (x, y)
                if symbol == ".":
                    continue
                self.playable.add(cell)
                if symbol == "S":
                    self.source = cell
                elif symbol == "O":
                    self.open_cells.add(cell)
                else:
                    self.targets[cell] = int(symbol)
        self.active = {self.source}
        self.result_count = {}
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self.undo_history = []
        self.animation_phases = []
        self.animation_cell = None
        self.changed_previews = set()
        self.feedback_point = None
        self._render("settled")

    @property
    def origin(self) -> tuple[int, int]:
        width = max(len(row) for row in self.spec.rows)
        height = len(self.spec.rows)
        return (VIEW_SIZE - CELL_SIZE * width) // 2, (VIEW_SIZE - CELL_SIZE * height) // 2

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = cell
        return [(x + dx, y + dy) for dx, dy in N4]

    def _neighbor_count(self, cell: tuple[int, int], active: set[tuple[int, int]] | None = None) -> int:
        support = self.active if active is None else active
        return sum(neighbor in support for neighbor in self._neighbors(cell))

    def _frontier_counts(self, active: set[tuple[int, int]] | None = None) -> dict[tuple[int, int], int]:
        support = self.active if active is None else active
        return {
            cell: self._neighbor_count(cell, support)
            for cell in self.playable - support
            if self._neighbor_count(cell, support) > 0
        }

    def _slot(self, cell: tuple[int, int]) -> tuple[int, int]:
        ox, oy = self.origin
        return ox + CELL_SIZE * cell[0], oy + CELL_SIZE * cell[1]

    def _draw_connectors(self, frame: np.ndarray, active: set[tuple[int, int]]) -> None:
        for x, y in active:
            x0, y0 = self._slot((x, y))
            if (x + 1, y) in active:
                frame[y0 + 2 : y0 + 5, x0 + 6 : x0 + 8] = COLOR_CONNECTOR
            if (x, y + 1) in active:
                frame[y0 + 6 : y0 + 8, x0 + 2 : x0 + 5] = COLOR_CONNECTOR

    def _draw_pips(
        self, frame: np.ndarray, x0: int, y0: int, count: int, *, center: bool, ink: int | None = None
    ) -> None:
        color = PIP_INK[count] if ink is None else ink
        locations = CENTER_PIPS[count] if center else RING_PIPS[count]
        for px, py in locations:
            frame[y0 + py, x0 + px] = color

    def _draw_cell(
        self,
        frame: np.ndarray,
        cell: tuple[int, int],
        active: set[tuple[int, int]],
        frontier: dict[tuple[int, int], int],
        *,
        flash_preview: bool = False,
    ) -> None:
        x0, y0 = self._slot(cell)
        if cell == self.source:
            frame[y0 + 1 : y0 + 6, x0 + 1 : x0 + 6] = COLOR_SOURCE
            frame[y0 + 3, x0 + 3] = COLOR_WHITE
            return

        if cell in self.open_cells:
            frame[y0 + 1, x0 + 2 : x0 + 5] = COLOR_OPEN
            frame[y0 + 5, x0 + 2 : x0 + 5] = COLOR_OPEN
            frame[y0 + 2 : y0 + 5, x0 + 1] = COLOR_OPEN
            frame[y0 + 2 : y0 + 5, x0 + 5] = COLOR_OPEN
        else:
            target = self.targets[cell]
            ring_color = RESULT_COLORS[target]
            frame[y0 + 1, x0 + 1 : x0 + 6] = ring_color
            frame[y0 + 5, x0 + 1 : x0 + 6] = ring_color
            frame[y0 + 1 : y0 + 6, x0 + 1] = ring_color
            frame[y0 + 1 : y0 + 6, x0 + 5] = ring_color
            self._draw_pips(frame, x0, y0, target, center=False)

        count = self.result_count.get(cell) if cell in active else frontier.get(cell)
        center_color = RESULT_COLORS[count] if count is not None else COLOR_INACTIVE
        frame[y0 + 2 : y0 + 5, x0 + 2 : x0 + 5] = center_color
        if count is not None:
            self._draw_pips(frame, x0, y0, count, center=True, ink=COLOR_WHITE if flash_preview else None)
        if cell in frontier:
            for px, py in BRACKETS:
                frame[y0 + py, x0 + px] = COLOR_WHITE

    def _draw_budget(self, frame: np.ndarray) -> None:
        left, width = 8, 48
        frame[1:3, left : left + width] = COLOR_INACTIVE
        filled = int(width * self.remaining_steps / max(1, self.step_budget))
        if filled:
            color = COLOR_SOURCE if self.remaining_steps > self.step_budget // 6 else COLOR_INVALID
            frame[1:3, left : left + filled] = color

    def _render(self, phase: str) -> None:
        if self.board_sprite is None:
            return
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8)
        clicked = self.animation_cell
        old_active = self.active - ({clicked} if clicked is not None else set())
        connector_active = old_active if phase in {"seed", "center_flash"} else self.active
        visible_active = old_active if phase == "seed" else self.active
        frontier_active = old_active if phase == "seed" else self.active
        frontier = self._frontier_counts(frontier_active)
        self._draw_connectors(frame, connector_active)
        for cell in self.playable:
            self._draw_cell(
                frame,
                cell,
                visible_active,
                frontier,
                flash_preview=phase == "center_flash" and cell in self.changed_previews,
            )

        hint_index = len(self.active) - 1
        if phase == "settled" and hint_index < len(self.spec.solution):
            hint = self.spec.solution[hint_index]
            if hint in frontier:
                x0, y0 = self._slot(hint)
                frame[y0, x0 + 2 : x0 + 5] = COLOR_SOURCE
                frame[y0 + 6, x0 + 2 : x0 + 5] = COLOR_SOURCE
                frame[y0 + 2 : y0 + 5, x0] = COLOR_SOURCE
                frame[y0 + 2 : y0 + 5, x0 + 6] = COLOR_SOURCE

        self._draw_budget(frame)
        if phase == "seed" and clicked is not None:
            x0, y0 = self._slot(clicked)
            for px, py in BRACKETS:
                frame[y0 + py, x0 + px] = COLOR_BACKGROUND
            frame[y0 + 2 : y0 + 5, x0 + 2 : x0 + 5] = COLOR_INACTIVE
            frame[y0 + 3, x0 + 3] = RESULT_COLORS[self.result_count[clicked]]
        elif phase.startswith("mismatch") and clicked is not None:
            x0, y0 = self._slot(clicked)
            for px, py in ((1, 1), (5, 1), (1, 5), (5, 5)):
                frame[y0 + py, x0 + px] = COLOR_INVALID
        elif phase == "active_feedback" and self.feedback_point is not None:
            x0, y0 = self._slot(self.feedback_point)
            frame[y0 + 1, x0 + 1 : x0 + 6] = COLOR_CONNECTOR
            frame[y0 + 5, x0 + 1 : x0 + 6] = COLOR_CONNECTOR
            frame[y0 + 1 : y0 + 6, x0 + 1] = COLOR_CONNECTOR
            frame[y0 + 1 : y0 + 6, x0 + 5] = COLOR_CONNECTOR
        elif phase == "contact_feedback" and self.feedback_point is not None:
            x0, y0 = self._slot(self.feedback_point)
            frame[y0 + 3, x0 : x0 + 2] = COLOR_INVALID
            frame[y0 + 3, x0 + 5 : x0 + 7] = COLOR_INVALID
            frame[y0 : y0 + 2, x0 + 3] = COLOR_INVALID
            frame[y0 + 5 : y0 + 7, x0 + 3] = COLOR_INVALID
        elif phase == "outside_feedback" and self.feedback_point is not None:
            px, py = self.feedback_point
            left = min(61, max(0, px - 1))
            top = min(61, max(0, py - 1))
            frame[top, left : left + 3] = COLOR_INVALID
            frame[top + 2, left : left + 3] = COLOR_INVALID
            frame[top : top + 3, left] = COLOR_INVALID
            frame[top : top + 3, left + 2] = COLOR_INVALID

        if phase == "completion_white":
            mask = np.zeros_like(frame, dtype=bool)
            ox, oy = self.origin
            board_width = CELL_SIZE * max(len(row) for row in self.spec.rows)
            board_height = CELL_SIZE * len(self.spec.rows)
            board_pixels = frame[oy : oy + board_height, ox : ox + board_width]
            mask[oy : oy + board_height, ox : ox + board_width] = board_pixels != COLOR_BACKGROUND
            boundary = np.zeros_like(mask)
            boundary[1:, :] |= mask[1:, :] & ~mask[:-1, :]
            boundary[:-1, :] |= mask[:-1, :] & ~mask[1:, :]
            boundary[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]
            boundary[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]
            frame[boundary] = COLOR_WHITE

        self.board_sprite.pixels = frame

    def _finish_animation(self) -> None:
        won = self.active == self.playable and all(
            self.result_count.get(cell) == target for cell, target in self.targets.items()
        )
        self.animation_cell = None
        self.changed_previews = set()
        self.feedback_point = None
        if won:
            self.next_level()
        self.complete_action()

    def _play_animation(self) -> None:
        phase = self.animation_phases.pop(0)
        self._render(phase)
        if not self.animation_phases:
            self._finish_animation()

    def _queue_feedback(self, phase: str, point: tuple[int, int], repeats: int) -> None:
        self.feedback_point = point
        self.animation_phases = [phase] * repeats + ["settled"]
        self._play_animation()

    def _hit_test(self, px: int, py: int) -> tuple[int, int] | None:
        ox, oy = self.origin
        width = CELL_SIZE * max(len(row) for row in self.spec.rows)
        height = CELL_SIZE * len(self.spec.rows)
        if not (ox <= px < ox + width and oy <= py < oy + height):
            return None
        return (px - ox) // CELL_SIZE, (py - oy) // CELL_SIZE

    def _spend_action(self) -> bool:
        self.remaining_steps -= 1
        if self.remaining_steps > 0:
            return True
        self._render("settled")
        self.lose()
        self.complete_action()
        return False

    def _activate(self, cell: tuple[int, int], count: int) -> None:
        before = self._frontier_counts()
        self.undo_history.append((set(self.active), dict(self.result_count)))
        self.active.add(cell)
        self.result_count[cell] = count
        after = self._frontier_counts()
        self.changed_previews = {candidate for candidate, value in after.items() if before.get(candidate) != value}
        self.animation_cell = cell
        mismatch = cell in self.targets and self.targets[cell] != count
        won = self.active == self.playable and all(
            self.result_count.get(target) == value for target, value in self.targets.items()
        )
        self.animation_phases = ["seed", "center_flash", "settled"]
        if mismatch:
            self.animation_phases += ["mismatch_a", "mismatch_b", "settled"]
        if won:
            self.animation_phases += ["completion_white", "completion_orig", "completion_white", "completion_orig"]
        self._play_animation()

    def _click(self) -> None:
        px = int(self.action.data.get("x", 0))
        py = int(self.action.data.get("y", 0))
        cell = self._hit_test(px, py)
        if cell is None or cell not in self.playable:
            self._queue_feedback("outside_feedback", (px, py), 2)
            return
        if cell in self.active:
            self._queue_feedback("active_feedback", cell, 1)
            return
        count = self._neighbor_count(cell)
        if count == 0:
            self._queue_feedback("contact_feedback", cell, 2)
            return
        self._activate(cell, count)

    def _undo(self) -> None:
        if self.undo_history:
            self.active, self.result_count = self.undo_history.pop()
        self.animation_phases = []
        self.animation_cell = None
        self.changed_previews = set()
        self.feedback_point = None
        self._render("settled")
        self.complete_action()

    def step(self) -> None:
        if self.animation_phases:
            self._play_animation()
            return
        action = _action_id(self.action.id)
        if action == 0:
            self._render("settled")
            self.complete_action()
            return
        if action == 7:
            self._undo()
            return
        if not self._spend_action():
            return
        if action == 6:
            self._click()
            return
        self._render("settled")
        self.complete_action()
