from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

GAME_ID = "contact_alchemy-0001"
VIEW_SIZE = 64
PITCH = 7
TILE_SIZE = 6

ACTION_CLICK = 6
ACTION_UNDO = 7

COLOR_WHITE = 0
COLOR_BACKGROUND = 4
COLOR_RED = 8
COLOR_BLUE = 9
COLOR_LIGHT_BLUE = 10
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_GREEN = 14
COLOR_PURPLE = 15
COLOR_GRAY = 5


@dataclass(frozen=True)
class LevelSpec:
    title: str
    current: tuple[str, ...]
    target: tuple[str, ...]
    baseline_actions: int

    @property
    def budget(self) -> int:
        return 6 * self.baseline_actions


LEVEL_SPECS = (
    LevelSpec("First Transmutation", ("A",), ("V",), 1),
    LevelSpec("Unequal Witnesses", (".A.", "VAV"), (".A.", "AVA"), 1),
    LevelSpec(
        "Keep the Heart",
        ("..V..", "..A..", "VAAAV", "..A..", "..V.."),
        ("..V..", "..V..", "VVAVV", "..V..", "..V.."),
        2,
    ),
    LevelSpec(
        "Diamond of Stamps",
        ("..V.A..", "..A.V..", "..AVV..", "..V.A..", "..VAA..", "..V.A..", "..A.V.."),
        ("..V.A..", "..V.A..", "..AAV..", "..A.V..", "..VVA..", "..A.V..", "..A.V.."),
        8,
    ),
    LevelSpec(
        "The Keystone",
        (
            "..V.A.V..",
            "..A.V.A..",
            "..AAVAA..",
            "..A.A.V..",
            "AVVAAVVVA",
            "..A.A.V..",
            "..AAVAA..",
            "..A.V.A..",
            "..V.A.V..",
        ),
        (
            "..V.A.V..",
            "..V.A.V..",
            "..AVVAA..",
            "..A.A.A..",
            "AAAVAVVAA",
            "..A.V.A..",
            "..AAVAA..",
            "..V.A.V..",
            "..V.A.V..",
        ),
        25,
    ),
    LevelSpec(
        "Borrowed Seams",
        (
            ".V.A.V.A.",
            ".A.V.A.V.",
            ".AAVAAVV.",
            ".A.A.V.A.",
            ".VAAAVVA.",
            ".A.V.V.V.",
            ".AVVAAVV.",
            ".A.V.A.V.",
            ".V.A.V.A.",
        ),
        (
            ".V.A.V.A.",
            ".V.A.V.A.",
            ".AAVAAAV.",
            ".V.V.A.V.",
            ".VVAVVVA.",
            ".A.V.A.V.",
            ".AVVVAVV.",
            ".V.A.V.A.",
            ".V.A.V.V.",
        ),
        41,
    ),
    LevelSpec(
        "Bent Crucible",
        (
            ".V.A.V.A.",
            ".A.V.A.V.",
            ".AAVAAVV.",
            "AA.A.V.A.",
            "AVAAAVVAA",
            ".A.V.V.VA",
            ".AVVAAVV.",
            ".A.V.A.V.",
            ".V.A.V.A.",
        ),
        (
            ".V.A.V.A.",
            ".V.A.V.A.",
            ".AAVAAAV.",
            "VA.A.A.V.",
            "AVVAAVAAV",
            ".A.V.A.VV",
            ".AVVVAVV.",
            ".V.A.V.V.",
            ".V.A.V.A.",
        ),
        55,
    ),
)


def _decode(rows: tuple[str, ...]) -> dict[tuple[int, int], int]:
    return {(r, c): int(symbol == "V") for r, row in enumerate(rows) for c, symbol in enumerate(row) if symbol != "."}


def _origin(rows: int, columns: int) -> tuple[int, int]:
    return (VIEW_SIZE - (PITCH * columns - 1)) // 2, (VIEW_SIZE - (PITCH * rows - 1)) // 2


def _tile_pixels(face: int, target: int) -> np.ndarray:
    frame_color = COLOR_ORANGE if target == 0 else COLOR_BLUE
    pixels = np.full((TILE_SIZE, TILE_SIZE), frame_color, dtype=np.int8)
    if target == 0:
        for y, x in ((0, 2), (0, 3), (5, 2), (5, 3), (2, 0), (3, 0), (2, 5), (3, 5)):
            pixels[y, x] = COLOR_YELLOW
    else:
        for y, x in ((0, 0), (0, 5), (5, 0), (5, 5)):
            pixels[y, x] = COLOR_LIGHT_BLUE

    pixels[1:5, 1:5] = COLOR_YELLOW if face == 0 else COLOR_PURPLE
    if face == 0:
        mark = ((1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3))
        mark_color = COLOR_BACKGROUND
    else:
        mark = ((1, 2), (1, 3), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3))
        mark_color = COLOR_WHITE
    for y, x in mark:
        pixels[y, x] = mark_color
    return pixels


class ContactAlchemyView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ContactAlchemy | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_BACKGROUND
        for cell in game.present:
            x, y = game._cell_top_left(cell)
            frame[y : y + TILE_SIZE, x : x + TILE_SIZE] = _tile_pixels(game.faces[cell], game.targets[cell])

        self._draw_feedback(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _outline(self, frame: np.ndarray, game: ContactAlchemy, cell: tuple[int, int], color: int) -> None:
        x, y = game._cell_top_left(cell)
        frame[y, x : x + TILE_SIZE] = color
        frame[y + TILE_SIZE - 1, x : x + TILE_SIZE] = color
        frame[y : y + TILE_SIZE, x] = color
        frame[y : y + TILE_SIZE, x + TILE_SIZE - 1] = color

    def _draw_feedback(self, frame: np.ndarray, game: ContactAlchemy) -> None:
        if game.feedback_kind == "valid" and game.clicked_cell is not None:
            if game.feedback_phase == 0:
                self._outline(frame, game, game.clicked_cell, COLOR_WHITE)
            elif game.feedback_phase == 1:
                self._draw_waves(frame, game)
            elif game.feedback_phase == 2:
                for cell, changed in game.considered_neighbors:
                    self._outline(frame, game, cell, COLOR_WHITE if changed else COLOR_GRAY)
            elif game.feedback_phase == 4:
                for cell in game.present:
                    self._outline(frame, game, cell, COLOR_GREEN)
        elif game.feedback_kind == "invalid" and game.invalid_point is not None:
            x, y = game.invalid_point
            radius = game.feedback_phase + 1
            for delta in range(-radius, radius + 1):
                for px, py in ((x + delta, y + delta), (x + delta, y - delta)):
                    if 0 <= px < VIEW_SIZE and 0 <= py < VIEW_SIZE - 1:
                        frame[py, px] = COLOR_RED

        if game.failed:
            for i in range(12, 52):
                frame[i, i] = COLOR_RED
                frame[i, VIEW_SIZE - 1 - i] = COLOR_RED

    def _draw_waves(self, frame: np.ndarray, game: ContactAlchemy) -> None:
        if game.clicked_cell is None or game.clicked_old_face is None:
            return
        r, c = game.clicked_cell
        sx, sy = game._cell_top_left((r, c))
        color = COLOR_ORANGE if game.clicked_old_face == 0 else COLOR_PURPLE
        for nr, nc in (cell for cell, _changed in game.considered_neighbors):
            if nr == r - 1:
                frame[sy - 1, sx + 2 : sx + 4] = color
            elif nr == r + 1:
                frame[sy + TILE_SIZE, sx + 2 : sx + 4] = color
            elif nc == c - 1:
                frame[sy + 2 : sy + 4, sx - 1] = color
            elif nc == c + 1:
                frame[sy + 2 : sy + 4, sx + TILE_SIZE] = color

    def _draw_budget(self, frame: np.ndarray, game: ContactAlchemy) -> None:
        filled = round(VIEW_SIZE * game.remaining_steps / max(1, game.step_budget))
        frame[VIEW_SIZE - 1, :] = COLOR_RED
        if filled:
            frame[VIEW_SIZE - 1, :filled] = COLOR_GREEN


def _build_level(spec: LevelSpec) -> Level:
    return Level(
        sprites=[],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.title,
        data={"rows": spec.current, "targets": spec.target, "step_budget": spec.budget},
    )


def _action_id(action_id: object) -> int:
    return int(getattr(action_id, "value", action_id))


class ContactAlchemy(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.view = ContactAlchemyView()
        self.view.game = self
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self.view]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, ACTION_CLICK, ACTION_UNDO],
            seed,
        )

    @property
    def num_levels(self) -> int:
        return len(LEVEL_SPECS)

    def on_set_level(self, level: Level) -> None:
        rows = tuple(level.get_data("rows"))
        target_rows = tuple(level.get_data("targets"))
        self.rows = len(rows)
        self.columns = len(rows[0])
        self.origin_x, self.origin_y = _origin(self.rows, self.columns)
        self.faces = _decode(rows)
        self.targets = _decode(target_rows)
        self.present = tuple(sorted(self.faces))
        self.step_budget = int(level.get_data("step_budget"))
        self.remaining_steps = self.step_budget
        self.terminal = False
        self.failed = False
        self.undo_history: list[dict[tuple[int, int], int]] = []
        self._clear_feedback()

    def _clear_feedback(self) -> None:
        self.feedback_kind: str | None = None
        self.feedback_phase = 0
        self.feedback_finished = False
        self.clicked_cell: tuple[int, int] | None = None
        self.clicked_old_face: int | None = None
        self.considered_neighbors: list[tuple[tuple[int, int], bool]] = []
        self.invalid_point: tuple[int, int] | None = None
        self.pending_win = False
        self.pending_failure = False

    def step(self) -> None:
        if self.feedback_finished:
            self._clear_feedback()
        if self.feedback_kind is not None:
            self._advance_feedback()
            return

        action_id = _action_id(self.action.id)
        if action_id == 0:
            self.complete_action()
            return

        self.remaining_steps = max(0, self.remaining_steps - 1)

        if action_id == ACTION_UNDO:
            if self.undo_history:
                self.faces = self.undo_history.pop()
            self._finish_or_fail()
            return

        if action_id != ACTION_CLICK:
            self._finish_or_fail()
            return

        display_x = int(self.action.data.get("x", 0))
        display_y = int(self.action.data.get("y", 0))
        point = self.camera.display_to_grid(display_x, display_y)
        clicked = self._cell_at_pixel(point) if point is not None else None
        if clicked is None:
            self.feedback_kind = "invalid"
            self.invalid_point = (display_x, display_y)
            self.feedback_phase = 0
            self.pending_failure = self.remaining_steps == 0
            return

        self.undo_history.append(dict(self.faces))
        old_faces = dict(self.faces)
        old = old_faces[clicked]
        neighbors = self._neighbors(clicked)
        self.faces[clicked] = 1 - old
        self.considered_neighbors = []
        for neighbor in neighbors:
            changed = old_faces[neighbor] != old
            self.faces[neighbor] = old
            self.considered_neighbors.append((neighbor, changed))

        self.clicked_cell = clicked
        self.clicked_old_face = old
        self.feedback_kind = "valid"
        self.feedback_phase = 0
        self.pending_win = self.faces == self.targets
        self.pending_failure = not self.pending_win and self.remaining_steps == 0

    def _advance_feedback(self) -> None:
        if self.feedback_kind == "valid" and self.feedback_phase < 3:
            self.feedback_phase += 1
            if self.feedback_phase == 3 and not self.pending_win:
                self._complete_feedback()
            return

        if self.feedback_kind == "valid" and self.pending_win:
            self.feedback_phase = 4
            self.terminal = True
            self.next_level()
            self.feedback_finished = True
            self.complete_action()
            return

        if self.feedback_kind == "invalid" and self.feedback_phase < 2:
            self.feedback_phase += 1
            if self.feedback_phase == 2:
                self._complete_feedback()
            return

    def _complete_feedback(self) -> None:
        if self.pending_failure:
            self.failed = True
            self.lose()
        self.feedback_finished = True
        self.complete_action()

    def _finish_or_fail(self) -> None:
        if self.remaining_steps == 0:
            self.failed = True
            self.lose()
        self.complete_action()

    def _cell_top_left(self, cell: tuple[int, int]) -> tuple[int, int]:
        row, column = cell
        return self.origin_x + PITCH * column, self.origin_y + PITCH * row

    def _cell_at_pixel(self, point: tuple[int, int]) -> tuple[int, int] | None:
        x, y = point
        relative_x = x - self.origin_x
        relative_y = y - self.origin_y
        if relative_x < 0 or relative_y < 0:
            return None
        row, local_y = divmod(relative_y, PITCH)
        column, local_x = divmod(relative_x, PITCH)
        cell = (row, column)
        if local_x >= TILE_SIZE or local_y >= TILE_SIZE or cell not in self.faces:
            return None
        return cell

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        row, column = cell
        return [
            neighbor
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
            if neighbor in self.faces
        ]

    def _get_hidden_state(self) -> np.ndarray:
        state = [self.faces[cell] for cell in self.present]
        state.extend((self.remaining_steps, int(self.terminal), int(self.failed)))
        return np.asarray(state, dtype=np.int16)

    def _get_valid_actions(self) -> list[ActionInput]:
        actions = [ActionInput(id=GameAction.from_id(action_id)) for action_id in (1, 2, 3, 4, 5, 7)]
        for cell in self.present:
            x, y = self._cell_top_left(cell)
            actions.append(ActionInput(id=GameAction.ACTION6, data={"x": x + 2, "y": y + 2}))
        return actions
