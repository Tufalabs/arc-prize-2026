from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "mirror_run-0001"
VIEW_SIZE = 64
BOARD_SIZE = 13
BOARD_ORIGIN = 6
CELL_SIZE = 4

COLOR_MIRROR = 0
COLOR_HOUSING = 1
COLOR_FRAME = 2
COLOR_FADE = 3
COLOR_FIELD = 4
COLOR_MARGIN = 5
COLOR_IMPACT = 8
COLOR_RECEIVER = 10
COLOR_BEAM = 11
COLOR_SOURCE = 12
COLOR_ACTIVE = 14

UP = "UP"
DOWN = "DOWN"
LEFT = "LEFT"
RIGHT = "RIGHT"
SLASH = "/"
BACKSLASH = "\\"

DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
REFLECTIONS = {
    SLASH: {RIGHT: UP, LEFT: DOWN, UP: RIGHT, DOWN: LEFT},
    BACKSLASH: {RIGHT: DOWN, LEFT: UP, UP: LEFT, DOWN: RIGHT},
}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    source_direction: str
    baseline_actions: int

    @property
    def step_budget(self) -> int:
        return 6 * self.baseline_actions


LEVEL_SPECS = (
    LevelSpec(
        "First Turn",
        (
            ".............",
            ".............",
            ".....R.......",
            ".............",
            ".............",
            ".............",
            "S....\\.......",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
        ),
        RIGHT,
        1,
    ),
    LevelSpec(
        "Second Bounce",
        (
            ".............",
            ".............",
            ".............",
            ".............",
            "R.../........",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            "S...\\........",
            ".............",
            ".............",
        ),
        RIGHT,
        2,
    ),
    LevelSpec(
        "Same Face, New Side",
        (
            ".............",
            ".............",
            ".............",
            "..../..../.R.",
            ".............",
            ".............",
            ".............",
            ".............",
            "S.../........",
            ".............",
            ".............",
            "....\\..../...",
            ".............",
        ),
        RIGHT,
        1,
    ),
    LevelSpec(
        "Dark Suffix",
        (
            ".....S.......",
            ".............",
            "...\\.\\..././.",
            ".............",
            ".............",
            ".............",
            ".............",
            ".............",
            ".../.....\\...",
            ".............",
            ".............",
            "...........R.",
            ".............",
        ),
        DOWN,
        5,
    ),
    LevelSpec(
        "Reverse Fold",
        (
            ".............",
            ".............",
            ".......\\../..",
            ".............",
            ".............",
            "...../.\\.....",
            ".............",
            "S....../.....",
            ".............",
            ".....\\./../.R",
            ".............",
            ".............",
            ".............",
        ),
        RIGHT,
        5,
    ),
    LevelSpec(
        "Cross-Weave",
        (
            ".............",
            "...\\../......",
            ".............",
            ".............",
            ".............",
            "....../....\\.",
            ".R...........",
            ".............",
            "../.../......",
            ".............",
            "./.\\..\\..../.",
            ".............",
            "..S..........",
        ),
        UP,
        6,
    ),
    LevelSpec(
        "Knotted Return",
        (
            "..S.....R....",
            ".............",
            "..\\....\\/./..",
            "..\\../.......",
            ".............",
            ".............",
            "..\\.....\\....",
            ".............",
            ".............",
            "...../.\\/.\\..",
            ".............",
            ".............",
            ".............",
        ),
        DOWN,
        8,
    ),
)


def _parse_rows(rows: tuple[str, ...]) -> tuple[tuple[int, int], tuple[int, int], dict[tuple[int, int], str]]:
    if len(rows) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in rows):
        raise ValueError("Mirror Run levels must be 13x13")
    source = None
    receiver = None
    mirrors: dict[tuple[int, int], str] = {}
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            if symbol == "S":
                source = (x, y)
            elif symbol == "R":
                receiver = (x, y)
            elif symbol in (SLASH, BACKSLASH):
                mirrors[(x, y)] = symbol
    if source is None or receiver is None:
        raise ValueError("Mirror Run level is missing its source or receiver")
    return source, receiver, mirrors


def _make_level(spec: LevelSpec) -> Level:
    _parse_rows(spec.rows)
    canvas = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_MARGIN, dtype=np.int8),
        name="mirror_run_canvas",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["canvas"],
    )
    return Level(sprites=[canvas], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name)


def _action_id(action: GameAction | int) -> int:
    return int(action.value) if isinstance(action, GameAction) else int(action)


class MirrorRun(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.canvas: Sprite | None = None
        self.source = (0, 0)
        self.receiver = (0, 0)
        self.source_direction = RIGHT
        self.mirror_orientation: dict[tuple[int, int], str] = {}
        self.beam_legs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.beam_hits: list[tuple[tuple[int, int], str, str]] = []
        self.terminal_kind = "FRAME"
        self.terminal_location = (0, 0)
        self.completed = False
        self.last_clicked_mirror: tuple[int, int] | None = None
        self.old_beam_legs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.animation_phase = "SETTLED"
        self.animation_frame = 0
        self._animation_frames: list[tuple[str, np.ndarray]] = []
        self._finish_after_animation = False
        self._undo_history: list[dict[str, object]] = []
        self.step_budget = 1
        self.remaining_steps = 1
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_MARGIN, COLOR_MARGIN)
        super().__init__(
            GAME_ID,
            [_make_level(spec) for spec in LEVEL_SPECS],
            camera,
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        self.canvas = level.get_sprites_by_tag("canvas")[0]
        self.source, self.receiver, self.mirror_orientation = _parse_rows(spec.rows)
        self.source_direction = spec.source_direction
        self.step_budget = spec.step_budget
        self.remaining_steps = self.step_budget
        self.last_clicked_mirror = None
        self.old_beam_legs = []
        self.animation_phase = "SETTLED"
        self.animation_frame = 0
        self._animation_frames = []
        self._finish_after_animation = False
        self._undo_history = []
        self.completed = False
        self._recompute()
        self._show(self._render())

    def step(self) -> None:
        if self._animation_frames:
            self.animation_frame += 1
            self.animation_phase, frame = self._animation_frames.pop(0)
            self._show(frame)
            if not self._animation_frames:
                self.animation_phase = "SETTLED"
                if self._finish_after_animation:
                    self.next_level()
                self.complete_action()
            return

        action_id = _action_id(self.action.id)
        if action_id == int(GameAction.RESET.value):
            self._show(self._render())
            self.complete_action()
            return

        if action_id == int(GameAction.ACTION7.value):
            self._undo()
            return

        if action_id != int(GameAction.ACTION6.value):
            self.complete_action()
            return

        clicked = self._clicked_mirror()
        if clicked is None:
            self.complete_action()
            return

        self._rotate(clicked)

    def _clicked_mirror(self) -> tuple[int, int] | None:
        px = int(self.action.data.get("x", 0))
        py = int(self.action.data.get("y", 0))
        if not (BOARD_ORIGIN <= px <= 57 and BOARD_ORIGIN <= py <= 57):
            return None
        cell = ((px - BOARD_ORIGIN) // CELL_SIZE, (py - BOARD_ORIGIN) // CELL_SIZE)
        return cell if cell in self.mirror_orientation else None

    def _snapshot(self) -> dict[str, object]:
        return {
            "orientations": dict(self.mirror_orientation),
            "beam_legs": list(self.beam_legs),
            "beam_hits": list(self.beam_hits),
            "terminal_kind": self.terminal_kind,
            "terminal_location": self.terminal_location,
            "completed": self.completed,
        }

    def _restore(self, snapshot: dict[str, object]) -> None:
        self.mirror_orientation = dict(snapshot["orientations"])
        self.beam_legs = list(snapshot["beam_legs"])
        self.beam_hits = list(snapshot["beam_hits"])
        self.terminal_kind = str(snapshot["terminal_kind"])
        self.terminal_location = tuple(snapshot["terminal_location"])
        self.completed = bool(snapshot["completed"])
        self.last_clicked_mirror = None
        self.old_beam_legs = []
        self.animation_phase = "SETTLED"
        self.animation_frame = 0
        self._animation_frames = []
        self._finish_after_animation = False
        self._show(self._render())

    def _undo(self) -> None:
        if not self._undo_history:
            self.complete_action()
            return
        self._restore(self._undo_history.pop())
        self.remaining_steps -= 1
        self._show(self._render())
        if self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _rotate(self, cell: tuple[int, int]) -> None:
        snapshot = self._snapshot()
        self._undo_history.append(snapshot)
        old_legs = list(self.beam_legs)
        self.old_beam_legs = old_legs
        self.last_clicked_mirror = cell
        self.mirror_orientation[cell] = SLASH if self.mirror_orientation[cell] == BACKSLASH else BACKSLASH
        self._recompute()
        self.remaining_steps -= 1

        frames = [
            ("ROTATION_FLASH", self._render(legs=old_legs, beam_color=COLOR_FADE, flash_cell=cell)),
            ("ROTATION_FLASH", self._render(legs=[])),
        ]
        leg_count = len(self.beam_legs)
        legs_per_frame = max(1, ceil(leg_count / 6))
        for count in range(legs_per_frame, leg_count + legs_per_frame, legs_per_frame):
            frames.append(("BEAM_REVEAL", self._render(legs=self.beam_legs[: min(count, leg_count)])))

        if self.completed:
            frames.extend(
                [
                    ("COMPLETE", self._render(receiver_glow=True)),
                    ("COMPLETE", self._render()),
                    ("COMPLETE", self._render(receiver_glow=True)),
                    ("COMPLETE", self._render()),
                ]
            )
            self._finish_after_animation = True
        elif self.remaining_steps <= 0:
            frames.append(("IMPACT", self._render()))
            self._animation_frames = frames
            self._finish_after_animation = False
            self.lose()
        else:
            frames.extend(
                [
                    ("IMPACT", self._render(impact_pulse=True)),
                    ("IMPACT", self._render(impact_pulse=True)),
                    ("IMPACT", self._render()),
                ]
            )
            self._finish_after_animation = False
        self._animation_frames = frames
        self.animation_frame = 0
        self.animation_phase, first_frame = self._animation_frames.pop(0)
        self._show(first_frame)

    def _recompute(self) -> None:
        position = self.source
        direction = self.source_direction
        leg_start = position
        legs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        hits: list[tuple[tuple[int, int], str, str]] = []
        seen_entries: set[tuple[tuple[int, int], str]] = set()
        while True:
            dx, dy = DELTAS[direction]
            next_cell = (position[0] + dx, position[1] + dy)
            if not (0 <= next_cell[0] < BOARD_SIZE and 0 <= next_cell[1] < BOARD_SIZE):
                legs.append((leg_start, next_cell))
                self.beam_legs = legs
                self.beam_hits = hits
                self.terminal_kind = "FRAME"
                self.terminal_location = next_cell
                self.completed = False
                return
            if next_cell == self.receiver:
                legs.append((leg_start, next_cell))
                self.beam_legs = legs
                self.beam_hits = hits
                self.terminal_kind = "RECEIVER"
                self.terminal_location = next_cell
                self.completed = True
                return
            if next_cell in self.mirror_orientation:
                legs.append((leg_start, next_cell))
                key = (next_cell, direction)
                if key in seen_entries:
                    raise RuntimeError("Authored Mirror Run route contains an infinite loop")
                seen_entries.add(key)
                outgoing = REFLECTIONS[self.mirror_orientation[next_cell]][direction]
                hits.append((next_cell, direction, outgoing))
                direction = outgoing
                leg_start = next_cell
            position = next_cell

    def _show(self, frame: np.ndarray) -> None:
        if self.canvas is not None:
            self.canvas.pixels[:, :] = frame

    def _render(
        self,
        *,
        legs: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
        beam_color: int = COLOR_BEAM,
        flash_cell: tuple[int, int] | None = None,
        impact_pulse: bool = False,
        receiver_glow: bool = False,
    ) -> np.ndarray:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_MARGIN, dtype=np.int8)
        frame[BOARD_ORIGIN:58, BOARD_ORIGIN:58] = COLOR_FIELD
        frame[5, 5:59] = COLOR_FRAME
        frame[58, 5:59] = COLOR_FRAME
        frame[5:59, 5] = COLOR_FRAME
        frame[5:59, 58] = COLOR_FRAME

        shown_legs = self.beam_legs if legs is None else legs
        for start, end in shown_legs:
            x1, y1 = self._beam_point(start)
            x2, y2 = self._beam_point(end)
            frame[min(y1, y2) : max(y1, y2) + 1, min(x1, x2) : max(x1, x2) + 1] = beam_color

        if shown_legs == self.beam_legs and self.terminal_kind == "FRAME":
            self._draw_impact(frame, pulse=impact_pulse)

        self._draw_receiver(frame, active=self.completed, glow=receiver_glow)
        for cell, orientation in self.mirror_orientation.items():
            self._draw_mirror(frame, cell, orientation, flash=cell == flash_cell)
        self._draw_source(frame)
        self._draw_budget(frame)
        return frame

    def _beam_point(self, cell: tuple[int, int]) -> tuple[int, int]:
        x, y = cell
        px = 5 if x < 0 else 58 if x >= BOARD_SIZE else 7 + CELL_SIZE * x
        py = 5 if y < 0 else 58 if y >= BOARD_SIZE else 7 + CELL_SIZE * y
        return px, py

    def _draw_mirror(self, frame: np.ndarray, cell: tuple[int, int], orientation: str, *, flash: bool) -> None:
        x0 = BOARD_ORIGIN + CELL_SIZE * cell[0]
        y0 = BOARD_ORIGIN + CELL_SIZE * cell[1]
        frame[y0 : y0 + 4, x0 : x0 + 4] = COLOR_HOUSING
        if flash:
            for idx in range(4):
                frame[y0 + idx, x0 + idx] = COLOR_MIRROR
                frame[y0 + idx, x0 + 3 - idx] = COLOR_MIRROR
        elif orientation == SLASH:
            for idx in range(4):
                frame[y0 + idx, x0 + 3 - idx] = COLOR_MIRROR
        else:
            for idx in range(4):
                frame[y0 + idx, x0 + idx] = COLOR_MIRROR

    def _draw_source(self, frame: np.ndarray) -> None:
        x0 = BOARD_ORIGIN + CELL_SIZE * self.source[0]
        y0 = BOARD_ORIGIN + CELL_SIZE * self.source[1]
        frame[y0 : y0 + 4, x0 : x0 + 4] = COLOR_SOURCE
        right = {(0, 1), (1, 1), (2, 1), (2, 0), (3, 1), (2, 2)}
        turns = {RIGHT: 0, DOWN: 1, LEFT: 2, UP: 3}[self.source_direction]
        points = right
        for _ in range(turns):
            points = {(3 - py, px) for px, py in points}
        for px, py in points:
            frame[y0 + py, x0 + px] = COLOR_MIRROR

    def _draw_receiver(self, frame: np.ndarray, *, active: bool, glow: bool) -> None:
        x0 = BOARD_ORIGIN + CELL_SIZE * self.receiver[0]
        y0 = BOARD_ORIGIN + CELL_SIZE * self.receiver[1]
        ring = COLOR_ACTIVE if active else COLOR_RECEIVER
        frame[y0 : y0 + 4, x0 : x0 + 4] = ring
        center = COLOR_BEAM if active else COLOR_FIELD
        frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = center
        if active and glow:
            if y0 > BOARD_ORIGIN:
                frame[y0 - 1, x0 : x0 + 4] = COLOR_ACTIVE
            if y0 + 4 < 58:
                frame[y0 + 4, x0 : x0 + 4] = COLOR_ACTIVE
            if x0 > BOARD_ORIGIN:
                frame[y0 : y0 + 4, x0 - 1] = COLOR_ACTIVE
            if x0 + 4 < 58:
                frame[y0 : y0 + 4, x0 + 4] = COLOR_ACTIVE

    def _draw_impact(self, frame: np.ndarray, *, pulse: bool) -> None:
        x, y = self._beam_point(self.terminal_location)
        outer = COLOR_SOURCE if pulse else COLOR_IMPACT
        if x in (5, 58):
            frame[max(5, y - 1) : min(59, y + 2), x] = outer
        else:
            frame[y, max(5, x - 1) : min(59, x + 2)] = outer
        frame[y, x] = COLOR_SOURCE

    def _draw_budget(self, frame: np.ndarray) -> None:
        width = 52
        filled = max(0, min(width, round(width * self.remaining_steps / max(1, self.step_budget))))
        frame[61:63, 6:58] = COLOR_FRAME
        if filled:
            frame[61:63, 6 : 6 + filled] = COLOR_BEAM

    def _get_hidden_state(self) -> np.ndarray:
        bits = [1 if self.mirror_orientation[cell] == BACKSLASH else 0 for cell in sorted(self.mirror_orientation)]
        return np.asarray([self.level_index, self.remaining_steps, *bits], dtype=np.int16)
