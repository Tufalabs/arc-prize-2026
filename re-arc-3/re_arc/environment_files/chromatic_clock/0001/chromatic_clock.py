from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "chromatic_clock-0001"
VIEW_SIZE = 64
BACKGROUND = 5

PHASE_COLORS = (8, 14, 9, 11, 15, 12, 6, 10)
PAD_ORIGINS = ((6, 1), (10, 3), (11, 6), (10, 9), (6, 11), (2, 9), (1, 6), (2, 3))
SYMBOL_PIXELS = (
    ((1, 1),),
    ((1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 1), (2, 1)),
    ((0, 0), (1, 1), (2, 2)),
    ((2, 0), (1, 1), (0, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2)),
    ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2)),
    ((0, 0), (2, 0), (1, 1), (0, 2), (2, 2)),
)
NOTCH_PIXELS = (
    ((6, 1), (7, 1), (8, 1)),
    ((11, 3), (12, 3), (12, 4)),
    ((13, 6), (13, 7), (13, 8)),
    ((12, 10), (12, 11), (11, 11)),
    ((6, 13), (7, 13), (8, 13)),
    ((2, 10), (2, 11), (3, 11)),
    ((1, 6), (1, 7), (1, 8)),
    ((2, 3), (3, 3), (2, 4)),
)
HUB_MARKERS = ((7, 6), (8, 6), (8, 7), (8, 8), (7, 8), (6, 8), (6, 7), (6, 6))
RAYS = (
    ((7, 5), (7, 4)),
    ((9, 5), (10, 5)),
    ((9, 7), (10, 7)),
    ((9, 9), (10, 9)),
    ((7, 9), (7, 10)),
    ((5, 9), (4, 9)),
    ((5, 7), (4, 7)),
    ((5, 5), (4, 5)),
)
CHECK_PIXELS = ((6, 7), (7, 8), (8, 7), (8, 6))
JAM_X_PIXELS = ((6, 6), (8, 6), (7, 7), (6, 8), (8, 8))

MODULE_ORIGINS = {chr(ord("A") + row * 4 + col): (1 + col * 16, 1 + row * 20) for row in range(3) for col in range(4)}


@dataclass(frozen=True)
class ClockLevelSpec:
    name: str
    rows: tuple[str, str, str]
    phases: tuple[tuple[str, int], ...]
    targets: tuple[tuple[str, int], ...]
    baseline: int


LEVEL_SPECS = (
    ClockLevelSpec("One Tick", ("....", ".F..", "...."), (("F", 0),), (("F", 1),), 1),
    ClockLevelSpec(
        "Distant Echo", (".B..", "E...", "..K."), (("B", 1), ("K", 1), ("E", 7)), (("B", 6), ("K", 6), ("E", 0)), 6
    ),
    ClockLevelSpec("Vacate Seven", ("....", ".FG.", "...."), (("F", 6), ("G", 7)), (("F", 1), ("G", 2)), 6),
    ClockLevelSpec(
        "Safe Collision",
        ("AB.D", ".F..", "I..L"),
        (("A", 0), ("B", 1), ("D", 3), ("F", 5), ("I", 6), ("L", 7)),
        (("A", 7), ("B", 7), ("D", 0), ("F", 1), ("I", 3), ("L", 5)),
        27,
    ),
    ClockLevelSpec(
        "Break the Checks",
        (".BC.", "E..H", "I..L"),
        (("B", 1), ("C", 3), ("E", 4), ("H", 5), ("I", 6), ("L", 7)),
        (("B", 1), ("C", 2), ("E", 3), ("H", 4), ("I", 5), ("L", 7)),
        44,
    ),
    ClockLevelSpec(
        "Two Rendezvous",
        ("A.CD", ".F.H", ".J.L"),
        (("A", 0), ("C", 1), ("D", 2), ("F", 3), ("H", 4), ("J", 6), ("L", 7)),
        (("A", 2), ("C", 3), ("D", 3), ("F", 4), ("H", 5), ("J", 5), ("L", 0)),
        47,
    ),
    ClockLevelSpec(
        "No Empty Hour",
        ("AB.D", "E.GH", ".J.L"),
        (("A", 0), ("B", 1), ("D", 2), ("E", 3), ("G", 4), ("H", 5), ("J", 6), ("L", 7)),
        (("A", 1), ("B", 2), ("D", 3), ("E", 4), ("G", 4), ("H", 5), ("J", 5), ("L", 7)),
        52,
    ),
)


def _plate_mask() -> np.ndarray:
    mask = np.zeros((15, 15), dtype=bool)
    for v in range(15):
        if v in (1, 13):
            mask[v, 5:10] = True
        elif v in (2, 12):
            mask[v, 3:12] = True
        elif v in (3, 11):
            mask[v, 2:13] = True
        elif 4 <= v <= 10:
            mask[v, 1:14] = True
    return mask


PLATE_MASK = _plate_mask()


def _dial_base(target: int) -> np.ndarray:
    dial = np.full((15, 15), -1, dtype=np.int8)
    padded = np.pad(PLATE_MASK, 1)
    surrounded = padded[1:-1, 2:] & padded[1:-1, :-2] & padded[2:, 1:-1] & padded[:-2, 1:-1]
    dial[PLATE_MASK] = 2
    dial[PLATE_MASK & surrounded] = 4
    for index, (pad_u, pad_v) in enumerate(PAD_ORIGINS):
        dial[pad_v : pad_v + 3, pad_u : pad_u + 3] = PHASE_COLORS[index]
        for du, dv in SYMBOL_PIXELS[index]:
            dial[pad_v + dv, pad_u + du] = BACKGROUND
    for u, v in NOTCH_PIXELS[target]:
        dial[v, u] = 7
    return dial


DIAL_BASES = tuple(_dial_base(target) for target in range(8))


def _active_modules(rows: tuple[str, str, str]) -> tuple[str, ...]:
    modules: list[str] = []
    for row_index, row in enumerate(rows):
        if len(row) != 4:
            raise ValueError("Clock level rows must be four cells wide")
        for col_index, symbol in enumerate(row):
            expected = chr(ord("A") + row_index * 4 + col_index)
            if symbol == expected:
                modules.append(symbol)
            elif symbol != ".":
                raise ValueError(f"Unexpected module {symbol!r} at slot {expected}")
    return tuple(modules)


def _make_level(spec: ClockLevelSpec) -> Level:
    active = _active_modules(spec.rows)
    phase_map = dict(spec.phases)
    target_map = dict(spec.targets)
    if set(active) != set(phase_map) or set(active) != set(target_map):
        raise ValueError(f"{spec.name}: layout and dial metadata disagree")
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), BACKGROUND, dtype=np.int8),
        name="clock_board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level(
        sprites=[board],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        data={
            "rows": spec.rows,
            "modules": active,
            "phases": spec.phases,
            "targets": spec.targets,
            "baseline": spec.baseline,
            "budget": spec.baseline * 6,
        },
        name=spec.name,
    )


class ClockDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ChromaticClock | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame

        frame[60:63, 1:63] = 2
        frame[61, 2:62] = 4
        fill = int(60 * max(0, game.remaining_steps) / max(1, game.step_budget))
        if fill:
            frame[61, 2 : 2 + fill] = 14

        if game.render_effect.startswith("success"):
            pulse_index = int(game.render_effect[-1])
            if pulse_index % 2 == 0:
                self._perimeter(frame, 14)
        elif game.mode == "JAMMED" and game.render_effect == "settle1":
            self._perimeter(frame, 13)
        elif game.render_effect == "timeout":
            self._perimeter(frame, 2)
        elif game.render_effect == "keypulse":
            frame[63, 30:33] = 2
        elif game.render_effect.startswith("ripple") and game.last_background_click is not None:
            radius = int(game.render_effect[-1])
            self._ring(frame, *game.last_background_click, radius)
        return frame

    @staticmethod
    def _perimeter(frame: np.ndarray, color: int) -> None:
        frame[0, :] = color
        frame[63, :] = color
        frame[:, 0] = color
        frame[:, 63] = color

    @staticmethod
    def _ring(frame: np.ndarray, x: int, y: int, radius: int) -> None:
        for dy in range(-radius, radius + 1):
            dx = radius - abs(dy)
            for px in {x - dx, x + dx}:
                py = y + dy
                if 0 <= px < VIEW_SIZE and 0 <= py < VIEW_SIZE:
                    frame[py, px] = 2


class ChromaticClock(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.display = ClockDisplay()
        levels = [_make_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BACKGROUND, BACKGROUND, [self.display]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )
        self.display.game = self

    def on_set_level(self, level: Level) -> None:
        self.board = level.get_sprites_by_tag("board")[0]
        self.modules = tuple(level.get_data("modules") or ())
        self.initial_phases = dict(level.get_data("phases") or ())
        self.phases = dict(self.initial_phases)
        self.targets = dict(level.get_data("targets") or ())
        self.step_budget = int(level.get_data("budget") or 1)
        self.remaining_steps = self.step_budget
        self.step_count = 0
        self.mode = "PLAY"
        self.jammed_phases: set[int] = set()
        self.selected_dial: str | None = None
        self.old_phase: int | None = None
        self.affected_dials: tuple[str, ...] = ()
        self.pulse_dials: set[str] = set()
        self.last_background_click: tuple[int, int] | None = None
        self.render_effect = "base"
        self._animation: list[str] = []
        self._pending_outcome: str | None = None
        self._undo_history: list[tuple[dict[str, int], str, set[int]]] = []
        self._render_board()

    def step(self) -> None:
        if self._animation:
            self._play_animation_frame()
            return

        if self.action.id == GameAction.RESET:
            self._render_board()
            self.complete_action()
            return

        if self.mode == "JAMMED":
            if self.action.id == GameAction.ACTION7 and self._undo_history:
                self._spend_step()
                phases, mode, jammed = self._undo_history.pop()
                self.phases = dict(phases)
                self.mode = mode
                self.jammed_phases = set(jammed)
                self._finish_immediate_action()
            else:
                self._reset_attempt()
                self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            self._spend_step()
            if self._undo_history:
                phases, mode, jammed = self._undo_history.pop()
                self.phases = dict(phases)
                self.mode = mode
                self.jammed_phases = set(jammed)
            self._finish_immediate_action()
            return

        if self.action.id == GameAction.ACTION6:
            x = int(self.action.data.get("x", 0))
            y = int(self.action.data.get("y", 0))
            clicked = self._dial_at(x, y)
            if clicked is not None:
                self._begin_dial_click(clicked)
            else:
                self._spend_step()
                self.last_background_click = (x, y)
                self._queue_feedback(["ripple1", "ripple2", "base"])
        else:
            self._spend_step()
            self._queue_feedback(["keypulse", "base"])

        if self._animation:
            self._play_animation_frame()
        else:
            self._finish_immediate_action()

    def _spend_step(self) -> None:
        self.step_count += 1
        self.remaining_steps = max(0, self.step_budget - self.step_count)
        if self.remaining_steps == 0 and self._pending_outcome != "win":
            self._pending_outcome = "lose"

    def _begin_dial_click(self, clicked: str) -> None:
        self._undo_history.append((dict(self.phases), self.mode, set(self.jammed_phases)))
        self._spend_step()
        self.selected_dial = clicked
        self.old_phase = self.phases[clicked]
        self.affected_dials = tuple(module for module in self.modules if self.phases[module] == self.old_phase)
        destination = (self.old_phase + 1) % 8
        destination_before = {module for module in self.modules if self.phases[module] == destination}
        for module in self.affected_dials:
            self.phases[module] = destination

        self.jammed_phases = self._find_jammed_phases()
        if self.jammed_phases:
            self.mode = "JAMMED"
            self._pending_outcome = "jam"
        else:
            merged = destination_before | set(self.affected_dials)
            self.pulse_dials = merged if destination_before else set()
            if all(self.phases[module] == self.targets[module] for module in self.modules):
                self.mode = "COMPLETE"
                self._pending_outcome = "win"

        if self.remaining_steps == 0 and self.mode != "COMPLETE":
            self._pending_outcome = "lose"

        self._animation = ["identify", "sweep", "commit", "settle1", "settle2"]
        if self._pending_outcome == "win":
            self._animation.extend(f"success{index}" for index in range(6))

    def _find_jammed_phases(self) -> set[int]:
        jammed: set[int] = set()
        for phase in range(8):
            represented_targets = {self.targets[module] for module in self.modules if self.phases[module] == phase}
            if len(represented_targets) > 1:
                jammed.add(phase)
        return jammed

    def _queue_feedback(self, effects: list[str]) -> None:
        if self._pending_outcome == "lose":
            self._animation = ["timeout", "timeout"]
        else:
            self._animation = effects

    def _play_animation_frame(self) -> None:
        effect = self._animation.pop(0)
        self.render_effect = effect
        self._render_board()
        if self._animation:
            return

        outcome = self._pending_outcome
        self._pending_outcome = None
        if outcome == "win":
            self.next_level()
            self.complete_action()
        elif outcome == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _finish_immediate_action(self) -> None:
        self.render_effect = "base"
        self._render_board()
        if self._pending_outcome == "lose":
            self._pending_outcome = None
            self.lose()
        self.complete_action()

    def _reset_attempt(self) -> None:
        self.phases = dict(self.initial_phases)
        self.step_count = 0
        self.remaining_steps = self.step_budget
        self.mode = "PLAY"
        self.jammed_phases.clear()
        self._undo_history.clear()
        self._pending_outcome = None
        self.render_effect = "base"
        self._render_board()

    def _dial_at(self, x: int, y: int) -> str | None:
        for module in self.modules:
            ox, oy = MODULE_ORIGINS[module]
            u, v = x - ox, y - oy
            if 0 <= u < 15 and 0 <= v < 15 and PLATE_MASK[v, u]:
                return module
        return None

    def _render_board(self) -> None:
        pixels = np.full((VIEW_SIZE, VIEW_SIZE), BACKGROUND, dtype=np.int8)
        for module in self.modules:
            phase = self.phases[module]
            offset = 0
            show_status = self.render_effect not in {"identify", "sweep", "commit"}
            old_ray: int | None = None
            if module in self.affected_dials and self.render_effect == "identify":
                phase = int(self.old_phase or 0)
                offset = -1
            elif module in self.affected_dials and self.render_effect == "sweep":
                offset = 1
                old_ray = int(self.old_phase or 0)
            self._draw_dial(pixels, module, phase, offset, show_status, old_ray)
        self.board.pixels[:, :] = pixels

    def _draw_dial(
        self, canvas: np.ndarray, module: str, phase: int, offset: int, show_status: bool, old_ray: int | None
    ) -> None:
        dial = DIAL_BASES[self.targets[module]].copy()

        matched = self.phases[module] == self.targets[module]
        jammed = self.phases[module] in self.jammed_phases
        hub_color = 1
        if show_status and matched:
            hub_color = 14
        if show_status and jammed:
            hub_color = 13
        if show_status and module in self.pulse_dials and self.render_effect == "settle1":
            hub_color = 7
        dial[6:9, 6:9] = hub_color
        marker_u, marker_v = HUB_MARKERS[phase]
        dial[marker_v, marker_u] = 0
        for u, v in RAYS[phase]:
            dial[v, u] = 0
        if old_ray is not None:
            for u, v in RAYS[old_ray]:
                dial[v, u] = 1

        if show_status and matched and not jammed:
            for u, v in CHECK_PIXELS:
                dial[v, u] = 0
        if show_status and jammed:
            for u, v in JAM_X_PIXELS:
                dial[v, u] = 0

        # Keep the actual click plate visible independently of the hand rays.
        plate_color = 14 if matched and not jammed else (13 if jammed else 11)
        dial[5, 5:10] = plate_color
        dial[9, 5:10] = plate_color
        dial[5:10, 5] = plate_color
        dial[5:10, 9] = plate_color
        dial[7, 7] = plate_color

        ox, oy = MODULE_ORIGINS[module]
        ox += offset
        source_x0 = max(0, -ox)
        source_y0 = max(0, -oy)
        source_x1 = min(15, VIEW_SIZE - ox)
        source_y1 = min(15, VIEW_SIZE - oy)
        source = dial[source_y0:source_y1, source_x0:source_x1]
        destination = canvas[oy + source_y0 : oy + source_y1, ox + source_x0 : ox + source_x1]
        destination[:, :] = np.where(source >= 0, source, destination)

    def _get_hidden_state(self) -> np.ndarray:
        values = [self.phases[module] for module in self.modules]
        values.extend((self.step_count, {"PLAY": 0, "JAMMED": 1, "COMPLETE": 2}[self.mode]))
        return np.asarray(values, dtype=np.int16)

    def _get_valid_actions(self) -> list[ActionInput]:
        actions = [ActionInput(id=GameAction.from_id(action_id)) for action_id in (1, 2, 3, 4, 5, 7)]
        actions.extend(
            ActionInput(
                id=GameAction.ACTION6, data={"x": MODULE_ORIGINS[module][0] + 7, "y": MODULE_ORIGINS[module][1] + 7}
            )
            for module in self.modules
        )
        return actions
