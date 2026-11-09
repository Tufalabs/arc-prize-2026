from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "charge_relay-0001"
VIEW_SIZE = 64
CELL_SIZE = 5
BOARD_LEFT = 4
BOARD_TOP = 3

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7
MOVE_DELTAS = {ACTION_UP: (-1, 0), ACTION_DOWN: (1, 0), ACTION_LEFT: (0, -1), ACTION_RIGHT: (0, 1)}

COLOR_CORONA = 0
COLOR_CHARGED = 1
COLOR_DETAIL = 2
COLOR_TRACK = 3
COLOR_RELAY = 4
COLOR_BACKGROUND = 5
COLOR_C = 6
COLOR_INVALID = 8
COLOR_A = 9
COLOR_LIVE = 10
COLOR_B = 11
COLOR_SOURCE = 12
COLOR_ACTIVE = 14
TYPE_COLORS = {"A": COLOR_A, "B": COLOR_B, "C": COLOR_C}

TYPE_GLYPHS = {
    "A": ((2, 1), (2, 2), (2, 3)),
    "B": ((1, 2), (2, 2), (3, 2)),
    "C": ((1, 1), (3, 1), (2, 2), (1, 3), (3, 3)),
}
CORONA = ((0, 0), (1, 0), (0, 1), (3, 0), (4, 0), (4, 1), (0, 3), (0, 4), (1, 4), (4, 3), (3, 4), (4, 4))

Cell = tuple[int, int]
Gate = tuple[str, int, int, str]
Receptor = tuple[str, int, int, int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    start: Cell
    gates: tuple[Gate, ...]
    receptors: tuple[Receptor, ...]
    baseline_actions: int

    @property
    def max_steps(self) -> int:
        return self.baseline_actions * 6


LEVEL_SPECS = (
    LevelSpec(
        "Crooked Spark",
        (
            "...........",
            "...........",
            "...........",
            "...........",
            "....B......",
            "...AA......",
            "...A.......",
            "...........",
            "...........",
            "...........",
            "...........",
        ),
        (6, 3),
        (),
        (("H", 4, 4, 4, 4),),
        3,
    ),
    LevelSpec(
        "The Live Rail",
        (
            "...........",
            "...........",
            "...........",
            "....AAAAA..",
            "....A......",
            "...AA......",
            "..AA.......",
            "..A........",
            "...........",
            "...........",
            "...........",
        ),
        (7, 2),
        (),
        (("H", 3, 4, 3, 8),),
        6,
    ),
    LevelSpec(
        "Retune on Landing",
        (
            "...........",
            "...........",
            "...........",
            "...........",
            "...ABACA...",
            "...A.......",
            "..AA.......",
            "..A........",
            "...........",
            "...........",
            "...........",
        ),
        (7, 2),
        (("H", 4, 3, "A"), ("H", 4, 5, "B")),
        (("H", 4, 4, 4, 7),),
        5,
    ),
    LevelSpec(
        "Enter at the Middle",
        (
            ".ACABCAC.",
            ".A..AA...",
            ".AABACBA.",
            ".......A.",
            ".AACABAB.",
            ".A.......",
            ".CAABA...",
            ".....C...",
            ".AABAA...",
        ),
        (8, 1),
        (
            *(("H", 0, c, "B") for c in range(1, 7)),
            ("V", 7, 5, "A"),
            ("V", 5, 1, "C"),
            ("V", 3, 7, "B"),
            ("V", 1, 5, "C"),
        ),
        (("H", 0, 1, 0, 7),),
        25,
    ),
    LevelSpec(
        "The Crossing",
        (
            "AAAABABAA",
            "B.A.A.A.A",
            "AAB.B.AAA",
            "A...A...A",
            "AAAACAAAA",
            "A...A...A",
            "ABA.AAABA",
            "..A.A....",
            "AAB.A....",
        ),
        (8, 0),
        (
            *(("H", 4, c, "C") for c in range(8)),
            *(("V", r, 4, "C") for r in (0, 1, 2, 3, 6, 7)),
            ("H", 6, 1, "A"),
            ("V", 1, 2, "B"),
            ("V", 0, 6, "B"),
            ("H", 6, 7, "A"),
            ("V", 1, 0, "B"),
            ("H", 0, 6, "A"),
        ),
        (("H", 4, 0, 4, 8), ("V", 0, 4, 8, 4)),
        32,
    ),
    LevelSpec(
        "Beyond the Bracket",
        (
            "AABAAACA.",
            "A.....AA.",
            "A.BAA.AAA",
            "AACAAAABC",
            "BAA.A..A.",
            "....A..A.",
            "ACA.CABAA",
            "A.C....AA",
            "A.AABACAB",
        ),
        (8, 0),
        (
            *(("H", 3, c, "B") for c in range(7)),
            *(("V", r, 7, "B") for r in range(8)),
            ("V", 0, 6, "C"),
            ("V", 2, 2, "B"),
            ("V", 2, 8, "A"),
            ("V", 5, 4, "C"),
            ("V", 6, 2, "A"),
            ("V", 7, 8, "B"),
        ),
        (("H", 3, 0, 3, 5), ("V", 0, 7, 8, 7)),
        44,
    ),
    LevelSpec(
        "Four Arms, One Frame",
        (
            "AABABCBAA.B",
            "A...A.B.A.A",
            "BAAABAAACAA",
            "A...A.B.A.A",
            "CAA.AAA.AAB",
            "..B.....A..",
            "AAA.CAAABAA",
            "A...A...A.A",
            "BACAA.AABAB",
            "......B.A.C",
            "AABAAAACAAB",
        ),
        (10, 0),
        (
            *(("H", 2, c, "C") for c in range(10)),
            *(("V", r, 8, "C") for r in range(2, 10)),
            ("V", 9, 6, "A"),
            ("V", 7, 10, "B"),
            ("V", 6, 4, "C"),
            ("V", 7, 0, "B"),
            ("V", 5, 2, "A"),
            ("V", 3, 0, "C"),
            ("V", 0, 4, "B"),
            ("V", 3, 6, "A"),
            ("H", 0, 6, "B"),
            ("H", 0, 4, "C"),
            ("H", 10, 6, "C"),
        ),
        (("H", 2, 0, 2, 3), ("H", 2, 5, 2, 10), ("V", 0, 8, 2, 8), ("V", 4, 8, 10, 8)),
        56,
    ),
)


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


def _token_cells(spec: LevelSpec) -> dict[Cell, str]:
    return {
        (row, col): value
        for row, line in enumerate(spec.rows)
        for col, value in enumerate(line)
        if value in TYPE_COLORS
    }


def _build_level(spec: LevelSpec) -> Level:
    background = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8), name="background", layer=-20, collidable=False
    )
    return Level(grid_size=(VIEW_SIZE, VIEW_SIZE), sprites=[background], name=spec.name, data={"spec": spec})


class ChargeRelayDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ChargeRelay | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.game is not None:
            frame[:, :] = self.game.display_frame
        return frame


class ChargeRelay(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = ChargeRelayDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._display]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK, ACTION_UNDO],
            seed,
        )
        self._display.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.tokens = _token_cells(self.spec)
        self.gates: dict[tuple[str, int, int], str] = {
            (orientation, row, col): key for orientation, row, col, key in self.spec.gates
        }
        self.holder = self.spec.start
        self.step_count = 0
        self.max_steps = self.spec.max_steps
        self.done = False
        self.truncated = False
        self.last_transition: dict[str, object] | None = None
        self.undo_history: list[Cell] = []
        self.animation_frames: list[np.ndarray] = []
        self._pending_outcome: str | None = None
        self._recompute()
        self.display_frame = self._render_frame()

    @property
    def current_type(self) -> str:
        return self.tokens[self.holder]

    def _contact(self, first: Cell, second: Cell) -> tuple[str, int, int]:
        r1, c1 = first
        r2, c2 = second
        if r1 == r2:
            return ("H", r1, min(c1, c2))
        return ("V", min(r1, r2), c1)

    def _contact_open(self, first: Cell, second: Cell, charge_type: str | None = None) -> bool:
        key = self.gates.get(self._contact(first, second))
        return key is None or key == (charge_type or self.current_type)

    def _segment(self, delta: Cell) -> set[Cell]:
        out = {self.holder}
        for sign in (-1, 1):
            current = self.holder
            while True:
                nxt = (current[0] + delta[0] * sign, current[1] + delta[1] * sign)
                if nxt not in self.tokens or not self._contact_open(current, nxt):
                    break
                out.add(nxt)
                current = nxt
        return out

    def _recompute(self) -> None:
        self.horizontal_energized = self._segment((0, 1))
        self.vertical_energized = self._segment((1, 0))
        active: list[bool] = []
        for orientation, r1, c1, r2, c2 in self.spec.receptors:
            if orientation == "H":
                interval = {(r1, col) for col in range(min(c1, c2), max(c1, c2) + 1)}
                active.append(interval <= self.horizontal_energized)
            else:
                interval = {(row, c1) for row in range(min(r1, r2), max(r1, r2) + 1)}
                active.append(interval <= self.vertical_energized)
        self.receptor_active = active

    def _gate_open_map(self, charge_type: str | None = None) -> dict[tuple[str, int, int], bool]:
        active_type = charge_type or self.current_type
        return {contact: key == active_type for contact, key in self.gates.items()}

    def _draw_gate(
        self,
        frame: np.ndarray,
        contact: tuple[str, int, int],
        key: str,
        is_open: bool,
        horizontal: set[Cell],
        vertical: set[Cell],
        center_pixels: int | None = None,
        color: int | None = None,
    ) -> None:
        orientation, row, col = contact
        gate_color = TYPE_COLORS[key] if color is None else color
        if orientation == "H":
            x = BOARD_LEFT + CELL_SIZE * col + CELL_SIZE - 1
            y = BOARD_TOP + CELL_SIZE * row + 1
            frame[y : y + 3, x : x + 2] = gate_color
            if is_open:
                frame[y + 1, x : x + 2] = COLOR_LIVE if (row, col) in horizontal else COLOR_RELAY
            elif center_pixels is not None:
                clear = max(0, min(2, center_pixels))
                frame[y + 1, x : x + clear] = COLOR_RELAY
        else:
            x = BOARD_LEFT + CELL_SIZE * col + 1
            y = BOARD_TOP + CELL_SIZE * row + CELL_SIZE - 1
            frame[y : y + 2, x : x + 3] = gate_color
            if is_open:
                frame[y : y + 2, x + 1] = COLOR_LIVE if (row, col) in vertical else COLOR_RELAY
            elif center_pixels is not None:
                clear = max(0, min(2, center_pixels))
                frame[y : y + clear, x + 1] = COLOR_RELAY

    def _draw_contact_highlight(self, frame: np.ndarray, first: Cell, second: Cell) -> None:
        orientation, row, col = self._contact(first, second)
        if orientation == "H":
            x = BOARD_LEFT + CELL_SIZE * col + CELL_SIZE - 1
            y = BOARD_TOP + CELL_SIZE * row + 2
            frame[y, x : x + 2] = COLOR_LIVE
        else:
            x = BOARD_LEFT + CELL_SIZE * col + 2
            y = BOARD_TOP + CELL_SIZE * row + CELL_SIZE - 1
            frame[y : y + 2, x] = COLOR_LIVE

    def _render_frame(
        self,
        *,
        holder: Cell | None = None,
        horizontal: set[Cell] | None = None,
        vertical: set[Cell] | None = None,
        receptors: list[bool] | None = None,
        gate_open: dict[tuple[str, int, int], bool] | None = None,
        gate_centers: dict[tuple[str, int, int], int] | None = None,
        stripe_color: int = COLOR_LIVE,
        feedback: dict[str, object] | None = None,
    ) -> np.ndarray:
        holder = self.holder if holder is None else holder
        horizontal = self.horizontal_energized if horizontal is None else horizontal
        vertical = self.vertical_energized if vertical is None else vertical
        receptors = self.receptor_active if receptors is None else receptors
        gate_open = self._gate_open_map() if gate_open is None else gate_open
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8)

        for (row, col), token_type in self.tokens.items():
            x = BOARD_LEFT + CELL_SIZE * col
            y = BOARD_TOP + CELL_SIZE * row
            frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = COLOR_CHARGED if (row, col) == holder else COLOR_RELAY
            if (row, col) in horizontal:
                frame[y + 2, x : x + CELL_SIZE] = stripe_color
            if (row, col) in vertical:
                frame[y : y + CELL_SIZE, x + 2] = stripe_color
            for dx, dy in TYPE_GLYPHS[token_type]:
                frame[y + dy, x + dx] = TYPE_COLORS[token_type]

        blocked_gate = feedback.get("blocked_gate") if feedback else None
        for contact, key in self.gates.items():
            pulse_color = COLOR_INVALID if contact == blocked_gate else None
            self._draw_gate(
                frame,
                contact,
                key,
                gate_open[contact],
                horizontal,
                vertical,
                None if gate_centers is None else gate_centers.get(contact),
                pulse_color,
            )

        hx = BOARD_LEFT + CELL_SIZE * holder[1]
        hy = BOARD_TOP + CELL_SIZE * holder[0]
        for dx, dy in CORONA:
            frame[hy + dy, hx + dx] = COLOR_CORONA

        if feedback and feedback.get("holder_glyph_zero"):
            for dx, dy in TYPE_GLYPHS[self.tokens[holder]]:
                frame[hy + dy, hx + dx] = COLOR_CORONA

        for index, (orientation, r1, c1, r2, c2) in enumerate(self.spec.receptors):
            sx = BOARD_LEFT + CELL_SIZE * c1
            sy = BOARD_TOP + CELL_SIZE * r1
            frame[sy : sy + 2, sx : sx + 2] = COLOR_SOURCE
            rx = BOARD_LEFT + CELL_SIZE * c2
            ry = BOARD_TOP + CELL_SIZE * r2
            if orientation == "H":
                ys = (ry, ry + 1)
                xs = (rx + 3, rx + 4)
                inactive = ((ys[0], xs[0]), (ys[0], xs[1]), (ys[1], xs[1]))
            else:
                ys = (ry + 3, ry + 4)
                xs = (rx, rx + 1)
                inactive = ((ys[0], xs[0]), (ys[1], xs[0]), (ys[1], xs[1]))
            if receptors[index]:
                frame[ys[0] : ys[1] + 1, xs[0] : xs[1] + 1] = COLOR_ACTIVE
            else:
                for py, px in inactive:
                    frame[py, px] = COLOR_DETAIL

        if feedback:
            for first, second in feedback.get("legal_contacts", []):
                self._draw_contact_highlight(frame, first, second)
            gap_delta = feedback.get("gap_delta")
            if gap_delta is not None:
                dr, dc = gap_delta
                if dr < 0:
                    frame[hy, hx + 1 : hx + 4] = COLOR_INVALID
                elif dr > 0:
                    frame[hy + 4, hx + 1 : hx + 4] = COLOR_INVALID
                elif dc < 0:
                    frame[hy + 1 : hy + 4, hx] = COLOR_INVALID
                else:
                    frame[hy + 1 : hy + 4, hx + 4] = COLOR_INVALID

        frame[61:63, 4:60] = COLOR_TRACK
        remaining = max(0, self.max_steps - self.step_count)
        filled = int(56 * remaining / max(1, self.max_steps))
        if filled:
            frame[61:63, 4 : 4 + filled] = COLOR_LIVE
        return frame

    def _legal_neighbors(self, holder: Cell, charge_type: str) -> list[tuple[Cell, Cell]]:
        legal = []
        for dr, dc in MOVE_DELTAS.values():
            nxt = (holder[0] + dr, holder[1] + dc)
            if nxt in self.tokens and self._contact_open(holder, nxt, charge_type):
                legal.append((holder, nxt))
        return legal

    def _queue_blocked(self, delta: Cell, blocked_gate: tuple[str, int, int] | None, departure_type: str) -> None:
        common = {
            "legal_contacts": self._legal_neighbors(self.holder, departure_type),
            "blocked_gate": blocked_gate,
            "gap_delta": None if blocked_gate else delta,
        }
        self.animation_frames = [
            self._render_frame(feedback={**common, "holder_glyph_zero": bool(blocked_gate)}),
            self._render_frame(feedback={**common, "blocked_gate": None, "holder_glyph_zero": False}),
            self._render_frame(feedback={**common, "holder_glyph_zero": bool(blocked_gate)}),
        ]

    def _partial_segments(self, radius: int) -> tuple[set[Cell], set[Cell]]:
        hr, hc = self.holder
        horizontal = {cell for cell in self.horizontal_energized if abs(cell[1] - hc) <= radius}
        vertical = {cell for cell in self.vertical_energized if abs(cell[0] - hr) <= radius}
        return horizontal, vertical

    def _queue_transition(
        self, old_type: str, old_horizontal: set[Cell], old_vertical: set[Cell], old_receptors: list[bool]
    ) -> None:
        changed = old_type != self.current_type or old_receptors != self.receptor_active
        if not changed:
            self.animation_frames = [self._render_frame()]
            return

        old_gate_open = self._gate_open_map(old_type)
        new_gate_open = self._gate_open_map()
        self.animation_frames = [
            self._render_frame(
                horizontal=old_horizontal, vertical=old_vertical, receptors=old_receptors, gate_open=old_gate_open
            )
        ]
        changed_contacts = {contact for contact in self.gates if old_gate_open[contact] != new_gate_open[contact]}
        centers = dict.fromkeys(changed_contacts, 1)
        intermediate_gate_open = dict(old_gate_open)
        for contact in changed_contacts:
            intermediate_gate_open[contact] = False
        self.animation_frames.append(
            self._render_frame(
                horizontal=set(),
                vertical=set(),
                receptors=old_receptors,
                gate_open=intermediate_gate_open,
                gate_centers=centers,
            )
        )
        self.animation_frames.append(
            self._render_frame(horizontal=set(), vertical=set(), receptors=old_receptors, gate_open=new_gate_open)
        )
        max_radius = max(
            [abs(cell[1] - self.holder[1]) for cell in self.horizontal_energized]
            + [abs(cell[0] - self.holder[0]) for cell in self.vertical_energized]
            + [0]
        )
        for radius in range(max_radius + 1):
            horizontal, vertical = self._partial_segments(radius)
            final_growth = radius == max_radius
            self.animation_frames.append(
                self._render_frame(
                    horizontal=horizontal,
                    vertical=vertical,
                    receptors=self.receptor_active if final_growth else old_receptors,
                    gate_open=new_gate_open,
                )
            )

    def _append_completion_frames(self) -> None:
        for index in range(6):
            color = COLOR_ACTIVE if index % 2 else COLOR_LIVE
            self.animation_frames.append(self._render_frame(stripe_color=color))

    def _finish_or_queue(self) -> None:
        if self.done:
            self._append_completion_frames()
            self._pending_outcome = "win"
        elif self.truncated:
            self._pending_outcome = "lose"
        if self.animation_frames:
            self._play_animation_frame()
        else:
            self._complete_pending()

    def _complete_pending(self) -> None:
        if self._pending_outcome == "win":
            self.next_level()
        elif self._pending_outcome == "lose":
            self.lose()
        self._pending_outcome = None
        self.complete_action()

    def _play_animation_frame(self) -> None:
        self.display_frame = self.animation_frames.pop(0)
        if not self.animation_frames:
            self._complete_pending()

    def _commit_action_result(self) -> None:
        self._recompute()
        self.done = bool(self.receptor_active) and all(self.receptor_active)
        self.truncated = not self.done and self.step_count >= self.max_steps

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return

        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self.display_frame = self._render_frame()
            self.complete_action()
            return

        old_holder = self.holder
        old_type = self.current_type
        old_horizontal = set(self.horizontal_energized)
        old_vertical = set(self.vertical_energized)
        old_receptors = list(self.receptor_active)
        self.last_transition = {"action": action, "departure": old_holder}

        blocked_delta: Cell | None = None
        blocked_gate: tuple[str, int, int] | None = None
        if action in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[action]
            destination = (old_holder[0] + dr, old_holder[1] + dc)
            if destination not in self.tokens:
                blocked_delta = (dr, dc)
            else:
                contact = self._contact(old_holder, destination)
                key = self.gates.get(contact)
                if key is not None and key != old_type:
                    blocked_delta = (dr, dc)
                    blocked_gate = contact
                else:
                    self.undo_history.append(old_holder)
                    self.holder = destination
        elif action == ACTION_UNDO:
            if self.undo_history:
                self.holder = self.undo_history.pop()

        self.step_count += 1
        self._commit_action_result()
        self.last_transition.update(
            {
                "holder": self.holder,
                "blocked": blocked_delta is not None,
                "current_type": self.current_type,
                "receptors": tuple(self.receptor_active),
            }
        )

        if blocked_delta is not None:
            self._queue_blocked(blocked_delta, blocked_gate, old_type)
        else:
            self._queue_transition(old_type, old_horizontal, old_vertical, old_receptors)
        self._finish_or_queue()

    def _get_hidden_state(self) -> np.ndarray:
        return np.asarray(
            [self.holder[0], self.holder[1], self.step_count, *[int(value) for value in self.receptor_active]],
            dtype=np.int16,
        )
