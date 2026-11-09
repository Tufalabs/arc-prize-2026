from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

BLACK = 5
WHITE = 0
HALO = 1
NEUTRAL = 2
RAIL = 3
SHADOW = 4
RED = 8
BLUE = 9
MAROON = 13
LIGHT_BLUE = 10
SUCCESS = 14


def _origin(value: int) -> int:
    return 10 + 5 * (value - 1)


@dataclass(frozen=True)
class Emitter:
    emitter_id: str
    side: str
    lane: int
    color: int

    @property
    def hitbox(self) -> tuple[int, int, int, int]:
        lo = _origin(self.lane)
        hi = lo + 3
        if self.side == "W":
            return (5, 8, lo, hi) if self.color == RED else (0, 3, lo, hi)
        if self.side == "E":
            return (55, 58, lo, hi) if self.color == RED else (60, 63, lo, hi)
        if self.side == "N":
            return (lo, hi, 5, 8) if self.color == RED else (lo, hi, 0, 3)
        return (lo, hi, 55, 58) if self.color == RED else (lo, hi, 60, 63)

    @property
    def center(self) -> tuple[int, int]:
        x0, x1, y0, y1 = self.hitbox
        return ((x0 + x1) // 2, (y0 + y1) // 2)


@dataclass
class Prism:
    prism_id: str
    x: int
    y: int
    current: int
    target: int


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    emitters: tuple[tuple[str, str, int, str], ...]
    baseline: int


# ASCII prism legend: r=R/R, R=B/R, b=B/B, B=R/B, .=empty.
LEVEL_SPECS = (
    LevelSpec(
        "One Visible Correction",
        (
            ".........",
            ".........",
            "..r......",
            ".........",
            "....R....",
            ".........",
            ".........",
            ".........",
            ".........",
        ),
        (("W5-R", "W", 5, "R"),),
        1,
    ),
    LevelSpec(
        "Penetration",
        (
            ".........",
            ".........",
            ".........",
            ".........",
            "R.R.R.R.R",
            ".........",
            ".........",
            ".........",
            ".........",
        ),
        (("W5-R", "W", 5, "R"),),
        5,
    ),
    LevelSpec(
        "Borrowed Color",
        (
            ".........",
            ".........",
            ".........",
            ".........",
            ".b.R.R.RR",
            ".........",
            ".........",
            ".........",
            ".........",
        ),
        (("W5-R", "W", 5, "R"), ("N2-B", "N", 2, "B")),
        6,
    ),
    LevelSpec(
        "Last Shield",
        (
            "...B.....",
            ".........",
            ".........",
            ".........",
            "Br.rB.r.B",
            ".........",
            "...r.....",
            ".........",
            "...B.....",
        ),
        (("H-R", "W", 5, "R"), ("H-B", "W", 5, "B"), ("V-R", "N", 4, "R"), ("V-B", "N", 4, "B")),
        25,
    ),
    LevelSpec(
        "Two-Gate Spine",
        (
            "...r.....",
            ".........",
            "Br.rB.r.B",
            "...B.....",
            ".........",
            "...r.....",
            "BrBB.r..B",
            ".........",
            "...B.....",
        ),
        (
            ("H1-R", "W", 3, "R"),
            ("H1-B", "W", 3, "B"),
            ("H2-R", "W", 7, "R"),
            ("H2-B", "W", 7, "B"),
            ("V-R", "N", 4, "R"),
            ("V-B", "N", 4, "B"),
        ),
        47,
    ),
    LevelSpec(
        "Opposite Approaches",
        (
            "..r...B..",
            "......r..",
            "rBr.r.B.B",
            ".........",
            "..B...B..",
            ".........",
            "B.BrB.r.B",
            "..r......",
            "..B...B..",
        ),
        (
            ("H1-R", "W", 3, "R"),
            ("H1-B", "W", 3, "B"),
            ("H2-R", "E", 7, "R"),
            ("H2-B", "E", 7, "B"),
            ("V1-R", "N", 3, "R"),
            ("V1-B", "N", 3, "B"),
            ("V2-R", "S", 7, "R"),
            ("V2-B", "S", 7, "B"),
        ),
        54,
    ),
    LevelSpec(
        "Closed Restoration Loop",
        (
            "..r...B..",
            "......r..",
            "BrrB.rB.B",
            ".........",
            "..B...B..",
            "..r......",
            "r.rBr.r.B",
            ".........",
            "..B...r..",
        ),
        (
            ("H1-R", "E", 3, "R"),
            ("H1-B", "E", 3, "B"),
            ("H2-R", "W", 7, "R"),
            ("H2-B", "W", 7, "B"),
            ("V1-R", "N", 3, "R"),
            ("V1-B", "N", 3, "B"),
            ("V2-R", "S", 7, "R"),
            ("V2-B", "S", 7, "B"),
        ),
        62,
    ),
)


class PrismBlockersDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: PrismBlockers | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = BLACK
        self._draw_rails(frame, game)
        self._draw_budget(frame, game)
        for prism in game.prisms:
            self._draw_prism(frame, prism, game)
        for emitter in game.emitters:
            self._draw_emitter(frame, emitter, game)
        self._draw_feedback(frame, game)
        return frame

    def _draw_rails(self, frame: np.ndarray, game: PrismBlockers) -> None:
        horizontal = {e.lane for e in game.emitters if e.side in {"W", "E"}}
        vertical = {e.lane for e in game.emitters if e.side in {"N", "S"}}
        for lane in horizontal:
            frame[_origin(lane) + 1, 4:60] = RAIL
        for lane in vertical:
            frame[4:60, _origin(lane) + 1] = RAIL

        if game.visual_phase == "path" and game.active_emitter is not None:
            e = game.active_emitter
            color = e.color
            hit = game.active_hit
            if e.side == "W":
                end = _origin(hit.x) + 1 if hit else 59
                frame[_origin(e.lane) + 1, 4 : end + 1] = color
            elif e.side == "E":
                end = _origin(hit.x) + 1 if hit else 4
                frame[_origin(e.lane) + 1, end:60] = color
            elif e.side == "N":
                end = _origin(hit.y) + 1 if hit else 59
                frame[4 : end + 1, _origin(e.lane) + 1] = color
            else:
                end = _origin(hit.y) + 1 if hit else 4
                frame[end:60, _origin(e.lane) + 1] = color

    def _draw_budget(self, frame: np.ndarray, game: PrismBlockers) -> None:
        frame[54, 10:54] = SHADOW
        filled = int(44 * game.remaining_steps / max(1, game.step_budget))
        if filled:
            frame[54, 10 : 10 + filled] = HALO

    def _pattern(self, color: int, x: int, y: int) -> int:
        if color == RED:
            return RED if x <= 1 else MAROON
        return BLUE if y <= 1 else LIGHT_BLUE

    def _draw_prism(self, frame: np.ndarray, prism: Prism, game: PrismBlockers) -> None:
        x0, y0 = _origin(prism.x), _origin(prism.y)
        facet_color = prism.current
        if game.visual_phase in {"source", "path", "impact"}:
            facet_color = game.old_colors.get(prism.prism_id, prism.current)
        for ly in range(4):
            for lx in range(4):
                border = lx in {0, 3} or ly in {0, 3}
                color = self._pattern(prism.target if border else facet_color, lx, ly)
                frame[y0 + ly, x0 + lx] = color

        if prism.current == prism.target:
            frame[y0, x0 : x0 + 4] = WHITE
            frame[y0 + 3, x0 : x0 + 4] = WHITE
        else:
            for px, py in ((x0 - 1, y0), (x0 - 1, y0 + 3), (x0 + 4, y0), (x0 + 4, y0 + 3)):
                if 0 <= px < 64 and 0 <= py < 64:
                    frame[py, px] = HALO

        if game.visual_phase == "path" and prism in game.passed_prisms:
            frame[y0, x0 : x0 + 4] = WHITE
            frame[y0 : y0 + 4, x0] = WHITE
        if game.visual_phase == "impact" and prism is game.active_hit:
            frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = WHITE
        if game.visual_phase == "success":
            frame[y0, x0 : x0 + 4] = WHITE
            frame[y0 + 3, x0 : x0 + 4] = WHITE
            frame[y0 : y0 + 4, x0] = WHITE
            frame[y0 : y0 + 4, x0 + 3] = WHITE

    def _draw_emitter(self, frame: np.ndarray, emitter: Emitter, game: PrismBlockers) -> None:
        x0, x1, y0, y1 = emitter.hitbox
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                frame[y, x] = self._pattern(emitter.color, x - x0, y - y0)
        if emitter.side in {"W", "E"}:
            cy = y0 + 1
            tip = x1 if emitter.side == "W" else x0
            tail = tip - 2 if emitter.side == "W" else tip + 2
            frame[cy, min(tip, tail) : max(tip, tail) + 1] = WHITE
            frame[cy - 1 : cy + 3, tip] = WHITE
        else:
            cx = x0 + 1
            tip = y1 if emitter.side == "N" else y0
            tail = tip - 2 if emitter.side == "N" else tip + 2
            frame[min(tip, tail) : max(tip, tail) + 1, cx] = WHITE
            frame[tip, cx - 1 : cx + 3] = WHITE

        if game.visual_phase == "source" and emitter is game.active_emitter:
            frame[max(0, y0 - 1) : min(64, y1 + 2), max(0, x0 - 1)] = HALO
            frame[max(0, y0 - 1) : min(64, y1 + 2), min(63, x1 + 1)] = HALO
            frame[max(0, y0 - 1), max(0, x0 - 1) : min(64, x1 + 2)] = HALO
            frame[min(63, y1 + 1), max(0, x0 - 1) : min(64, x1 + 2)] = HALO

    def _draw_feedback(self, frame: np.ndarray, game: PrismBlockers) -> None:
        if game.visual_phase == "exit" and game.active_emitter is not None:
            e = game.active_emitter
            if e.side == "W":
                frame[_origin(e.lane) + 1, 59] = WHITE
            elif e.side == "E":
                frame[_origin(e.lane) + 1, 4] = WHITE
            elif e.side == "N":
                frame[59, _origin(e.lane) + 1] = WHITE
            else:
                frame[4, _origin(e.lane) + 1] = WHITE
        elif game.visual_phase.startswith("ripple") and game.ripple_point is not None:
            x, y = game.ripple_point
            radius = 1 if game.visual_phase == "ripple1" else 2
            for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
                if 0 <= x + dx < 64 and 0 <= y + dy < 64:
                    frame[y + dy, x + dx] = NEUTRAL
        elif game.visual_phase == "perimeter":
            frame[0, :] = NEUTRAL
            frame[63, :] = NEUTRAL
            frame[:, 0] = NEUTRAL
            frame[:, 63] = NEUTRAL
        elif game.visual_phase == "success":
            frame[0, :] = SUCCESS
            frame[63, :] = SUCCESS
            frame[:, 0] = SUCCESS
            frame[:, 63] = SUCCESS
        elif game.visual_phase == "failure":
            frame[0, :] = MAROON
            frame[63, :] = MAROON
            frame[:, 0] = MAROON
            frame[:, 63] = MAROON


class PrismBlockers(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.display = PrismBlockersDisplay()
        background = Sprite(np.full((64, 64), BLACK, dtype=np.int8), name="background", collidable=False)
        levels = [
            Level([background.clone()], (64, 64), {"spec": spec}, f"Level {idx + 1}: {spec.name}")
            for idx, spec in enumerate(LEVEL_SPECS)
        ]
        camera = Camera(0, 0, 64, 64, BLACK, BLACK, [self.display])
        super().__init__("prism_blockers-0001", levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6, 7], seed)
        self.display.game = self

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.prisms: list[Prism] = []
        legend = {"r": (RED, RED), "R": (BLUE, RED), "b": (BLUE, BLUE), "B": (RED, BLUE)}
        for y, row in enumerate(spec.rows, 1):
            if len(row) != 9:
                raise ValueError("Prism Blockers rows must be exactly nine characters")
            for x, symbol in enumerate(row, 1):
                if symbol in legend:
                    current, target = legend[symbol]
                    self.prisms.append(Prism(f"p{x}_{y}", x, y, current, target))
        self.emitters = [
            Emitter(eid, side, lane, RED if color == "R" else BLUE) for eid, side, lane, color in spec.emitters
        ]
        self.step_budget = spec.baseline * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[int, ...]] = []
        self.visual_phase = "settled"
        self.animation_queue: list[str] = []
        self.active_emitter: Emitter | None = None
        self.active_hit: Prism | None = None
        self.passed_prisms: list[Prism] = []
        self.old_colors: dict[str, int] = {}
        self.ripple_point: tuple[int, int] | None = None

    def _ordered_prisms(self, emitter: Emitter) -> list[Prism]:
        if emitter.side in {"W", "E"}:
            prisms = [p for p in self.prisms if p.y == emitter.lane]
            return sorted(prisms, key=lambda p: p.x, reverse=emitter.side == "E")
        prisms = [p for p in self.prisms if p.x == emitter.lane]
        return sorted(prisms, key=lambda p: p.y, reverse=emitter.side == "S")

    def _emitter_at(self, px: int, py: int) -> Emitter | None:
        for emitter in self.emitters:
            x0, x1, y0, y1 = emitter.hitbox
            if x0 <= px <= x1 and y0 <= py <= y1:
                return emitter
        return None

    def _consume_step(self) -> bool:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.remaining_steps == 0:
            self.visual_phase = "failure"
            self.lose()
            self.complete_action()
            return True
        return False

    def step(self) -> None:
        if self.animation_queue:
            phase = self.animation_queue.pop(0)
            if phase == "terminal":
                self.visual_phase = "settled"
                self.next_level()
                self.complete_action()
                return
            self.visual_phase = phase
            if not self.animation_queue:
                if phase == "settled" and all(p.current == p.target for p in self.prisms):
                    self.animation_queue = ["success", "terminal"]
                else:
                    self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self.visual_phase = "settled"
            self.complete_action()
            return

        if self._consume_step():
            return

        if self.action.id == GameAction.ACTION7:
            if self.undo_history:
                colors = self.undo_history.pop()
                for prism, color in zip(self.prisms, colors, strict=True):
                    prism.current = color
            self.visual_phase = "settled"
            self.complete_action()
            return

        if self.action.id != GameAction.ACTION6:
            self.visual_phase = "perimeter"
            self.animation_queue = ["settled"]
            return

        px = int(self.action.data.get("x", 0))
        py = int(self.action.data.get("y", 0))
        emitter = self._emitter_at(px, py)
        if emitter is None:
            self.ripple_point = (px, py)
            self.visual_phase = "ripple1"
            self.animation_queue = ["ripple2"]
            return

        self.active_emitter = emitter
        self.active_hit = None
        self.passed_prisms = []
        self.old_colors = {p.prism_id: p.current for p in self.prisms}
        for prism in self._ordered_prisms(emitter):
            self.passed_prisms.append(prism)
            if prism.current != emitter.color:
                self.active_hit = prism
                self.undo_history.append(tuple(p.current for p in self.prisms))
                prism.current = emitter.color
                break
        self.visual_phase = "source"
        impact = "impact" if self.active_hit is not None else "exit"
        self.animation_queue = ["path", impact, "settled"]

    def _get_hidden_state(self) -> np.ndarray:
        values = [0 if p.current == RED else 1 for p in self.prisms]
        values.extend([self.level_index, self.remaining_steps])
        return np.asarray(values, dtype=np.int16)
