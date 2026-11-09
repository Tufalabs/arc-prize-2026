from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "powered_switchboard-0001"
VIEW_SIZE = 64

ACTION_CLICK = 6
ACTION_UNDO = 7

BACKGROUND = 4
HOT_CENTER = 0
PLATE_CORNER = 1
DARK_CABLE = 2
DARK_SWITCH = 3
DARK_CENTER = 5
CLICK_OUTLINE = 10
POWER = 11
SOURCE_PAD = 12
UNLIT_BULB = 13
COMPLETE_HALO = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    # Readable logical diagrams are the source of the authored module set.  The
    # pixel centers preserve the deliberately irregular 64x64 composition.
    layout: tuple[str, ...]
    centers: dict[str, tuple[int, int]]
    switches: tuple[str, ...]
    bulbs: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    initially_closed: tuple[tuple[str, str], ...]
    solution: tuple[str, ...]

    @property
    def budget(self) -> int:
        return 6 * len(self.solution)


def _chain(prefix: str, count: int) -> tuple[tuple[str, str], ...]:
    return tuple((f"{prefix}{idx}", f"{prefix}{idx + 1}") for idx in range(1, count))


def _words(*chunks: str) -> tuple[str, ...]:
    return tuple(" ".join(chunks).split())


LEVEL_SPECS = (
    LevelSpec(
        "First Contact", ("R -------- G",), {"R": (18, 32), "G": (46, 32)}, ("R",), ("G",), (("R", "G"),), (), ("R",)
    ),
    LevelSpec(
        "Cut Your Own Lead",
        ("R ------ A ------ G",),
        {"R": (12, 32), "A": (34, 32), "G": (54, 32)},
        ("R", "A"),
        ("G",),
        (("R", "A"), ("A", "G")),
        (("R", "A"),),
        ("A", "R"),
    ),
    LevelSpec(
        "Hand-Off",
        ("    U", "  /   \\", "R       H --- Q --- G", "  \\   /", "    V"),
        {"R": (8, 32), "U": (22, 16), "V": (22, 48), "H": (38, 32), "Q": (50, 32), "G": (60, 32)},
        ("R", "U", "V", "H", "Q"),
        ("G",),
        (("R", "U"), ("R", "V"), ("U", "H"), ("V", "H"), ("H", "Q"), ("Q", "G")),
        (("R", "U"), ("R", "V"), ("U", "H")),
        ("H", "Q", "H"),
    ),
    LevelSpec(
        "Branch Repair",
        (
            "              P--GP",
            "            /",
            "R--J--U--Q--GQ",
            "   \\",
            "    K--M--N--T--GT",
            "     \\",
            "      S--GS",
        ),
        {
            "R": (5, 34),
            "J": (15, 34),
            "K": (25, 44),
            "U": (25, 20),
            "M": (35, 44),
            "P": (37, 12),
            "N": (44, 44),
            "Q": (37, 28),
            "T": (54, 44),
            "S": (35, 56),
            "GP": (50, 12),
            "GQ": (50, 28),
            "GT": (61, 44),
            "GS": (49, 56),
        },
        ("R", "J", "K", "U", "M", "P", "N", "Q", "T", "S"),
        ("GP", "GQ", "GT", "GS"),
        (
            ("R", "J"),
            ("J", "K"),
            ("J", "U"),
            ("K", "M"),
            ("U", "P"),
            ("M", "N"),
            ("U", "Q"),
            ("N", "T"),
            ("K", "S"),
            ("P", "GP"),
            ("Q", "GQ"),
            ("T", "GT"),
            ("S", "GS"),
        ),
        (),
        tuple("R J R K J R M K J R U N M K J R P Q U T N M S K J R".split()),
    ),
    LevelSpec(
        "Alternating Clutch",
        (
            "    U       N1--N2--N3--N7--G7",
            "  /   \\         \\",
            "R       H          N4--N5--N6--N8--G8",
            "  /   /               /",
            "    V      N9--N10--G10   N11--G11",
        ),
        {
            "R": (5, 32),
            "U": (15, 15),
            "V": (15, 49),
            "H": (26, 32),
            "N1": (34, 22),
            "N2": (42, 22),
            "N3": (49, 12),
            "N4": (49, 28),
            "N5": (49, 38),
            "N6": (49, 48),
            "N7": (56, 12),
            "N8": (56, 56),
            "N9": (34, 46),
            "N10": (38, 56),
            "N11": (56, 44),
            "G7": (62, 12),
            "G8": (62, 56),
            "G10": (38, 62),
            "G11": (62, 44),
        },
        ("R", "U", "V", "H", *(f"N{i}" for i in range(1, 12))),
        ("G7", "G8", "G10", "G11"),
        (
            ("R", "U"),
            ("R", "V"),
            ("U", "H"),
            ("V", "H"),
            ("H", "N1"),
            ("N1", "N2"),
            ("N2", "N3"),
            ("N2", "N4"),
            ("N4", "N5"),
            ("N5", "N6"),
            ("N3", "N7"),
            ("N6", "N8"),
            ("H", "N9"),
            ("N9", "N10"),
            ("N6", "N11"),
            ("N7", "G7"),
            ("N8", "G8"),
            ("N10", "G10"),
            ("N11", "G11"),
        ),
        (("R", "U"), ("R", "V"), ("U", "H")),
        _words(
            "H N1 H N2 N1 H N3 N4 N2 N1 H N5 N4 N2 N1 H N6 N5", "N4 N2 N1 N9 H N7 N3 N8 N10 N9 N11 N6 N5 N4 N2 N1 H"
        ),
    ),
    LevelSpec(
        "Offset Bridge",
        ("    A1--A2--A3--A4--A5--A6--A7--A8--GA", "R  /              |", "    B1--B2--B3--B4--B5--B6--B7--B8--GB"),
        {
            "R": (3, 32),
            **{f"A{i}": (7 + 7 * (i - 1), 17) for i in range(1, 9)},
            **{f"B{i}": (7 + 7 * (i - 1), 47) for i in range(1, 9)},
            "GA": (62, 17),
            "GB": (62, 47),
        },
        ("R", *(f"A{i}" for i in range(1, 9)), *(f"B{i}" for i in range(1, 9))),
        ("GA", "GB"),
        (("R", "A1"), ("R", "B1"), *_chain("A", 8), *_chain("B", 8), ("A4", "B4"), ("A8", "GA"), ("B8", "GB")),
        (("A4", "A5"),),
        _words(
            "R A1 B1 R A2 A1 B2 B1 R A3 A2 A1 B3 B2 B1 R A5 A4 A6 A5 A4 A7",
            "A6 A5 A4 A8 A7 A6 A5 A4 B4 B5 B4 B6 B5 B4 B7 B6 B5 B4 B8 B7 B6 B5 B4",
        ),
    ),
    LevelSpec(
        "Finished Side",
        (
            "    A1--A2--A3--A4--A5--A6--A7--A8--A9--GA",
            "R  /     |           |",
            "    B1--B2--B3--B4--B5--B6--B7--B8--B9--GB",
        ),
        {
            "R": (3, 32),
            **{f"A{i}": (7 + 6 * (i - 1), 17) for i in range(1, 10)},
            **{f"B{i}": (7 + 6 * (i - 1), 47) for i in range(1, 10)},
            "GA": (62, 17),
            "GB": (62, 47),
        },
        ("R", *(f"A{i}" for i in range(1, 10)), *(f"B{i}" for i in range(1, 10))),
        ("GA", "GB"),
        (
            ("R", "A1"),
            ("R", "B1"),
            *_chain("A", 9),
            *_chain("B", 9),
            ("A2", "B2"),
            ("A5", "B5"),
            ("A9", "GA"),
            ("B9", "GB"),
        ),
        (("R", "A1"), ("A5", "A6")),
        _words(
            "R B1 R A1 B2 R B3 B2 R B4 B3 B2 R B5 B4 B3 B2 A2 R A3 A2 A6 A5 B6 B5 R",
            "A7 A6 B7 B6 B5 A5 R A8 A7 A6 A5 B8 B7 B6 B5 R A9 A8 A7 A6 B9 B8 B7 B6 B5 A5 R",
        ),
    ),
)


def _edge_key(a: str, b: str) -> frozenset[str]:
    return frozenset((a, b))


def _parse_layout(spec: LevelSpec) -> set[str]:
    return set(re.findall(r"[A-Z]+\d*", "\n".join(spec.layout)))


def _validate_specs() -> None:
    for spec in LEVEL_SPECS:
        authored = set(spec.switches) | set(spec.bulbs)
        parsed = _parse_layout(spec)
        if not authored <= parsed:
            raise ValueError(f"{spec.name}: ASCII layout is missing {sorted(authored - parsed)}")
        if spec.switches[0] != "R" or len(spec.solution) * 6 != spec.budget:
            raise ValueError(f"{spec.name}: invalid root or budget")


_validate_specs()


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """8-connected Bresenham; exact decision ties take the x update first."""
    x, y = a
    x2, y2 = b
    dx, dy = abs(x2 - x), abs(y2 - y)
    sx, sy = (1 if x < x2 else -1), (1 if y < y2 else -1)
    err = dx - dy
    points = [(x, y)]
    while (x, y) != (x2, y2):
        twice = 2 * err
        if twice >= -dy:  # x is deliberately evaluated before y on a tie
            err -= dy
            x += sx
        if twice <= dx:
            err += dx
            y += sy
        points.append((x, y))
    return points


class StepBar(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: PoweredSwitchboard | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.game is None:
            return frame
        frame[0:2, :] = DARK_CABLE
        width = int(VIEW_SIZE * self.game.remaining_steps / max(1, self.game.step_budget))
        if width:
            frame[0:2, :width] = SOURCE_PAD
        return frame


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), BACKGROUND, dtype=np.int8),
        name="circuit",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["circuit"],
    )
    return Level([board], (VIEW_SIZE, VIEW_SIZE), {"spec": spec}, spec.name)


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


class PoweredSwitchboard(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = StepBar()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BACKGROUND, BACKGROUND, [self.hud]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )
        self.hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.board = level.get_sprites_by_tag("circuit")[0]
        self.edge_closed = {
            _edge_key(*edge): _edge_key(*edge) in {_edge_key(*e) for e in self.spec.initially_closed}
            for edge in self.spec.edges
        }
        self.incident: dict[str, list[frozenset[str]]] = defaultdict(list)
        for edge in self.edge_closed:
            for endpoint in edge:
                self.incident[endpoint].append(edge)
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self.step_count = 0
        self.done = False
        self.undo_history: list[dict[frozenset[str], bool]] = []
        self._animation: deque[tuple[str, object]] = deque()
        self._pending_result: str | None = None
        self._render_settled()

    def _powered(self, edge_state: dict[frozenset[str], bool] | None = None) -> set[str]:
        state = self.edge_closed if edge_state is None else edge_state
        powered = {"R"}
        queue = deque(["R"])
        while queue:
            vertex = queue.popleft()
            for edge in self.incident[vertex]:
                if not state[edge]:
                    continue
                other = next(item for item in edge if item != vertex)
                if other not in powered:
                    powered.add(other)
                    queue.append(other)
        return powered

    def _route(self, edge: frozenset[str]) -> list[tuple[int, int]]:
        a, b = tuple(edge)
        route = _bresenham(self.spec.centers[a], self.spec.centers[b])
        switch_set = set(self.spec.switches)

        def in_module(point: tuple[int, int], vertex: str) -> bool:
            px, py = point
            cx, cy = self.spec.centers[vertex]
            if vertex in switch_set:
                return abs(px - cx) <= 1 and abs(py - cy) <= 1
            return abs(px - cx) + abs(py - cy) <= 1

        return [point for point in route if not in_module(point, a) and not in_module(point, b)]

    @staticmethod
    def _draw_outline(frame: np.ndarray, center: tuple[int, int], color: int) -> None:
        cx, cy = center
        for x in range(max(0, cx - 2), min(VIEW_SIZE, cx + 3)):
            for y in (cy - 2, cy + 2):
                if 0 <= y < VIEW_SIZE:
                    frame[y, x] = color
        for y in range(max(0, cy - 2), min(VIEW_SIZE, cy + 3)):
            for x in (cx - 2, cx + 2):
                if 0 <= x < VIEW_SIZE:
                    frame[y, x] = color

    def _draw_cable(
        self,
        frame: np.ndarray,
        edge: frozenset[str],
        closed: bool,
        powered: set[str],
        structural: bool = False,
        progress: float = 1.0,
    ) -> None:
        route = self._route(edge)
        a, b = tuple(edge)
        if structural and progress < 1.0:
            maximum_gap = min(3, len(route) - 2)
            old_gap = 0 if closed else maximum_gap
            new_gap = maximum_gap - old_gap
            shown_gap = round(old_gap + (new_gap - old_gap) * progress)
            if shown_gap:
                gap_start = (len(route) - shown_gap) // 2
                shown = route[:gap_start] + route[gap_start + shown_gap :]
            else:
                shown = route
            for x, y in shown:
                frame[y, x] = DARK_CABLE
            return
        if closed:
            color = DARK_CABLE if structural else (POWER if a in powered and b in powered else DARK_CABLE)
            for x, y in route:
                frame[y, x] = color
            return

        gap = min(3, len(route) - 2)
        start = (len(route) - gap) // 2
        left, right = route[:start], route[start + gap :]
        for x, y in left:
            frame[y, x] = DARK_CABLE if structural else (POWER if a in powered else DARK_CABLE)
        for x, y in right:
            frame[y, x] = DARK_CABLE if structural else (POWER if b in powered else DARK_CABLE)

    def _compose(
        self,
        *,
        edge_state: dict[frozenset[str], bool] | None = None,
        powered: set[str] | None = None,
        structural_edges: set[frozenset[str]] | None = None,
        progress: float = 1.0,
        click_outline: str | None = None,
        ripple: tuple[int, int, int] | None = None,
        halo: bool = False,
    ) -> np.ndarray:
        state = self.edge_closed if edge_state is None else edge_state
        live = self._powered(state) if powered is None else powered
        structural_edges = structural_edges or set()
        frame = np.full((VIEW_SIZE, VIEW_SIZE), BACKGROUND, dtype=np.int8)

        for edge, closed in state.items():
            self._draw_cable(
                frame, edge, closed, live, edge in structural_edges, progress if edge in structural_edges else 1.0
            )

        root_x, root_y = self.spec.centers["R"]
        for y in range(max(0, root_y - 3), min(VIEW_SIZE, root_y + 4)):
            for x in range(max(0, root_x - 3), min(VIEW_SIZE, root_x + 4)):
                if abs(x - root_x) + abs(y - root_y) <= 3:
                    frame[y, x] = SOURCE_PAD

        for switch in self.spec.switches:
            cx, cy = self.spec.centers[switch]
            is_powered = switch in live or switch == "R"
            pixels = (
                (PLATE_CORNER, POWER if is_powered else DARK_SWITCH, PLATE_CORNER),
                (
                    POWER if is_powered else DARK_SWITCH,
                    HOT_CENTER if is_powered else DARK_CENTER,
                    POWER if is_powered else DARK_SWITCH,
                ),
                (PLATE_CORNER, POWER if is_powered else DARK_SWITCH, PLATE_CORNER),
            )
            frame[cy - 1 : cy + 2, cx - 1 : cx + 2] = np.asarray(pixels, dtype=np.int8)

        for bulb in self.spec.bulbs:
            cx, cy = self.spec.centers[bulb]
            lit = bulb in live
            outer = POWER if lit else UNLIT_BULB
            for x, y in ((cx, cy - 1), (cx - 1, cy), (cx, cy), (cx + 1, cy), (cx, cy + 1)):
                if 0 <= x < VIEW_SIZE and 0 <= y < VIEW_SIZE:
                    frame[y, x] = (
                        HOT_CENTER if (x, y) == (cx, cy) and lit else (DARK_CENTER if (x, y) == (cx, cy) else outer)
                    )
            if halo:
                self._draw_outline(frame, (cx, cy), COMPLETE_HALO)

        if click_outline is not None:
            self._draw_outline(frame, self.spec.centers[click_outline], CLICK_OUTLINE)
        if ripple is not None:
            cx, cy, radius = ripple
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                for x, y in ((cx + delta, cy + delta), (cx + delta, cy - delta)):
                    if 0 <= x < VIEW_SIZE and 0 <= y < VIEW_SIZE:
                        frame[y, x] = DARK_CABLE
        return frame

    def _render_settled(self) -> None:
        self.board.pixels = self._compose()

    def _switch_at(self, x: int, y: int) -> str | None:
        for switch in self.spec.switches:
            cx, cy = self.spec.centers[switch]
            if abs(x - cx) <= 1 and abs(y - cy) <= 1:
                return switch
        return None

    def _spend_action(self) -> None:
        self.step_count += 1
        self.remaining_steps = max(0, self.remaining_steps - 1)

    def _finish_after_animation(self) -> None:
        if self._pending_result == "win":
            self.done = True
            self.next_level()
        elif self._pending_result == "lose":
            self.lose()
        self._pending_result = None
        self.complete_action()

    def _play_animation(self) -> None:
        kind, payload = self._animation.popleft()
        if kind == "frame":
            self.board.pixels = payload
        if not self._animation:
            self._finish_after_animation()

    def _queue_invalid(self, point: tuple[int, int]) -> None:
        for radius in (1, 2, 1):
            self._animation.append(("frame", self._compose(ripple=(*point, radius))))

    def _queue_toggle(self, switch: str, old_state: dict[frozenset[str], bool]) -> None:
        changed = set(self.incident[switch])
        old_power = self._powered(old_state)
        final_power = self._powered()
        self._animation.extend(
            (
                ("frame", self._compose(edge_state=old_state, powered=old_power, click_outline=switch)),
                (
                    "frame",
                    self._compose(
                        edge_state=old_state,
                        powered=old_power,
                        structural_edges=changed,
                        progress=0.35,
                        click_outline=switch,
                    ),
                ),
                (
                    "frame",
                    self._compose(
                        edge_state=old_state,
                        powered=old_power,
                        structural_edges=changed,
                        progress=0.7,
                        click_outline=switch,
                    ),
                ),
                ("frame", self._compose(powered=set(), structural_edges=changed, progress=1.0, click_outline=switch)),
                ("frame", self._compose(powered=final_power, click_outline=switch)),
            )
        )
        if set(self.spec.bulbs) <= final_power:
            self._pending_result = "win"
            for halo_visible in (True, True, False, False, True, True):
                self._animation.append(("frame", self._compose(powered=final_power, halo=halo_visible)))
        elif self.remaining_steps == 0:
            self._pending_result = "lose"

    def step(self) -> None:
        if self._animation:
            self._play_animation()
            return

        action_id = _action_id(self.action.id)
        if action_id == GameAction.RESET.value:
            self._render_settled()
            self.complete_action()
            return

        self._spend_action()

        if action_id == ACTION_UNDO:
            if self.undo_history:
                self.edge_closed = self.undo_history.pop()
            self._render_settled()
            if self.remaining_steps == 0:
                self.lose()
            self.complete_action()
            return

        if action_id != ACTION_CLICK:
            self._render_settled()
            if self.remaining_steps == 0:
                self.lose()
            self.complete_action()
            return

        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        switch = self._switch_at(x, y)
        pre_power = self._powered()
        if switch is None or switch not in pre_power:
            self._queue_invalid((x, y))
            if self.remaining_steps == 0:
                self._pending_result = "lose"
            self._play_animation()
            return

        old_state = dict(self.edge_closed)
        self.undo_history.append(old_state)
        for edge in self.incident[switch]:
            self.edge_closed[edge] = not self.edge_closed[edge]
        self._queue_toggle(switch, old_state)
        self._play_animation()
