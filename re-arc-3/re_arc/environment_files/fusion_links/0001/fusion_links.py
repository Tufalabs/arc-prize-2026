from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "fusion_links-0001"
BACKGROUND = 5
BASELINES = (2, 5, 6, 31, 47, 55, 63)


@dataclass(frozen=True)
class SocketSpec:
    socket_id: str
    center: tuple[int, int]
    tier: int


@dataclass(frozen=True)
class LinkSpec:
    link_id: str
    u: str
    v: str
    target: str
    bead: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    sockets: tuple[SocketSpec, ...]
    links: tuple[LinkSpec, ...]
    goal: str
    goal_tier: int
    budget: int
    solution: tuple[str, ...]


POD_OFFSETS = {
    "r": (0, 0),
    "a": (-5, -5),
    "b": (-5, 0),
    "c": (-10, -10),
    "d": (-5, 5),
    "e": (-15, -10),
    "f": (-5, -10),
    "g": (-10, -5),
}
POD_LINKS = {
    "rd": ("r", "d", "r", (-3, 3)),
    "bg": ("b", "g", "b", (-8, -3)),
    "rb": ("r", "b", "r", (-3, 0)),
    "af": ("a", "f", "a", (-5, -8)),
    "ce": ("c", "e", "c", (-13, -10)),
    "ac": ("a", "c", "a", (-8, -8)),
    "ra": ("r", "a", "r", (-3, -3)),
}
VARIANTS = {
    "P8": ({k: 0 for k in "rabcdefg"}, tuple(POD_LINKS)),
    "P8+1": ({k: 1 for k in "rabcdefg"}, tuple(POD_LINKS)),
    "P7R": ({"r": 1, **{k: 0 for k in "abcefg"}}, ("bg", "rb", "af", "ce", "ac", "ra")),
    "P7A": ({"a": 1, **{k: 0 for k in "rbcdeg"}}, ("rd", "bg", "rb", "ce", "ac", "ra")),
    "P6RA": ({"r": 1, "a": 1, **{k: 0 for k in "bceg"}}, ("bg", "rb", "ce", "ac", "ra")),
    "P5R": ({"r": 2, **{k: 0 for k in "acef"}}, ("af", "ce", "ac", "ra")),
    "P5A": ({"a": 2, **{k: 0 for k in "rbdg"}}, ("rd", "bg", "rb", "ra")),
    "P4": ({k: 1 for k in "rbac"}, ("rb", "ac", "ra")),
    "P2": ({"r": 2, "a": 2}, ("ra",)),
}


def _transform(point: tuple[int, int], orientation: str) -> tuple[int, int]:
    x, y = point
    if orientation == "W":
        return x, y
    if orientation == "N":
        return -y, x
    if orientation == "E":
        return -x, -y
    return y, -x


def _pod(name: str, root: tuple[int, int], orientation: str, variant: str) -> tuple[list[SocketSpec], list[LinkSpec]]:
    tiers, visible_links = VARIANTS[variant]
    positions: dict[str, tuple[int, int]] = {}
    sockets: list[SocketSpec] = []
    for local, tier in tiers.items():
        dx, dy = _transform(POD_OFFSETS[local], orientation)
        positions[local] = (root[0] + dx, root[1] + dy)
        sockets.append(SocketSpec(f"{name}.{local}", positions[local], tier))
    links: list[LinkSpec] = []
    for local_link in visible_links:
        u, v, target, bead_offset = POD_LINKS[local_link]
        bx, by = _transform(bead_offset, orientation)
        links.append(
            LinkSpec(
                f"{name}.{local_link}", f"{name}.{u}", f"{name}.{v}", f"{name}.{target}", (root[0] + bx, root[1] + by)
            )
        )
    return sockets, links


def _simple_level(
    name: str,
    rows: tuple[str, ...],
    coordinates: dict[str, tuple[int, int]],
    tiers: dict[str, int],
    raw_links: tuple[tuple[str, str, str, str, tuple[int, int]], ...],
    goal: str,
    goal_tier: int,
    solution: tuple[str, ...],
) -> LevelSpec:
    # The ASCII diagram is the authoritative logical membership declaration.
    symbols = {char for row in rows for char in row if char.isalpha()}
    if symbols != set(coordinates) or symbols != set(tiers):
        raise ValueError(f"ASCII/socket mismatch in {name}")
    sockets = tuple(SocketSpec(key, coordinates[key], tiers[key]) for key in coordinates)
    links = tuple(LinkSpec(link_id, u, v, target, bead) for link_id, u, v, target, bead in raw_links)
    return LevelSpec(name, rows, sockets, links, goal, goal_tier, len(solution) * 6, solution)


def _pod_level(
    name: str,
    rows: tuple[str, ...],
    pods: tuple[tuple[str, tuple[int, int], str, str], ...],
    macros: tuple[tuple[str, str, str, str, tuple[int, int]], ...],
    goal: str,
    goal_tier: int,
    solution: tuple[str, ...],
) -> LevelSpec:
    declared = {char for row in rows for char in row if char.isupper()}
    if declared != {pod[0] for pod in pods}:
        raise ValueError(f"ASCII/pod mismatch in {name}")
    sockets: list[SocketSpec] = []
    links: list[LinkSpec] = []
    for pod_name, root, orientation, variant in pods:
        pod_sockets, pod_links = _pod(pod_name, root, orientation, variant)
        sockets.extend(pod_sockets)
        links.extend(pod_links)
    links.extend(LinkSpec(*macro) for macro in macros)
    return LevelSpec(name, rows, tuple(sockets), tuple(links), goal, goal_tier, len(solution) * 6, solution)


def _seq(pod: str, links: str = "rd bg rb af ce ac ra") -> tuple[str, ...]:
    return tuple(f"{pod}.{link}" for link in links.split())


L1 = ("ab", "bg")
L2 = ("ab", "bc", "cd", "de", "eg")
L3 = ("ab", "bx", "xy", "yz", "zh", "hg")
L4 = _seq("F") + _seq("D") + ("M.FD",) + _seq("B") + _seq("E") + ("M.BE", "M.FB")
L5 = (
    _seq("A", "ra")
    + _seq("E")
    + ("M.AE",)
    + _seq("C", "rb ac ra")
    + _seq("G", "bg rb ce ac ra")
    + ("M.CG", "M.AC")
    + _seq("R", "af ce ac ra")
    + _seq("D")
    + ("M.RD",)
    + _seq("B", "rd bg rb ce ac ra")
    + _seq("F")
    + ("M.BF", "M.RB", "M.RA")
)
L6 = (
    _seq("F", "rd bg rb ra")
    + _seq("D", "bg rb af ce ac ra")
    + ("M.FD",)
    + _seq("B")
    + _seq("R")
    + ("M.BR", "M.DR")
    + _seq("E", "af ce ac ra")
    + _seq("A")
    + ("M.EA",)
    + _seq("G", "rd bg rb ce ac ra")
    + _seq("C")
    + ("M.GC", "M.AC", "M.RA")
)
L7 = (
    _seq("F")
    + _seq("D")
    + ("M.FD",)
    + _seq("B")
    + _seq("E")
    + ("M.BE", "M.FB")
    + _seq("R")
    + _seq("G")
    + ("M.RG",)
    + _seq("A")
    + _seq("C")
    + ("M.AC", "M.RA", "M.BR")
)

LEVEL_SPECS = (
    _simple_level(
        "One Size Short",
        ("a---b---g",),
        {"a": (18, 31), "b": (32, 31), "g": (46, 31)},
        {"a": 0, "b": 0, "g": 1},
        (("ab", "a", "b", "b", (25, 31)), ("bg", "b", "g", "g", (39, 31))),
        "g",
        2,
        L1,
    ),
    _simple_level(
        "Carry the Fusion",
        ("a--b", "...|", "...c--d", "......|", "......e--g"),
        {"a": (10, 11), "b": (22, 11), "c": (22, 25), "d": (38, 25), "e": (38, 41), "g": (52, 41)},
        {"a": 0, "b": 0, "c": 1, "d": 2, "e": 3, "g": 4},
        (
            ("ab", "a", "b", "b", (16, 11)),
            ("bc", "b", "c", "c", (22, 18)),
            ("cd", "c", "d", "d", (30, 25)),
            ("de", "d", "e", "e", (38, 33)),
            ("eg", "e", "g", "g", (45, 41)),
        ),
        "g",
        5,
        L2,
    ),
    _simple_level(
        "Meet the Large Front",
        ("x--b--a", "|......", "y---g..", "|...|..", "z---h.."),
        {"a": (50, 13), "b": (36, 13), "x": (22, 13), "y": (22, 29), "z": (22, 45), "h": (38, 45), "g": (38, 29)},
        {"a": 0, "b": 0, "x": 1, "y": 2, "z": 3, "h": 4, "g": 5},
        (
            ("ab", "a", "b", "b", (43, 13)),
            ("bx", "b", "x", "x", (29, 13)),
            ("xy", "x", "y", "y", (22, 21)),
            ("yz", "y", "z", "z", (22, 37)),
            ("zh", "z", "h", "h", (30, 45)),
            ("hg", "h", "g", "g", (38, 37)),
        ),
        "g",
        6,
        L3,
    ),
    _pod_level(
        "Shared Roots",
        ("F B E", "D . ."),
        (
            ("F", (16, 16), "W", "P8+1"),
            ("D", (16, 32), "W", "P8+1"),
            ("B", (32, 16), "N", "P8+1"),
            ("E", (48, 16), "N", "P8+1"),
        ),
        (
            ("M.FD", "F.r", "D.r", "F.r", (16, 24)),
            ("M.BE", "B.r", "E.r", "B.r", (40, 16)),
            ("M.FB", "F.r", "B.r", "B.r", (24, 16)),
        ),
        "B.r",
        6,
        L4,
    ),
    _pod_level(
        "Backward Split",
        ("F B E", "D R A", ". G C"),
        (
            ("R", (32, 32), "S", "P5R"),
            ("D", (16, 32), "W", "P8"),
            ("B", (32, 16), "N", "P7A"),
            ("F", (16, 16), "W", "P8"),
            ("A", (47, 32), "E", "P2"),
            ("E", (47, 16), "N", "P8"),
            ("C", (47, 47), "E", "P4"),
            ("G", (32, 47), "S", "P6RA"),
        ),
        (
            ("M.RD", "R.r", "D.r", "R.r", (24, 32)),
            ("M.AE", "A.r", "E.r", "A.r", (47, 24)),
            ("M.BF", "B.r", "F.r", "B.r", (24, 16)),
            ("M.CG", "C.r", "G.r", "C.r", (39, 47)),
            ("M.RB", "R.r", "B.r", "R.r", (32, 24)),
            ("M.AC", "A.r", "C.r", "A.r", (47, 39)),
            ("M.RA", "R.r", "A.r", "R.r", (39, 32)),
        ),
        "R.r",
        6,
        L5,
    ),
    _pod_level(
        "Carry the Whole Half",
        ("F B E", "D R A", ". G C"),
        (
            ("F", (16, 16), "W", "P5A"),
            ("D", (16, 32), "W", "P7R"),
            ("B", (32, 16), "N", "P8"),
            ("R", (32, 32), "S", "P8"),
            ("E", (47, 16), "N", "P5R"),
            ("A", (47, 32), "E", "P8"),
            ("G", (32, 47), "S", "P7A"),
            ("C", (47, 47), "E", "P8"),
        ),
        (
            ("M.FD", "F.r", "D.r", "D.r", (16, 24)),
            ("M.BR", "B.r", "R.r", "R.r", (32, 24)),
            ("M.DR", "D.r", "R.r", "R.r", (24, 32)),
            ("M.EA", "E.r", "A.r", "A.r", (47, 24)),
            ("M.GC", "G.r", "C.r", "C.r", (39, 47)),
            ("M.AC", "A.r", "C.r", "A.r", (47, 39)),
            ("M.RA", "R.r", "A.r", "A.r", (39, 32)),
        ),
        "A.r",
        6,
        L6,
    ),
    _pod_level(
        "Full Fusion Network",
        ("F B E", "D R A", ". G C"),
        (
            ("F", (16, 16), "W", "P8"),
            ("D", (16, 32), "W", "P8"),
            ("B", (32, 16), "N", "P8"),
            ("E", (47, 16), "N", "P8"),
            ("R", (32, 32), "W", "P8"),
            ("G", (32, 47), "S", "P8"),
            ("A", (47, 32), "E", "P8"),
            ("C", (47, 47), "E", "P8"),
        ),
        (
            ("M.FD", "F.r", "D.r", "F.r", (16, 24)),
            ("M.BE", "B.r", "E.r", "B.r", (39, 16)),
            ("M.FB", "F.r", "B.r", "B.r", (24, 16)),
            ("M.RG", "R.r", "G.r", "R.r", (32, 39)),
            ("M.AC", "A.r", "C.r", "A.r", (47, 39)),
            ("M.RA", "R.r", "A.r", "R.r", (39, 32)),
            ("M.BR", "B.r", "R.r", "R.r", (32, 24)),
        ),
        "R.r",
        6,
        L7,
    ),
)


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    points = []
    while True:
        points.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return points
        double = 2 * error
        if double >= dy:
            error += dy
            x0 += sx
        if double <= dx:
            error += dx
            y0 += sy


class FusionDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: FusionLinks | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.game is not None:
            return self.game._render()
        return frame


class FusionLinks(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = FusionDisplay()
        levels = []
        for spec in LEVEL_SPECS:
            markers = [
                Sprite(
                    [[-1]],
                    name=link.link_id,
                    x=link.bead[0],
                    y=link.bead[1],
                    visible=False,
                    collidable=False,
                    tags=["fusion_bead"],
                )
                for link in spec.links
            ]
            levels.append(Level(markers, (64, 64), {"spec": spec}, spec.name))
        camera = Camera(0, 0, 64, 64, BACKGROUND, BACKGROUND, [self._display])
        super().__init__(GAME_ID, levels, camera, False, 7, [1, 2, 3, 4, 5, 6, 7], seed)
        self._display.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.tokens = {socket.socket_id: socket.tier for socket in self.spec.sockets}
        self.positions = {socket.socket_id: socket.center for socket in self.spec.sockets}
        self.links = {link.link_id: link for link in self.spec.links}
        self.remaining_steps = self.spec.budget
        self.undo_history: list[dict[str, int | None]] = []
        self.animation: dict[str, object] | None = None
        self.flash_point: tuple[int, int] | None = None
        self.failure_reason: str | None = None
        self.failure_components: set[str] = set()
        self.completion_color = 14

    def _spend(self) -> bool:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        return self.remaining_steps == 0

    def _mergeable(self, link: LinkSpec) -> bool:
        return (
            self.tokens[link.u] is not None and self.tokens[link.u] == self.tokens[link.v] and self.tokens[link.u] < 6
        )

    def _occupied_components(self) -> list[set[str]]:
        occupied = {socket_id for socket_id, tier in self.tokens.items() if tier is not None}
        components: list[set[str]] = []
        while occupied:
            component = {occupied.pop()}
            frontier = list(component)
            while frontier:
                current = frontier.pop()
                for link in self.links.values():
                    if link.u == current:
                        other = link.v
                    elif link.v == current:
                        other = link.u
                    else:
                        continue
                    if self.tokens[other] is not None and other in occupied:
                        occupied.remove(other)
                        component.add(other)
                        frontier.append(other)
            components.append(component)
        return components

    def _commit_merge(self, link: LinkSpec, old_tier: int) -> str | None:
        source = link.v if link.target == link.u else link.u
        self.tokens[source] = None
        self.tokens[link.target] = old_tier + 1
        occupied = [socket_id for socket_id, tier in self.tokens.items() if tier is not None]
        if len(occupied) == 1 and self.tokens[self.spec.goal] == self.spec.goal_tier:
            return "success"
        components = self._occupied_components()
        if len(components) > 1:
            self.failure_reason = "SEVERED"
            goal_component = next((part for part in components if self.spec.goal in part), set())
            self.failure_components = set().union(*(part for part in components if part is not goal_component))
            return "failure"
        if not any(self._mergeable(candidate) for candidate in self.links.values()):
            self.failure_reason = "STALLED"
            return "failure"
        return None

    def _finish_budget_failure(self) -> None:
        self.failure_reason = "STALLED"
        self.lose()
        self.complete_action()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return
        if self.animation is not None:
            self._advance_animation()
            return
        self.flash_point = None
        action_id = self.action.id.value
        if action_id == 7:
            expired = self._spend()
            if self.undo_history:
                self.tokens = self.undo_history.pop()
                self.failure_reason = None
                self.failure_components.clear()
            if expired:
                self._finish_budget_failure()
            else:
                self.complete_action()
            return
        expired = self._spend()
        if action_id != 6:
            if expired:
                self._finish_budget_failure()
            else:
                self.complete_action()
            return
        x, y = int(self.action.data.get("x", 0)), int(self.action.data.get("y", 0))
        link = next(
            (
                candidate
                for candidate in self.links.values()
                if abs(x - candidate.bead[0]) <= 1 and abs(y - candidate.bead[1]) <= 1
            ),
            None,
        )
        if link is None:
            self.flash_point = (x, y)
            if expired:
                self._finish_budget_failure()
            else:
                self.complete_action()
            return
        if not self._mergeable(link):
            self.animation = {"kind": "invalid", "link": link, "phase": 1, "expired": expired}
            return
        self.undo_history.append(dict(self.tokens))
        self.animation = {"kind": "merge", "link": link, "phase": 1, "tier": self.tokens[link.u], "expired": expired}

    def _advance_animation(self) -> None:
        assert self.animation is not None
        phase = int(self.animation["phase"])
        kind = str(self.animation["kind"])
        if kind == "invalid":
            if phase < 2:
                self.animation["phase"] = phase + 1
                return
            expired = bool(self.animation["expired"])
            self.animation = None
            if expired:
                self._finish_budget_failure()
            else:
                self.complete_action()
            return
        link = self.animation["link"]
        assert isinstance(link, LinkSpec)
        if phase < 3:
            self.animation["phase"] = phase + 1
            return
        if phase == 3:
            outcome = self._commit_merge(link, int(self.animation["tier"]))
            if bool(self.animation["expired"]):
                outcome = "failure"
                self.failure_reason = "STALLED"
            self.animation["outcome"] = outcome
            if outcome == "success":
                self.animation["phase"] = 4
                return
            expired = bool(self.animation["expired"])
            self.animation = None
            if outcome == "failure" or expired:
                if expired and outcome is None:
                    self.failure_reason = "STALLED"
                self.lose()
            self.complete_action()
            return
        if phase == 4:
            self.completion_color = 0
            self.animation["phase"] = 5
            return
        self.completion_color = 14
        self.animation = None
        self.next_level()
        self.complete_action()

    @staticmethod
    def _paint_square(frame: np.ndarray, center: tuple[int, int], size: int, color: int) -> None:
        x, y = center
        left = x - size // 2
        top = y - size // 2
        frame[max(0, top) : min(62, top + size), max(0, left) : min(64, left + size)] = color

    @staticmethod
    def _paint_token(
        frame: np.ndarray,
        center: tuple[int, int],
        tier: int,
        shift: tuple[int, int] = (0, 0),
        shrink: int = 0,
        brighten: bool = False,
    ) -> None:
        diameter = max(2, tier + 2 - shrink)
        cx, cy = center[0] + shift[0], center[1] + shift[1]
        left, top = cx - diameter // 2, cy - diameter // 2
        radius = diameter / 2
        for py in range(diameter):
            for px in range(diameter):
                dx, dy = px + 0.5 - radius, py + 0.5 - radius
                if dx * dx + dy * dy > (radius + 0.15) ** 2:
                    continue
                distance = (dx * dx + dy * dy) ** 0.5
                band = max(0, int(radius - distance))
                color = 10 if brighten else (9 if band % 2 == 0 else 10)
                x, y = left + px, top + py
                if 0 <= x < 64 and 0 <= y < 62:
                    frame[y, x] = color

    def _render(self) -> np.ndarray:
        frame = np.full((64, 64), BACKGROUND, dtype=np.int8)
        invalid_link = None
        invalid_phase = 0
        merge_link = None
        merge_phase = 0
        if self.animation:
            if self.animation["kind"] == "invalid":
                invalid_link = self.animation["link"]
                invalid_phase = int(self.animation["phase"])
            else:
                merge_link = self.animation["link"]
                merge_phase = int(self.animation["phase"])
        hint_index = len(self.undo_history)
        hinted_id = self.spec.solution[hint_index] if hint_index < len(self.spec.solution) else None
        # Routes, beads and direction caps.
        for link in self.links.values():
            equal = self._mergeable(link)
            occupied = self.tokens[link.u] is not None and self.tokens[link.v] is not None
            body = 1 if equal else (3 if occupied else 4)
            route = _line(self.positions[link.u], self.positions[link.v])
            for x, y in route:
                if 0 <= x < 64 and 0 <= y < 62:
                    frame[y, x] = body if (x + y) % 2 else (2 if equal else body)
            bead_color = 11 if equal else 4
            if link is invalid_link and invalid_phase == 1:
                bead_color = 8
            self._paint_square(frame, link.bead, 3, bead_color)
            if equal:
                x, y = link.bead
                cue = 0 if link.link_id == hinted_id else 14
                for px, py in ((x - 2, y - 2), (x + 2, y - 2), (x - 2, y + 2), (x + 2, y + 2)):
                    if 0 <= px < 64 and 0 <= py < 62:
                        frame[py, px] = cue
            if occupied and not equal:
                x, y = link.bead
                if abs(self.positions[link.u][0] - self.positions[link.v][0]) >= abs(
                    self.positions[link.u][1] - self.positions[link.v][1]
                ):
                    frame[max(0, y - 1) : min(62, y + 2), x] = 3
                else:
                    frame[y, max(0, x - 1) : min(64, x + 2)] = 3
            target = self.positions[link.target]
            cap_path = _line(link.bead, target)
            cap = cap_path[-4] if len(cap_path) >= 4 else cap_path[-2]
            self._paint_square(frame, cap, 2, 12)
        # Socket plates.
        for center in self.positions.values():
            self._paint_square(frame, center, 3, 4)
        # Goal aperture outline.
        goal_center = self.positions[self.spec.goal]
        goal_diameter = self.spec.goal_tier + 2
        left, top = goal_center[0] - goal_diameter // 2 - 1, goal_center[1] - goal_diameter // 2 - 1
        size = goal_diameter + 2
        goal_ports: set[tuple[int, int]] = set()
        for link in self.links.values():
            if self.spec.goal not in (link.u, link.v):
                continue
            other = link.v if link.u == self.spec.goal else link.u
            route = _line(self.positions[other], goal_center)
            goal_ports.update(route[-(goal_diameter // 2 + 3) :])
        for offset in range(size):
            for x, y in (
                (left + offset, top),
                (left + offset, top + size - 1),
                (left, top + offset),
                (left + size - 1, top + offset),
            ):
                if 0 <= x < 64 and 0 <= y < 62 and (x, y) not in goal_ports:
                    frame[y, x] = self.completion_color
        # Tokens and invalid shake.
        for socket_id, tier in self.tokens.items():
            if tier is None:
                continue
            shift = (0, 0)
            if invalid_link is not None and socket_id in (invalid_link.u, invalid_link.v) and invalid_phase == 2:
                a, b = self.positions[invalid_link.u], self.positions[invalid_link.v]
                shift = (0, 1) if abs(a[0] - b[0]) >= abs(a[1] - b[1]) else (1, 0)
            brighten = merge_link is not None and socket_id == merge_link.target and merge_phase == 1
            shrink = (
                1
                if merge_link is not None
                and socket_id != merge_link.target
                and socket_id in (merge_link.u, merge_link.v)
                and merge_phase == 1
                else 0
            )
            self._paint_token(frame, self.positions[socket_id], tier, shift, shrink, brighten)
            if invalid_link is not None and invalid_phase == 1 and socket_id in (invalid_link.u, invalid_link.v):
                diameter = tier + 4
                x, y = self.positions[socket_id]
                for offset in range(diameter):
                    for px, py in (
                        (x - diameter // 2 + offset, y - diameter // 2),
                        (x - diameter // 2 + offset, y + diameter // 2),
                        (x - diameter // 2, y - diameter // 2 + offset),
                        (x + diameter // 2, y - diameter // 2 + offset),
                    ):
                        if 0 <= px < 64 and 0 <= py < 62:
                            frame[py, px] = 8
        # Stable source identity drives the travelling ghost.
        if merge_link is not None and merge_phase in (2, 3):
            source = merge_link.v if merge_link.target == merge_link.u else merge_link.u
            start, end = self.positions[source], self.positions[merge_link.target]
            fraction = merge_phase - 1
            ghost = (
                round(start[0] + (end[0] - start[0]) * fraction / 3),
                round(start[1] + (end[1] - start[1]) * fraction / 3),
            )
            self._paint_token(frame, ghost, int(self.animation["tier"]), brighten=True)
        if self.failure_reason == "SEVERED":
            for socket_id in self.failure_components:
                x, y = self.positions[socket_id]
                tier = self.tokens[socket_id]
                assert tier is not None
                diameter = tier + 4
                for offset in range(diameter):
                    for px, py in (
                        (x - diameter // 2 + offset, y - diameter // 2),
                        (x - diameter // 2 + offset, y + diameter // 2),
                        (x - diameter // 2, y - diameter // 2 + offset),
                        (x + diameter // 2, y - diameter // 2 + offset),
                    ):
                        if 0 <= px < 64 and 0 <= py < 62:
                            frame[py, px] = 13
        elif self.failure_reason == "STALLED":
            for link in self.links.values():
                self._paint_square(frame, link.bead, 3, 13)
        if self.flash_point is not None:
            x, y = self.flash_point
            if 0 <= x < 64 and 0 <= y < 62:
                frame[y, x] = 1
        # Required action-time bar; intentionally unlabelled and outside graph space.
        frame[62:64, :] = 4
        width = round(64 * self.remaining_steps / self.spec.budget)
        if width:
            frame[62:64, :width] = 12
        return frame
