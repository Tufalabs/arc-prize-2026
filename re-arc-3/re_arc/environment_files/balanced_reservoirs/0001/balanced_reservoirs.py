from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
BACKGROUND = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

GROUP_COLORS = (9, 6, 15, 12, 13)
BASELINES = (1, 6, 6, 27, 41, 44, 52)


@dataclass(frozen=True)
class GroupSpec:
    members: tuple[str, ...]
    badge: tuple[str, str]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    symbols: tuple[tuple[str, str, int], ...]
    groups: tuple[GroupSpec, ...]
    links: tuple[tuple[str, str, str], ...]
    reference_panel: bool = False


LEVEL_SPECS = (
    LevelSpec(
        "One Unit Level",
        ("....", "AB..", "....", "...."),
        (("A", "A", 2), ("B", "B", 0)),
        (GroupSpec(("A", "B"), ("B", "E")),),
        (("AB", "A", "B"),),
        True,
    ),
    LevelSpec(
        "Two Arrowheads",
        ("....", "LCR.", "....", "...."),
        (("L", "L", 0), ("C", "C", 9), ("R", "R", 0)),
        (GroupSpec(("L", "C", "R"), ("C", "N")),),
        (("LC", "L", "C"), ("CL", "C", "L"), ("CR", "C", "R"), ("RC", "R", "C")),
    ),
    LevelSpec(
        "Shared Rim",
        ("....", "ASB.", "....", "...."),
        (("A", "A", 7), ("S", "S", 1), ("B", "B", 1)),
        (GroupSpec(("A", "S"), ("A", "N")), GroupSpec(("S", "B"), ("B", "N"))),
        (("AS", "A", "S"), ("SA", "S", "A"), ("SB", "S", "B"), ("BS", "B", "S")),
    ),
    LevelSpec(
        "Raise the Zeroes",
        ("..GH", "..FE", ".ACD", "..B."),
        (
            ("H", "H", 8),
            ("G", "G", 0),
            ("F", "F", 0),
            ("E", "E", 0),
            ("D", "D", 0),
            ("C", "C", 0),
            ("A", "A", 0),
            ("B", "B", 0),
        ),
        (
            GroupSpec(("H", "G", "F", "E"), ("G", "W")),
            GroupSpec(("E", "D", "C"), ("E", "N")),
            GroupSpec(("F", "C", "A", "B"), ("F", "W")),
        ),
        (
            ("HG", "H", "G"),
            ("GF", "G", "F"),
            ("FE", "F", "E"),
            ("ED", "E", "D"),
            ("DC", "D", "C"),
            ("CA", "C", "A"),
            ("CB", "C", "B"),
        ),
    ),
    LevelSpec(
        "Balanced Conduit",
        (".21S", ".3..", "AJ14", "a.23"),
        (
            ("S", "S", 8),
            ("1", "C1", 1),
            ("2", "C2", 1),
            ("3", "C3", 1),
            ("J", "J", 0),
            ("A", "A1", 0),
            ("a", "A2", 0),
            ("1", "C1", 1),
            ("4", "B4", 0),
            ("2", "C2", 1),
            ("3", "C3", 1),
        ),
        (
            GroupSpec(("C1", "C2", "C3"), ("C3", "W")),
            GroupSpec(("S", "C1"), ("S", "S")),
            GroupSpec(("C3", "J", "A1", "B1"), ("C3", "E")),
            GroupSpec(("A1", "A2"), ("A1", "N")),
            GroupSpec(("B1", "B2", "B3", "B4"), ("B1", "N")),
        ),
        (
            ("SC1", "S", "C1"),
            ("C1C2", "C1", "C2"),
            ("C2C3", "C2", "C3"),
            ("C3J", "C3", "J"),
            ("JA1", "J", "A1"),
            ("A1A2", "A1", "A2"),
            ("JB1", "J", "B1"),
            ("B1B2", "B1", "B2"),
            ("B2B3", "B2", "B3"),
            ("B3B4", "B3", "B4"),
        ),
    ),
    LevelSpec(
        "Different Heights",
        ("AB..", ".DC.", ".EGJ", "..HI"),
        (
            ("A", "A", 5),
            ("B", "B", 5),
            ("D", "D", 1),
            ("C", "C", 3),
            ("G", "G", 0),
            ("E", "E", 0),
            ("H", "H", 0),
            ("I", "I", 0),
            ("J", "J", 0),
        ),
        (
            GroupSpec(("A", "B", "D"), ("A", "S")),
            GroupSpec(("D", "C"), ("D", "S")),
            GroupSpec(("G", "E"), ("E", "S")),
            GroupSpec(("G", "H", "I", "J"), ("J", "N")),
        ),
        (
            ("AB", "A", "B"),
            ("BD", "B", "D"),
            ("DC", "D", "C"),
            ("CG", "C", "G"),
            ("GC", "G", "C"),
            ("GE", "G", "E"),
            ("GH", "G", "H"),
            ("HI", "H", "I"),
            ("IJ", "I", "J"),
        ),
    ),
    LevelSpec(
        "Pinned Exchange",
        ("12B.", "43DC", "5FEX", "..G."),
        (
            ("1", "a1", 5),
            ("2", "a2", 4),
            ("3", "a3", 3),
            ("4", "a4", 2),
            ("5", "a5", 1),
            ("B", "b1", 4),
            ("D", "b3", 0),
            ("C", "b2", 0),
            ("X", "b4", 0),
            ("E", "b5", 0),
            ("F", "b6", 0),
            ("G", "b7", 0),
        ),
        (
            GroupSpec(("a1", "a2", "a3"), ("a3", "S")),
            GroupSpec(("a3", "a4", "a5"), ("a5", "E")),
            GroupSpec(("b1", "b3", "b2"), ("b1", "E")),
            GroupSpec(("b2", "b4", "b5"), ("b4", "S")),
            GroupSpec(("b5", "b6", "b7"), ("b7", "E")),
        ),
        (
            ("a1a2", "a1", "a2"),
            ("a2a3", "a2", "a3"),
            ("a4a3", "a4", "a3"),
            ("a3b3", "a3", "b3"),
            ("b1b3", "b1", "b3"),
            ("b3b2", "b3", "b2"),
            ("b2b4", "b2", "b4"),
            ("b4b5", "b4", "b5"),
            ("b5b6", "b5", "b6"),
            ("b5b7", "b5", "b7"),
        ),
    ),
)


def _positions(spec: LevelSpec) -> dict[str, tuple[int, int]]:
    symbol_to_id: dict[str, str] = {}
    for symbol, reservoir_id, _count in spec.symbols:
        symbol_to_id.setdefault(symbol, reservoir_id)
    # Levels with repeated digit symbols provide the ambiguous B reservoirs below.
    if spec.name == "Balanced Conduit":
        symbol_to_id.update({"1": "C1", "2": "C2", "3": "C3"})
    out: dict[str, tuple[int, int]] = {}
    for row, line in enumerate(spec.rows):
        for col, symbol in enumerate(line):
            if symbol == ".":
                continue
            reservoir_id = symbol_to_id[symbol]
            if spec.name == "Balanced Conduit":
                overrides = {(2, 2): "B1", (3, 2): "B4", (2, 3): "B2", (3, 3): "B3"}
                reservoir_id = overrides.get((col, row), reservoir_id)
            out[reservoir_id] = (col, row)
    return out


def _initial_counts(spec: LevelSpec) -> dict[str, int]:
    counts = {reservoir_id: count for _symbol, reservoir_id, count in spec.symbols}
    if spec.name == "Balanced Conduit":
        counts.update({"B1": 0, "B2": 0, "B3": 0, "B4": 0})
    return counts


def _center(cell: tuple[int, int]) -> tuple[int, int]:
    return 8 + 16 * cell[0], 8 + 16 * cell[1]


def _link_centers(spec: LevelSpec, positions: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    directions = {(source, destination) for _lid, source, destination in spec.links}
    centers: dict[str, tuple[int, int]] = {}
    for link_id, source, destination in spec.links:
        sx, sy = _center(positions[source])
        dx, dy = _center(positions[destination])
        cx, cy = (sx + dx) // 2, (sy + dy) // 2
        if (destination, source) in directions:
            if sy == dy:
                cy += -4 if dx > sx else 4
            else:
                cx += 4 if dy > sy else -4
        centers[link_id] = (cx, cy)
    return centers


def _assert_levels() -> None:
    assert len(LEVEL_SPECS) == 7
    for spec in LEVEL_SPECS:
        assert len(spec.rows) == 4
        assert all(len(row) == 4 for row in spec.rows)
        positions = _positions(spec)
        counts = _initial_counts(spec)
        assert len(positions) == len(set(positions.values())) == len(counts)
        assert all(value >= 0 for value in counts.values())
        memberships = {reservoir_id: 0 for reservoir_id in positions}
        for group in spec.groups:
            members = set(group.members)
            assert members <= positions.keys()
            reached = {next(iter(members))}
            while True:
                expanded = reached | {
                    other
                    for member in reached
                    for other in members
                    if abs(positions[member][0] - positions[other][0]) + abs(positions[member][1] - positions[other][1])
                    == 1
                }
                if expanded == reached:
                    break
                reached = expanded
            assert reached == members
            for reservoir_id in members:
                memberships[reservoir_id] += 1
        assert all(1 <= value <= 2 for value in memberships.values())
        for _link_id, source, destination in spec.links:
            assert sum(abs(a - b) for a, b in zip(positions[source], positions[destination], strict=True)) == 1
        centers = _link_centers(spec, positions)
        boxes = [(x - 2, y - 2, x + 2, y + 2) for x, y in centers.values()]
        assert len(boxes) == len(set(boxes))
        badge_boxes = []
        for group in spec.groups:
            reservoir_id, side = group.badge
            x, y = _center(positions[reservoir_id])
            dx, dy = {"N": (0, -8), "S": (0, 8), "W": (-8, 0), "E": (8, 0)}[side]
            badge_boxes.append((x + dx - 2, y + dy - 2, x + dx + 2, y + dy + 2))
        for box in badge_boxes:
            assert all(
                box[2] < other[0] or other[2] < box[0] or box[3] < other[1] or other[3] < box[1] for other in boxes
            )


_assert_levels()


class ReservoirView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: BalancedReservoirs | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = BACKGROUND
        counts = game.counts
        moving = None
        phase = None
        if game.visual is not None:
            phase = game.visual["phase"]
            counts = game.visual.get("counts", counts)
            moving = game.visual.get("moving")
        self._draw_group_contours(frame, game, phase)
        self._draw_links(frame, game, phase)
        self._draw_reservoirs(frame, game, counts)
        self._draw_badges(frame, game, counts, phase)
        if moving is not None:
            x, y, color = moving
            frame[y, x] = color
            if phase == "transfer_plate":
                sx, sy = game.visual["trail"]
                frame[sy, sx] = 10
        if game.spec.reference_panel:
            self._draw_reference(frame)
        self._draw_budget(frame, game)
        return frame

    def _draw_group_contours(self, frame: np.ndarray, game: BalancedReservoirs, phase: str | None) -> None:
        for group_idx, side, col, row, depth in game.contour_segments:
            color = 0 if phase == "completion" else GROUP_COLORS[group_idx]
            if side == "N":
                y = row * 16 + depth
                frame[y, col * 16 : min(64, (col + 1) * 16)] = color
            elif side == "S":
                y = (row + 1) * 16 - 1 - depth
                frame[y, col * 16 : min(64, (col + 1) * 16)] = color
            elif side == "W":
                x = col * 16 + depth
                frame[row * 16 : min(64, (row + 1) * 16), x] = color
            else:
                x = (col + 1) * 16 - 1 - depth
                frame[row * 16 : min(64, (row + 1) * 16), x] = color

    def _draw_links(self, frame: np.ndarray, game: BalancedReservoirs, phase: str | None) -> None:
        for link_id, source, destination in game.spec.links:
            cx, cy = game.link_centers[link_id]
            plate_color = 3
            arrow_color = 0
            shift_x = shift_y = 0
            blocked = False
            if game.visual and game.visual.get("link") == link_id:
                if phase in ("blocked_flash", "blocked_jolt"):
                    blocked = True
                    arrow_color = 8
                elif phase == "transfer_plate":
                    plate_color = 7
                if phase == "blocked_jolt":
                    sx, sy = game.positions[source]
                    dx, dy = game.positions[destination]
                    shift_x, shift_y = sx - dx, sy - dy
            frame[cy - 2 : cy + 3, cx - 2 : cx + 3] = plate_color
            if blocked:
                frame[cy - 2, cx - 2 : cx + 3] = 8
                frame[cy + 2, cx - 2 : cx + 3] = 8
                frame[cy - 2 : cy + 3, cx - 2] = 8
                frame[cy - 2 : cy + 3, cx + 2] = 8
            sx, sy = game.positions[source]
            dx, dy = game.positions[destination]
            vx, vy = dx - sx, dy - sy
            ax, ay = cx + shift_x, cy + shift_y
            for offset in (-1, 0, 1):
                px, py = ax + vx * offset, ay + vy * offset
                if cx - 2 <= px <= cx + 2 and cy - 2 <= py <= cy + 2:
                    frame[py, px] = arrow_color
            tip_x, tip_y = ax + vx, ay + vy
            for side in (-1, 1):
                px, py = tip_x + (side if vy else 0), tip_y + (side if vx else 0)
                if cx - 2 <= px <= cx + 2 and cy - 2 <= py <= cy + 2:
                    frame[py, px] = arrow_color

    def _draw_reservoirs(self, frame: np.ndarray, game: BalancedReservoirs, counts: dict[str, int]) -> None:
        memberships: dict[str, list[int]] = {reservoir_id: [] for reservoir_id in game.positions}
        for group_idx, group in enumerate(game.spec.groups):
            for reservoir_id in group.members:
                memberships[reservoir_id].append(group_idx)
        for reservoir_id, cell in game.positions.items():
            cx, cy = _center(cell)
            frame[cy - 5 : cy + 6, cx - 5 : cx + 6] = 1
            frame[cy - 4 : cy + 5, cx - 4 : cx + 5] = 4
            for index in range(counts[reservoir_id]):
                x = cx - 4 + 2 * (index % 5)
                y = cy + 4 - 2 * (index // 5)
                frame[y, x] = 11
            for tab_idx, group_idx in enumerate(sorted(memberships[reservoir_id])):
                color = GROUP_COLORS[group_idx]
                if tab_idx == 0:
                    frame[cy - 6 : cy - 4, cx - 1 : cx + 1] = color
                else:
                    frame[cy - 1 : cy + 1, cx + 5 : cx + 7] = color

    def _draw_badges(
        self, frame: np.ndarray, game: BalancedReservoirs, counts: dict[str, int], phase: str | None
    ) -> None:
        for group_idx, group in enumerate(game.spec.groups):
            reservoir_id, side = group.badge
            cx, cy = _center(game.positions[reservoir_id])
            dx, dy = {"N": (0, -8), "S": (0, 8), "W": (-8, 0), "E": (8, 0)}[side]
            x, y = cx + dx, cy + dy
            frame[y - 2 : y + 3, x - 2 : x + 3] = GROUP_COLORS[group_idx]
            frame[y - 1 : y + 2, x - 1 : x + 2] = 4
            values = [counts[member] for member in group.members]
            changed = game.visual is not None and group_idx in game.visual.get("changed_groups", ())
            if phase == "transfer_destination" and changed:
                frame[y, x - 1 : x + 2] = 0
            elif all(value == values[0] for value in values):
                frame[y, x - 1 : x + 2] = 14
                frame[y + 1, x] = 14
            else:
                frame[y - 1, x - 1] = 8
                frame[y + 1, x + 1] = 8

    def _draw_reference(self, frame: np.ndarray) -> None:
        frame[1:16, 36:64] = 2
        frame[2:15, 37:63] = 4
        frame[4:12, 39] = 9
        frame[4:12, 58] = 9
        frame[4, 39:59] = 9
        frame[11, 39:59] = 9
        for cx in (44, 53):
            frame[6:11, cx - 2 : cx + 3] = 1
            frame[7:10, cx - 1 : cx + 2] = 4
            frame[9, cx] = 11
        frame[6:11, 59:64] = 9
        frame[7:10, 60:63] = 4
        frame[8, 60:63] = 14
        frame[9, 61] = 14

    def _draw_budget(self, frame: np.ndarray, game: BalancedReservoirs) -> None:
        width = round(VIEW_SIZE * game.remaining_steps / game.step_budget)
        frame[63, :] = 8
        if width:
            frame[63, :width] = 11


class BalancedReservoirs(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.view = ReservoirView()
        levels = [
            Level(sprites=[], grid_size=(64, 64), data={"spec": spec, "budget": BASELINES[idx] * 6}, name=spec.name)
            for idx, spec in enumerate(LEVEL_SPECS)
        ]
        self.view.game = self
        super().__init__(
            "balanced_reservoirs-0001",
            levels,
            Camera(0, 0, 64, 64, BACKGROUND, BACKGROUND, [self.view]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.positions = _positions(self.spec)
        self.counts = _initial_counts(self.spec)
        self.total_charge = sum(self.counts.values())
        self.link_centers = _link_centers(self.spec, self.positions)
        self.links_by_id = {link_id: (source, destination) for link_id, source, destination in self.spec.links}
        self.contour_segments = self._build_contour_segments()
        self.step_budget = int(level.get_data("budget"))
        self.remaining_steps = self.step_budget
        self.undo_history: list[dict[str, int]] = []
        self.visual: dict[str, object] | None = None
        self.visual_queue: list[dict[str, object]] = []
        self.pending_result: str | None = None
        self.last_event: str | None = None

    def _build_contour_segments(self) -> list[tuple[int, str, int, int, int]]:
        geometric: dict[tuple[str, int, int], list[tuple[int, str, int, int]]] = {}
        for group_idx, group in enumerate(self.spec.groups):
            cells = {self.positions[member] for member in group.members}
            for col, row in cells:
                for side, dc, dr in (("N", 0, -1), ("S", 0, 1), ("W", -1, 0), ("E", 1, 0)):
                    if (col + dc, row + dr) in cells:
                        continue
                    if side in ("N", "S"):
                        edge = row * 16 if side == "N" else (row + 1) * 16 - 1
                        key = ("H", edge, col)
                    else:
                        edge = col * 16 if side == "W" else (col + 1) * 16 - 1
                        key = ("V", edge, row)
                    geometric.setdefault(key, []).append((group_idx, side, col, row))
        segments: list[tuple[int, str, int, int, int]] = []
        for owners in geometric.values():
            owners.sort()
            segments.extend((*owner, depth) for depth, owner in enumerate(owners))
        return segments

    def step(self) -> None:
        if self.visual_queue:
            self.visual = self.visual_queue.pop(0)
            if not self.visual_queue:
                result = self.pending_result
                self.pending_result = None
                if result == "win":
                    self.next_level()
                elif result == "lose":
                    self.lose()
                self.complete_action()
            return

        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id == int(GameAction.RESET.value):
            self.visual = None
            self.complete_action()
            return
        if action_id == ACTION_UNDO:
            if self.undo_history:
                self.counts = self.undo_history.pop()
            self.last_event = None
            self.visual = None
            self.complete_action()
            return
        if action_id != ACTION_CLICK:
            self.visual = None
            self.complete_action()
            return

        link_id = self._clicked_link()
        if link_id is None:
            self.last_event = "irrelevant_click"
            self.visual = None
            self.complete_action()
            return

        self.remaining_steps = max(0, self.remaining_steps - 1)
        source, destination = self.links_by_id[link_id]
        if self.counts[source] == 0:
            self.last_event = "blocked"
            base = dict(self.counts)
            self.visual = {"phase": "blocked_flash", "counts": base, "link": link_id}
            self.visual_queue = [
                {"phase": "blocked_jolt", "counts": base, "link": link_id},
                {"phase": "settled", "counts": base, "link": link_id},
            ]
            if self.remaining_steps == 0:
                self.pending_result = "lose"
            return

        old_counts = dict(self.counts)
        old_balanced = self._balanced_vector(old_counts)
        self.undo_history.append(old_counts)
        self.counts[source] -= 1
        self.counts[destination] += 1
        assert sum(self.counts.values()) == self.total_charge
        new_balanced = self._balanced_vector(self.counts)
        changed = tuple(
            idx for idx, pair in enumerate(zip(old_balanced, new_balanced, strict=True)) if pair[0] != pair[1]
        )
        self.last_event = "transfer"
        source_center = _center(self.positions[source])
        destination_center = _center(self.positions[destination])
        plate_center = self.link_centers[link_id]
        vx = destination_center[0] - source_center[0]
        vy = destination_center[1] - source_center[1]
        rim_source = (
            source_center[0] + (5 if vx > 0 else -5 if vx < 0 else 0),
            source_center[1] + (5 if vy > 0 else -5 if vy < 0 else 0),
            11,
        )
        rim_destination = (
            destination_center[0] + (-5 if vx > 0 else 5 if vx < 0 else 0),
            destination_center[1] + (-5 if vy > 0 else 5 if vy < 0 else 0),
            11,
        )
        stationary = dict(old_counts)
        stationary[source] -= 1
        self.visual = {"phase": "transfer_pre", "counts": old_counts, "link": link_id}
        self.visual_queue = [
            {"phase": "transfer_source", "counts": stationary, "link": link_id, "moving": rim_source},
            {
                "phase": "transfer_plate",
                "counts": stationary,
                "link": link_id,
                "moving": (*plate_center, 11),
                "trail": (
                    plate_center[0] - (1 if vx > 0 else -1 if vx < 0 else 0),
                    plate_center[1] - (1 if vy > 0 else -1 if vy < 0 else 0),
                ),
            },
            {
                "phase": "transfer_destination",
                "counts": stationary,
                "link": link_id,
                "moving": rim_destination,
                "changed_groups": changed,
            },
            {"phase": "transfer_settled", "counts": dict(self.counts), "link": link_id, "changed_groups": changed},
        ]
        if all(new_balanced):
            self.visual_queue.extend(
                {"phase": "completion", "counts": dict(self.counts), "link": link_id} for _ in range(4)
            )
            self.pending_result = "win"
        elif self.remaining_steps == 0:
            self.pending_result = "lose"

    def _balanced_vector(self, counts: dict[str, int]) -> tuple[bool, ...]:
        return tuple(
            all(counts[member] == counts[group.members[0]] for member in group.members) for group in self.spec.groups
        )

    def _clicked_link(self) -> str | None:
        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        for link_id, (cx, cy) in self.link_centers.items():
            if cx - 2 <= x <= cx + 2 and cy - 2 <= y <= cy + 2:
                return link_id
        return None

    def _get_hidden_state(self) -> np.ndarray:
        return np.asarray([self.counts[key] for key in sorted(self.counts)] + [self.remaining_steps], dtype=np.int16)
