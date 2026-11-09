from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
BACKGROUND = 5
SOURCE_CENTER = (3, 32)
EXIT_CENTER = (60, 32)
EXIT_HITBOX = (57, 27, 63, 37)
LOGICAL_COLORS = (6, 9, 11, 14)
ALL_ACTIONS = (1, 2, 3, 4, 5, 6, 7)


@dataclass(frozen=True)
class RotorState:
    color: int
    target: str


@dataclass(frozen=True)
class RotorSpec:
    name: str
    center: tuple[int, int]
    states: tuple[RotorState, ...]


@dataclass(frozen=True)
class LevelSpec:
    title: str
    rows: tuple[str, ...]
    source_edges: tuple[tuple[int, str], ...]
    exit_demand: int
    rotors: tuple[RotorSpec, ...]
    baseline_actions: int


def _s(color: int, target: str) -> RotorState:
    return RotorState(color, target)


LEVEL_SPECS = (
    LevelSpec("Live Latch", ("S-----------X",), ((11, "X"),), 11, (), 1),
    LevelSpec(
        "One Turn", ("S-----A-----X",), ((9, "A"),), 11, (RotorSpec("A", (32, 32), (_s(9, "X"), _s(11, "X"))),), 2
    ),
    LevelSpec(
        "Two Handoffs",
        ("      B      ", "S-------A---X", "      C      "),
        ((9, "A"), (9, "B"), (9, "C")),
        6,
        (
            RotorSpec("A", (40, 32), (_s(9, "X"), _s(11, "X"), _s(14, "X"), _s(6, "X"))),
            RotorSpec("B", (22, 16), (_s(9, "A"), _s(11, "A"))),
            RotorSpec("C", (22, 48), (_s(9, "A"), _s(14, "A"))),
        ),
        6,
    ),
    LevelSpec(
        "Borrowed Return",
        ("    B        ", "S-A----D--E-X", "    C        "),
        ((14, "A"),),
        11,
        (
            RotorSpec("A", (11, 32), (_s(14, "D"), _s(11, "D"), _s(11, "C"), _s(14, "B"))),
            RotorSpec("B", (23, 16), (_s(14, "A"), _s(11, "A"), _s(9, "D"), _s(9, "E"))),
            RotorSpec("C", (23, 48), (_s(11, "D"), _s(14, "E"), _s(11, "A"), _s(11, "E"))),
            RotorSpec("D", (36, 32), (_s(14, "B"), _s(14, "C"), _s(11, "A"), _s(9, "E"))),
            RotorSpec("E", (50, 32), (_s(11, "D"), _s(9, "C"), _s(11, "B"), _s(11, "X"))),
        ),
        28,
    ),
    LevelSpec(
        "Leave by Another Side",
        ("    B         ", "S-A----D--E-FX", "    C         "),
        ((14, "A"),),
        11,
        (
            RotorSpec("A", (11, 32), (_s(9, "C"), _s(14, "B"), _s(14, "C"), _s(14, "D"))),
            RotorSpec("B", (22, 15), (_s(11, "A"), _s(9, "A"), _s(14, "A"), _s(11, "D"))),
            RotorSpec("C", (22, 49), (_s(9, "A"), _s(9, "E"), _s(14, "E"), _s(9, "D"))),
            RotorSpec("D", (34, 32), (_s(11, "E"), _s(14, "C"), _s(9, "C"), _s(9, "F"))),
            RotorSpec("E", (45, 32), (_s(11, "B"), _s(11, "F"), _s(14, "B"), _s(9, "B"))),
            RotorSpec("F", (52, 32), (_s(9, "D"), _s(11, "D"), _s(9, "E"), _s(11, "X"))),
        ),
        42,
    ),
    LevelSpec(
        "False Finish",
        ("   B-----E     ", "S-A----D----G-X", "   C-----F     "),
        ((9, "A"),),
        14,
        (
            RotorSpec("A", (11, 32), (_s(9, "C"), _s(6, "B"), _s(9, "D"), _s(11, "C"))),
            RotorSpec("B", (21, 15), (_s(6, "E"), _s(6, "A"), _s(11, "D"), _s(6, "D"))),
            RotorSpec("C", (21, 49), (_s(9, "E"), _s(11, "E"), _s(11, "A"), _s(14, "E"))),
            RotorSpec("D", (32, 32), (_s(6, "E"), _s(11, "A"), _s(11, "E"), _s(11, "F"))),
            RotorSpec("E", (42, 16), (_s(11, "B"), _s(14, "D"), _s(6, "G"), _s(6, "F"))),
            RotorSpec("F", (42, 48), (_s(6, "E"), _s(6, "D"), _s(11, "D"), _s(9, "G"))),
            RotorSpec("G", (52, 32), (_s(9, "E"), _s(6, "E"), _s(6, "F"), _s(14, "X"))),
        ),
        46,
    ),
    LevelSpec(
        "Full Lockstep",
        ("   B---C       ", "S-A--F---G-H--X", "   D---E       "),
        ((11, "A"),),
        14,
        (
            RotorSpec("A", (11, 32), (_s(11, "F"), _s(9, "B"), _s(9, "F"), _s(9, "D"))),
            RotorSpec("B", (21, 15), (_s(9, "C"), _s(9, "A"), _s(6, "F"), _s(11, "F"))),
            RotorSpec("C", (35, 15), (_s(9, "B"), _s(9, "G"), _s(9, "F"), _s(11, "F"))),
            RotorSpec("D", (21, 49), (_s(9, "F"), _s(6, "A"), _s(6, "E"), _s(9, "A"))),
            RotorSpec("E", (35, 49), (_s(6, "G"), _s(11, "G"), _s(6, "D"), _s(9, "D"))),
            RotorSpec("F", (28, 32), (_s(11, "E"), _s(9, "E"), _s(6, "E"), _s(6, "D"))),
            RotorSpec("G", (44, 32), (_s(11, "E"), _s(11, "C"), _s(6, "H"), _s(9, "H"))),
            RotorSpec("H", (52, 32), (_s(6, "G"), _s(9, "G"), _s(14, "X"), _s(11, "G"))),
        ),
        56,
    ),
)


def _validate_ascii(spec: LevelSpec) -> None:
    symbols = {char for row in spec.rows for char in row if char.isalpha()}
    expected = {"S", "X", *(rotor.name for rotor in spec.rotors)}
    if symbols != expected:
        raise ValueError(f"ASCII layout mismatch for {spec.title}: {symbols} != {expected}")


def _build_level(spec: LevelSpec) -> Level:
    _validate_ascii(spec)
    return Level(
        sprites=[],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.title,
        data={"spec": spec, "step_budget": spec.baseline_actions * 6},
    )


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    points: list[tuple[int, int]] = []
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return points
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _paint(frame: np.ndarray, x: int, y: int, color: int, radius: int = 0) -> None:
    x0, x1 = max(0, x - radius), min(VIEW_SIZE, x + radius + 1)
    y0, y1 = max(0, y - radius), min(VIEW_SIZE, y + radius + 1)
    frame[y0:y1, x0:x1] = color


class LockstepDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: LockstepDoors | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        frame[:, :] = BACKGROUND
        if game is None:
            return frame

        spec, indices, powered, ready, glowing = game._visual_state()
        self._draw_budget(frame, game)
        self._draw_all_channels(frame, game, spec, indices, glowing)
        self._draw_source(frame, game)
        self._draw_exit(frame, game, spec, glowing)
        for rotor in spec.rotors:
            self._draw_rotor(frame, game, rotor, indices, powered, ready)
        return frame

    def _draw_budget(self, frame: np.ndarray, game: LockstepDoors) -> None:
        frame[1:3, 2:55] = 4
        width = round(53 * game.remaining_steps / max(1, game.step_budget))
        if width:
            frame[1:3, 2 : 2 + width] = 0
        for index in range(len(LEVEL_SPECS)):
            frame[1:3, 57 + index % 7] = 0 if index <= game.level_index else 3

    def _route_end(self, spec: LevelSpec, target: str) -> tuple[int, int]:
        if target == "X":
            return (57, 32)
        return next(rotor.center for rotor in spec.rotors if rotor.name == target)

    def _route_points(
        self, start: tuple[int, int], end: tuple[int, int], lane: int, source_name: str, target: str
    ) -> list[tuple[int, int]]:
        del source_name
        dx, dy = end[0] - start[0], end[1] - start[1]
        offset = (-2, -1, 1, 2)[lane % 4]
        if abs(dx) >= abs(dy):
            shifted_start = (start[0], start[1] + offset)
            shifted_end = (end[0], end[1] + offset)
        else:
            shifted_start = (start[0] + offset, start[1])
            shifted_end = (end[0] + offset, end[1])
        points = _line_points(shifted_start, shifted_end)
        trim = 1 if target == "X" else 4
        return points[:-trim] if len(points) > trim + 3 else points

    def _draw_crossing_bridges(
        self,
        frame: np.ndarray,
        routes: list[tuple[tuple[int, int], tuple[int, int], int, int, str, str]],
        indices: dict[str, int],
    ) -> None:
        occupied: list[tuple[int, int]] = []
        route_points = [
            self._route_points(start, end, lane, source, target) for start, end, _, lane, source, target in routes
        ]
        for first_index, first in enumerate(routes):
            start_a, end_a, color_a, lane_a, source_a, target_a = first
            selected_a = source_a == "S" or indices.get(source_a) == lane_a
            for second_index in range(first_index + 1, len(routes)):
                start_b, end_b, _, _, source_b, target_b = routes[second_index]
                if source_a == source_b or target_a == target_b:
                    continue
                cross = (end_a[0] - start_a[0]) * (end_b[1] - start_b[1]) - (end_a[1] - start_a[1]) * (
                    end_b[0] - start_b[0]
                )
                if abs(cross) < 20:
                    continue
                common = set(route_points[first_index]).intersection(route_points[second_index])
                candidates = [
                    point
                    for point in common
                    if all(
                        abs(point[0] - end[0]) + abs(point[1] - end[1]) > 5 for end in (start_a, end_a, start_b, end_b)
                    )
                ]
                if not candidates:
                    continue
                x, y = candidates[len(candidates) // 2]
                if any(abs(x - ox) <= 3 and abs(y - oy) <= 3 for ox, oy in occupied):
                    continue
                occupied.append((x, y))
                bridge_color = color_a if selected_a else 4
                frame[max(0, y - 1) : min(VIEW_SIZE, y + 2), max(0, x - 1) : min(VIEW_SIZE, x + 2)] = 2
                if abs(end_a[0] - start_a[0]) >= abs(end_a[1] - start_a[1]):
                    frame[y, max(0, x - 1) : min(VIEW_SIZE, x + 2)] = bridge_color
                else:
                    frame[max(0, y - 1) : min(VIEW_SIZE, y + 2), x] = bridge_color

    def _draw_route(
        self, frame: np.ndarray, points: list[tuple[int, int]], color: int, *, powered: bool, selected: bool
    ) -> None:
        for x, y in points:
            _paint(frame, x, y, color if powered else 4, radius=1)
        if selected and not powered:
            for x, y in points:
                _paint(frame, x, y, color)
        if len(points) > 3:
            tx, ty = points[-3]
            ex, ey = points[-1]
            vx = 0 if ex == tx else (1 if ex > tx else -1)
            vy = 0 if ey == ty else (1 if ey > ty else -1)
            arrow_color = color if selected else 4
            for side in (-1, 1):
                _paint(frame, ex - 2 * vx + side * vy, ey - 2 * vy - side * vx, arrow_color)

    def _draw_all_channels(
        self,
        frame: np.ndarray,
        game: LockstepDoors,
        spec: LevelSpec,
        indices: dict[str, int],
        glowing: set[tuple[str, int]],
    ) -> None:
        routes: list[tuple[tuple[int, int], tuple[int, int], int, int, str, str]] = []
        for lane, (color, target) in enumerate(spec.source_edges):
            routes.append((SOURCE_CENTER, self._route_end(spec, target), color, lane, "S", target))
        for rotor in spec.rotors:
            for lane, state in enumerate(rotor.states):
                routes.append(
                    (rotor.center, self._route_end(spec, state.target), state.color, lane, rotor.name, state.target)
                )

        for start, end, color, lane, source, target in routes:
            self._draw_route(
                frame, self._route_points(start, end, lane, source, target), color, powered=False, selected=False
            )

        collapse = game._animation_kind == "rotate" and game._animation_phase in (1, 2, 3)
        for lane, (color, target) in enumerate(spec.source_edges):
            points = self._route_points(SOURCE_CENTER, self._route_end(spec, target), lane, "S", target)
            self._draw_route(frame, points, color, powered=not collapse, selected=True)
        for rotor in spec.rotors:
            state_index = indices[rotor.name]
            state = rotor.states[state_index]
            points = self._route_points(
                rotor.center, self._route_end(spec, state.target), state_index, rotor.name, state.target
            )
            edge_key = (rotor.name, state_index)
            selected = not (
                game._animation_kind == "rotate" and game._animation_phase == 2 and rotor.name == game._clicked_name
            )
            is_powered = edge_key in glowing and not collapse
            if selected:
                self._draw_route(frame, points, state.color, powered=is_powered, selected=True)
        self._draw_crossing_bridges(frame, routes, indices)

    def _draw_source(self, frame: np.ndarray, game: LockstepDoors) -> None:
        core = 8 if game.terminal_failure else 0
        frame[28:37, 0:7] = 2
        frame[30:35, 1:6] = 0
        frame[31:34, 2:5] = core

    def _draw_exit(
        self, frame: np.ndarray, game: LockstepDoors, spec: LevelSpec, glowing: set[tuple[str, int]]
    ) -> None:
        ready = game._exit_ready_for(spec, glowing)
        door = np.full((11, 7), 3, dtype=np.int8)
        if ready:
            door[2:9, 0] = 0
        demand = spec.exit_demand
        door[4:7, 2:5] = demand
        opening = game.door_open
        if opening:
            gap = 1 + (1 if game._animation_phase >= 1 else 0)
            door[2:9, 3 - gap : 4 + gap] = 0
            door[4:7, 2:5] = 0
        if game.rejection == "X":
            door[0, :] = 8
            door[-1, :] = 8
            door[:, 0] = 8
        shift = game._shake_offset() if game._animation_kind == "reject" and game.rejection == "X" else 0
        x0 = 57 + shift
        destination_start = max(0, x0)
        destination_end = min(VIEW_SIZE, x0 + 7)
        source_start = destination_start - x0
        source_end = source_start + destination_end - destination_start
        frame[27:38, destination_start:destination_end] = door[:, source_start:source_end]

    def _draw_rotor(
        self,
        frame: np.ndarray,
        game: LockstepDoors,
        rotor: RotorSpec,
        indices: dict[str, int],
        powered: set[str],
        ready: set[str],
    ) -> None:
        cx, cy = rotor.center
        if game._animation_kind == "reject" and game.rejection == rotor.name:
            cx += game._shake_offset()
        shell = 1 if rotor.name in powered and not game.terminal_failure else 3
        if game.rejection == rotor.name:
            shell = 8
        frame[cy - 2 : cy + 3, cx - 2 : cx + 3] = shell
        petal_positions = ((0, -3), (3, 0), (0, 3), (-3, 0)) if len(rotor.states) != 2 else ((0, -3), (0, 3))
        for idx, state in enumerate(rotor.states):
            px, py = petal_positions[idx]
            _paint(frame, cx + px, cy + py, state.color)
        selected = indices[rotor.name]
        face = rotor.states[selected].color
        frame[cy - 1 : cy + 2, cx - 1 : cx + 2] = face
        px, py = petal_positions[selected]
        _paint(frame, cx + (0 if px == 0 else px // 2), cy + (0 if py == 0 else py // 2), face)
        if rotor.name in ready and not game.terminal_failure:
            frame[cy - 3, cx - 3 : cx + 4] = 0
            frame[cy + 3, cx - 3 : cx + 4] = 0
            frame[cy - 3 : cy + 4, cx - 3] = 0
            frame[cy - 3 : cy + 4, cx + 3] = 0
            for dx, dy in ((-2, -2), (2, -2), (-2, 2), (2, 2)):
                _paint(frame, cx + dx, cy + dy, 0)
            frame[cy - 1 : cy + 2, cx - 3] = 0


class LockstepDoors(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.display = LockstepDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "lockstep_doors-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BACKGROUND, BACKGROUND, [self.display]),
            False,
            len(levels),
            list(ALL_ACTIONS),
            seed,
        )
        self.display.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.indices = {rotor.name: 0 for rotor in self.spec.rotors}
        self.step_budget = int(level.get_data("step_budget"))
        self.remaining_steps = self.step_budget
        self.door_open = False
        self.terminal_failure = False
        self.rejection: str | None = None
        self.undo_history: list[tuple[dict[str, int], bool]] = []
        self._animation_kind: str | None = None
        self._animation_phase = 0
        self._animation_queue: deque[int] = deque()
        self._animation_before: tuple[dict[str, int], set[str], set[str], set[tuple[str, int]]] | None = None
        self._clicked_name: str | None = None
        self._pending_exit = False
        self._pending_loss = False
        self.powered, self.ready, self.glowing = self._compute_power(self.indices)

    def _compute_power(self, indices: dict[str, int]) -> tuple[set[str], set[str], set[tuple[str, int]]]:
        powered: set[str] = set()
        incoming: dict[str, set[int]] = {rotor.name: set() for rotor in self.spec.rotors}
        incoming["X"] = set()
        queue: deque[str] = deque()
        for color, target in self.spec.source_edges:
            incoming[target].add(color)
            if target != "X" and target not in powered:
                powered.add(target)
                queue.append(target)
        glowing: set[tuple[str, int]] = set()
        rotor_map = {rotor.name: rotor for rotor in self.spec.rotors}
        while queue:
            name = queue.popleft()
            state_index = indices[name]
            state = rotor_map[name].states[state_index]
            glowing.add((name, state_index))
            incoming[state.target].add(state.color)
            if state.target != "X" and state.target not in powered:
                powered.add(state.target)
                queue.append(state.target)
        ready = {
            rotor.name
            for rotor in self.spec.rotors
            if rotor.name in powered and rotor.states[indices[rotor.name]].color in incoming[rotor.name]
        }
        return powered, ready, glowing

    def _exit_ready_for(self, spec: LevelSpec, glowing: set[tuple[str, int]]) -> bool:
        if any(color == spec.exit_demand and target == "X" for color, target in spec.source_edges):
            return True
        rotor_map = {rotor.name: rotor for rotor in spec.rotors}
        return any(
            rotor_map[name].states[index].target == "X" and rotor_map[name].states[index].color == spec.exit_demand
            for name, index in glowing
        )

    @property
    def exit_ready(self) -> bool:
        return self._exit_ready_for(self.spec, self.glowing)

    def _visual_state(self) -> tuple[LevelSpec, dict[str, int], set[str], set[str], set[tuple[str, int]]]:
        if self._animation_kind == "rotate" and self._animation_before is not None and self._animation_phase <= 1:
            indices, powered, ready, glowing = self._animation_before
            return self.spec, indices, powered, ready, glowing
        if self._animation_kind == "rotate" and self._animation_phase == 4:
            return self.spec, self.indices, self.powered, set(), self.glowing
        return self.spec, self.indices, self.powered, self.ready, self.glowing

    def _shake_offset(self) -> int:
        return (1, -1, 1, 0)[min(self._animation_phase, 3)]

    def _spend_step(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.remaining_steps == 0:
            self.terminal_failure = True
            self._pending_loss = True

    def _clicked_object(self) -> str | None:
        x = int(self.action.data.get("x", -1))
        y = int(self.action.data.get("y", -1))
        grid = self.camera.display_to_grid(x, y)
        if grid is None:
            return None
        gx, gy = grid
        if EXIT_HITBOX[0] <= gx <= EXIT_HITBOX[2] and EXIT_HITBOX[1] <= gy <= EXIT_HITBOX[3]:
            return "X"
        for rotor in self.spec.rotors:
            cx, cy = rotor.center
            if cx - 3 <= gx <= cx + 3 and cy - 3 <= gy <= cy + 3:
                return rotor.name
        return None

    def _begin_animation(self, kind: str, phases: int) -> None:
        self._animation_kind = kind
        self._animation_queue = deque(range(phases))
        self._play_animation_frame()

    def _play_animation_frame(self) -> None:
        self._animation_phase = self._animation_queue.popleft()
        if self._animation_queue:
            return
        kind = self._animation_kind
        self._animation_kind = None
        self._animation_before = None
        if self._pending_loss:
            self.lose()
        elif self._pending_exit and kind == "exit":
            self.next_level()
        self.complete_action()

    def _restore_undo(self) -> None:
        self.rejection = None
        if self.undo_history:
            indices, door_open = self.undo_history.pop()
            self.indices = dict(indices)
            self.door_open = door_open
            self.terminal_failure = False
            self._pending_loss = False
            self.powered, self.ready, self.glowing = self._compute_power(self.indices)

    def step(self) -> None:
        if self._animation_queue:
            self._play_animation_frame()
            return

        action_id = _action_id(self.action.id)
        if action_id == _action_id(GameAction.RESET):
            self.complete_action()
            return

        self._spend_step()
        if action_id == 7:
            self._restore_undo()
            if self.remaining_steps == 0:
                self.terminal_failure = True
                self._pending_loss = True
                self.lose()
            self.complete_action()
            return
        if action_id != 6:
            if self._pending_loss:
                self.lose()
            self.complete_action()
            return

        self.rejection = None
        clicked = self._clicked_object()
        if clicked is None:
            if self._pending_loss:
                self.lose()
            self.complete_action()
            return
        if clicked == "X":
            if not self.exit_ready:
                self.rejection = "X"
                self._begin_animation("reject", 4)
                return
            self.undo_history.append((dict(self.indices), self.door_open))
            self.door_open = True
            self._pending_exit = True
            self._begin_animation("exit", 3)
            return

        if clicked not in self.ready:
            self.rejection = clicked
            self._begin_animation("reject", 4)
            return

        self.undo_history.append((dict(self.indices), self.door_open))
        self._clicked_name = clicked
        self._animation_before = (dict(self.indices), set(self.powered), set(self.ready), set(self.glowing))
        rotor = next(rotor for rotor in self.spec.rotors if rotor.name == clicked)
        self.indices[clicked] = (self.indices[clicked] + 1) % len(rotor.states)
        self.powered, self.ready, self.glowing = self._compute_power(self.indices)
        if not self.exit_ready and not self.ready:
            self.terminal_failure = True
            self._pending_loss = True
        self._begin_animation("rotate", 6)

    def _get_hidden_state(self) -> np.ndarray:
        ordered = [self.indices[rotor.name] for rotor in self.spec.rotors]
        return np.asarray([*ordered, self.remaining_steps, int(self.door_open)], dtype=np.int16)
