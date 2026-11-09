from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
CELL_SIZE = 4
ORIGINS = {"A": (2, 24), "B": (35, 24)}

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

COLOR_FLASH = 0
COLOR_FLOOR = 1
COLOR_FRAME = 2
COLOR_WALL = 3
COLOR_BACKGROUND = 4
COLOR_SHADOW = 5
COLOR_A = 6
COLOR_A_GOAL = 7
COLOR_BLOCKED = 8
COLOR_B = 9
COLOR_B_GOAL = 10

MOTIF_COLORS = {"Diamond": 11, "Ring": 12, "Bars": 14}
MOTIF_ROWS = {"Diamond": 12, "Ring": 16, "Bars": 20}


Cell = tuple[int, int]
Edge = frozenset[Cell]


@dataclass(frozen=True)
class LinkSpec:
    motif: str
    plate_room: str
    plate_cell: Cell
    gate_room: str
    gate_cells: tuple[Cell, Cell]

    @property
    def gate_edge(self) -> Edge:
        return frozenset(self.gate_cells)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows_a: tuple[str, ...]
    rows_b: tuple[str, ...]
    links: tuple[LinkSpec, ...]
    baseline: int

    @property
    def budget(self) -> int:
        return self.baseline * 6


LEVEL_SPECS = (
    LevelSpec(
        "Inward Reflection",
        (".......", ".......", ".####..", ".S..G..", ".####..", ".......", "......."),
        (".......", ".......", "..####.", "..G..S.", "..####.", ".......", "......."),
        (),
        3,
    ),
    LevelSpec(
        "The Shared Rise",
        (".......", ".....G.", ".......", "######.", ".......", ".S.....", "......."),
        (".......", ".G.....", ".......", ".######", ".......", ".....S.", "......."),
        (),
        10,
    ),
    LevelSpec(
        "The Clutch",
        (".......", ".......", "...G...", "#......", "#S.....", "#......", "......."),
        (".......", ".......", "..G....", ".......", "...S...", ".......", "......."),
        (),
        5,
    ),
    LevelSpec(
        "Hold the Diamond",
        (".......", ".......", "...##..", "...D#.G", "...##..", "...#...", "S......"),
        (".G#....", "..#....", "..#....", ".......", "..#....", "..#....", "..#...S"),
        (LinkSpec("Diamond", "A", (3, 3), "B", ((2, 3), (3, 3))),),
        26,
    ),
    LevelSpec(
        "Reciprocal Custody",
        ("....#..", ".##.#..", ".#D.#..", ".####.G", "......#", "....##.", "...S#.."),
        ("S..#...", "...#...", ".......", "..###.G", "...###.", "..###O.", "...###."),
        (LinkSpec("Diamond", "A", (2, 2), "B", ((2, 2), (3, 2))), LinkSpec("Ring", "B", (5, 5), "A", ((3, 4), (4, 4)))),
        45,
    ),
    LevelSpec(
        "Dock First",
        ("..#S##.", "....#..", ".......", ".####..", ".#D.#..", ".##.#..", "....#G."),
        ("...###.", "...##O.", "...###.", "...#...", ".......", "...##..", "S..#G.."),
        (LinkSpec("Diamond", "A", (2, 4), "B", ((2, 4), (3, 4))), LinkSpec("Ring", "B", (5, 1), "A", ((3, 2), (4, 2)))),
        50,
    ),
    LevelSpec(
        "Relay Chamber",
        (".#####.", ".D###T.", ".#####.", "...#...", ".......", "...##.#", "S..#..G"),
        ("#O..##.", "##.....", "....#..", "....#.S", "##.##..", ".......", "......G"),
        (
            LinkSpec("Diamond", "A", (1, 1), "B", ((4, 1), (5, 1))),
            LinkSpec("Ring", "B", (1, 0), "A", ((2, 4), (3, 4))),
            LinkSpec("Bars", "A", (5, 1), "B", ((2, 3), (2, 4))),
        ),
        54,
    ),
)


def _parse_room(rows: tuple[str, ...]) -> dict[str, object]:
    if len(rows) != 7 or any(len(row) != 7 for row in rows):
        raise ValueError("Twin Shadow rooms must be 7x7")
    walls: set[Cell] = set()
    start: Cell | None = None
    goal: Cell | None = None
    plates: dict[str, Cell] = {}
    plate_symbols = {"D": "Diamond", "O": "Ring", "T": "Bars"}
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            if symbol == "#":
                walls.add((x, y))
            elif symbol == "S":
                start = (x, y)
            elif symbol == "G":
                goal = (x, y)
            elif symbol in plate_symbols:
                plates[plate_symbols[symbol]] = (x, y)
    if start is None or goal is None:
        raise ValueError("Every Twin Shadow room needs one start and one goal")
    return {"walls": walls, "start": start, "goal": goal, "plates": plates}


def _build_level(spec: LevelSpec) -> Level:
    room_a = _parse_room(spec.rows_a)
    room_b = _parse_room(spec.rows_b)
    for link in spec.links:
        room = room_a if link.plate_room == "A" else room_b
        if room["plates"].get(link.motif) != link.plate_cell:
            raise ValueError(f"{spec.name}: {link.motif} plate metadata does not match ASCII map")
    return Level(
        sprites=[],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={"spec": spec, "rooms": {"A": room_a, "B": room_b}, "budget": spec.budget},
    )


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


class TwinShadowDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: TwinShadow | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        frame[:, :] = COLOR_BACKGROUND
        if game is None:
            return frame
        self._draw_completion_emblem(frame, game)
        self._draw_rails(frame, game)
        self._draw_rooms(frame, game)
        self._draw_mirror(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _draw_completion_emblem(self, frame: np.ndarray, game: TwinShadow) -> None:
        # Two unmistakable sockets joined at the mirror seam.
        frame[3:9, 25:32] = COLOR_A_GOAL
        frame[4:8, 26:32] = COLOR_BACKGROUND
        frame[3:9, 32:39] = COLOR_B_GOAL
        frame[4:8, 32:38] = COLOR_BACKGROUND
        if game.pos_a == game.goal_a:
            frame[4:8, 27:32] = COLOR_A
        if game.pos_b == game.goal_b:
            frame[4:8, 32:37] = COLOR_B
        frame[2:10, 31:33] = COLOR_FRAME
        if game.visual_phase.get("completion_flash"):
            frame[1:11, 31:33] = COLOR_FLASH

    def _draw_rails(self, frame: np.ndarray, game: TwinShadow) -> None:
        for index, link in enumerate(game.links):
            y = MOTIF_ROWS[link.motif]
            color = MOTIF_COLORS[link.motif]
            open_now = game._link_open(index)
            stage = game._transition_stage(index)
            old_open = game.visual_phase.get("old_open", {}).get(index, open_now)
            shown_open = old_open if stage == 1 else open_now
            pulse = index in game.visual_phase.get("rail_pulses", ())
            rail_color = COLOR_FLASH if pulse else color
            source_color = COLOR_A if link.plate_room == "A" else COLOR_B
            target_color = COLOR_A if link.gate_room == "A" else COLOR_B
            frame[y : y + 2, 10:54] = rail_color
            frame[y - 1 : y + 3, 6:10] = COLOR_BACKGROUND
            self._draw_mini_motif(frame, 6, y - 1, link.motif, rail_color)
            frame[y - 1 : y + 3, 54:58] = COLOR_BACKGROUND
            if shown_open:
                frame[y - 1, 54:58] = COLOR_FRAME
                frame[y + 2, 54:58] = COLOR_FRAME
            else:
                self._draw_mini_gate(frame, 54, y - 1, link.motif, rail_color)
            frame[y, 4] = source_color
            frame[y + 1, 59] = target_color

    def _draw_mini_motif(self, frame: np.ndarray, x: int, y: int, motif: str, color: int) -> None:
        if motif == "Diamond":
            frame[y, x + 1 : x + 3] = color
            frame[y + 1 : y + 3, x : x + 4] = color
            frame[y + 3, x + 1 : x + 3] = color
        elif motif == "Ring":
            frame[y : y + 4, x] = color
            frame[y : y + 4, x + 3] = color
            frame[y, x : x + 4] = color
            frame[y + 3, x : x + 4] = color
        else:
            frame[y : y + 4, x] = color
            frame[y : y + 4, x + 3] = color

    def _draw_mini_gate(self, frame: np.ndarray, x: int, y: int, motif: str, color: int) -> None:
        if motif == "Bars":
            frame[y : y + 4, x] = color
            frame[y : y + 4, x + 3] = color
        else:
            frame[y : y + 4, x : x + 4] = color
            frame[y + 1 : y + 3, x + 1 : x + 3] = COLOR_SHADOW

    def _draw_rooms(self, frame: np.ndarray, game: TwinShadow) -> None:
        for room_name in ("A", "B"):
            ox, oy = ORIGINS[room_name]
            frame[oy - 1, ox - 1 : ox + 29] = COLOR_FRAME
            frame[oy + 28, ox - 1 : ox + 29] = COLOR_FRAME
            frame[oy - 1 : oy + 29, ox - 1] = COLOR_FRAME
            frame[oy - 1 : oy + 29, ox + 28] = COLOR_FRAME
            walls = game.walls_a if room_name == "A" else game.walls_b
            for y in range(7):
                for x in range(7):
                    px, py = ox + x * CELL_SIZE, oy + y * CELL_SIZE
                    frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = COLOR_WALL if (x, y) in walls else COLOR_FLOOR

        self._draw_goal(frame, "A", game.goal_a, game.pos_a == game.goal_a)
        self._draw_goal(frame, "B", game.goal_b, game.pos_b == game.goal_b)
        for index, link in enumerate(game.links):
            self._draw_plate(frame, game, index, link)
        for index, link in enumerate(game.links):
            self._draw_gate(frame, game, index, link)
        self._draw_obstruction_flash(frame, game)
        self._draw_agent(frame, "A", game.pos_a, COLOR_A)
        self._draw_agent(frame, "B", game.pos_b, COLOR_B)

    def _draw_goal(self, frame: np.ndarray, room: str, cell: Cell, occupied: bool) -> None:
        x, y = self._cell_pixel(room, cell)
        color = COLOR_FLASH if occupied else (COLOR_A_GOAL if room == "A" else COLOR_B_GOAL)
        frame[y, x : x + 4] = color
        frame[y + 3, x : x + 4] = color
        frame[y : y + 4, x] = color
        frame[y : y + 4, x + 3] = color

    def _draw_plate(self, frame: np.ndarray, game: TwinShadow, index: int, link: LinkSpec) -> None:
        x, y = self._cell_pixel(link.plate_room, link.plate_cell)
        color = MOTIF_COLORS[link.motif]
        occupied = game._link_open(index)
        stage = game._transition_stage(index)
        if occupied and stage >= 2:
            frame[y : y + 4, x : x + 4] = color
            frame[y + 1 : y + 3, x + 1 : x + 3] = COLOR_FLASH
            return
        if link.motif == "Diamond":
            frame[y, x + 1 : x + 3] = color
            frame[y + 1 : y + 3, x : x + 4] = color
            frame[y + 3, x + 1 : x + 3] = color
        elif link.motif == "Ring":
            frame[y, x : x + 4] = color
            frame[y + 3, x : x + 4] = color
            frame[y : y + 4, x] = color
            frame[y : y + 4, x + 3] = color
        else:
            frame[y : y + 4, x] = color
            frame[y : y + 4, x + 3] = color

    def _draw_gate(self, frame: np.ndarray, game: TwinShadow, index: int, link: LinkSpec) -> None:
        (x1, y1), (x2, y2) = link.gate_cells
        color = MOTIF_COLORS[link.motif]
        open_now = game._link_open(index)
        stage = game._transition_stage(index)
        # During a transition, stage 1 favors the old shape, stage 2 is half-way,
        # and stage 3 is the committed final shape.
        old_open = game.visual_phase.get("old_open", {}).get(index, open_now)
        shown_open = old_open if stage == 1 else open_now
        partial = stage == 2 and old_open != open_now
        ox, oy = ORIGINS[link.gate_room]
        if y1 == y2:
            boundary_x = ox + max(x1, x2) * CELL_SIZE
            top = oy + y1 * CELL_SIZE
            if shown_open:
                frame[top, boundary_x - 1 : boundary_x + 1] = COLOR_FRAME
                frame[top + 3, boundary_x - 1 : boundary_x + 1] = COLOR_FRAME
            elif partial:
                frame[top : top + 2, boundary_x - 1 : boundary_x + 1] = color
                frame[top + 3, boundary_x - 1 : boundary_x + 1] = COLOR_FRAME
            else:
                frame[top : top + 4, boundary_x - 1 : boundary_x + 1] = color
        else:
            boundary_y = oy + max(y1, y2) * CELL_SIZE
            left = ox + x1 * CELL_SIZE
            if shown_open:
                frame[boundary_y - 1 : boundary_y + 1, left] = COLOR_FRAME
                frame[boundary_y - 1 : boundary_y + 1, left + 3] = COLOR_FRAME
            elif partial:
                frame[boundary_y - 1 : boundary_y + 1, left : left + 2] = color
                frame[boundary_y - 1 : boundary_y + 1, left + 3] = COLOR_FRAME
            else:
                frame[boundary_y - 1 : boundary_y + 1, left : left + 4] = color

    def _draw_obstruction_flash(self, frame: np.ndarray, game: TwinShadow) -> None:
        blocked = game.visual_phase.get("blocked", {})
        for room, detail in blocked.items():
            current, delta, kind, target = detail
            ox, oy = ORIGINS[room]
            x, y = current
            dx, dy = delta
            if kind == "wall":
                tx, ty = target
                px, py = ox + tx * 4, oy + ty * 4
                if dx > 0:
                    frame[py : py + 4, px] = COLOR_BLOCKED
                elif dx < 0:
                    frame[py : py + 4, px + 3] = COLOR_BLOCKED
                elif dy > 0:
                    frame[py, px : px + 4] = COLOR_BLOCKED
                else:
                    frame[py + 3, px : px + 4] = COLOR_BLOCKED
            elif kind == "boundary":
                px, py = ox + x * 4, oy + y * 4
                if dx > 0:
                    frame[py : py + 4, ox + 28] = COLOR_BLOCKED
                elif dx < 0:
                    frame[py : py + 4, ox - 1] = COLOR_BLOCKED
                elif dy > 0:
                    frame[oy + 28, px : px + 4] = COLOR_BLOCKED
                else:
                    frame[oy - 1, px : px + 4] = COLOR_BLOCKED
            else:  # gate
                link_index = int(target)
                link = game.links[link_index]
                (x1, y1), (x2, y2) = link.gate_cells
                ox, oy = ORIGINS[link.gate_room]
                if y1 == y2:
                    boundary_x = ox + max(x1, x2) * CELL_SIZE
                    top = oy + y1 * CELL_SIZE
                    frame[top : top + 4, boundary_x - 1 : boundary_x + 1] = COLOR_BLOCKED
                else:
                    boundary_y = oy + max(y1, y2) * CELL_SIZE
                    left = ox + x1 * CELL_SIZE
                    frame[boundary_y - 1 : boundary_y + 1, left : left + 4] = COLOR_BLOCKED

    def _draw_agent(self, frame: np.ndarray, room: str, cell: Cell, color: int) -> None:
        x, y = self._cell_pixel(room, cell)
        detail = self.game.visual_phase.get("blocked", {}).get(room) if self.game else None
        if detail:
            dx, dy = detail[1]
            x += dx
            y += dy
        # Connected L-shaped 2x2 twin leaves one center pixel visible on a plate.
        frame[y + 1, x + 1 : x + 3] = color
        frame[y + 2, x + 1] = color

    def _draw_mirror(self, frame: np.ndarray, game: TwinShadow) -> None:
        seam_color = COLOR_FLASH if game.visual_phase.get("completion_flash") else COLOR_FRAME
        frame[25:51, 32] = seam_color
        frame[34:37, 31] = COLOR_A
        frame[35, 33] = COLOR_A
        frame[39:42, 33] = COLOR_B
        frame[40, 31] = COLOR_B

    def _draw_budget(self, frame: np.ndarray, game: TwinShadow) -> None:
        # A compact hourglass plus a continuous depletion bar.
        frame[56, 2:6] = COLOR_FRAME
        frame[62, 2:6] = COLOR_FRAME
        frame[57, 2] = frame[57, 5] = COLOR_FRAME
        frame[61, 2] = frame[61, 5] = COLOR_FRAME
        frame[58, 3:5] = COLOR_FLASH
        frame[60, 3:5] = COLOR_SHADOW
        frame[57:62, 8:62] = COLOR_SHADOW
        frame[56, 8:62] = COLOR_FRAME
        frame[62, 8:62] = COLOR_FRAME
        frame[56:63, 8] = COLOR_FRAME
        frame[56:63, 61] = COLOR_FRAME
        width = int(52 * game.remaining_steps / max(1, game.step_budget))
        if width:
            frame[58:61, 9 : 9 + width] = COLOR_FLASH

    @staticmethod
    def _cell_pixel(room: str, cell: Cell) -> tuple[int, int]:
        ox, oy = ORIGINS[room]
        return ox + cell[0] * CELL_SIZE, oy + cell[1] * CELL_SIZE


class TwinShadow(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = TwinShadowDisplay()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "twin_shadow-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._display]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK, ACTION_UNDO],
            seed,
        )
        self._display.game = self

    def on_set_level(self, _level: Level) -> None:
        self.spec: LevelSpec = self.current_level.get_data("spec")
        rooms = self.current_level.get_data("rooms")
        self.walls_a = set(rooms["A"]["walls"])
        self.walls_b = set(rooms["B"]["walls"])
        self.pos_a = tuple(rooms["A"]["start"])
        self.pos_b = tuple(rooms["B"]["start"])
        self.goal_a = tuple(rooms["A"]["goal"])
        self.goal_b = tuple(rooms["B"]["goal"])
        self.links = tuple(self.spec.links)
        self.step_budget = int(self.current_level.get_data("budget"))
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[Cell, Cell]] = []
        self.visual_phase: dict[str, object] = {}
        self.animation_queue: list[dict[str, object]] = []
        self._pending_outcome: str | None = None
        self.won = False

    def _link_open(self, index: int) -> bool:
        link = self.links[index]
        holder = self.pos_a if link.plate_room == "A" else self.pos_b
        return holder == link.plate_cell

    def _transition_stage(self, index: int) -> int:
        transitions = self.visual_phase.get("transitions", {})
        return int(transitions.get(index, 3))

    def _blocked_reason(
        self, room: str, current: Cell, delta: Cell, open_at_start: tuple[bool, ...]
    ) -> tuple[str, object] | None:
        candidate = (current[0] + delta[0], current[1] + delta[1])
        if not (0 <= candidate[0] <= 6 and 0 <= candidate[1] <= 6):
            return "boundary", candidate
        walls = self.walls_a if room == "A" else self.walls_b
        if candidate in walls:
            return "wall", candidate
        edge = frozenset((current, candidate))
        for index, link in enumerate(self.links):
            if link.gate_room == room and link.gate_edge == edge and not open_at_start[index]:
                return "gate", index
        return None

    def _resolve_move(self, action: int) -> None:
        deltas_a = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}
        dx, dy = deltas_a[action]
        delta_a = (dx, dy)
        delta_b = ((-dx if dx else 0), dy)
        old_a, old_b = self.pos_a, self.pos_b
        open_before = tuple(self._link_open(index) for index in range(len(self.links)))
        reason_a = self._blocked_reason("A", old_a, delta_a, open_before)
        reason_b = self._blocked_reason("B", old_b, delta_b, open_before)
        new_a = old_a if reason_a else (old_a[0] + delta_a[0], old_a[1] + delta_a[1])
        new_b = old_b if reason_b else (old_b[0] + delta_b[0], old_b[1] + delta_b[1])
        if (new_a, new_b) != (old_a, old_b):
            self.undo_history.append((old_a, old_b))
        self.pos_a, self.pos_b = new_a, new_b
        open_after = tuple(self._link_open(index) for index in range(len(self.links)))
        changed = [index for index in range(len(self.links)) if open_before[index] != open_after[index]]
        blocked: dict[str, object] = {}
        if reason_a:
            blocked["A"] = (old_a, delta_a, reason_a[0], reason_a[1])
        if reason_b:
            blocked["B"] = (old_b, delta_b, reason_b[0], reason_b[1])

        phases: list[dict[str, object]] = []
        if blocked:
            phases.append({"blocked": blocked})
        if changed:
            old_map = {index: open_before[index] for index in changed}
            for stage in (1, 2, 3):
                phases.append(
                    {
                        "transitions": {index: stage for index in changed},
                        "old_open": old_map,
                        "rail_pulses": tuple(changed) if stage == 2 else (),
                    }
                )

        self.remaining_steps -= 1
        self.won = self.pos_a == self.goal_a and self.pos_b == self.goal_b
        if self.won:
            for _ in range(6):
                phases.append({"completion_flash": True})
            self._pending_outcome = "next"
        elif self.remaining_steps <= 0:
            self._pending_outcome = "lose"
        phases.append({})
        self._start_phases(phases)

    def _resolve_undo(self) -> None:
        if not self.undo_history:
            self.complete_action()
            return
        open_before = tuple(self._link_open(index) for index in range(len(self.links)))
        self.pos_a, self.pos_b = self.undo_history.pop()
        self.won = False
        open_after = tuple(self._link_open(index) for index in range(len(self.links)))
        changed = [index for index in range(len(self.links)) if open_before[index] != open_after[index]]
        if not changed:
            self.visual_phase = {}
            self.complete_action()
            return
        old_map = {index: open_before[index] for index in changed}
        phases = [
            {
                "transitions": {index: stage for index in changed},
                "old_open": old_map,
                "rail_pulses": tuple(changed) if stage == 2 else (),
            }
            for stage in (1, 2, 3)
        ]
        self._start_phases(phases)

    def _start_phases(self, phases: list[dict[str, object]]) -> None:
        self.visual_phase = phases.pop(0) if phases else {}
        self.animation_queue = phases
        if not self.animation_queue:
            self._finish_action()

    def _advance_animation(self) -> None:
        self.visual_phase = self.animation_queue.pop(0)
        if not self.animation_queue:
            self._finish_action()

    def _finish_action(self) -> None:
        outcome = self._pending_outcome
        self._pending_outcome = None
        if outcome == "next":
            self.next_level()
            # Mark the submitted action complete as well.  For non-final levels
            # ArcEngine still performs its deferred level swap before returning,
            # but must not re-apply this action in the freshly loaded level.
            self.complete_action()
        elif outcome == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def step(self) -> None:
        if self.animation_queue:
            self._advance_animation()
            return
        action = _action_id(self.action.id)
        if action == int(GameAction.RESET.value):
            self.visual_phase = {}
            self.complete_action()
        elif action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            self._resolve_move(action)
        elif action == ACTION_UNDO:
            self._resolve_undo()
        else:
            # SPACE and valid in-bounds clicks are intentionally inert.
            self.visual_phase = {}
            self.complete_action()
