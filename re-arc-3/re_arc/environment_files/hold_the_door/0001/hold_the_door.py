from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "hold_the_door-0001"
VIEW_SIZE = 64
CELL_SIZE = 4
BOARD_LEFT = 4
BOARD_TOP = 8

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7

MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

COLOR_HIGHLIGHT = 0
COLOR_WALL_TOP = 1
COLOR_WALL = 2
COLOR_FLOOR = 3
COLOR_BACKGROUND = 4
COLOR_SHADOW = 5
COLOR_PAIR_B = 6
COLOR_PAIR_D = 8
COLOR_PAIR_A = 9
COLOR_AGENT = 11
COLOR_STATUE = 12
COLOR_EXIT = 14
COLOR_PAIR_C = 15

PAIR_COLORS = {"A": COLOR_PAIR_A, "B": COLOR_PAIR_B, "C": COLOR_PAIR_C, "D": COLOR_PAIR_D}
PAIR_MOTIFS = {"A": ((1, 0), (2, 0)), "B": ((0, 1), (0, 2)), "C": ((1, 3), (2, 3)), "D": ((3, 1), (3, 2))}


@dataclass(frozen=True)
class PairSpec:
    name: str
    trace: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    facing: str
    pairs: tuple[PairSpec, ...] = ()
    exit_under_statue: tuple[int, int] | None = None
    baseline_actions: int = 1


LEVEL_SPECS = (
    LevelSpec(
        "The Covered Exit",
        (
            "##############",
            "##############",
            "##############",
            "##############",
            "#####...######",
            "#####0@.######",
            "##############",
            "##############",
            "##############",
            "##############",
            "##############",
            "##############",
        ),
        "LEFT",
        exit_under_statue=(6, 6),
        baseline_actions=7,
    ),
    LevelSpec(
        "Change Sides",
        (
            "##############",
            "##############",
            "##############",
            "##############",
            "##############",
            "###E0@.#######",
            "#####..#######",
            "#####..#######",
            "##############",
            "##############",
            "##############",
            "##############",
        ),
        "LEFT",
        baseline_actions=15,
    ),
    LevelSpec(
        "Occupied Means Open",
        (
            "##############",
            "##############",
            "##############",
            "##############",
            "###...########",
            "###@A.0.a.E###",
            "##############",
            "##############",
            "##############",
            "##############",
            "##############",
            "##############",
        ),
        "RIGHT",
        (PairSpec("A", ((5, 6), (5, 7), (9, 7), (9, 6))),),
        baseline_actions=15,
    ),
    LevelSpec(
        "Before You Leave It There",
        (
            "##############",
            "##############",
            "##############",
            "##############",
            "#####..#######",
            "#####...a...bE",
            "####1A.##..###",
            "#####0@##B.###",
            "#########..###",
            "##############",
            "##############",
            "##############",
        ),
        "LEFT",
        (PairSpec("A", ((6, 7), (6, 4), (9, 4), (9, 6))), PairSpec("B", ((10, 8), (10, 10), (13, 10), (13, 6)))),
        baseline_actions=36,
    ),
    LevelSpec(
        "Borrow the Holder",
        (
            "##############",
            "##############",
            "##############",
            "##############",
            "##############",
            "######..a1..bE",
            "#####.0A@#..##",
            "#####...##B.##",
            "##########..##",
            "##############",
            "##############",
            "##############",
        ),
        "LEFT",
        (PairSpec("A", ((8, 7), (8, 4), (9, 4), (9, 6))), PairSpec("B", ((11, 8), (11, 10), (13, 10), (13, 6)))),
        baseline_actions=43,
    ),
    LevelSpec(
        "The Handoff",
        (
            "##############",
            "#####Ec#..####",
            "######.C..####",
            "#########b####",
            "########...###",
            "########...###",
            "####...#.B.###",
            "####...a...###",
            "###1A.2.######",
            "####0@########",
            "##############",
            "##############",
        ),
        "UP",
        (
            PairSpec("A", ((5, 9), (3, 9), (3, 11), (8, 11), (8, 8))),
            PairSpec("B", ((10, 7), (13, 7), (13, 4), (10, 4))),
            PairSpec("C", ((8, 3), (8, 2), (7, 2))),
        ),
        baseline_actions=77,
    ),
    LevelSpec(
        "Everybody Has a Job",
        (
            "##############",
            "###Ed.c#..####",
            "######.C..####",
            "#########b####",
            "#########..###",
            "########...###",
            "####...#.BD.##",
            "###3...a....##",
            "###1A.2.######",
            "####0@########",
            "##############",
            "##############",
        ),
        "UP",
        (
            PairSpec("A", ((5, 9), (3, 9), (3, 11), (8, 11), (8, 8))),
            PairSpec("B", ((10, 7), (10, 6), (13, 6), (13, 4), (10, 4))),
            PairSpec("C", ((8, 3), (8, 2), (7, 2))),
            PairSpec("D", ((11, 7), (11, 11), (14, 11), (14, 12), (2, 12), (2, 2), (5, 2))),
        ),
        baseline_actions=102,
    ),
)


def _cell_top_left(cell: tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    return BOARD_LEFT + CELL_SIZE * (x - 1), BOARD_TOP + CELL_SIZE * (y - 1)


def _action_id(action_id: object) -> int:
    return int(getattr(action_id, "value", action_id))


def _parse_spec(spec: LevelSpec) -> dict[str, object]:
    if len(spec.rows) != 12 or any(len(row) != 14 for row in spec.rows):
        raise ValueError(f"{spec.name}: layout must be 14x12")
    floor: set[tuple[int, int]] = set()
    statues: list[tuple[int, int]] = []
    player = None
    exit_cell = spec.exit_under_statue
    plates: dict[str, tuple[int, int]] = {}
    gates: dict[str, set[tuple[int, int]]] = {}
    for y, row in enumerate(spec.rows, start=1):
        for x, symbol in enumerate(row, start=1):
            if symbol == "#":
                continue
            cell = (x, y)
            floor.add(cell)
            if symbol == "@":
                player = cell
            elif symbol.isdigit():
                statue_id = int(symbol)
                while len(statues) <= statue_id:
                    statues.append((-1, -1))
                statues[statue_id] = cell
            elif symbol == "E":
                exit_cell = cell
            elif symbol in PAIR_COLORS:
                plates[symbol] = cell
            elif symbol.upper() in PAIR_COLORS and symbol.islower():
                gates.setdefault(symbol.upper(), set()).add(cell)
    if player is None or exit_cell is None or any(cell == (-1, -1) for cell in statues):
        raise ValueError(f"{spec.name}: incomplete player, statue, or exit data")
    pair_names = {pair.name for pair in spec.pairs}
    if pair_names != set(plates) or pair_names != set(gates):
        raise ValueError(f"{spec.name}: plate/gate/trace mismatch")
    return {
        "floor": floor,
        "player": player,
        "statues": tuple(statues),
        "exit": exit_cell,
        "plates": plates,
        "gates": gates,
    }


def _make_level(spec: LevelSpec) -> Level:
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level(
        sprites=[board],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={"spec": spec, "parsed": _parse_spec(spec)},
    )


class HoldTheDoorHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: HoldTheDoor | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        # Exploration is unlimited in this game.  A previous implementation
        # rendered the host safety budget as an in-world countdown, which made
        # the intended persistent plate/gate state look time-dependent.
        return frame


class HoldTheDoor(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = HoldTheDoorHud()
        levels = [_make_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._hud]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )
        self._hud.game = self

    def on_set_level(self, _level: Level) -> None:
        self.spec: LevelSpec = self.current_level.get_data("spec")
        parsed = self.current_level.get_data("parsed")
        self.floor = set(parsed["floor"])
        self.agent_position = tuple(parsed["player"])
        self.agent_facing = self.spec.facing
        self.statue_positions = tuple(tuple(cell) for cell in parsed["statues"])
        self.exit_position = tuple(parsed["exit"])
        self.plates = {name: tuple(cell) for name, cell in parsed["plates"].items()}
        self.gates = {name: set(cells) for name, cells in parsed["gates"].items()}
        self.gripped_statue_id: int | None = None
        self.won = False
        self.undo_history: list[tuple[tuple[int, int], str, tuple[tuple[int, int], ...], int | None, bool]] = []
        self.animation_frames: list[dict[str, object]] = []
        self._finish_after_animation: str | None = None
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self._render_settled()

    def _snapshot(self) -> tuple[tuple[int, int], str, tuple[tuple[int, int], ...], int | None, bool]:
        return (self.agent_position, self.agent_facing, self.statue_positions, self.gripped_statue_id, self.won)

    def _restore(self, snapshot) -> None:
        self.agent_position, self.agent_facing, self.statue_positions, self.gripped_statue_id, self.won = snapshot

    def _gate_states(
        self, agent: tuple[int, int] | None = None, statues: tuple[tuple[int, int], ...] | None = None
    ) -> dict[str, bool]:
        agent = self.agent_position if agent is None else agent
        statues = self.statue_positions if statues is None else statues
        occupied = {agent, *statues}
        return {name: plate in occupied for name, plate in self.plates.items()}

    def _is_traversable(
        self, cell: tuple[int, int], statues: tuple[tuple[int, int], ...], gate_states: dict[str, bool]
    ) -> bool:
        if cell not in self.floor or cell in statues:
            return False
        return all(gate_states[name] or cell not in cells for name, cells in self.gates.items())

    def _draw_trace(self, frame: np.ndarray, pair: PairSpec) -> None:
        color = PAIR_COLORS[pair.name]
        points = pair.trace
        for start, end in pairwise(points):
            x1, y1 = start
            x2, y2 = end
            if x1 == x2:
                px, py1 = _cell_top_left((x1, min(y1, y2)))
                _, py2 = _cell_top_left((x2, max(y1, y2)))
                frame[py1 : py2 + CELL_SIZE, px] = color
            elif y1 == y2:
                px1, py = _cell_top_left((min(x1, x2), y1))
                px2, _ = _cell_top_left((max(x1, x2), y2))
                frame[py + CELL_SIZE - 1, px1 : px2 + CELL_SIZE] = color
            else:
                raise ValueError("Trace waypoints must be orthogonal")

    def _draw_terrain(self, frame: np.ndarray) -> None:
        for cell in self.floor:
            px, py = _cell_top_left(cell)
            frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = COLOR_FLOOR
        for y in range(1, 13):
            for x in range(1, 15):
                cell = (x, y)
                if cell in self.floor:
                    continue
                neighbors = {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}
                if not (neighbors & self.floor):
                    continue
                px, py = _cell_top_left(cell)
                frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = COLOR_WALL
                if (x, y + 1) in self.floor:
                    frame[py, px : px + CELL_SIZE] = COLOR_WALL_TOP
                if (x, y - 1) in self.floor:
                    frame[py + CELL_SIZE - 1, px : px + CELL_SIZE] = COLOR_SHADOW
        for pair in self.spec.pairs:
            self._draw_trace(frame, pair)

    def _draw_wall_shudder(self, frame: np.ndarray, cell: tuple[int, int], shift: tuple[int, int]) -> None:
        px, py = _cell_top_left(cell)
        dx, dy = shift
        x0, y0 = px + dx, py + dy
        frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = COLOR_WALL
        frame[y0, x0 : x0 + CELL_SIZE] = COLOR_WALL_TOP
        frame[y0 + CELL_SIZE - 1, x0 : x0 + CELL_SIZE] = COLOR_SHADOW

    def _draw_exit(self, frame: np.ndarray, flash: bool) -> None:
        px, py = _cell_top_left(self.exit_position)
        edge = COLOR_HIGHLIGHT if flash else COLOR_EXIT
        for u, v in ((1, 0), (2, 0), (0, 1), (3, 1), (0, 2), (3, 2), (1, 3), (2, 3)):
            frame[py + v, px + u] = edge
        frame[py + 1 : py + 3, px + 1 : px + 3] = COLOR_HIGHLIGHT

    def _draw_plate(self, frame: np.ndarray, name: str, occupied: bool) -> None:
        px, py = _cell_top_left(self.plates[name])
        color = PAIR_COLORS[name]
        frame[py, px : px + CELL_SIZE] = color
        frame[py + 3, px : px + CELL_SIZE] = color
        frame[py : py + CELL_SIZE, px] = color
        frame[py : py + CELL_SIZE, px + 3] = color
        frame[py + 1 : py + 3, px + 1 : px + 3] = COLOR_WALL if occupied else COLOR_SHADOW
        for u, v in PAIR_MOTIFS[name]:
            frame[py + v, px + u] = COLOR_HIGHLIGHT

    def _draw_gate(self, frame: np.ndarray, name: str, openness: int, shift: tuple[int, int] = (0, 0)) -> None:
        color = PAIR_COLORS[name]
        for cell in self.gates[name]:
            px, py = _cell_top_left(cell)
            if openness <= 0:
                dx, dy = shift
                x0, y0 = px + dx, py + dy
                frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE] = color
            elif openness == 1:
                frame[py, px : px + CELL_SIZE] = color
                frame[py + 3, px : px + CELL_SIZE] = color
            else:
                for u, v in ((0, 0), (3, 0), (0, 3), (3, 3)):
                    frame[py + v, px + u] = color
            for u, v in PAIR_MOTIFS[name]:
                frame[py + v, px + u] = COLOR_HIGHLIGHT

    def _draw_statue(
        self, frame: np.ndarray, statue_id: int, pixel: tuple[int, int], tilt: tuple[int, int] = (0, 0)
    ) -> None:
        px, py = pixel
        dx, dy = tilt
        px += dx
        py += dy
        caps = (((1, 0), (2, 0)), ((0, 0), (1, 0)), ((2, 0), (3, 0)), ((0, 0), (1, 0), (2, 0), (3, 0)))
        for u, v in caps[statue_id % 4]:
            if 0 <= px + u < VIEW_SIZE and 0 <= py + v < VIEW_SIZE:
                frame[py + v, px + u] = COLOR_STATUE
        for v in (1, 2):
            if 0 <= py + v < VIEW_SIZE:
                frame[py + v, max(0, px) : min(VIEW_SIZE, px + CELL_SIZE)] = COLOR_STATUE
        for u in (1, 2):
            if 0 <= px + u < VIEW_SIZE and 0 <= py + 3 < VIEW_SIZE:
                frame[py + 3, px + u] = COLOR_WALL

    def _draw_agent(self, frame: np.ndarray, pixel: tuple[int, int], facing: str) -> None:
        px, py = pixel
        frame[py + 1 : py + 3, px + 1 : px + 3] = COLOR_AGENT
        tabs = {"UP": ((1, 0), (2, 0)), "DOWN": ((1, 3), (2, 3)), "LEFT": ((0, 1), (0, 2)), "RIGHT": ((3, 1), (3, 2))}
        for u, v in tabs[facing]:
            frame[py + v, px + u] = COLOR_HIGHLIGHT

    def _draw_chain(self, frame: np.ndarray, agent_pixel: tuple[int, int], statue_pixel: tuple[int, int]) -> None:
        ax, ay = agent_pixel
        sx, sy = statue_pixel
        ac = (ax + 1, ay + 1)
        sc = (sx + 1, sy + 1)
        if abs(ax - sx) >= abs(ay - sy):
            y = max(0, min(VIEW_SIZE - 1, (ac[1] + sc[1]) // 2))
            frame[y, min(ac[0], sc[0]) : max(ac[0], sc[0]) + 1] = COLOR_HIGHLIGHT
        else:
            x = max(0, min(VIEW_SIZE - 1, (ac[0] + sc[0]) // 2))
            frame[min(ac[1], sc[1]) : max(ac[1], sc[1]) + 1, x] = COLOR_HIGHLIGHT

    def _render(self, visual: dict[str, object] | None = None) -> None:
        visual = visual or {}
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BACKGROUND, dtype=np.int8)
        self._draw_terrain(frame)
        wall_shudder = visual.get("wall_shudder")
        if wall_shudder is not None:
            self._draw_wall_shudder(frame, tuple(wall_shudder[0]), tuple(wall_shudder[1]))
        self._draw_exit(frame, bool(visual.get("exit_flash", False)))

        agent_cell = visual.get("agent_cell", self.agent_position)
        statue_cells = visual.get("statue_cells", self.statue_positions)
        gate_states = visual.get("gate_states") or self._gate_states(agent_cell, statue_cells)
        gate_progress = visual.get("gate_progress", {})
        gate_shift = visual.get("gate_shift", {})
        occupied = {agent_cell, *statue_cells}
        for name in self.plates:
            self._draw_plate(frame, name, self.plates[name] in occupied)
        for name, is_open in gate_states.items():
            default_progress = 2 if is_open else 0
            self._draw_gate(
                frame, name, int(gate_progress.get(name, default_progress)), tuple(gate_shift.get(name, (0, 0)))
            )

        agent_pixel = tuple(visual.get("agent_pixel", _cell_top_left(agent_cell)))
        statue_pixels = list(visual.get("statue_pixels", [_cell_top_left(cell) for cell in statue_cells]))
        statue_tilts = visual.get("statue_tilts", {})
        for idx, pixel in enumerate(statue_pixels):
            self._draw_statue(frame, idx, tuple(pixel), tuple(statue_tilts.get(idx, (0, 0))))
        self._draw_agent(frame, agent_pixel, self.agent_facing)
        grip_id = visual.get("grip_id", self.gripped_statue_id)
        if grip_id is not None:
            self._draw_chain(frame, agent_pixel, tuple(statue_pixels[int(grip_id)]))
        else:
            dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[self.agent_facing]
            faced = (agent_cell[0] + dx, agent_cell[1] + dy)
            if faced in statue_cells:
                statue_id = statue_cells.index(faced)
                px, py = statue_pixels[statue_id]
                # Persistent white handle marks the statue SPACE will grip.
                for u, v in ((0, 0), (3, 0), (0, 3), (3, 3)):
                    frame[py + v, px + u] = COLOR_HIGHLIGHT

        for name in self.plates:
            if self.plates[name] in occupied:
                px, py = _cell_top_left(self.plates[name])
                for u, v in ((0, 0), (3, 0), (0, 3), (3, 3)):
                    frame[py + v, px + u] = PAIR_COLORS[name]
                for u, v in PAIR_MOTIFS[name]:
                    frame[py + v, px + u] = COLOR_HIGHLIGHT
        self.board.pixels = frame

    def _render_settled(self) -> None:
        self._render()

    def _queue_frames(self, frames: list[dict[str, object]], finish: str | None = None) -> None:
        self.animation_frames = frames
        self._finish_after_animation = finish
        self._play_animation_frame()

    def _play_animation_frame(self) -> None:
        visual = self.animation_frames.pop(0)
        self._render(visual)
        if self.animation_frames:
            return
        finish = self._finish_after_animation
        self._finish_after_animation = None
        if finish == "win":
            self.won = True
            self.next_level()
            self.complete_action()
        elif finish == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _spend_step(self) -> bool:
        return False

    def _finish_frames(self, frames: list[dict[str, object]]) -> None:
        if self.agent_position == self.exit_position:
            frames.extend([{"exit_flash": True}, {}])
            self._queue_frames(frames, "win")
        else:
            self._queue_frames(frames)

    def _invalid_frames(
        self, action_delta: tuple[int, int], *, blocked_statue: int | None = None, blocked_gate: str | None = None
    ) -> list[dict[str, object]]:
        dx, dy = action_delta
        first: dict[str, object] = {}
        if blocked_statue is not None:
            first["statue_tilts"] = {blocked_statue: (dx, dy)}
        elif blocked_gate is not None:
            first["gate_shift"] = {blocked_gate: (dx, dy)}
        else:
            target = (self.agent_position[0] + dx, self.agent_position[1] + dy)
            first["wall_shudder"] = (target, (dx, dy))
        return [first, {}]

    def _resolve_arrow(self, action: int) -> None:
        dx, dy = MOVE_DELTAS[action]
        self.agent_facing = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}[action]
        old_agent = self.agent_position
        old_statues = self.statue_positions
        old_gates = self._gate_states()
        proposed_agent = old_agent
        proposed_statues = old_statues
        pulling = False

        if self.gripped_statue_id is None:
            proposed_agent = (old_agent[0] + dx, old_agent[1] + dy)
        else:
            statue = old_statues[self.gripped_statue_id]
            away = (old_agent[0] - statue[0], old_agent[1] - statue[1])
            if (dx, dy) != away:
                self._finish_frames([{}])
                return
            pulling = True
            proposed_agent = (old_agent[0] + dx, old_agent[1] + dy)
            moved = list(old_statues)
            moved[self.gripped_statue_id] = old_agent
            proposed_statues = tuple(moved)

        blocked_statue = None
        blocked_gate = None
        if proposed_agent in old_statues:
            blocked_statue = old_statues.index(proposed_agent)
        for name, cells in self.gates.items():
            if proposed_agent in cells and not old_gates[name]:
                blocked_gate = name
        if not self._is_traversable(proposed_agent, old_statues, old_gates):
            self._finish_frames(
                self._invalid_frames((dx, dy), blocked_statue=blocked_statue, blocked_gate=blocked_gate)
            )
            return

        new_gates = self._gate_states(proposed_agent, proposed_statues)
        occupants = {proposed_agent, *proposed_statues}
        closing_gate = next(
            (
                name
                for name, was_open in old_gates.items()
                if was_open and not new_gates[name] and bool(self.gates[name] & occupants)
            ),
            None,
        )
        if closing_gate is not None:
            self._finish_frames(self._invalid_frames((dx, dy), blocked_gate=closing_gate))
            return

        self.agent_position = proposed_agent
        self.statue_positions = proposed_statues
        changed_gates = [name for name in old_gates if old_gates[name] != new_gates[name]]
        frames: list[dict[str, object]] = []
        if pulling:
            agent_start = _cell_top_left(old_agent)
            statue_start = _cell_top_left(old_statues[self.gripped_statue_id])
            frames = [
                {
                    "agent_cell": old_agent,
                    "statue_cells": old_statues,
                    "gate_states": old_gates,
                    "grip_id": self.gripped_statue_id,
                },
                {
                    "agent_cell": old_agent,
                    "statue_cells": old_statues,
                    "gate_states": old_gates,
                    "agent_pixel": (agent_start[0] + dx, agent_start[1] + dy),
                    "statue_pixels": [
                        statue_start if idx == self.gripped_statue_id else _cell_top_left(cell)
                        for idx, cell in enumerate(old_statues)
                    ],
                    "statue_tilts": {self.gripped_statue_id: (dx, dy)},
                    "grip_id": self.gripped_statue_id,
                },
                {
                    "agent_cell": old_agent,
                    "statue_cells": old_statues,
                    "gate_states": new_gates,
                    "gate_progress": {name: 1 for name in changed_gates},
                    "agent_pixel": (agent_start[0] + dx * 2, agent_start[1] + dy * 2),
                    "statue_pixels": [
                        (statue_start[0] + dx * 2, statue_start[1] + dy * 2)
                        if idx == self.gripped_statue_id
                        else _cell_top_left(cell)
                        for idx, cell in enumerate(old_statues)
                    ],
                    "grip_id": self.gripped_statue_id,
                },
                {},
            ]
        elif changed_gates:
            frames = [
                {"agent_cell": old_agent, "statue_cells": old_statues, "gate_states": old_gates},
                {"gate_states": new_gates, "gate_progress": {name: 1 for name in changed_gates}},
                {},
            ]
        else:
            frames = [{}]
        self._finish_frames(frames)

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._render_settled()
            self.complete_action()
            return
        if self.won:
            self.complete_action()
            return
        if action == ACTION_UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            if self._spend_step():
                self._finish_frames([{}])
            else:
                self._render_settled()
                self.complete_action()
            return
        if action == ACTION_CLICK:
            self._spend_step()
            self._finish_frames([{}])
            return

        if action in MOVE_DELTAS or action == ACTION_SPACE:
            self.undo_history.append(self._snapshot())
        self._spend_step()
        if action == ACTION_SPACE:
            if self.gripped_statue_id is not None:
                self.gripped_statue_id = None
            else:
                dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[self.agent_facing]
                faced = (self.agent_position[0] + dx, self.agent_position[1] + dy)
                if faced in self.statue_positions:
                    self.gripped_statue_id = self.statue_positions.index(faced)
            self._finish_frames([{}])
            return
        if action in MOVE_DELTAS:
            self._resolve_arrow(action)
            return
        self._finish_frames([{}])
