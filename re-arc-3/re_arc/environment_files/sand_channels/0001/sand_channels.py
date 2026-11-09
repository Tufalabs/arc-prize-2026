from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = range(1, 8)
MOVE = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
CELL = 4
LEFT_EDGE, TOP_EDGE = 1, 6

WHITE, LIGHT_GRAY, GRAY, STONE, RECESS, BLACK = 0, 1, 2, 3, 4, 5
MAGENTA, PINK, RED, BLUE, LIGHT_BLUE, YELLOW, ORANGE, GREEN = 6, 7, 8, 9, 10, 11, 12, 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    basin_paths: dict[str, tuple[tuple[int, int], ...]]
    forks: dict[str, tuple[tuple[int, int], tuple[str, str], int]]
    root: str
    hopper: int
    baseline: int
    initially_filled: tuple[str, ...] = ()
    exit_under: str | None = None


LEVEL_SPECS = (
    LevelSpec(
        "Last Grain",
        (
            "#############",
            "#############",
            "#############",
            "#############",
            "###.#########",
            "##@.AX#######",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
        ),
        {"A": ((5, 4), (5, 5))},
        {},
        "A",
        12,
        3,
        ("A0",),
        "A1",
    ),
    LevelSpec(
        "Loose Is Not Floor",
        (
            "#############",
            "#############",
            "#############",
            "#############",
            "##..#########",
            "##@.AAX######",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
        ),
        {"A": ((5, 4), (5, 5), (5, 6))},
        {},
        "A",
        16,
        6,
        (),
        "A2",
    ),
    LevelSpec(
        "The Paddle",
        (
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "#BBfAAX######",
            "###@.########",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
        ),
        {"A": ((5, 4), (5, 5)), "B": ((5, 2), (5, 1))},
        {"f": ((5, 3), ("A", "B"), 1)},
        "f",
        14,
        7,
    ),
    LevelSpec(
        "One Step Before the Branch",
        (
            "#U###########",
            "#U####B######",
            "#UUUU#B######",
            "#UU#UUB######",
            "#####UfXAA###",
            "####...##A###",
            "###....##A###",
            "##.....AAA###",
            "#..#.#.#..###",
            "#....#....###",
            "#@###########",
            "#############",
            "#############",
        ),
        {
            "U": ((0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (2, 2), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)),
            "A": ((4, 7), (4, 8), (4, 9), (5, 9), (6, 9), (7, 9), (7, 8), (7, 7)),
            "B": ((3, 6), (2, 6), (1, 6)),
        },
        {"f": ((4, 6), ("A", "B"), 1)},
        "U",
        19,
        27,
        (),
        "A0",
    ),
    LevelSpec(
        "Bridge, Then Conduit",
        (
            "########D####",
            "########D####",
            "###.....gC###",
            "###B#AAAAC###",
            "###B#A###C###",
            "###B#A###C###",
            "##.B#A###C###",
            "#PPfAA###C###",
            "#P##AA#CCC###",
            "#P##CCCC#####",
            "#@..C########",
            "####X########",
            "#############",
        ),
        {
            "P": ((9, 1), (8, 1), (7, 1), (7, 2)),
            "B": ((6, 3), (5, 3), (4, 3), (3, 3)),
            "A": ((7, 4), (8, 4), (8, 5), (7, 5), (6, 5), (5, 5), (4, 5), (3, 5), (3, 6), (3, 7), (3, 8)),
            "C": (
                (2, 9),
                (3, 9),
                (4, 9),
                (5, 9),
                (6, 9),
                (7, 9),
                (8, 9),
                (8, 8),
                (8, 7),
                (9, 7),
                (9, 6),
                (9, 5),
                (9, 4),
                (10, 4),
                (11, 4),
            ),
            "D": ((1, 8), (0, 8)),
        },
        {"f": ((7, 3), ("A", "B"), 1), "g": ((2, 8), ("C", "D"), 1)},
        "P",
        30,
        34,
        ("P0", "P1", "P2", "P3"),
        "C14",
    ),
    LevelSpec(
        "Buy the Shortcut",
        (
            "#############",
            "#######.....#",
            "####....###.#",
            "###..MMgTTT.#",
            "###S#M#W##T.#",
            "###S#M#W##T.#",
            "###S#M#W##T.#",
            "##.S#M#WTTT.#",
            "#.@fMM##TTT.#",
            "#..#MM##TTT.#",
            "#.##MM#.XTT.#",
            "#.#####.#TT.#",
            "#...........#",
        ),
        {
            "S": ((7, 3), (6, 3), (5, 3), (4, 3)),
            "M": ((8, 4), (9, 4), (10, 4), (10, 5), (9, 5), (8, 5), (7, 5), (6, 5), (5, 5), (4, 5), (3, 5), (3, 6)),
            "T": (
                (3, 8),
                (3, 9),
                (3, 10),
                (4, 10),
                (5, 10),
                (6, 10),
                (7, 10),
                (8, 10),
                (9, 10),
                (10, 10),
                (11, 10),
                (11, 9),
                (10, 9),
                (9, 9),
                (8, 9),
                (7, 9),
                (7, 8),
                (8, 8),
                (9, 8),
                (10, 8),
            ),
            "W": ((4, 7), (5, 7), (6, 7), (7, 7)),
        },
        {"f": ((8, 3), ("M", "S"), 0), "g": ((3, 7), ("T", "W"), 1)},
        "f",
        36,
        41,
        (),
        "T19",
    ),
    LevelSpec(
        "Weave the Network",
        (
            "#############",
            "##########Q##",
            "######RR.#Q##",
            "###...R#.#Q##",
            "##....gN.#Q##",
            "##S#..MN..hE#",
            "##S#MMMN##NE#",
            "##S#MMMN##NE#",
            "..SMM##NNNNE#",
            ".@fM#EEEEEEE#",
            "#..##E#######",
            "##XEEE#######",
            "##EE#########",
        ),
        {
            "S": ((8, 2), (7, 2), (6, 2), (5, 2)),
            "M": ((9, 3), (8, 3), (8, 4), (7, 4), (6, 4), (6, 5), (7, 5), (7, 6), (6, 6), (5, 6)),
            "R": ((3, 6), (2, 6), (2, 7)),
            "N": ((4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (8, 8), (8, 9), (8, 10), (7, 10), (6, 10)),
            "E": (
                (5, 11),
                (6, 11),
                (7, 11),
                (8, 11),
                (9, 11),
                (9, 10),
                (9, 9),
                (9, 8),
                (9, 7),
                (9, 6),
                (9, 5),
                (10, 5),
                (11, 5),
                (11, 4),
                (11, 3),
                (12, 3),
                (12, 2),
                (11, 2),
            ),
            "Q": ((4, 10), (3, 10), (2, 10), (1, 10)),
        },
        {"f": ((9, 2), ("M", "S"), 0), "g": ((4, 6), ("N", "R"), 0), "h": ((5, 10), ("E", "Q"), 1)},
        "f",
        45,
        52,
        (),
        "E17",
    ),
)


def _xy(cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    return LEFT_EDGE + CELL * c, TOP_EDGE + CELL * r


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(
        np.full((64, 64), -1, dtype=np.int8), name="board", x=0, y=0, layer=0, collidable=False, tags=["board"]
    )
    return Level([board], (64, 64), {"spec": spec}, spec.name)


class SandChannelsView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: SandChannels | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.game is not None:
            self.game._draw(frame)
        return frame


class SandChannels(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = SandChannelsView()
        super().__init__(
            "sand_channels-0001",
            [_build_level(s) for s in LEVEL_SPECS],
            Camera(0, 0, 64, 64, BLACK, BLACK, [self._view]),
            False,
            len(LEVEL_SPECS),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed,
        )
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.stone: set[tuple[int, int]] = set()
        self.start = (-1, -1)
        self.exit = (-1, -1)
        for r, row in enumerate(self.spec.rows):
            for c, ch in enumerate(row):
                if ch in ".@X":
                    self.stone.add((r, c))
                if ch == "@":
                    self.start = (r, c)
                if ch == "X":
                    self.exit = (r, c)
        self.basin_at = {
            cell: (name, idx) for name, path in self.spec.basin_paths.items() for idx, cell in enumerate(path)
        }
        if self.spec.exit_under:
            name = self.spec.exit_under[0]
            idx = int(self.spec.exit_under[1:])
            self.basin_at[self.exit] = (name, idx)
            self.stone.discard(self.exit)
        self.fork_at = {data[0]: name for name, data in self.spec.forks.items()}
        self.player = self.start
        self.hopper = self.spec.hopper
        self.filled = set(self.spec.initially_filled)
        self.orientations = {name: data[2] for name, data in self.spec.forks.items()}
        self.spills = 0
        self.step_budget = self.spec.baseline * 6
        self.steps_remaining = self.step_budget
        self.undo_history: list[dict[str, Any]] = []
        self._anim: list[str] = []
        self._anim_index = 0
        self._anim_data: dict[str, Any] = {}
        self._pending_success = False
        self._pending_loss = False
        self.selected_object = None
        self.cursor_cell = None

    def _complete_basins(self) -> set[str]:
        return {
            name
            for name, path in self.spec.basin_paths.items()
            if all(f"{name}{i}" in self.filled for i in range(len(path)))
        }

    def _walkable(self, cell: tuple[int, int]) -> bool:
        if cell in self.stone:
            return True
        hit = self.basin_at.get(cell)
        return bool(hit and hit[0] in self._complete_basins())

    def _frontier(self) -> tuple[str, tuple[int, int] | None]:
        node = self.spec.root
        seen: set[str] = set()
        while node not in seen:
            seen.add(node)
            if node in self.spec.forks:
                branches = self.spec.forks[node][1]
                node = branches[self.orientations[node]]
                continue
            path = self.spec.basin_paths[node]
            for idx, cell in enumerate(path):
                if f"{node}{idx}" not in self.filled:
                    return f"{node}{idx}", cell
            successor = self._basin_successor(node)
            if successor is None:
                return "SPILL", path[-1]
            node = successor
        raise RuntimeError("Sand network must be acyclic")

    def _basin_successor(self, basin: str) -> str | None:
        # A basin feeds a fork when that fork is geometrically downstream in the authored graph.
        if basin == "U":
            return "f"
        if basin == "P":
            return "f"
        if basin == "A" and "g" in self.spec.forks:
            return "g"
        if basin == "M":
            return "g"
        if basin == "N":
            return "h"
        return None

    def _snapshot(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "hopper": self.hopper,
            "filled": set(self.filled),
            "orientations": dict(self.orientations),
            "spills": self.spills,
        }

    def _restore(self, state: dict[str, Any]) -> None:
        self.player = state["player"]
        self.hopper = state["hopper"]
        self.filled = set(state["filled"])
        self.orientations = dict(state["orientations"])
        self.spills = state["spills"]

    def _spend_action(self) -> None:
        self.steps_remaining -= 1
        self._pending_loss = self.steps_remaining <= 0

    def _start_anim(self, phases: list[str], data: dict[str, Any] | None = None) -> None:
        self._anim = phases
        self._anim_index = 0
        self._anim_data = data or {}

    def _finish_anim(self) -> None:
        self._anim = []
        self._anim_index = 0
        self._anim_data = {}
        if self._pending_success:
            self.next_level()
            self.complete_action()
        elif self._pending_loss:
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _advance_anim(self) -> None:
        if self._anim_index + 1 < len(self._anim):
            self._anim_index += 1
            if self._anim_index == len(self._anim) - 1:
                self._finish_anim()
        else:
            self._finish_anim()

    def step(self) -> None:
        if self._anim:
            self._advance_anim()
            return
        action = int(getattr(self.action.id, "value", self.action.id))
        if action == 0:
            self.complete_action()
            return
        self._spend_action()
        phases: list[str]
        data: dict[str, Any] = {}
        completed: set[str] = set()
        if action in MOVE:
            dr, dc = MOVE[action]
            target = (self.player[0] + dr, self.player[1] + dc)
            if not self._walkable(target):
                phases = ["bump", "settled"]
                data = {"direction": (dr, dc), "target": target}
            else:
                self.undo_history.append(self._snapshot())
                before = self._complete_basins()
                self.player = target
                route, frontier = self._frontier()
                if self.hopper > 0:
                    self.hopper -= 1
                    if route == "SPILL":
                        self.spills += 1
                    else:
                        self.filled.add(route)
                completed = self._complete_basins() - before
                phases = ["trace", "settled"]
                if completed:
                    phases = ["trace", "compact1", "compact2", "settled"]
                data = {"frontier": frontier, "completed": completed}
        elif action == SPACE:
            adjacent = [
                name
                for cell, name in self.fork_at.items()
                if abs(cell[0] - self.player[0]) + abs(cell[1] - self.player[1]) == 1
            ]
            if len(adjacent) == 1:
                name = adjacent[0]
                self.undo_history.append(self._snapshot())
                old = self.orientations[name]
                self.orientations[name] = 1 - old
                phases = ["paddle_old", "paddle_hub", "paddle_new"]
                data = {"fork": name, "old": old}
            else:
                phases = ["shake1", "shake2"]
        elif action == CLICK:
            x = int(self.action.data.get("x", 0))
            y = int(self.action.data.get("y", 0))
            phases = ["reticle1", "reticle2"]
            data = {"click": (x, y)}
        elif action == UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            phases = ["undo", "settled"]
        else:
            phases = ["settled"]
        self._pending_success = self.player == self.exit
        if self._pending_success:
            phases += ["halo1", "halo2", "halo3", "settled"]
            self._pending_loss = False
        self._start_anim(phases, data)
        if len(phases) == 1:
            self._finish_anim()

    def _draw(self, frame: np.ndarray) -> None:
        frame[:, :] = BLACK
        phase = self._anim[self._anim_index] if self._anim else ""
        complete = self._complete_basins()
        # Dry stone is one connected visual material; only exposed silhouette edges are accented.
        for cell in self.stone:
            x, y = _xy(cell)
            frame[y : y + CELL, x : x + CELL] = STONE
        for r, c in self.stone:
            x, y = _xy((r, c))
            if (r - 1, c) not in self.stone:
                frame[y, x : x + CELL] = GRAY
            if (r + 1, c) not in self.stone:
                frame[y + 3, x : x + CELL] = GRAY
            if (r, c - 1) not in self.stone:
                frame[y : y + CELL, x] = GRAY
            if (r, c + 1) not in self.stone:
                frame[y : y + CELL, x + 3] = GRAY
        # Basin silhouettes and directional grooves.
        for name, path in self.spec.basin_paths.items():
            is_complete = name in complete
            for idx, cell in enumerate(path):
                x, y = _xy(cell)
                frame[y : y + CELL, x : x + CELL] = ORANGE if is_complete else GRAY
                if not is_complete:
                    frame[y + 1 : y + 3, x + 1 : x + 3] = YELLOW if f"{name}{idx}" in self.filled else RECESS
                else:
                    frame[y + 1 : y + 3, x + 1 : x + 3] = ORANGE
                successor = path[idx + 1] if idx + 1 < len(path) else None
                if successor:
                    dr, dc = successor[0] - cell[0], successor[1] - cell[1]
                    frame[y + 2, x + 2] = RECESS if is_complete else LIGHT_GRAY
                    if dr < 0:
                        frame[y, x + 2] = RECESS if is_complete else LIGHT_GRAY
                    if dr > 0:
                        frame[y + 3, x + 2] = RECESS if is_complete else LIGHT_GRAY
                    if dc < 0:
                        frame[y + 2, x] = RECESS if is_complete else LIGHT_GRAY
                    if dc > 0:
                        frame[y + 2, x + 3] = RECESS if is_complete else LIGHT_GRAY
        _route, frontier = self._frontier()
        if frontier is not None:
            x, y = _xy(frontier)
            color = YELLOW if self.hopper else RED
            frame[y, x + 1 : x + 3] = color
            frame[y + 1, x + 1] = WHITE
        # Diverters.
        for name, (cell, branches, _) in self.spec.forks.items():
            x, y = _xy(cell)
            frame[y : y + CELL, x : x + CELL] = STONE
            shown_orientation = self.orientations[name]
            if phase == "paddle_old" and self._anim_data.get("fork") == name:
                shown_orientation = self._anim_data["old"]
            selected = branches[shown_orientation]
            target = self.spec.basin_paths[selected][0]
            dr, dc = target[0] - cell[0], target[1] - cell[1]
            frame[y + 1 : y + 3, x + 1 : x + 3] = YELLOW
            if not (phase == "paddle_hub" and self._anim_data.get("fork") == name):
                if dr < 0:
                    frame[y, x + 2] = YELLOW
                elif dr > 0:
                    frame[y + 3, x + 2] = YELLOW
                elif dc < 0:
                    frame[y + 2, x] = YELLOW
                else:
                    frame[y + 2, x + 3] = YELLOW
            if abs(cell[0] - self.player[0]) + abs(cell[1] - self.player[1]) == 1:
                frame[y, x] = frame[y, x + 3] = frame[y + 3, x] = frame[y + 3, x + 3] = PINK
        # Exit beacon overlays its terrain.
        ex, ey = _xy(self.exit)
        frame[ey : ey + 3, ex : ex + 3] = GREEN
        frame[ey + 1, ex + 1] = WHITE
        # Surveyor.
        px, py = _xy(self.player)
        ox = oy = 0
        if phase == "bump":
            dr, dc = self._anim_data.get("direction", (0, 0))
            ox, oy = dc, dr
            tx, ty = _xy(self._anim_data.get("target", self.player))
            frame[ty + 1 : ty + 3, tx + 1 : tx + 3] = RED
        if phase.startswith("shake"):
            ox = -1 if phase == "shake1" else 1
        px += ox
        py += oy
        frame[py + 1 : py + 4, px + 1 : px + 4] = LIGHT_BLUE
        frame[py + 1 : py + 3, px + 1] = BLUE
        frame[py + 1, px + 2] = WHITE
        # Transient route, click, compaction, and completion cues.
        if phase in {"trace", "compact1"}:
            for name in self._anim_data.get("completed", set()):
                path = self.spec.basin_paths[name]
                compacted_count = 0 if phase == "trace" else max(1, (len(path) + 1) // 2)
                for idx, cell in enumerate(path):
                    x, y = _xy(cell)
                    if idx < compacted_count:
                        frame[y : y + CELL, x : x + CELL] = ORANGE
                        frame[y + 2, x + 2] = RECESS
                    else:
                        frame[y : y + CELL, x : x + CELL] = GRAY
                        frame[y + 1 : y + 3, x + 1 : x + 3] = YELLOW
        trace_frontier = self._anim_data.get("frontier")
        if phase == "trace" and trace_frontier is not None:
            fx, fy = _xy(trace_frontier)
            frame[fy : fy + 2, fx : fx + 4] = YELLOW
        if phase.startswith("reticle"):
            cx, cy = self._anim_data.get("click", (0, 0))
            radius = 1 if phase == "reticle1" else 2
            for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
                if 0 <= cx + dx < 64 and 0 <= cy + dy < 64:
                    frame[cy + dy, cx + dx] = LIGHT_GRAY
        if phase.startswith("halo"):
            radius = int(phase[-1]) + 1
            x0, x1 = max(0, ex - radius), min(64, ex + 3 + radius)
            y0, y1 = max(0, ey - radius), min(64, ey + 3 + radius)
            frame[y0, x0:x1] = GREEN
            frame[y1 - 1, x0:x1] = GREEN
            frame[y0:y1, x0] = GREEN
            frame[y0:y1, x1 - 1] = GREEN
        # Hopper: exact 47-row reservoir with five-unit notches.
        frame[6:58, 56:63] = STONE
        frame[7:57, 57:62] = BLACK
        frame[8:55, 58:61] = RECESS
        if self.hopper:
            frame[55 - self.hopper : 55, 58:61] = YELLOW
        else:
            frame[56:58, 56:63] = RED
        for tick in range(5, 46, 5):
            frame[55 - tick, 55:58] = LIGHT_GRAY
        # Mandatory non-text action clock, underneath the board.
        frame[60:63, 1:53] = STONE
        width = max(0, min(50, (50 * self.steps_remaining) // max(1, self.step_budget)))
        if width:
            frame[61, 2 : 2 + width] = GREEN if self.steps_remaining > self.step_budget // 5 else RED


__all__ = ["SandChannels"]
