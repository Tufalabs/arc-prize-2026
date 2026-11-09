from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite
from arcengine.enums import GameAction

GAME_ID = "beam_compass-0001"
SIZE = 64
BOARD = 14
CELL = 4
ORIGIN = 4

MARGIN = 5
FIELD = 4
WALL = 2
EMITTER = 9
BEAM = 10
TERMINAL = 11
HALO = 12
CAP = 0
RIPPLE = 1
FLASH = 7
PULSE = 14

DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
N, E, S, W = range(4)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    orientations: tuple[int, ...]
    winning: tuple[int, ...]
    baseline: int

    @property
    def budget(self) -> int:
        return self.baseline * 6


LEVELS = (
    LevelSpec(
        "One Quarter Turn",
        (
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "...A......a...",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
        ),
        (N,),
        (E,),
        1,
    ),
    LevelSpec(
        "The Arrow in the Way",
        (
            "..............",
            "..............",
            "..a...........",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "..A....B....b.",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
        ),
        (E, S),
        (N, E),
        6,
    ),
    LevelSpec(
        "First Socket Wins",
        (
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "..............",
            "..A...a.....b.",
            "..............",
            "..............",
            "..............",
            "...c..C.....B.",
            "..............",
            "..............",
        ),
        (N, E, E),
        (E, N, W),
        6,
    ),
    LevelSpec(
        "Cross Without Colliding",
        (
            "..............",
            ".......B....d.",
            ".c..C.........",
            "..............",
            "............D.",
            "..............",
            "..............",
            ".A..........a.",
            "..............",
            "..............",
            "..............",
            "..............",
            ".......b......",
            "..............",
        ),
        (S, W, N, E),
        (E, S, W, N),
        12,
    ),
    LevelSpec(
        "Outward Pair",
        (
            "..............",
            "....a.....B...",
            "..............",
            ".eA.......c...",
            "...dD..E...g..",
            ".......b......",
            "..............",
            "..............",
            "..............",
            ".G............",
            ".......C......",
            "..............",
            ".......f..F...",
            "..............",
        ),
        (S, N, E, N, S, N, E),
        (E, W, N, W, E, W, N),
        21,
    ),
    LevelSpec(
        "The Blocked Approach",
        (
            "..............",
            "...d....g.H.I.",
            "..............",
            ".....e....E...",
            "..............",
            ".a...F..A...j.",
            "..............",
            "..............",
            "...D.f....J...",
            "..............",
            ".B.b.c......G.",
            ".#............",
            ".h.C....i.....",
            "..............",
        ),
        (W, S, N, E, N, N, N, N, W, N),
        (S, E, W, N, W, W, W, W, S, W),
        30,
    ),
    LevelSpec(
        "Borrowed Light",
        (
            "..............",
            ".k...J...H..A.",
            "..............",
            ".d.n...L...a..",
            "..............",
            ".E...c.F.m....",
            "..............",
            ".l.......I.Nh.",
            "..............",
            ".j.........i..",
            "..............",
            ".B.D.C#b.g.G..",
            ".e.......K.fM.",
            "..............",
        ),
        (W, E, E, E, E, W, N, W, N, N, N, S, N, W),
        (S, N, N, N, N, S, W, S, W, W, W, E, W, S),
        42,
    ),
)


def _parse(spec: LevelSpec) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]], set[tuple[int, int]]]:
    emitters: dict[str, tuple[int, int]] = {}
    terminals: dict[str, tuple[int, int]] = {}
    walls: set[tuple[int, int]] = set()
    for y, row in enumerate(spec.rows):
        if len(row) != BOARD:
            raise ValueError(f"{spec.name}: row {y} is not {BOARD} cells")
        for x, symbol in enumerate(row):
            if "A" <= symbol <= "N":
                emitters[symbol] = (x, y)
            elif "a" <= symbol <= "n":
                terminals[symbol] = (x, y)
            elif symbol == "#":
                walls.add((x, y))
    return emitters, terminals, walls


def _level(spec: LevelSpec) -> Level:
    board = Sprite(
        np.full((SIZE, SIZE), MARGIN, dtype=np.int8), name="board", x=0, y=0, layer=0, collidable=False, tags=["board"]
    )
    return Level([board], grid_size=(SIZE, SIZE), data={"spec": spec}, name=spec.name)


class BeamCompass(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            GAME_ID,
            [_level(spec) for spec in LEVELS],
            Camera(0, 0, SIZE, SIZE, MARGIN, MARGIN),
            False,
            len(LEVELS),
            [1, 2, 3, 4, 5, 6, 7],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.emitters, self.terminals, self.walls = _parse(self.spec)
        self.orientations = list(self.spec.orientations)
        self.remaining_steps = self.spec.budget
        self._history: list[tuple[int, ...]] = []
        self._frames: list[dict[str, object]] = []
        self._finish_after_frames = False
        self._board = level.get_sprites_by_tag("board")[0]
        self.rays, self.hits = self._raycast(self.orientations)
        self._render({"kind": "settled"})

    def _raycast(self, orientations: list[int] | tuple[int, ...]):
        occupied_emitters = {cell: emitter_id for emitter_id, cell in self.emitters.items()}
        occupied_terminals = {cell: terminal_id for terminal_id, cell in self.terminals.items()}
        rays: dict[str, dict[str, object]] = {}
        hits = {terminal_id: set() for terminal_id in self.terminals}
        for emitter_id in sorted(self.emitters):
            x, y = self.emitters[emitter_id]
            direction = orientations[ord(emitter_id) - ord("A")]
            dx, dy = DIRS[direction]
            cells: list[tuple[int, int]] = []
            while True:
                x, y = x + dx, y + dy
                if not (0 <= x < BOARD and 0 <= y < BOARD):
                    collision = ("rim", None, (x, y))
                    break
                if (x, y) in self.walls:
                    collision = ("wall", None, (x, y))
                    break
                if (x, y) in occupied_emitters:
                    collision = ("emitter", occupied_emitters[(x, y)], (x, y))
                    break
                if (x, y) in occupied_terminals:
                    terminal_id = occupied_terminals[(x, y)]
                    hits[terminal_id].add(emitter_id)
                    collision = ("terminal", terminal_id, (x, y))
                    break
                cells.append((x, y))
            rays[emitter_id] = {"cells": cells, "collision": collision, "direction": direction}
        return rays, hits

    @staticmethod
    def _p(cell: tuple[int, int]) -> tuple[int, int]:
        return ORIGIN + CELL * cell[0], ORIGIN + CELL * cell[1]

    def _draw_ray(self, frame: np.ndarray, emitter_id: str, ray: dict[str, object], fraction: float = 1.0) -> None:
        sx, sy = self._p(self.emitters[emitter_id])
        direction = int(ray["direction"])
        cx, cy = sx + 2, sy + 2
        kind, _, collision_cell = ray["collision"]
        tx, ty = collision_cell
        if direction == E:
            endpoint = ORIGIN + CELL * tx - (1 if kind == "rim" else 0)
            start, end = sx + 2, endpoint
        elif direction == W:
            endpoint = ORIGIN if kind == "rim" else ORIGIN + CELL * tx + CELL - 1
            start, end = endpoint, sx + 2
        elif direction == S:
            endpoint = ORIGIN + CELL * ty - (1 if kind == "rim" else 0)
            start, end = sy + 2, endpoint
        else:
            endpoint = ORIGIN if kind == "rim" else ORIGIN + CELL * ty + CELL - 1
            start, end = endpoint, sy + 2
        if direction in (E, W):
            full = max(0, end - start + 1)
            shown = max(1, int(full * fraction))
            if direction == E:
                frame[cy, start : start + shown] = BEAM
            else:
                frame[cy, end - shown + 1 : end + 1] = BEAM
        else:
            full = max(0, end - start + 1)
            shown = max(1, int(full * fraction))
            if direction == S:
                frame[start : start + shown, cx] = BEAM
            else:
                frame[end - shown + 1 : end + 1, cx] = BEAM

    def _draw_impact_cap(self, frame: np.ndarray, ray: dict[str, object]) -> None:
        kind, _, collision_cell = ray["collision"]
        if kind not in {"wall", "emitter"}:
            return
        direction = int(ray["direction"])
        px, py = self._p(collision_cell)
        if direction == E:
            frame[py + 1 : py + 3, px] = CAP
        elif direction == W:
            frame[py + 1 : py + 3, px + 3] = CAP
        elif direction == S:
            frame[py, px + 1 : px + 3] = CAP
        else:
            frame[py + 3, px + 1 : px + 3] = CAP

    def _render(self, mode: dict[str, object]) -> None:
        frame = np.full((SIZE, SIZE), MARGIN, dtype=np.int8)
        frame[ORIGIN : ORIGIN + BOARD * CELL, ORIGIN : ORIGIN + BOARD * CELL] = FIELD
        kind = str(mode.get("kind", "settled"))
        rays = self.rays
        orientations = self.orientations
        lit_hits = self.hits
        clicked = mode.get("emitter")
        if kind == "turn":
            orientations = list(mode["old_orientations"])
            rays, lit_hits = self._raycast(orientations)
        for emitter_id in sorted(rays):
            fraction = 0.5 if kind == "growth" and emitter_id == clicked else 1.0
            self._draw_ray(frame, emitter_id, rays[emitter_id], fraction)
        if kind == "growth" and clicked is not None:
            old_rays, old_hits = self._raycast(list(mode["old_orientations"]))
            del old_rays
            lit_hits = {key: {source for source in value if source != clicked} for key, value in old_hits.items()}
        for cell in self.walls:
            x, y = self._p(cell)
            frame[y : y + CELL, x : x + CELL] = WALL
        for terminal_id, cell in self.terminals.items():
            x, y = self._p(cell)
            lit = bool(lit_hits[terminal_id])
            if lit:
                halo = PULSE if kind == "pulse" else HALO
                frame[y : y + CELL, x : x + CELL] = halo
                frame[y + 1 : y + 3, x + 1 : x + 3] = TERMINAL
            else:
                frame[y : y + CELL, x : x + CELL] = TERMINAL
                frame[y + 1 : y + 3, x + 1 : x + 3] = FIELD
        north = np.array(
            [
                [FIELD, BEAM, BEAM, FIELD],
                [EMITTER] * 4,
                [FIELD, EMITTER, EMITTER, FIELD],
                [FIELD, EMITTER, EMITTER, FIELD],
            ],
            dtype=np.int8,
        )
        for emitter_id, cell in self.emitters.items():
            direction = orientations[ord(emitter_id) - ord("A")]
            pixels = np.rot90(north, -direction).copy()
            if kind == "turn" and emitter_id == clicked:
                pixels[pixels == EMITTER] = FLASH
            x, y = self._p(cell)
            frame[y : y + CELL, x : x + CELL] = pixels
        for emitter_id, ray in rays.items():
            if kind != "growth" or emitter_id != clicked:
                self._draw_impact_cap(frame, ray)
        if kind == "ripple":
            px, py = mode["pixel"]
            frame[max(0, py - 1) : min(SIZE, py + 2), px] = RIPPLE
            frame[py, max(0, px - 1) : min(SIZE, px + 2)] = RIPPLE
        frame[61:63, 4:60] = CAP
        width = int(56 * self.remaining_steps / max(1, self.spec.budget))
        if width:
            frame[61:63, 4 : 4 + width] = BEAM
        self._board.pixels = frame

    def _queue_and_show(self, frames: list[dict[str, object]], finish: bool = False) -> None:
        self._frames = frames
        self._finish_after_frames = finish
        self._show_next_frame()

    def _show_next_frame(self) -> None:
        mode = self._frames.pop(0)
        self._render(mode)
        if self._frames:
            return
        if self._finish_after_frames:
            self.next_level()
            # Mark the submitted click complete as well.  The engine will still
            # render the deferred next-level arrival, but must not replay that
            # click against the new board.
            self.complete_action()
        elif self.remaining_steps <= 0:
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def step(self) -> None:
        if self._frames:
            self._show_next_frame()
            return
        if self.action.id == GameAction.RESET:
            self._render({"kind": "settled"})
            self.complete_action()
            return

        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.action.id == GameAction.ACTION7:
            if self._history:
                self.orientations = list(self._history.pop())
                self.rays, self.hits = self._raycast(self.orientations)
            self._render({"kind": "settled"})
            if self.remaining_steps <= 0:
                self.lose()
            self.complete_action()
            return

        if self.action.id != GameAction.ACTION6:
            self._render({"kind": "settled"})
            if self.remaining_steps <= 0:
                self.lose()
            self.complete_action()
            return

        px = int(self.action.data.get("x", 0))
        py = int(self.action.data.get("y", 0))
        cell = ((px - ORIGIN) // CELL, (py - ORIGIN) // CELL)
        clicked = next(
            (key for key, value in self.emitters.items() if value == cell and ORIGIN <= px < 60 and ORIGIN <= py < 60),
            None,
        )
        if clicked is None:
            self._queue_and_show([{"kind": "ripple", "pixel": (px, py)}, {"kind": "settled"}])
            return

        old = tuple(self.orientations)
        self._history.append(old)
        idx = ord(clicked) - ord("A")
        self.orientations[idx] = (self.orientations[idx] + 1) % 4
        self.rays, self.hits = self._raycast(self.orientations)
        won = all(bool(value) for value in self.hits.values())
        frames: list[dict[str, object]] = [
            {"kind": "turn", "emitter": clicked, "old_orientations": old},
            {"kind": "growth", "emitter": clicked, "old_orientations": old},
        ]
        if won:
            frames.extend([{"kind": "pulse"}, {"kind": "pulse"}])
        frames.append({"kind": "settled"})
        self._queue_and_show(frames, finish=won)
