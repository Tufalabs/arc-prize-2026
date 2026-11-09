from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

GAME_ID = "forked_seeds-0001"
SIZE = 64
CELL = 6
ORIGIN = 2

WHITE = 0
PLANTING = 1
FAINT = 2
SHADOW = 3
GROUND = 4
OUTLINE = 5
INVALID = 8
EMPTY_FEEDBACK = 10
GOLD = 11
GREEN = 14

DIRECTION_VECTOR = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
BASELINES = (1, 5, 5, 25, 41, 46, 52)


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    explicit_forks: tuple[tuple[int, int, str], ...] = ()
    fork_style: str = ""


LEVEL_SPECS = (
    LevelSpec(
        "First Fork",
        (
            "##########",
            "##########",
            "##########",
            "##########",
            "####ST####",
            "####T#####",
            "##########",
            "##########",
            "##########",
            "##########",
        ),
        ((5, 5, "SE"), (5, 6, "NE"), (6, 5, "SW")),
    ),
    LevelSpec(
        "Forks Have Addresses",
        (
            "##########",
            "##########",
            "###T#T####",
            "##T...T###",
            "####S#####",
            "###T.T####",
            "##########",
            "##########",
            "##########",
            "##########",
        ),
        (
            (5, 5, "NS"),
            (4, 5, "EW"),
            (6, 5, "EW"),
            (4, 4, "NW"),
            (4, 6, "NE"),
            (6, 4, "SW"),
            (6, 6, "SE"),
            (3, 4, "NW"),
            (4, 3, "SW"),
            (3, 6, "NE"),
            (4, 7, "SE"),
        ),
    ),
    LevelSpec(
        "Room for Both",
        (
            "##########",
            "##########",
            "####T#####",
            "####.T####",
            "###SS.T###",
            "###T#.T###",
            "#####T####",
            "##########",
            "##########",
            "##########",
        ),
        (
            (5, 4, "ES"),
            (5, 5, "NE"),
            (4, 5, "NE"),
            (5, 6, "SE"),
            (6, 6, "SE"),
            (6, 4, "SW"),
            (3, 5, "NW"),
            (4, 6, "NE"),
            (5, 7, "NE"),
            (7, 6, "SW"),
            (6, 7, "SE"),
        ),
    ),
    LevelSpec(
        "Borrowed Rings",
        (
            "##TTTT####",
            "##.TTTT###",
            "##.TTTT###",
            "##....T###",
            "##S#######",
            "##...T####",
            "##.TTT####",
            "##.TTT####",
            "##.TTT####",
            "##TTT#####",
        ),
        fork_style="borrowed",
    ),
    LevelSpec(
        "Keep One, Spend One",
        (
            "##TTT#####",
            "##.TT..T##",
            "##.T.TTT##",
            "#T..TBT.T#",
            "#.TTBT..TT",
            "TTTTTTTTT#",
            "TT.TT.T.##",
            "#T..T.T.##",
            "####TTTT##",
            "#####T####",
        ),
        fork_style="clockwise",
    ),
    LevelSpec(
        "Sideways Shelf",
        (
            "#TTTT#####",
            "TTTT.#####",
            "TTTT.#####",
            "TTTT.S...T",
            "T...ST.TTT",
            "T.TTTTTT.#",
            "T.TTTTTTT#",
            "T.TTTT####",
            "T.TTTT####",
            "#####T####",
        ),
        fork_style="shelf",
    ),
    LevelSpec(
        "Canopy Lock",
        (
            "####T#####",
            "####.....T",
            "####.TTTTT",
            "####.TTTTT",
            "#TTTBTTTTT",
            "TTTTTTTT.#",
            "TTTTTTTT.#",
            "TTTTTTTT.#",
            "T....TTT.#",
            "#####TTTT#",
        ),
        fork_style="clockwise",
    ),
)


def _fork_for(spec: LevelSpec, row: int, col: int) -> str:
    if spec.explicit_forks:
        return {(r, c): directions for r, c, directions in spec.explicit_forks}[(row, col)]
    if spec.fork_style == "borrowed":
        if (row, col) == (5, 3):
            return "NS"
        return "NE" if row < 5 else "SE"
    if spec.fork_style == "shelf":
        if row <= 5 and col <= 5:
            return "NW"
        if row <= 5:
            return "SE"
        if col >= 6:
            return "SW"
        return "EW"
    if row <= 5 and col <= 5:
        return "NE"
    if row <= 5:
        return "SE"
    if col >= 6:
        return "SW"
    return "NW"


def _parse_level(spec: LevelSpec) -> dict[str, object]:
    usable: set[tuple[int, int]] = set()
    targets: set[tuple[int, int]] = set()
    seeds: dict[tuple[int, int], int] = {}
    for row, line in enumerate(spec.rows, 1):
        if len(line) != 10:
            raise ValueError(f"{spec.name}: row {row} is not ten cells wide")
        for col, symbol in enumerate(line, 1):
            if symbol == "#":
                continue
            usable.add((row, col))
            if symbol in "TB":
                targets.add((row, col))
            if symbol in "SB":
                seeds[(row, col)] = 0
    forks = {cell: _fork_for(spec, *cell) for cell in usable}
    return {
        "name": spec.name,
        "rows": spec.rows,
        "usable": frozenset(usable),
        "targets": frozenset(targets),
        "forks": forks,
        "seeds": seeds,
    }


class ForkedSeedsDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ForkedSeeds | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        frame[:, :] = GROUND
        if game is None:
            return frame
        self._draw_cells(frame, game)
        self._draw_seeds(frame, game)
        self._draw_feedback(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _draw_cells(self, frame: np.ndarray, game: ForkedSeeds) -> None:
        ring_color = GOLD
        if game.animation and game.animation["kind"] == "pulse":
            ring_color = (GOLD, WHITE, GREEN, GOLD)[int(game.animation["phase"])]
        for row, col in game.usable:
            x, y = _top_left(row, col)
            frame[y : y + CELL, x : x + CELL] = PLANTING
            self._draw_fork(frame, x, y, game.forks[(row, col)], FAINT)
            if (row, col) in game.targets:
                self._draw_ring(frame, x, y, ring_color, corners_only=False)

    def _draw_seeds(self, frame: np.ndarray, game: ForkedSeeds) -> None:
        animation = game.animation
        if animation and animation["kind"] == "split" and int(animation["phase"]) < 3:
            parent = animation["parent"]
            old_seeds = animation["old_seeds"]
            for cell, generation in old_seeds.items():
                if cell != parent:
                    self._draw_seed(frame, game, cell, generation)
            phase = int(animation["phase"])
            if phase == 0:
                self._draw_kernel(frame, parent, 0, 0)
            else:
                distance = 2 if phase == 1 else 4
                for direction in game.forks[parent]:
                    dr, dc = DIRECTION_VECTOR[direction]
                    self._draw_kernel(frame, parent, dc * distance, dr * distance)
            return
        if animation and animation["kind"] == "invalid":
            parent = animation["parent"]
            for cell, generation in game.seeds.items():
                if cell != parent:
                    self._draw_seed(frame, game, cell, generation)
            return
        for cell, generation in game.seeds.items():
            self._draw_seed(frame, game, cell, generation)

    def _draw_seed(
        self, frame: np.ndarray, game: ForkedSeeds, cell: tuple[int, int], generation: int, dx: int = 0, dy: int = 0
    ) -> None:
        row, col = cell
        x, y = _top_left(row, col)
        body = _body_mask(generation, game.forks[cell])
        for u, v in body:
            sx, sy = x + u + 1 + dx, y + v + 1 + dy
            if x <= sx < x + CELL and y <= sy < y + CELL:
                frame[sy, sx] = SHADOW
        for u, v in body:
            sx, sy = x + u + dx, y + v + dy
            if x <= sx < x + CELL and y <= sy < y + CELL:
                frame[sy, sx] = GREEN
        self._draw_arrowheads(frame, x + dx, y + dy, game.forks[cell], WHITE, clip=(x, y))
        if cell in game.targets:
            corner_color = GOLD
            if game.animation and game.animation["kind"] == "pulse":
                corner_color = (GOLD, WHITE, GREEN, GOLD)[int(game.animation["phase"])]
            self._draw_ring(frame, x, y, corner_color, corners_only=True)

    def _draw_kernel(self, frame: np.ndarray, cell: tuple[int, int], dx: int, dy: int) -> None:
        x, y = _top_left(*cell)
        cx, cy = x + 2 + dx, y + 2 + dy
        frame[max(0, cy) : min(SIZE, cy + 2), max(0, cx) : min(SIZE, cx + 2)] = GREEN

    def _draw_feedback(self, frame: np.ndarray, game: ForkedSeeds) -> None:
        animation = game.animation
        if not animation:
            return
        kind = animation["kind"]
        if kind == "empty":
            row, col = animation["cell"]
            x, y = _top_left(row, col)
            frame[y : y + CELL, x : x + CELL] = EMPTY_FEEDBACK
            self._draw_fork(frame, x, y, game.forks[(row, col)], FAINT)
            if (row, col) in game.targets:
                self._draw_ring(frame, x, y, GOLD, corners_only=False)
        elif kind == "invalid":
            parent = animation["parent"]
            phase = int(animation["phase"])
            shake = (1, -1, 1, -1, 1, 0)[phase]
            blocked = animation["blocked"]
            direction = blocked[0][0] if blocked else "E"
            dr, dc = DIRECTION_VECTOR[direction]
            self._draw_seed(frame, game, parent, game.seeds[parent], dx=shake * dc, dy=shake * dr)
            x, y = _top_left(*parent)
            stems = {"N": ((2, 1), (3, 1)), "E": ((4, 2), (4, 3)), "S": ((2, 4), (3, 4)), "W": ((1, 2), (1, 3))}
            for blocked_direction, destination, occupied in blocked:
                for u, v in stems[blocked_direction]:
                    frame[y + v, x + u] = INVALID
                self._draw_arrowheads(frame, x, y, blocked_direction, INVALID, clip=(x, y))
                if occupied:
                    self._draw_rim(frame, destination)
                else:
                    self._draw_ground_patch(frame, parent, blocked_direction)

    def _draw_budget(self, frame: np.ndarray, game: ForkedSeeds) -> None:
        width = int(60 * game.remaining_steps / max(1, game.step_budget))
        if width:
            frame[0:2, 2 : 2 + width] = GREEN
        if game.remaining_steps <= max(1, game.step_budget // 6) and width:
            frame[0:2, 2 : 2 + width] = INVALID

    def _draw_fork(self, frame: np.ndarray, x: int, y: int, directions: str, color: int) -> None:
        for u in (2, 3):
            for v in (2, 3):
                frame[y + v, x + u] = color
        stems = {"N": ((2, 1), (3, 1)), "E": ((4, 2), (4, 3)), "S": ((2, 4), (3, 4)), "W": ((1, 2), (1, 3))}
        for direction in directions:
            for u, v in stems[direction]:
                frame[y + v, x + u] = color
        self._draw_arrowheads(frame, x, y, directions, color)

    def _draw_arrowheads(
        self, frame: np.ndarray, x: int, y: int, directions: str, color: int, clip: tuple[int, int] | None = None
    ) -> None:
        heads = {
            "N": ((2, 0), (3, 0), (1, 1), (4, 1)),
            "E": ((5, 2), (5, 3), (4, 1), (4, 4)),
            "S": ((2, 5), (3, 5), (1, 4), (4, 4)),
            "W": ((0, 2), (0, 3), (1, 1), (1, 4)),
        }
        for direction in directions:
            for u, v in heads[direction]:
                px, py = x + u, y + v
                if clip is not None:
                    left, top = clip
                    if not (left <= px < left + CELL and top <= py < top + CELL):
                        continue
                if 0 <= px < SIZE and 0 <= py < SIZE:
                    frame[py, px] = color

    def _draw_ring(self, frame: np.ndarray, x: int, y: int, color: int, *, corners_only: bool) -> None:
        pixels = [(0, 0), (5, 0), (0, 5), (5, 5)]
        if not corners_only:
            pixels += [(2, 0), (3, 0), (5, 2), (5, 3), (2, 5), (3, 5), (0, 2), (0, 3)]
        for u, v in pixels:
            frame[y + v, x + u] = color

    def _draw_rim(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        x, y = _top_left(*cell)
        frame[y, x : x + CELL] = INVALID
        frame[y + CELL - 1, x : x + CELL] = INVALID
        frame[y : y + CELL, x] = INVALID
        frame[y : y + CELL, x + CELL - 1] = INVALID

    def _draw_ground_patch(self, frame: np.ndarray, cell: tuple[int, int], direction: str) -> None:
        x, y = _top_left(*cell)
        if direction == "N" and y > 0:
            frame[y - 1, x + 2 : x + 4] = INVALID
        elif direction == "S" and y + CELL < SIZE:
            frame[y + CELL, x + 2 : x + 4] = INVALID
        elif direction == "W" and x > 0:
            frame[y + 2 : y + 4, x - 1] = INVALID
        elif direction == "E" and x + CELL < SIZE:
            frame[y + 2 : y + 4, x + CELL] = INVALID


def _top_left(row: int, col: int) -> tuple[int, int]:
    return ORIGIN + CELL * (col - 1), ORIGIN + CELL * (row - 1)


def _body_mask(generation: int, directions: str) -> set[tuple[int, int]]:
    if generation == 0:
        body = {(u, v) for u in range(1, 5) for v in range(1, 5)}
    elif generation == 1:
        body = {(u, v) for u in (2, 3) for v in (1, 2, 3, 4)}
        body |= {(u, v) for u in (1, 2, 3, 4) for v in (2, 3)}
    else:
        body = {(u, v) for u in (2, 3) for v in (2, 3)}
    stems = {"N": {(2, 1), (3, 1)}, "E": {(4, 2), (4, 3)}, "S": {(2, 4), (3, 4)}, "W": {(1, 2), (1, 3)}}
    for direction in directions:
        body |= stems[direction]
    return body


class ForkedSeeds(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        parsed = tuple(_parse_level(spec) for spec in LEVEL_SPECS)
        levels = [
            Level([], grid_size=(SIZE, SIZE), data={"spec": spec, "budget": BASELINES[index] * 6}, name=spec["name"])
            for index, spec in enumerate(parsed)
        ]
        self._display = ForkedSeedsDisplay()
        camera = Camera(0, 0, SIZE, SIZE, GROUND, GROUND, [self._display])
        self.usable: frozenset[tuple[int, int]] = frozenset()
        self.targets: frozenset[tuple[int, int]] = frozenset()
        self.forks: dict[tuple[int, int], str] = {}
        self.seeds: dict[tuple[int, int], int] = {}
        self.animation: dict[str, object] | None = None
        self.feedback: dict[str, object] | None = None
        self.selected_cell: None = None
        self.undo_history: list[dict[tuple[int, int], int]] = []
        self.step_budget = 1
        self.remaining_steps = 1
        self.step_count = 0
        self.done = False
        self._pending_progress = False
        self._pending_loss = False
        super().__init__(GAME_ID, levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6, 7], seed=seed)
        self._display.game = self

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        self.usable = spec["usable"]
        self.targets = spec["targets"]
        self.forks = dict(spec["forks"])
        self.seeds = dict(spec["seeds"])
        self.animation = None
        self.feedback = None
        self.selected_cell = None
        self.undo_history = []
        self.step_budget = int(level.get_data("budget"))
        self.remaining_steps = self.step_budget
        self.step_count = 0
        self.done = self.targets.issubset(self.seeds)
        self._pending_progress = False
        self._pending_loss = False

    def step(self) -> None:
        if self.animation is not None:
            self._advance_animation()
            return
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        self.feedback = None
        self.step_count += 1
        self.remaining_steps = max(0, self.remaining_steps - 1)
        action_id = int(self.action.id.value)
        if action_id == 7:
            if self.undo_history:
                self.seeds = self.undo_history.pop()
            self.done = self.targets.issubset(self.seeds)
            self._finish_or_lose()
            return
        if action_id != 6:
            self._finish_or_lose()
            return

        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        cell = self._cell_at_pixel(x, y)
        if cell is None:
            self._finish_or_lose()
        elif cell not in self.seeds:
            self.feedback = {"kind": "empty", "cell": cell}
            self.animation = {"kind": "empty", "phase": 0, "cell": cell}
            self._pending_loss = self.remaining_steps == 0
        else:
            self._attempt_split(cell)

    def _cell_at_pixel(self, x: int, y: int) -> tuple[int, int] | None:
        if not (ORIGIN <= x < ORIGIN + 10 * CELL and ORIGIN <= y < ORIGIN + 10 * CELL):
            return None
        cell = ((y - ORIGIN) // CELL + 1, (x - ORIGIN) // CELL + 1)
        return cell if cell in self.usable else None

    def _attempt_split(self, parent: tuple[int, int]) -> None:
        destinations: list[tuple[str, tuple[int, int]]] = []
        blocked: list[tuple[str, tuple[int, int], bool]] = []
        for direction in self.forks[parent]:
            dr, dc = DIRECTION_VECTOR[direction]
            destination = (parent[0] + dr, parent[1] + dc)
            destinations.append((direction, destination))
            if destination not in self.usable or destination in self.seeds:
                blocked.append((direction, destination, destination in self.seeds))
        if blocked:
            self.feedback = {
                "kind": "invalid",
                "cell": parent,
                "blocked": tuple(direction for direction, _, _ in blocked),
            }
            self.animation = {"kind": "invalid", "phase": 0, "parent": parent, "blocked": blocked}
            self._pending_loss = self.remaining_steps == 0
            return

        old_seeds = dict(self.seeds)
        self.undo_history.append(old_seeds)
        generation = self.seeds.pop(parent) + 1
        for _, destination in destinations:
            self.seeds[destination] = generation
        self.done = self.targets.issubset(self.seeds)
        self._pending_progress = self.done
        self._pending_loss = self.remaining_steps == 0 and not self._pending_progress
        self.animation = {"kind": "split", "phase": 0, "parent": parent, "old_seeds": old_seeds}

    def _advance_animation(self) -> None:
        if self.animation is None:
            self._finish_or_lose()
            return
        kind = str(self.animation["kind"])
        phase = int(self.animation["phase"])
        last_phase = {"split": 3, "empty": 2, "invalid": 5, "pulse": 3}[kind]
        if phase < last_phase - 1:
            self.animation["phase"] = phase + 1
            return
        if kind == "split" and self._pending_progress:
            self.animation = {"kind": "pulse", "phase": 0}
            return
        if kind == "pulse":
            self.animation = None
            self.next_level()
            self.complete_action()
            return
        self.animation = None
        self._finish_or_lose()

    def _finish_or_lose(self) -> None:
        if self._pending_loss or self.remaining_steps == 0:
            self._pending_loss = False
            self.lose()
        self.complete_action()
