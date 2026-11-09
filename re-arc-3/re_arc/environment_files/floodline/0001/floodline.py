from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

UP, DOWN, LEFT, RIGHT, SPACE, UNDO = 1, 2, 3, 4, 5, 7
MOVES = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))

WHITE = 0
GROUND = 1
ROCK_BEVEL = 3
ROCK = 4
BACKGROUND = 5
AGENT = 6
BAD = 8
WATER = 9
FOAM = 10
BRACKET = 11
ACTIVE = 12
DAM = 13
EXIT = 14
VICTORY = 15


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    cell_size: int
    origin: tuple[int, int]
    budget: int


def _open_chain(grid: list[list[str]], waypoints: tuple[tuple[int, int], ...]) -> None:
    """Open a coarse chain on a 31x31 fine grid."""
    for (ax, ay), (bx, by) in pairwise(waypoints):
        if ax != bx and ay != by:
            raise ValueError("Coarse chain segments must be axis aligned")
        dx = 0 if ax == bx else (1 if bx > ax else -1)
        dy = 0 if ay == by else (1 if by > ay else -1)
        x, y = ax, ay
        while (x, y) != (bx, by):
            nx, ny = x + dx, y + dy
            grid[2 * y + 1][2 * x + 1] = "."
            grid[y + ny + 1][x + nx + 1] = "."
            grid[2 * ny + 1][2 * nx + 1] = "."
            x, y = nx, ny


def _fine_path(waypoints: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for (ax, ay), (bx, by) in pairwise(waypoints):
        x, y = 2 * ax + 1, 2 * ay + 1
        tx, ty = 2 * bx + 1, 2 * by + 1
        dx = 0 if x == tx else (1 if tx > x else -1)
        dy = 0 if y == ty else (1 if ty > y else -1)
        if not out:
            out.append((x, y))
        while (x, y) != (tx, ty):
            x, y = x + dx, y + dy
            out.append((x, y))
    return out


def _late_rows(
    chains: tuple[tuple[tuple[int, int], ...], ...],
    *,
    player: tuple[int, int],
    exit_cell: tuple[int, int],
    sites: tuple[tuple[int, int], ...],
    water_prefixes: tuple[tuple[tuple[tuple[int, int], ...], int], ...],
    water_cells: tuple[tuple[int, int], ...] = (),
) -> tuple[str, ...]:
    """Build a readable ASCII logical grid from the normative sparse graph."""
    grid = [["#" for _ in range(31)] for _ in range(31)]
    for chain in chains:
        _open_chain(grid, chain)
    for path, distance_to_target in water_prefixes:
        fine = _fine_path(path)
        end = max(0, len(fine) - 1 - distance_to_target)
        for x, y in fine[: end + 1]:
            grid[y][x] = "~"
    for x, y in water_cells:
        grid[y][x] = "~"
    for x, y in sites:
        grid[y][x] = "b"
    grid[exit_cell[1]][exit_cell[0]] = "E"
    grid[player[1]][player[0]] = "A"
    return tuple("".join(row) for row in grid)


L1 = (
    "###############",
    "###############",
    "##~..........##",
    "##~......A..E##",
    "##~..........##",
    "##~..........##",
    "###############",
    "###############",
    "###############",
)
L2 = (
    "############",
    "############",
    "############",
    "##~~########",
    "##~~bA.#####",
    "####.#.#####",
    "####.#.#####",
    "####E#.#####",
    "####...#####",
    "############",
    "############",
    "############",
)
L3 = (
    "############",
    "############",
    "############",
    "#####E..####",
    "#####.#.####",
    "###~~bA.####",
    "############",
    "############",
    "############",
    "############",
    "############",
    "############",
)
L4 = (
    "##############",
    "#............#",
    "#.##########.#",
    "#.##########.#",
    "#.##########.#",
    "#A##########.#",
    "#b......E....#",
    "#~##########.#",
    "#.##########.#",
    "#.##########.#",
    "#.##########.#",
    "#.##########.#",
    "#............#",
    "##############",
)

L5_PLAYER = (
    ((0, 2), (5, 2)),
    (
        (5, 2),
        (6, 2),
        (6, 3),
        (7, 3),
        (7, 4),
        (8, 4),
        (8, 3),
        (10, 3),
        (10, 5),
        (11, 5),
        (11, 7),
        (9, 7),
        (8, 7),
        (8, 9),
        (5, 9),
    ),
)
L5_BETA = ((2, 8), (5, 8), (5, 7), (6, 7), (6, 6), (7, 6), (7, 4), (8, 4))
L5_GAMMA = ((14, 14), (11, 14), (11, 12), (14, 12), (14, 9), (12, 9), (12, 10), (9, 10), (9, 7))
L5 = _late_rows(
    (*L5_PLAYER, L5_BETA, L5_GAMMA),
    player=(5, 5),
    exit_cell=(11, 19),
    sites=((10, 5), (16, 9), (18, 15)),
    # B is the penultimate fine cell: 19 steps to the far node means 18 to B.
    water_prefixes=((L5_BETA, 19),),
    water_cells=((1, 5), (29, 29)),
)

L6_MAIN = (
    (5, 5),
    (5, 7),
    (2, 7),
    (2, 9),
    (3, 9),
    (3, 8),
    (4, 8),
    (4, 6),
    (8, 6),
    (8, 9),
    (11, 9),
    (12, 9),
    (12, 10),
    (10, 10),
    (10, 12),
    (12, 12),
)
L6_STEM = ((4, 14), (4, 12), (1, 12), (1, 10), (2, 10), (2, 9))
L6_LEFT = ((2, 9), (1, 9), (1, 8), (1, 7), (2, 7))
L6_RIGHT = ((2, 9), (3, 9), (3, 8), (3, 7), (2, 7))
L6_REAR = (
    (14, 0),
    (9, 0),
    (9, 2),
    (14, 2),
    (14, 5),
    (12, 5),
    (12, 7),
    (14, 7),
    (14, 11),
    (13, 11),
    (13, 8),
    (11, 8),
    (11, 9),
    (12, 9),
)
L6 = _late_rows(
    (L6_MAIN, L6_STEM, L6_LEFT, L6_RIGHT, L6_REAR),
    player=(11, 11),
    exit_cell=(25, 25),
    sites=((5, 20), (3, 18), (7, 18), (24, 19)),
    # Both target sites precede their path's final far-side node by one cell.
    water_prefixes=((L6_STEM, 18), (L6_REAR, 49)),
)

L7_MAIN = (
    (2, 12),
    (2, 10),
    (3, 10),
    (5, 10),
    (5, 7),
    (7, 7),
    (7, 8),
    (8, 8),
    (10, 8),
    (10, 10),
    (11, 10),
    (11, 9),
    (13, 9),
    (13, 12),
    (10, 12),
    (10, 10),
    (7, 10),
    (7, 6),
    (10, 6),
)
L7_WEST = ((3, 4), (1, 4), (1, 8), (3, 8), (3, 9), (3, 10))
L7_WL = ((3, 10), (2, 10), (2, 9), (1, 9), (1, 11), (5, 11), (5, 9))
L7_WR = ((3, 10), (4, 10), (4, 9), (5, 9), (5, 10))
# The pressure corridors use unused coarse rows so they meet the route only at
# their named chokepoints. This keeps the visible graph faithful to the dam
# lesson instead of creating accidental crossings with the return loop.
L7_CENTRAL = ((0, 0), (4, 0), (4, 8), (6, 8), (7, 8), (8, 8))
L7_EAST = ((14, 14), (8, 14), (8, 13), (14, 13), (14, 8), (11, 8))
L7 = _late_rows(
    (L7_MAIN, L7_WEST, L7_WL, L7_WR, L7_CENTRAL, L7_EAST),
    player=(5, 25),
    exit_cell=(21, 13),
    sites=((7, 20), (5, 20), (9, 20), (16, 17), (23, 18)),
    # W/G precede their final far-side nodes; E lies one step beyond its
    # water-side endpoint.
    water_prefixes=((L7_WEST, 10), (L7_EAST, 39)),
)

LEVEL_SPECS = (
    LevelSpec("Three Dry Steps", L1, 4, (2, 14), 18),
    LevelSpec("The Plug", L2, 5, (2, 2), 54),
    LevelSpec("The Wall You Made", L3, 5, (2, 2), 36),
    LevelSpec("Around the World", L4, 4, (4, 4), 150),
    LevelSpec("Seal, Work, Seal", L5, 2, (1, 1), 306),
    LevelSpec("One Before the Fork", L6, 2, (1, 1), 372),
    LevelSpec("The Floodline", L7, 2, (1, 1), 474),
)


class FloodlineHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: Floodline | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.game is None:
            return frame
        game = self.game
        frame[63, :] = BRACKET
        filled = int(64 * max(0, game.remaining_steps) / max(1, game.step_budget))
        frame[63, :filled] = ACTIVE
        if game.remaining_steps <= max(1, game.step_budget // 6):
            frame[63, :filled] = BAD
        return frame


class Floodline(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = FloodlineHud()
        levels = []
        for spec in LEVEL_SPECS:
            board = Sprite(
                np.full((64, 64), -1, dtype=np.int8), name="board", x=0, y=0, layer=0, collidable=False, tags=["board"]
            )
            levels.append(Level([board], grid_size=(64, 64), data={"spec": spec}, name=spec.name))
        camera = Camera(0, 0, 64, 64, BACKGROUND, BACKGROUND, [self._hud])
        super().__init__("floodline-0001", levels, camera, False, 7, [1, 2, 3, 4, 5, 7], seed=seed)
        self._hud.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.board = level.get_sprites_by_tag("board")[0]
        self.rocks: set[tuple[int, int]] = set()
        self.ground: set[tuple[int, int]] = set()
        self.water: set[tuple[int, int]] = set()
        self.sites: set[tuple[int, int]] = set()
        self.dams: set[tuple[int, int]] = set()
        self.exit_cell = (0, 0)
        self.player = (0, 0)
        for y, row in enumerate(self.spec.rows):
            for x, char in enumerate(row):
                cell = (x, y)
                if char == "#":
                    self.rocks.add(cell)
                else:
                    self.ground.add(cell)
                if char == "~":
                    self.water.add(cell)
                elif char == "b":
                    self.sites.add(cell)
                elif char == "A":
                    self.player = cell
                elif char == "E":
                    self.exit_cell = cell
        self.step_count = 0
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self.status = "RUNNING"
        self._history: list[tuple[tuple[int, int], frozenset[tuple[int, int]], frozenset[tuple[int, int]], str]] = []
        self._pending_water: set[tuple[int, int]] | None = None
        self._phase = 0
        self._feedback: tuple[str, tuple[int, int] | None] | None = None
        self._terminal = ""
        self._sync_board()

    def _inside(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= y < len(self.spec.rows) and 0 <= x < len(self.spec.rows[0])

    def _eligible_sites(self) -> list[tuple[int, int]]:
        px, py = self.player
        return [
            site
            for site in self.sites
            if site not in self.dams and site not in self.water and abs(site[0] - px) + abs(site[1] - py) == 1
        ]

    def _expanded_water(self) -> set[tuple[int, int]]:
        expanded = set(self.water)
        for x, y in self.water:
            for dx, dy in DIRS:
                cell = (x + dx, y + dy)
                if self._inside(cell) and cell not in self.rocks and cell not in self.dams:
                    expanded.add(cell)
        return expanded

    def _snapshot(self) -> tuple[tuple[int, int], frozenset[tuple[int, int]], frozenset[tuple[int, int]], str]:
        return self.player, frozenset(self.water), frozenset(self.dams), self.status

    def _resolve_input(self) -> None:
        action_id = self.action.id.value
        self._feedback = None
        self._history.append(self._snapshot())
        if action_id in MOVES:
            dx, dy = MOVES[action_id]
            target = (self.player[0] + dx, self.player[1] + dy)
            if (
                self._inside(target)
                and target not in self.rocks
                and target not in self.water
                and target not in self.dams
            ):
                self.player = target
            else:
                self._feedback = ("blocked", (dx, dy))
        elif action_id == SPACE:
            eligible = self._eligible_sites()
            if len(eligible) == 1:
                self.dams.add(eligible[0])
            else:
                self._feedback = ("invalid", None)
        self.step_count += 1
        self.remaining_steps = self.step_budget - self.step_count
        self._pending_water = self._expanded_water()
        self._phase = 1
        self._sync_board()

    def _resolve_undo(self) -> None:
        self._feedback = None
        self._pending_water = None
        self._phase = 0
        if self._history:
            self.player, old_water, old_dams, self.status = self._history.pop()
            self.water = set(old_water)
            self.dams = set(old_dams)
        self.step_count += 1
        self.remaining_steps = self.step_budget - self.step_count
        self._sync_board()
        if self.remaining_steps <= 0:
            self.status = "LOST"
            self.lose()
        self.complete_action()

    def _advance_animation(self) -> None:
        if self._phase == 1:
            self._phase = 2
            self._feedback = None
            self._sync_board()
            return
        if self._phase == 2:
            assert self._pending_water is not None
            self.water = self._pending_water
            self._pending_water = None
            self._phase = 3
            if self.player in self.water:
                self.status = "LOST"
                self._terminal = "loss"
            elif self.player == self.exit_cell:
                self.status = "WON"
                self._terminal = "victory"
            elif self.remaining_steps <= 0:
                self.status = "LOST"
                self._terminal = "loss"
            self._sync_board()
            if not self._terminal:
                self._phase = 0
                self.complete_action()
            return
        # One brief terminal feedback frame has already rendered.
        self._phase = 0
        terminal = self._terminal
        self._terminal = ""
        self._sync_board()
        if terminal == "victory":
            self.next_level()
            self.complete_action()
        else:
            self.lose()
            self.complete_action()

    def step(self) -> None:
        if self.action.id == GameAction.RESET:
            self._sync_board()
            self.complete_action()
            return
        if self._phase:
            self._advance_animation()
            return
        if self.action.id == GameAction.ACTION7:
            self._resolve_undo()
            return
        if self.action.id.value not in (UP, DOWN, LEFT, RIGHT, SPACE):
            self.complete_action()
            return
        self._resolve_input()

    def _paint_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x, y = cell
        size = self.spec.cell_size
        ox, oy = self.spec.origin
        frame[oy + y * size : oy + (y + 1) * size, ox + x * size : ox + (x + 1) * size] = color

    def _bounds(self, cell: tuple[int, int]) -> tuple[int, int, int, int]:
        x, y = cell
        size = self.spec.cell_size
        ox, oy = self.spec.origin
        return ox + x * size, oy + y * size, ox + (x + 1) * size, oy + (y + 1) * size

    def _draw_rock_bevels(self, frame: np.ndarray) -> None:
        for x, y in self.rocks:
            x0, y0, x1, y1 = self._bounds((x, y))
            if (x - 1, y) not in self.rocks:
                frame[y0:y1, x0] = ROCK_BEVEL
            if (x + 1, y) not in self.rocks:
                frame[y0:y1, x1 - 1] = ROCK_BEVEL
            if (x, y - 1) not in self.rocks:
                frame[y0, x0:x1] = ROCK_BEVEL
            if (x, y + 1) not in self.rocks:
                frame[y1 - 1, x0:x1] = ROCK_BEVEL

    def _draw_exit(self, frame: np.ndarray) -> None:
        x0, y0, x1, y1 = self._bounds(self.exit_cell)
        if self.exit_cell in self.water:
            frame[y0, x0] = EXIT
            frame[y1 - 1, x1 - 1] = EXIT
            return
        color = VICTORY if self._terminal == "victory" and self._phase == 3 else EXIT
        frame[y0:y1, x0:x1] = color
        if x1 - x0 >= 4:
            frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = WHITE
        else:
            frame[y0, x0] = WHITE

    def _draw_sites(self, frame: np.ndarray) -> None:
        eligible = set(self._eligible_sites())
        for cell in self.sites:
            if cell in self.dams or cell in self.water:
                continue
            x0, y0, x1, y1 = self._bounds(cell)
            color = ACTIVE if cell in eligible and len(eligible) == 1 else BRACKET
            if self.spec.cell_size == 2:
                frame[y0, x0] = color
                frame[y1 - 1, x1 - 1] = color
            else:
                frame[y0, x0 : x0 + 2] = color
                frame[y0 : y0 + 2, x0] = color
                frame[y0, x1 - 2 : x1] = color
                frame[y0 : y0 + 2, x1 - 1] = color
                frame[y1 - 1, x0 : x0 + 2] = color
                frame[y1 - 2 : y1, x0] = color
                frame[y1 - 1, x1 - 2 : x1] = color
                frame[y1 - 2 : y1, x1 - 1] = color

    def _draw_dams(self, frame: np.ndarray) -> None:
        for cell in self.dams:
            x0, y0, x1, y1 = self._bounds(cell)
            frame[y0:y1, x0:x1] = DAM
            mid = x0 + (x1 - x0) // 2
            frame[y0:y1, mid] = ACTIVE

    def _draw_water(self, frame: np.ndarray) -> None:
        for cell in self.water:
            self._paint_cell(frame, cell, WATER)
        for x, y in self.water:
            x0, y0, x1, y1 = self._bounds((x, y))
            if (x - 1, y) not in self.water:
                frame[y0:y1, x0] = FOAM
            if (x + 1, y) not in self.water:
                frame[y0:y1, x1 - 1] = FOAM
            if (x, y - 1) not in self.water:
                frame[y0, x0:x1] = FOAM
            if (x, y + 1) not in self.water:
                frame[y1 - 1, x0:x1] = FOAM
        if self._phase == 2 and self._pending_water is not None:
            new_cells = self._pending_water - self.water
            depth = max(1, self.spec.cell_size // 2)
            for x, y in new_cells:
                x0, y0, x1, y1 = self._bounds((x, y))
                if (x - 1, y) in self.water:
                    frame[y0:y1, x0 : x0 + depth] = FOAM
                if (x + 1, y) in self.water:
                    frame[y0:y1, x1 - depth : x1] = FOAM
                if (x, y - 1) in self.water:
                    frame[y0 : y0 + depth, x0:x1] = FOAM
                if (x, y + 1) in self.water:
                    frame[y1 - depth : y1, x0:x1] = FOAM

    def _draw_player(self, frame: np.ndarray) -> None:
        if self.player in self.water and not (self._terminal == "loss" and self._phase == 3):
            return
        x0, y0, x1, y1 = self._bounds(self.player)
        if self._feedback and self._feedback[0] == "blocked" and self._phase == 1:
            dx, dy = self._feedback[1] or (0, 0)
            x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
        color = BAD if self._terminal == "loss" and self._phase == 3 else AGENT
        inset = 0 if self.spec.cell_size == 2 else 1
        frame[y0 + inset : y1 - inset, x0 + inset : x1 - inset] = color
        frame[y0 + inset, x0 + inset] = WHITE
        if self._terminal == "loss" and self._phase == 3:
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if y0 > 0:
                frame[y0 - 1, cx] = FOAM
            if y1 < 64:
                frame[y1, cx] = FOAM
            if x0 > 0:
                frame[cy, x0 - 1] = FOAM
            if x1 < 64:
                frame[cy, x1] = FOAM

    def _draw_feedback(self, frame: np.ndarray) -> None:
        if not self._feedback or self._phase != 1:
            return
        kind, direction = self._feedback
        x0, y0, x1, y1 = self._bounds(self.player)
        if kind == "blocked":
            dx, dy = direction or (0, 0)
            if dx < 0:
                frame[y0:y1, x0] = BAD
            elif dx > 0:
                frame[y0:y1, x1 - 1] = BAD
            elif dy < 0:
                frame[y0, x0:x1] = BAD
            else:
                frame[y1 - 1, x0:x1] = BAD
            return
        targets = [self.player]
        px, py = self.player
        targets.extend(cell for cell in self.sites if abs(cell[0] - px) + abs(cell[1] - py) == 1)
        for cell in targets:
            bx0, by0, bx1, by1 = self._bounds(cell)
            for offset in range(min(bx1 - bx0, by1 - by0)):
                frame[by0 + offset, bx0 + offset] = BAD
                frame[by0 + offset, bx1 - 1 - offset] = BAD

    def _sync_board(self) -> None:
        frame = np.full((64, 64), -1, dtype=np.int8)
        for cell in self.rocks:
            self._paint_cell(frame, cell, ROCK)
        for cell in self.ground:
            self._paint_cell(frame, cell, GROUND)
        self._draw_rock_bevels(frame)
        self._draw_exit(frame)
        self._draw_sites(frame)
        self._draw_dams(frame)
        self._draw_water(frame)
        self._draw_player(frame)
        self._draw_feedback(frame)
        self.board.pixels = frame
