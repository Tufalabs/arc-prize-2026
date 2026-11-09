"""jc26 — a click-only maze/turret puzzle (see spec.md).

Self-contained: the environment loader execs this file with no package context,
so everything lives here and there are no sibling imports. The pure pathfinding
+ round simulation helpers take plain data and avoid arcengine so they can be
unit-tested directly (see test_round.py).
"""

from __future__ import annotations

import json
import math
import pathlib
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

# --- geometry / board ---
TILE = 3
VIEW = 64
GRID = 21  # 21x21 tiles -> 63x63 px; row 63 is a free UI strip, drawn black

# --- palette indices (re_arc COLOR_MAP) ---
C_LOW = 5  # low ground (black) — enemies traverse
C_HIGH = 4  # high ground (dark grey) — turret-placeable
C_HIGHEST = 3  # highest ground (one shade lighter) — wall for movers, no turrets
C_BLOCK = 15  # togglable blocker (purple)
C_ENEMY = 10  # enemy body (light blue)
C_TARGET = 14  # goal the enemies walk to (green)
C_PATH = 9  # planned path line (blue — no dark blue in the palette)
C_TURRET = 13  # turret body (dark red)
C_RANGE = 13  # turret range ring (dark red)
C_LASER = 8  # turret laser / barrel (red)
C_SELECT = 0  # selected-turret outline (white)
C_FLASH = 8  # full-screen red flash on loss
C_UI = 5  # bottom UI strip
C_ENERGY = 0  # energy-bar fill (white)
C_ENERGY_TRACK = 4  # depleted energy (dark grey)

DEFAULT_LIVES = 27  # lives per mover; per-level via the "lives" key. One shot removes 1.
TURRET_RADIUS = 1.8  # tiles; Euclidean
ENERGY_MAX = 256  # mandatory time bar in the bottom UI row; drains 1 per action, 0 = loss (generous on purpose)
# Loss flash: red on/off, ~0.2 s each at the player's ~20 fps, repeated 3x.
FLASH_REPEATS = 3
FLASH_ON_FRAMES = 4
FLASH_OFF_FRAMES = 4
_INF = float("inf")

# 4-neighbour order: up, down, left, right. Fixes the deterministic shortest-path
# tie-break (and BFS parent order for the partial-path display).
NEIGH = ((0, -1), (0, 1), (-1, 0), (1, 0))

# Spiral over the 3x3 body, outer ring clockwise from the top-left, centre last.
# Body pixels are dropped from the front as the mover loses lives (centre last).
SPIRAL = ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (1, 1))
BODY_PIXELS = len(SPIRAL)  # 9


def pixels_removed(hp, lives):
    """How many of the 9 body pixels have vanished at `hp` of `lives` remaining,
    spread evenly so any per-level life count maps onto the 9 pixels."""
    return BODY_PIXELS * (lives - hp) // lives


# ---------------------------------------------------------------------------
# Pure helpers (no arcengine)
# ---------------------------------------------------------------------------
def _line(x0, y0, x1, y1):
    """4-connected raster line (one axis per step) so diagonals read solid."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx - dy
    pts = []
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            return pts
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        elif e2 < dx:
            err += dx
            y += sy


def dist_field(target, open_tiles):
    """BFS distance-to-target over the open-tile set. Order-independent."""
    dist = {target: 0}
    q = deque([target])
    while q:
        cx, cy = q.popleft()
        for dx, dy in NEIGH:
            nb = (cx + dx, cy + dy)
            if nb in open_tiles and nb not in dist:
                dist[nb] = dist[(cx, cy)] + 1
                q.append(nb)
    return dist


def greedy_path(spawn, target, dist, open_tiles):
    """Deterministic shortest path spawn->target by descending `dist`, breaking
    ties in NEIGH order. None if the spawn can't reach the target."""
    if dist.get(spawn, _INF) == _INF:
        return None
    path = [spawn]
    cur = spawn
    while cur != target:
        best, bestd = None, dist[cur]
        for dx, dy in NEIGH:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb in open_tiles and dist.get(nb, _INF) < bestd:
                best, bestd = nb, dist[nb]
        if best is None:
            return None
        path.append(best)
        cur = best
        if len(path) > GRID * GRID:
            return None
    return path


def reachable_path(spawn, target, open_tiles):
    """When the spawn can't reach the target: a path from the spawn to the
    reachable tile closest to the target (Manhattan), for the display line."""
    seen = {spawn: None}
    q = deque([spawn])
    while q:
        cx, cy = q.popleft()
        for dx, dy in NEIGH:
            nb = (cx + dx, cy + dy)
            if nb in open_tiles and nb not in seen:
                seen[nb] = (cx, cy)
                q.append(nb)
    best = min(seen, key=lambda t: (abs(t[0] - target[0]) + abs(t[1] - target[1]), t[1], t[0]))
    path = []
    c = best
    while c is not None:
        path.append(c)
        c = seen[c]
    return list(reversed(path))


def _pixel_track(path):
    """Per-frame centre-pixel positions for a tile path: movers advance one pixel
    per frame, so each tile-to-tile hop (centres TILE px apart) is TILE frames."""
    if not path:
        return [(0, 0)]
    track = [(path[0][0] * TILE + 1, path[0][1] * TILE + 1)]
    for a, b in pairwise(path):
        ax, ay = a[0] * TILE + 1, a[1] * TILE + 1
        bx, by = b[0] * TILE + 1, b[1] * TILE + 1
        for s in range(1, TILE + 1):
            track.append((ax + (bx - ax) * s // TILE, ay + (by - ay) * s // TILE))
    return track


def simulate_round(spawns, paths, turrets, *, hp=DEFAULT_LIVES, radius=TURRET_RADIUS):
    """Pure round simulation in pixel space. Each frame: every turret fires one
    shot (never missing) at the in-range mover closest to the target, then every
    survivor advances ONE pixel along its path. Returns (frames, outcome) where
    outcome is 'win' (all movers destroyed) or 'loss' (a mover reached the
    target). Each frame is a dict {'pos': {i: centre_px}, 'hp': {i: hp},
    'alive': [i...], 'lasers': [(turret_centre_px, mover_centre_px)]}.
    Range is Euclidean <= radius tiles, measured between tile/pixel centres.
    """
    n = len(spawns)
    tracks = {i: _pixel_track(paths[i]) for i in range(n)}
    prog = {i: 0 for i in range(n)}  # pixel index along the track
    life = {i: hp for i in range(n)}
    alive = list(range(n))
    radius_px = radius * TILE
    tcentres = [(t[0] * TILE + 1, t[1] * TILE + 1) for t in turrets]
    frames = []

    def cur(i):
        return tracks[i][prog[i]]

    def remaining(i):
        return len(tracks[i]) - 1 - prog[i]  # pixels left to the target -> "closeness"

    while True:
        lasers = []
        for tc in tcentres:
            in_range = [i for i in alive if math.hypot(tc[0] - cur(i)[0], tc[1] - cur(i)[1]) <= radius_px + 1e-9]
            if not in_range:
                continue
            victim = min(in_range, key=lambda i: (remaining(i), i))
            life[victim] -= 1
            lasers.append((tc, cur(victim)))
        alive = [i for i in alive if life[i] > 0]
        frames.append({"pos": {i: cur(i) for i in range(n)}, "hp": dict(life), "alive": list(alive), "lasers": lasers})
        if not alive:
            return frames, "win"
        arrived = False
        for i in alive:
            if prog[i] < len(tracks[i]) - 1:
                prog[i] += 1
            if remaining(i) == 0 and len(tracks[i]) > 1:
                arrived = True
        if arrived:
            frames.append({"pos": {i: cur(i) for i in range(n)}, "hp": dict(life), "alive": list(alive), "lasers": []})
            return frames, "loss"


# ---------------------------------------------------------------------------
# Levels (ASCII).  Legend per tile:
#   '.' low ground   '#' high ground   'H' highest ground   'E' enemy spawn
#   'X' target   'G' fixed turret   'M' movable turret   (turrets on high ground)
#   'o' blocker slot starting ON   '*' blocker slot starting OPEN
# Only slots and movable turrets can be acted on. Highest ground walls movers
# (like high ground) but can't hold a turret.
# Terrain under E/X/o/* is low ground; under G/M is high ground.
# A per-level "lives" key sets each mover's life count (default 27).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple
    lives: int = DEFAULT_LIVES
    meta: dict = field(default_factory=dict)


def _data_dir():
    # The env loader execs this file with no __file__, so locate our own folder
    # via the re_arc package (snapshot-safe for SLURM/deploy copies).
    import re_arc

    return pathlib.Path(re_arc.__file__).resolve().parent / "environment_files" / "jc26" / "0001"


def _load_level_specs(path=None):
    path = pathlib.Path(path) if path is not None else (_data_dir() / "levels.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        LevelSpec(
            name=lv["name"],
            rows=tuple(lv["grid"]),
            lives=int(lv.get("lives", DEFAULT_LIVES)),
            meta=dict(lv.get("meta", {})),
        )
        for lv in data
    ]


_HIGH_CHARS = {"#", "G", "M"}


def _parse_level(spec):
    rows = [(r + "." * GRID)[:GRID] for r in list(spec.rows)[:GRID]]
    while len(rows) < GRID:
        rows.append("." * GRID)
    high, highest, spawns, turrets, slots, blockers = set(), set(), [], [], set(), set()
    target = None
    for ty, row in enumerate(rows):
        for tx, ch in enumerate(row):
            if ch in _HIGH_CHARS:
                high.add((tx, ty))
            elif ch == "H":
                highest.add((tx, ty))
            if ch == "E":
                spawns.append((tx, ty))
            elif ch == "X":
                target = (tx, ty)
            elif ch == "G":
                turrets.append({"tile": (tx, ty), "movable": False})
            elif ch == "M":
                turrets.append({"tile": (tx, ty), "movable": True})
            elif ch == "o":  # designated slot, starting blocked
                slots.add((tx, ty))
                blockers.add((tx, ty))
            elif ch == "*":  # designated slot, starting open
                slots.add((tx, ty))
    if target is None:
        raise ValueError(f"Level {spec.name!r} needs a target 'X'.")
    if not spawns:
        raise ValueError(f"Level {spec.name!r} needs at least one enemy 'E'.")
    walls = high | highest
    low = {(tx, ty) for ty in range(GRID) for tx in range(GRID) if (tx, ty) not in walls}
    return {
        "high": high,
        "highest": highest,
        "low": low,
        "spawns": spawns,
        "target": target,
        "turrets": turrets,
        "slots": slots,
        "blockers": blockers,
        "lives": spec.lives,
    }


def _build_level(spec: LevelSpec) -> Level:
    parsed = _parse_level(spec)
    board = Sprite(np.full((VIEW, VIEW), C_UI, dtype=np.int8), name="screen", layer=0, collidable=False)
    return Level(grid_size=(VIEW, VIEW), sprites=[board], name=spec.name, data={"parsed": parsed})


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class JC26(ARCBaseGame):
    def __init__(self, seed: int = 0, specs=None) -> None:
        if specs is None:
            levels = []
            for s in _load_level_specs():
                try:
                    levels.append(_build_level(s))
                except Exception as exc:
                    import warnings

                    warnings.warn(f"jc26: skipping level {getattr(s, 'name', '?')!r}: {exc}", stacklevel=2)
            if not levels:
                levels = [
                    _build_level(
                        LevelSpec(
                            name="(no levels)",
                            rows=("#" * GRID,) + ("#" + "." * (GRID - 2) + "#",) * (GRID - 2) + ("#" * GRID,),
                        )
                    )
                ]
        else:
            levels = [_build_level(s) for s in specs]
        super().__init__(
            "jc26",
            levels,
            Camera(0, 0, VIEW, VIEW, C_UI, C_UI),
            False,
            len(levels),
            [5, 6],  # 5 = SPACE (start the round), 6 = click (toggle gate / move turret)
            seed,
        )

    # -- level setup -------------------------------------------------------
    def on_set_level(self, _level: Level) -> None:
        p = self.current_level.get_data("parsed")
        self.high = set(p["high"])
        self.highest = set(p["highest"])
        self.low = set(p["low"])
        self.spawns = list(p["spawns"])
        self.target = p["target"]
        self.turrets = [dict(t) for t in p["turrets"]]  # {'tile', 'movable'}
        self.slots = set(p["slots"])  # the only tiles a blocker may occupy
        self.blockers = set(p["blockers"])
        self.lives = p["lives"]
        self.selected = None
        self._range_turret = None  # turret whose range ring shows (just placed); cleared each action
        self._show_all_ranges = False  # editor sets True to show every turret's range
        self.screen = self.current_level.get_sprites_by_name("screen")[0]
        self._frames_queue = None
        self._pending_next_level = False
        self._pending_lose = False
        self.energy = ENERGY_MAX
        self.screen.pixels = self._compose_plan()

    # -- model helpers -----------------------------------------------------
    def _open_tiles(self):
        return self.low - self.blockers

    def _turret_at(self, tile):
        for i, t in enumerate(self.turrets):
            if t["tile"] == tile:
                return i
        return None

    def _paths(self):
        """Per-enemy movement path (None if blocked) under the current blockers."""
        open_tiles = self._open_tiles()
        dist = dist_field(self.target, open_tiles)
        return {i: greedy_path(sp, self.target, dist, open_tiles) for i, sp in enumerate(self.spawns)}

    # -- action processing -------------------------------------------------
    def step(self) -> None:
        if self._frames_queue is None:
            self._frames_queue = self._process_action()
        self.screen.pixels = self._frames_queue.pop(0)
        if not self._frames_queue:
            self._frames_queue = None
            if self._pending_next_level:
                self._pending_next_level = False
                self.next_level()
            elif self._pending_lose:
                self._pending_lose = False
                self.lose()
            self.complete_action()

    def _process_action(self):
        action = int(getattr(self.action.id, "value", self.action.id))
        self._range_turret = None  # range ring only persists for the action that places a turret
        if action == GameAction.RESET.value:
            return [self._compose_plan()]
        self.energy = max(0, self.energy - 1)  # every action costs 1 energy
        if action == 5:  # SPACE — try to start the round
            frames = self._start_round()
        else:  # action == 6: a click toggles a gate / selects-or-moves a turret (no round)
            data = self.action.data or {}
            x, y = int(data.get("x", -1)), int(data.get("y", -1))
            tx, ty = x // TILE, y // TILE
            if 0 <= tx < GRID and 0 <= ty < GRID:
                self._handle_click((tx, ty))
            frames = [self._compose_plan()]
        if self.energy <= 0 and not self._pending_next_level:  # ran out of time -> loss
            self._pending_lose = True
        return frames

    def _start_round(self):
        paths = self._paths()
        stuck = [i for i, p in paths.items() if p is None]
        if stuck:  # can't start — flash the stuck movers and their lines red
            return self._stuck_flash_frames(set(stuck))
        frames, outcome = self._run_round(paths)
        if outcome == "win":
            self._pending_next_level = True
            return frames
        frames += self._flash_frames(frames[-1])  # loss: a mover reached the target
        frames.append(self._compose_plan())  # state is unchanged — back to planning
        return frames

    def _flash_frames(self, base):
        red = np.full((VIEW, VIEW), C_FLASH, dtype=np.int8)
        seq = []
        for _ in range(FLASH_REPEATS):
            seq += [red.copy() for _ in range(FLASH_ON_FRAMES)]
            seq += [base.copy() for _ in range(FLASH_OFF_FRAMES)]
        return seq

    def _stuck_flash_frames(self, stuck):
        # the stuck movers (no route) and their best-effort lines blink red 3x
        on = self._compose_plan(flash=stuck)
        off = self._compose_plan()
        seq = []
        for _ in range(FLASH_REPEATS):
            seq += [on.copy() for _ in range(FLASH_ON_FRAMES)]
            seq += [off.copy() for _ in range(FLASH_OFF_FRAMES)]
        return seq

    def _handle_click(self, tile):
        ti = self._turret_at(tile)
        occupied = {t["tile"] for t in self.turrets}
        if self.selected is not None:
            if ti == self.selected:
                self.selected = None  # click the held turret -> drop it
            elif ti is not None and self.turrets[ti]["movable"]:
                self.selected = ti  # click another movable turret -> switch
            elif ti is None and tile in self.high and tile not in occupied:
                self.turrets[self.selected]["tile"] = tile  # reposition onto empty high ground
                self._range_turret = self.selected  # show its range ring this frame
                self.selected = None
            else:
                self.selected = None  # anything else -> deselect
            return
        if ti is not None:
            if self.turrets[ti]["movable"]:  # fixed turrets can't be picked up
                self.selected = ti
        elif tile in self.slots:
            self.blockers ^= {tile}  # toggle a blocker (only designated slots)

    def _run_round(self, paths):
        tiles = [t["tile"] for t in self.turrets]
        ticks, outcome = simulate_round(self.spawns, paths, tiles, hp=self.lives)
        frames = [self._compose_round(t) for t in ticks]
        return frames, outcome

    # -- rendering ---------------------------------------------------------
    def _fill_tile(self, frame, tile, color):
        tx, ty = tile
        x0, y0 = tx * TILE, ty * TILE
        frame[y0 : y0 + TILE, x0 : x0 + TILE] = color

    def _center(self, tile):
        return (tile[0] * TILE + 1, tile[1] * TILE + 1)

    def _draw_enemy(self, frame, center, hp, color=C_ENEMY):
        # 3x3 body around the centre pixel; body pixels vanish as lives are lost
        # (centre last). `center` is a pixel, so the body moves sub-tile.
        x0, y0 = center[0] - 1, center[1] - 1
        for dx, dy in SPIRAL[pixels_removed(hp, self.lives) :]:  # surviving pixels
            px, py = x0 + dx, y0 + dy
            if 0 <= px < VIEW and 0 <= py < VIEW:
                frame[py, px] = color

    def _corners(self, frame, tile, color):
        x0, y0 = tile[0] * TILE, tile[1] * TILE
        for dx in (0, TILE - 1):
            for dy in (0, TILE - 1):
                frame[y0 + dy, x0 + dx] = color

    def _draw_turret(self, frame, tile, selected):
        # fixed and movable turrets look identical; level design tells them apart
        x0, y0 = tile[0] * TILE, tile[1] * TILE
        frame[y0 : y0 + TILE, x0 : x0 + TILE] = C_TURRET
        if selected:
            for dx, dy in SPIRAL[:8]:  # white outer ring
                frame[y0 + dy, x0 + dx] = C_SELECT
        frame[y0 + 1, x0 + 1] = C_LASER  # barrel

    def _draw_line(self, frame, a, b, color):
        for px, py in _line(a[0], a[1], b[0], b[1]):
            if 0 <= px < VIEW and 0 <= py < VIEW:
                frame[py, px] = color

    def _draw_path_end(self, frame, path, open_tiles, color=C_PATH):
        # Push the route line one pixel past its last tile centre, toward the
        # target, so it touches the blocking tile. Prefer the heading it was
        # already going; among target-ward axes pick one that hits an obstacle.
        last = path[-1]
        sx = (self.target[0] > last[0]) - (self.target[0] < last[0])
        sy = (self.target[1] > last[1]) - (self.target[1] < last[1])
        cands = []
        if len(path) >= 2:
            ld = (last[0] - path[-2][0], last[1] - path[-2][1])
            if ld in ((sx, 0), (0, sy)):  # was already heading toward the target
                cands.append(ld)
        cands += [(sx, 0)] * (sx != 0) + [(0, sy)] * (sy != 0)
        if not cands:  # already on the target (no obstacle) -> nothing to extend
            return
        d = next((d for d in cands if (last[0] + d[0], last[1] + d[1]) not in open_tiles), cands[0])
        cx, cy = self._center(last)
        px, py = cx + d[0], cy + d[1]
        if 0 <= px < VIEW and 0 <= py < VIEW:
            frame[py, px] = color

    def _draw_ranges(self, frame, which=None):
        # dark-red ring at a turret's firing radius (Euclidean, tiles->px). Drawn
        # as a 1-px distance band (closed at any radius, unlike angle sampling).
        # `which` = a turret index (just-placed) or None for every turret (editor).
        r = TURRET_RADIUS * TILE
        rr = int(r) + 1
        turrets = self.turrets if which is None else [self.turrets[which]]
        for t in turrets:
            cx, cy = self._center(t["tile"])
            for py in range(max(0, cy - rr), min(VIEW, cy + rr + 1)):
                for px in range(max(0, cx - rr), min(VIEW, cx + rr + 1)):
                    if abs(math.hypot(px - cx, py - cy) - r) <= 0.5:
                        frame[py, px] = C_RANGE

    def _draw_path_coverage(self, frame, which=None):
        # recolour the route line red wherever it falls inside a turret's radius,
        # so you can see exactly which stretch of path that turret can shoot
        rp = TURRET_RADIUS * TILE
        turrets = self.turrets if which is None else [self.turrets[which]]
        for t in turrets:
            cx, cy = self._center(t["tile"])
            lo, hi = int(rp) + 1, int(rp) + 1
            for py in range(max(0, cy - hi), min(VIEW, cy + hi + 1)):
                for px in range(max(0, cx - lo), min(VIEW, cx + lo + 1)):
                    if frame[py, px] == C_PATH and math.hypot(px - cx, py - cy) <= rp + 1e-9:
                        frame[py, px] = C_FLASH

    def _terrain_color(self, tile):
        if tile in self.high:
            return C_HIGH
        if tile in self.highest:
            return C_HIGHEST
        return C_LOW

    def _base_frame(self):
        frame = np.full((VIEW, VIEW), C_UI, dtype=np.int8)
        for ty in range(GRID):
            for tx in range(GRID):
                frame[ty * TILE : ty * TILE + TILE, tx * TILE : tx * TILE + TILE] = self._terrain_color((tx, ty))
        for s in self.slots:  # designated blocker slots
            if s in self.blockers:
                self._fill_tile(frame, s, C_BLOCK)  # toggled on -> filled
            else:
                self._corners(frame, s, C_BLOCK)  # open -> 4 corner pixels
        self._draw_energy_bar(frame)
        return frame

    def _draw_energy_bar(self, frame):
        # bottom UI row (the grid is 63 px in a 64 px frame, so row 63 is free)
        w = round(VIEW * getattr(self, "energy", ENERGY_MAX) / ENERGY_MAX)
        frame[VIEW - 1, :] = C_ENERGY_TRACK
        if w > 0:
            frame[VIEW - 1, :w] = C_ENERGY

    def _draw_paths(self, frame, flash=()):
        # blue route line per mover (kept visible during the round animation too);
        # a mover in `flash` (stuck, no route) draws its best-effort line red
        open_tiles = self._open_tiles()
        dist = dist_field(self.target, open_tiles)
        for i, sp in enumerate(self.spawns):
            path = greedy_path(sp, self.target, dist, open_tiles) or reachable_path(sp, self.target, open_tiles)
            color = C_FLASH if i in flash else C_PATH
            for a, b in pairwise(path):
                self._draw_line(frame, self._center(a), self._center(b), color)
            self._draw_path_end(frame, path, open_tiles, color)

    def _compose_plan(self, flash=()):
        frame = self._base_frame()
        self._draw_paths(frame, flash)
        if self._show_all_ranges:  # editor design view: every turret's range + coverage
            self._draw_ranges(frame)
            self._draw_path_coverage(frame)
        elif self._range_turret is not None:  # game: only the just-placed turret
            self._draw_ranges(frame, self._range_turret)
            self._draw_path_coverage(frame, self._range_turret)
        self._fill_tile(frame, self.target, C_TARGET)
        if self.selected is not None:  # carrying a turret: mark every free high-ground tile
            occupied = {t["tile"] for t in self.turrets}
            for tile in self.high - occupied:
                cx, cy = self._center(tile)
                frame[cy, cx] = C_BLOCK  # purple centre dot = a place you can drop it
        for i, t in enumerate(self.turrets):
            self._draw_turret(frame, t["tile"], i == self.selected)
        for i, sp in enumerate(self.spawns):
            self._draw_enemy(frame, self._center(sp), self.lives, C_FLASH if i in flash else C_ENEMY)
        return frame

    def _compose_round(self, tick):
        frame = self._base_frame()
        self._draw_paths(frame)
        self._fill_tile(frame, self.target, C_TARGET)
        for t in self.turrets:
            self._draw_turret(frame, t["tile"], False)
        for turret_px, mover_px in tick["lasers"]:  # already centre pixels
            self._draw_line(frame, turret_px, mover_px, C_LASER)
        for i in tick["alive"]:
            self._draw_enemy(frame, tick["pos"][i], tick["hp"][i])
        return frame
