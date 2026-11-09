"""Talos Beams — a connector/laser puzzle (see spec.md).

Self-contained: the environment loader execs this file with no package
context, so everything lives here and there are no sibling imports. The pure
beam resolver (`resolve_beams`) takes plain data and avoids arcengine so it
can be unit-tested directly.
"""

from __future__ import annotations

import json
import math
import pathlib
import random
from dataclasses import dataclass, field

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

# --- geometry / board ---
TILE = 5
VIEW = 64
MAX_TILES = 12
FRAMES_PER_TIER = 10
CHARGE_FULL = 3  # one notch per internal target pixel
_C = TILE // 2  # tile-centre offset (=2)
CONNECTOR_RADIUS = TILE / 2.0  # a placed connector blocks unrelated beams within this circle
GRID_W = GRID_H = 12  # all levels are 12x12 tiles


def _mask(pred):
    return frozenset((dx, dy) for dx in range(TILE) for dy in range(TILE) if pred(dx, dy))


# 5x5 stamp masks (dx, dy)
DIAMOND_FILL = _mask(lambda dx, dy: abs(dx - _C) + abs(dy - _C) <= _C)
DIAMOND_LINE = _mask(lambda dx, dy: abs(dx - _C) + abs(dy - _C) == _C)
DIAMOND_IN = frozenset(DIAMOND_FILL - DIAMOND_LINE)
CIRCLE_FILL = _mask(lambda dx, dy: not (dx in (0, TILE - 1) and dy in (0, TILE - 1)))
CENTER3 = _mask(lambda dx, dy: 1 <= dx <= TILE - 2 and 1 <= dy <= TILE - 2)
CIRCLE_RING = frozenset(CIRCLE_FILL - CENTER3)

# --- palette indices (re_arc COLOR_MAP) ---
C_FLOOR = 5  # black
C_WHITE_WALL = 0  # white
C_GREY_WALL = 2  # mid grey
C_DOOR = 11  # yellow
C_INACTIVE = 3  # dark grey (wired-but-dead beam line)
C_PLAYER = 15  # purple
C_EXIT = 12  # orange
C_CONN = 3  # grey connector ring (distinct from white centre and 153 grey walls)
C_LETTERBOX = 5
C_ENERGY = 6  # energy bar fill (pink — unused elsewhere in this game)
C_ENERGY_TRACK = 4  # depleted part of the energy bar (dark grey)

BEAM_COLOR = {"R": 8, "G": 14, "B": 9}  # red / green / blue

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

ENERGY_MAX = 256  # drains 1 per action; reaching 0 is a loss. Bar fills the 64 px row at full

_GEO_EPS = 1e-9
_FRAC_EPS = 1e-7
# blockers are shrunk by this many pixels for clipping, so a beam that only
# touches a wall/connector boundary exactly (float equality) counts as connected
_CLIP_EPS = 1e-6


# ---------------------------------------------------------------------------
# Geometry helpers (pure)
# ---------------------------------------------------------------------------
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _seg_cross(p, p2, q, q2):
    """Intersection of segments p->p2 and q->q2. Returns (t, u, point) with
    t,u the params on each segment, or None (incl. parallel/collinear)."""
    rx, ry = p2[0] - p[0], p2[1] - p[1]
    sx, sy = q2[0] - q[0], q2[1] - q[1]
    rxs = rx * sy - ry * sx
    if abs(rxs) < _GEO_EPS:
        return None
    qpx, qpy = q[0] - p[0], q[1] - p[1]
    t = (qpx * sy - qpy * sx) / rxs
    u = (qpx * ry - qpy * rx) / rxs
    if -_GEO_EPS <= t <= 1 + _GEO_EPS and -_GEO_EPS <= u <= 1 + _GEO_EPS:
        return (t, u, (p[0] + t * rx, p[1] + t * ry))
    return None


def _circle_entry(p0, p1, center, r):
    """Param t in (0,1] where segment p0->p1 first enters the circle, else None."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    fx, fy = p0[0] - center[0], p0[1] - center[1]
    a = dx * dx + dy * dy
    if a < _GEO_EPS:
        return None
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    hits = [t for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if _FRAC_EPS < t <= 1 + _FRAC_EPS]
    return min(hits) if hits else None


def _line4(x0, y0, x1, y1):
    """4-connected (orthogonally connected) raster line — no corner-only gaps, so
    diagonal lines read as solid rather than dashed."""
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
        if e2 > -dy:  # step in x ...
            err -= dy
            x += sx
        elif e2 < dx:  # ... otherwise step in y (one axis per step = 4-connected)
            err += dx
            y += sy


def _target_conn(tx, ty, rot, offx, offy):
    """A target's single connection point: the float64 midpoint of the facing
    edge, exactly on the tile boundary. The beam resolver and the renderer both
    use this same point (the renderer just rounds it to a pixel), so collision
    math and drawing can never diverge. Being on the boundary, a wall tile
    adjacent on another side can't graze-block the approach. rot 0=down 1=left
    2=up 3=right."""
    x0, y0 = offx + tx * TILE, offy + ty * TILE
    c = TILE / 2.0
    return {
        0: (x0 + c, float(y0 + TILE)),
        1: (float(x0), y0 + c),
        2: (x0 + c, float(y0)),
        3: (float(x0 + TILE), y0 + c),
    }[rot % 4]


def _box_entry_t(p, q, x0, y0, x1, y1):
    """Liang-Barsky: param t in [0,1] where segment p->q enters box, or None.
    t==0 means p is already inside."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if abs(pp) < _GEO_EPS:
            if qq < 0:
                return None
        else:
            r = qq / pp
            if pp < 0:
                if r > t1:
                    return None
                t0 = max(t0, r)
            else:
                if r < t0:
                    return None
                t1 = min(t1, r)
    if t0 > t1:
        return None
    return t0


# ---------------------------------------------------------------------------
# Beam resolver (pure — no arcengine)
# ---------------------------------------------------------------------------
@dataclass
class NodeState:
    kind: str  # 'source' | 'connector' | 'target' | 'inverter'
    color: str | None  # 'R'/'G'/'B' for source/target; None for connector/inverter
    tx: int
    ty: int
    held: bool = False
    rot: int = 0  # target only: 0=down 1=left 2=up 3=right — which edge the beam connects to


@dataclass
class ResolvedBeam:
    emitter: int
    target: int
    color: str
    tier: int
    p0: tuple  # emitter anchor (px)
    pT: tuple  # target anchor (px)
    live: tuple  # live end (cut point or target)
    reached: bool
    length: float  # full p0->pT length
    cut_by_beam: bool = False  # cut by another beam (crossing/head-on), not a wall


@dataclass
class Resolution:
    powered: dict  # nid -> color
    beams: list  # ResolvedBeam
    dead: set  # connector nids killed by 2-colour conflict
    signature: tuple


def _anchor(node, offx, offy):
    # integer tile-centre pixel (TILE is odd) so beams and base lines align exactly
    return (offx + node.tx * TILE + TILE // 2, offy + node.ty * TILE + TILE // 2)


def resolve_beams(nodes, edges, blocking_tiles, offx, offy, seed=0):
    """Tiered hop-distance wavefront. See spec.md for the rules."""
    rng = random.Random(seed)
    anchors = {nid: _anchor(n, offx, offy) for nid, n in nodes.items()}

    adj = {nid: set() for nid in nodes}
    for e in edges:
        a, b = tuple(e)
        if a in nodes and b in nodes:
            adj[a].add(b)
            adj[b].add(a)

    def is_inert(nid):
        n = nodes[nid]
        return n.kind in ("connector", "inverter") and n.held

    def wall_cut(p0, pT):
        best = None
        for tx, ty in blocking_tiles:
            x0 = offx + tx * TILE
            y0 = offy + ty * TILE
            # box inset by _CLIP_EPS: a beam grazing the exact boundary passes
            t = _box_entry_t(p0, pT, x0 + _CLIP_EPS, y0 + _CLIP_EPS, x0 + TILE - _CLIP_EPS, y0 + TILE - _CLIP_EPS)
            if t is not None and t > _FRAC_EPS:
                best = t if best is None else min(best, t)
        return best

    # placed connectors and inverters are circular obstacles for beams not wired into them
    conn_blockers = [
        (nid, anchors[nid]) for nid, n in nodes.items() if n.kind in ("connector", "inverter") and not n.held
    ]

    def connector_cut(p0, pT, a, b):
        best = None
        for nid, ctr in conn_blockers:
            if nid == a or nid == b:
                continue
            t = _circle_entry(p0, pT, ctr, CONNECTOR_RADIUS - _CLIP_EPS)  # shrink: exact grazes pass
            if t is not None:
                best = t if best is None else min(best, t)
        return best

    powered = {nid: (n.color, 0) for nid, n in nodes.items() if n.kind == "source"}
    arrivals = {}  # nid -> set of delivered colours
    dead = set()
    conducting = []  # (p0, live_end, color, tier) — blocks strictly-higher tiers
    beams_out = []

    cur_emitters = [nid for nid, (_c, t) in powered.items() if t == 0]
    tier = 0
    guard = len(nodes) + 2
    while cur_emitters and guard > 0:
        guard -= 1
        btier = tier + 1
        cands = []
        for e_id in cur_emitters:
            color = powered[e_id][0]
            p0 = anchors[e_id]
            for nb in adj[e_id]:
                if is_inert(nb):
                    continue
                if nb in powered and powered[nb][1] <= btier - 2:
                    continue  # don't beam back upstream; peers (same tier) kept
                pT = anchors[nb]
                if nodes[nb].kind == "target":
                    pT = _target_conn(nodes[nb].tx, nodes[nb].ty, nodes[nb].rot, offx, offy)
                cands.append(
                    {"emitter": e_id, "target": nb, "color": color, "p0": p0, "pT": pT, "L": _dist(p0, pT), "tcut": 1.0}
                )
        if not cands:
            break

        # clip vs hard blockers — walls + unrelated connectors (wall_t) — and vs
        # already-conducting lower-tier beams (beam_t)
        for c in cands:
            tw = wall_cut(c["p0"], c["pT"])
            tc = connector_cut(c["p0"], c["pT"], c["emitter"], c["target"])
            hard = [t for t in (tw, tc) if t is not None]
            c["wall_t"] = min(hard) if hard else 1.0
            c["beam_t"] = 1.0
            for cp0, cp1, _cc, _ct in conducting:
                hit = _seg_cross(c["p0"], c["pT"], cp0, cp1)
                if hit is None:
                    continue
                t, _u, _pt = hit
                if t > _FRAC_EPS:
                    c["beam_t"] = min(c["beam_t"], t)
            c["tcut"] = min(c["wall_t"], c["beam_t"])

        def _cut(c, f):  # record a beam-cut at fraction f
            c["beam_t"] = min(c["beam_t"], f)
            c["tcut"] = min(c["tcut"], f)

        # head-on: same edge driven from both ends this tier -> both stop at the middle
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                if a["emitter"] == b["target"] and a["target"] == b["emitter"]:
                    _cut(a, 0.5)
                    _cut(b, 0.5)

        # mutual crossings among same-tier candidates, resolved by arrival fraction
        events = []
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                shared = {a["emitter"], a["target"]} & {b["emitter"], b["target"]}
                hit = _seg_cross(a["p0"], a["pT"], b["p0"], b["pT"])
                if hit is None:
                    continue
                _t, _u, pt = hit
                if any(_dist(pt, anchors[s]) < 1e-6 for s in shared):
                    continue
                events.append((min(_t, _u), i, j, _t, _u))
        events.sort(key=lambda ev: (ev[0], ev[1], ev[2]))
        for _tmin, i, j, fa, fb in events:
            a, b = cands[i], cands[j]
            if fa > a["tcut"] + _FRAC_EPS or fb > b["tcut"] + _FRAC_EPS:
                continue  # one was already cut before the crossing
            if abs(fa - fb) < _FRAC_EPS:
                _cut(b if rng.random() < 0.5 else a, fb)  # tie -> seeded-random loser
            elif fa < fb:
                _cut(b, fb)
            else:
                _cut(a, fa)

        newly = []
        for c in cands:
            reached = c["tcut"] >= 1.0 - _FRAC_EPS
            live = _lerp(c["p0"], c["pT"], c["tcut"])
            cut_by_beam = (not reached) and (c["beam_t"] <= c["wall_t"] + _FRAC_EPS)
            beams_out.append(
                ResolvedBeam(
                    c["emitter"], c["target"], c["color"], btier, c["p0"], c["pT"], live, reached, c["L"], cut_by_beam
                )
            )
            conducting.append((c["p0"], live, c["color"], btier))
            if reached:
                arrivals.setdefault(c["target"], set()).add(c["color"])

        for tgt, cols in arrivals.items():
            if tgt in powered or tgt in dead:
                continue
            n = nodes[tgt]
            if n.kind == "connector":
                if len(cols) >= 2:
                    dead.add(tgt)
                elif len(cols) == 1:
                    powered[tgt] = (next(iter(cols)), btier)
                    newly.append(tgt)
            elif n.kind == "inverter":
                eff = cols - {"G"}  # green does nothing
                if len(eff) >= 2:
                    dead.add(tgt)
                elif len(eff) == 1:
                    powered[tgt] = (INVERT_COLOR[next(iter(eff))], btier)  # emits the opposite
                    newly.append(tgt)
            elif n.kind == "target":
                if n.color in cols:
                    powered[tgt] = (n.color, btier)
        cur_emitters = newly
        tier = btier

    # Signature is the BEAM PATTERN only (powered nodes + live beams) — NOT the
    # blocking geometry, so opening/closing a door that doesn't actually reroute
    # any beam produces no animation.
    sig = (
        tuple(sorted((nid, c) for nid, (c, _t) in powered.items())),
        tuple(
            sorted(
                (b.emitter, b.target, b.color, round(b.live[0], 2), round(b.live[1], 2), b.reached) for b in beams_out
            )
        ),
    )
    return Resolution(powered={k: v[0] for k, v in powered.items()}, beams=beams_out, dead=dead, signature=sig)


# ---------------------------------------------------------------------------
# Levels (ASCII).  Legend per tile:
#   '.' floor   '#' white wall   '%' grey wall   '@' player   'X' exit
#   'r/g/b' source   'R/G/B' target   'o' connector   '1'-'9' door tiles
# `doors` maps a door digit to the target colour that opens it.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple
    doors: dict = field(default_factory=dict)  # door digit -> target id(s) / colour
    charge_full: int = CHARGE_FULL
    edges: tuple = ()  # initial wiring: pairs of node ids
    ids: dict = field(default_factory=dict)  # "tx,ty" -> stable entity id (editor-assigned)
    rot: dict = field(default_factory=dict)  # target id -> rotation 0..3
    click_hint: bool = False  # while holding a connector, outline clickable nodes in purple


def _data_dir():
    # The env loader execs this file with no __file__, so locate our own folder
    # via the re_arc package (snapshot-safe for SLURM/deploy copies).
    import re_arc

    return pathlib.Path(re_arc.__file__).resolve().parent / "environment_files" / "talos_beams" / "0001"


def _load_level_specs(path=None):
    """Level layouts + manual wiring live in levels.json (split from code)."""
    path = pathlib.Path(path) if path is not None else (_data_dir() / "levels.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for lv in data:
        specs.append(
            LevelSpec(
                name=lv["name"],
                rows=tuple(lv["grid"]),
                doors=dict(lv.get("doors", {})),
                charge_full=int(lv.get("charge_full", CHARGE_FULL)),
                edges=tuple(tuple(e) for e in lv.get("edges", [])),  # pairs of node ids
                ids=dict(lv.get("ids", {})),
                rot=dict(lv.get("rot", {})),
                click_hint=bool(lv.get("click_hint", False)),
            )
        )
    return specs


_ENTITY_KIND = {
    "r": "source",
    "g": "source",
    "b": "source",
    "R": "target",
    "G": "target",
    "B": "target",
    "o": "connector",
    "i": "inverter",
}
INVERT_COLOR = {"R": "B", "B": "R"}  # inverter swaps red<->blue; green does nothing


def _scan_entities(rows, ids=None):
    """Entities with stable ids. An entity's id comes from the `ids` map
    (`"tx,ty" -> id`, editor-assigned) when present; any unmapped cell gets the
    lowest-free char+index id (so deleting one never renumbers the others).
    Returns dicts {id, kind, color, tx, ty}."""
    ids = ids or {}
    used = set(ids.values())
    out = []
    for ty, row in enumerate(rows):
        for tx, ch in enumerate(row):
            kind = _ENTITY_KIND.get(ch)
            if kind is None:
                continue
            eid = ids.get(f"{tx},{ty}")
            if eid is None:
                n = 1
                while f"{ch}{n}" in used:
                    n += 1
                eid = f"{ch}{n}"
                used.add(eid)
            color = ch.upper() if ch in "rgbRGB" else None
            out.append({"id": eid, "kind": kind, "color": color, "tx": tx, "ty": ty})
    return out


def _parse_level(spec):
    # normalize to exactly GRID_W x GRID_H (pad/truncate) so size slips never crash
    rows = [(r + "#" * GRID_W)[:GRID_W] for r in list(spec.rows)[:GRID_H]]
    while len(rows) < GRID_H:
        rows.append("#" * GRID_W)
    white, grey, door_tiles = set(), set(), {}
    player = exit_tile = None
    for ty, row in enumerate(rows):
        for tx, ch in enumerate(row):
            if ch == "#":
                white.add((tx, ty))
            elif ch == "%":
                grey.add((tx, ty))
            elif ch == "@":
                player = (tx, ty)
            elif ch == "X":
                exit_tile = (tx, ty)
            elif ch.isdigit():
                door_tiles.setdefault(ch, set()).add((tx, ty))
    if player is None or exit_tile is None:
        raise ValueError(f"Level {spec.name!r} needs both '@' and 'X'.")
    ents = _scan_entities(rows, spec.ids)
    nodes = {
        e["id"]: NodeState(e["kind"], e["color"], e["tx"], e["ty"], rot=int(spec.rot.get(e["id"], 0))) for e in ents
    }
    target_ids = {e["id"] for e in ents if e["kind"] == "target"}
    by_color = {}
    for e in ents:
        if e["kind"] == "target":
            by_color.setdefault(e["color"], []).append(e["id"])

    def expand(items):
        out = set()
        for item in items:
            if item in target_ids:
                out.add(item)
            elif item in ("R", "G", "B"):
                out.update(by_color.get(item, []))
            else:
                raise ValueError(f"Level {spec.name!r}: door references unknown target {item!r}.")
        return out

    def door_req(req):
        # Parse a door requirement into (op, targets). Each item is a target id
        # ("B1") or a colour ("R" = every target of that colour).
        #   None            -> never opens
        #   "B1" / ["a","b"] -> ALL listed targets full (default)
        #   {"all": [...]}   -> ALL listed full
        #   {"any": [...]}   -> ANY listed full (OR)
        if req is None:
            return "all", set()
        if isinstance(req, dict):
            if len(req) != 1 or next(iter(req)) not in ("all", "any"):
                raise ValueError(f"Level {spec.name!r}: door req {req!r} must be {{'all':[...]}} or {{'any':[...]}}.")
            op, items = next(iter(req.items()))
            items = [items] if isinstance(items, str) else list(items)
            return op, expand(items)
        items = [req] if isinstance(req, str) else list(req)
        return "all", expand(items)

    doors = {}
    for d, tiles in door_tiles.items():
        op, tgts = door_req(spec.doors.get(d))
        doors[d] = {"tiles": tiles, "op": op, "targets": tgts, "open": False}
    init_edges = set()
    for a, b in spec.edges:
        if a in nodes and b in nodes and a != b:
            init_edges.add(frozenset((a, b)))
    return {
        "w": GRID_W,
        "h": GRID_H,
        "white": white,
        "grey": grey,
        "doors": doors,
        "nodes": nodes,
        "player": player,
        "exit": exit_tile,
        "charge_full": spec.charge_full,
        "edges": init_edges,
        "click_hint": spec.click_hint,
    }


def _build_level(spec: LevelSpec) -> Level:
    parsed = _parse_level(spec)
    board = Sprite(np.full((VIEW, VIEW), C_LETTERBOX, dtype=np.int8), name="screen", layer=0, collidable=False)
    return Level(grid_size=(VIEW, VIEW), sprites=[board], name=spec.name, data={"parsed": parsed})


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class TalosBeams(ARCBaseGame):
    def __init__(self, seed: int = 0, specs=None) -> None:
        if specs is None:  # loading levels.json: skip malformed levels (e.g. mid-edit)
            levels = []
            for s in _load_level_specs():
                try:
                    levels.append(_build_level(s))
                except Exception as exc:
                    import warnings

                    warnings.warn(f"talos_beams: skipping level {getattr(s, 'name', '?')!r}: {exc}", stacklevel=2)
            if not levels:  # empty / all-malformed levels.json -> a placeholder room
                levels = [
                    _build_level(
                        LevelSpec(
                            name="(no levels — add some in editor.ipynb)",
                            rows=(
                                "############",
                                "#@.........#",
                                "#..........#",
                                "#..........#",
                                "#..........#",
                                "#..........#",
                                "#..........#",
                                "#..........#",
                                "#..........#",
                                "#..........#",
                                "#.........X#",
                                "############",
                            ),
                        )
                    )
                ]
        else:  # explicit specs (editor preview): surface errors directly
            levels = [_build_level(s) for s in specs]
        super().__init__(
            "talos_beams",
            levels,
            Camera(0, 0, VIEW, VIEW, C_LETTERBOX, C_LETTERBOX),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6],
            seed,
        )

    # -- level setup -------------------------------------------------------
    def on_set_level(self, _level: Level) -> None:
        p = self.current_level.get_data("parsed")
        self.W, self.H = p["w"], p["h"]
        self.offx = (VIEW - self.W * TILE) // 2
        self.offy = (VIEW - self.H * TILE) // 2
        self.white = set(p["white"])
        self.grey = set(p["grey"])
        self.exit_tile = p["exit"]
        self.charge_full = p["charge_full"]
        self.click_hint = p["click_hint"]
        # deep-ish copy of mutable per-run state
        self.doors = {
            d: {"tiles": set(v["tiles"]), "op": v["op"], "targets": set(v["targets"]), "open": False}
            for d, v in p["doors"].items()
        }
        self.nodes = {nid: NodeState(n.kind, n.color, n.tx, n.ty, False, n.rot) for nid, n in p["nodes"].items()}
        self.edges = set(p.get("edges", set()))
        self.player = p["player"]
        self.held = None
        self.charge = {nid: 0 for nid, n in self.nodes.items() if n.kind == "target"}
        self.energy = ENERGY_MAX
        self.screen = self.current_level.get_sprites_by_name("screen")[0]
        self._frames_queue = None
        self._pending_next_level = False
        self._pending_lose = False
        self._cur_res = self._resolve()
        self._prev_sig = self._cur_res.signature
        self.screen.pixels = self._compose(self._cur_res)

    # -- geometry helpers --------------------------------------------------
    def _beam_blocking_tiles(self):
        tiles = set(self.white)
        for d in self.doors.values():
            if not d["open"]:
                tiles |= d["tiles"]
        return tiles

    def _player_blocked(self, tx, ty):
        if not (0 <= tx < self.W and 0 <= ty < self.H):
            return True
        if (tx, ty) in self.white or (tx, ty) in self.grey:
            return True
        for d in self.doors.values():
            if not d["open"] and (tx, ty) in d["tiles"]:
                return True
        for n in self.nodes.values():
            if n.kind in ("source", "target") and not n.held and (n.tx, n.ty) == (tx, ty):
                return True
        return False

    def _node_at(self, tx, ty):
        for nid, n in self.nodes.items():
            if not n.held and (n.tx, n.ty) == (tx, ty):
                return nid
        return None

    def _resolve(self):
        res = resolve_beams(
            self.nodes, self.edges, self._beam_blocking_tiles(), self.offx, self.offy, getattr(self, "_seed", 0)
        )
        self._cur_res = res
        return res

    def _sync_doors(self):
        """Open a door per its op ('all'/'any') over its targets; True if any flipped."""
        changed = False
        for d in self.doors.values():
            tgts = d["targets"]
            full = (self.charge.get(t, 0) >= self.charge_full for t in tgts)
            new_open = bool(tgts) and (any(full) if d["op"] == "any" else all(full))
            if new_open != d["open"]:
                d["open"] = new_open
                changed = True
        return changed

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
        if action == GameAction.RESET.value:  # level start / reset: play the build animation
            res = self._resolve()
            self._prev_sig = res.signature
            return self._animate(res) if res.beams else [self._compose(res)]

        self.energy = max(0, self.energy - 1)  # every action drains the energy bar
        did_move = False
        if action in MOVE_DELTAS:
            did_move = True
            dx, dy = MOVE_DELTAS[action]
            nx, ny = self.player[0] + dx, self.player[1] + dy
            if not self._player_blocked(nx, ny):
                self.player = (nx, ny)
                if self.held is not None:  # carry the held connector along
                    self.nodes[self.held].tx, self.nodes[self.held].ty = nx, ny
            if self.player == self.exit_tile:
                self._pending_next_level = True
        elif action == 5:
            self._toggle_carry()
        elif action == 6:
            self._click_wire()
        frames = self._settle_and_frames(did_move)
        if self.energy <= 0 and not self._pending_next_level:  # ran out of energy -> loss
            self._pending_lose = True
        return frames

    def _settle_and_frames(self, did_move):
        """Resolve beams, charge targets, toggle doors. A door flip re-routes
        beams (a fresh calculation) and can flip another door -> each changed
        stage is animated in sequence."""
        frames = []
        prev = self._prev_sig
        fill = did_move  # the move charges targets once; cascade stages only empty
        door_flipped = False
        # Termination: after the first (fill) pass, charge is monotonically
        # non-increasing, so open doors can only ever close from here, and each
        # door closes at most once -> the cascade is bounded by the door count.
        for _ in range(len(self.doors) + 3):
            res = self._resolve()
            self._apply_charges(res, fill)
            fill = False
            if res.signature != prev:
                frames += self._animate(res)
                prev = res.signature
            if self._sync_doors():
                door_flipped = True
            else:
                break
        self._prev_sig = prev
        if not frames:
            return [self._compose(self._cur_res)]
        if door_flipped:
            # doors flipped after the last animation frame was composed -> add a
            # settled frame so the new door state is shown this action, not next
            frames.append(self._compose(self._cur_res))
        return frames

    def _apply_charges(self, res, fill):
        for nid, n in self.nodes.items():
            if n.kind != "target":
                continue
            if res.powered.get(nid) == n.color:
                if fill:
                    self.charge[nid] = min(self.charge_full, self.charge[nid] + 1)
            else:
                self.charge[nid] = 0  # an unpowered target empties immediately

    def _toggle_carry(self):
        if self.held is not None:
            self.nodes[self.held].held = False
            self.nodes[self.held].tx, self.nodes[self.held].ty = self.player
            self.held = None
            return
        here = self._node_at(*self.player)
        if here is not None and self.nodes[here].kind in ("connector", "inverter"):
            self.nodes[here].held = True
            self.held = here

    def _click_wire(self):
        if self.held is None:
            return
        data = self.action.data or {}
        cx, cy = int(data.get("x", -1)), int(data.get("y", -1))
        tx = (cx - self.offx) // TILE
        ty = (cy - self.offy) // TILE
        nid = self._node_at(tx, ty)
        if nid is None or nid == self.held:
            return
        edge = frozenset((self.held, nid))
        self.edges ^= {edge}  # toggle

    # -- rendering ---------------------------------------------------------
    def _fill_tile(self, frame, tx, ty, color):
        x0, y0 = self.offx + tx * TILE, self.offy + ty * TILE
        frame[y0 : y0 + TILE, x0 : x0 + TILE] = color

    def _tile_center(self, tx, ty):
        return (self.offx + tx * TILE + TILE // 2, self.offy + ty * TILE + TILE // 2)

    def _edge_path(self, a, b):
        """Canonical 4-connected pixel path between the two nodes' connection
        points (a target connects at its rotation-chosen edge, others at the
        centre) — identical regardless of direction, so colour lines up on grey.
        The float connection points are the same ones the resolver uses; each is
        mapped to the pixel that contains it (floor), so the drawn line is just
        the pixelisation of the collision geometry — never a different shape."""

        def conn(nid):
            n = self.nodes[nid]
            if n.kind == "target":
                return _target_conn(n.tx, n.ty, n.rot, self.offx, self.offy)  # float edge point
            return self._tile_center(n.tx, n.ty)

        A, B = conn(a), conn(b)
        A = (math.floor(A[0]), math.floor(A[1]))
        B = (math.floor(B[0]), math.floor(B[1]))
        if (B[0], B[1]) < (A[0], A[1]):
            A, B = B, A
        return _line4(A[0], A[1], B[0], B[1])

    def _compose(self, res, T=None):
        frame = np.full((VIEW, VIEW), C_LETTERBOX, dtype=np.int8)
        # board floor
        for ty in range(self.H):
            for tx in range(self.W):
                self._fill_tile(frame, tx, ty, C_FLOOR)
        # walls / doors
        for tx, ty in self.white:
            self._fill_tile(frame, tx, ty, C_WHITE_WALL)
        for tx, ty in self.grey:
            self._fill_tile(frame, tx, ty, C_GREY_WALL)
        for d in self.doors.values():
            for tx, ty in d["tiles"]:
                if not d["open"]:
                    self._fill_tile(frame, tx, ty, C_DOOR)
        block = self._beam_blocking_tiles()

        def ok(px, py):
            return (
                0 <= px < VIEW and 0 <= py < VIEW and ((px - self.offx) // TILE, (py - self.offy) // TILE) not in block
            )

        # one canonical path per edge, shared by grey + colour so they line up
        paths = {e: self._edge_path(*tuple(e)) for e in self.edges if all(nid in self.nodes for nid in e)}
        # solid grey line for every wired connection
        for path in paths.values():
            for px, py in path:
                if ok(px, py):
                    frame[py, px] = C_INACTIVE

        # colour over the lit part of each beam: same path, emitter -> cut
        def front_frac(bm):
            return 1.0 if T is None else max(0.0, min(1.0, T - (bm.tier - 1)))

        for bm in res.beams:
            path = paths.get(frozenset((bm.emitter, bm.target)))
            if path is None:
                continue
            # The lit extent is purely the collision cut point (bm.live): colour the
            # path up to that distance from the emitter. Wall pixels are only *skipped*
            # cosmetically (not painted over), never used to shorten the beam — so the
            # rendered reach always matches what the resolver computed, even when the
            # raster grazes a wall corner the float segment cleared.
            cutf = (_dist(bm.p0, bm.live) / bm.length) if bm.length > 1e-9 else 0.0
            end_dist = (cutf if T is None else min(front_frac(bm), cutf)) * bm.length
            for px, py in path:
                if _dist((px, py), bm.p0) <= end_dist + 0.75 and ok(px, py):
                    frame[py, px] = BEAM_COLOR[bm.color]
        # white block marker at a beam-vs-beam cut, once the front reaches it
        for bm in res.beams:
            if not bm.cut_by_beam:
                continue
            cutf = (_dist(bm.p0, bm.live) / bm.length) if bm.length > 1e-9 else 0.0
            if T is None or front_frac(bm) >= cutf - 1e-9:
                x, y = math.floor(bm.live[0]), math.floor(bm.live[1])
                if 0 <= x < VIEW and 0 <= y < VIEW:
                    frame[y, x] = C_WHITE_WALL
        # nodes
        for nid, n in self.nodes.items():
            if n.held:
                continue
            if n.kind == "source":
                self._stamp(frame, n.tx, n.ty, CIRCLE_FILL, BEAM_COLOR[n.color])
            elif n.kind == "target":
                self._draw_target(frame, nid, n)
            elif n.kind == "connector":
                self._draw_connector(frame, nid, n, res)
            elif n.kind == "inverter":
                self._draw_inverter(frame, nid, n, res)
        # exit / goal (orange diamond)
        self._stamp(frame, self.exit_tile[0], self.exit_tile[1], DIAMOND_FILL, C_EXIT)
        # player (purple diamond), with a centre dot if carrying a connector
        self._stamp(frame, self.player[0], self.player[1], DIAMOND_FILL, C_PLAYER)
        if self.held is not None:
            cx, cy = self._tile_center(*self.player)
            frame[cy, cx] = C_CONN
        self._draw_click_hint(frame)
        self._draw_energy_bar(frame)
        return frame

    def _draw_click_hint(self, frame):
        # while holding a controller, ring every clickable node with a 1px purple
        # 7x7 outline (extends 1px outside the tile) so players see they can click.
        if self.held is None or not getattr(self, "click_hint", False):
            return
        for n in self.nodes.values():
            if n.held:
                continue
            x0, y0 = self.offx + n.tx * TILE, self.offy + n.ty * TILE
            x1, y1, x2, y2 = x0 - 1, y0 - 1, x0 + TILE, y0 + TILE  # 7x7 ring corners
            for x in range(x1, x2 + 1):
                for y in (y1, y2):
                    if 0 <= x < VIEW and 0 <= y < VIEW:
                        frame[y, x] = C_PLAYER
            for y in range(y1, y2 + 1):
                for x in (x1, x2):
                    if 0 <= x < VIEW and 0 <= y < VIEW:
                        frame[y, x] = C_PLAYER

    def _draw_energy_bar(self, frame):
        # bottom two letterbox rows (grid is 60 px tall, centred -> rows 62/63 free)
        w = round(VIEW * getattr(self, "energy", ENERGY_MAX) / ENERGY_MAX)
        frame[VIEW - 2 : VIEW, :] = C_ENERGY_TRACK
        if w > 0:
            frame[VIEW - 2 : VIEW, :w] = C_ENERGY

    def _stamp(self, frame, tx, ty, cells, color):
        x0, y0 = self.offx + tx * TILE, self.offy + ty * TILE
        for dx, dy in cells:
            frame[y0 + dy, x0 + dx] = color

    def _draw_target(self, frame, nid, n):  # bar (long axis perp. to facing) + 3-pixel
        col = BEAM_COLOR[n.color]  # charge fill + a connection nub
        x0, y0 = self.offx + n.tx * TILE, self.offy + n.ty * TILE
        charge = self.charge.get(nid, 0)
        if n.rot in (0, 2):  # faces down / up -> horizontal bar (5 wide x 3 tall)
            top, bot = y0 + 1, y0 + 3
            frame[top, x0 : x0 + TILE] = col
            frame[bot, x0 : x0 + TILE] = col
            frame[top : bot + 1, x0] = col
            frame[top : bot + 1, x0 + TILE - 1] = col
            for i in range(3):
                frame[y0 + 2, x0 + 1 + i] = col if charge > i else C_FLOOR
            frame[y0 + (TILE - 1 if n.rot == 0 else 0), x0 + TILE // 2] = col  # nub
        else:  # faces left / right -> vertical bar (3 wide x 5 tall)
            lft, rgt = x0 + 1, x0 + 3
            frame[y0 : y0 + TILE, lft] = col
            frame[y0 : y0 + TILE, rgt] = col
            frame[y0, lft : rgt + 1] = col
            frame[y0 + TILE - 1, lft : rgt + 1] = col
            for i in range(3):
                frame[y0 + 1 + i, x0 + 2] = col if charge > i else C_FLOOR
            frame[y0 + TILE // 2, x0 + (0 if n.rot == 1 else TILE - 1)] = col  # nub

    def _draw_connector(self, frame, nid, n, res):  # grey ring; centre = colour / white
        self._stamp(frame, n.tx, n.ty, CIRCLE_RING, C_CONN)  # grey border always
        if nid in res.dead:
            self._stamp(frame, n.tx, n.ty, CENTER3, C_WHITE_WALL)  # double-powered: white centre
        elif nid in res.powered:
            self._stamp(frame, n.tx, n.ty, CENTER3, BEAM_COLOR[res.powered[nid]])

    def _draw_inverter(self, frame, nid, n, res):  # grey ring, blue top/bottom, red sides
        self._stamp(frame, n.tx, n.ty, CIRCLE_RING, C_CONN)
        x0, y0 = self.offx + n.tx * TILE, self.offy + n.ty * TILE
        frame[y0, x0 + _C] = BEAM_COLOR["B"]  # top middle
        frame[y0 + TILE - 1, x0 + _C] = BEAM_COLOR["B"]  # bottom middle
        frame[y0 + _C, x0] = BEAM_COLOR["R"]  # left middle
        frame[y0 + _C, x0 + TILE - 1] = BEAM_COLOR["R"]  # right middle
        if nid in res.dead:
            self._stamp(frame, n.tx, n.ty, CENTER3, C_WHITE_WALL)
        elif nid in res.powered:
            self._stamp(frame, n.tx, n.ty, CENTER3, BEAM_COLOR[res.powered[nid]])  # emitted colour

    def _animate(self, res):
        tmax = max((b.tier for b in res.beams), default=0)
        frames = []
        for k in range(1, tmax * FRAMES_PER_TIER + 1):
            frames.append(self._compose(res, T=k / FRAMES_PER_TIER))
        frames.append(self._compose(res))  # settled
        return frames or [self._compose(res)]

    def step_action_name(self):  # convenience for debugging/tests
        return None
