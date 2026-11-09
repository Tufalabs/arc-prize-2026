from __future__ import annotations

from collections import deque

from ..core import DslAgent, camera_grid_to_display, observation_level_index

OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}
ROT_CW = {"U": "R", "R": "D", "D": "L", "L": "U"}


def _rot(dirs, n=1):
    d = set(dirs)
    for _ in range(n % 4):
        d = {ROT_CW[x] for x in d}
    return frozenset(d)


def _ncell(cell, d):
    c, r = cell
    return {"R": (c + 1, r), "L": (c - 1, r), "D": (c, r + 1), "U": (c, r - 1)}[d]


class _Planner:
    """A faithful, animation-free model of one en18 level used to search for a
    winning click sequence.

    A tile is represented purely by its set of wire directions (its identity is
    irrelevant for adjacency, teleport and rotation), so a state is just
    ``{gid: {cell: frozenset(dirs)}}`` plus the teleport flag and the set of
    energised wires. The transition rules mirror ``En18``:

      * a slide / belt-tick / empty click re-settles power afterwards
        (``_finish_action_or_flow``), so ``energised = {wi: src solved}``;
      * a teleport or rotate click does NOT re-settle power (those code paths
        return early), so newly-solved sources stay un-energised until the next
        ordinary action.

    Search is decomposed into one subgoal per group in solve order (teleport-pair
    groups merged, since a single swap must satisfy both). Only the current
    subgoal's groups -- plus belt sources, which must stay slideable for phase
    control -- are slid during a subgoal's BFS, which keeps the branching factor
    small.
    """

    def __init__(self, game):
        self.order_groups = list(game._groups)
        self.cells = {gid: list(g.cells) for gid, g in game._groups.items()}
        self.lines = {gid: [(a, i, tuple(lc)) for a, i, lc in g.lines()] for gid, g in game._groups.items()}
        self.green = dict(game._green)
        self.rot = dict(game._rotate_spots)
        self.belts = [(gid, a, i, dr) for (gid, a, i), dr in game._belt.items()]  # preserve order
        self.wires = game._wires
        self.gate = dict(game._gate)
        self.tp = game._teleport
        self.tp_gated = set(getattr(game, "_teleport_gated", set()))
        self.goal_wire = game._goal_wire
        self.solve_order = list(game._solve_order)
        self.trigger_wire = None
        if self.tp:
            b = next((b for b in game._bulbs if b["id"] == self.tp["trigger"]), None)
            self.trigger_wire = b["wire"] if b else None
        rb = next((b for b in game._bulbs if b["role"] == "rotate"), None)
        self.rotate_wire = rb["wire"] if rb else None
        self.init = {gid: {c: frozenset(g.tiles[g.placement[c]]) for c in g.cells} for gid, g in game._groups.items()}
        # groups whose wire gates a belt -- kept slideable so phase can be retimed
        self.belt_sources = set()
        for gid, a, i, _ in self.belts:
            w = self.gate.get((gid, a, i))
            if w is not None:
                self.belt_sources.add(self.wires[w]["src"])

    # -- state queries -----------------------------------------------------
    def _solved(self, st, gid):
        d = st[gid]
        cs = set(self.cells[gid])
        for cell, dirs in d.items():
            for dr in dirs:
                nb = _ncell(cell, dr)
                if nb not in cs or OPP[dr] not in d[nb]:
                    return False
        return True

    def _line_enabled(self, st, en, gid, axis, index):
        if gid in self.tp_gated and not st["_tp"]:
            return False
        w = self.gate.get((gid, axis, index))
        return w is None or w in en

    def _is_belt(self, gid, axis, index):
        return any(g == gid and a == axis and i == index for g, a, i, _ in self.belts)

    def _clickable_lines(self, st, en):
        out = []
        for gid in self.order_groups:
            for a, i, lc in self.lines[gid]:
                if self._is_belt(gid, a, i):
                    continue
                if self._line_enabled(st, en, gid, a, i):
                    out.append((gid, a, i, lc))
        return out

    def _active_belts(self, st, en):
        return [(gid, a, i, dr) for gid, a, i, dr in self.belts if self._line_enabled(st, en, gid, a, i)]

    @staticmethod
    def _slide(d, lc, direction):
        ids = [d[c] for c in lc]
        ids = [ids[-1], *ids[:-1]] if direction in ("R", "D") else [*ids[1:], ids[0]]
        for c, t in zip(lc, ids, strict=False):
            d[c] = t

    def _copy(self, st):
        ns = {gid: dict(st[gid]) for gid in self.order_groups}
        ns["_tp"] = st["_tp"]
        ns["_en"] = st["_en"]
        return ns

    def _settle_power(self, ns):
        ns["_en"] = frozenset(wi for wi, w in enumerate(self.wires) if self._solved(ns, w["src"]))

    # -- transitions -------------------------------------------------------
    def successors(self, st, allowed=None):
        """Yield (label, next_state). ``allowed`` restricts which groups may be
        slid (belt sources are always allowed); None means no restriction."""
        en = st["_en"]
        out = []
        belts_now = self._active_belts(st, en)

        def tick_belts(ns):
            for gid, a, i, dr in belts_now:
                lc = next(line for aa, ii, line in self.lines[gid] if aa == a and ii == i)
                self._slide(ns[gid], lc, dr)

        for gid, a, i, lc in self._clickable_lines(st, en):
            if allowed is not None and gid not in allowed and gid not in self.belt_sources:
                continue
            for dr in ("L", "R") if a == "row" else ("U", "D"):
                ns = self._copy(st)
                self._slide(ns[gid], lc, dr)
                tick_belts(ns)
                self._settle_power(ns)
                out.append((("slide", gid, a, i, dr), ns))
        # NOTE: belts have no standalone "click the purple button" action any more --
        # they only advance alongside a player's blue-button slide (handled above).
        if self.tp and self.trigger_wire in en:
            g1, g2 = self.tp["groups"]
            c1, c2 = self.green[g1], self.green[g2]
            ns = self._copy(st)
            ns[g1][c1], ns[g2][c2] = st[g2][c2], st[g1][c1]
            ns["_tp"] = not st["_tp"]
            out.append((("teleport",), ns))
        if self.rotate_wire is not None and self.rotate_wire in en:
            ns = self._copy(st)
            for gid, cell in self.rot.items():
                ns[gid][cell] = _rot(ns[gid][cell], 1)
            out.append((("rotate",), ns))
        return out

    def _key(self, st):
        return (
            tuple((gid, tuple(st[gid][c] for c in self.cells[gid])) for gid in self.order_groups),
            st["_tp"],
            st["_en"],
        )

    def _start(self):
        st = {gid: dict(self.init[gid]) for gid in self.order_groups}
        st["_tp"] = False
        st["_en"] = frozenset()
        return st

    # -- search ------------------------------------------------------------
    def _subgoals(self):
        """Ordered (group_set, wire_set) pairs; teleport-pair groups merged."""
        pair = set(self.tp["groups"]) if self.tp else set()
        out = []
        done = set()
        for gid in self.solve_order:
            if gid in done:
                continue
            grp = pair if gid in pair else {gid}
            wires = frozenset(wi for wi, w in enumerate(self.wires) if w["src"] in grp)
            done |= grp
            if wires:
                out.append((frozenset(grp), wires))
        return out

    def _bfs(self, start, target_wires, allowed, max_states=800_000):
        if target_wires <= start["_en"]:
            return []
        seen = {self._key(start)}
        q = deque([(start, [])])
        n = 0
        while q:
            s, path = q.popleft()
            n += 1
            if n > max_states:
                return None
            for lab, ns in self.successors(s, allowed):
                k = self._key(ns)
                if k in seen:
                    continue
                if target_wires <= ns["_en"]:
                    return [*path, lab]
                seen.add(k)
                q.append((ns, [*path, lab]))
        return None

    def plan(self):
        """Return the full list of semantic action labels that win the level, or
        raise if no plan is found."""
        st = self._start()
        labels = []
        for grp, target in self._subgoals():
            seg = self._bfs(st, target, grp)
            if seg is None:
                raise RuntimeError(f"en18 planner: no path to subgoal {sorted(grp)} (wires {sorted(target)}).")
            succ_lookup = {lab: ns for lab, ns in self.successors(st, grp)}
            for lab in seg:
                st = succ_lookup[lab]
                labels.append(lab)
                succ_lookup = {lab2: ns for lab2, ns in self.successors(st, grp)}
        return labels


def _label_to_click(game, label):
    kind = label[0]
    if kind in ("slide", "belt"):
        _, gid, axis, index, direction = label
        gx, gy = game._groups[gid].button_center(axis, index, direction)
    elif kind == "teleport":
        b = next(b for b in game._bulbs if b["id"] == game._teleport["trigger"])
        gx, gy = b["pos"]
    else:  # rotate
        b = next(b for b in game._bulbs if b["role"] == "rotate")
        gx, gy = b["pos"]
    dx, dy = camera_grid_to_display(game.camera, int(gx), int(gy))
    return (6, {"x": int(dx), "y": int(dy)})


class En18DslAgent(DslAgent):
    """Oracle: plan each level once (a BFS per group/teleport-pair in dependency
    order) and replay the resulting clicks."""

    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)
        self._level_idx = None
        self._queue: list = []

    def reset_episode(self):
        super().reset_episode()
        self._level_idx = None
        self._queue = []

    def _build_plan(self, game):
        planner = _Planner(game)
        return [_label_to_click(game, lab) for lab in planner.plan()]

    def next_action(self, env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Missing `levels_completed` in en18 observation.")
        self.mark_levels_solved(level_idx)
        # Rebuild the plan on a new level OR a full reset, since the latter restores the
        # board without changing levels_completed and would otherwise replay a stale queue.
        reset_level = bool(getattr(observation, "full_reset", False))
        if level_idx != self._level_idx or reset_level:
            self._level_idx = level_idx
            self._queue = self._build_plan(env._game)
        if not self._queue:
            raise RuntimeError("en18 DSL plan exhausted before solving the level.")
        return self._queue.pop(0)


AGENT_CLASS = En18DslAgent
