"""Winning-count test for Level 3 ("Moving turrets").

The player places the two movable turrets. This test exhaustively tries every
placement (every pair of turret-placeable tiles that can actually fire on the
path) and asserts that exactly FOUR placements clear the mover — the intended
"one turret in the U, one at an orange position below it" solutions.

Run with:  uv run python re_arc/environment_files/jc26/0001/test_level3.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("jc26_level3", _HERE / "jc26.py")
tb = importlib.util.module_from_spec(_spec)
sys.modules["jc26_level3"] = tb
_spec.loader.exec_module(tb)

LEVEL_INDEX = 2  # level 3 is the third entry in levels.json
EXPECTED_WINS = 4


def _level3_game():
    lv = json.loads((_HERE / "levels.json").read_text(encoding="utf-8"))[LEVEL_INDEX]
    spec = tb.LevelSpec(name=lv["name"], rows=tuple(lv["grid"]), lives=int(lv.get("lives", tb.DEFAULT_LIVES)))
    g = tb.JC26(specs=[spec])
    g.handle_reset()
    return g


def _paths(g):
    ot = g._open_tiles()
    dist = tb.dist_field(g.target, ot)
    return {i: tb.greedy_path(sp, g.target, dist, ot) for i, sp in enumerate(g.spawns)}


def _candidate_tiles(g, paths):
    # turret-placeable tiles that can actually shoot some pixel of some mover's
    # path; a tile out of range of the whole path deals no damage and can't win,
    # so restricting to these is exact (and keeps the enumeration fast).
    rp = tb.TURRET_RADIUS * tb.TILE
    pathpx = {pp for p in paths.values() for pp in tb._pixel_track(p)}
    return [
        t
        for t in g.high
        if any(math.hypot(t[0] * tb.TILE + 1 - px, t[1] * tb.TILE + 1 - py) <= rp + 1e-9 for px, py in pathpx)
    ]


def _winning_placements(g):
    paths = _paths(g)
    cand = _candidate_tiles(g, paths)
    wins = []
    for a, b in itertools.combinations(cand, 2):
        if tb.simulate_round(g.spawns, paths, [a, b], hp=g.lives)[1] == "win":
            wins.append((a, b))
    return wins


def test_winning_placement_count():
    g = _level3_game()
    wins = _winning_placements(g)
    assert len(wins) == EXPECTED_WINS, (f"expected {EXPECTED_WINS} winning placements", len(wins), sorted(wins))


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _main()
