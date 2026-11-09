"""Property test for Level 6 ("Misdirection").

The intended solution puts both turrets on the central corner tiles
(9,13)/(11,13) — the only spot that reaches the column both movers must climb
to the goal. With every right-side gate held open, exactly ONE configuration of
the left-side gates wins, so the left half has a unique solution (no other left
gate configuration clears the movers).

Run with:  uv run python re_arc/environment_files/jc26/0001/test_level6.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("jc26_level6", _HERE / "jc26.py")
tb = importlib.util.module_from_spec(_spec)
sys.modules["jc26_level6"] = tb
_spec.loader.exec_module(tb)

LEVEL_INDEX = 5  # level 6 is the sixth entry in levels.json
TURRETS = [(9, 13), (11, 13)]  # the two central corner tiles (the intended placement)


def _level6_game():
    lv = json.loads((_HERE / "levels.json").read_text(encoding="utf-8"))[LEVEL_INDEX]
    spec = tb.LevelSpec(name=lv["name"], rows=tuple(lv["grid"]), lives=int(lv.get("lives", tb.DEFAULT_LIVES)))
    g = tb.JC26(specs=[spec])
    g.handle_reset()
    return g


def _single_tiles(g):
    # turret-placeable cells fully ringed by gate slots (two per side)
    nbr = ((1, 0), (-1, 0), (0, 1), (0, -1))
    return [t for t in g.high if all((t[0] + dx, t[1] + dy) in g.slots for dx, dy in nbr)]


def _wins(g, turrets, open_gates):
    g.blockers = set(g.slots) - set(open_gates)
    ot = g._open_tiles()
    dist = tb.dist_field(g.target, ot)
    paths = {i: tb.greedy_path(sp, g.target, dist, ot) for i, sp in enumerate(g.spawns)}
    if any(p is None for p in paths.values()):
        return False
    return tb.simulate_round(g.spawns, paths, turrets, hp=g.lives)[1] == "win"


def test_unique_left_gate_solution_with_right_open():
    g = _level6_game()
    assert all(t in g.high for t in TURRETS), ("turret tiles must be placeable", TURRETS)
    left = sorted(t for t in g.slots if t[0] < g.target[0])
    right = [t for t in g.slots if t[0] > g.target[0]]
    # every right gate held open; sweep all left-gate combinations
    wins = sum(
        1
        for k in range(len(left) + 1)
        for op in itertools.combinations(left, k)
        if _wins(g, TURRETS, set(op) | set(right))
    )
    assert wins == 1, ("exactly one left-gate configuration should win (right held open)", wins)


def test_no_symmetric_single_tile_solution():
    # the "misdirection": putting one turret on a single tile each side never
    # works. Try both symmetric pairings (both outer, both inner) and, for each,
    # every symmetric gate configuration (left subset mirrored onto the right).
    g = _level6_game()

    def mirror(t):
        return (2 * g.target[0] - t[0], t[1])

    left_singles = sorted(t for t in _single_tiles(g) if t[0] < g.target[0])
    assert len(left_singles) == 2, ("expected an outer and an inner single tile on the left", left_singles)
    left = sorted(t for t in g.slots if t[0] < g.target[0])
    wins = sum(
        1
        for left_tile in left_singles
        for k in range(len(left) + 1)
        for op in itertools.combinations(left, k)
        if _wins(g, [left_tile, mirror(left_tile)], set(op) | {mirror(t) for t in op})
    )
    assert wins == 0, ("a symmetric single-tile placement should never win", wins)


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
