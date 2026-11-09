"""Puzzle-property test for Level 4 ("Grid").

The three movable turrets belong on the "single tiles" — the high-ground cells
embedded in the gate grid (each surrounded on all four sides by gate slots).
With the turrets there, exactly ONE configuration of the gates routes the movers
so that all of them are destroyed.

Run with:  uv run python re_arc/environment_files/jc26/0001/test_level4.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("jc26_level4", _HERE / "jc26.py")
tb = importlib.util.module_from_spec(_spec)
sys.modules["jc26_level4"] = tb
_spec.loader.exec_module(tb)

LEVEL_INDEX = 3  # level 4 is the fourth entry in levels.json


def _level4_game():
    lv = json.loads((_HERE / "levels.json").read_text(encoding="utf-8"))[LEVEL_INDEX]
    spec = tb.LevelSpec(name=lv["name"], rows=tuple(lv["grid"]), lives=int(lv.get("lives", tb.DEFAULT_LIVES)))
    g = tb.JC26(specs=[spec])
    g.handle_reset()
    return g


def _single_tiles(g):
    # the "single tiles": turret-placeable cells fully ringed by gate slots
    return sorted(
        t for t in g.high if all((t[0] + dx, t[1] + dy) in g.slots for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    )


def _winning_gate_configs(g, turret_tiles):
    gates = sorted(g.slots)
    allg = set(gates)
    wins = []
    for k in range(len(gates) + 1):
        for opened in itertools.combinations(gates, k):
            g.blockers = allg - set(opened)
            ot = g._open_tiles()
            dist = tb.dist_field(g.target, ot)
            paths = {i: tb.greedy_path(sp, g.target, dist, ot) for i, sp in enumerate(g.spawns)}
            if any(p is None for p in paths.values()):
                continue
            if tb.simulate_round(g.spawns, paths, turret_tiles, hp=g.lives)[1] == "win":
                wins.append(set(opened))
    return wins


def test_unique_winning_gate_config_with_turrets_on_single_tiles():
    g = _level4_game()
    tiles = _single_tiles(g)
    assert len(tiles) == 3, ("expected three single tiles for the three turrets", tiles)
    wins = _winning_gate_configs(g, tiles)
    assert len(wins) == 1, ("exactly one gate configuration should win", len(wins), wins)


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
