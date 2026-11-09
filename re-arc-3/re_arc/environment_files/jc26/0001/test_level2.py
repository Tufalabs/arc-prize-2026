"""Puzzle-property test for Level 2 (the "is it a maze game?" turret puzzle).

Checks, against the shipped levels.json, that exactly TWO gate configurations
win at the level's life count (the two symmetric left/right "U around the middle
turret" solutions).

Run with:  uv run python re_arc/environment_files/jc26/0001/test_level2.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("jc26_level2", _HERE / "jc26.py")
tb = importlib.util.module_from_spec(_spec)
sys.modules["jc26_level2"] = tb
_spec.loader.exec_module(tb)

LEVEL_INDEX = 1  # level 2 is the second entry in levels.json


def _level2_game():
    lv = json.loads((_HERE / "levels.json").read_text(encoding="utf-8"))[LEVEL_INDEX]
    spec = tb.LevelSpec(name=lv["name"], rows=tuple(lv["grid"]), lives=int(lv.get("lives", tb.DEFAULT_LIVES)))
    g = tb.JC26(specs=[spec])
    g.handle_reset()
    return g


def _path(g):
    ot = g._open_tiles()
    return tb.greedy_path(g.spawns[0], g.target, tb.dist_field(g.target, ot), ot)


def _winning_configs(g):
    """Every open/closed gate combo whose round ends in a win (mover destroyed)."""
    gates = sorted(g.slots)
    tiles = [t["tile"] for t in g.turrets]
    wins = []
    for k in range(len(gates) + 1):
        for opened in itertools.combinations(gates, k):
            g.blockers = set(gates) - set(opened)
            path = _path(g)
            if path is None:
                continue
            _ticks, outcome = tb.simulate_round(g.spawns, {0: path}, tiles, hp=g.lives)
            if outcome == "win":
                wins.append(set(opened))
    return wins


def test_exactly_two_winning_configurations():
    g = _level2_game()
    wins = _winning_configs(g)
    assert len(wins) == 2, ("exactly two gate configs should win (the symmetric U routes)", wins)


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
