"""Property test for Level 1 ("Is it a maze game?").

Level 1's whole point: it looks like a maze, but opening every gate (the
trivial "just open all the blocks" move) sends the mover straight down the
maze, where the hidden turret cuts it down before the goal — a win. This test
asserts that opening all gates wins.

Run with:  uv run python re_arc/environment_files/jc26/0001/test_level1.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("jc26_level1", _HERE / "jc26.py")
tb = importlib.util.module_from_spec(_spec)
sys.modules["jc26_level1"] = tb
_spec.loader.exec_module(tb)

LEVEL_INDEX = 0  # level 1 is the first entry in levels.json


def _level1_game():
    lv = json.loads((_HERE / "levels.json").read_text(encoding="utf-8"))[LEVEL_INDEX]
    spec = tb.LevelSpec(name=lv["name"], rows=tuple(lv["grid"]), lives=int(lv.get("lives", tb.DEFAULT_LIVES)))
    g = tb.JC26(specs=[spec])
    g.handle_reset()
    return g


def test_wins_with_all_gates_open():
    g = _level1_game()
    g.blockers = set()  # open every gate
    ot = g._open_tiles()
    dist = tb.dist_field(g.target, ot)
    paths = {i: tb.greedy_path(sp, g.target, dist, ot) for i, sp in enumerate(g.spawns)}
    assert all(p is not None for p in paths.values()), "opening all gates should give every mover a path"
    outcome = tb.simulate_round(g.spawns, paths, [t["tile"] for t in g.turrets], hp=g.lives)[1]
    assert outcome == "win", ("opening all gates should win (turret kills the mover before the goal)", outcome)


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
