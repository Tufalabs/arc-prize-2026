"""Door logic tests: targets referenced by ID, multiple targets per door, and
multiple same-colour targets driving different doors. Run with `uv run`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("talos_beams_mod", _HERE / "talos_beams.py")
tb = importlib.util.module_from_spec(_spec)
sys.modules["talos_beams_mod"] = tb
_spec.loader.exec_module(tb)

# Two R targets (t1@(7,2), t2@(7,4)) + one B target (t3@(7,5)); three doors.
GRID = (
    "############",
    "#@.........#",
    "#r.....R...#",
    "#..........#",
    "#r.....R...#",
    "#b.....B...#",
    "#.........1#",
    "#.........2#",
    "#.........3#",
    "#..........#",
    "#.........X#",
    "############",
)


def _game(doors):
    return tb.TalosBeams(specs=[tb.LevelSpec(name="t", rows=GRID, doors=doors, charge_full=3)])


def test_same_colour_targets_drive_distinct_doors():
    g = _game({"1": ["R1"], "2": ["R2"]})  # two red targets, different doors
    f = g.charge_full
    g.charge["R1"] = f
    g._sync_doors()
    assert g.doors["1"]["open"], g.doors
    assert not g.doors["2"]["open"], g.doors
    g.charge["R2"] = f
    g._sync_doors()
    assert g.doors["1"]["open"]
    assert g.doors["2"]["open"]


def test_multi_target_door_needs_all():
    g = _game({"3": ["R2", "B1"]})
    f = g.charge_full
    g.charge["R2"] = f
    g._sync_doors()
    assert not g.doors["3"]["open"], "door needs B1 too"
    g.charge["B1"] = f
    g._sync_doors()
    assert g.doors["3"]["open"]


def test_any_door_opens_on_either():
    g = _game({"1": {"any": ["R1", "R2"]}})  # OR: either red target full opens it
    f = g.charge_full
    g._sync_doors()
    assert not g.doors["1"]["open"], "neither full -> closed"
    g.charge["R2"] = f
    g._sync_doors()
    assert g.doors["1"]["open"], "R2 alone should open an 'any' door"
    g.charge["R2"] = 0
    g.charge["R1"] = f
    g._sync_doors()
    assert g.doors["1"]["open"], "R1 alone should open it too"


def test_all_explicit_matches_default():
    g = _game({"1": {"all": ["R1", "R2"]}})  # explicit AND == default list form
    f = g.charge_full
    g.charge["R1"] = f
    g._sync_doors()
    assert not g.doors["1"]["open"], "'all' needs both"
    g.charge["R2"] = f
    g._sync_doors()
    assert g.doors["1"]["open"]


def test_colour_shortcut_means_all_of_colour():
    g = _game({"1": "R"})  # both red targets must be full
    f = g.charge_full
    g.charge["R1"] = f
    g._sync_doors()
    assert not g.doors["1"]["open"], "colour shortcut requires every R target"
    g.charge["R2"] = f
    g._sync_doors()
    assert g.doors["1"]["open"]


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
