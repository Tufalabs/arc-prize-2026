"""Unit tests for the pure beam resolver in talos_beams.py.

Run with:  uv run python re_arc/environment_files/talos_beams/0001/test_resolver.py
(loads the game module by path so no package install is needed).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("talos_beams_mod", _HERE / "talos_beams.py")
tb = importlib.util.module_from_spec(_spec)
sys.modules["talos_beams_mod"] = tb  # dataclasses needs this to resolve string annotations
_spec.loader.exec_module(tb)

NodeState = tb.NodeState
resolve = tb.resolve_beams


def _e(*pairs):
    return {frozenset(p) for p in pairs}


def test_simple_chain():
    # source R --conn-- target R, colinear, clear
    nodes = {
        0: NodeState("source", "R", 0, 0),
        1: NodeState("connector", None, 2, 0),
        2: NodeState("target", "R", 4, 0),
    }
    res = resolve(nodes, _e((0, 1), (1, 2)), set(), 0, 0)
    assert res.powered.get(1) == "R", res.powered
    assert res.powered.get(2) == "R", res.powered
    assert all(b.reached for b in res.beams), res.beams


def test_white_wall_blocks():
    nodes = {
        0: NodeState("source", "R", 0, 0),
        1: NodeState("connector", None, 2, 0),
        2: NodeState("target", "R", 4, 0),
    }
    res = resolve(nodes, _e((0, 1), (1, 2)), {(3, 0)}, 0, 0)
    assert res.powered.get(1) == "R", res.powered
    assert 2 not in res.powered, "wall should stop conn->target"


def test_wall_adjacent_to_target_does_not_graze_block():
    # A right-facing target with a wall directly to its LEFT (a different side
    # than the open, facing side). A beam approaching the facing edge from below
    # must connect — the float edge point sits on the open boundary, so the wall
    # can't graze-block it (the old interior-nub point did, via a corner graze).
    nodes = {
        0: NodeState("source", "R", 1, 6),
        1: NodeState("target", "R", 4, 4, rot=3),  # faces right; open side is (5,4)
    }
    res = resolve(nodes, _e((0, 1)), {(3, 4)}, 2, 2)  # wall left of the target
    assert res.powered.get(1) == "R", ("adjacent wall must not block the facing edge", res.powered)


def test_boundary_touch_counts_as_connected():
    # Right-facing target with a wall directly to its RIGHT: the connection edge
    # point lies exactly on that wall's boundary. Float-equality at the boundary
    # must count as connected (clip shapes are inset), so the target powers.
    nodes = {
        0: NodeState("source", "R", 1, 4),
        1: NodeState("target", "R", 4, 4, rot=3),  # edge point == left boundary of (5,4)
    }
    res = resolve(nodes, _e((0, 1)), {(5, 4)}, 2, 2)
    assert res.powered.get(1) == "R", ("boundary touch must connect, not block", res.powered)


def test_grey_wall_transparent():
    # same geometry but the (3,0) tile is a GREY wall -> not in blocking set
    nodes = {
        0: NodeState("source", "R", 0, 0),
        1: NodeState("connector", None, 2, 0),
        2: NodeState("target", "R", 4, 0),
    }
    res = resolve(nodes, _e((0, 1), (1, 2)), set(), 0, 0)
    assert res.powered.get(2) == "R"


def test_crossing_earliest_wins():
    # beam A reaches the crossing at fraction 0.375, beam B at 0.5 -> A wins,
    # B is cut before reaching its connector.
    nodes = {
        0: NodeState("source", "R", 0, 0),
        1: NodeState("connector", None, 8, 8),
        2: NodeState("source", "G", 6, 0),
        3: NodeState("connector", None, 0, 6),
    }
    res = resolve(nodes, _e((0, 1), (2, 3)), set(), 0, 0)
    assert res.powered.get(1) == "R", ("A should reach connA", res.powered)
    assert 3 not in res.powered, ("B should be cut before connB", res.powered)


def test_head_on_both_stop():
    # two sources wired directly -> beams collide head-on at the midpoint
    nodes = {0: NodeState("source", "R", 0, 0), 1: NodeState("source", "R", 4, 0)}
    res = resolve(nodes, _e((0, 1)), set(), 0, 0)
    assert len(res.beams) == 2, res.beams
    for b in res.beams:
        assert not b.reached, "head-on beams must not reach"
        assert abs(b.live[0] - 12.5) < 0.6, ("cut at midpoint", b.live)


def test_two_colour_connector_dead():
    nodes = {
        0: NodeState("source", "R", 0, 0),
        1: NodeState("source", "B", 4, 0),
        2: NodeState("connector", None, 2, 2),
    }
    res = resolve(nodes, _e((0, 2), (1, 2)), set(), 0, 0)
    assert 2 in res.dead, res.dead
    assert 2 not in res.powered, res.powered


def test_lower_tier_beats_higher():
    # chain1: tier-1 horizontal beam.  chain2: tier-2 vertical beam crossing it.
    nodes = {
        0: NodeState("source", "R", 0, 2),
        1: NodeState("connector", None, 8, 2),
        2: NodeState("source", "G", 4, 0),
        3: NodeState("connector", None, 4, 1),
        4: NodeState("connector", None, 4, 4),
    }
    res = resolve(nodes, _e((0, 1), (2, 3), (3, 4)), set(), 0, 0)
    assert res.powered.get(1) == "R", res.powered
    assert res.powered.get(3) == "G", res.powered
    assert 4 not in res.powered, ("tier-2 beam cut by tier-1", res.powered)


def test_connector_blocks_unrelated_beam():
    # C1 sits on the straight path from the source to C2 but is NOT wired in -> blocks
    nodes = {
        0: NodeState("source", "R", 0, 0),
        1: NodeState("connector", None, 4, 0),  # in the way, unwired
        2: NodeState("connector", None, 8, 0),
    }  # wired to source
    res = resolve(nodes, _e((0, 2)), set(), 0, 0)
    assert 2 not in res.powered, ("unrelated connector should block source->C2", res.powered)


def test_connector_endpoint_not_self_blocked():
    # a beam wired straight into a connector still reaches it (own circle exempt)
    nodes = {0: NodeState("source", "R", 0, 0), 1: NodeState("connector", None, 4, 0)}
    res = resolve(nodes, _e((0, 1)), set(), 0, 0)
    assert res.powered.get(1) == "R", res.powered


def test_held_connector_inert():
    nodes = {
        0: NodeState("source", "R", 0, 0),
        1: NodeState("connector", None, 2, 0, True),  # held
        2: NodeState("target", "R", 4, 0),
    }
    res = resolve(nodes, _e((0, 1), (1, 2)), set(), 0, 0)
    assert 1 not in res.powered, res.powered
    assert 2 not in res.powered, res.powered
    assert not res.beams, ("held connector blocks all propagation", res.beams)


def test_inverter_blue_to_red():
    nodes = {0: NodeState("source", "B", 0, 0), 1: NodeState("inverter", None, 2, 0), 2: NodeState("target", "R", 4, 0)}
    res = resolve(nodes, _e((0, 1), (1, 2)), set(), 0, 0)
    assert res.powered.get(1) == "R", ("inverter emits red from blue", res.powered)
    assert res.powered.get(2) == "R", ("red target lit by inverter", res.powered)


def test_inverter_red_to_blue():
    nodes = {0: NodeState("source", "R", 0, 0), 1: NodeState("inverter", None, 2, 0), 2: NodeState("target", "B", 4, 0)}
    res = resolve(nodes, _e((0, 1), (1, 2)), set(), 0, 0)
    assert res.powered.get(2) == "B", res.powered


def test_inverter_green_does_nothing():
    nodes = {0: NodeState("source", "G", 0, 0), 1: NodeState("inverter", None, 2, 0), 2: NodeState("target", "G", 4, 0)}
    res = resolve(nodes, _e((0, 1), (1, 2)), set(), 0, 0)
    assert 1 not in res.powered, ("green is inert", res.powered)
    assert 2 not in res.powered, ("green is inert", res.powered)


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
