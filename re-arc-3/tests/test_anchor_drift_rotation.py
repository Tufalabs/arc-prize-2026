from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_anchor_drift_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "re_arc"
        / "environment_files"
        / "anchor_drift"
        / "0001"
        / "anchor_drift.py"
    )
    spec = importlib.util.spec_from_file_location("test_anchor_drift_env", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_group_rotates_when_anchor_touches_passive_block() -> None:
    env = _load_anchor_drift_module()

    for rel in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        anchor = (5, 5)
        hinge = (anchor[0] + rel[0], anchor[1] + rel[1])
        active = (anchor[0] + 2 * rel[0], anchor[1] + 2 * rel[1])
        delta = (rel[1], -rel[0])
        sailor = (active[0] - delta[0], active[1] - delta[1])
        expected = {
            "A": (anchor[0] + 2 * delta[0], anchor[1] + 2 * delta[1]),
            "B": (anchor[0] + delta[0], anchor[1] + delta[1]),
        }

        game = env.AnchorDrift.__new__(env.AnchorDrift)
        game._hud = env.AnchorDriftHud()
        game.sailor = sailor
        game.anchor_state = "dropped"
        game.anchor_pos = anchor
        game.requires_anchor = True
        game.reefs = set()
        game.color_pads = {}
        game.crates = {"A": env.Crate("A", 12, active, (0, 0), "G1"), "B": env.Crate("B", 11, hinge, (0, 0), "G1")}
        game.groups = {"G1": {"members": ["A", "B"], "active": "A"}}
        game.crate_animation_pixels = {}
        game._pivot_animation_frames = []
        game._pivot_final_positions = None
        game._pivot_final_sailor = None

        assert game._move_or_push(delta) is True
        assert game._pivot_final_positions == expected


def test_group_cannot_slide_into_anchor() -> None:
    env = _load_anchor_drift_module()

    game = env.AnchorDrift.__new__(env.AnchorDrift)
    game._hud = env.AnchorDriftHud()
    game.sailor = (7, 5)
    game.anchor_state = "dropped"
    game.anchor_pos = (4, 5)
    game.requires_anchor = False
    game.reefs = set()
    game.color_pads = {}
    game.crates = {"A": env.Crate("A", 12, (6, 5), (0, 0), "G1"), "B": env.Crate("B", 11, (5, 5), (0, 0), "G1")}
    game.groups = {"G1": {"members": ["A", "B"], "active": "A"}}
    game.crate_animation_pixels = {}
    game._pivot_animation_frames = []
    game._pivot_final_positions = None
    game._pivot_final_sailor = None

    assert game._move_or_push((-1, 0)) is False
    assert game.crates["A"].pos == (6, 5)
    assert game.crates["B"].pos == (5, 5)
    assert game.anchor_pos == (4, 5)


def test_anchored_group_prefers_rotation_over_slide() -> None:
    env = _load_anchor_drift_module()

    game = env.AnchorDrift.__new__(env.AnchorDrift)
    game._hud = env.AnchorDriftHud()
    game.sailor = (6, 4)
    game.anchor_state = "dropped"
    game.anchor_pos = (5, 5)
    game.requires_anchor = False
    game.reefs = set()
    game.color_pads = {}
    game.crates = {"A": env.Crate("A", 12, (6, 5), (0, 0), "G1"), "B": env.Crate("B", 11, (6, 6), (0, 0), "G1")}
    game.groups = {"G1": {"members": ["A", "B"], "active": "A"}}
    game.crate_animation_pixels = {}
    game._pivot_animation_frames = []
    game._pivot_final_positions = None
    game._pivot_final_sailor = None

    assert game._move_or_push((0, 1)) is True
    assert game._pivot_final_positions == {"A": (5, 6), "B": (4, 6)}
    assert game.crates["A"].pos == (6, 5)
    assert game.crates["B"].pos == (6, 6)
