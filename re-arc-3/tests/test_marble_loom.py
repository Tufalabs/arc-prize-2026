from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_marble_loom_module():
    path = (
        Path(__file__).resolve().parents[1] / "re_arc" / "environment_files" / "marble_loom" / "0001" / "marble_loom.py"
    )
    spec = importlib.util.spec_from_file_location("test_marble_loom_env", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selected_side_falling_marble_settles_through_open_gap() -> None:
    env = _load_marble_loom_module()
    game = env.MarbleLoom()
    game.on_set_level(game._levels[3])

    blue_idx = next(idx for idx, marble in enumerate(game.marbles) if marble.color == "blue")
    game.selected_marble_idx = blue_idx
    blue = game.marbles[blue_idx]
    assert (blue.x, blue.y) == (1, 3)

    game.vertical[1] = "###.####"

    game._settle()

    assert (blue.x, blue.y) == (3, 3)
