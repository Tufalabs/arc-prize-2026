from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_sliders_module():
    path = Path(__file__).resolve().parents[1] / "re_arc" / "environment_files" / "sliders" / "0001" / "sliders.py"
    spec = importlib.util.spec_from_file_location("test_sliders_env", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_child_cap_does_not_hide_parent_marker_tip() -> None:
    env = _load_sliders_module()
    game = env.Sliders()
    game.on_set_level(game._levels[7])

    game._compute_world()
    marker_tip = game.world["C"]["tip"]
    px, py = env._cell_center(marker_tip)

    assert game.canvas.pixels[py, px] == env.COLOR_MARKER
