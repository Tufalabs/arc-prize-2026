from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_skewer_module():
    path = Path(__file__).resolve().parents[1] / "re_arc" / "environment_files" / "skewer" / "0001" / "skewer.py"
    spec = importlib.util.spec_from_file_location("test_skewer_env", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vertical_skewer_moves_left_and_right_on_arrow_actions() -> None:
    env = _load_skewer_module()

    for level_idx in (6, 7):
        for delta, expected in [((-1, 0), (-1, 0)), ((1, 0), (1, 0))]:
            game = env.Skewer()
            game.on_set_level(game._levels[level_idx])
            game.selected_id = "V"
            before = game.skewers["V"].handle

            assert game._try_move_selected(delta) is True
            assert game.skewers["V"].handle == (before[0] + expected[0], before[1] + expected[1])
