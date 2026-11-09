"""``tools/framediff_corpus.py``: the animation frames a move keeps, as the kernel's ``history`` keeps them.

What is pinned: a move that won a level (not the game) ends its turn on the win screen, the frame before the
new level's opening, so its animation is what came before the win screen and the opening is the next move's
board (``platypus.contract.partition``); a move that won the game, or none, is not split.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools" / "framediff_corpus.py"


def _tool() -> Any:
    """The corpus tool, loaded from its file."""
    spec = importlib.util.spec_from_file_location("framediff_corpus", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def _board(value: int) -> np.ndarray:
    """A board of one colour."""
    return np.full((4, 4), value, dtype=np.int8)


def _state(frames: list[int], completed: int, *, won_level: bool = False, won: bool = False) -> SimpleNamespace:
    """A recorded state: the frames the engine sent for the move, each a board of one colour."""
    return SimpleNamespace(
        raw=SimpleNamespace(frame=[_board(value) for value in frames], levels_completed=completed),
        previous_action=SimpleNamespace(id=SimpleNamespace(name="ACTION1")),
        just_won_level=won_level,
        won=won,
    )


def test_a_winning_move_animates_up_to_its_win_screen() -> None:
    game = _tool()._Game(
        [
            _state([1], 0),
            _state([2, 3], 0),  # a plain move: one frame, then its settled board
            _state([4, 5, 5, 9], 1, won_level=True),  # a frame, the win screen twice, the next level's opening
            _state([5, 5, 9], 2, won_level=True),  # the win screen alone: no animation
            _state([7, 8], 3, won=True),  # the game won: no split
        ]
    )
    assert [[int(frame[0, 0]) for frame in game.kept(index)] for index in range(1, 5)] == [[2], [4], [], [7]]
    assert int(game.settled[2][0, 0]) == 9, "the next move starts from the new level's opening"
