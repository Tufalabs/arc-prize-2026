from __future__ import annotations

from ..core import CachedProgramDslAgent


def _click_cell(x: int, y: int) -> tuple[int, dict[str, int]]:
    return 6, {"x": 5 + 6 * int(x), "y": 5 + 6 * int(y)}


def _moves(text: str) -> list[tuple[int, dict[str, int]]]:
    action_ids = {"U": 1, "D": 2, "L": 3, "R": 4, "SP": 5}
    out: list[tuple[int, dict[str, int]]] = []
    for token in text.replace(",", " ").split():
        out.append((action_ids[token], {}))
    return out


LEVEL_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    _moves("R R SP U R R U R"),
    _moves("R R U SP R R D R R R"),
    _moves("U R U R D L D D SP U R R R R"),
    _moves("R SP U R R D R U R D D D D R U L U R U U SP L D R SP U R"),
    [
        *_moves("R SP U U"),
        _click_cell(4, 4),
        *_moves("R R R D R"),
        _click_cell(6, 4),
        *_moves("U D R D L D D D R U L U R R U SP L D R SP U R"),
    ],
    [
        *_moves("R SP U U U"),
        _click_cell(4, 4),
        *_moves("R R R D R"),
        _click_cell(6, 4),
        *_moves("U D L D D D R U U U L U"),
        _click_cell(6, 4),
        *_moves("R D R D R R SP D L SP D R SP U U U"),
    ],
    [
        *_moves("R SP U U U"),
        _click_cell(4, 4),
        *_moves("R R D R"),
        _click_cell(5, 4),
        *_moves("U D L L L D D D R R R U U U L U"),
        _click_cell(5, 4),
        *_moves("R D L L L D D D R R R U SP R D R R U R U SP D L D SP U R D SP U U U"),
    ],
    [
        *_moves("R SP U U U"),
        _click_cell(4, 4),
        *_moves("R R D R"),
        _click_cell(5, 4),
        *_moves("U D L L L D D D R R R U U U L U"),
        _click_cell(5, 4),
        *_moves("R D L L L D D D R R R U SP"),
        _click_cell(7, 7),
        *_moves("R R D R R U U SP D D SP L L SP R R U U U L U"),
    ],
]


class PocketElevatorsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env):
        level_idx = int(env._game.level_index)
        return LEVEL_PROGRAMS[level_idx]


AGENT_CLASS = PocketElevatorsDslAgent
