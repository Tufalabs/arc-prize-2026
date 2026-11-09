from __future__ import annotations

from ..core import DslAgent, observation_level_index


def _moves(text: str) -> list[int]:
    action = {"U": 1, "D": 2, "L": 3, "R": 4}
    return [action[token] for token in text.split()]


LEVEL_PROGRAMS = [
    _moves("D R U U"),
    _moves("U R R D U L L D R R"),
    _moves("U U R R D D D L L U R R"),
    _moves("U L D D L L L U L L D R R R R R D R D L U U U U L D D D R D D L U U U U U"),
    _moves(
        "D D L L U L L D R R R D R U "
        "D L L L D U U R R R R R D R R U L L L L L L L "
        "U D R R R R R R U U L L L D L D R R R R"
    ),
    _moves(
        "D D D L L L L D D R R R R D U U L L L L L L U "
        "R R R R R D R U U U U L L L L L L L L D D D "
        "R R R R R R R U R D D D D"
    ),
    _moves(
        "D D L L L L U R R R R U U U D L L L L L D D L D "
        "R R R R R D R U U U R D D R D D "
        "U U L D L L L L D U R R R R U U L L L L U U L L L L L D D D "
        "R R R R R R R R D D D D L L L U U U U U U U"
    ),
]


class PaintedFootprintsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._level = None
        self._index = 0

    def reset_episode(self):
        super().reset_episode()
        self._level = None
        self._index = 0

    def next_action(self, _env, observation):
        level = observation_level_index(observation, self.total_levels)
        if level is None:
            raise RuntimeError("Missing levels_completed in Painted Footprints observation.")
        self.mark_levels_solved(level)
        if level != self._level:
            self._level = level
            self._index = 0
        program = LEVEL_PROGRAMS[level]
        if self._index >= len(program):
            raise RuntimeError("Painted Footprints program exhausted before WIN.")
        action = program[self._index]
        self._index += 1
        return action, {}
