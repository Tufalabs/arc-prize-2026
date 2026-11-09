from __future__ import annotations

from ..core import DslAgent, observation_level_index


def _program(trace: str) -> list[int]:
    actions = {"U": 1, "D": 2, "L": 3, "R": 4, "S": 5}
    return [actions[token] for token in trace.replace("|", " ").split()]


LEVEL_PROGRAMS = [
    _program("S | R S | R R R"),
    _program("R S | R R R R S | R U S | R D D"),
    _program("R S | R U S | D R R S | L L L L D"),
    _program("U S | D R R U S | D L L D S | U R R U R R S | L L S | R R R S | L S | D D S | L D S | U U R R R"),
    _program(
        "U U U S | D D L L U S | L S | D R R R R S | L L L D L S | U R R R S | "
        "L L U S | D R R U R S | D S | R R R U S | D L L L L S | R R R R D R"
    ),
    _program(
        "U S | D R R U S | D L L D S | U R R R R S | L U L S | D R R R U S | "
        "D L L D S | U R R R S | L L S | R S | U S | D D S | U S | D R D R S | "
        "L U U S | R R D"
    ),
    _program(
        "U S | D D R R U S | D L L D S | U R R R D D R S | D S | U L U U R R D S | "
        "U L L D D R S | L U U U R R S | L L L S | R R R R R S | L L S | D D R S | "
        "L S | R R R U"
    ),
]


class OneHandCourierDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is not None:
            self.mark_levels_solved(level_idx)
            if level_idx != self._current_level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing levels_completed in One-Hand Courier observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("One-Hand Courier DSL program exhausted before WIN.")
        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}
