from __future__ import annotations

from ..core import DslAgent, observation_level_index


def _actions(program: str) -> list[int]:
    action_ids = {"U": 1, "D": 2, "L": 3, "R": 4}
    return [action_ids[token] for token in program.split()]


LEVEL_PROGRAMS = (
    _actions("R R R L L U D U D"),
    _actions("R R R L U U D D R L R L"),
    _actions("R L R R D U U U U L U D D D D L R D U"),
    _actions("D U R L L R R U U L R R U D U D D U D L R L U U D D U D"),
    _actions("L L L R R L L L R L U D D D R L L L L D U R L R R L U U D U U L R L R D U D U R L R L"),
    _actions(
        "U R L U D U D D U D R L R D D U D D L R U D R L R L R L L L U D D D R L R L L R L U D U U L L D U U R L R L"
    ),
    _actions(
        "U U U U D U R R L R L D D U D U L L R R U U D L R L R R L R L R R L "
        "D U D U U U D U L R R L U D U D D U D L L R D U D R L R L"
    ),
)


class VectorWakeDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._level_index: int | None = None
        self._action_index = 0

    def reset_episode(self):
        super().reset_episode()
        self._level_index = None
        self._action_index = 0

    def next_action(self, _env, observation):
        level_index = observation_level_index(observation, self.total_levels)
        if level_index is None:
            raise RuntimeError("Missing levels_completed in Vector Wake observation.")
        self.mark_levels_solved(level_index)
        if level_index != self._level_index:
            self._level_index = level_index
            self._action_index = 0
        program = LEVEL_PROGRAMS[level_index]
        if self._action_index >= len(program):
            raise RuntimeError(f"Vector Wake level {level_index + 1} program exhausted before completion.")
        action = program[self._action_index]
        self._action_index += 1
        return action, {}


AGENT_CLASS = VectorWakeDslAgent
