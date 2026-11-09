from __future__ import annotations

from ..core import DslAgent, observation_level_index

LEVEL_PROGRAMS = (
    "RRLLLLDDL",
    "RDDRRUULLLDD",
    "RDUUDUUURRDDDLLR",
    "LDRURRRULUDDDDDLURRRRUULLLU",
    "RRRRLLLLLLDDRRUULRRRRRRDLUUUUUDUURRDDRDD",
    "RDDRRDDUUUURRLLLLLUUDUURRLLLLLUUDDDDDRRRRUUDDD",
    "RRRRLLLLLLDDRRUULRRRRRRDLUUUUUDUURRDDRDDDDDLDDUUUUULRRU",
)

ACTION_IDS = {"U": 1, "D": 2, "L": 3, "R": 4}


class CompassWeaveDslAgent(DslAgent):
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
            raise RuntimeError("Compass Weave observation has no level progress.")
        self.mark_levels_solved(level_index)
        if level_index != self._level_index:
            self._level_index = level_index
            self._action_index = 0
        program = LEVEL_PROGRAMS[level_index]
        if self._action_index >= len(program):
            raise RuntimeError("Compass Weave DSL program exhausted before level completion.")
        action_id = ACTION_IDS[program[self._action_index]]
        self._action_index += 1
        return action_id, {}
