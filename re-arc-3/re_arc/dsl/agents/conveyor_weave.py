from __future__ import annotations

from ..core import DslAgent, observation_level_index

_ACTIONS = {"U": 1, "D": 2, "L": 3, "R": 4, "S": 5}
_PROGRAM_TEXT = (
    "R",
    "URSSR",
    "URRUSU",
    "RDUSSSDDLRSSR UUULLSSSL".replace(" ", ""),
    "LUUURSSSRDDSSURSSSRLDDDRRUDSSSURSRLDDDRR",
    "SSRRRRDSLLULUDSSDLUUURRSSSRRLSDSSSDLUUURRR",
    "SSRRRRDDRSSUULLLULUSURRRRRRSSSDSRRSSRDSSDLDDDRSSSR",
)
LEVEL_PROGRAMS = [[_ACTIONS[char] for char in text] for text in _PROGRAM_TEXT]


class ConveyorWeaveDslAgent(DslAgent):
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
            raise RuntimeError("Missing levels_completed in Conveyor Weave observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Conveyor Weave DSL program exhausted before WIN.")
        action = program[self._action_idx]
        self._action_idx += 1
        return action, {}
