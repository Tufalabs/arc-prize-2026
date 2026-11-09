from __future__ import annotations

from ..core import DslAgent, observation_level_index

PROGRAM_STRINGS = (
    "DDLSDLDD",
    "LLDLLDRSDRDDDRR",
    "RRDSLLLDSLLDDDDRRRULSDDRRD",
    "DLLSDLDDDDRRRUSDRRULSSURRR",
    "SDDLLULULUUURRURRDRRRRSLLLLULLDRSDDRSUULLLDDDRDRDRRRRRURRDD",
    "RDDLLULUUUUUURRSURRRRRRDSLLLLLLDLLDDDDDDRDRRUUURRRSLLLUSRRRDDRRD",
    "RDDLLULUUUUUURURRDRRURRRRDRSULLLLDLLULLDRSLLDDDDDDRDRRUUURRSLLUSRRDDRRRRRD",
)

ACTION_BY_SYMBOL = {"U": 1, "D": 2, "L": 3, "R": 4, "S": 5}
LEVEL_PROGRAMS = [[ACTION_BY_SYMBOL[symbol] for symbol in program] for program in PROGRAM_STRINGS]


class PrismPathsDslAgent(DslAgent):
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
            if self._current_level_idx != level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in Prism Paths observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Prism Paths DSL program exhausted before reaching WIN.")
        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}
