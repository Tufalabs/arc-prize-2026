from __future__ import annotations

from ..core import DslAgent, observation_level_index


def _program(text: str) -> list[int]:
    actions = {"U": 1, "D": 2, "L": 3, "R": 4, "S": 5}
    return [actions[token] for token in text if token in actions]


LEVEL_PROGRAMS = [
    _program("RSRR"),
    _program("RRRS RR S RR S RR"),
    _program("RS RRRR S LLLLLL"),
    _program("RS RS RRRRS LLLULS DS RS ULS DS RS RRDRRDDSL"),
    _program("RSRS RRRDRS DDDDS UUUUULLLLS DDDDDRRRUUURRS DDDLS UUURRRUUURSUL"),
    _program("RSRS UUUULLS DDDDRRRRRRRRS UUUUULS DDDDDLLLLLS UUUUURRRRRDDS UUS DDRRR"),
    _program("RSRS RRRDRS DDDDS UUUUULLLLS DDDDDRRRURRS DLSURRRUUURS LDDDLLUSDSDSUUSDSDS URRDRS ULLLS UUUUURR"),
]


class PressureRelayDslAgent(DslAgent):
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
            raise RuntimeError("Missing level progress in Pressure Relay observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Pressure Relay DSL program exhausted before reaching WIN.")
        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}
