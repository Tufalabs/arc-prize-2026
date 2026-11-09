from __future__ import annotations

from ..core import DslAgent, observation_level_index

_RAW_PROGRAMS = (
    "RURRRR",
    "DRULURURRR",
    "URDDRRUUDDLLLDL",
    "DRUUURUUURRDLULDDDDDDLDDRRURRUUDDLLLLDLL",
    "RRDRUUDRUUUURUURRLLLDDDDDDLDRDRRUUURDDDLDDRR",
    "ULLLUUUURRRDDLLLDDDRRRRRDRRDRRUUUDRURRUULLDDDDDRDDDLLDLL",
    "ULLLUURRRLLLDDDRRRRRURRURRRRRDLLULDLDDRDDDDDLLRRRDLLLLLLLLDLL",
)
_ACTION = {"U": 1, "D": 2, "L": 3, "R": 4}
LEVEL_PROGRAMS = tuple(tuple(_ACTION[ch] for ch in replay) for replay in _RAW_PROGRAMS)


class FuseDoorsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)
        self._level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._level_idx = None
        self._action_idx = 0

    def next_action(self, env, observation):
        del env
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Fuse Doors observation is missing levels_completed")
        if self._level_idx != level_idx:
            self._level_idx = level_idx
            self._action_idx = 0
        program = LEVEL_PROGRAMS[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"Fuse Doors program exhausted on level {level_idx + 1}")
        action = program[self._action_idx]
        self._action_idx += 1
        return action, {}


AGENT_CLASS = FuseDoorsDslAgent
