from __future__ import annotations

from ..core import DslAgent, observation_level_index

PROGRAMS = (
    "RURD",
    "URDDLDRRDRU",
    "ULURR",
    "URDDDDDLDRRRRRDRUUUUURULLL",
    "RURDRDLDLULURRRRDRRUULDLDRURDLDRURDDDLDDRRULLLLLL",
    "DRRUULLDRRURDDDLDDRRULULDUUUULUURDDLDRRRDRULUURDLDR",
    "DRRUUUDLLDRRRRDRULUULUULDDDDDLDRURDLDDRRULDLUUUUURURRDLDDRRULLLULDDDDUUUULUURDDLDRRRR",
)
ACTION_BY_LETTER = {"U": 1, "D": 2, "L": 3, "R": 4}


class TumblingBlocksDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(PROGRAMS))
        self._level_index: int | None = None
        self._action_index = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._level_index = None
        self._action_index = 0

    def next_action(self, _env, observation):
        level_index = observation_level_index(observation, self.total_levels)
        if level_index is None:
            raise RuntimeError("Tumbling Blocks observation does not expose level progress.")
        self.mark_levels_solved(level_index)
        if self._level_index != level_index:
            self._level_index = level_index
            self._action_index = 0
        program = PROGRAMS[level_index]
        if self._action_index >= len(program):
            raise RuntimeError("Tumbling Blocks DSL program exhausted before level completion.")
        action = ACTION_BY_LETTER[program[self._action_index]]
        self._action_index += 1
        return action, {}
