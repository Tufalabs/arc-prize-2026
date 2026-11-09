from __future__ import annotations

from ..core import DslAgent, observation_level_index

CELL = 4
ORIGIN = 4

# (emitter cell, clockwise click count), in the explanatory order from the spec.
PROGRAMS = (
    (((3, 7), 1),),
    (((7, 8), 3), ((2, 8), 3)),
    (((2, 7), 1), ((12, 11), 3), ((6, 11), 2)),
    (((4, 2), 3), ((1, 7), 3), ((12, 4), 3), ((7, 1), 3)),
    (((1, 9), 3), ((2, 3), 3), ((10, 1), 3), ((10, 12), 3), ((7, 10), 3), ((4, 4), 3), ((7, 4), 3)),
    (
        ((10, 3), 3),
        ((10, 1), 3),
        ((12, 1), 3),
        ((10, 8), 3),
        ((5, 5), 3),
        ((8, 5), 3),
        ((12, 10), 3),
        ((1, 10), 3),
        ((3, 8), 3),
        ((3, 12), 3),
    ),
    (
        ((12, 1), 3),
        ((5, 11), 3),
        ((3, 11), 3),
        ((9, 1), 3),
        ((7, 5), 3),
        ((1, 5), 3),
        ((5, 1), 3),
        ((7, 3), 3),
        ((12, 12), 3),
        ((11, 7), 3),
        ((11, 11), 3),
        ((9, 12), 3),
        ((1, 11), 3),
        ((9, 7), 3),
    ),
)


def _expand(program):
    actions = []
    for (x, y), count in program:
        actions.extend([(6, {"x": ORIGIN + CELL * x + 2, "y": ORIGIN + CELL * y + 2})] * count)
    return actions


class BeamCompassDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(PROGRAMS))
        self._level = None
        self._index = 0

    def reset_episode(self):
        super().reset_episode()
        self._level = None
        self._index = 0

    def next_action(self, _env, observation):
        level = observation_level_index(observation, self.total_levels)
        if level is None:
            raise RuntimeError("Beam Compass observation has no level progress.")
        self.mark_levels_solved(level)
        if level != self._level:
            self._level = level
            self._index = 0
        actions = _expand(PROGRAMS[level])
        if self._index >= len(actions):
            raise RuntimeError("Beam Compass solver exhausted before WIN.")
        action = actions[self._index]
        self._index += 1
        return action
