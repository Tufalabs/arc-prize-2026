from __future__ import annotations

import gc

from ..core import CachedProgramDslAgent

S = 5

PROGRAMS = (
    (S, 1, 1),
    (S, 4, 4, S, 1),
    (4, 4, S, 1, 1, 1),
    (S, 4, 4, 4, 4, 4, S, 2, 3, 3, 1, 3, 3, 3, S, 1, 4, 4, 4, 4, 4, 4, 4, S, 1, 1),
    (4, 1, 1, 4, 4, 4, 4, 4, 1, S, 2, 3, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4, 4, 4, S, 3, 1, 1, 1, 4, 4, 4),
    (
        S,
        1,
        1,
        1,
        3,
        1,
        1,
        4,
        4,
        S,
        3,
        3,
        2,
        2,
        S,
        3,
        1,
        S,
        2,
        4,
        4,
        2,
        2,
        2,
        S,
        4,
        1,
        4,
        4,
        4,
        1,
        4,
        S,
        1,
        1,
        3,
        3,
        1,
        3,
        3,
        3,
        3,
        2,
        2,
        3,
        1,
        3,
        3,
        3,
    ),
    (
        S,
        1,
        1,
        1,
        3,
        1,
        1,
        1,
        3,
        3,
        S,
        4,
        4,
        4,
        4,
        S,
        3,
        3,
        2,
        2,
        S,
        3,
        1,
        S,
        2,
        4,
        2,
        4,
        2,
        2,
        2,
        S,
        4,
        1,
        4,
        4,
        4,
        1,
        1,
        1,
        4,
        S,
        1,
        3,
        3,
        1,
        3,
        3,
        3,
        3,
        2,
        2,
        3,
        1,
        3,
        3,
        1,
        1,
        1,
    ),
)


class TakeItBackDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id, total_levels=7)

    def _build_level_program(self, env):
        del env
        assert self._current_level_idx is not None
        return [(action, {}) for action in PROGRAMS[self._current_level_idx]]

    def next_action(self, env, observation):
        # Frame payloads contain many short-lived Python lists.  Reclaim them
        # between inputs so collection never interrupts an animated game step.
        gc.collect()
        return super().next_action(env, observation)


AGENT_CLASS = TakeItBackDslAgent
