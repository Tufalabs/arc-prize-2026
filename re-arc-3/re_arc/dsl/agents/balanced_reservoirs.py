from __future__ import annotations

import gc

from ..core import CachedProgramDslAgent

PROGRAMS = (
    (("AB", 1),),
    (("CL", 3), ("CR", 3)),
    (("AS", 4), ("SB", 2)),
    (("HG", 7), ("GF", 6), ("FE", 5), ("ED", 4), ("DC", 3), ("CA", 1), ("CB", 1)),
    (
        ("SC1", 7),
        ("C1C2", 7),
        ("C2C3", 7),
        ("C3J", 7),
        ("JA1", 2),
        ("A1A2", 1),
        ("JB1", 4),
        ("B1B2", 3),
        ("B2B3", 2),
        ("B3B4", 1),
    ),
    (("AB", 4), ("BD", 8), ("DC", 8), ("CG", 10), ("GE", 2), ("GH", 6), ("HI", 4), ("IJ", 2)),
    (
        ("a1a2", 4),
        ("a2a3", 7),
        ("a4a3", 1),
        ("b1b3", 2),
        ("a3b3", 10),
        ("b3b2", 10),
        ("b2b4", 8),
        ("b4b5", 6),
        ("b5b6", 2),
        ("b5b7", 2),
    ),
)


class BalancedReservoirsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)

    def reset_episode(self):
        super().reset_episode()
        # Multi-frame observations contain many short-lived nested lists. Collect
        # between episodes so a host's older garbage cannot pause a later action.
        gc.collect()

    def _build_level_program(self, env):
        level_idx = int(env._game.level_index)
        centers = env._game.link_centers
        program = []
        for link_id, repeats in PROGRAMS[level_idx]:
            x, y = centers[link_id]
            program.extend((6, {"x": int(x), "y": int(y)}) for _ in range(repeats))
        return program


AGENT_CLASS = BalancedReservoirsDslAgent
