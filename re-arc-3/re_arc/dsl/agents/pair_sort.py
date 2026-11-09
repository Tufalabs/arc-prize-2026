from __future__ import annotations

from ..core import CachedProgramDslAgent

LEVEL_ROUTES = (
    ("x",),
    ("L", "R", "D", "L"),
    ("c0", "c1", "c3", "c2", "c5", "c0", "c1", "c4", "c5"),
    tuple("e6 e5 e4 e3 e2 e0 e1 b e2 e3 e4 e5 e6 e1 e2 e3 e4 e5 e1 e2 e3 e4 e1 e2 e3 e1 e2 e1 e0".split()),
    tuple(
        "e6 e5 e4 e3 e2 b e1 e0 e7 e6 e5 e4 e3 e2 e1 b "
        "e7 e6 e5 e4 e3 e2 e7 e6 e5 e4 e3 e7 e6 e5 e4 e7 e6 e5 e7 e6 e7".split()
    ),
    tuple(
        "e5 e4 e2 e3 e4 e5 e7 e6 e5 e4 e3 e2 e0 e1 e0 e2 e3 e4 e5 e6 e7 e8 "
        "e7 e6 e5 e4 e3 e2 b e1 b e2 e3 e4 e5 e6 e7 e6 e5 e4 e2 e3 e2 e4 e5 e4".split()
    ),
    tuple(
        "e3 e2 e1 e3 e4 e5 e6 bR e7 e6 e4 e3 e5 e4 e3 e2 e0 e1 e0 e2 e3 e4 e5 e4 e6 "
        "e8 e7 bR e6 e2 e3 e5 e4 e3 e2 bL e1 bL e2 e3 e4 e5 e6 e7 e6 e8 e4 e5 e4 e3 e2".split()
    ),
)


class PairSortDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_ROUTES))

    def _build_level_program(self, env):
        level_idx = self._current_level_idx
        game = env._game
        centers = {}
        for name, _a, _b, (x0, y0, x1, y1) in game.edges:
            centers[name] = ((x0 + x1) // 2, (y0 + y1) // 2)
        return [(6, {"x": centers[name][0], "y": centers[name][1]}) for name in LEVEL_ROUTES[level_idx]]


AGENT_CLASS = PairSortDslAgent
