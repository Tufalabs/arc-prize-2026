from __future__ import annotations

from ..core import CachedProgramDslAgent

PALETTE_CLICKS = {9: {"x": 4, "y": 4}, 12: {"x": 11, "y": 4}, 15: {"x": 18, "y": 4}}

RIGHT_BOARD_CLICKS = {
    (4, 1): {"x": 55, "y": 21},
    (4, 2): {"x": 55, "y": 26},
    (3, 2): {"x": 50, "y": 26},
    (3, 1): {"x": 50, "y": 21},
    (2, 2): {"x": 45, "y": 26},
    (3, 3): {"x": 50, "y": 31},
    (2, 3): {"x": 45, "y": 31},
    (1, 3): {"x": 40, "y": 31},
    (1, 2): {"x": 40, "y": 26},
    (2, 4): {"x": 45, "y": 36},
    (1, 4): {"x": 40, "y": 36},
    (2, 5): {"x": 45, "y": 41},
    (1, 5): {"x": 40, "y": 41},
}


class MirrorMatchDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "mirror_match-0001") -> None:
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        _ = env
        level_idx = int(self._current_level_idx or 0)
        if level_idx == 0:
            return [
                (6, RIGHT_BOARD_CLICKS[(4, 1)]),
                (6, RIGHT_BOARD_CLICKS[(4, 2)]),
                (6, RIGHT_BOARD_CLICKS[(3, 2)]),
                (1, {}),
            ]
        if level_idx == 1:
            return [
                (6, RIGHT_BOARD_CLICKS[(4, 1)]),
                (6, RIGHT_BOARD_CLICKS[(2, 2)]),
                (6, RIGHT_BOARD_CLICKS[(3, 3)]),
                (6, RIGHT_BOARD_CLICKS[(1, 3)]),
                (6, PALETTE_CLICKS[12]),
                (6, RIGHT_BOARD_CLICKS[(3, 1)]),
                (6, RIGHT_BOARD_CLICKS[(3, 2)]),
                (6, RIGHT_BOARD_CLICKS[(2, 3)]),
                (1, {}),
            ]
        return [
            (6, RIGHT_BOARD_CLICKS[(3, 1)]),
            (6, RIGHT_BOARD_CLICKS[(1, 2)]),
            (6, RIGHT_BOARD_CLICKS[(2, 4)]),
            (6, RIGHT_BOARD_CLICKS[(2, 5)]),
            (6, PALETTE_CLICKS[12]),
            (6, RIGHT_BOARD_CLICKS[(3, 2)]),
            (6, RIGHT_BOARD_CLICKS[(1, 3)]),
            (6, RIGHT_BOARD_CLICKS[(1, 5)]),
            (6, PALETTE_CLICKS[15]),
            (6, RIGHT_BOARD_CLICKS[(4, 1)]),
            (6, RIGHT_BOARD_CLICKS[(2, 2)]),
            (6, RIGHT_BOARD_CLICKS[(1, 4)]),
            (1, {}),
        ]


AGENT_CLASS = MirrorMatchDslAgent
