from __future__ import annotations

from ..core import DslAgent, observation_level_index

PROGRAMS = (
    ("R",),
    ("R", "B"),
    ("B", "R", "G", "O"),
    tuple("MCGBOGPOMPCGOMP"),
    tuple("POGCM" * 4),
    tuple("ROGBY" + "MPOGC" * 2 + "MPOGBC" + "MPOGC" * 3 + "MPOG"),
    tuple("YOC" + "GMPOC" * 2 + "GBR" + "MPOGC" * 2 + "MPOGBC" + "MPOGC" * 3 + "MPOG"),
)


class ColorCensusDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)
        self._level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._level_idx = None
        self._action_idx = 0

    def next_action(self, env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Color Census observation has no level index")
        if level_idx != self._level_idx:
            self._level_idx = level_idx
            self._action_idx = 0
        program = PROGRAMS[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"Color Census program exhausted on level {level_idx + 1}")
        symbol = program[self._action_idx]
        self._action_idx += 1
        color = {"R": 8, "B": 9, "G": 14, "O": 12, "P": 15, "M": 6, "C": 10, "Y": 11}[symbol]
        candidates = [piece for piece in env._game.pieces if not piece.covered and piece.color == color]
        if not candidates:
            raise RuntimeError(f"No exposed {symbol} representative on level {level_idx + 1}")
        piece = min(candidates, key=lambda item: (item.row, item.col))
        return 6, {"x": 5 + 7 * piece.col, "y": 7 + 7 * piece.row}


AGENT_CLASS = ColorCensusDslAgent
