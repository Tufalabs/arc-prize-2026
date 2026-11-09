from __future__ import annotations

from ..core import CachedProgramDslAgent


def _trace(groups: tuple[str, ...]) -> list[int]:
    actions: list[int] = []
    action_for = {"U": 1, "D": 2, "L": 3, "R": 4, "X": 5}
    for group in groups:
        actions.extend(action_for[token] for token in group)
    return actions


LEVEL_PROGRAMS = (
    _trace(("X",)),
    _trace(("X", "RRX")),
    _trace(("X", "LX", "RRX")),
    _trace(("X", "UX", "DRX", "DLX", "ULX", "DDX", "RRX", "RX", "UX", "UUX")),
    _trace(
        (
            "X",
            "ULX",
            "DLX",
            "DRX",
            "LX",
            "DLX",
            "UUX",
            "UUX",
            "DX",
            "RX",
            "UX",
            "RX",
            "RRX",
            "DDX",
            "DDX",
            "LLX",
            "URX",
            "ULX",
            "URX",
        )
    ),
    _trace(
        (
            "X",
            "LLX",
            "DRRRX",
            "LX",
            "URX",
            "UUX",
            "RX",
            "DX",
            "DRX",
            "DDX",
            "DX",
            "LX",
            "UX",
            "LX",
            "DLX",
            "LX",
            "LX",
            "UX",
            "RX",
            "UUX",
            "URX",
            "LLX",
        )
    ),
    _trace(
        (
            "X",
            "URX",
            "DDX",
            "URX",
            "ULLX",
            "DLX",
            "DRX",
            "URX",
            "URX",
            "RX",
            "UX",
            "DDX",
            "DX",
            "LX",
            "DDX",
            "LX",
            "UX",
            "LX",
            "DLX",
            "ULX",
            "UX",
            "UX",
            "UX",
            "RX",
            "UX",
            "RX",
        )
    ),
)


class NeighborCensusDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)

    def _build_level_program(self, env):
        del env
        if self._current_level_idx is None:
            raise RuntimeError("Neighbor Census level index is unavailable.")
        return [(action, {}) for action in LEVEL_PROGRAMS[self._current_level_idx]]


AGENT_CLASS = NeighborCensusDslAgent
