from __future__ import annotations

from ..core import CachedProgramDslAgent


def _expand(program: str) -> list[int]:
    actions = {"U": 1, "D": 2, "L": 3, "R": 4}
    out: list[int] = []
    for token in program.split():
        count = int(token[1:] or "1")
        out.extend([actions[token[0]]] * count)
    return out


LEVEL_PROGRAMS = tuple(
    _expand(program)
    for program in (
        "R2",
        "R2 L3",
        "R3 U D R3",
        "L2 D4 L2 D2 R4 L2 U6 R3",
        "R4 D2 R2 D2 L6 D2 R6 L6 U4 R4 U2 R3",
        "R2 D2 R2 U2 R4 D2 L2 D2 L6 D2 R2 D2 R6 U8 R",
        "L4 D2 L2 D2 L2 D2 R2 D2 R2 U2 R2 U4 R2 D6 L6 U2 L2 U2 R2 U2 R2 U2 R5",
    )
)


class TurnbackGatesDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str) -> None:
        super().__init__(game_id=game_id, total_levels=7)

    def _build_level_program(self, env):
        del env
        if self._current_level_idx is None:
            raise RuntimeError("Turnback Gates level index is unavailable.")
        return [(action, {}) for action in LEVEL_PROGRAMS[self._current_level_idx]]
