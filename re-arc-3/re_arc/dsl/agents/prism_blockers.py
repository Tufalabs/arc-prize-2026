from __future__ import annotations

from ..core import CachedProgramDslAgent

RUNS = (
    (("W5-R", 1),),
    (("W5-R", 5),),
    (("W5-R", 5), ("N2-B", 1)),
    (("H-B", 6), ("H-R", 5), ("H-B", 4), ("H-R", 2), ("H-B", 1), ("V-B", 3), ("V-R", 3), ("V-B", 1)),
    (
        ("H1-B", 6),
        ("H1-R", 2),
        ("H2-B", 6),
        ("H2-R", 5),
        ("H2-B", 3),
        ("H2-R", 2),
        ("H2-B", 1),
        ("V-B", 5),
        ("H1-R", 3),
        ("H1-B", 2),
        ("V-R", 3),
        ("H1-B", 2),
        ("H1-R", 2),
        ("H1-B", 1),
        ("V-B", 2),
        ("V-R", 2),
    ),
    (
        ("H1-B", 6),
        ("H1-R", 2),
        ("H2-B", 6),
        ("H2-R", 1),
        ("V1-B", 4),
        ("H1-R", 2),
        ("H1-B", 2),
        ("H1-R", 1),
        ("V1-R", 4),
        ("V1-B", 4),
        ("V1-R", 2),
        ("V2-B", 4),
        ("H2-R", 3),
        ("H2-B", 1),
        ("V2-R", 4),
        ("H2-B", 2),
        ("H2-R", 1),
        ("V2-B", 3),
        ("H2-R", 1),
        ("H2-B", 1),
    ),
    (
        ("H1-B", 7),
        ("H1-R", 4),
        ("H2-B", 6),
        ("H2-R", 4),
        ("H2-B", 3),
        ("H2-R", 1),
        ("V1-B", 4),
        ("H1-R", 2),
        ("H1-B", 4),
        ("H1-R", 1),
        ("H2-R", 1),
        ("V1-R", 3),
        ("V1-B", 3),
        ("V1-R", 2),
        ("V2-B", 4),
        ("H1-R", 2),
        ("H1-B", 1),
        ("V2-R", 4),
        ("H1-B", 1),
        ("V2-B", 3),
        ("V2-R", 2),
    ),
)


class PrismBlockersDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id, total_levels=7)

    def _build_level_program(self, env):
        level_idx = int(self._current_level_idx or 0)
        emitters = {emitter.emitter_id: emitter for emitter in env._game.emitters}
        program = []
        for emitter_id, repeats in RUNS[level_idx]:
            emitter = emitters[emitter_id]
            x, y = emitter.center
            program.extend([(6, {"x": x, "y": y})] * repeats)
        return program


AGENT_CLASS = PrismBlockersDslAgent
