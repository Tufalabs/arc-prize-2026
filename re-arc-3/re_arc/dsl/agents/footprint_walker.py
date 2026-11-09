from __future__ import annotations

from ..core import CachedProgramDslAgent

LEVEL_SPECS = [
    {"start": (16, 28), "target": (22, 30)},
    {"start": (14, 20), "target": (26, 24)},
    {"start": (16, 18), "target": (28, 24)},
]


class FootprintWalkerDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env):
        level_index = int(getattr(env, "_current_level", self.solved_levels))
        spec = LEVEL_SPECS[level_index]
        start_x, start_y = spec["start"]
        target_x, target_y = spec["target"]

        program: list[tuple[int, dict[str, int]]] = []
        dx = target_x - start_x
        dy = target_y - start_y
        horizontal_action = 4 if dx >= 0 else 3
        vertical_action = 2 if dy >= 0 else 1
        for _ in range(abs(dx)):
            program.append((horizontal_action, {}))
        for _ in range(abs(dy)):
            program.append((vertical_action, {}))
        return program


AGENT_CLASS = FootprintWalkerDslAgent
