from __future__ import annotations

from ..core import CachedProgramDslAgent, observation_level_index

CLICK_PROGRAMS = (
    [((32, 29), 2), ((32, 35), 2), ((32, 41), 2)],
    [((14, 23), 3), ((32, 23), 5), ((32, 41), 3), ((44, 41), 5)],
    [((20, 23), 3), ((32, 23), 5), ((32, 35), 6), ((20, 35), 4), ((20, 47), 3), ((38, 47), 5), ((38, 59), 6)],
)


class WireTheCircuitDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "wire_the_circuit-0001"):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        observation = getattr(env, "_obs", None)
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise self._missing_level_error()

        program: list[tuple[int, dict[str, int]]] = []
        for (x, y), repeats in CLICK_PROGRAMS[level_idx]:
            for _ in range(repeats):
                program.append((6, {"x": x, "y": y}))
        program.append((1, {}))
        return program


AGENT_CLASS = WireTheCircuitDslAgent
