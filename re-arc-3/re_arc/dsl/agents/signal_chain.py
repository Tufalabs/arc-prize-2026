from __future__ import annotations

from ..core import CachedProgramDslAgent

_LEVEL_PROGRAMS = {
    0: [(6, {"x": 14, "y": 22}), (6, {"x": 32, "y": 22}), (6, {"x": 32, "y": 42}), (5, {})],
    1: [(6, {"x": 12, "y": 20}), (6, {"x": 28, "y": 20}), (6, {"x": 28, "y": 44}), (6, {"x": 48, "y": 44}), (5, {})],
    2: [
        (6, {"x": 12, "y": 18}),
        (6, {"x": 48, "y": 18}),
        (6, {"x": 48, "y": 48}),
        (6, {"x": 16, "y": 48}),
        (6, {"x": 16, "y": 30}),
        (5, {}),
    ],
}


class SignalChainDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, _env):
        if self._current_level_idx is None:
            raise RuntimeError("Signal Chain DSL could not determine the current level.")
        return list(_LEVEL_PROGRAMS[self._current_level_idx])


AGENT_CLASS = SignalChainDslAgent
