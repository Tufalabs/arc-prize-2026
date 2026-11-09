from __future__ import annotations

from ..core import CachedProgramDslAgent

PROGRAMS = {
    0: [1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4],
    1: [4, 1, 1, 1, 1, 1, 1, 4, 4, 4],
    2: [4, 1, 1, 4, 1, 1, 1, 1, 1, 4, 4, 4],
}


class MirrorGreedyFriendDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, _env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(self._current_level_idx or 0)
        return [(action_id, {}) for action_id in PROGRAMS[level_idx]]


AGENT_CLASS = MirrorGreedyFriendDslAgent
