from __future__ import annotations

from ..core import DslAgent, observation_level_index

# Center clicks for NODE_TOP_LEFTS in zigclick-0001.
_CLICK_SEQUENCE = [(7, 8), (54, 15), (10, 23), (51, 31), (13, 39), (48, 47)]


class ZigclickDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=1)
        self._cursor = 0
        self._level_idx = None

    def reset_episode(self):
        super().reset_episode()
        self._cursor = 0
        self._level_idx = None

    def _sync_level(self, observation) -> None:
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            level_idx = 0

        reset_level = bool(getattr(observation, "full_reset", False))
        if self._level_idx != level_idx or reset_level:
            self._level_idx = level_idx
            self._cursor = 0

    def next_action(self, _env, observation):
        self._sync_level(observation)
        if self._cursor < len(_CLICK_SEQUENCE):
            x, y = _CLICK_SEQUENCE[self._cursor]
            self._cursor += 1
            return 6, {"x": x, "y": y}
        return 6, {"x": 0, "y": 0}


AGENT_CLASS = ZigclickDslAgent
