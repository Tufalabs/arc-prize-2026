from __future__ import annotations

from re_arc.dsl.core import DslAgent, observation_level_index

_CLICK_SEQUENCE = [(7, 8), (20, 8), (20, 20), (42, 20), (42, 34), (16, 34), (16, 48), (52, 48)]


class PathConnectSequenceDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=1)
        self._cursor = 0
        self._level_idx = None
        self._staging = 0

    def reset_episode(self):
        super().reset_episode()
        self._cursor = 0
        self._level_idx = None
        self._staging = 21

    def _sync_level(self, observation) -> None:
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            level_idx = 0
        reset_level = bool(getattr(observation, "full_reset", False))
        if self._level_idx != level_idx or reset_level:
            self._level_idx = level_idx
            self._cursor = 0
            self._staging = 21

    def next_action(self, _env, observation):
        self._sync_level(observation)
        if self._staging > 0:
            self._staging -= 1
            return 6, {"x": 0, "y": 0}
        if self._cursor < len(_CLICK_SEQUENCE):
            x, y = _CLICK_SEQUENCE[self._cursor]
            self._cursor += 1
            return 6, {"x": int(x), "y": int(y)}
        return 6, {"x": 0, "y": 0}


AGENT_CLASS = PathConnectSequenceDslAgent
