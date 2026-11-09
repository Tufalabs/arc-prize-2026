from __future__ import annotations

from arcengine import GameAction

from ..core import DslAgent, observation_level_index

_CLICK_SEQUENCE = [(11, 37), (27, 37), (43, 37), (27, 51)]


class PatternMatchSequenceDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=1)
        self._cursor = 0
        self._level_idx = None
        self._warmup_remaining = 0

    def reset_episode(self):
        super().reset_episode()
        self._cursor = 0
        self._level_idx = None
        self._warmup_remaining = 24

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

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            return int(GameAction.ACTION6.value), {"x": 0, "y": 0}

        if self._cursor < len(_CLICK_SEQUENCE):
            x, y = _CLICK_SEQUENCE[self._cursor]
            self._cursor += 1
            return int(GameAction.ACTION6.value), {"x": int(x), "y": int(y)}

        x, y = _CLICK_SEQUENCE[-1]
        return int(GameAction.ACTION6.value), {"x": int(x), "y": int(y)}


AGENT_CLASS = PatternMatchSequenceDslAgent
