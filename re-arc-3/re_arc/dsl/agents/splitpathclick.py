from __future__ import annotations

from arcengine import GameAction

from ..core import DslAgent, observation_level_index

CLICK_PLAN = [(10, 11), (31, 12), (51, 30), (32, 51)]


class SplitpathclickDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=1)
        self._cursor = 0
        self._level_idx = None

    def reset_episode(self):
        super().reset_episode()
        self._cursor = 0
        self._level_idx = None

    def _sync_level(self, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            level_idx = 0
        if level_idx != self._level_idx:
            self._level_idx = level_idx
            self._cursor = 0

    def next_action(self, _env, observation):
        self._sync_level(observation)
        if self._cursor < len(CLICK_PLAN):
            x, y = CLICK_PLAN[self._cursor]
            self._cursor += 1
            return int(GameAction.ACTION6.value), {"x": int(x), "y": int(y)}
        x, y = CLICK_PLAN[-1]
        return int(GameAction.ACTION6.value), {"x": int(x), "y": int(y)}


AGENT_CLASS = SplitpathclickDslAgent
