from __future__ import annotations

from arcengine import GameAction

from ..core import DslAgent, observation_level_index

ACTION_PLAN = [4, 4, 2, 2, 5, 4, 4, 4, 4, 4, 1, 4, 5, 3, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 5, 2, 4, 4, 4, 4, 4]


class SplitlinkDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=1)
        self._actions = ACTION_PLAN
        self._idx = 0
        self._level_idx = None

    def reset_episode(self):
        super().reset_episode()
        self._idx = 0
        self._level_idx = None

    def _sync_level(self, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            level_idx = 0
        if level_idx != self._level_idx:
            self._level_idx = level_idx
            self._idx = 0

    def next_action(self, _env, observation):
        self._sync_level(observation)
        if self._idx < len(self._actions):
            action_id = self._actions[self._idx]
            self._idx += 1
            return action_id, {}
        return int(GameAction.ACTION5.value), {}


AGENT_CLASS = SplitlinkDslAgent
