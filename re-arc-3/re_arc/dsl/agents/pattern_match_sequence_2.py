from __future__ import annotations

from re_arc.dsl.core import DslAgent, observation_level_index

SEQUENCE = [1, 3, 2, 4, 2, 1]
BUTTON_POSITIONS = {1: (10, 41), 2: (23, 41), 3: (36, 41), 4: (49, 41)}
BUTTON_SIZE = 7


class PatternMatchSequence2DslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=1)
        self._plan: list[tuple[int, int]] = []
        self._warmup: list[tuple[int, int]] = []
        self._cursor = 0
        self._warmup_cursor = 0
        self._last_level_index: int | None = None

    def reset_episode(self):
        super().reset_episode()
        self._warmup = [(0, 0)] * 12
        self._plan = []
        for token in SEQUENCE:
            x, y = BUTTON_POSITIONS[token]
            self._plan.append((x + BUTTON_SIZE // 2, y + BUTTON_SIZE // 2))
        self._cursor = 0
        self._warmup_cursor = 0
        self._last_level_index = None

    def _sync_level(self, observation) -> None:
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx != self._last_level_index:
            self._cursor = 0
            self._warmup_cursor = 0
            self._last_level_index = level_idx

    def next_action(self, _env, observation):
        self._sync_level(observation)
        if self._warmup_cursor < len(self._warmup):
            x, y = self._warmup[self._warmup_cursor]
            self._warmup_cursor += 1
            return 6, {"x": int(x), "y": int(y)}
        if self._cursor < len(self._plan):
            x, y = self._plan[self._cursor]
            self._cursor += 1
            return 6, {"x": int(x), "y": int(y)}

        x, y = self._plan[-1] if self._plan else (0, 0)
        return 6, {"x": int(x), "y": int(y)}


AGENT_CLASS = PatternMatchSequence2DslAgent
