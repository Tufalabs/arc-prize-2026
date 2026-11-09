from __future__ import annotations

from typing import Any

from ..core import DslAgent

LEFT_KNOB: dict[str, int] = {"x": 27, "y": 10}
BLUE_PAD: dict[str, int] = {"x": 54, "y": 10}

LEVEL_PROGRAMS: tuple[list[int], ...] = (
    [1, 1, 1, 4, 5, 6, 4, 2, 5, 1],
    [1, 1, 1, 4, 5, 6, 2, 5, 1],
    [2, 2, 3, 3, 5, 6, 6, 5, 1],
)


class StampPressDslAgent(DslAgent):
    def __init__(self, game_id: str = "stamp_press-0001"):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def _level_idx(self, observation: Any) -> int | None:
        raw = getattr(observation, "levels_completed", None)
        if raw is None:
            return None
        try:
            level_idx = int(raw)
        except (TypeError, ValueError):
            return None
        return max(0, min(level_idx, len(LEVEL_PROGRAMS) - 1))

    def _sync(self, observation: Any) -> None:
        level_idx = self._level_idx(observation)
        if level_idx is None:
            return
        self.mark_levels_solved(level_idx)
        if self._current_level_idx != level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0

    def next_action(self, _env: Any, observation: Any) -> tuple[int, dict[str, int]]:
        self._sync(observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing levels_completed in observation for stamp_press DSL agent.")

        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"stamp_press DSL program exhausted on level {self._current_level_idx}.")

        action_id = int(program[self._action_idx])
        self._action_idx += 1
        if action_id != 6:
            return action_id, {}

        if self._current_level_idx == 2 and self._action_idx == 6:
            return 6, BLUE_PAD
        return 6, LEFT_KNOB


AGENT_CLASS = StampPressDslAgent
