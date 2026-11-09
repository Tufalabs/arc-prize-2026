from __future__ import annotations

from ..core import DslAgent

_LEVEL_PROGRAMS = ((4, 2, 4), (4, 1, 4), (4, 2, 3, 4), (4, 1, 4, 2, 4), (4, 1, 3, 4, 2, 4))


class IceStopDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def _level_idx(self, observation):
        raw = getattr(observation, "levels_completed", None)
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            return None
        return max(0, min(idx, len(_LEVEL_PROGRAMS) - 1))

    def _sync(self, observation):
        level_idx = self._level_idx(observation)
        if level_idx is None:
            return
        self.mark_levels_solved(level_idx)
        reset_level = bool(getattr(observation, "full_reset", False))

        if self._current_level_idx is None or self._current_level_idx != level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
            return

        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def next_action(self, _env, observation):
        self._sync(observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` for ice_stop DSL agent.")

        program = _LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"ice_stop DSL program exhausted at level={self._current_level_idx} steps={len(program)}"
            )

        action_id = int(program[self._action_idx])
        self._action_idx += 1
        return action_id, {}
