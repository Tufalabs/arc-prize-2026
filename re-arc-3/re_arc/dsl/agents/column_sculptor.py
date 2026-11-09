from __future__ import annotations

from ..core import DslAgent

_PROGRAMS: list[list[tuple[int, dict[str, int]]]] = [
    [(6, {"x": 54, "y": 30}), (5, {}), (1, {}), (1, {}), (1, {}), (1, {}), (1, {})],
    [(6, {"x": 10, "y": 30}), (6, {"x": 54, "y": 30}), (5, {}), (1, {}), (1, {}), (1, {})],
    [(6, {"x": 32, "y": 30}), (5, {}), (1, {}), (1, {}), (1, {}), (1, {}), (1, {})],
]


class ColumnSculptorDslAgent(DslAgent):
    def __init__(self, game_id: str = "column_sculptor-0001"):
        super().__init__(game_id=game_id, total_levels=len(_PROGRAMS))
        self._level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._level_idx = None
        self._action_idx = 0

    def _sync_level(self, observation) -> None:
        raw_idx = getattr(observation, "levels_completed", None)
        if raw_idx is None:
            return
        try:
            level_idx = max(0, min(int(raw_idx), len(_PROGRAMS) - 1))
        except (TypeError, ValueError):
            return

        self.mark_levels_solved(level_idx)
        if self._level_idx is None or level_idx != self._level_idx:
            self._level_idx = level_idx
            self._action_idx = 0
            return

        if bool(getattr(observation, "full_reset", False)) and self._action_idx > 0:
            self._action_idx = 0

    def next_action(self, _env, observation):
        self._sync_level(observation)
        if self._level_idx is None:
            raise RuntimeError("Column Sculptor DSL could not determine the current level.")

        program = _PROGRAMS[self._level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"Column Sculptor DSL program exhausted before completion level={self._level_idx} steps={len(program)}"
            )

        action = program[self._action_idx]
        self._action_idx += 1
        return action


AGENT_CLASS = ColumnSculptorDslAgent
