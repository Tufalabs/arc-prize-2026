from __future__ import annotations

from ..core import DslAgent, observation_level_index

UP_CLICK = {"x": 31, "y": 5}
RIGHT_CLICK = {"x": 58, "y": 31}

LEVEL_PROGRAMS = {
    0: [(6, RIGHT_CLICK), (6, RIGHT_CLICK), (6, RIGHT_CLICK), (6, RIGHT_CLICK)],
    1: [(6, RIGHT_CLICK), (6, RIGHT_CLICK), (6, UP_CLICK), (6, UP_CLICK), (6, RIGHT_CLICK)],
    2: [(6, RIGHT_CLICK), (6, RIGHT_CLICK), (6, UP_CLICK), (6, UP_CLICK), (6, RIGHT_CLICK), (6, RIGHT_CLICK)],
}


class OutlinePainterPadsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)
        self._level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Missing `levels_completed` in outline painter pads observation.")
        self.mark_levels_solved(level_idx)

        if self._level_idx is None or self._level_idx != level_idx:
            self._level_idx = level_idx
            self._action_idx = 0
        elif bool(getattr(observation, "full_reset", False)) and self._action_idx > 0:
            self._action_idx = 0

        program = LEVEL_PROGRAMS.get(level_idx)
        if program is None:
            raise RuntimeError(f"Missing outline painter pads program for level {level_idx}.")
        if self._action_idx >= len(program):
            raise RuntimeError(
                "outline_painter_pads DSL program exhausted before level advance "
                f"level={level_idx} steps={len(program)}"
            )

        action_id, payload = program[self._action_idx]
        self._action_idx += 1
        return action_id, dict(payload)


AGENT_CLASS = OutlinePainterPadsDslAgent
