from __future__ import annotations

from ..core import DslAgent

CLICK_WAIT = {"x": 0, "y": 0}

_LEVEL_PROGRAMS = [
    [5, 4, 4, 2, 2, 2, 4, 4, 4, 6],
    [5, 4, 4, 4, 2, 2, 2, 2, 5, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 6],
    [
        5,
        4,
        4,
        4,
        4,
        2,
        2,
        2,
        5,
        4,
        4,
        4,
        4,
        4,
        2,
        2,
        2,
        2,
        5,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        5,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        6,
    ],
]


class EchoMazeGptDslAgent(DslAgent):
    def __init__(self, game_id: str = "echo_maze_gpt-0001") -> None:
        super().__init__(game_id=game_id, total_levels=len(_LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def _level_index(self, observation) -> int | None:
        raw = getattr(observation, "levels_completed", None)
        if raw is None:
            return None
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            return None
        return max(0, min(idx, len(_LEVEL_PROGRAMS) - 1))

    def _sync_level(self, observation) -> None:
        level_idx = self._level_index(observation)
        if level_idx is None:
            return

        self.mark_levels_solved(level_idx)
        reset_level = bool(getattr(observation, "full_reset", False))
        if self._current_level_idx is None or level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
            return
        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def next_action(self, _env, observation):
        self._sync_level(observation)
        if self._current_level_idx is None:
            raise RuntimeError("Echo Maze GPT DSL requires `levels_completed` in the observation.")

        program = _LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(
                "Echo Maze GPT DSL program exhausted before the next level transition. "
                f"level={self._current_level_idx} steps={len(program)}"
            )

        action_id = int(program[self._action_idx])
        self._action_idx += 1
        return action_id, (CLICK_WAIT if action_id == 6 else {})


AGENT_CLASS = EchoMazeGptDslAgent
