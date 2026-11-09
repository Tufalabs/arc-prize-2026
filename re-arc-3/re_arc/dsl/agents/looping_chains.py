from __future__ import annotations

from ..core import DslAgent, observation_level_index

CELL = 8


def _click(cell_x: int, cell_y: int = 7) -> tuple[int, dict[str, int]]:
    return 6, {"x": cell_x * CELL + 4, "y": cell_y * CELL + 4}


LEVEL_PROGRAMS = [
    [_click(4), _click(4), _click(4)],
    [_click(1), _click(1), _click(7), _click(7), _click(7)],
    [_click(4), _click(4), _click(0), _click(1)],
    [_click(7), _click(7), _click(7), _click(7), _click(1), _click(1), _click(1), _click(7)],
    [_click(1), _click(1), _click(1), _click(1), _click(7), _click(7), _click(7), _click(7), _click(1)],
    [_click(7), _click(7), _click(1), _click(1), _click(1), _click(1), _click(1), _click(1), _click(1)],
    [_click(7), _click(7), *[_click(0) for _ in range(13)]],
    [
        *[_click(0) for _ in range(11)],
        _click(3),
        _click(3),
        *[_click(0) for _ in range(4)],
        _click(6),
        _click(6),
        _click(6),
    ],
]


class LoopingChainsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is not None:
            self.mark_levels_solved(level_idx)
            if self._current_level_idx != level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in Looping Chains observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Looping Chains DSL program exhausted before reaching WIN.")
        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
