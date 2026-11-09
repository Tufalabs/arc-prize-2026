from __future__ import annotations

from ..core import DslAgent, observation_level_index

LEVEL_PROGRAMS = (
    ("A",),
    ("A", "B", "D", "E", "G"),
    ("E", "B", "D", "D", "F", "C", "B"),
    (
        "E",
        "C",
        "B",
        "A",
        "D",
        "B",
        "B",
        "D",
        "G",
        "E",
        "H",
        "C",
        "B",
        "F",
        "F",
        "B",
        "F",
        "I",
        "D",
        "D",
        "I",
        "D",
        "B",
        "B",
        "D",
        "F",
        "F",
        "D",
    ),
    (
        "I",
        "F",
        "H",
        "H",
        "G",
        "H",
        "I",
        "F",
        "F",
        "H",
        "E",
        "A",
        "D",
        "G",
        "E",
        "H",
        "H",
        "C",
        "B",
        "A",
        "C",
        "E",
        "D",
        "D",
        "D",
        "B",
        "E",
        "H",
        "H",
        "F",
        "F",
        "E",
        "B",
        "D",
        "D",
        "H",
        "K",
        "K",
        "H",
        "K",
    ),
    (
        "G",
        "H",
        "K",
        "I",
        "H",
        "H",
        "I",
        "F",
        "E",
        "E",
        "H",
        "G",
        "D",
        "A",
        "D",
        "D",
        "E",
        "A",
        "E",
        "E",
        "B",
        "B",
        "D",
        "I",
        "I",
        "F",
        "C",
        "B",
        "E",
        "E",
        "F",
        "F",
        "B",
        "B",
        "D",
        "H",
        "I",
        "H",
        "H",
        "I",
        "K",
        "L",
    ),
    (
        "J",
        "K",
        "E",
        "H",
        "H",
        "J",
        "G",
        "G",
        "G",
        "D",
        "A",
        "D",
        "D",
        "E",
        "C",
        "B",
        "A",
        "E",
        "C",
        "E",
        "B",
        "B",
        "B",
        "D",
        "E",
        "E",
        "F",
        "F",
        "E",
        "G",
        "H",
        "F",
        "E",
        "H",
        "G",
        "G",
        "I",
        "I",
        "E",
        "G",
        "E",
        "H",
        "I",
        "F",
        "F",
        "F",
        "B",
        "B",
        "D",
        "J",
        "L",
    ),
)


class PipeRotorsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is not None:
            self.mark_levels_solved(level_idx)
            if self._current_level_idx != level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing level progress in Pipe Rotors observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Pipe Rotors DSL program exhausted before reaching WIN.")
        pipe_id = program[self._action_idx]
        self._action_idx += 1
        row, col = env._game.pipe_cells[pipe_id]
        return 6, {"x": 8 + 8 * col, "y": 8 + 8 * row}


AGENT_CLASS = PipeRotorsDslAgent
