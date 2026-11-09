from __future__ import annotations

from ..core import DslAgent, observation_level_index

_ACTION = {"U": 1, "D": 2, "L": 3, "R": 4, "S": 5}

_REPLAYS = (
    "S U U R R S R R",
    "S R R D R U S R S R S L U S R R R S R R",
    "S U S R U R S R S L S R R R U S D R L S U U U R S R R",
    "S U R S U R S R S L D S R R R U U U R S D D D R L S U U U R S R R",
    (
        "S R S R R R R U U L S D S U L S L U U R S D D R R D S U L L U U R "
        "S L D D R R D D L L S R R U U L L U U R R S R R R"
    ),
    (
        "S U R S U S D R S L L U S D R R R S R L S R R U U L L U U R S D D R R D D L S R U U L L U U R "
        "S L D D R R D R S U L L L U U R R S R L S R S R R"
    ),
    (
        "S U L S L L U S R R U U S D D R L S U U U S D D R S D R R S R R U S L L L S R S U L L S D S U L "
        "U U R S D D S U U R S R L S R R U U L S D D L S R U U L S R R D S U L L L S L R S L S L R S L S L L"
    ),
)

LEVEL_PROGRAMS = [[_ACTION[token] for token in replay.split()] for replay in _REPLAYS]


class SandFerryDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self) -> None:
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
            raise RuntimeError("Missing `levels_completed` in Sand Ferry observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Sand Ferry DSL program exhausted before reaching WIN.")
        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}
