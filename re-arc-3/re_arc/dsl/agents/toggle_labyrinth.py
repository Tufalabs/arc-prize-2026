from __future__ import annotations

from ..core import DslAgent, observation_level_index


def _expand(program: str) -> list[int]:
    action = {"U": 1, "D": 2, "L": 3, "R": 4}
    out: list[int] = []
    for token in program.split():
        out.extend([action[token[0]]] * int(token[1:] or "1"))
    return out


LEVEL_PROGRAMS = [
    _expand("R L3"),
    _expand("U2 D R3 L5"),
    _expand("U2 R4 L7"),
    _expand("R2 U5 R4 U D L2 D2 R3 U R3 D4"),
    _expand("R2 U4 R2 U2 R U R4 U D R D2 L6 D L2 D3 R2 D R3 U R2 D R"),
    _expand("R2 U4 R2 U2 R U R7 D2 L8 D L2 D3 R2 D R3 U R U R4 D2"),
    _expand("R2 U4 R2 U R U R6 D3 L7 U L2 D6 R10 U8 L7 D L D5 L4 D4"),
]


class ToggleLabyrinthDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._level_index: int | None = None
        self._action_index = 0

    def reset_episode(self):
        super().reset_episode()
        self._level_index = None
        self._action_index = 0

    def next_action(self, _env, observation):
        level_index = observation_level_index(observation, self.total_levels)
        if level_index is None:
            raise RuntimeError("Missing levels_completed in Toggle Labyrinth observation")
        self.mark_levels_solved(level_index)
        if level_index != self._level_index:
            self._level_index = level_index
            self._action_index = 0
        program = LEVEL_PROGRAMS[level_index]
        if self._action_index >= len(program):
            raise RuntimeError("Toggle Labyrinth DSL program exhausted before level advance")
        action_id = program[self._action_index]
        self._action_index += 1
        return action_id, {}
