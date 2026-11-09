from __future__ import annotations

from ..core import DslAgent, observation_level_index

ACTION_CLICK = 6

LEVEL1_CLICKS = ((59, 13), (3, 19), (59, 25), (3, 31), (59, 37), (3, 43), (59, 49))
RECORDED_LEVEL_2_TO_6 = """
29,11 4,15 22,11 60,32 60,39 60,43 60,50 61,14 40,16 61,19 61,19 61,19 3,20 53,17 4,26 4,26 61,25 60,19 60,19 60,19
5,15 5,15 17,17 3,18 3,18 3,19 50,60 50,60 52,17 3,20 3,20 3,20 3,20 61,18 3,26 3,32 3,37 3,44 3,49 2,19 2,19 2,19
17,11 13,3 13,3 13,3 13,3 17,11 3,13 4,20 2,32 20,3 20,3 42,34 43,60 43,60 40,47 43,3 32,4 41,27 43,4 39,23 44,4 44,4
44,4 44,4 51,3 50,3 50,3 50,3 50,3 46,16 46,16 46,16 6,21 42,17 1,19 1,19 2,13 60,20 60,20 60,20 60,32 60,32 60,32
62,50 62,50 62,50 62,3 31,4 29,29 59,31 35,30 43,59 41,29 59,32 47,27 4,38 60,36 60,36 3,32 3,32 3,32 59,50 3,32 2,39
2,39 3,51 3,51 60,19 53,17 60,19 52,17 61,20 29,24 3,31 29,24 32,59 32,59 28,34 44,59 31,3 53,17 17,11 15,10 59,12
4,21 4,21 2,32 3,37 3,37 2,49 2,49 4,12 52,16 61,19 61,19 52,17 43,3 43,3 43,3 31,5 31,5 31,5 14,1 14,1 51,16 5,20
5,20 5,20 53,17 3,19 54,17 53,18 60,4 60,4 44,3 61,5 47,17 62,21 62,21 62,21 62,21 62,21 62,21 60,30 60,30 60,30 60,30
46,18 44,58 44,58 44,58 46,34 32,59 32,59 32,59 15,59 15,59 15,59 44,59 44,59 44,59 50,2 59,36 59,36 59,36 59,36 60,43
60,43 60,43 60,43 16,35 14,58 14,58 14,58 60,1 32,5 29,29 45,2 41,29 61,31 29,28 28,30 60,32 59,38 59,38 59,38 60,31
60,31 60,31 60,31 60,44 60,44 60,44 60,32 60,32 60,32 58,37 58,37 61,44 61,44 61,50 3,49 3,49 28,23 61,26 61,26 6,32
6,32 6,32 6,32 4,38 4,38 4,38 27,23 31,59 31,59 31,59 44,60 44,60 49,58 31,4 60,14 5,13 59,20 2,25 3,32 4,37 3,44 3,44
3,44 3,44 3,50 3,50 17,10 62,13 21,10 4,13 47,16 43,3 43,3 43,3 43,3 31,2 31,2 31,2 31,2 13,3 13,3 13,3 13,3 13,3 50,3
50,3 50,3 50,3 50,3 50,60 60,19 46,18 59,19 59,19 8,17 3,19 3,19 3,19 3,19 3,19 3,26 3,26 2,31 2,31 2,38 2,38 2,44
2,44 3,50 3,50
"""
RECORDED_LEVEL_LENGTHS = (20, 22, 46, 93, 118)
REDUCED_LEVEL_5 = """
60,38 61,50 44,60 41,29 59,32 30,4 4,31 30,30 60,31 4,50 4,50 33,59 61,20 61,20 61,20 61,20
61,20 46,17 44,61 5,32 47,23 45,60 45,60 47,35 43,5 43,5 14,59 60,31 60,31 60,31 16,23 15,60
15,60 17,34 60,4 4,38 4,38 60,31 60,31 60,31 5,50 5,50 4,38 4,38 3,31 3,31 5,20 5,20
5,20 5,20 15,11 60,14 23,11 5,14
"""
REDUCED_LEVEL_6 = """
44,59 44,59 49,4 43,4 61,37 61,44 41,28 58,31 46,28 3,31 49,3 49,3 48,1 47,17 60,19 60,19
10,18 62,20 60,18 60,18 60,18 60,18 61,32 61,32 60,37 60,37 60,43 60,43 59,37 60,44 17,35 14,60
14,60 17,47 60,31 60,31 59,38 60,43 59,19 59,19 17,11 60,13 25,11 3,13 3,32 3,38 3,45 18,23
14,59 14,59 14,59 61,6 32,60 44,60 44,60 49,3 49,3 49,3 49,3
"""


def _parse_clicks(text: str) -> list[dict[str, int]]:
    clicks = []
    for token in text.split():
        x_raw, y_raw = token.split(",", maxsplit=1)
        clicks.append({"x": int(x_raw), "y": int(y_raw)})
    return clicks


def _split_clicks(clicks: list[dict[str, int]], lengths: tuple[int, ...]) -> list[list[dict[str, int]]]:
    programs = []
    start = 0
    for length in lengths:
        end = start + length
        programs.append(clicks[start:end])
        start = end
    if start != len(clicks):
        raise ValueError("Recorded Marble Loom solution length does not match level lengths.")
    return programs


LEVEL_PROGRAMS = [[{"x": x, "y": y} for x, y in LEVEL1_CLICKS]]
LEVEL_PROGRAMS.extend(_split_clicks(_parse_clicks(RECORDED_LEVEL_2_TO_6), RECORDED_LEVEL_LENGTHS))
LEVEL_PROGRAMS[1].insert(0, {"x": 40, "y": 11})
LEVEL_PROGRAMS[4] = _parse_clicks(REDUCED_LEVEL_5)
LEVEL_PROGRAMS[5] = _parse_clicks(REDUCED_LEVEL_6)


class MarbleLoomDslAgent(DslAgent):
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
            raise RuntimeError("Missing `levels_completed` in Marble Loom observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Marble Loom DSL program exhausted before reaching WIN.")
        click = program[self._action_idx]
        self._action_idx += 1
        return ACTION_CLICK, dict(click)
