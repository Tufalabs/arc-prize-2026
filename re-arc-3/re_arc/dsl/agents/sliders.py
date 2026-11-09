from __future__ import annotations

from ..core import DslAgent, observation_level_index

CLICKS = {
    "B+": (55, 4),
    "B-": (61, 4),
    "G0+": (55, 4),
    "G0-": (61, 4),
    "G+": (55, 10),
    "G-": (61, 10),
    "P1ccw": (55, 10),
    "P1cw": (61, 10),
    "Pccw": (55, 16),
    "Pcw": (61, 16),
    "Y2left": (55, 16),
    "Y2right": (61, 16),
    "Yleft": (55, 22),
    "Yright": (61, 22),
    "Oup1": (55, 10),
    "Odown1": (61, 10),
    "Oleft1": (55, 10),
    "Oright1": (61, 10),
    "Oup2": (55, 16),
    "Odown2": (61, 16),
    "Oleft3": (55, 22),
    "Oright3": (61, 22),
    "Oup4": (55, 22),
    "Odown4": (61, 22),
    "Oup5": (55, 28),
    "Odown5": (61, 28),
}

LEVEL_PROGRAMS = [
    ["B+", "B+", "B+"],
    ["B+", "B+", "G+", "G+"],
    ["Oup1", "B+", "B+", "B+"],
    ["Oleft1", "Oleft1", "B+", "B+", "B+", "B+"],
    ["Oup2", "Oup2", "Oup2", "Oup2", "B+", "B+", "Oright3", "Oright3", "B+", "G+", "G+", "G+"],
    ["G0-", "G0-", "Oright1", "Oright1", "Oright1", "Oright1", "G0+", "G0+", "G0+", "G0+"],
    [
        "G0-",
        "G0-",
        "G0-",
        "P1cw",
        "Oup4",
        "Oup4",
        "Oup4",
        "Oup4",
        "Y2right",
        "Y2right",
        "Y2right",
        "Y2right",
        "G0+",
        "G0+",
        "G0+",
    ],
    [
        "B+",
        "B+",
        "B+",
        "Yright",
        "Yright",
        "Yright",
        "G-",
        "G-",
        "G-",
        "B-",
        "B-",
        "B-",
        "Pcw",
        "Oup5",
        "Oup5",
        "Oup5",
        "Oup5",
        "Yright",
        "Yright",
        "Yright",
        "Yright",
        "B+",
        "G+",
        "G+",
        "G+",
        "G+",
    ],
]


class SlidersDslAgent(DslAgent):
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
            raise RuntimeError("Missing `levels_completed` in Sliders observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Sliders DSL program exhausted before reaching WIN.")
        click_name = program[self._action_idx]
        self._action_idx += 1
        x, y = CLICKS[click_name]
        return 6, {"x": x, "y": y}
