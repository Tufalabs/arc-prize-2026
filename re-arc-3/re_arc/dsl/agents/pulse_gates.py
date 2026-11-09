from __future__ import annotations

from ..core import DslAgent, observation_level_index

GATE_CELLS = (
    {"A": (4, 4)},
    {"A": (3, 6), "B": (3, 3), "C": (1, 3)},
    {"A": (4, 3), "B": (4, 5), "C": (6, 5)},
    {"A": (2, 1), "B": (4, 1), "C": (2, 3), "D": (4, 3), "E": (2, 5), "F": (4, 5)},
    {"A": (2, 1), "B": (4, 1), "C": (2, 3), "D": (4, 3), "E": (6, 3), "F": (2, 5), "G": (4, 5), "H": (6, 5)},
    {
        "A": (2, 1),
        "B": (4, 1),
        "C": (6, 1),
        "D": (2, 3),
        "E": (4, 3),
        "F": (6, 3),
        "G": (2, 5),
        "H": (4, 5),
        "I": (6, 5),
    },
    {
        "A": (2, 1),
        "B": (4, 1),
        "C": (6, 1),
        "D": (2, 3),
        "E": (4, 3),
        "F": (6, 3),
        "G": (2, 5),
        "H": (4, 5),
        "I": (6, 5),
    },
)


def _expand(*parts: tuple[str, int] | str) -> list[str]:
    out: list[str] = []
    for part in parts:
        if isinstance(part, str):
            out.append(part)
        else:
            name, count = part
            out.extend([name] * count)
    return out


LEVEL_PROGRAMS = (
    ["A"],
    ["A", "B", "C"],
    ["A", "B", "A", "C"],
    _expand("C", "A", ("C", 2), "E", "B", ("A", 3), ("C", 2), ("D", 2), "A", ("C", 3), "E", "F", "D", ("C", 2), "F"),
    _expand(
        "C",
        "A",
        ("C", 2),
        "F",
        "B",
        ("A", 3),
        ("C", 2),
        ("D", 2),
        "A",
        ("C", 3),
        "F",
        "G",
        "D",
        ("C", 2),
        ("F", 3),
        "C",
        "G",
        "H",
        "E",
        "D",
        ("A", 3),
        ("C", 2),
        "D",
        "H",
    ),
    _expand(
        "G",
        "E",
        "B",
        ("A", 2),
        ("D", 2),
        ("E", 2),
        "A",
        ("D", 3),
        "G",
        "H",
        "E",
        ("D", 2),
        ("G", 3),
        "D",
        "H",
        "I",
        "F",
        "E",
        ("A", 3),
        ("D", 2),
        "E",
        ("H", 2),
        "E",
        ("B", 2),
        "A",
        ("D", 3),
        "G",
        "H",
        "F",
        "C",
        "B",
        "C",
        ("E", 2),
        ("D", 2),
    ),
    _expand(
        "D",
        "A",
        ("D", 2),
        "G",
        "E",
        ("A", 3),
        ("D", 2),
        ("E", 2),
        ("B", 2),
        "A",
        ("D", 3),
        "G",
        "H",
        "C",
        "B",
        "E",
        ("D", 2),
        ("G", 3),
        "D",
        "E",
        ("F", 2),
        "B",
        ("A", 3),
        ("D", 2),
        ("E", 2),
        "A",
        ("D", 3),
        "G",
        "E",
        ("D", 2),
        ("G", 3),
        "D",
        "H",
        "I",
        "F",
        "E",
        ("A", 3),
        ("D", 2),
        "E",
        "I",
    ),
)


class PulseGatesDslAgent(DslAgent):
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
            raise RuntimeError("Pulse Gates observation has no level progress.")
        self.mark_levels_solved(level_index)
        if self._level_index != level_index:
            self._level_index = level_index
            self._action_index = 0
        program = LEVEL_PROGRAMS[level_index]
        if self._action_index >= len(program):
            raise RuntimeError(f"Pulse Gates level {level_index} program exhausted before completion.")
        gate_name = program[self._action_index]
        self._action_index += 1
        cell_x, cell_y = GATE_CELLS[level_index][gate_name]
        return 6, {"x": 6 + 5 * cell_x, "y": 11 + 5 * cell_y}
