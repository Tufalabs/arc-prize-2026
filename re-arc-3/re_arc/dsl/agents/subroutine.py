from __future__ import annotations

from ..core import CachedProgramDslAgent

ACTION_SPACE = 5
ACTION_CLICK = 6

COMPONENTS = [
    {"R": (20, 16, 3)},
    {"R": (18, 16, 3), "B": (24, 28, 2)},
    {"R": (15, 16, 4), "B": (21, 28, 2)},
    {"R": (18, 15, 3), "B": (10, 27, 2), "G": (34, 39, 2)},
    {"R": (9, 15, 5), "B": (24, 29, 2)},
    {"R": (2, 15, 7), "B": (47, 25, 2), "G": (18, 35, 1), "P": (35, 45, 2)},
    {"R": (17, 15, 4), "G": (5, 29, 1), "P": (18, 29, 2), "B": (39, 29, 3)},
    {"R": (4, 15, 4), "B": (39, 25, 3), "P": (24, 35, 2), "G": (5, 45, 5)},
]

SOLUTIONS = [
    [("T0", "R0"), ("T2", "R1"), ("T1", "R2")],
    [("T1", "R0"), ("T3", "B0"), ("T2", "B1"), ("T0", "R2")],
    [("T2", "R0"), ("T1", "R1"), ("T0", "R2"), ("T3", "R3"), ("T5", "B0"), ("T4", "B1")],
    [("T2", "R0"), ("T0", "R1"), ("T3", "R2"), ("T4", "B0"), ("T5", "G0"), ("T1", "G1")],
    [("T1", "R0"), ("T0", "R1"), ("T3", "R2"), ("T4", "R3"), ("T5", "R4"), ("T6", "B0"), ("T2", "B1")],
    [
        ("T0", "R0"),
        ("T6", "R1"),
        ("T7", "R2"),
        ("T1", "R3"),
        ("T8", "R4"),
        ("T2", "R5"),
        ("T9", "R6"),
        ("T3", "G0"),
        ("T4", "P0"),
        ("T5", "P1"),
    ],
    [
        ("R0", "B0"),
        ("R0", "G0"),
        ("T0", "B1"),
        ("T1", "P0"),
        ("T2", "P1"),
        ("T5", "B2"),
        ("T3", "R1"),
        ("T4", "R2"),
        ("T6", "R3"),
    ],
    [
        ("G0", "B0"),
        ("P0", "G2"),
        ("T0", "P0"),
        ("T1", "P1"),
        ("T2", "G3"),
        ("T3", "G4"),
        ("T4", "R1"),
        ("T5", "B1"),
        ("T6", "R0"),
        ("T7", "R2"),
        ("T8", "R3"),
        ("T9", "B2"),
    ],
]


def _well_center(level_idx: int, well: str) -> tuple[int, int]:
    if well.startswith("T"):
        index = int(well[1:])
        return 4 + 6 * index, 58
    component = well[0]
    slot = int(well[1:])
    x, y, _slots = COMPONENTS[level_idx][component]
    return x + 4 + 6 * slot, y + 4


class SubroutineDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=8)

    def _build_level_program(self, env):
        del env
        level_idx = int(self._current_level_idx or 0)
        program: list[tuple[int, dict[str, int]]] = []
        for source, destination in SOLUTIONS[level_idx]:
            sx, sy = _well_center(level_idx, source)
            dx, dy = _well_center(level_idx, destination)
            program.append((ACTION_CLICK, {"x": sx, "y": sy}))
            program.append((ACTION_CLICK, {"x": dx, "y": dy}))
        program.append((ACTION_SPACE, {}))
        return program


AGENT_CLASS = SubroutineDslAgent
