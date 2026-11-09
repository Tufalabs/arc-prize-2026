from __future__ import annotations

from ..core import CachedProgramDslAgent

CLICK_SLOT_0 = {"x": 4, "y": 27}
CLICK_SLOT_1 = {"x": 12, "y": 27}
CLICK_SLOT_2 = {"x": 20, "y": 27}
CLICK_SLOT_3 = {"x": 4, "y": 35}
CLICK_SLOT_4 = {"x": 12, "y": 35}
CLICK_SLOT_5 = {"x": 20, "y": 35}
CLICK_SMALL = {"x": 12, "y": 54}


def simple(action_id: int) -> tuple[int, dict[str, int]]:
    return action_id, {}


def click(payload: dict[str, int]) -> tuple[int, dict[str, int]]:
    return 6, payload


LEVEL_PROGRAMS: tuple[tuple[tuple[int, dict[str, int]], ...], ...] = (
    (simple(5), click(CLICK_SLOT_1), simple(4), simple(2), simple(5)),
    (simple(5), click(CLICK_SLOT_1), simple(4), simple(4), simple(2), simple(2), simple(5)),
    (simple(5), click(CLICK_SLOT_1), click(CLICK_SMALL), simple(5)),
    (
        simple(5),
        click(CLICK_SLOT_1),
        click(CLICK_SMALL),
        simple(5),
        click(CLICK_SLOT_2),
        click(CLICK_SMALL),
        simple(4),
        simple(2),
        simple(2),
        simple(5),
    ),
    (
        simple(5),
        click(CLICK_SLOT_1),
        simple(4),
        simple(2),
        simple(5),
        click(CLICK_SLOT_2),
        click(CLICK_SMALL),
        simple(1),
        simple(3),
        simple(5),
        click(CLICK_SLOT_3),
        simple(4),
        simple(2),
        simple(5),
        click(CLICK_SLOT_4),
        click(CLICK_SMALL),
        simple(2),
        simple(3),
        simple(3),
        simple(5),
    ),
    (
        simple(5),
        click(CLICK_SLOT_1),
        simple(4),
        simple(4),
        simple(2),
        simple(5),
        click(CLICK_SLOT_2),
        click(CLICK_SMALL),
        simple(1),
        simple(3),
        simple(5),
        click(CLICK_SLOT_3),
        simple(4),
        simple(2),
        simple(5),
        click(CLICK_SLOT_4),
        click(CLICK_SMALL),
        simple(2),
        simple(3),
        simple(3),
        simple(5),
        click(CLICK_SLOT_5),
        click(CLICK_SMALL),
        simple(1),
        simple(5),
    ),
)


class CraneDipperDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        del env
        level_idx = self._current_level_idx
        if level_idx is None or level_idx < 0 or level_idx >= len(LEVEL_PROGRAMS):
            raise RuntimeError(f"Crane Dipper DSL cannot resolve level index {level_idx!r}.")
        return list(LEVEL_PROGRAMS[level_idx])


AGENT_CLASS = CraneDipperDslAgent
