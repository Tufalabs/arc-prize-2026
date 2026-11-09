from __future__ import annotations

from dataclasses import dataclass

from ..core import DslAgent
from ..solvers.search import bfs_plan

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
WAIT = 5

ACTION_TO_DELTA = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0), WAIT: (0, 0)}


@dataclass(frozen=True)
class PortalSpec:
    entry: tuple[tuple[int, int], tuple[int, int]]
    exit: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class LevelSpec:
    start: tuple[int, int]
    budget: int
    hazards: frozenset[tuple[int, int]]
    dock: frozenset[tuple[int, int]]
    portals: tuple[PortalSpec, ...]


LEVELS: tuple[LevelSpec, ...] = (
    LevelSpec(
        start=(3, 0),
        budget=12,
        hazards=frozenset((x, y) for y in (2, 3, 4) for x in range(8)),
        dock=frozenset({(5, 5), (6, 5), (5, 6), (6, 6)}),
        portals=(PortalSpec(entry=((3, 1), (4, 1)), exit=((2, 5), (3, 5))),),
    ),
    LevelSpec(
        start=(0, 5),
        budget=24,
        hazards=frozenset(
            {(x, y) for x in (2, 3) for y in range(7)}
            | {(x, y) for x in (5, 6) for y in range(7)}
            | {(7, 0), (7, 1), (7, 2)}
        ),
        dock=frozenset({(7, 5), (7, 6)}),
        portals=(
            PortalSpec(entry=((1, 1), (1, 2)), exit=((4, 1), (4, 2))),
            PortalSpec(entry=((4, 4), (4, 5)), exit=((7, 3), (7, 4))),
        ),
    ),
    LevelSpec(
        start=(0, 3),
        budget=12,
        hazards=frozenset(
            {
                (x, y)
                for y in range(7)
                for x in range(8)
                if (x, y)
                not in {
                    (0, 2),
                    (1, 2),
                    (0, 3),
                    (1, 3),
                    (0, 4),
                    (1, 4),
                    (3, 0),
                    (4, 0),
                    (3, 1),
                    (4, 1),
                    (3, 3),
                    (4, 3),
                    (3, 4),
                    (4, 4),
                    (3, 5),
                    (4, 5),
                    (3, 6),
                    (4, 6),
                    (6, 2),
                    (7, 2),
                    (6, 3),
                    (7, 3),
                    (6, 4),
                    (7, 4),
                }
            }
        ),
        dock=frozenset({(6, 2), (7, 2), (6, 3), (7, 3)}),
        portals=(
            PortalSpec(entry=((0, 2), (1, 2)), exit=((3, 0), (4, 0))),
            PortalSpec(entry=((0, 4), (1, 4)), exit=((3, 5), (4, 5))),
            PortalSpec(entry=((3, 1), (4, 1)), exit=((3, 3), (4, 3))),
            PortalSpec(entry=((3, 4), (4, 4)), exit=((6, 4), (7, 4))),
        ),
    ),
)


def _build_portal_lookup(spec: LevelSpec) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    lookup: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for portal in spec.portals:
        for idx, entry_cell in enumerate(portal.entry):
            lookup[entry_cell] = (portal.exit[idx], portal.entry[idx])
    return lookup


class HazardSkipPortalsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVELS))
        self._planned_actions = {idx: self._plan_level(spec) for idx, spec in enumerate(LEVELS)}
        self._current_level: int | None = None
        self._index = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level = None
        self._index = 0

    def _sync_level(self, observation) -> int:
        level_idx = int(getattr(observation, "levels_completed", 0) or 0)
        if self._current_level != level_idx:
            self._current_level = level_idx
            self._index = 0
        return level_idx

    def _plan_level(self, spec: LevelSpec) -> list[int]:
        portal_lookup = _build_portal_lookup(spec)
        start_state = (spec.start, spec.budget)

        def is_goal(state: tuple[tuple[int, int], int]) -> bool:
            return state[0] in spec.dock

        def expand(state: tuple[tuple[int, int], int]):
            (x, y), remaining = state
            if remaining <= 0:
                return
            for action_id, (dx, dy) in ACTION_TO_DELTA.items():
                nx = x + dx
                ny = y + dy
                if 0 <= nx < 8 and 0 <= ny < 7:
                    next_pos = (nx, ny)
                else:
                    next_pos = (x, y)
                if next_pos != (x, y) and next_pos in portal_lookup:
                    next_pos = portal_lookup[next_pos][0]
                next_remaining = remaining - 1
                if next_pos in spec.hazards:
                    continue
                if next_pos not in spec.dock and next_remaining <= 0:
                    continue
                yield (action_id, (next_pos, next_remaining), 1.0)

        plan = bfs_plan(start_state, is_goal, expand)
        if plan is None:
            raise RuntimeError(f"No DSL plan found for level starting at {spec.start}.")
        return [int(action) for action in plan]

    def next_action(self, _env, observation):
        state_name = str(getattr(getattr(observation, "state", None), "name", "")).upper()
        level_idx = self._sync_level(observation)

        if state_name == "GAME_OVER":
            self._index = 0
            return WAIT, {}

        if state_name == "WIN":
            return WAIT, {}

        plan = self._planned_actions.get(level_idx, [])
        if self._index >= len(plan):
            return WAIT, {}

        action = plan[self._index]
        self._index += 1
        return action, {}
