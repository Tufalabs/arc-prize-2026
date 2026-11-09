from __future__ import annotations

from dataclasses import dataclass

from ..core import MOVE_ACTION_BY_DELTA, CachedProgramDslAgent, find_shortest_action_plan


@dataclass(frozen=True)
class PortalPairSpec:
    a: tuple[int, int]
    b: tuple[int, int]
    color: int


@dataclass(frozen=True)
class LevelSpec:
    floor_cells: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]
    move_budget: int
    portal_pairs: tuple[PortalPairSpec, ...]


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        floor_cells=frozenset({(0, 2), (1, 2), (0, 3), (1, 3), (4, 2), (5, 2), (4, 3), (5, 3)}),
        start=(0, 3),
        goal=(5, 3),
        move_budget=8,
        portal_pairs=(PortalPairSpec(a=(1, 2), b=(4, 2), color=11),),
    ),
    LevelSpec(
        floor_cells=frozenset(
            {
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3),
                (0, 4),
                (1, 4),
                (3, 1),
                (4, 1),
                (5, 1),
                (3, 2),
                (4, 2),
                (5, 2),
                (3, 3),
                (4, 3),
                (5, 3),
                (6, 4),
                (7, 4),
                (6, 5),
                (7, 5),
            }
        ),
        start=(0, 3),
        goal=(7, 5),
        move_budget=12,
        portal_pairs=(PortalPairSpec(a=(1, 2), b=(3, 2), color=11), PortalPairSpec(a=(5, 1), b=(6, 4), color=15)),
    ),
    LevelSpec(
        floor_cells=frozenset(
            {
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3),
                (0, 4),
                (1, 4),
                (3, 1),
                (4, 1),
                (5, 1),
                (3, 2),
                (4, 2),
                (5, 2),
                (3, 3),
                (4, 3),
                (5, 3),
                (6, 1),
                (7, 1),
                (6, 2),
                (7, 2),
                (3, 5),
                (4, 5),
                (3, 6),
                (4, 6),
                (6, 4),
                (7, 4),
                (6, 5),
                (7, 5),
            }
        ),
        start=(0, 3),
        goal=(7, 2),
        move_budget=13,
        portal_pairs=(
            PortalPairSpec(a=(1, 2), b=(3, 1), color=11),
            PortalPairSpec(a=(1, 4), b=(3, 5), color=15),
            PortalPairSpec(a=(5, 2), b=(6, 1), color=12),
            PortalPairSpec(a=(5, 3), b=(6, 4), color=13),
        ),
    ),
)


def build_portal_map(spec: LevelSpec) -> dict[tuple[int, int], tuple[int, int]]:
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for pair in spec.portal_pairs:
        mapping[pair.a] = pair.b
        mapping[pair.b] = pair.a
    return mapping


class IslandPortalDockDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, _env) -> list[tuple[int, dict[str, int]]]:
        level_idx = getattr(self, "_current_level_idx", None)
        if level_idx is None:
            raise RuntimeError("Missing current level for island_portal_dock DSL program build.")
        spec = LEVEL_SPECS[int(level_idx)]
        portal_map = build_portal_map(spec)
        start_state = (spec.start[0], spec.start[1], spec.move_budget)

        def is_goal(state: tuple[int, int, int]) -> bool:
            return (state[0], state[1]) == spec.goal

        def expand(state: tuple[int, int, int]):
            x, y, moves_left = state
            if moves_left <= 0:
                return []

            out = []
            for delta, action_id in MOVE_ACTION_BY_DELTA.items():
                nx = x + delta[0]
                ny = y + delta[1]
                if (nx, ny) in spec.floor_cells:
                    final = portal_map.get((nx, ny), (nx, ny))
                else:
                    final = (x, y)
                out.append((action_id, (final[0], final[1], moves_left - 1)))
            return out

        plan = find_shortest_action_plan(
            start_state=start_state,
            is_goal=is_goal,
            expand=expand,
            dominance_key=lambda state: (state[0], state[1]),
            dominance_score=lambda state: state[2],
        )
        if plan is None:
            raise RuntimeError(f"island_portal_dock level {level_idx} has no DSL solution.")
        if len(plan) > spec.move_budget:
            raise RuntimeError(f"island_portal_dock level {level_idx} solution exceeds move budget.")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = IslandPortalDockDslAgent
