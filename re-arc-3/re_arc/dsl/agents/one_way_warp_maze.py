from __future__ import annotations

from dataclasses import dataclass

from ..core import CachedProgramDslAgent, find_shortest_action_plan

MOVE_BY_ACTION = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


@dataclass(frozen=True)
class LevelPlanSpec:
    rows: tuple[str, ...]
    move_budget: int


LEVELS = (
    LevelPlanSpec(
        rows=("########", "#S.A...#", "###.####", "#...#..#", "#.###aG#", "#......#", "########"), move_budget=9
    ),
    LevelPlanSpec(
        rows=("########", "#S.A#G##", "#..##..#", "#.###..#", "#b##Ba.#", "########", "########"), move_budget=15
    ),
    LevelPlanSpec(
        rows=("########", "#####G##", "#####.c#", "#b##aB##", "#.##..##", "#SA##C##", "########"), move_budget=18
    ),
)


def _parse_level(spec: LevelPlanSpec) -> dict[str, object]:
    walls: set[tuple[int, int]] = set()
    markers: dict[str, tuple[int, int]] = {}
    for y, row in enumerate(spec.rows):
        for x, tile in enumerate(row):
            if tile == "#":
                walls.add((x, y))
            elif tile in "SGABCabc":
                markers[tile] = (x, y)
    warps: dict[tuple[int, int], tuple[int, int]] = {}
    for entry_symbol, exit_symbol in (("A", "a"), ("B", "b"), ("C", "c")):
        if entry_symbol in markers and exit_symbol in markers:
            warps[markers[entry_symbol]] = markers[exit_symbol]
    return {
        "walls": frozenset(walls),
        "start": markers["S"],
        "goal": markers["G"],
        "warps": warps,
        "move_budget": spec.move_budget,
    }


PARSED_LEVELS = tuple(_parse_level(spec) for spec in LEVELS)


class OneWayWarpMazeDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, _env) -> list[tuple[int, dict[str, int]]]:
        if self._current_level_idx is None:
            raise RuntimeError("one_way_warp_maze: missing current level index")
        level_idx = int(self._current_level_idx)
        parsed = PARSED_LEVELS[level_idx]
        walls = parsed["walls"]
        start = parsed["start"]
        goal = parsed["goal"]
        warps = parsed["warps"]
        move_budget = int(parsed["move_budget"])

        def is_goal(state: tuple[int, int, int]) -> bool:
            return (state[0], state[1]) == goal

        def expand(state: tuple[int, int, int]):
            x, y, remaining = state
            if remaining <= 0:
                return []
            out = []
            for action_id, delta in MOVE_BY_ACTION.items():
                nx = x + delta[0]
                ny = y + delta[1]
                if not (0 <= nx < 8 and 0 <= ny < 7) or (nx, ny) in walls:
                    nx, ny = x, y
                target = warps.get((nx, ny), (nx, ny))
                out.append((action_id, (target[0], target[1], remaining - 1)))
            return out

        plan = find_shortest_action_plan(
            (start[0], start[1], move_budget),
            is_goal,
            expand,
            dominance_key=lambda state: (state[0], state[1]),
            dominance_score=lambda state: int(state[2]),
        )
        if not plan:
            raise RuntimeError(f"one_way_warp_maze: no winning plan for level {level_idx}")

        program = [(int(action_id), {}) for action_id in plan]
        program.append((1, {}))
        return program


AGENT_CLASS = OneWayWarpMazeDslAgent
