from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.windmolen.0001.windmolen")

LEVEL_SPECS = _ENV_MOD.LEVEL_SPECS
UP = _ENV_MOD.UP
DOWN = _ENV_MOD.DOWN
LEFT = _ENV_MOD.LEFT
RIGHT = _ENV_MOD.RIGHT


class WindmolenDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(getattr(env._game, "level_index", 0))
        spec = LEVEL_SPECS[level_idx]
        base = _ENV_MOD._parse_layout(spec["layout"])
        goal_set = {(g[0], g[1]) for g in spec["goals"]}
        start_state = _ENV_MOD.search_initial_state(spec)

        def is_goal(state):
            px, py, _ = state
            return (px, py) in goal_set

        def expand(state):
            for action_id in (UP, DOWN, LEFT, RIGHT):
                result, _won = _ENV_MOD.search_apply_action(spec, state, action_id, base)
                if result is None:
                    continue
                yield action_id, result, 1.0

        plan = bfs_plan(start_state, is_goal, expand)
        if plan is None:
            raise RuntimeError(f"windmolen DSL could not solve level {level_idx}")
        return [(int(a), {}) for a in plan]


AGENT_CLASS = WindmolenDslAgent
