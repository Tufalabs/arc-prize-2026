from __future__ import annotations

from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent
from re_arc.dsl.solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.deliver_the_package.0001.deliverthepackage")

COMPILED_LEVELS = _ENV_MOD.COMPILED_LEVELS
initial_sim_state = _ENV_MOD.initial_sim_state
package_cell = _ENV_MOD.package_cell
simulate_step = _ENV_MOD.simulate_step


class DeliverThePackageDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(COMPILED_LEVELS))

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(getattr(env, "_current_level", len(self._programs)))
        level_idx = max(0, min(level_idx, len(COMPILED_LEVELS) - 1))

        level = COMPILED_LEVELS[level_idx]
        start = initial_sim_state(level_idx)

        def expand(state):
            for action_id in (1, 2, 3, 4, 5):
                result = simulate_step(level, state, action_id)
                if result.failed:
                    continue
                yield action_id, result.state, 1.0

        plan = bfs_plan(
            start,
            lambda s: package_cell(level, s) in level.bay_cells[level.target_bay],
            expand,
            dominance_key=lambda s: (
                s.player_x,
                s.player_y,
                s.facing,
                s.carrying,
                s.package_x,
                s.package_y,
                s.door_state,
                s.drone_x,
                s.drone_dir,
                s.anim_phase,
            ),
            dominance_score=lambda s: float(s.time_remaining),
        )
        if not plan:
            raise RuntimeError(f"deliver_the_package: no plan found for level {level_idx + 1}")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = DeliverThePackageDslAgent
