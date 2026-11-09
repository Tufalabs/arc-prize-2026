from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent

_env_mod = import_module("re_arc.environment_files.territory_capture.0001.territorycapture")
WAIT_ACTION_ID = int(_env_mod.WAIT_ACTION_ID)
make_level_macro_plan = _env_mod.make_level_macro_plan
simulate_plan_until_win = _env_mod.simulate_plan_until_win


class TerritoryCaptureDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=1)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level_idx = int(getattr(game, "level_index", 0))
        actions = [int(action_id) for action_id in make_level_macro_plan(level_idx)]
        if not actions:
            raise RuntimeError(f"territory_capture DSL produced an empty macro plan for level={level_idx}.")

        # Pad with deterministic waits so replay still completes when goal-opening
        # timing jitters.
        actions.extend([WAIT_ACTION_ID] * 96)
        if not simulate_plan_until_win(level_idx, actions, max_steps=max(5000, len(actions) + 2500)):
            raise RuntimeError(f"territory_capture DSL macro plan failed simulation for level={level_idx}.")
        return [(int(action_id), {}) for action_id in actions]


AGENT_CLASS = TerritoryCaptureDslAgent
