from __future__ import annotations

import importlib

from ..core import CachedProgramDslAgent

_mod = importlib.import_module("re_arc.environment_files.limited_turns.0001.limitedturns")
FREEZE_STEPS = int(_mod.FREEZE_STEPS)
build_limited_turns_level_model = _mod.build_limited_turns_level_model
plan_limited_turns_actions = _mod.plan_limited_turns_actions


class LimitedTurnsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = build_limited_turns_level_model(
            name=str(level.get_data("name") or "Level"),
            layout=list(level.get_data("layout") or []),
            turn_budget=int(level.get_data("turn_budget") or 1),
            time_max_steps=int(level.get_data("time_max_steps") or 1),
        )
        actions = plan_limited_turns_actions(model)
        actions.extend([5] * FREEZE_STEPS)
        return [(action_id, {}) for action_id in actions]
