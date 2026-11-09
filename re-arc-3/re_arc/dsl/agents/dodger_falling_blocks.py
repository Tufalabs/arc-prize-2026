from __future__ import annotations

from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent

_build_survival_plan = import_module(
    "re_arc.environment_files.dodger_falling_blocks.0001.dodgerfallingblocks"
).build_survival_plan


class DodgerFallingBlocksDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    @property
    def _agent_tag(self) -> str:
        return "dodger-falling-blocks"

    def _build_level_program(self, env):
        model = env._game.export_solver_state()
        plan = _build_survival_plan(model)
        if plan is None:
            raise RuntimeError("dodger_falling_blocks solver could not find a survival plan")
        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = DodgerFallingBlocksDslAgent
