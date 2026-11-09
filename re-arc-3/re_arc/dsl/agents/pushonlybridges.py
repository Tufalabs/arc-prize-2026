from __future__ import annotations

from ..core import CachedProgramDslAgent


class PushonlybridgesDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        if not hasattr(game, "solver_program_for_current_level"):
            raise RuntimeError("pushonlybridges game does not expose solver program hook.")
        return list(game.solver_program_for_current_level())


AGENT_CLASS = PushonlybridgesDslAgent
