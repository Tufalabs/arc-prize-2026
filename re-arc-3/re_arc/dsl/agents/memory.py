from __future__ import annotations

from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent

_env_mod = import_module("re_arc.environment_files.memory.0001.memory")

TOTAL_LEVELS = int(_env_mod.LEVEL_COUNT)
CLICK_ACTION_ID = int(_env_mod.ACTION_CLICK_ID)


class MemoryDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=TOTAL_LEVELS)

    def reset_episode(self) -> None:
        super().reset_episode()
        # Memory layouts are regenerated per episode, so clear cached plans.
        self._programs.clear()

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = getattr(env, "_game", None)
        if game is None or not hasattr(game, "solver_program_for_level"):
            raise RuntimeError("memory DSL expected env._game.solver_program_for_level.")

        level_idx = int(getattr(game, "level_index", 0))
        action_ids = game.solver_program_for_level(level_idx)
        if not action_ids:
            raise RuntimeError(f"memory DSL got an empty solver program for level {level_idx}.")
        final_pad = game.final_pad_position_for_level(level_idx)

        out: list[tuple[int, dict[str, int]]] = []
        for action_id in action_ids:
            aid = int(action_id)
            payload: dict[str, int] = {}
            if aid == CLICK_ACTION_ID:
                payload = {"x": int(final_pad[0]), "y": int(final_pad[1])}
            out.append((aid, payload))
        return out


AGENT_CLASS = MemoryDslAgent
