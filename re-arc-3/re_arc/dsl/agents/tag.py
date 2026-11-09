from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.search import bfs_plan

_tag_mod = import_module("re_arc.environment_files.tag.0001.tag")

TagSimState = _tag_mod.TagSimState
TagLevelModel = _tag_mod.TagLevelModel
advance_tag_state = _tag_mod.advance_tag_state
ACTION_WAIT = int(_tag_mod.ACTION_WAIT)


class TagDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level_idx = int(game.current_level.get_data("level_idx") or 0)
        models: list[TagLevelModel] = list(_tag_mod.TAG_LEVEL_MODELS)
        if level_idx < 0 or level_idx >= len(models):
            raise RuntimeError(f"Tag DSL could not resolve level model for index {level_idx}.")

        model = models[level_idx]
        start = _tag_mod._initial_state(model)

        def is_goal(state: TagSimState) -> bool:
            return bool(state.won)

        def expand(state: TagSimState):
            if state.won:
                return
            for action_id in (1, 2, 3, 4, 5, 6):
                next_state, effects = advance_tag_state(model, state, int(action_id))
                if effects.lose:
                    continue
                yield int(action_id), next_state, 1.0

        def dominance_key(state: TagSimState) -> tuple:
            return (
                bool(state.won),
                int(state.player_x),
                int(state.player_y),
                int(state.player_dir),
                int(state.target_x),
                int(state.target_y),
                int(state.target_prev_dir),
                bool(state.door_open),
                bool(state.moving_gate_open),
                bool(state.exhausted),
            )

        def dominance_score(state: TagSimState) -> float:
            return float(state.time_left)

        actions = bfs_plan(start, is_goal, expand, dominance_key=dominance_key, dominance_score=dominance_score)
        if actions is None:
            raise RuntimeError(f"Tag DSL could not find a winning plan for level {level_idx + 1}.")

        program: list[tuple[int, dict[str, int]]] = []
        for action_id in actions:
            if int(action_id) == ACTION_WAIT:
                program.append((int(action_id), {"x": -1, "y": -1}))
            else:
                program.append((int(action_id), {}))
        return program


AGENT_CLASS = TagDslAgent
