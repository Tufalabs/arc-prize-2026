from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent

_ENV_MOD = import_module("re_arc.environment_files.pong_solo_training_walls.0001.pongsolotrainingwalls")

_deserialize_model = _ENV_MOD._deserialize_model
apply_action_transition = _ENV_MOD.apply_action_transition
choose_policy_action = _ENV_MOD.choose_policy_action
initial_search_state_from_model = _ENV_MOD.initial_search_state_from_model


class PongSoloTrainingWallsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)

        state = initial_search_state_from_model(model)
        plan: list[tuple[int, dict[str, int]]] = []
        max_steps = int(model["time_limit"]) * 4

        for _ in range(max_steps):
            action_id = int(choose_policy_action(model, state))
            plan.append((action_id, {}))

            next_state, won = apply_action_transition(model, state, action_id)
            if next_state is None:
                break
            state = next_state
            if won:
                return plan

        raise RuntimeError("pong_solo_training_walls DSL could not solve the current level")


AGENT_CLASS = PongSoloTrainingWallsDslAgent
