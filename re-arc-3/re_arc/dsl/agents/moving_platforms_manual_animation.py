from __future__ import annotations

from collections import deque
from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent

_ENV = import_module("re_arc.environment_files.moving_platforms_manual_animation.0001.movingplatformsmanualanimation")

ActionToken = _ENV.ActionToken
CLICK_ACTION = int(_ENV.CLICK_ACTION)
LEVEL_MODELS = _ENV.LEVEL_MODELS
MOVE_DELTAS = _ENV.MOVE_DELTAS
WAIT_ACTION = int(_ENV.WAIT_ACTION)
initial_level_state = _ENV.initial_level_state
transition_level_state = _ENV.transition_level_state


class MovingPlatformsManualAnimationDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_MODELS))
        self._building_level_idx: int | None = None

    @property
    def _agent_tag(self) -> str:
        return "moving_platforms_manual_animation"

    def _on_new_level(self, env, level_idx: int):
        self._building_level_idx = int(level_idx)
        super()._on_new_level(env, level_idx)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        if self._building_level_idx is None:
            raise RuntimeError("moving_platforms_manual_animation: missing level index while building program")

        level_idx = int(self._building_level_idx)
        if level_idx < 0 or level_idx >= len(LEVEL_MODELS):
            raise RuntimeError(f"moving_platforms_manual_animation: invalid level index {level_idx}")

        model = LEVEL_MODELS[level_idx]
        start = initial_level_state(model)

        base_tokens = [ActionToken(action_id=int(aid)) for aid in sorted(MOVE_DELTAS)]
        base_tokens.append(ActionToken(action_id=WAIT_ACTION))
        for lx, ly in model.levers:
            base_tokens.append(ActionToken(action_id=CLICK_ACTION, click_x=int(lx), click_y=int(ly)))

        queue: deque[tuple[object, int]] = deque([(start, 0)])
        previous: dict[object, object | None] = {start: None}
        previous_token: dict[object, ActionToken] = {}

        goal_state = None

        while queue:
            state, depth = queue.popleft()
            if depth >= int(model.time_limit):
                continue

            for token in base_tokens:
                next_state, won = transition_level_state(model, state, token, camera=env._game.camera)
                if next_state is None:
                    continue
                if next_state in previous:
                    continue
                previous[next_state] = state
                previous_token[next_state] = token
                if won:
                    goal_state = next_state
                    queue.clear()
                    break
                queue.append((next_state, depth + 1))

        if goal_state is None:
            raise RuntimeError(
                f"moving_platforms_manual_animation solver failed to find a level program for level={level_idx}."
            )

        token_plan: list[ActionToken] = []
        cursor = goal_state
        while previous[cursor] is not None:
            token_plan.append(previous_token[cursor])
            cursor = previous[cursor]
        token_plan.reverse()

        actions: list[tuple[int, dict[str, int]]] = []
        for token in token_plan:
            if int(token.action_id) == CLICK_ACTION:
                actions.append((CLICK_ACTION, {"x": int(token.click_x), "y": int(token.click_y)}))
            else:
                actions.append((int(token.action_id), {}))
        return actions


AGENT_CLASS = MovingPlatformsManualAnimationDslAgent
