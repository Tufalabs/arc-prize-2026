from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.grid import grid_to_display_click
from ..solvers.search import bfs_plan

_ENV_MOD = import_module("re_arc.environment_files.timed_patrol_crossing.0001.timedpatrolcrossing")

CLICK_ACTION = _ENV_MOD.CLICK_ACTION
MOVE_DELTAS = _ENV_MOD.MOVE_DELTAS
WAIT_ACTION = _ENV_MOD.WAIT_ACTION
SearchModel = _ENV_MOD.SearchModel
SearchState = _ENV_MOD.SearchState
apply_action_transition = _ENV_MOD.apply_action_transition


class TimedPatrolCrossingDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        model: SearchModel = game.export_search_model()
        start_state: SearchState = game.export_initial_search_state()

        available_tokens: list[tuple[int, int]] = [(action_id, -1) for action_id in sorted(int(k) for k in MOVE_DELTAS)]
        available_tokens.append((int(WAIT_ACTION), -1))
        if bool(model.controls_click):
            for lever_idx in range(len(model.levers)):
                available_tokens.append((int(CLICK_ACTION), int(lever_idx)))

        GOAL = "__goal__"

        def is_goal(state: SearchState | str) -> bool:
            return state == GOAL

        def expand(state: SearchState | str):
            if state == GOAL:
                return
            for action_id, click_idx in available_tokens:
                click_pos = None
                if action_id == CLICK_ACTION and click_idx >= 0:
                    lever = model.levers[int(click_idx)]
                    click_pos = (int(lever.x), int(lever.y))
                next_state, outcome = apply_action_transition(model, state, action_id, click_pos)
                if outcome == "fail" or next_state is None:
                    continue
                next_node = GOAL if outcome == "win" else next_state
                yield (action_id, click_idx), next_node, 1.0

        def dominance_key(state: SearchState | str) -> tuple:
            if state == GOAL:
                return (GOAL,)
            return (
                int(state.player_x),
                int(state.player_y),
                int(state.door_mask),
                int(state.lever_mask),
                tuple(int(v) for guard in state.guards for v in guard),
            )

        def dominance_score(state: SearchState | str) -> float:
            if state == GOAL:
                return 0.0
            return float(state.time_remaining)

        plan = bfs_plan(start_state, is_goal, expand, dominance_key=dominance_key, dominance_score=dominance_score)
        if plan is None:
            raise RuntimeError("timed_patrol_crossing DSL could not find a valid plan.")

        program: list[tuple[int, dict[str, int]]] = []
        for action_id, click_idx in plan:
            if action_id == CLICK_ACTION and click_idx >= 0:
                lever = model.levers[int(click_idx)]
                payload = grid_to_display_click(game.camera, (int(lever.x), int(lever.y)))
                program.append((int(action_id), payload))
            else:
                program.append((int(action_id), {}))

        # One extra action is required to transition from WIN_FLASH into next level.
        program.append((int(WAIT_ACTION), {}))
        return program


AGENT_CLASS = TimedPatrolCrossingDslAgent
