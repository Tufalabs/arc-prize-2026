from __future__ import annotations

from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent, find_shortest_action_plan

_game_mod = import_module("re_arc.environment_files.hinge_bridge.0001.hingebridge")

ACTION_CLICK = _game_mod.ACTION_CLICK
MOVE_DELTAS = _game_mod.MOVE_DELTAS
LEVEL_SPECS = _game_mod.LEVEL_SPECS
anchor_click_point = _game_mod.anchor_click_point
apply_abstract_action = _game_mod.apply_abstract_action
make_initial_state = _game_mod.make_initial_state


class HingeBridgeDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "hinge_bridge-0001"):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_SPECS))

    def _build_level_program(self, env):
        game = env._game
        level_idx = int(game.level_index)
        level_spec = game.describe_level_for_solver(level_idx)
        start_state = make_initial_state(level_spec)

        click_actions = [(ACTION_CLICK, idx) for idx in range(len(level_spec.anchors))]
        move_actions = [(action_id, None) for action_id in MOVE_DELTAS]
        candidate_actions = move_actions + click_actions

        def is_goal(state):
            return state.walker in level_spec.goals

        def expand(state):
            for action_id, anchor_index in candidate_actions:
                next_state = apply_abstract_action(level_spec, state, action_id, anchor_index=anchor_index)
                if next_state.failed:
                    continue
                yield action_id * 100 + (anchor_index if anchor_index is not None else 99), next_state

        def dominance_key(state):
            return (state.walker, state.horizontal)

        def dominance_score(state):
            return int(state.remaining_budget)

        plan = find_shortest_action_plan(
            start_state=start_state,
            is_goal=is_goal,
            expand=expand,
            dominance_key=dominance_key,
            dominance_score=dominance_score,
        )
        if plan is None:
            raise RuntimeError(f"hinge_bridge DSL could not solve level {level_idx}.")

        out = []
        for encoded in plan:
            action_id = encoded // 100
            aux = encoded % 100
            if action_id == ACTION_CLICK:
                anchor = level_spec.anchors[aux]
                x, y = anchor_click_point(anchor)
                out.append((ACTION_CLICK, {"x": int(x), "y": int(y)}))
            else:
                out.append((action_id, {}))
        return out


AGENT_CLASS = HingeBridgeDslAgent
