from __future__ import annotations

from itertools import product

from ..core import CachedProgramDslAgent, observation_level_index

GAME_ID = "conveyor_sort_gpt-0001"


class ConveyorSortGptDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = GAME_ID):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env):
        game = env._game
        level = game.current_level
        splitters = list(level.get_data("splitters") or [])
        goal_id = str(level.get_data("goal_id"))
        if not splitters:
            return [(5, {})]

        initial = {spec["id"]: str(spec["initial_state"]) for spec in splitters}
        best_states = None
        best_cost = None

        for picks in product((0, 1), repeat=len(splitters)):
            candidate = {}
            for idx, spec in enumerate(splitters):
                states = [str(state) for state in spec["states"]]
                candidate[str(spec["id"])] = states[picks[idx]]
            if self._trace_terminal(splitters, candidate) != goal_id:
                continue
            cost = sum(1 for spec in splitters if candidate[str(spec["id"])] != initial[str(spec["id"])])
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_states = candidate

        if best_states is None:
            raise RuntimeError("Conveyor Sort GPT DSL could not find a route to the goal.")

        program = []
        for spec in splitters:
            splitter_id = str(spec["id"])
            if best_states[splitter_id] == initial[splitter_id]:
                continue
            col, row = spec["module"]
            program.append((6, {"x": int(col) * 8 + 4, "y": int(row) * 8 + 4}))
        program.append((5, {}))
        return program

    def _trace_terminal(self, splitters, states):
        mapping = {str(spec["id"]): spec for spec in splitters}
        current = str(splitters[0]["id"])
        while True:
            spec = mapping[current]
            state = states[current]
            target = str(spec["targets"][state])
            if target not in mapping:
                return target
            current = target

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise self._missing_level_error()
        program = self._programs[level_idx]
        if self._action_idx >= len(program):
            return (5, {})
        action = program[self._action_idx]
        self._action_idx += 1
        return action


AGENT_CLASS = ConveyorSortGptDslAgent
