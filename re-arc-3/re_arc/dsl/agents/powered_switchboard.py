from __future__ import annotations

from ..core import DslAgent, camera_grid_to_display, observation_level_index

LEVEL_CLICKS = (
    "R",
    "A R",
    "H Q H",
    "R J R K J R M K J R U N M K J R P Q U T N M S K J R",
    "H N1 H N2 N1 H N3 N4 N2 N1 H N5 N4 N2 N1 H N6 N5 N4 N2 N1 N9 H N7 N3 N8 N10 N9 N11 N6 N5 N4 N2 N1 H",
    "R A1 B1 R A2 A1 B2 B1 R A3 A2 A1 B3 B2 B1 R A5 A4 A6 A5 A4 A7 A6 A5 A4 "
    "A8 A7 A6 A5 A4 B4 B5 B4 B6 B5 B4 B7 B6 B5 B4 B8 B7 B6 B5 B4",
    "R B1 R A1 B2 R B3 B2 R B4 B3 B2 R B5 B4 B3 B2 A2 R A3 A2 A6 A5 B6 B5 R "
    "A7 A6 B7 B6 B5 A5 R A8 A7 A6 A5 B8 B7 B6 B5 R A9 A8 A7 A6 B9 B8 B7 B6 B5 A5 R",
)


class PoweredSwitchboardDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_CLICKS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Powered Switchboard observation is missing level progress.")
        self.mark_levels_solved(level_idx)
        if level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0

        program = LEVEL_CLICKS[level_idx].split()
        if self._action_idx >= len(program):
            raise RuntimeError(f"Powered Switchboard program exhausted on level {level_idx}.")
        switch_name = program[self._action_idx]
        self._action_idx += 1

        game = env._game
        cx, cy = game.spec.centers[switch_name]
        x, y = camera_grid_to_display(game.camera, cx, cy)
        return 6, {"x": int(x), "y": int(y)}


AGENT_CLASS = PoweredSwitchboardDslAgent
