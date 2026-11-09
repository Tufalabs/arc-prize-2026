from __future__ import annotations

from ..core import DslAgent, camera_grid_to_display, observation_level_index

LEVEL_CLICKS = (
    ((0, 0),),
    ((0, 0), (2, 3)),
    ((0, 2), (0, 5), (2, 4)),
    ((0, 2), (1, 3), (4, 2), (3, 5), (2, 6), (4, 4)),
    ((0, 3), (0, 0), (5, 3), (10, 4), (10, 0), (5, 4), (0, 7), (4, 10), (5, 5), (5, 0), (5, 6)),
    (
        (0, 2),
        (0, 4),
        (5, 3),
        (10, 2),
        (10, 5),
        (5, 4),
        (0, 6),
        (10, 7),
        (5, 0),
        (5, 5),
        (5, 5),
        (0, 2),
        (0, 4),
        (5, 3),
        (10, 2),
        (10, 5),
        (5, 4),
        (0, 6),
        (5, 0),
        (5, 7),
        (5, 5),
        (5, 7),
        (0, 2),
        (0, 4),
        (5, 3),
        (10, 2),
        (10, 5),
        (5, 4),
        (5, 0),
        (5, 9),
    ),
    (
        (0, 2),
        (0, 4),
        (6, 3),
        (13, 3),
        (13, 5),
        (6, 4),
        (0, 8),
        (0, 13),
        (6, 5),
        (13, 6),
        (13, 8),
        (13, 10),
        (6, 0),
        (6, 6),
        (6, 6),
        (0, 2),
        (0, 4),
        (6, 3),
        (13, 3),
        (13, 5),
        (6, 4),
        (0, 8),
        (0, 13),
        (6, 5),
        (13, 6),
        (6, 0),
        (6, 8),
        (6, 6),
        (6, 8),
        (0, 2),
        (0, 4),
        (6, 3),
        (13, 3),
        (13, 5),
        (6, 4),
        (0, 8),
        (0, 13),
        (6, 5),
        (13, 6),
        (13, 8),
        (6, 0),
        (6, 10),
        (6, 6),
        (6, 8),
        (6, 10),
        (0, 2),
        (0, 4),
        (6, 3),
        (13, 3),
        (13, 5),
        (6, 4),
        (0, 8),
        (0, 13),
        (6, 5),
        (6, 0),
        (6, 12),
    ),
)


class IslandInverterDslAgent(DslAgent):
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
            raise RuntimeError("Missing `levels_completed` in Island Inverter observation.")
        self.mark_levels_solved(level_idx)
        if self._current_level_idx != level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
        program = LEVEL_CLICKS[level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"Island Inverter DSL exhausted on level {level_idx + 1}.")

        row, col = program[self._action_idx]
        self._action_idx += 1
        game = env._game
        pitch, origin_x, origin_y = game._geometry()
        grid_x = origin_x + col * pitch + pitch // 2
        grid_y = origin_y + row * pitch + pitch // 2
        display_x, display_y = camera_grid_to_display(game.camera, grid_x, grid_y)
        return 6, {"x": display_x, "y": display_y}
