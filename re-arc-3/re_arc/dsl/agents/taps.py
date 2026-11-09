from __future__ import annotations

from ..core import DslAgent, camera_grid_to_display, observation_level_index


class TapsDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=8)
        self._current_level_idx = None
        self._action_idx = 0
        self._programs: dict[int, list[tuple[int, dict[str, int]]]] = {}

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def _sync_level(self, env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            return
        self.mark_levels_solved(level_idx)
        reset_level = bool(getattr(observation, "full_reset", False))
        if self._current_level_idx is None or level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
            if level_idx not in self._programs:
                self._programs[level_idx] = self._build_level_program(env)
            return
        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def _build_level_program(self, env):
        game = env._game
        level = game.current_level
        targets = list(level.get_data("targets") or [])
        out: list[tuple[int, dict[str, int]]] = []
        for gx, gy in targets:
            dx, dy = camera_grid_to_display(game.camera, int(gx), int(gy))
            out.append((6, {"x": int(dx), "y": int(dy)}))
        return out

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in taps observation.")
        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                "Taps DSL program exhausted before advancing to the next level. "
                f"level={self._current_level_idx} steps={len(program)}"
            )
        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
