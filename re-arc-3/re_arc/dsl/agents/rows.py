from __future__ import annotations

from ..core import DslAgent, observation_level_index


class RowsDslAgent(DslAgent):
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

        if self._current_level_idx is None or self._current_level_idx != level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
            if level_idx not in self._programs:
                self._programs[level_idx] = self._build_program(env)
            return

        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def _build_program(self, env):
        game = env._game
        level = game.current_level

        n = int(level.get_data("size"))
        cell = int(level.get_data("cell"))
        ox = int(level.get_data("ox"))
        oy = int(level.get_data("oy"))
        mode = int(level.get_data("start_mode") or 0) % 2
        recipe = [
            {"mode": int(step["mode"]) % 2, "index": int(step["index"])} for step in (level.get_data("recipe") or [])
        ]

        program: list[tuple[int, dict[str, int]]] = []
        for step in recipe:
            step_mode = int(step["mode"])
            idx = int(step["index"])

            if mode != step_mode:
                program.append((5, {}))
                mode ^= 1

            if step_mode == 0:
                gx = 1
                gy = idx + 1
            else:
                gx = idx + 1
                gy = 1

            if gx < 1 or gx > n or gy < 1 or gy > n:
                raise RuntimeError(f"rows DSL invalid recipe step: {step}")

            click_x = ox + gx * cell + cell // 2
            click_y = oy + gy * cell + cell // 2
            program.append((6, {"x": click_x, "y": click_y}))

        return program

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in rows observation")

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"rows DSL program exhausted before level advance level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
