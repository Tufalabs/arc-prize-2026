from __future__ import annotations

from collections import deque

from ..core import DslAgent, observation_level_index


class WeavDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)
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
                self._programs[level_idx] = self._build_program(env)
            return

        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def _build_program(self, env):
        level = env._game.current_level
        width = int(level.get_data("width"))
        height = int(level.get_data("height"))
        goal = tuple(int(v) for v in level.get_data("goal"))
        start_mode = int(level.get_data("start_mode") or 0) % 2

        walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        phase_tiles = {
            0: {tuple(int(v) for v in item) for item in (level.get_data("phase0_tiles") or [])},
            1: {tuple(int(v) for v in item) for item in (level.get_data("phase1_tiles") or [])},
        }

        player = level.get_sprites_by_name("player")[0]
        start = (int(player.x), int(player.y), start_mode)

        def blocked(x: int, y: int, mode: int):
            if x < 0 or y < 0 or x >= width or y >= height:
                return True
            if (x, y) in walls:
                return True
            if (x, y) in phase_tiles[0] and mode != 0:
                return True
            return bool((x, y) in phase_tiles[1] and mode != 1)

        queue = deque([start])
        previous = {start: None}
        previous_action: dict[tuple[int, int, int], int] = {}
        goal_state = None

        while queue:
            state = queue.popleft()
            x, y, mode = state
            if (x, y) == goal:
                goal_state = state
                break

            for action_id, (dx, dy) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))):
                nx, ny = x + dx, y + dy
                if blocked(nx, ny, mode):
                    continue
                nxt = (nx, ny, mode)
                if nxt in previous:
                    continue
                previous[nxt] = state
                previous_action[nxt] = action_id
                queue.append(nxt)

            toggled = (x, y, mode ^ 1)
            if toggled not in previous:
                previous[toggled] = state
                previous_action[toggled] = 5
                queue.append(toggled)

        if goal_state is None:
            raise RuntimeError("weav DSL could not find a path for current level")

        actions: list[int] = []
        cursor = goal_state
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]  # type: ignore[index]
        actions.reverse()

        return [(action_id, {}) for action_id in actions]

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in weav observation")

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"weav DSL program exhausted before level advance level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
