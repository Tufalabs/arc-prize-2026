from __future__ import annotations

from collections import deque

from ..core import DslAgent, observation_level_index


class OrbxDslAgent(DslAgent):
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
                self._programs[level_idx] = self._build_program(env)
            return

        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def _build_program(self, env):
        level = env._game.current_level
        width = int(level.get_data("width"))
        height = int(level.get_data("height"))
        walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}

        pilot = level.get_sprites_by_name("pilot")[0]
        orb = level.get_sprites_by_name("orb")[0]

        start = (int(pilot.x), int(pilot.y), int(orb.x), int(orb.y))
        pilot_goal = tuple(int(v) for v in level.get_data("pilot_goal"))
        orb_goal = tuple(int(v) for v in level.get_data("orb_goal"))
        goal = (pilot_goal[0], pilot_goal[1], orb_goal[0], orb_goal[1])

        def blocked(x: int, y: int):
            if x < 0 or y < 0 or x >= width or y >= height:
                return True
            return (x, y) in walls

        queue = deque([start])
        previous = {start: None}
        previous_action: dict[tuple[int, int, int, int], int] = {}

        while queue:
            state = queue.popleft()
            if state == goal:
                break

            px, py, ox, oy = state
            for action_id, (dx, dy) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))):
                npx, npy = px + dx, py + dy
                if blocked(npx, npy) or (npx, npy) == (ox, oy):
                    npx, npy = px, py

                nox, noy = ox - dx, oy - dy
                if blocked(nox, noy) or (nox, noy) == (npx, npy):
                    nox, noy = ox, oy

                nxt = (npx, npy, nox, noy)
                if nxt in previous:
                    continue
                previous[nxt] = state
                previous_action[nxt] = action_id
                queue.append(nxt)

        if goal not in previous:
            raise RuntimeError("orbx DSL could not find a path for current level")

        actions: list[int] = []
        cursor = goal
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]  # type: ignore[index]
        actions.reverse()

        return [(action_id, {}) for action_id in actions]

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in orbx observation")

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"orbx DSL program exhausted before level advance level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
