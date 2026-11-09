from __future__ import annotations

from collections import deque

from ..core import DslAgent, observation_level_index

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class SyncDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)
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
        level = env._game.current_level

        width = int(level.get_data("width"))
        height = int(level.get_data("height"))
        walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        alpha_goal = tuple(int(v) for v in level.get_data("alpha_goal"))
        beta_goal = tuple(int(v) for v in level.get_data("beta_goal"))
        start_active = int(level.get_data("start_active") or 0) % 2

        alpha = level.get_sprites_by_name("alpha")[0]
        beta = level.get_sprites_by_name("beta")[0]
        start = (int(alpha.x), int(alpha.y), int(beta.x), int(beta.y), start_active)

        queue = deque([start])
        previous = {start: None}
        previous_action: dict[tuple[int, int, int, int, int], int] = {}
        goal_state = None

        def blocked(x: int, y: int):
            if x < 0 or y < 0 or x >= width or y >= height:
                return True
            return (x, y) in walls

        while queue:
            ax, ay, bx, by, active = queue.popleft()
            if (ax, ay) == alpha_goal and (bx, by) == beta_goal:
                goal_state = (ax, ay, bx, by, active)
                break

            toggled = (ax, ay, bx, by, active ^ 1)
            if toggled not in previous:
                previous[toggled] = (ax, ay, bx, by, active)
                previous_action[toggled] = 5
                queue.append(toggled)

            for action_id in (1, 2, 3, 4):
                dx, dy = MOVE_DELTAS[action_id]
                nax, nay, nbx, nby = ax, ay, bx, by

                if active == 0:
                    tx, ty = ax + dx, ay + dy
                    if not blocked(tx, ty) and (tx, ty) != (bx, by):
                        nax, nay = tx, ty
                else:
                    tx, ty = bx + dx, by + dy
                    if not blocked(tx, ty) and (tx, ty) != (ax, ay):
                        nbx, nby = tx, ty

                nxt = (nax, nay, nbx, nby, active)
                if nxt in previous:
                    continue
                previous[nxt] = (ax, ay, bx, by, active)
                previous_action[nxt] = action_id
                queue.append(nxt)

        if goal_state is None:
            raise RuntimeError("sync DSL could not find a path for current level")

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
            raise RuntimeError("Missing `levels_completed` in sync observation")

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"sync DSL program exhausted before level advance level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
