from __future__ import annotations

from collections import deque

from ..core import DELTA_BY_MOVE_ACTION, DslAgent, observation_level_index


class PushDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)
        self._current_level_idx = None
        self._action_idx = 0
        self._programs: dict[int, list[tuple[int, dict[str, int]]] | None] = {}

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
        width = int(level.grid_size[0])
        height = int(level.grid_size[1])
        goal = tuple(int(v) for v in (level.get_data("goal") or ()))
        if len(goal) != 2:
            raise RuntimeError("Push DSL could not find goal coordinates.")

        walls = set()
        ice = set()
        for y in range(height):
            for x in range(width):
                if level.get_sprite_at(x, y, tag="wall") is not None:
                    walls.add((x, y))
                tile = level.get_sprite_at(x, y, tag="ice", ignore_collidable=True)
                if tile is None:
                    continue
                pixels = tile.render()
                if pixels[y - tile.y][x - tile.x] == 10:
                    ice.add((x, y))

        player = level.get_sprites_by_name("player")[0]
        crate = level.get_sprites_by_name("crate")[0]
        start_state = (int(player.x), int(player.y), int(crate.x), int(crate.y))

        def is_wall(x: int, y: int):
            return x < 0 or y < 0 or x >= width or y >= height or (x, y) in walls

        def slide_to_stop(x: int, y: int, dx: int, dy: int, blocked):
            while (x, y) in ice:
                nx, ny = x + dx, y + dy
                if blocked(nx, ny):
                    break
                x, y = nx, ny
            return x, y

        def advance(state: tuple[int, int, int, int], action_id: int):
            dx, dy = DELTA_BY_MOVE_ACTION[action_id]
            px, py, cx, cy = state

            next_x = px + dx
            next_y = py + dy
            if is_wall(next_x, next_y):
                return None

            crate_moved = False
            next_crate_x, next_crate_y = cx, cy

            if (next_x, next_y) == (cx, cy):
                beyond_x = next_x + dx
                beyond_y = next_y + dy
                if is_wall(beyond_x, beyond_y):
                    return None

                moved_crate_x, moved_crate_y = slide_to_stop(beyond_x, beyond_y, dx, dy, lambda tx, ty: is_wall(tx, ty))
                next_crate_x, next_crate_y = moved_crate_x, moved_crate_y
                crate_moved = True

            moved_player_x, moved_player_y = slide_to_stop(
                next_x, next_y, dx, dy, lambda tx, ty: is_wall(tx, ty) or (tx, ty) == (next_crate_x, next_crate_y)
            )

            next_state = (int(moved_player_x), int(moved_player_y), int(next_crate_x), int(next_crate_y))
            if next_state == state:
                return None
            won = crate_moved and (next_crate_x, next_crate_y) == goal
            return next_state, won

        queue = deque([start_state])
        previous = {start_state: None}
        previous_action: dict[tuple[int, int, int, int], int] = {}
        goal_state = None

        while queue:
            state = queue.popleft()
            for action_id in (1, 2, 3, 4):
                out = advance(state, action_id)
                if out is None:
                    continue
                next_state, won = out
                if next_state in previous:
                    continue
                previous[next_state] = state
                previous_action[next_state] = action_id
                if won:
                    goal_state = next_state
                    queue.clear()
                    break
                queue.append(next_state)

        if goal_state is None:
            return None

        actions: list[int] = []
        cursor = goal_state
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]
        actions.reverse()
        return [(action_id, {}) for action_id in actions]

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in push observation.")

        program = self._programs.get(self._current_level_idx)
        if program is None:
            raise RuntimeError(f"Push DSL has no valid action program for level={self._current_level_idx}.")

        if self._action_idx >= len(program):
            raise RuntimeError(
                "Push DSL program exhausted before advancing to the next level. "
                f"level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
