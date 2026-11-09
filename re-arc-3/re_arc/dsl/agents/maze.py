from __future__ import annotations

from collections import deque

from ..core import MOVE_ACTION_BY_DELTA, DslAgent, camera_grid_to_display, observation_level_index


class MazeDslAgent(DslAgent):
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
        width = int(level.grid_size[0])
        height = int(level.grid_size[1])

        walls = set()
        for y in range(height):
            for x in range(width):
                if level.get_sprite_at(x, y, tag="wall") is not None:
                    walls.add((x, y))

        player = level.get_sprites_by_name("player")[0]
        start = (int(player.x), int(player.y))
        goal = tuple(int(v) for v in (level.get_data("goal") or ()))
        if len(goal) != 2:
            raise RuntimeError("Maze DSL could not find goal coordinates.")

        doors = sorted(level.get_sprites_by_tag("door"), key=lambda sprite: (sprite.y, sprite.x))

        def in_bounds(x: int, y: int):
            return 0 <= x < width and 0 <= y < height

        queue = deque([start])
        previous = {start: None}
        previous_action: dict[tuple[int, int], int] = {}

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                break
            for (dx, dy), action_id in MOVE_ACTION_BY_DELTA.items():
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny):
                    continue
                if (nx, ny) in walls:
                    continue
                if (nx, ny) in previous:
                    continue
                previous[(nx, ny)] = (x, y)
                previous_action[(nx, ny)] = action_id
                queue.append((nx, ny))

        if goal not in previous:
            raise RuntimeError("Maze DSL could not find a path from player to goal.")

        move_actions: list[int] = []
        cursor = goal
        while previous[cursor] is not None:
            move_actions.append(previous_action[cursor])
            cursor = previous[cursor]
        move_actions.reverse()

        program: list[tuple[int, dict[str, int]]] = []
        for door in doors:
            click_x, click_y = camera_grid_to_display(game.camera, int(door.x), int(door.y))
            program.append((6, {"x": click_x, "y": click_y}))
        program.extend((action_id, {}) for action_id in move_actions)
        return program

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in maze observation.")
        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                "Maze DSL program exhausted before advancing to the next level. "
                f"level={self._current_level_idx} steps={len(program)}"
            )
        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
