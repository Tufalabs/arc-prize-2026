from __future__ import annotations

from collections import deque

from ..core import MOVE_ACTION_BY_DELTA, DslAgent, observation_level_index


class KeysDslAgent(DslAgent):
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
            raise RuntimeError("Keys DSL could not find goal coordinates.")
        key_order = [tuple(int(v) for v in point) for point in (level.get_data("key_order") or [])]

        def in_bounds(x: int, y: int):
            return 0 <= x < width and 0 <= y < height

        def shortest_moves(
            origin: tuple[int, int], target: tuple[int, int], block_goal: bool, blocked_cells: set[tuple[int, int]]
        ):
            queue = deque([origin])
            previous = {origin: None}
            previous_action: dict[tuple[int, int], int] = {}

            while queue:
                x, y = queue.popleft()
                if (x, y) == target:
                    break
                for (dx, dy), action_id in MOVE_ACTION_BY_DELTA.items():
                    nx, ny = x + dx, y + dy
                    if not in_bounds(nx, ny):
                        continue
                    if (nx, ny) in walls:
                        continue
                    if block_goal and (nx, ny) == goal:
                        continue
                    if (nx, ny) in blocked_cells and (nx, ny) != target:
                        continue
                    if (nx, ny) in previous:
                        continue
                    previous[(nx, ny)] = (x, y)
                    previous_action[(nx, ny)] = action_id
                    queue.append((nx, ny))

            if target not in previous:
                return None

            actions: list[int] = []
            cursor = target
            while previous[cursor] is not None:
                actions.append(previous_action[cursor])
                cursor = previous[cursor]
            actions.reverse()
            return actions

        program: list[tuple[int, dict[str, int]]] = []
        cursor = start
        for idx, key_pos in enumerate(key_order):
            future_keys = set(key_order[idx + 1 :])
            segment = shortest_moves(cursor, key_pos, block_goal=True, blocked_cells=future_keys)
            if segment is None:
                raise RuntimeError(f"Keys DSL could not reach key at {key_pos}.")
            program.extend((action_id, {}) for action_id in segment)
            cursor = key_pos

        final_segment = shortest_moves(cursor, goal, block_goal=False, blocked_cells=set())
        if final_segment is None:
            raise RuntimeError("Keys DSL could not find a path to the goal.")
        program.extend((action_id, {}) for action_id in final_segment)

        return program

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in keys observation.")
        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                "Keys DSL program exhausted before advancing to the next level. "
                f"level={self._current_level_idx} steps={len(program)}"
            )
        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
