from __future__ import annotations

from collections import deque

from ..core import DslAgent, observation_level_index


class TurnDslAgent(DslAgent):
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
        start_phase = int(level.get_data("start_phase") or 0) % 4

        gate_phase = {}
        for entry in level.get_data("gate_phases") or []:
            gate_phase[(int(entry["x"]), int(entry["y"]))] = int(entry["phase"]) % 4

        player = level.get_sprites_by_name("player")[0]
        start = (int(player.x), int(player.y), start_phase)
        goal = tuple(int(v) for v in level.get_data("goal"))

        def blocked(x: int, y: int, phase: int):
            if x < 0 or y < 0 or x >= width or y >= height:
                return True
            if (x, y) in walls:
                return True
            required = gate_phase.get((x, y))
            return bool(required is not None and required != phase)

        queue = deque([start])
        previous = {start: None}
        previous_action: dict[tuple[int, int, int], int] = {}
        goal_state = None

        while queue:
            state = queue.popleft()
            x, y, phase = state
            if (x, y) == goal:
                goal_state = state
                break

            for action_id, (dx, dy) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))):
                nx, ny = x + dx, y + dy
                moved = not blocked(nx, ny, phase)
                if not moved:
                    nx, ny = x, y

                next_phase = phase if (nx, ny) == goal else (phase + 1) % 4

                nxt = (nx, ny, next_phase)
                if nxt in previous:
                    continue
                previous[nxt] = state
                previous_action[nxt] = action_id
                queue.append(nxt)

        if goal_state is None:
            for state in previous:
                if (state[0], state[1]) == goal:
                    goal_state = state
                    break

        if goal_state is None:
            raise RuntimeError("turn DSL could not find a path for current level")

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
            raise RuntimeError("Missing `levels_completed` in turn observation")

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"turn DSL program exhausted before level advance level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
