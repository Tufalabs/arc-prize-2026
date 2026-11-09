from __future__ import annotations

from collections import deque

from ..core import DslAgent, observation_level_index

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class RookDslAgent(DslAgent):
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
        goal = tuple(int(v) for v in level.get_data("goal"))
        beacons = [tuple(int(v) for v in item) for item in (level.get_data("beacons") or [])]

        beacon_index = {pos: idx for idx, pos in enumerate(beacons)}
        target_mask = (1 << len(beacons)) - 1

        player = level.get_sprites_by_name("player")[0]
        start_mask = 0
        bit = beacon_index.get((int(player.x), int(player.y)))
        if bit is not None:
            start_mask |= 1 << bit

        start = (int(player.x), int(player.y), start_mask)

        def blocked(x: int, y: int, mask: int):
            if x < 0 or y < 0 or x >= width or y >= height:
                return True
            if (x, y) in walls:
                return True
            return bool((x, y) == goal and mask != target_mask)

        def slide(x: int, y: int, dx: int, dy: int, mask: int):
            moved = False
            out_mask = int(mask)
            won = False
            while True:
                nx = x + dx
                ny = y + dy
                if blocked(nx, ny, out_mask):
                    break
                x, y = nx, ny
                moved = True
                bit = beacon_index.get((x, y))
                if bit is not None:
                    out_mask |= 1 << bit
                if (x, y) == goal and out_mask == target_mask:
                    won = True
                    break
            return x, y, out_mask, moved, won

        queue = deque([start])
        previous = {start: None}
        previous_action: dict[tuple[int, int, int], int] = {}
        goal_state = None

        while queue:
            x, y, mask = queue.popleft()
            if (x, y) == goal and mask == target_mask:
                goal_state = (x, y, mask)
                break

            for action_id in (1, 2, 3, 4):
                dx, dy = MOVE_DELTAS[action_id]
                nx, ny, nmask, moved, won = slide(x, y, dx, dy, mask)
                if not moved:
                    continue
                nxt = (nx, ny, nmask)
                if nxt in previous:
                    continue
                previous[nxt] = (x, y, mask)
                previous_action[nxt] = action_id
                if won:
                    goal_state = nxt
                    queue.clear()
                    break
                queue.append(nxt)
            if goal_state is not None:
                break

        if goal_state is None:
            raise RuntimeError("rook DSL could not find a path for current level")

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
            raise RuntimeError("Missing `levels_completed` in rook observation")

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"rook DSL program exhausted before level advance level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
