from __future__ import annotations

from collections import deque

from ..core import DslAgent, camera_grid_to_display, observation_level_index


class GlyphDslAgent(DslAgent):
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
        game = env._game
        level = game.current_level

        width = int(level.get_data("width"))
        height = int(level.get_data("height"))
        goal = tuple(int(v) for v in level.get_data("goal"))
        start_mode = int(level.get_data("start_mode") or 0) % 2
        runes = [
            {"x": int(entry["x"]), "y": int(entry["y"]), "mode": int(entry["mode"]) % 2}
            for entry in (level.get_data("runes") or [])
        ]

        walls = set()
        for y in range(height):
            for x in range(width):
                if level.get_sprite_at(x, y, tag="wall") is not None:
                    walls.add((x, y))

        player = level.get_sprites_by_name("player")[0]
        start = (int(player.x), int(player.y), start_mode, 0)

        def blocked(x: int, y: int, progress: int):
            if x < 0 or y < 0 or x >= width or y >= height:
                return True
            if (x, y) in walls:
                return True
            return bool(progress < len(runes) and (x, y) == goal)

        queue = deque([start])
        previous = {start: None}
        previous_action: dict[tuple[int, int, int, int], tuple[str, int | None]] = {}
        goal_state = None

        while queue:
            state = queue.popleft()
            x, y, mode, progress = state
            if progress >= len(runes) and (x, y) == goal:
                goal_state = state
                break

            for action_id, (dx, dy) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))):
                nx, ny = x + dx, y + dy
                if blocked(nx, ny, progress):
                    continue
                nxt = (nx, ny, mode, progress)
                if nxt in previous:
                    continue
                previous[nxt] = state
                previous_action[nxt] = ("move", action_id)
                queue.append(nxt)

            toggled = (x, y, mode ^ 1, progress)
            if toggled not in previous:
                previous[toggled] = state
                previous_action[toggled] = ("mode", 5)
                queue.append(toggled)

            if progress < len(runes):
                expected = runes[progress]
                if mode == int(expected["mode"]) and abs(x - int(expected["x"])) + abs(y - int(expected["y"])) <= 1:
                    nxt = (x, y, mode, progress + 1)
                    if nxt not in previous:
                        previous[nxt] = state
                        previous_action[nxt] = ("click", progress)
                        queue.append(nxt)

        if goal_state is None:
            raise RuntimeError("glyph DSL could not find a path for current level")

        actions = []
        cursor = goal_state
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]  # type: ignore[index]
        actions.reverse()

        program: list[tuple[int, dict[str, int]]] = []
        for kind, payload in actions:
            if kind == "move" or kind == "mode":
                program.append((int(payload), {}))
                continue

            rune_index = int(payload)
            rune = runes[rune_index]
            cx, cy = camera_grid_to_display(game.camera, int(rune["x"]), int(rune["y"]))
            program.append((6, {"x": int(cx), "y": int(cy)}))

        return program

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in glyph observation")

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"glyph DSL program exhausted before level advance level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
