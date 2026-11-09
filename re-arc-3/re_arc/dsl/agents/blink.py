from __future__ import annotations

from collections import deque

from ..core import DslAgent, camera_grid_to_display, observation_level_index


class BlinkDslAgent(DslAgent):
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
        game = env._game
        level = game.current_level

        width = int(level.get_data("width"))
        height = int(level.get_data("height"))
        goal = tuple(int(v) for v in level.get_data("goal"))
        walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}

        player = level.get_sprites_by_name("player")[0]
        start = (int(player.x), int(player.y), -1)

        anchor_ids: list[str] = []
        anchor_points: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        for entry in level.get_data("anchors") or []:
            aid = str(entry["id"])
            p0 = (int(entry["points"][0][0]), int(entry["points"][0][1]))
            p1 = (int(entry["points"][1][0]), int(entry["points"][1][1]))
            anchor_ids.append(aid)
            anchor_points[aid] = (p0, p1)
        anchor_ids.sort()

        def blocked(x: int, y: int):
            if x < 0 or y < 0 or x >= width or y >= height:
                return True
            return (x, y) in walls

        queue = deque([start])
        previous = {start: None}
        previous_action: dict[tuple[int, int, int], tuple[str, int | None]] = {}
        goal_state = None

        while queue:
            state = queue.popleft()
            x, y, selected = state
            if (x, y) == goal:
                goal_state = state
                break

            for action_id, (dx, dy) in ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))):
                nx, ny = x + dx, y + dy
                if blocked(nx, ny):
                    continue
                nxt = (nx, ny, selected)
                if nxt in previous:
                    continue
                previous[nxt] = state
                previous_action[nxt] = ("move", action_id)
                queue.append(nxt)

            for idx, _aid in enumerate(anchor_ids):
                if idx == selected:
                    continue
                nxt = (x, y, idx)
                if nxt in previous:
                    continue
                previous[nxt] = state
                previous_action[nxt] = ("select", idx)
                queue.append(nxt)

            if selected >= 0:
                aid = anchor_ids[selected]
                p0, p1 = anchor_points[aid]
                if (x, y) == p0:
                    nxt = (p1[0], p1[1], selected)
                elif (x, y) == p1:
                    nxt = (p0[0], p0[1], selected)
                else:
                    nxt = None
                if nxt is not None and nxt not in previous:
                    previous[nxt] = state
                    previous_action[nxt] = ("blink", None)
                    queue.append(nxt)

        if goal_state is None:
            raise RuntimeError("blink DSL could not find a path for current level")

        actions = []
        cursor = goal_state
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]  # type: ignore[index]
        actions.reverse()

        program: list[tuple[int, dict[str, int]]] = []
        for kind, payload in actions:
            if kind == "move":
                program.append((int(payload), {}))
                continue
            if kind == "blink":
                program.append((5, {}))
                continue

            idx = int(payload)
            aid = anchor_ids[idx]
            anchor_pos = anchor_points[aid][0]
            cx, cy = camera_grid_to_display(game.camera, int(anchor_pos[0]), int(anchor_pos[1]))
            program.append((6, {"x": int(cx), "y": int(cy)}))

        return program

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in blink observation")

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                f"blink DSL program exhausted before level advance level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
