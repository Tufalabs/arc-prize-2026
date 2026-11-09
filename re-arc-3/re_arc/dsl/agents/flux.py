from __future__ import annotations

from collections import deque

from ..core import DslAgent, camera_grid_to_display, observation_level_index

MOVE_ACTIONS = (
    (1, 0, -1),  # W
    (2, 0, 1),  # S
    (3, -1, 0),  # A
    (4, 1, 0),  # D
)


class FluxDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=5)
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

        width = int(game._width)
        height = int(game._height)
        start_x, start_y = (int(game._player_cell[0]), int(game._player_cell[1]))
        goal = (int(game._goal_cell[0]), int(game._goal_cell[1]))
        start_phase = int(game._phase) & 1
        start_bridge_mask = int(game._bridge_mask)

        walls = {tuple(int(v) for v in cell) for cell in game._wall_cells}
        phase0_tiles = {tuple(int(v) for v in cell) for cell in game._phase_tiles[0]}
        phase1_tiles = {tuple(int(v) for v in cell) for cell in game._phase_tiles[1]}
        bridge_cell_to_index = {
            tuple(int(v) for v in cell): int(idx) for cell, idx in game._bridge_cell_to_index.items()
        }

        node_specs = []
        for node_id in sorted(game._node_positions.keys()):
            nx, ny = game._node_positions[node_id]
            node_specs.append((str(node_id), (int(nx), int(ny)), int(game._node_masks.get(node_id, 0))))

        def is_passable(x: int, y: int, phase: int, bridge_mask: int) -> bool:
            if x < 0 or y < 0 or x >= width or y >= height:
                return False
            if (x, y) in walls:
                return False
            bridge_idx = bridge_cell_to_index.get((x, y))
            if bridge_idx is not None and ((bridge_mask >> bridge_idx) & 1) == 0:
                return False
            if (x, y) in phase0_tiles and phase != 0:
                return False
            return not ((x, y) in phase1_tiles and phase != 1)

        start_state = (start_x, start_y, start_phase, start_bridge_mask)
        queue = deque([start_state])
        previous: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None] = {start_state: None}
        previous_action: dict[tuple[int, int, int, int], tuple[str, int, int] | tuple[str, str] | tuple[str, int]] = {}

        goal_state = None
        while queue:
            state = queue.popleft()
            x, y, phase, bridge_mask = state
            if (x, y) == goal:
                goal_state = state
                break

            for action_id, dx, dy in MOVE_ACTIONS:
                nx, ny = x + dx, y + dy
                if not is_passable(nx, ny, phase, bridge_mask):
                    continue
                next_state = (nx, ny, phase, bridge_mask)
                if next_state in previous:
                    continue
                previous[next_state] = state
                previous_action[next_state] = ("move", int(action_id))
                queue.append(next_state)

            toggled_phase = phase ^ 1
            phase_state = (x, y, toggled_phase, bridge_mask)
            if phase_state not in previous:
                previous[phase_state] = state
                previous_action[phase_state] = ("phase", 5)
                queue.append(phase_state)

            for node_id, (_nx, _ny), node_mask in node_specs:
                if node_mask == 0:
                    continue
                next_state = (x, y, phase, bridge_mask ^ node_mask)
                if next_state in previous:
                    continue
                previous[next_state] = state
                previous_action[next_state] = ("click", str(node_id))
                queue.append(next_state)

        if goal_state is None:
            raise RuntimeError("Flux DSL could not find a solution for current level.")

        action_plan: list[tuple[str, int] | tuple[str, str]] = []
        cursor = goal_state
        while previous[cursor] is not None:
            action_plan.append(previous_action[cursor])
            cursor = previous[cursor]  # type: ignore[index]
        action_plan.reverse()

        program: list[tuple[int, dict[str, int]]] = []
        for action in action_plan:
            kind = action[0]
            if kind == "move" or kind == "phase":
                program.append((int(action[1]), {}))
                continue

            node_id = str(action[1])
            node_pos = game._node_positions.get(node_id)
            if node_pos is None:
                raise RuntimeError(f"Flux DSL could not resolve node `{node_id}`.")
            click_x, click_y = camera_grid_to_display(game.camera, int(node_pos[0]), int(node_pos[1]))
            program.append((6, {"x": int(click_x), "y": int(click_y)}))

        if not program:
            raise RuntimeError("Flux DSL built an empty program for a non-terminal level.")
        return program

    def next_action(self, env, observation):
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in flux observation.")

        program = self._programs.get(self._current_level_idx, [])
        if self._action_idx >= len(program):
            raise RuntimeError(
                "Flux DSL program exhausted before advancing to the next level. "
                f"level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
