from __future__ import annotations

from typing import Any

from ..core import DslAgent, camera_grid_to_display, observation_level_index

_LEVEL_BLOCKS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9)),
    ((7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (8, 4), (9, 5), (10, 6), (11, 7)),
    ((6, 3), (5, 4), (4, 5), (3, 6), (7, 4), (8, 5), (9, 7), (8, 8), (7, 9), (10, 8), (11, 9)),
)

_FLOW_TICKS: tuple[int, ...] = (17, 15, 16)


def _cell_center(cell_x: int, cell_y: int) -> tuple[int, int]:
    return 4 + cell_x * 4 + 1, 4 + cell_y * 4 + 1


class Sf01DslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_LEVEL_BLOCKS))
        self._current_level_idx: int | None = None
        self._action_idx = 0
        self._programs = self._build_programs()

    def _build_programs(self) -> dict[int, list[tuple[int, dict[str, int]]]]:
        programs: dict[int, list[tuple[int, dict[str, int]]]] = {}
        for idx, blocks in enumerate(_LEVEL_BLOCKS):
            program: list[tuple[int, dict[str, int]]] = []
            for cell_x, cell_y in blocks:
                x, y = _cell_center(cell_x, cell_y)
                program.append((6, {"x": x, "y": y}))
            program.append((5, {}))
            for _ in range(_FLOW_TICKS[idx]):
                program.append((5, {}))
            programs[idx] = program
        return programs

    def reset_episode(self) -> None:
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def _sync_level(self, _env: Any, observation: Any) -> None:
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            return
        self.mark_levels_solved(level_idx)
        reset_level = bool(getattr(observation, "full_reset", False))

        if self._current_level_idx is None or level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
            return

        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def next_action(self, env: Any, observation: Any) -> tuple[int, dict[str, int]]:
        self._sync_level(env, observation)
        if self._current_level_idx is None:
            raise RuntimeError("SF01 DSL could not determine the current level index.")

        program = self._programs[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(f"SF01 DSL program exhausted on level {self._current_level_idx}.")

        action_id, data = program[self._action_idx]
        self._action_idx += 1

        if action_id != 6:
            return action_id, data

        camera = env._game.camera
        cell_x = (int(data["x"]) - 4) // 4
        cell_y = (int(data["y"]) - 4) // 4
        grid_x, grid_y = _cell_center(cell_x, cell_y)
        display_x, display_y = camera_grid_to_display(camera, grid_x, grid_y)
        return action_id, {"x": int(display_x), "y": int(display_y)}


AGENT_CLASS = Sf01DslAgent
