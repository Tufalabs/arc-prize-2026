from __future__ import annotations

from importlib import import_module

from ..core import DslAgent

ENV_MOD = import_module("re_arc.environment_files.sliding_rows.0001.slid")
LEVEL_SPECS = ENV_MOD.LEVEL_SPECS

LEFT_PAD_X = int(ENV_MOD.LEFT_PAD_X)
RIGHT_PAD_X = int(ENV_MOD.RIGHT_PAD_X)
BODY_Y = int(ENV_MOD.BODY_Y)
ROW_STRIDE = int(ENV_MOD.ROW_STRIDE)


def _row_center_y(strip_idx: int) -> int:
    return BODY_Y + (ROW_STRIDE * strip_idx) + 1


def _build_program(offsets: tuple[int, ...]) -> list[tuple[int, dict[str, int]]]:
    best_target = 0
    best_cost = None
    for target in range(12):
        cost = 0
        for offset in offsets:
            right = (target - offset) % 12
            left = (offset - target) % 12
            cost += min(left, right)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_target = target

    program: list[tuple[int, dict[str, int]]] = []
    for strip_idx, offset in enumerate(offsets):
        right = (best_target - offset) % 12
        left = (offset - best_target) % 12
        if left <= right:
            x = LEFT_PAD_X + 1
            steps = left
        else:
            x = RIGHT_PAD_X + 1
            steps = right
        y = _row_center_y(strip_idx)
        for _ in range(steps):
            program.append((6, {"x": x, "y": y}))

    program.append((1, {}))
    return program


LEVEL_PROGRAMS = [_build_program(spec.offsets) for spec in LEVEL_SPECS]


class SlidingRowsDslAgent(DslAgent):
    def __init__(self, game_id: str = "sliding_rows-0001"):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def _get_level_idx(self, observation):
        raw = getattr(observation, "levels_completed", None)
        if raw is None:
            return None
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            return None
        return max(0, min(idx, len(LEVEL_PROGRAMS) - 1))

    def _sync_level(self, observation):
        level_idx = self._get_level_idx(observation)
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

    def next_action(self, _env, observation):
        self._sync_level(observation)
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in sliding_rows observation.")

        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(
                "sliding_rows DSL program exhausted before level advance "
                f"level={self._current_level_idx} steps={len(program)}"
            )

        action = program[self._action_idx]
        self._action_idx += 1
        return action


AGENT_CLASS = SlidingRowsDslAgent
