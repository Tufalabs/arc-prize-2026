from __future__ import annotations

from collections import deque
from importlib import import_module

from ..core import DslAgent

_ENV_MOD = import_module("re_arc.environment_files.szgt.0001.szgt")
LEVEL_SPECS = _ENV_MOD.LEVEL_SPECS
MOVE_DELTAS = _ENV_MOD.MOVE_DELTAS
TOGGLE_ACTION = _ENV_MOD.TOGGLE_ACTION


def _occupied_cells(cx: int, cy: int, is_large: bool) -> tuple[tuple[int, int], ...]:
    if not is_large:
        return ((cx, cy),)
    return tuple((x, y) for y in range(cy - 1, cy + 2) for x in range(cx - 1, cx + 2))


def _gate_open(level_index: int, gate_index: int, mask: int) -> bool:
    return bool(mask & (1 << LEVEL_SPECS[level_index].gates[gate_index].plate_index))


def _is_passable(level_index: int, x: int, y: int, mask: int) -> bool:
    spec = LEVEL_SPECS[level_index]
    if not (0 <= x < 16 and 0 <= y < 15):
        return False
    if (x, y) in spec.floor_cells or (x, y) in spec.exit_cells:
        return True
    for plate in spec.plates:
        if (x, y) in plate.cells:
            return True
    for gate_index, gate in enumerate(spec.gates):
        if (x, y) in gate.cells:
            return _gate_open(level_index, gate_index, mask)
    return False


def _can_occupy(level_index: int, cx: int, cy: int, is_large: bool, mask: int) -> bool:
    return all(_is_passable(level_index, x, y, mask) for x, y in _occupied_cells(cx, cy, is_large))


def _activate_mask(level_index: int, cx: int, cy: int, is_large: bool, mask: int) -> int:
    if not is_large:
        return mask
    occupied = frozenset(_occupied_cells(cx, cy, True))
    next_mask = int(mask)
    for plate_index, plate in enumerate(LEVEL_SPECS[level_index].plates):
        if occupied == plate.cells:
            next_mask |= 1 << plate_index
    return next_mask


def _bfs_program(level_index: int) -> list[int]:
    spec = LEVEL_SPECS[level_index]
    start = (spec.start_center[0], spec.start_center[1], bool(spec.start_large), 0)
    queue = deque([(start, [])])
    visited = {start}
    actions = [1, 2, 3, 4, TOGGLE_ACTION]

    while queue:
        (cx, cy, is_large, mask), path = queue.popleft()
        if (cx, cy) in spec.exit_cells:
            return path

        for action_id in actions:
            next_cx, next_cy, next_large = cx, cy, is_large
            if action_id in MOVE_DELTAS:
                dx, dy = MOVE_DELTAS[action_id]
                trial_cx = cx + dx
                trial_cy = cy + dy
                if _can_occupy(level_index, trial_cx, trial_cy, is_large, mask):
                    next_cx, next_cy = trial_cx, trial_cy
            elif is_large:
                next_large = False
            elif _can_occupy(level_index, cx, cy, True, mask):
                next_large = True

            next_mask = _activate_mask(level_index, next_cx, next_cy, next_large, mask)
            state = (next_cx, next_cy, next_large, next_mask)
            if state in visited:
                continue
            visited.add(state)
            queue.append((state, [*path, action_id]))

    raise RuntimeError(f"szgt level {level_index} has no BFS solution")


_LEVEL_PROGRAMS = tuple(_bfs_program(level_index) for level_index in range(len(LEVEL_SPECS)))


class SzgtDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_LEVEL_PROGRAMS))
        self._level_index: int | None = None
        self._action_index = 0

    def reset_episode(self):
        super().reset_episode()
        self._level_index = None
        self._action_index = 0

    def _sync_level(self, observation) -> None:
        raw_level = getattr(observation, "levels_completed", None)
        if raw_level is None:
            return
        try:
            level_index = int(raw_level)
        except (TypeError, ValueError):
            return
        level_index = max(0, min(level_index, len(_LEVEL_PROGRAMS) - 1))
        if self._level_index != level_index:
            self._level_index = level_index
            self._action_index = 0

    def next_action(self, _env, observation):
        self._sync_level(observation)
        if self._level_index is None:
            raise RuntimeError("szgt DSL agent requires observation.levels_completed")
        program = _LEVEL_PROGRAMS[self._level_index]
        if self._action_index >= len(program):
            raise RuntimeError(f"szgt level {self._level_index} program exhausted")
        action_id = int(program[self._action_index])
        self._action_index += 1
        return action_id, {}


AGENT_CLASS = SzgtDslAgent
