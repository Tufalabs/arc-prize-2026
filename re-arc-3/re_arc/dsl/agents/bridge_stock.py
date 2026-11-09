from __future__ import annotations

from ..core import DslAgent, observation_level_index


def _expand(*parts: tuple[int, int] | int) -> list[int]:
    out: list[int] = []
    for part in parts:
        if isinstance(part, int):
            out.append(part)
        else:
            action, count = part
            out.extend([action] * count)
    return out


U, D, L, R = 1, 2, 3, 4

LEVEL_PROGRAMS = [
    _expand((R, 2)),
    _expand((U, 2), (D, 2), (R, 2)),
    _expand((U, 2), (L, 2), (R, 2), D, (R, 5)),
    _expand((U, 3), (R, 2), D, (R, 3), (L, 3), U, (L, 2), (D, 3), (L, 7)),
    _expand(
        (R, 4),
        (D, 2),
        (L, 2),
        D,
        L,
        R,
        U,
        (R, 2),
        (U, 2),
        (U, 5),
        (L, 2),
        (R, 4),
        (L, 2),
        (D, 5),
        (R, 5),
        (U, 3),
        (R, 2),
    ),
    _expand(
        (L, 3),
        D,
        L,
        D,
        (R, 2),
        (U, 2),
        (R, 2),
        (U, 3),
        (L, 2),
        (R, 4),
        D,
        (R, 4),
        (D, 3),
        (L, 2),
        D,
        (L, 2),
        (D, 2),
        (R, 2),
        (U, 3),
        (R, 2),
        (U, 3),
        (L, 4),
        (L, 2),
        (D, 2),
        (L, 2),
        (D, 2),
        (L, 4),
        (D, 2),
        (R, 6),
    ),
    _expand(
        (R, 2),
        (D, 2),
        (L, 2),
        D,
        U,
        (R, 2),
        (U, 2),
        (U, 4),
        (R, 2),
        U,
        (R, 2),
        (D, 2),
        U,
        (L, 4),
        (D, 4),
        (R, 4),
        (D, 3),
        L,
        (D, 2),
        L,
        U,
        L,
        R,
        U,
        (R, 2),
        (U, 3),
        U,
        R,
        U,
        (R, 3),
        (L, 3),
        (D, 2),
        (R, 2),
        (D, 2),
        (R, 2),
        (D, 2),
        (L, 2),
    ),
]


class BridgeStockDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_PROGRAMS))
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is not None:
            self.mark_levels_solved(level_idx)
            if self._current_level_idx != level_idx:
                self._current_level_idx = level_idx
                self._action_idx = 0
        if self._current_level_idx is None:
            raise RuntimeError("Missing `levels_completed` in Bridge Stock observation.")
        program = LEVEL_PROGRAMS[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError("Bridge Stock DSL program exhausted before reaching WIN.")
        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}


AGENT_CLASS = BridgeStockDslAgent
