from __future__ import annotations

from ..core import DslAgent

# fmt: off
_LS20_LEVEL_PROGRAMS = [
    {
        "name": "krg",
        "actions": [
            1, 1, 1, 1, 3, 3, 2, 3, 1, 4, 4, 4, 1, 1, 1,
        ],
    },
    {
        "name": "mgu",
        "actions": [
            4, 1, 1, 1, 1, 1, 1, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 3, 2, 3, 4, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3,
            3, 3, 3, 2, 3, 2, 2, 2, 2, 2,
        ],
    },
    {
        "name": "puq",
        "actions": [
            1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 3, 4, 4, 4, 4, 1, 4, 4, 4, 1, 1, 3, 1, 2,
            1, 4, 1, 2,
        ],
    },
    {
        "name": "tmx",
        "actions": [
            3, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 2, 3, 2, 1, 2, 3, 1, 3, 1, 1, 1, 1, 4, 1, 2, 3, 2, 2, 2, 4, 4, 4, 4,
            2, 1, 3, 3, 1, 4, 1, 1, 1, 1, 4, 1, 4, 4, 4, 2, 2, 2, 3, 3, 1, 2, 1, 2, 1, 2, 1, 4, 2, 3, 2, 3, 3, 1, 1,
            3, 3, 3,
        ],
    },
    {
        "name": "zba",
        "actions": [
            1, 3, 1, 1, 3, 2, 3, 2, 3, 3, 3, 1, 2, 1, 2, 4, 4, 4, 1, 1, 4, 4, 3, 3, 3, 4, 3, 4, 3, 4, 4, 2, 3, 2, 3,
            3, 3, 3, 1, 3, 4, 2, 4, 2, 2, 2, 2, 4, 4, 2, 2, 3, 1, 4, 4, 4, 4, 1, 4, 4, 2, 2, 2, 1,
        ],
    },
    {
        "name": "lyd",
        "actions": [
            1, 3, 1, 3, 3, 1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 1, 1, 4, 2, 2, 1, 1, 3, 1, 1, 1, 3, 3, 3, 3, 4, 4,
            4, 3, 1, 4, 4, 2, 2, 2, 2, 3, 3, 2, 1, 4, 4, 2, 2, 2, 3, 3, 4, 3, 4, 4, 1, 1, 1, 1, 1, 4, 4, 1, 1, 4, 2,
            2, 2, 2, 2,
        ],
    },
    {
        "name": "fij",
        "actions": [
            3, 3, 2, 2, 2, 2, 2, 4, 4, 3, 4, 3, 3, 4, 3, 4, 2, 1, 4, 1, 2, 1, 2, 1, 3, 3, 1, 1, 4, 4, 4, 4, 1, 4, 4,
            4, 4, 2, 2, 1, 1, 4, 1, 1, 1, 3, 2, 2, 2, 2, 2, 4, 3, 1, 1, 3, 3, 1, 1, 2, 2, 2, 2,
        ],
    },
]
# fmt: on


class Ls20DslAgent(DslAgent):
    """
    Observation-only LS20 agent.

    It uses `observation.levels_completed` to select the current level program and
    executes a precomputed action sequence for that level.
    """

    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_LS20_LEVEL_PROGRAMS))
        self._current_level_idx = None
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
        if idx < 0:
            return 0
        max_idx = len(_LS20_LEVEL_PROGRAMS) - 1
        if idx > max_idx:
            return max_idx
        return idx

    def _sync_level_from_observation(self, observation):
        new_level_idx = self._get_level_idx(observation)
        if new_level_idx is None:
            return

        self.mark_levels_solved(new_level_idx)
        reset_level = bool(getattr(observation, "full_reset", False))

        if self._current_level_idx is None:
            self._current_level_idx = new_level_idx
            self._action_idx = 0
            return

        if new_level_idx != self._current_level_idx:
            self._current_level_idx = new_level_idx
            self._action_idx = 0
            return

        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def next_action(self, _env, observation):
        self._sync_level_from_observation(observation)

        if self._current_level_idx is None:
            raise RuntimeError(
                "Missing or invalid `levels_completed` in observation. Cannot select LS20 level program."
            )

        program = _LS20_LEVEL_PROGRAMS[self._current_level_idx]["actions"]
        if self._action_idx >= len(program):
            raise RuntimeError(
                "LS20 DSL program exhausted before a new level-start frame appeared. "
                f"level={_LS20_LEVEL_PROGRAMS[self._current_level_idx]['name']} "
                f"steps_in_program={len(program)}"
            )

        action_id = int(program[self._action_idx])
        self._action_idx += 1
        return action_id, {}
