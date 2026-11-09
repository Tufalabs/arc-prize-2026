from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..core import DslAgent, observation_level_index

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0), 5: (0, 0)}


@dataclass(frozen=True)
class LevelSpec:
    player_start: tuple[int, int]
    helper_start: tuple[int, int]
    marker: tuple[int, int]
    walls: frozenset[tuple[int, int]]


LEVEL_SPECS = (
    LevelSpec(player_start=(0, 0), helper_start=(6, 6), marker=(3, 3), walls=frozenset()),
    LevelSpec(player_start=(0, 3), helper_start=(6, 5), marker=(3, 3), walls=frozenset()),
    LevelSpec(player_start=(1, 3), helper_start=(0, 6), marker=(5, 3), walls=frozenset({(3, 2), (3, 3), (3, 4)})),
)


def _in_bounds(position: tuple[int, int]) -> bool:
    x, y = position
    return 0 <= x < 7 and 0 <= y < 7


def _blocked(
    position: tuple[int, int],
    *,
    mover: str,
    player_pos: tuple[int, int],
    helper_pos: tuple[int, int],
    marker: tuple[int, int],
    marker_locked: bool,
    walls: frozenset[tuple[int, int]],
) -> bool:
    if not _in_bounds(position):
        return True
    if position in walls:
        return True
    if marker_locked and position == marker:
        return True
    other = helper_pos if mover == "player" else player_pos
    if mover == "helper" and position == marker and other == marker and not marker_locked:
        return False
    return position == other


def _helper_step(
    helper_pos: tuple[int, int],
    *,
    player_pos: tuple[int, int],
    marker: tuple[int, int],
    marker_locked: bool,
    walls: frozenset[tuple[int, int]],
) -> tuple[int, int]:
    if helper_pos == marker:
        return helper_pos

    if helper_pos[0] != marker[0]:
        step_x = 1 if marker[0] > helper_pos[0] else -1
        horizontal = (helper_pos[0] + step_x, helper_pos[1])
        if not _blocked(
            horizontal,
            mover="helper",
            player_pos=player_pos,
            helper_pos=helper_pos,
            marker=marker,
            marker_locked=marker_locked,
            walls=walls,
        ):
            return horizontal
        if helper_pos[1] != marker[1]:
            step_y = 1 if marker[1] > helper_pos[1] else -1
            vertical = (helper_pos[0], helper_pos[1] + step_y)
            if not _blocked(
                vertical,
                mover="helper",
                player_pos=player_pos,
                helper_pos=helper_pos,
                marker=marker,
                marker_locked=marker_locked,
                walls=walls,
            ):
                return vertical
        return helper_pos

    if helper_pos[1] != marker[1]:
        step_y = 1 if marker[1] > helper_pos[1] else -1
        vertical = (helper_pos[0], helper_pos[1] + step_y)
        if not _blocked(
            vertical,
            mover="helper",
            player_pos=player_pos,
            helper_pos=helper_pos,
            marker=marker,
            marker_locked=marker_locked,
            walls=walls,
        ):
            return vertical

    return helper_pos


def _advance(
    state: tuple[tuple[int, int], tuple[int, int], int, bool], action_id: int, spec: LevelSpec
) -> tuple[tuple[int, int], tuple[int, int], int, bool] | None:
    player_pos, helper_pos, remaining, marker_locked = state
    if remaining <= 0:
        return None

    next_remaining = remaining - 1
    delta = MOVE_DELTAS[action_id]
    next_player = player_pos
    if not (marker_locked and player_pos == spec.marker):
        candidate = (player_pos[0] + delta[0], player_pos[1] + delta[1])
        if not _blocked(
            candidate,
            mover="player",
            player_pos=player_pos,
            helper_pos=helper_pos,
            marker=spec.marker,
            marker_locked=marker_locked,
            walls=spec.walls,
        ):
            next_player = candidate

    next_helper = _helper_step(
        helper_pos, player_pos=next_player, marker=spec.marker, marker_locked=marker_locked, walls=spec.walls
    )

    player_on_marker = next_player == spec.marker
    helper_on_marker = next_helper == spec.marker
    next_locked = marker_locked or (player_on_marker ^ helper_on_marker)
    return (next_player, next_helper, next_remaining, next_locked)


def _is_goal(state: tuple[tuple[int, int], tuple[int, int], int, bool], marker: tuple[int, int]) -> bool:
    return state[0] == marker and state[1] == marker


def _solve_level(level_idx: int) -> list[int]:
    spec = LEVEL_SPECS[level_idx]
    budgets = (10, 8, 11)
    start = (spec.player_start, spec.helper_start, budgets[level_idx], False)
    queue = deque([(start, [])])
    seen = {start}

    while queue:
        state, path = queue.popleft()
        if _is_goal(state, spec.marker):
            return [*path, 1]
        for action_id in (1, 2, 3, 4, 5):
            nxt = _advance(state, action_id, spec)
            if nxt is None or nxt in seen:
                continue
            if nxt[2] == 0 and not _is_goal(nxt, spec.marker):
                continue
            seen.add(nxt)
            queue.append((nxt, [*path, action_id]))

    raise RuntimeError(f"No solution found for level {level_idx}.")


class GreedyFriendMeetingDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)
        self._programs = {idx: _solve_level(idx) for idx in range(3)}
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def next_action(self, _env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("Missing `levels_completed` in greedy_friend_meeting observation.")

        self.mark_levels_solved(level_idx)
        if self._current_level_idx is None or self._current_level_idx != level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
        elif bool(getattr(observation, "full_reset", False)) and self._action_idx > 0:
            self._action_idx = 0

        program = self._programs[level_idx]
        if self._action_idx >= len(program):
            return 1, {}

        action_id = program[self._action_idx]
        self._action_idx += 1
        return action_id, {}


AGENT_CLASS = GreedyFriendMeetingDslAgent
