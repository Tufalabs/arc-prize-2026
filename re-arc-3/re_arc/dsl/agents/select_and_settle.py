from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..core import CachedProgramDslAgent

LEVEL_SPECS = [
    {"blue_pos": (5, 6), "blue_mark": (1, 6), "magenta_pos": (9, 6), "magenta_mark": (9, 6), "active": "blue"},
    {"blue_pos": (0, 9), "blue_mark": (0, 9), "magenta_pos": (12, 7), "magenta_mark": (12, 7), "active": "blue"},
    {"blue_pos": (1, 10), "blue_mark": (1, 10), "magenta_pos": (11, 5), "magenta_mark": (11, 5), "active": "blue"},
]

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


@dataclass(frozen=True)
class State:
    active: str
    blue_pos: tuple[int, int]
    blue_mark: tuple[int, int]
    magenta_pos: tuple[int, int]
    magenta_mark: tuple[int, int]


def _legal(candidate: tuple[int, int], mark: tuple[int, int]) -> bool:
    if not (0 <= candidate[0] <= 12 and 0 <= candidate[1] <= 12):
        return False
    return abs(candidate[0] - mark[0]) + abs(candidate[1] - mark[1]) <= 4


def _is_goal(state: State) -> bool:
    return state.blue_pos == state.magenta_mark or state.magenta_pos == state.blue_mark


def _click_payload(position: tuple[int, int]) -> dict[str, int]:
    x = 4 + position[0] * 4 + 3
    y = 8 + position[1] * 4 + 3
    return {"x": int(x), "y": int(y)}


def _expand(state: State) -> list[tuple[int, dict[str, int], State]]:
    out: list[tuple[int, dict[str, int], State]] = []
    for action_id, (dx, dy) in MOVE_DELTAS.items():
        if state.active == "blue":
            candidate = (state.blue_pos[0] + dx, state.blue_pos[1] + dy)
            if not _legal(candidate, state.blue_mark):
                continue
            out.append(
                (
                    action_id,
                    {},
                    State(
                        active=state.active,
                        blue_pos=candidate,
                        blue_mark=state.blue_mark,
                        magenta_pos=state.magenta_pos,
                        magenta_mark=state.magenta_mark,
                    ),
                )
            )
        else:
            candidate = (state.magenta_pos[0] + dx, state.magenta_pos[1] + dy)
            if not _legal(candidate, state.magenta_mark):
                continue
            out.append(
                (
                    action_id,
                    {},
                    State(
                        active=state.active,
                        blue_pos=state.blue_pos,
                        blue_mark=state.blue_mark,
                        magenta_pos=candidate,
                        magenta_mark=state.magenta_mark,
                    ),
                )
            )

    if state.active == "blue":
        out.append(
            (
                6,
                _click_payload(state.magenta_pos),
                State(
                    active="magenta",
                    blue_pos=state.blue_pos,
                    blue_mark=state.blue_pos,
                    magenta_pos=state.magenta_pos,
                    magenta_mark=state.magenta_mark,
                ),
            )
        )
    else:
        out.append(
            (
                6,
                _click_payload(state.blue_pos),
                State(
                    active="blue",
                    blue_pos=state.blue_pos,
                    blue_mark=state.blue_mark,
                    magenta_pos=state.magenta_pos,
                    magenta_mark=state.magenta_pos,
                ),
            )
        )
    return out


def _solve_level(level_idx: int) -> list[tuple[int, dict[str, int]]]:
    raw = LEVEL_SPECS[level_idx]
    start = State(
        active=raw["active"],
        blue_pos=raw["blue_pos"],
        blue_mark=raw["blue_mark"],
        magenta_pos=raw["magenta_pos"],
        magenta_mark=raw["magenta_mark"],
    )
    queue = deque([start])
    previous: dict[State, State | None] = {start: None}
    previous_action: dict[State, tuple[int, dict[str, int]]] = {}

    while queue:
        state = queue.popleft()
        if _is_goal(state):
            actions: list[tuple[int, dict[str, int]]] = []
            cursor = state
            while previous[cursor] is not None:
                actions.append(previous_action[cursor])
                cursor = previous[cursor]  # type: ignore[assignment]
            actions.reverse()
            return actions

        for action_id, payload, next_state in _expand(state):
            if next_state in previous:
                continue
            previous[next_state] = state
            previous_action[next_state] = (action_id, payload)
            queue.append(next_state)

    raise RuntimeError(f"No DSL solution found for level {level_idx}.")


class SelectAndSettleDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _on_new_level(self, _env, level_idx: int):
        if level_idx not in self._programs:
            self._programs[level_idx] = _solve_level(level_idx)


AGENT_CLASS = SelectAndSettleDslAgent
