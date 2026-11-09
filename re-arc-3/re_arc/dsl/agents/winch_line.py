from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..core import CachedProgramDslAgent


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    def contains(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class BarSpec:
    key: str
    state0: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    state1: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    initial_state: int

    def cells_for_state(self, state: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        return self.state1 if state else self.state0


@dataclass(frozen=True)
class WinchSpec:
    footprint: Rect
    bar_key: str

    def click_payload(self) -> dict[str, int]:
        return {"x": self.footprint.x0 * 4 + 4, "y": 8 + self.footprint.y0 * 4 + 4}


@dataclass(frozen=True)
class LevelSpec:
    budget: int
    floor_cells: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: Rect
    winches: tuple[WinchSpec, ...]
    bars: tuple[BarSpec, ...]


def _rect_cells(x0: int, y0: int, x1: int, y1: int) -> frozenset[tuple[int, int]]:
    return frozenset((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        budget=16,
        floor_cells=_rect_cells(6, 8, 8, 10)
        | _rect_cells(8, 6, 8, 7)
        | _rect_cells(8, 5, 11, 5)
        | _rect_cells(12, 8, 14, 10),
        start=(8, 9),
        goal=Rect(12, 8, 13, 9),
        winches=(WinchSpec(Rect(1, 2, 2, 3), "B1"),),
        bars=(BarSpec("B1", ((9, 4), (10, 4), (11, 4)), ((9, 9), (10, 9), (11, 9)), 0),),
    ),
    LevelSpec(
        budget=18,
        floor_cells=_rect_cells(6, 8, 8, 10) | _rect_cells(9, 5, 11, 6) | _rect_cells(12, 8, 14, 10),
        start=(8, 9),
        goal=Rect(12, 8, 13, 9),
        winches=(WinchSpec(Rect(1, 2, 2, 3), "B1"), WinchSpec(Rect(1, 4, 2, 5), "B2")),
        bars=(
            BarSpec("B1", ((9, 5), (10, 5), (11, 5)), ((9, 9), (10, 9), (11, 9)), 0),
            BarSpec("B2", ((11, 9), (12, 9), (13, 9)), ((13, 9), (14, 9), (15, 9)), 0),
        ),
    ),
    LevelSpec(
        budget=30,
        floor_cells=_rect_cells(1, 8, 4, 12)
        | _rect_cells(4, 6, 5, 7)
        | _rect_cells(5, 5, 7, 5)
        | _rect_cells(8, 8, 8, 12)
        | _rect_cells(14, 8, 15, 11),
        start=(4, 10),
        goal=Rect(14, 9, 15, 10),
        winches=(WinchSpec(Rect(1, 2, 2, 3), "B1"), WinchSpec(Rect(6, 11, 7, 12), "B2")),
        bars=(
            BarSpec("B1", ((5, 5), (6, 5), (7, 5)), ((5, 10), (6, 10), (7, 10)), 0),
            BarSpec("B2", ((9, 10), (10, 10), (11, 10)), ((11, 10), (12, 10), (13, 10)), 0),
        ),
    ),
)

MOVE_ACTIONS = ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0)))


class WinchLineDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "winch_line-0001"):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, _env):
        if self._current_level_idx is None:
            raise RuntimeError("winch_line DSL missing current level index")
        level_index = int(self._current_level_idx)
        spec = LEVEL_SPECS[level_index]
        actions = self._solve_level(spec)
        program = [self._encode_action(step) for step in actions]
        if level_index < len(LEVEL_SPECS) - 1:
            program.append((1, {}))
        return program

    def _encode_action(self, action):
        kind = action[0]
        if kind == "move":
            return action[1], {}
        return 6, action[1].click_payload()

    def _solve_level(self, spec: LevelSpec):
        start_bars = tuple(bar.initial_state for bar in spec.bars)
        start_state = (spec.start[0], spec.start[1], start_bars, 0)
        queue = deque([start_state])
        previous: dict[tuple[int, int, tuple[int, ...], int], tuple[int, int, tuple[int, ...], int] | None] = {
            start_state: None
        }
        previous_action: dict[tuple[int, int, tuple[int, ...], int], tuple[str, object]] = {}

        while queue:
            state = queue.popleft()
            x, y, _bar_states, used = state
            if spec.goal.contains((x, y)):
                return self._reconstruct(state, previous, previous_action)
            if used >= spec.budget:
                continue

            for next_action, next_state in self._expand(spec, state):
                if next_state in previous:
                    continue
                previous[next_state] = state
                previous_action[next_state] = next_action
                queue.append(next_state)

        raise RuntimeError("winch_line DSL failed to find a solution")

    def _reconstruct(self, goal_state, previous, previous_action):
        out = []
        cursor = goal_state
        while previous[cursor] is not None:
            out.append(previous_action[cursor])
            cursor = previous[cursor]
        out.reverse()
        return out

    def _expand(self, spec: LevelSpec, state):
        x, y, bar_states, used = state
        active = self._active_cells(spec, bar_states)
        winch_cells = self._winch_cells(spec)

        for action_id, delta in MOVE_ACTIONS:
            target = (x + delta[0], y + delta[1])
            if self._walkable(spec, target, active, winch_cells):
                yield (("move", action_id), (target[0], target[1], bar_states, used + 1))
            else:
                yield (("move", action_id), (x, y, bar_states, used + 1))

        for winch in spec.winches:
            next_x, next_y, next_bars = self._toggle(spec, (x, y), bar_states, winch, winch_cells)
            yield (("click", winch), (next_x, next_y, next_bars, used + 1))

    def _active_cells(self, spec: LevelSpec, bar_states: tuple[int, ...]) -> set[tuple[int, int]]:
        occupied: set[tuple[int, int]] = set()
        for index, bar in enumerate(spec.bars):
            occupied.update(bar.cells_for_state(bar_states[index]))
        return occupied

    def _winch_cells(self, spec: LevelSpec) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for winch in spec.winches:
            for gx in range(winch.footprint.x0, winch.footprint.x1 + 1):
                for gy in range(winch.footprint.y0, winch.footprint.y1 + 1):
                    cells.add((gx, gy))
        return cells

    def _walkable(self, spec: LevelSpec, cell, active, winch_cells) -> bool:
        x, y = cell
        if not (0 <= x < 16 and 0 <= y < 14):
            return False
        if cell in spec.floor_cells:
            return cell not in active and cell not in winch_cells
        return cell in active

    def _toggle(self, spec: LevelSpec, avatar, bar_states, winch, winch_cells):
        bar_index = next(index for index, bar in enumerate(spec.bars) if bar.key == winch.bar_key)
        bar = spec.bars[bar_index]
        current_state = bar_states[bar_index]
        current_cells = bar.cells_for_state(current_state)
        target_cells = bar.cells_for_state(1 - current_state)
        dx = target_cells[0][0] - current_cells[0][0]
        dy = target_cells[0][1] - current_cells[0][1]

        other_active: set[tuple[int, int]] = set()
        for index, other_bar in enumerate(spec.bars):
            if index == bar_index:
                continue
            other_active.update(other_bar.cells_for_state(bar_states[index]))

        for cell in target_cells:
            tx, ty = cell
            if not (0 <= tx < 16 and 0 <= ty < 14):
                return avatar[0], avatar[1], bar_states
            if cell in winch_cells or cell in other_active:
                return avatar[0], avatar[1], bar_states

        avatar_on_bar = avatar in set(current_cells)
        if not avatar_on_bar and avatar in target_cells:
            return avatar[0], avatar[1], bar_states

        next_avatar = avatar
        if avatar_on_bar:
            carried = (avatar[0] + dx, avatar[1] + dy)
            if not (0 <= carried[0] < 16 and 0 <= carried[1] < 14):
                return avatar[0], avatar[1], bar_states
            if carried not in target_cells or carried in spec.floor_cells:
                return avatar[0], avatar[1], bar_states
            next_avatar = carried

        next_states = list(bar_states)
        next_states[bar_index] = 1 - current_state
        return next_avatar[0], next_avatar[1], tuple(next_states)


AGENT_CLASS = WinchLineDslAgent
