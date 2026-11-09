from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..core import DslAgent, observation_level_index

BOARD_SIZE = 7


@dataclass(frozen=True)
class EmitterSpec:
    name: str
    kind: str
    index: int
    direction: str
    click: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    discs: tuple[tuple[int, int], ...]
    pads: frozenset[tuple[int, int]]
    walls: frozenset[tuple[int, int]]
    emitters: tuple[EmitterSpec, ...]
    budget: int


def _level_specs() -> tuple[LevelSpec, ...]:
    level3_open = {(2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1), (3, 2), (4, 2), (3, 3), (3, 4), (3, 5)}
    all_cells = {(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)}
    return (
        LevelSpec(
            discs=((1, 5),),
            pads=frozenset({(4, 2)}),
            walls=frozenset(),
            emitters=(EmitterSpec("R5", "row", 5, "RIGHT", (7, 44)), EmitterSpec("U4", "col", 4, "UP", (38, 57))),
            budget=18,
        ),
        LevelSpec(
            discs=((1, 5), (3, 5)),
            pads=frozenset({(4, 1), (5, 1)}),
            walls=frozenset(),
            emitters=(
                EmitterSpec("R5", "row", 5, "RIGHT", (7, 44)),
                EmitterSpec("U4", "col", 4, "UP", (38, 57)),
                EmitterSpec("U5", "col", 5, "UP", (44, 57)),
            ),
            budget=33,
        ),
        LevelSpec(
            discs=((3, 3), (3, 4), (3, 5)),
            pads=frozenset({(2, 0), (3, 0), (4, 0)}),
            walls=frozenset(all_cells - level3_open),
            emitters=(
                EmitterSpec("U2", "col", 2, "UP", (26, 57)),
                EmitterSpec("U3", "col", 3, "UP", (32, 57)),
                EmitterSpec("U4", "col", 4, "UP", (38, 57)),
                EmitterSpec("L1", "row", 1, "LEFT", (57, 20)),
                EmitterSpec("R2", "row", 2, "RIGHT", (7, 26)),
            ),
            budget=27,
        ),
    )


def _scan_positions(emitter: EmitterSpec) -> list[tuple[int, int]]:
    if emitter.kind == "row":
        xs = range(5, -1, -1) if emitter.direction == "RIGHT" else range(1, 7)
        return [(x, emitter.index) for x in xs]
    ys = range(5, -1, -1) if emitter.direction == "DOWN" else range(1, 7)
    return [(emitter.index, y) for y in ys]


def _next_position(position: tuple[int, int], direction: str) -> tuple[int, int]:
    x, y = position
    if direction == "RIGHT":
        return x + 1, y
    if direction == "LEFT":
        return x - 1, y
    if direction == "DOWN":
        return x, y + 1
    return x, y - 1


def _apply_pulse(
    discs: tuple[tuple[int, int], ...], emitter: EmitterSpec, walls: frozenset[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    occupied = set(discs)
    for position in _scan_positions(emitter):
        if position not in occupied:
            continue
        nx, ny = _next_position(position, emitter.direction)
        next_position = (nx, ny)
        if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
            continue
        if next_position in walls or next_position in occupied:
            continue
        occupied.remove(position)
        occupied.add(next_position)
    return tuple(sorted(occupied))


def _is_solved(discs: tuple[tuple[int, int], ...], pads: frozenset[tuple[int, int]]) -> bool:
    return all(pad in discs for pad in pads)


def _solve_level(spec: LevelSpec) -> list[EmitterSpec]:
    start_discs = tuple(sorted(spec.discs))
    queue: deque[tuple[tuple[tuple[int, int], ...], int, list[EmitterSpec]]] = deque([(start_discs, spec.budget, [])])
    seen = {(start_discs, spec.budget)}
    while queue:
        discs, budget, path = queue.popleft()
        if _is_solved(discs, spec.pads):
            return path
        if budget <= 0:
            continue
        for emitter in spec.emitters:
            next_discs = _apply_pulse(discs, emitter, spec.walls)
            next_state = (next_discs, budget - 1)
            if next_state in seen:
                continue
            seen.add(next_state)
            queue.append((next_discs, budget - 1, [*path, emitter]))
    raise RuntimeError("Row Pulse Packing DSL could not find a solution.")


class RowPulsePackingDslAgent(DslAgent):
    def __init__(self, game_id: str = "row_pulse_packing-0001"):
        super().__init__(game_id=game_id, total_levels=3)
        self._specs = _level_specs()
        self._programs = {idx: _solve_level(spec) for idx, spec in enumerate(self._specs)}
        self._current_level_idx: int | None = None
        self._action_idx = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def _sync_level(self, observation: Any) -> int:
        level_idx = observation_level_index(observation, self.total_levels) or 0
        self.mark_levels_solved(level_idx)
        if self._current_level_idx is None or self._current_level_idx != level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
        elif bool(getattr(observation, "full_reset", False)):
            self._action_idx = 0
        return level_idx

    def next_action(self, env: Any, observation: Any) -> tuple[int, dict[str, int] | None]:
        del env
        level_idx = self._sync_level(observation)
        program = self._programs[level_idx]
        if self._action_idx >= len(program):
            return 1, None
        emitter = program[self._action_idx]
        self._action_idx += 1
        return 6, {"x": emitter.click[0], "y": emitter.click[1]}
