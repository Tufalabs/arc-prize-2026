from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..core import CachedProgramDslAgent

RED = 8
BLUE = 9
GREEN = 14

PENDING_NONE = "none"
PENDING_WIN = "win"
PENDING_FINAL_WIN = "final_win"


@dataclass(frozen=True)
class BandSpec:
    lane: int
    x: int
    width: int
    color: int


@dataclass(frozen=True)
class LevelSpec:
    budget: int
    active_lanes: tuple[int, ...]
    selected_lane: int
    target_queue: tuple[int, ...]
    bands: tuple[BandSpec, ...]


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        budget=24,
        active_lanes=(1,),
        selected_lane=1,
        target_queue=(BLUE,),
        bands=(BandSpec(lane=1, x=22, width=10, color=BLUE),),
    ),
    LevelSpec(
        budget=46,
        active_lanes=(1,),
        selected_lane=1,
        target_queue=(RED, BLUE, RED),
        bands=(
            BandSpec(lane=1, x=17, width=4, color=RED),
            BandSpec(lane=1, x=23, width=4, color=BLUE),
            BandSpec(lane=1, x=29, width=4, color=RED),
            BandSpec(lane=1, x=40, width=4, color=GREEN),
        ),
    ),
    LevelSpec(
        budget=50,
        active_lanes=(0, 1, 2),
        selected_lane=0,
        target_queue=(BLUE, RED, GREEN, BLUE),
        bands=(
            BandSpec(lane=0, x=17, width=4, color=BLUE),
            BandSpec(lane=0, x=25, width=4, color=GREEN),
            BandSpec(lane=1, x=19, width=4, color=RED),
            BandSpec(lane=1, x=25, width=4, color=BLUE),
            BandSpec(lane=2, x=20, width=4, color=GREEN),
            BandSpec(lane=2, x=26, width=4, color=RED),
        ),
    ),
)


def _overlaps_strike(x: int, width: int) -> bool:
    return x <= 17 and (x + width - 1) >= 16


def _has_future_band(spec: LevelSpec, target_queue: tuple[int, ...], bands: tuple[tuple[int, bool], ...]) -> bool:
    if not target_queue:
        return False
    wanted = target_queue[0]
    for (x, claimed), band_spec in zip(bands, spec.bands, strict=True):
        if claimed or band_spec.color != wanted:
            continue
        if x + band_spec.width - 1 >= 16:
            return True
    return False


def _apply_action(
    spec: LevelSpec, state: tuple[int, int, tuple[int, ...], tuple[tuple[int, bool], ...]], action_id: int
) -> tuple[int, int, tuple[int, ...], tuple[tuple[int, bool], ...], str] | None:
    budget, selected_lane, target_queue, bands = state
    if budget <= 0:
        return None

    next_budget = budget
    next_lane = selected_lane
    next_queue = list(target_queue)
    next_bands = [list(pair) for pair in bands]
    budgeted = False

    if action_id in {3, 4}:
        budgeted = True
        next_budget -= 1
        for idx, _band_spec in enumerate(spec.bands):
            if not next_bands[idx][1]:
                next_bands[idx][0] -= 1
    elif action_id == 5:
        budgeted = True
        next_budget -= 1
        wanted = next_queue[0]
        lane = spec.active_lanes[0] if len(spec.active_lanes) == 1 else next_lane
        candidates = []
        for idx, ((x, claimed), band_spec) in enumerate(zip(next_bands, spec.bands, strict=True)):
            if (
                claimed
                or band_spec.lane != lane
                or band_spec.color != wanted
                or not _overlaps_strike(x, band_spec.width)
            ):
                continue
            candidates.append((x, idx))
        if candidates:
            candidates.sort()
            _, claim_idx = candidates[0]
            next_bands[claim_idx][1] = True
            next_queue.pop(0)
    elif action_id == 1 and len(spec.active_lanes) > 1:
        new_lane = max(spec.active_lanes[0], next_lane - 1)
        if new_lane != next_lane:
            budgeted = True
            next_budget -= 1
            next_lane = new_lane
    elif action_id == 2 and len(spec.active_lanes) > 1:
        new_lane = min(spec.active_lanes[-1], next_lane + 1)
        if new_lane != next_lane:
            budgeted = True
            next_budget -= 1
            next_lane = new_lane
    else:
        return None

    if not next_queue:
        pending = PENDING_FINAL_WIN if spec is LEVEL_SPECS[-1] else PENDING_WIN
        return next_budget, next_lane, tuple(next_queue), tuple((int(x), bool(c)) for x, c in next_bands), pending

    if budgeted and next_budget <= 0:
        return None
    if budgeted and not _has_future_band(spec, tuple(next_queue), tuple((int(x), bool(c)) for x, c in next_bands)):
        return None

    return next_budget, next_lane, tuple(next_queue), tuple((int(x), bool(c)) for x, c in next_bands), PENDING_NONE


class FrequencyLockDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_SPECS))

    def _plan_level(self, level_idx: int) -> list[tuple[int, dict[str, int]]]:
        spec = LEVEL_SPECS[level_idx]
        start_state = (
            spec.budget,
            spec.selected_lane,
            tuple(spec.target_queue),
            tuple((band.x, False) for band in spec.bands),
        )

        queue = deque([start_state])
        previous: dict[tuple[int, int, tuple[int, ...], tuple[tuple[int, bool], ...]], tuple | None] = {
            start_state: None
        }
        previous_action: dict[tuple[int, int, tuple[int, ...], tuple[tuple[int, bool], ...]], int] = {}
        goal_state = None
        pending_by_state: dict[tuple[int, int, tuple[int, ...], tuple[tuple[int, bool], ...]], str] = {
            start_state: PENDING_NONE
        }

        while queue:
            state = queue.popleft()
            for action_id in (1, 2, 3, 4, 5):
                next_state = _apply_action(spec, state, action_id)
                if next_state is None:
                    continue
                budget, selected_lane, target_queue, bands, pending = next_state
                packed = (budget, selected_lane, target_queue, bands)
                if packed in previous:
                    continue
                previous[packed] = state
                previous_action[packed] = action_id
                pending_by_state[packed] = pending
                if pending in {PENDING_WIN, PENDING_FINAL_WIN}:
                    goal_state = packed
                    queue.clear()
                    break
                queue.append(packed)
            if goal_state is not None:
                break

        if goal_state is None:
            raise RuntimeError(f"frequency_lock DSL could not solve level {level_idx}.")

        actions: list[int] = []
        cursor = goal_state
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]
        actions.reverse()
        actions.append(5)
        return [(action_id, {}) for action_id in actions]

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level_idx = int(getattr(env._game, "level_index", 0))
        return self._plan_level(level_idx)


AGENT_CLASS = FrequencyLockDslAgent
