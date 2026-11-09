from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from arcengine import GameAction

from ..core import CachedProgramDslAgent

MOVE_AND_WAIT = (
    int(GameAction.ACTION1.value),
    int(GameAction.ACTION2.value),
    int(GameAction.ACTION3.value),
    int(GameAction.ACTION4.value),
    int(GameAction.ACTION5.value),
)

DELTA_BY_ACTION = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}


@dataclass(frozen=True)
class SimGate:
    coord: tuple[int, int]
    mode: str
    cycle: object | None
    phase: int
    pulse_steps: int


@dataclass(frozen=True)
class SimSpike:
    down_steps: int
    up_steps: int
    phase: int


@dataclass(frozen=True)
class SimLevel:
    width: int
    height: int
    spawn: tuple[int, int]
    exits: frozenset[tuple[int, int]]
    walls: frozenset[tuple[int, int]]
    posts: frozenset[tuple[int, int]]
    spikes: frozenset[tuple[int, int]]
    spike_spec: SimSpike | None
    gates: tuple[SimGate, ...]
    plate: tuple[int, int] | None
    plate_target_gate: int | None
    max_steps: int


@dataclass(frozen=True)
class SimState:
    x: int
    y: int
    time_left: int
    tick: int
    pulse_remaining: tuple[int, ...]


class TimingGateDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        model = self._extract_level_model(env)
        actions = self._bfs_plan(model)
        if actions is None:
            raise RuntimeError("timing_gate DSL could not find a valid plan for the current level")
        program = [(action_id, {}) for action_id in actions]
        program.extend((int(GameAction.ACTION5.value), {}) for _ in range(8))
        return program

    @staticmethod
    def _extract_level_model(env) -> SimLevel:
        game = env._game
        raw = game.export_level_model()

        gates = tuple(
            SimGate(
                coord=tuple(gate["coord"]),
                mode=str(gate["mode"]),
                cycle=gate["cycle"],
                phase=int(gate["phase"]),
                pulse_steps=int(gate["pulse_steps"]),
            )
            for gate in raw["gates"]
        )

        spike_spec_raw = raw.get("spike_spec")
        spike_spec = None
        if spike_spec_raw is not None:
            spike_spec = SimSpike(
                down_steps=int(spike_spec_raw["down"]),
                up_steps=int(spike_spec_raw["up"]),
                phase=int(spike_spec_raw["phase"]),
            )

        plate_raw = raw.get("plate")
        plate = tuple(plate_raw) if plate_raw is not None else None

        return SimLevel(
            width=int(raw["width"]),
            height=int(raw["height"]),
            spawn=tuple(raw["spawn"]),
            exits=frozenset(tuple(cell) for cell in raw["exits"]),
            walls=frozenset(tuple(cell) for cell in raw["walls"]),
            posts=frozenset(tuple(cell) for cell in raw["posts"]),
            spikes=frozenset(tuple(cell) for cell in raw["spikes"]),
            spike_spec=spike_spec,
            gates=gates,
            plate=plate,
            plate_target_gate=raw.get("plate_target_gate"),
            max_steps=int(raw["max_steps"]),
        )

    def _bfs_plan(self, model: SimLevel) -> list[int] | None:
        start = SimState(
            x=int(model.spawn[0]),
            y=int(model.spawn[1]),
            time_left=int(model.max_steps),
            tick=0,
            pulse_remaining=tuple(0 for _ in model.gates),
        )

        queue = deque([start])
        previous: dict[SimState, SimState | None] = {start: None}
        previous_action: dict[SimState, int] = {}

        goal_state: SimState | None = None

        while queue:
            state = queue.popleft()
            for action_id in MOVE_AND_WAIT:
                nxt, won = self._simulate_step(model, state, action_id)
                if nxt is None:
                    continue
                if nxt not in previous:
                    previous[nxt] = state
                    previous_action[nxt] = action_id
                    if won:
                        goal_state = nxt
                        queue.clear()
                        break
                    queue.append(nxt)
            if goal_state is not None:
                break

        if goal_state is None:
            return None

        actions: list[int] = []
        cursor = goal_state
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]  # type: ignore[assignment]
        actions.reverse()
        return actions

    def _simulate_step(self, model: SimLevel, state: SimState, action_id: int) -> tuple[SimState | None, bool]:
        x, y = int(state.x), int(state.y)

        dx, dy = DELTA_BY_ACTION.get(int(action_id), (0, 0))
        tx, ty = x + dx, y + dy

        gate_passable_before = self._gate_passable_before_update(model, state)

        if self._can_enter(model, tx, ty, gate_passable_before):
            x, y = tx, ty

        time_left = int(state.time_left) - 1

        pulse = list(state.pulse_remaining)
        if model.plate is not None and (x, y) == model.plate:
            target = model.plate_target_gate
            if target is not None and 0 <= int(target) < len(model.gates):
                gate = model.gates[int(target)]
                if gate.mode == "plate":
                    pulse[int(target)] = int(gate.pulse_steps) + 1

        tick = int(state.tick) + 1

        gate_passable_after = self._gate_passable_after_update(model, tick, tuple(pulse))
        for idx, gate in enumerate(model.gates):
            if gate.mode == "plate" and pulse[idx] > 0:
                pulse[idx] -= 1

        if time_left <= 0:
            return None, False

        if model.spike_spec is not None and (x, y) in model.spikes and self._spike_is_up(model.spike_spec, tick):
            return None, False

        for idx, gate in enumerate(model.gates):
            if (x, y) == gate.coord and not gate_passable_after[idx]:
                return None, False

        won = (x, y) in model.exits

        return (SimState(x=x, y=y, time_left=time_left, tick=tick, pulse_remaining=tuple(pulse)), won)

    @staticmethod
    def _can_enter(model: SimLevel, x: int, y: int, gate_passable_before: tuple[bool, ...]) -> bool:
        if x < 0 or y < 0 or x >= model.width or y >= model.height:
            return False
        if (x, y) in model.walls or (x, y) in model.posts:
            return False
        for idx, gate in enumerate(model.gates):
            if (x, y) == gate.coord and not gate_passable_before[idx]:
                return False
        return True

    def _gate_passable_before_update(self, model: SimLevel, state: SimState) -> tuple[bool, ...]:
        out: list[bool] = []
        for idx, gate in enumerate(model.gates):
            if gate.mode == "timed":
                out.append(self._timed_gate_is_open(gate, int(state.tick)))
            else:
                out.append(int(state.pulse_remaining[idx]) > 0)
        return tuple(out)

    def _gate_passable_after_update(
        self, model: SimLevel, tick_after_increment: int, pulse_after_plate_trigger: tuple[int, ...]
    ) -> tuple[bool, ...]:
        out: list[bool] = []
        for idx, gate in enumerate(model.gates):
            if gate.mode == "timed":
                out.append(self._timed_gate_is_open(gate, tick_after_increment))
            else:
                out.append(int(pulse_after_plate_trigger[idx]) > 0)
        return tuple(out)

    @staticmethod
    def _timed_gate_is_open(gate: SimGate, tick: int) -> bool:
        cycle = gate.cycle
        if cycle is None:
            return False
        closed, warning_a, open_steps, warning_b = tuple(cycle)
        period = int(closed + warning_a + open_steps + warning_b)
        if period <= 0:
            return False
        slot = int((tick + gate.phase) % period)
        if slot < int(closed):
            return False
        slot -= int(closed)
        if slot < int(warning_a):
            return False
        slot -= int(warning_a)
        return slot < int(open_steps)

    @staticmethod
    def _spike_is_up(spike: SimSpike, tick: int) -> bool:
        period = int(spike.down_steps + spike.up_steps)
        if period <= 0:
            return False
        slot = int((tick + spike.phase) % period)
        return slot >= int(spike.down_steps)


AGENT_CLASS = TimingGateDslAgent
