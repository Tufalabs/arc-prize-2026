from __future__ import annotations

from collections import deque
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from ..core import CachedProgramDslAgent

_ENV_PATH = Path(__file__).resolve().parents[2] / "environment_files" / "rotor_junction" / "0001" / "rotorjunction.py"
_ENV_SPEC = spec_from_file_location("re_arc_rotor_junction_env", _ENV_PATH)
if _ENV_SPEC is None or _ENV_SPEC.loader is None:
    raise RuntimeError(f"Unable to load Rotor Junction environment module from {_ENV_PATH}.")
_ENV_MODULE = module_from_spec(_ENV_SPEC)
_ENV_SPEC.loader.exec_module(_ENV_MODULE)

BOARD_ORIGIN_X = _ENV_MODULE.BOARD_ORIGIN_X
BOARD_ORIGIN_Y = _ENV_MODULE.BOARD_ORIGIN_Y
CELL_SIZE = _ENV_MODULE.CELL_SIZE
LEVEL_SPECS = _ENV_MODULE.LEVEL_SPECS
PuckState = _ENV_MODULE.PuckState
ROTATE_CW = _ENV_MODULE.ROTATE_CW
level_state_from_spec = _ENV_MODULE.level_state_from_spec
simulate_step = _ENV_MODULE.simulate_step

WAIT_ACTION_ID = 5


def _state_key(
    *, remaining_budget: int, ack_state: str, pucks: list[PuckState], rotors: dict[tuple[int, int], str]
) -> tuple[object, ...]:
    return (
        remaining_budget,
        ack_state,
        tuple((pos[0], pos[1], heading) for pos, heading in sorted(rotors.items())),
        tuple(puck.as_tuple() for puck in pucks),
    )


def _click_data(pos: tuple[int, int]) -> dict[str, int]:
    return {"x": BOARD_ORIGIN_X + CELL_SIZE * pos[0] + 2, "y": BOARD_ORIGIN_Y + CELL_SIZE * pos[1] + 2}


def _build_level_program(level_index: int) -> list[tuple[int, dict[str, int]]]:
    start_pucks, start_rotors, remaining_budget = level_state_from_spec(level_index)
    start_state = (remaining_budget, "none", start_pucks, start_rotors)
    queue = deque([(start_state, [])])
    seen = {_state_key(remaining_budget=remaining_budget, ack_state="none", pucks=start_pucks, rotors=start_rotors)}

    while queue:
        (budget, ack_state, pucks, rotors), actions = queue.popleft()
        if level_index == len(LEVEL_SPECS) - 1:
            if all(puck.docked for puck in pucks):
                return actions
        elif ack_state == "win":
            return [*actions, (WAIT_ACTION_ID, {})]

        if ack_state != "none" or budget <= 0:
            continue

        candidates: list[tuple[tuple[int, int] | None, tuple[int, dict[str, int]]]] = [(None, (WAIT_ACTION_ID, {}))]
        for rotor_pos in rotors:
            candidates.append((rotor_pos, (6, _click_data(rotor_pos))))

        for clicked_rotor, action in candidates:
            next_pucks, flashes, invalid_route = simulate_step(
                level_index=level_index,
                pucks=[
                    PuckState(puck_id=puck.puck_id, x=puck.x, y=puck.y, heading=puck.heading, docked=puck.docked)
                    for puck in pucks
                ],
                rotors=rotors,
                clicked_rotor=clicked_rotor,
            )
            next_budget = budget - 1
            next_rotors = dict(rotors)
            if clicked_rotor in next_rotors:
                next_rotors[clicked_rotor] = ROTATE_CW[next_rotors[clicked_rotor]]

            next_ack_state = "none"
            if all(puck.docked for puck in next_pucks):
                if level_index == len(LEVEL_SPECS) - 1:
                    key = _state_key(
                        remaining_budget=next_budget, ack_state="none", pucks=next_pucks, rotors=next_rotors
                    )
                    if key not in seen:
                        seen.add(key)
                        queue.append(((next_budget, "none", next_pucks, next_rotors), [*actions, action]))
                    continue
                next_ack_state = "win"
            elif invalid_route or next_budget == 0:
                next_ack_state = "fail"

            _ = flashes
            key = _state_key(
                remaining_budget=next_budget, ack_state=next_ack_state, pucks=next_pucks, rotors=next_rotors
            )
            if key in seen:
                continue
            seen.add(key)
            queue.append(((next_budget, next_ack_state, next_pucks, next_rotors), [*actions, action]))

    raise RuntimeError(f"No DSL solution found for rotor_junction level {level_index}.")


class RotorJunctionDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "rotor_junction-0001") -> None:
        super().__init__(game_id=game_id, total_levels=len(LEVEL_SPECS))

    def _build_level_program(self, _env) -> list[tuple[int, dict[str, int]]]:
        if self._current_level_idx is None:
            raise RuntimeError("Rotor Junction level index is unavailable.")
        return _build_level_program(self._current_level_idx)


AGENT_CLASS = RotorJunctionDslAgent
