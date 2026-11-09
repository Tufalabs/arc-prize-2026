from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.grid import cardinal_neighbors, grid_to_display_click
from ..solvers.search import dijkstra_plan

_ENV_MOD = import_module(
    "re_arc.environment_files.circuit_connect_path_drawing_by_clicks.0001.circuitconnectpathdrawingbyclicks"
)

WIDTH = int(_ENV_MOD.WIDTH)
HEIGHT = int(_ENV_MOD.HEIGHT)
DOOR_OPEN_FLASH_STEPS = int(_ENV_MOD.DOOR_OPEN_FLASH_STEPS)
DIODE_DIR = dict(_ENV_MOD.DIODE_DIR)

build_level_model = _ENV_MOD.build_level_model
LEVEL_LAYOUTS = list(_ENV_MOD.LEVEL_LAYOUTS)
LEVEL_DOOR_CONTROLS = list(_ENV_MOD.LEVEL_DOOR_CONTROLS)


def _no_op() -> tuple[int, dict[str, int]]:
    return 6, {"x": -1, "y": -1}


def _objective_chain(level_idx: int) -> list[str]:
    if level_idx <= 3:
        return ["E"]
    if level_idx == 4:
        return ["R", "E"]
    return ["R", "Q", "E"]


def _base_char(model, pos: tuple[int, int], opened_doors: set[int]) -> str:
    x, y = pos
    ch = model.rows[y][x]
    if ch == "|":
        group_idx = model.door_group_for_pos.get(pos)
        if group_idx is not None and group_idx in opened_doors:
            return "."
    return ch


def _is_traversable(model, pos: tuple[int, int], opened_doors: set[int]) -> bool:
    x, y = pos
    if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
        return False
    if y == 0:
        return False
    ch = _base_char(model, pos, opened_doors)
    return ch not in {"#", "|", "X", "~"}


def _adjacent_to_ground(model, pos: tuple[int, int]) -> bool:
    return any(nxt in model.ground_tiles for nxt in cardinal_neighbors(pos))


def _is_conductor(model, pos: tuple[int, int], opened_doors: set[int], wires: set[tuple[int, int]]) -> bool:
    if pos in wires:
        return True
    ch = _base_char(model, pos, opened_doors)
    return ch in {"S", "E", "R", "Q", "<", ">", "^", "v"}


def _diode_allows_exit(model, pos: tuple[int, int], dx: int, dy: int, opened_doors: set[int]) -> bool:
    ch = _base_char(model, pos, opened_doors)
    if ch not in DIODE_DIR:
        return True
    return DIODE_DIR[ch] == (dx, dy)


def _diode_allows_entry(model, pos: tuple[int, int], dx: int, dy: int, opened_doors: set[int]) -> bool:
    ch = _base_char(model, pos, opened_doors)
    if ch not in DIODE_DIR:
        return True
    ddx, ddy = DIODE_DIR[ch]
    return (ddx, ddy) == (dx, dy)


def _flow_allowed(model, src: tuple[int, int], dst: tuple[int, int], opened_doors: set[int]) -> bool:
    dx = int(dst[0] - src[0])
    dy = int(dst[1] - src[1])
    if not _diode_allows_exit(model, src, dx, dy, opened_doors):
        return False
    return _diode_allows_entry(model, dst, dx, dy, opened_doors)


def _find_click_path(
    model, *, opened_doors: set[int], wires: set[tuple[int, int]], goals: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    starts = [pos for pos in model.source_tiles if _is_traversable(model, pos, opened_doors)]
    if not starts:
        raise RuntimeError("circuit_connect DSL has no valid source tiles.")

    best_path: list[tuple[int, int]] | None = None
    best_clicks: int | None = None

    for start in starts:

        def is_goal(state: tuple[int, int]) -> bool:
            return state in goals

        def expand(state: tuple[int, int]):
            for nxt in cardinal_neighbors(state):
                if not _is_traversable(model, nxt, opened_doors):
                    continue
                if not _flow_allowed(model, state, nxt, opened_doors):
                    continue
                ch = _base_char(model, nxt, opened_doors)
                if ch == "." and _adjacent_to_ground(model, nxt):
                    continue
                step_cost = 1.0 if ch == "." and nxt not in wires else 0.0
                yield nxt, nxt, step_cost

        plan = dijkstra_plan(start, is_goal, expand)
        if plan is None:
            continue

        clicks = 0
        for pos in plan:
            ch = _base_char(model, pos, opened_doors)
            if ch == "." and pos not in wires:
                clicks += 1

        if best_clicks is None or clicks < best_clicks:
            best_clicks = clicks
            best_path = list(plan)

    if best_path is None:
        raise RuntimeError("circuit_connect DSL could not find a valid path to objective.")
    return best_path


def _pulse_distance(model, *, opened_doors: set[int], wires: set[tuple[int, int]], goals: set[tuple[int, int]]) -> int:
    from collections import deque

    queue = deque((pos, 0) for pos in model.source_tiles)
    seen = set(model.source_tiles)

    while queue:
        pos, dist = queue.popleft()
        if pos in goals:
            return int(dist)
        for nxt in cardinal_neighbors(pos):
            if nxt in seen:
                continue
            if not _is_conductor(model, nxt, opened_doors, wires):
                continue
            if not _flow_allowed(model, pos, nxt, opened_doors):
                continue
            seen.add(nxt)
            queue.append((nxt, dist + 1))

    raise RuntimeError("circuit_connect DSL could not measure pulse distance to objective.")


class CircuitConnectPathDrawingByClicksDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level = game.current_level
        level_idx = int(level.get_data("level_index") or 0)

        model = build_level_model(LEVEL_LAYOUTS[level_idx], LEVEL_DOOR_CONTROLS[level_idx])

        opened_doors: set[int] = set()
        wires: set[tuple[int, int]] = set()
        chain = _objective_chain(level_idx)

        program: list[tuple[int, dict[str, int]]] = [_no_op()]

        for stage in chain:
            if stage == "E":
                goals = set(model.target_tiles)
            else:
                goals = set(model.relay_tiles_by_id.get(stage, set()))
            if not goals:
                raise RuntimeError(f"circuit_connect DSL missing objective tiles for stage {stage!r}.")

            path = _find_click_path(model, opened_doors=opened_doors, wires=wires, goals=goals)

            for pos in path:
                ch = _base_char(model, pos, opened_doors)
                if ch != "." or pos in wires:
                    continue
                wires.add(pos)
                program.append((6, grid_to_display_click(game.camera, pos)))

            pulse_steps = _pulse_distance(model, opened_doors=opened_doors, wires=wires, goals=goals)
            wait_steps = max(2, pulse_steps + 2)
            for _ in range(wait_steps):
                program.append(_no_op())

            if stage != "E":
                for door_idx, controller in enumerate(model.door_control_by_group):
                    if controller == stage:
                        opened_doors.add(door_idx)
                for _ in range(DOOR_OPEN_FLASH_STEPS + 2):
                    program.append(_no_op())

        # In this environment win freezes until clicked.
        program.append(_no_op())
        program.append((6, {"x": 0, "y": 0}))
        return program


AGENT_CLASS = CircuitConnectPathDrawingByClicksDslAgent
