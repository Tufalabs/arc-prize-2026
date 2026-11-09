from __future__ import annotations

from collections import deque

from ..core import CachedProgramDslAgent

LEVEL_SPECS = (
    {
        "name": "level_1",
        "emitter": ("LEFT", 5),
        "mirrors": ((3, 5, "\\"),),
        "crystals": ((3, 1, True),),
        "blockers": (),
    },
    {
        "name": "level_2",
        "emitter": ("BOTTOM", 1),
        "mirrors": ((1, 4, "\\"), (4, 4, "/")),
        "crystals": ((4, 6, True),),
        "blockers": (),
    },
    {
        "name": "level_3",
        "emitter": ("LEFT", 6),
        "mirrors": ((2, 6, "\\"), (2, 3, "\\"), (6, 3, "/")),
        "crystals": ((6, 5, True),),
        "blockers": ((1, 1), (2, 1), (3, 1), (4, 4), (4, 5), (4, 6)),
    },
    {
        "name": "level_4",
        "emitter": ("BOTTOM", 3),
        "mirrors": ((3, 5, "\\"), (1, 5, "\\"), (6, 5, "\\")),
        "crystals": ((1, 2, False), (6, 2, True)),
        "blockers": ((4, 3), (4, 4)),
    },
    {
        "name": "level_5",
        "emitter": ("TOP", 1),
        "mirrors": ((1, 2, "/"), (5, 2, "\\"), (5, 6, "\\"), (2, 6, "/")),
        "crystals": ((2, 4, True),),
        "blockers": ((3, 3), (4, 3), (3, 4), (4, 4), (6, 5), (6, 6)),
    },
)

BOARD_SIZE = 8
BOARD_ORIGIN_X = 8
BOARD_ORIGIN_Y = 12
CELL_SIZE = 6

DIR_UP = (0, -1)
DIR_DOWN = (0, 1)
DIR_LEFT = (-1, 0)
DIR_RIGHT = (1, 0)


def mirror_turn(orientation: str, direction: tuple[int, int]) -> tuple[int, int]:
    if orientation == "/":
        return {DIR_UP: DIR_RIGHT, DIR_RIGHT: DIR_UP, DIR_DOWN: DIR_LEFT, DIR_LEFT: DIR_DOWN}[direction]
    return {DIR_UP: DIR_LEFT, DIR_LEFT: DIR_UP, DIR_DOWN: DIR_RIGHT, DIR_RIGHT: DIR_DOWN}[direction]


def socket_click(cell_x: int, cell_y: int) -> dict[str, int]:
    return {"x": BOARD_ORIGIN_X + cell_x * CELL_SIZE + 3, "y": BOARD_ORIGIN_Y + cell_y * CELL_SIZE + 3}


class MirrorShotDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "mirrorshot-0001") -> None:
        super().__init__(game_id=game_id, total_levels=len(LEVEL_SPECS))

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        del env
        spec = LEVEL_SPECS[self._current_level_idx]
        plan = self._solve_level(spec)
        return [*plan, (5, {})]

    def _solve_level(self, spec) -> list[tuple[int, dict[str, int]]]:
        mirrors = spec["mirrors"]
        initial = tuple(mirror[2] for mirror in mirrors)
        queue = deque([(initial, [])])
        seen = {initial}

        while queue:
            state, path = queue.popleft()
            if self._fires_success(spec, state):
                return [*path, (5, {})]
            for idx, mirror in enumerate(mirrors):
                toggled = list(state)
                toggled[idx] = "/" if toggled[idx] == "\\" else "\\"
                toggled_state = tuple(toggled)
                if toggled_state in seen:
                    continue
                seen.add(toggled_state)
                queue.append((toggled_state, [*path, (6, socket_click(mirror[0], mirror[1]))]))
        raise RuntimeError(f"mirrorshot DSL found no solution for {spec['name']}")

    def _fires_success(self, spec, state: tuple[str, ...]) -> bool:
        mirrors = {(mirror[0], mirror[1]): state[idx] for idx, mirror in enumerate(spec["mirrors"])}
        crystals = {(crystal[0], crystal[1]): crystal[2] for crystal in spec["crystals"]}
        blockers = set(spec["blockers"])
        emitter_side, emitter_index = spec["emitter"]

        if emitter_side == "LEFT":
            cell_x, cell_y, direction = 0, emitter_index, DIR_RIGHT
        elif emitter_side == "RIGHT":
            cell_x, cell_y, direction = BOARD_SIZE - 1, emitter_index, DIR_LEFT
        elif emitter_side == "TOP":
            cell_x, cell_y, direction = emitter_index, 0, DIR_DOWN
        else:
            cell_x, cell_y, direction = emitter_index, BOARD_SIZE - 1, DIR_UP

        visited: set[tuple[int, int, tuple[int, int]]] = set()
        while True:
            if not (0 <= cell_x < BOARD_SIZE and 0 <= cell_y < BOARD_SIZE):
                return False
            state_key = (cell_x, cell_y, direction)
            if state_key in visited:
                return False
            visited.add(state_key)
            position = (cell_x, cell_y)
            if position in blockers:
                return False
            if position in crystals:
                return bool(crystals[position])
            if position in mirrors:
                direction = mirror_turn(mirrors[position], direction)
            cell_x += direction[0]
            cell_y += direction[1]


AGENT_CLASS = MirrorShotDslAgent
