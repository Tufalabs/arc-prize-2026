from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "tron_light_cycles-0001"

WIDTH = 32
HEIGHT = 22
ARENA_Y_MIN = 2
ARENA_Y_MAX = 21

COLORS = {
    "outside": 0,
    "floor": 1,
    "wall": 2,
    "player_rear": 3,
    "player_front": 4,
    "player_trail": 5,
    "ai_rear": 6,
    "ai_front": 7,
    "ai_trail": 8,
    "time_fill": 9,
    "time_empty": 10,
    "gate_closed": 11,
    "highlight": 12,
    "portal_a": 13,
    "portal_b": 14,
    "crash": 15,
}

DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
REV_DIR = {(0, -1): (0, 1), (0, 1): (0, -1), (-1, 0): (1, 0), (1, 0): (-1, 0)}
LEFT_TURN = {(0, -1): (-1, 0), (-1, 0): (0, 1), (0, 1): (1, 0), (1, 0): (0, -1)}
RIGHT_TURN = {(0, -1): (1, 0), (1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1)}

PHASE_OPEN = 0
PHASE_WARN_TO_CLOSED = 1
PHASE_CLOSED = 2
PHASE_WARN_TO_OPEN = 3

FRONT_TO_DIR = {"^": (0, -1), "v": (0, 1), "<": (-1, 0), ">": (1, 0)}

LEVEL_LAYOUTS = [
    [
        "================================",
        "                                ",
        "################################",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#              @>              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "################################",
    ],
    [
        "================================",
        "                                ",
        "################################",
        "#                              #",
        "#   ########################   #",
        "#   #                      #   #",
        "#   #  ######      ######  #   #",
        "#   #  #                #  #   #",
        "#   #  #  ####    ####  #  #   #",
        "#   #  #  #          #  #  #   #",
        "#   #  #  #  ######  #  #  #   #",
        "#   #  #  #  #    #  #  #  #   #",
        "#   #  #  #  ######  #  #  #   #",
        "#   #  #  #          #  #  #   #",
        "#   #  #  ####    ####  #  #   #",
        "#   #  #                #  #   #",
        "#   #  ######      ######  #   #",
        "#   #                      #   #",
        "#   ########################   #",
        "#          ^                   #",
        "#          @                   #",
        "################################",
    ],
    [
        "================================",
        "                                ",
        "################################",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#            ######            #",
        "#            ######            #",
        "#            ######            #",
        "#    @>      ######      <&    #",
        "#            ######            #",
        "#            ######            #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#                              #",
        "################################",
    ],
    [
        "================================",
        "                                ",
        "################################",
        "#                              #",
        "#     ####          ####       #",
        "#     ####          ####       #",
        "#              &               #",
        "#              v               #",
        "#                              #",
        "#                              #",
        "#                              #",
        "#              |               #",
        "###############|################",
        "#              |               #",
        "#                              #",
        "#     ####          ####       #",
        "#     ####          ####       #",
        "#                              #",
        "#              ^               #",
        "#              @               #",
        "#                              #",
        "################################",
    ],
    [
        "================================",
        "                                ",
        "################################",
        "#              #               #",
        "#              #               #",
        "#              #               #",
        "#   ##         #               #",
        "#   ##         #               #",
        "#              #               #",
        "#              #               #",
        "#              #    ##         #",
        "#              #    ##         #",
        "#           @>   <&            #",
        "#              #               #",
        "#        ##    #               #",
        "#        ##    #               #",
        "#              #          ##   #",
        "#              #          ##   #",
        "#              #            ^  #",
        "#              #            &  #",
        "#              #               #",
        "################################",
    ],
    [
        "================================",
        "                                ",
        "################################",
        "#              #               #",
        "#   ***        #               #",
        "#   * *        #          <&   #",
        "#   ***        #               #",
        "#      ######  #  ######       #",
        "#         &    #               #",
        "#         v    #     ######    #",
        "#      ######  |               #",
        "#              |               #",
        "#              |               #",
        "#   ######     #     ######    #",
        "#              #               #",
        "#              |               #",
        "#  ######      |       ***     #",
        "#              |       * *     #",
        "#              #       ***     #",
        "#  @>          #            ^  #",
        "#              #            &  #",
        "################################",
    ],
]

LEVEL_CONFIGS = [
    {"time_limit": 120, "ai_types": []},
    {"time_limit": 150, "ai_types": []},
    {"time_limit": 180, "ai_types": [1]},
    {"time_limit": 200, "ai_types": [1], "gate_cycles": [{"open": 6, "closed": 6, "initial_open": False}]},
    {"time_limit": 230, "ai_types": [1, 1], "gap_enabled": True},
    {
        "time_limit": 260,
        "ai_types": [2, 2, 2],
        "gap_enabled": True,
        "gate_cycles": [
            {"open": 5, "closed": 5, "initial_open": False},
            {"open": 3, "closed": 3, "initial_open": True},
        ],
    },
]


@dataclass(frozen=True)
class ParsedLevel:
    level_idx: int
    time_limit: int
    walls_mask: int
    gate_masks: tuple[int, ...]
    gate_cycle: tuple[tuple[int, int, bool], ...]
    portal_ring_mask: int
    portal_pair: tuple[tuple[int, int], tuple[int, int]] | None
    player_front: tuple[int, int]
    player_rear: tuple[int, int]
    player_dir: tuple[int, int]
    ai_starts: tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int], ...]
    gap_enabled: bool


def _cell_bit(x: int, y: int) -> int:
    return 1 << (y * WIDTH + x)


def _mask_has(mask: int, x: int, y: int) -> bool:
    if x < 0 or y < 0 or x >= WIDTH or y >= HEIGHT:
        return False
    return bool(mask & _cell_bit(x, y))


def _iter_mask(mask: int):
    idx = 0
    data = int(mask)
    while data:
        if data & 1:
            y, x = divmod(idx, WIDTH)
            yield (x, y)
        idx += 1
        data >>= 1


def _connected_components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    comps: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        queue: deque[tuple[int, int]] = deque([start])
        remaining.remove(start)
        comp: list[tuple[int, int]] = [start]
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                nxt = (nx, ny)
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
                    comp.append(nxt)
        comps.append(sorted(comp))
    comps.sort(key=lambda comp: comp[0])
    return comps


def _front_for_rear(lines: list[str], rear: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    rx, ry = rear
    for dx, dy in FRONT_TO_DIR.values():
        fx, fy = rx + dx, ry + dy
        if 0 <= fx < WIDTH and 0 <= fy < HEIGHT and lines[fy][fx] in FRONT_TO_DIR:
            front_dir = FRONT_TO_DIR[lines[fy][fx]]
            if front_dir == (dx, dy):
                return (fx, fy), front_dir
    raise RuntimeError(f"rear bike cell at {rear} missing front marker")


def parse_level(level_idx: int) -> ParsedLevel:
    lines = LEVEL_LAYOUTS[level_idx]
    if len(lines) != HEIGHT or any(len(row) != WIDTH for row in lines):
        raise RuntimeError(f"level {level_idx + 1} has invalid layout dimensions")

    cfg = LEVEL_CONFIGS[level_idx]

    wall_mask = 0
    gate_cells: set[tuple[int, int]] = set()
    portal_cells: set[tuple[int, int]] = set()
    player_rear = None
    ai_rears: list[tuple[int, int]] = []

    for y, row in enumerate(lines):
        for x, ch in enumerate(row):
            if ch == "#":
                wall_mask |= _cell_bit(x, y)
            elif ch == "|":
                gate_cells.add((x, y))
            elif ch == "*":
                portal_cells.add((x, y))
            elif ch == "@":
                player_rear = (x, y)
            elif ch == "&":
                ai_rears.append((x, y))

    if player_rear is None:
        raise RuntimeError(f"level {level_idx + 1} missing player rear")

    player_front, player_dir = _front_for_rear(lines, player_rear)

    ai_types = list(cfg.get("ai_types", []))
    if len(ai_types) != len(ai_rears):
        raise RuntimeError(f"level {level_idx + 1} AI count mismatch")

    ai_starts: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]] = []
    for i, rear in enumerate(sorted(ai_rears)):
        front, ai_dir = _front_for_rear(lines, rear)
        ai_starts.append((front, rear, ai_dir, int(ai_types[i])))

    gate_comps = _connected_components(gate_cells)
    gate_cycles_cfg = list(cfg.get("gate_cycles", []))
    if gate_comps and len(gate_comps) != len(gate_cycles_cfg):
        raise RuntimeError(f"level {level_idx + 1} gate cycle mismatch")

    gate_masks: list[int] = []
    gate_cycle: list[tuple[int, int, bool]] = []
    for idx, comp in enumerate(gate_comps):
        mask = 0
        for x, y in comp:
            mask |= _cell_bit(x, y)
        gate_masks.append(mask)
        gc = gate_cycles_cfg[idx]
        gate_cycle.append((int(gc["open"]), int(gc["closed"]), bool(gc["initial_open"])))

    portal_pair = None
    portal_mask = 0
    if portal_cells:
        comps = _connected_components(portal_cells)
        if len(comps) >= 2:
            centers: list[tuple[int, int]] = []
            for comp in comps[:2]:
                xs = [x for x, _ in comp]
                ys = [y for _, y in comp]
                centers.append(((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2))
            portal_pair = (centers[0], centers[1])
        for x, y in portal_cells:
            portal_mask |= _cell_bit(x, y)

    return ParsedLevel(
        level_idx=int(level_idx),
        time_limit=int(cfg["time_limit"]),
        walls_mask=wall_mask,
        gate_masks=tuple(gate_masks),
        gate_cycle=tuple(gate_cycle),
        portal_ring_mask=portal_mask,
        portal_pair=portal_pair,
        player_front=player_front,
        player_rear=player_rear,
        player_dir=player_dir,
        ai_starts=tuple(ai_starts),
        gap_enabled=bool(cfg.get("gap_enabled", False)),
    )


PARSED_LEVELS = tuple(parse_level(i) for i in range(6))


def initial_state(model: ParsedLevel) -> dict:
    gate_state: list[tuple[int, int]] = []
    for open_steps, closed_steps, initial_open in model.gate_cycle:
        if initial_open:
            gate_state.append((PHASE_OPEN, int(open_steps)))
        else:
            gate_state.append((PHASE_CLOSED, int(closed_steps)))

    return {
        "time_left": int(model.time_limit),
        "player_front": tuple(model.player_front),
        "player_rear": tuple(model.player_rear),
        "player_dir": tuple(model.player_dir),
        "player_alive": True,
        "player_trail": 0,
        "ai_trail": 0,
        "ais": [
            {"front": tuple(front), "rear": tuple(rear), "dir": tuple(ai_dir), "alive": True, "ai_type": int(ai_type)}
            for front, rear, ai_dir, ai_type in model.ai_starts
        ],
        "gate_state": gate_state,
        "gap_active": 0,
        "gap_cooldown": 0,
        "pulse": 0,
        "failed": False,
        "flash_cell": None,
        "flash_timer": 0,
    }


def _player_rear_color(state: dict, model: ParsedLevel) -> int:
    if not model.gap_enabled:
        return COLORS["player_rear"]
    if int(state["gap_active"]) > 0:
        return COLORS["highlight"]
    if int(state["gap_cooldown"]) > 0:
        return COLORS["time_empty"]
    return COLORS["player_rear"]


def _gate_passable_for_phase(phase: int) -> bool:
    return phase in (PHASE_OPEN, PHASE_WARN_TO_CLOSED)


def _gate_color_for_phase(phase: int) -> int | None:
    if phase in (PHASE_OPEN,):
        return None
    if phase in (PHASE_WARN_TO_CLOSED, PHASE_WARN_TO_OPEN):
        return COLORS["highlight"]
    return COLORS["gate_closed"]


def _is_static_blocked(model: ParsedLevel, state: dict, x: int, y: int, include_trails: bool = True) -> bool:
    if x < 0 or x >= WIDTH or y < ARENA_Y_MIN or y > ARENA_Y_MAX:
        return True
    bit = _cell_bit(x, y)
    if model.walls_mask & bit:
        return True
    if include_trails and (((int(state["player_trail"]) | int(state["ai_trail"])) & bit) != 0):
        return True
    for gate_idx, gate_mask in enumerate(model.gate_masks):
        if gate_mask & bit:
            phase, _remaining = state["gate_state"][gate_idx]
            return not _gate_passable_for_phase(int(phase))
    return False


def _occupied_cells_current(state: dict) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    if bool(state["player_alive"]):
        out.add(tuple(state["player_front"]))
        out.add(tuple(state["player_rear"]))
    for ai in state["ais"]:
        if bool(ai["alive"]):
            out.add(tuple(ai["front"]))
            out.add(tuple(ai["rear"]))
    return out


def _free_neighbors(model: ParsedLevel, state: dict, x: int, y: int, occupied: set[tuple[int, int]]) -> int:
    count = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if (nx, ny) in occupied:
            continue
        if _is_static_blocked(model, state, nx, ny):
            continue
        count += 1
    return count


def _ai_choose_dir(
    model: ParsedLevel, state: dict, ai: dict, occupied: set[tuple[int, int]]
) -> tuple[tuple[int, int] | None, bool]:
    current_dir = tuple(ai["dir"])
    options = [current_dir, LEFT_TURN[current_dir], RIGHT_TURN[current_dir]]

    open_dirs: list[tuple[int, int]] = []
    for d in options:
        nx, ny = int(ai["front"][0]) + d[0], int(ai["front"][1]) + d[1]
        if (nx, ny) in occupied:
            continue
        if _is_static_blocked(model, state, nx, ny):
            continue
        open_dirs.append(d)

    if not open_dirs:
        return None, True

    if int(ai["ai_type"]) == 1 or len(open_dirs) <= 1:
        return open_dirs[0], False

    best_dir = open_dirs[0]
    best_score = -1
    for d in open_dirs:
        nx, ny = int(ai["front"][0]) + d[0], int(ai["front"][1]) + d[1]
        score = _free_neighbors(model, state, nx, ny, occupied)
        if score > best_score:
            best_dir = d
            best_score = score
    return best_dir, False


def _advance_gates(model: ParsedLevel, state: dict) -> set[int]:
    closed_now: set[int] = set()
    new_state: list[tuple[int, int]] = []
    for gate_idx, (phase, remaining) in enumerate(state["gate_state"]):
        open_steps, closed_steps, _initial_open = model.gate_cycle[gate_idx]
        phase = int(phase)
        remaining = int(remaining) - 1
        if remaining > 0:
            new_state.append((phase, remaining))
            continue

        if phase == PHASE_OPEN:
            new_state.append((PHASE_WARN_TO_CLOSED, 1))
        elif phase == PHASE_WARN_TO_CLOSED:
            new_state.append((PHASE_CLOSED, int(closed_steps)))
            closed_now.add(gate_idx)
        elif phase == PHASE_CLOSED:
            new_state.append((PHASE_WARN_TO_OPEN, 1))
        else:
            new_state.append((PHASE_OPEN, int(open_steps)))

    state["gate_state"] = new_state
    return closed_now


def simulate_action(model: ParsedLevel, state: dict, action_id: int) -> tuple[dict, str]:
    nxt = {
        "time_left": int(state["time_left"]),
        "player_front": tuple(state["player_front"]),
        "player_rear": tuple(state["player_rear"]),
        "player_dir": tuple(state["player_dir"]),
        "player_alive": bool(state["player_alive"]),
        "player_trail": int(state["player_trail"]),
        "ai_trail": int(state["ai_trail"]),
        "ais": [
            {
                "front": tuple(ai["front"]),
                "rear": tuple(ai["rear"]),
                "dir": tuple(ai["dir"]),
                "alive": bool(ai["alive"]),
                "ai_type": int(ai["ai_type"]),
            }
            for ai in state["ais"]
        ],
        "gate_state": [(int(phase), int(remaining)) for phase, remaining in state["gate_state"]],
        "gap_active": int(state["gap_active"]),
        "gap_cooldown": int(state["gap_cooldown"]),
        "pulse": int(state["pulse"]),
        "failed": bool(state["failed"]),
        "flash_cell": state["flash_cell"],
        "flash_timer": int(state["flash_timer"]),
    }

    if nxt["failed"]:
        return nxt, "fail"

    action_id = int(action_id)

    if action_id == int(GameAction.RESET.value):
        return nxt, "running"

    if action_id in DIRS:
        requested = DIRS[action_id]
        if requested != REV_DIR[tuple(nxt["player_dir"])]:
            nxt["player_dir"] = requested
    elif action_id == int(GameAction.ACTION5.value) and model.gap_enabled:
        if int(nxt["gap_active"]) <= 0 and int(nxt["gap_cooldown"]) <= 0:
            nxt["gap_active"] = 3
            nxt["gap_cooldown"] = 12

    occupied_now = _occupied_cells_current(nxt)

    movers: list[tuple[str, int, tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    movers.append(("player", -1, tuple(nxt["player_front"]), tuple(nxt["player_rear"]), tuple(nxt["player_dir"])))

    forced_crash: set[tuple[str, int]] = set()
    for ai_idx, ai in enumerate(nxt["ais"]):
        if not bool(ai["alive"]):
            continue
        chosen, blocked = _ai_choose_dir(model, nxt, ai, occupied_now)
        if blocked or chosen is None:
            forced_crash.add(("ai", ai_idx))
            chosen = tuple(ai["dir"])
        ai["dir"] = tuple(chosen)
        movers.append(("ai", ai_idx, tuple(ai["front"]), tuple(ai["rear"]), tuple(ai["dir"])))

    proposed: dict[tuple[str, int], dict] = {}
    crashed: set[tuple[str, int]] = set(forced_crash)

    for who, idx, front, rear, direction in movers:
        fx, fy = front
        nx, ny = fx + direction[0], fy + direction[1]
        proposal = {"old_front": front, "old_rear": rear, "new_front": (nx, ny), "new_rear": front, "dir": direction}
        proposed[(who, idx)] = proposal
        if _is_static_blocked(model, nxt, nx, ny):
            crashed.add((who, idx))

    def _apply_dynamic_collisions() -> None:
        cell_map: dict[tuple[int, int], list[tuple[str, int]]] = {}
        for key, proposal in proposed.items():
            if key in crashed:
                continue
            for cell in (proposal["new_front"], proposal["new_rear"]):
                cell_map.setdefault(cell, []).append(key)
        for _cell, owners in cell_map.items():
            if len(owners) > 1:
                for owner in owners:
                    crashed.add(owner)

    _apply_dynamic_collisions()

    portal_map: dict[tuple[int, int], tuple[int, int]] = {}
    if model.portal_pair is not None:
        a, b = model.portal_pair
        portal_map[a] = b
        portal_map[b] = a

    if portal_map:
        for key, proposal in proposed.items():
            if key in crashed:
                continue
            src = tuple(proposal["new_front"])
            dst = portal_map.get(src)
            if dst is None:
                continue
            direction = tuple(proposal["dir"])
            dst_front = tuple(dst)
            dst_rear = (dst_front[0] - direction[0], dst_front[1] - direction[1])
            if _is_static_blocked(model, nxt, dst_front[0], dst_front[1]):
                crashed.add(key)
                continue
            if _is_static_blocked(model, nxt, dst_rear[0], dst_rear[1]):
                crashed.add(key)
                continue
            proposal["new_front"] = dst_front
            proposal["new_rear"] = dst_rear

        _apply_dynamic_collisions()

    player_crashed = False
    for key, proposal in proposed.items():
        who, idx = key
        if key in crashed:
            if who == "player":
                player_crashed = True
                nxt["player_alive"] = False
                nxt["flash_cell"] = tuple(proposal["new_front"])
                nxt["flash_timer"] = 1
            else:
                nxt["ais"][idx]["alive"] = False
            continue

        old_rear = tuple(proposal["old_rear"])
        if who == "player":
            if int(nxt["gap_active"]) <= 0:
                nxt["player_trail"] |= _cell_bit(old_rear[0], old_rear[1])
            nxt["player_front"] = tuple(proposal["new_front"])
            nxt["player_rear"] = tuple(proposal["new_rear"])
            nxt["player_dir"] = tuple(proposal["dir"])
            nxt["player_alive"] = True
        else:
            nxt["ai_trail"] |= _cell_bit(old_rear[0], old_rear[1])
            ai = nxt["ais"][idx]
            ai["front"] = tuple(proposal["new_front"])
            ai["rear"] = tuple(proposal["new_rear"])
            ai["dir"] = tuple(proposal["dir"])
            ai["alive"] = True

    closed_now = _advance_gates(model, nxt)
    if closed_now:
        occupied_after: dict[tuple[int, int], tuple[str, int]] = {}
        if bool(nxt["player_alive"]):
            occupied_after[tuple(nxt["player_front"])] = ("player", -1)
            occupied_after[tuple(nxt["player_rear"])] = ("player", -1)
        for ai_idx, ai in enumerate(nxt["ais"]):
            if bool(ai["alive"]):
                occupied_after[tuple(ai["front"])] = ("ai", ai_idx)
                occupied_after[tuple(ai["rear"])] = ("ai", ai_idx)

        for gate_idx in closed_now:
            for x, y in _iter_mask(model.gate_masks[gate_idx]):
                owner = occupied_after.get((x, y))
                if owner is None:
                    continue
                if owner[0] == "player":
                    player_crashed = True
                    nxt["player_alive"] = False
                    nxt["flash_cell"] = (x, y)
                    nxt["flash_timer"] = 1
                else:
                    nxt["ais"][owner[1]]["alive"] = False

    nxt["time_left"] = int(nxt["time_left"]) - 1
    nxt["pulse"] = int(nxt["pulse"]) + 1
    if int(nxt["gap_active"]) > 0:
        nxt["gap_active"] = int(nxt["gap_active"]) - 1
    if int(nxt["gap_cooldown"]) > 0:
        nxt["gap_cooldown"] = int(nxt["gap_cooldown"]) - 1

    if player_crashed:
        nxt["failed"] = True
        return nxt, "fail"

    ai_alive = sum(1 for ai in nxt["ais"] if bool(ai["alive"]))
    if len(nxt["ais"]) > 0 and ai_alive == 0:
        return nxt, "win"

    if int(nxt["time_left"]) <= 0:
        return nxt, "win"

    return nxt, "running"


def render_board(model: ParsedLevel, state: dict) -> np.ndarray:
    board = np.full((HEIGHT, WIDTH), COLORS["floor"], dtype=np.int8)

    fill = round((max(0, int(state["time_left"])) / float(max(1, model.time_limit))) * WIDTH)
    fill = max(0, min(WIDTH, fill))
    board[0, :] = COLORS["time_empty"]
    if fill > 0:
        board[0, :fill] = COLORS["time_fill"]

    for x in range(WIDTH):
        board[1, x] = COLORS["floor"]

    for x, y in _iter_mask(model.walls_mask):
        board[y, x] = COLORS["wall"]

    for x, y in _iter_mask(int(state["player_trail"])):
        board[y, x] = COLORS["player_trail"]
    for x, y in _iter_mask(int(state["ai_trail"])):
        board[y, x] = COLORS["ai_trail"]

    if model.portal_ring_mask:
        portal_color = COLORS["portal_a"] if (int(state["pulse"]) % 2 == 0) else COLORS["portal_b"]
        for x, y in _iter_mask(model.portal_ring_mask):
            board[y, x] = portal_color

    for gate_idx, gate_mask in enumerate(model.gate_masks):
        phase, _remaining = state["gate_state"][gate_idx]
        gate_color = _gate_color_for_phase(int(phase))
        if gate_color is None:
            continue
        for x, y in _iter_mask(gate_mask):
            board[y, x] = gate_color

    for ai in state["ais"]:
        if not bool(ai["alive"]):
            continue
        rx, ry = ai["rear"]
        fx, fy = ai["front"]
        if 0 <= rx < WIDTH and 0 <= ry < HEIGHT:
            board[ry, rx] = COLORS["ai_rear"]
        if 0 <= fx < WIDTH and 0 <= fy < HEIGHT:
            board[fy, fx] = COLORS["ai_front"]

    if bool(state["player_alive"]):
        prx, pry = state["player_rear"]
        pfx, pfy = state["player_front"]
        if 0 <= prx < WIDTH and 0 <= pry < HEIGHT:
            board[pry, prx] = _player_rear_color(state, model)
        if 0 <= pfx < WIDTH and 0 <= pfy < HEIGHT:
            board[pfy, pfx] = COLORS["player_front"]

    if int(state["flash_timer"]) > 0 and state["flash_cell"] is not None:
        fx, fy = state["flash_cell"]
        if 0 <= fx < WIDTH and 0 <= fy < HEIGHT:
            board[fy, fx] = COLORS["crash"]

    return board


def _build_level(level_idx: int) -> Level:
    return Level(
        grid_size=(WIDTH, HEIGHT),
        sprites=[
            Sprite(
                np.full((HEIGHT, WIDTH), COLORS["floor"], dtype=np.int8),
                name="board",
                x=0,
                y=0,
                layer=1,
                tags=["board"],
                collidable=False,
            )
        ],
        data={"level_idx": int(level_idx)},
    )


class TronLightCycles(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._level_idx = 0
        self._model = PARSED_LEVELS[0]
        self._runtime = initial_state(self._model)
        super().__init__(
            game_id=GAME_ID,
            levels=[_build_level(i) for i in range(6)],
            camera=Camera(width=WIDTH, height=HEIGHT, background=COLORS["outside"]),
            win_score=6,
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        idx = int(level.get_data("level_idx") or 0)
        self._level_idx = idx
        self._model = PARSED_LEVELS[idx]
        self._runtime = initial_state(self._model)
        self._sync_view()

    def _sync_view(self) -> None:
        board_sprite = self.current_level.get_sprites_by_name("board")[0]
        board_sprite.pixels = render_board(self._model, self._runtime)

    def step(self) -> None:
        raw_action = self.action.id
        action_id = int(getattr(raw_action, "value", raw_action))
        if action_id == int(GameAction.RESET.value):
            self.complete_action()
            return
        self._runtime, outcome = simulate_action(self._model, self._runtime, action_id)

        self._sync_view()

        if int(self._runtime["flash_timer"]) > 0:
            self._runtime["flash_timer"] = max(0, int(self._runtime["flash_timer"]) - 1)

        if outcome == "win":
            self.next_level()
        elif outcome == "fail":
            self.lose()
        self.complete_action()
