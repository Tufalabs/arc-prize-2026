from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "one_enemy_duel-0001"
GRID_W = 26
GRID_H = 20

COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_HUD_BG = 3
COLOR_BORDER = 4
COLOR_TIME_FILL = 5
COLOR_TIME_EMPTY = 6
COLOR_HP = 7
COLOR_PLAYER_BODY = 8
COLOR_PLAYER_OUTLINE = 9
COLOR_PLAYER_GUARD = 10
COLOR_ENEMY_BODY = 11
COLOR_ENEMY_OUTLINE = 12
COLOR_SPARK = 13
COLOR_WARN = 14
COLOR_DANGER = 15

MOVE_DELTAS = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}

LEVEL_GRIDS = [
    [
        "==========================",
        "=$$$,||||||||||||||||,,,,=",
        "=,,,,,,,,,,,,,,,,,,,,,,,,=",
        "==========================",
        "=........................=",
        "=................%%%%%...=",
        "=..+++...........%&&&%...=",
        "=..+@+...........%&&&%...=",
        "=..+++...........%&&&%...=",
        "=................%%%%%...=",
        "=........................=",
        "=........................=",
        "=........................=",
        "=........................=",
        "=........................=",
        "=........................=",
        "=........................=",
        "=........................=",
        "=........................=",
        "==========================",
    ],
    [
        "==========================",
        "=$$$,||||||||||||||||,,,,=",
        "=,,,,,,,,,,,,,,,,,,,,,,,,=",
        "==========================",
        "=............#...........=",
        "=............#...........=",
        "=............#...%%%%%...=",
        "=............#...%&&&%...=",
        "=............#...%&&&%...=",
        "=......##....#...%&&&%...=",
        "=......##........%%%%%...=",
        "=........................=",
        "=........................=",
        "=..+++.......#...........=",
        "=..+@+.......#...........=",
        "=..+++.......#...........=",
        "=............#...........=",
        "=............#...........=",
        "=............#...........=",
        "==========================",
    ],
    [
        "==========================",
        "=$$$,||||||||||||||||,,,,=",
        "=,,,,,,,,,,,,,,,,,,,,,,,,=",
        "==========================",
        "=........................=",
        "=........................=",
        "=.........##.....%%%%%...=",
        "=.........##.....%&&&%...=",
        "=................%&&&%...=",
        "=................%&&&%...=",
        "=............##..%%%%%...=",
        "=............##..........=",
        "=........................=",
        "=......##................=",
        "=..+++.##................=",
        "=..+@+.##................=",
        "=..+++...................=",
        "=........................=",
        "=........................=",
        "==========================",
    ],
    [
        "==========================",
        "=$$$,||||||||||||||||,,,,=",
        "=,,,,,,,,,,,,,,,,,,,,,,,,=",
        "==========================",
        "=........................=",
        "=........................=",
        "=................%%%%%...=",
        "=...........##...%&&&%...=",
        "=...........##...%&&&%...=",
        "=...........##...%&&&%...=",
        "=................%%%%%...=",
        "=........................=",
        "=...........##...........=",
        "=.....##.................=",
        "=..+++##.................=",
        "=..+@+......##...........=",
        "=..+++......##...........=",
        "=...........##...........=",
        "=........................=",
        "==========================",
    ],
    [
        "==========================",
        "=$$$,||||||||||||||||,,,,=",
        "=,,,,,,,,,,,,,,,,,,,,,,,,=",
        "==========================",
        "=........................=",
        "=................%%%%%...=",
        "=................%&&&%...=",
        "=................%&&&%...=",
        "=................%&&&%...=",
        "=................%%%%%...=",
        "=......;.;.;.;.;.;.......=",
        "=.......;.;.;.;.;.;......=",
        "=......;.;.;.;.;.;.......=",
        "=.......;.;.;.;.;.;......=",
        "=..+++.;.;.;.;.;.;.......=",
        "=..+@+..;.;.;.;.;.;......=",
        "=..+++.;.;.;.;.;.;.......=",
        "=........................=",
        "=........................=",
        "==========================",
    ],
    [
        "==========================",
        "=$$$,||||||||||||||||,,,,=",
        "=,,,,,,,,,,,,,,,,,,,,,,,,=",
        "==========================",
        "=........................=",
        "=................%%%%%...=",
        "=.........##.....%&&&%...=",
        "=.........##.....%&&&%...=",
        "=................%&&&%...=",
        "=................%%%%%...=",
        "=##;;;##############;;;##=",
        "=........................=",
        "=....;......##.....;.....=",
        "=..+++......##.##........=",
        "=..+@+.;....##.##........=",
        "=..+++......##.....;.....=",
        "=....;......##...........=",
        "=........................=",
        "=........................=",
        "==========================",
    ],
]

LEVEL_RULES = [
    {
        "timer": 160,
        "hp": 3,
        "melee": True,
        "cleave": False,
        "projectile": False,
        "dash": False,
        "melee_telegraph": 2,
        "move_idle": False,
        "spikes": False,
    },
    {
        "timer": 150,
        "hp": 4,
        "melee": True,
        "cleave": True,
        "projectile": False,
        "dash": False,
        "melee_telegraph": 1,
        "move_idle": True,
        "spikes": False,
    },
    {
        "timer": 140,
        "hp": 5,
        "melee": True,
        "cleave": True,
        "projectile": True,
        "dash": False,
        "melee_telegraph": 1,
        "move_idle": True,
        "spikes": False,
    },
    {
        "timer": 130,
        "hp": 6,
        "melee": True,
        "cleave": True,
        "projectile": False,
        "dash": True,
        "melee_telegraph": 1,
        "move_idle": True,
        "spikes": False,
    },
    {
        "timer": 120,
        "hp": 7,
        "melee": True,
        "cleave": True,
        "projectile": True,
        "dash": False,
        "melee_telegraph": 1,
        "move_idle": True,
        "spikes": True,
    },
    {
        "timer": 110,
        "hp": 8,
        "melee": True,
        "cleave": True,
        "projectile": True,
        "dash": True,
        "melee_telegraph": 1,
        "move_idle": True,
        "spikes": True,
        "enrage": True,
    },
]


@dataclass
class DuelAction:
    action_id: int
    click_x: int | None = None
    click_y: int | None = None


def _enemy_stage(max_hp: int, hp: int) -> int:
    if hp <= 1:
        return 1
    if hp <= max(2, max_hp // 2):
        return 3
    return 5


def _rect_cells(cx: int, cy: int, size: int) -> set[tuple[int, int]]:
    half = size // 2
    out: set[tuple[int, int]] = set()
    for y in range(cy - half, cy + half + 1):
        for x in range(cx - half, cx + half + 1):
            out.add((x, y))
    return out


def _player_cells(state: dict[str, Any]) -> set[tuple[int, int]]:
    return _rect_cells(int(state["player_x"]), int(state["player_y"]), 3)


def _enemy_cells(model: dict[str, Any], state: dict[str, Any]) -> set[tuple[int, int]]:
    size = _enemy_stage(int(model["enemy_max_hp"]), int(state["enemy_hp"]))
    return _rect_cells(int(state["enemy_x"]), int(state["enemy_y"]), size)


def _enemy_body_cells(model: dict[str, Any], state: dict[str, Any]) -> set[tuple[int, int]]:
    size = _enemy_stage(int(model["enemy_max_hp"]), int(state["enemy_hp"]))
    cx = int(state["enemy_x"])
    cy = int(state["enemy_y"])
    if size == 5:
        return _rect_cells(cx, cy, 3)
    if size == 3:
        return {(cx, cy)}
    return {(cx, cy)}


def _walkable_arena(model: dict[str, Any], x: int, y: int) -> bool:
    if x <= 0 or y <= 3 or x >= GRID_W - 1 or y >= GRID_H - 1:
        return False
    return (x, y) not in model["walls"]


def _line_of_sight(
    model: dict[str, Any], start: tuple[int, int], direction: tuple[int, int], targets: set[tuple[int, int]]
) -> bool:
    x, y = start
    dx, dy = direction
    while True:
        x += dx
        y += dy
        if not _walkable_arena(model, x, y):
            return False
        if (x, y) in targets:
            return True


def _snap_dir(from_xy: tuple[int, int], to_xy: tuple[int, int], fallback: tuple[int, int]) -> tuple[int, int]:
    dx = int(to_xy[0] - from_xy[0])
    dy = int(to_xy[1] - from_xy[1])
    if dx == 0 and dy == 0:
        return fallback
    if abs(dx) >= abs(dy):
        return (1, 0) if dx > 0 else (-1, 0)
    return (0, 1) if dy > 0 else (0, -1)


def _slash_cells(px: int, py: int, direction: tuple[int, int]) -> set[tuple[int, int]]:
    dx, dy = direction
    out: set[tuple[int, int]] = set()
    if dx != 0:
        sign = 1 if dx > 0 else -1
        for i in range(2, 5):
            x = px + sign * i
            out.add((x, py - 1))
            out.add((x, py))
            out.add((x, py + 1))
    else:
        sign = 1 if dy > 0 else -1
        for i in range(2, 5):
            y = py + sign * i
            out.add((px - 1, y))
            out.add((px, y))
            out.add((px + 1, y))
    return out


def _melee_strip(model: dict[str, Any], state: dict[str, Any], direction: tuple[int, int]) -> set[tuple[int, int]]:
    size = _enemy_stage(int(model["enemy_max_hp"]), int(state["enemy_hp"]))
    half = size // 2
    ex = int(state["enemy_x"])
    ey = int(state["enemy_y"])
    dx, dy = direction
    out: set[tuple[int, int]] = set()
    if dx != 0:
        sign = 1 if dx > 0 else -1
        for i in range(half + 1, half + 4):
            out.add((ex + sign * i, ey))
    else:
        sign = 1 if dy > 0 else -1
        for i in range(half + 1, half + 4):
            out.add((ex, ey + sign * i))
    return out


def _cleave_box(model: dict[str, Any], state: dict[str, Any], direction: tuple[int, int]) -> set[tuple[int, int]]:
    size = _enemy_stage(int(model["enemy_max_hp"]), int(state["enemy_hp"]))
    half = size // 2
    ex = int(state["enemy_x"])
    ey = int(state["enemy_y"])
    dx, dy = direction
    cx = ex + dx * (half + 1)
    cy = ey + dy * (half + 1)
    return _rect_cells(cx, cy, 3)


def _dash_path(
    model: dict[str, Any], state: dict[str, Any], direction: tuple[int, int], max_dist: int = 10
) -> list[tuple[int, int]]:
    ex = int(state["enemy_x"])
    ey = int(state["enemy_y"])
    dx, dy = direction
    path: list[tuple[int, int]] = []
    x = ex
    y = ey
    for _ in range(max_dist):
        x += dx
        y += dy
        if not _walkable_arena(model, x, y):
            break
        path.append((x, y))
    return path


def _choose_enemy_move(model: dict[str, Any], state: dict[str, Any]) -> tuple[int, int]:
    ex = int(state["enemy_x"])
    ey = int(state["enemy_y"])
    px = int(state["player_x"])
    py = int(state["player_y"])
    options: list[tuple[int, int]] = []
    dx = 0 if ex == px else (1 if px > ex else -1)
    dy = 0 if ey == py else (1 if py > ey else -1)
    if abs(px - ex) >= abs(py - ey):
        if dx != 0:
            options.append((dx, 0))
        if dy != 0:
            options.append((0, dy))
    else:
        if dy != 0:
            options.append((0, dy))
        if dx != 0:
            options.append((dx, 0))
    options.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])

    p_cells = _player_cells(state)
    for odx, ody in options:
        nx = ex + odx
        ny = ey + ody
        if not _walkable_arena(model, nx, ny):
            continue
        if (nx, ny) in p_cells:
            continue
        return nx, ny
    return ex, ey


def _spike_state(model: dict[str, Any], spike_tick: int) -> str:
    if not model["rules"].get("spikes"):
        return "off"
    phase = spike_tick % 6
    if phase == 4:
        return "warn"
    if phase == 5:
        return "active"
    return "idle"


def _empty_state(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_x": int(model["player_start"][0]),
        "player_y": int(model["player_start"][1]),
        "player_guard": False,
        "player_hp": 3,
        "player_flash": 0,
        "player_facing": (1, 0),
        "enemy_x": int(model["enemy_start"][0]),
        "enemy_y": int(model["enemy_start"][1]),
        "enemy_hp": int(model["enemy_max_hp"]),
        "enemy_flash": 0,
        "enemy_stun": 0,
        "enemy_cooldown": 0,
        "enemy_attack": None,
        "enemy_warn": [],
        "enemy_danger": [],
        "player_slash": [],
        "projectiles": [],
        "timer": int(model["rules"]["timer"]),
        "spike_tick": 0,
        "fail_flash": 0,
    }


def _compute_player_move(model: dict[str, Any], state: dict[str, Any], dx: int, dy: int) -> tuple[int, int]:
    nx = int(state["player_x"]) + dx
    ny = int(state["player_y"]) + dy
    p_cells = _rect_cells(nx, ny, 3)
    if any(not _walkable_arena(model, x, y) for (x, y) in p_cells):
        return int(state["player_x"]), int(state["player_y"])
    if p_cells.intersection(_enemy_cells(model, state)):
        return int(state["player_x"]), int(state["player_y"])
    return nx, ny


def _spawn_enemy_attack(model: dict[str, Any], state: dict[str, Any]) -> None:
    if int(state["enemy_stun"]) > 0:
        return
    if int(state["enemy_hp"]) <= 0:
        return
    if state["enemy_attack"] is not None:
        return
    if int(state["enemy_cooldown"]) > 0:
        state["enemy_cooldown"] = int(state["enemy_cooldown"]) - 1
        if model["rules"].get("move_idle"):
            nx, ny = _choose_enemy_move(model, state)
            state["enemy_x"] = nx
            state["enemy_y"] = ny
        return

    rules = model["rules"]
    ex, ey = int(state["enemy_x"]), int(state["enemy_y"])
    px, py = int(state["player_x"]), int(state["player_y"])
    direction = _snap_dir((ex, ey), (px, py), (1, 0))
    p_cells = _player_cells(state)
    manhattan = abs(px - ex) + abs(py - ey)

    if rules.get("dash"):
        same_axis = (px == ex) or (py == ey)
        if same_axis and manhattan <= 10:
            dash_dir = _snap_dir((ex, ey), (px, py), direction)
            path = _dash_path(model, state, dash_dir, 10)
            if path and p_cells.intersection(set(path)):
                state["enemy_attack"] = {
                    "type": "dash",
                    "phase": "telegraph",
                    "timer": 1,
                    "dir": dash_dir,
                    "path": path,
                    "remaining": 2,
                }
                return

    if rules.get("melee") and manhattan <= 4:
        is_cleave = bool(rules.get("cleave"))
        state["enemy_attack"] = {
            "type": "cleave" if is_cleave else "melee",
            "phase": "telegraph",
            "timer": int(rules.get("melee_telegraph", 1)),
            "dir": direction,
        }
        return

    if rules.get("projectile") and manhattan >= 8:
        spawn = (ex + direction[0], ey + direction[1])
        if _walkable_arena(model, spawn[0], spawn[1]) and _line_of_sight(model, spawn, direction, p_cells):
            state["enemy_attack"] = {
                "type": "projectile",
                "phase": "telegraph",
                "timer": 1,
                "dir": direction,
                "spawn": spawn,
            }
            return

    if rules.get("move_idle"):
        nx, ny = _choose_enemy_move(model, state)
        state["enemy_x"] = nx
        state["enemy_y"] = ny


def _advance_enemy_attack(model: dict[str, Any], state: dict[str, Any]) -> bool:
    state["enemy_warn"] = []
    state["enemy_danger"] = []
    counterable_hit = False

    attack = state["enemy_attack"]
    if attack is None:
        return False

    attack_type = str(attack["type"])
    phase = str(attack["phase"])
    direction = tuple(int(v) for v in attack.get("dir", (1, 0)))

    if phase == "telegraph":
        if attack_type == "melee":
            state["enemy_warn"] = sorted(_melee_strip(model, state, direction))
        elif attack_type == "cleave":
            state["enemy_warn"] = sorted(_cleave_box(model, state, direction))
        elif attack_type == "projectile":
            state["enemy_warn"] = [tuple(int(v) for v in attack["spawn"])]
        elif attack_type == "dash":
            state["enemy_warn"] = list(attack.get("path", []))

        attack["timer"] = int(attack["timer"]) - 1
        if int(attack["timer"]) <= 0:
            attack["phase"] = "execute"
        return False

    if attack_type == "melee":
        state["enemy_danger"] = sorted(_melee_strip(model, state, direction))
        counterable_hit = True
        enraged = (
            bool(model["rules"].get("enrage")) and _enemy_stage(int(model["enemy_max_hp"]), int(state["enemy_hp"])) == 1
        )
        state["enemy_cooldown"] = 0 if enraged else 2
        state["enemy_attack"] = None
    elif attack_type == "cleave":
        state["enemy_danger"] = sorted(_cleave_box(model, state, direction))
        counterable_hit = True
        enraged = (
            bool(model["rules"].get("enrage")) and _enemy_stage(int(model["enemy_max_hp"]), int(state["enemy_hp"])) == 1
        )
        state["enemy_cooldown"] = 0 if enraged else 2
        state["enemy_attack"] = None
    elif attack_type == "projectile":
        spawn = tuple(int(v) for v in attack["spawn"])
        state["projectiles"].append((spawn[0], spawn[1], direction[0], direction[1]))
        enraged = (
            bool(model["rules"].get("enrage")) and _enemy_stage(int(model["enemy_max_hp"]), int(state["enemy_hp"])) == 1
        )
        state["enemy_cooldown"] = 0 if enraged else 2
        state["enemy_attack"] = None
    elif attack_type == "dash":
        ex = int(state["enemy_x"])
        ey = int(state["enemy_y"])
        dx, dy = direction
        step_cells: list[tuple[int, int]] = []
        for _ in range(2):
            nx, ny = ex + dx, ey + dy
            if not _walkable_arena(model, nx, ny):
                break
            ex, ey = nx, ny
            step_cells.append((ex, ey))
        state["enemy_x"] = ex
        state["enemy_y"] = ey
        state["enemy_danger"] = step_cells
        counterable_hit = True
        attack["remaining"] = int(attack.get("remaining", 1)) - 1
        if int(attack["remaining"]) <= 0:
            enraged = (
                bool(model["rules"].get("enrage"))
                and _enemy_stage(int(model["enemy_max_hp"]), int(state["enemy_hp"])) == 1
            )
            state["enemy_cooldown"] = 0 if enraged else 2
            state["enemy_attack"] = None
    return counterable_hit


def _move_projectiles(model: dict[str, Any], state: dict[str, Any]) -> None:
    out: list[tuple[int, int, int, int]] = []
    for x, y, dx, dy in list(state["projectiles"]):
        nx, ny = x + dx, y + dy
        if not _walkable_arena(model, nx, ny):
            continue
        out.append((nx, ny, dx, dy))
    state["projectiles"] = out


def apply_action_transition(
    model: dict[str, Any], state: dict[str, Any], action: DuelAction
) -> tuple[dict[str, Any], bool, bool]:
    s = deepcopy(state)

    if int(s["fail_flash"]) > 0:
        return s, False, True

    s["player_slash"] = []

    action_id = int(action.action_id)
    if action_id in MOVE_DELTAS:
        dx, dy = MOVE_DELTAS[action_id]
        nx, ny = _compute_player_move(model, s, dx, dy)
        s["player_x"] = nx
        s["player_y"] = ny
        s["player_facing"] = (dx, dy)
    elif action_id == int(GameAction.ACTION5.value):
        s["player_guard"] = not bool(s["player_guard"])
    elif action_id == int(GameAction.ACTION6.value) and not bool(s["player_guard"]):
        px, py = int(s["player_x"]), int(s["player_y"])
        cx = px if action.click_x is None else int(action.click_x)
        cy = py if action.click_y is None else int(action.click_y)
        direction = _snap_dir((px, py), (cx, cy), tuple(s["player_facing"]))
        s["player_facing"] = direction
        s["player_slash"] = sorted(_slash_cells(px, py, direction))

    counterable = False
    if int(s["enemy_stun"]) <= 0:
        if s["enemy_attack"] is None:
            _spawn_enemy_attack(model, s)
        counterable = _advance_enemy_attack(model, s)

    _move_projectiles(model, s)

    p_cells = _player_cells(s)
    e_cells = _enemy_cells(model, s)

    spike_mode = _spike_state(model, int(s["spike_tick"]))
    spike_active = model["spikes"] if spike_mode == "active" else set()

    hit_enemy = bool(set(s["player_slash"]).intersection(e_cells))
    if hit_enemy:
        s["enemy_hp"] = max(0, int(s["enemy_hp"]) - 1)
        s["enemy_flash"] = 2
        if int(s["enemy_hp"]) <= 0:
            return s, True, False

    enemy_hit_player = bool(set(s["enemy_danger"]).intersection(p_cells))
    projectile_hit = False
    kept_projectiles: list[tuple[int, int, int, int]] = []
    for proj in s["projectiles"]:
        px, py = int(proj[0]), int(proj[1])
        if (px, py) in p_cells:
            projectile_hit = True
            continue
        kept_projectiles.append(proj)
    s["projectiles"] = kept_projectiles

    spike_hit = bool(p_cells.intersection(spike_active))

    took_damage = False
    guarded = bool(s["player_guard"])
    if spike_hit and not guarded:
        took_damage = True
    if enemy_hit_player and not guarded:
        took_damage = True
    if projectile_hit and not guarded:
        took_damage = True

    if enemy_hit_player and guarded and counterable:
        s["enemy_stun"] = 2
        s["enemy_attack"] = None
        s["enemy_cooldown"] = 0

    if took_damage:
        s["player_hp"] = max(0, int(s["player_hp"]) - 1)
        s["player_flash"] = 2

    if int(s["player_hp"]) <= 0:
        s["fail_flash"] = 2
        return s, False, True

    if int(s["player_flash"]) > 0:
        s["player_flash"] = int(s["player_flash"]) - 1
    if int(s["enemy_flash"]) > 0:
        s["enemy_flash"] = int(s["enemy_flash"]) - 1
    if int(s["enemy_stun"]) > 0:
        s["enemy_stun"] = int(s["enemy_stun"]) - 1

    s["enemy_warn"] = []
    s["enemy_danger"] = []
    s["player_slash"] = []

    s["spike_tick"] = int(s["spike_tick"]) + 1
    s["timer"] = int(s["timer"]) - 1
    if int(s["timer"]) <= 0:
        s["fail_flash"] = 2
        return s, False, True

    return s, False, False


def _parse_level_model(level_idx: int) -> dict[str, Any]:
    lines = LEVEL_GRIDS[level_idx]
    walls: set[tuple[int, int]] = set()
    spikes: set[tuple[int, int]] = set()
    player_start = None
    enemy_cells: list[tuple[int, int]] = []

    for y, row in enumerate(lines):
        if len(row) != GRID_W:
            raise ValueError("one_enemy_duel level row width mismatch")
        for x, ch in enumerate(row):
            if ch in {"=", "#"}:
                walls.add((x, y))
            if ch == ";":
                spikes.add((x, y))
            if ch == "@":
                player_start = (x, y)
            if ch in {"%", "&"}:
                enemy_cells.append((x, y))

    if player_start is None or not enemy_cells:
        raise ValueError("one_enemy_duel level missing player/enemy")

    min_x = min(x for x, _ in enemy_cells)
    max_x = max(x for x, _ in enemy_cells)
    min_y = min(y for _, y in enemy_cells)
    max_y = max(y for _, y in enemy_cells)
    enemy_start = ((min_x + max_x) // 2, (min_y + max_y) // 2)

    rules = dict(LEVEL_RULES[level_idx])
    return {
        "level_idx": level_idx,
        "lines": list(lines),
        "walls": walls,
        "spikes": spikes,
        "player_start": player_start,
        "enemy_start": enemy_start,
        "enemy_max_hp": int(rules["hp"]),
        "rules": rules,
    }


def _build_levels() -> list[Level]:
    levels: list[Level] = []
    for idx in range(len(LEVEL_GRIDS)):
        model = _parse_level_model(idx)
        board = np.full((GRID_H, GRID_W), COLOR_HUD_BG, dtype=np.int8)
        sprites = [Sprite(board, name="board", x=0, y=0, layer=0, tags=["board"], collidable=False)]
        levels.append(
            Level(
                name=f"OneEnemyDuel L{idx + 1}",
                grid_size=(GRID_W, GRID_H),
                sprites=sprites,
                data={
                    "model": {
                        "level_idx": model["level_idx"],
                        "lines": model["lines"],
                        "walls": sorted((int(x), int(y)) for (x, y) in model["walls"]),
                        "spikes": sorted((int(x), int(y)) for (x, y) in model["spikes"]),
                        "player_start": tuple(int(v) for v in model["player_start"]),
                        "enemy_start": tuple(int(v) for v in model["enemy_start"]),
                        "enemy_max_hp": int(model["enemy_max_hp"]),
                        "rules": dict(model["rules"]),
                    }
                },
            )
        )
    return levels


def _deserialize_model(level: Level) -> dict[str, Any]:
    raw = dict(level.get_data("model") or {})
    return {
        "level_idx": int(raw["level_idx"]),
        "lines": list(raw["lines"]),
        "walls": {tuple(int(v) for v in p) for p in (raw.get("walls") or [])},
        "spikes": {tuple(int(v) for v in p) for p in (raw.get("spikes") or [])},
        "player_start": tuple(int(v) for v in raw["player_start"]),
        "enemy_start": tuple(int(v) for v in raw["enemy_start"]),
        "enemy_max_hp": int(raw["enemy_max_hp"]),
        "rules": dict(raw["rules"]),
    }


def initial_search_state_from_model(model: dict[str, Any]) -> dict[str, Any]:
    return _empty_state(model)


def _danger_now(_model: dict[str, Any], state: dict[str, Any]) -> bool:
    if int(state["enemy_stun"]) > 0:
        return False
    attack = state["enemy_attack"]
    if not attack:
        return False
    if str(attack.get("phase")) != "execute":
        return False
    return str(attack.get("type")) in {"melee", "cleave", "dash"}


def _best_attack_dir(model: dict[str, Any], state: dict[str, Any]) -> tuple[int, int] | None:
    px, py = int(state["player_x"]), int(state["player_y"])
    enemy_tiles = _enemy_cells(model, state)
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if _slash_cells(px, py, d).intersection(enemy_tiles):
            return d
    return None


def choose_solver_action(model: dict[str, Any], state: dict[str, Any]) -> DuelAction:
    spikes_level = bool(model["rules"].get("spikes"))

    if bool(state["player_guard"]):
        if int(state["enemy_stun"]) > 0:
            return DuelAction(int(GameAction.ACTION5.value))
        if spikes_level and int(state["enemy_hp"]) <= 1:
            return DuelAction(int(GameAction.ACTION5.value))
        if spikes_level:
            px, py = int(state["player_x"]), int(state["player_y"])
            ex, ey = int(state["enemy_x"]), int(state["enemy_y"])
            if abs(px - ex) + abs(py - ey) > 5:
                direction = _snap_dir((px, py), (ex, ey), tuple(state["player_facing"]))
                move_order = [
                    (direction[0], direction[1]),
                    (direction[1], direction[0]),
                    (-direction[1], -direction[0]),
                ]
                for mdx, mdy in move_order:
                    if (mdx, mdy) == (0, 0):
                        continue
                    aid = None
                    if (mdx, mdy) == (0, -1):
                        aid = int(GameAction.ACTION1.value)
                    elif (mdx, mdy) == (0, 1):
                        aid = int(GameAction.ACTION2.value)
                    elif (mdx, mdy) == (-1, 0):
                        aid = int(GameAction.ACTION3.value)
                    elif (mdx, mdy) == (1, 0):
                        aid = int(GameAction.ACTION4.value)
                    if aid is None:
                        continue
                    nx, ny = _compute_player_move(model, state, mdx, mdy)
                    if (nx, ny) != (px, py):
                        return DuelAction(aid)
            return DuelAction(int(GameAction.ACTION6.value), px, py)
        if not _danger_now(model, state):
            close_projectile = False
            p_cells = _player_cells(state)
            for x, y, _dx, _dy in state["projectiles"]:
                if (x, y) in p_cells:
                    close_projectile = True
                    break
            if not close_projectile:
                return DuelAction(int(GameAction.ACTION5.value))

    if _danger_now(model, state) and not bool(state["player_guard"]):
        return DuelAction(int(GameAction.ACTION5.value))

    attack_dir = _best_attack_dir(model, state)
    if attack_dir is not None and not bool(state["player_guard"]):
        px, py = int(state["player_x"]), int(state["player_y"])
        return DuelAction(int(GameAction.ACTION6.value), px + attack_dir[0] * 8, py + attack_dir[1] * 8)

    if spikes_level and int(state["enemy_hp"]) > 1 and not bool(state["player_guard"]):
        return DuelAction(int(GameAction.ACTION5.value))

    if bool(state["player_guard"]):
        return DuelAction(int(GameAction.ACTION5.value))

    px, py = int(state["player_x"]), int(state["player_y"])
    ex, ey = int(state["enemy_x"]), int(state["enemy_y"])
    direction = _snap_dir((px, py), (ex, ey), tuple(state["player_facing"]))
    candidates: list[tuple[int, int]] = []
    if direction == (1, 0):
        candidates = [
            (int(GameAction.ACTION4.value), 1, 0),
            (int(GameAction.ACTION1.value), 0, -1),
            (int(GameAction.ACTION2.value), 0, 1),
            (int(GameAction.ACTION3.value), -1, 0),
        ]
    elif direction == (-1, 0):
        candidates = [
            (int(GameAction.ACTION3.value), -1, 0),
            (int(GameAction.ACTION1.value), 0, -1),
            (int(GameAction.ACTION2.value), 0, 1),
            (int(GameAction.ACTION4.value), 1, 0),
        ]
    elif direction == (0, 1):
        candidates = [
            (int(GameAction.ACTION2.value), 0, 1),
            (int(GameAction.ACTION3.value), -1, 0),
            (int(GameAction.ACTION4.value), 1, 0),
            (int(GameAction.ACTION1.value), 0, -1),
        ]
    else:
        candidates = [
            (int(GameAction.ACTION1.value), 0, -1),
            (int(GameAction.ACTION3.value), -1, 0),
            (int(GameAction.ACTION4.value), 1, 0),
            (int(GameAction.ACTION2.value), 0, 1),
        ]

    for action_id, dx, dy in candidates:
        nx, ny = _compute_player_move(model, state, dx, dy)
        if (nx, ny) == (px, py):
            continue
        return DuelAction(action_id)

    return DuelAction(int(GameAction.ACTION6.value), px, py)


def build_solver_program(model: dict[str, Any], max_steps: int | None = None) -> list[tuple[int, dict[str, int]]]:
    state = initial_search_state_from_model(model)
    budget = int(max_steps or (int(model["rules"]["timer"]) * 2))
    program: list[tuple[int, dict[str, int]]] = []
    for _ in range(budget):
        action = choose_solver_action(model, state)
        payload = {}
        if action.action_id == int(GameAction.ACTION6.value):
            payload = {"x": int(action.click_x or 0), "y": int(action.click_y or 0)}
        program.append((int(action.action_id), payload))
        state, won, lost = apply_action_transition(model, state, action)
        if won:
            return program
        if lost:
            break
    raise RuntimeError("one_enemy_duel solver could not produce a winning plan within budget")


class OneEnemyDuel(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = _build_levels()
        camera = Camera(width=GRID_W, height=GRID_H, background=COLOR_HUD_BG)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )
        self._model: dict[str, Any] = {}
        self._duel_state: dict[str, Any] = {}
        self._board: Sprite | None = None

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level)
        self._duel_state = _empty_state(self._model)
        boards = level.get_sprites_by_name("board")
        self._board = boards[0] if boards else None
        self._render()

    def _action_to_duel_action(self) -> DuelAction:
        action_id = int(self.action.id.value)
        if action_id != int(GameAction.ACTION6.value):
            return DuelAction(action_id)

        data = self.action.data if isinstance(self.action.data, dict) else {}
        try:
            raw_x = int(data.get("x", 0))
            raw_y = int(data.get("y", 0))
        except (TypeError, ValueError):
            return DuelAction(action_id)

        if 0 <= raw_x < GRID_W and 0 <= raw_y < GRID_H:
            return DuelAction(action_id, raw_x, raw_y)

        grid_pos = self.camera.display_to_grid(raw_x, raw_y)
        if grid_pos is None:
            return DuelAction(action_id)
        return DuelAction(action_id, int(grid_pos[0]), int(grid_pos[1]))

    def _render(self) -> None:
        if self._board is None:
            return

        state = self._duel_state
        model = self._model
        rules = model["rules"]
        grid = np.full((GRID_H, GRID_W), COLOR_HUD_BG, dtype=np.int8)

        for y in range(GRID_H):
            for x in range(GRID_W):
                if y in (0, 3, GRID_H - 1) or x in (0, GRID_W - 1):
                    grid[y, x] = COLOR_BORDER
                elif y in (1, 2):
                    grid[y, x] = COLOR_HUD_BG
                else:
                    grid[y, x] = COLOR_FLOOR

        for x in range(1, 4):
            grid[1, x] = COLOR_HP if x <= int(state["player_hp"]) else COLOR_HUD_BG
        for x in range(5, 21):
            steps_per = max(1, (int(rules["timer"]) + 15) // 16)
            remaining_segments = max(0, min(16, (int(state["timer"]) + steps_per - 1) // steps_per))
            seg_idx = x - 5
            grid[1, x] = COLOR_TIME_FILL if seg_idx < remaining_segments else COLOR_TIME_EMPTY

        for x, y in model["walls"]:
            grid[y, x] = COLOR_WALL if y > 3 else COLOR_BORDER

        spike_mode = _spike_state(model, int(state["spike_tick"]))
        for x, y in model["spikes"]:
            if spike_mode == "warn":
                grid[y, x] = COLOR_WARN
            elif spike_mode == "active":
                grid[y, x] = COLOR_DANGER
            else:
                grid[y, x] = COLOR_TIME_EMPTY

        for x, y in state.get("enemy_warn", []):
            if _walkable_arena(model, x, y):
                grid[y, x] = COLOR_WARN
        for x, y in state.get("enemy_danger", []):
            if _walkable_arena(model, x, y):
                grid[y, x] = COLOR_DANGER
        for x, y in state.get("player_slash", []):
            if _walkable_arena(model, x, y):
                grid[y, x] = COLOR_DANGER

        for x, y, _dx, _dy in state.get("projectiles", []):
            if _walkable_arena(model, int(x), int(y)):
                grid[int(y), int(x)] = COLOR_SPARK

        if int(state["enemy_stun"]) > 0:
            ex, ey = int(state["enemy_x"]), int(state["enemy_y"])
            for sx, sy in ((ex + 1, ey), (ex - 1, ey), (ex, ey + 1), (ex, ey - 1)):
                if _walkable_arena(model, sx, sy):
                    grid[sy, sx] = COLOR_SPARK

        enemy_size = _enemy_stage(int(model["enemy_max_hp"]), int(state["enemy_hp"]))
        for x, y in _enemy_cells(model, state):
            if not _walkable_arena(model, x, y):
                continue
            grid[y, x] = COLOR_ENEMY_OUTLINE
        enemy_body = _enemy_body_cells(model, state)
        body_color = COLOR_WARN if (int(state["enemy_flash"]) % 2 == 1) else COLOR_ENEMY_BODY
        for x, y in enemy_body:
            if _walkable_arena(model, x, y):
                grid[y, x] = body_color
        if enemy_size == 1:
            ex, ey = int(state["enemy_x"]), int(state["enemy_y"])
            if _walkable_arena(model, ex, ey):
                grid[ey, ex] = body_color

        outline_color = COLOR_PLAYER_GUARD if bool(state["player_guard"]) else COLOR_PLAYER_OUTLINE
        if int(state["player_flash"]) > 0:
            outline_color = COLOR_DANGER if int(state["player_flash"]) % 2 == 1 else COLOR_PLAYER_OUTLINE
        px, py = int(state["player_x"]), int(state["player_y"])
        for y in range(py - 1, py + 2):
            for x in range(px - 1, px + 2):
                if _walkable_arena(model, x, y):
                    grid[y, x] = outline_color
        if _walkable_arena(model, px, py):
            grid[py, px] = COLOR_PLAYER_BODY

        if int(state["fail_flash"]) > 0:
            for y in range(4, 19):
                for x in range(1, GRID_W - 1):
                    if (x, y) not in model["walls"]:
                        grid[y, x] = COLOR_DANGER

        self._board.pixels = grid

    def step(self) -> None:
        if getattr(self.action.id, "value", self.action.id) == 0 and self.level_index > 0:
            # arcengine's perform_action loop runs step() once after
            # handle_reset; without this guard the sim would advance a
            # tick, so mid-play RESET on any level entered via
            # next_level() would land one tick past the frame the
            # client saw on arrival. Level 0 keeps the legacy tick to
            # preserve env.reset()'s observation and the DSL trace.
            self.complete_action()
            return

        action = self._action_to_duel_action()
        self._duel_state, won, lost = apply_action_transition(self._model, self._duel_state, action)
        if won:
            self.next_level()
        elif lost:
            self.lose()
        self._render()
        self.complete_action()


__all__ = [
    "OneEnemyDuel",
    "_deserialize_model",
    "apply_action_transition",
    "build_solver_program",
    "initial_search_state_from_model",
]
