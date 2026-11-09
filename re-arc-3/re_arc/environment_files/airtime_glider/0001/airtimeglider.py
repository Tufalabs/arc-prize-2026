from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "airtime_glider-0001"

WIDTH = 32
HEIGHT = 15
LANES = 7
LANE_HEIGHT = 2
PLAYER_X = 6
TIMEBAR_STEPS = 112
TIMEBAR_CELLS = 28
STEPS_PER_CELL = 4
MAX_HEALTH = 3
SHIELD_STEPS = 24

COLOR_SKY = 0
COLOR_STRIPE = 1
COLOR_GLIDER_BODY = 2
COLOR_GLIDER_NOSE = 3
COLOR_GLIDER_TILT = 4
COLOR_BALLOON = 5
COLOR_TOWER = 6
COLOR_BIRD_A = 7
COLOR_BIRD_B = 8
COLOR_WIND_UP = 9
COLOR_WIND_DOWN = 10
COLOR_SHIELD = 11
COLOR_GLIDER_SHIELDED = 12
COLOR_TIME_REMAIN = 13
COLOR_TIME_ELAPSED = 14
COLOR_CRASH = 15

ACTION_UP = int(GameAction.ACTION1.value)
ACTION_DOWN = int(GameAction.ACTION2.value)
ACTION_IDLE = int(GameAction.ACTION5.value)

ENTITY_BALLOON = "balloon"
ENTITY_TOWER = "tower"
ENTITY_BIRD = "bird"
ENTITY_WIND = "wind"
ENTITY_SHIELD = "shield"


@dataclass(frozen=True)
class SpawnEvent:
    tick: int
    kind: str
    args: tuple[int, ...]


@dataclass(frozen=True)
class LevelModel:
    initial_lane: int
    entities: tuple[tuple, ...]
    spawns: tuple[SpawnEvent, ...]


def _rows_for_lane(lane: int) -> tuple[int, int]:
    y = 1 + lane * LANE_HEIGHT
    return y, y + 1


def _lane_for_row(row: int) -> int:
    return max(0, min(LANES - 1, (int(row) - 1) // LANE_HEIGHT))


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _entity_rect(entity: tuple) -> tuple[int, int, int, int]:
    kind = str(entity[0])
    if kind == ENTITY_BALLOON:
        _, x, lane = entity
        y0, _ = _rows_for_lane(int(lane))
        return int(x), y0, 2, 2
    if kind == ENTITY_BIRD:
        _, x, _la, _lb, current_lane, _counter = entity
        y0, _ = _rows_for_lane(int(current_lane))
        return int(x), y0, 2, 2
    if kind == ENTITY_TOWER:
        _, x, lane_min, lane_max = entity
        y0 = 1 + int(lane_min) * LANE_HEIGHT
        h = (int(lane_max) - int(lane_min) + 1) * LANE_HEIGHT
        return int(x), y0, 2, h
    if kind == ENTITY_WIND:
        _, x, lane, _direction = entity
        y0, _ = _rows_for_lane(int(lane))
        return int(x), y0, 1, 2
    if kind == ENTITY_SHIELD:
        _, x, lane = entity
        y0, _ = _rows_for_lane(int(lane))
        return int(x), y0, 1, 2
    raise ValueError(f"Unknown entity type: {kind}")


def _player_rect(lane: int) -> tuple[int, int, int, int]:
    y0, _ = _rows_for_lane(int(lane))
    return PLAYER_X, y0, 2, 2


def _parse_initial_grid(lines: list[str]) -> tuple[int, tuple[tuple, ...]]:
    # Skip UI row for parsing world objects.
    world = [list(row) for row in lines[1:]]
    if len(world) != 14:
        raise ValueError("Expected 14 world rows after UI row.")

    seen: set[tuple[int, int]] = set()
    entities: list[tuple] = []
    player_lane = 3

    for wy in range(14):
        for x in range(WIDTH):
            if (x, wy) in seen:
                continue
            ch = world[wy][x]
            gy = wy + 1

            if ch == "]" and x + 1 < WIDTH and world[wy][x + 1] == ">":
                lane = _lane_for_row(gy)
                player_lane = lane
                if wy + 1 < 14 and world[wy + 1][x] == "]" and world[wy + 1][x + 1] == ">":
                    seen.update({(x, wy), (x + 1, wy), (x, wy + 1), (x + 1, wy + 1)})
                continue

            if (
                ch == "0"
                and x + 1 < WIDTH
                and wy + 1 < 14
                and (world[wy][x + 1] == "0" and world[wy + 1][x] == "0" and world[wy + 1][x + 1] == "0")
            ):
                entities.append((ENTITY_BALLOON, x, _lane_for_row(gy)))
                seen.update({(x, wy), (x + 1, wy), (x, wy + 1), (x + 1, wy + 1)})
                continue

            if (
                ch in {"x", "X"}
                and x + 1 < WIDTH
                and wy + 1 < 14
                and (
                    world[wy][x + 1] in {"x", "X"}
                    and world[wy + 1][x] in {"x", "X"}
                    and world[wy + 1][x + 1] in {"x", "X"}
                )
            ):
                lane = _lane_for_row(gy)
                if lane == 0:
                    lane_a, lane_b = 0, 1
                elif lane == LANES - 1:
                    lane_a, lane_b = LANES - 2, LANES - 1
                else:
                    lane_a, lane_b = lane, lane + 1
                entities.append((ENTITY_BIRD, x, lane_a, lane_b, lane, 0))
                seen.update({(x, wy), (x + 1, wy), (x, wy + 1), (x + 1, wy + 1)})
                continue

            if ch in {"^", "v"} and wy + 1 < 14 and world[wy + 1][x] == ch:
                direction = -1 if ch == "^" else 1
                entities.append((ENTITY_WIND, x, _lane_for_row(gy), direction))
                seen.update({(x, wy), (x, wy + 1)})
                continue

            if ch == "*" and wy + 1 < 14 and world[wy + 1][x] == "*":
                entities.append((ENTITY_SHIELD, x, _lane_for_row(gy)))
                seen.update({(x, wy), (x, wy + 1)})
                continue

    # Towers are parsed by contiguous '|' segments over two adjacent columns.
    tower_cols: list[int] = []
    for x in range(WIDTH - 1):
        has_pair = any(world[wy][x] == "|" and world[wy][x + 1] == "|" for wy in range(14))
        if has_pair:
            tower_cols.append(x)

    for x in tower_cols:
        ys = [wy + 1 for wy in range(14) if world[wy][x] == "|" and world[wy][x + 1] == "|"]
        if not ys:
            continue
        lane_min = _lane_for_row(min(ys))
        lane_max = _lane_for_row(max(ys))
        entities.append((ENTITY_TOWER, x, lane_min, lane_max))

    return player_lane, tuple(sorted(entities, key=lambda item: (str(item[0]), int(item[1]))))


def _spawn_events(raw: list[tuple[int, str, tuple[int, ...]]]) -> tuple[SpawnEvent, ...]:
    return tuple(SpawnEvent(int(t), str(kind), tuple(int(v) for v in args)) for t, kind, args in raw)


def _event_to_entity(event: SpawnEvent) -> tuple:
    if event.kind == ENTITY_BALLOON:
        lane = int(event.args[0])
        return (ENTITY_BALLOON, 30, lane)
    if event.kind == ENTITY_TOWER:
        lane_min, lane_max = int(event.args[0]), int(event.args[1])
        return (ENTITY_TOWER, 30, lane_min, lane_max)
    if event.kind == ENTITY_BIRD:
        lane_a, lane_b, start_lane = (int(event.args[0]), int(event.args[1]), int(event.args[2]))
        return (ENTITY_BIRD, 30, lane_a, lane_b, start_lane, 0)
    if event.kind == ENTITY_WIND:
        lane, direction = int(event.args[0]), int(event.args[1])
        return (ENTITY_WIND, 31, lane, direction)
    if event.kind == ENTITY_SHIELD:
        lane = int(event.args[0])
        return (ENTITY_SHIELD, 31, lane)
    raise ValueError(f"Unsupported spawn kind: {event.kind}")


def _scroll_entities(entities: tuple[tuple, ...]) -> tuple[tuple, ...]:
    out: list[tuple] = []
    for entity in entities:
        kind = str(entity[0])
        if kind == ENTITY_BALLOON:
            _, x, lane = entity
            nx = int(x) - 1
            if nx + 1 >= 0:
                out.append((ENTITY_BALLOON, nx, int(lane)))
        elif kind == ENTITY_TOWER:
            _, x, lane_min, lane_max = entity
            nx = int(x) - 1
            if nx + 1 >= 0:
                out.append((ENTITY_TOWER, nx, int(lane_min), int(lane_max)))
        elif kind == ENTITY_BIRD:
            _, x, lane_a, lane_b, current_lane, counter = entity
            nx = int(x) - 1
            if nx + 1 >= 0:
                out.append((ENTITY_BIRD, nx, int(lane_a), int(lane_b), int(current_lane), int(counter)))
        elif kind == ENTITY_WIND:
            _, x, lane, direction = entity
            nx = int(x) - 1
            if nx >= 0:
                out.append((ENTITY_WIND, nx, int(lane), int(direction)))
        elif kind == ENTITY_SHIELD:
            _, x, lane = entity
            nx = int(x) - 1
            if nx >= 0:
                out.append((ENTITY_SHIELD, nx, int(lane)))
    return tuple(out)


def _animate_entities(entities: tuple[tuple, ...]) -> tuple[tuple, ...]:
    out: list[tuple] = []
    for entity in entities:
        if str(entity[0]) != ENTITY_BIRD:
            out.append(entity)
            continue
        _, x, lane_a, lane_b, current_lane, counter = entity
        next_counter = int(counter) + 1
        lane_now = int(current_lane)
        if next_counter >= 6:
            next_counter = 0
            lane_now = int(lane_b) if lane_now == int(lane_a) else int(lane_a)
        out.append((ENTITY_BIRD, int(x), int(lane_a), int(lane_b), lane_now, next_counter))
    return tuple(out)


def _apply_spawns(entities: tuple[tuple, ...], spawns: tuple[SpawnEvent, ...], tick: int) -> tuple[tuple, ...]:
    out = list(entities)
    for event in spawns:
        if int(event.tick) == int(tick):
            out.append(_event_to_entity(event))
    return tuple(out)


def _collect_shields(entities: tuple[tuple, ...], lane: int, shield_timer: int) -> tuple[tuple[tuple, ...], int]:
    player = _player_rect(lane)
    out: list[tuple] = []
    timer = int(shield_timer)
    for entity in entities:
        if str(entity[0]) == ENTITY_SHIELD and _rects_overlap(player, _entity_rect(entity)):
            timer = SHIELD_STEPS
            continue
        out.append(entity)
    return tuple(out), timer


def _apply_collisions(
    entities: tuple[tuple, ...], lane: int, health: int, shield_timer: int, damage_flash: int
) -> tuple[tuple[tuple, ...], int, int, int, bool]:
    player = _player_rect(lane)
    hazard_hits: list[tuple] = []
    for entity in entities:
        kind = str(entity[0])
        if kind in {ENTITY_BALLOON, ENTITY_TOWER, ENTITY_BIRD} and _rects_overlap(player, _entity_rect(entity)):
            hazard_hits.append(entity)

    if not hazard_hits:
        return entities, int(health), int(shield_timer), int(damage_flash), False

    remove_non_tower = set(entity for entity in hazard_hits if str(entity[0]) != ENTITY_TOWER)
    out = tuple(entity for entity in entities if entity not in remove_non_tower)

    if int(shield_timer) > 0:
        return out, int(health), 0, int(damage_flash), True

    next_health = max(0, int(health) - 1)
    return out, next_health, int(shield_timer), 2, True


def _wind_push_from_overlap(entities: tuple[tuple, ...], lane: int) -> int:
    player = _player_rect(lane)
    push = 0
    for entity in entities:
        if str(entity[0]) != ENTITY_WIND:
            continue
        if _rects_overlap(player, _entity_rect(entity)):
            push = int(entity[3])
    return push


def _simulate_tick(
    model: LevelModel, state: tuple[int, int, int, int, int, tuple[tuple, ...]], action_id: int
) -> tuple[tuple[int, int, int, int, int, tuple[tuple, ...]] | None, bool]:
    tick, lane, health, shield_timer, pending_wind, entities = state

    lane_now = int(lane)
    health_now = int(health)
    shield_now = max(0, int(shield_timer) - 1)

    # 1) pending wind from previous step
    if int(pending_wind) != 0:
        target = lane_now + int(pending_wind)
        if target < 0 or target >= LANES:
            health_now -= 1
            if health_now <= 0:
                return None, False
        else:
            lane_now = target

    # 2) player action lane change
    if int(action_id) == ACTION_UP and lane_now > 0:
        lane_now -= 1
    elif int(action_id) == ACTION_DOWN and lane_now < LANES - 1:
        lane_now += 1

    # 3) world scroll + spawn + internal movement
    ents = _scroll_entities(tuple(entities))
    ents = _apply_spawns(ents, model.spawns, int(tick))
    ents = _animate_entities(ents)

    # 4) shield collection
    ents, shield_now = _collect_shields(ents, lane_now, shield_now)

    # 5) hazard collisions
    ents, health_now, shield_now, _damage, _hit = _apply_collisions(
        ents, lane_now, health_now, shield_now, damage_flash=0
    )
    if health_now <= 0:
        return None, False

    # 6) set next pending wind from current overlap
    next_pending = _wind_push_from_overlap(ents, lane_now)

    next_tick = int(tick) + 1
    won = next_tick >= TIMEBAR_STEPS
    return (next_tick, lane_now, health_now, shield_now, next_pending, tuple(ents)), won


def _serialize_model(model: LevelModel) -> dict:
    return {
        "initial_lane": int(model.initial_lane),
        "entities": [list(entity) for entity in model.entities],
        "spawns": [{"tick": int(ev.tick), "kind": ev.kind, "args": list(ev.args)} for ev in model.spawns],
    }


def _deserialize_model(level: Level) -> LevelModel:
    raw = dict(level.get_data("model") or {})
    initial_lane = int(raw.get("initial_lane", 3))
    entities = tuple(tuple(entity) for entity in raw.get("entities", []))
    spawns = tuple(
        SpawnEvent(int(item["tick"]), str(item["kind"]), tuple(int(v) for v in item.get("args", [])))
        for item in raw.get("spawns", [])
    )
    return LevelModel(initial_lane=initial_lane, entities=entities, spawns=spawns)


def initial_search_state_from_model(model: LevelModel) -> tuple[int, int, int, int, int, tuple[tuple, ...]]:
    return (0, int(model.initial_lane), MAX_HEALTH, 0, 0, tuple(model.entities))


def apply_action_transition(
    model: LevelModel, state: tuple[int, int, int, int, int, tuple[tuple, ...]], action_id: int
) -> tuple[tuple[int, int, int, int, int, tuple[tuple, ...]] | None, bool]:
    if int(action_id) not in {ACTION_UP, ACTION_DOWN, ACTION_IDLE}:
        return None, False
    return _simulate_tick(model, state, int(action_id))


def _level_from_ascii(lines: list[str], spawns: list[tuple[int, str, tuple[int, ...]]], idx: int) -> Level:
    initial_lane, entities = _parse_initial_grid(lines)
    model = LevelModel(initial_lane=initial_lane, entities=entities, spawns=_spawn_events(spawns))

    sprite = Sprite(
        np.full((HEIGHT, WIDTH), COLOR_SKY, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        layer=0,
        tags=["board"],
        collidable=False,
    )
    return Level(
        name=f"Airtime Glider {idx + 1}",
        grid_size=(WIDTH, HEIGHT),
        sprites=[sprite],
        data={"level_idx": int(idx), "model": _serialize_model(model), "time_limit": TIMEBAR_STEPS},
    )


def _make_levels() -> list[Level]:
    level_lines = [
        [
            "OOO|============================",
            "................................",
            "................................",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            "........................00......",
            "........................00......",
            ",,,,,,]>,,,,,,,,,,00,,,,,,,,,,,,",
            ",,,,,,]>,,,,,,,,,,00,,,,,,,,,,,,",
            "................................",
            "................................",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            "................................",
            "................................",
        ],
        [
            "OOO|============================",
            "..............00................",
            "..............00................",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            "....................||....00....",
            "....................||....00....",
            ",,,,,,]>,,,,,,,,,,,,||,,,,,,,,,,",
            ",,,,,,]>,,,,,,,,,,,,||,,,,,,,,,,",
            "....................||..........",
            "....................||..........",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            "................................",
            "................................",
        ],
        [
            "OOO|============================",
            "................................",
            "................................",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            "..................00............",
            "..................00............",
            ",,,,,,]>,,,,,,,,,,,,,,,,,,,,,,,,",
            ",,,,,,]>,,,,,,,,,,,,,,,,,,,,,,,,",
            "........................xx......",
            "........................xx......",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,",
            "................................",
            "................................",
        ],
        [
            "OOO|============================",
            "................................",
            "................................",
            ",,,,,,,,,,,,,,,,,,,,,v,,,,,,,,,,",
            ",,,,,,,,,,,,,,,,,,,,,v,,,,,,,,,,",
            "................................",
            "................................",
            ",,,,,,]>,,,,,,,,,,00,,,,,,,,,,,,",
            ",,,,,,]>,,,,,,,,,,00,,,,,,,,,,,,",
            "................................",
            "................................",
            ",,,,,,,,,,,,,,,,^,,,,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,,^,,,,,,,,,,,,,,,",
            "................................",
            "................................",
        ],
        [
            "OOO|============================",
            "........................||......",
            "........................||......",
            ",,,,,,,,,,,,,,,,,,,,,,,,||,,,,,,",
            ",,,,,,,,,,,,,,,,,,,,,,,,||,,,,,,",
            "........................||......",
            "........................||......",
            ",,,,,,]>,,,,,,*,,,,,,,xx,,,,,,,,",
            ",,,,,,]>,,,,,,*,,,,,,,xx,,,,,,,,",
            "........................||......",
            "........................||......",
            ",,,,,,,,,,,,,,,,,,,,,,,,||,,,,,,",
            ",,,,,,,,,,,,,,,,,,,,,,,,||,,,,,,",
            "........................||......",
            "........................||......",
        ],
        [
            "OOO|============================",
            "..................||............",
            "..................||............",
            ",,,,,,,,,,,,,,,,,,||,,,,,,,,,,,,",
            ",,,,,,,,,,,,,,,,,,||,,,,,,,,,,,,",
            "..................||...v........",
            "..................||...v........",
            ",,,,,,]>,,,,,,,,,,||,,,,,,||,,,,",
            ",,,,,,]>,,,,,,,,,,||,,,,,,||,,,,",
            "............*.............||....",
            "............*.............||....",
            ",,,,,,,,,,,,,,,,^,,,,,xx,,||,,,,",
            ",,,,,,,,,,,,,,,,^,,,,,xx,,||,,,,",
            "..........................||....",
            "..........................||....",
        ],
    ]

    level_spawns: list[list[tuple[int, str, tuple[int, ...]]]] = [
        [
            (8, ENTITY_BALLOON, (3,)),
            (22, ENTITY_BALLOON, (2,)),
            (36, ENTITY_BALLOON, (4,)),
            (50, ENTITY_BALLOON, (1,)),
            (64, ENTITY_BALLOON, (5,)),
            (78, ENTITY_BALLOON, (3,)),
            (88, ENTITY_BALLOON, (2,)),
        ],
        [
            (16, ENTITY_TOWER, (2, 4)),
            (48, ENTITY_TOWER, (1, 3)),
            (72, ENTITY_TOWER, (3, 6)),
            (10, ENTITY_BALLOON, (5,)),
            (26, ENTITY_BALLOON, (0,)),
            (34, ENTITY_BALLOON, (6,)),
            (42, ENTITY_BALLOON, (2,)),
            (58, ENTITY_BALLOON, (4,)),
            (66, ENTITY_BALLOON, (1,)),
            (82, ENTITY_BALLOON, (5,)),
        ],
        [
            (12, ENTITY_BIRD, (2, 3, 2)),
            (36, ENTITY_BIRD, (4, 5, 5)),
            (60, ENTITY_BIRD, (1, 2, 1)),
            (76, ENTITY_BIRD, (5, 6, 6)),
            (20, ENTITY_BALLOON, (4,)),
            (28, ENTITY_BALLOON, (0,)),
            (44, ENTITY_BALLOON, (3,)),
            (52, ENTITY_BALLOON, (6,)),
            (68, ENTITY_BALLOON, (2,)),
            (84, ENTITY_BALLOON, (4,)),
            (48, ENTITY_TOWER, (0, 2)),
        ],
        [
            (10, ENTITY_WIND, (5, -1)),
            (22, ENTITY_WIND, (1, 1)),
            (38, ENTITY_WIND, (2, -1)),
            (54, ENTITY_WIND, (6, 1)),
            (70, ENTITY_WIND, (4, -1)),
            (82, ENTITY_WIND, (0, 1)),
            (28, ENTITY_BIRD, (3, 4, 3)),
            (64, ENTITY_BIRD, (1, 2, 2)),
            (16, ENTITY_BALLOON, (4,)),
            (32, ENTITY_BALLOON, (0,)),
            (48, ENTITY_BALLOON, (6,)),
            (60, ENTITY_BALLOON, (2,)),
            (76, ENTITY_BALLOON, (5,)),
            (44, ENTITY_TOWER, (2, 5)),
        ],
        [
            (8, ENTITY_SHIELD, (3,)),
            (52, ENTITY_SHIELD, (2,)),
            (18, ENTITY_TOWER, (0, 2)),
            (18, ENTITY_TOWER, (4, 6)),
            (60, ENTITY_TOWER, (1, 3)),
            (60, ENTITY_TOWER, (5, 6)),
            (26, ENTITY_BIRD, (3, 4, 3)),
            (72, ENTITY_BIRD, (2, 3, 2)),
            (34, ENTITY_WIND, (2, 1)),
            (80, ENTITY_WIND, (5, -1)),
            (12, ENTITY_BALLOON, (5,)),
            (40, ENTITY_BALLOON, (0,)),
            (88, ENTITY_BALLOON, (4,)),
        ],
        [
            (12, ENTITY_TOWER, (0, 3)),
            (28, ENTITY_TOWER, (3, 6)),
            (44, ENTITY_TOWER, (0, 2)),
            (60, ENTITY_TOWER, (4, 6)),
            (76, ENTITY_TOWER, (1, 4)),
            (20, ENTITY_BIRD, (5, 6, 6)),
            (36, ENTITY_BIRD, (2, 3, 2)),
            (56, ENTITY_BIRD, (4, 5, 4)),
            (72, ENTITY_BIRD, (1, 2, 1)),
            (84, ENTITY_BIRD, (3, 4, 4)),
            (16, ENTITY_WIND, (5, -1)),
            (24, ENTITY_WIND, (2, 1)),
            (40, ENTITY_WIND, (1, -1)),
            (52, ENTITY_WIND, (6, 1)),
            (68, ENTITY_WIND, (3, -1)),
            (80, ENTITY_WIND, (0, 1)),
            (8, ENTITY_SHIELD, (4,)),
            (48, ENTITY_SHIELD, (3,)),
            (32, ENTITY_BALLOON, (0,)),
            (64, ENTITY_BALLOON, (6,)),
            (88, ENTITY_BALLOON, (2,)),
        ],
    ]

    levels: list[Level] = []
    for idx, lines in enumerate(level_lines):
        levels.append(_level_from_ascii(lines, level_spawns[idx], idx))
    return levels


class AirtimeGlider(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._model: LevelModel | None = None
        self._entities: list[tuple] = []
        self._lane = 3
        self._pending_wind_push = 0
        self._health = MAX_HEALTH
        self._shield_timer = 0
        self._damage_flash = 0
        self._tilt_timer = 0
        self._crash_timer = 0
        self._tick = 0
        self._anim_phase = 0
        self._route_score = 0
        self._board_sprite: Sprite | None = None

        levels = _make_levels()
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=Camera(width=WIDTH, height=HEIGHT, background=COLOR_SKY),
            win_score=len(levels),
            available_actions=[1, 2, 5],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._model = _deserialize_model(level)
        self._board_sprite = level.get_sprites_by_name("board")[0]
        self._reset_level_runtime()

    def _reset_level_runtime(self) -> None:
        if self._model is None:
            return
        self._entities = list(self._model.entities)
        self._lane = int(self._model.initial_lane)
        self._pending_wind_push = 0
        self._health = MAX_HEALTH
        self._shield_timer = 0
        self._damage_flash = 0
        self._tilt_timer = 0
        self._crash_timer = 0
        self._tick = 0
        self._anim_phase = 0
        self._route_score = 0
        self._draw()

    def _draw(self) -> None:
        if self._board_sprite is None:
            return
        board = np.full((HEIGHT, WIDTH), COLOR_SKY, dtype=np.int8)

        for lane in range(LANES):
            if lane % 2 == 1:
                y0, y1 = _rows_for_lane(lane)
                board[y0, :] = COLOR_STRIPE
                board[y1, :] = COLOR_STRIPE

        for entity in self._entities:
            kind = str(entity[0])
            if kind == ENTITY_BALLOON:
                x, lane = int(entity[1]), int(entity[2])
                y0, y1 = _rows_for_lane(lane)
                if 0 <= x < WIDTH:
                    board[y0 : y1 + 1, x : min(WIDTH, x + 2)] = COLOR_BALLOON
            elif kind == ENTITY_TOWER:
                x, lane_min, lane_max = int(entity[1]), int(entity[2]), int(entity[3])
                y0 = 1 + lane_min * LANE_HEIGHT
                y1 = 1 + (lane_max + 1) * LANE_HEIGHT
                board[y0:y1, max(0, x) : min(WIDTH, x + 2)] = COLOR_TOWER
            elif kind == ENTITY_BIRD:
                x, current_lane = int(entity[1]), int(entity[4])
                y0, y1 = _rows_for_lane(current_lane)
                c = COLOR_BIRD_A if self._anim_phase == 0 else COLOR_BIRD_B
                board[y0 : y1 + 1, max(0, x) : min(WIDTH, x + 2)] = c
            elif kind == ENTITY_WIND:
                x, lane, direction = int(entity[1]), int(entity[2]), int(entity[3])
                y0, y1 = _rows_for_lane(lane)
                if self._anim_phase == 0:
                    c = COLOR_WIND_UP if direction < 0 else COLOR_WIND_DOWN
                else:
                    c = COLOR_GLIDER_TILT
                if 0 <= x < WIDTH:
                    board[y0 : y1 + 1, x] = c
            elif kind == ENTITY_SHIELD:
                x, lane = int(entity[1]), int(entity[2])
                y0, y1 = _rows_for_lane(lane)
                if 0 <= x < WIDTH:
                    board[y0 : y1 + 1, x] = COLOR_SHIELD

        py0, py1 = _rows_for_lane(self._lane)
        if self._crash_timer > 0:
            body_color = COLOR_CRASH
            nose_color = COLOR_CRASH
        elif self._damage_flash > 0:
            body_color = COLOR_GLIDER_TILT
            nose_color = COLOR_GLIDER_NOSE
        elif self._shield_timer > 0:
            body_color = COLOR_GLIDER_SHIELDED
            nose_color = COLOR_GLIDER_NOSE
        elif self._tilt_timer > 0:
            body_color = COLOR_GLIDER_TILT
            nose_color = COLOR_GLIDER_NOSE
        else:
            body_color = COLOR_GLIDER_BODY
            nose_color = COLOR_GLIDER_NOSE

        board[py0 : py1 + 1, PLAYER_X] = body_color
        board[py0 : py1 + 1, PLAYER_X + 1] = nose_color

        board[0, :] = COLOR_SKY
        for i in range(MAX_HEALTH):
            board[0, i] = COLOR_GLIDER_NOSE if i < self._health else COLOR_TIME_ELAPSED
        board[0, 3] = COLOR_TOWER

        elapsed_cells = min(TIMEBAR_CELLS, self._tick // STEPS_PER_CELL)
        for i in range(TIMEBAR_CELLS):
            board[0, 4 + i] = COLOR_TIME_ELAPSED if i < elapsed_cells else COLOR_TIME_REMAIN

        self._board_sprite.pixels = board

    def _apply_pending_wind(self) -> None:
        if self._pending_wind_push == 0:
            return
        target_lane = self._lane + int(self._pending_wind_push)
        if target_lane < 0 or target_lane >= LANES:
            self._handle_hazard_collision(collided=[])
            self._pending_wind_push = 0
            return
        self._lane = target_lane
        self._pending_wind_push = 0

    def _apply_action_lane_change(self) -> None:
        moved = False
        if self.action.id == GameAction.ACTION1 and self._lane > 0:
            self._lane -= 1
            moved = True
        elif self.action.id == GameAction.ACTION2 and self._lane < LANES - 1:
            self._lane += 1
            moved = True
        if moved:
            self._tilt_timer = 1

    def _apply_scroll_and_spawns(self) -> None:
        if self._model is None:
            return
        self._entities = list(_scroll_entities(tuple(self._entities)))
        self._entities = list(_apply_spawns(tuple(self._entities), self._model.spawns, self._tick))

    def _animate_entities(self) -> None:
        self._entities = list(_animate_entities(tuple(self._entities)))

    def _collect_pickups(self) -> None:
        ents, shield = _collect_shields(tuple(self._entities), self._lane, self._shield_timer)
        self._entities = list(ents)
        self._shield_timer = int(shield)

    def _handle_hazard_collision(self, collided: list[tuple]) -> None:
        remove_non_tower = {entity for entity in collided if str(entity[0]) != ENTITY_TOWER}
        if remove_non_tower:
            self._entities = [entity for entity in self._entities if entity not in remove_non_tower]

        if self._shield_timer > 0:
            self._shield_timer = 0
            return

        self._health = max(0, self._health - 1)
        self._damage_flash = 2
        if self._health <= 0:
            self._crash_timer = 3

    def _resolve_collisions(self) -> None:
        player = _player_rect(self._lane)
        collided: list[tuple] = []
        for entity in self._entities:
            if str(entity[0]) not in {ENTITY_BALLOON, ENTITY_TOWER, ENTITY_BIRD}:
                continue
            if _rects_overlap(player, _entity_rect(entity)):
                collided.append(entity)
        if collided:
            self._handle_hazard_collision(collided)

    def _update_pending_wind(self) -> None:
        self._pending_wind_push = int(_wind_push_from_overlap(tuple(self._entities), self._lane))

    def _advance_timebar(self) -> None:
        self._tick += 1
        self._route_score += 1
        if self._tick >= TIMEBAR_STEPS:
            self.next_level()

    def step(self) -> None:
        if self._model is None:
            self.complete_action()
            return

        if self.action.id == GameAction.RESET and self.level_index > 0:
            # arcengine's perform_action loop runs step() once with the RESET
            # action after handle_reset() has already restored the pristine
            # level state. Without this guard the simulation would advance a
            # tick, so mid-play RESET on a level entered via next_level()
            # (all levels except level 0) would land on a frame one tick
            # past the one the client saw on arrival. On level 0 we keep the
            # legacy behaviour: env.reset() is the only way in, and both
            # env.reset() and mid-play RESET go through step(RESET), so
            # skipping the tick here would change env.reset()'s observation
            # and invalidate recorded DSL traces.
            self.complete_action()
            return

        if self._crash_timer > 0:
            self._anim_phase = 1 - self._anim_phase
            self._crash_timer -= 1
            if self._crash_timer == 0:
                self.lose()
            self._draw()
            self.complete_action()
            return

        self._anim_phase = 1 - self._anim_phase

        if self._shield_timer > 0:
            self._shield_timer -= 1
        if self._damage_flash > 0:
            self._damage_flash -= 1
        if self._tilt_timer > 0:
            self._tilt_timer -= 1

        self._apply_pending_wind()
        if self._crash_timer == 0:
            self._apply_action_lane_change()

        self._apply_scroll_and_spawns()
        self._animate_entities()
        self._collect_pickups()
        self._resolve_collisions()
        self._update_pending_wind()

        if self._crash_timer == 0:
            self._advance_timebar()

        self._draw()
        self.complete_action()
