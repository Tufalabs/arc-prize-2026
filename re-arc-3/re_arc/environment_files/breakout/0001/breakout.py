from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "breakout-0001"

WIDTH = 32
HEIGHT = 24
UI_TIME_Y = 0
UI_LIVES_Y = 1
SEPARATOR_Y = 2
PLAY_MIN_Y = 3
PLAY_MAX_Y = 22
DRAIN_Y = 23

A_LEFT = int(GameAction.ACTION3.value)
A_RIGHT = int(GameAction.ACTION4.value)
A_SPACE = int(GameAction.ACTION5.value)

ACTION_BY_ID = {A_LEFT: GameAction.ACTION3, A_RIGHT: GameAction.ACTION4, A_SPACE: GameAction.ACTION5}

C_BG = 0
C_WALL = 1
C_PADDLE = 2
C_BALL_A = 3
C_FLASH = 4
C_BRICK_NORMAL = 5
C_BRICK_STRONG_2 = 6
C_BRICK_STRONG_1 = 7
C_BRICK_HEAVY_3 = 8
C_BRICK_HEAVY_2 = 9
C_BRICK_HEAVY_1 = 10
C_BRICK_UNBREAK = 11
C_TIME_HIGH = 12
C_TIME_MID = 13
C_TIME_LOW = 14
C_PULSE = 15

LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "paddle_width": 9,
        "ball_speed_substeps": 1,
        "max_time": 2000,
        "time_bonus_per_brick": 4,
        "rows": [
            "#|::::::::::::::::::::|........#",
            "#........................ooo...#",
            "################################",
            "#..............................#",
            "#..............................#",
            "#.............****.............#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............O...............#",
            "#..........=========...........#",
            "#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#",
        ],
    },
    {
        "name": "Level 2",
        "paddle_width": 9,
        "ball_speed_substeps": 1,
        "max_time": 2000,
        "time_bonus_per_brick": 4,
        "rows": [
            "#|::::::::::::::::::::|........#",
            "#........................ooo...#",
            "################################",
            "#..............................#",
            "#..............................#",
            "#...........********...........#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............O...............#",
            "#..........=========...........#",
            "#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#",
        ],
    },
    {
        "name": "Level 3",
        "paddle_width": 9,
        "ball_speed_substeps": 1,
        "max_time": 2000,
        "time_bonus_per_brick": 4,
        "rows": [
            "#|::::::::::::::::::::|........#",
            "#........................ooo...#",
            "################################",
            "#..............................#",
            "#..............................#",
            "#.........************.........#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............O...............#",
            "#..........=========...........#",
            "#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#",
        ],
    },
    {
        "name": "Level 4",
        "paddle_width": 9,
        "ball_speed_substeps": 1,
        "max_time": 2000,
        "time_bonus_per_brick": 4,
        "rows": [
            "#|::::::::::::::::::::|........#",
            "#........................ooo...#",
            "################################",
            "#..............................#",
            "#..............................#",
            "#.........**++++++**...........#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............O...............#",
            "#..........=========...........#",
            "#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#",
        ],
    },
    {
        "name": "Level 5",
        "paddle_width": 9,
        "ball_speed_substeps": 1,
        "max_time": 2000,
        "time_bonus_per_brick": 4,
        "rows": [
            "#|::::::::::::::::::::|........#",
            "#........................ooo...#",
            "################################",
            "#..............................#",
            "#..............................#",
            "#........**@@++++@@**..........#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............O...............#",
            "#..........=========...........#",
            "#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#",
        ],
    },
    {
        "name": "Level 6",
        "paddle_width": 9,
        "ball_speed_substeps": 1,
        "max_time": 2000,
        "time_bonus_per_brick": 4,
        "rows": [
            "#|::::::::::::::::::::|........#",
            "#........................ooo...#",
            "################################",
            "#..............................#",
            "#..............................#",
            "#.......**@@++^^++@@**.........#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............................#",
            "#..............O...............#",
            "#..........=========...........#",
            "#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#",
        ],
    },
]


class Brick:
    def __init__(self, brick_id: int, x: int, y: int, kind: str, hp: int) -> None:
        self.brick_id = int(brick_id)
        self.x = int(x)
        self.y = int(y)
        self.kind = str(kind)
        self.hp = int(hp)
        self.pending_damage = 0
        self.flash = 0
        self.removed = False


class Capsule:
    def __init__(self, cap_type: str, x: int, y: int, active: bool = False) -> None:
        self.cap_type = str(cap_type)
        self.x = int(x)
        self.y = int(y)
        self.active = bool(active)


class Bumper:
    def __init__(self, x: int, y: int, direction: int = 1) -> None:
        self.x = int(x)
        self.y = int(y)
        self.direction = int(direction)


def _full(color: int, width: int, height: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _brick_hp_for_kind(kind: str) -> int:
    if kind == "normal":
        return 1
    if kind == "strong":
        return 1
    if kind == "heavy":
        return 1
    if kind == "unbreakable":
        return 10**9
    if kind in {"expand", "shrink"}:
        return 1
    raise ValueError(f"Unsupported brick kind: {kind}")


def _brick_is_breakable(kind: str) -> bool:
    return kind in {"normal", "strong", "heavy", "expand", "shrink"}


def _brick_color(brick: Brick) -> int:
    if brick.flash > 0:
        return C_FLASH
    if brick.kind == "normal":
        return C_BRICK_NORMAL
    if brick.kind == "strong":
        return C_BRICK_STRONG_2 if brick.hp >= 2 else C_BRICK_STRONG_1
    if brick.kind == "heavy":
        if brick.hp >= 3:
            return C_BRICK_HEAVY_3
        if brick.hp == 2:
            return C_BRICK_HEAVY_2
        return C_BRICK_HEAVY_1
    if brick.kind == "unbreakable":
        return C_BRICK_UNBREAK
    if brick.kind == "expand":
        return C_TIME_HIGH
    if brick.kind == "shrink":
        return C_TIME_MID
    return C_BG


def _parse_level_rows(rows: list[str]) -> dict:
    if len(rows) != HEIGHT:
        raise ValueError(f"breakout level must have {HEIGHT} rows")
    for row in rows:
        if len(row) != WIDTH:
            raise ValueError(f"breakout level row has length {len(row)} != {WIDTH}")

    walls: set[tuple[int, int]] = set()
    drains: set[tuple[int, int]] = set()
    bricks: list[dict] = []
    bumpers: list[dict] = []
    ball_pos: tuple[int, int] | None = None
    paddle_x = 0
    paddle_y = 0
    paddle_width = 0

    token_map = {"*": "normal", "+": "strong", "@": "heavy", "%": "unbreakable", "^": "expand", "v": "shrink"}

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == "#" or ch == "|":
                walls.add((x, y))
            elif ch == "!":
                drains.add((x, y))
            elif ch == "O":
                ball_pos = (x, y)

        start = row.find("=")
        if start >= 0:
            end = start
            while end < WIDTH and row[end] == "=":
                end += 1
            paddle_x = start
            paddle_y = y
            paddle_width = end - start

        x = 0
        while x < WIDTH - 1:
            ch = row[x]
            if ch in token_map and row[x + 1] == ch:
                bricks.append({"x": x, "y": y, "kind": token_map[ch]})
                x += 2
                continue
            x += 1

    bumper_seen: set[tuple[int, int]] = set()
    for y in range(HEIGHT - 1):
        for x in range(WIDTH - 1):
            if (x, y) in bumper_seen:
                continue
            if rows[y][x] == "~" and rows[y][x + 1] == "~" and rows[y + 1][x] == "~" and rows[y + 1][x + 1] == "~":
                bumpers.append({"x": x, "y": y, "direction": 1})
                bumper_seen.update({(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)})

    if ball_pos is None:
        raise ValueError("breakout level must include initial ball position")
    if paddle_width <= 0:
        raise ValueError("breakout level must include initial paddle")

    return {
        "walls": sorted((int(x), int(y)) for x, y in walls),
        "drains": sorted((int(x), int(y)) for x, y in drains),
        "bricks": bricks,
        "bumpers": bumpers,
        "ball": ball_pos,
        "paddle": (paddle_x, paddle_y, paddle_width),
    }


class BreakoutModel:
    def __init__(self, level_data: dict):
        self.width = int(level_data["width"])
        self.height = int(level_data["height"])
        self.max_time = int(level_data["max_time"])
        self.time_bonus_per_brick = int(level_data["time_bonus_per_brick"])
        self.base_paddle_width = int(level_data["base_paddle_width"])
        self.ball_speed_substeps = int(level_data["ball_speed_substeps"])

        self.wall_cells = {tuple(int(v) for v in item) for item in (level_data.get("walls") or [])}
        self.drain_cells = {tuple(int(v) for v in item) for item in (level_data.get("drains") or [])}

        px, py, pw = tuple(int(v) for v in level_data["paddle"])
        self.paddle_x = px
        self.paddle_y = py
        self.paddle_width = int(pw)

        bx, by = tuple(int(v) for v in level_data["ball"])
        self.ball_x = bx
        self.ball_y = by
        self.ball_dx = 0
        self.ball_dy = -1
        self.ball_docked = True
        self.launch_dx = 0
        self.respawn_pending = False

        self.ball_frame_is_a = True
        self.capsule_blink = False
        self.bumper_alt_frame = False
        self.step_counter = 0

        self.lives = 3
        self.time_left = self.max_time
        self.destroyed_this_step = 0

        self.power_timer = 0
        self.power_mode: str | None = None

        self.bricks: dict[int, Brick] = {}
        self.brick_cells: dict[tuple[int, int], int] = {}
        self.capsules: list[Capsule] = []
        self.bumpers: list[Bumper] = []

        for index, brick_spec in enumerate(level_data.get("bricks") or []):
            kind = str(brick_spec["kind"])
            brick = Brick(
                brick_id=index,
                x=int(brick_spec["x"]),
                y=int(brick_spec["y"]),
                kind=kind,
                hp=int(_brick_hp_for_kind(kind)),
            )
            self.bricks[index] = brick
            self.brick_cells[(brick.x, brick.y)] = index
            self.brick_cells[(brick.x + 1, brick.y)] = index
            if kind in {"expand", "shrink"}:
                self.capsules.append(Capsule(cap_type=kind, x=-10, y=-10, active=False))

        for bumper_spec in level_data.get("bumpers") or []:
            self.bumpers.append(
                Bumper(
                    x=int(bumper_spec["x"]),
                    y=int(bumper_spec["y"]),
                    direction=int(bumper_spec.get("direction", 1) or 1),
                )
            )

        self.breakable_remaining = sum(1 for brick in self.bricks.values() if _brick_is_breakable(brick.kind))

        self.is_win = False
        self.is_fail = False

    @classmethod
    def from_level_data(cls, level_data: dict) -> BreakoutModel:
        return cls(level_data)

    def clone(self) -> BreakoutModel:
        data = {
            "width": self.width,
            "height": self.height,
            "max_time": self.max_time,
            "time_bonus_per_brick": self.time_bonus_per_brick,
            "base_paddle_width": self.base_paddle_width,
            "ball_speed_substeps": self.ball_speed_substeps,
            "walls": sorted(self.wall_cells),
            "drains": sorted(self.drain_cells),
            "paddle": (self.paddle_x, self.paddle_y, self.paddle_width),
            "ball": (self.ball_x, self.ball_y),
            "bricks": [{"x": brick.x, "y": brick.y, "kind": brick.kind} for brick in self.bricks.values()],
            "bumpers": [{"x": bumper.x, "y": bumper.y, "direction": bumper.direction} for bumper in self.bumpers],
        }
        cloned = BreakoutModel(data)
        cloned.ball_dx = self.ball_dx
        cloned.ball_dy = self.ball_dy
        cloned.ball_docked = self.ball_docked
        cloned.launch_dx = self.launch_dx
        cloned.respawn_pending = self.respawn_pending
        cloned.ball_frame_is_a = self.ball_frame_is_a
        cloned.capsule_blink = self.capsule_blink
        cloned.bumper_alt_frame = self.bumper_alt_frame
        cloned.step_counter = self.step_counter
        cloned.lives = self.lives
        cloned.time_left = self.time_left
        cloned.destroyed_this_step = self.destroyed_this_step
        cloned.power_timer = self.power_timer
        cloned.power_mode = self.power_mode
        cloned.breakable_remaining = self.breakable_remaining
        cloned.is_win = self.is_win
        cloned.is_fail = self.is_fail

        for bid, brick in self.bricks.items():
            copied = cloned.bricks[bid]
            copied.hp = brick.hp
            copied.pending_damage = brick.pending_damage
            copied.flash = brick.flash
            copied.removed = brick.removed
        for idx, capsule in enumerate(self.capsules):
            cloned.capsules[idx].cap_type = capsule.cap_type
            cloned.capsules[idx].x = capsule.x
            cloned.capsules[idx].y = capsule.y
            cloned.capsules[idx].active = capsule.active
        for idx, bumper in enumerate(self.bumpers):
            cloned.bumpers[idx].x = bumper.x
            cloned.bumpers[idx].y = bumper.y
            cloned.bumpers[idx].direction = bumper.direction
        return cloned

    @property
    def paddle_center(self) -> int:
        return int(self.paddle_x + self.paddle_width // 2)

    def paddle_cells(self) -> set[tuple[int, int]]:
        return {(x, self.paddle_y) for x in range(self.paddle_x, self.paddle_x + self.paddle_width)}

    def brick_id_at(self, x: int, y: int) -> int | None:
        bid = self.brick_cells.get((x, y))
        if bid is None:
            return None
        brick = self.bricks.get(bid)
        if brick is None or brick.removed:
            return None
        return bid

    def bumper_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for bumper in self.bumpers:
            for dy in (0, 1):
                for dx in (0, 1):
                    cells.add((bumper.x + dx, bumper.y + dy))
        return cells

    def _set_paddle_width(self, new_width: int) -> None:
        clamped = max(3, min(9, int(new_width)))
        old_center = self.paddle_center
        self.paddle_width = clamped
        self.paddle_x = max(1, min(self.width - 1 - self.paddle_width, old_center - self.paddle_width // 2))
        if self.ball_docked:
            self.ball_x = self.paddle_center
            self.ball_y = self.paddle_y - 1

    def _spawn_ball_on_paddle(self) -> None:
        self.ball_docked = True
        self.ball_x = self.paddle_center
        self.ball_y = self.paddle_y - 1
        self.ball_dx = 0
        self.ball_dy = -1
        self.launch_dx = 0

    def _mark_failed(self) -> None:
        self.is_fail = True
        self.respawn_pending = False

    def _queue_damage(self, brick_id: int) -> None:
        brick = self.bricks.get(brick_id)
        if brick is None or brick.removed or not _brick_is_breakable(brick.kind):
            return
        brick.pending_damage += 1
        brick.flash = 1

    def _remove_brick(self, brick: Brick) -> None:
        brick.removed = True
        self.brick_cells.pop((brick.x, brick.y), None)
        self.brick_cells.pop((brick.x + 1, brick.y), None)
        if _brick_is_breakable(brick.kind):
            self.breakable_remaining = max(0, self.breakable_remaining - 1)
            self.destroyed_this_step += 1
        if brick.kind in {"expand", "shrink"}:
            for capsule in self.capsules:
                if not capsule.active and capsule.cap_type == brick.kind:
                    capsule.active = True
                    capsule.x = brick.x
                    capsule.y = brick.y
                    break

    def _resolve_pending(self) -> None:
        self.destroyed_this_step = 0
        if self.respawn_pending and self.lives > 0:
            self._spawn_ball_on_paddle()
            self.respawn_pending = False

        for brick in self.bricks.values():
            if brick.removed:
                continue
            if brick.pending_damage <= 0:
                continue
            brick.hp -= int(brick.pending_damage)
            brick.pending_damage = 0
            brick.flash = 0
            if brick.hp <= 0:
                self._remove_brick(brick)

    def _apply_input(self, action_id: int) -> None:
        move = 0
        if action_id == A_LEFT:
            move = -1
        elif action_id == A_RIGHT:
            move = 1

        if move != 0:
            nx = self.paddle_x + move
            nx = max(1, min(self.width - 1 - self.paddle_width, nx))
            actual_move = nx - self.paddle_x
            self.paddle_x = nx
            if self.ball_docked:
                self.ball_x += actual_move
                self.launch_dx = _sign(actual_move)

        if action_id == A_SPACE and self.ball_docked:
            self.ball_docked = False
            self.ball_dx = int(self.launch_dx)
            self.ball_dy = -1

    def _apply_capsule_effect(self, cap_type: str) -> None:
        if cap_type == "expand":
            self._set_paddle_width(self.paddle_width + 2)
            self.power_mode = "expand"
            self.power_timer = 300
        elif cap_type == "shrink":
            self._set_paddle_width(self.paddle_width - 2)
            self.power_mode = "shrink"
            self.power_timer = 300

    def _move_capsules(self) -> None:
        paddle_cells = self.paddle_cells()
        for capsule in self.capsules:
            if not capsule.active:
                continue
            capsule.y += 1
            if (capsule.x, capsule.y) in paddle_cells:
                self._apply_capsule_effect(capsule.cap_type)
                capsule.active = False
                capsule.x = -10
                capsule.y = -10
                continue
            if capsule.y >= DRAIN_Y or (capsule.x, capsule.y) in self.drain_cells:
                capsule.active = False
                capsule.x = -10
                capsule.y = -10

    def _bumper_can_occupy(self, nx: int, ny: int, moving_index: int) -> bool:
        cells = {(nx + dx, ny + dy) for dy in (0, 1) for dx in (0, 1)}
        for cx, cy in cells:
            if cx <= 0 or cy <= SEPARATOR_Y or cx >= self.width - 1 or cy >= DRAIN_Y:
                return False
            if (cx, cy) in self.wall_cells:
                return False
            bid = self.brick_id_at(cx, cy)
            if bid is not None:
                return False
        for idx, bumper in enumerate(self.bumpers):
            if idx == moving_index:
                continue
            other = {(bumper.x + dx, bumper.y + dy) for dy in (0, 1) for dx in (0, 1)}
            if cells & other:
                return False
        return not cells & self.paddle_cells()

    def _move_bumpers(self) -> None:
        if not self.bumpers:
            return
        if self.step_counter % 6 != 0:
            return
        for idx, bumper in enumerate(self.bumpers):
            trial_x = bumper.x + bumper.direction
            if self._bumper_can_occupy(trial_x, bumper.y, idx):
                bumper.x = trial_x
            else:
                bumper.direction *= -1

    def _bounce(self, predicate) -> None:
        block_x = predicate(self.ball_x + self.ball_dx, self.ball_y)
        block_y = predicate(self.ball_x, self.ball_y + self.ball_dy)
        if block_x and not block_y:
            self.ball_dx *= -1
        elif block_y and not block_x:
            self.ball_dy *= -1
        else:
            self.ball_dx *= -1
            self.ball_dy *= -1

    def _move_ball(self) -> None:
        if self.ball_docked or self.respawn_pending:
            return

        for _ in range(max(1, self.ball_speed_substeps)):
            nx = self.ball_x + self.ball_dx
            ny = self.ball_y + self.ball_dy

            bumper_cells = self.bumper_cells()

            def blocker_pred(x: int, y: int, bumper_cells: set[tuple[int, int]] = bumper_cells) -> bool:
                if (x, y) in self.wall_cells:
                    return True
                bid = self.brick_id_at(x, y)
                if bid is not None and self.bricks[bid].kind == "unbreakable":
                    return True
                return (x, y) in bumper_cells

            if blocker_pred(nx, ny):
                self._bounce(blocker_pred)
                continue

            bid = self.brick_id_at(nx, ny)
            if bid is not None:

                def brick_pred(x: int, y: int) -> bool:
                    hit = self.brick_id_at(x, y)
                    if hit is None:
                        return False
                    return self.bricks[hit].kind != "unbreakable"

                self._queue_damage(bid)
                self._bounce(brick_pred)
                continue

            if self.ball_dy > 0 and ny == self.paddle_y and self.paddle_x <= nx < self.paddle_x + self.paddle_width:
                offset = self.ball_x - self.paddle_center
                self.ball_dx = max(-1, min(1, _sign(offset)))
                self.ball_dy = -1
                continue

            self.ball_x = nx
            self.ball_y = ny

    def _drain_check(self) -> None:
        if self.ball_docked:
            return
        if self.ball_y == DRAIN_Y or (self.ball_x, self.ball_y) in self.drain_cells:
            self.lives -= 1
            self.respawn_pending = self.lives > 0
            if self.lives <= 0:
                self._mark_failed()

    def _tick_time(self) -> None:
        bonus = self.destroyed_this_step * self.time_bonus_per_brick
        self.time_left = max(0, min(self.max_time, self.time_left - 1 + bonus))
        if self.time_left <= 0:
            self._mark_failed()

    def _tick_power(self) -> None:
        if self.power_timer > 0:
            self.power_timer -= 1
            if self.power_timer == 0:
                self.power_mode = None
        elif self.paddle_width != self.base_paddle_width:
            if self.paddle_width < self.base_paddle_width:
                self._set_paddle_width(self.paddle_width + 1)
            else:
                self._set_paddle_width(self.paddle_width - 1)

    def step(self, action_id: int) -> None:
        if self.is_win or self.is_fail:
            return

        self.step_counter += 1
        self.ball_frame_is_a = not self.ball_frame_is_a
        self.capsule_blink = not self.capsule_blink
        self.bumper_alt_frame = not self.bumper_alt_frame

        self._resolve_pending()
        self._apply_input(int(action_id))
        self._move_capsules()
        self._move_bumpers()
        self._move_ball()
        self._drain_check()
        if self.is_fail:
            return
        self._tick_time()
        if self.is_fail:
            return
        self._tick_power()

        if self.breakable_remaining <= 0:
            self.is_win = True
        if self.lives <= 0 or self.time_left <= 0:
            self._mark_failed()

    def timebar_color(self) -> int:
        frac = float(self.time_left) / float(max(1, self.max_time))
        if frac > (2.0 / 3.0):
            return C_TIME_HIGH
        if frac > (1.0 / 3.0):
            return C_TIME_MID
        return C_TIME_LOW

    def predict_drain_x(self, horizon: int = 120) -> int | None:
        sim = self.clone()
        for _ in range(max(1, horizon)):
            if sim.is_fail or sim.is_win:
                return None
            if sim.ball_docked:
                return None
            if sim.ball_dy > 0 and sim.ball_y >= sim.paddle_y - 2:
                return int(sim.ball_x)
            sim.step(A_SPACE)
        return None


def _build_level(spec: dict) -> Level:
    parsed = _parse_level_rows(list(spec["rows"]))

    data = {
        "name": str(spec["name"]),
        "width": WIDTH,
        "height": HEIGHT,
        "max_time": int(spec["max_time"]),
        "time_bonus_per_brick": int(spec["time_bonus_per_brick"]),
        "base_paddle_width": int(spec["paddle_width"]),
        "ball_speed_substeps": int(spec["ball_speed_substeps"]),
        "walls": parsed["walls"],
        "drains": parsed["drains"],
        "bricks": parsed["bricks"],
        "bumpers": parsed["bumpers"],
        "ball": parsed["ball"],
        "paddle": parsed["paddle"],
    }

    wall_pixels = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
    for x, y in parsed["walls"]:
        wall_pixels[y, x] = C_WALL

    drain_pixels = np.full((1, WIDTH), -1, dtype=np.int8)
    for x, y in parsed["drains"]:
        if y == DRAIN_Y:
            drain_pixels[0, x] = C_TIME_LOW

    sprites: list[Sprite] = [
        Sprite(pixels=_full(C_BG, WIDTH, HEIGHT), name="bg", x=0, y=0, layer=0, tags=["bg"], collidable=False),
        Sprite(pixels=wall_pixels, name="walls", x=0, y=0, layer=1, tags=["wall"], collidable=True),
        Sprite(pixels=drain_pixels, name="drain", x=0, y=DRAIN_Y, layer=2, tags=["drain"], collidable=False),
        Sprite(
            pixels=_full(C_TIME_HIGH, 20, 1),
            name="timebar",
            x=2,
            y=UI_TIME_Y,
            layer=3,
            tags=["hud", "timer"],
            collidable=False,
        ),
        Sprite(
            pixels=_full(C_BALL_A, 3, 1),
            name="lives",
            x=25,
            y=UI_LIVES_Y,
            layer=3,
            tags=["hud", "lives"],
            collidable=False,
        ),
        Sprite(
            pixels=_full(C_PADDLE, int(parsed["paddle"][2]), 1),
            name="paddle",
            x=int(parsed["paddle"][0]),
            y=int(parsed["paddle"][1]),
            layer=6,
            tags=["paddle"],
            collidable=False,
        ),
        Sprite(
            pixels=_full(C_BALL_A, 1, 1),
            name="ball",
            x=int(parsed["ball"][0]),
            y=int(parsed["ball"][1]),
            layer=7,
            tags=["ball"],
            collidable=False,
        ),
    ]

    for idx, brick in enumerate(parsed["bricks"]):
        kind = str(brick["kind"])
        temp = Brick(idx, int(brick["x"]), int(brick["y"]), kind, _brick_hp_for_kind(kind))
        sprites.append(
            Sprite(
                pixels=_full(_brick_color(temp), 2, 1),
                name=f"brick_{idx}",
                x=int(brick["x"]),
                y=int(brick["y"]),
                layer=4,
                tags=["brick", kind],
                collidable=False,
            )
        )

    capsule_count = sum(1 for brick in parsed["bricks"] if brick["kind"] in {"expand", "shrink"})
    for idx in range(capsule_count):
        sprites.append(
            Sprite(
                pixels=_full(C_TIME_HIGH, 1, 1),
                name=f"capsule_{idx}",
                x=-10,
                y=-10,
                layer=5,
                tags=["capsule"],
                collidable=False,
            )
        )

    for idx, bumper in enumerate(parsed["bumpers"]):
        sprites.append(
            Sprite(
                pixels=_full(C_PULSE, 2, 2),
                name=f"bumper_{idx}",
                x=int(bumper["x"]),
                y=int(bumper["y"]),
                layer=5,
                tags=["bumper"],
                collidable=False,
            )
        )

    return Level(name=str(spec["name"]), sprites=sprites, grid_size=(WIDTH, HEIGHT), data=data)


class Breakout(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, WIDTH, HEIGHT, 5, 5, [])
        super().__init__(
            "breakout",
            levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[A_LEFT, A_RIGHT, A_SPACE],
        )
        self._breakout_score = 0

    def on_set_level(self, level: Level) -> None:
        data = dict(getattr(level, "_data", {}) or {})
        self._model = BreakoutModel.from_level_data(data)
        self._breakout_score = int(getattr(self, "levels_completed", 0) or 0)

        self._timebar = level.get_sprites_by_name("timebar")[0]
        self._lives = level.get_sprites_by_name("lives")[0]
        self._paddle = level.get_sprites_by_name("paddle")[0]
        self._ball = level.get_sprites_by_name("ball")[0]

        self._brick_sprites: dict[int, Sprite] = {}
        for brick in self._model.bricks.values():
            self._brick_sprites[brick.brick_id] = level.get_sprites_by_name(f"brick_{brick.brick_id}")[0]

        self._capsule_sprites: list[Sprite] = []
        for idx in range(len(self._model.capsules)):
            self._capsule_sprites.append(level.get_sprites_by_name(f"capsule_{idx}")[0])

        self._bumper_sprites: list[Sprite] = []
        for idx in range(len(self._model.bumpers)):
            self._bumper_sprites.append(level.get_sprites_by_name(f"bumper_{idx}")[0])

        self._sync_view()

    def _sync_view(self) -> None:
        bar_fill = round((self._model.time_left / max(1, self._model.max_time)) * 20.0)
        bar_fill = max(0, min(20, bar_fill))
        bar_color = self._model.timebar_color()
        pixels = np.full((1, 20), C_BG, dtype=np.int8)
        if bar_fill > 0:
            pixels[0, :bar_fill] = int(bar_color)
        self._timebar.pixels = pixels

        life_count = max(0, min(3, int(self._model.lives)))
        life_pixels = np.full((1, 3), C_BG, dtype=np.int8)
        if life_count > 0:
            life_pixels[0, :life_count] = C_BALL_A
        self._lives.pixels = life_pixels

        self._paddle.pixels = _full(C_PADDLE, self._model.paddle_width, 1)
        self._paddle.set_position(self._model.paddle_x, self._model.paddle_y)

        ball_color = C_BALL_A if self._model.ball_frame_is_a else C_FLASH
        self._ball.pixels[0][0] = int(ball_color)
        self._ball.set_position(self._model.ball_x, self._model.ball_y)

        for brick_id, sprite in self._brick_sprites.items():
            brick = self._model.bricks[brick_id]
            if brick.removed:
                sprite.set_position(-20, -20)
            else:
                sprite.set_position(brick.x, brick.y)
                sprite.pixels = _full(_brick_color(brick), 2, 1)

        for idx, capsule in enumerate(self._model.capsules):
            sprite = self._capsule_sprites[idx]
            if not capsule.active:
                sprite.set_position(-20, -20)
                continue
            base = C_TIME_HIGH if capsule.cap_type == "expand" else C_TIME_MID
            color = C_PULSE if self._model.capsule_blink else base
            sprite.pixels[0][0] = int(color)
            sprite.set_position(capsule.x, capsule.y)

        for idx, bumper in enumerate(self._model.bumpers):
            sprite = self._bumper_sprites[idx]
            color = C_FLASH if self._model.bumper_alt_frame else C_PULSE
            sprite.pixels = _full(color, 2, 2)
            sprite.set_position(bumper.x, bumper.y)

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

        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id not in (A_LEFT, A_RIGHT, A_SPACE):
            action_id = A_SPACE

        self._model.step(action_id)
        self._sync_view()

        if self._model.is_fail:
            self.lose()
            self.complete_action()
            return

        if self._model.is_win:
            self._breakout_score += 1
            self.next_level()

        self.complete_action()
