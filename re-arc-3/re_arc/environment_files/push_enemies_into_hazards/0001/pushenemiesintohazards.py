from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

from re_arc.environment_files.push_enemies_into_hazards.common import (
    ACTION_DELTAS,
    LEVEL_SPECS,
    LevelModel,
    build_level_models,
)

GAME_ID = "push_enemies_into_hazards-0001"

COLOR_FLOOR = 1
COLOR_WALL = 2
COLOR_SPIKES_A = 3
COLOR_SPIKES_B = 4
COLOR_PLAYER_IDLE = 5
COLOR_PLAYER_MOVE = 6
COLOR_ENEMY_STATIONARY = 7
COLOR_ENEMY_STUN = 8
COLOR_ENEMY_PATROLLER = 9
COLOR_ENEMY_CHASER = 10
COLOR_BRUTE_BODY = 11
COLOR_BRUTE_HIGHLIGHT = 12
COLOR_EFFECT_TRAIL = 13
COLOR_EFFECT_IMPACT = 14
COLOR_TIME_FULL = 15
COLOR_TIME_EMPTY = 0

PLAY_STATE = "PLAY"
CLEAR_STATE = "CLEAR"
FAIL_STATE = "FAIL"
ANIMATION_STEPS = 6


@dataclass
class Enemy:
    enemy_type: int  # 0=stationary, 1=patroller, 2=chaser
    x: int
    y: int
    stun: int
    dir_x: int
    dir_y: int


def _full_grid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _make_level(index: int, model: LevelModel) -> Level:
    width = int(model.width)
    height = int(model.height)
    board = Sprite(
        pixels=_full_grid(width, height, COLOR_FLOOR), name="board", x=0, y=0, collidable=False, layer=0, tags=["board"]
    )
    return Level(
        name=str(LEVEL_SPECS[index]["name"]),
        sprites=[board],
        grid_size=(width, height),
        data={"level_index": int(index), "time_max_steps": int(model.time_max_steps)},
    )


class PushEnemiesIntoHazards(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._models = build_level_models()
        levels = [_make_level(i, model) for i, model in enumerate(self._models)]
        first = levels[0].grid_size or (28, 18)
        camera = Camera(0, 0, int(first[0]), int(first[1]), COLOR_TIME_EMPTY, COLOR_TIME_EMPTY, [])

        self._level_idx = 0
        self._mode = PLAY_STATE
        self._anim_ticks = 0
        self._tick = 0
        self._spikes_phase_a = True

        self._player_x = 0
        self._player_y = 0
        self._player_moved = False

        self._enemies: list[Enemy] = []
        self._brute_x = -1
        self._brute_y = -1
        self._brute_shake = 0

        self._time_max_steps = 1
        self._time_left_steps = 1

        self._trails: set[tuple[int, int]] = set()
        self._impacts: set[tuple[int, int]] = set()

        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        idx = int(level.get_data("level_index") or 0)
        model = self._models[idx]
        self._level_idx = idx
        self._mode = PLAY_STATE
        self._anim_ticks = 0
        self._tick = 0
        self._spikes_phase_a = True

        self._player_x = int(model.player_start[0])
        self._player_y = int(model.player_start[1])
        self._player_moved = False

        self._enemies = [
            Enemy(
                enemy_type=int(enemy[0]),
                x=int(enemy[1]),
                y=int(enemy[2]),
                stun=int(enemy[3]),
                dir_x=int(enemy[4]),
                dir_y=int(enemy[5]),
            )
            for enemy in model.enemies_start
        ]

        if model.has_brute:
            self._brute_x = int(model.brute_start[0])
            self._brute_y = int(model.brute_start[1])
        else:
            self._brute_x = -1
            self._brute_y = -1
        self._brute_shake = 0

        self._time_max_steps = int(level.get_data("time_max_steps") or model.time_max_steps)
        self._time_left_steps = int(self._time_max_steps)

        self._trails = set()
        self._impacts = set()

        self._render_board()

    def _model(self) -> LevelModel:
        return self._models[self._level_idx]

    def _is_wall(self, x: int, y: int) -> bool:
        model = self._model()
        if x < 0 or y < 0 or x >= model.width or y >= model.height:
            return True
        return (x, y) in model.walls

    def _is_spike(self, x: int, y: int) -> bool:
        return (x, y) in self._model().spikes

    def _brute_cells(self) -> set[tuple[int, int]]:
        if self._brute_x < 0 or self._brute_y < 0:
            return set()
        return {
            (self._brute_x, self._brute_y),
            (self._brute_x + 1, self._brute_y),
            (self._brute_x, self._brute_y + 1),
            (self._brute_x + 1, self._brute_y + 1),
        }

    def _enemy_index_at(self, x: int, y: int) -> int | None:
        for idx, enemy in enumerate(self._enemies):
            if enemy.x == x and enemy.y == y:
                return idx
        return None

    def _remaining_targets(self) -> int:
        count = len(self._enemies)
        if self._brute_x >= 0 and self._brute_y >= 0:
            count += 1
        return count

    def _enter_fail(self) -> None:
        if self._mode != PLAY_STATE:
            return
        self._mode = FAIL_STATE
        self._anim_ticks = ANIMATION_STEPS

    def _enter_clear(self) -> None:
        if self._mode != PLAY_STATE:
            return
        self._mode = CLEAR_STATE
        self._anim_ticks = ANIMATION_STEPS

    def _begin_new_effects(self):
        return set(), set()

    def _blocked_for_enemy(self, x: int, y: int, enemy_idx: int) -> bool:
        if self._is_wall(x, y):
            return True
        if (x, y) in self._brute_cells():
            return True
        for idx, enemy in enumerate(self._enemies):
            if idx == enemy_idx:
                continue
            if enemy.x == x and enemy.y == y:
                return True
        return False

    def _resolve_player_action(
        self, action_id: int, trails: set[tuple[int, int]], impacts: set[tuple[int, int]]
    ) -> None:
        self._player_moved = False
        if action_id not in ACTION_DELTAS:
            return

        dx, dy = ACTION_DELTAS[action_id]
        tx = self._player_x + dx
        ty = self._player_y + dy

        if self._is_wall(tx, ty):
            impacts.add((tx, ty))
            return

        if self._is_spike(tx, ty):
            trails.add((self._player_x, self._player_y))
            self._player_x = tx
            self._player_y = ty
            self._player_moved = True
            self._enter_fail()
            return

        enemy_idx = self._enemy_index_at(tx, ty)
        if enemy_idx is not None:
            nx = tx + dx
            ny = ty + dy
            occupied_enemy = self._enemy_index_at(nx, ny) is not None
            occupied_brute = (nx, ny) in self._brute_cells()
            if self._is_wall(nx, ny) or occupied_enemy or occupied_brute:
                impacts.add((tx, ty))
                return

            trails.add((self._player_x, self._player_y))
            enemy = self._enemies[enemy_idx]
            if self._is_spike(nx, ny):
                self._enemies.pop(enemy_idx)
            else:
                trails.add((enemy.x, enemy.y))
                enemy.x = nx
                enemy.y = ny
                enemy.stun = 2
            self._player_x = tx
            self._player_y = ty
            self._player_moved = True
            return

        brute_cells = self._brute_cells()
        if (tx, ty) in brute_cells:
            old_cells = set(brute_cells)
            shifted_cells = {(bx + dx, by + dy) for bx, by in old_cells}

            blocked = False
            for cx, cy in shifted_cells:
                if self._is_wall(cx, cy):
                    blocked = True
                    break
                if (cx, cy) == (self._player_x, self._player_y) and (cx, cy) not in old_cells:
                    blocked = True
                    break
                if self._enemy_index_at(cx, cy) is not None:
                    blocked = True
                    break

            if blocked:
                impacts.add((tx, ty))
                return

            trails.add((self._player_x, self._player_y))
            self._player_x = tx
            self._player_y = ty
            self._player_moved = True

            if self._brute_x >= 0 and self._brute_y >= 0:
                trails.update(old_cells)
                if any((cx, cy) in self._model().spikes for cx, cy in shifted_cells):
                    self._brute_x = -1
                    self._brute_y = -1
                    self._brute_shake = 0
                else:
                    self._brute_x += dx
                    self._brute_y += dy
                    self._brute_shake = 1
            return

        trails.add((self._player_x, self._player_y))
        self._player_x = tx
        self._player_y = ty
        self._player_moved = True

    def _enemy_ai(self, trails: set[tuple[int, int]]) -> None:
        for idx, enemy in enumerate(self._enemies):
            if enemy.stun > 0:
                continue
            if enemy.enemy_type == 0:
                continue

            if enemy.enemy_type == 1:
                nx = enemy.x + enemy.dir_x
                ny = enemy.y + enemy.dir_y
                if self._blocked_for_enemy(nx, ny, idx):
                    enemy.dir_x = -enemy.dir_x
                    enemy.dir_y = -enemy.dir_y
                else:
                    trails.add((enemy.x, enemy.y))
                    enemy.x = nx
                    enemy.y = ny
                continue

            if enemy.enemy_type == 2:
                dx = self._player_x - enemy.x
                dy = self._player_y - enemy.y
                candidates: list[tuple[int, int]] = []
                if dx != 0:
                    candidates.append((1 if dx > 0 else -1, 0))
                if dy != 0:
                    candidates.append((0, 1 if dy > 0 else -1))

                for mx, my in candidates:
                    nx = enemy.x + mx
                    ny = enemy.y + my
                    if self._blocked_for_enemy(nx, ny, idx):
                        continue
                    trails.add((enemy.x, enemy.y))
                    enemy.x = nx
                    enemy.y = ny
                    break

    def _resolve_collisions_and_hazards(self) -> None:
        for enemy in self._enemies:
            if enemy.x == self._player_x and enemy.y == self._player_y:
                self._enter_fail()
                return

        survivors: list[Enemy] = []
        for enemy in self._enemies:
            if self._is_spike(enemy.x, enemy.y):
                continue
            survivors.append(enemy)
        self._enemies = survivors

    def _animate_step(self) -> None:
        self._spikes_phase_a = not self._spikes_phase_a
        for enemy in self._enemies:
            if enemy.stun > 0:
                enemy.stun -= 1
        if self._brute_shake > 0:
            self._brute_shake -= 1

    def _timebar_filled_cells(self) -> int:
        steps_per_cell = max(1, math.ceil(float(self._time_max_steps) / 26.0))
        return max(0, min(26, math.ceil(float(self._time_left_steps) / float(steps_per_cell))))

    def _render_board(self) -> None:
        model = self._model()
        board = _full_grid(model.width, model.height, COLOR_FLOOR)

        floor_override = None
        if self._mode == CLEAR_STATE:
            floor_override = COLOR_EFFECT_TRAIL
        elif self._mode == FAIL_STATE:
            floor_override = COLOR_EFFECT_IMPACT
        if floor_override is not None:
            board[:, :] = floor_override

        for wx, wy in model.walls:
            board[wy, wx] = COLOR_WALL

        spike_color = COLOR_SPIKES_A if self._spikes_phase_a else COLOR_SPIKES_B
        for sx, sy in model.spikes:
            board[sy, sx] = spike_color

        filled = self._timebar_filled_cells()
        for offset in range(26):
            x = 1 + offset
            board[1, x] = COLOR_TIME_FULL if offset < filled else COLOR_TIME_EMPTY

        for tx, ty in self._trails:
            if 0 <= tx < model.width and 0 <= ty < model.height:
                board[ty, tx] = COLOR_EFFECT_TRAIL
        for ix, iy in self._impacts:
            if 0 <= ix < model.width and 0 <= iy < model.height:
                board[iy, ix] = COLOR_EFFECT_IMPACT

        flash = (self._tick % 2) == 0

        if self._brute_x >= 0 and self._brute_y >= 0:
            body_color = COLOR_BRUTE_HIGHLIGHT if self._brute_shake > 0 else COLOR_BRUTE_BODY
            for cx, cy in self._brute_cells():
                board[cy, cx] = body_color
            board[self._brute_y, self._brute_x] = COLOR_BRUTE_HIGHLIGHT

        for enemy in self._enemies:
            if enemy.enemy_type == 0:
                color = COLOR_ENEMY_STUN if enemy.stun > 0 and flash else COLOR_ENEMY_STATIONARY
            elif enemy.enemy_type == 1:
                color = COLOR_ENEMY_STUN if enemy.stun > 0 and flash else COLOR_ENEMY_PATROLLER
            else:
                color = COLOR_ENEMY_STUN if enemy.stun > 0 and flash else COLOR_ENEMY_CHASER
            board[enemy.y, enemy.x] = color

        board[self._player_y, self._player_x] = COLOR_PLAYER_MOVE if self._player_moved else COLOR_PLAYER_IDLE

        sprite = self.current_level.get_sprites_by_name("board")[0]
        sprite.pixels = board

    def _step_play(self, action_id: int) -> None:
        trails, impacts = self._begin_new_effects()

        self._resolve_player_action(action_id, trails, impacts)
        if self._mode == FAIL_STATE:
            self._trails = trails
            self._impacts = impacts
            return

        self._enemy_ai(trails)
        self._resolve_collisions_and_hazards()

        self._trails = trails
        self._impacts = impacts

        if self._mode != PLAY_STATE:
            return

        if self._remaining_targets() == 0:
            self._enter_clear()
            return

        self._animate_step()
        self._time_left_steps -= 1
        if self._time_left_steps <= 0:
            self._enter_fail()

    def _step_animation(self) -> None:
        self._player_moved = False
        self._trails = set()
        self._impacts = set()
        self._spikes_phase_a = not self._spikes_phase_a
        self._anim_ticks -= 1
        if self._anim_ticks > 0:
            return

        if self._mode == CLEAR_STATE:
            self.next_level()
            return

        if self._mode == FAIL_STATE:
            self.lose()

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
        if self._mode == PLAY_STATE:
            self._step_play(action_id)
        else:
            self._step_animation()

        self._tick += 1
        self._render_board()
        self.complete_action()


__all__ = ["PushEnemiesIntoHazards"]
