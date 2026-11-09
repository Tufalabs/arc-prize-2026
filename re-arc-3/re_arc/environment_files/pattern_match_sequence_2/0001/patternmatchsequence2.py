from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "pattern_match_sequence_2-0001"
GRID_WIDTH = 64
GRID_HEIGHT = 64
TIME_LIMIT = 180

COLOR_BG = 0
COLOR_PANEL = 1
COLOR_FRAME = 5
COLOR_TIMER_FILL = 10
COLOR_TIMER_EMPTY = 2
COLOR_PROGRESS = 3
COLOR_SLOT_PENDING = 8
COLOR_SLOT_SOLVED = 11
COLOR_SLOT_ERROR = 14

BUTTON_COLORS = {1: 9, 2: 6, 3: 4, 4: 7}

SEQUENCE = [1, 3, 2, 4, 2, 1]
BUTTON_POSITIONS = {1: (10, 41), 2: (23, 41), 3: (36, 41), 4: (49, 41)}
BUTTON_SIZE = 7
TARGET_START_X = 10
TARGET_Y = 16
TARGET_STEP_X = 8


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _slot_pixels(fill: int, border: int) -> np.ndarray:
    px = np.full((5, 5), int(fill), dtype=np.int8)
    px[0, :] = np.int8(border)
    px[-1, :] = np.int8(border)
    px[:, 0] = np.int8(border)
    px[:, -1] = np.int8(border)
    return px


def _button_pixels(color: int, active: bool, error: bool) -> np.ndarray:
    border = COLOR_FRAME
    if error:
        border = COLOR_SLOT_ERROR
    elif active:
        border = COLOR_PROGRESS
    px = np.full((BUTTON_SIZE, BUTTON_SIZE), np.int8(color), dtype=np.int8)
    px[0, :] = np.int8(border)
    px[-1, :] = np.int8(border)
    px[:, 0] = np.int8(border)
    px[:, -1] = np.int8(border)
    px[BUTTON_SIZE // 2, BUTTON_SIZE // 2] = np.int8(0)
    return px


def _build_level() -> Level:
    sprites: list[Sprite] = []

    sprites.append(
        Sprite(
            _solid(GRID_WIDTH, GRID_HEIGHT, COLOR_PANEL),
            name="panel",
            x=0,
            y=0,
            layer=0,
            tags=["background"],
            collidable=False,
        )
    )

    sprites.append(
        Sprite(
            _solid(GRID_WIDTH, 1, COLOR_TIMER_FILL),
            name="timer_bar",
            x=0,
            y=0,
            layer=5,
            tags=["hud", "timer"],
            collidable=False,
        )
    )

    for idx, token in enumerate(SEQUENCE):
        sprites.append(
            Sprite(
                _slot_pixels(BUTTON_COLORS[token], COLOR_FRAME),
                name=f"target_{idx}",
                x=TARGET_START_X + idx * TARGET_STEP_X,
                y=TARGET_Y,
                layer=2,
                tags=["target", f"target_{idx}", f"token_{token}"],
                collidable=False,
            )
        )

    sprites.append(
        Sprite(
            _solid(len(SEQUENCE) * TARGET_STEP_X - 3, 1, COLOR_SLOT_PENDING),
            name="progress_bar",
            x=TARGET_START_X,
            y=25,
            layer=2,
            tags=["hud", "progress"],
            collidable=False,
        )
    )

    for token, (x, y) in BUTTON_POSITIONS.items():
        sprites.append(
            Sprite(
                _button_pixels(BUTTON_COLORS[token], active=False, error=False),
                name=f"button_{token}",
                x=x,
                y=y,
                layer=3,
                tags=["button", "sys_click", "sys_every_pixel", f"button_{token}"],
                collidable=False,
            )
        )

    return Level(
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=sprites,
        data={
            "time_limit": TIME_LIMIT,
            "sequence": list(SEQUENCE),
            "button_positions": dict(BUTTON_POSITIONS),
            "button_size": BUTTON_SIZE,
        },
    )


class PatternMatchSequence2(ARCBaseGame):
    def __init__(self, seed: int = 0):
        level = _build_level()
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_BG)
        super().__init__(game_id=GAME_ID, levels=[level], camera=camera, win_score=1, available_actions=[6], seed=seed)
        self._time_left = TIME_LIMIT
        self._sequence = list(SEQUENCE)
        self._progress = 0
        self._error_flash = 0
        self._buttons: dict[int, Sprite] = {}
        self._targets: list[Sprite] = []
        self._timer: Sprite | None = None
        self._progress_bar: Sprite | None = None

    def on_set_level(self, level: Level) -> None:
        self._time_left = int(level.get_data("time_limit") or TIME_LIMIT)
        self._sequence = list(level.get_data("sequence") or SEQUENCE)
        self._progress = 0
        self._error_flash = 0

        self._buttons = {}
        for sprite in self.current_level.get_sprites_by_tag("button"):
            token = self._button_token(sprite)
            if token is not None:
                self._buttons[token] = sprite

        self._targets = sorted(self.current_level.get_sprites_by_tag("target"), key=lambda s: int(s.x))

        timers = self.current_level.get_sprites_by_name("timer_bar")
        self._timer = timers[0] if timers else None

        bars = self.current_level.get_sprites_by_name("progress_bar")
        self._progress_bar = bars[0] if bars else None

        self._sync_visuals()

    def _button_token(self, sprite: Sprite) -> int | None:
        for tag in getattr(sprite, "tags", []) or []:
            if not tag.startswith("button_"):
                continue
            try:
                return int(tag.split("_", 1)[1])
            except ValueError:
                return None
        return None

    def _clicked_token(self, x: int, y: int) -> int | None:
        for token, sprite in self._buttons.items():
            sx = int(sprite.x)
            sy = int(sprite.y)
            if sx <= x < sx + int(sprite.width) and sy <= y < sy + int(sprite.height):
                return token
        return None

    def _apply_click(self, x: int, y: int) -> None:
        token = self._clicked_token(x, y)
        if token is None:
            return

        expected = self._sequence[self._progress]
        if token == expected:
            self._progress += 1
            self._error_flash = 0
            if self._progress >= len(self._sequence):
                self.next_level()
            return

        self._progress = 0
        self._error_flash = 2

    def _sync_visuals(self) -> None:
        for idx, target in enumerate(self._targets):
            token = self._sequence[idx]
            if self._error_flash > 0:
                border = COLOR_SLOT_ERROR
            elif idx < self._progress:
                border = COLOR_SLOT_SOLVED
            elif idx == self._progress:
                border = COLOR_PROGRESS
            else:
                border = COLOR_FRAME
            target.pixels = _slot_pixels(BUTTON_COLORS[token], border)

        for token, button in self._buttons.items():
            active = self._progress < len(self._sequence) and token == self._sequence[self._progress]
            button.pixels = _button_pixels(BUTTON_COLORS[token], active=active, error=self._error_flash > 0)

        if self._progress_bar is not None:
            width = int(self._progress_bar.width)
            filled = round((self._progress / max(1, len(self._sequence))) * width)
            filled = max(0, min(width, filled))
            meter = np.full((1, width), np.int8(COLOR_SLOT_PENDING), dtype=np.int8)
            meter[:, :filled] = np.int8(COLOR_PROGRESS)
            self._progress_bar.pixels = meter

        if self._timer is not None:
            filled = round((self._time_left / max(1, TIME_LIMIT)) * GRID_WIDTH)
            filled = max(0, min(GRID_WIDTH, filled))
            row = np.full((1, GRID_WIDTH), np.int8(COLOR_TIMER_EMPTY), dtype=np.int8)
            row[:, :filled] = np.int8(COLOR_TIMER_FILL)
            self._timer.pixels = row

    def step(self) -> None:
        action_id = int(self.action.id.value)
        if action_id == int(GameAction.ACTION6.value):
            payload = self.action.data if isinstance(self.action.data, dict) else {}
            raw_x = payload.get("x", -1)
            raw_y = payload.get("y", -1)
            try:
                click_x = int(raw_x)
                click_y = int(raw_y)
            except (TypeError, ValueError):
                click_x = -1
                click_y = -1
            self._apply_click(click_x, click_y)

        self._time_left -= 1
        if self._error_flash > 0:
            self._error_flash -= 1

        state_name = getattr(getattr(self, "_state", None), "name", "")
        if self._time_left <= 0 and state_name != "WIN":
            self.lose()

        self._sync_visuals()
        self.complete_action()
