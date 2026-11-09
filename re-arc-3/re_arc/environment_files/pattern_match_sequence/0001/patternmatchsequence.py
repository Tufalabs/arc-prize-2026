from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "pattern_match_sequence-0001"

GRID_WIDTH = 64
GRID_HEIGHT = 64
TIME_LIMIT = 180

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_TIMER_EMPTY = 2
COLOR_TIMER_FILL = 3
COLOR_CARD_BG = 4
COLOR_WALL = 5
COLOR_FRAME_IDLE = 6
COLOR_GOAL_OFF = 7
COLOR_STATUS_NEUTRAL = 8
COLOR_MOTIF_A = 9
COLOR_FRAME_ACTIVE = 10
COLOR_GOAL_ON = 11
COLOR_FRAME_EXPECTED = 12
COLOR_MOTIF_B = 13
COLOR_STATUS_BAD = 14

TOKEN_SIZE = 7
MOTIF_SIZE = 5

SEQUENCE_NODE_NAMES = ["node_1", "node_2", "node_3", "node_4"]

NODE_LAYOUT = [
    ("node_1", 8, 34),
    ("node_2", 24, 34),
    ("node_3", 40, 34),
    ("node_4", 24, 48),
    ("decoy_rotate", 8, 48),
    ("decoy_swap", 40, 48),
]

TARGET_LAYOUT = [("target_1", 8, 8), ("target_2", 20, 8), ("target_3", 32, 8), ("target_4", 44, 8)]


def _l_motif(primary: int, secondary: int, rotation: int) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), -1, dtype=np.int8)
    motif[:, 0] = primary
    motif[-1, :4] = primary
    motif[1:4, 1] = secondary
    motif[3, 1:4] = secondary
    return np.rot90(motif, k=rotation)


def _swap_colors(motif: np.ndarray, a: int, b: int) -> np.ndarray:
    out = motif.copy()
    out[motif == a] = b
    out[motif == b] = a
    return out


MOTIFS: dict[str, np.ndarray] = {
    "node_1": _l_motif(COLOR_MOTIF_A, COLOR_MOTIF_B, 0),
    "node_2": _l_motif(COLOR_MOTIF_B, COLOR_MOTIF_A, 1),
    "node_3": _l_motif(COLOR_MOTIF_A, COLOR_MOTIF_B, 2),
    "node_4": _l_motif(COLOR_MOTIF_B, COLOR_MOTIF_A, 3),
}
MOTIFS["decoy_rotate"] = np.rot90(MOTIFS["node_1"], k=1)
MOTIFS["decoy_swap"] = _swap_colors(MOTIFS["node_3"], COLOR_MOTIF_A, COLOR_MOTIF_B)


def _timer_pixels(fill: int) -> np.ndarray:
    row = [COLOR_TIMER_FILL if x < fill else COLOR_TIMER_EMPTY for x in range(GRID_WIDTH)]
    return np.array([row], dtype=np.int8)


def _token_pixels(motif: np.ndarray, border_color: int) -> np.ndarray:
    pixels = np.full((TOKEN_SIZE, TOKEN_SIZE), COLOR_CARD_BG, dtype=np.int8)
    pixels[0, :] = border_color
    pixels[-1, :] = border_color
    pixels[:, 0] = border_color
    pixels[:, -1] = border_color

    for y in range(MOTIF_SIZE):
        for x in range(MOTIF_SIZE):
            value = int(motif[y, x])
            if value >= 0:
                pixels[y + 1, x + 1] = value
    return pixels


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _path_segments() -> list[Sprite]:
    segments: list[Sprite] = []
    segments.append(
        Sprite(
            _solid(17, 1, COLOR_WALL), name="path_1", x=11, y=37, layer=2, tags=["path_seg", "to_1"], collidable=False
        )
    )
    segments.append(
        Sprite(
            _solid(17, 1, COLOR_WALL), name="path_2", x=27, y=37, layer=2, tags=["path_seg", "to_2"], collidable=False
        )
    )
    segments.append(
        Sprite(
            _solid(17, 1, COLOR_WALL), name="path_3_h", x=27, y=51, layer=2, tags=["path_seg", "to_3"], collidable=False
        )
    )
    segments.append(
        Sprite(
            _solid(1, 15, COLOR_WALL), name="path_3_v", x=43, y=37, layer=2, tags=["path_seg", "to_3"], collidable=False
        )
    )
    return segments


def _build_level() -> Level:
    sprites: list[Sprite] = []

    sprites.append(
        Sprite(
            _solid(GRID_WIDTH, GRID_HEIGHT, COLOR_FLOOR),
            name="floor",
            x=0,
            y=0,
            layer=0,
            tags=["floor", "sys_static"],
            collidable=False,
        )
    )
    sprites.append(
        Sprite(_timer_pixels(GRID_WIDTH), name="timer_bar", x=0, y=0, layer=10, tags=["hud", "timer"], collidable=False)
    )

    sprites.append(
        Sprite(_solid(GRID_WIDTH, 1, COLOR_WALL), name="wall_top", x=0, y=1, layer=1, tags=["wall"], collidable=False)
    )
    sprites.append(
        Sprite(
            _solid(GRID_WIDTH, 1, COLOR_WALL),
            name="wall_bottom",
            x=0,
            y=GRID_HEIGHT - 1,
            layer=1,
            tags=["wall"],
            collidable=False,
        )
    )
    sprites.append(
        Sprite(
            _solid(1, GRID_HEIGHT - 1, COLOR_WALL), name="wall_left", x=0, y=1, layer=1, tags=["wall"], collidable=False
        )
    )
    sprites.append(
        Sprite(
            _solid(1, GRID_HEIGHT - 1, COLOR_WALL),
            name="wall_right",
            x=GRID_WIDTH - 1,
            y=1,
            layer=1,
            tags=["wall"],
            collidable=False,
        )
    )
    sprites.append(
        Sprite(_solid(30, 1, COLOR_WALL), name="wall_mid_left", x=1, y=28, layer=1, tags=["wall"], collidable=False)
    )
    sprites.append(
        Sprite(_solid(30, 1, COLOR_WALL), name="wall_mid_right", x=33, y=28, layer=1, tags=["wall"], collidable=False)
    )

    for idx, (name, x, y) in enumerate(TARGET_LAYOUT):
        motif = MOTIFS[SEQUENCE_NODE_NAMES[idx]]
        sprites.append(
            Sprite(
                _token_pixels(motif, COLOR_FRAME_EXPECTED),
                name=name,
                x=x,
                y=y,
                layer=4,
                tags=["target", f"step_{idx + 1}", "sys_static"],
                collidable=False,
            )
        )

    for name, x, y in NODE_LAYOUT:
        sprites.append(
            Sprite(
                _token_pixels(MOTIFS[name], COLOR_FRAME_IDLE),
                name=name,
                x=x,
                y=y,
                layer=6,
                tags=["node", "sys_click", "sys_every_pixel"],
                collidable=False,
            )
        )

    sprites.extend(_path_segments())

    sprites.append(
        Sprite(_solid(9, 5, COLOR_GOAL_OFF), name="goal", x=50, y=48, layer=5, tags=["goal"], collidable=False)
    )

    sprites.append(
        Sprite(
            _solid(7, 3, COLOR_STATUS_NEUTRAL),
            name="status",
            x=52,
            y=6,
            layer=7,
            tags=["status", "hud"],
            collidable=False,
        )
    )

    return Level(
        name="PatternMatchSequence 0001",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=sprites,
        data={"time_limit": TIME_LIMIT, "sequence": list(SEQUENCE_NODE_NAMES)},
    )


class PatternMatchSequence(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._time_limit = TIME_LIMIT
        self._time_left = TIME_LIMIT
        self._progress = 0
        self._sequence = list(SEQUENCE_NODE_NAMES)
        self._nodes_by_name: dict[str, Sprite] = {}
        self._path_segments: list[tuple[Sprite, int]] = []
        self._goal: Sprite | None = None
        self._status: Sprite | None = None

        level = _build_level()
        camera = Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_BG)
        super().__init__(game_id=GAME_ID, levels=[level], camera=camera, win_score=1, available_actions=[6], seed=seed)

    def on_set_level(self, level: Level) -> None:
        self._time_limit = int(level.get_data("time_limit") or TIME_LIMIT)
        self._time_left = self._time_limit
        self._progress = 0

        seq_data = level.get_data("sequence")
        if isinstance(seq_data, list) and seq_data:
            self._sequence = [str(name) for name in seq_data]
        else:
            self._sequence = list(SEQUENCE_NODE_NAMES)

        self._nodes_by_name = {sprite.name: sprite for sprite in level.get_sprites_by_tag("node")}

        self._path_segments = []
        for sprite in level.get_sprites_by_tag("path_seg"):
            to_step_tag = next((tag for tag in sprite.tags if tag.startswith("to_")), None)
            if to_step_tag is None:
                continue
            self._path_segments.append((sprite, int(to_step_tag.split("_", 1)[1])))

        goals = level.get_sprites_by_name("goal")
        status = level.get_sprites_by_name("status")
        self._goal = goals[0] if goals else None
        self._status = status[0] if status else None

        self._sync_visuals()

    def _sync_timer(self) -> None:
        timers = self.current_level.get_sprites_by_name("timer_bar")
        if not timers:
            return
        fill = round((self._time_left / max(1, self._time_limit)) * GRID_WIDTH)
        fill = max(0, min(GRID_WIDTH, fill))
        timers[0].pixels = _timer_pixels(fill)

    def _set_status(self, color: int) -> None:
        if self._status is not None:
            self._status.pixels = _solid(self._status.width, self._status.height, color)

    def _set_goal(self, is_open: bool) -> None:
        if self._goal is not None:
            color = COLOR_GOAL_ON if is_open else COLOR_GOAL_OFF
            self._goal.pixels = _solid(self._goal.width, self._goal.height, color)

    def _paint_node(self, name: str, border_color: int) -> None:
        node = self._nodes_by_name.get(name)
        motif = MOTIFS.get(name)
        if node is None or motif is None:
            return
        node.pixels = _token_pixels(motif, border_color)

    def _sync_visuals(self) -> None:
        for idx, name in enumerate(self._sequence):
            if idx < self._progress:
                border = COLOR_FRAME_ACTIVE
            elif idx == self._progress:
                border = COLOR_FRAME_EXPECTED
            else:
                border = COLOR_FRAME_IDLE
            self._paint_node(name, border)

        for name in self._nodes_by_name:
            if name not in self._sequence:
                self._paint_node(name, COLOR_FRAME_IDLE)

        for segment, to_step in self._path_segments:
            color = COLOR_FRAME_ACTIVE if self._progress >= to_step else COLOR_WALL
            segment.pixels = _solid(segment.width, segment.height, color)

        self._set_goal(self._progress >= len(self._sequence))

    def _sprite_at_click(self, x: int, y: int) -> Sprite | None:
        if x < 0 or y < 0 or x >= GRID_WIDTH or y >= GRID_HEIGHT:
            return None

        clickables = sorted(self.current_level.get_sprites_by_tag("sys_click"), key=lambda s: s.layer, reverse=True)
        for sprite in clickables:
            if x < sprite.x or y < sprite.y or x >= sprite.x + sprite.width or y >= sprite.y + sprite.height:
                continue
            rendered = sprite.render()
            local_x = x - sprite.x
            local_y = y - sprite.y
            if int(rendered[local_y][local_x]) < 0:
                continue
            return sprite
        return None

    def _apply_click(self, x: int, y: int) -> None:
        clicked = self._sprite_at_click(x, y)
        if clicked is None or "node" not in clicked.tags:
            self._set_status(COLOR_STATUS_NEUTRAL)
            return

        expected_name = self._sequence[self._progress] if self._progress < len(self._sequence) else None
        if clicked.name != expected_name:
            self._progress = 0
            self._set_status(COLOR_STATUS_BAD)
            self._sync_visuals()
            return

        self._progress += 1
        self._set_status(COLOR_FRAME_ACTIVE)
        self._sync_visuals()

        if self._progress >= len(self._sequence):
            self.next_level()

    def current_state_name(self) -> str:
        state = getattr(self, "_state", None)
        return getattr(state, "name", str(state))

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))

        if action_id == int(GameAction.ACTION6.value):
            raw_x = self.action.data.get("x") if isinstance(self.action.data, dict) else None
            raw_y = self.action.data.get("y") if isinstance(self.action.data, dict) else None
            try:
                click_x = int(raw_x)
                click_y = int(raw_y)
            except (TypeError, ValueError):
                click_x = -1
                click_y = -1
            self._apply_click(click_x, click_y)

        if self._time_left > 0 and self.current_state_name() != "WIN":
            self._time_left -= 1
        if self._time_left <= 0 and self.current_state_name() != "WIN":
            self.lose()

        self._sync_timer()
        self.complete_action()
