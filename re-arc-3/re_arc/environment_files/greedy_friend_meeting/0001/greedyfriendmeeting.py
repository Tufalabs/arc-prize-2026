from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

BACKGROUND_COLOR = 0
PLAYFIELD_X0 = 4
PLAYFIELD_Y0 = 8
CELL_SIZE = 8
BOARD_SIZE = 7

PLAYER_NAME = "player"
HELPER_NAME = "helper"
MARKER_NAME = "marker"

PLAYER_PATTERN = np.array(
    [
        [0, 0, 0, 9, 9, 0, 0, 0],
        [0, 0, 9, 10, 10, 9, 0, 0],
        [0, 9, 10, 9, 9, 10, 9, 0],
        [0, 9, 9, 9, 9, 9, 9, 0],
        [0, 9, 10, 10, 10, 10, 9, 0],
        [0, 0, 9, 10, 10, 9, 0, 0],
        [0, 0, 0, 9, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)
HELPER_PATTERN = np.array(
    [
        [0, 0, 14, 14, 14, 14, 0, 0],
        [0, 14, 10, 14, 14, 10, 14, 0],
        [0, 14, 14, 14, 14, 14, 14, 0],
        [0, 14, 14, 14, 14, 14, 14, 0],
        [0, 14, 14, 14, 14, 14, 14, 0],
        [0, 0, 14, 14, 14, 14, 0, 0],
        [0, 0, 0, 14, 14, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)
OPEN_MARKER_PATTERN = np.array(
    [
        [0, 0, 11, 11, 11, 11, 0, 0],
        [0, 11, 12, 12, 12, 12, 11, 0],
        [0, 11, 12, 0, 0, 12, 11, 0],
        [0, 11, 12, 0, 0, 12, 11, 0],
        [0, 11, 12, 12, 12, 12, 11, 0],
        [0, 0, 11, 11, 11, 11, 0, 0],
        [0, 0, 0, 11, 11, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)
LOCKED_MARKER_PATTERN = np.array(
    [
        [0, 0, 8, 8, 8, 8, 0, 0],
        [0, 8, 13, 13, 13, 13, 8, 0],
        [0, 8, 13, 0, 0, 13, 8, 0],
        [0, 8, 13, 0, 0, 13, 8, 0],
        [0, 8, 13, 13, 13, 13, 8, 0],
        [0, 0, 8, 8, 8, 8, 0, 0],
        [0, 0, 0, 8, 8, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)
WALL_PATTERN = np.array(
    [
        [4, 4, 4, 4, 4, 4, 4, 4],
        [4, 3, 3, 3, 3, 3, 3, 4],
        [4, 3, 4, 4, 4, 4, 3, 4],
        [4, 3, 4, 3, 3, 4, 3, 4],
        [4, 3, 4, 3, 3, 4, 3, 4],
        [4, 3, 4, 4, 4, 4, 3, 4],
        [4, 3, 3, 3, 3, 3, 3, 4],
        [4, 4, 4, 4, 4, 4, 4, 4],
    ],
    dtype=np.int8,
)

MOVE_DELTAS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def _action_id_value(value: object) -> int:
    raw = getattr(value, "value", value)
    return int(raw)


LEVEL_SPECS = (
    {"player_start": (0, 0), "helper_start": (6, 6), "marker": (3, 3), "walls": (), "move_budget": 10},
    {"player_start": (0, 3), "helper_start": (6, 5), "marker": (3, 3), "walls": (), "move_budget": 8},
    {
        "player_start": (1, 3),
        "helper_start": (0, 6),
        "marker": (5, 3),
        "walls": ((3, 2), (3, 3), (3, 4)),
        "move_budget": 11,
    },
)


def _cell_to_pixels(position: tuple[int, int]) -> tuple[int, int]:
    return (PLAYFIELD_X0 + (position[0] * CELL_SIZE), PLAYFIELD_Y0 + (position[1] * CELL_SIZE))


def _sprite(pattern: np.ndarray, *, name: str, position: tuple[int, int], layer: int) -> Sprite:
    px, py = _cell_to_pixels(position)
    return Sprite(pattern.copy(), name=name, x=px, y=py, layer=layer, collidable=False)


def _make_level(index: int, spec: dict[str, object]) -> Level:
    marker = tuple(spec["marker"])
    walls = tuple(tuple(wall) for wall in spec["walls"])
    sprites = [_sprite(OPEN_MARKER_PATTERN, name=MARKER_NAME, position=marker, layer=0)]
    for wall_index, wall in enumerate(walls):
        sprites.append(_sprite(WALL_PATTERN, name=f"wall_{wall_index}", position=wall, layer=1))
    sprites.append(_sprite(HELPER_PATTERN, name=HELPER_NAME, position=tuple(spec["helper_start"]), layer=2))
    sprites.append(_sprite(PLAYER_PATTERN, name=PLAYER_NAME, position=tuple(spec["player_start"]), layer=3))
    return Level(
        sprites=sprites,
        grid_size=(64, 64),
        name=f"Level {index + 1}",
        data={
            "player_start": tuple(spec["player_start"]),
            "helper_start": tuple(spec["helper_start"]),
            "marker": marker,
            "walls": walls,
            "move_budget": int(spec["move_budget"]),
        },
    )


class MoveBudgetDisplay(RenderableUserDisplay):
    def __init__(self, game: GreedyFriendMeeting | None = None) -> None:
        self.game = game

    def attach(self, game: GreedyFriendMeeting) -> None:
        self.game = game

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.game is None:
            return frame

        frame[7, 3:61] = 1
        frame[7:64, 3] = 1
        frame[7:64, 60] = 1
        frame[63, 3:61] = 1

        budget = int(self.game.level_budget)
        remaining = int(self.game.remaining_moves)
        for slot in range(16):
            if slot >= budget:
                continue
            x0 = slot * 4
            color = 3
            if slot < remaining:
                if remaining == 1 and slot == 0:
                    color = 8
                elif remaining <= 3:
                    color = 11
                else:
                    color = 14
            frame[2:5, x0 : x0 + 3] = color

        border_color = None
        if self.game.level_status == "won":
            border_color = 15
        elif self.game.level_status == "lost":
            border_color = 8

        if border_color is not None:
            frame[7, 3:61] = border_color
            frame[7:64, 3] = border_color
            frame[7:64, 60] = border_color
            frame[63, 3:61] = border_color
        return frame


class GreedyFriendMeeting(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.hud = MoveBudgetDisplay()
        self.level_status = "playing"
        self.marker_locked = False
        self.player_pos = (0, 0)
        self.helper_pos = (0, 0)
        self.marker_pos = (0, 0)
        self.walls: frozenset[tuple[int, int]] = frozenset()
        self.remaining_moves = 0
        self.level_budget = 0
        self._route_score = 0
        super().__init__(
            "greedy_friend_meeting",
            [_make_level(index, spec) for index, spec in enumerate(LEVEL_SPECS)],
            Camera(0, 0, 64, 64, BACKGROUND_COLOR, BACKGROUND_COLOR, [self.hud]),
            False,
            len(LEVEL_SPECS),
            [1, 2, 3, 4, 5],
            seed,
        )
        self.hud.attach(self)

    def on_set_level(self, _level: Level) -> None:
        self.level_status = "playing"
        self.marker_locked = False
        self.marker_sprite = self.current_level.get_sprites_by_name(MARKER_NAME)[0]
        self.player_sprite = self.current_level.get_sprites_by_name(PLAYER_NAME)[0]
        self.helper_sprite = self.current_level.get_sprites_by_name(HELPER_NAME)[0]

        self.player_pos = tuple(self.current_level.get_data("player_start"))
        self.helper_pos = tuple(self.current_level.get_data("helper_start"))
        self.marker_pos = tuple(self.current_level.get_data("marker"))
        self.walls = frozenset(tuple(wall) for wall in self.current_level.get_data("walls"))
        self.level_budget = int(self.current_level.get_data("move_budget"))
        self.remaining_moves = self.level_budget
        self._route_score = 0
        self._set_marker_visual()
        self._sync_dynamic_sprites()

    def _set_marker_visual(self) -> None:
        self.marker_sprite.pixels = (LOCKED_MARKER_PATTERN if self.marker_locked else OPEN_MARKER_PATTERN).copy()

    def _sync_dynamic_sprites(self) -> None:
        self.player_sprite.set_position(*_cell_to_pixels(self.player_pos))
        self.helper_sprite.set_position(*_cell_to_pixels(self.helper_pos))
        self.marker_sprite.set_position(*_cell_to_pixels(self.marker_pos))

    def _in_bounds(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def _cell_blocked(self, position: tuple[int, int], *, mover: str) -> bool:
        if not self._in_bounds(position):
            return True
        if position in self.walls:
            return True
        if self.marker_locked and position == self.marker_pos:
            return True
        other = self.helper_pos if mover == "player" else self.player_pos
        if mover == "helper" and position == self.marker_pos and other == self.marker_pos and not self.marker_locked:
            return False
        return position == other

    def _move_player(self) -> None:
        if self.marker_locked and self.player_pos == self.marker_pos:
            return

        delta = MOVE_DELTAS.get(_action_id_value(self.action.id))
        if delta is None:
            return

        destination = (self.player_pos[0] + delta[0], self.player_pos[1] + delta[1])
        if self._cell_blocked(destination, mover="player"):
            return
        self.player_pos = destination

    def _helper_vertical_target(self) -> tuple[int, int] | None:
        if self.helper_pos[1] == self.marker_pos[1]:
            return None
        step = 1 if self.marker_pos[1] > self.helper_pos[1] else -1
        return (self.helper_pos[0], self.helper_pos[1] + step)

    def _helper_horizontal_target(self) -> tuple[int, int] | None:
        if self.helper_pos[0] == self.marker_pos[0]:
            return None
        step = 1 if self.marker_pos[0] > self.helper_pos[0] else -1
        return (self.helper_pos[0] + step, self.helper_pos[1])

    def _move_helper(self) -> None:
        if self.helper_pos == self.marker_pos:
            return

        horizontal = self._helper_horizontal_target()
        vertical = self._helper_vertical_target()

        if horizontal is not None:
            if not self._cell_blocked(horizontal, mover="helper"):
                self.helper_pos = horizontal
                return
            if vertical is not None and not self._cell_blocked(vertical, mover="helper"):
                self.helper_pos = vertical
                return
            return

        if vertical is not None and not self._cell_blocked(vertical, mover="helper"):
            self.helper_pos = vertical

    def _resolve_end_of_turn(self) -> None:
        player_on_marker = self.player_pos == self.marker_pos
        helper_on_marker = self.helper_pos == self.marker_pos

        if player_on_marker and helper_on_marker:
            self.level_status = "won"
            return

        if not self.marker_locked and (player_on_marker ^ helper_on_marker):
            self.marker_locked = True
            self._set_marker_visual()

        if self.remaining_moves == 0:
            self.level_status = "lost"
            self.lose()

    def step(self) -> None:
        if _action_id_value(self.action.id) == 0:
            self.complete_action()
            return

        if self.level_status == "won":
            self.next_level()
            self.complete_action()
            return

        if self.level_status == "lost":
            self.lose()
            self.complete_action()
            return

        self.remaining_moves = max(0, self.remaining_moves - 1)
        self._route_score += 1
        self._move_player()
        self._move_helper()
        self._sync_dynamic_sprites()
        self._resolve_end_of_turn()
        self.complete_action()
