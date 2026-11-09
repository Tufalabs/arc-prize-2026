from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameState, Level, RenderableUserDisplay, Sprite

GRID_WIDTH = 8
GRID_HEIGHT = 7
CELL_SIZE = 8
HUD_HEIGHT = 8
SCREEN_SIZE = 64
MAX_BUDGET = 13

COLOR_WATER = 10
COLOR_FLOOR = 1
COLOR_BORDER = 3
COLOR_BLACK = 5
COLOR_AVATAR = 6
COLOR_FAILURE = 8
COLOR_PORTAL_A = 11
COLOR_PORTAL_C = 12
COLOR_PORTAL_D = 13
COLOR_GOAL = 14
COLOR_PORTAL_B = 15

ACTION_TO_DELTA = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

AVATAR_PATTERN = np.array(
    [
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, COLOR_AVATAR, -1, -1, -1, -1],
        [-1, -1, COLOR_AVATAR, COLOR_AVATAR, COLOR_AVATAR, -1, -1, -1],
        [-1, COLOR_AVATAR, COLOR_AVATAR, COLOR_AVATAR, COLOR_AVATAR, COLOR_AVATAR, -1, -1],
        [-1, -1, COLOR_AVATAR, COLOR_AVATAR, COLOR_AVATAR, -1, -1, -1],
        [-1, -1, -1, COLOR_AVATAR, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ],
    dtype=np.int8,
)

PORTAL_PATTERN = np.array(
    [
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 1, 1, 1, 1, -1, -1],
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, 1, -1, -1, -1, -1, 1, -1],
        [-1, 1, -1, -1, -1, -1, 1, -1],
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, 1, 1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ],
    dtype=np.int8,
)

GOAL_PATTERN = np.array(
    [
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, COLOR_GOAL, COLOR_GOAL, COLOR_GOAL, COLOR_GOAL, -1, -1],
        [-1, -1, COLOR_GOAL, -1, -1, COLOR_GOAL, -1, -1],
        [-1, -1, COLOR_GOAL, -1, -1, COLOR_GOAL, -1, -1],
        [-1, -1, COLOR_GOAL, -1, -1, COLOR_GOAL, -1, -1],
        [-1, -1, COLOR_GOAL, COLOR_GOAL, COLOR_GOAL, COLOR_GOAL, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
    ],
    dtype=np.int8,
)

LEVEL_SPECS: tuple[dict[str, object], ...] = (
    {
        "floor_cells": frozenset({(0, 2), (1, 2), (0, 3), (1, 3), (4, 2), (5, 2), (4, 3), (5, 3)}),
        "start": (0, 3),
        "goal": (5, 3),
        "move_budget": 8,
        "portal_pairs": ({"a": (1, 2), "b": (4, 2), "color": COLOR_PORTAL_A},),
    },
    {
        "floor_cells": frozenset(
            {
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3),
                (0, 4),
                (1, 4),
                (3, 1),
                (4, 1),
                (5, 1),
                (3, 2),
                (4, 2),
                (5, 2),
                (3, 3),
                (4, 3),
                (5, 3),
                (6, 4),
                (7, 4),
                (6, 5),
                (7, 5),
            }
        ),
        "start": (0, 3),
        "goal": (7, 5),
        "move_budget": 12,
        "portal_pairs": (
            {"a": (1, 2), "b": (3, 2), "color": COLOR_PORTAL_A},
            {"a": (5, 1), "b": (6, 4), "color": COLOR_PORTAL_B},
        ),
    },
    {
        "floor_cells": frozenset(
            {
                (0, 2),
                (1, 2),
                (0, 3),
                (1, 3),
                (0, 4),
                (1, 4),
                (3, 1),
                (4, 1),
                (5, 1),
                (3, 2),
                (4, 2),
                (5, 2),
                (3, 3),
                (4, 3),
                (5, 3),
                (6, 1),
                (7, 1),
                (6, 2),
                (7, 2),
                (3, 5),
                (4, 5),
                (3, 6),
                (4, 6),
                (6, 4),
                (7, 4),
                (6, 5),
                (7, 5),
            }
        ),
        "start": (0, 3),
        "goal": (7, 2),
        "move_budget": 13,
        "portal_pairs": (
            {"a": (1, 2), "b": (3, 1), "color": COLOR_PORTAL_A},
            {"a": (1, 4), "b": (3, 5), "color": COLOR_PORTAL_B},
            {"a": (5, 2), "b": (6, 1), "color": COLOR_PORTAL_C},
            {"a": (5, 3), "b": (6, 4), "color": COLOR_PORTAL_D},
        ),
    },
)


def cell_origin(cell: tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    return x * CELL_SIZE, HUD_HEIGHT + y * CELL_SIZE


def draw_pattern(canvas: np.ndarray, origin: tuple[int, int], pattern: np.ndarray, color: int | None = None) -> None:
    ox, oy = origin
    if color is None:
        block = pattern
    else:
        block = np.where(pattern > 0, color, pattern)
    mask = block >= 0
    canvas[oy : oy + CELL_SIZE, ox : ox + CELL_SIZE][mask] = block[mask]


def render_level_board(spec: dict[str, object]) -> np.ndarray:
    floor_cells = spec["floor_cells"]
    canvas = np.full((SCREEN_SIZE, SCREEN_SIZE), COLOR_WATER, dtype=np.int8)

    for x, y in floor_cells:
        px, py = cell_origin((x, y))
        canvas[py : py + CELL_SIZE, px : px + CELL_SIZE] = COLOR_FLOOR

    for x, y in floor_cells:
        px, py = cell_origin((x, y))
        neighbors = {"up": (x, y - 1), "down": (x, y + 1), "left": (x - 1, y), "right": (x + 1, y)}
        if neighbors["up"] not in floor_cells:
            canvas[py, px : px + CELL_SIZE] = COLOR_BORDER
        if neighbors["down"] not in floor_cells:
            canvas[py + CELL_SIZE - 1, px : px + CELL_SIZE] = COLOR_BORDER
        if neighbors["left"] not in floor_cells:
            canvas[py : py + CELL_SIZE, px] = COLOR_BORDER
        if neighbors["right"] not in floor_cells:
            canvas[py : py + CELL_SIZE, px + CELL_SIZE - 1] = COLOR_BORDER

    for portal in spec["portal_pairs"]:
        draw_pattern(canvas, cell_origin(portal["a"]), PORTAL_PATTERN, portal["color"])
        draw_pattern(canvas, cell_origin(portal["b"]), PORTAL_PATTERN, portal["color"])

    draw_pattern(canvas, cell_origin(spec["goal"]), GOAL_PATTERN)
    return canvas


def portal_lookup(spec: dict[str, object]) -> dict[tuple[int, int], tuple[int, int]]:
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for pair in spec["portal_pairs"]:
        mapping[pair["a"]] = pair["b"]
        mapping[pair["b"]] = pair["a"]
    return mapping


class MovePipHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.capacity = 0
        self.remaining = 0
        self.border_color: int | None = None

    def configure(self, capacity: int, remaining: int, border_color: int | None) -> None:
        self.capacity = int(capacity)
        self.remaining = int(remaining)
        self.border_color = border_color

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        for idx in range(MAX_BUDGET):
            x0 = idx * 5
            x1 = x0 + 4
            color = COLOR_BLACK
            if idx < self.capacity:
                color = COLOR_GOAL if idx < self.remaining else COLOR_FAILURE
            frame[1:7, x0:x1] = color

        if self.border_color is not None:
            frame[0, :] = self.border_color
            frame[-1, :] = self.border_color
            frame[:, 0] = self.border_color
            frame[:, -1] = self.border_color

        return frame


def build_levels() -> list[Level]:
    levels: list[Level] = []
    for index, spec in enumerate(LEVEL_SPECS, start=1):
        board = Sprite(render_level_board(spec), name="board", x=0, y=0, layer=0, collidable=False, tags=["sys_static"])
        avatar_x, avatar_y = cell_origin(spec["start"])
        avatar = Sprite(AVATAR_PATTERN, name="avatar", x=avatar_x, y=avatar_y, layer=5, collidable=False)
        levels.append(
            Level(
                sprites=[board, avatar],
                grid_size=(SCREEN_SIZE, SCREEN_SIZE),
                data={"spec": spec, "portal_map": portal_lookup(spec)},
                name=f"Level {index}",
            )
        )
    return levels


levels = build_levels()


class IslandPortalDock(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = MovePipHud()
        self._remaining_moves = 0
        self._player_cell = (0, 0)
        self._portal_map: dict[tuple[int, int], tuple[int, int]] = {}
        self._level_spec = LEVEL_SPECS[0]
        camera = Camera(0, 0, SCREEN_SIZE, SCREEN_SIZE, COLOR_WATER, COLOR_WATER, [self._hud])
        super().__init__("island_portal_dock", levels, camera, False, 1, [1, 2, 3, 4, 5, 6], seed=seed)

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        self._level_spec = spec
        self._portal_map = dict(level.get_data("portal_map"))
        self._player_cell = tuple(spec["start"])
        self._remaining_moves = int(spec["move_budget"])
        # arcengine's level_reset() (and full_reset()) call set_level → on_set_level
        # *before* clearing self._state to NOT_FINISHED. Without this defensive
        # clear, RESET after a mid-play LOSE makes _sync_hud() read GAME_OVER,
        # draw the failure-red border, and persist it onto the post-RESET frame.
        self._state = GameState.NOT_FINISHED
        self._sync_avatar()
        self._sync_hud()

    def _sync_avatar(self) -> None:
        avatar = self.current_level.get_sprites_by_name("avatar")[0]
        avatar.set_position(*cell_origin(self._player_cell))

    def _terminal_border_color(self) -> int | None:
        if self._state.name == "WIN":
            return COLOR_GOAL
        if self._state.name == "GAME_OVER":
            return COLOR_FAILURE
        return None

    def _sync_hud(self) -> None:
        self._hud.configure(int(self._level_spec["move_budget"]), self._remaining_moves, self._terminal_border_color())

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

        if self._state.name in {"WIN", "GAME_OVER"}:
            self._sync_hud()
            self.complete_action()
            return

        self._remaining_moves = max(0, self._remaining_moves - 1)

        action_id = int(self.action.id.value)
        delta = ACTION_TO_DELTA.get(action_id, (0, 0))
        moved_onto_portal = False

        if action_id in ACTION_TO_DELTA:
            target = (self._player_cell[0] + delta[0], self._player_cell[1] + delta[1])
            if target in self._level_spec["floor_cells"]:
                self._player_cell = target
                moved_onto_portal = target in self._portal_map

        if moved_onto_portal:
            self._player_cell = self._portal_map[self._player_cell]

        self._sync_avatar()

        if self._player_cell == self._level_spec["goal"]:
            self.next_level()
        elif self._remaining_moves == 0:
            self.lose()

        self._sync_hud()
        self.complete_action()
