from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
BOARD_SIZE = 14
CELL_SIZE = 4
BOARD_ORIGIN_X = 4
BOARD_ORIGIN_Y = 8
HUD_BAR_X0 = 4
HUD_BAR_X1 = 60
HUD_BAR_Y0 = 2
HUD_BAR_Y1 = 6

COLOR_FLOOR = 0
COLOR_SHADOW_OUTER = 2
COLOR_SHADOW_INNER = 3
COLOR_FRAME = 4
COLOR_AVATAR_CORE = 9
COLOR_AVATAR_HIGHLIGHT = 10
COLOR_EXIT_FRAME = 11
COLOR_WARNING = 12
COLOR_SUCCESS = 14
COLOR_FAILURE = 8

AVATAR_PATTERN = np.array([[0, 10, 10, 0], [10, 9, 9, 10], [10, 9, 9, 10], [0, 10, 10, 0]], dtype=np.int8)
SHADOW_PATTERN = np.array([[2, 2, 2, 2], [2, 3, 3, 2], [2, 3, 3, 2], [2, 2, 2, 2]], dtype=np.int8)
EXIT_PATTERN = np.array([[11, 11, 11, 11], [11, 0, 0, 11], [11, 0, 0, 11], [11, 12, 12, 11]], dtype=np.int8)

LEVEL_MAPS = (
    (
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
        "..S.....E.....",
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
        "..............",
    ),
    (
        "E....#########",
        ".....#########",
        ".....#########",
        ".....#########",
        ".........#####",
        "####......####",
        "#####.....####",
        "#####.....####",
        "#####........#",
        "########......",
        "#########.....",
        "#########.....",
        "#########....S",
        "#########.....",
    ),
    (
        "#########....E",
        "#########.....",
        "#########.....",
        "#########.....",
        "#########.....",
        "#########..###",
        "#########..###",
        "#####.....####",
        "#####..#..####",
        "####......####",
        ".....#########",
        "....##########",
        "....##########",
        "S...##########",
    ),
)
LEVEL_BUDGETS = (20, 80, 84)


class LevelSpec:
    def __init__(
        self,
        rows: tuple[str, ...],
        start_pos: tuple[int, int],
        exit_pos: tuple[int, int],
        initial_shadow_set: frozenset[tuple[int, int]],
        budget: int,
    ) -> None:
        self.rows = rows
        self.start_pos = start_pos
        self.exit_pos = exit_pos
        self.initial_shadow_set = initial_shadow_set
        self.budget = int(budget)


def _parse_level(rows: tuple[str, ...], budget: int) -> LevelSpec:
    if len(rows) != BOARD_SIZE:
        raise ValueError(f"Expected {BOARD_SIZE} rows, got {len(rows)}.")

    start_pos: tuple[int, int] | None = None
    exit_pos: tuple[int, int] | None = None
    initial_shadow_set: set[tuple[int, int]] = set()

    for y, row in enumerate(rows):
        if len(row) != BOARD_SIZE:
            raise ValueError(f"Expected row width {BOARD_SIZE}, got {len(row)} at y={y}.")
        for x, tile in enumerate(row):
            if tile == "S":
                if start_pos is not None:
                    raise ValueError("Each level must define exactly one start.")
                start_pos = (x, y)
            elif tile == "E":
                if exit_pos is not None:
                    raise ValueError("Each level must define exactly one exit.")
                exit_pos = (x, y)
            elif tile == "#":
                initial_shadow_set.add((x, y))
            elif tile != ".":
                raise ValueError(f"Unsupported tile {tile!r} at {(x, y)}.")

    if start_pos is None or exit_pos is None:
        raise ValueError("Each level must define exactly one start and exit.")
    if start_pos in initial_shadow_set or exit_pos in initial_shadow_set:
        raise ValueError("Start and exit cannot overlap pre-placed shadows.")

    return LevelSpec(
        rows=rows,
        start_pos=start_pos,
        exit_pos=exit_pos,
        initial_shadow_set=frozenset(initial_shadow_set),
        budget=int(budget),
    )


LEVEL_SPECS = tuple(_parse_level(rows, budget) for rows, budget in zip(LEVEL_MAPS, LEVEL_BUDGETS, strict=True))


class ShadowStep(ARCBaseGame):
    def __init__(self) -> None:
        self._canvas_sprite: Sprite | None = None
        self._level_spec: LevelSpec | None = None
        self._shadow_set: set[tuple[int, int]] = set()
        self._avatar_pos = (0, 0)
        self._remaining_actions = 0
        self._route_score = 0
        self._flash_type = "none"
        self._clear_flash_on_next_step = False
        self._terminal_frame_tint = COLOR_FRAME
        self._pending_transition_flash = "none"

        levels = [
            Level(
                name=f"Shadow Step {index + 1}",
                grid_size=(GRID_SIZE, GRID_SIZE),
                sprites=[Sprite(np.full((GRID_SIZE, GRID_SIZE), COLOR_FRAME, dtype=np.int8), name="canvas", layer=0)],
                data={"level_spec": level_spec, "level_index": index},
            )
            for index, level_spec in enumerate(LEVEL_SPECS)
        ]

        super().__init__(
            game_id="shadow_step-0001",
            levels=levels,
            camera=Camera(0, 0, GRID_SIZE, GRID_SIZE, COLOR_FRAME, COLOR_FRAME),
            debug=False,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
        )

    def on_set_level(self, level: Level) -> None:
        self._canvas_sprite = level.get_sprites_by_name("canvas")[0]
        self._level_spec = level.get_data("level_spec")
        self._route_score = int(level.get_data("level_index") or 0)
        self._terminal_frame_tint = COLOR_FRAME
        flash = self._pending_transition_flash
        self._pending_transition_flash = "none"
        self._reset_runtime_state(flash_type=flash)

    def _reset_runtime_state(self, *, flash_type: str) -> None:
        if self._level_spec is None:
            raise RuntimeError("Level spec is not initialized.")
        self._avatar_pos = self._level_spec.start_pos
        self._shadow_set = set(self._level_spec.initial_shadow_set)
        self._remaining_actions = int(self._level_spec.budget)
        self._flash_type = flash_type
        self._clear_flash_on_next_step = flash_type in {"fail", "success"}
        self._render_board()

    def _cell_to_pixel(self, gx: int, gy: int) -> tuple[int, int]:
        return BOARD_ORIGIN_X + gx * CELL_SIZE, BOARD_ORIGIN_Y + gy * CELL_SIZE

    def _draw_pattern(self, frame: np.ndarray, gx: int, gy: int, pattern: np.ndarray) -> None:
        px, py = self._cell_to_pixel(gx, gy)
        frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = pattern

    def _frame_tint(self) -> int:
        if self._terminal_frame_tint != COLOR_FRAME:
            return self._terminal_frame_tint
        if self._flash_type == "fail":
            return COLOR_FAILURE
        if self._flash_type == "success":
            return COLOR_SUCCESS
        return COLOR_FRAME

    def _render_board(self) -> None:
        if self._canvas_sprite is None or self._level_spec is None:
            raise RuntimeError("Canvas sprite is not initialized.")

        frame = np.full((GRID_SIZE, GRID_SIZE), self._frame_tint(), dtype=np.int8)
        frame[
            BOARD_ORIGIN_Y : BOARD_ORIGIN_Y + BOARD_SIZE * CELL_SIZE,
            BOARD_ORIGIN_X : BOARD_ORIGIN_X + BOARD_SIZE * CELL_SIZE,
        ] = COLOR_FLOOR
        self._render_budget_bar(frame)

        for gx, gy in self._shadow_set:
            self._draw_pattern(frame, gx, gy, SHADOW_PATTERN)

        exit_x, exit_y = self._level_spec.exit_pos
        self._draw_pattern(frame, exit_x, exit_y, EXIT_PATTERN)

        avatar_x, avatar_y = self._avatar_pos
        self._draw_pattern(frame, avatar_x, avatar_y, AVATAR_PATTERN)
        self._canvas_sprite.pixels[:, :] = frame

    def _render_budget_bar(self, frame: np.ndarray) -> None:
        if self._level_spec is None:
            raise RuntimeError("Level spec is not initialized.")

        frame[HUD_BAR_Y0:HUD_BAR_Y1, HUD_BAR_X0:HUD_BAR_X1] = COLOR_SHADOW_OUTER
        total_width = HUD_BAR_X1 - HUD_BAR_X0
        fill_ratio = self._remaining_actions / max(1, self._level_spec.budget)
        fill_width = max(0, min(total_width, round(total_width * fill_ratio)))
        if fill_width <= 0:
            return

        fill_color = COLOR_WARNING if self._remaining_actions * 4 <= self._level_spec.budget else COLOR_SUCCESS
        frame[HUD_BAR_Y0:HUD_BAR_Y1, HUD_BAR_X0 : HUD_BAR_X0 + fill_width] = fill_color

    def _consume_action(self) -> None:
        self._remaining_actions -= 1

    def _restart_current_level(self, *, flash_type: str) -> None:
        self._reset_runtime_state(flash_type=flash_type)

    def _complete_level(self) -> None:
        if self.is_last_level():
            self._terminal_frame_tint = COLOR_SUCCESS
            self._flash_type = "none"
            self._clear_flash_on_next_step = False
            self._render_board()
            self.next_level()
            return

        # Frame 1: render the old level showing the avatar at the exit
        # cell. Frame 2 is emitted by arcengine's auto-transition: it
        # loops once more, calls _really_set_next_level → on_set_level,
        # which consumes _pending_transition_flash and renders the new
        # level's initial state with the success flash.
        self._render_board()
        self._pending_transition_flash = "success"
        self.next_level()

    def _handle_move(self, delta_x: int, delta_y: int) -> None:
        if self._level_spec is None:
            raise RuntimeError("Level spec is not initialized.")

        current_x, current_y = self._avatar_pos
        target = (current_x + delta_x, current_y + delta_y)

        if not (0 <= target[0] < BOARD_SIZE and 0 <= target[1] < BOARD_SIZE):
            self._consume_action()
            if self._remaining_actions <= 0:
                self.lose()
            else:
                self._render_board()
            return

        if target in self._shadow_set:
            self.lose()
            return

        self._shadow_set.add(self._avatar_pos)
        self._avatar_pos = target
        self._consume_action()

        if target == self._level_spec.exit_pos:
            self._complete_level()
            return

        if self._remaining_actions <= 0:
            self.lose()
            return

        self._render_board()

    def step(self) -> None:
        if getattr(self.action.id, "value", self.action.id) == 0 and self.level_index > 0:
            # arcengine's perform_action loop runs step() once after
            # handle_reset; without this branch the simulation would
            # advance a tick. Level arrival via next_level leaves the
            # game with a transient success-flash (set by _complete_level),
            # so to make mid-play RESET land on the same frame we emulate
            # that post-arrival state here — set flash_type="success" and
            # re-render. The flash is cleared on the player's next action
            # by the usual _clear_flash_on_next_step mechanism, matching
            # the arrival flow exactly. Level 0 falls through to preserve
            # env.reset()'s observation and the stored DSL trace.
            self._flash_type = "success"
            self._clear_flash_on_next_step = True
            self._render_board()
            self.complete_action()
            return

        if self._clear_flash_on_next_step:
            self._flash_type = "none"
            self._clear_flash_on_next_step = False

        action_id = int(self.action.id.value)

        if action_id == int(GameAction.ACTION5.value):
            self._restart_current_level(flash_type="none")
            self.complete_action()
            return

        if action_id == int(GameAction.ACTION6.value):
            self._render_board()
            self.complete_action()
            return

        if action_id == int(GameAction.ACTION1.value):
            self._handle_move(0, -1)
        elif action_id == int(GameAction.ACTION2.value):
            self._handle_move(0, 1)
        elif action_id == int(GameAction.ACTION3.value):
            self._handle_move(-1, 0)
        elif action_id == int(GameAction.ACTION4.value):
            self._handle_move(1, 0)
        else:
            self._render_board()

        self.complete_action()
