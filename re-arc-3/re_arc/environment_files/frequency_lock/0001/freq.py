from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
FRAME_BG = 0
DEFAULT_BORDER = 4
LANE_BORDER = 3
LANE_INTERIOR = 1
BAND_OUTLINE = 5
STRIKE_NORMAL = 11
STRIKE_SUCCESS = 14
STRIKE_ERROR = 8
SELECTOR = 15
PIP_FILLED = 14
PIP_EMPTY = 3
CURRENT_BORDER_NORMAL = 0
CURRENT_BORDER_SUCCESS = 14
CURRENT_BORDER_ERROR = 8
UPCOMING_BORDER = 2

RED = 8
BLUE = 9
GREEN = 14

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
SPACE = GameAction.ACTION5
CLICK = GameAction.ACTION6

PENDING_NONE = "none"
PENDING_WIN = "win"
PENDING_FAIL = "fail"
PENDING_FINAL_WIN = "final_win"

LEVEL_SPECS = (
    {
        "name": "Level 1",
        "budget": 24,
        "active_lanes": (1,),
        "selected_lane": 1,
        "target_queue": (BLUE,),
        "bands": ((1, 22, 10, BLUE),),
    },
    {
        "name": "Level 2",
        "budget": 46,
        "active_lanes": (1,),
        "selected_lane": 1,
        "target_queue": (RED, BLUE, RED),
        "bands": ((1, 17, 4, RED), (1, 23, 4, BLUE), (1, 29, 4, RED), (1, 40, 4, GREEN)),
    },
    {
        "name": "Level 3",
        "budget": 50,
        "active_lanes": (0, 1, 2),
        "selected_lane": 0,
        "target_queue": (BLUE, RED, GREEN, BLUE),
        "bands": (
            (0, 17, 4, BLUE),
            (0, 25, 4, GREEN),
            (1, 19, 4, RED),
            (1, 25, 4, BLUE),
            (2, 20, 4, GREEN),
            (2, 26, 4, RED),
        ),
    },
)

LANE_ROWS = {0: (14, 20, 15, 19), 1: (28, 34, 29, 33), 2: (42, 48, 43, 47)}


class Freq(ARCBaseGame):
    def __init__(self, seed: int = 0, **kwargs):
        self._budget_max = 0
        self._budget_remaining = 0
        self._selected_lane = 0
        self._active_lanes = ()
        self._target_queue = []
        self._bands = []
        self._success_flash_timer = 0
        self._error_flash_timer = 0
        self._claimed_flash_band = None
        self._pending_transition = PENDING_NONE
        self._last_target_color = 0
        self._board_sprite = None

        levels = [Level(name=spec["name"], grid_size=(GRID_SIZE, GRID_SIZE)) for spec in LEVEL_SPECS]
        camera = Camera(width=GRID_SIZE, height=GRID_SIZE, background=FRAME_BG, letter_box=FRAME_BG)
        super().__init__(
            game_id="frequency_lock-0001",
            levels=levels,
            camera=camera,
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
            **kwargs,
        )
        self._refresh_board()

    def on_set_level(self, _level):
        spec = LEVEL_SPECS[self.level_index]
        self._budget_max = int(spec["budget"])
        self._budget_remaining = int(spec["budget"])
        self._active_lanes = tuple(spec["active_lanes"])
        self._selected_lane = int(spec["selected_lane"])
        self._target_queue = list(spec["target_queue"])
        self._bands = [
            {"lane": lane, "x": x, "width": width, "color": color, "claimed": False}
            for lane, x, width, color in spec["bands"]
        ]
        self._success_flash_timer = 0
        self._error_flash_timer = 0
        self._claimed_flash_band = None
        self._pending_transition = PENDING_NONE
        self._last_target_color = self._target_queue[0] if self._target_queue else 0
        self._board_sprite = Sprite(
            pixels=np.full((GRID_SIZE, GRID_SIZE), FRAME_BG, dtype=np.int8),
            x=0,
            y=0,
            name=f"frequency_lock_board_{self.level_index}",
            visible=True,
            collidable=False,
            layer=0,
        )
        self.current_level.remove_all_sprites()
        self.current_level.add_sprite(self._board_sprite)
        self._refresh_board()

    def _refresh_board(self):
        if self._board_sprite is not None:
            self._board_sprite.pixels = self._render_board()

    def _clear_transient_feedback(self):
        if self._pending_transition != PENDING_NONE:
            return
        self._success_flash_timer = 0
        self._error_flash_timer = 0
        self._claimed_flash_band = None

    def _render_board(self):
        board = np.full((GRID_SIZE, GRID_SIZE), FRAME_BG, dtype=np.int8)

        border_color = DEFAULT_BORDER
        if self._pending_transition in {PENDING_WIN, PENDING_FINAL_WIN}:
            border_color = PIP_FILLED
        elif self._pending_transition == PENDING_FAIL:
            border_color = STRIKE_ERROR

        board[0, :] = border_color
        board[-1, :] = border_color
        board[:, 0] = border_color
        board[:, -1] = border_color
        board[0:9, :] = DEFAULT_BORDER
        board[0, :] = border_color
        board[-1, :] = border_color
        board[:, 0] = border_color
        board[:, -1] = border_color

        for idx in range(50):
            y = 1 if idx < 25 else 3
            x = 2 + (idx % 25)
            if idx < self._budget_max:
                board[y, x] = PIP_FILLED if idx < self._budget_remaining else PIP_EMPTY

        self._draw_target_queue(board)
        self._draw_lanes(board)
        self._draw_bands(board)
        self._draw_strike_line(board)
        self._draw_selector(board)
        self._draw_claimed_flash(board)
        return board

    def _draw_box(self, board, top, bottom, left, right, border, fill):
        board[top : bottom + 1, left : right + 1] = border
        if fill is not None and bottom - top > 1 and right - left > 1:
            board[top + 1 : bottom, left + 1 : right] = fill

    def _draw_target_queue(self, board):
        current_color = self._target_queue[0] if self._target_queue else self._last_target_color
        current_border = CURRENT_BORDER_NORMAL
        if self._pending_transition in {PENDING_WIN, PENDING_FINAL_WIN} or self._success_flash_timer:
            current_border = CURRENT_BORDER_SUCCESS
        elif self._pending_transition == PENDING_FAIL or self._error_flash_timer:
            current_border = CURRENT_BORDER_ERROR

        self._draw_box(board, 1, 7, 35, 41, current_border, None)
        if current_color:
            board[2:7, 36:41] = current_color

        upcoming_boxes = ((2, 6, 44, 48), (2, 6, 51, 55), (2, 6, 58, 62))
        for idx, (top, bottom, left, right) in enumerate(upcoming_boxes, start=1):
            upcoming = self._target_queue[idx] if idx < len(self._target_queue) else None
            self._draw_box(board, top, bottom, left, right, UPCOMING_BORDER, None)
            if upcoming is not None:
                board[top + 1 : bottom, left + 1 : right] = upcoming

    def _draw_lanes(self, board):
        for lane in self._active_lanes:
            top, bottom, inner_top, inner_bottom = LANE_ROWS[lane]
            board[top : bottom + 1, 8:61] = LANE_BORDER
            board[inner_top : inner_bottom + 1, 9:60] = LANE_INTERIOR

    def _draw_band(self, board, band):
        _, _, inner_top, inner_bottom = LANE_ROWS[band["lane"]]
        start_x = max(9, int(band["x"]))
        end_x = min(59, int(band["x"]) + int(band["width"]) - 1)
        if start_x > end_x:
            return
        board[inner_top, start_x : end_x + 1] = BAND_OUTLINE
        board[inner_bottom, start_x : end_x + 1] = BAND_OUTLINE
        board[inner_top + 1 : inner_bottom, start_x : end_x + 1] = int(band["color"])

    def _draw_bands(self, board):
        for band in self._bands:
            if band["claimed"]:
                continue
            self._draw_band(board, band)

    def _draw_claimed_flash(self, board):
        if self._claimed_flash_band is not None:
            self._draw_band(board, self._claimed_flash_band)

    def _draw_strike_line(self, board):
        top = min(LANE_ROWS[lane][0] for lane in self._active_lanes)
        bottom = max(LANE_ROWS[lane][1] for lane in self._active_lanes)
        color = STRIKE_NORMAL
        if self._pending_transition in {PENDING_WIN, PENDING_FINAL_WIN} or self._success_flash_timer:
            color = STRIKE_SUCCESS
        elif self._pending_transition == PENDING_FAIL or self._error_flash_timer:
            color = STRIKE_ERROR
        board[top : bottom + 1, 16:18] = color

    def _draw_selector(self, board):
        if len(self._active_lanes) <= 1:
            return
        top, bottom, _, _ = LANE_ROWS[self._selected_lane]
        left = 14
        right = 19
        board[top, left : left + 2] = SELECTOR
        board[top, right - 1 : right + 1] = SELECTOR
        board[bottom, left : left + 2] = SELECTOR
        board[bottom, right - 1 : right + 1] = SELECTOR
        board[top : bottom + 1, left] = SELECTOR
        board[top : bottom + 1, right] = SELECTOR

    def _consume_budget(self):
        if self._budget_remaining > 0:
            self._budget_remaining -= 1

    def _has_future_target_band(self):
        if not self._target_queue:
            return False
        wanted = self._target_queue[0]
        for band in self._bands:
            if band["claimed"] or band["color"] != wanted:
                continue
            if int(band["x"]) + int(band["width"]) - 1 >= 16:
                return True
        return False

    def _band_overlaps_strike(self, band):
        return int(band["x"]) <= 17 and (int(band["x"]) + int(band["width"]) - 1) >= 16

    def _resolve_space(self):
        self._consume_budget()
        self._last_target_color = self._target_queue[0] if self._target_queue else self._last_target_color
        lane = self._active_lanes[0] if len(self._active_lanes) == 1 else self._selected_lane
        wanted = self._target_queue[0]
        candidates = [
            band
            for band in self._bands
            if (not band["claimed"])
            and band["lane"] == lane
            and band["color"] == wanted
            and self._band_overlaps_strike(band)
        ]
        candidates.sort(key=lambda band: int(band["x"]))

        if candidates:
            claimed = candidates[0]
            claimed["claimed"] = True
            self._claimed_flash_band = {
                "lane": claimed["lane"],
                "x": int(claimed["x"]),
                "width": int(claimed["width"]),
                "color": int(claimed["color"]),
                "claimed": True,
            }
            self._success_flash_timer = 1
            self._error_flash_timer = 0
            self._target_queue.pop(0)
            if self._target_queue:
                self._last_target_color = self._target_queue[0]
        else:
            self._error_flash_timer = 1
            self._success_flash_timer = 0

    def _resolve_shift(self):
        self._consume_budget()
        for band in self._bands:
            if not band["claimed"]:
                band["x"] -= 1

    def _resolve_lane_change(self, delta):
        if len(self._active_lanes) <= 1:
            return False
        new_lane = min(max(self._selected_lane + delta, self._active_lanes[0]), self._active_lanes[-1])
        if new_lane == self._selected_lane:
            return False
        self._selected_lane = new_lane
        self._consume_budget()
        return True

    def _enter_win_transition(self):
        self._pending_transition = PENDING_FINAL_WIN if self.is_last_level() else PENDING_WIN

    def _enter_fail_transition(self):
        self._pending_transition = PENDING_FAIL
        self._success_flash_timer = 0
        self._error_flash_timer = 1

    def _acknowledge_pending_transition(self):
        if self._pending_transition == PENDING_FAIL:
            self.lose()
            self.complete_action()
            return
        if self._pending_transition == PENDING_WIN:
            self._pending_transition = PENDING_NONE
            self._success_flash_timer = 0
            self._error_flash_timer = 0
            self._claimed_flash_band = None
            # End the action here so perform_action's loop doesn't run step()
            # a second time on the new level with the transition-triggering
            # action still in self.action. Without complete_action() the next
            # action gets processed on the new level, and the player-visible
            # "level N initial frame" then depends on which action triggered
            # the transition — making mid-play RESET unable to restore it.
            self.next_level()
            self.complete_action()
            return
        if self._pending_transition == PENDING_FINAL_WIN:
            self._pending_transition = PENDING_NONE
            self._success_flash_timer = 0
            self._error_flash_timer = 0
            self._claimed_flash_band = None
            self.next_level()
            self._refresh_board()
            self.complete_action()

    def step(self):
        if self._pending_transition != PENDING_NONE:
            self._acknowledge_pending_transition()
            return

        self._clear_transient_feedback()

        action = self.action.id
        budgeted = False
        if action in {LEFT, RIGHT}:
            budgeted = True
            self._resolve_shift()
        elif action == SPACE:
            budgeted = True
            self._resolve_space()
        elif action == UP:
            budgeted = self._resolve_lane_change(-1)
        elif action == DOWN:
            budgeted = self._resolve_lane_change(1)
        elif action == CLICK:
            pass

        if not self._target_queue:
            self._enter_win_transition()
            self._refresh_board()
            self.complete_action()
            return

        if budgeted and self._budget_remaining == 0:
            self._enter_fail_transition()
            self._refresh_board()
            self.complete_action()
            return

        if budgeted and not self._has_future_target_band():
            self._enter_fail_transition()
            self._refresh_board()
            self.complete_action()
            return

        self._refresh_board()
        self.complete_action()
