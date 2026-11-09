from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GRID_SIZE = 64
PLAYFIELD_ORIGIN_X = 4
PLAYFIELD_ORIGIN_Y = 4
PLAYFIELD_WIDTH = 14
PLAYFIELD_HEIGHT = 12
CELL_SIZE = 4
PLAYFIELD_PIXEL_WIDTH = PLAYFIELD_WIDTH * CELL_SIZE
PLAYFIELD_PIXEL_HEIGHT = PLAYFIELD_HEIGHT * CELL_SIZE
UI_TOP = 52
UI_BAR_X = 4
UI_BAR_Y = 56
UI_BAR_WIDTH = 56
UI_BAR_HEIGHT = 5
STATUS_X = 60
STATUS_Y = 56
STATUS_SIZE = 4

COLOR_BG = 0
COLOR_BOARD = 2
COLOR_SPENT = 3
COLOR_BLOCK_SHADE = 4
COLOR_FAIL = 8
COLOR_WATER = 9
COLOR_WATER_LIGHT = 10
COLOR_BUDGET = 11
COLOR_BLOCK = 12
COLOR_WIN = 14

Dir = Literal["DOWN", "LEFT", "RIGHT"]
_DIR_PRIORITY: dict[Dir, int] = {"RIGHT": 0, "LEFT": 1, "DOWN": 2}


class BucketSpec(NamedTuple):
    idx: int
    x: int
    y: int

    @property
    def receiver_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset({(self.x, self.y), (self.x + 1, self.y)})

    @property
    def body_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset({(self.x, self.y), (self.x + 1, self.y), (self.x, self.y + 1), (self.x + 1, self.y + 1)})


class LevelSpec(NamedTuple):
    name: str
    budget: int
    source_col: int
    buckets: tuple[BucketSpec, ...]
    solution_blocks: tuple[tuple[int, int], ...]


class WaterFront(NamedTuple):
    x: int
    y: int
    direction: Dir


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        name="shrink-to-fit-1",
        budget=75,
        source_col=0,
        buckets=(BucketSpec(0, 7, 10),),
        solution_blocks=((0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9)),
    ),
    LevelSpec(
        name="shrink-to-fit-2",
        budget=75,
        source_col=7,
        buckets=(BucketSpec(0, 2, 10), BucketSpec(1, 11, 10)),
        solution_blocks=((7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (8, 4), (9, 5), (10, 6), (11, 7)),
    ),
    LevelSpec(
        name="shrink-to-fit-3",
        budget=84,
        source_col=6,
        buckets=(BucketSpec(0, 1, 10), BucketSpec(1, 6, 10), BucketSpec(2, 11, 10)),
        solution_blocks=((6, 3), (5, 4), (4, 5), (3, 6), (7, 4), (8, 5), (9, 7), (8, 8), (7, 9), (10, 8), (11, 9)),
    ),
)


def _cell_rect(cell_x: int, cell_y: int) -> tuple[int, int, int, int]:
    px = PLAYFIELD_ORIGIN_X + cell_x * CELL_SIZE
    py = PLAYFIELD_ORIGIN_Y + cell_y * CELL_SIZE
    return px, py, CELL_SIZE, CELL_SIZE


class Sf01(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._level_specs = LEVEL_SPECS
        self._level_budget = 0
        self._budget_remaining = 0
        self._route_score = 0
        self._phase: Literal["build", "flow"] = "build"
        self._blocks: set[tuple[int, int]] = set()
        self._wet: set[tuple[int, int]] = set()
        self._active_fronts: list[WaterFront] = []
        self._filled_buckets: set[int] = set()
        self._frame_sprite: Sprite | None = None
        self._bucket_by_receiver: dict[tuple[int, int], int] = {}
        self._bucket_body_cells: set[tuple[int, int]] = set()
        self._source_cell: tuple[int, int] = (0, 0)

        levels = [self._make_level(idx, spec) for idx, spec in enumerate(self._level_specs)]
        camera = Camera(width=GRID_SIZE, height=GRID_SIZE, background=COLOR_BG)
        super().__init__(
            game_id="sf01",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def _make_level(self, idx: int, spec: LevelSpec) -> Level:
        frame = np.full((GRID_SIZE, GRID_SIZE), COLOR_BG, dtype=np.int8)
        sprite = Sprite(frame, name=f"level_{idx}_frame", x=0, y=0, layer=0, visible=True, collidable=False)
        return Level(
            name=spec.name,
            grid_size=(GRID_SIZE, GRID_SIZE),
            sprites=[sprite],
            data={"spec_idx": idx, "budget": spec.budget, "source_col": spec.source_col},
        )

    def on_set_level(self, level: Level) -> None:
        spec_idx = int(level.get_data("spec_idx") or 0)
        spec = self._level_specs[spec_idx]
        self._level_budget = int(spec.budget)
        self._budget_remaining = int(spec.budget)
        self._route_score = 0
        self._phase = "build"
        self._blocks = set()
        self._wet = set()
        self._active_fronts = []
        self._filled_buckets = set()
        self._source_cell = (int(spec.source_col), 0)
        self._bucket_by_receiver = {}
        self._bucket_body_cells = set()
        for bucket in spec.buckets:
            for cell in bucket.receiver_cells:
                self._bucket_by_receiver[cell] = bucket.idx
            self._bucket_body_cells.update(bucket.body_cells)
        frame_sprites = level.get_sprites_by_name(f"level_{spec_idx}_frame")
        self._frame_sprite = frame_sprites[0] if frame_sprites else None
        self._render_frame()

    def _is_inside(self, x: int, y: int) -> bool:
        return 0 <= x < PLAYFIELD_WIDTH and 0 <= y < PLAYFIELD_HEIGHT

    def _bucket_receiver_idx(self, x: int, y: int) -> int | None:
        return self._bucket_by_receiver.get((x, y))

    def _is_buildable(self, x: int, y: int) -> bool:
        return self._is_inside(x, y) and (x, y) != self._source_cell and (x, y) not in self._bucket_body_cells

    def _is_enterable(self, x: int, y: int) -> bool:
        if not self._is_inside(x, y):
            return False
        if (x, y) == self._source_cell:
            return False
        if (x, y) in self._blocks:
            return False
        if (x, y) in self._wet:
            return False
        bucket_idx = self._bucket_receiver_idx(x, y)
        if bucket_idx is not None:
            return bucket_idx not in self._filled_buckets
        return (x, y) not in self._bucket_body_cells

    def _click_to_cell(self, x: int, y: int) -> tuple[int, int] | None:
        if not (
            PLAYFIELD_ORIGIN_X <= x < PLAYFIELD_ORIGIN_X + PLAYFIELD_PIXEL_WIDTH
            and PLAYFIELD_ORIGIN_Y <= y < PLAYFIELD_ORIGIN_Y + PLAYFIELD_PIXEL_HEIGHT
        ):
            return None
        cell_x = (x - PLAYFIELD_ORIGIN_X) // CELL_SIZE
        cell_y = (y - PLAYFIELD_ORIGIN_Y) // CELL_SIZE
        if not self._is_inside(cell_x, cell_y):
            return None
        return int(cell_x), int(cell_y)

    def _toggle_block(self, x: int, y: int) -> None:
        if not self._is_buildable(x, y):
            return
        cell = (x, y)
        if cell in self._blocks:
            self._blocks.remove(cell)
        else:
            self._blocks.add(cell)

    def _start_flow(self) -> None:
        if self._phase != "build":
            return
        self._phase = "flow"
        self._active_fronts = [WaterFront(self._source_cell[0], self._source_cell[1], "DOWN")]
        self._wet = {self._source_cell}

    def _record_front(
        self, candidate_fronts: dict[tuple[int, int], WaterFront], bucket_hits: set[int], x: int, y: int, direction: Dir
    ) -> None:
        if not self._is_inside(x, y):
            return
        bucket_idx = self._bucket_receiver_idx(x, y)
        if bucket_idx is not None:
            if bucket_idx not in self._filled_buckets:
                bucket_hits.add(bucket_idx)
            return
        if not self._is_enterable(x, y):
            return
        key = (x, y)
        prior = candidate_fronts.get(key)
        if prior is None or _DIR_PRIORITY[direction] > _DIR_PRIORITY[prior.direction]:
            candidate_fronts[key] = WaterFront(x, y, direction)

    def _advance_down_front(
        self, front: WaterFront, candidate_fronts: dict[tuple[int, int], WaterFront], bucket_hits: set[int]
    ) -> None:
        below = (front.x, front.y + 1)
        if not self._is_inside(*below):
            return
        if self._bucket_receiver_idx(*below) is not None:
            self._record_front(candidate_fronts, bucket_hits, below[0], below[1], "DOWN")
            return
        if self._is_enterable(*below):
            self._record_front(candidate_fronts, bucket_hits, below[0], below[1], "DOWN")
            return

        side_candidates: tuple[tuple[int, Dir], ...] = ((front.x - 1, "LEFT"), (front.x + 1, "RIGHT"))
        for next_x, direction in side_candidates:
            if not self._is_inside(next_x, front.y):
                continue
            self._record_front(candidate_fronts, bucket_hits, next_x, front.y, direction)

    def _advance_side_front(
        self, front: WaterFront, candidate_fronts: dict[tuple[int, int], WaterFront], bucket_hits: set[int]
    ) -> None:
        below = (front.x, front.y + 1)
        if self._is_inside(*below):
            if self._bucket_receiver_idx(*below) is not None:
                self._record_front(candidate_fronts, bucket_hits, below[0], below[1], "DOWN")
                return
            if self._is_enterable(*below):
                self._record_front(candidate_fronts, bucket_hits, below[0], below[1], "DOWN")
                return

        step_x = front.x - 1 if front.direction == "LEFT" else front.x + 1
        if not self._is_inside(step_x, front.y):
            return
        self._record_front(candidate_fronts, bucket_hits, step_x, front.y, front.direction)

    def _advance_flow(self) -> None:
        if not self._active_fronts:
            return

        candidate_fronts: dict[tuple[int, int], WaterFront] = {}
        bucket_hits: set[int] = set()
        prior_fronts = list(self._active_fronts)

        for front in prior_fronts:
            if front.direction == "DOWN":
                self._advance_down_front(front, candidate_fronts, bucket_hits)
            else:
                self._advance_side_front(front, candidate_fronts, bucket_hits)

        self._filled_buckets.update(bucket_hits)
        self._wet.update(candidate_fronts.keys())
        self._active_fronts = list(candidate_fronts.values())

    def _status_color(self) -> int:
        state_name = str(getattr(getattr(self, "_state", None), "name", ""))
        if state_name == "WIN":
            return COLOR_WIN
        if state_name == "LOSE":
            return COLOR_FAIL
        return COLOR_BLOCK if self._phase == "build" else COLOR_WATER

    def _budget_segments(self) -> int:
        if self._level_budget <= 0:
            return 0
        used = self._level_budget - self._budget_remaining
        remaining_ratio = self._budget_remaining / float(self._level_budget)
        if used <= 0:
            return UI_BAR_WIDTH
        if self._budget_remaining <= 0:
            return 0
        return max(0, min(UI_BAR_WIDTH, round(remaining_ratio * UI_BAR_WIDTH)))

    def _draw_block(self, frame: np.ndarray, cell_x: int, cell_y: int) -> None:
        px, py, w, h = _cell_rect(cell_x, cell_y)
        frame[py : py + h, px : px + w] = np.int8(COLOR_BLOCK)
        frame[py + 1 : py + h - 1, px + 1 : px + w - 1] = np.int8(COLOR_BLOCK_SHADE)
        frame[py, px : px + w] = np.int8(COLOR_BLOCK)
        frame[py : py + h, px] = np.int8(COLOR_BLOCK)

    def _draw_source(self, frame: np.ndarray, source_col: int) -> None:
        px = PLAYFIELD_ORIGIN_X + source_col * CELL_SIZE
        frame[1, px + 1 : px + 3] = np.int8(COLOR_WATER)
        frame[2, px : px + 4] = np.int8(COLOR_WATER)
        frame[3, px + 1 : px + 3] = np.int8(COLOR_WATER_LIGHT)
        frame[4, px + 1 : px + 3] = np.int8(COLOR_WATER_LIGHT)

    def _draw_bucket(self, frame: np.ndarray, bucket: BucketSpec, filled: bool) -> None:
        px = PLAYFIELD_ORIGIN_X + bucket.x * CELL_SIZE
        py = PLAYFIELD_ORIGIN_Y + bucket.y * CELL_SIZE
        frame[py : py + 8, px : px + 8] = np.int8(COLOR_BG)
        frame[py : py + 8, px : px + 1] = np.int8(COLOR_WIN)
        frame[py : py + 8, px + 7 : px + 8] = np.int8(COLOR_WIN)
        frame[py + 7 : py + 8, px : px + 8] = np.int8(COLOR_WIN)
        frame[py + 1 : py + 7, px + 1 : px + 7] = np.int8(COLOR_WATER_LIGHT if filled else COLOR_BG)
        frame[py + 5 : py + 7, px + 1 : px + 7] = np.int8(COLOR_WATER_LIGHT if filled else COLOR_BG)
        frame[py, px : px + 2] = np.int8(COLOR_WIN)
        frame[py, px + 6 : px + 8] = np.int8(COLOR_WIN)

    def _draw_water_cell(self, frame: np.ndarray, x: int, y: int, color: int) -> None:
        px, py, w, h = _cell_rect(x, y)
        frame[py + 1 : py + h - 1, px + 1 : px + w - 1] = np.int8(color)

    def _render_frame(self) -> None:
        if self._frame_sprite is None:
            return

        spec_idx = int(self.current_level.get_data("spec_idx") or 0)
        spec = self._level_specs[spec_idx]
        frame = np.full((GRID_SIZE, GRID_SIZE), COLOR_BG, dtype=np.int8)

        frame[PLAYFIELD_ORIGIN_Y - 1 : PLAYFIELD_ORIGIN_Y + PLAYFIELD_PIXEL_HEIGHT + 1, PLAYFIELD_ORIGIN_X - 1] = (
            np.int8(COLOR_BOARD)
        )
        frame[
            PLAYFIELD_ORIGIN_Y - 1 : PLAYFIELD_ORIGIN_Y + PLAYFIELD_PIXEL_HEIGHT + 1,
            PLAYFIELD_ORIGIN_X + PLAYFIELD_PIXEL_WIDTH,
        ] = np.int8(COLOR_BOARD)
        frame[PLAYFIELD_ORIGIN_Y - 1, PLAYFIELD_ORIGIN_X - 1 : PLAYFIELD_ORIGIN_X + PLAYFIELD_PIXEL_WIDTH + 1] = (
            np.int8(COLOR_BOARD)
        )
        frame[
            PLAYFIELD_ORIGIN_Y + PLAYFIELD_PIXEL_HEIGHT,
            PLAYFIELD_ORIGIN_X - 1 : PLAYFIELD_ORIGIN_X + PLAYFIELD_PIXEL_WIDTH + 1,
        ] = np.int8(COLOR_BOARD)

        for bucket in spec.buckets:
            self._draw_bucket(frame, bucket, bucket.idx in self._filled_buckets)

        for block_x, block_y in sorted(self._blocks):
            self._draw_block(frame, block_x, block_y)

        for wet_x, wet_y in sorted(self._wet):
            if (wet_x, wet_y) == self._source_cell:
                continue
            if self._bucket_receiver_idx(wet_x, wet_y) is not None:
                continue
            self._draw_water_cell(frame, wet_x, wet_y, COLOR_WATER_LIGHT)

        for front in self._active_fronts:
            if self._bucket_receiver_idx(front.x, front.y) is None:
                self._draw_water_cell(frame, front.x, front.y, COLOR_WATER)

        self._draw_source(frame, spec.source_col)

        frame[UI_TOP:GRID_SIZE, :] = np.maximum(frame[UI_TOP:GRID_SIZE, :], np.int8(COLOR_BG))
        remaining_segments = self._budget_segments()
        frame[UI_BAR_Y : UI_BAR_Y + UI_BAR_HEIGHT, UI_BAR_X : UI_BAR_X + UI_BAR_WIDTH] = np.int8(COLOR_SPENT)
        if remaining_segments > 0:
            frame[UI_BAR_Y : UI_BAR_Y + UI_BAR_HEIGHT, UI_BAR_X : UI_BAR_X + remaining_segments] = np.int8(COLOR_BUDGET)
        frame[STATUS_Y : STATUS_Y + STATUS_SIZE, STATUS_X : STATUS_X + STATUS_SIZE] = np.int8(self._status_color())

        self._frame_sprite.pixels = frame

    def _all_buckets_filled(self) -> bool:
        spec_idx = int(self.current_level.get_data("spec_idx") or 0)
        return len(self._filled_buckets) == len(self._level_specs[spec_idx].buckets)

    def _handle_post_action_state(self) -> None:
        if self._all_buckets_filled():
            self._route_score += 1
            self.next_level()
            return
        if self._budget_remaining <= 0:
            self.lose()
            return
        if self._phase == "flow" and not self._active_fronts:
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

        payload = self.action.data if isinstance(self.action.data, dict) else {}

        self._budget_remaining -= 1

        if self._phase == "build":
            if self.action.id == GameAction.ACTION6:
                cell = self._click_to_cell(int(payload.get("x", -1)), int(payload.get("y", -1)))
                if cell is not None:
                    self._toggle_block(*cell)
            elif self.action.id == GameAction.ACTION5:
                self._start_flow()
        elif self.action.id == GameAction.ACTION5:
            self._advance_flow()

        self._handle_post_action_state()
        self._render_frame()
        self.complete_action()
