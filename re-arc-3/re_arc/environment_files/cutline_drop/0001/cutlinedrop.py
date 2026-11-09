from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

LOGICAL_SIZE = 16
CELL_SIZE = 4
RENDER_SIZE = LOGICAL_SIZE * CELL_SIZE
UI_ROWS = 2

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

PLAY_BACKGROUND = 0
UI_BACKGROUND = 1

TERRAIN_CELL = [[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]
ROPE_CELL = [[0, 10, 10, 0], [0, 9, 9, 0], [0, 9, 9, 0], [0, 10, 10, 0]]
ROPE_ANCHOR_CELL = [[4, 4, 4, 4], [4, 9, 9, 4], [0, 9, 9, 0], [0, 10, 10, 0]]
WALKER_CELL = [[0, 14, 14, 0], [14, 5, 5, 14], [14, 14, 14, 14], [0, 14, 14, 0]]
GOAL_SUPPORT_CELL = [[11, 6, 11, 6], [6, 11, 6, 11], [11, 6, 11, 6], [6, 11, 6, 11]]
GOAL_FLAG_CELL = [[0, 11, 11, 0], [0, 11, 6, 0], [0, 11, 6, 0], [0, 0, 6, 0]]
REMAINING_PIP = [[1, 1, 1, 1], [1, 14, 14, 1], [1, 14, 14, 1], [1, 1, 1, 1]]
SPENT_PIP = [[1, 1, 1, 1], [1, 3, 3, 1], [1, 3, 3, 1], [1, 1, 1, 1]]


class BarSpec:
    def __init__(self, *, name: str, x0: int, x1: int, y: int, rope_x: int) -> None:
        self.name = name
        self.x0 = x0
        self.x1 = x1
        self.y = y
        self.rope_x = rope_x


class LevelSpec:
    def __init__(
        self,
        *,
        name: str,
        budget: int,
        start: tuple[int, int],
        goal_stand: tuple[int, int],
        goal_support: tuple[int, int],
        goal_flag: tuple[int, int],
        terrain: frozenset[tuple[int, int]],
        bars: tuple[BarSpec, ...],
    ) -> None:
        self.name = name
        self.budget = budget
        self.start = start
        self.goal_stand = goal_stand
        self.goal_support = goal_support
        self.goal_flag = goal_flag
        self.terrain = terrain
        self.bars = bars


class BarState:
    def __init__(self, *, name: str, x0: int, x1: int, y: int, rope_x: int, cut: bool = False) -> None:
        self.name = name
        self.x0 = x0
        self.x1 = x1
        self.y = y
        self.rope_x = rope_x
        self.cut = cut


LEVEL_SPECS = (
    LevelSpec(
        name="Level 1",
        budget=24,
        start=(2, 13),
        goal_stand=(9, 13),
        goal_support=(9, 14),
        goal_flag=(10, 13),
        terrain=frozenset({(1, 14), (2, 14), (3, 14), (8, 14), (9, 14), (10, 14)}),
        bars=(BarSpec(name="A", x0=3, x1=7, y=8, rope_x=5),),
    ),
    LevelSpec(
        name="Level 2",
        budget=27,
        start=(2, 13),
        goal_stand=(10, 13),
        goal_support=(10, 14),
        goal_flag=(10, 13),
        terrain=frozenset({(1, 14), (2, 14), (3, 14), (8, 14), (9, 14), (10, 14), (9, 12), (10, 12)}),
        bars=(BarSpec(name="A", x0=3, x1=7, y=8, rope_x=5), BarSpec(name="B", x0=9, x1=10, y=9, rope_x=9)),
    ),
    LevelSpec(
        name="Level 3",
        budget=27,
        start=(2, 13),
        goal_stand=(9, 10),
        goal_support=(9, 11),
        goal_flag=(9, 10),
        terrain=frozenset({(1, 14), (2, 14), (3, 14), (6, 14), (9, 11), (10, 11), (10, 12), (10, 13), (10, 14)}),
        bars=(BarSpec(name="A", x0=3, x1=6, y=8, rope_x=4), BarSpec(name="B", x0=6, x1=8, y=5, rope_x=7)),
    ),
)


def _logical_to_pixels(cell_x: int, cell_y: int) -> tuple[int, int]:
    return cell_x * CELL_SIZE, cell_y * CELL_SIZE


def _cell_sprite(
    pixels: list[list[int]], cell_x: int, cell_y: int, *, name: str, layer: int, collidable: bool = False
) -> Sprite:
    pixel_x, pixel_y = _logical_to_pixels(cell_x, cell_y)
    return Sprite(
        pixels=np.array(pixels, dtype=np.int8), name=name, x=pixel_x, y=pixel_y, layer=layer, collidable=collidable
    )


def _bar_cell_pixels(index: int, span: int) -> list[list[int]]:
    pixels = [[13, 12, 12, 13], [12, 12, 12, 12], [12, 12, 12, 12], [13, 12, 12, 13]]
    if index == 0:
        for row in pixels:
            row[0] = 13
    if index == span - 1:
        for row in pixels:
            row[3] = 13
    return pixels


class CutlineDrop(ARCBaseGame):
    """Click ropes, drop bars, then walk the rebuilt route to the goal."""

    def __init__(self) -> None:
        levels = [
            Level(grid_size=(RENDER_SIZE, RENDER_SIZE), data={"level_index": level_index}, name=spec.name)
            for level_index, spec in enumerate(LEVEL_SPECS)
        ]
        super().__init__(
            "cutline_drop",
            levels,
            Camera(0, 0, RENDER_SIZE, RENDER_SIZE, PLAY_BACKGROUND, UI_BACKGROUND),
            available_actions=[ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
        )

    def on_set_level(self, level: Level) -> None:
        level_index = int(level.get_data("level_index") or 0)
        self._level_spec = LEVEL_SPECS[level_index]
        self._budget_max = int(self._level_spec.budget)
        self._budget_remaining = int(self._level_spec.budget)
        self._walker = tuple(self._level_spec.start)
        self._bars = {
            spec.name: BarState(name=spec.name, x0=spec.x0, x1=spec.x1, y=spec.y, rope_x=spec.rope_x)
            for spec in self._level_spec.bars
        }
        self._rebuild_level_sprites()

    def _bar_cells(self, bar: BarState) -> set[tuple[int, int]]:
        return {(cell_x, bar.y) for cell_x in range(bar.x0, bar.x1 + 1)}

    def _terrain_cells(self) -> set[tuple[int, int]]:
        return set(self._level_spec.terrain)

    def _goal_support_cell(self) -> tuple[int, int]:
        return self._level_spec.goal_support

    def _solid_cells(self, *, exclude_bar: str | None = None) -> set[tuple[int, int]]:
        solids = self._terrain_cells()
        solids.add(self._goal_support_cell())
        for name, bar in self._bars.items():
            if exclude_bar is not None and name == exclude_bar:
                continue
            solids.update(self._bar_cells(bar))
        return solids

    def _in_play_bounds(self, cell_x: int, cell_y: int) -> bool:
        return 0 <= cell_x < LOGICAL_SIZE and UI_ROWS <= cell_y < LOGICAL_SIZE

    def _is_solid(self, cell_x: int, cell_y: int, *, exclude_bar: str | None = None) -> bool:
        if not self._in_play_bounds(cell_x, cell_y):
            return True
        return (cell_x, cell_y) in self._solid_cells(exclude_bar=exclude_bar)

    def _is_empty(self, cell_x: int, cell_y: int, *, exclude_bar: str | None = None) -> bool:
        return self._in_play_bounds(cell_x, cell_y) and not self._is_solid(cell_x, cell_y, exclude_bar=exclude_bar)

    def _try_move_walker(self, dx: int) -> None:
        walker_x, walker_y = self._walker
        target_x = walker_x + dx

        if self._is_empty(target_x, walker_y) and self._is_solid(target_x, walker_y + 1):
            self._walker = (target_x, walker_y)
            return

        if self._is_solid(target_x, walker_y) and self._is_empty(target_x, walker_y - 1):
            self._walker = (target_x, walker_y - 1)
            return

        if (
            self._is_empty(target_x, walker_y)
            and self._is_empty(target_x, walker_y + 1)
            and self._is_solid(target_x, walker_y + 2)
        ):
            self._walker = (target_x, walker_y + 1)

    def _cut_bar_from_click(self, click_x: int, click_y: int) -> BarState | None:
        cell_x = max(0, min(RENDER_SIZE - 1, int(click_x))) // CELL_SIZE
        cell_y = max(0, min(RENDER_SIZE - 1, int(click_y))) // CELL_SIZE

        for bar in self._bars.values():
            if bar.cut or cell_x != bar.rope_x:
                continue
            if cell_y == UI_ROWS or 3 <= cell_y < bar.y:
                bar.cut = True
                return bar
        return None

    def _drop_bar(self, bar: BarState) -> None:
        landing_y = bar.y
        while True:
            next_y = landing_y + 1
            next_cells = {(cell_x, next_y) for cell_x in range(bar.x0, bar.x1 + 1)}
            if next_cells & self._solid_cells(exclude_bar=bar.name):
                break
            if next_y >= LOGICAL_SIZE:
                break
            landing_y = next_y
        bar.y = landing_y

    def _walker_crushed(self, bar: BarState) -> bool:
        return self._walker in self._bar_cells(bar)

    def _budget_cell(self, index: int) -> tuple[int, int]:
        return index % LOGICAL_SIZE, index // LOGICAL_SIZE

    def _rebuild_level_sprites(self) -> None:
        self.current_level.remove_all_sprites()

        self.current_level.add_sprite(
            Sprite(
                pixels=np.full((UI_ROWS * CELL_SIZE, RENDER_SIZE), UI_BACKGROUND, dtype=np.int8),
                name="ui_background",
                x=0,
                y=0,
                layer=-5,
                collidable=False,
            )
        )

        for cell_x, cell_y in sorted(self._terrain_cells()):
            if (cell_x, cell_y) == self._goal_support_cell():
                continue
            self.current_level.add_sprite(
                _cell_sprite(TERRAIN_CELL, cell_x, cell_y, name=f"terrain_{cell_x}_{cell_y}", layer=0, collidable=False)
            )

        goal_x, goal_y = self._goal_support_cell()
        self.current_level.add_sprite(
            _cell_sprite(GOAL_SUPPORT_CELL, goal_x, goal_y, name="goal_support", layer=1, collidable=False)
        )

        for bar in sorted(self._bars.values(), key=lambda value: value.name):
            span = bar.x1 - bar.x0 + 1
            for index, cell_x in enumerate(range(bar.x0, bar.x1 + 1)):
                self.current_level.add_sprite(
                    _cell_sprite(
                        _bar_cell_pixels(index, span),
                        cell_x,
                        bar.y,
                        name=f"bar_{bar.name}_{cell_x}",
                        layer=2,
                        collidable=False,
                    )
                )

        for bar in sorted(self._bars.values(), key=lambda value: value.name):
            if bar.cut:
                continue
            self.current_level.add_sprite(
                _cell_sprite(
                    ROPE_ANCHOR_CELL, bar.rope_x, UI_ROWS, name=f"rope_anchor_{bar.name}", layer=3, collidable=False
                )
            )
            for cell_y in range(UI_ROWS + 1, bar.y):
                self.current_level.add_sprite(
                    _cell_sprite(
                        ROPE_CELL, bar.rope_x, cell_y, name=f"rope_{bar.name}_{cell_y}", layer=3, collidable=False
                    )
                )

        walker_x, walker_y = self._walker
        self.current_level.add_sprite(
            _cell_sprite(WALKER_CELL, walker_x, walker_y, name="walker", layer=4, collidable=False)
        )

        flag_x, flag_y = self._level_spec.goal_flag
        self.current_level.add_sprite(
            _cell_sprite(GOAL_FLAG_CELL, flag_x, flag_y, name="goal_flag", layer=5, collidable=False)
        )

        for pip_index in range(self._budget_max):
            pip_x, pip_y = self._budget_cell(pip_index)
            sprite_pixels = REMAINING_PIP if pip_index < self._budget_remaining else SPENT_PIP
            self.current_level.add_sprite(
                _cell_sprite(sprite_pixels, pip_x, pip_y, name=f"budget_{pip_index}", layer=6, collidable=False)
            )

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

        self._budget_remaining = max(0, self._budget_remaining - 1)

        dropped_bar: BarState | None = None
        action_id = int(self.action.id.value)

        if action_id == ACTION_LEFT:
            self._try_move_walker(-1)
        elif action_id == ACTION_RIGHT:
            self._try_move_walker(1)
        elif action_id == ACTION_CLICK:
            click_x = int(self.action.data.get("x", 0))
            click_y = int(self.action.data.get("y", 0))
            dropped_bar = self._cut_bar_from_click(click_x, click_y)

        if dropped_bar is not None:
            self._drop_bar(dropped_bar)
            if self._walker_crushed(dropped_bar):
                self._rebuild_level_sprites()
                self.lose()
                self.complete_action()
                return

        self._rebuild_level_sprites()

        if self._walker == self._level_spec.goal_stand:
            self.next_level()
            self.complete_action()
            return

        if self._budget_remaining == 0:
            self.lose()

        self.complete_action()
