from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ActionInput, ARCBaseGame, Camera, GameAction, Level, Sprite

BOARD_WIDTH = 16
BOARD_HEIGHT = 15
CELL_SIZE = 4
PLAYFIELD_HEIGHT = 60
GRID_WIDTH = 64
GRID_HEIGHT = 64
HUD_Y = 60

COLOR_FLOOR = 0
COLOR_SEPARATOR = 1
COLOR_FAN_OFF = 2
COLOR_WALL = 4
COLOR_BLACK = 5
COLOR_RED = 8
COLOR_FAN_ON = 9
COLOR_BEAM = 10
COLOR_YELLOW = 11
COLOR_BIN = 12
COLOR_LEAF = 14
COLOR_TURBULENCE = 15

WAIT_ACTION_IDS = {1, 2, 3, 4, 5}
DIR_TO_DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

LEAF_TILE = np.array([[0, 14, 14, 0], [14, 14, 14, 0], [0, 14, 14, 14], [0, 14, 0, 0]], dtype=np.int8)

HORIZONTAL_BEAM_TILE = np.array([[0, 0, 0, 0], [10, 10, 10, 10], [10, 10, 10, 10], [0, 0, 0, 0]], dtype=np.int8)

VERTICAL_BEAM_TILE = np.array([[0, 10, 10, 0], [0, 10, 10, 0], [0, 10, 10, 0], [0, 10, 10, 0]], dtype=np.int8)

TURBULENCE_TILE = np.array([[15, 10, 10, 15], [10, 15, 15, 10], [10, 15, 15, 10], [15, 10, 10, 15]], dtype=np.int8)

FAN_RIGHT_ON = np.array(
    [
        [4, 4, 4, 4, 4, 4, 4, 4],
        [4, 9, 9, 9, 9, 11, 0, 4],
        [4, 9, 11, 11, 11, 11, 0, 4],
        [4, 9, 9, 9, 11, 0, 0, 4],
        [4, 9, 9, 9, 11, 0, 0, 4],
        [4, 9, 11, 11, 11, 11, 0, 4],
        [4, 9, 9, 9, 9, 11, 0, 4],
        [4, 4, 4, 4, 4, 4, 4, 4],
    ],
    dtype=np.int8,
)

BIN_EMPTY_TILE = np.array(
    [
        [12, 0, 0, 0, 0, 0, 0, 12],
        [12, 0, 0, 0, 0, 0, 0, 12],
        [12, 0, 0, 0, 0, 0, 0, 12],
        [12, 0, 0, 0, 0, 0, 0, 12],
        [12, 0, 0, 0, 0, 0, 0, 12],
        [12, 0, 0, 0, 0, 0, 0, 12],
        [12, 0, 0, 0, 0, 0, 0, 12],
        [12, 12, 12, 12, 12, 12, 12, 12],
    ],
    dtype=np.int8,
)

BIN_FILLED_TILE = BIN_EMPTY_TILE.copy()
BIN_FILLED_TILE[1:7, 1:7] = np.int8(COLOR_LEAF)


@dataclass(frozen=True)
class FanSpec:
    x: int
    y: int
    direction: str
    starts_on: bool = False


@dataclass(frozen=True)
class BinSpec:
    x: int
    y: int


@dataclass(frozen=True)
class LevelSpec:
    name: str
    budget: int
    fans: tuple[FanSpec, ...]
    leaves: tuple[tuple[int, int], ...]
    bins: tuple[BinSpec, ...]
    walls: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class LevelState:
    fan_mask: int
    leaves: tuple[tuple[int, int], ...]
    filled_mask: int
    remaining_budget: int


def _perimeter_walls() -> set[tuple[int, int]]:
    walls: set[tuple[int, int]] = set()
    for x in range(BOARD_WIDTH):
        walls.add((x, 0))
        walls.add((x, BOARD_HEIGHT - 1))
    for y in range(BOARD_HEIGHT):
        walls.add((0, y))
        walls.add((BOARD_WIDTH - 1, y))
    return walls


def _level_specs() -> tuple[LevelSpec, ...]:
    perimeter = _perimeter_walls()

    level_1 = LevelSpec(
        name="Level 1",
        budget=15,
        fans=(FanSpec(2, 6, "RIGHT"), FanSpec(11, 10, "UP")),
        leaves=((6, 6),),
        bins=(BinSpec(11, 5),),
        walls=frozenset(perimeter),
    )

    level_2_walls = set(perimeter)
    for y in (2, 3, 4, 5, 8, 9, 10, 11, 12):
        level_2_walls.add((8, y))
    level_2 = LevelSpec(
        name="Level 2",
        budget=33,
        fans=(FanSpec(1, 6, "RIGHT"), FanSpec(10, 1, "DOWN")),
        leaves=((4, 6), (3, 7)),
        bins=(BinSpec(5, 5), BinSpec(10, 11)),
        walls=frozenset(level_2_walls),
    )

    level_3_walls = set(perimeter)
    for x in range(5, 11):
        for y in range(5, 10):
            level_3_walls.add((x, y))
    level_3 = LevelSpec(
        name="Level 3",
        budget=48,
        fans=(FanSpec(1, 3, "RIGHT"), FanSpec(11, 1, "DOWN"), FanSpec(13, 10, "LEFT"), FanSpec(3, 12, "UP")),
        leaves=((4, 3), (10, 11)),
        bins=(BinSpec(3, 1), BinSpec(11, 11)),
        walls=frozenset(level_3_walls),
    )
    return (level_1, level_2, level_3)


LEVEL_SPECS = _level_specs()


def initial_level_state(level_spec: LevelSpec) -> LevelState:
    fan_mask = 0
    for index, fan in enumerate(level_spec.fans):
        if fan.starts_on:
            fan_mask |= 1 << index
    return LevelState(
        fan_mask=fan_mask, leaves=tuple(sorted(level_spec.leaves)), filled_mask=0, remaining_budget=level_spec.budget
    )


def _fan_cells(fan: FanSpec) -> tuple[tuple[int, int], ...]:
    return ((fan.x, fan.y), (fan.x + 1, fan.y), (fan.x, fan.y + 1), (fan.x + 1, fan.y + 1))


def _bin_cells(bin_spec: BinSpec) -> tuple[tuple[int, int], ...]:
    return (
        (bin_spec.x, bin_spec.y),
        (bin_spec.x + 1, bin_spec.y),
        (bin_spec.x, bin_spec.y + 1),
        (bin_spec.x + 1, bin_spec.y + 1),
    )


def _all_fan_cells(level_spec: LevelSpec) -> frozenset[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for fan in level_spec.fans:
        cells.update(_fan_cells(fan))
    return frozenset(cells)


def _all_bin_cells(level_spec: LevelSpec) -> tuple[frozenset[tuple[int, int]], ...]:
    return tuple(frozenset(_bin_cells(bin_spec)) for bin_spec in level_spec.bins)


def _frontier_cells(fan: FanSpec, step: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if fan.direction == "RIGHT":
        return ((fan.x + 1 + step, fan.y), (fan.x + 1 + step, fan.y + 1))
    if fan.direction == "LEFT":
        return ((fan.x - step, fan.y), (fan.x - step, fan.y + 1))
    if fan.direction == "DOWN":
        return ((fan.x, fan.y + 1 + step), (fan.x + 1, fan.y + 1 + step))
    return ((fan.x, fan.y - step), (fan.x + 1, fan.y - step))


def compute_beam_dirs(level_spec: LevelSpec, fan_mask: int) -> dict[tuple[int, int], set[str]]:
    blocked = set(level_spec.walls)
    blocked.update(_all_fan_cells(level_spec))
    beam_dirs: dict[tuple[int, int], set[str]] = {}
    for index, fan in enumerate(level_spec.fans):
        if ((fan_mask >> index) & 1) == 0:
            continue
        step = 1
        while True:
            cells = _frontier_cells(fan, step)
            if any(
                cell in blocked or not (0 <= cell[0] < BOARD_WIDTH and 0 <= cell[1] < BOARD_HEIGHT) for cell in cells
            ):
                break
            for cell in cells:
                beam_dirs.setdefault(cell, set()).add(fan.direction)
            step += 1
    return beam_dirs


def simulate_level_step(level_spec: LevelSpec, state: LevelState, action: str | int) -> tuple[LevelState, bool, bool]:
    fan_mask = int(state.fan_mask)
    leaves = list(state.leaves)
    filled_mask = int(state.filled_mask)

    if isinstance(action, str) and action.startswith("fan:"):
        fan_index = int(action.split(":", 1)[1])
        if 0 <= fan_index < len(level_spec.fans):
            fan_mask ^= 1 << fan_index

    remaining_budget = int(state.remaining_budget) - 1
    beam_dirs = compute_beam_dirs(level_spec, fan_mask)
    fan_cells = _all_fan_cells(level_spec)
    bin_cells = _all_bin_cells(level_spec)
    filled_cells: set[tuple[int, int]] = set()
    for index, cells in enumerate(bin_cells):
        if (filled_mask >> index) & 1:
            filled_cells.update(cells)

    start_positions = set(leaves)
    intents: list[tuple[int, int]] = []
    for leaf_x, leaf_y in leaves:
        dirs = beam_dirs.get((leaf_x, leaf_y), set())
        if len(dirs) != 1:
            intents.append((leaf_x, leaf_y))
            continue
        dx, dy = DIR_TO_DELTA[next(iter(dirs))]
        target = (leaf_x + dx, leaf_y + dy)
        blocked = (
            target[0] < 0
            or target[0] >= BOARD_WIDTH
            or target[1] < 0
            or target[1] >= BOARD_HEIGHT
            or target in level_spec.walls
            or target in fan_cells
            or target in filled_cells
            or target in start_positions
        )
        intents.append((leaf_x, leaf_y) if blocked else target)

    conflict_counts: dict[tuple[int, int], int] = {}
    for target in intents:
        conflict_counts[target] = conflict_counts.get(target, 0) + 1

    moved: list[tuple[int, int]] = []
    for original, target in zip(leaves, intents, strict=True):
        if target != original and conflict_counts[target] > 1:
            moved.append(original)
        else:
            moved.append(target)

    surviving: list[tuple[int, int]] = []
    for leaf in moved:
        captured = False
        for bin_index, cells in enumerate(bin_cells):
            if (filled_mask >> bin_index) & 1:
                continue
            if leaf in cells:
                filled_mask |= 1 << bin_index
                captured = True
                break
        if not captured:
            surviving.append(leaf)

    next_state = LevelState(
        fan_mask=fan_mask, leaves=tuple(sorted(surviving)), filled_mask=filled_mask, remaining_budget=remaining_budget
    )
    solved = filled_mask == (1 << len(level_spec.bins)) - 1
    failed = remaining_budget <= 0 and not solved
    return next_state, solved, failed


def logical_to_render(x: int, y: int) -> tuple[int, int]:
    return x * CELL_SIZE, y * CELL_SIZE


def fan_click_action_data(fan: FanSpec) -> dict[str, int]:
    rx, ry = logical_to_render(fan.x, fan.y)
    return {"x": rx + 3, "y": ry + 3}


def _rotate_tile(tile: np.ndarray, direction: str) -> np.ndarray:
    if direction == "RIGHT":
        return tile
    if direction == "DOWN":
        return np.rot90(tile, 3)
    if direction == "LEFT":
        return np.rot90(tile, 2)
    return np.rot90(tile, 1)


def _fan_tile(direction: str, *, active: bool) -> np.ndarray:
    tile = FAN_RIGHT_ON.copy()
    if not active:
        tile[tile == COLOR_FAN_ON] = np.int8(COLOR_FAN_OFF)
    return _rotate_tile(tile, direction)


def _draw_tile(frame: np.ndarray, x: int, y: int, tile: np.ndarray) -> None:
    height, width = tile.shape
    frame[y : y + height, x : x + width] = tile


def _draw_cell(frame: np.ndarray, x: int, y: int, color: int) -> None:
    rx, ry = logical_to_render(x, y)
    frame[ry : ry + CELL_SIZE, rx : rx + CELL_SIZE] = np.int8(color)


def render_level(level_spec: LevelSpec, state: LevelState) -> np.ndarray:
    frame = np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_FLOOR, dtype=np.int8)
    frame[PLAYFIELD_HEIGHT:, :] = np.int8(COLOR_WALL)

    beam_dirs = compute_beam_dirs(level_spec, state.fan_mask)

    for bin_index, bin_spec in enumerate(level_spec.bins):
        rx, ry = logical_to_render(bin_spec.x, bin_spec.y)
        tile = BIN_FILLED_TILE if ((state.filled_mask >> bin_index) & 1) else BIN_EMPTY_TILE
        _draw_tile(frame, rx, ry, tile)

    for (x, y), dirs in beam_dirs.items():
        rx, ry = logical_to_render(x, y)
        if len(dirs) >= 2:
            tile = TURBULENCE_TILE
        elif next(iter(dirs)) in {"LEFT", "RIGHT"}:
            tile = HORIZONTAL_BEAM_TILE
        else:
            tile = VERTICAL_BEAM_TILE
        _draw_tile(frame, rx, ry, tile)

    for wall_x, wall_y in level_spec.walls:
        _draw_cell(frame, wall_x, wall_y, COLOR_WALL)

    for index, fan in enumerate(level_spec.fans):
        rx, ry = logical_to_render(fan.x, fan.y)
        active = ((state.fan_mask >> index) & 1) == 1
        _draw_tile(frame, rx, ry, _fan_tile(fan.direction, active=active))

    for leaf_x, leaf_y in state.leaves:
        rx, ry = logical_to_render(leaf_x, leaf_y)
        _draw_tile(frame, rx, ry, LEAF_TILE)

    for bin_index, bin_spec in enumerate(level_spec.bins):
        if ((state.filled_mask >> bin_index) & 1) == 0:
            continue
        rx, ry = logical_to_render(bin_spec.x, bin_spec.y)
        _draw_tile(frame, rx, ry, BIN_FILLED_TILE)

    total_budget = max(1, level_spec.budget)
    frame[PLAYFIELD_HEIGHT:, :total_budget] = np.int8(COLOR_FAN_OFF)
    remaining_color = COLOR_YELLOW
    if state.remaining_budget <= max(1, round(total_budget * 0.10)):
        remaining_color = COLOR_RED
    elif state.remaining_budget <= max(1, round(total_budget * 0.25)):
        remaining_color = COLOR_BIN
    remaining = max(0, min(total_budget, state.remaining_budget))
    frame[PLAYFIELD_HEIGHT:, :remaining] = np.int8(remaining_color)
    if total_budget < GRID_WIDTH:
        frame[PLAYFIELD_HEIGHT:, total_budget:] = np.int8(COLOR_WALL)
    return frame


def build_levels() -> list[Level]:
    levels: list[Level] = []
    for level_index, level_spec in enumerate(LEVEL_SPECS):
        state = initial_level_state(level_spec)
        canvas = Sprite(
            pixels=render_level(level_spec, state),
            name=f"canvas_{level_index}",
            x=0,
            y=0,
            layer=0,
            tags=["canvas"],
            collidable=False,
        )
        clickers: list[Sprite] = []
        for fan_index, fan in enumerate(level_spec.fans):
            rx, ry = logical_to_render(fan.x, fan.y)
            clickers.append(
                Sprite(
                    pixels=np.full((CELL_SIZE * 2, CELL_SIZE * 2), COLOR_SEPARATOR, dtype=np.int8),
                    name=f"fan_click_{fan_index}",
                    x=rx,
                    y=ry,
                    layer=-1,
                    tags=["fan_click", "sys_click", "sys_every_pixel", f"fan_{fan_index}"],
                    collidable=False,
                    visible=False,
                )
            )
        levels.append(
            Level(
                name=level_spec.name,
                grid_size=(GRID_WIDTH, GRID_HEIGHT),
                sprites=[canvas, *clickers],
                data={"level_index": level_index},
            )
        )
    return levels


class FanDrift(ARCBaseGame):
    def __init__(self) -> None:
        self._route_score = 0
        self._state_by_level: list[LevelState] = []
        self._canvas: Sprite | None = None
        super().__init__(
            game_id="fan_drift",
            levels=build_levels(),
            camera=Camera(0, 0, GRID_WIDTH, GRID_HEIGHT, COLOR_FLOOR),
            win_score=len(LEVEL_SPECS),
            available_actions=[1, 2, 3, 4, 5, 6],
        )

    def on_set_level(self, level: Level) -> None:
        level_index = int(level.get_data("level_index") or self.level_index)
        while len(self._state_by_level) < len(LEVEL_SPECS):
            self._state_by_level.append(initial_level_state(LEVEL_SPECS[len(self._state_by_level)]))
        self._reset_level_state(level_index)
        self._canvas = level.get_sprites_by_name(f"canvas_{level_index}")[0]
        self._sync_canvas()

    def _sync_canvas(self) -> None:
        if self._canvas is None:
            return
        spec = LEVEL_SPECS[self.level_index]
        self._canvas.pixels = render_level(spec, self._state_by_level[self.level_index])

    def _clicked_fan_index(self, x: int, y: int) -> int | None:
        if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
            return None
        if y >= HUD_Y:
            return None
        logical = (x // CELL_SIZE, y // CELL_SIZE)
        for index, fan in enumerate(LEVEL_SPECS[self.level_index].fans):
            if logical in _fan_cells(fan):
                return index
        return None

    def _decode_action(self) -> str:
        action_id = int(self.action.id.value)
        if action_id in WAIT_ACTION_IDS:
            return "wait"
        if action_id != int(GameAction.ACTION6.value):
            return "wait"
        payload = self.action.data if isinstance(self.action.data, dict) else {}
        fan_index = self._clicked_fan_index(int(payload.get("x", -1)), int(payload.get("y", -1)))
        if fan_index is None:
            return "wait"
        return f"fan:{fan_index}"

    def _reset_level_state(self, level_index: int) -> None:
        self._state_by_level[level_index] = initial_level_state(LEVEL_SPECS[level_index])

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

        action = self._decode_action()
        level_spec = LEVEL_SPECS[self.level_index]
        current_state = self._state_by_level[self.level_index]
        next_state, solved, failed = simulate_level_step(level_spec, current_state, action)
        self._state_by_level[self.level_index] = next_state

        if solved:
            self._route_score += 1
            self._sync_canvas()
            self.next_level()
            self.complete_action()
            return

        if failed:
            self._sync_canvas()
            self.lose()
            self.complete_action()
            return

        self._sync_canvas()
        self.complete_action()

    def _get_hidden_state(self) -> np.ndarray:
        state = self._state_by_level[self.level_index]
        hidden = np.zeros(8, dtype=np.int16)
        hidden[0] = np.int16(state.fan_mask)
        hidden[1] = np.int16(state.filled_mask)
        hidden[2] = np.int16(state.remaining_budget)
        for index, (x, y) in enumerate(state.leaves[:2]):
            hidden[3 + index * 2] = np.int16(x)
            hidden[4 + index * 2] = np.int16(y)
        return hidden

    def _get_valid_actions(self) -> list[ActionInput]:
        actions = [ActionInput(id=GameAction.from_id(action_id)) for action_id in (1, 2, 3, 4, 5)]
        for fan in LEVEL_SPECS[self.level_index].fans:
            actions.append(ActionInput(id=GameAction.ACTION6, data=fan_click_action_data(fan)))
        return actions
