from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BOARD_WIDTH: Final[int] = 16
BOARD_HEIGHT: Final[int] = 14
CELL_SIZE: Final[int] = 4
HUD_HEIGHT: Final[int] = 8
FRAME_WIDTH: Final[int] = 64
FRAME_HEIGHT: Final[int] = 64

COLOR_VOID: Final[int] = 5
COLOR_FLOOR: Final[int] = 1
COLOR_OUTLINE: Final[int] = 3
COLOR_AVATAR: Final[int] = 9
COLOR_AVATAR_HIGHLIGHT: Final[int] = 10
COLOR_AVATAR_EYE: Final[int] = 4
COLOR_BLOCK: Final[int] = 12
COLOR_BLOCK_SHADE: Final[int] = 13
COLOR_PAN_RIM: Final[int] = 11
COLOR_ACTIVE_BRIDGE: Final[int] = 14
COLOR_TARGET: Final[int] = 15
COLOR_TARGET_ACCENT: Final[int] = 6
COLOR_TARGET_SPARK: Final[int] = 7
COLOR_FAILURE: Final[int] = 8
COLOR_SUCCESS: Final[int] = 14

ACTION_UP: Final[int] = int(GameAction.ACTION1.value)
ACTION_DOWN: Final[int] = int(GameAction.ACTION2.value)
ACTION_LEFT: Final[int] = int(GameAction.ACTION3.value)
ACTION_RIGHT: Final[int] = int(GameAction.ACTION4.value)
ACTION_SPACE: Final[int] = int(GameAction.ACTION5.value)
ACTION_CLICK: Final[int] = int(GameAction.ACTION6.value)

FACING_UP: Final[int] = 0
FACING_DOWN: Final[int] = 1
FACING_LEFT: Final[int] = 2
FACING_RIGHT: Final[int] = 3

MOVE_TO_FACING: Final[dict[int, int]] = {
    ACTION_UP: FACING_UP,
    ACTION_DOWN: FACING_DOWN,
    ACTION_LEFT: FACING_LEFT,
    ACTION_RIGHT: FACING_RIGHT,
}
FACING_TO_DELTA: Final[dict[int, tuple[int, int]]] = {
    FACING_UP: (0, -1),
    FACING_DOWN: (0, 1),
    FACING_LEFT: (-1, 0),
    FACING_RIGHT: (1, 0),
}

FLOOR_TILE: Final[np.ndarray] = np.array([[3, 3, 3, 3], [3, 1, 1, 3], [3, 1, 1, 3], [3, 3, 3, 3]], dtype=np.int8)
PAN_TILE: Final[np.ndarray] = np.array(
    [[11, 11, 11, 11], [11, 12, 12, 11], [11, 12, 12, 11], [3, 3, 3, 3]], dtype=np.int8
)
ACTIVE_SPAN_TILE: Final[np.ndarray] = np.array(
    [[3, 14, 14, 3], [14, 14, 14, 14], [14, 14, 14, 14], [3, 14, 14, 3]], dtype=np.int8
)
INACTIVE_SPAN_TILE: Final[np.ndarray] = np.array(
    [[5, 13, 13, 5], [13, 5, 5, 13], [13, 5, 5, 13], [5, 13, 13, 5]], dtype=np.int8
)
BLOCK_TILE: Final[np.ndarray] = np.array(
    [[12, 12, 12, 12], [12, 13, 13, 12], [12, 13, 13, 12], [12, 12, 12, 12]], dtype=np.int8
)
GHOST_TILE: Final[np.ndarray] = np.array(
    [[10, 10, 10, 10], [10, 5, 5, 10], [10, 5, 5, 10], [10, 10, 10, 10]], dtype=np.int8
)
TARGET_TILE: Final[np.ndarray] = np.array([[5, 15, 5, 5], [15, 6, 15, 5], [5, 15, 7, 5], [5, 5, 5, 5]], dtype=np.int8)
AVATAR_TILES: Final[dict[int, np.ndarray]] = {
    FACING_RIGHT: np.array([[5, 9, 9, 5], [9, 10, 4, 9], [9, 9, 10, 9], [5, 9, 5, 5]], dtype=np.int8),
    FACING_LEFT: np.array([[5, 9, 9, 5], [9, 4, 10, 9], [9, 10, 9, 9], [5, 5, 9, 5]], dtype=np.int8),
    FACING_UP: np.array([[5, 9, 4, 5], [9, 10, 9, 9], [9, 9, 10, 9], [5, 9, 9, 5]], dtype=np.int8),
    FACING_DOWN: np.array([[5, 9, 9, 5], [9, 9, 10, 9], [9, 10, 9, 9], [5, 9, 4, 5]], dtype=np.int8),
}
AVATAR_CARRY_TILES: Final[dict[int, np.ndarray]] = {
    facing: np.array([[12, 12, 12, 12], [12, 13, 13, 12], *tile[2:]], dtype=np.int8)
    for facing, tile in AVATAR_TILES.items()
}


def _rect(x0: int, x1: int, y0: int, y1: int) -> set[tuple[int, int]]:
    return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


@dataclass(frozen=True)
class BridgeSpec:
    x0: int
    y: int
    req_left: int
    req_right: int

    @property
    def anchors(self) -> tuple[tuple[int, int], ...]:
        return ((self.x0 + 0, self.y), (self.x0 + 1, self.y), (self.x0 + 5, self.y), (self.x0 + 6, self.y))

    @property
    def span(self) -> tuple[tuple[int, int], ...]:
        return ((self.x0 + 2, self.y), (self.x0 + 3, self.y), (self.x0 + 4, self.y))

    @property
    def left_pan(self) -> tuple[tuple[int, int], ...]:
        return ((self.x0 + 0, self.y + 1), (self.x0 + 1, self.y + 1))

    @property
    def right_pan(self) -> tuple[tuple[int, int], ...]:
        return ((self.x0 + 5, self.y + 1), (self.x0 + 6, self.y + 1))


@dataclass(frozen=True)
class LevelSpec:
    name: str
    budget: int
    floor_cells: frozenset[tuple[int, int]]
    bridges: tuple[BridgeSpec, ...]
    start: tuple[int, int]
    facing: int
    target: tuple[int, int]
    blocks: frozenset[tuple[int, int]]

    @property
    def anchor_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset(cell for bridge in self.bridges for cell in bridge.anchors)

    @property
    def pan_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset(cell for bridge in self.bridges for cell in (*bridge.left_pan, *bridge.right_pan))

    @property
    def span_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset(cell for bridge in self.bridges for cell in bridge.span)

    @property
    def static_walkable(self) -> frozenset[tuple[int, int]]:
        return frozenset({*self.floor_cells, *self.anchor_cells, *self.pan_cells, self.target})


@dataclass(frozen=True)
class SearchState:
    player: tuple[int, int]
    facing: int
    carrying: bool
    blocks: tuple[tuple[int, int], ...]


def _build_level_specs() -> list[LevelSpec]:
    level_1 = LevelSpec(
        name="Scale Lock 1",
        budget=30,
        floor_cells=frozenset(_rect(1, 4, 4, 10) | _rect(12, 14, 4, 10)),
        bridges=(BridgeSpec(x0=5, y=7, req_left=2, req_right=0),),
        start=(2, 7),
        facing=FACING_RIGHT,
        target=(13, 7),
        blocks=frozenset({(2, 5), (3, 9)}),
    )
    level_2 = LevelSpec(
        name="Scale Lock 2",
        budget=44,
        floor_cells=frozenset(_rect(1, 3, 1, 5) | _rect(11, 14, 1, 12) | _rect(1, 3, 8, 12)),
        bridges=(BridgeSpec(x0=4, y=3, req_left=1, req_right=0), BridgeSpec(x0=4, y=9, req_left=0, req_right=2)),
        start=(2, 3),
        facing=FACING_RIGHT,
        target=(2, 9),
        blocks=frozenset({(1, 1), (1, 5), (3, 5)}),
    )
    level_3 = LevelSpec(
        name="Scale Lock 3",
        budget=56,
        floor_cells=frozenset(_rect(2, 3, 1, 5) | {(1, 2), (1, 4), (1, 5)} | _rect(11, 14, 1, 12) | _rect(1, 3, 8, 12)),
        bridges=(BridgeSpec(x0=4, y=3, req_left=1, req_right=1), BridgeSpec(x0=4, y=9, req_left=0, req_right=2)),
        start=(3, 3),
        facing=FACING_LEFT,
        target=(2, 9),
        blocks=frozenset({(1, 2), (1, 5), (9, 4)}),
    )
    return [level_1, level_2, level_3]


LEVEL_SPECS: Final[list[LevelSpec]] = _build_level_specs()


def _sorted_blocks(blocks: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(blocks))


def initial_search_state(level_spec: LevelSpec) -> SearchState:
    return SearchState(
        player=level_spec.start, facing=level_spec.facing, carrying=False, blocks=_sorted_blocks(level_spec.blocks)
    )


def _bridge_active(bridge: BridgeSpec, blocks: set[tuple[int, int]]) -> bool:
    left_count = sum(1 for cell in bridge.left_pan if cell in blocks)
    right_count = sum(1 for cell in bridge.right_pan if cell in blocks)
    return left_count == bridge.req_left and right_count == bridge.req_right


def active_spans(level_spec: LevelSpec, blocks: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {cell for bridge in level_spec.bridges if _bridge_active(bridge, blocks) for cell in bridge.span}


def bridge_actives(level_spec: LevelSpec, blocks: set[tuple[int, int]]) -> tuple[bool, ...]:
    return tuple(_bridge_active(bridge, blocks) for bridge in level_spec.bridges)


def _in_bounds(cell: tuple[int, int]) -> bool:
    x, y = cell
    return 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT


def _front_cell(player: tuple[int, int], facing: int) -> tuple[int, int]:
    dx, dy = FACING_TO_DELTA[facing]
    return (player[0] + dx, player[1] + dy)


def _walkable(level_spec: LevelSpec, cell: tuple[int, int], blocks: set[tuple[int, int]]) -> bool:
    if not _in_bounds(cell) or cell in blocks:
        return False
    if cell in level_spec.static_walkable:
        return True
    return cell in active_spans(level_spec, blocks)


def simulate_search_action(level_spec: LevelSpec, state: SearchState, action_id: int) -> SearchState | None:
    if action_id not in {ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE}:
        return state

    player = state.player
    facing = state.facing
    carrying = state.carrying
    blocks = set(state.blocks)

    if action_id in MOVE_TO_FACING:
        facing = MOVE_TO_FACING[action_id]
        destination = _front_cell(player, facing)
        if _walkable(level_spec, destination, blocks):
            player = destination
    elif action_id == ACTION_SPACE:
        front = _front_cell(player, facing)
        if not carrying and front in blocks:
            blocks.remove(front)
            carrying = True
        elif carrying and front in level_spec.floor_cells | level_spec.pan_cells and front not in blocks:
            blocks.add(front)
            carrying = False

    active = active_spans(level_spec, blocks)
    if player in level_spec.span_cells and player not in active:
        return None

    return SearchState(player=player, facing=facing, carrying=carrying, blocks=_sorted_blocks(blocks))


def find_optimal_actions(level_spec: LevelSpec) -> list[int]:
    start = initial_search_state(level_spec)
    queue: list[SearchState] = [start]
    previous: dict[SearchState, SearchState | None] = {start: None}
    previous_action: dict[SearchState, int] = {}
    cursor = 0

    while cursor < len(queue):
        state = queue[cursor]
        cursor += 1
        if state.player == level_spec.target:
            actions: list[int] = []
            node = state
            while previous[node] is not None:
                actions.append(previous_action[node])
                node = previous[node]  # type: ignore[assignment]
            actions.reverse()
            return actions

        for action_id in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE):
            next_state = simulate_search_action(level_spec, state, action_id)
            if next_state is None or next_state in previous:
                continue
            previous[next_state] = state
            previous_action[next_state] = action_id
            queue.append(next_state)

    raise RuntimeError(f"No solution found for {level_spec.name}.")


SOLVER_PROGRAMS: Final[dict[int, list[int]]] = {
    0: [1, 5, 2, 2, 4, 4, 4, 5, 3, 3, 2, 5, 4, 5, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6],
    1: [
        1,
        1,
        3,
        5,
        2,
        2,
        2,
        4,
        5,
        3,
        2,
        3,
        5,
        1,
        1,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        5,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        5,
        1,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        3,
        5,
        1,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        6,
    ],
    2: [
        1,
        3,
        3,
        5,
        2,
        2,
        4,
        5,
        3,
        2,
        3,
        5,
        1,
        1,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        5,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        3,
        5,
        4,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        3,
        5,
        1,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        6,
    ],
}
ESTIMATED_OPTIMAL_ACTION_COUNTS: Final[dict[int, int]] = {0: 24, 1: 81, 2: 61}
TUNED_LEVEL_SPECS: Final[list[LevelSpec]] = [
    replace(spec, budget=max(spec.budget, ESTIMATED_OPTIMAL_ACTION_COUNTS[idx] * 3))
    for idx, spec in enumerate(LEVEL_SPECS)
]


class ScaleLock(ARCBaseGame):
    def __init__(self) -> None:
        self._route_score = 0
        self._phase = "play"
        self._budget_remaining = 0
        self._player = (0, 0)
        self._facing = FACING_RIGHT
        self._carrying = False
        self._blocks: set[tuple[int, int]] = set()
        self._bridge_states: tuple[bool, ...] = ()
        self._board_sprite: Sprite | None = None
        self._level_specs = TUNED_LEVEL_SPECS
        levels = [self._build_level(level_idx, spec) for level_idx, spec in enumerate(self._level_specs)]
        camera = Camera(0, 0, FRAME_WIDTH, FRAME_HEIGHT, COLOR_VOID)
        super().__init__(
            game_id="scale_lock-0001",
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_CLICK],
        )

    def _build_level(self, level_idx: int, spec: LevelSpec) -> Level:
        board = Sprite(
            pixels=np.full((FRAME_HEIGHT, FRAME_WIDTH), COLOR_VOID, dtype=np.int8),
            name="board",
            x=0,
            y=0,
            layer=0,
            collidable=False,
            tags=["board"],
        )
        return Level(
            name=spec.name,
            grid_size=(FRAME_WIDTH, FRAME_HEIGHT),
            sprites=[board],
            data={
                "level_index": level_idx,
                "budget": spec.budget,
                "optimal_actions": ESTIMATED_OPTIMAL_ACTION_COUNTS[level_idx],
            },
        )

    def on_set_level(self, level: Level) -> None:
        level_idx = int(level.get_data("level_index"))
        spec = self._level_specs[level_idx]
        self._route_score = level_idx
        self._phase = "play"
        self._budget_remaining = int(spec.budget)
        self._player = spec.start
        self._facing = spec.facing
        self._carrying = False
        self._blocks = set(spec.blocks)
        self._bridge_states = bridge_actives(spec, self._blocks)
        boards = level.get_sprites_by_name("board")
        if not boards:
            raise RuntimeError("Scale Lock board sprite is missing.")
        self._board_sprite = boards[0]
        self._redraw()

    @property
    def _spec(self) -> LevelSpec:
        return self._level_specs[self.level_index]

    def _apply_terminal_input(self) -> None:
        if self._phase == "success":
            self.next_level()
        elif self._phase == "failure":
            self.lose()

    def _active_spans(self) -> set[tuple[int, int]]:
        return active_spans(self._spec, self._blocks)

    def _deactivated_under_player(self) -> bool:
        return self._player in self._spec.span_cells and self._player not in self._active_spans()

    def _try_move(self, facing: int) -> None:
        self._facing = facing
        destination = _front_cell(self._player, facing)
        if _walkable(self._spec, destination, self._blocks):
            self._player = destination

    def _try_space(self) -> None:
        front = _front_cell(self._player, self._facing)
        if not _in_bounds(front):
            return
        if not self._carrying and front in self._blocks:
            self._blocks.remove(front)
            self._carrying = True
            return
        if self._carrying and front not in self._blocks and front in self._spec.floor_cells | self._spec.pan_cells:
            self._blocks.add(front)
            self._carrying = False

    def _non_click_turn(self) -> None:
        self._budget_remaining -= 1
        action_id = int(self.action.id.value)
        if action_id in MOVE_TO_FACING:
            self._try_move(MOVE_TO_FACING[action_id])
        elif action_id == ACTION_SPACE:
            self._try_space()

        self._bridge_states = bridge_actives(self._spec, self._blocks)
        if self._deactivated_under_player():
            self._phase = "failure"
        elif self._player == self._spec.target:
            self._phase = "success"
        elif self._budget_remaining <= 0:
            self._phase = "failure"

    def step(self) -> None:
        if self._phase != "play":
            self._apply_terminal_input()
            self._redraw()
            self.complete_action()
            return

        action_id = int(self.action.id.value)
        if action_id != ACTION_CLICK:
            self._non_click_turn()
        self._redraw()
        self.complete_action()

    def _cell_pixels(self, cell: tuple[int, int], walkable_neighbors: set[tuple[int, int]]) -> np.ndarray:
        base = np.full((CELL_SIZE, CELL_SIZE), COLOR_FLOOR, dtype=np.int8)
        x, y = cell
        if (x, y - 1) not in walkable_neighbors:
            base[0, :] = COLOR_OUTLINE
        if (x, y + 1) not in walkable_neighbors:
            base[-1, :] = COLOR_OUTLINE
        if (x - 1, y) not in walkable_neighbors:
            base[:, 0] = COLOR_OUTLINE
        if (x + 1, y) not in walkable_neighbors:
            base[:, -1] = COLOR_OUTLINE
        return base

    def _draw_tile(self, frame: np.ndarray, cell: tuple[int, int], tile: np.ndarray) -> None:
        px = cell[0] * CELL_SIZE
        py = HUD_HEIGHT + cell[1] * CELL_SIZE
        frame[py : py + CELL_SIZE, px : px + CELL_SIZE] = tile

    def _draw_ghosts(self, frame: np.ndarray, bridge: BridgeSpec) -> None:
        if bridge.req_left == 1:
            self._draw_tile(frame, (bridge.x0, bridge.y - 1), GHOST_TILE)
        elif bridge.req_left == 2:
            self._draw_tile(frame, (bridge.x0, bridge.y - 1), GHOST_TILE)
            self._draw_tile(frame, (bridge.x0 + 1, bridge.y - 1), GHOST_TILE)

        if bridge.req_right == 1:
            self._draw_tile(frame, (bridge.x0 + 5, bridge.y - 1), GHOST_TILE)
        elif bridge.req_right == 2:
            self._draw_tile(frame, (bridge.x0 + 5, bridge.y - 1), GHOST_TILE)
            self._draw_tile(frame, (bridge.x0 + 6, bridge.y - 1), GHOST_TILE)

    def _draw_budget(self, frame: np.ndarray) -> None:
        lit = max(0, min(56, self._budget_remaining))
        frame[2:6, 4:60] = COLOR_OUTLINE
        if lit:
            frame[2:6, 4 : 4 + lit] = COLOR_PAN_RIM

    def _draw_border(self, frame: np.ndarray, color: int) -> None:
        frame[:2, :] = color
        frame[-2:, :] = color
        frame[:, :2] = color
        frame[:, -2:] = color

    def _redraw(self) -> None:
        if self._board_sprite is None:
            return

        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), COLOR_VOID, dtype=np.int8)
        spec = self._spec
        active = self._active_spans()
        terrain_walkable = set(spec.floor_cells | spec.anchor_cells | spec.pan_cells | {spec.target} | active)

        for cell in spec.floor_cells | spec.anchor_cells:
            self._draw_tile(frame, cell, self._cell_pixels(cell, terrain_walkable))

        for cell in spec.pan_cells:
            self._draw_tile(frame, cell, PAN_TILE)

        for bridge in spec.bridges:
            self._draw_ghosts(frame, bridge)
            for cell in bridge.span:
                self._draw_tile(frame, cell, ACTIVE_SPAN_TILE if cell in active else INACTIVE_SPAN_TILE)

        self._draw_tile(frame, spec.target, TARGET_TILE)

        for block in sorted(self._blocks):
            self._draw_tile(frame, block, BLOCK_TILE)

        avatar_tile = AVATAR_CARRY_TILES[self._facing] if self._carrying else AVATAR_TILES[self._facing]
        self._draw_tile(frame, self._player, avatar_tile)
        self._draw_budget(frame)

        if self._phase == "success":
            self._draw_border(frame, COLOR_SUCCESS)
        elif self._phase == "failure":
            self._draw_border(frame, COLOR_FAILURE)

        self._board_sprite.pixels = frame


class Scalelock(ScaleLock):
    pass
