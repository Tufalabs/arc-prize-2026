from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay

GAME_ID = "bridge_stock-0001"
VIEW_SIZE = 64
BOARD_LEFT = 2
BOARD_TOP = 14
CELL_SIZE = 4

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

VOID = "VOID"
WALL = "WALL"
FLOOR = "FLOOR"
GAP = "GAP"
BRIDGE = "BRIDGE"
RUNNING = "RUNNING"
WON = "WON"

COLOR_WHITE = 0
COLOR_FLOOR = 1
COLOR_WALL_HIGHLIGHT = 2
COLOR_WALL = 3
COLOR_HUD = 4
COLOR_VOID = 5
COLOR_INVALID = 8
COLOR_PLAYER = 9
COLOR_PLAYER_HIGHLIGHT = 10
COLOR_PLANK_HIGHLIGHT = 11
COLOR_WOOD = 12
COLOR_WOOD_END = 13
COLOR_EXIT = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    initial_inventory: int
    baseline_actions: int

    @property
    def step_budget(self) -> int:
        return self.baseline_actions * 6


LEVEL_SPECS = (
    LevelSpec(
        "The First Crossing",
        (
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~SGE~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
        ),
        1,
        2,
    ),
    LevelSpec(
        "Fetch Before Crossing",
        (
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~P~~~~~~~~~",
            "~~~~~.~~~~~~~~~",
            "~~~~~SGE~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
        ),
        0,
        6,
    ),
    LevelSpec(
        "Paid Once",
        (
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~PP.~~~~~~~~~",
            "~~~~~G.G.GE~~~~",
            "~~~~~S~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
        ),
        1,
        12,
    ),
    LevelSpec(
        "The Expensive Shortcut",
        (
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~...~~~~~",
            "~~~~~~~.#.P.P~~",
            "~~~~~~~.#.~~~~~",
            "E.G.G.GSG.~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
        ),
        1,
        25,
    ),
    LevelSpec(
        "Junction Investment",
        (
            "~~~~~~~~~~~~~~~",
            "~~~~~PP.PP~~~~~",
            "~~~~~~~G~~~~~~~",
            "~~~~~~~.~~~~.GE",
            "~~~~~~~G~~~~G~~",
            "~~~~~~~.~~~~.~~",
            "~~~S...G.G.G.~~",
            "~~~~~~~.~~~~~~~",
            "~~~~~...~~~~~~~",
            "~~~~P.~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
        ),
        2,
        44,
    ),
    LevelSpec(
        "The Route You Need Twice",
        (
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~~~~~~~~~~~",
            "~~~~~P...P~~~~~",
            "~~~~~~~...G.G.~",
            "~~~~~~~G~~~~~.~",
            "~~~...GS~~~~~G~",
            "~~~P..~~~~~.G.~",
            "~.G..P~~~PPP~~~",
            "~G~~~~~~~...~~~",
            "~.G.G.GE~P.P~~~",
            "~~~~~~~~~~~~~~~",
        ),
        2,
        70,
    ),
    LevelSpec(
        "Shared Stock Network",
        (
            "~~~~~~~~~~~~~~~",
            "~~~~~~~P.P~~~~~",
            "~~~~~.G...~~~~~",
            "~~~~~G~..P~~~~~",
            "~~.G..~~~~.P.P~",
            "~~.~~.~~~..~~~~",
            "~~.S.G...G.G.~~",
            "~~~~~.~~~.~~G~~",
            "~~~...~~~G~~.G.",
            "~~~P~~~.G.~~~~G",
            "~~~~~~P.P~~~EG.",
            "~~~~~~~P.~~~~~~",
        ),
        2,
        73,
    ),
)


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


def _cell_top_left(cell: tuple[int, int]) -> tuple[int, int]:
    return BOARD_LEFT + cell[0] * CELL_SIZE, BOARD_TOP + cell[1] * CELL_SIZE


def _parse_level(spec: LevelSpec) -> dict[str, object]:
    if len(spec.rows) != 12 or any(len(row) != 15 for row in spec.rows):
        raise ValueError(f"{spec.name} must be exactly 15x12")
    terrain: list[list[str]] = []
    loose: set[tuple[int, int]] = set()
    player = None
    exit_cell = None
    for y, row in enumerate(spec.rows):
        terrain_row: list[str] = []
        for x, symbol in enumerate(row):
            terrain_row.append({"~": VOID, "#": WALL, "G": GAP}.get(symbol, FLOOR))
            if symbol == "P":
                loose.add((x, y))
            elif symbol == "S":
                player = (x, y)
            elif symbol == "E":
                exit_cell = (x, y)
        terrain.append(terrain_row)
    if player is None or exit_cell is None:
        raise ValueError(f"{spec.name} needs one start and exit")
    return {"terrain": terrain, "loose": loose, "player": player, "exit": exit_cell}


class BridgeStockView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: BridgeStock | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = COLOR_VOID
        self._draw_board(frame, game)
        self._draw_hud(frame, game)
        return frame

    def _draw_board(self, frame: np.ndarray, game: BridgeStock) -> None:
        for y in range(12):
            for x in range(15):
                x0, y0 = _cell_top_left((x, y))
                tile = game.terrain[y][x]
                if tile == FLOOR:
                    frame[y0 : y0 + 4, x0 : x0 + 4] = COLOR_FLOOR
                elif tile == WALL:
                    frame[y0 : y0 + 4, x0 : x0 + 4] = COLOR_WALL
                    if y == 0 or game.terrain[y - 1][x] != WALL:
                        frame[y0, x0 : x0 + 4] = COLOR_WALL_HIGHLIGHT
                elif tile == GAP:
                    frame[y0 : y0 + 4, x0 : x0 + 4] = np.array(
                        [[5, 12, 12, 5], [12, 5, 5, 12], [12, 5, 5, 12], [5, 12, 12, 5]], dtype=np.int8
                    )
                elif tile == BRIDGE:
                    self._draw_bridge(frame, x0, y0)

        self._draw_exit(frame, game)
        for cell in game.loose_planks:
            self._draw_loose_plank(frame, cell, game)
        self._draw_install_animation(frame, game)
        self._draw_player(frame, game)
        self._draw_block_feedback(frame, game)

    def _draw_bridge(self, frame: np.ndarray, x0: int, y0: int) -> None:
        frame[y0 : y0 + 4, x0 : x0 + 4] = np.array(
            [[13, 12, 12, 13], [12, 13, 13, 12], [12, 13, 13, 12], [13, 12, 12, 13]], dtype=np.int8
        )

    def _draw_exit(self, frame: np.ndarray, game: BridgeStock) -> None:
        x0, y0 = _cell_top_left(game.exit_position)
        ring = COLOR_EXIT
        center = COLOR_WHITE
        if game.animation_kind == "completion":
            if game.animation_phase == 1:
                ring = COLOR_WHITE
            elif game.animation_phase in (2, 4):
                ring, center = COLOR_EXIT, COLOR_WHITE
            else:
                ring, center = COLOR_WHITE, COLOR_EXIT
        frame[y0 : y0 + 4, x0 : x0 + 4] = np.array(
            [[1, ring, ring, 1], [ring, center, center, ring], [ring, center, center, ring], [1, ring, ring, 1]],
            dtype=np.int8,
        )

    def _draw_loose_plank(self, frame: np.ndarray, cell: tuple[int, int], game: BridgeStock) -> None:
        x0, y0 = _cell_top_left(cell)
        color = COLOR_WOOD
        if game.animation_kind == "collect" and game.pending_target == cell:
            color = COLOR_PLANK_HIGHLIGHT
            if game.animation_phase == 2:
                frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = color
                return
        frame[y0 + 1 : y0 + 3, x0 : x0 + 4] = [COLOR_WOOD_END, color, color, COLOR_WOOD_END]

    def _draw_player(self, frame: np.ndarray, game: BridgeStock) -> None:
        cell = game.player_position
        x0, y0 = _cell_top_left(cell)
        dx = dy = 0
        if game.animation_kind in {"wall", "void"} and game.animation_phase == 1:
            dx, dy = game.feedback_delta
        body_color = COLOR_PLAYER
        center_color = COLOR_PLAYER_HIGHLIGHT
        if game.animation_kind == "completion":
            if game.animation_phase in (1, 3):
                body_color, center_color = COLOR_EXIT, COLOR_WHITE
            else:
                body_color, center_color = COLOR_WHITE, COLOR_EXIT
        elif game.status == WON:
            body_color, center_color = COLOR_WHITE, COLOR_EXIT
        frame[y0 + dy : y0 + dy + 3, x0 + 1 + dx : x0 + 3 + dx] = body_color
        frame[y0 + 1 + dy, x0 + dx : x0 + 4 + dx] = body_color
        frame[y0 + 1 + dy, x0 + 2 + dx] = center_color
        if game.inventory > 0:
            frame[y0 + 1 + dy : y0 + 3 + dy, x0 + dx] = [COLOR_WOOD, COLOR_WOOD_END]

    def _draw_install_animation(self, frame: np.ndarray, game: BridgeStock) -> None:
        if game.animation_kind != "install" or game.pending_target is None:
            return
        x0, y0 = _cell_top_left(game.pending_target)
        dx, _dy = game.feedback_delta
        if game.animation_phase == 1:
            if dx:
                frame[y0 + 1 : y0 + 3, x0 : x0 + 4] = COLOR_PLANK_HIGHLIGHT
            else:
                frame[y0 : y0 + 4, x0 + 1 : x0 + 3] = COLOR_PLANK_HIGHLIGHT
        elif game.animation_phase == 2:
            if dx:
                frame[y0 + 1 : y0 + 3, x0 : x0 + 4] = COLOR_WOOD
            else:
                frame[y0 : y0 + 4, x0 + 1 : x0 + 3] = COLOR_WOOD
        elif game.animation_phase == 3:
            self._draw_bridge(frame, x0, y0)

    def _draw_block_feedback(self, frame: np.ndarray, game: BridgeStock) -> None:
        if game.animation_kind != "empty" or game.pending_target is None:
            return
        if game.animation_phase == 1:
            x0, y0 = _cell_top_left(game.pending_target)
            tile = frame[y0 : y0 + 4, x0 : x0 + 4]
            tile[tile == COLOR_WOOD] = COLOR_INVALID

    def _draw_hud(self, frame: np.ndarray, game: BridgeStock) -> None:
        frame[2:10, 2:42] = COLOR_HUD
        frame[4:8, 4:8] = COLOR_PLAYER
        frame[5:7, 5:7] = COLOR_PLAYER_HIGHLIGHT
        shake = 1 if game.animation_kind == "empty" and game.animation_phase == 2 else 0
        preview_inventory = game.inventory
        if game.animation_kind == "collect" and game.animation_phase == 2:
            preview_inventory += 1
        for index in range(5):
            x0 = 10 + index * 6 + shake
            frame[4:7, x0 : x0 + 4] = COLOR_WALL
            if index < preview_inventory:
                frame[4:7, x0 : x0 + 4] = COLOR_WOOD
                frame[4:7, x0] = COLOR_WOOD_END
                frame[4:7, x0 + 3] = COLOR_WOOD_END

        frame[3:8, 43:62] = COLOR_HUD
        frame[4:7, 44:61] = COLOR_WALL
        filled = int(17 * game.remaining_steps / max(1, game.step_budget))
        if filled:
            frame[4:7, 44 : 44 + filled] = COLOR_EXIT


class BridgeStock(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = BridgeStockView()
        levels = [Level(grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_VOID, COLOR_VOID, [self._view])
        super().__init__(GAME_ID, levels, camera, False, len(levels), [1, 2, 3, 4, 5, 6, 7], seed=seed)
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        parsed = _parse_level(spec)
        self.terrain = [list(row) for row in parsed["terrain"]]
        self.player_position = parsed["player"]
        self.exit_position = parsed["exit"]
        self.loose_planks = set(parsed["loose"])
        self.inventory = spec.initial_inventory
        self.status = RUNNING
        self.step_budget = spec.step_budget
        self.remaining_steps = self.step_budget
        self.undo_history: list[
            tuple[tuple[int, int], tuple[tuple[str, ...], ...], frozenset[tuple[int, int]], int, str]
        ] = []
        self.animation_kind: str | None = None
        self.animation_phase = 0
        self.pending_target: tuple[int, int] | None = None
        self.feedback_delta = (0, 0)
        self._pending_terminal = False
        if hasattr(self, "_view"):
            self._view.game = self

    def _snapshot(self):
        return (
            self.player_position,
            tuple(tuple(row) for row in self.terrain),
            frozenset(self.loose_planks),
            self.inventory,
            self.status,
        )

    def _restore(self, snapshot) -> None:
        player, terrain, loose, inventory, status = snapshot
        self.player_position = player
        self.terrain = [list(row) for row in terrain]
        self.loose_planks = set(loose)
        self.inventory = inventory
        self.status = status
        self._clear_animation()

    def _clear_animation(self) -> None:
        self.animation_kind = None
        self.animation_phase = 0
        self.pending_target = None
        self.feedback_delta = (0, 0)

    def _begin_animation(self, kind: str, target: tuple[int, int] | None, delta: tuple[int, int]) -> None:
        self.animation_kind = kind
        self.animation_phase = 1
        self.pending_target = target
        self.feedback_delta = delta

    def _spend_directional_step(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if self.remaining_steps == 0:
            self._pending_terminal = True

    def _finish_or_fail(self) -> None:
        if self._pending_terminal and self.status != WON:
            self._pending_terminal = False
            self.lose()
        self.complete_action()

    def _advance_animation(self) -> None:
        limits = {"install": 4, "collect": 3, "empty": 3, "wall": 2, "void": 2, "completion": 4}
        kind = self.animation_kind
        if kind is None:
            self._finish_or_fail()
            return
        if self.animation_phase < limits[kind] - 1:
            self.animation_phase += 1
            return

        # Commit on the final frame of the action, not on an extra hidden tick.
        self.animation_phase = limits[kind]
        if kind == "install":
            assert self.pending_target is not None
            x, y = self.pending_target
            self.inventory -= 1
            self.terrain[y][x] = BRIDGE
            self.player_position = self.pending_target
        elif kind == "collect":
            assert self.pending_target is not None
            self.player_position = self.pending_target
            self.loose_planks.remove(self.pending_target)
            self.inventory += 1
        elif kind == "completion":
            self.next_level()
            self.complete_action()
            return
        self._clear_animation()
        self._finish_or_fail()

    def _resolve_move(self, dx: int, dy: int) -> None:
        self._spend_directional_step()
        x, y = self.player_position
        target = (x + dx, y + dy)
        tx, ty = target
        if not (0 <= tx < 15 and 0 <= ty < 12):
            self._begin_animation("void", None, (dx, dy))
            return
        terrain = self.terrain[ty][tx]
        if terrain == VOID:
            self._begin_animation("void", target, (dx, dy))
            return
        if terrain == WALL:
            self._begin_animation("wall", target, (dx, dy))
            return
        if terrain == GAP:
            if self.inventory == 0:
                self._begin_animation("empty", target, (dx, dy))
                return
            self.undo_history.append(self._snapshot())
            self._begin_animation("install", target, (dx, dy))
            return

        self.undo_history.append(self._snapshot())
        if target in self.loose_planks:
            self._begin_animation("collect", target, (dx, dy))
            return
        self.player_position = target
        if target == self.exit_position:
            self.status = WON
            self._begin_animation("completion", target, (dx, dy))
            return
        self._finish_or_fail()

    def step(self) -> None:
        if self.animation_kind is not None:
            self._advance_animation()
            return
        action = _action_id(self.action.id)
        if action == 0:
            self.complete_action()
            return
        if self.status == WON:
            self.complete_action()
            return
        if action == ACTION_UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self.complete_action()
            return
        if action in MOVE_DELTAS:
            self._resolve_move(*MOVE_DELTAS[action])
            return
        # SPACE and every valid canvas click are intentional, exact no-ops.
        self.complete_action()
