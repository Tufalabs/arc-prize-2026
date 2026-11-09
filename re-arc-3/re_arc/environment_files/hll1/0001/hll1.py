from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, GameState, Level, Sprite

BOARD_W = 16
BOARD_H = 14
CELL = 4
HUD_H = 8
FRAME_W = 64
FRAME_H = 64

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SWITCH = 5

HEAVY = "heavy"
LIGHT = "light"

WALL = "wall"
FLOOR = "floor"
RENDEZVOUS = "rendezvous"
PLATE = "plate"
DOOR = "door"
NARROW = "narrow"

PAIR_A = "A"
PAIR_B = "B"

COLORS = {
    "white": 0,
    "floor": 1,
    "spent": 3,
    "wall": 4,
    "black": 5,
    "pad": 6,
    "pad_hi": 7,
    "red": 8,
    "blue": 9,
    "blue_hi": 10,
    "yellow": 11,
    "orange": 12,
    "maroon": 13,
    "green": 14,
    "purple": 15,
}

HEAVY_MOTIF = np.array([[-1, 12, 12, -1], [12, 12, 12, 12], [12, 13, 13, 12], [-1, 12, 12, -1]], dtype=np.int8)
LIGHT_MOTIF = np.array([[-1, -1, 9, -1], [-1, 9, 10, 9], [-1, 9, 10, 9], [-1, -1, 9, -1]], dtype=np.int8)
MERGED_MOTIF = np.array([[12, 12, 9, 9], [12, 13, 10, 9], [12, 13, 10, 9], [12, 12, 9, 9]], dtype=np.int8)
PAD_MOTIF = np.array([[6, 6, 6, 6], [6, 0, 0, 6], [6, 0, 0, 6], [6, 6, 6, 6]], dtype=np.int8)
OPEN_DOOR_MOTIF = np.array([[14, 14, 14, 14], [14, 1, 1, 14], [14, 1, 1, 14], [14, 14, 14, 14]], dtype=np.int8)
NARROW_MOTIF = np.array([[15, 4, 4, 15], [15, 0, 0, 15], [15, 0, 0, 15], [15, 4, 4, 15]], dtype=np.int8)
FLOOR_TILE = np.array([[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=np.int8)
WALL_TILE = np.array([[5, 5, 5, 5], [5, 4, 4, 5], [5, 4, 4, 5], [5, 5, 5, 5]], dtype=np.int8)


LEVEL_SPECS = (
    {
        "budget": 14,
        "floor_rects": ((1, 6, 3, 10), (8, 14, 3, 10)),
        "heavy_start": (4, 6),
        "light_start": (10, 6),
        "rendezvous": (6, 6),
        "plates": {PAIR_A: (5, 6)},
        "doors": {PAIR_A: (7, 6)},
        "narrow": (),
    },
    {
        "budget": 44,
        "floor_rects": ((1, 5, 1, 6), (3, 5, 7, 11), (5, 14, 11, 11), (11, 14, 4, 10), (7, 9, 1, 3), (7, 9, 5, 6)),
        "heavy_start": (2, 3),
        "light_start": (8, 2),
        "rendezvous": (11, 9),
        "plates": {PAIR_A: (4, 3)},
        "doors": {PAIR_A: (8, 4)},
        "narrow": ((10, 5),),
    },
    {
        "budget": 60,
        "floor_rects": (
            (1, 4, 1, 4),
            (1, 4, 5, 11),
            (4, 13, 11, 11),
            (11, 13, 8, 10),
            (7, 9, 6, 10),
            (6, 8, 1, 2),
            (6, 8, 4, 4),
            (10, 13, 1, 4),
            (11, 13, 6, 6),
        ),
        "heavy_start": (2, 2),
        "light_start": (7, 1),
        "rendezvous": (8, 8),
        "plates": {PAIR_A: (3, 3), PAIR_B: (12, 9)},
        "doors": {PAIR_A: (7, 3), PAIR_B: (12, 5)},
        "narrow": ((9, 4), (10, 6)),
    },
)


def _rect_cells(x0: int, x1: int, y0: int, y1: int) -> set[tuple[int, int]]:
    return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


def _pair_color(pair_id: str) -> int:
    return COLORS["red"] if pair_id == PAIR_A else COLORS["purple"]


def _plate_motif(pair_id: str) -> np.ndarray:
    accent = _pair_color(pair_id)
    return np.array(
        [[11, 11, 11, 11], [11, accent, accent, 11], [11, accent, accent, 11], [11, 11, 11, 11]], dtype=np.int8
    )


def _door_closed_motif(pair_id: str) -> np.ndarray:
    accent = _pair_color(pair_id)
    return np.array(
        [
            [accent, accent, accent, accent],
            [accent, 5, 5, accent],
            [accent, 5, 5, accent],
            [accent, accent, accent, accent],
        ],
        dtype=np.int8,
    )


class Hll1(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._route_score = 0
        self._board = np.zeros((FRAME_H, FRAME_W), dtype=np.int8)
        self._board_sprite: Sprite | None = None
        self._spec = LEVEL_SPECS[0]
        self._floor_cells: set[tuple[int, int]] = set()
        self._cell_kind: dict[tuple[int, int], str] = {}
        self._heavy = (0, 0)
        self._light = (0, 0)
        self._selected = HEAVY
        self._remaining_moves = 0
        self._static_layers = [self._build_static_frame(spec) for spec in LEVEL_SPECS]
        levels = [Level(grid_size=(FRAME_W, FRAME_H), name=f"Level {idx + 1}") for idx in range(len(LEVEL_SPECS))]
        super().__init__(
            game_id="hll1-0001",
            levels=levels,
            win_score=len(LEVEL_SPECS),
            available_actions=[UP, DOWN, LEFT, RIGHT, SWITCH],
            seed=seed,
        )

    def on_set_level(self, _level: Level) -> None:
        idx = self._current_level_index
        self._spec = LEVEL_SPECS[idx]
        self._floor_cells = set()
        for rect in self._spec["floor_rects"]:
            self._floor_cells.update(_rect_cells(*rect))
        self._cell_kind = {(x, y): FLOOR for (x, y) in self._floor_cells}
        self._cell_kind[self._spec["rendezvous"]] = RENDEZVOUS
        for pair_id, cell in self._spec["plates"].items():
            self._cell_kind[cell] = f"{PLATE}_{pair_id}"
        for pair_id, cell in self._spec["doors"].items():
            self._cell_kind[cell] = f"{DOOR}_{pair_id}"
        for cell in self._spec["narrow"]:
            self._cell_kind[cell] = NARROW

        self._heavy = self._spec["heavy_start"]
        self._light = self._spec["light_start"]
        self._selected = HEAVY
        self._remaining_moves = int(self._spec["budget"])
        self._sync_level_sprite()

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id == 0:
            # RESET path: arcengine's handle_reset restored the pristine
            # level state BEFORE self._state was reset to NOT_FINISHED, so
            # on_set_level rendered the board with the stale loss/win
            # overlay still drawn (GAME_OVER → red X, WIN → green
            # brackets). Re-sync now that state is clean so the returned
            # frame shows the pristine level without the terminal overlay.
            self._sync_level_sprite()
            self.complete_action()
            return

        if self._state in {GameState.WIN, GameState.GAME_OVER}:
            self.complete_action()
            return

        if action_id == SWITCH:
            self._selected = LIGHT if self._selected == HEAVY else HEAVY
            self._consume_move_and_finalize(level_won=False)
            return

        delta = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}.get(action_id)
        if delta is None:
            self.complete_action()
            return

        level_won = self._attempt_move(delta[0], delta[1])
        self._consume_move_and_finalize(level_won=level_won)

    def _consume_move_and_finalize(self, *, level_won: bool) -> None:
        self._remaining_moves -= 1
        if self._remaining_moves < 0:
            self._remaining_moves = 0

        if level_won:
            self._route_score += 1
            self._sync_level_sprite()
            self.next_level()
            self.complete_action()
            return

        if self._remaining_moves == 0:
            self.lose()

        self._sync_level_sprite()
        self.complete_action()

    def _attempt_move(self, dx: int, dy: int) -> bool:
        actor = self._selected
        current = self._heavy if actor == HEAVY else self._light
        other = self._light if actor == HEAVY else self._heavy
        target = (current[0] + dx, current[1] + dy)

        if not self._is_target_passable(actor, current, target, other):
            return False

        if actor == HEAVY:
            self._heavy = target
        else:
            self._light = target

        return self._heavy == self._spec["rendezvous"] and self._light == self._spec["rendezvous"]

    def _is_target_passable(
        self, actor: str, current: tuple[int, int], target: tuple[int, int], other: tuple[int, int]
    ) -> bool:
        tx, ty = target
        if not (0 <= tx < BOARD_W and 0 <= ty < BOARD_H):
            return False

        kind = self._cell_kind.get(target, WALL)
        if kind == WALL:
            return False
        if kind == NARROW and actor == HEAVY:
            return False
        if kind.startswith(f"{DOOR}_") and not self._is_door_open(kind.split("_", 1)[1]):
            return False

        if target == other and target != self._spec["rendezvous"]:
            return False

        if actor == HEAVY and self._heavy_departure_would_close_on_pawn(current, target):
            return False
        return True

    def _heavy_departure_would_close_on_pawn(self, current: tuple[int, int], target: tuple[int, int]) -> bool:
        if current == target:
            return False
        for pair_id, plate_cell in self._spec["plates"].items():
            if current != plate_cell:
                continue
            door_cell = self._spec["doors"][pair_id]
            if self._light == door_cell or self._heavy == door_cell:
                return True
        return False

    def _is_door_open(self, pair_id: str) -> bool:
        return self._heavy == self._spec["plates"][pair_id]

    def _build_static_frame(self, spec: dict[str, object]) -> np.ndarray:
        frame = np.full((FRAME_H, FRAME_W), COLORS["black"], dtype=np.int8)
        frame[:HUD_H, :] = COLORS["wall"]
        frame[0, :] = COLORS["black"]
        frame[7, :] = COLORS["black"]

        floor_cells = set()
        for rect in spec["floor_rects"]:
            floor_cells.update(_rect_cells(*rect))

        for y in range(BOARD_H):
            for x in range(BOARD_W):
                tile = FLOOR_TILE if (x, y) in floor_cells else WALL_TILE
                self._blit(frame, tile, x, y)
        return frame

    def _sync_level_sprite(self) -> None:
        frame = self._static_layers[self._current_level_index].copy()
        self._draw_hud(frame)
        self._draw_specials(frame)
        self._draw_pawns(frame)
        self._draw_selection_brackets(frame)
        self._draw_terminal_overlay(frame)
        self._board = frame

        sprites = self.current_level.get_sprites()
        if self._board_sprite is None or self._board_sprite not in sprites:
            self.current_level.remove_all_sprites()
            self._board_sprite = Sprite(
                pixels=frame,
                name="board",
                x=0,
                y=0,
                layer=0,
                blocking=BlockingMode.NOT_BLOCKED,
                collidable=False,
                tags=["sys_static"],
            )
            self.current_level.add_sprite(self._board_sprite)
        else:
            self._board_sprite.pixels = frame.copy()

    def _draw_hud(self, frame: np.ndarray) -> None:
        heavy_icon = HEAVY_MOTIF.copy()
        light_icon = LIGHT_MOTIF.copy()
        if self._selected != HEAVY:
            heavy_icon = np.where(heavy_icon == 12, 13, heavy_icon)
        if self._selected != LIGHT:
            light_icon = np.where(light_icon == 9, 10, light_icon)
        self._blit_pixels(frame, heavy_icon, 1, 2)
        self._blit_pixels(frame, light_icon, 7, 2)
        self._draw_hud_border(frame, 0, 1, selected=self._selected == HEAVY)
        self._draw_hud_border(frame, 6, 1, selected=self._selected == LIGHT)

        total = int(self._spec["budget"])
        remaining = self._remaining_moves
        for idx in range(total):
            x = 20 + (idx % 32)
            y = 1 if idx < 32 else 4
            if idx < remaining:
                color = COLORS["orange"] if remaining - idx <= 5 else COLORS["green"]
            else:
                color = COLORS["spent"]
            frame[y : y + 3, x] = color

    def _draw_hud_border(self, frame: np.ndarray, x0: int, y0: int, *, selected: bool) -> None:
        color = COLORS["white"] if selected else COLORS["spent"]
        frame[y0, x0 : x0 + 6] = color
        frame[y0 + 5, x0 : x0 + 6] = color
        frame[y0 : y0 + 6, x0] = color
        frame[y0 : y0 + 6, x0 + 5] = color

    def _draw_specials(self, frame: np.ndarray) -> None:
        self._blit(frame, PAD_MOTIF, *self._spec["rendezvous"])
        for pair_id, cell in self._spec["plates"].items():
            self._blit(frame, _plate_motif(pair_id), *cell)
        for pair_id, cell in self._spec["doors"].items():
            motif = OPEN_DOOR_MOTIF if self._is_door_open(pair_id) else _door_closed_motif(pair_id)
            self._blit(frame, motif, *cell)
        for cell in self._spec["narrow"]:
            self._blit(frame, NARROW_MOTIF, *cell)

    def _draw_pawns(self, frame: np.ndarray) -> None:
        if self._heavy == self._light:
            self._blit(frame, MERGED_MOTIF, *self._heavy)
            return
        self._blit(frame, HEAVY_MOTIF, *self._heavy)
        self._blit(frame, LIGHT_MOTIF, *self._light)

    def _draw_selection_brackets(self, frame: np.ndarray) -> None:
        if self._heavy == self._light:
            cell = self._heavy
        else:
            cell = self._heavy if self._selected == HEAVY else self._light
        px = cell[0] * CELL
        py = HUD_H + cell[1] * CELL
        c = COLORS["white"]
        frame[py, px : px + 2] = c
        frame[py : py + 2, px] = c
        frame[py, px + 2 : px + 4] = c
        frame[py : py + 2, px + 3] = c
        frame[py + 3, px : px + 2] = c
        frame[py + 2 : py + 4, px] = c
        frame[py + 3, px + 2 : px + 4] = c
        frame[py + 2 : py + 4, px + 3] = c

    def _draw_terminal_overlay(self, frame: np.ndarray) -> None:
        if self._state == GameState.GAME_OVER:
            for i in range(56):
                x = min(63, i)
                y = HUD_H + i
                if y < FRAME_H:
                    frame[y, x] = COLORS["red"]
                    frame[y, 63 - x] = COLORS["red"]
            return
        if self._state == GameState.WIN:
            frame[HUD_H : HUD_H + 2, 0:8] = COLORS["green"]
            frame[HUD_H : HUD_H + 2, 56:64] = COLORS["green"]
            frame[56:64, 0:8] = COLORS["green"]
            frame[56:64, 56:64] = COLORS["green"]

    def _blit(self, frame: np.ndarray, tile: np.ndarray, cell_x: int, cell_y: int) -> None:
        self._blit_pixels(frame, tile, cell_x * CELL, HUD_H + cell_y * CELL)

    def _blit_pixels(self, frame: np.ndarray, tile: np.ndarray, px: int, py: int) -> None:
        mask = tile >= 0
        target = frame[py : py + tile.shape[0], px : px + tile.shape[1]]
        target[mask] = tile[mask]
