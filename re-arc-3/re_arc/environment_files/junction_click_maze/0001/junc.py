from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

GRID_SIZE = 64

COLOR_BG = 0
COLOR_JUNCTION_FILL = 1
COLOR_CORRIDOR = 2
COLOR_JUNCTION_OUTLINE = 3
COLOR_SPENT_PIP = 4
COLOR_CURRENT_OUTLINE = 9
COLOR_TOKEN = 10
COLOR_GOAL_HALO = 11
COLOR_GOAL_CORE = 12
COLOR_REACHABLE = 14

ORDINARY_DIAMOND = ((0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0))
INNER_DIAMOND = ((0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (0, 1, 1, 1, 0), (0, 0, 1, 0, 0), (0, 0, 0, 0, 0))
GOAL_HALO = (
    (0, 0, 0, 1, 0, 0, 0),
    (0, 0, 1, 1, 1, 0, 0),
    (0, 1, 1, 1, 1, 1, 0),
    (1, 1, 1, 1, 1, 1, 1),
    (0, 1, 1, 1, 1, 1, 0),
    (0, 0, 1, 1, 1, 0, 0),
    (0, 0, 0, 1, 0, 0, 0),
)
TOKEN_MASK = ((1, 1, 1), (1, 1, 1), (1, 1, 1))

LEVEL_SPECS = (
    {
        "nodes": {"A": (14, 30), "B": (30, 30), "C": (30, 46)},
        "edges": (("A", "B"), ("B", "C")),
        "start": "A",
        "destination": "C",
        "budget": 5,
    },
    {
        "nodes": {"A": (14, 34), "B": (30, 34), "C": (46, 34), "D": (30, 18), "E": (46, 50)},
        "edges": (("A", "B"), ("B", "C"), ("B", "D"), ("C", "E")),
        "start": "A",
        "destination": "E",
        "budget": 7,
    },
    {
        "nodes": {
            "A": (14, 18),
            "B": (30, 18),
            "C": (46, 18),
            "D": (46, 34),
            "E": (30, 34),
            "F": (14, 34),
            "G": (30, 24),
        },
        "edges": (("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "A"), ("E", "G")),
        "start": "A",
        "destination": "G",
        "budget": 5,
    },
)


def _solid_frame() -> np.ndarray:
    return np.full((GRID_SIZE, GRID_SIZE), COLOR_BG, dtype=np.int8)


def _dummy_level(index: int) -> Level:
    sprite = Sprite(_solid_frame(), name="board", x=0, y=0, layer=0, collidable=False, tags=["board"])
    return Level(
        name=f"junction_click_maze_{index}", grid_size=(GRID_SIZE, GRID_SIZE), sprites=[sprite], data={"index": index}
    )


class Junc(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._level_specs = LEVEL_SPECS
        self._board_sprite: Sprite | None = None
        self._neighbors: dict[str, tuple[str, ...]] = {}
        self._current_node_id = ""
        self._moves_remaining = 0
        self._mode = "play"
        self._level_index_state = 0
        self._initial_node_id = ""
        self._destination_node_id = ""
        self._move_budget = 0
        levels = [_dummy_level(index) for index in range(len(self._level_specs))]
        super().__init__(
            game_id="junction_click_maze-0001",
            levels=levels,
            camera=Camera(0, 0, GRID_SIZE, GRID_SIZE, COLOR_BG),
            win_score=len(levels),
            available_actions=[6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        index = int(level.get_data("index") or 0)
        self._level_index_state = index
        spec = self._level_specs[index]
        self._board_sprite = level.get_sprites_by_name("board")[0]
        self._neighbors = {name: [] for name in spec["nodes"]}
        for left, right in spec["edges"]:
            self._neighbors[left].append(right)
            self._neighbors[right].append(left)
        self._neighbors = {name: tuple(sorted(adjacent)) for name, adjacent in self._neighbors.items()}
        self._initial_node_id = spec["start"]
        self._destination_node_id = spec["destination"]
        self._move_budget = spec["budget"]
        self._reset_level_state()
        self._redraw()

    def _reset_level_state(self) -> None:
        self._current_node_id = self._initial_node_id
        self._moves_remaining = self._move_budget
        self._mode = "play"

    def _current_spec(self) -> dict[str, object]:
        return self._level_specs[self._level_index_state]

    def _decode_click(self) -> tuple[int, int] | None:
        data = getattr(self.action, "data", None) or {}
        try:
            display_x = int(data.get("x"))
            display_y = int(data.get("y"))
        except (TypeError, ValueError, AttributeError):
            return None
        return self.camera.display_to_grid(display_x, display_y)

    def _node_hit(self, x: int, y: int, node_id: str) -> bool:
        cx, cy = self._current_spec()["nodes"][node_id]
        return (cx - 3) <= x <= (cx + 3) and (cy - 3) <= y <= (cy + 3)

    def _clicked_neighbor(self, x: int, y: int) -> str | None:
        for neighbor_id in self._neighbors[self._current_node_id]:
            if self._node_hit(x, y, neighbor_id):
                return neighbor_id
        return None

    def _draw_mask(self, frame: np.ndarray, cx: int, cy: int, mask: tuple[tuple[int, ...], ...], color: int) -> None:
        mask_h = len(mask)
        mask_w = len(mask[0])
        top = cy - (mask_h // 2)
        left = cx - (mask_w // 2)
        for row_idx, row in enumerate(mask):
            py = top + row_idx
            if py < 0 or py >= GRID_SIZE:
                continue
            for col_idx, value in enumerate(row):
                if not value:
                    continue
                px = left + col_idx
                if 0 <= px < GRID_SIZE:
                    frame[py, px] = color

    def _draw_corridor(self, frame: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
        x1, y1 = start
        x2, y2 = end
        if x1 == x2:
            y_low, y_high = sorted((y1, y2))
            frame[y_low : y_high + 1, x1 : x1 + 2] = COLOR_CORRIDOR
        elif y1 == y2:
            x_low, x_high = sorted((x1, x2))
            frame[y1 : y1 + 2, x_low : x_high + 1] = COLOR_CORRIDOR

    def _draw_junction(self, frame: np.ndarray, node_id: str) -> None:
        cx, cy = self._current_spec()["nodes"][node_id]
        reachable = node_id in self._neighbors[self._current_node_id]
        is_current = node_id == self._current_node_id
        is_destination = node_id == self._destination_node_id

        if is_destination:
            self._draw_mask(frame, cx, cy, GOAL_HALO, COLOR_GOAL_HALO)
            if reachable and not is_current:
                for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3)):
                    frame[cy + dy, cx + dx] = COLOR_REACHABLE
            self._draw_mask(frame, cx, cy, ORDINARY_DIAMOND, COLOR_GOAL_CORE)
            self._draw_mask(frame, cx, cy, INNER_DIAMOND, COLOR_JUNCTION_FILL)
            return

        outline_color = COLOR_JUNCTION_OUTLINE
        if reachable:
            outline_color = COLOR_REACHABLE
        if is_current:
            outline_color = COLOR_CURRENT_OUTLINE
        self._draw_mask(frame, cx, cy, ORDINARY_DIAMOND, outline_color)
        self._draw_mask(frame, cx, cy, INNER_DIAMOND, COLOR_JUNCTION_FILL)

    def _draw_token(self, frame: np.ndarray) -> None:
        cx, cy = self._current_spec()["nodes"][self._current_node_id]
        self._draw_mask(frame, cx, cy, TOKEN_MASK, COLOR_TOKEN)

    def _draw_pips(self, frame: np.ndarray) -> None:
        for index in range(self._move_budget):
            color = COLOR_GOAL_HALO if index < self._moves_remaining else COLOR_SPENT_PIP
            left = 4 + (index * 4)
            frame[4:7, left : left + 3] = color

    def _draw_border(self, frame: np.ndarray, color: int) -> None:
        frame[0, :] = color
        frame[-1, :] = color
        frame[:, 0] = color
        frame[:, -1] = color

    def _redraw(self) -> None:
        frame = _solid_frame()
        spec = self._current_spec()
        for left, right in spec["edges"]:
            self._draw_corridor(frame, spec["nodes"][left], spec["nodes"][right])
        for node_id in spec["nodes"]:
            self._draw_junction(frame, node_id)
        self._draw_token(frame)
        self._draw_pips(frame)
        if self._mode in {"cleared_hold", "final_win"}:
            self._draw_border(frame, COLOR_REACHABLE)
        assert self._board_sprite is not None
        self._board_sprite.pixels = frame

    def step(self) -> None:
        if self._mode == "cleared_hold":
            self.next_level()
            self.complete_action()
            return

        if self._mode == "final_win":
            self.complete_action()
            return

        click = self._decode_click()
        if click is not None:
            clicked_neighbor = self._clicked_neighbor(click[0], click[1])
            if clicked_neighbor is not None:
                self._current_node_id = clicked_neighbor
                self._moves_remaining -= 1
                if self._current_node_id == self._destination_node_id:
                    self._mode = (
                        "final_win" if self._level_index_state == len(self._level_specs) - 1 else "cleared_hold"
                    )
                    if self._mode == "final_win":
                        self.next_level()
                elif self._moves_remaining == 0:
                    self.lose()
        self._redraw()
        self.complete_action()
