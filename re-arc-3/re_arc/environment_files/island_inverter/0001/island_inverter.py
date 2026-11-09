from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

GAME_ID = "island_inverter-0001"
VIEW_SIZE = 64
CANVAS = 5
WHITE = 0
LOCKED = 4
ACCESSIBLE = 11
RED = 8
STAR = 11

COLOR_BY_CODE = {"J": 1, "H": 2, "I": 3, "D": 6, "Y": 7, "X": 8, "A": 9, "G": 10, "B": 12, "Z": 13, "C": 14, "E": 15}

NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class IslandLevelSpec:
    name: str
    face: tuple[str, ...]
    back: tuple[str, ...]
    special: tuple[int, int]
    baseline: int

    @property
    def budget(self) -> int:
        return self.baseline * 6


LEVEL_SPECS = (
    IslandLevelSpec("First Turn", ("AAB", "AAB"), ("JDA", "CDA"), (0, 0), 1),
    IslandLevelSpec(
        "Borrowed Shore",
        ("ACCCC", "ACCCC", "AAABC", "CCCCC", "CCCCC"),
        ("BDDDD", "BDDDD", "BBBJD", "DDDDD", "DDDDD"),
        (2, 3),
        2,
    ),
    IslandLevelSpec(
        "Castaway Fragment",
        ("HHAHHD", "HHADDD", "HHAACH", "HHHHHH", "HHHHHH"),
        ("IIBIIC", "IIBCCC", "IIBCJI", "IIIIII", "IIIIII"),
        (2, 4),
        3,
    ),
    IslandLevelSpec(
        "Prepare the Inlet",
        ("HHAHHHH", "HHACCCH", "HAAACCG", "HAAACDG", "HHBAEAH", "BBBAAAH", "HHHHHHH"),
        ("IICIIII", "IICDDDI", "IBCCDDE", "IBBDEEE", "IIDDJDI", "DDDDDDI", "IIIIIII"),
        (4, 4),
        6,
    ),
    IslandLevelSpec(
        "Three Strands",
        (
            "BHHAHHHDHHH",
            "BHHAHHHDHHH",
            "BHAAHHDDHHH",
            "BHAAHDDHHHH",
            "BBBAHDBBBBB",
            "GGGXYZEHHHH",
            "BBBBCHHHHHH",
            "BHHHCCHHHHH",
            "BHHHCCHHHHH",
            "BHHHCHHHHHH",
            "BHHHCHHHHHH",
        ),
        (
            "XIIBIIIBIII",
            "XIIBIIIBIII",
            "XIBBIIBBIII",
            "XIBBIBBIIII",
            "XXXXIZZZZZZ",
            "EEEEEEJIIII",
            "YYYYYIIIIII",
            "YIIIBBIIIII",
            "YIIIBBIIIII",
            "YIIIBIIIIII",
            "YIIIBIIIIII",
        ),
        (5, 6),
        11,
    ),
    IslandLevelSpec(
        "The Shore Closes",
        (
            "HHAHBHAHHHH",
            "HHAHBHAHHHH",
            "HHAHBHAHHHH",
            "HHAABHAHHHH",
            "HHHABAAHHHH",
            "GGGXYEXEYEH",
            "HHHHCBHCHHH",
            "HHCCCBHCHHH",
            "HHCHHBHCHHH",
            "HHCHHBHCHHH",
            "HHCHHBHCHHH",
        ),
        (
            "IIBIXIXIIII",
            "IIBIXIXIIII",
            "IIBIXIXIIII",
            "IIBBXIXIIII",
            "IIIXXXXIIII",
            "EEEEEXEYEJI",
            "IIIIYYIYIII",
            "IIBBBYIYIII",
            "IIBIIYIYIII",
            "IIBIIYIYIII",
            "IIBIIYIYIII",
        ),
        (5, 9),
        30,
    ),
    IslandLevelSpec(
        "Nested Tides",
        (
            "HHAHBHHHDHHHHB",
            "HHAHBHHHDHHHHB",
            "HHAHBHHHDHHHHB",
            "HHAHBHHDDHHHHB",
            "HHAABHHDBBBBBB",
            "HHHABDDDBHHHHH",
            "GGGXYZEXEYEZEH",
            "HHHHCBAHCHDHHH",
            "HHHCCBAHCHDHHH",
            "HHHCHBAHCHDHHH",
            "HHHCHBAHCHDHHH",
            "HHHCHBAHCHDHHH",
            "HHHCHBAHCHDHHH",
            "HHHCHBAHCHDHHH",
        ),
        (
            "IIBIXIIIBIIIIZ",
            "IIBIXIIIBIIIIZ",
            "IIBIXIIIBIIIIZ",
            "IIBIXIIBBIIIIZ",
            "IIBBXIIBZZZZZZ",
            "IIIXXZZZZIIIII",
            "EEEEEEXEYEZEJI",
            "IIIIYYXIYIZIII",
            "IIIBBYXIYIZIII",
            "IIIBIYXIYIZIII",
            "IIIBIYXIYIZIII",
            "IIIBIYXIYIZIII",
            "IIIBIYXIYIZIII",
            "IIIBIYXIYIZIII",
        ),
        (6, 12),
        56,
    ),
)


class StepBar(RenderableUserDisplay):
    def __init__(self) -> None:
        self.remaining = 1
        self.budget = 1

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        width = 60
        filled = max(0, min(width, round(width * self.remaining / max(1, self.budget))))
        frame[62:64, 2:62] = LOCKED
        if filled:
            color = ACCESSIBLE if self.remaining * 4 > self.budget else RED
            frame[62:64, 2 : 2 + filled] = color
        return frame


def _build_level(spec: IslandLevelSpec) -> Level:
    # FACE and BACK remain readable ASCII logical grids in Level.data.
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), CANVAS, dtype=np.int8),
        name="island_board",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["board"],
    )
    return Level(sprites=[board], grid_size=(VIEW_SIZE, VIEW_SIZE), name=spec.name, data={"spec": spec})


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


class IslandInverter(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._step_bar = StepBar()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, CANVAS, CANVAS, [self._step_bar]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: IslandLevelSpec = level.get_data("spec")
        self.height = len(self.spec.face)
        self.width = len(self.spec.face[0])
        if any(len(row) != self.width for row in (*self.spec.face, *self.spec.back)):
            raise ValueError(f"Non-rectangular Island Inverter level: {self.spec.name}")
        if len(self.spec.back) != self.height:
            raise ValueError(f"FACE/BACK height mismatch: {self.spec.name}")

        self.side_0 = [[COLOR_BY_CODE[ch] for ch in row] for row in self.spec.face]
        self.side_1 = [[COLOR_BY_CODE[ch] for ch in row] for row in self.spec.back]
        self.orientation = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.special_tile = self.spec.special
        self.terminal = False
        self.step_count = 0
        self.feedback_kind = "NONE"
        self.feedback_cells: set[tuple[int, int]] = set()
        self.feedback_frames_remaining = 0
        self.selected_cell = None
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self._step_bar.budget = self.step_budget
        self._step_bar.remaining = self.remaining_steps
        self._undo_history: list[tuple[tuple[int, ...], ...]] = []
        self._animation: list[dict[str, object]] = []
        self._pending_result: str | None = None
        self._board_sprite = self.current_level.get_sprites_by_tag("board")[0]
        self._sync_board()

    def _current_color(self, row: int, col: int, orientations: list[list[int]] | None = None) -> int:
        state = self.orientation if orientations is None else orientations
        return self.side_1[row][col] if state[row][col] else self.side_0[row][col]

    def _reverse_color(self, row: int, col: int, orientations: list[list[int]] | None = None) -> int:
        state = self.orientation if orientations is None else orientations
        return self.side_0[row][col] if state[row][col] else self.side_1[row][col]

    def _component_at(self, row: int, col: int, orientations: list[list[int]] | None = None) -> set[tuple[int, int]]:
        state = self.orientation if orientations is None else orientations
        color = self._current_color(row, col, state)
        component = {(row, col)}
        frontier = [(row, col)]
        while frontier:
            cr, cc = frontier.pop()
            for dr, dc in NEIGHBORS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < self.height and 0 <= nc < self.width):
                    continue
                if (nr, nc) in component or self._current_color(nr, nc, state) != color:
                    continue
                component.add((nr, nc))
                frontier.append((nr, nc))
        return component

    def _is_accessible(self, component: set[tuple[int, int]]) -> bool:
        return any(r in (0, self.height - 1) or c in (0, self.width - 1) for r, c in component)

    def _components(self, orientations: list[list[int]] | None = None) -> list[set[tuple[int, int]]]:
        state = self.orientation if orientations is None else orientations
        unseen = {(r, c) for r in range(self.height) for c in range(self.width)}
        out: list[set[tuple[int, int]]] = []
        while unseen:
            r, c = min(unseen)
            component = self._component_at(r, c, state)
            unseen.difference_update(component)
            out.append(component)
        return out

    def _geometry(self) -> tuple[int, int, int]:
        pitch = min(10, 56 // max(self.height, self.width))
        origin_x = (VIEW_SIZE - self.width * pitch) // 2
        origin_y = (VIEW_SIZE - self.height * pitch) // 2
        return pitch, origin_x, origin_y

    def _draw_icon(self, frame: np.ndarray, orientation: int, full_star: bool = False) -> None:
        pitch, ox, oy = self._geometry()
        row, col = self.special_tile
        x = ox + col * pitch
        y = oy + row * pitch
        cx, cy = x + pitch // 2, y + pitch // 2
        if orientation == 0:
            if pitch == 4:
                frame[cy, cx] = CANVAS
                frame[max(y + 1, cy - 1), cx] = 13
                frame[cy, max(x + 1, cx - 1)] = 13
            else:
                frame[cy - 1 : cy + 2, cx - 2 : cx + 3] = 13
                frame[cy, cx - 1 : cx + 2] = CANVAS
                frame[cy - 2, cx - 1 : cx + 2] = CANVAS
        else:
            radius = 2 if full_star and pitch >= 5 else 1
            frame[cy, cx] = STAR
            for delta in range(1, radius + 1):
                if cy - delta >= y + 1:
                    frame[cy - delta, cx] = STAR
                if cy + delta < y + pitch - 1:
                    frame[cy + delta, cx] = STAR
                if cx - delta >= x + 1:
                    frame[cy, cx - delta] = STAR
                if cx + delta < x + pitch - 1:
                    frame[cy, cx + delta] = STAR

    def _draw_reverse_mark(self, frame: np.ndarray, orientation: int) -> None:
        pitch, ox, oy = self._geometry()
        row, col = self.special_tile
        x = ox + col * pitch
        y = oy + row * pitch
        size = 2 if pitch >= 5 else 1
        sx, sy = x + pitch - size - 1, y + pitch - size - 1
        if orientation == 0:
            frame[sy : sy + size, sx : sx + size] = STAR
        else:
            frame[sy : sy + size, sx : sx + size] = CANVAS
            if size == 2:
                frame[sy + 1, sx : sx + 2] = 13

    def _sync_board(
        self,
        *,
        orientations: list[list[int]] | None = None,
        affected: set[tuple[int, int]] | None = None,
        mode: str = "settled",
        outline_override: int | None = None,
        halo_cells: set[tuple[int, int]] | None = None,
        full_star: bool = False,
    ) -> None:
        state = self.orientation if orientations is None else orientations
        pitch, ox, oy = self._geometry()
        frame = np.full((VIEW_SIZE, VIEW_SIZE), CANVAS, dtype=np.int8)
        affected = affected or set()

        components = self._components(state)
        access_by_cell: dict[tuple[int, int], bool] = {}
        for component in components:
            accessible = self._is_accessible(component)
            for cell in component:
                access_by_cell[cell] = accessible

        for row in range(self.height):
            for col in range(self.width):
                x, y = ox + col * pitch, oy + row * pitch
                current = self._current_color(row, col, state)
                reverse = self._reverse_color(row, col, state)
                if (row, col) in affected and mode in {"compress", "expand"}:
                    frame[y : y + pitch, x : x + pitch] = CANVAS
                    strip_width = 1 if mode == "compress" else max(1, pitch // 2)
                    left = x + (pitch - strip_width) // 2
                    frame[y : y + pitch, left : left + strip_width] = current
                else:
                    frame[y : y + pitch, x : x + pitch] = current
                    size = 2 if pitch >= 5 else 1
                    sx, sy = x + pitch - size - 1, y + pitch - size - 1
                    frame[sy : sy + size, sx : sx + size] = reverse

        draw_outlines = mode in {"settled", "flash", "halo"}
        if draw_outlines:
            for row in range(self.height):
                for col in range(self.width):
                    x, y = ox + col * pitch, oy + row * pitch
                    color = ACCESSIBLE if access_by_cell[(row, col)] else LOCKED
                    if outline_override is not None and (row, col) in affected:
                        color = outline_override
                    if halo_cells is not None and (row, col) in halo_cells:
                        color = outline_override if outline_override is not None else STAR
                    current = self._current_color(row, col, state)
                    for dr, dc, edge in ((-1, 0, "top"), (1, 0, "bottom"), (0, -1, "left"), (0, 1, "right")):
                        nr, nc = row + dr, col + dc
                        boundary = not (0 <= nr < self.height and 0 <= nc < self.width)
                        partition = boundary or self._current_color(nr, nc, state) != current
                        if not partition:
                            continue
                        if edge == "top":
                            frame[y, x : x + pitch] = color
                        elif edge == "bottom":
                            frame[y + pitch - 1, x : x + pitch] = color
                        elif edge == "left":
                            frame[y : y + pitch, x] = color
                        else:
                            frame[y : y + pitch, x + pitch - 1] = color

            # One outward tab per maximal board-boundary run of each accessible component.
            for component in components:
                if not self._is_accessible(component):
                    continue
                tab_color = outline_override if outline_override is not None and component & affected else ACCESSIBLE
                for edge, cells in (
                    ("top", sorted(c for r, c in component if r == 0)),
                    ("bottom", sorted(c for r, c in component if r == self.height - 1)),
                    ("left", sorted(r for r, c in component if c == 0)),
                    ("right", sorted(r for r, c in component if c == self.width - 1)),
                ):
                    if not cells:
                        continue
                    runs: list[list[int]] = [[cells[0]]]
                    for value in cells[1:]:
                        if value == runs[-1][-1] + 1:
                            runs[-1].append(value)
                        else:
                            runs.append([value])
                    for run in runs:
                        midpoint = run[len(run) // 2]
                        if edge == "top" and oy > 0:
                            frame[oy - 1, ox + midpoint * pitch + pitch // 2] = tab_color
                        elif edge == "bottom" and oy + self.height * pitch < VIEW_SIZE - 2:
                            frame[oy + self.height * pitch, ox + midpoint * pitch + pitch // 2] = tab_color
                        elif edge == "left" and ox > 0:
                            frame[oy + midpoint * pitch + pitch // 2, ox - 1] = tab_color
                        elif edge == "right" and ox + self.width * pitch < VIEW_SIZE:
                            frame[oy + midpoint * pitch + pitch // 2, ox + self.width * pitch] = tab_color

        special_orientation = state[self.special_tile[0]][self.special_tile[1]]
        if mode not in {"compress", "expand"} or self.special_tile not in affected:
            self._draw_icon(frame, special_orientation, full_star=full_star)
            self._draw_reverse_mark(frame, special_orientation)
        if mode == "flash" and affected:
            for row, col in affected:
                x, y = ox + col * pitch, oy + row * pitch
                frame[y + pitch // 2, x + pitch // 2] = WHITE

        self._board_sprite.pixels = frame
        self._step_bar.remaining = self.remaining_steps

    def _spend_action(self) -> None:
        self.step_count += 1
        self.remaining_steps = max(0, self.remaining_steps - 1)
        self._step_bar.remaining = self.remaining_steps

    def _clicked_cell(self) -> tuple[int, int] | None:
        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        grid = self.camera.display_to_grid(x, y)
        if grid is None:
            return None
        gx, gy = grid
        pitch, ox, oy = self._geometry()
        col = (gx - ox) // pitch
        row = (gy - oy) // pitch
        if gx < ox or gy < oy or not (0 <= row < self.height and 0 <= col < self.width):
            return None
        return row, col

    def _snapshot(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self.orientation)

    def _restore(self, snapshot: tuple[tuple[int, ...], ...]) -> None:
        self.orientation = [list(row) for row in snapshot]

    def _queue_locked(self, component: set[tuple[int, int]]) -> None:
        self.feedback_kind = "LOCKED"
        self.feedback_cells = set(component)
        self._animation = [
            {"affected": component, "mode": "settled"},
            {"affected": component, "mode": "settled", "outline_override": WHITE},
            {"affected": component, "mode": "settled", "outline_override": RED},
            {"affected": component, "mode": "settled"},
        ]
        self.feedback_frames_remaining = len(self._animation)

    def _queue_flip(self, component: set[tuple[int, int]], old_state: list[list[int]], winning: bool) -> None:
        self.feedback_kind = "WIN" if winning else "FLIP"
        self.feedback_cells = set(component)
        self._animation = [
            {"orientations": old_state, "affected": component, "mode": "flash", "outline_override": WHITE},
            {"orientations": old_state, "affected": component, "mode": "compress"},
            {"affected": component, "mode": "expand"},
            {"affected": component, "mode": "faces"},
            {"affected": component, "mode": "settled"},
        ]
        if winning:
            star_component = self._component_at(*self.special_tile)
            self._animation.extend(
                [
                    {"mode": "halo", "halo_cells": star_component, "outline_override": STAR, "full_star": True},
                    {"mode": "halo", "halo_cells": star_component, "outline_override": WHITE, "full_star": True},
                    {"mode": "halo", "halo_cells": star_component, "outline_override": STAR, "full_star": True},
                    {"mode": "halo", "halo_cells": star_component, "outline_override": WHITE, "full_star": True},
                ]
            )
            self._pending_result = "win"
        elif self.remaining_steps == 0:
            self._pending_result = "lose"
        self.feedback_frames_remaining = len(self._animation)

    def _play_animation_frame(self) -> None:
        config = self._animation.pop(0)
        self.feedback_frames_remaining = len(self._animation)
        self._sync_board(**config)
        if self._animation:
            return
        self.feedback_kind = "NONE"
        self.feedback_cells.clear()
        result = self._pending_result
        self._pending_result = None
        if result == "win":
            self.terminal = True
            self.next_level()
        elif result == "lose":
            self.terminal = True
            self.lose()
        self.complete_action()

    def step(self) -> None:
        if self._animation:
            self._play_animation_frame()
            return

        action = _action_id(self.action.id)
        if action == 0:
            self._sync_board()
            self.complete_action()
            return

        self._spend_action()

        if action == 7:
            if self._undo_history:
                self._restore(self._undo_history.pop())
            self.feedback_kind = "NONE"
            self.feedback_cells.clear()
            self._sync_board()
            if self.remaining_steps == 0:
                self.terminal = True
                self.lose()
            self.complete_action()
            return

        if action != 6:
            self._sync_board()
            if self.remaining_steps == 0:
                self.terminal = True
                self.lose()
            self.complete_action()
            return

        cell = self._clicked_cell()
        if cell is None:
            self._sync_board()
            if self.remaining_steps == 0:
                self.terminal = True
                self.lose()
            self.complete_action()
            return

        component = self._component_at(*cell)
        if not self._is_accessible(component):
            self._queue_locked(component)
            if self.remaining_steps == 0:
                self._pending_result = "lose"
            self._play_animation_frame()
            return

        old_state = [row[:] for row in self.orientation]
        self._undo_history.append(self._snapshot())
        for row, col in component:
            self.orientation[row][col] ^= 1
        winning = self.orientation[self.special_tile[0]][self.special_tile[1]] == 1
        self._queue_flip(component, old_state, winning)
        self._play_animation_frame()

    def _get_hidden_state(self) -> np.ndarray:
        flat = [value for row in self.orientation for value in row]
        return np.asarray([*flat, self.remaining_steps], dtype=np.int16)
