from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "painttheroom-0001"

C_VOID = 0
C_WALL = 1
C_FLOOR = 2
C_TARGET = 3
C_PAINTED = 4
C_CURSOR_A = 5
C_CURSOR_B = 6
C_FORBIDDEN = 7
C_GATE_CLOSED = 8
C_GATE_OPEN = 9
C_LEVER = 10
C_TIME_FILL = 11
C_TIME_EMPTY = 12
C_CLEANER = 13
C_EFFECT = 14
C_STAIN = 15

MOVE_DELTAS: dict[int, tuple[int, int]] = {
    int(GameAction.ACTION1.value): (0, -1),
    int(GameAction.ACTION2.value): (0, 1),
    int(GameAction.ACTION3.value): (-1, 0),
    int(GameAction.ACTION4.value): (1, 0),
}


class LevelSpec:
    def __init__(
        self, *, name: str, time_limit: int, rows: tuple[str, ...], cleaner_dirs: tuple[tuple[int, int], ...] = ()
    ) -> None:
        self.name = str(name)
        self.time_limit = int(time_limit)
        self.rows = tuple(str(row) for row in rows)
        self.cleaner_dirs = tuple((int(dx), int(dy)) for (dx, dy) in cleaner_dirs)


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        name="Level 1 - First coat",
        time_limit=1000,
        rows=(
            "========================",
            "========================",
            "########################",
            "#S.....................#",
            "#......................#",
            "#..........TTTT........#",
            "#..........TTTT........#",
            "#..........TTTT........#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "#......................#",
            "########################",
        ),
    ),
    LevelSpec(
        name="Level 2 - Rooms and corridors",
        time_limit=1000,
        rows=(
            "========================",
            "====================----",
            "########################",
            "#S.....#...............#",
            "#......#..TTTTT........#",
            "#......#..TTTTT........#",
            "#......#...............#",
            "#......#######..####...#",
            "#......#.....#..#......#",
            "#......#.....#..#..TT..#",
            "#......#.....#..#..TT..#",
            "#......#.....#..#......#",
            "#......#........#......#",
            "#..........#######.....#",
            "#......#...............#",
            "#......#....TTTT.......#",
            "#......#....TTTT.......#",
            "########################",
        ),
    ),
    LevelSpec(
        name="Level 3 - Switch and gate",
        time_limit=1000,
        rows=(
            "========================",
            "==================------",
            "########################",
            "#S.............#.......#",
            "#.######.......#..L....#",
            "#......#.......#.......#",
            "#......#...............#",
            "#......###########.....#",
            "#......................#",
            "##########DD############",
            "#......................#",
            "#.....TTTTTT...........#",
            "#.....TTTTTT..#####....#",
            "#.............#........#",
            "#....#####....#....TT..#",
            "#....#####....#....TT..#",
            "#.............#........#",
            "########################",
        ),
    ),
    LevelSpec(
        name="Level 4 - Paint carefully",
        time_limit=1000,
        rows=(
            "========================",
            "================--------",
            "########################",
            "#S.....................#",
            "#.............L........#",
            "#......................#",
            "#....######............#",
            "#....#....#............#",
            "#....#....#............#",
            "##########DD############",
            "#......................#",
            "#.....TTTTXTTTT........#",
            "#.....TTTXXXTTT........#",
            "#.....TTTTXTTTT........#",
            "#......................#",
            "#..........TTTT........#",
            "#..........TTTT........#",
            "########################",
        ),
    ),
    LevelSpec(
        name="Level 5 - Stains and a cleaner",
        time_limit=1000,
        rows=(
            "========================",
            "==============----------",
            "########################",
            "#S.....................#",
            "#..........L...........#",
            "#......................#",
            "#....######..######....#",
            "#....#............#....#",
            "#....#..*****TT...#....#",
            "#....#..*...*TT...#....#",
            "#....#..*H..*TT...#....#",
            "#....#..*H..*TT...#....#",
            "#....#..*...*TT...#....#",
            "#....#..*****TT...#....#",
            "#....#............#....#",
            "#....######DD######....#",
            "#......................#",
            "#..........TTTT........#",
            "#..........TTTT........#",
            "########################",
        ),
        cleaner_dirs=((1, 0),),
    ),
    LevelSpec(
        name="Level 6 - Everything together",
        time_limit=1000,
        rows=(
            "============================",
            "============----------------",
            "############################",
            "#S.............#...........#",
            "#.######.......#..L........#",
            "#......#.......#...........#",
            "#......#...................#",
            "#......###########DD########",
            "#.............#............#",
            "#..TTTTT......#..TTTXTT....#",
            "#..TTTTT......#..TTXXXTT...#",
            "#.............#..TTTXTT....#",
            "#....#..H......#...........#",
            "#....#..H......#...........#",
            "#....#.........#...........#",
            "#....###########DD##########",
            "#..##########..............#",
            "#..#*****..H#....TTTT......#",
            "#..#*...*..H....TTTXTT.....#",
            "#..#*****...#...TTXXXTT..X.#",
            "#..##########..............#",
            "############################",
        ),
        cleaner_dirs=((1, 0), (0, 1)),
    ),
)

ACTIVE_LEVEL_COUNT = 4


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _build_level(spec: LevelSpec) -> Level:
    rows = list(spec.rows)
    height = len(rows)
    width = len(rows[0])
    for row in rows:
        if len(row) != width:
            raise ValueError(f"Inconsistent row width in level {spec.name}")

    start: tuple[int, int] | None = None
    lever: tuple[int, int] | None = None
    gate_cells: list[tuple[int, int]] = []
    cleaner_cells: list[tuple[int, int]] = []

    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if y < 2:
                continue
            if ch == "S":
                start = (x, y)
            elif ch == "L":
                lever = (x, y)
            elif ch == "D":
                gate_cells.append((x, y))
            elif ch == "H":
                cleaner_cells.append((x, y))

    if start is None:
        raise ValueError(f"Missing start position in level {spec.name}")

    gate_cells_set = set(gate_cells)
    gates: list[list[tuple[int, int]]] = []
    while gate_cells_set:
        x0, y0 = next(iter(gate_cells_set))
        if (x0 + 1, y0) in gate_cells_set:
            pair = [(x0, y0), (x0 + 1, y0)]
        elif (x0 - 1, y0) in gate_cells_set:
            pair = [(x0 - 1, y0), (x0, y0)]
        elif (x0, y0 + 1) in gate_cells_set:
            pair = [(x0, y0), (x0, y0 + 1)]
        elif (x0, y0 - 1) in gate_cells_set:
            pair = [(x0, y0 - 1), (x0, y0)]
        else:
            raise ValueError(f"Unpaired gate cell {(x0, y0)} in level {spec.name}")
        for cell in pair:
            gate_cells_set.discard(cell)
        gates.append(pair)

    cleaner_cells_set = set(cleaner_cells)
    cleaners: list[tuple[int, int]] = []
    while cleaner_cells_set:
        x0, y0 = next(iter(cleaner_cells_set))
        if (x0, y0 + 1) in cleaner_cells_set:
            top = (x0, y0)
            bottom = (x0, y0 + 1)
        elif (x0, y0 - 1) in cleaner_cells_set:
            top = (x0, y0 - 1)
            bottom = (x0, y0)
        else:
            raise ValueError(f"Cleaner is not vertical at {(x0, y0)} in level {spec.name}")
        cleaner_cells_set.discard(top)
        cleaner_cells_set.discard(bottom)
        cleaners.append(top)

    if len(cleaners) != len(spec.cleaner_dirs):
        raise ValueError(
            f"Cleaner direction count mismatch in {spec.name}: cells={len(cleaners)} dirs={len(spec.cleaner_dirs)}"
        )

    sprites = [
        Sprite(
            pixels=_solid(width, height, C_VOID),
            name="board",
            x=0,
            y=0,
            layer=0,
            tags=["board", "sys_click", "sys_every_pixel"],
            collidable=False,
        )
    ]

    return Level(
        name=spec.name,
        grid_size=(width, height),
        sprites=sprites,
        data={
            "rows": rows,
            "time_limit": int(spec.time_limit),
            "start": [int(start[0]), int(start[1])],
            "lever": [int(lever[0]), int(lever[1])] if lever is not None else None,
            "gates": [[[int(x), int(y)] for (x, y) in gate] for gate in gates],
            "cleaners": [
                {"top": [int(top[0]), int(top[1])], "dir": [int(direction[0]), int(direction[1])]}
                for top, direction in zip(cleaners, spec.cleaner_dirs, strict=False)
            ],
        },
    )


class Painttheroom(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [_build_level(spec) for spec in LEVEL_SPECS[:ACTIVE_LEVEL_COUNT]]
        first = levels[0]
        width, height = first.grid_size or (64, 64)
        camera = Camera(width=width, height=height, background=C_VOID)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

        self._width = 0
        self._height = 0
        self._time_limit = 0
        self._time_left = 0
        self._turn = 0

        self._board: Sprite | None = None
        self._rows: list[str] = []
        self._cursor = (0, 0)
        self._lever_cell: tuple[int, int] | None = None
        self._lever_on = False
        self._lever_flash_turn = -1

        self._wall_cells: set[tuple[int, int]] = set()
        self._forbidden_cells: set[tuple[int, int]] = set()
        self._gate_cells: set[tuple[int, int]] = set()
        self._gate_groups: list[list[tuple[int, int]]] = []
        self._gate_open = False
        self._gate_pending_target: bool | None = None
        self._gate_flash_turn = -1
        self._gate_apply_turn = -1

        self._cleaners: list[dict[str, object]] = []

        self._required_cells: list[tuple[int, int]] = []
        self._required_kind: dict[tuple[int, int], str] = {}
        self._cell_state: dict[tuple[int, int], int] = {}

        self._pending_paint: list[dict[str, int]] = []
        self._overlay_cells: set[tuple[int, int]] = set()

        self._fail_active = False
        self._fail_cells: set[tuple[int, int]] = set()

    def on_set_level(self, level: Level) -> None:
        rows = [str(row) for row in (level.get_data("rows") or [])]
        if not rows:
            raise ValueError("painttheroom level is missing rows")

        self._rows = rows
        self._height = len(rows)
        self._width = len(rows[0])
        self.camera.width = self._width
        self.camera.height = self._height

        self._time_limit = int(level.get_data("time_limit") or (2 * self._width))
        self._time_left = int(self._time_limit)
        self._turn = 0

        start = level.get_data("start") or [1, 2]
        self._cursor = (int(start[0]), int(start[1]))

        lever = level.get_data("lever")
        self._lever_cell = (int(lever[0]), int(lever[1])) if lever is not None else None
        self._lever_on = False
        self._lever_flash_turn = -1

        self._wall_cells = set()
        self._forbidden_cells = set()
        self._required_cells = []
        self._required_kind = {}
        self._cell_state = {}

        for y in range(2, self._height):
            row = rows[y]
            for x, ch in enumerate(row):
                cell = (x, y)
                if ch == "#":
                    self._wall_cells.add(cell)
                elif ch == "X":
                    self._forbidden_cells.add(cell)
                    self._cell_state[cell] = C_FORBIDDEN
                elif ch == "T":
                    self._required_cells.append(cell)
                    self._required_kind[cell] = "target"
                    self._cell_state[cell] = C_TARGET
                elif ch == "*":
                    self._required_cells.append(cell)
                    self._required_kind[cell] = "stain"
                    self._cell_state[cell] = C_STAIN
                elif ch in {".", "S", "H"}:
                    self._cell_state[cell] = C_FLOOR
                elif ch == "L":
                    self._cell_state[cell] = C_LEVER
                elif ch == "D":
                    self._cell_state[cell] = C_GATE_CLOSED
                else:
                    self._cell_state[cell] = C_FLOOR

        gates = level.get_data("gates") or []
        self._gate_groups = [[(int(cell[0]), int(cell[1])) for cell in gate] for gate in gates]
        self._gate_cells = {cell for gate in self._gate_groups for cell in gate}
        self._gate_open = False
        self._gate_pending_target = None
        self._gate_flash_turn = -1
        self._gate_apply_turn = -1

        self._cleaners = []
        for item in level.get_data("cleaners") or []:
            top = (int(item["top"][0]), int(item["top"][1]))
            direction = (int(item["dir"][0]), int(item["dir"][1]))
            self._cleaners.append({"top": top, "dir": direction})

        self._pending_paint = []
        self._overlay_cells = set()

        self._fail_active = False
        self._fail_cells = set()

        boards = self.current_level.get_sprites_by_name("board")
        if not boards:
            raise ValueError("painttheroom is missing board sprite")
        self._board = boards[0]

        self._render()

    def _is_in_playfield(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 2 <= y < self._height

    def _is_closed_gate(self, x: int, y: int) -> bool:
        return (x, y) in self._gate_cells and not self._gate_open

    def _is_wall_like(self, x: int, y: int) -> bool:
        return (x, y) in self._wall_cells or self._is_closed_gate(x, y)

    def _cursor_walkable(self, x: int, y: int) -> bool:
        if not self._is_in_playfield(x, y):
            return False
        if (x, y) in self._wall_cells:
            return False
        return not ((x, y) in self._gate_cells and not self._gate_open)

    def _cleaner_blocked(self, top: tuple[int, int]) -> bool:
        tx, ty = top
        bx, by = tx, ty + 1
        for x, y in ((tx, ty), (bx, by)):
            if not self._is_in_playfield(x, y):
                return True
            if (x, y) in self._wall_cells:
                return True
            if (x, y) in self._gate_cells and not self._gate_open:
                return True
        return False

    def _advance_pending_transitions(self) -> None:
        if self._gate_pending_target is not None and self._turn >= self._gate_apply_turn:
            self._gate_open = bool(self._gate_pending_target)
            self._gate_pending_target = None

        if not self._pending_paint:
            return
        self._pending_paint = []

    def _trigger_fail(self, fail_cells: set[tuple[int, int]]) -> None:
        self._fail_active = True
        self._fail_cells = set(fail_cells)

    def _toggle_lever(self) -> None:
        self._lever_on = not self._lever_on
        self._lever_flash_turn = self._turn

        source_gate_state = (
            bool(self._gate_pending_target) if self._gate_pending_target is not None else bool(self._gate_open)
        )
        self._gate_pending_target = not source_gate_state
        self._gate_flash_turn = self._turn + 1
        self._gate_apply_turn = self._turn + 2

    def _splash_cells(self, x: int, y: int) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for dy in (-1, 0):
            for dx in (-1, 0):
                px = x + dx
                py = y + dy
                if self._is_in_playfield(px, py):
                    cells.add((px, py))
        return cells

    def _beam_cells(self, x0: int, y0: int, x1: int, y1: int) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        if x0 == x1:
            step = 1 if y1 >= y0 else -1
            for y in range(y0, y1 + step, step):
                if self._is_in_playfield(x0, y):
                    out.add((x0, y))
        elif y0 == y1:
            step = 1 if x1 >= x0 else -1
            for x in range(x0, x1 + step, step):
                if self._is_in_playfield(x, y0):
                    out.add((x, y0))
        return out

    def _apply_paint(self, x: int, y: int, *, with_beam_from: tuple[int, int] | None) -> None:
        if not self._is_in_playfield(x, y):
            return
        if (x, y) in self._wall_cells:
            return
        if (x, y) in self._gate_cells:
            return
        if (x, y) in self._forbidden_cells:
            self._trigger_fail({(x, y), self._cursor})
            return

        state = int(self._cell_state.get((x, y), C_FLOOR))
        next_state = state

        if state == C_STAIN:
            next_state = C_TARGET
        elif state in (C_TARGET, C_PAINTED, C_FLOOR):
            next_state = C_PAINTED
        elif state == C_LEVER:
            return
        else:
            next_state = C_PAINTED

        self._cell_state[(x, y)] = int(next_state)

        overlay = self._splash_cells(x, y)
        if with_beam_from is not None:
            overlay |= self._beam_cells(with_beam_from[0], with_beam_from[1], x, y)
        self._overlay_cells |= overlay

    def _parse_click(self) -> tuple[int, int] | None:
        payload = self.action.data if isinstance(self.action.data, dict) else None
        if not payload:
            return None

        try:
            raw_x = int(payload.get("x", -999))
            raw_y = int(payload.get("y", -999))
        except (TypeError, ValueError):
            return None

        if 0 <= raw_x < self._width and 0 <= raw_y < self._height:
            return (raw_x, raw_y)

        if raw_x < 0 or raw_y < 0:
            return None

        to_grid = getattr(self.camera, "display_to_grid", None)
        if callable(to_grid):
            grid = to_grid(raw_x, raw_y)
            if grid is None:
                return None
            gx, gy = int(grid[0]), int(grid[1])
            if 0 <= gx < self._width and 0 <= gy < self._height:
                return (gx, gy)
        return None

    def _click_valid_path(self, target: tuple[int, int]) -> bool:
        cx, cy = self._cursor
        tx, ty = target
        if tx != cx and ty != cy:
            return False

        if tx == cx:
            step = 1 if ty > cy else -1
            for y in range(cy + step, ty, step):
                if self._is_wall_like(tx, y):
                    return False
        else:
            step = 1 if tx > cx else -1
            for x in range(cx + step, tx, step):
                if self._is_wall_like(x, ty):
                    return False
        return True

    def _move_cleaners(self) -> None:
        for cleaner in self._cleaners:
            tx, ty = cleaner["top"]  # type: ignore[index]
            dx, dy = cleaner["dir"]  # type: ignore[index]

            candidate = (int(tx + dx), int(ty + dy))
            if self._cleaner_blocked(candidate):
                dx, dy = -int(dx), -int(dy)
                cleaner["dir"] = (dx, dy)
                candidate = (int(tx + dx), int(ty + dy))

            if not self._cleaner_blocked(candidate):
                cleaner["top"] = candidate

            top_x, top_y = cleaner["top"]  # type: ignore[index]
            body = ((int(top_x), int(top_y)), (int(top_x), int(top_y) + 1))
            for cell in body:
                state = int(self._cell_state.get(cell, C_FLOOR))
                if state != C_PAINTED:
                    continue
                kind = self._required_kind.get(cell)
                if kind == "target" or kind == "stain":
                    self._cell_state[cell] = C_TARGET
                else:
                    self._cell_state[cell] = C_FLOOR

    def _check_win(self) -> bool:
        for cell in self._required_cells:
            if int(self._cell_state.get(cell, C_FLOOR)) != C_PAINTED:
                return False
        return True

    def _timebar(self, frame: np.ndarray) -> None:
        width = self._width
        total = 2 * width
        filled = max(0, min(total, int(self._time_left)))

        row1 = min(width, filled)
        row0 = max(0, filled - width)

        frame[0, :] = np.int8(C_TIME_EMPTY)
        frame[1, :] = np.int8(C_TIME_EMPTY)
        if row0 > 0:
            frame[0, :row0] = np.int8(C_TIME_FILL)
        if row1 > 0:
            frame[1, :row1] = np.int8(C_TIME_FILL)

    def _render(self) -> None:
        if self._board is None:
            return

        frame = np.full((self._height, self._width), np.int8(C_VOID), dtype=np.int8)
        self._timebar(frame)

        show_gate_flash = self._gate_pending_target is not None and self._turn == self._gate_flash_turn
        show_lever_flash = self._lever_cell is not None and self._turn == self._lever_flash_turn

        for y in range(2, self._height):
            for x in range(self._width):
                cell = (x, y)
                if cell in self._wall_cells:
                    frame[y, x] = np.int8(C_WALL)
                    continue
                if cell in self._gate_cells:
                    if show_gate_flash:
                        frame[y, x] = np.int8(C_EFFECT)
                    else:
                        frame[y, x] = np.int8(C_GATE_OPEN if self._gate_open else C_GATE_CLOSED)
                    continue

                state = int(self._cell_state.get(cell, C_FLOOR))
                if self._lever_cell is not None and cell == self._lever_cell:
                    frame[y, x] = np.int8(C_EFFECT if show_lever_flash else C_LEVER)
                else:
                    frame[y, x] = np.int8(state)

        for idx, cleaner in enumerate(self._cleaners):
            top_x, top_y = cleaner["top"]  # type: ignore[index]
            tx = int(top_x)
            ty = int(top_y)
            top_color = C_EFFECT if ((self._turn + idx) % 2 == 0) else C_CLEANER
            if self._is_in_playfield(tx, ty):
                frame[ty, tx] = np.int8(top_color)
            if self._is_in_playfield(tx, ty + 1):
                frame[ty + 1, tx] = np.int8(C_CLEANER)

        for x, y in self._overlay_cells:
            if self._is_in_playfield(x, y):
                frame[y, x] = np.int8(C_EFFECT)

        if self._fail_active:
            for x, y in self._fail_cells:
                if 0 <= x < self._width and 0 <= y < self._height:
                    frame[y, x] = np.int8(C_FORBIDDEN)
            cx, cy = self._cursor
            if 0 <= cx < self._width and 0 <= cy < self._height:
                frame[cy, cx] = np.int8(C_FORBIDDEN)
        else:
            cx, cy = self._cursor
            if 0 <= cx < self._width and 0 <= cy < self._height:
                cursor_color = C_CURSOR_A if (self._turn % 2 == 0) else C_CURSOR_B
                frame[cy, cx] = np.int8(cursor_color)

        self._board.pixels = frame

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

        self._turn += 1
        self._overlay_cells = set()
        self._advance_pending_transitions()

        action_id = int(self.action.id.value)

        if action_id in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action_id]
            nx = int(self._cursor[0] + dx)
            ny = int(self._cursor[1] + dy)
            if self._cursor_walkable(nx, ny):
                self._cursor = (nx, ny)

        elif action_id == int(GameAction.ACTION5.value):
            if self._lever_cell is not None and self._cursor == self._lever_cell:
                self._toggle_lever()
            else:
                self._apply_paint(self._cursor[0], self._cursor[1], with_beam_from=None)

        elif action_id == int(GameAction.ACTION6.value):
            clicked = self._parse_click()
            if (
                clicked is not None
                and self._is_in_playfield(clicked[0], clicked[1])
                and self._click_valid_path(clicked)
            ):
                self._apply_paint(clicked[0], clicked[1], with_beam_from=(self._cursor[0], self._cursor[1]))

        if self._fail_active:
            self._render()
            self.lose()
            self.complete_action()
            return

        self._move_cleaners()
        self._time_left -= 1

        if not self._fail_active and self._check_win():
            self.next_level()
        elif not self._fail_active and self._time_left <= 0:
            self._trigger_fail({self._cursor})

        self._render()
        if self._fail_active:
            self.lose()
        self.complete_action()
