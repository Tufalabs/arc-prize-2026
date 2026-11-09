from __future__ import annotations

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

BOARD_ORIGIN_X = 12
BOARD_ORIGIN_Y = 12
CELL_STRIDE = 6
CELL_SIZE = 5
BOARD_SIZE = 7

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_FRAME = 2
COLOR_WALL = 4
COLOR_WALL_HIGHLIGHT = 3
COLOR_DISC = 6
COLOR_DISC_HIGHLIGHT = 7
COLOR_PAD_OUTLINE = 9
COLOR_PAD_GLOW = 10
COLOR_ARROW = 11
COLOR_EMITTER = 12
COLOR_WIN = 14

DISC_SPRITE = [[1, 1, 6, 1, 1], [1, 6, 7, 6, 1], [6, 7, 7, 7, 6], [1, 6, 7, 6, 1], [1, 1, 6, 1, 1]]

PAD_SPRITE = [[9, 9, 9, 9, 9], [9, 1, 1, 1, 9], [9, 1, 10, 1, 9], [9, 1, 1, 1, 9], [9, 9, 9, 9, 9]]

WALL_SPRITE = [[4, 4, 4, 4, 4], [4, 3, 3, 3, 4], [4, 3, 4, 3, 4], [4, 3, 3, 3, 4], [4, 4, 4, 4, 4]]

EMITTER_RIGHT = [
    [12, 12, 11, 12, 12],
    [12, 12, 11, 11, 12],
    [11, 11, 11, 11, 11],
    [12, 12, 11, 11, 12],
    [12, 12, 11, 12, 12],
]

EMITTER_LEFT = [
    [12, 12, 11, 12, 12],
    [12, 11, 11, 12, 12],
    [11, 11, 11, 11, 11],
    [12, 11, 11, 12, 12],
    [12, 12, 11, 12, 12],
]

EMITTER_DOWN = [
    [12, 12, 11, 12, 12],
    [12, 11, 11, 11, 12],
    [12, 12, 11, 12, 12],
    [12, 12, 11, 12, 12],
    [12, 12, 11, 12, 12],
]

EMITTER_UP = [
    [12, 12, 11, 12, 12],
    [12, 12, 11, 12, 12],
    [12, 12, 11, 12, 12],
    [12, 11, 11, 11, 12],
    [12, 12, 11, 12, 12],
]


class EmitterSpec:
    def __init__(
        self,
        name: str,
        kind: str,
        index: int,
        direction: str,
        click_x: int,
        click_y: int,
        rect: tuple[int, int, int, int],
    ) -> None:
        self.name = name
        self.kind = kind
        self.index = index
        self.direction = direction
        self.click_x = click_x
        self.click_y = click_y
        self.rect = rect


class LevelSpec:
    def __init__(
        self,
        name: str,
        walls: frozenset[tuple[int, int]],
        discs: tuple[tuple[int, int], ...],
        pads: frozenset[tuple[int, int]],
        emitters: tuple[EmitterSpec, ...],
        budget: int,
    ) -> None:
        self.name = name
        self.walls = walls
        self.discs = discs
        self.pads = pads
        self.emitters = emitters
        self.budget = budget


def _cell_xy(x: int, y: int) -> tuple[int, int]:
    return BOARD_ORIGIN_X + (CELL_STRIDE * x), BOARD_ORIGIN_Y + (CELL_STRIDE * y)


def _make_sprite(pixels: list[list[int]], x: int, y: int, *, name: str, layer: int) -> Sprite:
    return Sprite(pixels=pixels, x=x, y=y, name=name, layer=layer, collidable=False)


def _rect_sprite(width: int, height: int, color: int, *, name: str, x: int, y: int, layer: int) -> Sprite:
    return Sprite(pixels=[[color for _ in range(width)] for _ in range(height)], x=x, y=y, name=name, layer=layer)


def _outline_sprite(width: int, height: int, color: int, *, name: str, x: int, y: int, layer: int) -> Sprite:
    pixels = [[-1 for _ in range(width)] for _ in range(height)]
    for px in range(width):
        pixels[0][px] = color
        pixels[height - 1][px] = color
    for py in range(height):
        pixels[py][0] = color
        pixels[py][width - 1] = color
    return Sprite(pixels=pixels, x=x, y=y, name=name, layer=layer, collidable=False)


def _level_specs() -> list[LevelSpec]:
    level1_emitters = (
        EmitterSpec("R5", "row", 5, "RIGHT", 7, 14 + (6 * 5), (5, 12 + (6 * 5), 9, 16 + (6 * 5))),
        EmitterSpec("U4", "col", 4, "UP", 14 + (6 * 4), 57, (12 + (6 * 4), 55, 16 + (6 * 4), 59)),
    )
    level2_emitters = (
        EmitterSpec("R5", "row", 5, "RIGHT", 7, 14 + (6 * 5), (5, 12 + (6 * 5), 9, 16 + (6 * 5))),
        EmitterSpec("U4", "col", 4, "UP", 14 + (6 * 4), 57, (12 + (6 * 4), 55, 16 + (6 * 4), 59)),
        EmitterSpec("U5", "col", 5, "UP", 14 + (6 * 5), 57, (12 + (6 * 5), 55, 16 + (6 * 5), 59)),
    )
    level3_emitters = (
        EmitterSpec("U2", "col", 2, "UP", 14 + (6 * 2), 57, (12 + (6 * 2), 55, 16 + (6 * 2), 59)),
        EmitterSpec("U3", "col", 3, "UP", 14 + (6 * 3), 57, (12 + (6 * 3), 55, 16 + (6 * 3), 59)),
        EmitterSpec("U4", "col", 4, "UP", 14 + (6 * 4), 57, (12 + (6 * 4), 55, 16 + (6 * 4), 59)),
        EmitterSpec("L1", "row", 1, "LEFT", 57, 14 + (6 * 1), (55, 12 + (6 * 1), 59, 16 + (6 * 1))),
        EmitterSpec("R2", "row", 2, "RIGHT", 7, 14 + (6 * 2), (5, 12 + (6 * 2), 9, 16 + (6 * 2))),
    )
    level3_open = {(2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1), (3, 2), (4, 2), (3, 3), (3, 4), (3, 5)}
    all_cells = {(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)}
    return [
        LevelSpec(
            name="Level 1",
            walls=frozenset(),
            discs=((1, 5),),
            pads=frozenset({(4, 2)}),
            emitters=level1_emitters,
            budget=18,
        ),
        LevelSpec(
            name="Level 2",
            walls=frozenset(),
            discs=((1, 5), (3, 5)),
            pads=frozenset({(4, 1), (5, 1)}),
            emitters=level2_emitters,
            budget=33,
        ),
        LevelSpec(
            name="Level 3",
            walls=frozenset(all_cells - level3_open),
            discs=((3, 3), (3, 4), (3, 5)),
            pads=frozenset({(2, 0), (3, 0), (4, 0)}),
            emitters=level3_emitters,
            budget=27,
        ),
    ]


class RowPulsePacking(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._specs = _level_specs()
        self._remaining_budget = 0
        self._discs: set[tuple[int, int]] = set()
        levels = [Level(grid_size=(64, 64), name=spec.name) for spec in self._specs]
        super().__init__(
            game_id="row_pulse_packing-0001",
            levels=levels,
            camera=Camera(width=64, height=64, background=COLOR_BG),
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        del level
        spec = self._specs[self.level_index]
        self._remaining_budget = spec.budget
        self._discs = set(spec.discs)
        self._rebuild_level()

    def _rebuild_level(self) -> None:
        spec = self._specs[self.level_index]
        sprites: list[Sprite] = []
        sprites.append(_rect_sprite(64, 64, COLOR_BG, name="bg", x=0, y=0, layer=-10))
        sprites.append(
            _outline_sprite(43, 43, COLOR_FRAME, name="frame", x=BOARD_ORIGIN_X - 1, y=BOARD_ORIGIN_Y - 1, layer=-4)
        )

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                sx, sy = _cell_xy(x, y)
                if (x, y) in spec.walls:
                    sprites.append(_make_sprite(WALL_SPRITE, sx, sy, name=f"wall_{x}_{y}", layer=-2))
                else:
                    sprites.append(
                        _rect_sprite(CELL_SIZE, CELL_SIZE, COLOR_FLOOR, name=f"floor_{x}_{y}", x=sx, y=sy, layer=-3)
                    )

        for x, y in sorted(spec.pads):
            sx, sy = _cell_xy(x, y)
            sprites.append(_make_sprite(PAD_SPRITE, sx, sy, name=f"pad_{x}_{y}", layer=-1))

        for x, y in sorted(self._discs):
            sx, sy = _cell_xy(x, y)
            sprites.append(_make_sprite(DISC_SPRITE, sx, sy, name=f"disc_{x}_{y}", layer=1))

        for emitter in spec.emitters:
            left, top, _right, _bottom = emitter.rect
            pixels = self._emitter_pixels(emitter.direction)
            sprites.append(_make_sprite(pixels, left, top, name=f"emitter_{emitter.name}", layer=2))

        self._add_budget_pips(sprites, spec.budget)
        if self._state.name == "WIN" and self.level_index == len(self._specs) - 1:
            self._add_win_overlay(sprites)

        self.current_level.remove_all_sprites()
        for sprite in sprites:
            self.current_level.add_sprite(sprite)

    def _add_budget_pips(self, sprites: list[Sprite], total_budget: int) -> None:
        for idx in range(total_budget):
            row = idx // 14
            col = idx % 14
            x = 12 + (3 * col)
            y = 1 + (3 * row)
            remaining = idx < self._remaining_budget
            color = COLOR_ARROW if remaining else COLOR_WALL_HIGHLIGHT
            sprites.append(_rect_sprite(2, 2, color, name=f"pip_{idx}", x=x, y=y, layer=3))

    def _add_win_overlay(self, sprites: list[Sprite]) -> None:
        border = _outline_sprite(64, 64, COLOR_WIN, name="win_border", x=0, y=0, layer=4)
        sprites.append(border)

    def _emitter_pixels(self, direction: str) -> list[list[int]]:
        if direction == "RIGHT":
            return EMITTER_RIGHT
        if direction == "LEFT":
            return EMITTER_LEFT
        if direction == "DOWN":
            return EMITTER_DOWN
        return EMITTER_UP

    def _find_clicked_emitter(self, click_x: int, click_y: int) -> EmitterSpec | None:
        spec = self._specs[self.level_index]
        for emitter in spec.emitters:
            left, top, right, bottom = emitter.rect
            if left <= click_x <= right and top <= click_y <= bottom:
                return emitter
        return None

    def _is_solved(self) -> bool:
        spec = self._specs[self.level_index]
        return all(pad in self._discs for pad in spec.pads)

    def _next_position(self, position: tuple[int, int], direction: str) -> tuple[int, int]:
        x, y = position
        if direction == "RIGHT":
            return x + 1, y
        if direction == "LEFT":
            return x - 1, y
        if direction == "DOWN":
            return x, y + 1
        return x, y - 1

    def _scan_positions(self, emitter: EmitterSpec) -> list[tuple[int, int]]:
        if emitter.kind == "row":
            y = emitter.index
            xs = range(5, -1, -1) if emitter.direction == "RIGHT" else range(1, 7)
            return [(x, y) for x in xs]
        x = emitter.index
        ys = range(5, -1, -1) if emitter.direction == "DOWN" else range(1, 7)
        return [(x, y) for y in ys]

    def _apply_pulse(self, emitter: EmitterSpec) -> None:
        spec = self._specs[self.level_index]
        occupied = set(self._discs)
        for position in self._scan_positions(emitter):
            if position not in occupied:
                continue
            nx, ny = self._next_position(position, emitter.direction)
            next_position = (nx, ny)
            if not (0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE):
                continue
            if next_position in spec.walls or next_position in occupied:
                continue
            occupied.remove(position)
            occupied.add(next_position)
        self._discs = occupied

    def step(self) -> None:
        action_id = getattr(self.action, "id", GameAction.RESET)
        if action_id != GameAction.ACTION6:
            self.complete_action()
            return

        raw_x = self.action.data.get("x")
        raw_y = self.action.data.get("y")
        if raw_x is None or raw_y is None:
            self.complete_action()
            return

        click_x = int(raw_x)
        click_y = int(raw_y)
        emitter = self._find_clicked_emitter(click_x, click_y)
        if emitter is None:
            self.complete_action()
            return

        self._remaining_budget -= 1
        self._apply_pulse(emitter)

        if self._is_solved():
            self.next_level()
            if self.level_index == len(self._specs) - 1 and self._state.name == "WIN":
                self._rebuild_level()
            self.complete_action()
            return

        if self._remaining_budget <= 0:
            self.lose()
            self.complete_action()
            return

        self._rebuild_level()
        self.complete_action()
