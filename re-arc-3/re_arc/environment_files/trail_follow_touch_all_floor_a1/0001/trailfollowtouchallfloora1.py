from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "trail_follow_touch_all_floor_a1-0001"

COLORS = {
    "void": 0,
    "wall": 1,
    "normal_unpainted": 2,
    "solid_trail": 3,
    "player_a": 4,
    "player_b": 5,
    "soft_painted": 6,
    "soft_unpainted": 7,
    "switch_off": 8,
    "switch_on": 9,
    "gate_closed": 10,
    "gate_open": 11,
    "teleport": 12,
    "enemy": 13,
    "timebar": 14,
    "pulse": 15,
}

MOVE_DELTAS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "time_limit": 80,
        "layout": [
            "################",
            "#██████████████#",
            "##@...##########",
            "##....##########",
            "##....##########",
            "##....##########",
            "################",
            "################",
            "################",
            "################",
            "################",
        ],
    },
    {
        "name": "Level 2",
        "time_limit": 110,
        "layout": [
            "################",
            "#██████████████#",
            "##@....#########",
            "##..#..#########",
            "##.....#########",
            "##.....#########",
            "################",
            "################",
            "################",
            "################",
            "################",
        ],
    },
    {
        "name": "Level 3",
        "time_limit": 170,
        "layout": [
            "####################",
            "#██████████████████#",
            "########..##########",
            "########~.##########",
            "########~###########",
            "########~###########",
            "###..###~~~###..####",
            "###.~~~~~@~~~~~.####",
            "########~~~#########",
            "####################",
            "####################",
            "####################",
            "####################",
        ],
    },
    {
        "name": "Level 4",
        "time_limit": 210,
        "layout": [
            "######################",
            "#████████████████████#",
            "######################",
            "######################",
            "######################",
            "###########~~~########",
            "###########~@~########",
            "####**~~~~~~~~########",
            "####**######.#########",
            "############.#########",
            "############||########",
            "############||########",
            "##########......######",
            "##########......######",
            "######################",
        ],
    },
    {
        "name": "Level 5",
        "time_limit": 260,
        "layout": [
            "##########################",
            "#████████████████████████#",
            "##########################",
            "##########################",
            "##**~~~###################",
            "##**~~~###################",
            "##~~@~~~||~~~~&&~~~~######",
            "##~~~~~~||~~~~&&~~~~######",
            "##~~~~~###########..######",
            "##################..######",
            "##########################",
            "##########################",
            "##########################",
            "##########################",
            "##########################",
            "##########################",
        ],
    },
    {
        "name": "Level 6",
        "time_limit": 320,
        "layout": [
            "##############################",
            "#████████████████████████████#",
            "##############################",
            "##############################",
            "##############################",
            "###**~~~~~####################",
            "###**~~~~~######~~~###########",
            "###~~~@~~~||OO##OO~~~&&~~~####",
            "###~~~~~~~||OO##OO~~~&&~~~####",
            "###~~~~.~~######~~~~~~~~~.####",
            "###....##################..###",
            "##############################",
            "##############################",
            "##############################",
            "##############################",
            "##############################",
            "##############################",
            "##############################",
        ],
    },
]


@dataclass(frozen=True)
class LevelModel:
    width: int
    height: int
    time_limit: int
    walls: frozenset[tuple[int, int]]
    timebar_cells: tuple[tuple[int, int], ...]
    timebar_row: int
    normal_cells: frozenset[tuple[int, int]]
    soft_cells: frozenset[tuple[int, int]]
    switch_cells: frozenset[tuple[int, int]]
    gate_cells: frozenset[tuple[int, int]]
    teleport_cells: frozenset[tuple[int, int]]
    teleport_map: dict[tuple[int, int], tuple[int, int]]
    player_spawn: tuple[int, int]
    enemy_spawn: tuple[int, int] | None

    @property
    def required_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            set(self.normal_cells)
            | set(self.soft_cells)
            | set(self.switch_cells)
            | set(self.gate_cells)
            | set(self.teleport_cells)
        )


def _parse_layout(layout: list[str], time_limit: int) -> LevelModel:
    if not layout:
        raise ValueError("layout cannot be empty")

    width = len(layout[0])
    height = len(layout)
    for row in layout:
        if len(row) != width:
            raise ValueError("layout rows must have consistent width")

    walls: set[tuple[int, int]] = set()
    timebar_cells: list[tuple[int, int]] = []
    normal_cells: set[tuple[int, int]] = set()
    soft_cells: set[tuple[int, int]] = set()
    switch_cells: set[tuple[int, int]] = set()
    gate_cells: set[tuple[int, int]] = set()
    teleport_cells: set[tuple[int, int]] = set()
    enemy_cells: set[tuple[int, int]] = set()
    player_spawn: tuple[int, int] | None = None

    for y, row in enumerate(layout):
        for x, ch in enumerate(row):
            cell = (x, y)
            if ch == "#":
                walls.add(cell)
            elif ch == "█":
                timebar_cells.append(cell)
            elif ch == ".":
                normal_cells.add(cell)
            elif ch == "~":
                soft_cells.add(cell)
            elif ch == "@":
                player_spawn = cell
                soft_cells.add(cell)
            elif ch == "*":
                switch_cells.add(cell)
            elif ch == "|":
                gate_cells.add(cell)
            elif ch == "O":
                teleport_cells.add(cell)
            elif ch == "&":
                enemy_cells.add(cell)
                soft_cells.add(cell)
            else:
                raise ValueError(f"unknown layout tile {ch!r}")

    if player_spawn is None:
        raise ValueError("layout must include player spawn '@'")

    if not timebar_cells:
        raise ValueError("layout must include a timebar row")

    timebar_rows = {y for _, y in timebar_cells}
    if len(timebar_rows) != 1:
        raise ValueError("timebar must occupy exactly one row")
    timebar_row = next(iter(timebar_rows))

    teleport_map: dict[tuple[int, int], tuple[int, int]] = {}
    if teleport_cells:
        if len(teleport_cells) != 8:
            raise ValueError("teleport levels must define exactly two 2x2 pads")
        pads = _split_teleport_pads(teleport_cells)
        if len(pads) != 2:
            raise ValueError("teleport cells must form exactly two pads")
        pad_a = sorted(pads[0])
        pad_b = sorted(pads[1])
        top_left_a = (min(x for x, _ in pad_a), min(y for _, y in pad_a))
        top_left_b = (min(x for x, _ in pad_b), min(y for _, y in pad_b))
        for src in pad_a:
            offset = (src[0] - top_left_a[0], src[1] - top_left_a[1])
            teleport_map[src] = (top_left_b[0] + offset[0], top_left_b[1] + offset[1])
        for src in pad_b:
            offset = (src[0] - top_left_b[0], src[1] - top_left_b[1])
            teleport_map[src] = (top_left_a[0] + offset[0], top_left_a[1] + offset[1])

    enemy_spawn: tuple[int, int] | None = None
    if enemy_cells:
        xs = sorted(x for x, _ in enemy_cells)
        ys = sorted(y for _, y in enemy_cells)
        if len(enemy_cells) != 4 or max(xs) - min(xs) != 1 or max(ys) - min(ys) != 1:
            raise ValueError("enemy must be represented as a 2x2 block")
        enemy_spawn = (min(xs), min(ys))

    return LevelModel(
        width=width,
        height=height,
        time_limit=int(time_limit),
        walls=frozenset(walls),
        timebar_cells=tuple(sorted(timebar_cells)),
        timebar_row=int(timebar_row),
        normal_cells=frozenset(normal_cells),
        soft_cells=frozenset(soft_cells),
        switch_cells=frozenset(switch_cells),
        gate_cells=frozenset(gate_cells),
        teleport_cells=frozenset(teleport_cells),
        teleport_map=dict(teleport_map),
        player_spawn=player_spawn,
        enemy_spawn=enemy_spawn,
    )


def _split_teleport_pads(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    parts: list[set[tuple[int, int]]] = []
    while remaining:
        seed = next(iter(remaining))
        stack = [seed]
        comp: set[tuple[int, int]] = set()
        while stack:
            x, y = stack.pop()
            if (x, y) in comp:
                continue
            comp.add((x, y))
            remaining.discard((x, y))
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nb = (x + dx, y + dy)
                if nb in remaining:
                    stack.append(nb)
        parts.append(comp)
    for comp in parts:
        xs = [x for x, _ in comp]
        ys = [y for _, y in comp]
        if len(comp) != 4 or max(xs) - min(xs) != 1 or max(ys) - min(ys) != 1:
            raise ValueError("teleport pads must each be 2x2")
    return parts


def _build_level(spec: dict) -> Level:
    model = _parse_layout(list(spec["layout"]), int(spec["time_limit"]))
    pixels = np.zeros((model.height, model.width), dtype=np.int8)
    board = Sprite(pixels=pixels, name="board", x=0, y=0, layer=0, tags=["board"], collidable=False)
    return Level(
        name=str(spec["name"]),
        grid_size=(model.width, model.height),
        sprites=[board],
        data={"layout": list(spec["layout"]), "time_limit": int(spec["time_limit"])},
    )


class TrailFollowTouchAllFloorA1(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._models = [_parse_layout(list(spec["layout"]), int(spec["time_limit"])) for spec in LEVEL_SPECS]
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        first = self._models[0]
        self._route_score = 0
        self._time_remaining = first.time_limit
        self._anim_phase = False
        self._model = first
        self._player = first.player_spawn
        self._solid: set[tuple[int, int]] = set()
        self._visited: set[tuple[int, int]] = set()
        self._gate_open = False
        self._switch_on = False
        self._enemy_top_left: tuple[int, int] | None = None
        self._enemy_dir = 1
        self._board: Sprite | None = None

        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=Camera(width=first.width, height=first.height, background=COLORS["void"]),
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        model = _parse_layout(list(level.get_data("layout") or []), int(level.get_data("time_limit") or 1))
        self._model = model
        self._time_remaining = model.time_limit
        self._anim_phase = False
        self._board = self.current_level.get_sprites_by_name("board")[0]
        self._reset_attempt_state()
        self._render()

    def _reset_attempt_state(self) -> None:
        self._player = self._model.player_spawn
        self._solid = set()
        self._visited = set()
        if self._player in self._model.required_cells:
            self._visited.add(self._player)
        self._route_score = len(self._visited)
        self._gate_open = False
        self._switch_on = False
        self._enemy_top_left = self._model.enemy_spawn
        self._enemy_dir = 1

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._model.width and 0 <= y < self._model.height

    def _is_passable(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        if not self._in_bounds(x, y):
            return False
        if pos in self._model.walls:
            return False
        if y == self._model.timebar_row:
            return False
        if pos in self._solid:
            return False
        return not (pos in self._model.gate_cells and not self._gate_open)

    def _enemy_cells(self, top_left: tuple[int, int] | None = None) -> set[tuple[int, int]]:
        if top_left is None:
            top_left = self._enemy_top_left
        if top_left is None:
            return set()
        x, y = top_left
        return {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}

    def _enemy_can_move(self, dx: int) -> bool:
        if self._enemy_top_left is None:
            return False
        nx = self._enemy_top_left[0] + dx
        ny = self._enemy_top_left[1]
        for cell in ((nx, ny), (nx + 1, ny), (nx, ny + 1), (nx + 1, ny + 1)):
            if not self._is_passable(cell):
                return False
        return True

    def _move_enemy(self) -> None:
        if self._enemy_top_left is None:
            return
        if self._enemy_can_move(self._enemy_dir):
            self._enemy_top_left = (self._enemy_top_left[0] + self._enemy_dir, self._enemy_top_left[1])
            return
        self._enemy_dir *= -1
        if self._enemy_can_move(self._enemy_dir):
            self._enemy_top_left = (self._enemy_top_left[0] + self._enemy_dir, self._enemy_top_left[1])

    def _mark_visited(self, pos: tuple[int, int]) -> None:
        if pos not in self._model.required_cells:
            return
        if pos in self._model.gate_cells and not self._gate_open:
            return
        self._visited.add(pos)
        self._route_score = len(self._visited)

    def _is_click_restart(self) -> bool:
        if self.action.id != GameAction.ACTION6:
            return False
        data = self.action.data if isinstance(self.action.data, dict) else {}
        try:
            x = int(data.get("x", -1))
            y = int(data.get("y", -1))
        except (TypeError, ValueError):
            return False
        return 0 <= x < self._model.width and y == self._model.timebar_row

    def _timebar_lit(self) -> int:
        total = max(1, self._model.time_limit)
        segment_count = len(self._model.timebar_cells)
        if self._time_remaining <= 0:
            return 0
        return math.ceil((self._time_remaining * segment_count) / total)

    def _all_required_painted(self) -> bool:
        return len(self._visited) >= len(self._model.required_cells)

    def _has_legal_move(self) -> bool:
        px, py = self._player
        for dx, dy in MOVE_DELTAS.values():
            if self._is_passable((px + dx, py + dy)):
                return True
        return False

    def _render(self) -> None:
        if self._board is None:
            return

        board = np.full((self._model.height, self._model.width), COLORS["void"], dtype=np.int8)

        for x, y in self._model.walls:
            board[y, x] = COLORS["wall"]

        lit = self._timebar_lit()
        low_time = self._time_remaining <= max(1, self._model.time_limit // 6)
        time_color = COLORS["pulse"] if low_time and self._anim_phase else COLORS["timebar"]
        for idx, (x, y) in enumerate(self._model.timebar_cells):
            board[y, x] = time_color if idx < lit else COLORS["void"]

        for cell in self._model.normal_cells:
            board[cell[1], cell[0]] = COLORS["solid_trail"] if cell in self._solid else COLORS["normal_unpainted"]

        for cell in self._model.soft_cells:
            board[cell[1], cell[0]] = COLORS["soft_painted"] if cell in self._visited else COLORS["soft_unpainted"]

        for cell in self._model.gate_cells:
            board[cell[1], cell[0]] = COLORS["gate_open"] if self._gate_open else COLORS["gate_closed"]

        switch_on_color = COLORS["pulse"] if self._switch_on and self._anim_phase else COLORS["switch_on"]
        for cell in self._model.switch_cells:
            board[cell[1], cell[0]] = switch_on_color if self._switch_on else COLORS["switch_off"]

        teleport_color = COLORS["pulse"] if self._anim_phase else COLORS["teleport"]
        for cell in self._model.teleport_cells:
            board[cell[1], cell[0]] = teleport_color

        enemy_color = COLORS["pulse"] if self._anim_phase else COLORS["enemy"]
        for ex, ey in self._enemy_cells():
            if self._in_bounds(ex, ey):
                board[ey, ex] = enemy_color

        px, py = self._player
        board[py, px] = COLORS["player_b"] if self._anim_phase else COLORS["player_a"]

        self._board.pixels = board

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

        collision = False
        moved = False
        left_cell: tuple[int, int] | None = None

        if self._is_click_restart():
            self._reset_attempt_state()
        else:
            move = MOVE_DELTAS.get(self.action.id)
            if move is not None:
                dx, dy = move
                target = (self._player[0] + dx, self._player[1] + dy)
                if self._is_passable(target):
                    left_cell = self._player
                    self._player = target
                    moved = True
                    if self._player in self._enemy_cells():
                        collision = True

                    self._mark_visited(self._player)
                    if self._player in self._model.switch_cells:
                        self._switch_on = not self._switch_on
                        self._gate_open = self._switch_on

                    if self._player in self._model.teleport_map:
                        self._player = self._model.teleport_map[self._player]
                        self._mark_visited(self._player)
                        if self._player in self._enemy_cells():
                            collision = True

            if moved and left_cell is not None and left_cell in self._model.normal_cells:
                self._solid.add(left_cell)

        self._move_enemy()
        if self._player in self._enemy_cells():
            collision = True

        self._time_remaining -= 1

        if self._all_required_painted():
            self.next_level()
        else:
            unpainted_remaining = len(self._model.required_cells - self._visited)
            no_moves = unpainted_remaining > 0 and not self._has_legal_move()
            if collision or self._time_remaining <= 0 or no_moves:
                self.lose()

        self._anim_phase = not self._anim_phase
        self._render()
        self.complete_action()
