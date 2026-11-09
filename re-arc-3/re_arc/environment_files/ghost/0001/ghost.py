from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6

MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

VIEW_SIZE = 64
CELL = 5
HUD_HEIGHT = 9

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_SLOT = 2
COLOR_SHADOW = 3
COLOR_WALL = 4
COLOR_BLACK = 5
COLOR_MAGENTA = 6
COLOR_INVALID = 8
COLOR_BLUE = 9
COLOR_GHOST = 10
COLOR_YELLOW = 11
COLOR_GREEN = 14
COLOR_PURPLE = 15

CIRCUIT_COLORS = {"yellow": COLOR_YELLOW, "green": COLOR_GREEN, "magenta": COLOR_MAGENTA}


Command = tuple[Literal["walk"], int] | tuple[Literal["rewind"], tuple[int, int]]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    width: int
    height: int
    max_ghosts: int
    start: tuple[int, int]
    goal: tuple[int, int]
    floor: frozenset[tuple[int, int]]
    plates: dict[str, tuple[tuple[int, int], ...]]
    gates: dict[str, tuple[tuple[int, int], ...]]
    teleporters: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
    budget: int = 30


@dataclass
class GhostState:
    pos: tuple[int, int]
    phase: int
    commands: list[Command]
    walk_length: int


LEVEL_SPECS = (
    LevelSpec(
        name="First Outline",
        width=7,
        height=7,
        max_ghosts=0,
        start=(1, 5),
        goal=(5, 2),
        floor=frozenset({(2, 5), (3, 5), (3, 4), (3, 3), (4, 3), (5, 3)}),
        plates={},
        gates={},
        budget=42,
    ),
    LevelSpec(
        name="Borrowed Foot",
        width=7,
        height=6,
        max_ghosts=1,
        start=(1, 3),
        goal=(5, 3),
        floor=frozenset({(2, 3), (4, 3)}),
        plates={"yellow": ((1, 2),)},
        gates={"yellow": ((3, 3),)},
        budget=36,
    ),
    LevelSpec(
        name="The Rewind Moves Time",
        width=7,
        height=7,
        max_ghosts=2,
        start=(1, 4),
        goal=(5, 5),
        floor=frozenset({(2, 4), (3, 5)}),
        plates={"yellow": ((1, 3),), "green": ((4, 4),)},
        gates={"yellow": ((3, 4),), "green": ((4, 5),)},
        budget=66,
    ),
    LevelSpec(
        name="Purple Jump",
        width=8,
        height=8,
        max_ghosts=1,
        start=(1, 5),
        goal=(5, 6),
        floor=frozenset({(6, 3), (2, 5), (3, 5), (3, 6)}),
        plates={"green": ((6, 4),)},
        gates={"green": ((4, 6),)},
        teleporters=(((1, 4), (6, 2)),),
        budget=54,
    ),
    LevelSpec(
        name="The Wrong Shortcut",
        width=9,
        height=9,
        max_ghosts=2,
        start=(1, 5),
        goal=(7, 7),
        floor=frozenset({(1, 4), (2, 5), (3, 5), (5, 5), (6, 4), (4, 6), (4, 7), (5, 7)}),
        plates={"yellow": ((1, 3),), "green": ((6, 5),)},
        gates={"yellow": ((4, 5),), "green": ((6, 7),)},
        teleporters=(((4, 4), (6, 3)),),
        budget=108,
    ),
    LevelSpec(
        name="Long Echo",
        width=8,
        height=7,
        max_ghosts=2,
        start=(1, 4),
        goal=(4, 5),
        floor=frozenset({(2, 4), (2, 3), (2, 2), (3, 4), (3, 3), (4, 3), (6, 4), (5, 4)}),
        plates={"yellow": ((1, 2),), "green": ((6, 3),)},
        gates={"yellow": ((5, 3),), "green": ((5, 5),)},
        budget=144,
    ),
    LevelSpec(
        name="Three Windows",
        width=11,
        height=10,
        max_ghosts=3,
        start=(1, 6),
        goal=(10, 9),
        floor=frozenset(
            {
                (1, 5),
                (2, 6),
                (3, 6),
                (7, 5),
                (4, 7),
                (4, 8),
                (5, 8),
                (7, 8),
                (7, 7),
                (8, 7),
                (8, 6),
                (6, 9),
                (7, 9),
                (8, 9),
                (8, 8),
            }
        ),
        plates={"yellow": ((1, 4),), "green": ((7, 6),), "magenta": ((9, 6),)},
        gates={"yellow": ((4, 6),), "green": ((6, 8),), "magenta": ((9, 9),)},
        teleporters=(((4, 5), (7, 4)),),
        budget=222,
    ),
)


class Ghost(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        canvas = Sprite(
            np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BG, dtype=np.int8),
            name="canvas",
            layer=0,
            visible=True,
            collidable=False,
            tags=["canvas"],
        )
        levels = [
            Level(sprites=[canvas.clone()], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name)
            for spec in LEVEL_SPECS
        ]
        super().__init__(
            "ghost-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, background=COLOR_BG, letter_box=COLOR_BG),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.canvas = level.get_sprites_by_tag("canvas")[0]
        self.board_left = (VIEW_SIZE - self.spec.width * CELL) // 2
        self.board_top = HUD_HEIGHT + max(0, (52 - self.spec.height * CELL) // 2)
        self.live_pos = self.spec.start
        self.live_commands: list[int] = []
        self.live_positions: list[tuple[int, int]] = [self.spec.start]
        self.ghosts: list[GhostState | None] = [None for _ in range(self.spec.max_ghosts)]
        self.remaining_steps = self.spec.budget
        self.active_gates = self._compute_gates()
        self.invalid_cell: tuple[int, int] | None = None
        self.teleport_flash: tuple[tuple[int, int], tuple[int, int]] | None = None
        self.animation_frames: list[
            tuple[tuple[int, int], list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]] | None]
        ] = []
        self._sync_visuals()

    def step(self) -> None:
        if self.animation_frames:
            live, ghost_positions, flash = self.animation_frames.pop(0)
            self.live_pos = live
            for ghost, pos in zip((g for g in self.ghosts if g is not None), ghost_positions, strict=False):
                ghost.pos = pos
            self.teleport_flash = flash
            self.active_gates = self._compute_gates()
            self._sync_visuals()
            if not self.animation_frames:
                self.teleport_flash = None
                self._sync_visuals()
                self.complete_action()
            return

        if self.action.id == GameAction.RESET:
            self._sync_visuals()
            self.complete_action()
            return

        self.invalid_cell = None
        self.teleport_flash = None
        action_id = int(self.action.id.value)
        if action_id in MOVE_DELTAS:
            self._handle_move(action_id)
        elif action_id == ACTION_SPACE:
            self._handle_space()
        else:
            self._sync_visuals()
            self.complete_action()

    def _handle_move(self, action_id: int) -> None:
        gates = self._compute_gates()
        target = self._move_target(self.live_pos, action_id, gates, check_ghosts=True)
        if target is None:
            dx, dy = MOVE_DELTAS[action_id]
            self.invalid_cell = (self.live_pos[0] + dx, self.live_pos[1] + dy)
            self._sync_visuals()
            self.complete_action()
            return

        self.live_pos, flash = target
        self.live_commands.append(action_id)
        self.live_positions.append(self.live_pos)
        self._advance_ghosts(gates)
        self.remaining_steps -= 1
        self.active_gates = self._compute_gates()
        self.teleport_flash = flash
        self._sync_visuals()
        if self.live_pos == self.spec.goal:
            self.next_level()
            self.complete_action()
        elif self.remaining_steps <= 0:
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _handle_space(self) -> None:
        if not self.live_commands or self.spec.max_ghosts <= 0:
            self._sync_visuals()
            self.complete_action()
            return

        commands = self._compile_ghost_commands()
        rewind_targets = list(reversed(self.live_positions[:-1]))
        for target in rewind_targets:
            gates = self._compute_gates()
            self.live_pos = target
            self._advance_ghosts(gates)
            ghost_positions = [ghost.pos for ghost in self.ghosts if ghost is not None]
            self.animation_frames.append((self.live_pos, ghost_positions, None))

        self.remaining_steps -= 1
        self.live_pos = self.spec.start
        self._store_ghost(GhostState(self.spec.start, 0, commands, len(self.live_commands)))
        self.live_commands = []
        self.live_positions = [self.spec.start]
        self.active_gates = self._compute_gates()
        if self.remaining_steps <= 0:
            self._sync_visuals()
            self.lose()
            self.complete_action()
        elif self.animation_frames:
            self.step()
        else:
            self._sync_visuals()
            self.complete_action()

    def _compile_ghost_commands(self) -> list[Command]:
        commands: list[Command] = [("walk", command) for command in self.live_commands]
        commands.extend(("rewind", pos) for pos in reversed(self.live_positions[:-1]))
        return commands

    def _store_ghost(self, ghost: GhostState) -> None:
        for idx, existing in enumerate(self.ghosts):
            if existing is None:
                self.ghosts[idx] = ghost
                return
        self.ghosts[-1] = ghost

    def _advance_ghosts(self, gates: set[tuple[int, int]]) -> None:
        for ghost in self.ghosts:
            if ghost is None or not ghost.commands:
                continue
            command = ghost.commands[ghost.phase]
            if command[0] == "walk":
                target = self._move_target(ghost.pos, command[1], gates, check_ghosts=False)
                if target is not None:
                    ghost.pos = target[0]
            else:
                ghost.pos = command[1]
            ghost.phase = (ghost.phase + 1) % len(ghost.commands)

    def _move_target(
        self, pos: tuple[int, int], action_id: int, gates: set[tuple[int, int]], *, check_ghosts: bool
    ) -> tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]] | None] | None:
        dx, dy = MOVE_DELTAS[action_id]
        entry = (pos[0] + dx, pos[1] + dy)
        if not self._is_passable(entry, gates):
            return None
        if check_ghosts and entry != self.spec.start and entry in self._ghost_positions():
            return None
        paired = self._paired_teleporter(entry)
        if paired is None:
            return entry, None
        if not self._is_passable(paired, gates):
            return None
        if check_ghosts and paired != self.spec.start and paired in self._ghost_positions():
            return None
        return paired, (entry, paired)

    def _is_passable(self, cell: tuple[int, int], gates: set[tuple[int, int]]) -> bool:
        if not (0 <= cell[0] < self.spec.width and 0 <= cell[1] < self.spec.height):
            return False
        return cell in self._base_floor() or cell in gates

    def _base_floor(self) -> set[tuple[int, int]]:
        cells = set(self.spec.floor)
        cells.add(self.spec.start)
        cells.add(self.spec.goal)
        for positions in self.spec.plates.values():
            cells.update(positions)
        for pair in self.spec.teleporters:
            cells.update(pair)
        return cells

    def _ghost_positions(self) -> set[tuple[int, int]]:
        return {ghost.pos for ghost in self.ghosts if ghost is not None}

    def _compute_gates(self) -> set[tuple[int, int]]:
        occupied = {self.live_pos} | self._ghost_positions()
        open_cells: set[tuple[int, int]] = set()
        for circuit, plate_cells in self.spec.plates.items():
            if any(cell in occupied for cell in plate_cells):
                open_cells.update(self.spec.gates.get(circuit, ()))
        return open_cells

    def _paired_teleporter(self, cell: tuple[int, int]) -> tuple[int, int] | None:
        for a, b in self.spec.teleporters:
            if cell == a:
                return b
            if cell == b:
                return a
        return None

    def _cell_rect(self, cell: tuple[int, int]) -> tuple[int, int, int, int]:
        x = self.board_left + cell[0] * CELL
        y = self.board_top + cell[1] * CELL
        return x, y, x + CELL, y + CELL

    def _sync_visuals(self) -> None:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BG, dtype=np.int8)
        self._draw_hud(frame)
        self._draw_board(frame)
        self._draw_route(frame)
        for ghost in self.ghosts:
            if ghost is not None:
                self._draw_body(frame, ghost.pos, COLOR_GHOST, hollow=ghost.phase >= ghost.walk_length)
        self._draw_body(frame, self.live_pos, COLOR_BLUE, hollow=False)
        if self.invalid_cell is not None:
            self._draw_invalid(frame, self.invalid_cell)
            self.invalid_cell = None
        if self.teleport_flash is not None:
            for cell in self.teleport_flash:
                self._draw_cell_center(frame, cell, COLOR_PURPLE)
        self.canvas.pixels = frame

    def _draw_hud(self, frame: np.ndarray) -> None:
        for idx in range(self.spec.max_ghosts):
            x = 2 + idx * 7
            frame[1:7, x : x + 6] = COLOR_BG
            frame[1, x : x + 6] = COLOR_SLOT
            frame[6, x : x + 6] = COLOR_SLOT
            frame[1:7, x] = COLOR_SLOT
            frame[1:7, x + 5] = COLOR_SLOT
            if idx < len(self.ghosts) and self.ghosts[idx] is not None:
                frame[3:5, x + 2 : x + 4] = COLOR_GHOST
        bar_x, bar_y, bar_w = 24, 3, 37
        frame[bar_y : bar_y + 3, bar_x : bar_x + bar_w] = COLOR_SHADOW
        filled = max(0, min(bar_w, int(bar_w * self.remaining_steps / max(1, self.spec.budget))))
        if filled:
            frame[bar_y : bar_y + 3, bar_x : bar_x + filled] = COLOR_BLUE

    def _draw_board(self, frame: np.ndarray) -> None:
        for y in range(self.spec.height):
            for x in range(self.spec.width):
                cell = (x, y)
                if (
                    cell in self._base_floor()
                    or cell in self.active_gates
                    or any(cell in g for g in self.spec.gates.values())
                ):
                    self._fill_cell(frame, cell, COLOR_FLOOR)
                elif self._touches_floor(cell):
                    self._fill_cell(frame, cell, COLOR_WALL)
        self._draw_start(frame)
        self._draw_goal(frame)
        for circuit, cells in self.spec.plates.items():
            for cell in cells:
                self._draw_plate(
                    frame, cell, CIRCUIT_COLORS[circuit], cell in ({self.live_pos} | self._ghost_positions())
                )
        for circuit, cells in self.spec.gates.items():
            for cell in cells:
                self._draw_gate(frame, cell, CIRCUIT_COLORS[circuit], cell in self.active_gates)
        for pair in self.spec.teleporters:
            for cell in pair:
                self._draw_teleporter(frame, cell)

    def _touches_floor(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        nearby = {(x + dx, y + dy) for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))}
        all_walkable = self._base_floor() | {gate for gates in self.spec.gates.values() for gate in gates}
        return bool(nearby & all_walkable)

    def _fill_cell(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x0, y0, x1, y1 = self._cell_rect(cell)
        frame[y0:y1, x0:x1] = color

    def _draw_start(self, frame: np.ndarray) -> None:
        x0, y0, x1, y1 = self._cell_rect(self.spec.start)
        frame[y0, x0 : x0 + 2] = COLOR_BLUE
        frame[y0 : y0 + 2, x0] = COLOR_BLUE
        frame[y1 - 1, x1 - 2 : x1] = COLOR_BLUE
        frame[y1 - 2 : y1, x1 - 1] = COLOR_BLUE

    def _draw_goal(self, frame: np.ndarray) -> None:
        x0, y0, x1, y1 = self._cell_rect(self.spec.goal)
        frame[y0, x0:x1] = COLOR_BLUE
        frame[y1 - 1, x0:x1] = COLOR_BLUE
        frame[y0:y1, x0] = COLOR_BLUE
        frame[y0:y1, x1 - 1] = COLOR_BLUE
        frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = COLOR_FLOOR

    def _draw_plate(self, frame: np.ndarray, cell: tuple[int, int], color: int, active: bool) -> None:
        x0, y0, x1, y1 = self._cell_rect(cell)
        frame[y0:y1, x0:x1] = color if active else COLOR_FLOOR
        frame[y0, x0:x1] = color
        frame[y1 - 1, x0:x1] = color
        frame[y0:y1, x0] = color
        frame[y0:y1, x1 - 1] = color

    def _draw_gate(self, frame: np.ndarray, cell: tuple[int, int], color: int, open_gate: bool) -> None:
        x0, y0, x1, y1 = self._cell_rect(cell)
        if open_gate:
            frame[y0:y1, x0:x1] = COLOR_FLOOR
            frame[y0:y1, x0] = color
            frame[y0:y1, x1 - 1] = color
        else:
            frame[y0:y1, x0:x1] = color
            frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = COLOR_SHADOW

    def _draw_teleporter(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        x0, y0, x1, y1 = self._cell_rect(cell)
        frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = COLOR_PURPLE
        frame[y0 + 2, x0 + 2] = COLOR_GHOST

    def _draw_route(self, frame: np.ndarray) -> None:
        for cell in self.live_positions[1:]:
            self._draw_cell_center(frame, cell, COLOR_GHOST)

    def _draw_body(self, frame: np.ndarray, cell: tuple[int, int], color: int, *, hollow: bool) -> None:
        x0, y0, _x1, _y1 = self._cell_rect(cell)
        if hollow:
            frame[y0 + 1, x0 + 1 : x0 + 4] = color
            frame[y0 + 3, x0 + 1 : x0 + 4] = color
            frame[y0 + 1 : y0 + 4, x0 + 1] = color
            frame[y0 + 1 : y0 + 4, x0 + 3] = color
        else:
            frame[y0 + 1 : y0 + 4, x0 + 1 : x0 + 4] = color
            frame[y0 + 2, x0 + 2] = COLOR_BLACK if color == COLOR_BLUE else color

    def _draw_cell_center(self, frame: np.ndarray, cell: tuple[int, int], color: int) -> None:
        x0, y0, _x1, _y1 = self._cell_rect(cell)
        frame[y0 + 2, x0 + 2] = color

    def _draw_invalid(self, frame: np.ndarray, cell: tuple[int, int]) -> None:
        if 0 <= cell[0] < self.spec.width and 0 <= cell[1] < self.spec.height:
            x0, y0, x1, y1 = self._cell_rect(cell)
            frame[y0:y1, x0:x1] = COLOR_INVALID
