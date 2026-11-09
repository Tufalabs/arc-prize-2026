from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite
from arcengine.enums import GameAction

VIEW_SIZE = 64
BOARD_LEFT = 4
BOARD_TOP = 4
CELL_SIZE = 8

COLOR_CANVAS = 4
COLOR_BORE = 5
COLOR_COLLAR = 3
COLOR_INACTIVE = 2
COLOR_LIVE_WALL = 9
COLOR_FLUID = 10
COLOR_TERMINAL = 1
COLOR_FILLED = 14
COLOR_ROTATE = 11
COLOR_BLOCKED = 8

DIRS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
PORT_STATES = {
    "I": (("N", "S"), ("E", "W")),
    "L": (("N", "E"), ("E", "S"), ("S", "W"), ("W", "N")),
    "T": (("N", "E", "W"), ("N", "E", "S"), ("E", "S", "W"), ("N", "S", "W")),
}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    pipes: dict[str, tuple[str, int]]
    source_port: str
    terminals: dict[str, str]
    baseline: int


LEVEL_SPECS = (
    LevelSpec(
        "Quarter Turn",
        ("..S....", "..An...", ".......", ".......", ".......", ".......", "......."),
        {"A": ("L", 3)},
        "S",
        {"n": "W"},
        1,
    ),
    LevelSpec(
        "Live Edge",
        (".....nG", "......F", "......E", "..SABCD", ".......", ".......", "......."),
        {"A": ("T", 3), "B": ("T", 3), "C": ("I", 1), "D": ("L", 2), "E": ("T", 2), "F": ("T", 1), "G": ("T", 1)},
        "E",
        {"n": "E"},
        5,
    ),
    LevelSpec(
        "Prepare, Then Restore",
        (".nS....", "ABC....", "DEF....", "b......", ".......", ".......", "......."),
        {"A": ("T", 2), "B": ("T", 1), "C": ("T", 0), "D": ("L", 3), "E": ("T", 3), "F": ("T", 2)},
        "S",
        {"n": "S", "b": "N"},
        7,
    ),
    LevelSpec(
        "Borrowed Loop",
        ("..n....", ".ABC...", "wDEFe..", ".GHI...", "..Sb...", ".......", "......."),
        {
            "A": ("T", 0),
            "B": ("L", 1),
            "C": ("T", 0),
            "D": ("L", 3),
            "E": ("T", 2),
            "F": ("T", 0),
            "G": ("T", 3),
            "H": ("T", 1),
            "I": ("T", 1),
        },
        "N",
        {"n": "S", "w": "E", "e": "W", "b": "N"},
        28,
    ),
    LevelSpec(
        "Lower Handoff",
        ("..n....", ".ABC...", "wDEFe..", ".GHI...", "..JK...", "..Sb...", "......."),
        {
            "A": ("T", 3),
            "B": ("L", 1),
            "C": ("T", 0),
            "D": ("L", 0),
            "E": ("L", 1),
            "F": ("T", 0),
            "G": ("T", 2),
            "H": ("L", 0),
            "I": ("T", 1),
            "J": ("T", 1),
            "K": ("T", 0),
        },
        "N",
        {"n": "S", "w": "E", "e": "W", "b": "N"},
        40,
    ),
    LevelSpec(
        "Side-Spur Staging",
        ("..n....", ".ABC...", "wDEFe..", ".GHI...", "..JKLt.", "..S....", "......."),
        {
            "A": ("T", 3),
            "B": ("L", 2),
            "C": ("T", 1),
            "D": ("T", 2),
            "E": ("L", 0),
            "F": ("L", 1),
            "G": ("T", 2),
            "H": ("T", 3),
            "I": ("L", 1),
            "J": ("T", 1),
            "K": ("T", 2),
            "L": ("T", 3),
        },
        "N",
        {"n": "S", "w": "E", "e": "W", "t": "W"},
        42,
    ),
    LevelSpec(
        "Five-Way Relay",
        ("..n....", ".ABC...", "wDEFe..", ".GHI...", ".JKL...", ".lSr...", "......."),
        {
            "A": ("T", 3),
            "B": ("L", 1),
            "C": ("T", 0),
            "D": ("T", 2),
            "E": ("L", 1),
            "F": ("L", 3),
            "G": ("L", 1),
            "H": ("T", 1),
            "I": ("T", 0),
            "J": ("T", 2),
            "K": ("T", 2),
            "L": ("T", 0),
        },
        "N",
        {"n": "S", "w": "E", "e": "W", "l": "N", "r": "N"},
        51,
    ),
)


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(
        np.full((VIEW_SIZE, VIEW_SIZE), COLOR_CANVAS, dtype=np.int8),
        name="board",
        x=0,
        y=0,
        collidable=False,
        tags=["board"],
    )
    return Level([board], (VIEW_SIZE, VIEW_SIZE), {"spec": spec}, spec.name)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


class PipeRotors(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "pipe_rotors-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_CANVAS, COLOR_CANVAS),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )

    def on_set_level(self, _level: Level) -> None:
        self.spec: LevelSpec = self.current_level.get_data("spec")
        self.board = self.current_level.get_sprites_by_tag("board")[0]
        self.pipe_cells: dict[str, tuple[int, int]] = {}
        self.pipe_at: dict[tuple[int, int], str] = {}
        self.source_cell = (-1, -1)
        self.terminal_cells: dict[str, tuple[int, int]] = {}
        for row, line in enumerate(self.spec.rows):
            if len(line) != 7:
                raise ValueError(f"{self.spec.name}: every ASCII row must have seven cells")
            for col, symbol in enumerate(line):
                if symbol in self.spec.pipes:
                    self.pipe_cells[symbol] = (row, col)
                    self.pipe_at[(row, col)] = symbol
                elif symbol == "S":
                    self.source_cell = (row, col)
                elif symbol in self.spec.terminals:
                    self.terminal_cells[symbol] = (row, col)
        self.orientations = {pipe_id: data[1] for pipe_id, data in self.spec.pipes.items()}
        self.step_budget = self.spec.baseline * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[dict[str, int]] = []
        self.animation_frames: list[tuple[str, dict[str, object]]] = []
        self._pending_win = False
        self._pending_failure = False
        self.live_pipes, self.filled_terminals = self._connectivity(self.orientations)
        self._render("settled", {})

    def _ports_for_pipe(self, pipe_id: str, orientations: dict[str, int]) -> tuple[str, ...]:
        kind = self.spec.pipes[pipe_id][0]
        states = PORT_STATES[kind]
        return states[orientations[pipe_id] % len(states)]

    def _ports_at(self, cell: tuple[int, int], orientations: dict[str, int]) -> tuple[str, ...]:
        if cell == self.source_cell:
            return (self.spec.source_port,)
        pipe_id = self.pipe_at.get(cell)
        if pipe_id is not None:
            return self._ports_for_pipe(pipe_id, orientations)
        for terminal_id, terminal_cell in self.terminal_cells.items():
            if cell == terminal_cell:
                return (self.spec.terminals[terminal_id],)
        return ()

    def _connectivity(self, orientations: dict[str, int]) -> tuple[set[str], set[str]]:
        live: set[str] = set()
        visited = {self.source_cell}
        frontier = [self.source_cell]
        while frontier:
            cell = frontier.pop()
            for direction in self._ports_at(cell, orientations):
                dr, dc = DIRS[direction]
                neighbor = (cell[0] + dr, cell[1] + dc)
                if OPPOSITE[direction] not in self._ports_at(neighbor, orientations):
                    continue
                pipe_id = self.pipe_at.get(neighbor)
                if pipe_id is None:
                    continue
                live.add(pipe_id)
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)

        filled: set[str] = set()
        supplied_cells = {self.source_cell, *(self.pipe_cells[pipe_id] for pipe_id in live)}
        for terminal_id, cell in self.terminal_cells.items():
            direction = self.spec.terminals[terminal_id]
            dr, dc = DIRS[direction]
            neighbor = (cell[0] + dr, cell[1] + dc)
            if neighbor in supplied_cells and OPPOSITE[direction] in self._ports_at(neighbor, orientations):
                filled.add(terminal_id)
        return live, filled

    def _spend_action(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        self._pending_failure = self.remaining_steps == 0

    def _click_cell(self) -> tuple[int, int] | None:
        payload = self.action.data or {}
        x = int(payload.get("x", 0))
        y = int(payload.get("y", 0))
        if not (BOARD_LEFT <= x < 60 and BOARD_TOP <= y < 60):
            return None
        return ((y - BOARD_TOP) // CELL_SIZE, (x - BOARD_LEFT) // CELL_SIZE)

    def _finish_action(self) -> None:
        self.complete_action()
        if self._pending_failure:
            self.lose()
        elif self._pending_win:
            self.next_level()

    def _queue_rotation(self, pipe_id: str) -> None:
        old_orientations = self.orientations.copy()
        old_live = set(self.live_pipes)
        self.undo_history.append(old_orientations)
        kind = self.spec.pipes[pipe_id][0]
        self.orientations[pipe_id] = (self.orientations[pipe_id] + 1) % len(PORT_STATES[kind])
        self.live_pipes, self.filled_terminals = self._connectivity(self.orientations)
        newly_live = self.live_pipes - old_live
        self._pending_win = len(self.filled_terminals) == len(self.terminal_cells)
        payload: dict[str, object] = {
            "pipe": pipe_id,
            "old_orientations": old_orientations,
            "old_live": old_live,
            "newly_live": newly_live,
        }
        modes = ["quarter_ring", "port_overlay", "live_temp", "settled_pulse"]
        if self._pending_win:
            modes.extend(("win_low", "win_high"))
        self.animation_frames = [(mode, payload) for mode in modes]

    def _play_animation_frame(self) -> None:
        mode, payload = self.animation_frames.pop(0)
        self._render(mode, payload)
        if not self.animation_frames:
            self._finish_action()

    def _draw_arm(self, module: np.ndarray, direction: str, outer: int, bore: int) -> None:
        if direction == "N":
            module[0:4, 2:6] = outer
            module[0:4, 3:5] = bore
        elif direction == "E":
            module[2:6, 4:8] = outer
            module[3:5, 4:8] = bore
        elif direction == "S":
            module[4:8, 2:6] = outer
            module[4:8, 3:5] = bore
        else:
            module[2:6, 0:4] = outer
            module[3:5, 0:4] = bore

    def _draw_pipe(
        self,
        frame: np.ndarray,
        pipe_id: str,
        orientations: dict[str, int],
        live: set[str],
        bore_override: int | None = None,
    ) -> None:
        row, col = self.pipe_cells[pipe_id]
        y0, x0 = BOARD_TOP + row * CELL_SIZE, BOARD_LEFT + col * CELL_SIZE
        module = frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE]
        module[1:7, 1:7] = COLOR_COLLAR
        module[1, 1] = module[1, 6] = module[6, 1] = module[6, 6] = COLOR_CANVAS
        outer = COLOR_LIVE_WALL if pipe_id in live else COLOR_INACTIVE
        bore = COLOR_FLUID if pipe_id in live else COLOR_BORE
        if bore_override is not None:
            bore = bore_override
        module[2:6, 2:6] = outer
        module[3:5, 3:5] = bore
        for direction in self._ports_for_pipe(pipe_id, orientations):
            self._draw_arm(module, direction, outer, bore)

    def _draw_source(self, frame: np.ndarray) -> None:
        row, col = self.source_cell
        y0, x0 = BOARD_TOP + row * CELL_SIZE, BOARD_LEFT + col * CELL_SIZE
        module = frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE]
        self._draw_arm(module, self.spec.source_port, COLOR_LIVE_WALL, COLOR_FLUID)
        module[1:7, 1:7] = COLOR_LIVE_WALL
        module[2:6, 2:6] = COLOR_FLUID

    def _draw_terminal(self, frame: np.ndarray, terminal_id: str, interior: int | None = None) -> None:
        row, col = self.terminal_cells[terminal_id]
        y0, x0 = BOARD_TOP + row * CELL_SIZE, BOARD_LEFT + col * CELL_SIZE
        module = frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE]
        fill = COLOR_FILLED if terminal_id in self.filled_terminals else COLOR_BORE
        if interior is not None and terminal_id in self.filled_terminals:
            fill = interior
        self._draw_arm(module, self.spec.terminals[terminal_id], COLOR_TERMINAL, fill)
        module[1:7, 1:7] = COLOR_TERMINAL
        module[2:6, 2:6] = fill

    def _module(self, frame: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
        row, col = cell
        y0, x0 = BOARD_TOP + row * CELL_SIZE, BOARD_LEFT + col * CELL_SIZE
        return frame[y0 : y0 + CELL_SIZE, x0 : x0 + CELL_SIZE]

    def _draw_budget(self, frame: np.ndarray) -> None:
        frame[61:63, 4:60] = COLOR_BORE
        width = int(56 * self.remaining_steps / max(1, self.step_budget))
        if width:
            frame[61:63, 4 : 4 + width] = COLOR_ROTATE

    def _render(self, mode: str, payload: dict[str, object]) -> None:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_CANVAS, dtype=np.int8)
        orientations = self.orientations
        live = self.live_pipes
        if mode in {"quarter_ring", "port_overlay"}:
            orientations = payload["old_orientations"]  # type: ignore[assignment]
            live = payload["old_live"]  # type: ignore[assignment]
        for pipe_id in self.spec.pipes:
            bore_override = COLOR_LIVE_WALL if mode == "live_temp" and pipe_id == payload.get("pipe") else None
            self._draw_pipe(frame, pipe_id, orientations, live, bore_override)
            if pipe_id in live and mode == "settled":
                # Persistent clockwise arrows identify both the actionable
                # pipes and the only supported rotation direction.
                module = self._module(frame, self.pipe_cells[pipe_id])
                module[0, 1:5] = COLOR_ROTATE
                module[1:4, 6] = COLOR_ROTATE
                module[1, 4:7] = COLOR_ROTATE
                module[2, 5] = COLOR_ROTATE
        self._draw_source(frame)
        terminal_override = COLOR_FLUID if mode == "win_low" else None
        for terminal_id in self.terminal_cells:
            self._draw_terminal(frame, terminal_id, terminal_override)

        pipe_id = payload.get("pipe")
        if isinstance(pipe_id, str):
            module = self._module(frame, self.pipe_cells[pipe_id])
            if mode == "quarter_ring":
                module[0, 1:4] = COLOR_ROTATE
                module[1:4, 7] = COLOR_ROTATE
                module[0, 3] = COLOR_ROTATE
                module[1, 4] = COLOR_ROTATE
            elif mode == "port_overlay":
                old_ports = set(self._ports_for_pipe(pipe_id, payload["old_orientations"]))  # type: ignore[arg-type]
                new_ports = set(self._ports_for_pipe(pipe_id, self.orientations))
                for direction in old_ports:
                    self._draw_arm(module, direction, COLOR_INACTIVE, COLOR_BORE)
                for direction in new_ports - old_ports:
                    self._draw_arm(module, direction, COLOR_ROTATE, COLOR_ROTATE)
            elif mode == "settled_pulse":
                for newly_lit in payload["newly_live"]:  # type: ignore[union-attr]
                    pulse = self._module(frame, self.pipe_cells[newly_lit])
                    pulse[1, 2:6] = pulse[6, 2:6] = COLOR_ROTATE
                    pulse[2:6, 1] = pulse[2:6, 6] = COLOR_ROTATE
        feedback_cell = payload.get("feedback_cell")
        if isinstance(feedback_cell, tuple):
            module = self._module(frame, feedback_cell)
            color = COLOR_BLOCKED if mode == "blocked" else COLOR_TERMINAL
            module[0, 1:7] = module[7, 1:7] = color
            module[1:7, 0] = module[1:7, 7] = color
        self._draw_budget(frame)
        self.board.pixels = frame

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return

        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._render("settled", {})
            self.complete_action()
            return

        self._pending_win = False
        self._pending_failure = False
        self._spend_action()

        if action == GameAction.ACTION7.value:
            if self.undo_history:
                self.orientations = self.undo_history.pop()
                self.live_pipes, self.filled_terminals = self._connectivity(self.orientations)
            self._render("settled", {})
            self._finish_action()
            return

        if action != GameAction.ACTION6.value:
            self._render("settled", {})
            self._finish_action()
            return

        cell = self._click_cell()
        if cell is None:
            self._render("settled", {})
            self._finish_action()
            return
        pipe_id = self.pipe_at.get(cell)
        if pipe_id is not None:
            if pipe_id not in self.live_pipes:
                self._render("blocked", {"feedback_cell": cell})
                self._finish_action()
                return
            self._queue_rotation(pipe_id)
            self._play_animation_frame()
            return
        if cell == self.source_cell or cell in self.terminal_cells.values():
            self._render("fixed", {"feedback_cell": cell})
            self._finish_action()
            return
        self._render("settled", {})
        self._finish_action()
