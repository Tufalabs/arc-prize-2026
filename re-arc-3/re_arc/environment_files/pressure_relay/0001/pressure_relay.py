from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

GAME_ID = "pressure_relay-0001"
VIEW_SIZE = 64
CELL = 4
BOARD_X = 6
BOARD_Y = 9

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = range(1, 8)
DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

BG = 5
WALL = 4
WALL_EDGE = 3
FLOOR = 1
SHADOW = 2
WHITE = 0
FAIL = 8
AGENT_OUTLINE = 9
AGENT_FILL = 10
KEY_RED = 8
KEY_DARK = 13
ARTIFACT = 12
EXIT = 14

CHANNEL_COLORS = {"a": (9, 10), "b": (6, 7), "c": (11, 12), "d": (15, 0)}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    conduits: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = ()
    extra_terrain: tuple[tuple[int, int, str], ...] = ()
    baseline: int = 1


LEVEL_SPECS = (
    LevelSpec(
        "Carry It Home",
        (
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "###....######",
            "###sa.e######",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
        ),
        baseline=4,
    ),
    LevelSpec(
        "Held, Not Toggled",
        (
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "#..#...#...##",
            "#s.Aw.2Ba.e##",
            "#..#...#...##",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
        ),
        conduits=(("a", ((5, 7), (4, 7))), ("b", ((7, 7), (8, 7)))),
        extra_terrain=((5, 7, "plate_a"),),
        baseline=12,
    ),
    LevelSpec(
        "A Spent Key Leaves a Passage",
        (
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "##....#...###",
            "##esk.L.a.###",
            "##....#...###",
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
        ),
        baseline=13,
    ),
    LevelSpec(
        "Borrowed Pressure",
        (
            "#############",
            "#############",
            "#############",
            "#############",
            "#############",
            "##....#....##",
            "##sk1.A.w..##",
            "##....#....##",
            "#########L###",
            "########ea.##",
            "########...##",
            "#############",
            "#############",
        ),
        conduits=(("a", ((5, 7), (7, 7))),),
        baseline=35,
    ),
    LevelSpec(
        "The Relay Loop",
        (
            "#############",
            "#.....#...#e.",
            "#.....#...#.a",
            "#.sw1.A...#L#",
            "#.....#.k.#..",
            "#.....#..3C..",
            "#.....#...#..",
            "#.....#...#..",
            "#.....B.2.###",
            "#.....#...###",
            "#############",
            "#############",
            "#############",
        ),
        conduits=(("a", ((5, 4), (7, 4))), ("b", ((9, 9), (7, 9))), ("c", ((10, 6), (11, 6)))),
        baseline=57,
    ),
    LevelSpec(
        "Cargo as Support",
        (
            "#############",
            "#.....#######",
            "#.....###..##",
            "#.....B..2.##",
            "#.k...###..##",
            "#.....###3.Ce",
            "#.....###..##",
            "#.....###..##",
            "#.sw1.A.L.a##",
            "#.....###..##",
            "#.....#######",
            "#############",
            "#############",
        ),
        conduits=(("a", ((5, 9), (7, 9))), ("b", ((10, 4), (7, 4))), ("c", ((10, 6), (12, 6)))),
        baseline=63,
    ),
    LevelSpec(
        "Closed Circuit",
        (
            "#############",
            "###########..",
            "#.....#...De.",
            "#.sw1.A...###",
            "#.....#.k.#.a",
            "#.....#...#L#",
            "#.....#...#..",
            "#.....#..3C..",
            "#.....B.2.#.4",
            "#.....#...#..",
            "#############",
            "#############",
            "#############",
        ),
        conduits=(
            ("a", ((5, 4), (7, 4))),
            ("b", ((9, 9), (7, 9))),
            ("c", ((10, 8), (11, 8))),
            ("d", ((13, 9), (13, 4), (11, 4), (11, 3))),
        ),
        baseline=86,
    ),
)


def _pixel(cell: tuple[int, int]) -> tuple[int, int]:
    x, y = cell
    return BOARD_X + (x - 1) * CELL, BOARD_Y + (y - 1) * CELL


class PressureRelayView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: PressureRelay | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = BG
        self._draw_inventory(frame, game)
        self._draw_budget(frame, game)
        self._draw_board(frame, game)
        return frame

    def _draw_inventory(self, frame: np.ndarray, game: PressureRelay) -> None:
        flash = game._anim_kind == "invalid_space" and game._anim_phase % 2 == 0
        frame[1:7, 29:35] = SHADOW if flash else WALL_EDGE
        frame[2:6, 30:34] = FAIL if flash else WHITE
        frame[3:5, 31:33] = SHADOW
        if game.inventory is not None:
            self._draw_item_icon(frame, 30, 2, game.item_kind[game.inventory], size=4)

    def _draw_budget(self, frame: np.ndarray, game: PressureRelay) -> None:
        frame[7:9, BOARD_X : BOARD_X + 52] = SHADOW
        width = int(52 * game.remaining_steps / max(1, game.step_budget))
        if width:
            frame[7:9, BOARD_X : BOARD_X + width] = EXIT
        if game._anim_kind == "failure":
            frame[7:9, BOARD_X : BOARD_X + 52] = FAIL if game._anim_phase % 2 == 0 else SHADOW

    def _draw_board(self, frame: np.ndarray, game: PressureRelay) -> None:
        frame[BOARD_Y : BOARD_Y + 52, BOARD_X : BOARD_X + 52] = WALL
        for cell in game.floor:
            x0, y0 = _pixel(cell)
            frame[y0 : y0 + CELL, x0 : x0 + CELL] = FLOOR

        # Only floor/wall boundaries are outlined; there is no logical-cell lattice.
        for x, y in game.floor:
            x0, y0 = _pixel((x, y))
            if (x, y - 1) not in game.floor:
                frame[y0, x0 : x0 + CELL] = WALL_EDGE
            if (x, y + 1) not in game.floor:
                frame[y0 + 3, x0 : x0 + CELL] = WALL_EDGE
            if (x - 1, y) not in game.floor:
                frame[y0 : y0 + CELL, x0] = WALL_EDGE
            if (x + 1, y) not in game.floor:
                frame[y0 : y0 + CELL, x0 + 3] = WALL_EDGE

        for channel, points in game.conduits.items():
            self._draw_conduit(frame, points, channel, game.plate_active(channel))

        plate_override = None
        gate_override = None
        if game._anim_kind == "plate":
            channel = str(game._anim_data["channel"])
            old = bool(game._anim_data["old"])
            new = game.plate_active(channel)
            if game._anim_phase == 0:
                plate_override, gate_override = (channel, old), (channel, old, 0)
            elif game._anim_phase == 1:
                plate_override, gate_override = (channel, new), (channel, old, 0)
            elif game._anim_phase == 2:
                gate_override = (channel, new, 1)

        for channel, cell in game.plates.items():
            active = game.plate_active(channel)
            if plate_override and plate_override[0] == channel:
                active = plate_override[1]
            self._draw_plate(frame, cell, channel, active)

        for channel, cell in game.gates.items():
            opened = game.plate_active(channel)
            half = False
            if gate_override and gate_override[0] == channel:
                opened, half = gate_override[1], bool(gate_override[2])
            self._draw_gate(frame, cell, channel, opened, half)
            if game._anim_kind == "invalid_move" and game._anim_data.get("gate") == channel:
                x0, y0 = _pixel(cell)
                frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = FLOOR
                shake_y = y0 + 1 + game._anim_phase % 2
                frame[shake_y, x0 : x0 + 4] = CHANNEL_COLORS[channel][0]

        for lock_id, cell in game.locks.items():
            self._draw_lock(frame, cell, lock_id in game.open_locks, game)

        self._draw_exit(frame, game.exit_position, game)

        for item_id, cell in game.loose_position.items():
            self._draw_item(frame, cell, game.item_kind[item_id])
        if game._anim_kind == "plate" and game._anim_phase == 0 and bool(game._anim_data["old"]):
            channel = str(game._anim_data["channel"])
            self._draw_item(frame, game.plates[channel], str(game._anim_data["item_kind"]))

        player_cell = game.player_pos
        player_offset = (0, 0)
        if game._anim_kind == "invalid_move":
            dx, dy = game._anim_data["delta"]
            if game._anim_phase % 2 == 0:
                player_offset = (dx, dy)
        if game._anim_kind == "lock" and game._anim_phase < 4:
            player_cell = game._anim_data["source"]
            if game._anim_phase == 3:
                dx = game.player_pos[0] - player_cell[0]
                dy = game.player_pos[1] - player_cell[1]
                player_offset = (dx * 2, dy * 2)
        self._draw_player(frame, player_cell, game, player_offset)
        if game._anim_kind == "lock" and game._anim_phase == 0:
            x0, y0 = _pixel(game._anim_data["source"])
            target_x, target_y = game._anim_data["lock"]
            source_x, source_y = game._anim_data["source"]
            frame[y0 + 1 + (target_y - source_y), x0 + 2 + (target_x - source_x)] = KEY_RED

        if game._anim_kind == "invalid_space":
            cell = game.player_pos
            x0, y0 = _pixel(cell)
            frame[y0, x0 : x0 + CELL] = FAIL
            frame[y0 + 3, x0 : x0 + CELL] = FAIL
        if game._anim_kind == "completion":
            x0, y0 = _pixel(game.exit_position)
            pulse = EXIT if game._anim_phase % 2 == 0 else WHITE
            frame[y0, x0 : x0 + 4] = pulse
            frame[y0 + 3, x0 : x0 + 4] = pulse

    def _draw_conduit(self, frame: np.ndarray, points: tuple[tuple[int, int], ...], channel: str, active: bool) -> None:
        primary, accent = CHANNEL_COLORS[channel]
        pixel_points = []
        for point in points:
            x0, y0 = _pixel(point)
            pixel_points.append((x0 + 1, y0 + 1))
        for index in range(len(pixel_points) - 1):
            x1, y1 = pixel_points[index]
            x2, y2 = pixel_points[index + 1]
            if x1 == x2:
                lo, hi = sorted((y1, y2))
                for pos in range(lo, hi + 1):
                    if active or (pos - lo) % 2 == 0:
                        frame[pos, x1] = accent if active and (pos - lo) % 3 == 0 else primary
            else:
                lo, hi = sorted((x1, x2))
                for pos in range(lo, hi + 1):
                    if active or (pos - lo) % 2 == 0:
                        frame[y1, pos] = accent if active and (pos - lo) % 3 == 0 else primary

    def _draw_plate(self, frame: np.ndarray, cell: tuple[int, int], channel: str, active: bool) -> None:
        x0, y0 = _pixel(cell)
        primary, accent = CHANNEL_COLORS[channel]
        frame[y0 : y0 + 4, x0 : x0 + 4] = primary
        frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = accent if active else SHADOW
        if active:
            frame[y0 + 1, x0 + 1 : x0 + 3] = primary

    def _draw_gate(self, frame: np.ndarray, cell: tuple[int, int], channel: str, opened: bool, half: bool) -> None:
        x0, y0 = _pixel(cell)
        primary, accent = CHANNEL_COLORS[channel]
        frame[y0 : y0 + 4, x0] = WALL_EDGE
        frame[y0 : y0 + 4, x0 + 3] = WALL_EDGE
        if opened:
            frame[y0 + 1 : y0 + 3, x0] = accent
            frame[y0 + 1 : y0 + 3, x0 + 3] = accent
        elif half:
            frame[y0 + 1, x0 : x0 + 2] = primary
            frame[y0 + 2, x0 + 2 : x0 + 4] = primary
        else:
            frame[y0 + 1 : y0 + 3, x0 : x0 + 4] = primary
            frame[y0 + 2, x0 + 1 : x0 + 3] = accent

    def _draw_lock(self, frame: np.ndarray, cell: tuple[int, int], opened: bool, game: PressureRelay) -> None:
        x0, y0 = _pixel(cell)
        phase = game._anim_phase if game._anim_kind == "lock" and game._anim_data["lock"] == cell else None
        invalid = game._anim_kind == "invalid_lock" and game._anim_data["lock"] == cell
        if opened and phase is None:
            frame[y0 : y0 + 4, x0 : x0 + 4] = KEY_DARK
            frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = SHADOW
            return
        if phase is not None and phase >= 2:
            frame[y0 : y0 + 4, x0 : x0 + 4] = KEY_DARK
            frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = FLOOR
            return
        color = FAIL if invalid and game._anim_phase % 2 == 0 else KEY_RED
        frame[y0 : y0 + 4, x0 : x0 + 4] = KEY_DARK
        if phase == 1:
            frame[y0 : y0 + 4, x0] = color
            frame[y0 : y0 + 4, x0 + 3] = color
        else:
            frame[y0, x0 : x0 + 2] = color
            frame[y0 + 1, x0 : x0 + 3] = color
            frame[y0 + 2 : y0 + 4, x0 + 1] = color
            frame[y0 + 3, x0 + 2] = color

    def _draw_exit(self, frame: np.ndarray, cell: tuple[int, int], game: PressureRelay) -> None:
        x0, y0 = _pixel(cell)
        frame[y0 : y0 + 4, x0 : x0 + 4] = EXIT
        center = WHITE if game._anim_kind == "completion" and game._anim_phase % 2 else ARTIFACT
        frame[y0, x0 + 1 : x0 + 3] = center
        frame[y0 + 1 : y0 + 3, x0 : x0 + 4] = center
        frame[y0 + 3, x0 + 1 : x0 + 3] = center
        frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = EXIT

    def _draw_item(self, frame: np.ndarray, cell: tuple[int, int], kind: str) -> None:
        x0, y0 = _pixel(cell)
        self._draw_item_icon(frame, x0, y0, kind, size=4)

    def _draw_item_icon(self, frame: np.ndarray, x0: int, y0: int, kind: str, *, size: int) -> None:
        del size
        if kind == "weight":
            corner = frame[y0, x0]
            frame[y0 : y0 + 3, x0 : x0 + 3] = SHADOW
            frame[y0 + 2, x0 : x0 + 3] = WALL_EDGE
            frame[y0 : y0 + 3, x0 + 2] = WALL_EDGE
            frame[y0, x0] = corner
        elif kind == "key":
            frame[y0 : y0 + 2, x0 : x0 + 2] = KEY_RED
            frame[y0 + 1 : y0 + 4, x0 + 1] = KEY_DARK
            frame[y0 + 3, x0 + 2] = KEY_DARK
        else:
            frame[y0, x0 + 1 : x0 + 3] = ARTIFACT
            frame[y0 + 1 : y0 + 3, x0 : x0 + 4] = ARTIFACT
            frame[y0 + 3, x0 + 1 : x0 + 3] = ARTIFACT
            frame[y0 + 1, x0 + 1] = WHITE

    def _draw_player(
        self, frame: np.ndarray, cell: tuple[int, int], game: PressureRelay, offset: tuple[int, int]
    ) -> None:
        x0, y0 = _pixel(cell)
        x0 += offset[0]
        y0 += offset[1]
        frame[y0 + 1, x0 + 1 : x0 + 4] = AGENT_OUTLINE
        frame[y0 + 2 : y0 + 4, x0 + 1] = AGENT_OUTLINE
        frame[y0 + 2 : y0 + 4, x0 + 3] = AGENT_OUTLINE
        frame[y0 + 2 : y0 + 4, x0 + 2] = AGENT_FILL
        if game.inventory is not None:
            kind = game.item_kind[game.inventory]
            color = SHADOW if kind == "weight" else KEY_RED if kind == "key" else ARTIFACT
            frame[y0, x0 + 1 : x0 + 4] = color
            if kind == "artifact":
                frame[y0, x0 + 2] = WHITE


class PressureRelay(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = PressureRelayView()
        levels = [Level([], grid_size=(64, 64), data={"spec": spec}, name=spec.name) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, BG, BG, [self._view])
        super().__init__(GAME_ID, levels, camera, False, len(levels), list(range(1, 8)), seed=seed)
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        spec: LevelSpec = level.get_data("spec")
        self.spec = spec
        self.floor: set[tuple[int, int]] = set()
        self.terrain: dict[tuple[int, int], str] = {}
        self.plates: dict[str, tuple[int, int]] = {}
        self.gates: dict[str, tuple[int, int]] = {}
        self.locks: dict[str, tuple[int, int]] = {}
        self.item_kind: dict[str, str] = {}
        self.loose_position: dict[str, tuple[int, int]] = {}
        self.player_pos = (1, 1)
        self.exit_position = (1, 1)
        symbol_terrain = {"1": "plate_a", "2": "plate_b", "3": "plate_c", "4": "plate_d"}
        gate_symbols = {"A": "a", "B": "b", "C": "c", "D": "d"}
        item_symbols = {"w": ("weight", "weight"), "k": ("key", "key"), "a": ("artifact", "artifact")}
        for y, row in enumerate(spec.rows, 1):
            if len(row) != 13:
                raise ValueError(f"{spec.name}: row {y} is not 13 cells")
            for x, symbol in enumerate(row, 1):
                if symbol == "#":
                    continue
                cell = (x, y)
                self.floor.add(cell)
                self.terrain[cell] = "floor"
                if symbol == "s":
                    self.player_pos = cell
                elif symbol == "e":
                    self.exit_position = cell
                    self.terrain[cell] = "exit"
                elif symbol in item_symbols:
                    item_id, kind = item_symbols[symbol]
                    self.item_kind[item_id] = kind
                    self.loose_position[item_id] = cell
                elif symbol in symbol_terrain:
                    terrain = symbol_terrain[symbol]
                    channel = terrain[-1]
                    self.terrain[cell] = terrain
                    self.plates[channel] = cell
                elif symbol in gate_symbols:
                    channel = gate_symbols[symbol]
                    self.terrain[cell] = f"gate_{channel}"
                    self.gates[channel] = cell
                elif symbol == "L":
                    lock_id = f"lock_{x}_{y}"
                    self.terrain[cell] = lock_id
                    self.locks[lock_id] = cell
        for x, y, terrain in spec.extra_terrain:
            cell = (x, y)
            self.terrain[cell] = terrain
            if terrain.startswith("plate_"):
                self.plates[terrain[-1]] = cell
        self.conduits = {channel: points for channel, points in spec.conduits}
        self.inventory: str | None = None
        self.open_locks: set[str] = set()
        self.completed = False
        self.step_budget = spec.baseline * 6
        self.remaining_steps = self.step_budget
        self._history: list[tuple[tuple[int, int], str | None, dict[str, tuple[int, int]], set[str], bool]] = []
        self._anim_kind: str | None = None
        self._anim_phase = 0
        self._anim_queue: list[int] = []
        self._anim_data: dict[str, object] = {}
        self._pending_terminal: str | None = None

    def plate_active(self, channel: str) -> bool:
        plate = self.plates.get(channel)
        return plate is not None and plate in self.loose_position.values()

    def _snapshot(self) -> tuple[tuple[int, int], str | None, dict[str, tuple[int, int]], set[str], bool]:
        return (self.player_pos, self.inventory, dict(self.loose_position), set(self.open_locks), self.completed)

    def _restore(self, snapshot) -> None:
        self.player_pos, self.inventory, loose, locks, self.completed = snapshot
        self.loose_position = dict(loose)
        self.open_locks = set(locks)

    def _begin_animation(self, kind: str, frames: int, data: dict[str, object] | None = None) -> None:
        self._anim_kind = kind
        self._anim_phase = 0
        self._anim_queue = list(range(1, frames))
        self._anim_data = data or {}
        if frames == 1:
            self._anim_data["single_frame"] = True
            self.complete_action()

    def _finish_or_animate(self) -> None:
        if self._anim_kind is not None:
            return
        self._finish_action()

    def _finish_action(self) -> None:
        terminal = self._pending_terminal
        self._pending_terminal = None
        if terminal == "next":
            self.next_level()
            self.complete_action()
        elif terminal == "lose":
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _advance_animation(self) -> None:
        if self._anim_queue:
            self._anim_phase = self._anim_queue.pop(0)
            if self._anim_queue:
                return
        self._anim_kind = None
        self._anim_data = {}
        self._finish_action()

    def _lock_at(self, cell: tuple[int, int]) -> str | None:
        for lock_id, position in self.locks.items():
            if position == cell:
                return lock_id
        return None

    def _resolve_move(self, action_id: int) -> None:
        dx, dy = DELTAS[action_id]
        target = (self.player_pos[0] + dx, self.player_pos[1] + dy)
        terrain = self.terrain.get(target)
        if target not in self.floor:
            self._begin_animation("invalid_move", 2, {"delta": (dx, dy)})
            return
        if terrain and terrain.startswith("gate_"):
            channel = terrain[-1]
            if not self.plate_active(channel):
                self._begin_animation("invalid_move", 3, {"delta": (dx, dy), "gate": channel})
                return
        lock_id = self._lock_at(target)
        if lock_id is not None and lock_id not in self.open_locks:
            if self.inventory is None or self.item_kind[self.inventory] != "key":
                self._begin_animation("invalid_lock", 2, {"lock": target})
                return
            self._history.append(self._snapshot())
            source = self.player_pos
            self.inventory = None
            self.open_locks.add(lock_id)
            self.player_pos = target
            self._begin_animation("lock", 5, {"lock": target, "source": source})
            return
        self._history.append(self._snapshot())
        self.player_pos = target

    def _resolve_space(self) -> None:
        old_active = {channel: self.plate_active(channel) for channel in self.plates}
        if self.inventory is None:
            item_id = next((item for item, cell in self.loose_position.items() if cell == self.player_pos), None)
            if item_id is None:
                self._begin_animation("invalid_space", 1)
                return
            self._history.append(self._snapshot())
            self.loose_position.pop(item_id)
            self.inventory = item_id
        else:
            terrain = self.terrain.get(self.player_pos, "wall")
            occupied = self.player_pos in self.loose_position.values()
            droppable = terrain == "floor" or terrain.startswith("plate_")
            if occupied or not droppable:
                self._begin_animation("invalid_space", 2 if occupied else 1)
                return
            self._history.append(self._snapshot())
            self.loose_position[self.inventory] = self.player_pos
            self.inventory = None
        changed = [channel for channel, was in old_active.items() if was != self.plate_active(channel)]
        if changed:
            channel = changed[0]
            if self.inventory is not None:
                transition_item_kind = self.item_kind[self.inventory]
            else:
                transition_item_kind = next(
                    self.item_kind[item]
                    for item, position in self.loose_position.items()
                    if position == self.plates[channel]
                )
            self._begin_animation(
                "plate", 4, {"channel": channel, "old": old_active[channel], "item_kind": transition_item_kind}
            )

    def _check_success(self) -> None:
        if self.completed:
            return
        if (
            self.player_pos == self.exit_position
            and self.inventory is not None
            and self.item_kind[self.inventory] == "artifact"
        ):
            self.completed = True
            self._pending_terminal = "next"
            self._begin_animation("completion", 6)

    def step(self) -> None:
        if self._anim_kind is not None and self._anim_data.get("single_frame"):
            self._anim_kind = None
            self._anim_data = {}
            self._anim_queue = []
        if self._anim_kind is not None:
            self._advance_animation()
            return
        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        action_id = self.action.id.value
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if action_id == UNDO:
            if self._history:
                self._restore(self._history.pop())
        elif action_id in DELTAS:
            self._resolve_move(action_id)
        elif action_id == SPACE:
            self._resolve_space()
        elif action_id == CLICK:
            pass

        self._check_success()
        if not self.completed and self.remaining_steps == 0:
            self._pending_terminal = "lose"
            self._begin_animation("failure", 2)
        self._finish_or_animate()
