from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

VIEW_SIZE = 64
CELL = 4

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = range(1, 8)

AIR = 0
WASH = 1
SOCKET = 2
TERRAIN_EDGE = 3
TERRAIN = 4
HATCH_INNER = 5
PLAYER = 6
FACING = 7
ALERT = 8
BRACE = 9
BRACE_DIAGONAL = 10
CONTACT = 11
SLAB = 12
SLAB_EDGE = 13
GOAL = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    facing: str
    carrying: bool
    baseline: int

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def budget(self) -> int:
        return self.baseline * 6


# Top row first.  # terrain, B occupied socket, o empty socket, lowercase
# letters loose slabs, P climber, and E exit hatch.
LEVEL_SPECS = (
    LevelSpec("First Rung", ("........", "........", "...E#...", "..PB#...", "########"), "R", False, 2),
    LevelSpec("Missing Rung", ("........", "........", "...E#...", "..Po#...", "########"), "R", True, 3),
    LevelSpec(
        "Drop Bridge",
        (".....aa.......", ".....B........", "..P.#....E....", "#####..#######", "#####..#######", "##############"),
        "R",
        False,
        8,
    ),
    LevelSpec(
        "Chosen Height",
        (
            ".......aa.....",
            "........B.....",
            ".........##B..",
            ".....E.....B..",
            "#######....B..",
            "#######.oP.B..",
            "#######..#####",
            "##############",
        ),
        "L",
        True,
        13,
    ),
    LevelSpec(
        "Second Landing First",
        (
            "......aa......",
            ".......BP.....",
            "........##B...",
            "..........B...",
            "......B...B...",
            ".....#.E..B...",
            ".....#..##....",
            ".......o......",
            "........###...",
            "##############",
        ),
        "L",
        False,
        25,
    ),
    LevelSpec(
        "Seal, Then Climb",
        (
            "..............",
            "B####.........",
            "......bbb.....",
            "B.....B.......",
            ".....#...E....",
            "B..aaaa..#....",
            "...B..........",
            "o.#....P......",
            "##############",
        ),
        "L",
        False,
        26,
    ),
    LevelSpec(
        "One Pull, Three Steps",
        (
            ".......cc......",
            ".......B.......",
            "B######........",
            "...............",
            "B..............",
            "...............",
            "B.......BB.....",
            "......bb.......",
            "o.....B..o.....",
            "######.E..###B.",
            ".............B.",
            ".....aa......B.",
            "......BP.......",
            "###############",
        ),
        "L",
        False,
        51,
    ),
)


@dataclass(frozen=True)
class Snapshot:
    player: tuple[int, int]
    facing: str
    carrying: bool
    braces: frozenset[tuple[int, int]]
    slabs: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RenderEvent:
    state: Snapshot
    flash: tuple[tuple[int, int], ...] = ()
    flash_color: int = ALERT
    bump: tuple[int, int] = (0, 0)
    pulse: bool = False
    crush: bool = False


def _parse(spec: LevelSpec) -> dict[str, object]:
    if any(len(row) != spec.width for row in spec.rows):
        raise ValueError(f"Ragged layout: {spec.name}")
    terrain: set[tuple[int, int]] = set()
    sockets: set[tuple[int, int]] = set()
    braces: set[tuple[int, int]] = set()
    slab_cells: dict[str, list[tuple[int, int]]] = {}
    player = exit_cell = None
    for row_index, row in enumerate(spec.rows):
        y = spec.height - 1 - row_index
        for x, symbol in enumerate(row):
            cell = (x, y)
            if symbol == "#":
                terrain.add(cell)
            elif symbol == "B":
                sockets.add(cell)
                braces.add(cell)
            elif symbol == "o":
                sockets.add(cell)
            elif symbol == "P":
                player = cell
            elif symbol == "E":
                exit_cell = cell
            elif symbol.islower():
                slab_cells.setdefault(symbol, []).append(cell)
    if player is None or exit_cell is None:
        raise ValueError(f"Missing player or exit: {spec.name}")
    slabs: dict[str, tuple[int, int, int]] = {}
    for slab_id, cells in slab_cells.items():
        xs = sorted(x for x, _ in cells)
        ys = {y for _, y in cells}
        if len(ys) != 1 or xs != list(range(xs[0], xs[-1] + 1)):
            raise ValueError(f"Slab {slab_id} is not horizontal and connected")
        slabs[slab_id] = (xs[0], ys.pop(), len(xs))
    return {
        "spec": spec,
        "terrain": terrain,
        "sockets": sockets,
        "braces": braces,
        "slabs": slabs,
        "player": player,
        "exit": exit_cell,
    }


def _build_level(spec: LevelSpec) -> Level:
    data = _parse(spec)
    canvas = Sprite(
        np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8),
        name="scene",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["scene"],
    )
    return Level([canvas], (VIEW_SIZE, VIEW_SIZE), data=data, name=spec.name)


def _action_id(action: object) -> int:
    return int(getattr(action, "value", action))


class FallingScaffold(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            "falling_scaffold-0001",
            [_build_level(spec) for spec in LEVEL_SPECS],
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, AIR, AIR),
            False,
            len(LEVEL_SPECS),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed,
        )

    def on_set_level(self, _level: Level) -> None:
        data = {
            key: self.current_level.get_data(key)
            for key in ("spec", "terrain", "sockets", "braces", "slabs", "player", "exit")
        }
        spec: LevelSpec = data["spec"]
        self.scene = self.current_level.get_sprites_by_tag("scene")[0]
        self.width = spec.width
        self.height = spec.height
        self.origin_x = (VIEW_SIZE - CELL * self.width) // 2
        self.origin_y = (VIEW_SIZE - CELL * self.height) // 2
        self.terrain = set(data["terrain"])
        self.sockets = set(data["sockets"])
        self.exit = tuple(data["exit"])
        self.slab_geometry = {key: (value[0], value[2]) for key, value in data["slabs"].items()}
        self.player = tuple(data["player"])
        self.facing = spec.facing
        self.carrying = spec.carrying
        self.braces = set(data["braces"])
        self.slabs = {key: value[1] for key, value in data["slabs"].items()}
        self.step_budget = spec.budget
        self.remaining_steps = self.step_budget
        self.undo_history: list[Snapshot] = []
        self._events: list[RenderEvent] = []
        self._terminal_after_events: str | None = None
        self._display_event: RenderEvent | None = None
        self._sync_visuals()

    def _snapshot(self) -> Snapshot:
        return Snapshot(
            self.player, self.facing, self.carrying, frozenset(self.braces), tuple(sorted(self.slabs.items()))
        )

    def _restore(self, state: Snapshot) -> None:
        self.player = state.player
        self.facing = state.facing
        self.carrying = state.carrying
        self.braces = set(state.braces)
        self.slabs = dict(state.slabs)

    def _slab_cells(self, state: Snapshot | None = None) -> set[tuple[int, int]]:
        slab_ys = dict(state.slabs) if state else self.slabs
        cells: set[tuple[int, int]] = set()
        for slab_id, y in slab_ys.items():
            x, width = self.slab_geometry[slab_id]
            cells.update((xx, y) for xx in range(x, x + width))
        return cells

    def _loaded_braces(self, state: Snapshot | None = None) -> set[tuple[int, int]]:
        braces = set(state.braces) if state else self.braces
        slab_cells = self._slab_cells(state)
        return {cell for cell in braces if (cell[0], cell[1] + 1) in slab_cells}

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def _passable(self, cell: tuple[int, int]) -> bool:
        return (
            self._in_bounds(cell)
            and cell not in self.terrain
            and cell not in self._slab_cells()
            and cell not in self._loaded_braces()
        )

    def _exposed_brace(self, cell: tuple[int, int]) -> bool:
        return cell in self.braces and cell not in self._loaded_braces()

    def _player_supported(self) -> bool:
        if self._exposed_brace(self.player):
            return True
        below = (self.player[0], self.player[1] - 1)
        return below in self.terrain or below in self.braces or below in self._slab_cells()

    def _anchored_slabs(self) -> set[str]:
        anchored: set[str] = set()
        occupied_by_slab = {
            cell: slab_id for slab_id, y in self.slabs.items() for cell in self._cells_for_slab(slab_id, y)
        }
        for slab_id, y in self.slabs.items():
            if any(
                (x, y - 1) in self.terrain or (x, y - 1) in self.braces for x, _ in self._cells_for_slab(slab_id, y)
            ):
                anchored.add(slab_id)
        changed = True
        while changed:
            changed = False
            for slab_id, y in self.slabs.items():
                if slab_id in anchored:
                    continue
                supporters = {occupied_by_slab.get((x, y - 1)) for x, _ in self._cells_for_slab(slab_id, y)}
                if any(supporter in anchored for supporter in supporters if supporter is not None):
                    anchored.add(slab_id)
                    changed = True
        return anchored

    def _cells_for_slab(self, slab_id: str, y: int) -> tuple[tuple[int, int], ...]:
        x, width = self.slab_geometry[slab_id]
        return tuple((xx, y) for xx in range(x, x + width))

    def _resolve_gravity(self) -> tuple[list[RenderEvent], bool, bool]:
        events: list[RenderEvent] = []
        moved_any = False
        while True:
            moving = set(self.slabs) - self._anchored_slabs()
            player_moves = not self._player_supported()
            if not moving and not player_moves:
                return events, False, moved_any
            moved_any = True
            old_cells = {cell for slab_id in moving for cell in self._cells_for_slab(slab_id, self.slabs[slab_id])}
            riding = any(
                (self.player[0], self.player[1] - 1) in self._cells_for_slab(slab_id, self.slabs[slab_id])
                for slab_id in moving
            )
            if riding:
                self.player = (self.player[0], self.player[1] - 1)
            for slab_id in moving:
                self.slabs[slab_id] -= 1
                if self.slabs[slab_id] < 0:
                    event = RenderEvent(self._snapshot(), tuple(old_cells), crush=True)
                    return events + [event] * 4, True, moved_any
            slab_cells = self._slab_cells()
            pinched = self.player in self.braces and (self.player[0], self.player[1] + 1) in slab_cells
            if (not riding and self.player in slab_cells) or pinched:
                event = RenderEvent(self._snapshot(), (self.player,), crush=True)
                return events + [event] * 4, True, moved_any
            if not riding and not self._player_supported():
                self.player = (self.player[0], self.player[1] - 1)
            if self.player[1] < 0:
                event = RenderEvent(self._snapshot(), (self.player,), crush=True)
                return events + [event] * 4, True, moved_any
            tick = RenderEvent(self._snapshot())
            events.extend((tick, tick))

    def _contact_cells(self) -> tuple[tuple[int, int], ...]:
        contacts: set[tuple[int, int]] = set()
        slab_cells = self._slab_cells()
        for slab_id, y in self.slabs.items():
            for x, _ in self._cells_for_slab(slab_id, y):
                below = (x, y - 1)
                if below in self.terrain or below in self.braces or below in slab_cells:
                    contacts.add(below)
        below_player = (self.player[0], self.player[1] - 1)
        if below_player in self.braces or below_player in slab_cells:
            contacts.add(below_player)
        return tuple(sorted(contacts))

    def _pixel_box(self, cell: tuple[int, int]) -> tuple[int, int]:
        x, y = cell
        return self.origin_x + CELL * x, self.origin_y + CELL * (self.height - 1 - y)

    def _draw_cell_pattern(self, frame: np.ndarray, cell: tuple[int, int], pattern: np.ndarray) -> None:
        x, y = self._pixel_box(cell)
        target = frame[y : y + CELL, x : x + CELL]
        target[:] = np.where(pattern >= 0, pattern, target)

    def _render(self, event: RenderEvent) -> np.ndarray:
        state = event.state
        frame = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8)
        slab_cells = self._slab_cells(state)
        loaded = self._loaded_braces(state)
        # Connected terrain: no cell borders, only genuinely exposed top edges.
        for cell in self.terrain:
            x, y = self._pixel_box(cell)
            frame[y : y + CELL, x : x + CELL] = TERRAIN
            if (cell[0], cell[1] + 1) not in self.terrain:
                frame[y, x : x + CELL] = GOAL if event.pulse else TERRAIN_EDGE
        hatch = np.array(
            [[GOAL] * 4, [GOAL, HATCH_INNER, HATCH_INNER, GOAL], [GOAL, HATCH_INNER, HATCH_INNER, GOAL], [GOAL] * 4],
            dtype=np.int8,
        )
        self._draw_cell_pattern(frame, self.exit, hatch)
        for socket in self.sockets - set(state.braces):
            self._draw_cell_pattern(
                frame,
                socket,
                np.array([[SOCKET] * 4, [SOCKET, -1, -1, SOCKET], [SOCKET, -1, -1, SOCKET], [SOCKET] * 4]),
            )
        for brace in state.braces:
            top = CONTACT if brace in loaded else BRACE
            self._draw_cell_pattern(
                frame,
                brace,
                np.array(
                    [[top] * 4, [BRACE, -1, BRACE_DIAGONAL, BRACE], [BRACE, BRACE_DIAGONAL, -1, BRACE], [BRACE] * 4]
                ),
            )
        dx = -1 if state.facing == "L" else 1
        space_target = (state.player[0] + dx, state.player[1])
        can_pick_up = not state.carrying and space_target in state.braces
        can_place = (
            state.carrying
            and space_target in self.sockets
            and space_target not in (self.terrain | slab_cells | set(state.braces) | {state.player})
        )
        if can_pick_up or can_place:
            px, py = self._pixel_box(space_target)
            frame[py, px : px + CELL] = CONTACT
            frame[py + CELL - 1, px : px + CELL] = CONTACT
            frame[py : py + CELL, px] = CONTACT
            frame[py : py + CELL, px + CELL - 1] = CONTACT
        for slab_id, y in state.slabs:
            x0, width = self.slab_geometry[slab_id]
            px, py = self._pixel_box((x0, y))
            frame[py : py + CELL, px : px + width * CELL] = SLAB
            frame[py + CELL - 1, px : px + width * CELL] = SLAB_EDGE
            frame[py + 1 : py + 3, px] = SLAB_EDGE
            frame[py + 1 : py + 3, px + width * CELL - 1] = SLAB_EDGE
        for cell in event.flash:
            if not self._in_bounds(cell):
                continue
            px, py = self._pixel_box(cell)
            if event.flash_color == WASH:
                frame[py : py + CELL, px : px + CELL] = WASH
            else:
                frame[py, px : px + CELL] = event.flash_color
                frame[py + CELL - 1, px : px + CELL] = event.flash_color
                frame[py : py + CELL, px] = event.flash_color
                frame[py : py + CELL, px + CELL - 1] = event.flash_color
        if event.crush:
            for cell in slab_cells | {state.player}:
                if self._in_bounds(cell):
                    px, py = self._pixel_box(cell)
                    frame[py : py + CELL, px : px + CELL] = ALERT
        # Player is intentionally compact so hatch frames and brace rails remain legible.
        px, py = self._pixel_box(state.player)
        px += event.bump[0]
        py += event.bump[1]
        if 0 <= px <= VIEW_SIZE - CELL and 0 <= py <= VIEW_SIZE - CELL:
            body = np.full((CELL, CELL), -1, dtype=np.int8)
            body[0, 1:3] = PLAYER
            body[1:4, 1:3] = PLAYER
            nub_col = 3 if state.facing == "R" else 0
            body[1, nub_col] = FACING
            if state.carrying:
                pack_col = 0 if state.facing == "R" else 3
                body[0, pack_col] = BRACE
                body[1, pack_col] = BRACE_DIAGONAL
                body[2, pack_col] = BRACE
            target = frame[py : py + CELL, px : px + CELL]
            target[:] = np.where(body >= 0, body, target)
        # Host safety budget, separated from the active board.
        frame[62:64, 2:62] = SOCKET
        fill = int(58 * max(0, self.remaining_steps) / max(1, self.step_budget))
        if fill:
            frame[62:64, 3 : 3 + fill] = CONTACT
        return frame

    def _sync_visuals(self) -> None:
        event = self._display_event or RenderEvent(self._snapshot())
        self.scene.pixels = self._render(event)

    def _show_next_event(self) -> None:
        self._display_event = self._events.pop(0)
        self._sync_visuals()
        if not self._events:
            terminal = self._terminal_after_events
            if terminal == "success":
                self.next_level()
            elif terminal == "failure":
                self.lose()
            self._terminal_after_events = None
            self.complete_action()

    def _animate(self, events: list[RenderEvent], terminal: str | None = None) -> None:
        self._events = events
        self._terminal_after_events = terminal
        self._show_next_event()

    def _move_action(self, action: int) -> tuple[bool, tuple[int, int]]:
        old = self._snapshot()
        dx = -1 if action == LEFT else 1 if action == RIGHT else 0
        dy = 1 if action == UP else -1 if action == DOWN else 0
        if dx:
            self.facing = "L" if dx < 0 else "R"
            adjacent = (self.player[0] + dx, self.player[1])
            if self._passable(adjacent):
                self.player = adjacent
                return True, (0, 0)
            above = (adjacent[0], adjacent[1] + 1)
            if not self._passable(adjacent) and self._passable(above):
                self.player = above
                return True, (0, 0)
            changed = old != self._snapshot()
            return changed, (dx, 0)
        destination = (self.player[0], self.player[1] + dy)
        if self._passable(destination) and (self._exposed_brace(self.player) or self._exposed_brace(destination)):
            self.player = destination
            return True, (0, 0)
        return False, (0, -dy)

    def _space_action(self) -> tuple[bool, tuple[int, int]]:
        dx = -1 if self.facing == "L" else 1
        target = (self.player[0] + dx, self.player[1])
        if not self.carrying and target in self.braces:
            self.braces.remove(target)
            self.carrying = True
            return True, target
        occupied = (
            target in self.terrain or target in self._slab_cells() or target in self.braces or target == self.player
        )
        if self.carrying and target in self.sockets and not occupied:
            self.braces.add(target)
            self.carrying = False
            return True, target
        return False, target

    def _click_cell(self) -> tuple[int, int] | None:
        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        if not (self.origin_x <= x < self.origin_x + self.width * CELL):
            return None
        if not (self.origin_y <= y < self.origin_y + self.height * CELL):
            return None
        return ((x - self.origin_x) // CELL, self.height - 1 - (y - self.origin_y) // CELL)

    def step(self) -> None:
        if self._events:
            self._show_next_event()
            return
        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self._display_event = None
            self._sync_visuals()
            self.complete_action()
            return

        if action == CLICK:
            # Click is not a gameplay control. Keep it silent so a transient
            # wash cannot be mistaken for brace selection or relocation.
            self._display_event = None
            self._sync_visuals()
            self.complete_action()
            return

        self.remaining_steps -= 1
        before = self._snapshot()
        events: list[RenderEvent] = []
        terminal: str | None = None

        if action == UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self._display_event = None
        elif action in (UP, DOWN, LEFT, RIGHT):
            changed, bump = self._move_action(action)
            if changed:
                self.undo_history.append(before)
            if self.player == before.player:
                target = (before.player[0] + (-1 if action == LEFT else 1 if action == RIGHT else 0), before.player[1])
                if action in (UP, DOWN):
                    target = (before.player[0], before.player[1] + (1 if action == UP else -1))
                events.extend(
                    [RenderEvent(self._snapshot(), (target,), bump=bump), RenderEvent(self._snapshot(), (target,))]
                )
        elif action == SPACE:
            changed, target = self._space_action()
            if changed:
                self.undo_history.append(before)
                events.append(RenderEvent(self._snapshot(), (target,), CONTACT))
            else:
                events.append(RenderEvent(self._snapshot(), (target,)))
        gravity_events, crushed, moved_any = self._resolve_gravity()
        events.extend(gravity_events)
        if crushed:
            terminal = "failure"
        elif moved_any:
            events.append(RenderEvent(self._snapshot(), self._contact_cells(), CONTACT))

        if terminal is None and self.player == self.exit:
            final = self._snapshot()
            events.extend(RenderEvent(final, pulse=index % 2 == 0) for index in range(6))
            terminal = "success"
        elif terminal is None and self.remaining_steps <= 0:
            final = self._snapshot()
            events.extend(RenderEvent(final, tuple(self.terrain), crush=True) for _ in range(4))
            terminal = "failure"

        if events:
            self._animate(events, terminal)
            return
        self._display_event = None
        self._sync_visuals()
        self.complete_action()
