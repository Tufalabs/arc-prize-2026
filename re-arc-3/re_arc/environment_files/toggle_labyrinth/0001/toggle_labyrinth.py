from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

VIEW = 64
CELL = 4
BACKGROUND = 5
WALL = 4
EDGE = 3
FLOOR = 1
JAMB = 2
DOOR = 13
LATCH = 8
PLAYER = 14
PURPLE = 15
GEM = 7
PLATE_COLORS = {"p": 11, "q": 6, "r": 9}
FLASH_COLORS = {"p": 0, "q": 7, "r": 10}
MOVE = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
FACING = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
PORTS = ((1, 0), (2, 1), (1, 2), (0, 1))
PATH_DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    links: dict[str, tuple[str, ...]]
    initial_open: tuple[str, ...]
    baseline: int


LEVEL_SPECS = (
    LevelSpec("The Matching Recess", ("#######", "#.....#", "#x.so.#", "#.....#", "#######"), {}, (), 4),
    LevelSpec(
        "One Cable, One Door", ("#########", "#..p#...#", "#x..A.o.#", "#..s#...#", "#########"), {"p": ("A",)}, (), 11
    ),
    LevelSpec(
        "The Door You Closed Behind You",
        ("###########", "###########", "###########", "#xB.p.A.o.#", "####.######", "####s######", "###########"),
        {"p": ("A", "B")},
        ("B",),
        13,
    ),
    LevelSpec(
        "Shared Cable",
        (
            "#############",
            "#############",
            "#############",
            "#######q#####",
            "###...A.#####",
            "###...##.o..#",
            "###.#.B..##.#",
            "###p#######C#",
            "###.######..#",
            "#s..######.x#",
            "#############",
        ),
        {"p": ("A", "C"), "q": ("B", "C")},
        ("C",),
        28,
    ),
    LevelSpec(
        "Do Not Retrace the Turnstile",
        (
            "#############",
            "#############",
            "#############",
            "##########o.#",
            "######.AqD..#",
            "#####..####.#",
            "#####.....B.#",
            "###p..#######",
            "###.#########",
            "###.#########",
            "###...##...##",
            "#s..#..C.#.x#",
            "#############",
        ),
        {"p": ("A", "B"), "q": ("A", "B", "C", "D")},
        ("B",),
        44,
    ),
    LevelSpec(
        "State Handoff",
        (
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "###############",
            "######...AqE..#",
            "#####..######.#",
            "#####.......B.#",
            "###p..#########",
            "###.###########",
            "###.#####..or.#",
            "###...##..###D#",
            "#s..#..C.####x#",
            "###############",
        ),
        {"p": ("A", "B"), "q": ("A", "B", "C", "E"), "r": ("C", "D")},
        ("B",),
        50,
    ),
    LevelSpec(
        "The Entire Journey Is the Switch",
        (
            "###############",
            "###############",
            "######.AqF....#",
            "#####..#####..#",
            "###...######..#",
            "###G#......B..#",
            "###p#.#######.#",
            "###H#.#######.#",
            "#s..I.#######.#",
            "#.#.#########.#",
            "#.#..C..oDrE..#",
            "#J#############",
            "#x#############",
            "###############",
            "###############",
        ),
        {"p": ("B", "C", "G", "H"), "q": ("A", "B", "C", "F"), "r": ("C", "D", "E", "I", "J")},
        ("A", "B", "D", "H"),
        76,
    ),
)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _parse(spec: LevelSpec) -> dict[str, object]:
    starts: list[tuple[int, int]] = []
    artifacts: list[tuple[int, int]] = []
    exits: list[tuple[int, int]] = []
    plates: dict[tuple[int, int], str] = {}
    doors: dict[str, tuple[int, int]] = {}
    for y, row in enumerate(spec.rows):
        for x, char in enumerate(row):
            if char == "s":
                starts.append((x, y))
            elif char == "o":
                artifacts.append((x, y))
            elif char == "x":
                exits.append((x, y))
            elif char in PLATE_COLORS:
                plates[(x, y)] = char
            elif "A" <= char <= "J":
                doors[char] = (x, y)
    if len(starts) != 1 or len(artifacts) != 1 or len(exits) != 1:
        raise ValueError(f"{spec.name} requires one start, artifact, and exit")
    return {
        "spec": spec,
        "start": starts[0],
        "artifact": artifacts[0],
        "exit": exits[0],
        "plates": plates,
        "doors": doors,
        "budget": spec.baseline * 6,
    }


class ToggleLabyrinthView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ToggleLabyrinth | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.game is not None:
            frame[:, :] = self.game._render_board()
        return frame


class ToggleLabyrinth(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = ToggleLabyrinthView()
        blank = np.full((VIEW, VIEW), BACKGROUND, dtype=np.int8)
        levels = [
            Level(
                sprites=[Sprite(blank, name="canvas", collidable=False)],
                grid_size=(VIEW, VIEW),
                name=spec.name,
                data=_parse(spec),
            )
            for spec in LEVEL_SPECS
        ]
        super().__init__(
            "toggle_labyrinth-0001",
            levels,
            Camera(0, 0, VIEW, VIEW, BACKGROUND, BACKGROUND, [self._view]),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )
        self._cache_level_visuals()
        self._view.game = self

    def _cache_level_visuals(self) -> None:
        for index in range(len(self._levels)):
            if self.level_index != index:
                self.set_level(index)
            cached_routes = {key: list(path) for key, path in self._cable_routes.items()}
            cached_order = list(self._cable_order)
            cached_frame = self._static_frame.copy()
            for level in (self._levels[index], self._clean_levels[index]):
                level._data["cable_routes"] = cached_routes
                level._data["cable_order"] = cached_order
                level._data["static_frame"] = cached_frame
        self.set_level(0)

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.rows = self.spec.rows
        self.height = len(self.rows)
        self.width = len(self.rows[0])
        self.origin_x = (VIEW - CELL * self.width) // 2
        self.origin_y = (VIEW - CELL * self.height) // 2
        self.start = tuple(level.get_data("start"))
        self.player_position = self.start
        self.artifact_position = tuple(level.get_data("artifact"))
        self.exit_position = tuple(level.get_data("exit"))
        self.plates = {tuple(cell): plate for cell, plate in level.get_data("plates").items()}
        self.doors = {door: tuple(cell) for door, cell in level.get_data("doors").items()}
        self.door_at = {cell: door for door, cell in self.doors.items()}
        self.door_open = {door: door in self.spec.initial_open for door in self.doors}
        self.door_axes = {door: self._door_axis(cell) for door, cell in self.doors.items()}
        self.has_artifact = False
        self.artifact_present = True
        self.blocked_facing = "NONE"
        self.won = False
        self.step_budget = int(level.get_data("budget"))
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[object, ...]] = []
        self._animation: list[int] = []
        self._animation_kind: str | None = None
        self._animation_phase = -1
        self._animation_plate: str | None = None
        self._old_doors: dict[str, bool] = {}
        self._pending_completion = False
        cached_routes = level.get_data("cable_routes")
        if cached_routes is None:
            self._cable_routes, self._cable_order = self._route_cables()
            self._static_frame = np.full((VIEW, VIEW), BACKGROUND, dtype=np.int8)
            self._draw_walls(self._static_frame)
            self._draw_cables(self._static_frame)
        else:
            self._cable_routes = cached_routes
            self._cable_order = level.get_data("cable_order")
            self._static_frame = level.get_data("static_frame")

    def _door_axis(self, cell: tuple[int, int]) -> str:
        x, y = cell
        left_right = self.rows[y][x - 1] != "#" and self.rows[y][x + 1] != "#"
        return "horizontal" if left_right else "vertical"

    def _snapshot(self) -> tuple[object, ...]:
        return (
            self.player_position,
            self.has_artifact,
            self.artifact_present,
            dict(self.door_open),
            self.blocked_facing,
            self.won,
        )

    def _restore(self, snapshot: tuple[object, ...]) -> None:
        position, carried, present, doors, facing, won = snapshot
        self.player_position = position
        self.has_artifact = bool(carried)
        self.artifact_present = bool(present)
        self.door_open = dict(doors)
        self.blocked_facing = str(facing)
        self.won = bool(won)
        self._clear_animation()

    def _clear_animation(self) -> None:
        self._animation.clear()
        self._animation_kind = None
        self._animation_phase = -1
        self._animation_plate = None
        self._old_doors = {}
        self._pending_completion = False

    def _blocked(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        if not (0 <= x < self.width and 0 <= y < self.height) or self.rows[y][x] == "#":
            return True
        door = self.door_at.get(cell)
        return door is not None and not self.door_open[door]

    def _spend_action(self) -> bool:
        self.remaining_steps = max(0, self.remaining_steps - 1)
        return self.remaining_steps == 0

    def _begin_animation(self, kind: str, phases: int, *, plate: str | None = None) -> None:
        self._animation_kind = kind
        self._animation = list(range(phases))
        self._animation_plate = plate
        self._advance_animation()

    def _advance_animation(self) -> None:
        self._animation_phase = self._animation.pop(0)
        if self._animation:
            return
        self._animation_kind = None if self._animation_kind != "completion" else self._animation_kind
        if self._pending_completion:
            self.next_level()
            self.complete_action()
            return
        self.complete_action()

    def _resolve_move(self, action: int) -> None:
        self.undo_history.append(self._snapshot())
        dx, dy = MOVE[action]
        target = (self.player_position[0] + dx, self.player_position[1] + dy)
        if self._blocked(target):
            self.blocked_facing = FACING[action]
            self._begin_animation("blocked", 2)
            return

        self.player_position = target
        self.blocked_facing = "NONE"
        plate = self.plates.get(target)
        if plate is not None:
            self._old_doors = dict(self.door_open)
            for door in self.spec.links[plate]:
                self.door_open[door] = not self.door_open[door]

        picked_up = target == self.artifact_position and self.artifact_present
        if picked_up:
            self.artifact_present = False
            self.has_artifact = True

        if target == self.exit_position and self.has_artifact:
            self.won = True
            self._pending_completion = True
            self._begin_animation("completion", 6)
        elif plate is not None:
            self._begin_animation("plate", 4, plate=plate)
        elif picked_up:
            self._begin_animation("pickup", 3)
        else:
            self.complete_action()

    def step(self) -> None:
        if self._animation:
            self._advance_animation()
            return
        action = _action_id(self.action.id)
        if action == _action_id(GameAction.RESET):
            self.complete_action()
            return
        if self.won:
            self.complete_action()
            return

        expired = self._spend_action()
        if expired:
            self._clear_animation()
            self.lose()
            self.complete_action()
            return

        if action in MOVE:
            self._resolve_move(action)
        elif action == 7:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self.complete_action()
        else:
            self.complete_action()

    def _cell_kind(self, x: int, y: int) -> str:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return "outside"
        return "wall" if self.rows[y][x] == "#" else "floor"

    def _edge_cost(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        u1, v1 = a
        u2, v2 = b
        if v1 == v2:
            y = v1 // 2
            x = min(u1, u2) // 2
            sides = (self._cell_kind(x, y - 1), self._cell_kind(x, y))
        else:
            x = u1 // 2
            y = min(v1, v2) // 2
            sides = (self._cell_kind(x - 1, y), self._cell_kind(x, y))
        if "outside" in sides:
            return 3
        if sides == ("wall", "wall"):
            return 1
        if "wall" in sides:
            return 2
        return 6

    def _neighbors(self, node: tuple[int, int]):
        u, v = node
        for direction, (du, dv) in enumerate(PATH_DIRS):
            nxt = (u + du, v + dv)
            if not (0 <= nxt[0] <= 2 * self.width and 0 <= nxt[1] <= 2 * self.height):
                continue
            if dv == 0 and v % 2 != 0:
                continue
            if du == 0 and u % 2 != 0:
                continue
            yield direction, nxt

    @staticmethod
    def _edge(a: tuple[int, int], b: tuple[int, int]):
        return tuple(sorted((a, b)))

    def _shortest_route(
        self,
        source: tuple[int, int],
        target: tuple[int, int],
        plate: str,
        edge_colors: dict[tuple[tuple[int, int], tuple[int, int]], set[str]],
        vertex_colors: dict[tuple[int, int], set[str]],
    ) -> tuple[int, tuple[int, ...], list[tuple[int, int]]]:
        queue = [(0, (), source, [source])]
        best: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {source: (0, ())}
        while queue:
            cost, directions, node, path = heapq.heappop(queue)
            if node == target:
                return cost, directions, path
            if best.get(node) != (cost, directions):
                continue
            for direction, nxt in self._neighbors(node):
                edge = self._edge(node, nxt)
                colors = edge_colors.get(edge, set())
                extra = 2 if plate in colors else 0
                if colors - {plate}:
                    extra += 20
                if vertex_colors.get(nxt, set()) - {plate}:
                    extra += 6
                next_key = (cost + self._edge_cost(node, nxt) + extra, (*directions, direction))
                if nxt not in best or next_key < best[nxt]:
                    best[nxt] = next_key
                    heapq.heappush(queue, (*next_key, nxt, [*path, nxt]))
        raise RuntimeError("cable route unavailable")

    def _route_cables(self):
        routes: dict[tuple[str, str], list[tuple[int, int]]] = {}
        order: list[tuple[str, str]] = []
        edge_colors: dict[tuple[tuple[int, int], tuple[int, int]], set[str]] = {}
        vertex_colors: dict[tuple[int, int], set[str]] = {}
        plate_cells = {plate: cell for cell, plate in self.plates.items()}
        for plate in ("p", "q", "r"):
            for door in sorted(self.spec.links.get(plate, ())):
                px, py = plate_cells[plate]
                dx, dy = self.doors[door]
                candidates = []
                for source_idx, source_delta in enumerate(PORTS):
                    source = (2 * px + source_delta[0], 2 * py + source_delta[1])
                    for target_idx, target_delta in enumerate(PORTS):
                        target = (2 * dx + target_delta[0], 2 * dy + target_delta[1])
                        cost, directions, path = self._shortest_route(source, target, plate, edge_colors, vertex_colors)
                        candidates.append((cost, source_idx, target_idx, directions, path))
                path = min(candidates)[-1]
                routes[(plate, door)] = path
                order.append((plate, door))
                for a, b in pairwise(path):
                    edge_colors.setdefault(self._edge(a, b), set()).add(plate)
                for vertex in path:
                    vertex_colors.setdefault(vertex, set()).add(plate)
        return routes, order

    def _paint_cell(self, frame: np.ndarray, cell: tuple[int, int], pixels: np.ndarray) -> None:
        x, y = cell
        x0 = self.origin_x + CELL * x
        y0 = self.origin_y + CELL * y
        mask = pixels >= 0
        patch = frame[y0 : y0 + CELL, x0 : x0 + CELL]
        patch[mask] = pixels[mask]

    def _draw_walls(self, frame: np.ndarray) -> None:
        for y, row in enumerate(self.rows):
            for x, char in enumerate(row):
                x0, y0 = self.origin_x + CELL * x, self.origin_y + CELL * y
                frame[y0 : y0 + CELL, x0 : x0 + CELL] = WALL if char == "#" else FLOOR
        for y, row in enumerate(self.rows):
            for x, char in enumerate(row):
                if char != "#":
                    continue
                x0, y0 = self.origin_x + CELL * x, self.origin_y + CELL * y
                if self._cell_kind(x, y - 1) == "floor":
                    frame[y0, x0 : x0 + CELL] = EDGE
                if self._cell_kind(x, y + 1) == "floor":
                    frame[y0 + 3, x0 : x0 + CELL] = EDGE
                if self._cell_kind(x - 1, y) == "floor":
                    frame[y0 : y0 + CELL, x0] = EDGE
                if self._cell_kind(x + 1, y) == "floor":
                    frame[y0 : y0 + CELL, x0 + 3] = EDGE

    def _draw_cables(self, frame: np.ndarray) -> None:
        for plate, door in self._cable_order:
            flashing = self._animation_kind == "plate" and self._animation_plate == plate and self._animation_phase == 0
            color = FLASH_COLORS[plate] if flashing else PLATE_COLORS[plate]
            path = self._cable_routes[(plate, door)]
            points = [(self.origin_x + 2 * u, self.origin_y + 2 * v) for u, v in path]
            for (x1, y1), (x2, y2) in pairwise(points):
                if x1 == x2:
                    frame[min(y1, y2) : max(y1, y2) + 1, x1] = color
                else:
                    frame[y1, min(x1, x2) : max(x1, x2) + 1] = color

    def _draw_plate(self, frame: np.ndarray, cell: tuple[int, int], plate: str) -> None:
        color = PLATE_COLORS[plate]
        pixels = np.array(
            [[color] * 4, [color, EDGE, EDGE, color], [color, EDGE, EDGE, color], [color] * 4], dtype=np.int8
        )
        if self._animation_kind == "plate" and self._animation_plate == plate and self._animation_phase == 0:
            pixels[1:3, 1:3] = FLASH_COLORS[plate]
        self._paint_cell(frame, cell, pixels)

    def _door_visual_state(self, door: str) -> tuple[bool, int]:
        if self._animation_kind != "plate" or door not in self.spec.links.get(self._animation_plate or "", ()):
            return self.door_open[door], 3
        if self._animation_phase == 0:
            return self._old_doors[door], 0
        return self.door_open[door], self._animation_phase

    def _draw_door(self, frame: np.ndarray, door: str, cell: tuple[int, int]) -> None:
        opened, phase = self._door_visual_state(door)
        pixels = np.full((4, 4), FLOOR, dtype=np.int8)
        horizontal = self.door_axes[door] == "horizontal"
        if opened:
            if horizontal:
                pixels[0, :] = JAMB
                pixels[3, :] = JAMB
            else:
                pixels[:, 0] = JAMB
                pixels[:, 3] = JAMB
            if phase in (1, 2):
                if horizontal:
                    if phase == 1:
                        pixels[1:3, 1:3] = DOOR
                    else:
                        pixels[0, 1:3] = DOOR
                        pixels[3, 1:3] = DOOR
                else:
                    if phase == 1:
                        pixels[1:3, 1:3] = DOOR
                    else:
                        pixels[1:3, 0] = DOOR
                        pixels[1:3, 3] = DOOR
        else:
            if horizontal:
                pixels[:, 1:3] = DOOR
                pixels[1:3, 2] = LATCH
                if phase in (1, 2):
                    pixels[phase : 4 - phase, 1:3] = FLOOR
            else:
                pixels[1:3, :] = DOOR
                pixels[2, 1:3] = LATCH
                if phase in (1, 2):
                    pixels[1:3, phase : 4 - phase] = FLOOR
        controllers = [p for p in ("p", "q", "r") if door in self.spec.links.get(p, ())]
        for idx, plate in enumerate(controllers):
            if horizontal:
                pixels[0 if idx % 2 == 0 else 3, min(3, 1 + idx)] = PLATE_COLORS[plate]
            else:
                pixels[min(3, 1 + idx), 0 if idx % 2 == 0 else 3] = PLATE_COLORS[plate]
        self._paint_cell(frame, cell, pixels)

    def _draw_exit(self, frame: np.ndarray) -> None:
        pixels = np.array(
            [
                [-1, PURPLE, PURPLE, -1],
                [PURPLE, WALL, WALL, PURPLE],
                [PURPLE, WALL, WALL, PURPLE],
                [-1, PURPLE, PURPLE, -1],
            ],
            dtype=np.int8,
        )
        if self._animation_kind == "completion":
            phase = self._animation_phase
            if phase >= 3:
                pixels[1:3, 1:3] = PURPLE
            if phase >= 4:
                pixels[0, :] = PLAYER
                pixels[3, :] = PLAYER
            if phase >= 5:
                pixels[:, 0] = PLAYER
                pixels[:, 3] = PLAYER
        self._paint_cell(frame, self.exit_position, pixels)

    def _draw_completion_load(self, frame: np.ndarray) -> None:
        x, y = self.exit_position
        x0 = self.origin_x + CELL * x
        y0 = self.origin_y + CELL * y
        if self._animation_phase == 1:
            frame[y0, x0 + 3] = PURPLE
            frame[y0 + 1, x0 + 2] = PURPLE
        elif self._animation_phase == 2:
            frame[y0 + 1 : y0 + 3, x0 + 1 : x0 + 3] = GEM

    def _draw_artifact(self, frame: np.ndarray) -> None:
        if not self.artifact_present and self._animation_kind != "pickup":
            return
        pixels = np.array(
            [
                [-1, PURPLE, PURPLE, -1],
                [PURPLE, GEM, GEM, PURPLE],
                [PURPLE, GEM, GEM, PURPLE],
                [-1, PURPLE, PURPLE, -1],
            ],
            dtype=np.int8,
        )
        if self._animation_kind == "pickup" and self._animation_phase == 1:
            pixels = np.roll(pixels, -1, axis=0)
            pixels[3, :] = -1
        self._paint_cell(frame, self.artifact_position, pixels)

    def _draw_player(self, frame: np.ndarray) -> None:
        pixels = np.array(
            [[-1, PLAYER, PLAYER, -1], [PLAYER] * 4, [PLAYER] * 4, [-1, PLAYER, PLAYER, -1]], dtype=np.int8
        )
        carried_visible = self.has_artifact and not (self._animation_kind == "pickup" and self._animation_phase < 2)
        if self._animation_kind == "completion" and self._animation_phase >= 1:
            carried_visible = False
        if carried_visible:
            pixels[0, 3] = PURPLE
            pixels[1, 3] = PURPLE
        dx = dy = 0
        if self._animation_kind == "blocked" and self._animation_phase == 0:
            dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[self.blocked_facing]
        x, y = self.player_position
        x0, y0 = self.origin_x + CELL * x + dx, self.origin_y + CELL * y + dy
        mask = pixels >= 0
        patch = frame[y0 : y0 + CELL, x0 : x0 + CELL]
        if patch.shape == pixels.shape:
            patch[mask] = pixels[mask]
        if self.blocked_facing != "NONE" and not (self._animation_kind == "blocked" and self._animation_phase == 0):
            if self.blocked_facing == "UP":
                frame[y0, x0 + 1 : x0 + 3] = LATCH
            elif self.blocked_facing == "DOWN":
                frame[y0 + 3, x0 + 1 : x0 + 3] = LATCH
            elif self.blocked_facing == "LEFT":
                frame[y0 + 1 : y0 + 3, x0] = LATCH
            else:
                frame[y0 + 1 : y0 + 3, x0 + 3] = LATCH

    def _render_board(self) -> np.ndarray:
        frame = self._static_frame.copy()
        if self._animation_kind == "plate" and self._animation_phase == 0:
            self._draw_cables(frame)
        for cell, plate in self.plates.items():
            self._draw_plate(frame, cell, plate)
        for door, cell in self.doors.items():
            self._draw_door(frame, door, cell)
        self._draw_exit(frame)
        self._draw_artifact(frame)
        self._draw_player(frame)
        if self._animation_kind == "completion":
            if self._animation_phase in (1, 2):
                self._draw_completion_load(frame)
            elif self._animation_phase >= 3:
                self._draw_exit(frame)
        filled = int(VIEW * self.remaining_steps / max(1, self.step_budget))
        frame[63, :] = LATCH
        if filled:
            frame[63, :filled] = 12
        return frame
