from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

ACTION_CLICK = 6

VIEW_SIZE = 64
WORK_RIGHT = 50
PANEL_X = 51
LOGICAL_W = 12
LOGICAL_H = 15

COLOR_BG = 0
COLOR_PANEL = 1
COLOR_CAP = 2
COLOR_BLOCKER = 3
COLOR_WALL = 4
COLOR_MARKER = 6
COLOR_TARGET = COLOR_MARKER
COLOR_FLASH = 8
COLOR_BLUE = 9
COLOR_YELLOW = 11
COLOR_ORANGE = 12
COLOR_GREEN = 14
COLOR_PURPLE = 15

DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}
DIR_ORDER = ("N", "E", "S", "W")
REL_TURN = {"straight": 0, "right": 1, "back": 2, "left": -1}
BUTTON_RECTS = {
    (row, col): (53 + col * 6, 2 + row * 6, 57 + col * 6, 6 + row * 6) for row in range(10) for col in range(2)
}


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    parent: str | None
    cap: tuple[int, int] | None
    direction: str | None
    relative: str | None
    length: int
    min_length: int
    max_length: int
    end_kind: str
    length_channel: int | None = None
    move_channel: int | None = None
    rotation_channel: int | None = None


@dataclass
class NodeState:
    node_id: str
    parent: str | None
    cap: tuple[int, int] | None
    direction: str | None
    relative: str | None
    length: int
    min_length: int
    max_length: int
    end_kind: str
    length_channel: int | None = None
    move_channel: int | None = None
    rotation_channel: int | None = None


@dataclass(frozen=True)
class ButtonSpec:
    row: int
    col: int
    op: str
    channel: int
    direction: str | None = None


@dataclass(frozen=True)
class LevelSpec:
    name: str
    targets: tuple[tuple[int, int], ...]
    nodes: tuple[NodeSpec, ...]
    buttons: tuple[ButtonSpec, ...]
    walls: tuple[tuple[str, int, int, int], ...] = ()
    budget: int = 30


def _cell_center(cell: tuple[int, int]) -> tuple[int, int]:
    return 4 + cell[0] * 4, 4 + cell[1] * 4


def _add(a: tuple[int, int], b: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return a[0] + b[0] * scale, a[1] + b[1] * scale


def _rotate_dir(direction: str, clockwise_steps: int) -> str:
    idx = DIR_ORDER.index(direction)
    return DIR_ORDER[(idx + clockwise_steps) % 4]


def _child_dir(parent_dir: str, relative: str) -> str:
    return _rotate_dir(parent_dir, REL_TURN[relative])


def _action_id(action: Any) -> int:
    return int(action.value if isinstance(action, GameAction) else action)


def _node(spec: NodeSpec) -> NodeState:
    return NodeState(**spec.__dict__)


class Sliders(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.canvas = Sprite(
            np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BG, dtype=np.int8),
            name="sliders_canvas",
            layer=0,
            collidable=False,
            tags=["canvas"],
        )
        self.level_spec = LEVEL_SPECS[0]
        self.nodes: dict[str, NodeState] = {}
        self.children: dict[str, list[str]] = {}
        self.world: dict[str, dict[str, Any]] = {}
        self.remaining_steps = 1
        self.step_budget = 1
        self.flash_button: ButtonSpec | None = None
        self.flash_cell: tuple[int, int] | None = None
        levels = [
            Level(sprites=[self.canvas.clone()], grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name)
            for spec in LEVEL_SPECS
        ]
        super().__init__(
            "sliders-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BG, COLOR_BG),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.canvas = level.get_sprites_by_tag("canvas")[0]
        self.level_spec = level.get_data("spec")
        self.nodes = {spec.node_id: _node(spec) for spec in self.level_spec.nodes}
        self.children = {node_id: [] for node_id in self.nodes}
        for node in self.nodes.values():
            if node.parent is not None:
                self.children[node.parent].append(node.node_id)
        self.step_budget = self.level_spec.budget
        self.remaining_steps = self.step_budget
        self.flash_button = None
        self.flash_cell = None
        self._sync_visuals()

    def step(self) -> None:
        if _action_id(self.action.id) == 0:
            self.flash_button = None
            self.flash_cell = None
            self._sync_visuals()
            self.complete_action()
            return

        self.flash_button = None
        self.flash_cell = None
        button = self._clicked_button()
        if _action_id(self.action.id) == ACTION_CLICK and button is not None:
            ok, contact = self._apply_button(button)
            if not ok:
                self.flash_button = button
                self.flash_cell = contact

        self.remaining_steps -= 1
        self._sync_visuals()
        if self._is_solved():
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _clicked_button(self) -> ButtonSpec | None:
        if _action_id(self.action.id) != ACTION_CLICK:
            return None
        x = int(self.action.data.get("x", -1))
        y = int(self.action.data.get("y", -1))
        for button in self.level_spec.buttons:
            x1, y1, x2, y2 = BUTTON_RECTS[(button.row, button.col)]
            if x1 <= x <= x2 and y1 <= y <= y2:
                return button
        return None

    def _apply_button(self, button: ButtonSpec) -> tuple[bool, tuple[int, int] | None]:
        old_nodes = {node_id: replace(node) for node_id, node in self.nodes.items()}
        if button.op == "extend":
            for node in self.nodes.values():
                if node.length_channel == button.channel:
                    node.length += 1
        elif button.op == "retract":
            for node in self.nodes.values():
                if node.length_channel == button.channel:
                    node.length -= 1
        elif button.op == "translate":
            dx, dy = DIRS[button.direction or "N"]
            for node in self.nodes.values():
                if node.parent is None and node.move_channel == button.channel and node.cap is not None:
                    node.cap = (node.cap[0] + dx, node.cap[1] + dy)
        elif button.op == "rotate_cw":
            self._rotate_channel(button.channel, 1)
        elif button.op == "rotate_ccw":
            self._rotate_channel(button.channel, -1)

        valid, contact = self._validate_nodes()
        if not valid:
            self.nodes = old_nodes
            self._compute_world()
            return False, contact
        return True, None

    def _rotate_channel(self, channel: int, steps: int) -> None:
        old_world = self._compute_world()
        for node in self.nodes.values():
            if node.rotation_channel != channel:
                continue
            world_dir = old_world[node.node_id]["dir"]
            new_world_dir = _rotate_dir(world_dir, steps)
            if node.parent is None:
                node.direction = new_world_dir
            else:
                parent_dir = old_world[node.parent]["dir"]
                diff = (DIR_ORDER.index(new_world_dir) - DIR_ORDER.index(parent_dir)) % 4
                node.relative = {0: "straight", 1: "right", 2: "back", 3: "left"}[diff]

    def _compute_world(self) -> dict[str, dict[str, Any]]:
        world: dict[str, dict[str, Any]] = {}

        def visit(node_id: str) -> None:
            node = self.nodes[node_id]
            if node.parent is None:
                cap = node.cap
                direction = node.direction
            else:
                if node.parent not in world:
                    visit(node.parent)
                parent = world[node.parent]
                cap = parent["tip"]
                direction = _child_dir(parent["dir"], node.relative or "straight")
            if cap is None or direction is None:
                raise RuntimeError(f"Malformed slider node {node_id}.")
            tip = _add(cap, DIRS[direction], node.length)
            world[node_id] = {"cap": cap, "dir": direction, "tip": tip}
            for child_id in self.children.get(node_id, []):
                visit(child_id)

        for node_id, node in self.nodes.items():
            if node.parent is None:
                visit(node_id)
        self.world = world
        return world

    def _footprint(self, node_id: str, world: dict[str, dict[str, Any]] | None = None) -> set[tuple[int, int]]:
        world = self.world if world is None else world
        node = self.nodes[node_id]
        cap = world[node_id]["cap"]
        direction = world[node_id]["dir"]
        dx, dy = DIRS[direction]
        return {_add(cap, (dx, dy), step) for step in range(node.length + 1)}

    def _wall_cells(self) -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for kind, a, b, c in self.level_spec.walls:
            if kind == "H":
                cells.update((x, c) for x in range(a, b + 1))
            elif kind == "V":
                cells.update((a, y) for y in range(b, c + 1))
            else:
                cells.add((a, b))
        return cells

    def _validate_nodes(self) -> tuple[bool, tuple[int, int] | None]:
        for node in self.nodes.values():
            if not (node.min_length <= node.length <= node.max_length):
                return False, None
        world = self._compute_world()
        walls = self._wall_cells()
        occupancy: dict[tuple[int, int], list[str]] = {}
        for node_id in self.nodes:
            for cell in self._footprint(node_id, world):
                if not (0 <= cell[0] < LOGICAL_W and 0 <= cell[1] < LOGICAL_H):
                    return False, cell
                if cell in walls:
                    return False, cell
                occupancy.setdefault(cell, []).append(node_id)
        for cell, owners in occupancy.items():
            unique = sorted(set(owners))
            if len(unique) <= 1:
                continue
            if self._authorized_overlap(cell, unique, world):
                continue
            return False, cell
        return True, None

    def _authorized_overlap(self, cell: tuple[int, int], owners: list[str], world: dict[str, dict[str, Any]]) -> bool:
        if len(owners) != 2:
            return False
        a, b = owners
        return (self.nodes[b].parent == a and world[a]["tip"] == cell and world[b]["cap"] == cell) or (
            self.nodes[a].parent == b and world[b]["tip"] == cell and world[a]["cap"] == cell
        )

    def _is_solved(self) -> bool:
        world = self._compute_world()
        markers = {world[node_id]["tip"] for node_id, node in self.nodes.items() if node.end_kind == "marker"}
        return all(target in markers for target in self.level_spec.targets)

    def _sync_visuals(self) -> None:
        frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_BG, dtype=np.int8)
        frame[:, PANEL_X:VIEW_SIZE] = COLOR_PANEL
        frame[:, WORK_RIGHT] = COLOR_WALL
        self._draw_targets(frame)
        self._draw_walls(frame)
        self._compute_world()
        for node_id in self.nodes:
            self._draw_rod(frame, node_id)
        for node_id in self.nodes:
            self._draw_ends(frame, node_id)
        self._draw_step_bar(frame)
        self._draw_buttons(frame)
        if self.flash_cell is not None:
            self._draw_square(frame, self.flash_cell, 1, COLOR_FLASH)
        if self.flash_button is not None:
            x1, y1, x2, y2 = BUTTON_RECTS[(self.flash_button.row, self.flash_button.col)]
            frame[y1 : y2 + 1, x1 : x2 + 1] = COLOR_FLASH
        self.canvas.pixels = frame

    def _draw_targets(self, frame: np.ndarray) -> None:
        for target in self.level_spec.targets:
            self._draw_square(frame, target, 2, COLOR_TARGET)
            self._draw_square(frame, target, 1, COLOR_BG)

    def _draw_walls(self, frame: np.ndarray) -> None:
        for cell in self._wall_cells():
            self._draw_square(frame, cell, 1, COLOR_WALL)

    def _draw_rod(self, frame: np.ndarray, node_id: str) -> None:
        node = self.nodes[node_id]
        color = node.length_channel or node.move_channel or node.rotation_channel or COLOR_BLOCKER
        cap = self.world[node_id]["cap"]
        tip = self.world[node_id]["tip"]
        x1, y1 = _cell_center(cap)
        x2, y2 = _cell_center(tip)
        frame[min(y1, y2) - 1 : max(y1, y2) + 2, min(x1, x2) - 1 : max(x1, x2) + 2] = color

    def _draw_ends(self, frame: np.ndarray, node_id: str) -> None:
        node = self.nodes[node_id]
        cap = self.world[node_id]["cap"]
        tip = self.world[node_id]["tip"]
        cap_color = node.rotation_channel or node.move_channel or COLOR_CAP
        if node.parent is not None:
            parent = self.nodes[node.parent]
            if parent.end_kind == "marker" and self.world[node.parent]["tip"] == cap:
                cap_color = COLOR_MARKER
        self._draw_square(frame, cap, 1, cap_color)
        end_color = {"marker": COLOR_MARKER, "mount": COLOR_CAP, "blocker": COLOR_BLOCKER}[node.end_kind]
        self._draw_square(frame, tip, 1, end_color)

    def _draw_square(self, frame: np.ndarray, cell: tuple[int, int], radius: int, color: int) -> None:
        x, y = _cell_center(cell)
        frame[
            max(0, y - radius) : min(VIEW_SIZE, y + radius + 1), max(0, x - radius) : min(VIEW_SIZE, x + radius + 1)
        ] = color

    def _draw_step_bar(self, frame: np.ndarray) -> None:
        frame[61:63, 52:64] = COLOR_WALL
        filled = max(0, min(12, round(12 * self.remaining_steps / max(1, self.step_budget))))
        if filled:
            frame[61:63, 52 : 52 + filled] = COLOR_GREEN

    def _draw_buttons(self, frame: np.ndarray) -> None:
        for button in self.level_spec.buttons:
            x1, y1, x2, y2 = BUTTON_RECTS[(button.row, button.col)]
            frame[y1 : y2 + 1, x1 : x2 + 1] = COLOR_WALL
            frame[y1 + 1 : y2, x1 + 1 : x2] = button.channel
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if button.op == "extend":
                frame[cy, cx - 1 : cx + 2] = COLOR_BG
                frame[cy - 1 : cy + 2, cx] = COLOR_BG
            elif button.op == "retract":
                frame[cy, cx - 1 : cx + 2] = COLOR_BG
            elif button.op == "translate":
                self._draw_button_arrow(frame, cx, cy, button.direction or "N")
            elif button.op == "rotate_cw":
                frame[cy - 1, cx : cx + 2] = COLOR_BG
                frame[cy, cx + 1] = COLOR_BG
                frame[cy + 1, cx] = COLOR_BG
            elif button.op == "rotate_ccw":
                frame[cy - 1, cx - 1 : cx + 1] = COLOR_BG
                frame[cy, cx - 1] = COLOR_BG
                frame[cy + 1, cx] = COLOR_BG

    def _draw_button_arrow(self, frame: np.ndarray, cx: int, cy: int, direction: str) -> None:
        if direction == "N":
            pts = ((cx, cy - 1), (cx - 1, cy), (cx + 1, cy))
        elif direction == "S":
            pts = ((cx, cy + 1), (cx - 1, cy), (cx + 1, cy))
        elif direction == "W":
            pts = ((cx - 1, cy), (cx, cy - 1), (cx, cy + 1))
        else:
            pts = ((cx + 1, cy), (cx, cy - 1), (cx, cy + 1))
        for x, y in pts:
            frame[y, x] = COLOR_BG


def _b(row: int, col: int, op: str, channel: int, direction: str | None = None) -> ButtonSpec:
    return ButtonSpec(row, col, op, channel, direction)


LEVEL_SPECS = (
    LevelSpec(
        "First cap",
        ((6, 5),),
        (NodeSpec("S", None, (2, 5), "E", None, 1, 1, 4, "marker", length_channel=COLOR_BLUE),),
        (_b(0, 0, "extend", COLOR_BLUE), _b(0, 1, "retract", COLOR_BLUE)),
        budget=18,
    ),
    LevelSpec(
        "Child rides",
        ((6, 5),),
        (
            NodeSpec("P", None, (2, 8), "E", None, 2, 1, 9, "mount", length_channel=COLOR_BLUE),
            NodeSpec("C", "P", None, None, "left", 1, 1, 3, "marker", length_channel=COLOR_GREEN),
        ),
        (
            _b(0, 0, "extend", COLOR_BLUE),
            _b(0, 1, "retract", COLOR_BLUE),
            _b(1, 0, "extend", COLOR_GREEN),
            _b(1, 1, "retract", COLOR_GREEN),
        ),
        budget=24,
    ),
    LevelSpec(
        "Movable wall",
        ((7, 7),),
        (
            NodeSpec("S", None, (2, 7), "E", None, 2, 1, 5, "marker", length_channel=COLOR_BLUE),
            NodeSpec("W", None, (5, 4), "S", None, 3, 3, 3, "blocker", move_channel=COLOR_ORANGE),
        ),
        (
            _b(0, 0, "extend", COLOR_BLUE),
            _b(0, 1, "retract", COLOR_BLUE),
            _b(1, 0, "translate", COLOR_ORANGE, "N"),
            _b(1, 1, "translate", COLOR_ORANGE, "S"),
        ),
        budget=24,
    ),
    LevelSpec(
        "Coupled distances",
        ((7, 4), (11, 10)),
        (
            NodeSpec("A", None, (2, 4), "E", None, 1, 1, 5, "marker", length_channel=COLOR_BLUE),
            NodeSpec(
                "B", None, (7, 10), "E", None, 2, 1, 6, "marker", length_channel=COLOR_BLUE, move_channel=COLOR_ORANGE
            ),
        ),
        (
            _b(0, 0, "extend", COLOR_BLUE),
            _b(0, 1, "retract", COLOR_BLUE),
            _b(1, 0, "translate", COLOR_ORANGE, "W"),
            _b(1, 1, "translate", COLOR_ORANGE, "E"),
        ),
        budget=36,
    ),
    LevelSpec(
        "Wall twice",
        ((7, 6),),
        (
            NodeSpec("P", None, (2, 10), "E", None, 2, 1, 5, "mount", length_channel=COLOR_BLUE),
            NodeSpec("C", "P", None, None, "left", 1, 1, 4, "marker", length_channel=COLOR_GREEN),
            NodeSpec("W", None, (7, 13), "N", None, 3, 3, 3, "blocker", move_channel=COLOR_ORANGE),
        ),
        (
            _b(0, 0, "extend", COLOR_BLUE),
            _b(0, 1, "retract", COLOR_BLUE),
            _b(1, 0, "extend", COLOR_GREEN),
            _b(1, 1, "retract", COLOR_GREEN),
            _b(2, 0, "translate", COLOR_ORANGE, "N"),
            _b(2, 1, "translate", COLOR_ORANGE, "S"),
            _b(3, 0, "translate", COLOR_ORANGE, "W"),
            _b(3, 1, "translate", COLOR_ORANGE, "E"),
        ),
        budget=72,
    ),
    LevelSpec(
        "Retraction clearance",
        ((8, 5), (6, 3)),
        (
            NodeSpec("P", None, (2, 10), "E", None, 2, 2, 2, "mount", move_channel=COLOR_ORANGE),
            NodeSpec("C", "P", None, None, "left", 3, 1, 5, "marker", length_channel=COLOR_GREEN),
            NodeSpec("G2", None, (1, 3), "E", None, 3, 1, 5, "marker", length_channel=COLOR_GREEN),
        ),
        (
            _b(0, 0, "extend", COLOR_GREEN),
            _b(0, 1, "retract", COLOR_GREEN),
            _b(1, 0, "translate", COLOR_ORANGE, "W"),
            _b(1, 1, "translate", COLOR_ORANGE, "E"),
        ),
        walls=(("H", 5, 7, 8),),
        budget=60,
    ),
    LevelSpec(
        "Rotation pocket",
        ((11, 11),),
        (
            NodeSpec(
                "P", None, (3, 8), "E", None, 3, 3, 3, "mount", move_channel=COLOR_YELLOW, rotation_channel=COLOR_PURPLE
            ),
            NodeSpec("C", "P", None, None, "left", 4, 1, 4, "marker", length_channel=COLOR_GREEN),
            NodeSpec("W", None, (9, 14), "N", None, 3, 3, 3, "blocker", move_channel=COLOR_ORANGE),
        ),
        (
            _b(0, 0, "extend", COLOR_GREEN),
            _b(0, 1, "retract", COLOR_GREEN),
            _b(1, 0, "rotate_ccw", COLOR_PURPLE),
            _b(1, 1, "rotate_cw", COLOR_PURPLE),
            _b(2, 0, "translate", COLOR_YELLOW, "W"),
            _b(2, 1, "translate", COLOR_YELLOW, "E"),
            _b(3, 0, "translate", COLOR_ORANGE, "N"),
            _b(3, 1, "translate", COLOR_ORANGE, "S"),
        ),
        walls=(),
        budget=90,
    ),
    LevelSpec(
        "Build shift dismantle",
        ((10, 14), (5, 13)),
        (
            NodeSpec(
                "A", None, (1, 6), "E", None, 3, 3, 3, "mount", move_channel=COLOR_YELLOW, rotation_channel=COLOR_PURPLE
            ),
            NodeSpec("B", "A", None, None, "left", 1, 1, 4, "mount", length_channel=COLOR_BLUE),
            NodeSpec("C", "B", None, None, "right", 4, 1, 5, "marker", length_channel=COLOR_GREEN),
            NodeSpec("D", "C", None, None, "left", 1, 1, 1, "blocker"),
            NodeSpec("G2", None, (0, 13), "E", None, 4, 1, 5, "marker", length_channel=COLOR_GREEN),
            NodeSpec("W", None, (10, 12), "N", None, 1, 1, 1, "blocker", move_channel=COLOR_ORANGE),
        ),
        (
            _b(0, 0, "extend", COLOR_BLUE),
            _b(0, 1, "retract", COLOR_BLUE),
            _b(1, 0, "extend", COLOR_GREEN),
            _b(1, 1, "retract", COLOR_GREEN),
            _b(2, 0, "rotate_ccw", COLOR_PURPLE),
            _b(2, 1, "rotate_cw", COLOR_PURPLE),
            _b(3, 0, "translate", COLOR_YELLOW, "W"),
            _b(3, 1, "translate", COLOR_YELLOW, "E"),
            _b(4, 0, "translate", COLOR_ORANGE, "N"),
            _b(4, 1, "translate", COLOR_ORANGE, "S"),
        ),
        walls=(("H", 2, 3, 9), ("C", 9, 8, 0)),
        budget=156,
    ),
)
