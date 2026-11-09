from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

MODULE_PITCH = 7
MODULE_SIZE = 6
MODULE_X0 = 4
MODULE_Y0 = 8

BACKGROUND = 0
SOCKET_BORDER = 3
SOCKET_FILL = 1
DOMINO_DORMANT = 9
DOMINO_HILITE = 10
DOMINO_ACTIVE = 12
DOMINO_ACTIVE_HILITE = 11
START_DORMANT = 8
START_ACCENT = 13
TARGET_DORMANT = 14
TARGET_ACCENT = 4
JUNCTION_DORMANT = 2
WALL_FILL = 4
PIP_SPENT = 3

SocketState = Literal[0, 1, 2]
Phase = Literal["edit", "resolving_nodes_to_sockets", "resolving_sockets_to_nodes", "win_delay", "game_complete"]
NodeKind = Literal["start", "junction", "target"]


@dataclass(frozen=True)
class NodeSpec:
    row: int
    col: int
    kind: NodeKind


@dataclass(frozen=True)
class LevelSpec:
    name: str
    budget: int
    nodes: tuple[NodeSpec, ...]
    sockets: tuple[tuple[int, int], ...]
    walls: tuple[tuple[int, int], ...]


LEVEL_SPECS: tuple[LevelSpec, ...] = (
    LevelSpec(
        name="Level 1",
        budget=6,
        nodes=(NodeSpec(2, 0, "start"), NodeSpec(2, 2, "junction"), NodeSpec(4, 2, "target")),
        sockets=((2, 1), (3, 2)),
        walls=(),
    ),
    LevelSpec(
        name="Level 2",
        budget=10,
        nodes=(
            NodeSpec(2, 0, "start"),
            NodeSpec(4, 0, "junction"),
            NodeSpec(4, 2, "junction"),
            NodeSpec(4, 4, "junction"),
            NodeSpec(2, 4, "junction"),
            NodeSpec(2, 2, "junction"),
            NodeSpec(0, 2, "junction"),
            NodeSpec(0, 4, "junction"),
            NodeSpec(2, 6, "target"),
        ),
        sockets=((3, 0), (4, 1), (4, 3), (3, 4), (2, 5), (2, 1), (1, 2), (0, 3)),
        walls=((1, 3), (2, 3), (3, 3)),
    ),
    LevelSpec(
        name="Level 3",
        budget=13,
        nodes=(
            NodeSpec(2, 0, "start"),
            NodeSpec(2, 2, "junction"),
            NodeSpec(4, 2, "junction"),
            NodeSpec(4, 4, "junction"),
            NodeSpec(2, 4, "junction"),
            NodeSpec(6, 4, "junction"),
            NodeSpec(0, 2, "junction"),
            NodeSpec(2, 6, "target"),
            NodeSpec(6, 6, "target"),
        ),
        sockets=((2, 1), (3, 2), (4, 3), (3, 4), (2, 5), (5, 4), (6, 5), (1, 2)),
        walls=((4, 5),),
    ),
)


def module_xy(row: int, col: int) -> tuple[int, int]:
    return MODULE_X0 + MODULE_PITCH * col, MODULE_Y0 + MODULE_PITCH * row


def _sprite(name: str, pixels: list[list[int]], x: int, y: int, *, layer: int, tags: list[str] | None = None) -> Sprite:
    return Sprite(
        name=name,
        pixels=np.array(pixels, dtype=np.int8),
        x=x,
        y=y,
        layer=layer,
        visible=True,
        collidable=False,
        tags=(tags or []).copy(),
    )


def empty_socket_pixels() -> np.ndarray:
    return np.array(
        [
            [0, 3, 3, 3, 3, 0],
            [3, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 3],
            [3, 1, 1, 1, 1, 3],
            [0, 3, 3, 3, 3, 0],
        ],
        dtype=np.int8,
    )


def socket_pixels(state: SocketState, active: bool) -> np.ndarray:
    pixels = empty_socket_pixels()
    if state == 0:
        return pixels

    main = DOMINO_ACTIVE if active else DOMINO_DORMANT
    hilite = DOMINO_ACTIVE_HILITE if active else DOMINO_HILITE
    if state == 1:
        pixels[2:4, :] = main
        pixels[2, 1:5] = hilite
    else:
        pixels[:, 2:4] = main
        pixels[1:5, 2] = hilite
    return pixels


def start_pixels(active: bool) -> np.ndarray:
    main = DOMINO_ACTIVE if active else START_DORMANT
    core = DOMINO_ACTIVE_HILITE if active else START_ACCENT
    return np.array(
        [
            [0, 0, main, main, 0, 0],
            [0, 0, main, main, 0, 0],
            [0, main, main, main, main, 0],
            [0, main, core, core, main, 0],
            [0, main, core, core, main, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )


def target_pixels(active: bool) -> np.ndarray:
    center_a = DOMINO_ACTIVE_HILITE if active else TARGET_ACCENT
    center_b = DOMINO_ACTIVE if active else TARGET_ACCENT
    return np.array(
        [
            [0, 0, TARGET_DORMANT, TARGET_DORMANT, 0, 0],
            [0, TARGET_DORMANT, TARGET_DORMANT, TARGET_DORMANT, TARGET_DORMANT, 0],
            [0, TARGET_DORMANT, center_a, center_b, TARGET_DORMANT, 0],
            [0, TARGET_DORMANT, center_b, center_a, TARGET_DORMANT, 0],
            [0, 0, TARGET_DORMANT, TARGET_DORMANT, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )


def junction_pixels(active: bool) -> np.ndarray:
    outer = DOMINO_ACTIVE if active else JUNCTION_DORMANT
    inner = DOMINO_ACTIVE_HILITE if active else TARGET_ACCENT
    return np.array(
        [
            [0, 0, outer, outer, 0, 0],
            [0, outer, inner, inner, outer, 0],
            [0, outer, inner, inner, outer, 0],
            [0, 0, outer, outer, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )


def wall_pixels() -> np.ndarray:
    return np.array(
        [
            [4, 4, 4, 4, 4, 4],
            [4, 3, 3, 3, 3, 4],
            [4, 3, 3, 3, 3, 4],
            [4, 3, 3, 3, 3, 4],
            [4, 3, 3, 3, 3, 4],
            [4, 4, 4, 4, 4, 4],
        ],
        dtype=np.int8,
    )


def pip_pixels(color: int) -> np.ndarray:
    return np.array(
        [[color, color, color], [color, 0, color], [color, 0, color], [color, 0, color], [color, color, color]],
        dtype=np.int8,
    )


def reset_pixels() -> np.ndarray:
    return np.array(
        [
            [13, 13, 13, 13, 13, 13],
            [13, 0, 0, 13, 0, 13],
            [13, 0, 13, 13, 0, 13],
            [13, 0, 13, 0, 0, 13],
            [13, 0, 0, 0, 13, 13],
            [13, 13, 13, 13, 13, 13],
        ],
        dtype=np.int8,
    )


def _build_level(spec: LevelSpec) -> Level:
    sprites: list[Sprite] = []

    for row, col in spec.walls:
        x, y = module_xy(row, col)
        sprites.append(_sprite(f"wall_{row}_{col}", wall_pixels().tolist(), x, y, layer=2))

    for node in spec.nodes:
        x, y = module_xy(node.row, node.col)
        tags = [f"node:{node.kind}", "board"]
        if node.kind == "start":
            tags.extend(["sys_click", "sys_every_pixel", "start"])
            pixels = start_pixels(False)
            layer = 5
        elif node.kind == "target":
            pixels = target_pixels(False)
            layer = 4
        else:
            pixels = junction_pixels(False)
            layer = 4
        sprites.append(_sprite(f"node_{node.row}_{node.col}", pixels.tolist(), x, y, layer=layer, tags=tags))

    for row, col in spec.sockets:
        x, y = module_xy(row, col)
        sprites.append(
            _sprite(
                f"socket_{row}_{col}",
                socket_pixels(0, False).tolist(),
                x,
                y,
                layer=3,
                tags=["socket", "board", "sys_click", "sys_every_pixel"],
            )
        )

    for idx in range(spec.budget):
        x = 4 + idx * 4
        sprites.append(_sprite(f"pip_{idx}", pip_pixels(11).tolist(), x, 1, layer=20, tags=["hud"]))

    sprites.append(
        _sprite(
            "reset_button",
            reset_pixels().tolist(),
            57,
            1,
            layer=20,
            tags=["hud", "sys_click", "sys_every_pixel", "reset"],
        )
    )
    return Level(name=spec.name, sprites=sprites, grid_size=(64, 64), data={"budget": spec.budget})


class DominoSocket(ARCBaseGame):
    def __init__(self, game_id: str = "domino_socket-0001", seed: int = 0):
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            game_id=game_id,
            levels=levels,
            camera=Camera(width=64, height=64, background=BACKGROUND, letter_box=BACKGROUND),
            win_score=len(levels),
            available_actions=[5, 6],
            seed=seed,
        )
        self._phase: Phase = "edit"
        self._remaining_budget = 0
        self._frontier_nodes: set[tuple[int, int]] = set()
        self._frontier_sockets: set[tuple[int, int]] = set()
        self._active_nodes: set[tuple[int, int]] = set()
        self._active_sockets: set[tuple[int, int]] = set()
        self._socket_states: dict[tuple[int, int], SocketState] = {}
        self._node_kinds: dict[tuple[int, int], NodeKind] = {}
        self._socket_sprites: dict[tuple[int, int], Sprite] = {}
        self._node_sprites: dict[tuple[int, int], Sprite] = {}
        self._target_positions: tuple[tuple[int, int], ...] = ()
        self._start_position: tuple[int, int] | None = None
        self._win_delay_steps = 0
        self.on_set_level(self.current_level)

    def on_set_level(self, level: Level) -> None:
        spec = LEVEL_SPECS[self.level_index]
        self._phase = "edit"
        self._remaining_budget = spec.budget
        self._frontier_nodes = set()
        self._frontier_sockets = set()
        self._active_nodes = set()
        self._active_sockets = set()
        self._socket_states = {pos: 0 for pos in spec.sockets}
        self._socket_sprites = {}
        self._node_sprites = {}
        self._node_kinds = {}
        self._target_positions = tuple((node.row, node.col) for node in spec.nodes if node.kind == "target")
        self._start_position = None
        self._win_delay_steps = 0

        for node in spec.nodes:
            pos = (node.row, node.col)
            sprite = level.get_sprites_by_name(f"node_{node.row}_{node.col}")[0]
            self._node_sprites[pos] = sprite
            self._node_kinds[pos] = node.kind
            if node.kind == "start":
                self._start_position = pos

        for pos in spec.sockets:
            sprite = level.get_sprites_by_name(f"socket_{pos[0]}_{pos[1]}")[0]
            self._socket_sprites[pos] = sprite

        self._refresh_art()

    def _is_sprite_clickable_now(self, sprite: Sprite) -> bool:
        if "reset" in sprite.tags:
            return True
        if "socket" in sprite.tags:
            return self._phase == "edit"
        if "start" in sprite.tags:
            return self._phase == "edit"
        return False

    def _refresh_art(self) -> None:
        for pos, sprite in self._socket_sprites.items():
            sprite.pixels = socket_pixels(self._socket_states[pos], pos in self._active_sockets)

        for pos, sprite in self._node_sprites.items():
            kind = self._node_kinds[pos]
            active = pos in self._active_nodes
            if kind == "start":
                sprite.pixels = start_pixels(active)
            elif kind == "target":
                sprite.pixels = target_pixels(active)
            else:
                sprite.pixels = junction_pixels(active)

        pip_sprites = sorted(
            (sprite for sprite in self.current_level.get_sprites_by_tag("hud") if sprite.name.startswith("pip_")),
            key=lambda sprite: int(sprite.name.split("_")[1]),
        )
        for idx, sprite in enumerate(pip_sprites):
            color = 11 if idx < self._remaining_budget else PIP_SPENT
            sprite.pixels = pip_pixels(color)

    def _clicked_sprite(self) -> Sprite | None:
        data = self.action.data or {}
        x = int(data.get("x", -1))
        y = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(x, y)
        if grid_pos is None:
            return None
        return self.current_level.get_sprite_at(grid_pos[0], grid_pos[1], ignore_collidable=True)

    def _spend_budget(self, *, allow_zero: bool = False) -> bool:
        if self._remaining_budget <= 0:
            self.lose()
            self.complete_action()
            return False
        self._remaining_budget -= 1
        self._refresh_art()
        if self._remaining_budget == 0 and self._phase == "edit" and not allow_zero:
            self.lose()
            self.complete_action()
            return False
        return True

    def _socket_orientation_matches(self, socket: tuple[int, int], node: tuple[int, int]) -> bool:
        state = self._socket_states[socket]
        if state == 1:
            return node in {(socket[0], socket[1] - 1), (socket[0], socket[1] + 1)}
        if state == 2:
            return node in {(socket[0] - 1, socket[1]), (socket[0] + 1, socket[1])}
        return False

    def _adjacent_nodes_for_socket(self, socket: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        state = self._socket_states[socket]
        if state == 1:
            return ((socket[0], socket[1] - 1), (socket[0], socket[1] + 1))
        if state == 2:
            return ((socket[0] - 1, socket[1]), (socket[0] + 1, socket[1]))
        return ()

    def _trigger_chain(self) -> None:
        if self._phase != "edit" or self._start_position is None:
            self.complete_action()
            return
        if not self._spend_budget(allow_zero=True):
            return
        self._active_nodes = {self._start_position}
        self._active_sockets = set()
        self._frontier_nodes = {self._start_position}
        self._frontier_sockets = set()
        self._phase = "resolving_nodes_to_sockets"
        self._refresh_art()

    def _resolve_nodes_to_sockets(self) -> None:
        newly_active: set[tuple[int, int]] = set()
        for socket in self._socket_states:
            if socket in self._active_sockets:
                continue
            if self._socket_states[socket] == 0:
                continue
            if any(self._socket_orientation_matches(socket, node) for node in self._frontier_nodes):
                newly_active.add(socket)
        if not newly_active:
            self._end_propagation()
            return
        self._active_sockets.update(newly_active)
        self._frontier_sockets = newly_active
        self._phase = "resolving_sockets_to_nodes"
        self._refresh_art()

    def _resolve_sockets_to_nodes(self) -> None:
        newly_active: set[tuple[int, int]] = set()
        for socket in self._frontier_sockets:
            for node in self._adjacent_nodes_for_socket(socket):
                if node in self._active_nodes:
                    continue
                if node in self._node_sprites:
                    newly_active.add(node)
        if not newly_active:
            self._end_propagation()
            return
        self._active_nodes.update(newly_active)
        self._frontier_nodes = newly_active
        self._phase = "resolving_nodes_to_sockets"
        self._refresh_art()

    def _end_propagation(self) -> None:
        if all(target in self._active_nodes for target in self._target_positions):
            self._phase = "win_delay"
            self._win_delay_steps = 2
        else:
            self.lose()
            self.complete_action()

    def _advance_win_delay(self) -> None:
        self._win_delay_steps -= 1
        if self._win_delay_steps > 0:
            return
        if self.is_last_level():
            self._phase = "game_complete"
            self.next_level()
            self.complete_action()
            return
        self._phase = "edit"
        self.next_level()
        self.complete_action()

    def _restart_level(self) -> None:
        self.level_reset()
        self.complete_action()

    def _handle_click(self) -> None:
        clicked = self._clicked_sprite()
        if clicked is None:
            self.complete_action()
            return

        if "reset" in clicked.tags:
            self._restart_level()
            return

        if "socket" in clicked.tags and self._phase == "edit":
            if not self._spend_budget():
                return
            row, col = map(int, clicked.name.split("_")[1:])
            pos = (row, col)
            self._socket_states[pos] = cast_socket_state((self._socket_states[pos] + 1) % 3)
            self._refresh_art()
            self.complete_action()
            return

        if "start" in clicked.tags:
            if self._phase == "edit":
                self._trigger_chain()
                return

        self.complete_action()

    def step(self) -> None:
        if self._phase == "resolving_nodes_to_sockets":
            self._resolve_nodes_to_sockets()
            return
        if self._phase == "resolving_sockets_to_nodes":
            self._resolve_sockets_to_nodes()
            return
        if self._phase == "win_delay":
            self._advance_win_delay()
            return

        action_id = int(self.action.id.value)
        if self._phase == "game_complete":
            self.complete_action()
            return

        if action_id in {1, 2, 3, 4}:
            self.complete_action()
            return
        if action_id == 5:
            self._trigger_chain()
            return
        if action_id == 6:
            self._handle_click()
            return
        self.complete_action()


def cast_socket_state(value: int) -> SocketState:
    return 0 if value == 0 else 1 if value == 1 else 2
