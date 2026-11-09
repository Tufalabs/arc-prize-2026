from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

VIEW = 64
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 1, 2, 3, 4
ACTION_SPACE, ACTION_CLICK, ACTION_UNDO = 5, 6, 7
MOVES = {ACTION_UP: (-1, 0), ACTION_DOWN: (1, 0), ACTION_LEFT: (0, -1), ACTION_RIGHT: (0, 1)}

OUTSIDE, FLOOR, WALL = 5, 1, 3
WALL_LIGHT, WALL_DARK = 2, 4
WHITE, PLAYER, CORE, ACTIVE = 0, 14, 6, 7
COLOR_PIXELS = {"BLUE": (9, 10), "YELLOW": (11, 12), "RED": (8, 13)}


@dataclass(frozen=True)
class ModuleSpec:
    socket_id: str
    color: str
    socket: tuple[int, int]
    door_id: str
    door: tuple[int, int]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    fuse_colors: tuple[tuple[str, str, tuple[int, int]], ...]
    modules: tuple[ModuleSpec, ...]
    baseline: int


LEVEL_SPECS = (
    LevelSpec(
        "First Burn",
        ("#########", "#...#...#", "#...+C.E#", "#.PFs...#", "#...#...#", "#...#...#", "#########"),
        (("B1", "BLUE", (3, 3)),),
        (ModuleSpec("M1", "BLUE", (3, 4), "D1", (2, 4)),),
        6,
    ),
    LevelSpec(
        "Wrong Mouth",
        ("#########", "#....#..#", "#....+CE#", "#....s..#", "#..PFs..#", "#....+..#", "#########"),
        (("B1", "BLUE", (4, 4)),),
        (ModuleSpec("M1", "BLUE", (3, 5), "D1", (2, 5)), ModuleSpec("M2", "YELLOW", (4, 5), "D2", (5, 5))),
        10,
    ),
    LevelSpec(
        "Keep the Way Back",
        ("#########", "#...#...#", "#...+C..#", "#.PFs...#", "##......#", "#E+s#...#", "#########"),
        (("B1", "BLUE", (3, 3)),),
        (ModuleSpec("M1", "BLUE", (3, 4), "D1", (2, 4)), ModuleSpec("M2", "BLUE", (5, 3), "D2", (5, 2))),
        15,
    ),
    LevelSpec(
        "The Second Push Chooses",
        (
            "###########",
            "######....#",
            "######.F..#",
            "#####s+####",
            "#.......#.#",
            "#......#+.#",
            "#...PF..sC#",
            "#......##.#",
            "##.#s.....#",
            "###E+.....#",
            "###########",
        ),
        (("B1", "BLUE", (6, 5)), ("Y1", "YELLOW", (2, 7))),
        (
            ModuleSpec("M1", "BLUE", (3, 5), "D1", (3, 6)),
            ModuleSpec("M2", "BLUE", (6, 8), "D2", (5, 8)),
            ModuleSpec("M3", "YELLOW", (8, 4), "D3", (9, 4)),
        ),
        40,
    ),
    LevelSpec(
        "A Door to Stand In",
        (
            "#############",
            "#.........C.#",
            "#...........#",
            "#...........#",
            "#...........#",
            "#.#+s#s+#####",
            "#......F#...#",
            "#..PF...#.F.#",
            "#.......s...#",
            "#.......+...#",
            "#.......#.s##",
            "#.......#.+E#",
            "#############",
        ),
        (("B1", "BLUE", (7, 4)), ("Y1", "YELLOW", (6, 7)), ("B2", "BLUE", (7, 10))),
        (
            ModuleSpec("M1", "BLUE", (5, 4), "D1", (5, 3)),
            ModuleSpec("M2", "BLUE", (5, 6), "D2", (5, 7)),
            ModuleSpec("M3", "YELLOW", (8, 8), "D3", (9, 8)),
            ModuleSpec("M4", "BLUE", (10, 10), "D4", (11, 10)),
        ),
        44,
    ),
    LevelSpec(
        "Clear, Park, Reverse",
        (
            "###############",
            "#......#......#",
            "#...C..#......#",
            "#......#......#",
            "#......#......#",
            "#.##+s####s+#.#",
            "#......#...F..#",
            "#...PF.s......#",
            "#......+..F...#",
            "#......#......#",
            "###########s+##",
            "########......#",
            "#########s.F..#",
            "########E+....#",
            "###############",
        ),
        (("Y1", "YELLOW", (7, 5)), ("R1", "RED", (6, 11)), ("B1", "BLUE", (8, 10)), ("B2", "BLUE", (12, 11))),
        (
            ModuleSpec("M1", "YELLOW", (5, 5), "D1", (5, 4)),
            ModuleSpec("M2", "YELLOW", (7, 7), "D2", (8, 7)),
            ModuleSpec("M3", "BLUE", (5, 10), "D3", (5, 11)),
            ModuleSpec("M4", "RED", (10, 11), "D4", (10, 12)),
            ModuleSpec("M5", "BLUE", (12, 9), "D5", (13, 9)),
        ),
        56,
    ),
    LevelSpec(
        "No Spare Shortcut",
        (
            "###############",
            "#......#......#",
            "#...C..#......#",
            "#.##+s##......#",
            "#......+....F.#",
            "#...PF.s#+s##.#",
            "#......#.F....#",
            "#......##s+####",
            "#......#......#",
            "########......#",
            "#......#......#",
            "#......s.F....#",
            "##s....+..F...#",
            "#E+....#......#",
            "###############",
        ),
        (
            ("Y1", "YELLOW", (5, 5)),
            ("B1", "BLUE", (4, 12)),
            ("R1", "RED", (6, 9)),
            ("Y2", "YELLOW", (11, 9)),
            ("B2", "BLUE", (12, 10)),
        ),
        (
            ModuleSpec("M1", "YELLOW", (3, 5), "D1", (3, 4)),
            ModuleSpec("M2", "YELLOW", (5, 7), "D2", (4, 7)),
            ModuleSpec("M3", "BLUE", (5, 10), "D3", (5, 9)),
            ModuleSpec("M4", "RED", (7, 9), "D4", (7, 10)),
            ModuleSpec("M5", "YELLOW", (11, 7), "D5", (12, 7)),
            ModuleSpec("M6", "BLUE", (12, 2), "D6", (13, 2)),
        ),
        61,
    ),
)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _build_level(spec: LevelSpec) -> Level:
    if any(len(row) != len(spec.rows[0]) for row in spec.rows):
        raise ValueError(f"Ragged layout in {spec.name}")
    sprites = [
        Sprite(
            np.full((VIEW, VIEW), OUTSIDE, dtype=np.int8),
            name="board",
            x=0,
            y=0,
            layer=0,
            collidable=False,
            tags=["board"],
        )
    ]
    return Level(sprites=sprites, grid_size=(VIEW, VIEW), data={"spec": spec}, name=spec.name)


class FuseDoors(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "fuse_doors-0001",
            levels,
            Camera(0, 0, VIEW, VIEW, OUTSIDE, OUTSIDE),
            False,
            len(levels),
            [1, 2, 3, 4, 5, 6, 7],
            seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.board_sprite = self.current_level.get_sprites_by_tag("board")[0]
        self.height = len(self.spec.rows)
        self.width = len(self.spec.rows[0])
        self.cell_size = 60 // max(self.height, self.width)
        self.origin_x = (VIEW - self.width * self.cell_size) // 2
        self.origin_y = (VIEW - self.height * self.cell_size) // 2

        self.walls: set[tuple[int, int]] = set()
        self.player_pos: tuple[int, int] | None = None
        self.core_pos: tuple[int, int] | None = None
        self.exit_pos: tuple[int, int] | None = None
        for row, cells in enumerate(self.spec.rows):
            for col, symbol in enumerate(cells):
                if symbol == "#":
                    self.walls.add((row, col))
                elif symbol == "P":
                    self.player_pos = (row, col)
                elif symbol == "C":
                    self.core_pos = (row, col)
                elif symbol == "E":
                    self.exit_pos = (row, col)
        if self.player_pos is None or self.core_pos is None or self.exit_pos is None:
            raise ValueError(f"{self.spec.name} lacks P, C, or E")

        self.fuses = {fuse_id: {"color": color, "pos": tuple(pos)} for fuse_id, color, pos in self.spec.fuse_colors}
        self.sockets = {
            module.socket_id: {
                "color": module.color,
                "pos": module.socket,
                "used": False,
                "linked_door_id": module.door_id,
            }
            for module in self.spec.modules
        }
        self.doors = {
            module.door_id: {"pos": module.door, "open": False, "color": module.color} for module in self.spec.modules
        }
        self.socket_by_pos = {data["pos"]: sid for sid, data in self.sockets.items()}
        self.door_by_pos = {data["pos"]: did for did, data in self.doors.items()}
        self.carrying_core = False
        self.core_present = True
        self.step_count = 0
        self.step_budget = self.spec.baseline * 6
        self.won = False
        self.undo_history: list[dict[str, Any]] = []
        self._animation_frames: list[dict[str, Any]] = []
        self._visual: dict[str, Any] = {}
        self._pending_success = False
        self._pending_loss = False
        self._sync_frame()

    def _snapshot(self) -> dict[str, Any]:
        return {
            "player_pos": self.player_pos,
            "carrying_core": self.carrying_core,
            "core_present": self.core_present,
            "fuses": {fid: dict(data) for fid, data in self.fuses.items()},
            "sockets": {sid: dict(data) for sid, data in self.sockets.items()},
            "doors": {did: dict(data) for did, data in self.doors.items()},
            "won": self.won,
        }

    def _restore(self, snap: dict[str, Any]) -> None:
        self.player_pos = snap["player_pos"]
        self.carrying_core = snap["carrying_core"]
        self.core_present = snap["core_present"]
        self.fuses = {fid: dict(data) for fid, data in snap["fuses"].items()}
        self.sockets = {sid: dict(data) for sid, data in snap["sockets"].items()}
        self.doors = {did: dict(data) for did, data in snap["doors"].items()}
        self.won = snap["won"]

    def _fuse_at(self, pos: tuple[int, int]) -> str | None:
        for fuse_id, data in self.fuses.items():
            if data["pos"] == pos:
                return fuse_id
        return None

    def _player_walkable(self, pos: tuple[int, int]) -> bool:
        if pos in self.walls or pos in self.socket_by_pos or self._fuse_at(pos) is not None:
            return False
        door_id = self.door_by_pos.get(pos)
        return door_id is None or bool(self.doors[door_id]["open"])

    def _fuse_floor_free(self, pos: tuple[int, int]) -> bool:
        if pos in self.walls or pos in self.socket_by_pos or pos in (self.core_pos, self.exit_pos):
            return False
        if self._fuse_at(pos) is not None:
            return False
        door_id = self.door_by_pos.get(pos)
        return door_id is None or bool(self.doors[door_id]["open"])

    def _resolve_move(self, delta: tuple[int, int]) -> list[dict[str, Any]]:
        assert self.player_pos is not None
        old_player = self.player_pos
        dr, dc = delta
        target = (old_player[0] + dr, old_player[1] + dc)
        fuse_id = self._fuse_at(target)
        frames: list[dict[str, Any]] = []

        if fuse_id is not None:
            fuse = self.fuses[fuse_id]
            beyond = (target[0] + dr, target[1] + dc)
            socket_id = self.socket_by_pos.get(beyond)
            if socket_id is not None and not self.sockets[socket_id]["used"]:
                socket = self.sockets[socket_id]
                if fuse["color"] != socket["color"]:
                    return [{"rim_flash": socket_id}, {"fuse_shift": (fuse_id, dr, dc, 1), "rim_flash": socket_id}, {}]

                before = self._snapshot()
                color = str(fuse["color"])
                del self.fuses[fuse_id]
                socket["used"] = True
                door_id = str(socket["linked_door_id"])
                self.doors[door_id]["open"] = True
                self.player_pos = target
                self.undo_history.append(before)
                frames = [
                    {
                        "player_override": old_player,
                        "ghost_fuse": (color, target, dr, dc, 0.5, "normal"),
                        "socket_unused": socket_id,
                        "door_closed": door_id,
                        "rim_flash": socket_id,
                    },
                    {
                        "player_offset": (-dr * self.cell_size // 2, -dc * self.cell_size // 2),
                        "ghost_fuse": (color, beyond, 0, 0, 0.0, "small"),
                        "socket_unused": socket_id,
                        "door_closed": door_id,
                    },
                    {"door_half": door_id, "conduit_pulse": socket_id},
                    {},
                ]
            elif self._fuse_floor_free(beyond):
                before = self._snapshot()
                color = str(fuse["color"])
                fuse["pos"] = beyond
                self.player_pos = target
                self.undo_history.append(before)
                frames = [
                    {
                        "player_override": old_player,
                        "hide_fuse": fuse_id,
                        "ghost_fuse": (color, target, dr, dc, 0.5, "normal"),
                    },
                    {},
                ]
            else:
                frames = [{"fuse_shift": (fuse_id, dr, dc, 1), "blocked": target}, {}]
        elif self._player_walkable(target):
            before = self._snapshot()
            self.player_pos = target
            if self.core_present and self.player_pos == self.core_pos:
                self.core_present = False
                self.carrying_core = True
                frames = [{"ghost_core": True}, {}]
            if self.carrying_core and self.player_pos == self.exit_pos:
                self.won = True
                frames = [{"exit_pulse": 1}, {"exit_pulse": 2}, {"exit_pulse": 3}]
            self.undo_history.append(before)
        else:
            frames = [{"player_shake": (dr, dc), "blocked": target}, {}]
        return frames

    def _begin_frames(self, frames: list[dict[str, Any]]) -> None:
        if frames:
            self._visual = frames[0]
            self._animation_frames = frames[1:]
        else:
            self._visual = {}
            self._animation_frames = []
        self._sync_frame()
        if not self._animation_frames:
            self._finish_action()

    def _finish_action(self) -> None:
        self._visual = {}
        self._sync_frame()
        if self._pending_success:
            self._pending_success = False
            self.next_level()
            self.complete_action()
        elif self._pending_loss:
            self._pending_loss = False
            self.lose()
            self.complete_action()
        else:
            self.complete_action()

    def _play_animation_frame(self) -> None:
        self._visual = self._animation_frames.pop(0)
        self._sync_frame()
        if not self._animation_frames:
            if self._pending_success:
                self._pending_success = False
                self.next_level()
                self.complete_action()
            elif self._pending_loss:
                self._pending_loss = False
                self.lose()
                self.complete_action()
            else:
                self.complete_action()

    def step(self) -> None:
        if self._animation_frames:
            self._play_animation_frame()
            return

        action = _action_id(self.action.id)
        if action == 0:
            self._visual = {}
            self._sync_frame()
            self.complete_action()
            return

        self.step_count += 1
        frames: list[dict[str, Any]] = []
        if action in MOVES:
            frames = self._resolve_move(MOVES[action])
        elif action == ACTION_UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            frames = [{}]
        # SPACE and CLICK deliberately consume a step and do nothing.

        self._pending_success = self.won
        self._pending_loss = self.step_count >= self.step_budget and not self.won
        self._begin_frames(frames)

    def _cell_box(self, pos: tuple[int, int]) -> tuple[int, int, int, int]:
        row, col = pos
        x0 = self.origin_x + col * self.cell_size
        y0 = self.origin_y + row * self.cell_size
        return x0, y0, x0 + self.cell_size, y0 + self.cell_size

    def _draw_wall(self, frame: np.ndarray, pos: tuple[int, int]) -> None:
        x0, y0, x1, y1 = self._cell_box(pos)
        frame[y0:y1, x0:x1] = WALL
        row, col = pos
        if (row - 1, col) not in self.walls:
            frame[y0, x0:x1] = WALL_LIGHT
        if (row, col - 1) not in self.walls:
            frame[y0:y1, x0] = WALL_LIGHT
        if (row + 1, col) not in self.walls:
            frame[y1 - 1, x0:x1] = WALL_DARK
        if (row, col + 1) not in self.walls:
            frame[y0:y1, x1 - 1] = WALL_DARK

    def _draw_fuse(
        self,
        frame: np.ndarray,
        color: str,
        pos: tuple[int, int],
        pixel_offset: tuple[int, int] = (0, 0),
        small: bool = False,
        tint: int | None = None,
    ) -> None:
        x0, y0, x1, y1 = self._cell_box(pos)
        oy, ox = pixel_offset
        x0 += ox
        x1 += ox
        y0 += oy
        y1 += oy
        body, highlight = COLOR_PIXELS[color]
        if tint is not None:
            body = highlight = tint
        cx, cy = (x0 + x1 - 1) // 2, (y0 + y1 - 1) // 2
        if small:
            frame[cy, cx] = body
            return
        if color == "BLUE":
            frame[cy, max(x0, cx - 1) : min(x1, cx + 2)] = body
            frame[max(y0, cy - 1) : min(y1, cy + 2), cx] = body
            frame[max(y0, cy - 1), cx] = highlight
        elif color == "YELLOW":
            left = max(x0, cx - 1)
            frame[max(y0, cy - 1) : min(y1, cy + 2), left : min(x1, left + 2)] = body
            frame[max(y0, cy - 1), left : min(x1, left + 2)] = highlight
        else:
            frame[cy, cx] = body
            if cy - 1 >= y0:
                frame[cy - 1, cx] = body
            if cy + 1 < y1:
                frame[cy + 1, cx] = highlight
            if cx - 1 >= x0:
                frame[cy, cx - 1] = body
            if cx + 1 < x1:
                frame[cy, cx + 1] = body

    def _draw_socket(self, frame: np.ndarray, socket_id: str, used: bool, flash: bool) -> None:
        socket = self.sockets[socket_id]
        x0, y0, x1, y1 = self._cell_box(socket["pos"])
        frame[y0:y1, x0:x1] = WALL
        color = WHITE if flash else COLOR_PIXELS[str(socket["color"])][0]
        cavity = COLOR_PIXELS[str(socket["color"])][1] if used else WALL_DARK
        if self.cell_size >= 5:
            frame[y0 + 1 : y1 - 1, x0 + 1] = color
            frame[y0 + 1 : y1 - 1, x1 - 2] = color
            frame[y1 - 2, x0 + 1 : x1 - 1] = color
            frame[y0 + 2 : y1 - 2, x0 + 2 : x1 - 2] = cavity
        else:
            frame[y0 + 1 : y1 - 1, x0] = color
            frame[y0 + 1 : y1 - 1, x1 - 1] = color
            frame[y1 - 1, x0:x1] = color
            frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = cavity

    def _draw_door(self, frame: np.ndarray, door_id: str, mode: str) -> None:
        door = self.doors[door_id]
        x0, y0, x1, y1 = self._cell_box(door["pos"])
        color = COLOR_PIXELS[str(door["color"])][0]
        horizontal = any(
            module.door_id == door_id and module.door[0] == module.socket[0] for module in self.spec.modules
        )
        if mode == "open":
            frame[y0:y1, x0:x1] = FLOOR
            if horizontal:
                frame[y0:y1, x0] = color
                frame[y0:y1, x1 - 1] = color
            else:
                frame[y0, x0:x1] = color
                frame[y1 - 1, x0:x1] = color
            return
        frame[y0:y1, x0:x1] = WALL
        if horizontal:
            inset = 1 if mode == "half" else max(1, self.cell_size // 2 - 1)
            frame[y0:y1, x0 : min(x1, x0 + inset)] = color
            frame[y0:y1, max(x0, x1 - inset) : x1] = color
        else:
            inset = 1 if mode == "half" else max(1, self.cell_size // 2 - 1)
            frame[y0 : min(y1, y0 + inset), x0:x1] = color
            frame[max(y0, y1 - inset) : y1, x0:x1] = color

    def _draw_conduit(self, frame: np.ndarray, module: ModuleSpec, pulse: bool) -> None:
        sx0, sy0, sx1, sy1 = self._cell_box(module.socket)
        color = WHITE if pulse else COLOR_PIXELS[module.color][0]
        if module.socket[0] == module.door[0]:
            x = sx1 - 1 if module.door[1] > module.socket[1] else sx0
            y = (sy0 + sy1 - 1) // 2
        else:
            x = (sx0 + sx1 - 1) // 2
            y = sy1 - 1 if module.door[0] > module.socket[0] else sy0
        frame[y, x] = color

    def _draw_core(self, frame: np.ndarray, pos: tuple[int, int], center: int = WHITE) -> None:
        x0, y0, x1, y1 = self._cell_box(pos)
        cx, cy = (x0 + x1 - 1) // 2, (y0 + y1 - 1) // 2
        frame[cy, cx] = center
        for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if x0 <= cx + ox < x1 and y0 <= cy + oy < y1:
                frame[cy + oy, cx + ox] = CORE

    def _draw_exit(self, frame: np.ndarray, pulse: int = 0) -> None:
        assert self.exit_pos is not None
        x0, y0, x1, y1 = self._cell_box(self.exit_pos)
        active = self.carrying_core
        rim = ACTIVE if active else CORE
        if pulse % 2:
            rim = WHITE
        frame[y0, x0:x1] = rim
        frame[y1 - 1, x0:x1] = rim
        frame[y0:y1, x0] = rim
        frame[y0:y1, x1 - 1] = rim
        self._draw_core(frame, self.exit_pos, WHITE if active else WALL_DARK)

    def _draw_player(self, frame: np.ndarray, pos: tuple[int, int], pixel_offset: tuple[int, int]) -> None:
        x0, y0, x1, y1 = self._cell_box(pos)
        oy, ox = pixel_offset
        cx = (x0 + x1 - 1) // 2 + ox
        cy = (y0 + y1 - 1) // 2 + oy
        if 0 <= cy < VIEW and 0 <= cx < VIEW:
            frame[cy, max(0, cx - 1) : min(VIEW, cx + 2)] = PLAYER
            frame[max(0, cy - 1) : min(VIEW, cy + 2), cx] = PLAYER
            if cy + 1 < VIEW:
                frame[cy + 1, max(0, cx - 1) : min(VIEW, cx + 2)] = PLAYER
            if cy - 1 >= 0:
                frame[cy - 1, cx] = WHITE
            if self.carrying_core:
                frame[max(0, cy - 2), cx] = CORE
                frame[max(0, cy - 1), cx] = WHITE

    def _draw_push_hints(self, frame: np.ndarray) -> None:
        assert self.player_pos is not None
        for fuse in self.fuses.values():
            row, col = fuse["pos"]
            dr, dc = row - self.player_pos[0], col - self.player_pos[1]
            if abs(dr) + abs(dc) != 1:
                continue
            beyond = (row + dr, col + dc)
            socket_id = self.socket_by_pos.get(beyond)
            can_move = self._fuse_floor_free(beyond)
            cue = ACTIVE if can_move else WALL_LIGHT
            if socket_id is not None and not self.sockets[socket_id]["used"]:
                cue = WHITE if fuse["color"] == self.sockets[socket_id]["color"] else COLOR_PIXELS["RED"][0]

            x0, y0, x1, y1 = self._cell_box((row, col))
            # The near white edge identifies the pushable object; the far
            # colored edge previews whether its destination accepts the push.
            if dr:
                frame[y0 if dr > 0 else y1 - 1, x0:x1] = WHITE
                frame[y1 - 1 if dr > 0 else y0, x0:x1] = cue
            else:
                frame[y0:y1, x0 if dc > 0 else x1 - 1] = WHITE
                frame[y0:y1, x1 - 1 if dc > 0 else x0] = cue

            if socket_id is not None:
                sx0, sy0, sx1, sy1 = self._cell_box(beyond)
                frame[sy0, sx0:sx1] = cue
                frame[sy1 - 1, sx0:sx1] = cue
                frame[sy0:sy1, sx0] = cue
                frame[sy0:sy1, sx1 - 1] = cue

    def _sync_frame(self) -> None:
        frame = np.full((VIEW, VIEW), OUTSIDE, dtype=np.int8)
        visual = self._visual
        for row in range(self.height):
            for col in range(self.width):
                pos = (row, col)
                x0, y0, x1, y1 = self._cell_box(pos)
                if pos in self.walls:
                    self._draw_wall(frame, pos)
                else:
                    frame[y0:y1, x0:x1] = FLOOR

        for door_id, door in self.doors.items():
            if visual.get("door_closed") == door_id:
                mode = "closed"
            elif visual.get("door_half") == door_id:
                mode = "half"
            else:
                mode = "open" if door["open"] else "closed"
            self._draw_door(frame, door_id, mode)

        for socket_id, socket in self.sockets.items():
            used = bool(socket["used"]) and visual.get("socket_unused") != socket_id
            self._draw_socket(frame, socket_id, used, visual.get("rim_flash") == socket_id)

        pulse_socket = visual.get("conduit_pulse")
        for module in self.spec.modules:
            self._draw_conduit(frame, module, pulse_socket == module.socket_id)

        if self.core_present and self.core_pos is not None:
            self._draw_core(frame, self.core_pos)
        if visual.get("ghost_core") and self.core_pos is not None:
            self._draw_core(frame, self.core_pos, CORE)
        self._draw_exit(frame, int(visual.get("exit_pulse", 0)))

        hide_fuse = visual.get("hide_fuse")
        shift = visual.get("fuse_shift")
        for fuse_id, fuse in self.fuses.items():
            if fuse_id == hide_fuse:
                continue
            offset = (0, 0)
            tint = None
            if shift and shift[0] == fuse_id:
                offset = (int(shift[1]) * int(shift[3]), int(shift[2]) * int(shift[3]))
                tint = WALL_LIGHT if visual.get("blocked") else None
            self._draw_fuse(frame, str(fuse["color"]), fuse["pos"], offset, tint=tint)

        ghost = visual.get("ghost_fuse")
        if ghost:
            color, pos, dr, dc, fraction, size = ghost
            offset = (
                round(float(dr) * self.cell_size * float(fraction)),
                round(float(dc) * self.cell_size * float(fraction)),
            )
            self._draw_fuse(frame, color, pos, offset, small=size == "small")

        if not visual:
            self._draw_push_hints(frame)

        assert self.player_pos is not None
        player_pos = visual.get("player_override", self.player_pos)
        player_offset = visual.get("player_offset", (0, 0))
        if visual.get("player_shake"):
            dr, dc = visual["player_shake"]
            player_offset = (dr, dc)
        self._draw_player(frame, player_pos, player_offset)

        # Mandatory derived action bar: 6x the verified solver count.
        frame[62:64, 2:62] = WALL_DARK
        remaining = max(0, self.step_budget - self.step_count)
        filled = int(60 * remaining / max(1, self.step_budget))
        if filled:
            frame[62:64, 2 : 2 + filled] = ACTIVE
        if self.step_count >= self.step_budget:
            frame[62:64, 2:62] = COLOR_PIXELS["RED"][0]
            frame[0, :] = COLOR_PIXELS["RED"][0]
            frame[-1, :] = COLOR_PIXELS["RED"][0]
            frame[:, 0] = COLOR_PIXELS["RED"][0]
            frame[:, -1] = COLOR_PIXELS["RED"][0]
        self.board_sprite.pixels = frame


__all__ = ["FuseDoors"]
