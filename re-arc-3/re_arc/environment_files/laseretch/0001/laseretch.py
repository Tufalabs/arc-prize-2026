from __future__ import annotations

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

W = 24
H = 18
BAR_LEN = 22

C_FLOOR = 0
C_FLOOR_LIT = 1
C_WALL = 2
C_DANGER = 3
C_TIME_EMPTY = 4
C_TIME_FILLED = 5
C_NEUTRAL = 6
C_EMITTER = 7
C_DEVICE = 8
C_TARGET = 9
C_PORT = 10
C_BEAM = 11
C_BEAM_PWR = 12
C_BEAM_HEAD = 13
C_SUCCESS = 14
C_SPARK = 15

ACTION_WAIT = int(GameAction.ACTION4.value)
ACTION_SPACE = int(GameAction.ACTION5.value)
ACTION_CLICK = int(GameAction.ACTION6.value)

DIRS8 = {(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)}

PORT_OFFSETS = {"up": (1, 0), "down": (1, 2), "left": (0, 1), "right": (2, 1)}


class Segment:
    def __init__(
        self,
        seg_id: int,
        a: tuple[int, int],
        b: tuple[int, int],
        points: list[tuple[int, int]],
        direction: tuple[int, int],
        progress_a: int = 0,
        progress_b: int = 0,
    ):
        self.seg_id = seg_id
        self.a = a
        self.b = b
        self.points = points
        self.direction = direction
        self.progress_a = progress_a
        self.progress_b = progress_b


class TargetState:
    def __init__(self, key: str, top_left: tuple[int, int], port_dir: str, charge: int = 0, required: bool = True):
        self.key = key
        self.top_left = top_left
        self.port_dir = port_dir
        self.charge = charge
        self.required = required

    @property
    def port(self) -> tuple[int, int]:
        return _port_cell(self.top_left, self.port_dir)


class ReflectorState:
    def __init__(
        self,
        key: str,
        top_left: tuple[int, int],
        ports: list[str],
        rotatable: bool = False,
        turn_timer: int = 0,
        orientation_cycle: list[list[str]] | None = None,
        orientation_index: int = 0,
    ):
        self.key = key
        self.top_left = top_left
        self.ports = ports
        self.rotatable = rotatable
        self.turn_timer = turn_timer
        self.orientation_cycle = orientation_cycle or []
        self.orientation_index = orientation_index


class PrismState:
    def __init__(self, key: str, top_left: tuple[int, int], ports: list[str]):
        self.key = key
        self.top_left = top_left
        self.ports = ports


class SwitchState:
    def __init__(
        self, key: str, top_left: tuple[int, int], port_dir: str, door_key: str, was_powered_prev: bool = False
    ):
        self.key = key
        self.top_left = top_left
        self.port_dir = port_dir
        self.door_key = door_key
        self.was_powered_prev = was_powered_prev


class DoorState:
    def __init__(self, key: str, cells: list[tuple[int, int]], state: str = "closed", opening_timer: int = 0):
        self.key = key
        self.cells = cells
        self.state = state
        self.opening_timer = opening_timer


class HazardState:
    def __init__(self, key: str, top_left: tuple[int, int], port_dir: str, flash_steps: int = 0):
        self.key = key
        self.top_left = top_left
        self.port_dir = port_dir
        self.flash_steps = flash_steps


class EmitterState:
    def __init__(self, key: str, top_left: tuple[int, int], port_dir: str):
        self.key = key
        self.top_left = top_left
        self.port_dir = port_dir


class ShutterState:
    def __init__(self, cell: tuple[int, int]):
        self.cell = cell


class LevelSpec:
    def __init__(
        self,
        name: str,
        time_steps_max: int,
        walls: list[tuple[int, int]],
        emitters: list[EmitterState],
        reflectors: list[ReflectorState],
        prisms: list[PrismState],
        targets: list[TargetState],
        switches: list[SwitchState],
        doors: list[DoorState],
        hazards: list[HazardState],
        shutters: list[ShutterState],
        solution: list[dict[str, object]],
    ):
        self.name = name
        self.time_steps_max = time_steps_max
        self.walls = walls
        self.emitters = emitters
        self.reflectors = reflectors
        self.prisms = prisms
        self.targets = targets
        self.switches = switches
        self.doors = doors
        self.hazards = hazards
        self.shutters = shutters
        self.solution = solution


def _points_on_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dx == 0 and dy == 0:
        return []
    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
    if not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
        return []
    if (step_x, step_y) not in DIRS8:
        return []

    dist = max(abs(dx), abs(dy))
    out = []
    for i in range(dist + 1):
        out.append((a[0] + step_x * i, a[1] + step_y * i))
    return out


def _port_cell(top_left: tuple[int, int], direction: str) -> tuple[int, int]:
    ox, oy = PORT_OFFSETS[direction]
    return top_left[0] + ox, top_left[1] + oy


def _device_cells(top_left: tuple[int, int]) -> list[tuple[int, int]]:
    tx, ty = top_left
    return [(tx + dx, ty + dy) for dy in range(3) for dx in range(3)]


def _build_level_specs() -> list[LevelSpec]:
    return [
        LevelSpec(
            name="Laser Etch 1",
            time_steps_max=90,
            walls=[(x, 0) for x in range(W)]
            + [(0, y) for y in range(H)]
            + [(W - 1, y) for y in range(H)]
            + [(10, 4), (11, 4), (12, 4)],
            emitters=[EmitterState("e1", (2, 7), "right")],
            reflectors=[],
            prisms=[],
            targets=[TargetState("t1", (18, 7), "left")],
            switches=[],
            doors=[],
            hazards=[],
            shutters=[],
            solution=[
                {"type": "click_port", "pos": (4, 8)},
                {"type": "click_port", "pos": (18, 8)},
                {"type": "wait", "n": 30},
            ],
        ),
        LevelSpec(
            name="Laser Etch 2",
            time_steps_max=110,
            walls=[(x, 0) for x in range(W)]
            + [(0, y) for y in range(H)]
            + [(W - 1, y) for y in range(H)]
            + [(x, 6) for x in range(9, 14)],
            emitters=[EmitterState("e1", (2, 9), "right")],
            reflectors=[ReflectorState("r1", (13, 9), ["left", "up"], rotatable=False)],
            prisms=[],
            targets=[TargetState("t1", (13, 3), "down")],
            switches=[],
            doors=[],
            hazards=[],
            shutters=[],
            solution=[
                {"type": "click_port", "pos": (4, 10)},
                {"type": "click_port", "pos": (13, 10)},
                {"type": "click_port", "pos": (14, 9)},
                {"type": "click_port", "pos": (14, 5)},
                {"type": "wait", "n": 42},
            ],
        ),
        LevelSpec(
            name="Laser Etch 3",
            time_steps_max=130,
            walls=[(x, 0) for x in range(W)] + [(0, y) for y in range(H)] + [(W - 1, y) for y in range(H)],
            emitters=[EmitterState("e1", (2, 11), "right")],
            reflectors=[
                ReflectorState("r1", (8, 11), ["left", "up"], rotatable=False),
                ReflectorState("r2", (8, 6), ["down", "right"], rotatable=False),
            ],
            prisms=[],
            targets=[TargetState("t1", (15, 6), "left")],
            switches=[],
            doors=[],
            hazards=[HazardState("h1", (8, 2), "down")],
            shutters=[],
            solution=[
                {"type": "click_port", "pos": (4, 12)},
                {"type": "click_port", "pos": (8, 12)},
                {"type": "click_port", "pos": (9, 11)},
                {"type": "click_port", "pos": (9, 8)},
                {"type": "click_port", "pos": (10, 7)},
                {"type": "click_port", "pos": (15, 7)},
                {"type": "wait", "n": 46},
            ],
        ),
        LevelSpec(
            name="Laser Etch 4",
            time_steps_max=140,
            walls=[(x, 0) for x in range(W)] + [(0, y) for y in range(H)] + [(W - 1, y) for y in range(H)],
            emitters=[EmitterState("e1", (2, 8), "right")],
            reflectors=[
                ReflectorState(
                    "r1",
                    (13, 8),
                    ["left", "down"],
                    rotatable=True,
                    orientation_cycle=[["left", "down"], ["left", "up"]],
                    orientation_index=0,
                )
            ],
            prisms=[],
            targets=[TargetState("t1", (13, 3), "down")],
            switches=[],
            doors=[],
            hazards=[],
            shutters=[],
            solution=[
                {"type": "click_port", "pos": (4, 9)},
                {"type": "click_port", "pos": (13, 9)},
                {"type": "click_center", "pos": (14, 9)},
                {"type": "wait", "n": 2},
                {"type": "click_port", "pos": (14, 8)},
                {"type": "click_port", "pos": (14, 5)},
                {"type": "wait", "n": 46},
            ],
        ),
        LevelSpec(
            name="Laser Etch 5",
            time_steps_max=160,
            walls=[(x, 0) for x in range(W)] + [(0, y) for y in range(H)] + [(W - 1, y) for y in range(H)],
            emitters=[EmitterState("e1", (2, 9), "right")],
            reflectors=[],
            prisms=[PrismState("p1", (11, 9), ["left", "up", "right", "down"])],
            targets=[TargetState("tr", (18, 9), "left"), TargetState("tb", (11, 14), "up")],
            switches=[SwitchState("s1", (11, 3), "down", "d1")],
            doors=[DoorState("d1", [(12, 12), (13, 12), (14, 12)], "closed", 0)],
            hazards=[],
            shutters=[],
            solution=[
                {"type": "click_port", "pos": (4, 10)},
                {"type": "click_port", "pos": (11, 10)},
                {"type": "click_port", "pos": (13, 10)},
                {"type": "click_port", "pos": (18, 10)},
                {"type": "click_port", "pos": (12, 9)},
                {"type": "click_port", "pos": (12, 5)},
                {"type": "wait", "n": 22},
                {"type": "click_port", "pos": (12, 11)},
                {"type": "click_port", "pos": (12, 14)},
                {"type": "wait", "n": 52},
            ],
        ),
        LevelSpec(
            name="Laser Etch 6",
            time_steps_max=200,
            walls=[(x, 0) for x in range(W)]
            + [(0, y) for y in range(H)]
            + [(W - 1, y) for y in range(H)]
            + [(x, 14) for x in range(11, 15)],
            emitters=[EmitterState("e1", (2, 6), "right"), EmitterState("e2", (2, 11), "right")],
            reflectors=[
                ReflectorState(
                    "r1",
                    (10, 6),
                    ["left", "down"],
                    rotatable=True,
                    orientation_cycle=[["left", "down"], ["left", "up"]],
                    orientation_index=0,
                )
            ],
            prisms=[PrismState("p1", (15, 11), ["left", "up", "right", "down"])],
            targets=[
                TargetState("t1", (10, 2), "down"),
                TargetState("t2", (15, 7), "down"),
                TargetState("t3", (16, 14), "up"),
            ],
            switches=[SwitchState("s1", (8, 2), "down", "d1")],
            doors=[DoorState("d1", [(12, 9), (12, 10), (12, 11), (12, 12)], "closed", 0)],
            hazards=[HazardState("h1", (18, 10), "up")],
            shutters=[ShutterState((10, 5))],
            solution=[
                {"type": "click_port", "pos": (4, 7)},
                {"type": "click_port", "pos": (10, 7)},
                {"type": "click_center", "pos": (11, 7)},
                {"type": "wait", "n": 2},
                {"type": "click_port", "pos": (11, 6)},
                {"type": "click_port", "pos": (9, 4)},
                {"type": "wait", "n": 16},
                {"type": "space"},
                {"type": "wait", "n": 1},
                {"type": "click_port", "pos": (4, 7)},
                {"type": "click_port", "pos": (10, 7)},
                {"type": "click_port", "pos": (11, 6)},
                {"type": "click_port", "pos": (11, 4)},
                {"type": "click_port", "pos": (4, 12)},
                {"type": "click_port", "pos": (15, 12)},
                {"type": "click_port", "pos": (16, 11)},
                {"type": "click_port", "pos": (16, 9)},
                {"type": "click_port", "pos": (16, 13)},
                {"type": "click_port", "pos": (17, 14)},
                {"type": "wait", "n": 74},
            ],
        ),
    ]


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(
        pixels=np.zeros((H, W), dtype=np.int8), name="board", x=0, y=0, layer=1, tags=["board"], collidable=False
    )
    return Level(
        name=spec.name,
        grid_size=(W, H),
        sprites=[board],
        data={"time_steps_max": int(spec.time_steps_max), "solution": list(spec.solution)},
    )


class Laseretch(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._specs = _build_level_specs()
        levels = [_build_level(spec) for spec in self._specs]
        super().__init__(
            game_id="laseretch",
            levels=levels,
            camera=Camera(width=W, height=H, background=C_FLOOR),
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5, 6],
            seed=seed,
        )
        self._route_score = 0
        self._step_idx = 0
        self._time_max = 0
        self._time_left = 0
        self._steps_per_cell = 1
        self._selected_port: tuple[int, int] | None = None

        self._pending_segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self._pending_clear = False
        self._pending_rotate_keys: list[str] = []

        self._segments: list[Segment] = []
        self._next_segment_id = 1

        self._emitters: dict[str, EmitterState] = {}
        self._reflectors: dict[str, ReflectorState] = {}
        self._prisms: dict[str, PrismState] = {}
        self._targets: dict[str, TargetState] = {}
        self._switches: dict[str, SwitchState] = {}
        self._doors: dict[str, DoorState] = {}
        self._hazards: dict[str, HazardState] = {}
        self._shutters: list[ShutterState] = []

        self._board: Sprite | None = None
        self._walls: set[tuple[int, int]] = set()
        self._device_body_cells: set[tuple[int, int]] = set()
        self._port_map: dict[tuple[int, int], tuple[str, str]] = {}
        self._rot_center_map: dict[tuple[int, int], str] = {}
        self._target_top_left: dict[str, tuple[int, int]] = {}

        self._powered_ports_source: set[tuple[int, int]] = set()
        self._scheduled_power_ports: set[tuple[int, int]] = set()
        self._pulse_front_on = False

        self._win_streak = 0
        self._hazard_failure_active = False

    def on_set_level(self, level: Level) -> None:
        idx = 0
        for i, spec in enumerate(self._specs):
            if spec.name == level.name:
                idx = i
                break
        spec = self._specs[idx]

        self._step_idx = 0
        self._time_max = int(level.get_data("time_steps_max") or spec.time_steps_max)
        self._time_left = self._time_max
        self._steps_per_cell = max(1, (self._time_max + BAR_LEN - 1) // BAR_LEN)
        self._selected_port = None
        self._pending_segments = []
        self._pending_clear = False
        self._pending_rotate_keys = []
        self._segments = []
        self._next_segment_id = 1
        self._powered_ports_source = set()
        self._scheduled_power_ports = set()
        self._win_streak = 0
        self._hazard_failure_active = False
        self._pulse_front_on = False

        board_sprites = self.current_level.get_sprites_by_name("board")
        if not board_sprites:
            raise RuntimeError("laseretch level missing board sprite")
        self._board = board_sprites[0]

        self._emitters = {e.key: EmitterState(e.key, e.top_left, e.port_dir) for e in spec.emitters}
        self._reflectors = {}
        for r in spec.reflectors:
            self._reflectors[r.key] = ReflectorState(
                key=r.key,
                top_left=r.top_left,
                ports=list(r.ports),
                rotatable=bool(r.rotatable),
                turn_timer=int(r.turn_timer),
                orientation_cycle=[list(entry) for entry in r.orientation_cycle],
                orientation_index=int(r.orientation_index),
            )
        self._prisms = {p.key: PrismState(p.key, p.top_left, list(p.ports)) for p in spec.prisms}
        self._targets = {
            t.key: TargetState(t.key, t.top_left, t.port_dir, charge=0, required=bool(t.required)) for t in spec.targets
        }
        self._switches = {
            s.key: SwitchState(s.key, s.top_left, s.port_dir, s.door_key, was_powered_prev=False) for s in spec.switches
        }
        self._doors = {
            d.key: DoorState(d.key, list(d.cells), state=d.state, opening_timer=int(d.opening_timer))
            for d in spec.doors
        }
        self._hazards = {h.key: HazardState(h.key, h.top_left, h.port_dir, flash_steps=0) for h in spec.hazards}
        self._shutters = [ShutterState(s.cell) for s in spec.shutters]

        self._walls = set(spec.walls)
        self._target_top_left = {t.key: t.top_left for t in self._targets.values()}
        self._rebuild_device_maps()

        self._render()

    def _register_3x3_device(self, kind: str, key: str, top_left: tuple[int, int], ports: list[str]) -> None:
        for cell in _device_cells(top_left):
            self._device_body_cells.add(cell)
        for direction in ports:
            port = _port_cell(top_left, direction)
            self._port_map[port] = (kind, key)

    def _rebuild_device_maps(self) -> None:
        self._device_body_cells = set()
        self._port_map = {}
        self._rot_center_map = {}
        for emitter in self._emitters.values():
            self._register_3x3_device("emitter", emitter.key, emitter.top_left, [emitter.port_dir])
        for refl in self._reflectors.values():
            self._register_3x3_device("reflector", refl.key, refl.top_left, refl.ports)
            if refl.rotatable:
                cx, cy = refl.top_left[0] + 1, refl.top_left[1] + 1
                self._rot_center_map[(cx, cy)] = refl.key
        for prism in self._prisms.values():
            self._register_3x3_device("prism", prism.key, prism.top_left, prism.ports)
        for target in self._targets.values():
            self._register_3x3_device("target", target.key, target.top_left, [target.port_dir])
        for switch in self._switches.values():
            self._register_3x3_device("switch", switch.key, switch.top_left, [switch.port_dir])
        for hazard in self._hazards.values():
            self._register_3x3_device("hazard", hazard.key, hazard.top_left, [hazard.port_dir])

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < W and 0 <= y < H

    def _is_shutter_open(self, cell: tuple[int, int]) -> bool:
        for shutter in self._shutters:
            if shutter.cell == cell:
                return (self._step_idx % 2) == 1
        return True

    def _is_blocking_cell(self, cell: tuple[int, int], *, for_etch: bool) -> bool:
        x, y = cell
        if x <= 0 or x >= W - 1 or y <= 1 or y >= H - 1:
            return True
        if cell in self._walls:
            return True

        for door in self._doors.values():
            if cell in door.cells and door.state != "open":
                return True

        for shutter in self._shutters:
            if shutter.cell == cell and not self._is_shutter_open(cell):
                return True

        if cell in self._device_body_cells:
            return True

        if not for_etch:
            return False

        return False

    def _segment_overlaps(self, points: list[tuple[int, int]]) -> bool:
        check_cells = set(points[1:-1])
        if not check_cells:
            return False
        for seg in self._segments:
            existing = set(seg.points[1:-1])
            if check_cells & existing:
                return True
        return False

    def _can_etch(self, a: tuple[int, int], b: tuple[int, int]) -> tuple[bool, list[tuple[int, int]], tuple[int, int]]:
        if a == b:
            return False, [], (0, 0)
        if a not in self._port_map or b not in self._port_map:
            return False, [], (0, 0)

        points = _points_on_line(a, b)
        if len(points) < 2:
            return False, [], (0, 0)

        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        if (dx, dy) not in DIRS8:
            return False, [], (0, 0)

        for cell in points[1:-1]:
            if self._is_blocking_cell(cell, for_etch=True):
                return False, [], (0, 0)

        if self._segment_overlaps(points):
            return False, [], (0, 0)

        return True, points, (dx, dy)

    def _apply_pending_edits(self) -> None:
        if self._pending_clear:
            self._segments = []
            self._pending_clear = False

        for refl_key in self._pending_rotate_keys:
            refl = self._reflectors.get(refl_key)
            if refl is None or not refl.rotatable or refl.turn_timer > 0:
                continue
            refl.turn_timer = 2

        self._pending_rotate_keys = []

        for a, b in self._pending_segments:
            ok, points, direction = self._can_etch(a, b)
            if not ok:
                continue
            seg = Segment(
                seg_id=self._next_segment_id, a=a, b=b, points=points, direction=direction, progress_a=0, progress_b=0
            )
            self._segments.append(seg)
            self._next_segment_id += 1
        self._pending_segments = []

    def _update_timebar(self) -> None:
        self._time_left = max(0, self._time_left - 1)

    def _port_for_emitter(self, emitter: EmitterState) -> tuple[int, int]:
        return _port_cell(emitter.top_left, emitter.port_dir)

    def _ports_for_reflector(self, refl: ReflectorState) -> list[tuple[int, int]]:
        return [_port_cell(refl.top_left, d) for d in refl.ports]

    def _ports_for_prism(self, prism: PrismState) -> list[tuple[int, int]]:
        return [_port_cell(prism.top_left, d) for d in prism.ports]

    def _port_for_switch(self, switch: SwitchState) -> tuple[int, int]:
        return _port_cell(switch.top_left, switch.port_dir)

    def _port_for_hazard(self, hazard: HazardState) -> tuple[int, int]:
        return _port_cell(hazard.top_left, hazard.port_dir)

    def _segment_blocks_forward(self, seg: Segment, from_a: bool) -> bool:
        if len(seg.points) <= 1:
            return True
        idx = seg.progress_a + 1 if from_a else len(seg.points) - 1 - seg.progress_b
        if idx <= 0 or idx >= len(seg.points):
            return False
        cell = seg.points[idx]
        if from_a and cell == seg.b:
            return False
        if (not from_a) and cell == seg.a:
            return False
        return self._is_blocking_cell(cell, for_etch=False)

    def _advance_power(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
        source_ports = set(self._scheduled_power_ports)
        for emitter in self._emitters.values():
            source_ports.add(self._port_for_emitter(emitter))

        arrivals: set[tuple[int, int]] = set()
        front_cells: set[tuple[int, int]] = set()
        powered_cells: set[tuple[int, int]] = set()

        for seg in self._segments:
            total_edges = len(seg.points) - 1
            if total_edges <= 0:
                continue

            a_powered = seg.a in source_ports
            b_powered = seg.b in source_ports

            if a_powered and not self._segment_blocks_forward(seg, from_a=True):
                seg.progress_a = min(total_edges, seg.progress_a + 1)
            if b_powered and not self._segment_blocks_forward(seg, from_a=False):
                seg.progress_b = min(total_edges, seg.progress_b + 1)

            if seg.progress_a >= total_edges:
                arrivals.add(seg.b)
            if seg.progress_b >= total_edges:
                arrivals.add(seg.a)

            interior_n = max(0, len(seg.points) - 2)
            if interior_n <= 0:
                continue

            pa = max(0, min(interior_n, seg.progress_a))
            pb = max(0, min(interior_n, seg.progress_b))

            for i in range(pa):
                powered_cells.add(seg.points[1 + i])
            for i in range(pb):
                powered_cells.add(seg.points[len(seg.points) - 2 - i])

            front_idx_a = pa
            if 0 <= front_idx_a < interior_n:
                front_cells.add(seg.points[1 + front_idx_a])

            front_idx_b = pb
            if 0 <= front_idx_b < interior_n:
                front_cells.add(seg.points[len(seg.points) - 2 - front_idx_b])

        powered_ports = source_ports | arrivals
        return powered_ports, powered_cells, front_cells

    def _resolve_devices(self, powered_ports: set[tuple[int, int]]) -> None:
        next_scheduled: set[tuple[int, int]] = set()

        for refl in self._reflectors.values():
            if refl.turn_timer > 0:
                continue
            ports = self._ports_for_reflector(refl)
            if len(ports) != 2:
                continue
            a, b = ports
            if a in powered_ports:
                next_scheduled.add(b)
            if b in powered_ports:
                next_scheduled.add(a)

        for prism in self._prisms.values():
            ports = self._ports_for_prism(prism)
            active_inputs = [p for p in ports if p in powered_ports]
            if not active_inputs:
                continue
            for inp in active_inputs:
                for outp in ports:
                    if outp != inp:
                        next_scheduled.add(outp)

        for switch in self._switches.values():
            port = self._port_for_switch(switch)
            is_powered = port in powered_ports
            if is_powered and not switch.was_powered_prev:
                door = self._doors.get(switch.door_key)
                if door is not None:
                    if door.state == "closed":
                        door.state = "opening"
                        door.opening_timer = 1
                    elif door.state == "open":
                        door.state = "closed"
                        door.opening_timer = 0
            switch.was_powered_prev = is_powered

        for door in self._doors.values():
            if door.state == "opening":
                if door.opening_timer > 0:
                    door.opening_timer -= 1
                else:
                    door.state = "open"

        for hazard in self._hazards.values():
            port = self._port_for_hazard(hazard)
            if port in powered_ports and not self._hazard_failure_active:
                self._hazard_failure_active = True
                hazard.flash_steps = 2

        for target in self._targets.values():
            if target.port in powered_ports:
                target.charge = min(3, target.charge + 1)
            else:
                target.charge = max(0, target.charge - 1)

        self._scheduled_power_ports = next_scheduled

        for refl in self._reflectors.values():
            if refl.turn_timer > 0:
                refl.turn_timer -= 1
                if refl.turn_timer == 0 and refl.rotatable and refl.orientation_cycle:
                    refl.orientation_index = (refl.orientation_index + 1) % len(refl.orientation_cycle)
                    refl.ports = list(refl.orientation_cycle[refl.orientation_index])
                    self._rebuild_device_maps()

    def _all_targets_charged(self) -> bool:
        required = [t for t in self._targets.values() if t.required]
        if not required:
            return False
        return all(t.charge >= 3 for t in required)

    def _process_action_for_next_step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        if action_id == ACTION_SPACE:
            self._pending_clear = True
            self._selected_port = None
            return

        if action_id != ACTION_CLICK:
            return

        data = self.action.data or {}
        display_x = int(data.get("x", -1))
        display_y = int(data.get("y", -1))
        pos = self.camera.display_to_grid(display_x, display_y)
        if pos is None:
            return
        gx, gy = int(pos[0]), int(pos[1])
        cell = (gx, gy)

        rot_key = self._rot_center_map.get(cell)
        if rot_key is not None:
            self._pending_rotate_keys.append(rot_key)
            self._selected_port = None
            return

        if cell not in self._port_map:
            return

        if self._selected_port is None:
            self._selected_port = cell
            return

        start = self._selected_port
        self._selected_port = None
        self._pending_segments.append((start, cell))

    def _render(self) -> None:
        if self._board is None:
            return

        grid = np.full((H, W), C_FLOOR, dtype=np.int8)

        for x in range(W):
            grid[0, x] = C_WALL
        for y in range(H):
            grid[y, 0] = C_WALL
            grid[y, W - 1] = C_WALL

        for x in range(1, W - 1):
            grid[1, x] = C_TIME_EMPTY

        fill_cells = int((self._time_left + self._steps_per_cell - 1) // self._steps_per_cell)
        fill_cells = max(0, min(BAR_LEN, fill_cells))
        danger_blink = fill_cells <= max(1, BAR_LEN // 5) and (self._step_idx % 2 == 0)
        for i in range(fill_cells):
            grid[1, 1 + i] = C_DANGER if danger_blink else C_TIME_FILLED

        for x, y in self._walls:
            if 0 <= x < W and 0 <= y < H:
                grid[y, x] = C_WALL

        for door in self._doors.values():
            color = C_WALL
            if door.state == "opening":
                color = C_NEUTRAL
            if door.state == "open":
                color = C_FLOOR
            for x, y in door.cells:
                if 0 <= x < W and 0 <= y < H:
                    grid[y, x] = color

        for shutter in self._shutters:
            x, y = shutter.cell
            if 0 <= x < W and 0 <= y < H:
                grid[y, x] = C_FLOOR if self._is_shutter_open(shutter.cell) else C_WALL

        for emitter in self._emitters.values():
            for x, y in _device_cells(emitter.top_left):
                grid[y, x] = C_EMITTER

        for refl in self._reflectors.values():
            color = C_DEVICE
            if refl.turn_timer == 1:
                color = C_SPARK
            for x, y in _device_cells(refl.top_left):
                grid[y, x] = color

        for prism in self._prisms.values():
            for x, y in _device_cells(prism.top_left):
                grid[y, x] = C_DEVICE

        # Targets are rendered as explicit 3x3 blocks from level specs.
        for target in self._targets.values():
            top_left = target.top_left
            for x, y in _device_cells(top_left):
                if 0 <= x < W and 0 <= y < H:
                    grid[y, x] = C_TARGET

        for switch in self._switches.values():
            for x, y in _device_cells(switch.top_left):
                grid[y, x] = C_NEUTRAL

        for hazard in self._hazards.values():
            flash = self._hazard_failure_active and (hazard.flash_steps % 2 == 0)
            color = C_SPARK if flash else C_DANGER
            for x, y in _device_cells(hazard.top_left):
                grid[y, x] = color

        for seg in self._segments:
            for cell in seg.points[1:-1]:
                x, y = cell
                grid[y, x] = C_BEAM

        powered_ports, powered_cells, front_cells = self._advance_power_preview()

        for x, y in powered_cells:
            if 0 <= x < W and 0 <= y < H:
                grid[y, x] = C_BEAM_PWR

        front_color = C_SPARK if self._pulse_front_on else C_BEAM_HEAD
        for x, y in front_cells:
            if 0 <= x < W and 0 <= y < H:
                grid[y, x] = front_color

        for port_cell, (_kind, _key) in self._port_map.items():
            x, y = port_cell
            if 0 <= x < W and 0 <= y < H:
                grid[y, x] = C_BEAM_PWR if port_cell in powered_ports else C_PORT

        if self._selected_port is not None:
            sx, sy = self._selected_port
            if 0 <= sx < W and 0 <= sy < H:
                grid[sy, sx] = C_SPARK

        for target in self._targets.values():
            tx, ty = target.port[0], target.port[1]
            cx, cy = tx, ty
            if target.charge >= 3:
                grid[cy, cx] = C_SPARK if (self._step_idx % 2 == 0) else C_SUCCESS
            else:
                grid[cy, cx] = C_NEUTRAL

        self._board.pixels = grid

    def _advance_power_preview(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
        # Non-mutating approximation for rendering the current frame.
        source_ports = set(self._scheduled_power_ports)
        for emitter in self._emitters.values():
            source_ports.add(self._port_for_emitter(emitter))

        arrivals: set[tuple[int, int]] = set()
        powered_cells: set[tuple[int, int]] = set()
        front_cells: set[tuple[int, int]] = set()

        for seg in self._segments:
            total_edges = len(seg.points) - 1
            if total_edges <= 0:
                continue

            pa = seg.progress_a
            pb = seg.progress_b
            if seg.a in source_ports and not self._segment_blocks_forward(seg, from_a=True):
                pa = min(total_edges, pa + 1)
            if seg.b in source_ports and not self._segment_blocks_forward(seg, from_a=False):
                pb = min(total_edges, pb + 1)

            if pa >= total_edges:
                arrivals.add(seg.b)
            if pb >= total_edges:
                arrivals.add(seg.a)

            interior_n = max(0, len(seg.points) - 2)
            if interior_n <= 0:
                continue

            show_pa = max(0, min(interior_n, pa))
            show_pb = max(0, min(interior_n, pb))

            for i in range(show_pa):
                powered_cells.add(seg.points[1 + i])
            for i in range(show_pb):
                powered_cells.add(seg.points[len(seg.points) - 2 - i])

            if show_pa < interior_n:
                front_cells.add(seg.points[1 + show_pa])
            if show_pb < interior_n:
                front_cells.add(seg.points[len(seg.points) - 2 - show_pb])

        return source_ports | arrivals, powered_cells, front_cells

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

        self._apply_pending_edits()

        self._update_timebar()
        self._pulse_front_on = not self._pulse_front_on

        powered_ports, _, _ = self._advance_power()
        self._resolve_devices(powered_ports)

        all_charged = self._all_targets_charged()
        self._win_streak = self._win_streak + 1 if all_charged else 0

        if self._hazard_failure_active:
            active = False
            for hazard in self._hazards.values():
                if hazard.flash_steps > 0:
                    hazard.flash_steps -= 1
                    active = True
            if not active:
                self.lose()
                self.complete_action()
                return

        if self._time_left <= 0:
            self.lose()
            self.complete_action()
            return

        if self._win_streak >= 2:
            self._route_score += 1
            self.next_level()
            self.complete_action()
            return

        self._process_action_for_next_step()
        self._step_idx += 1
        self._render()
        self.complete_action()


GAME_CLASS = Laseretch
