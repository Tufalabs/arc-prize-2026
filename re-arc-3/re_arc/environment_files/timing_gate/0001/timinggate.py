from __future__ import annotations

import math

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "timing_gate-0001"

GRID_WIDTH = 28
GRID_HEIGHT = 18
TIMEBAR_X = 1
TIMEBAR_Y = 0
TIMEBAR_LEN = 26

COLOR_FLOOR = 0
COLOR_TIMEBAR_EMPTY = 1
COLOR_WALL = 2
COLOR_GATE_CLOSED = 3
COLOR_GATE_POST = 4
COLOR_GATE_OPEN = 5
COLOR_GATE_WARNING = 6
COLOR_PLAYER_A = 7
COLOR_PLAYER_B = 8
COLOR_SPIKE_UP = 9
COLOR_EXIT_A = 10
COLOR_EXIT_B = 11
COLOR_TIMEBAR_FILL = 12
COLOR_TIMEBAR_LOW = 13
COLOR_PLATE_ON = 14
COLOR_PLATE_OFF = 15

ACTION_TO_DELTA = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}

LAYOUT_1 = (
    "#tttttttttttttttttttttttttt#",
    "############################",
    "#..........................#",
    "#..P.......................#",
    "#..........................#",
    "#..........................#",
    "#..........................#",
    "#..........................#",
    "#............G.G...........#",
    "##############g#############",
    "#............G.G...........#",
    "#..........................#",
    "#..........................#",
    "#..........................#",
    "#.....................XX...#",
    "#.....................XX...#",
    "#..........................#",
    "############################",
)

LAYOUT_2 = (
    "#tttttttttttttttttttttttttt#",
    "############################",
    "#.............#............#",
    "#..P..........#............#",
    "#.............#............#",
    "#.............#............#",
    "#.............#............#",
    "#.............#............#",
    "#............G#G...........#",
    "#.............g............#",
    "#............G#G...........#",
    "#.............#............#",
    "#.............#............#",
    "#.............#........XX..#",
    "#.............#........XX..#",
    "#.............#............#",
    "#.............#............#",
    "############################",
)

LAYOUT_3 = (
    "#tttttttttttttttttttttttttt#",
    "############################",
    "#........#........#........#",
    "#..P.....#........#........#",
    "#........#........#........#",
    "#........#........#........#",
    "#........#........#........#",
    "#........#........#........#",
    "#.......G#G......G#G.......#",
    "#........g........g........#",
    "#.......G#G......G#G.......#",
    "#........#........#........#",
    "#........#........#........#",
    "#........#........#........#",
    "#........#........#....XX..#",
    "#........#........#....XX..#",
    "#........#........#........#",
    "############################",
)

LAYOUT_4 = (
    "#tttttttttttttttttttttttttt#",
    "############################",
    "############################",
    "############################",
    "############################",
    "############################",
    "############################",
    "############################",
    "#########G#G####G#G####XX###",
    "#....P....g...ss.g.....XX..#",
    "#########G#G####G#G#########",
    "############################",
    "############################",
    "############################",
    "############################",
    "############################",
    "############################",
    "############################",
)

LAYOUT_5 = (
    "#tttttttttttttttttttttttttt#",
    "############################",
    "#........#........#........#",
    "#..P.....#........#........#",
    "#........#........#........#",
    "#....p...#........#........#",
    "#........#........#........#",
    "#........#........#........#",
    "#.......G#G......G#G.......#",
    "#........g........g........#",
    "#.......G#G......G#G.......#",
    "#........#........#........#",
    "#........#........#........#",
    "#........#........#........#",
    "#........#........#....XX..#",
    "#........#........#....XX..#",
    "#........#........#........#",
    "############################",
)

LAYOUT_6 = (
    "#tttttttttttttttttttttttttt#",
    "############################",
    "#........#........##########",
    "#..P.....#........##########",
    "#.......G#G.......##########",
    "#........g........##########",
    "#.......G#G.......##########",
    "#........#........##########",
    "#........#.......G#G########",
    "#........#........g.....####",
    "#........#.......G#G###.####",
    "#........#...p....####s.####",
    "#........#........####G.G###",
    "#........#........#####g####",
    "#........#........####G.G###",
    "#........#........#####.XX.#",
    "#........#........#####.XX.#",
    "############################",
)

LEVEL_SPECS = (
    {
        "name": "Timing Gate 1",
        "max_steps": 120,
        "rows": LAYOUT_1,
        "gates": ({"coord": (14, 9), "mode": "timed", "cycle": (3, 1, 6, 1), "phase": 0, "pulse_steps": 0},),
        "spike": None,
        "plate_target_gate": None,
    },
    {
        "name": "Timing Gate 2",
        "max_steps": 110,
        "rows": LAYOUT_2,
        "gates": ({"coord": (14, 9), "mode": "timed", "cycle": (4, 1, 2, 1), "phase": 0, "pulse_steps": 0},),
        "spike": None,
        "plate_target_gate": None,
    },
    {
        "name": "Timing Gate 3",
        "max_steps": 120,
        "rows": LAYOUT_3,
        "gates": (
            {"coord": (9, 9), "mode": "timed", "cycle": (4, 1, 2, 1), "phase": 0, "pulse_steps": 0},
            {"coord": (18, 9), "mode": "timed", "cycle": (4, 1, 2, 1), "phase": 3, "pulse_steps": 0},
        ),
        "spike": None,
        "plate_target_gate": None,
    },
    {
        "name": "Timing Gate 4",
        "max_steps": 105,
        "rows": LAYOUT_4,
        "gates": (
            {"coord": (10, 9), "mode": "timed", "cycle": (4, 1, 2, 1), "phase": 0, "pulse_steps": 0},
            {"coord": (17, 9), "mode": "timed", "cycle": (5, 1, 2, 1), "phase": 2, "pulse_steps": 0},
        ),
        "spike": {"down": 3, "up": 2, "phase": 0},
        "plate_target_gate": None,
    },
    {
        "name": "Timing Gate 5",
        "max_steps": 115,
        "rows": LAYOUT_5,
        "gates": (
            {"coord": (9, 9), "mode": "plate", "cycle": None, "phase": 0, "pulse_steps": 8},
            {"coord": (18, 9), "mode": "timed", "cycle": (4, 1, 2, 1), "phase": 1, "pulse_steps": 0},
        ),
        "spike": None,
        "plate_target_gate": 0,
    },
    {
        "name": "Timing Gate 6",
        "max_steps": 95,
        "rows": LAYOUT_6,
        "gates": (
            {"coord": (9, 5), "mode": "timed", "cycle": (3, 1, 2, 1), "phase": 0, "pulse_steps": 0},
            {"coord": (18, 9), "mode": "plate", "cycle": None, "phase": 0, "pulse_steps": 7},
            {"coord": (23, 13), "mode": "timed", "cycle": (4, 1, 2, 1), "phase": 2, "pulse_steps": 0},
        ),
        "spike": {"down": 2, "up": 2, "phase": 1},
        "plate_target_gate": 1,
    },
)


def _pixel(color: int) -> np.ndarray:
    return np.array([[int(color)]], dtype=np.int8)


def _gate_cycle_state(cycle: tuple[int, int, int, int], tick: int, phase: int) -> str:
    closed, warning_a, open_steps, warning_b = cycle
    period = max(1, closed + warning_a + open_steps + warning_b)
    slot = int((tick + phase) % period)
    if slot < closed:
        return "closed"
    slot -= closed
    if slot < warning_a:
        return "warning"
    slot -= warning_a
    if slot < open_steps:
        return "open"
    return "warning"


def _gate_color_for_state(state: str, tick: int) -> int:
    blink_on = bool(tick % 2 == 0)
    if state == "open":
        return COLOR_GATE_OPEN if blink_on else COLOR_FLOOR
    if state == "warning":
        return COLOR_GATE_WARNING if blink_on else COLOR_GATE_CLOSED
    return COLOR_GATE_CLOSED


def _spike_is_up(spec: dict, tick: int) -> bool:
    period = max(1, int(spec["down"]) + int(spec["up"]))
    slot = int((tick + int(spec["phase"])) % period)
    return slot >= int(spec["down"])


class TimingGate(ARCBaseGame):
    def __init__(self, seed: int = 0):
        levels = [self._build_level(spec_index, spec) for spec_index, spec in enumerate(LEVEL_SPECS)]
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=Camera(width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_FLOOR),
            win_score=len(levels),
            available_actions=[1, 2, 3, 4, 5],
            seed=seed,
        )
        self._player = None
        self._timebar = None
        self._plate = None
        self._remaining_steps = 0
        self._max_steps = 0
        self._tick = 0
        self._gate_specs = ()
        self._gate_sprites = []
        self._gate_passable_now = []
        self._gate_pulse_remaining = []
        self._spike_spec = None
        self._spike_sprites = []
        self._spike_up_now = False
        self._exit_sprites = []
        self._plate_target_gate = None
        self._pending_outcome = None
        self._transition_ticks = 0

    @staticmethod
    def _build_level(spec_index: int, spec: dict) -> Level:
        rows = tuple(spec["rows"])
        if len(rows) != GRID_HEIGHT:
            raise ValueError(f"{spec['name']}: expected {GRID_HEIGHT} rows.")

        sprites = [
            Sprite(
                pixels=np.full((GRID_HEIGHT, GRID_WIDTH), COLOR_FLOOR, dtype=np.int8),
                name="floor",
                layer=-10,
                tags=["floor", "sys_static"],
                collidable=False,
            )
        ]

        player_pos = None
        gate_centers = []
        exit_tiles = []
        spike_tiles = []
        plate_pos = None

        for y, row in enumerate(rows):
            if len(row) != GRID_WIDTH:
                raise ValueError(f"{spec['name']}: row {y} expected {GRID_WIDTH} columns.")
            for x, ch in enumerate(row):
                if ch == "#":
                    sprites.append(
                        Sprite(
                            pixels=_pixel(COLOR_WALL),
                            name=f"wall-{x}-{y}",
                            x=x,
                            y=y,
                            layer=1,
                            tags=["wall", "blocker"],
                            collidable=True,
                        )
                    )
                elif ch == "G":
                    sprites.append(
                        Sprite(
                            pixels=_pixel(COLOR_GATE_POST),
                            name=f"gate-post-{x}-{y}",
                            x=x,
                            y=y,
                            layer=2,
                            tags=["gate_post", "blocker"],
                            collidable=True,
                        )
                    )
                elif ch == "g":
                    gate_centers.append((x, y))
                elif ch == "s":
                    spike_tiles.append((x, y))
                elif ch == "p":
                    if plate_pos is not None:
                        raise ValueError(f"{spec['name']}: only one plate is supported.")
                    plate_pos = (x, y)
                elif ch == "X":
                    exit_tiles.append((x, y))
                elif ch == "P":
                    if player_pos is not None:
                        raise ValueError(f"{spec['name']}: only one player spawn is supported.")
                    player_pos = (x, y)
                elif ch in {".", "t"}:
                    pass
                else:
                    raise ValueError(f"{spec['name']}: unsupported tile {ch!r}.")

        if player_pos is None:
            raise ValueError(f"{spec['name']}: missing player spawn.")
        if not exit_tiles:
            raise ValueError(f"{spec['name']}: missing exit tiles.")

        gate_specs = tuple(spec["gates"])
        if len(gate_centers) != len(gate_specs):
            raise ValueError(
                f"{spec['name']}: gate center count mismatch layout={len(gate_centers)} spec={len(gate_specs)}"
            )

        gate_centers = sorted(gate_centers)
        for idx, gate_cfg in enumerate(gate_specs):
            gx, gy = gate_centers[idx]
            if (gx, gy) != tuple(gate_cfg["coord"]):
                raise ValueError(
                    f"{spec['name']}: gate index {idx} coord mismatch layout={(gx, gy)} spec={gate_cfg['coord']}"
                )
            sprites.append(
                Sprite(
                    pixels=_pixel(COLOR_GATE_CLOSED),
                    name=f"gate-center-{idx}",
                    x=gx,
                    y=gy,
                    layer=3,
                    tags=["gate_center", f"gate_{idx}", str(gate_cfg["mode"])],
                    collidable=True,
                )
            )

        for idx, (sx, sy) in enumerate(spike_tiles):
            sprites.append(
                Sprite(
                    pixels=_pixel(COLOR_FLOOR),
                    name=f"spike-{idx}",
                    x=sx,
                    y=sy,
                    layer=2,
                    tags=["spike"],
                    collidable=False,
                )
            )

        for idx, (ex, ey) in enumerate(exit_tiles):
            sprites.append(
                Sprite(
                    pixels=_pixel(COLOR_EXIT_A),
                    name=f"exit-{idx}",
                    x=ex,
                    y=ey,
                    layer=2,
                    tags=["exit"],
                    collidable=False,
                )
            )

        if plate_pos is not None:
            sprites.append(
                Sprite(
                    pixels=_pixel(COLOR_PLATE_OFF),
                    name="plate",
                    x=plate_pos[0],
                    y=plate_pos[1],
                    layer=2,
                    tags=["plate"],
                    collidable=False,
                )
            )

        sprites.append(
            Sprite(
                pixels=np.full((1, TIMEBAR_LEN), COLOR_TIMEBAR_FILL, dtype=np.int8),
                name="timebar",
                x=TIMEBAR_X,
                y=TIMEBAR_Y,
                layer=9,
                tags=["hud", "timer"],
                collidable=False,
            )
        )

        sprites.append(
            Sprite(
                pixels=_pixel(COLOR_PLAYER_A),
                name="player",
                x=player_pos[0],
                y=player_pos[1],
                layer=10,
                tags=["player"],
                collidable=True,
            )
        )

        return Level(
            name=spec["name"],
            grid_size=(GRID_WIDTH, GRID_HEIGHT),
            sprites=sprites,
            data={
                "spec_index": int(spec_index),
                "max_steps": int(spec["max_steps"]),
                "spawn": player_pos,
                "exit_tiles": tuple(exit_tiles),
                "spike_tiles": tuple(spike_tiles),
                "plate": plate_pos,
                "gate_specs": gate_specs,
                "spike_spec": spec.get("spike"),
                "plate_target_gate": spec.get("plate_target_gate"),
            },
        )

    def export_level_model(self) -> dict:
        level = self.current_level
        walls = set()
        posts = set()
        exits = set(tuple(cell) for cell in (level.get_data("exit_tiles") or ()))
        spikes = set(tuple(cell) for cell in (level.get_data("spike_tiles") or ()))
        spawn = tuple(level.get_data("spawn") or ())
        plate_raw = level.get_data("plate")
        plate = tuple(plate_raw) if plate_raw is not None else None

        for sprite in level.get_sprites_by_tag("wall"):
            walls.add((int(sprite.x), int(sprite.y)))
        for sprite in level.get_sprites_by_tag("gate_post"):
            posts.add((int(sprite.x), int(sprite.y)))

        return {
            "width": GRID_WIDTH,
            "height": GRID_HEIGHT,
            "spawn": spawn,
            "exits": exits,
            "walls": walls,
            "posts": posts,
            "gates": tuple(level.get_data("gate_specs") or ()),
            "spikes": spikes,
            "spike_spec": level.get_data("spike_spec"),
            "plate": plate,
            "plate_target_gate": level.get_data("plate_target_gate"),
            "max_steps": int(level.get_data("max_steps") or 0),
        }

    def on_set_level(self, level: Level) -> None:
        player = level.get_sprites_by_name("player")
        if not player:
            raise RuntimeError("timing_gate: missing player sprite")
        self._player = player[0]

        bars = level.get_sprites_by_name("timebar")
        if not bars:
            raise RuntimeError("timing_gate: missing timebar sprite")
        self._timebar = bars[0]

        self._plate = next(iter(level.get_sprites_by_name("plate")), None)
        self._exit_sprites = sorted(level.get_sprites_by_tag("exit"), key=lambda s: (int(s.y), int(s.x)))
        self._spike_sprites = sorted(level.get_sprites_by_tag("spike"), key=lambda s: (int(s.y), int(s.x)))

        self._gate_specs = tuple(level.get_data("gate_specs") or ())
        self._gate_sprites = sorted(
            level.get_sprites_by_tag("gate_center"),
            key=lambda s: int(
                next(tag.split("_", 1)[1] for tag in s.tags if tag.startswith("gate_") and tag[5:].isdigit())
            ),
        )
        if len(self._gate_specs) != len(self._gate_sprites):
            raise RuntimeError("timing_gate: gate config mismatch")

        self._max_steps = int(level.get_data("max_steps") or 1)
        self._remaining_steps = int(self._max_steps)
        self._tick = 0
        self._gate_passable_now = [False for _ in self._gate_specs]
        self._gate_pulse_remaining = [0 for _ in self._gate_specs]
        self._spike_spec = level.get_data("spike_spec")
        self._spike_up_now = False
        self._plate_target_gate = level.get_data("plate_target_gate")
        self._pending_outcome = None
        self._transition_ticks = 0

        self._refresh_dynamic_visuals(pre_advance_pulses=self._gate_pulse_remaining)

    @staticmethod
    def _coords(sprite: Sprite) -> tuple[int, int]:
        return int(sprite.x), int(sprite.y)

    def _refresh_timebar(self) -> None:
        if self._timebar is None:
            return
        clamped_remaining = max(0, min(self._remaining_steps, self._max_steps))
        filled = math.ceil((clamped_remaining * TIMEBAR_LEN) / max(1, self._max_steps))
        filled = max(0, min(TIMEBAR_LEN, filled))

        low_time = clamped_remaining * 4 <= self._max_steps
        fill_color = COLOR_TIMEBAR_LOW if (low_time and self._tick % 2 == 1) else COLOR_TIMEBAR_FILL

        self._timebar.pixels = np.array(
            [[fill_color if x < filled else COLOR_TIMEBAR_EMPTY for x in range(TIMEBAR_LEN)]], dtype=np.int8
        )

    def _refresh_player_visual(self) -> None:
        if self._player is None:
            return
        if self._pending_outcome == "fail":
            self._player.pixels = _pixel(COLOR_SPIKE_UP)
            return
        self._player.pixels = _pixel(COLOR_PLAYER_A if self._tick % 2 == 0 else COLOR_PLAYER_B)

    def _refresh_exit_visual(self, fast: bool = False) -> None:
        if not self._exit_sprites:
            return
        if fast:
            color = COLOR_EXIT_A if self._transition_ticks % 2 == 0 else COLOR_EXIT_B
        else:
            color = COLOR_EXIT_A if self._tick % 2 == 0 else COLOR_EXIT_B
        for sprite in self._exit_sprites:
            sprite.pixels = _pixel(color)

    def _refresh_plate_visual(self) -> None:
        if self._plate is None or self._player is None:
            return
        self._plate.pixels = _pixel(
            COLOR_PLATE_ON if self._coords(self._player) == self._coords(self._plate) else COLOR_PLATE_OFF
        )

    def _refresh_gates(self, pre_advance_pulses: list[int]) -> None:
        self._gate_passable_now = [False for _ in self._gate_specs]

        for idx, gate_spec in enumerate(self._gate_specs):
            sprite = self._gate_sprites[idx]
            passable = False
            color = COLOR_GATE_CLOSED

            if gate_spec["mode"] == "timed":
                state = _gate_cycle_state(tuple(gate_spec["cycle"]), self._tick, int(gate_spec["phase"]))
                passable = state == "open"
                color = _gate_color_for_state(state, self._tick)
            else:
                active_now = pre_advance_pulses[idx] > 0
                passable = active_now
                if active_now and pre_advance_pulses[idx] <= 2:
                    color = _gate_color_for_state("warning", self._tick)
                elif active_now:
                    color = _gate_color_for_state("open", self._tick)
                else:
                    color = COLOR_GATE_CLOSED

            self._gate_passable_now[idx] = passable
            sprite.pixels = _pixel(color)
            sprite.set_collidable(not passable)

    def _refresh_spikes(self) -> None:
        if not self._spike_sprites:
            self._spike_up_now = False
            return
        if self._spike_spec is None:
            self._spike_up_now = False
            for sprite in self._spike_sprites:
                sprite.pixels = _pixel(COLOR_FLOOR)
            return

        self._spike_up_now = _spike_is_up(self._spike_spec, self._tick)
        color = COLOR_SPIKE_UP if self._spike_up_now else COLOR_FLOOR
        for sprite in self._spike_sprites:
            sprite.pixels = _pixel(color)

    def _refresh_dynamic_visuals(self, pre_advance_pulses: list[int]) -> None:
        self._refresh_gates(pre_advance_pulses)
        self._refresh_spikes()
        self._refresh_plate_visual()
        self._refresh_player_visual()
        self._refresh_exit_visual(self._pending_outcome == "win")
        self._refresh_timebar()

    def _gate_index_at(self, x: int, y: int) -> int | None:
        for idx, sprite in enumerate(self._gate_sprites):
            if int(sprite.x) == x and int(sprite.y) == y:
                return idx
        return None

    def _is_player_on_closed_gate(self) -> bool:
        if self._player is None:
            return False
        px, py = self._coords(self._player)
        gate_idx = self._gate_index_at(px, py)
        if gate_idx is None:
            return False
        return not self._gate_passable_now[gate_idx]

    def _is_player_on_up_spike(self) -> bool:
        if self._player is None or not self._spike_up_now:
            return False
        pos = self._coords(self._player)
        return any(self._coords(spike) == pos for spike in self._spike_sprites)

    def _is_player_on_exit(self) -> bool:
        if self._player is None:
            return False
        pos = self._coords(self._player)
        return any(self._coords(exit_tile) == pos for exit_tile in self._exit_sprites)

    def _apply_move_or_wait(self) -> None:
        if self._player is None:
            return
        action = self.action.id
        if action not in ACTION_TO_DELTA:
            return
        dx, dy = ACTION_TO_DELTA[action]
        self.try_move_sprite(self._player, dx, dy)

    def _apply_plate_effects(self) -> None:
        if self._plate is None or self._player is None:
            return
        if self._coords(self._plate) != self._coords(self._player):
            return
        target = self._plate_target_gate
        if target is None or not (0 <= int(target) < len(self._gate_specs)):
            return
        gate = self._gate_specs[int(target)]
        if gate["mode"] != "plate":
            return
        self._gate_pulse_remaining[int(target)] = int(gate["pulse_steps"]) + 1

    def _advance_timers(self) -> list[int]:
        pre = list(self._gate_pulse_remaining)
        self._tick += 1
        for idx, gate_spec in enumerate(self._gate_specs):
            if gate_spec["mode"] != "plate":
                continue
            if self._gate_pulse_remaining[idx] > 0:
                self._gate_pulse_remaining[idx] -= 1
        return pre

    def _begin_fail_transition(self) -> None:
        self._pending_outcome = "fail"
        self._transition_ticks = 2
        self._refresh_player_visual()

    def _begin_win_transition(self) -> None:
        self._pending_outcome = "win"
        self._transition_ticks = 4
        self._refresh_exit_visual(True)

    def _step_transition(self) -> None:
        if self._pending_outcome is None:
            return
        if self._pending_outcome == "fail":
            self._refresh_player_visual()
            self._transition_ticks -= 1
            if self._transition_ticks <= 0:
                self.lose()
            return

        self._refresh_exit_visual(True)
        self._transition_ticks -= 1
        if self._transition_ticks <= 0:
            self.next_level()

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

        if self._pending_outcome is not None:
            self._step_transition()
            self.complete_action()
            return

        self._apply_move_or_wait()
        self._remaining_steps -= 1
        self._apply_plate_effects()

        pre_advance_pulses = self._advance_timers()
        self._refresh_dynamic_visuals(pre_advance_pulses)

        if self._remaining_steps <= 0 or self._is_player_on_up_spike() or self._is_player_on_closed_gate():
            self._begin_fail_transition()
        elif self._is_player_on_exit():
            self._begin_win_transition()

        self.complete_action()
