from __future__ import annotations

import math

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

SIDES = ("top", "bottom", "left", "right")


class EnergyBar(RenderableUserDisplay):
    def __init__(
        self,
        *,
        side: str = "top",
        rows: int = 1,
        pip_width: int = 2,
        actions_per_tick: int = 1,
        pips_per_tick: int = 1,
        pip_color: int = 11,
        spent_color: int = 3,
        gap: int = 1,
        margin: int = 0,
        tier_colors: list[int] | None = None,
    ) -> None:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        self.side = side
        self.rows = max(1, min(int(rows), 3))
        self.pip_width = max(1, min(int(pip_width), 3))
        self.actions_per_tick = max(1, int(actions_per_tick))
        self.pips_per_tick = max(1, int(pips_per_tick))
        self.pip_color = int(pip_color)
        self.spent_color = int(spent_color)
        self.gap = max(0, int(gap))
        self.margin = max(0, int(margin))
        self.tier_colors: list[int] = list(tier_colors) if tier_colors else [self.pip_color]

        self.capacity_actions = 0
        self.remaining_actions = 0

    def set_capacity(self, capacity_actions: int) -> None:
        self.capacity_actions = max(0, int(capacity_actions))
        self.remaining_actions = self.capacity_actions

    def set_remaining_actions(self, remaining_actions: int) -> None:
        self.remaining_actions = max(0, min(int(remaining_actions), self.capacity_actions))

    def tick(self) -> int:
        if self.remaining_actions > 0:
            self.remaining_actions -= 1
        return self.remaining_actions

    def _actions_to_pips(self, actions: int) -> int:
        if actions <= 0:
            return 0
        return (actions * self.pips_per_tick + self.actions_per_tick - 1) // self.actions_per_tick

    @property
    def total_pips(self) -> int:
        return self._actions_to_pips(self.capacity_actions)

    @property
    def remaining_pips(self) -> int:
        return self._actions_to_pips(self.remaining_actions)

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.capacity_actions <= 0:
            return frame
        h, w = int(frame.shape[0]), int(frame.shape[1])
        total = self.total_pips
        remaining = self.remaining_pips
        if total <= 0:
            return frame

        pw = self.pip_width
        ph = self.pip_width
        stride = pw + self.gap
        horizontal = self.side in ("top", "bottom")
        long_dim = w if horizontal else h
        pips_per_row = max(1, (long_dim - self.margin) // stride)
        slot_count = pips_per_row * self.rows

        if total <= slot_count:
            visible = total
            colored = remaining
            color = self.pip_color
        else:
            visible = slot_count
            consumed = total - remaining
            tier_index = consumed // slot_count
            consumed_in_tier = consumed - tier_index * slot_count
            if tier_index >= len(self.tier_colors):
                tier_index = len(self.tier_colors) - 1
                consumed_in_tier = slot_count
            colored = slot_count - consumed_in_tier
            color = self.tier_colors[tier_index]

        for i in range(visible):
            row = i // pips_per_row
            col = i % pips_per_row
            if row >= self.rows:
                break
            cell_color = color if i < colored else self.spent_color
            if horizontal:
                x = self.margin + col * stride
                if self.side == "top":
                    y = self.margin + row * stride
                else:
                    y = h - self.margin - (row + 1) * pw - row * self.gap
                self._fill(frame, x, y, pw, ph, cell_color)
            else:
                y = self.margin + col * stride
                if self.side == "left":
                    x = self.margin + row * stride
                else:
                    x = w - self.margin - (row + 1) * pw - row * self.gap
                self._fill(frame, x, y, pw, ph, cell_color)
        return frame

    @staticmethod
    def _fill(frame: np.ndarray, x: int, y: int, w: int, h: int, color: int) -> None:
        h_frame, w_frame = int(frame.shape[0]), int(frame.shape[1])
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w_frame, x + w)
        y1 = min(h_frame, y + h)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = color


ENERGY_CONFIG = {
    "side": "top",
    "rows": 1,
    "pip_width": 1,
    "actions_per_tick": 1,
    "pips_per_tick": 2,
    "pip_color": 14,
    "spent_color": 15,
    "gap": 0,
    "margin": 0,
    "tier_colors": [14, 12, 11],
}
ENERGY_CAPACITIES = [78, 78, 84]

GAME_ID = "windmolen-0001"
W = H = 64
CELL = 4
GW = GH = 16

UP = int(GameAction.ACTION1.value)
DOWN = int(GameAction.ACTION2.value)
LEFT = int(GameAction.ACTION3.value)
RIGHT = int(GameAction.ACTION4.value)

MOVE = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}

C_FLOOR = 0
C_WALL = 5
C_PLAYER = 9
C_GOAL = 14
C_START = 4  # yellow — spawn point
C_HUB = 7  # orange — turbine center
C_ARM = 13  # dark red — turbine arm
C_DEATH = 8  # bright red — death flash

# Clockwise rotation in 8 steps: R, DR, D, DL, L, UL, U, UR
ARM_DIRS: list[tuple[int, int]] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _arm_phase(tick: int, period: int, offset: int) -> int:
    """Which direction the arm points at a given tick."""
    return ((tick + offset) * 8 // period) % 8


def _arm_cells(cx: int, cy: int, arm_length: int, phase: int) -> list[tuple[int, int]]:
    dx, dy = ARM_DIRS[phase]
    return [(cx + dx * i, cy + dy * i) for i in range(1, arm_length + 1)]


def _sweep_between(cx: int, cy: int, arm_length: int, phase_a: int, phase_b: int) -> set[tuple[int, int]]:
    """Cells swept by arm rotating from phase_a to adjacent phase_b."""
    cells: set[tuple[int, int]] = set()
    dx_a, dy_a = ARM_DIRS[phase_a]
    dx_b, dy_b = ARM_DIRS[phase_b]
    for i in range(1, arm_length + 1):
        x0, y0 = cx + dx_a * i, cy + dy_a * i
        x1, y1 = cx + dx_b * i, cy + dy_b * i
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            x = x0 + round((x1 - x0) * s / steps)
            y = y0 + round((y1 - y0) * s / steps)
            cells.add((x, y))
    return cells


def _turbine_blocked(turbines: list[dict], tick: int) -> set[tuple[int, int]]:
    """Cells blocked by turbine centers + arms at a given tick (for movement)."""
    blocked: set[tuple[int, int]] = set()
    for t in turbines:
        cx, cy = t["cx"], t["cy"]
        blocked.add((cx, cy))
        phase = _arm_phase(tick, t["period"], t["offset"])
        for cell in _arm_cells(cx, cy, t["arm_length"], phase):
            blocked.add(cell)
    return blocked


def _turbine_death_zone(turbines: list[dict], old_tick: int, new_tick: int) -> set[tuple[int, int]]:
    """All cells lethal this tick: current arm + everything the arm swept through."""
    zone: set[tuple[int, int]] = set()
    for t in turbines:
        cx, cy = t["cx"], t["cy"]
        zone.add((cx, cy))
        old_phase = _arm_phase(old_tick, t["period"], t["offset"])
        new_phase = _arm_phase(new_tick, t["period"], t["offset"])
        if old_phase == new_phase:
            for cell in _arm_cells(cx, cy, t["arm_length"], new_phase):
                zone.add(cell)
        else:
            phase = old_phase
            while phase != new_phase:
                nxt = (phase + 1) % 8
                zone.update(_sweep_between(cx, cy, t["arm_length"], phase, nxt))
                phase = nxt
    return zone


def _parse_layout(lines: list[str]) -> np.ndarray:
    grid = np.zeros((GH, GW), dtype=np.int8)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "W":
                grid[y, x] = 1
    return grid


def _render(
    base: np.ndarray,
    player: list[int],
    goals: list[list[int]],
    turbines: list[dict],
    tick: int,
    start: list[int] | None = None,
    death: bool = False,
) -> np.ndarray:
    canvas = np.full((H, W), C_FLOOR, dtype=np.int8)

    for gy in range(GH):
        for gx in range(GW):
            if base[gy, gx] == 1:
                y0, x0 = gy * CELL, gx * CELL
                canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_WALL

    # Start marker (yellow, drawn under everything else)
    if start is not None:
        sx, sy = start
        sy0, sx0 = sy * CELL, sx * CELL
        canvas[sy0 : sy0 + CELL, sx0 : sx0 + CELL] = C_START

    for gx, gy in goals:
        y0, x0 = gy * CELL, gx * CELL
        canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_GOAL

    for t in turbines:
        cx, cy = t["cx"], t["cy"]
        phase = _arm_phase(tick, t["period"], t["offset"])
        y0, x0 = cy * CELL, cx * CELL
        canvas[y0 : y0 + CELL, x0 : x0 + CELL] = C_HUB
        for ax, ay in _arm_cells(cx, cy, t["arm_length"], phase):
            ay0, ax0 = ay * CELL, ax * CELL
            canvas[ay0 : ay0 + CELL, ax0 : ax0 + CELL] = C_ARM

    px, py = player
    py0, px0 = py * CELL, px * CELL
    if death:
        canvas[py0 : py0 + CELL, px0 : px0 + CELL] = C_DEATH
    else:
        canvas[py0 : py0 + CELL, px0 : px0 + CELL] = C_PLAYER
        canvas[py0 + 1 : py0 + 3, px0 + 1 : px0 + 3] = C_FLOOR

    return canvas


# ---------------------------------------------------------------------------
# Layouts (16x16 logical grid, W=wall, .=floor)
# ---------------------------------------------------------------------------

LAYOUT_1 = [
    "WWWWWWWWWWWWWWWW",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "WWWWWWWWWWWWWWWW",
]

LAYOUT_2 = [
    "WWWWWWWWWWWWWWWW",
    "W..............W",
    "W..............W",
    "W..............W",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "W..............W",
    "W..............W",
    "W..............W",
    "W..............W",
    "WWWWWWWWWWWWWWWW",
]

LAYOUT_3 = [
    "WWWWWWWWWWWWWWWW",
    "W..............W",
    "W..............W",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "WWWW.......WWWWW",
    "W..............W",
    "W..............W",
    "WWWWWWWWWWWWWWWW",
]

LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "layout": LAYOUT_1,
        "start": [1, 1],
        "goals": [[14, 14]],
        "turbines": [{"cx": 8, "cy": 8, "period": 8, "offset": 0, "arm_length": 3}],
    },
    {
        "name": "Level 2",
        "layout": LAYOUT_2,
        "start": [1, 1],
        "goals": [[14, 14]],
        "turbines": [{"cx": 7, "cy": 7, "period": 8, "offset": 0, "arm_length": 3}],
    },
    {
        "name": "Level 3",
        "layout": LAYOUT_3,
        "start": [1, 1],
        "goals": [[14, 14]],
        "turbines": [
            {"cx": 7, "cy": 5, "period": 8, "offset": 0, "arm_length": 3},
            {"cx": 7, "cy": 10, "period": 16, "offset": 0, "arm_length": 3},
        ],
    },
]


# ---------------------------------------------------------------------------
# Search helpers (used by DSL agent for BFS planning)
# ---------------------------------------------------------------------------


def _level_period(turbines: list[dict]) -> int:
    period = 1
    for t in turbines:
        period = period * t["period"] // math.gcd(period, t["period"])
    return period


def search_initial_state(spec: dict) -> tuple[int, int, int]:
    return (spec["start"][0], spec["start"][1], 0)


def search_apply_action(
    spec: dict, state: tuple[int, int, int], action_id: int, base: np.ndarray
) -> tuple[tuple[int, int, int] | None, bool]:
    """Transition function for BFS. Returns (new_state, won) or (None, False) if dead."""
    px, py, tick = state
    turbines = spec["turbines"]
    period = _level_period(turbines)
    goal_set = {(g[0], g[1]) for g in spec["goals"]}

    # Frame 1: player moves (blocked by current arm + walls)
    blocked = _turbine_blocked(turbines, tick)
    nx, ny = px, py
    if action_id in MOVE:
        dx, dy = MOVE[action_id]
        tx, ty = px + dx, py + dy
        if 0 <= tx < GW and 0 <= ty < GH and base[ty, tx] == 0 and (tx, ty) not in blocked:
            nx, ny = tx, ty

    # Goal check before arm rotates
    if (nx, ny) in goal_set:
        return (nx, ny, (tick + 1) % period), True

    # Frame 2: arm rotates — check death zone
    death_zone = _turbine_death_zone(turbines, tick, tick + 1)
    if (nx, ny) in death_zone:
        return None, False

    return (nx, ny, (tick + 1) % period), False


def _build_level(spec: dict) -> Level:
    base = _parse_layout(spec["layout"])
    initial = _render(base, spec["start"], spec["goals"], spec["turbines"], 0, start=spec["start"])
    board = Sprite(pixels=initial, name="board", collidable=False, layer=0)
    return Level(
        name=spec["name"],
        grid_size=(W, H),
        sprites=[board],
        data={"layout": spec["layout"], "start": spec["start"], "goals": spec["goals"], "turbines": spec["turbines"]},
    )


class Windmolen(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(0, 0, W, H, background=C_FLOOR, letter_box=C_FLOOR, interfaces=[self._energy_bar])
        super().__init__(
            GAME_ID,
            levels,
            camera=camera,
            debug=False,
            win_score=len(levels),
            available_actions=[UP, DOWN, LEFT, RIGHT],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        self._base = _parse_layout(level.get_data("layout"))
        start = level.get_data("start")
        self._start = [int(start[0]), int(start[1])]
        self._pos = list(self._start)
        self._goals = level.get_data("goals")
        self._goal_set = {(int(g[0]), int(g[1])) for g in self._goals}
        self._turbines = level.get_data("turbines")
        self._tick = 0
        self._hit_timer = 0
        self._arm_pending = False
        self._board = level.get_sprites_by_name("board")[0]
        self._redraw()

    def _redraw(self, death: bool = False) -> None:
        self._board.pixels = _render(
            self._base, self._pos, self._goals, self._turbines, self._tick, start=self._start, death=death
        )

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        if getattr(self.action.id, "value", self.action.id) == 0 and self.level_index > 0:
            # arcengine's perform_action loop runs step() once after
            # handle_reset; without this guard the sim would advance a
            # tick, so mid-play RESET on any level entered via
            # next_level() would land one tick past the frame the
            # client saw on arrival. Level 0 keeps the legacy tick to
            # preserve env.reset()'s observation and the DSL trace.
            self.complete_action()
            return

        # Death animation: flash for a few frames, then end the level as a loss.
        if self._hit_timer > 0:
            self._hit_timer -= 1
            if self._hit_timer == 0:
                self.lose()
                self.complete_action()
            return

        # Frame 2: arm rotates (after player already moved last frame)
        if self._arm_pending:
            self._arm_pending = False
            old_tick = self._tick
            self._tick += 1
            death_zone = _turbine_death_zone(self._turbines, old_tick, self._tick)
            if (self._pos[0], self._pos[1]) in death_zone:
                self._hit_timer = 3
                self._redraw(death=True)
                return
            self._redraw()
            self.complete_action()
            return

        # Frame 1: player moves, arm stays
        action_id = int(getattr(self.action.id, "value", self.action.id))
        blocked = _turbine_blocked(self._turbines, self._tick)

        if action_id in MOVE:
            dx, dy = MOVE[action_id]
            nx, ny = self._pos[0] + dx, self._pos[1] + dy
            if 0 <= nx < GW and 0 <= ny < GH and self._base[ny, nx] == 0 and (nx, ny) not in blocked:
                self._pos = [nx, ny]

        # Reached goal → next level (before arm rotates)
        if (self._pos[0], self._pos[1]) in self._goal_set:
            self._redraw()
            self.next_level()
            self.complete_action()
            return

        self._arm_pending = True
        self._redraw()
        # No complete_action — arm rotation happens next frame
