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
    "side": "bottom",
    "rows": 2,
    "pip_width": 1,
    "actions_per_tick": 1,
    "pips_per_tick": 1,
    "pip_color": 9,
    "spent_color": 3,
    "gap": 0,
    "margin": 0,
    "tier_colors": [9],
}
ENERGY_CAPACITIES = [30, 72, 96]

GAME_ID = "home_finding-0001"
PX = 64

CLICK = int(GameAction.ACTION6.value)

C_FLOOR = 0
C_WALL = 5
C_RED = 8
C_BLUE = 9
C_PURPLE = 15

GRAB_RADIUS = 6
WIN_RADIUS = 2


def _agent_pixels(color: int) -> list[list[int]]:
    c = color
    return [[c, c, c], [c, C_WALL, c], [c, c, c]]


def _circle_pixels(color: int, outer_r: int = 5, inner_r: int = 3) -> list[list[int]]:
    size = outer_r * 2 + 1
    center = outer_r
    pixels: list[list[int]] = []
    for y in range(size):
        row: list[int] = []
        for x in range(size):
            dist = math.sqrt((x - center) ** 2 + (y - center) ** 2)
            if inner_r <= dist <= outer_r + 0.5:
                row.append(color)
            else:
                row.append(-1)
        pixels.append(row)
    return pixels


def _dist(x1: int, y1: int, x2: int, y2: int) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


C_GREEN = 14

LEVEL_SPECS: list[dict] = [
    {
        "name": "Level 1",
        "agents": [{"color": C_RED, "x": 15, "y": 30}],
        "circles": [{"color": C_RED, "cx": 48, "cy": 32}],
    },
    {
        "name": "Level 2",
        "agents": [{"color": C_RED, "x": 43, "y": 19}, {"color": C_BLUE, "x": 43, "y": 43}],
        "circles": [{"color": C_RED, "cx": 20, "cy": 44}, {"color": C_BLUE, "cx": 20, "cy": 20}],
    },
    {
        "name": "Level 3",
        "agents": [
            {"color": C_GREEN, "x": 31, "y": 43},
            {"color": C_RED, "x": 19, "y": 19},
            {"color": C_BLUE, "x": 43, "y": 19},
        ],
        "circles": [
            {"color": C_GREEN, "cx": 32, "cy": 20},
            {"color": C_RED, "cx": 44, "cy": 44},
            {"color": C_BLUE, "cx": 20, "cy": 44},
        ],
    },
]


def _build_level(spec: dict) -> Level:
    floor = Sprite(
        pixels=[[C_FLOOR] * PX for _ in range(PX)],
        name="floor",
        collidable=False,
        layer=-10,
        tags=["sys_click", "sys_every_pixel"],
    )

    sprites: list[Sprite] = [floor]

    for i, c in enumerate(spec["circles"]):
        sprites.append(
            Sprite(
                pixels=_circle_pixels(c["color"]),
                name=f"circle_{i}",
                x=c["cx"] - 5,
                y=c["cy"] - 5,
                collidable=False,
                layer=1,
                tags=["circle"],
            )
        )

    for i, a in enumerate(spec["agents"]):
        sprites.append(
            Sprite(
                pixels=_agent_pixels(a["color"]),
                name=f"agent_{i}",
                x=a["x"],
                y=a["y"],
                collidable=False,
                layer=2,
                tags=["agent"],
            )
        )

    return Level(
        name=spec["name"],
        grid_size=(PX, PX),
        sprites=sprites,
        data={
            "agents": [{"color": a["color"], "x": a["x"], "y": a["y"]} for a in spec["agents"]],
            "circles": [{"color": c["color"], "cx": c["cx"], "cy": c["cy"]} for c in spec["circles"]],
        },
    )


class HomeFinding(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        self._energy_bar = EnergyBar(**ENERGY_CONFIG)
        camera = Camera(0, 0, PX, PX, background=C_FLOOR, letter_box=C_WALL, interfaces=[self._energy_bar])
        super().__init__(
            GAME_ID, levels, camera=camera, debug=False, win_score=len(levels), available_actions=[CLICK], seed=seed
        )

    def on_set_level(self, level: Level) -> None:
        self._energy_last_action_count = 0
        self._energy_bar.set_capacity(ENERGY_CAPACITIES[self.level_index])
        self._agents = level.get_sprites_by_tag("agent")
        self._circles = level.get_sprites_by_tag("circle")
        raw_agents = level.get_data("agents") or []
        self._agent_colors = {f"agent_{i}": int(a["color"]) for i, a in enumerate(raw_agents)}
        raw_circles = level.get_data("circles") or []
        self._circle_colors = {f"circle_{i}": int(c["color"]) for i, c in enumerate(raw_circles)}
        self._circle_centers = {f"circle_{i}": (int(c["cx"]), int(c["cy"])) for i, c in enumerate(raw_circles)}

    def _agent_center(self, agent: object) -> tuple[int, int]:
        return (int(agent.x) + 1, int(agent.y) + 1)

    def _handle_click(self) -> None:
        data = self.action.data or {}
        try:
            raw_x = int(data.get("x", -1))
            raw_y = int(data.get("y", -1))
        except (TypeError, ValueError):
            return
        grid_pos = self.camera.display_to_grid(raw_x, raw_y)
        if grid_pos is None:
            return
        gx, gy = int(grid_pos[0]), int(grid_pos[1])

        best_agent = None
        best_dist = GRAB_RADIUS + 1.0
        for agent in self._agents:
            cx, cy = self._agent_center(agent)
            d = _dist(gx, gy, cx, cy)
            if d < best_dist:
                best_dist = d
                best_agent = agent

        if best_agent is None or best_dist > GRAB_RADIUS:
            return

        nx = max(0, min(PX - 3, gx - 1))
        ny = max(0, min(PX - 3, gy - 1))
        best_agent.set_position(nx, ny)

        if self._all_matched():
            self.next_level()

    def _all_matched(self) -> bool:
        for agent in self._agents:
            agent_color = self._agent_colors.get(agent.name)
            if agent_color is None:
                continue
            cx, cy = self._agent_center(agent)
            matched = False
            for circle in self._circles:
                circle_color = self._circle_colors.get(circle.name)
                if circle_color != agent_color:
                    continue
                tcx, tcy = self._circle_centers.get(circle.name, (-99, -99))
                if _dist(cx, cy, tcx, tcy) <= WIN_RADIUS:
                    matched = True
                    break
            if not matched:
                return False
        return True

    def step(self) -> None:
        if getattr(self, "_energy_last_action_count", -1) != self._action_count:
            self._energy_last_action_count = self._action_count
            if self._energy_bar.tick() == 0:
                self.lose()
                self.complete_action()
                return
        if self.action.id == GameAction.ACTION6:
            self._handle_click()
        self.complete_action()
