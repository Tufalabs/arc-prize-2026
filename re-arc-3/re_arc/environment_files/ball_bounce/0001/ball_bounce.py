from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
GRID_W = 64
GRID_H = 64

# Playfield interior cells (inclusive). The 1px frame sits just outside.
FIELD_LEFT = 1
FIELD_TOP = 1
FIELD_RIGHT = 61  # 61 - 1 is a multiple of GRID_PITCH: the lattice ends flush on a line
FIELD_BOTTOM = 52

TRAY_TOP = 54  # tray backdrop rows TRAY_TOP..63
TRAY_SLOT_X0 = 2
TRAY_SLOT_W = 10
TRAY_SLOTS = 6

BALL_SIZE = 4
BALL_RADIUS = BALL_SIZE / 2.0  # physics hitbox: a circle centered on the 4x4 sprite
GOAL_SIZE = 6  # 2 cells + 2 grid lines: sits cleanly on the lattice
GOAL_RADIUS = GOAL_SIZE / 2.0  # scoring: ball circle must intersect this goal circle

COLOR_FRAME = 3
COLOR_TRAY = 4
COLOR_BALL = 9
COLOR_GOAL = 11
COLOR_WALL = COLOR_FRAME  # placed walls match the outer frame
COLOR_RAMP = COLOR_WALL  # ramps match walls
COLOR_BOUNCER = 14
COLOR_STICKY = 13  # dark red/maroon — the palette's closest thing to brown
COLOR_PORTAL_A = 10
COLOR_PORTAL_B = 12
COLOR_KICKER = 7
COLOR_KICKER_HOT = 0  # white: a triggered kicker flashes white while its boost lasts
COLOR_HIGHLIGHT = 0
COLOR_GRID = 4  # faint placement grid in the field interior
COLOR_FLASH = 8  # red — illegal-click feedback

SQRT2 = math.sqrt(2.0)
# Ball radius plus ramp band half-thickness (a circle's extent along any normal is its radius).
RAMP_CONTACT = BALL_RADIUS + 1.0 / SQRT2

# Max ball travel between collision checks; keeps fast balls from tunneling walls.
MAX_MICRO_STEP = 1.0
# Upper bound on micro-steps per substep (a stuck-ball backstop; never hit in practice).
MICRO_STEP_GUARD = 256

# Rendered frames each red/clear phase of the illegal-click flash is held.
FLASH_PHASE_FRAMES = 4
# Rendered frames a triggered kicker is drawn white.
KICKER_FLASH_FRAMES = 3


# ---------------------------------------------------------------------------
# Object catalog
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObjectKind:
    name: str
    width: int
    height: int
    color: int
    behavior: str  # solid | bouncer | sticky | ramp_a | ramp_b | portal_a | portal_b | kicker_*
    placeable: bool = True


OBJECT_KINDS: dict[str, ObjectKind] = {
    "wall_h": ObjectKind("wall_h", 8, 2, COLOR_WALL, "solid"),
    "wall_v": ObjectKind("wall_v", 2, 8, COLOR_WALL, "solid"),
    # Square wall block; per-object "size" in the level spec overrides the default n.
    "wall_box": ObjectKind("wall_box", 4, 4, COLOR_WALL, "solid"),
    "ramp_a": ObjectKind("ramp_a", 6, 6, COLOR_RAMP, "ramp_a"),
    "ramp_b": ObjectKind("ramp_b", 6, 6, COLOR_RAMP, "ramp_b"),
    "bouncer": ObjectKind("bouncer", 8, 2, COLOR_BOUNCER, "bouncer"),
    "sticky": ObjectKind("sticky", 8, 2, COLOR_STICKY, "sticky"),
    "portal_a": ObjectKind("portal_a", 2, 6, COLOR_PORTAL_A, "portal_a"),
    "portal_b": ObjectKind("portal_b", 2, 6, COLOR_PORTAL_B, "portal_b"),
    # Kickers SET the ball's velocity to a fixed vector on contact (pinball
    # kicker): fully predictable redirection, one kind per direction.
    "kicker_r": ObjectKind("kicker_r", 4, 4, COLOR_KICKER, "kicker_r"),
    "kicker_l": ObjectKind("kicker_l", 4, 4, COLOR_KICKER, "kicker_l"),
    "kicker_u": ObjectKind("kicker_u", 4, 4, COLOR_KICKER, "kicker_u"),
    "kicker_d": ObjectKind("kicker_d", 4, 4, COLOR_KICKER, "kicker_d"),
}

KICKER_DIRECTIONS = {"kicker_r": (1.0, 0.0), "kicker_l": (-1.0, 0.0), "kicker_u": (0.0, -1.0), "kicker_d": (0.0, 1.0)}


WALL_BOX_MIN = 1  # 1x1 walls are designer-only (placed off-grid via the editor)
WALL_BOX_MAX = 12
WALL_BOX_TRAY_MAX = 8  # bigger boxes don't fit a tray slot

# The field is a lattice of 1px grid lines every GRID_PITCH pixels with 2x2
# cells between them. Objects are CENTERED on the clicked cell (even sizes
# straddle it symmetrically). A click whose cell is illegal does NOT place the
# piece (no nearest-cell fallback); clicks on the lines themselves do nothing.
GRID_PITCH = 3
CELL_SIZE = GRID_PITCH - 1  # interior cell side: (pitch - 1) px between adjacent lines


def snap_to_cell(value: int, origin: int) -> int:
    """Nearest cell top-left (cells sit at origin + 1 + k * GRID_PITCH)."""
    return origin + 1 + round((value - origin - 1) / GRID_PITCH) * GRID_PITCH


def on_grid_line(gx: int, gy: int) -> bool:
    return (gx - FIELD_LEFT) % GRID_PITCH == 0 or (gy - FIELD_TOP) % GRID_PITCH == 0


def in_field(gx: int, gy: int) -> bool:
    """True if the pixel lies inside the playfield interior (inclusive)."""
    return FIELD_LEFT <= gx <= FIELD_RIGHT and FIELD_TOP <= gy <= FIELD_BOTTOM


def cell_origin(value: int, origin: int) -> int:
    """Top-left of the 2x2 cell CONTAINING value (floor, not nearest)."""
    return origin + 1 + ((value - origin - 1) // GRID_PITCH) * GRID_PITCH


def cell_pixels(ox: int, oy: int) -> list[tuple[int, int]]:
    """The interior pixels of the cell whose top-left interior pixel is (ox, oy)."""
    return [(ox + i, oy + j) for i in range(CELL_SIZE) for j in range(CELL_SIZE)]


def cell_click_pixels(x: int, y: int, w: int, h: int) -> list[tuple[int, int]]:
    """The click pixels that place an even-sized piece's top-left at (x, y).

    Placement centers a piece on the clicked cell (top-left = cell_origin + 1 -
    w // 2), so any of the cell's interior pixels is an equivalent click. Shared
    with the DSL agent so its replay clicks can't drift from `_try_place`'s snap.
    Raises for odd sizes, which this symmetric centering cannot place exactly.
    """
    if w % 2 or h % 2:
        raise ValueError(f"cell_click_pixels requires even dimensions, got {w}x{h}")
    # The piece centers on a cell, so any of that cell's interior pixels clicks it.
    return cell_pixels(x - 1 + w // 2, y - 1 + h // 2)


def object_dims(kind: ObjectKind, size: int | None = None) -> tuple[int, int]:
    """Per-object dimensions; "size" makes wall_box an n x n square."""
    if size is not None and kind.name == "wall_box":
        n = max(WALL_BOX_MIN, min(WALL_BOX_MAX, int(size)))
        return n, n
    return kind.width, kind.height


def kind_pixels(kind: ObjectKind, size: int | None = None) -> list[list[int]]:
    w, h = object_dims(kind, size)
    c = kind.color
    if kind.behavior == "ramp_a":
        # "/" band: local cells with x + y in {w - 1, w}
        return [[c if x + y in (w - 1, w) else -1 for x in range(w)] for y in range(h)]
    if kind.behavior == "ramp_b":
        # "\" band: local cells with x - y in {0, 1}
        return [[c if x - y in (0, 1) else -1 for x in range(w)] for y in range(h)]
    if kind.behavior == "bouncer":
        # trampoline |---|: mat on top, legs at the ends
        return [[c] * w, [c] + [-1] * (w - 2) + [c]]
    if kind.behavior == "kicker_r":  # ">>"
        return [[c, -1, c, -1], [-1, c, -1, c], [-1, c, -1, c], [c, -1, c, -1]]
    if kind.behavior == "kicker_l":  # "<<"
        return [[-1, c, -1, c], [c, -1, c, -1], [c, -1, c, -1], [-1, c, -1, c]]
    if kind.behavior == "kicker_u":  # stacked "^^"
        return [[-1, c, c, -1], [c, -1, -1, c], [-1, c, c, -1], [c, -1, -1, c]]
    if kind.behavior == "kicker_d":  # stacked "vv"
        return [[c, -1, -1, c], [-1, c, c, -1], [c, -1, -1, c], [-1, c, c, -1]]
    return [[c] * w for _ in range(h)]


def occupied_cells(kind_name: str, x: int, y: int, w: int, h: int) -> set[tuple[int, int]]:
    """Grid cells a placed piece actually fills, for overlap tests. A diagonal
    ramp fills only its band, so its empty corners neither block nor get blocked;
    every other piece is treated as a solid w x h box."""
    kind = OBJECT_KINDS[kind_name]
    if kind.behavior in ("ramp_a", "ramp_b"):
        rows = kind_pixels(kind)
        return {(x + cx, y + cy) for cy, row in enumerate(rows) for cx, val in enumerate(row) if val != -1}
    return {(x + i, y + j) for i in range(w) for j in range(h)}


BALL_PIXELS = [
    [-1, COLOR_BALL, COLOR_BALL, -1],
    [COLOR_BALL, COLOR_BALL, COLOR_BALL, COLOR_BALL],
    [COLOR_BALL, COLOR_BALL, COLOR_BALL, COLOR_BALL],
    [-1, COLOR_BALL, COLOR_BALL, -1],
]

_G = COLOR_GOAL
GOAL_PIXELS = [
    [-1, -1, _G, _G, -1, -1],
    [-1, _G, -1, -1, _G, -1],
    [_G, -1, -1, -1, -1, _G],
    [_G, -1, -1, -1, -1, _G],
    [-1, _G, -1, -1, _G, -1],
    [-1, -1, _G, _G, -1, -1],
]


# ---------------------------------------------------------------------------
# Physics configuration (global; editable via the level editor)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicsConfig:
    energy: int = 10  # attempts per level: one consumed per launch (space); clicks are free and unlimited
    gravity: float = 0.10  # downward acceleration, px/frame^2
    substeps: int = 5  # physics integration steps per physics frame (accuracy, not speed)
    render_every: int = 2  # physics frames per rendered gameplay frame (2 = render every 10th substep)
    restitution: float = 0.62  # speed kept along the surface normal on wall/solid bounces
    tangential_keep: float = 0.92  # speed kept along the surface per contact (friction)
    ramp_restitution: float = 0.95  # same, for ramp faces (high = crisp mirror deflections)
    ramp_tangential_keep: float = 0.95
    air_keep: float = 1.0  # per-frame velocity multiplier in flight (1.0 = no air drag)
    max_frames: int = 240  # hard cap on flight length, frames
    rest_speed: float = 0.2  # flight ends as "rest" when speed stays below this...
    rest_frames: int = 10  # ...for this many consecutive frames
    bouncer_restitution: float = 0.62  # bouncer top face restitution
    bouncer_min_exit_speed: float = 2.2  # guaranteed pop speed off a bouncer top (height = v^2/2g)
    sticky_restitution: float = 0.02  # sticky top face: kills the bounce
    sticky_tangential_keep: float = 0.25  # sticky top face: kills most sliding
    sticky_min_speed: float = 0.5  # ...but the ball keeps rolling along sticky at least this fast
    portal_cooldown_frames: int = 8  # RENDERED gameplay frames both portals stay inert after a teleport
    kicker_speed: float = 2.0  # speed (px/physics frame) a kicker SETS the ball to, in its direction
    kicker_cooldown_frames: int = 8  # RENDERED frames a used kicker stays inert (then re-arms once clear)
    kicker_capture: float = 2.0  # cross-axis hitbox half-width (px); bigger = more forgiving trigger
    kicker_gravity_free_frames: int = 6  # physics frames gravity is suppressed after a kick (clean launch)

    @classmethod
    def from_dict(cls, raw: dict) -> PhysicsConfig:
        defaults = cls()
        kwargs = {}
        for name, default in defaults.__dict__.items():
            value = raw.get(name, default)
            kwargs[name] = int(value) if isinstance(default, int) else float(value)
        if not 1 <= kwargs["energy"] <= 64:
            raise ValueError(f"energy (attempts per level) must be in [1, 64], got {kwargs['energy']}")
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Flight simulation (the deterministic physics IR)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimObject:
    kind: ObjectKind
    x: int
    y: int
    w: int = 0  # 0 = use the kind's default dimensions
    h: int = 0
    index: int = -1  # original level-object index (so the game can map sim events back to sprites)

    @property
    def width(self) -> int:
        return self.w or self.kind.width

    @property
    def height(self) -> int:
        return self.h or self.kind.height

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (float(self.x), float(self.y), float(self.x + self.width), float(self.y + self.height))

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)


@dataclass(frozen=True)
class FlightResult:
    frames: list[tuple[float, float]]
    hit_goal: bool
    end_reason: str  # goal | rest | timeout | out


def normalize_borders(raw: dict | None) -> dict:
    """Per-level border walls; missing keys default to enabled."""
    raw = raw or {}
    return {side: bool(raw.get(side, True)) for side in ("top", "left", "right", "floor")}


def _surface_params(behavior: str, cfg: PhysicsConfig) -> tuple[float, float, float, float]:
    """Return (restitution, tangential_keep, min_exit_speed, min_roll_speed) for a solid surface."""
    if behavior == "bouncer":
        return cfg.bouncer_restitution, cfg.tangential_keep, cfg.bouncer_min_exit_speed, 0.0
    if behavior == "sticky":
        return cfg.sticky_restitution, cfg.sticky_tangential_keep, 0.0, cfg.sticky_min_speed
    return cfg.restitution, cfg.tangential_keep, 0.0, 0.0


class FlightSim:
    """Simulates one launch of the ball.

    step_physics_frame() advances one physics frame (the simulation unit all
    constants are expressed in); step_frame() advances one RENDERED gameplay
    frame, i.e. cfg.render_every physics frames.
    """

    def __init__(
        self,
        cfg: PhysicsConfig,
        start_center: tuple[float, float],
        velocity: tuple[float, float],
        objects: list[SimObject],
        goal_pos: tuple[int, int],
        borders: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.cx, self.cy = float(start_center[0]), float(start_center[1])
        self.vx, self.vy = float(velocity[0]), float(velocity[1])
        self.frame_count = 0
        self.slow_frames = 0
        self.done = False
        self.hit_goal = False
        self.end_reason = ""
        self.portal_clear = True
        self.portal_cooldown = 0
        self.gravity_free_frames = 0
        self._kicker_fired = False  # set per-frame; ends the frame at a kick's reset state

        gx, gy = goal_pos
        self.goal_center = (gx + GOAL_SIZE / 2.0, gy + GOAL_SIZE / 2.0)

        # Boundary walls of the playfield interior (interior spans continuous
        # x in [FIELD_LEFT, FIELD_RIGHT + 1), y in [FIELD_TOP, FIELD_BOTTOM + 1)).
        # Disabled borders are simply absent; the ball then exits and the flight
        # ends with reason "out".
        self.borders = normalize_borders(borders)
        margin = 8.0
        left, top = float(FIELD_LEFT), float(FIELD_TOP)
        right, bottom = float(FIELD_RIGHT + 1), float(FIELD_BOTTOM + 1)
        self.solids: list[tuple[float, float, float, float, str]] = []
        if self.borders["left"]:
            self.solids.append((left - margin, top - margin, left, bottom + margin, "solid"))
        if self.borders["right"]:
            self.solids.append((right, top - margin, right + margin, bottom + margin, "solid"))
        if self.borders["top"]:
            self.solids.append((left - margin, top - margin, right + margin, top, "solid"))
        if self.borders["floor"]:
            self.solids.append((left - margin, bottom, right + margin, bottom + margin, "solid"))
        # Ramps are thin 45-degree mirror segments: (ax, ay, bx, by, behavior, line_c).
        self.ramps: list[tuple[float, float, float, float, str, float]] = []
        self.kickers: list[dict] = []
        portal_in: SimObject | None = None
        portal_out: SimObject | None = None
        for obj in objects:
            behavior = obj.kind.behavior
            if behavior in ("solid", "bouncer", "sticky"):
                self.solids.append((*obj.box, behavior))
            if behavior in ("ramp_a", "ramp_b"):
                size = float(obj.kind.width)
                if behavior == "ramp_a":
                    # "/" centerline x + y = c, clipped to the object box.
                    line_c = obj.x + obj.y + size + 0.5
                    ax, ay = obj.x + size, obj.y + 0.5
                    bx, by = obj.x + 0.5, obj.y + size
                else:
                    # "\" centerline x - y = c, clipped to the object box.
                    line_c = obj.x - obj.y + 0.5
                    ax, ay = obj.x + 0.5, float(obj.y)
                    bx, by = obj.x + size, obj.y + size - 0.5
                self.ramps.append((ax, ay, bx, by, behavior, line_c))
            elif behavior == "portal_a":
                portal_in = obj
            elif behavior == "portal_b":
                portal_out = obj
            elif behavior in KICKER_DIRECTIONS:
                dx, dy = KICKER_DIRECTIONS[behavior]
                self.kickers.append(
                    {
                        "box": obj.box,
                        "center": obj.center,  # fixed launch point
                        "index": obj.index,  # level-object index, for the white-flash sprite
                        "vx": dx * cfg.kicker_speed,
                        "vy": dy * cfg.kicker_speed,
                        "horizontal": dx != 0.0,  # travel axis: x for kicker_l/r, else y
                        "cooldown": 0,
                        "white_frames": 0,  # rendered frames left flashing white after a kick
                        "armed": True,
                    }
                )
        self.portal_pair: tuple[SimObject, SimObject] | None = None
        if portal_in is not None and portal_out is not None:
            self.portal_pair = (portal_in, portal_out)

    # -- helpers --

    def _collide_solids(self) -> None:
        """Circle-vs-box: push out along the contact normal and reflect.

        Corner contacts get true diagonal normals. Pads (bouncer/sticky) are
        special only when the contact normal points mostly up (their top face).
        """
        for x0, y0, x1, y1, behavior in self.solids:
            nearest_x = min(max(self.cx, x0), x1)
            nearest_y = min(max(self.cy, y0), y1)
            dx, dy = self.cx - nearest_x, self.cy - nearest_y
            dist_sq = dx * dx + dy * dy
            if dist_sq >= BALL_RADIUS * BALL_RADIUS:
                continue
            if dist_sq > 1e-12:
                dist = math.sqrt(dist_sq)
                nx, ny = dx / dist, dy / dist
                pen = BALL_RADIUS - dist
            else:
                # Center inside the box: push out through the nearest face.
                gaps = (self.cx - x0, x1 - self.cx, self.cy - y0, y1 - self.cy)
                smallest = min(gaps)
                if smallest == gaps[0]:
                    nx, ny = -1.0, 0.0
                elif smallest == gaps[1]:
                    nx, ny = 1.0, 0.0
                elif smallest == gaps[2]:
                    nx, ny = 0.0, -1.0
                else:
                    nx, ny = 0.0, 1.0
                pen = BALL_RADIUS + smallest
            vn = self.vx * nx + self.vy * ny
            if vn >= 0:
                # Already separating; just resolve the overlap.
                self.cx += nx * pen
                self.cy += ny * pen
                continue
            hit_top = ny <= -0.707  # contact normal within ~45 degrees of straight up
            rest, keep, min_exit, min_roll = _surface_params(behavior if hit_top else "solid", self.cfg)
            tx, ty = -ny, nx
            vt = self.vx * tx + self.vy * ty
            vn_out = -vn * rest
            if hit_top and min_exit > 0 and vn_out < min_exit:
                vn_out = min_exit
            vt_out = vt * keep
            # Sticky tops never quite stop the ball: it keeps rolling along the
            # surface at min_roll (in the direction it was already moving).
            if hit_top and min_roll > 0 and vt_out != 0 and abs(vt_out) < min_roll:
                vt_out = math.copysign(min_roll, vt_out)
            self.vx = vn_out * nx + vt_out * tx
            self.vy = vn_out * ny + vt_out * ty
            self.cx += nx * pen
            self.cy += ny * pen

    def _collide_ramps(self) -> None:
        for ax, ay, bx, by, behavior, line_c in self.ramps:
            # Distance from ball center to the mirror SEGMENT, so the ball never
            # reflects off empty space beyond the drawn band's ends.
            abx, aby = bx - ax, by - ay
            t = ((self.cx - ax) * abx + (self.cy - ay) * aby) / (abx * abx + aby * aby)
            t = min(1.0, max(0.0, t))
            seg_dist = math.hypot(self.cx - (ax + t * abx), self.cy - (ay + t * aby))
            if seg_dist >= RAMP_CONTACT:
                continue
            if behavior == "ramp_a":
                nx, ny = 1.0 / SQRT2, 1.0 / SQRT2
                dist = (self.cx + self.cy - line_c) / SQRT2
            else:
                nx, ny = 1.0 / SQRT2, -1.0 / SQRT2
                dist = (self.cx - self.cy - line_c) / SQRT2
            side = 1.0 if dist >= 0 else -1.0
            vn = self.vx * nx + self.vy * ny
            if vn * side >= 0:
                continue
            tx, ty = -ny, nx
            vt = self.vx * tx + self.vy * ty
            vn_out = -vn * self.cfg.ramp_restitution
            vt_out = vt * self.cfg.ramp_tangential_keep
            self.vx = vn_out * nx + vt_out * tx
            self.vy = vn_out * ny + vt_out * ty
            push = (RAMP_CONTACT - abs(dist)) * side
            self.cx += push * nx
            self.cy += push * ny

    def _apply_portals(self) -> None:
        """Bidirectional teleport: the ball's center entering either portal exits
        at the other. After a teleport both portals stay inert for
        cfg.portal_cooldown_frames physics frames AND until the ball has left
        their vicinity, so the exiting ball is never instantly re-captured."""
        if self.portal_pair is None:
            return
        inside_any = False
        for portal in self.portal_pair:
            x0, y0, x1, y1 = portal.box
            if x0 - 2 <= self.cx <= x1 + 2 and y0 - 2 <= self.cy <= y1 + 2:
                inside_any = True
                break
        if not self.portal_clear:
            if self.portal_cooldown <= 0 and not inside_any:
                self.portal_clear = True
            return
        for entry, exit_portal in (self.portal_pair, self.portal_pair[::-1]):
            x0, y0, x1, y1 = entry.box
            if x0 <= self.cx <= x1 and y0 <= self.cy <= y1:
                self.cx, self.cy = exit_portal.center
                speed = math.hypot(self.vx, self.vy)
                if speed > 1e-6:
                    self.cx += (self.vx / speed) * (BALL_RADIUS + 1.0)
                    self.cy += (self.vy / speed) * (BALL_RADIUS + 1.0)
                self.portal_clear = False
                # The knob is in rendered frames; the counter ticks per physics frame.
                self.portal_cooldown = max(0, self.cfg.portal_cooldown_frames * max(1, self.cfg.render_every))
                return

    def _apply_kickers(self) -> None:
        """A kicker fully RESETS the ball when its center enters the (slightly
        padded) box: the ball is teleported to the pad's exact center and given
        the fixed kicker velocity, so the launch is identical no matter how the
        ball arrived. The kicker then stays inert for a grace period (and until
        the ball moves clear)."""
        cap = self.cfg.kicker_capture
        for kicker in self.kickers:
            x0, y0, x1, y1 = kicker["box"]
            cx_c, cy_c = kicker["center"]
            # Enlarged trigger: pad 1px along the travel axis, +-cap across it.
            if kicker["horizontal"]:
                axis_overlap = x0 - 1 <= self.cx <= x1 + 1
                cross_off = abs(self.cy - cy_c)
            else:
                axis_overlap = y0 - 1 <= self.cy <= y1 + 1
                cross_off = abs(self.cx - cx_c)
            if kicker["armed"]:
                if axis_overlap and cross_off <= cap:
                    # Launch from a fixed point (pad center) at a fixed velocity, so
                    # the kick is identical no matter how the ball arrived.
                    self.cx, self.cy = cx_c, cy_c
                    self.vx, self.vy = kicker["vx"], kicker["vy"]
                    self.gravity_free_frames = self.cfg.kicker_gravity_free_frames
                    kicker["armed"] = False
                    kicker["cooldown"] = max(0, self.cfg.kicker_cooldown_frames * max(1, self.cfg.render_every))
                    kicker["white_frames"] = KICKER_FLASH_FRAMES  # fixed white flash on trigger
                    self._kicker_fired = True  # end the frame at this reset state
            elif kicker["cooldown"] <= 0 and not (x0 - 2 <= self.cx <= x1 + 2 and y0 - 2 <= self.cy <= y1 + 2):
                kicker["armed"] = True

    def hot_kickers(self) -> set[int]:
        """Object indices of kickers still flashing white (their boost is active)."""
        return {kicker["index"] for kicker in self.kickers if kicker["white_frames"] > 0}

    # -- main loop --

    def step_frame(self) -> bool:
        """Advance one rendered gameplay frame. Returns True when the flight is over."""
        # The kicker white-flash is a display effect, so it counts down in rendered
        # frames here (not in the per-physics-frame bookkeeping below).
        for kicker in self.kickers:
            if kicker["white_frames"] > 0:
                kicker["white_frames"] -= 1
        for _ in range(max(1, self.cfg.render_every)):
            if self.step_physics_frame():
                return True
        return self.done

    def step_physics_frame(self) -> bool:
        """Advance one physics frame. Returns True when the flight is over."""
        if self.done:
            return True
        cfg = self.cfg
        dt = 1.0 / cfg.substeps
        air = cfg.air_keep**dt
        self._kicker_fired = False
        for _ in range(cfg.substeps):
            if self.gravity_free_frames <= 0:
                self.vy += cfg.gravity * dt
            self.vx *= air
            self.vy *= air
            # Travel this substep's dt in <=1px micro-steps, ALWAYS using the
            # current (possibly just-reflected) velocity, so a mid-substep bounce
            # is honored immediately and a fast ball can't tunnel a wall, ramp,
            # portal, kicker, or the goal between samples.
            remaining = dt
            guard = 0
            while remaining > 1e-9 and guard < MICRO_STEP_GUARD:
                guard += 1
                seg = remaining
                dist = math.hypot(self.vx * seg, self.vy * seg)
                if dist > MAX_MICRO_STEP:
                    seg *= MAX_MICRO_STEP / dist
                self.cx += self.vx * seg
                self.cy += self.vy * seg
                remaining -= seg
                self._collide_solids()
                self._collide_ramps()
                self._apply_portals()
                self._apply_kickers()
                # A kicker fully resets the ball to its pad center + fixed
                # velocity; end the frame here so every kicked ball resumes the
                # NEXT frame from an identical state, regardless of when in this
                # frame it arrived (no leftover sub-frame drift).
                if self._kicker_fired:
                    break
                goal_dx = self.cx - self.goal_center[0]
                goal_dy = self.cy - self.goal_center[1]
                if goal_dx * goal_dx + goal_dy * goal_dy < (BALL_RADIUS + GOAL_RADIUS) ** 2:
                    self.done = True
                    self.hit_goal = True
                    self.end_reason = "goal"
                    return True
            if self._kicker_fired:
                break
        self.frame_count += 1
        if self.portal_cooldown > 0:
            self.portal_cooldown -= 1
        if self.gravity_free_frames > 0:
            self.gravity_free_frames -= 1
        for kicker in self.kickers:
            if kicker["cooldown"] > 0:
                kicker["cooldown"] -= 1
        if self._out_of_bounds():
            self.done = True
            self.end_reason = "out"
            return True
        if math.hypot(self.vx, self.vy) < cfg.rest_speed:
            self.slow_frames += 1
        else:
            self.slow_frames = 0
        if self.slow_frames >= cfg.rest_frames:
            self.done = True
            self.end_reason = "rest"
        elif self.frame_count >= cfg.max_frames:
            self.done = True
            self.end_reason = "timeout"
        return self.done

    def _out_of_bounds(self) -> bool:
        """The ball escaped through a disabled border and cannot come back.

        Sideways and downward exits are final immediately; upward exits get a
        generous margin because gravity usually brings the ball back in.
        """
        if self.cx + BALL_RADIUS < FIELD_LEFT - 2 or self.cx - BALL_RADIUS > FIELD_RIGHT + 3:
            return True
        if self.cy - BALL_RADIUS > FIELD_BOTTOM + 3:
            return True
        return self.cy + BALL_RADIUS < FIELD_TOP - 32


def simulate_level(cfg: PhysicsConfig, level_spec: dict, placements: list[dict] | None = None) -> FlightResult:
    """Pure simulation entry point shared by the game, the DSL tooling and the editor.

    level_spec follows the LEVELS_JSON level schema. placements optionally place
    tray objects: [{"object": <index into level objects>, "pos": [x, y]}].
    """
    positions: dict[int, tuple[int, int] | None] = {}
    for idx, obj in enumerate(level_spec["objects"]):
        positions[idx] = None if obj.get("tray") else (int(obj["pos"][0]), int(obj["pos"][1]))
    for placement in placements or []:
        positions[int(placement["object"])] = (int(placement["pos"][0]), int(placement["pos"][1]))

    sim_objects = []
    for idx, obj in enumerate(level_spec["objects"]):
        pos = positions[idx]
        if pos is None:
            continue
        kind = OBJECT_KINDS[obj["kind"]]
        w, h = object_dims(kind, obj.get("size"))
        sim_objects.append(SimObject(kind, pos[0], pos[1], w, h))

    start = (level_spec["ball_start"][0] + BALL_RADIUS, level_spec["ball_start"][1] + BALL_RADIUS)
    velocity = (float(level_spec["launch_velocity"][0]), float(level_spec["launch_velocity"][1]))
    goal = (int(level_spec["goal"][0]), int(level_spec["goal"][1]))

    sim = FlightSim(cfg, start, velocity, sim_objects, goal, borders=level_spec.get("borders"))
    frames: list[tuple[float, float]] = []
    # Full physics-frame resolution: the editor preview and solution checks see
    # every physics frame even though gameplay renders only every render_every-th.
    while not sim.step_physics_frame():
        frames.append((sim.cx, sim.cy))
    frames.append((sim.cx, sim.cy))
    return FlightResult(frames=frames, hit_goal=sim.hit_goal, end_reason=sim.end_reason)


# ---------------------------------------------------------------------------
# Level data. Edited by dev/ball_bounce_editor.py — keep these plain literals.
# ---------------------------------------------------------------------------
# >>> LEVEL_SPECS_START >>>
PHYSICS_SPEC = {
    "air_keep": 0.995,
    "bouncer_min_exit_speed": 2.2,
    "bouncer_restitution": 1,
    "energy": 20,
    "gravity": 0.1,
    "kicker_gravity_free_frames": 5,
    "kicker_speed": 2,
    "max_frames": 200,
    "ramp_restitution": 0.6,
    "ramp_tangential_keep": 0.95,
    "rest_frames": 10,
    "rest_speed": 0.2,
    "restitution": 0.6,
    "sticky_min_speed": 0.3,
    "sticky_restitution": 0,
    "sticky_tangential_keep": 0.6,
    "substeps": 5,
    "tangential_keep": 0.95,
}

LEVEL_SPECS = [
    {
        "ball_start": [4, 5],
        "goal": [51, 24],
        "launch_velocity": [0, 0],
        "name": "Bend It",
        "objects": [{"kind": "ramp_b", "tray": True}, {"kind": "bouncer", "tray": True}],
        "solution": [{"object": 0, "pos": [3, 27]}, {"object": 1, "pos": [29, 50]}],
    },
    {
        "ball_start": [4, 2],
        "borders": {"floor": False, "left": True, "right": False, "top": True},
        "goal": [46, 46],
        "launch_velocity": [0, 0],
        "name": "Sticky",
        "objects": [
            {"kind": "ramp_b", "tray": True},
            {"kind": "bouncer", "tray": True},
            {"kind": "wall_box", "pos": [38, 38], "size": 8},
            {"kind": "wall_box", "pos": [38, 29], "size": 8},
            {"kind": "wall_v", "pos": [47, 2]},
            {"kind": "wall_v", "pos": [47, 11]},
            {"kind": "sticky", "pos": [38, 26]},
            {"kind": "wall_v", "pos": [47, 1]},
            {"kind": "wall_v", "pos": [47, 5]},
            {"kind": "wall_box", "pos": [38, 32], "size": 8},
            {"kind": "wall_v", "pos": [38, 44]},
            {"kind": "wall_v", "pos": [41, 44]},
            {"kind": "wall_box", "pos": [40, 46], "size": 1},
            {"kind": "wall_box", "pos": [40, 47], "size": 1},
            {"kind": "wall_box", "pos": [40, 49], "size": 1},
            {"kind": "wall_box", "pos": [40, 48], "size": 1},
            {"kind": "wall_box", "pos": [40, 50], "size": 1},
            {"kind": "wall_box", "pos": [40, 51], "size": 1},
            {"kind": "wall_v", "pos": [47, 14]},
            {"kind": "wall_h", "pos": [14, 26]},
        ],
        "solution": [{"object": 0, "pos": [3, 15]}, {"object": 1, "pos": [14, 23]}],
    },
    {
        "ball_start": [2, 2],
        "borders": {"floor": False, "left": False, "right": False, "top": False},
        "goal": [1, 43],
        "launch_velocity": [0.7, 0],
        "name": "Portal Drop",
        "objects": [
            {"kind": "ramp_b", "tray": True},
            {"kind": "bouncer", "tray": True},
            {"kind": "sticky", "tray": True},
            {"kind": "bouncer", "pos": [11, 14]},
            {"kind": "bouncer", "pos": [2, 14]},
            {"kind": "portal_b", "pos": [2, 21]},
            {"kind": "portal_a", "pos": [53, 18]},
            {"kind": "wall_h", "pos": [2, 17]},
            {"kind": "wall_h", "pos": [8, 17]},
            {"kind": "wall_h", "pos": [11, 17]},
            {"kind": "wall_v", "pos": [17, 17]},
            {"kind": "wall_v", "pos": [17, 20]},
            {"kind": "wall_h", "pos": [1, 17]},
            {"kind": "wall_h", "pos": [1, 29]},
            {"kind": "wall_h", "pos": [2, 29]},
            {"kind": "wall_v", "pos": [17, 26]},
            {"kind": "wall_v", "pos": [17, 29]},
            {"kind": "wall_v", "pos": [17, 32]},
            {"kind": "wall_v", "pos": [17, 35]},
            {"kind": "wall_v", "pos": [17, 38]},
            {"kind": "wall_v", "pos": [17, 41]},
            {"kind": "wall_v", "pos": [17, 44]},
            {"kind": "wall_h", "pos": [11, 50]},
            {"kind": "wall_h", "pos": [5, 50]},
            {"kind": "wall_h", "pos": [2, 50]},
            {"kind": "wall_h", "pos": [1, 50]},
            {"kind": "wall_box", "pos": [17, 2], "size": 2},
            {"kind": "wall_v", "pos": [56, 14]},
            {"kind": "wall_v", "pos": [56, 20]},
            {"kind": "wall_box", "pos": [53, 26], "size": 2},
            {"kind": "wall_h", "pos": [50, 14]},
            {"kind": "wall_box", "pos": [55, 26], "size": 1},
            {"kind": "wall_box", "pos": [55, 27], "size": 1},
            {"kind": "wall_h", "pos": [38, 50]},
            {"kind": "wall_box", "pos": [17, 5], "size": 2},
            {"kind": "wall_box", "pos": [18, 4], "size": 1},
            {"kind": "wall_box", "pos": [17, 4], "size": 1},
        ],
        "solution": [{"object": 1, "pos": [38, 47]}, {"object": 2, "pos": [11, 11]}, {"object": 0, "pos": [21, 15]}],
    },
    {
        "ball_start": [16, 5],
        "borders": {"floor": False, "left": True, "right": True, "top": False},
        "goal": [54, 27],
        "launch_velocity": [-0.4, 0],
        "name": "Kick Start",
        "objects": [
            {"kind": "ramp_a", "tray": True},
            {"kind": "ramp_b", "tray": True},
            {"kind": "bouncer", "tray": True},
            {"kind": "kicker_u", "tray": True},
            {"kind": "kicker_r", "tray": True},
            {"kind": "wall_h", "pos": [1, 2]},
            {"kind": "wall_h", "pos": [8, 2]},
            {"kind": "wall_h", "pos": [17, 2]},
            {"kind": "wall_h", "pos": [14, 2]},
            {"kind": "wall_v", "pos": [23, 44]},
            {"kind": "wall_v", "pos": [23, 32]},
            {"kind": "wall_v", "pos": [23, 26]},
            {"kind": "wall_v", "pos": [23, 20]},
            {"kind": "wall_v", "pos": [23, 14]},
            {"kind": "wall_v", "pos": [23, 8]},
            {"kind": "wall_v", "pos": [23, 2]},
            {"kind": "wall_h", "pos": [54, 35]},
            {"kind": "wall_h", "pos": [50, 35]},
            {"kind": "wall_v", "pos": [50, 29]},
            {"kind": "wall_v", "pos": [50, 23]},
            {"kind": "wall_v", "pos": [50, 20]},
            {"kind": "wall_h", "pos": [32, 35]},
            {"kind": "wall_v", "pos": [50, 17]},
            {"kind": "wall_v", "pos": [50, 14]},
            {"kind": "wall_box", "pos": [44, 29], "size": 8},
            {"kind": "wall_box", "pos": [38, 29], "size": 8},
            {"kind": "wall_box", "pos": [35, 29], "size": 8},
            {"kind": "wall_box", "pos": [8, 32], "size": 2},
        ],
        "solution": [
            {"object": 4, "pos": [10, 40]},
            {"object": 1, "pos": [3, 39]},
            {"object": 3, "pos": [28, 40]},
            {"object": 0, "pos": [27, 15]},
            {"object": 2, "pos": [35, 26]},
        ],
    },
    {
        "ball_start": [2, 4],
        "borders": {"floor": False, "left": True, "right": True, "top": True},
        "goal": [54, 33],
        "launch_velocity": [0.9, 0],
        "name": "Chain Reaction",
        "objects": [
            {"kind": "ramp_a", "tray": True},
            {"kind": "ramp_b", "tray": True},
            {"kind": "bouncer", "tray": True},
            {"kind": "sticky", "tray": True},
            {"kind": "wall_v", "pos": [26, 20]},
            {"kind": "wall_v", "pos": [23, 11]},
            {"kind": "wall_h", "pos": [23, 8]},
            {"kind": "portal_b", "pos": [26, 12]},
            {"kind": "portal_a", "tray": True},
            {"kind": "kicker_d", "tray": True},
            {"kind": "wall_v", "pos": [23, 8]},
            {"kind": "wall_v", "pos": [23, 14]},
            {"kind": "wall_v", "pos": [50, 26]},
            {"kind": "wall_v", "pos": [50, 29]},
            {"kind": "wall_v", "pos": [50, 32]},
            {"kind": "wall_v", "pos": [41, 44]},
            {"kind": "wall_v", "pos": [41, 38]},
            {"kind": "wall_v", "pos": [41, 32]},
            {"kind": "wall_v", "pos": [41, 26]},
            {"kind": "wall_v", "pos": [41, 20]},
            {"kind": "wall_v", "pos": [41, 14]},
            {"kind": "wall_v", "pos": [41, 8]},
            {"kind": "wall_box", "pos": [26, 35], "size": 2},
            {"kind": "wall_h", "pos": [54, 26]},
            {"kind": "wall_h", "pos": [53, 26]},
            {"kind": "wall_h", "pos": [50, 26]},
            {"kind": "wall_box", "pos": [25, 20], "size": 1},
            {"kind": "wall_box", "pos": [25, 21], "size": 1},
            {"kind": "wall_box", "pos": [59, 38], "size": 2},
            {"kind": "wall_box", "pos": [61, 38], "size": 1},
            {"kind": "wall_box", "pos": [61, 39], "size": 1},
            {"kind": "wall_h", "pos": [41, 8]},
            {"kind": "wall_h", "pos": [47, 8]},
            {"kind": "wall_h", "pos": [53, 8]},
            {"kind": "wall_h", "pos": [54, 8]},
            {"kind": "wall_box", "pos": [59, 50], "size": 2},
            {"kind": "wall_box", "pos": [44, 50], "size": 2},
            {"kind": "wall_box", "pos": [43, 50], "size": 1},
            {"kind": "wall_box", "pos": [43, 51], "size": 1},
            {"kind": "wall_box", "pos": [61, 50], "size": 1},
            {"kind": "wall_box", "pos": [61, 51], "size": 1},
            {"kind": "wall_box", "pos": [41, 52], "size": 1},
            {"kind": "wall_box", "pos": [42, 52], "size": 1},
            {"kind": "wall_box", "pos": [43, 52], "size": 1},
            {"kind": "wall_box", "pos": [44, 52], "size": 1},
            {"kind": "wall_box", "pos": [45, 52], "size": 1},
            {"kind": "wall_box", "pos": [46, 52], "size": 1},
            {"kind": "wall_box", "pos": [47, 52], "size": 1},
            {"kind": "wall_box", "pos": [49, 52], "size": 1},
            {"kind": "wall_box", "pos": [48, 52], "size": 1},
            {"kind": "wall_box", "pos": [50, 52], "size": 1},
            {"kind": "wall_box", "pos": [51, 52], "size": 1},
            {"kind": "wall_box", "pos": [52, 52], "size": 1},
            {"kind": "wall_box", "pos": [53, 52], "size": 1},
            {"kind": "wall_box", "pos": [54, 52], "size": 1},
            {"kind": "wall_box", "pos": [55, 52], "size": 1},
            {"kind": "wall_box", "pos": [56, 52], "size": 1},
            {"kind": "wall_box", "pos": [57, 52], "size": 1},
            {"kind": "wall_box", "pos": [58, 52], "size": 1},
            {"kind": "wall_box", "pos": [59, 52], "size": 1},
            {"kind": "wall_box", "pos": [60, 52], "size": 1},
            {"kind": "wall_box", "pos": [61, 52], "size": 1},
            {"kind": "wall_box", "pos": [52, 51], "size": 1},
            {"kind": "wall_box", "pos": [52, 50], "size": 1},
            {"kind": "wall_box", "pos": [53, 50], "size": 1},
            {"kind": "wall_box", "pos": [53, 51], "size": 1},
            {"kind": "wall_box", "pos": [51, 50], "size": 1},
            {"kind": "wall_box", "pos": [51, 51], "size": 1},
            {"kind": "wall_box", "pos": [50, 40], "size": 1},
            {"kind": "wall_box", "pos": [51, 40], "size": 1},
            {"kind": "wall_box", "pos": [59, 41], "size": 2},
            {"kind": "wall_box", "pos": [59, 40], "size": 1},
            {"kind": "wall_box", "pos": [60, 40], "size": 1},
            {"kind": "wall_box", "pos": [61, 40], "size": 1},
            {"kind": "wall_box", "pos": [61, 41], "size": 1},
            {"kind": "wall_box", "pos": [61, 42], "size": 1},
        ],
        "solution": [
            {"object": 1, "pos": [45, 45]},
            {"object": 3, "pos": [23, 32]},
            {"object": 0, "pos": [54, 45]},
            {"object": 2, "pos": [32, 32]},
            {"object": 8, "pos": [53, 12]},
            {"object": 9, "pos": [46, 19]},
        ],
    },
    {
        "ball_start": [1, 8],
        "borders": {"floor": False, "left": True, "right": True, "top": False},
        "goal": [54, 12],
        "launch_velocity": [1.1, 0],
        "name": "Moonshot",
        "objects": [
            {"kind": "ramp_a", "tray": True},
            {"kind": "ramp_b", "tray": True},
            {"kind": "bouncer", "tray": True},
            {"kind": "wall_h", "pos": [20, 2]},
            {"kind": "wall_box", "pos": [29, 50], "size": 2},
            {"kind": "wall_v", "pos": [29, 35]},
            {"kind": "wall_v", "pos": [29, 29]},
            {"kind": "wall_v", "pos": [29, 23]},
            {"kind": "wall_v", "pos": [29, 17]},
            {"kind": "wall_v", "pos": [29, 11]},
            {"kind": "wall_v", "pos": [29, 5]},
            {"kind": "wall_v", "pos": [29, 2]},
            {"kind": "wall_h", "pos": [44, 2]},
            {"kind": "wall_h", "pos": [38, 2]},
            {"kind": "wall_h", "pos": [32, 2]},
            {"kind": "wall_h", "pos": [26, 2]},
            {"kind": "wall_h", "pos": [53, 26]},
            {"kind": "wall_h", "pos": [54, 26]},
            {"kind": "kicker_r", "pos": [22, 46]},
            {"kind": "wall_v", "pos": [29, 38]},
            {"kind": "wall_box", "pos": [26, 38], "size": 2},
            {"kind": "wall_box", "pos": [28, 38], "size": 1},
            {"kind": "wall_box", "pos": [28, 39], "size": 1},
            {"kind": "wall_h", "pos": [50, 26]},
            {"kind": "wall_h", "pos": [2, 2]},
            {"kind": "wall_h", "pos": [1, 2]},
        ],
        "solution": [{"object": 2, "pos": [8, 14]}],
    },
]
# <<< LEVEL_SPECS_END <<<


def load_document() -> dict:
    return copy.deepcopy({"physics": PHYSICS_SPEC, "levels": LEVEL_SPECS})


# ---------------------------------------------------------------------------
# Energy bar (rendered as a camera interface overlay along the bottom edge)
# ---------------------------------------------------------------------------
class EnergyBar(RenderableUserDisplay):
    def __init__(self, pip_color: int = COLOR_GOAL, spent_color: int = COLOR_FRAME) -> None:
        self.pip_color = pip_color
        self.spent_color = spent_color
        self.capacity = 0
        self.remaining = 0

    def set_capacity(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self.remaining = self.capacity

    def tick(self) -> int:
        if self.remaining > 0:
            self.remaining -= 1
        return self.remaining

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """One chunk per remaining attempt, along the very bottom of the screen."""
        boxes = self.capacity
        if boxes <= 0:
            return frame
        if boxes <= 21:
            pip_w, gap = 2, 1
        elif boxes <= 32:
            pip_w, gap = 1, 1
        else:
            pip_w, gap = 1, 0
        rows = slice(frame.shape[0] - 2, frame.shape[0])
        for box in range(boxes):
            x0 = box * (pip_w + gap)
            if x0 + pip_w > frame.shape[1]:
                break
            color = self.pip_color if box < self.remaining else self.spent_color
            frame[rows, x0 : x0 + pip_w] = color
        return frame


# ---------------------------------------------------------------------------
# Level construction
# ---------------------------------------------------------------------------
def _chrome_pixels(borders: dict) -> list[list[int]]:
    pixels = np.full((GRID_H, GRID_W), -1, dtype=int)
    # Placement lattice: 1px lines every GRID_PITCH pixels, 2x2 cells between.
    for gx in range(FIELD_LEFT, FIELD_RIGHT + 1, GRID_PITCH):
        pixels[FIELD_TOP : FIELD_BOTTOM + 1, gx] = COLOR_GRID
    for gy in range(FIELD_TOP, FIELD_BOTTOM + 1, GRID_PITCH):
        pixels[gy, FIELD_LEFT : FIELD_RIGHT + 1] = COLOR_GRID
    if borders["top"]:
        pixels[FIELD_TOP - 1, :] = COLOR_FRAME
    if borders["floor"]:
        pixels[FIELD_BOTTOM + 1, :] = COLOR_FRAME
    if borders["left"]:
        pixels[FIELD_TOP - 1 : FIELD_BOTTOM + 2, FIELD_LEFT - 1] = COLOR_FRAME
    if borders["right"]:
        # Fill through the screen edge so no dead column remains.
        pixels[FIELD_TOP - 1 : FIELD_BOTTOM + 2, FIELD_RIGHT + 1 :] = COLOR_FRAME
    pixels[TRAY_TOP:, :] = COLOR_TRAY
    return pixels.tolist()


def tray_slot_position(slot: int, w: int, h: int) -> tuple[int, int]:
    # Slots sit flush under the field so the bottom two rows stay free for the
    # attempts bar.
    sx = TRAY_SLOT_X0 + slot * TRAY_SLOT_W
    return (sx + (8 - w) // 2, TRAY_TOP + (8 - h) // 2)


COLOR_ARROW = 2

_A = COLOR_ARROW
_ARROW_E = [
    [-1, -1, _A, -1, -1],
    [-1, -1, -1, _A, -1],
    [_A, _A, _A, _A, _A],
    [-1, -1, -1, _A, -1],
    [-1, -1, _A, -1, -1],
]
_ARROW_SE = [
    [_A, -1, -1, -1, -1],
    [-1, _A, -1, -1, -1],
    [-1, -1, _A, -1, _A],
    [-1, -1, -1, _A, _A],
    [-1, -1, _A, _A, _A],
]
# Eight compass arrows; index = round(atan2(dy, dx) / 45 deg) mod 8, i.e.
# E, SE, S, SW, W, NW, N, NE. np.rot90 turns counter-clockwise.
ARROW_GLYPHS = [
    np.array(_ARROW_E),
    np.array(_ARROW_SE),
    np.rot90(np.array(_ARROW_E), 3),
    np.rot90(np.array(_ARROW_SE), 3),
    np.rot90(np.array(_ARROW_E), 2),
    np.rot90(np.array(_ARROW_SE), 2),
    np.rot90(np.array(_ARROW_E), 1),
    np.rot90(np.array(_ARROW_SE), 1),
]
ARROW_OFFSETS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def launch_arrow_sprite(ball_start: tuple[int, int], launch: tuple[float, float]) -> Sprite:
    """Small arrow next to the ball showing its initial direction of travel.

    A launched ball points along its launch velocity; a ball with no launch
    velocity just falls under gravity, so it points straight down.
    """
    vx, vy = float(launch[0]), float(launch[1])
    if vx == 0.0 and vy == 0.0:
        idx = 2  # straight down (S): gravity is the only initial force
    else:
        idx = round(math.atan2(vy, vx) / (math.pi / 4)) % 8
    dx, dy = ARROW_OFFSETS[idx]
    cx = ball_start[0] + BALL_SIZE // 2 + dx * 6
    cy = ball_start[1] + BALL_SIZE // 2 + dy * 6
    x = max(FIELD_LEFT, min(cx - 2, FIELD_RIGHT - 4))
    y = max(FIELD_TOP, min(cy - 2, FIELD_BOTTOM - 4))
    return Sprite(pixels=ARROW_GLYPHS[idx].tolist(), name="launch_arrow", x=x, y=y, collidable=False, layer=2)


def _placed_sim_objects(objects_data: list[dict]) -> list[SimObject]:
    return [
        SimObject(OBJECT_KINDS[obj["kind"]], int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"]), index=idx)
        for idx, obj in enumerate(objects_data)
        if not obj["in_tray"]
    ]


def _build_level(spec: dict, index: int) -> Level:
    borders = normalize_borders(spec.get("borders"))
    sprites: list[Sprite] = [
        Sprite(
            pixels=_chrome_pixels(borders),
            name="chrome",
            collidable=False,
            layer=-5,
            tags=["sys_click", "sys_every_pixel"],
        )
    ]

    ball_start = (int(spec["ball_start"][0]), int(spec["ball_start"][1]))
    goal = (int(spec["goal"][0]), int(spec["goal"][1]))
    sprites.append(Sprite(pixels=GOAL_PIXELS, name="goal", x=goal[0], y=goal[1], collidable=False, layer=5))
    sprites.append(
        Sprite(pixels=BALL_PIXELS, name="ball", x=ball_start[0], y=ball_start[1], collidable=False, layer=20)
    )

    objects_data: list[dict] = []
    tray_slot = 0
    for obj_index, obj in enumerate(spec["objects"]):
        kind = OBJECT_KINDS[obj["kind"]]
        in_tray = bool(obj.get("tray"))
        size = obj.get("size")
        if in_tray and size is not None:
            size = min(int(size), WALL_BOX_TRAY_MAX)
        w, h = object_dims(kind, size)
        if in_tray:
            pos = tray_slot_position(tray_slot, w, h)
            tray_slot += 1
        else:
            pos = (int(obj["pos"][0]), int(obj["pos"][1]))
        movable = in_tray and kind.placeable
        objects_data.append(
            {
                "kind": kind.name,
                "x": pos[0],
                "y": pos[1],
                "w": w,
                "h": h,
                "in_tray": in_tray,
                "movable": movable,
                "home": [pos[0], pos[1]],  # tray slot to snap back to when returned
            }
        )
        sprites.append(
            Sprite(
                pixels=kind_pixels(kind, size),
                name=f"obj_{obj_index}",
                x=pos[0],
                y=pos[1],
                collidable=False,
                layer=10,
                tags=["obj", "sys_click"] if movable else ["obj"],
            )
        )

    launch = (float(spec["launch_velocity"][0]), float(spec["launch_velocity"][1]))
    sprites.append(launch_arrow_sprite(ball_start, launch))

    data = {
        "ball_start": list(ball_start),
        "launch": list(launch),
        "goal": list(goal),
        "borders": borders,
        "objects": objects_data,
        "solution": [{"object": int(s["object"]), "pos": list(s["pos"])} for s in spec.get("solution", [])],
    }
    return Level(name=spec.get("name", f"Level {index + 1}"), sprites=sprites, grid_size=(GRID_W, GRID_H), data=data)


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class BallBounce(ARCBaseGame):
    def __init__(self) -> None:
        document = load_document()
        self._physics = PhysicsConfig.from_dict(document.get("physics", {}))
        levels = [_build_level(spec, idx) for idx, spec in enumerate(document["levels"])]
        self._energy_bar = EnergyBar()
        camera = Camera(0, 0, GRID_W, GRID_H, 5, 5, [self._energy_bar])
        super().__init__(
            "ball_bounce", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[5, 6]
        )

    _selected: int | None = None
    _flight: FlightSim | None = None
    _ball_needs_reset: bool = False
    _pending_loss: bool = False
    _flash: dict | None = None
    # The single in-progress per-frame animation: a bound advancer method, or
    # None. Both the launch flight and the illegal-click flash route through it,
    # so step() needs one guard and a new animation costs no new step() branch.
    _anim = None
    _hot_kickers: frozenset[int] = frozenset()  # kicker objects currently drawn white

    def on_set_level(self, _level: Level) -> None:
        self._energy_bar.set_capacity(self._physics.energy)
        self._selected = None
        self._flight = None
        self._ball_needs_reset = False
        self._pending_loss = False
        self._flash = None
        self._anim = None
        self._hot_kickers = frozenset()

    # -- engine hooks --

    def step(self) -> None:
        # A running animation (flight or flash) owns the frame until it finishes.
        if self._anim is not None:
            self._anim()
            return

        if self._ball_needs_reset:
            self._reset_ball_sprite()
            if self._pending_loss:
                self._pending_loss = False
                self.lose()
            self.complete_action()
            return

        action = self.action.id
        if action == GameAction.ACTION5:
            if self._energy_bar.remaining <= 0:
                self.lose()
                self.complete_action()
                return
            self._energy_bar.tick()
            self._start_flight()  # sets self._anim
            self._anim()
            return
        if action == GameAction.ACTION6:
            self._handle_click()
            if self._anim is not None:  # a flash started — render its first frame
                self._anim()
                return
        self.complete_action()

    # -- launching --

    def _ball_sprite(self) -> Sprite:
        return self.current_level.get_sprites_by_name("ball")[0]

    def _reset_ball_sprite(self) -> None:
        start = self.current_level.get_data("ball_start")
        sprite = self._ball_sprite()
        sprite.set_position(int(start[0]), int(start[1]))
        sprite.pixels = np.array(BALL_PIXELS, dtype=np.int8)  # undo any bottom-exit clipping
        self._ball_needs_reset = False

    @staticmethod
    def _clip_ball_sprite(sprite: Sprite) -> None:
        """Rows that would draw over the tray are blanked, so a ball falling
        through an open floor sinks out of view instead of crossing the grey
        tray area."""
        pixels = np.array(BALL_PIXELS, dtype=np.int8)
        overhang = sprite.y + BALL_SIZE - TRAY_TOP
        if overhang > 0:
            pixels[BALL_SIZE - min(overhang, BALL_SIZE) :, :] = -1
        sprite.pixels = pixels

    def _rebuild_arrow(self) -> None:
        """The launch arrow hides during a flight and returns afterwards."""
        level = self.current_level
        for old in level.get_sprites_by_name("launch_arrow"):
            level.remove_sprite(old)
        if self._flight is not None:
            return
        start = level.get_data("ball_start")
        launch = level.get_data("launch")
        level.add_sprite(launch_arrow_sprite((int(start[0]), int(start[1])), (launch[0], launch[1])))

    def _start_flight(self) -> None:
        if self._selected is not None:
            self._set_selection(None)
        self._reset_ball_sprite()
        level = self.current_level
        start = level.get_data("ball_start")
        launch = level.get_data("launch")
        goal = level.get_data("goal")
        self._flight = FlightSim(
            self._physics,
            (start[0] + BALL_RADIUS, start[1] + BALL_RADIUS),
            (launch[0], launch[1]),
            _placed_sim_objects(level.get_data("objects")),
            (int(goal[0]), int(goal[1])),
            borders=level.get_data("borders"),
        )
        self._anim = self._advance_flight
        self._rebuild_arrow()  # hides the launch arrow while the ball flies

    def _advance_flight(self) -> None:
        flight = self._flight
        if flight is None:
            return
        done = flight.step_frame()
        sprite = self._ball_sprite()
        sprite.set_position(round(flight.cx - BALL_RADIUS), round(flight.cy - BALL_RADIUS))
        self._clip_ball_sprite(sprite)
        self._sync_kicker_flash(flight.hot_kickers())
        if not done:
            return
        self._reset_kicker_colors()
        self._flight = None
        self._anim = None
        if flight.hit_goal:
            self.next_level()
            self.complete_action()
            return
        self._rebuild_arrow()
        self._ball_needs_reset = True
        if self._energy_bar.remaining <= 0:
            self._pending_loss = True
        self.complete_action()

    def _sync_kicker_flash(self, hot: set[int]) -> None:
        """Draw kickers white while their boost is active. Only touches sprites on
        a change, so a steady (un)lit kicker costs nothing per frame."""
        hot = frozenset(hot)
        if hot == self._hot_kickers:
            return
        for idx in hot - self._hot_kickers:
            self._recolor_kicker(idx, hot=True)
        for idx in self._hot_kickers - hot:
            self._recolor_kicker(idx, hot=False)
        self._hot_kickers = hot

    def _reset_kicker_colors(self) -> None:
        """Restore every flashing kicker to its normal color (e.g. when a flight ends)."""
        for idx in self._hot_kickers:
            self._recolor_kicker(idx, hot=False)
        self._hot_kickers = frozenset()

    def _recolor_kicker(self, idx: int, hot: bool) -> None:
        kind = OBJECT_KINDS[self.current_level.get_data("objects")[idx]["kind"]]
        rows = kind_pixels(kind)
        if hot:
            rows = [[COLOR_KICKER_HOT if c != -1 else -1 for c in row] for row in rows]
        pixels = np.array(rows, dtype=np.int8)
        for sprite in self.current_level.get_sprites_by_name(f"obj_{idx}"):
            sprite.pixels = pixels

    # -- clicking --

    def _handle_click(self) -> None:
        data = self.action.data or {}
        pos = self.camera.display_to_grid(int(data.get("x", -1)), int(data.get("y", -1)))
        if pos is None:
            return
        gx, gy = pos
        hit = self._object_at(gx, gy)

        if self._selected is None:
            if hit is not None:
                self._set_selection(hit)
            return
        # Clicking any other piece (in the tray or placed) switches selection to it.
        if hit is not None and hit != self._selected:
            self._set_selection(hit)
            return
        # Clicking the empty tray returns the held piece to its home slot.
        if gy >= TRAY_TOP:
            self._return_to_tray(self._selected)
            self._set_selection(None)
            return
        self._try_place(gx, gy)

    def _object_at(self, gx: int, gy: int) -> int | None:
        """Topmost movable object whose bounding box (inflated by 1) contains the cell."""
        best: int | None = None
        for idx, obj in enumerate(self.current_level.get_data("objects")):
            if not obj["movable"]:
                continue
            if obj["x"] - 1 <= gx <= obj["x"] + obj["w"] and obj["y"] - 1 <= gy <= obj["y"] + obj["h"]:
                best = idx
        return best

    def _set_selection(self, idx: int | None) -> None:
        level = self.current_level
        for ring in level.get_sprites_by_name("sel_ring"):
            level.remove_sprite(ring)
        self._selected = idx
        if idx is None:
            return
        obj = level.get_data("objects")[idx]
        w, h = obj["w"] + 2, obj["h"] + 2
        ring = np.full((h, w), -1, dtype=int)
        ring[0, :] = COLOR_HIGHLIGHT
        ring[-1, :] = COLOR_HIGHLIGHT
        ring[:, 0] = COLOR_HIGHLIGHT
        ring[:, -1] = COLOR_HIGHLIGHT
        level.add_sprite(
            Sprite(pixels=ring.tolist(), name="sel_ring", x=obj["x"] - 1, y=obj["y"] - 1, collidable=False, layer=30)
        )

    def _placement_valid(self, idx: int, x: int, y: int) -> bool:
        level = self.current_level
        objects = level.get_data("objects")
        obj = objects[idx]
        w, h = obj["w"], obj["h"]
        if not (FIELD_LEFT <= x and x + w - 1 <= FIELD_RIGHT and FIELD_TOP <= y and y + h - 1 <= FIELD_BOTTOM):
            return False
        # Overlap is tested pixel-by-pixel, so a diagonal ramp may sit in a tight
        # corner as long as its band doesn't actually touch another piece.
        placed = occupied_cells(obj["kind"], x, y, w, h)
        blocked: set[tuple[int, int]] = set()
        for other_idx, other in enumerate(objects):
            if other_idx != idx and not other["in_tray"]:
                blocked |= occupied_cells(other["kind"], other["x"], other["y"], other["w"], other["h"])
        # The ball and goal block as solid boxes — nothing may sit on them.
        start = level.get_data("ball_start")
        goal = level.get_data("goal")
        blocked |= {(int(start[0]) + i, int(start[1]) + j) for i in range(BALL_SIZE) for j in range(BALL_SIZE)}
        blocked |= {(int(goal[0]) + i, int(goal[1]) + j) for i in range(GOAL_SIZE) for j in range(GOAL_SIZE)}
        return placed.isdisjoint(blocked)

    def _try_place(self, gx: int, gy: int) -> None:
        idx = self._selected
        if idx is None:
            return
        level = self.current_level
        obj = level.get_data("objects")[idx]

        # Clicks on a lattice line are illegal: flash the clicked pixel plus its
        # orthogonal neighbours that are also on a line (a cross/segment of the
        # lattice), keep holding the piece.
        if on_grid_line(gx, gy):
            pixels = [(gx, gy)]
            for nx, ny in ((gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)):
                if in_field(nx, ny) and on_grid_line(nx, ny):
                    pixels.append((nx, ny))
            self._start_flash(pixels)
            return
        # A click on a wall pixel is illegal too: flash just that pixel.
        if self._wall_pixel_at(gx, gy):
            self._start_flash([(gx, gy)])
            return

        w, h = obj["w"], obj["h"]
        # Center the piece on the clicked cell: even sizes straddle it symmetrically.
        x = snap_to_cell(gx, FIELD_LEFT) + 1 - w // 2
        y = snap_to_cell(gy, FIELD_TOP) + 1 - h // 2

        if not obj["in_tray"] and (x, y) == (obj["x"], obj["y"]):
            self._set_selection(None)  # same cell: just put it down
            return
        if self._placement_valid(idx, x, y):
            obj["x"], obj["y"] = x, y
            obj["in_tray"] = False
            for sprite in level.get_sprites_by_name(f"obj_{idx}"):
                sprite.set_position(x, y)
            self._set_selection(None)
            return
        # Cell is blocked (overlap / off-field): flash the whole cell, keep holding.
        self._start_flash(cell_pixels(cell_origin(gx, FIELD_LEFT), cell_origin(gy, FIELD_TOP)))

    def _return_to_tray(self, idx: int) -> None:
        obj = self.current_level.get_data("objects")[idx]
        obj["x"], obj["y"] = int(obj["home"][0]), int(obj["home"][1])
        obj["in_tray"] = True
        for sprite in self.current_level.get_sprites_by_name(f"obj_{idx}"):
            sprite.set_position(obj["x"], obj["y"])

    def _wall_pixel_at(self, gx: int, gy: int) -> bool:
        """True if (gx, gy) is the playfield frame or a placed solid wall pixel."""
        if not in_field(gx, gy):
            return True
        for o in self.current_level.get_data("objects"):
            if o["in_tray"] or o["kind"] not in ("wall_h", "wall_v", "wall_box"):
                continue
            if o["x"] <= gx < o["x"] + o["w"] and o["y"] <= gy < o["y"] + o["h"]:
                return True
        return False

    def _start_flash(self, pixels: list[tuple[int, int]]) -> None:
        self._flash = {"pixels": [(int(x), int(y)) for x, y in pixels], "frame": 0, "showing": False}
        self._anim = self._advance_flash

    def _flash_sprite(self, pixels: list[tuple[int, int]]) -> Sprite:
        """A red mask over exactly the given pixels (bounding box, others clear)."""
        x0 = min(p[0] for p in pixels)
        y0 = min(p[1] for p in pixels)
        grid = np.full((max(p[1] for p in pixels) - y0 + 1, max(p[0] for p in pixels) - x0 + 1), -1, dtype=int)
        for px, py in pixels:
            grid[py - y0, px - x0] = COLOR_FLASH
        return Sprite(pixels=grid.tolist(), name="click_flash", x=x0, y=y0, collidable=False, layer=40)

    def _set_flash_visible(self, show: bool) -> None:
        """Add/remove the red overlay only on a state change (no per-frame churn)."""
        flash = self._flash
        if show == flash["showing"]:
            return
        for old in self.current_level.get_sprites_by_name("click_flash"):
            self.current_level.remove_sprite(old)
        if show:
            self.current_level.add_sprite(self._flash_sprite(flash["pixels"]))
        flash["showing"] = show

    def _advance_flash(self) -> None:
        """Blink the illegal-click region red / original / red / original, each
        phase held for FLASH_PHASE_FRAMES rendered frames so it reads clearly."""
        flash = self._flash
        phase = flash["frame"] // FLASH_PHASE_FRAMES
        flash["frame"] += 1
        if phase >= 4:  # red, orig, red, orig -> done
            self._set_flash_visible(False)
            self._flash = None
            self._anim = None
            self.complete_action()
            return
        self._set_flash_visible(phase % 2 == 0)  # red on phases 0 and 2
