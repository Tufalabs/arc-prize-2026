"""jc26 level editor — headless-testable API plus an ipywidgets UI.

Unlike the talos editor (a paint grid + a separate preview image), this uses a
SINGLE grid: the official game render is shown as one clickable image, and
clicking it paints the tile under the cursor with the current brush, then
re-renders through the real `JC26` renderer. So what you edit is exactly what
the game draws (range rings included).

The module-level functions (`load_levels`, `save_levels`, `render_frame`,
`frame_to_png`, `preview_png`, …) need only numpy + PIL and run headless;
`build_editor` adds ipywidgets + ipyevents and is imported lazily inside it.
"""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
LEVELS_PATH = HERE / "levels.json"
SCALE = 12  # display px per game px (64x64 -> 768x768)


def _load_game():
    """Load the sibling jc26.py by path (the env loader execs games with no
    package context, so we mirror that and import it standalone)."""
    spec = importlib.util.spec_from_file_location("jc26_editor_game", HERE / "jc26.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_jc = _load_game()
GRID, TILE, VIEW, DEFAULT_LIVES = _jc.GRID, _jc.TILE, _jc.VIEW, _jc.DEFAULT_LIVES

# Brush palette: char -> human label. The official renderer turns these into
# pixels; the swatch colours below are only for the toolbar buttons.
LEGEND = [
    (".", "low ground"),
    ("#", "high ground"),
    ("H", "highest ground"),
    ("E", "mover spawn"),
    ("X", "target"),
    ("G", "fixed turret"),
    ("M", "movable turret"),
    ("o", "slot (on)"),
    ("*", "slot (open)"),
]
# rough swatch colours (palette index) per brush, for the toolbar only
_SWATCH = {".": 5, "#": 4, "H": 3, "E": 10, "X": 14, "G": 13, "M": 13, "o": 15, "*": 15}


def _color_map():
    from re_arc.cli import COLOR_MAP

    return COLOR_MAP


def hexcolor(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def load_levels(path=LEVELS_PATH):
    path = pathlib.Path(path)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_levels(levels, path=LEVELS_PATH):
    pathlib.Path(path).write_text(json.dumps(levels, indent=2) + "\n", encoding="utf-8")


def blank_grid():
    """A bordered arena with one spawn and one target so it renders validly.
    Walls default to highest ground (`H`) — high ground is reserved for where
    turrets may go."""
    rows = ["H" * GRID]
    rows += ["H" + "." * (GRID - 2) + "H" for _ in range(GRID - 2)]
    rows.append("H" * GRID)
    rows[1] = "HE" + "." * (GRID - 3) + "H"  # spawn top-left
    rows[GRID - 2] = "H" + "." * (GRID - 3) + "XH"  # target bottom-right
    return rows


def blank_level(name="New level"):
    return {"name": name, "lives": DEFAULT_LIVES, "grid": blank_grid()}


def entities_summary(level):
    flat = "".join(level.get("grid", []))
    n = {ch: flat.count(ch) for ch in "EXGMo*"}
    return (
        f"spawns:{n['E']}  targets:{n['X']}  turrets:{n['G'] + n['M']} "
        f"(fixed {n['G']}, movable {n['M']})  slots:{n['o'] + n['*']} "
        f"(on {n['o']}, open {n['*']})  lives:{level.get('lives', DEFAULT_LIVES)}"
    )


def level_to_spec(level):
    return _jc.LevelSpec(
        name=level.get("name", "level"), rows=tuple(level["grid"]), lives=int(level.get("lives", DEFAULT_LIVES))
    )


def render_frame(level):
    """The official render of a level: build a real JC26 from the spec and return
    its composed plan frame (raises on a malformed grid, e.g. no target). The
    editor shows every turret's range ring (the game only flashes one on place)."""
    game = _jc.JC26(specs=[level_to_spec(level)])
    game.handle_reset()
    game._show_all_ranges = True
    return np.array(game._compose_plan())


def frame_to_png(frame, scale=SCALE):
    arr = np.asarray(frame, dtype=np.uint8)
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, col in _color_map().items():
        rgb[arr == idx] = col
    im = Image.fromarray(rgb, "RGB").resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def preview_png(level, scale=SCALE):
    return frame_to_png(render_frame(level), scale)


def build_editor(levels_path=LEVELS_PATH):
    """Single-grid ipywidgets editor: click the official render to paint.

    Returns the widget; display it by leaving it as a cell's last expression
    (the notebook does `editor_lib.build_editor()`)."""
    import ipywidgets as W
    from ipyevents import Event

    cmap = _color_map()
    levels = load_levels(levels_path) or [blank_level("Level 1")]
    cur = {"i": 0}

    side = f"{VIEW * SCALE}px"  # fixed on-screen size so the PNG isn't CSS-stretched
    img = W.Image(format="png", layout=W.Layout(width=side, height=side, border="1px solid #888"))
    name_w = W.Text(description="name")
    lives_w = W.BoundedIntText(description="lives", min=1, max=9999)
    brush_w = W.ToggleButtons(
        options=[(f"{ch}  {label}", ch) for ch, label in LEGEND], description="brush", style={"button_width": "150px"}
    )
    dropdown = W.Dropdown(description="level")
    new_btn = W.Button(description="New", icon="plus")
    del_btn = W.Button(description="Delete", icon="trash")
    save_btn = W.Button(description="Save", icon="save", button_style="primary")
    info = W.HTML()
    status = W.HTML()

    def level():
        return levels[cur["i"]]

    def paint(tx, ty, ch):
        g = level()["grid"]
        row = list(g[ty])
        row[tx] = ch
        g[ty] = "".join(row)

    def set_cell(tx, ty, ch):
        if ch == "X":  # single target: clear any previous one
            g = level()["grid"]
            g[:] = [r.replace("X", ".") for r in g]
        paint(tx, ty, ch)

    def refresh_image():
        try:
            img.value = preview_png(level())
            status.value = ""
        except Exception as exc:  # malformed mid-edit (e.g. target deleted)
            status.value = f"<span style='color:#c00'>cannot render: {exc}</span>"
        info.value = "<code>" + entities_summary(level()) + "</code>"

    def refresh_meta():
        name_w.value = level().get("name", "")
        lives_w.value = int(level().get("lives", DEFAULT_LIVES))

    def refresh_dropdown():
        cur["loading"] = True  # suppress on_pick while we rebuild options
        dropdown.options = [(f"{i}: {lv.get('name', '?')}", i) for i, lv in enumerate(levels)]
        dropdown.index = cur["i"]
        cur["loading"] = False

    def event_tile(event):
        # Normalise by the element's actual size (robust to any CSS scaling):
        # click fraction across the image -> game pixel -> tile (clamped).
        bw, bh = event.get("boundingRectWidth"), event.get("boundingRectHeight")
        rx, ry = event.get("relativeX"), event.get("relativeY")
        if not bw or not bh or rx is None or ry is None:
            return None
        tx = min(GRID - 1, max(0, int(rx / bw * VIEW) // TILE))
        ty = min(GRID - 1, max(0, int(ry / bh * VIEW) // TILE))
        return tx, ty

    def on_mouse(event):
        # Press-drag-release paints a rectangle (a plain click = a 1x1 box).
        t = event_tile(event)
        if t is None:
            return
        if event.get("type") == "mousedown":
            cur["drag"] = t
            return
        start = cur.pop("drag", t)
        ch = brush_w.value
        if ch == "X":  # target is single-tile: just the release tile
            set_cell(t[0], t[1], "X")
        else:
            x0, x1 = sorted((start[0], t[0]))
            y0, y1 = sorted((start[1], t[1]))
            for ty in range(y0, y1 + 1):
                for tx in range(x0, x1 + 1):
                    paint(tx, ty, ch)
        refresh_image()

    def on_pick(_change):
        if cur.get("loading") or dropdown.index is None:
            return
        cur["i"] = int(dropdown.index)
        refresh_meta()
        refresh_image()

    def on_name(_change):
        level()["name"] = name_w.value
        refresh_dropdown()
        info.value = "<code>" + entities_summary(level()) + "</code>"

    def on_lives(_change):
        level()["lives"] = int(lives_w.value)
        refresh_image()

    def on_new(_b):
        levels.append(blank_level(f"Level {len(levels) + 1}"))
        cur["i"] = len(levels) - 1
        refresh_dropdown()
        refresh_meta()
        refresh_image()

    def on_delete(_b):
        if len(levels) <= 1:
            status.value = "<span style='color:#c00'>can't delete the last level</span>"
            return
        levels.pop(cur["i"])
        cur["i"] = max(0, cur["i"] - 1)
        refresh_dropdown()
        refresh_meta()
        refresh_image()

    def on_save(_b):
        save_levels(levels, levels_path)
        status.value = f"<span style='color:#070'>saved {len(levels)} level(s) to {levels_path.name}</span>"

    Event(
        source=img,
        watched_events=["mousedown", "mouseup"],
        xy_coordinate_system="relative",
        prevent_default_action=True,  # stop the browser's native image-drag ghost
    ).on_dom_event(on_mouse)
    dropdown.observe(on_pick, "index")
    name_w.observe(on_name, "value")
    lives_w.observe(on_lives, "value")
    new_btn.on_click(on_new)
    del_btn.on_click(on_delete)
    save_btn.on_click(on_save)

    # colour the brush buttons after layout (ipywidgets applies styles post-display)
    legend_html = "  ".join(
        f"<span style='background:{hexcolor(cmap[_SWATCH[ch]])};padding:1px 6px'>{ch}</span> {label}"
        for ch, label in LEGEND
    )

    refresh_dropdown()
    refresh_meta()
    refresh_image()
    return W.VBox(
        [
            W.HBox([dropdown, new_btn, del_btn, save_btn]),
            W.HBox([name_w, lives_w]),
            brush_w,
            W.HTML("<small>" + legend_html + "</small>"),
            img,
            info,
            status,
        ]
    )
