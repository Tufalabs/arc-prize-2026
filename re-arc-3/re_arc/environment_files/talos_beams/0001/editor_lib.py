"""Helpers for the Talos Beams level editor (editor.ipynb).

Pure / headless-testable: level JSON I/O, an editor palette, and rendering a
(possibly unsaved) level to a PNG via the real game renderer. The notebook
wires ipywidgets on top of these. Loaded normally (has __file__), so it can
import the game module by path — unlike the env loader.
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


def _load_game():
    spec = importlib.util.spec_from_file_location("talos_beams_editor_mod", HERE / "talos_beams.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["talos_beams_editor_mod"] = mod  # dataclasses need this for string annotations
    spec.loader.exec_module(mod)
    return mod


tb = _load_game()

# re_arc render palette (index -> RGB)
GAME_COLOR_MAP = {
    0: (255, 255, 255),
    1: (204, 204, 204),
    2: (153, 153, 153),
    3: (102, 102, 102),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (229, 58, 163),
    7: (255, 123, 204),
    8: (249, 60, 49),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 133, 27),
    13: (146, 18, 49),
    14: (79, 204, 48),
    15: (163, 86, 214),
}

# editor brush palette: char -> (label, display RGB swatch)
LEGEND = {
    ".": ("floor", (28, 28, 28)),
    "#": ("white wall", (255, 255, 255)),
    "%": ("grey wall", (150, 150, 150)),
    "@": ("player", (163, 86, 214)),
    "X": ("exit", (255, 133, 27)),
    "o": ("connector", (110, 110, 110)),
    "i": ("inverter", (170, 90, 170)),
    "r": ("source R", (150, 40, 40)),
    "g": ("source G", (40, 120, 40)),
    "b": ("source B", (40, 80, 170)),
    "R": ("target R", (249, 60, 49)),
    "G": ("target G", (79, 204, 48)),
    "B": ("target B", (30, 147, 255)),
    "1": ("door 1", (255, 220, 0)),
    "2": ("door 2", (235, 205, 0)),
    "3": ("door 3", (215, 185, 0)),
    "4": ("door 4", (195, 165, 0)),
    "5": ("door 5", (175, 150, 0)),
    "6": ("door 6", (155, 130, 0)),
}


def hexcolor(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def load_levels(path=LEVELS_PATH):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def save_levels(levels, path=LEVELS_PATH):
    pathlib.Path(path).write_text(json.dumps(levels, indent=2) + "\n", encoding="utf-8")


def blank_grid(w=None, h=None):
    """An empty w x h canvas — all floor, nothing placed (paint it yourself)."""
    w = tb.GRID_W if w is None else int(w)
    h = tb.GRID_H if h is None else int(h)
    return ["." * w for _ in range(h)]


def entities_summary(rows, ids=None, rot=None):
    """Human-readable list of entity IDs + positions for wiring (doors/edges)."""
    rot = rot or {}
    arrow = {0: "↓", 1: "←", 2: "↑", 3: "→"}  # down left up right
    parts = []
    for e in tb._scan_entities(rows, ids):
        s = f"{e['id']}@{e['tx'], e['ty']}"
        if e["kind"] == "target":
            s += arrow[rot.get(e["id"], 0) % 4]
        parts.append(s)
    doors = {}
    for ty, row in enumerate(rows):
        for tx, ch in enumerate(row):
            if ch.isdigit():
                doors.setdefault(ch, []).append((tx, ty))
    parts += [f"door {d}@{tiles}" for d, tiles in sorted(doors.items())]
    return "  ·  ".join(parts) if parts else "(no entities yet)"


def level_to_spec(level):
    return tb.LevelSpec(
        name=level.get("name", "Level"),
        rows=tuple(level["grid"]),
        doors=dict(level.get("doors", {})),
        charge_full=int(level.get("charge_full", tb.CHARGE_FULL)),
        edges=tuple(tuple(e) for e in level.get("edges", [])),  # pairs of node ids
        ids=dict(level.get("ids", {})),
        rot=dict(level.get("rot", {})),
        click_hint=bool(level.get("click_hint", False)),
    )


def render_frame(level):
    """Render a (possibly unsaved) level's initial frame. -> (frame, error_str)."""
    try:
        game = tb.TalosBeams(specs=[level_to_spec(level)])
        return np.asarray(game.screen.pixels, dtype=np.int16), None
    except Exception as exc:  # surface any layout error to the editor
        return None, f"{type(exc).__name__}: {exc}"


def frame_to_png(frame, scale=14):
    arr = np.asarray(frame, dtype=np.int16)
    rgb = np.zeros((*arr.shape, 3), np.uint8)
    for idx, col in GAME_COLOR_MAP.items():
        rgb[arr == idx] = col
    img = Image.fromarray(rgb).resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def preview_png(level, scale=14):
    """(png_bytes_or_None, error_str)."""
    frame, err = render_frame(level)
    if err is not None:
        return None, err
    return frame_to_png(frame, scale), None


def build_editor(levels_path=LEVELS_PATH):
    """Build the ipywidgets paint editor. Click a tile to paint it with the
    selected brush; the live preview uses the real game renderer. Save writes
    the grid (and the manual doors/charge_full/edges fields) back to the JSON.
    The notebook just does `editor_lib.build_editor()`.
    """
    import ipywidgets as W

    state = {"levels": load_levels(levels_path), "grid": [["#"]], "ids": {}, "rot": {}}

    brush = W.ToggleButtons(
        options=[(f"{lbl} [{ch}]", ch) for ch, (lbl, _c) in LEGEND.items()] + [("↻ rotate target", "ROT")],
        value="#",
        style={"button_width": "auto"},
    )
    level_dd = W.Dropdown(description="Level:")
    name_tx = W.Text(description="Name:")
    cf_tx = W.IntText(description="charge_full:", value=3, layout=W.Layout(width="170px"))
    click_hint_cb = W.Checkbox(description="click hint", value=False, indent=False, layout=W.Layout(width="120px"))
    doors_tx = W.Text(
        description="doors:",
        layout=W.Layout(width="46%"),
        placeholder='{"1": ["R1"], "2": ["R1","G1"]}  (door digit -> target IDs; bare colour ok)',
    )
    edges_tx = W.Textarea(
        description="edges:",
        layout=W.Layout(width="46%", height="56px"),
        placeholder='[["r1","o1"],["o1","B1"]]  (initial wires by node ID)',
    )
    examples_html = W.HTML(
        "<small><b>Examples</b> (need not match this level) &nbsp; "
        '<b>doors</b>: <code>{"1": ["R1"], "2": ["R1","G1"]}</code> '
        "— door 1 opens when R1 is full; door 2 needs R1 <i>and</i> G1; "
        '<code>{"3": {"any": ["R2","R3"]}}</code> — R2 <i>or</i> R3 &nbsp;|&nbsp; '
        '<b>edges</b>: <code>[["r1","o1"],["o1","B1"]]</code> '
        "— pre-wire source r1 → connector o1 → target B1</small>"
    )
    new_btn = W.Button(description="New 12x12", icon="plus")
    save_btn = W.Button(description="Save levels.json", button_style="success", icon="save")
    del_btn = W.Button(description="Delete level", button_style="danger", icon="trash")
    status = W.HTML()
    entities_html = W.HTML()
    preview = W.Image(format="png")
    grid_box = W.GridBox()

    def color_of(ch):
        return hexcolor(LEGEND.get(ch, (28, 28, 28))[1])

    def level_dict():
        try:
            doors = json.loads(doors_tx.value or "{}")
        except json.JSONDecodeError:
            doors = "ERR"
        try:
            edges = json.loads(edges_tx.value or "[]")
        except json.JSONDecodeError:
            edges = "ERR"
        return {
            "name": name_tx.value or "Level",
            "grid": ["".join(r) for r in state["grid"]],
            "doors": {} if doors == "ERR" else doors,
            "charge_full": int(cf_tx.value or 3),
            "edges": [] if edges == "ERR" else edges,
            "ids": dict(state["ids"]),
            "rot": dict(state["rot"]),
            "click_hint": bool(click_hint_cb.value),
            "_doors_ok": doors != "ERR",
            "_edges_ok": edges != "ERR",
        }

    def refresh_preview():
        lv = level_dict()
        entities_html.value = "<small><b>IDs:</b> " + entities_summary(lv["grid"], lv["ids"], lv["rot"]) + "</small>"
        warn = ""
        if not lv.pop("_doors_ok"):
            warn += " · doors JSON invalid"
        if not lv.pop("_edges_ok"):
            warn += " · edges JSON invalid"
        png, err = preview_png(lv)
        if err is not None:
            status.value = f"<b style='color:#c0392b'>{err}{warn}</b>"
        else:
            preview.value = png
            status.value = f"<span style='color:#27ae60'>ok</span>{warn}"

    def cell_label(ch, col, row):
        if ch in "rgbRGBoi":
            return state["ids"].get(f"{col},{row}", ch)  # the stable entity ID (e.g. "B1")
        if ch.isdigit() or ch in "@X":
            return ch  # door digit / player / exit
        return ""  # floor + walls: colour conveys it

    def relabel():
        for b in grid_box.children:
            r, c = b.rc
            b.description = cell_label(state["grid"][r][c], c, r)

    def on_cell(btn):
        r, c = btn.rc
        key = f"{c},{r}"
        if brush.value == "ROT":  # cycle a target's rotation
            tid = state["ids"].get(key)
            if tid and tid[0] in "RGB":
                state["rot"][tid] = (state["rot"].get(tid, 0) + 1) % 4
            refresh_preview()
            return
        ch = brush.value
        state["grid"][r][c] = ch
        if ch in "rgbRGBoi":
            cur = state["ids"].get(key)
            if not (cur and cur[0] == ch):  # new placement -> lowest-free id (reuses freed ones)
                if cur:
                    state["rot"].pop(cur, None)
                used = set(state["ids"].values())
                n = 1
                while f"{ch}{n}" in used:
                    n += 1
                state["ids"][key] = f"{ch}{n}"
        else:
            old = state["ids"].pop(key, None)  # deleting frees the id; others keep theirs
            if old:
                state["rot"].pop(old, None)
        btn.style.button_color = color_of(ch)
        relabel()
        refresh_preview()

    def build_grid():
        grid = state["grid"]
        h, w = len(grid), len(grid[0])
        cells = []
        for r in range(h):
            for c in range(w):
                b = W.Button(description="", layout=W.Layout(width="26px", height="26px", margin="0", padding="0"))
                b.style.button_color = color_of(grid[r][c])
                b.style.font_size = "9px"
                b.rc = (r, c)
                b.on_click(on_cell)
                cells.append(b)
        grid_box.layout.grid_template_columns = f"repeat({w}, 26px)"
        grid_box.layout.grid_gap = "1px"
        grid_box.children = cells
        relabel()
        refresh_preview()

    def load_level(level):
        # normalize to exactly GRID_W x GRID_H so the editor is always 12x12
        rows = [(r + "#" * tb.GRID_W)[: tb.GRID_W] for r in list(level["grid"])[: tb.GRID_H]]
        while len(rows) < tb.GRID_H:
            rows.append("#" * tb.GRID_W)
        state["grid"] = [list(r) for r in rows]
        # stable id map: use the saved one, else derive once from the grid
        if level.get("ids"):
            state["ids"] = dict(level["ids"])
        else:
            state["ids"] = {f"{e['tx']},{e['ty']}": e["id"] for e in tb._scan_entities(["".join(r) for r in rows])}
        state["rot"] = dict(level.get("rot", {}))
        name_tx.value = level.get("name", "Level")
        cf_tx.value = int(level.get("charge_full", 3))
        click_hint_cb.value = bool(level.get("click_hint", False))
        doors_tx.value = json.dumps(level.get("doors", {}))
        edges_tx.value = json.dumps(level.get("edges", []))
        build_grid()

    def refresh_dropdown():
        level_dd.options = [(lv["name"], i) for i, lv in enumerate(state["levels"])] + [("+ new", -1)]

    def on_pick(change):
        if change["new"] is None:
            return
        if change["new"] == -1:
            load_level(
                {"name": "New", "grid": blank_grid(), "doors": {}, "charge_full": int(cf_tx.value or 3), "edges": []}
            )
        else:
            load_level(state["levels"][change["new"]])

    def on_new(_b):
        load_level(
            {
                "name": name_tx.value or "New",
                "grid": blank_grid(),
                "doors": {},
                "charge_full": int(cf_tx.value or 3),
                "edges": [],
            }
        )

    def load_first_or_blank():
        if state["levels"]:
            load_level(state["levels"][0])
            level_dd.value = 0
        else:
            load_level(
                {
                    "name": "Level 1",
                    "grid": blank_grid(),
                    "doors": {},
                    "charge_full": int(cf_tx.value or 3),
                    "edges": [],
                }
            )
            level_dd.value = -1

    def on_delete(_b):
        names = [x["name"] for x in state["levels"]]
        if name_tx.value in names:
            del state["levels"][names.index(name_tx.value)]
            save_levels(state["levels"], levels_path)
        refresh_dropdown()
        load_first_or_blank()
        status.value = f"<b style='color:#c0392b'>deleted; {len(state['levels'])} level(s) left</b>"

    def on_save(_b):
        lv = level_dict()
        lv.pop("_doors_ok")
        lv.pop("_edges_ok")
        names = [x["name"] for x in state["levels"]]
        if lv["name"] in names:
            state["levels"][names.index(lv["name"])] = lv
        else:
            state["levels"].append(lv)
        save_levels(state["levels"], levels_path)
        state["levels"] = load_levels(levels_path)
        refresh_dropdown()
        status.value = f"<b style='color:#27ae60'>saved {len(state['levels'])} level(s) -> levels.json</b>"

    level_dd.observe(on_pick, names="value")
    new_btn.on_click(on_new)
    save_btn.on_click(on_save)
    del_btn.on_click(on_delete)
    for w in (name_tx, cf_tx, doors_tx, edges_tx, click_hint_cb):
        w.observe(lambda _c: refresh_preview(), names="value")

    refresh_dropdown()
    load_first_or_blank()

    legend_html = " &nbsp; ".join(
        f"<span style='background:{hexcolor(c)};color:#888;padding:0 4px'>{ch}</span> {lbl}"
        for ch, (lbl, c) in LEGEND.items()
    )
    return W.VBox(
        [
            W.HTML(
                "<b>Talos Beams level editor</b> (12x12) — pick a brush, click tiles to paint. "
                "Connections (doors/edges) are the JSON fields below, referenced by the IDs shown."
            ),
            W.HBox([level_dd, name_tx, cf_tx, click_hint_cb]),
            W.HBox([new_btn, save_btn, del_btn]),
            brush,
            W.HBox([grid_box, W.VBox([preview, status])]),
            entities_html,
            W.HBox([doors_tx, edges_tx]),
            examples_html,
            W.HTML(f"<small>{legend_html}</small>"),
        ]
    )
