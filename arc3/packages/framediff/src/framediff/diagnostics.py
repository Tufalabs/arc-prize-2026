"""Notebook views: frames drawn as images beside the code that edits them.

Development tooling only. Nothing here runs on the path from two frames to
the emitted block.
"""

from __future__ import annotations

import base64
import html
import io
import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from framediff.animation import AnimationDecomposition
from framediff.decompose import Decomposer
from framediff.grid import Grid
from framediff.transforms import WILDCARD, Composite, NoOp, Transform, with_colors

#: The ARC palette, by colour index.
_PALETTE: npt.NDArray[np.uint8] = np.array(
    [
        (255, 255, 255),
        (204, 204, 204),
        (153, 153, 153),
        (102, 102, 102),
        (51, 51, 51),
        (0, 0, 0),
        (229, 58, 163),
        (255, 123, 204),
        (249, 60, 49),
        (30, 147, 255),
        (136, 216, 241),
        (255, 220, 0),
        (255, 133, 27),
        (146, 18, 49),
        (79, 204, 48),
        (163, 86, 214),
    ],
    dtype=np.uint8,
)

#: The outline drawn around marked cells: a colour no ARC palette entry is close to.
_MARK = np.array((0, 229, 255), dtype=np.uint8)

#: How a wildcard cell is drawn: another colour the palette has nothing near.
_WILD = np.array((255, 0, 255), dtype=np.uint8)

# The view carries its own light panel: a notebook's output area may be dark,
# and black captions on it disappear.
_PANEL = "display: inline-block; background: #fff; color: #000; padding: 8px; border-radius: 4px"
_PRE = "font: 12px/1.35 monospace; background: #f6f8fa; color: #000; padding: 6px; margin: 6px 0; white-space: pre"
_CAPTION = "font: 12px monospace; color: #000"
_ARROW = '<div style="font: bold 20px monospace; color: #888; align-self: center; margin: 0 6px">→</div>'
_TABLE = "font: 12px monospace; color: #000; border-collapse: collapse; margin: 6px 0"
_CELL = "padding: 2px 10px 2px 0; text-align: right"


@dataclass
class View:
    """A piece of HTML a notebook displays as itself."""

    html: str

    def _repr_html_(self) -> str:
        """What a notebook displays."""
        return self.html


def frame_image(
    grid: npt.ArrayLike,
    caption: str = "",
    *,
    marked: npt.ArrayLike | None = None,
    scale: int = 5,
    axes: bool = False,
) -> str:
    """A figure of ``grid`` as a PNG, the regions where ``marked`` is true outlined.

    The outline runs only along a marked cell's sides that face an unmarked
    cell or the frame's edge, so a changed region shows as one contour and
    its cells keep their colours. With ``axes``, ticks around the frame mark
    every few cells, columns numbered above and rows to the left, so a
    block's slices can be found in it.
    """
    frame = np.asarray(grid).astype(np.int16)
    colours = np.where((frame < 0)[..., None], _WILD, _PALETTE[np.clip(frame, 0, len(_PALETTE) - 1)])
    pixels = np.repeat(np.repeat(colours, scale, axis=0), scale, axis=1)
    if marked is not None:
        inside = np.pad(np.asarray(marked, dtype=bool), 1)
        for row, col in np.argwhere(inside[1:-1, 1:-1]):
            top, left = row * scale, col * scale
            if not inside[row, col + 1]:
                pixels[top, left : left + scale] = _MARK
            if not inside[row + 2, col + 1]:
                pixels[top + scale - 1, left : left + scale] = _MARK
            if not inside[row + 1, col]:
                pixels[top : top + scale, left] = _MARK
            if not inside[row + 1, col + 2]:
                pixels[top : top + scale, left + scale - 1] = _MARK
    image = Image.fromarray(pixels, "RGB")
    if axes:
        image = _with_axes(image, scale)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<figure style="margin: 4px; text-align: center">'
        f'<img src="data:image/png;base64,{encoded}" style="image-rendering: pixelated; border: 1px solid #ddd"/>'
        f'<figcaption style="{_CAPTION}">{html.escape(caption)}</figcaption></figure>'
    )


#: The axes' ink, and the room their numbers take: above the frame, and to its left.
_AXIS = (90, 90, 90)
_AXIS_TOP = 13
_AXIS_LEFT = 17


def _with_axes(image: Image.Image, scale: int) -> Image.Image:
    """``image``, a frame drawn at ``scale`` pixels a cell, framed with ticks on its four sides at every eighth
    cell boundary (every sixteenth where the cells are too small for the numbers to fit), numbered above for
    columns and to the left for rows: a boundary's number is the index of the cell after it, as a slice reads."""
    step = 8 if scale >= 3 else 16
    font = ImageFont.load_default(size=9)
    canvas = Image.new("RGB", (image.width + _AXIS_LEFT + 4, image.height + _AXIS_TOP + 4), (255, 255, 255))
    canvas.paste(image, (_AXIS_LEFT, _AXIS_TOP))
    draw = ImageDraw.Draw(canvas)
    right, bottom = _AXIS_LEFT + image.width, _AXIS_TOP + image.height
    draw.rectangle((_AXIS_LEFT - 1, _AXIS_TOP - 1, right, bottom), outline=_AXIS)
    for col in range(0, image.width // scale + 1, step):
        x = _AXIS_LEFT + col * scale
        draw.line((x, _AXIS_TOP - 4, x, _AXIS_TOP - 1), fill=_AXIS)
        draw.line((x, bottom, x, bottom + 3), fill=_AXIS)
        draw.text((x, _AXIS_TOP - 4), str(col), fill=_AXIS, font=font, anchor="mb")
    for row in range(0, image.height // scale + 1, step):
        y = _AXIS_TOP + row * scale
        draw.line((_AXIS_LEFT - 4, y, _AXIS_LEFT - 1, y), fill=_AXIS)
        draw.line((right, y, right + 3, y), fill=_AXIS)
        draw.text((_AXIS_LEFT - 5, y), str(row), fill=_AXIS, font=font, anchor="rm")
    return canvas


def show(
    a: npt.ArrayLike,
    transform: Transform,
    b: npt.ArrayLike | None = None,
    *,
    names: tuple[str, str, str] = ("a", "b", "g"),
) -> View:
    """``a``, the transform's result with the cells it changed outlined, and
    ``b`` if given with the cells the result still gets wrong outlined; below
    them the emitted block and its bit cost."""
    start = np.asarray(a)
    wildcards = bool((start == WILDCARD).any())
    result: Grid = transform.transform(start)
    figures = [frame_image(start, names[0]), _ARROW, frame_image(result, "result", marked=result != start)]
    if b is not None:
        target = np.asarray(b)
        wrong = (result != target) & (start != WILDCARD)
        verdict = "matches" if not wrong.any() else f"{int(wrong.sum())} cells differ"
        figures.append(frame_image(target, f"{names[1]}: {verdict}", marked=wrong))
    code = with_colors(transform, start).to_code(*names, wildcards=wildcards)
    return View(
        f'<div style="{_PANEL}">'
        f'<div style="display: flex; flex-wrap: wrap; align-items: flex-start">{"".join(figures)}</div>'
        f'<pre style="{_PRE}">{html.escape(code)}</pre>'
        f'<div style="{_CAPTION}">bit cost {transform.bit_cost():g}</div>'
        "</div>"
    )


@dataclass
class Explanation:
    """One baseline's decomposition toward a target, and how long the search took."""

    name: str
    baseline: Grid
    transform: Transform
    seconds: float

    @property
    def cost(self) -> float:
        """The transform's bit cost."""
        return self.transform.bit_cost()

    @property
    def edits(self) -> int:
        """How many lines of edits the block carries."""
        return len(self.transform.transforms) if isinstance(self.transform, Composite) else 0


def explain_all(
    target: npt.ArrayLike, baselines: dict[str, npt.ArrayLike], decomposer: Decomposer | None = None
) -> list[Explanation]:
    """Decompose every baseline toward ``target``, timed, in the order given."""
    decomposer = decomposer or Decomposer()
    found: list[Explanation] = []
    for name, baseline in baselines.items():
        started = time.perf_counter()
        transform = decomposer.decompose(baseline, target)
        found.append(Explanation(name, np.asarray(baseline, dtype=np.int8), transform, time.perf_counter() - started))
    return found


def cheapest(explanations: list[Explanation]) -> Explanation:
    """The lowest-cost explanation; at equal cost the earliest, which is the caller's priority order."""
    return min(explanations, key=lambda explanation: explanation.cost)


def show_transition(
    target: npt.ArrayLike,
    baselines: dict[str, npt.ArrayLike],
    *,
    name_b: str = "current",
    title: str = "",
    decomposer: Decomposer | None = None,
) -> View:
    """Every baseline beside the target, each search's cost, edits and time, and the cheapest block.

    The baselines are keyed by the names the block is written against. On
    every baseline the cells that differ from the target are outlined, and
    on the target those that differ from the chosen one; a baseline that
    is the same frame as an earlier one says so.
    """
    explanations = explain_all(target, baselines, decomposer)
    chosen = cheapest(explanations)
    current = np.asarray(target, dtype=np.int8)
    figures: list[str] = []
    for i, explanation in enumerate(explanations):
        differs = (explanation.baseline != current) & (explanation.baseline != WILDCARD)
        same = [other.name for other in explanations[:i] if np.array_equal(other.baseline, explanation.baseline)]
        caption = explanation.name + (f" (same frame as {same[0]})" if same else "")
        figures.append(frame_image(explanation.baseline, caption, marked=differs))
    changed = (chosen.baseline != current) & (chosen.baseline != WILDCARD)
    figures += [
        _ARROW,
        frame_image(current, f"{name_b}: {int(changed.sum())} cells differ from {chosen.name}", marked=changed),
    ]
    rows = "".join(
        f'<tr><td style="{_CELL}; text-align: left">{html.escape(explanation.name)}'
        f"{' (shown)' if explanation is chosen else ''}</td>"
        f'<td style="{_CELL}">{int((explanation.baseline != current).sum())}</td>'
        f'<td style="{_CELL}">{explanation.cost:g}</td><td style="{_CELL}">{explanation.edits}</td>'
        f'<td style="{_CELL}">{explanation.seconds * 1000:.1f}</td></tr>'
        for explanation in explanations
    )
    header = "".join(
        f'<th style="{_CELL}">{column}</th>' for column in ("baseline", "cells", "bit cost", "edits", "ms")
    )
    code = with_colors(chosen.transform, chosen.baseline).to_code(
        chosen.name, name_b, wildcards=bool((chosen.baseline == WILDCARD).any())
    )
    return View(
        f'<div style="{_PANEL}">'
        + (f'<div style="{_CAPTION}; font-weight: bold">{html.escape(title)}</div>' if title else "")
        + f'<div style="display: flex; flex-wrap: wrap; align-items: flex-start">{"".join(figures)}</div>'
        f'<table style="{_TABLE}"><tr>{header}</tr>{rows}</table>'
        f'<pre style="{_PRE}">{html.escape(code)}</pre>'
        "</div>"
    )


def show_animation(
    before: npt.ArrayLike,
    frames: list[npt.ArrayLike],
    found: AnimationDecomposition,
    *,
    settled: npt.ArrayLike | None = None,
    title: str = "",
    seconds: float | None = None,
    max_images: int = 24,
    scale: int = 2,
    axes: bool = False,
) -> View:
    """An animation explained: the frame before it, each frame, the settled frame, and the block ``found``.

    Each frame is captioned with what it is built from and the script's
    cost, and the cells that differ from that source are outlined; a frame
    taken as it is is outlined against the frame built before it. Only the
    frames the block builds are drawn, every ``found.stride``-th; past
    ``max_images`` of them the strip stops, and the block is shown whole.
    ``axes`` numbers each frame's rows and columns (:func:`frame_image`).
    """
    start = np.asarray(before, dtype=np.int8)
    grids = [np.asarray(frame, dtype=np.int8) for frame in frames]
    figures = [frame_image(start, "before", scale=scale, axes=axes)]
    shown = found.frames[:max_images]
    for position, step in enumerate(shown):
        k, grid = step.index, grids[step.index]
        if step.cost is None:
            source = grids[shown[position - 1].index] if position else start
            caption = f"{k}: as it is"
        else:
            source = start if step.earlier is None else grids[step.earlier]
            origin = "before" if step.earlier is None else str(step.earlier)
            caption = f"{k}: {'= ' if isinstance(step.transform, NoOp) else 'from '}{origin} ({step.cost:g})"
        figures.append(frame_image(grid, caption, marked=grid != source, scale=scale, axes=axes))
    if len(found.frames) > max_images:
        figures.append(
            f'<div style="{_CAPTION}; align-self: center">… {len(found.frames) - max_images} more frames</div>'
        )
    if settled is not None:
        figures += [_ARROW, frame_image(settled, "settled", scale=scale, axes=axes)]
    built = [step.cost for step in found.frames if step.cost is not None]
    summary = (
        f"{len(grids)} frames, every {found.stride}: {len(built)} built ({sum(built):g} bits), "
        f"{len(found.frames) - len(built)} taken as they are; {len(found.code.splitlines())} lines, "
        f"{len(found.code)} characters" + (f"; {seconds:.1f} s" if seconds is not None else "")
    )
    return View(
        f'<div style="{_PANEL}">'
        + (f'<div style="{_CAPTION}; font-weight: bold">{html.escape(title)}</div>' if title else "")
        + f'<div style="display: flex; flex-wrap: wrap; align-items: flex-start">{"".join(figures)}</div>'
        f'<div style="{_CAPTION}">{html.escape(summary)}</div>'
        f'<pre style="{_PRE}; max-height: 480px; overflow: auto">{html.escape(found.code)}</pre>'
        "</div>"
    )
