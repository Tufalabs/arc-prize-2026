"""The viewer: pages over span streams, served live or exported
static.

**URL space = file space.** Every canonical diagnostics page has a path
under its run's job_dir:

    <job_dir>/viewer/index.html            run index (games × passes)
    <job_dir>/viewer/g_<run_id>.html       one (game, pass): move strip + summaries
    <job_dir>/viewer/g_<run_id>_m<N>.html  one move: the full chunk tree
    <job_dir>/viewer/png/<sha>.png         image blocks, as PNG

The server (:func:`serve`) is rooted at a directory of runs.  Two
reserved, underscore-prefixed prefixes sit outside the file space:
``/_replay/`` hosts the replay viewer — one puzzle's transcript scrolled
through context by context, beside the puzzle and the answer standing at
the selected point — and ``/_api/`` its JSON adapter over the run
layouts (``platypus.viewer_data``).  No TAAF run
layout produces such names, so nothing on disk is shadowed.  Any other
path matching the viewer scheme is ALWAYS rendered fresh from the JSONL
streams (live viewing — never served stale from disk); any other path is
served as a plain static file, which is what makes TAAF's central
``diagnostics.html`` and its relative links into these pages work in
both directions. :func:`export_viewer` materializes the same canonical
paths with the same renderers for serverless archival — exported files
only answer when no server is in front.

Pagination is a page decision: the PAGES registry (pattern → builder) is
shared by the server and the exporter, so a future page kind (e.g. a
combined per-level page) is one new entry, present in both hosts,
purely additive.
"""

from __future__ import annotations

import base64
import collections
import html
import io
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ullm.html_render

import platypus.live
from platypus import diagnostics
from platypus.chunks import (
    ViewContext,
    chunk_for,
    is_trivial,
    load_producer_chunks,
    peak_context,
)
from platypus.diagnostics import SpanNode, build_span_tree

__all__ = [
    "PAGES",
    "PageSpec",
    "benchmark_finished",
    "export_viewer",
    "render_viewer_path",
    "resolve_interactive_request",
    "resolve_request",
    "serve",
]

_PNG_MAGIC = b"\x89PNG"


def _esc(value: Any) -> str:
    """HTML-escape a value for interpolation into a page."""
    return html.escape(str(value), quote=True)


_CSS = """
body { font-family: system-ui, sans-serif; margin: 1.2em; background: #fafafa; color: #222; }
a { color: #06c; }
table { border-collapse: collapse; }
td, th { border: 1px solid #ccc; padding: 2px 8px; font-size: 0.9em; }
pre { background: #f0f0f0; padding: 6px; overflow-x: auto; white-space: pre-wrap; }
details { margin: 4px 0 4px 0.8em; border-left: 2px solid #ddd; padding-left: 6px; }
summary { cursor: pointer; }
.chunk-title { font-weight: 600; }
.chunk-kind { color: #888; font-size: 0.85em; }
.chunk-outcome { color: #060; }
.chunk-error > summary .chunk-outcome { color: #b00; font-weight: 600; }
.chunk-cancelled > summary .chunk-outcome { color: #b60; font-weight: 600; }
.live-link { color: #c00; font-weight: 600; text-decoration: none; }
.chunk-unclosed > summary .chunk-outcome { color: #b60; }
.chunk-tokens, .chunk-time, .chunk-llm-meta, .ullm-usage { color: #666; font-size: 0.85em; }
.chunk-action { font-family: monospace; }
.chunk-grid, .ullm-image { image-rendering: pixelated; border: 1px solid #999; max-width: 100%; }
.ullm-label { font-size: 0.8em; color: #666; text-transform: uppercase; display: block; }
.ullm-user pre { background: #eef4ff; }
.ullm-tool-result.ullm-error pre { background: #ffecec; }
.ullm-synthetic { color: #b60; }
.ullm-injected { color: #06b; font-size: 0.9em; }
.ullm-injected code { background: #e6f0ff; padding: 1px 4px; }
.ullm-code { background: #f6f8fa; border: 1px solid #ddd; padding: 6px 8px; overflow-x: auto; }
.ullm-arg-name { font-size: 0.8em; color: #666; font-family: monospace; margin-top: 4px; }
.ullm-args { font-size: 0.85em; color: #444; margin: 2px 0; }
.ullm-md :is(h1,h2,h3,h4) { margin: 6px 0 2px; font-size: 1em; }
.ullm-md pre { background: #f6f8fa; border: 1px solid #ddd; padding: 6px 8px; overflow-x: auto; }
.ullm-md code { background: #eef; padding: 0 3px; border-radius: 3px; }
.ullm-md pre code { background: none; padding: 0; }
.ullm-md p { margin: 4px 0; }
.ullm-md table { border-collapse: collapse; }
.ullm-md td, .ullm-md th { border: 1px solid #ccc; padding: 2px 6px; }
.ullm-preview { color: #888; font-weight: normal; font-family: monospace; font-size: 0.85em; }
.ullm-image-box { margin: 4px 0; }
.ullm-image-box figcaption { font-size: 0.8em; color: #666; }
.ullm-prehistory { border-left-color: #bbb; background: #f4f4f4; }
.ullm-tool-def summary { color: #666; }
.move-frames { display: flex; gap: 14px; margin: 6px 0; }
.move-frames figure { margin: 0; }
.move-frames img { width: 176px; height: 176px; }
.move-frames figcaption { font-size: 0.8em; color: #666; text-align: center; }
.frame-thumb { width: 40px; height: 40px; image-rendering: pixelated; border: 1px solid #999; vertical-align: middle; }
.trivial-row td { background: #f4f4f4; padding: 3px; }
.trivial-row a { margin-right: 3px; }
.move-nav a, .move-nav-off { margin: 0 2px; }
.move-nav-off { color: #aaa; }
.move-strip a { display: inline-block; width: 14px; height: 18px; margin: 1px; text-align: center;
  text-decoration: none; color: #222; background: #e5e5e5; border-radius: 2px; font-size: 0.75em; }
.move-strip a.error { background: #f5b5b5; }
.move-strip a.gave_up { background: #f5d9a8; }
.move-strip a.unclosed { background: #f0e68c; }
"""


def _page(title: str, body: str) -> bytes:
    """Wrap body HTML in the shared document shell."""
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{_esc(title)}</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    ).encode()


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def _is_run_dir(path: Path) -> bool:
    """Whether a directory looks like a finished or running job dir."""
    return diagnostics.has_diagnostics(path)


def _game_dirs(job_dir: Path) -> list[Path]:
    """A job dir's per-(game, pass) directories with a stream, loose or
    archived, in stable order. An archived game's dir need not exist: its
    name is the run id every reader keys on."""
    return [job_dir / "games" / run_id for run_id in diagnostics.game_run_ids(job_dir)]


def _game_tree(job_dir: Path, run_id: str) -> SpanNode | None:
    """One game's span tree rebuilt from its JSONL stream, or None if absent."""
    roots = build_span_tree(diagnostics.read_game_events(job_dir, run_id))
    return roots[0] if roots else None


@dataclass
class _GameRow:
    """What the run overview shows of one game's stream."""

    name: str
    closed: bool
    outcome: str | None
    moves: int
    incl_generated: int
    incl_uncached: int


# The run overview's rows by stream version: parsing every stream of a long
# run takes seconds, so a re-listed run parses only the streams that changed.
# Bounded only so a viewer left up across many runs does not grow forever.
_ROWS_MAX = 20000
_ROWS: collections.OrderedDict[tuple[Path, str], tuple[Any, _GameRow | None]] = collections.OrderedDict()
_ROWS_LOCK = threading.Lock()


def _game_row(job_dir: Path, run_id: str) -> _GameRow | None:
    """One game's overview row, or None if it has no stream."""
    key = (job_dir.resolve(), run_id)
    version = diagnostics.job_file_version(job_dir, f"games/{run_id}/spans.jsonl")
    with _ROWS_LOCK:
        cached = _ROWS.get(key)
    if cached is not None and version is not None and cached[0] == version:
        return cached[1]
    root = _game_tree(job_dir, run_id)
    row = (
        None
        if root is None
        else _GameRow(
            name=root.name,
            closed=root.closed,
            outcome=root.outcome,
            moves=sum(1 for child in root.children if child.kind == "infra/move"),
            incl_generated=root.incl_generated,
            incl_uncached=root.incl_uncached,
        )
    )
    with _ROWS_LOCK:
        _ROWS[key] = (version, row)
        _ROWS.move_to_end(key)
        while len(_ROWS) > _ROWS_MAX:
            _ROWS.popitem(last=False)
    return row


class _ImageCollector:
    """image_src for pages: returns the canonical png path and remembers
    which blocks were referenced (the exporter materializes exactly
    those)."""

    def __init__(self) -> None:
        """A collector with no images yet."""
        self.keys: set[str] = set()

    def __call__(self, key: str) -> str:
        """Record an image block and return the URL the page should use."""
        self.keys.add(key)
        return f"png/{key}.png"


def _view(
    job_dir: Path, collector: _ImageCollector, live_href: str | None = None, deterministic: bool = False
) -> ViewContext:
    """A ViewContext over a job dir, wired to collect image URLs."""
    return ViewContext(
        blocks=diagnostics.job_block_store(job_dir),
        image_src=collector,
        live_href=live_href,
        deterministic=deterministic,
    )


def _dur(duration_s: float | None, deterministic: bool, *, fmt: str = ".0f") -> str:
    """Wall-clock duration suffix, or nothing in deterministic mode (so the
    HTML reproduces across runs — see ViewContext.deterministic)."""
    if duration_s is None or deterministic:
        return ""
    return f" — {format(duration_s, fmt)}s"


def block_png_bytes(job_dir: Path, key: str) -> bytes:
    """An image block as PNG bytes: png-grid blocks are already PNG
    files; json image blocks (raw RGB) are converted."""
    import numpy as np
    from PIL import Image

    store = diagnostics.job_block_store(job_dir)
    raw = store.raw_bytes(key)
    if raw.startswith(_PNG_MAGIC):
        return raw
    payload = store.load(key, "json")
    data = np.frombuffer(base64.b64decode(payload["image_b64"]), dtype=np.uint8).reshape(payload["shape"])
    buffer = io.BytesIO()
    Image.fromarray(data, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _index_page(job_dir: Path, match: re.Match[str], collector: _ImageCollector, deterministic: bool = False) -> bytes:
    """The run overview: one row per hosted game.

    A game the stream never closed is shown by its hosted state rather
    than as in-flight, so a crashed run does not read as still running."""
    del match
    run_events = diagnostics.read_run_events(job_dir)
    run_roots = build_span_tree(run_events)
    hosted: dict[str, dict[str, Any]] = {}
    header = ""
    if run_roots:
        run = run_roots[0]
        status = run.outcome if run.closed else "in flight"
        header = (
            f"<p>run <b>{_esc(run.name)}</b> — {_esc(status)} — "
            f"{run.incl_generated} generated / {run.incl_uncached} uncached tokens"
            + _dur(run.duration_s, deterministic)
            + "</p>"
        )
        for event in run.events:
            if event["kind"] == "game_hosted":
                hosted[event["values"]["run_id"]] = event["values"]
    rows: list[str] = []
    for game_dir in _game_dirs(job_dir):
        run_id = game_dir.name
        row = _game_row(job_dir, run_id)
        info = hosted.get(run_id)
        # The game_hosted event snapshots run.state at hosting end, which
        # is still "playing" for a crashed game (TAAF stamps "crashed" at
        # teardown) — the stream's own root outcome is the truth then.
        state = (info or {}).get("state")
        if state in (None, "playing", "not_started"):
            state = row.outcome if row and row.closed else "in flight"
        score = (info or {}).get("final_score")
        rows.append(
            f'<tr><td><a href="g_{_esc(run_id)}.html">{_esc(row.name if row else run_id)}</a></td>'
            f"<td>{_esc(state)}</td><td>{'-' if score is None else _esc(score)}</td>"
            f"<td>{row.moves if row else 0}</td>"
            f"<td>{row.incl_generated if row else 0} / {row.incl_uncached if row else 0}</td>"
            f"<td><code>{_esc(run_id[:12])}</code></td></tr>"
        )
    body = (
        # ../../ = the listing of the directory holding this run dir (the
        # index lives at <run>/viewer/index.html). Works exported too.
        '<p><a href="../../">← all runs</a></p>'
        "<h1>diagnostics viewer</h1>"
        + header
        + "<table><tr><th>game</th><th>state</th><th>score</th><th>moves</th>"
        + "<th>tokens gen/unc</th><th>run id</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return _page("diagnostics viewer", body)


def _move_thumb(move: SpanNode, collector: _ImageCollector) -> str:
    """A small image standing for the task: its first question, which is
    what the agent was asked to answer. Empty when the span carries none."""
    asked = sorted(
        (int(a["name"][len("ask_") :]), a["block"])
        for a in move.artifacts
        if str(a["name"]).startswith("ask_") and a["name"][len("ask_") :].isdigit()
    )
    if not asked:
        return ""
    return f'<img class="frame-thumb" src="{_esc(collector(asked[0][1]))}" alt="{_esc(move.name)}">'


def _move_cells(root: SpanNode, run_id: str) -> str:
    """The move strip: one cell per move, linking to that move's page."""
    cells: list[str] = []
    for child in (c for c in root.children if c.kind == "infra/move"):
        outcome = child.outcome or ""
        css = ""
        if not child.closed:
            css = "unclosed"
        elif outcome == "error":
            css = "error"
        elif outcome in ("gave_up", "cancelled"):
            css = "gave_up"
        label = "·" if child.incl_generated == 0 else "L"
        title = f"{child.name}: {outcome or 'in flight'} — {child.incl_generated} gen tokens"
        cells.append(
            f'<a class="{css}" href="g_{_esc(run_id)}_m{child.span_id}.html" title="{_esc(title)}">{label}</a>'
        )
    return f'<div class="move-strip">{"".join(cells)}</div>'


def _game_page(
    job_dir: Path, match: re.Match[str], collector: _ImageCollector, deterministic: bool = False
) -> bytes | None:
    """One game: its move strip and per-move chunk trees.

    Consecutive trivial moves (queue drains, deliveries) buffer into a
    single folded row, so the interesting moves stay findable."""
    run_id = match["run_id"]
    root = _game_tree(job_dir, run_id)
    if root is None:
        return None
    live_href = None if root.closed else f"g_{run_id}_live.html"
    view = _view(job_dir, collector, live_href, deterministic)
    status = root.outcome if root.closed else "in flight / aborted"
    game_peak = peak_context(root)
    rows: list[str] = []
    others: list[str] = []
    trivial_buffer: list[str] = []

    def flush_trivial() -> None:
        """Emit the buffered run of trivial moves as one folded row."""
        if trivial_buffer:
            rows.append(f'<tr class="trivial-row"><td colspan="7">{"".join(trivial_buffer)}</td></tr>')
            trivial_buffer.clear()

    for child in root.children:
        if child.kind != "infra/move":
            others.append(chunk_for(child, view).html())
            continue
        href = f"g_{_esc(run_id)}_m{child.span_id}.html"
        thumb = _move_thumb(child, collector)
        n_submitted = sum(1 for e in child.events if e["kind"] == "infra/submit")
        action_text = f"{n_submitted} answer{'' if n_submitted == 1 else 's'}" if n_submitted else "no answer"
        if is_trivial(child):
            # Trivial moves don't get table rows: their frames run as a
            # line between the decision moves (each still a link).
            title = f"{child.name}: {action_text or child.outcome or 'in flight'}"
            cell = thumb if thumb else _esc(child.name)
            trivial_buffer.append(f'<a href="{href}" title="{_esc(title)}">{cell}</a>')
            continue
        flush_trivial()
        move_peak = peak_context(child)
        rows.append(
            f'<tr><td><a href="{href}">{_esc(child.name)}</a></td>'
            f'<td><a href="{href}">{thumb}</a></td>'
            f"<td>{_esc(action_text)}</td><td>{_esc(child.outcome or 'in flight')}</td>"
            f"<td>{child.incl_generated} / {child.incl_uncached}</td>"
            f"<td>{f'{move_peak:,}' if move_peak else ''}</td>"
            f"<td>{'' if child.duration_s is None or deterministic else f'{child.duration_s:.2f}s'}</td></tr>"
        )
    flush_trivial()
    live_link = "" if root.closed else f' — <a href="g_{_esc(run_id)}_live.html">⏵ live turn</a>'
    body = (
        f'<p><a href="index.html">← run index</a>{live_link}</p>'
        f"<h1>{_esc(root.name)}</h1>"
        f"<p>{_esc(status)} — {root.incl_generated} generated / {root.incl_uncached} uncached tokens"
        + (f" — peak ctx {game_peak:,}" if game_peak else "")
        + _dur(root.duration_s, deterministic)
        + "</p>"
        + _move_cells(root, run_id)
        + "<table><tr><th>call</th><th>task</th><th>answers</th><th>outcome</th><th>tokens gen/unc</th>"
        + "<th>peak ctx</th><th>time</th></tr>"
        + "".join(rows)
        + "</table>"
        + "".join(others)
    )
    return _page(f"{root.name} — {run_id[:12]}", body)


def _live_page(
    job_dir: Path,
    match: re.Match[str],
    collector: _ImageCollector,
    deterministic: bool = False,
) -> bytes | None:
    """The in-flight turn of a running game, refreshed by the page itself."""
    del deterministic
    run_id = match["run_id"]
    root = _game_tree(job_dir, run_id)
    if root is None:
        return None
    game_dir = job_dir / "games" / run_id
    if not root.closed:
        # "a viewer is watching" — the game streams while this is fresh. Not
        # written for a finished run: no process will ever read it, and a tab
        # left open on an archived run would rewrite it forever.
        platypus.live.touch_request(game_dir)
    snapshot = platypus.live.read_snapshot(game_dir)
    back = f'<p><a href="g_{_esc(run_id)}.html">← {_esc(root.name)}</a></p>'
    if snapshot is None or not snapshot.get("active"):
        reason = (
            "the run has finished"
            if root.closed
            else "no turn is running right now (between moves, or the streamer is just spinning up)"
        )
        # A finished run has nothing further to show, so its page is static:
        # refreshing it forever would poll a stream that will never move again.
        if root.closed:
            body = back + f"<h1>{_esc(root.name)} — live</h1><p>{_esc(reason)}.</p>"
            return _page(f"{root.name} — live", body)
        body = back + f"<h1>{_esc(root.name)} — live</h1><p>{_esc(reason)}. Auto-refreshing every 5 s.</p>"
        return _live_wrap(f"{root.name} — live", body)
    # Link to the in-flight MOVE (the last unclosed infra/move span), not
    # the turn span — the move page is keyed by move span id, and the
    # snapshot's span_id is the deeper llm/turn span.
    open_moves = [c for c in root.children if c.kind == "infra/move" and not c.closed]
    current = open_moves[-1] if open_moves else None
    move_link = (
        f'<a href="g_{_esc(run_id)}_m{current.span_id}.html">{_esc(current.name)}</a>'
        if current is not None
        else "current move"
    )
    parts = [
        back,
        f"<h1>{_esc(root.name)} — live turn</h1>",
        f"<p><b>{_esc(snapshot.get('label', ''))}</b> — {move_link} — "
        f"{snapshot.get('n_responses', 0)} responses completed — "
        f"{snapshot.get('elapsed_s', 0)}s elapsed — auto-refresh 5 s</p>",
    ]
    # The streaming tail of the in-flight response, up top so it's always
    # visible without scrolling the (possibly long) conversation.
    reasoning = snapshot.get("partial_reasoning") or ""
    final = snapshot.get("partial_final") or ""
    if reasoning:
        parts.append(f"<h3>streaming reasoning ({len(reasoning):,} chars)</h3><pre>{_esc(reasoning[-6000:])}</pre>")
    if final:
        parts.append(f"<h3>streaming response ({len(final):,} chars)</h3><pre>{_esc(final[-6000:])}</pre>")
    if not reasoning and not final:
        parts.append("<p>model call in flight, no streamed content yet (or the backend isn't streaming).</p>")
    # The full conversation so far (system prompt, pre-history, this
    # turn's completed rounds), rendered like the move page.
    conversation = platypus.live.read_conversation(game_dir)
    if conversation is not None and isinstance(conversation.get("structure"), dict):
        view = _view(job_dir, collector)
        parts.append("<h3>conversation so far</h3>")
        parts.append(
            ullm.html_render.conversation_html(
                conversation["structure"],
                text_of=view.text_of,
                image_src=view.image_src_for,
                image_label=view.image_label_for,
            )
        )
    return _live_wrap(f"{root.name} — live", "".join(parts))


def _live_wrap(title: str, body: str) -> bytes:
    """A page that refreshes itself, for a run still being written."""
    page = _page(title, body)
    return page.replace(b"<head>", b'<head><meta http-equiv="refresh" content="5">', 1)


def _move_page(
    job_dir: Path, match: re.Match[str], collector: _ImageCollector, deterministic: bool = False
) -> bytes | None:
    """One move: its chunk tree, frames, and links to its neighbours."""
    run_id, span_id = match["run_id"], int(match["span_id"])
    root = _game_tree(job_dir, run_id)
    if root is None:
        return None
    moves = [c for c in root.children if c.kind == "infra/move"]
    move = next((c for c in root.children if c.span_id == span_id), None)
    if move is None:
        return None
    live_href = None if root.closed else f"g_{run_id}_live.html"
    view = _view(job_dir, collector, live_href, deterministic)
    move_chunk = chunk_for(move, view)
    move_chunk.open_by_default = True
    body = (
        f'<p><a href="g_{_esc(run_id)}.html">← {_esc(root.name)}</a></p>'
        + _move_nav(moves, move, run_id)
        + f"<h1>{_esc(root.name)} — {_esc(move.name)}</h1>"
        + move_chunk.html()
    )
    return _page(f"{root.name} — {move.name}", body)


def _move_nav(moves: list[SpanNode], move: SpanNode, run_id: str) -> str:
    """prev/next move links, plus prev/next NON-TRIVIAL move."""
    try:
        index = next(i for i, m in enumerate(moves) if m.span_id == move.span_id)
    except StopIteration:
        return ""

    def link(target: SpanNode | None, label: str) -> str:
        """A navigation link to another move, or inert text when there is none."""
        if target is None:
            return f'<span class="move-nav-off">{_esc(label)}</span>'
        return f'<a href="g_{_esc(run_id)}_m{target.span_id}.html">{_esc(label)}</a>'

    def scan(direction: int, want_nontrivial: bool) -> SpanNode | None:
        """The nearest move in ``direction``, optionally skipping trivial ones."""
        i = index + direction
        while 0 <= i < len(moves):
            if not want_nontrivial or not is_trivial(moves[i]):
                return moves[i]
            i += direction
        return None

    return (
        '<p class="move-nav">'
        + link(scan(-1, False), "← prev")
        + " · "
        + link(scan(1, False), "next →")
        + " &nbsp;|&nbsp; non-trivial: "
        + link(scan(-1, True), "← prev")
        + " · "
        + link(scan(1, True), "next →")
        + "</p>"
    )


@dataclass(frozen=True)
class PageSpec:
    """One viewer-scheme page: the path pattern that claims it and the builder
    that renders it. ``PAGES`` is ordered, so an earlier pattern wins — the
    server and ``export_viewer`` both walk it, which is what keeps the served
    and exported URL spaces identical."""

    pattern: re.Pattern[str]
    build: Callable[[Path, re.Match[str], _ImageCollector, bool], bytes | None]


_INDEX_RE = re.compile(r"^index\.html$")
_LIVE_RE = re.compile(r"^g_(?P<run_id>[A-Za-z0-9]+)_live\.html$")
_MOVE_RE = re.compile(r"^g_(?P<run_id>[A-Za-z0-9_-]+)_m(?P<span_id>\d+)\.html$")
_GAME_RE = re.compile(r"^g_(?P<run_id>[A-Za-z0-9_-]+)\.html$")

PAGES: list[PageSpec] = [
    PageSpec(_INDEX_RE, _index_page),
    PageSpec(_LIVE_RE, _live_page),  # before _GAME_RE, which would swallow the name
    PageSpec(_MOVE_RE, _move_page),
    PageSpec(_GAME_RE, _game_page),
]

_PNG_ROUTE = re.compile(r"^png/(?P<key>[0-9a-f]{64})\.png$")


def render_viewer_path(job_dir: str | Path, page: str, *, deterministic: bool = False) -> tuple[bytes, str] | None:
    """Render one canonical viewer path for ``job_dir`` — ``(bytes,
    content type)``, or ``None`` if the path matches no page (or its
    subject does not exist)."""
    load_producer_chunks()
    job_dir = Path(job_dir)
    png = _PNG_ROUTE.match(page)
    if png is not None:
        try:
            return block_png_bytes(job_dir, png["key"]), "image/png"
        except FileNotFoundError:
            return None
    collector = _ImageCollector()
    for spec in PAGES:
        match = spec.pattern.match(page)
        if match is not None:
            rendered = spec.build(job_dir, match, collector, deterministic)
            return None if rendered is None else (rendered, "text/html; charset=utf-8")
    return None


# ---------------------------------------------------------------------------
# Static export
# ---------------------------------------------------------------------------


def export_viewer(job_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    """Materialize every canonical viewer path as files (default: into
    ``<job_dir>/viewer/``) — same renderers, same paths as the server, so
    every link keeps resolving with no server in front."""
    job_dir = Path(job_dir)
    out = Path(out_dir) if out_dir is not None else job_dir / "viewer"
    out.mkdir(parents=True, exist_ok=True)
    collector = _ImageCollector()
    pages: dict[str, bytes] = {}
    index_match = _INDEX_RE.match("index.html")
    assert index_match is not None
    pages["index.html"] = _index_page(job_dir, index_match, collector)
    for game_dir in _game_dirs(job_dir):
        run_id = game_dir.name
        root = _game_tree(job_dir, run_id)
        if root is None:
            continue
        game_match = _GAME_RE.match(f"g_{run_id}.html")
        assert game_match is not None
        rendered = _game_page(job_dir, game_match, collector)
        if rendered is not None:
            pages[f"g_{run_id}.html"] = rendered
        for child in root.children:
            if child.kind != "infra/move":
                continue
            move_match = _MOVE_RE.match(f"g_{run_id}_m{child.span_id}.html")
            assert move_match is not None
            move_rendered = _move_page(job_dir, move_match, collector)
            if move_rendered is not None:
                pages[f"g_{run_id}_m{child.span_id}.html"] = move_rendered
    for rel, content in pages.items():
        (out / rel).write_bytes(content)
    if collector.keys:
        (out / "png").mkdir(exist_ok=True)
        for key in collector.keys:
            (out / "png" / f"{key}.png").write_bytes(block_png_bytes(job_dir, key))
    return out


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


_VIEWER_ROUTE = re.compile(r"^(?P<run>(?:[^/]+/)*?)viewer/(?P<page>.+)$")


def resolve_request(root: str | Path, path: str) -> tuple[bytes, str] | None:
    """One request against the serving root: viewer paths render fresh;
    anything else falls through to the file on disk. ``None`` = 404."""
    root = Path(root).resolve()  # callers may pass a relative root; one canonical form throughout
    rel = path.lstrip("/")
    if not rel:
        rel = "index.html" if _is_run_dir(root) else ""
    match = _VIEWER_ROUTE.match(rel)
    if match is not None:
        run_dir = (root / match["run"].rstrip("/")).resolve() if match["run"] else root
        # Confined like the static fallthrough below: ".." is a valid route
        # segment, and the live page WRITES a request flag into the run dir it
        # resolves, so an unconfined prefix would reach outside the root.
        if (run_dir == root or run_dir.is_relative_to(root)) and _is_run_dir(run_dir):
            rendered = render_viewer_path(run_dir, match["page"])
            if rendered is not None:
                return rendered
    target = (root / rel).resolve() if rel else root
    if not target.is_relative_to(root):  # path traversal guard
        return None
    if target.is_file():
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css",
            ".png": "image/png",
            ".mp4": "video/mp4",
            ".json": "application/json",
            ".jsonl": "text/plain; charset=utf-8",
            # Plain-text run-dir sidecars render in the browser instead of
            # downloading.
            ".log": "text/plain; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
            ".py": "text/plain; charset=utf-8",
            ".sh": "text/plain; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        return target.read_bytes(), content_type
    if target.is_dir():
        return _listing(root, target), "text/html; charset=utf-8"
    return None


_FINISHED_LOCK = threading.Lock()
#: path -> (mtime_ns, size, finished). Keyed on what a rewrite changes, so a
#: file that has moved on is re-read and one that has not is answered from
#: the stat alone.
_FINISHED_CACHE: dict[str, tuple[int, int, bool]] = {}


def benchmark_finished(run_dir: Path | str) -> bool:
    """Whether the run's benchmark has ENDED, not merely been written.

    TAAF saves ``benchmark.json`` periodically from the first save on and
    stamps ``end_time`` immediately before the final one, so the file's
    existence says a run started and only the stamp says it stopped.

    One home for the rule because it has two callers: a listing that calls
    this per run dir per request, and the sweeper's collector, which decides
    from it whether a job can be read. Two copies of a predicate drift.

    Answered from a ``(mtime, size)`` memo, since the file is multi-MB and
    both callers ask repeatedly — a listing on every request, a watch loop on
    every poll. A file mid-rewrite reads as unfinished and is re-read once it
    settles.
    """
    path = Path(run_dir) / "benchmark.json"
    try:
        stat = path.stat()
    except OSError:
        return False
    key = str(path)
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _FINISHED_LOCK:
        cached = _FINISHED_CACHE.get(key)
        if cached is not None and cached[:2] == stamp:
            return cached[2]
    try:
        finished = json.loads(path.read_text()).get("end_time") is not None
    except (OSError, ValueError):
        return False
    with _FINISHED_LOCK:
        _FINISHED_CACHE[key] = (*stamp, finished)
    return finished


def _viewer_app_path() -> Path:
    """Installed package path of the replay-viewer shell."""
    return Path(__file__).with_name("viewer_app.html")


def _json_response(payload: dict[str, Any]) -> tuple[bytes, str]:
    """Compact UTF-8 JSON response body and content type."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(), "application/json; charset=utf-8"


def resolve_interactive_request(root: str | Path, path: str, query: str = "") -> tuple[int, bytes, str] | None:
    """Resolve the replay UI (``/_replay/``) and its adapter API (``/_api/``).

    Returns ``(status, body, content_type)`` for interactive routes and
    ``None`` for the canonical diagnostics/static resolver to handle.
    """
    from urllib.parse import parse_qs

    from platypus import viewer_data

    route = "/" + path.lstrip("/")
    if route in ("/_replay", "/_replay/", "/_replay/index.html"):
        return 200, _viewer_app_path().read_bytes(), "text/html; charset=utf-8"
    if not route.startswith("/_api/"):
        return None
    params = parse_qs(query)
    requested_run = params.get("run", [None])[0]
    try:
        if route == "/_api/run":
            payload = viewer_data.load_run_summary(root, requested_run)
            return 200, *_json_response(payload)
        if route == "/_api/game":
            game_index = int(str(params.get("index", [None])[0]))
            return 200, *_json_response(viewer_data.load_game_full(root, requested_run, game_index))
    except (TypeError, ValueError) as exc:
        return 400, *_json_response({"error": str(exc) or "Missing or invalid game index."})
    # OSError covers FileNotFoundError plus the filesystem errors a shared
    # tree serves up (permission denied, stale NFS handles) — any of them
    # must become a readable JSON error, never a reset connection.
    except OSError as exc:
        return 404, *_json_response({"error": str(exc), "games": []})
    return 404, *_json_response({"error": "Not found"})


def _listing(root: Path, directory: Path) -> bytes:
    """Directory listing with viewer links where a run dir is detected —
    the navigation glue for a directory of runs."""
    from urllib.parse import quote

    entries: list[str] = []
    for entry in sorted(directory.iterdir()):
        name = entry.name + ("/" if entry.is_dir() else "")
        extra = ""
        if entry.is_dir() and _is_run_dir(entry):
            # The replay viewer names a run by its path under the root.
            run_label = entry.resolve().relative_to(root.resolve()).as_posix()
            extra = (
                f' — <a href="{_esc(entry.name)}/viewer/index.html">viewer</a>'
                f' · <a href="/_replay/?run={_esc(quote(run_label, safe=""))}">replay</a>'
            )
            if (entry / "diagnostics.html").is_file():
                extra += f' · <a href="{_esc(entry.name)}/diagnostics.html">diagnostics</a>'
            extra += " ✓" if benchmark_finished(entry) else " ⏳"
        entries.append(f'<li><a href="{_esc(name)}">{_esc(name)}</a>{extra}</li>')
    up = "" if directory.resolve() == root.resolve() else '<p><a href="../">← up</a></p>'
    replay = '<p><a href="/_replay/">replay viewer</a></p>' if directory.resolve() == root.resolve() else ""
    title = _esc(str(directory.relative_to(root)) or ".")
    return _page(title, f"<h1>{title}</h1>{up}{replay}<ul>{''.join(entries)}</ul>")


def serve(root: str | Path, port: int = 8765, host: str = "127.0.0.1") -> None:
    """Serve a directory of runs (or one run dir). Viewer pages render
    fresh per request — in-flight runs included; everything else serves
    from disk with no-store caching.

    Strictly foreground: the server lives exactly as long as this call
    (Ctrl-C / terminal close stops it; connection threads are daemonic so
    shutdown never lingers). If ``port`` is taken (shared login nodes
    collect parked viewers), an OS-assigned free port is used instead —
    read the printed URL. ``port=0`` asks for a free port directly."""
    import errno
    import http.server

    load_producer_chunks()
    root_path = Path(root)

    class Handler(http.server.BaseHTTPRequestHandler):
        """Serves viewer-scheme paths rendered fresh from the JSONL streams,
        and everything else as static files under ``root_path``. Rendering per
        request rather than from disk is what makes an in-flight run viewable."""

        def _redirect_to_slash(self, path: str) -> bool:
            """Send the trailing-slash redirect a directory URL needs, if it is one.

            A listing's links are relative, and a browser resolves those against
            the *parent* of a URL with no trailing slash: served at ``/a/b/c``,
            a link to ``d.html`` is fetched from ``/a/b/d.html``. Clicking
            through never hits it, because a listing writes its directory links
            with the slash already on — it takes a typed or pasted URL, which is
            why it looks intermittent.
            """
            if path.endswith("/"):
                return False
            target = (root_path / path.lstrip("/")).resolve()
            if not (target.is_relative_to(root_path.resolve()) and target.is_dir()):
                return False
            # Built from the raw request path, not the decoded one handed in:
            # they differ the moment a segment needs percent-encoding, and
            # substituting the decoded form into the raw URL then matches
            # nothing and redirects the browser to where it already is.
            from urllib.parse import urlparse

            parts = urlparse(self.path)
            location = parts.path + "/" + (f"?{parts.query}" if parts.query else "")
            self.send_response(301)
            self.send_header("Location", location)
            self.end_headers()
            return True

        def do_GET(self) -> None:
            """Serve one request: viewer paths render fresh, everything else is a file."""
            from urllib.parse import unquote, urlparse

            parsed = urlparse(self.path)
            decoded = unquote(parsed.path)
            interactive = resolve_interactive_request(root_path, decoded, parsed.query)
            if interactive is not None:
                status, content, content_type = interactive
                self._send_content(status, content, content_type)
                return
            if self._redirect_to_slash(decoded):
                return
            result = resolve_request(root_path, decoded)
            if result is None:
                self.send_error(404)
                return
            content, content_type = result
            self._send_content(200, content, content_type)

        def _send_content(self, status: int, content: bytes, content_type: str) -> None:
            """Send one app, API, canonical-viewer, or static response."""
            import gzip

            self.send_response(status)
            self.send_header("Content-Type", content_type)
            # Conversations make big text pages; worth compressing over a
            # forwarded port (the duck viewer's trick). PNGs are already
            # compressed.
            if (
                len(content) > 1024
                and not content_type.startswith("image/")
                and "gzip" in (self.headers.get("Accept-Encoding") or "")
            ):
                content = gzip.compress(content)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: Any) -> None:
            """Silence the stdlib per-request logging."""
            # quiet by default

    try:
        server = http.server.ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        print(f"port {port} is already in use (another viewer on this machine?) — picking a free one", flush=True)
        server = http.server.ThreadingHTTPServer((host, 0), Handler)
    server.daemon_threads = True  # open connections never outlive the process
    bound_port = server.server_address[1]
    print(
        f"viewer serving {root_path} — Ctrl-C to stop\n"
        f"  http://{host}:{bound_port}/\n"
        f"  replay: http://{host}:{bound_port}/_replay/",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("viewer stopped", flush=True)
    finally:
        server.server_close()


def main() -> None:
    """CLI — the serve_diagnostics successor: point at a directory of
    runs (platypus job_dirs render live; anything else serves as files):

        uv run python -m platypus.viewer /data/shared/research/arc_3_results/...
    """
    import argparse

    parser = argparse.ArgumentParser(description="Serve the platypus diagnostics viewer over a directory of runs.")
    parser.add_argument("root", nargs="?", default=".", help="directory of run dirs (or one run dir)")
    parser.add_argument("--port", type=int, default=8765, help="preferred port; falls back to a free one")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    serve(args.root, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
