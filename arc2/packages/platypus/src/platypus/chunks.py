"""Chunks: producer-owned HTML rendering of spans (the viewer layer).

A chunk renders ONE span node (``platypus.diagnostics.SpanNode``) to
HTML at view time. The kind→class registry is how producers own their
presentation: the module that emits a span kind registers the chunk
class for it (``@chunk("bfs/search")``), co-located with the emitter.
Unregistered kinds fall back to :class:`GenericChunk` — header +
key-value table + artifacts + children — so new spans always show up;
a bespoke chunk is a progressive enhancement, never a prerequisite.

Every chunk carries the mandated header (name, kind, outcome, inclusive
generated/uncached tokens, wall time); subclasses extend via
``header_extra`` and replace the body via ``body_html``. Compound
rendering is the parent's choice: ``children_html()`` renders children
through the registry, and a parent may group, summarize, or inline
instead.

Chunks run wherever the viewer runs (server request or static export);
the stream never carries HTML.
"""

from __future__ import annotations

import base64
import html
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import ullm.html_render

from platypus.diagnostics import BlockStore, SpanNode

_REGISTRY: dict[str, type[Chunk]] = {}


def chunk(kind: str) -> Callable[[type[Chunk]], type[Chunk]]:
    """Register a chunk class for a span kind (producer-side decorator)."""

    def register(cls: type[Chunk]) -> type[Chunk]:
        """Register a chunk class for the span kind it declares."""
        assert kind not in _REGISTRY, f"chunk kind {kind!r} registered twice"
        _REGISTRY[kind] = cls
        return cls

    return register


def chunk_for(node: SpanNode, view: ViewContext) -> Chunk:
    """The registered chunk class for a span's kind, or the generic fallback
    — an unregistered kind always renders, never breaks the page."""
    return _REGISTRY.get(node.kind, GenericChunk)(node, view)


_producers_loaded = False


def load_producer_chunks() -> None:
    """Import every producer chunk module declared under the
    ``platypus.chunks`` entry-point group, filling the registry.

    A producer's chunks are co-located with its package (e.g.
    ``whale/chunks.py``), but the viewer runs standalone over a runs
    directory and imports no agent package on its own — the entry point is
    how a producer says "when rendering, load this". Called by the viewer's
    entry paths; idempotent. A broken producer module raises: rendering
    with silently missing renderers would look like working fallback."""
    global _producers_loaded
    if _producers_loaded:
        return
    _producers_loaded = True
    import importlib.metadata

    for entry_point in importlib.metadata.entry_points(group="platypus.chunks"):
        entry_point.load()


def _esc(value: Any) -> str:
    """HTML-escape a value for interpolation into a page."""
    return html.escape(str(value), quote=True)


def is_trivial(node: SpanNode) -> bool:
    """A move span the agent marked trivial (kv trivial=True). A move
    that spent LLM tokens is never trivial, whatever the agent said."""
    return node.kv.get("trivial") is True and node.incl_generated == 0


def peak_context(node: SpanNode) -> int:
    """Max context pressure (prompt + completion tokens, the whale's
    metric) across every llm/response event in this span's subtree.
    Prompt size is reconstructed as uncached + cached input; 0 when no
    LLM call happened (or the stream predates cached-token recording —
    then it undercounts by the cached share)."""
    peak = 0
    for event in node.events:
        if event["kind"] == "llm/response":
            values = event["values"]
            peak = max(
                peak,
                values.get("uncached_input_tokens", 0)
                + values.get("cached_input_tokens", 0)
                + values.get("completion_tokens", 0),
            )
    for child in node.children:
        peak = max(peak, peak_context(child))
    return peak


@dataclass
class ViewContext:
    """What rendering needs beyond the node: the block store, and how to
    address an image block in an ``<img src>``. The default embeds a
    PNG data URI (works in every host); the server/export may override
    with routes or files."""

    blocks: BlockStore
    image_src: Callable[[str], str] | None = None
    # Live page URL for this run, set only while the game is in flight;
    # an unclosed span renders a "watch live" link to it.
    live_href: str | None = None
    # Suppress wall-clock-derived output (span durations). Off by default —
    # durations are useful. On, the rendered HTML is reproducible across runs
    # of the same conversation, which is what lets a slice of it live in a
    # replay reference (token counts stay: they reproduce bit-exact).
    deterministic: bool = False

    # Event values dumped verbatim into the events table, and a span's own
    # key-values, may carry wall-clock fields; drop them in deterministic mode
    # so the HTML reproduces (a reference is a golden of the ORIGINAL run that
    # replay must reproduce, so only real behavioural differences may remain).
    _WALL_CLOCK_KEYS = ("duration_s", "t", "elapsed_s", "wall_s")

    def without_wall_clock(self, values: dict[str, Any]) -> dict[str, Any]:
        """``values`` minus wall-clock keys under deterministic rendering (a
        replay golden must not move with the clock); as given otherwise."""
        if self.deterministic:
            return {k: v for k, v in values.items() if k not in self._WALL_CLOCK_KEYS}
        return values

    def event_values_json(self, values: dict[str, Any]) -> str:
        """An event's values as JSON, minus wall-clock keys under deterministic
        rendering."""
        return json.dumps(self.without_wall_clock(values))

    def load_block(self, key: str, codec: str = "json") -> Any:
        """A block's payload, decoded. ``pickle`` is refused here: the codec
        travels in the stream, so honoring it would make rendering a run
        directory execute whatever that directory holds — and the viewer is
        routinely pointed at a shared results tree. Pickle artifacts are for
        offline repro bundles, loaded deliberately, not for page rendering."""
        if codec == "pickle":
            raise ValueError(f"block {key} is pickle-encoded; the viewer does not deserialize pickles")
        return self.blocks.load(key, codec)

    def image_src_for(self, key: str) -> str:
        """The URL for an image block: collected by the server, inlined otherwise."""
        if self.image_src is not None:
            return self.image_src(key)
        return self._data_uri(key)

    def _data_uri(self, key: str) -> str:
        """An image block re-encoded as an inline PNG data URL."""
        import numpy as np
        from PIL import Image

        payload = self.load_block(key)
        data = np.frombuffer(base64.b64decode(payload["image_b64"]), dtype=np.uint8).reshape(payload["shape"])
        buffer = io.BytesIO()
        Image.fromarray(data, mode="RGB").save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def text_of(self, field: Any) -> str:
        """Resolve a text field that may be an inline string or a
        ``{"$block": key}`` reference (the kv/serialization convention)."""
        if isinstance(field, dict) and "$block" in field:
            ref = cast("dict[str, Any]", field)
            codec = str(ref.get("codec", "json"))
            if codec == "pickle":
                return f"[pickle block {ref['$block']} — not rendered]"
            return str(self.load_block(ref["$block"], codec))
        return str(cast(object, field))

    def image_label_for(self, key: str) -> str:
        """Pixel dimensions + stored size of an image block — computed
        from the block itself, so streams recorded before this label
        existed get it too."""
        payload = self.load_block(key)
        height, width = payload["shape"][:2]
        stored = self.blocks.size(key)
        return f"{width}×{height} px, {stored / 1024:.0f} kB stored"


class Chunk:
    """Base renderer: one ``<details>`` block with the mandated header
    line, a body, and the children rendered through the registry."""

    open_by_default = False

    def __init__(self, node: SpanNode, view: ViewContext):
        """Bind a chunk to its span node and the view rendering it."""
        self.node = node
        self.view = view

    # -- override surface ---------------------------------------------------

    def header_extra(self) -> str:
        """Extra HTML appended to the header line."""
        return ""

    def body_html(self) -> str:
        """The span's own content (kv, artifacts, events) — everything
        between the header and the children."""
        return self._kv_html() + self._events_html() + self._artifacts_html()

    def children_html(self) -> str:
        """Default compound rendering: every child through the registry.
        Parents may override to group or summarize."""
        return "".join(chunk_for(child, self.view).html() for child in self.node.children)

    # -- framework ----------------------------------------------------------

    def html(self) -> str:
        """The whole chunk: header, body, and children."""
        node = self.node
        css = f"chunk chunk-{node.kind.replace('/', '-')}"
        if not node.closed:
            css += " chunk-unclosed"
        elif node.outcome == "error":
            css += " chunk-error"
        elif node.outcome == "cancelled":
            css += " chunk-cancelled"
        # Single-child spans auto-open: collapsing them costs a click
        # without hiding any breadth, so chains of one-child spans unroll.
        open_attr = " open" if self.open_by_default or len(node.children) == 1 else ""
        return (
            f'<details class="{css}"{open_attr}>'
            f"<summary>{self.header_line()}</summary>"
            f'<div class="chunk-body">{self.body_html()}{self.children_html()}</div>'
            "</details>"
        )

    def header_line(self) -> str:
        """The chunk's one-line header."""
        node = self.node
        title = _esc(node.name) if node.name else _esc(node.kind)
        status = _esc(node.outcome) if node.closed and node.outcome else "in flight / aborted"
        bits = [
            f'<span class="chunk-title">{title}</span>',
            f'<span class="chunk-kind">{_esc(node.kind)}</span>',
            f'<span class="chunk-outcome">{status}</span>',
        ]
        if node.incl_generated or node.incl_uncached:
            bits.append(f'<span class="chunk-tokens">{node.incl_generated} gen / {node.incl_uncached} unc</span>')
        if node.duration_s is not None and not self.view.deterministic:
            bits.append(f'<span class="chunk-time">{node.duration_s:.2f}s</span>')
        extra = self.header_extra()
        if extra:
            bits.append(extra)
        # An in-flight span links to the run's live streamer (set only
        # while the game is unfinished) — most usefully on the llm/turn
        # span, where the model is generating right now.
        if not node.closed and self.view.live_href:
            bits.append(f'<a class="live-link" href="{_esc(self.view.live_href)}">⏵ live</a>')
        return " ".join(bits)

    # -- shared pieces ------------------------------------------------------

    def _kv_html(self) -> str:
        """The span's key-values as a definition list."""
        kv = self.view.without_wall_clock(self.node.kv)
        if not kv:
            return ""
        rows = "".join(f"<tr><td>{_esc(key)}</td><td>{self._value_html(value)}</td></tr>" for key, value in kv.items())
        return f'<table class="chunk-kv">{rows}</table>'

    def _value_html(self, value: Any) -> str:
        """One key-value, resolving block references to their text."""
        if isinstance(value, dict) and "$block" in value:
            return f"<details><summary>…</summary><pre>{_esc(self.view.text_of(value))}</pre></details>"
        return _esc(value)

    def _events_html(self) -> str:
        """The span's events, in order."""
        if not self.node.events:
            return ""
        rows = "".join(
            f"<tr><td>{_esc(event['kind'])}</td><td>{_esc(self.view.event_values_json(event['values']))}</td></tr>"
            for event in self.node.events
        )
        return (
            f'<details class="chunk-events"><summary>{len(self.node.events)} events</summary>'
            f"<table>{rows}</table></details>"
        )

    def _artifacts_html(self, artifacts: list[dict[str, Any]] | None = None) -> str:
        """The span's artifacts as links or inline images."""
        parts: list[str] = []
        for artifact in self.node.artifacts if artifacts is None else artifacts:
            name, codec, key = artifact["name"], artifact["codec"], artifact["block"]
            if codec == "png-grid":
                parts.append(
                    f'<div class="chunk-artifact"><span class="chunk-label">{_esc(name)}</span>'
                    f'<img class="chunk-grid" src="{_esc(self.view.image_src_for(key))}" alt="{_esc(name)}"></div>'
                )
            elif codec == "json":
                payload = json.dumps(self.view.load_block(key), indent=1, ensure_ascii=False)
                parts.append(
                    f'<details class="chunk-artifact"><summary>{_esc(name)} (json)</summary>'
                    f"<pre>{_esc(payload if len(payload) < 20_000 else payload[:20_000] + ' …truncated')}</pre></details>"
                )
            else:  # pickle / bytes: not displayable — name the block for retrieval
                parts.append(
                    f'<div class="chunk-artifact">{_esc(name)} ({_esc(codec)} block '
                    f"<code>{_esc(key[:12])}…</code> — rehydrate from the block store)</div>"
                )
        return "".join(parts)


class GenericChunk(Chunk):
    """The fallback for unregistered kinds — the guarantee that new span
    shapes appear without viewer changes."""


@chunk("infra/game")
class GameChunk(Chunk):
    """The whole-game span. Expanded on arrival: it is the root of a game
    page, so collapsing it would hide the run behind a click."""

    open_by_default = True


def _grid_row(cells: list[tuple[dict[str, Any], str]], view: ViewContext) -> str:
    """A row of labelled grids — the task row and an answer's row share it."""
    if not cells:
        return ""
    figures = "".join(
        f'<figure class="move-frame"><img class="chunk-grid" '
        f'src="{_esc(view.image_src_for(artifact["block"]))}" alt="{_esc(label)}">'
        f"<figcaption>{_esc(label)}</figcaption></figure>"
        for artifact, label in cells
    )
    return f'<div class="move-frames">{figures}</div>'


def _attempt_key(name: str, prefix: str) -> tuple[int, int] | None:
    """``(question, attempt)`` from a ``<prefix>_<q>_attempt_<a>`` artifact
    name, or None for anything else."""
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != prefix or parts[2] != "attempt":
        return None
    return int(parts[1]), int(parts[3])


@chunk("infra/move")
class MoveChunk(Chunk):
    """The agent's play: the task it was shown and its standing answer as
    one row of grids, and how many answers it submitted.

    The grids are infra's own artifacts — ``given_*`` / ``ask_*`` for the
    task, ``final_*`` (last wins) for the answer that stood at the end — so
    this renders for any agent without one having to log anything. Each
    answer along the way is an ``infra/submit`` span somewhere below
    (:class:`SubmitChunk`), wherever in the agent's tree it was made.
    Whether any of it was *right* is TAAF's answer sheet: the truth never
    reaches this stream.
    """

    def _submit_events(self) -> list[dict[str, Any]]:
        """The infra/submit events this call made, in order."""
        return [e for e in self.node.events if e["kind"] == "infra/submit"]

    def _task_row(self) -> tuple[list[tuple[dict[str, Any], str]], set[str]]:
        """The task row's cells — each question's demonstrations, then the
        question itself followed by the standing answer's attempts for it —
        and the artifact names the row consumed."""
        given: dict[int, dict[str, dict[str, Any]]] = {}
        asked: dict[int, dict[str, Any]] = {}
        final: dict[tuple[int, int], dict[str, Any]] = {}  # last wins, in stream order
        shown: set[str] = set()
        for artifact in self.node.artifacts:
            name = str(artifact["name"])
            key = _attempt_key(name, "final")
            if name.startswith("given_"):
                _, index, part = name.split("_", 2)
                given.setdefault(int(index), {})[part] = artifact
            elif name.startswith("ask_") and name[len("ask_") :].isdigit():
                asked[int(name[len("ask_") :])] = artifact
            elif key is not None:
                final[key] = artifact
            else:
                continue
            shown.add(name)
        cells: list[tuple[dict[str, Any], str]] = []
        for i in sorted(given):
            if {"question", "answer"} <= given[i].keys():
                cells.append((given[i]["question"], f"given {i} in"))
                cells.append((given[i]["answer"], f"given {i} out"))
        for q in sorted(asked):
            cells.append((asked[q], f"ask {q}"))
            for key in sorted(k for k in final if k[0] == q):
                cells.append((final[key], f"final {q} attempt {key[1]}"))
        return cells, shown

    def body_html(self) -> str:
        """The task row followed by the generic body."""
        cells, shown = self._task_row()
        others = [a for a in self.node.artifacts if a["name"] not in shown]
        return _grid_row(cells, self.view) + self._kv_html() + self._events_html() + self._artifacts_html(others)

    def header_extra(self) -> str:
        """How many answers the agent submitted, for its header line."""
        n = len(self._submit_events())
        if not n:
            return '<span class="chunk-action">no answer</span>'
        return f'<span class="chunk-action">{n} answer{"" if n == 1 else "s"}</span>'

    def _events_html(self) -> str:
        """Events minus the submit events the header already counts."""
        events = [e for e in self.node.events if e["kind"] != "infra/submit"]
        if not events:
            return ""
        rows = "".join(
            f"<tr><td>{_esc(event['kind'])}</td><td>{_esc(self.view.event_values_json(event['values']))}</td></tr>"
            for event in events
        )
        return f'<details class="chunk-events"><summary>{len(events)} events</summary><table>{rows}</table></details>'


@chunk("infra/submit")
class SubmitChunk(Chunk):
    """One answer, as the row of grids the agent submitted: every attempt
    for every question, in ``Submission`` order. Open on arrival — the
    row is all there is, and it is what the page is for."""

    open_by_default = True

    def body_html(self) -> str:
        """The submitted grids as a labelled row, then anything else."""
        cells: list[tuple[dict[str, Any], str]] = []
        shown: set[str] = set()
        for artifact in self.node.artifacts:
            key = _attempt_key(str(artifact["name"]), "ask")
            if key is None:
                continue
            cells.append((artifact, f"ask {key[0]} attempt {key[1]}"))
            shown.add(str(artifact["name"]))
        others = [a for a in self.node.artifacts if a["name"] not in shown]
        return _grid_row(cells, self.view) + self._events_html() + self._artifacts_html(others)


@chunk("llm/turn")
class LLMTurnChunk(Chunk):
    """An LLM turn: response/kernel event summary plus the stored
    conversation rendered by ullm's own renderer (the same code path any
    canonical-format conversation renders through)."""

    def header_extra(self) -> str:
        """The turn's token spend and stop reason."""
        responses = [e for e in self.node.events if e["kind"] == "llm/response"]
        kernels = [e for e in self.node.events if e["kind"] == "kernel/exec"]
        tool_errors = [e for e in self.node.events if e["kind"] == "tool/result" and e["values"].get("is_error")]
        retries = sum(e["values"].get("malformed_retries", 0) for e in responses)
        injected = [e for e in responses if e["values"].get("response_injection")]
        bits = [f"{len(responses)} calls"]
        if injected:
            bits.append(f"{len(injected)} injected")
        peak = peak_context(self.node)
        if peak:
            bits.append(f"peak ctx {peak:,}")
        if kernels:
            bits.append(f"{len(kernels)} kernel execs")
        if tool_errors:
            bits.append(f"{len(tool_errors)} tool errors")
        if retries:
            bits.append(f"{retries} retries")
        return f'<span class="chunk-llm-meta">{_esc(", ".join(bits))}</span>'

    def body_html(self) -> str:
        """The conversation, rendered from its block-deduped artifact."""
        parts = [self._kv_html(), self._events_html()]
        others: list[dict[str, Any]] = []
        for artifact in self.node.artifacts:
            if artifact["name"] == "conversation" and artifact["codec"] == "json":
                structure = self.view.load_block(artifact["block"])
                parts.append(
                    ullm.html_render.conversation_html(
                        structure,
                        text_of=self.view.text_of,
                        image_src=self.view.image_src_for,
                        image_label=self.view.image_label_for,
                    )
                )
            else:
                others.append(artifact)
        parts.append(self._artifacts_html(others))
        return "".join(parts)
