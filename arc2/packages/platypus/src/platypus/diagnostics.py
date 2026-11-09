"""Hierarchical diagnostics: spans, the event stream, and the block
store.

One append-only JSONL stream per (game, pass), written as events happen
by whichever process hosts the game (the child, in subprocess mode —
bulk never rides the parent control pipe), plus a parent-written
run-level stream. Payloads too big for a spine line go to a shared
content-addressed block store under the job_dir — or all of it under a
scratch dir, archived into the job dir as zip parts ("Scratch and archive"). In memory only the
open-span stack (plus one token accumulator per open span) — never an
accumulating tree.

Producers open spans (``ctx.diag`` is a :class:`SpanLogger`), attach
small key-values, attach artifacts, and emit point events; every span's
close event carries self + inclusive token totals and wall time. The
ambient "current span" is a per-task contextvar, so infra's move span
and an agent's free-form spans nest without either knowing the other.

Spine events (one JSON object per line):

    {"e": "open", "id": N, "parent": P|null, "kind": ..., "name": ..., "t": ...}
    {"e": "kv", "span": N, "values": {...}}
    {"e": "artifact", "span": N, "name": ..., "codec": ..., "block": sha}
    {"e": "event", "span": N, "kind": ..., "t": ..., "values": {...}}
    {"e": "close", "id": N, "outcome": ..., "t": ..., "duration_s": ...,
     "self_generated": g, "self_uncached": u, "incl_generated": G, "incl_uncached": U}

Inline values are deliberately restricted: JSON scalars pass, numpy
scalars are coerced, strings above ``_INLINE_STR_LIMIT`` are auto-stored
as blocks, and anything structured (dict/list/ndarray) raises, pointing
at ``artifact()`` — the stream stays scannable and nothing large ever
rides a spine line. A span with no close event reads as in-flight (live
run) or aborted (dead run); the tree builder sums children for its
totals.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import threading
import time
import zipfile
from collections.abc import Collection, Generator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np

CODECS = ("json", "bytes", "png-grid", "pickle")

_INLINE_STR_LIMIT = 256


# ---------------------------------------------------------------------------
# Block store
# ---------------------------------------------------------------------------


# Payloads at or below this go into a pack instead of their own file. Most
# blocks are small — a measured run had 76% under 4 KiB and 52% under 1 KiB —
# and one file each turns a 25-game arm into ~14k files, which is minutes of
# NFS metadata to copy or delete and over a million files for a 10-pass tier.
PACK_THRESHOLD_BYTES = 4096

# The decompressed archived packs one store keeps, by bytes, least recently
# read evicted first. A packed block is a slice of its pack, and a page reads
# many blocks from a few packs, so those stay decompressed; but a viewer's
# store lives as long as the viewer (``job_block_store``), so what it keeps
# must be bounded by size rather than grow with every pack it has read. The
# pack read last stays even when it alone is larger.
ARCHIVED_PACK_CACHE_BYTES = 256 * 2**20

#: The block store's directory under a job dir (or a run's scratch dir), and
#: the per-process pack files in it.
BLOCKS_DIR = "diag_blocks"
_PACK_NAME = re.compile(r"pack-(\d+)\.(bin|idx)")


def _index_lines(index: dict[str, tuple[Path | str, int, int]], text: str, pack: Path | str) -> None:
    """Add one pack index's entries to ``index``, all pointing into ``pack``."""
    for line in text.splitlines():
        if not line.strip():
            continue  # a torn final line: the run is still writing
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        index[entry["k"]] = (pack, entry["o"], entry["n"])


class BlockStore:
    """Content-addressed payload store, keyed by sha256 of the encoded bytes.
    Shared across all of a run's streams so repeated content (system prompts,
    frames) stores once — conversation artifacts carry pre-history, so the
    same prompt is referenced by every later turn and dedup does real work.

    Large payloads get a file each, write-if-absent via tmp+rename (safe
    under concurrent writers on one filesystem). Small ones are appended to a
    pack with a sidecar index, because their COUNT is the cost, not their
    bytes. Packs are per-process (pid in the name) so concurrent game
    children never write the same file and no locking is needed; append-only
    means a live reader tailing the run still sees them.

    Dedup holds exactly within one store, and best-effort beyond it: a
    store checks the on-disk indexes before appending, but that view is a
    snapshot, so bytes another writer packed since can still be packed
    twice. Loose files dedup across processes by filename. The residual
    duplication is a bounded waste of space, traded for never
    coordinating."""

    def __init__(self, directory: str | Path, archive: DiagnosticsArchive | None = None):
        """A store rooted at ``directory``; packs are named per process.

        ``archive`` is where a reader finds what is no longer loose under
        ``directory``: a job whose blocks were archived into the job dir
        (``DiagnosticsArchiver``). Writing never touches it."""
        self.directory = Path(directory)
        self.archive = archive
        self._made_dirs: set[Path] = set()  # prefix dirs this instance has created
        self._pack_path = self.directory / f"pack-{os.getpid()}.bin"
        self._index_path = self.directory / f"pack-{os.getpid()}.idx"
        # A pack is a file under ``directory`` or, for an archived one, its
        # member name in ``archive``.
        self._packed: dict[str, tuple[Path | str, int, int]] = {}  # key -> (pack, offset, length)
        self._index: dict[str, tuple[Path | str, int, int]] | None = None  # lazily read, for load()
        # member name -> bytes, least recently read first (ARCHIVED_PACK_CACHE_BYTES)
        self._archived_packs: collections.OrderedDict[str, bytes] = collections.OrderedDict()
        self._archived_pack_bytes = 0
        self._archived_packs_lock = threading.Lock()  # readers share a store across threads

    def path(self, key: str) -> Path:
        """A block's loose-file path (whether or not it exists)."""
        return self.directory / key[:2] / key

    def store(self, payload: Any, codec: str) -> str:
        """Store a payload and return its content key; small payloads pack."""
        data = _encode_block(payload, codec)
        key = hashlib.sha256(data).hexdigest()
        if len(data) <= PACK_THRESHOLD_BYTES:
            self._pack(key, data)
            return key
        target = self.path(key)
        if not target.exists():
            # Only 256 possible prefix dirs; mkdir each at most once per
            # process instead of a mkdir syscall per new block (NFS RPC).
            parent = target.parent
            if parent not in self._made_dirs:
                parent.mkdir(parents=True, exist_ok=True)
                self._made_dirs.add(parent)
            tmp = target.with_name(f".tmp-{os.getpid()}-{key}")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        return key

    def _pack(self, key: str, data: bytes) -> None:
        """Append a small payload to this process's pack file and index it."""
        if key in self._packed:
            return
        # Consult the on-disk indexes too: several stores share one pack
        # file per process (one per hosted game in inprocess mode), so an
        # instance-local map alone would re-append what a sibling wrote.
        located = self._read_index().get(key)
        if located is not None:
            self._packed[key] = located
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        # Append data and index entry, flushing both: a reader tailing a live
        # run must never see an index entry pointing past the data.
        with open(self._pack_path, "ab") as fh:
            offset = fh.tell()
            fh.write(data)
            fh.flush()
        with open(self._index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": key, "o": offset, "n": len(data)}) + "\n")
            fh.flush()
        self._packed[key] = (self._pack_path, offset, len(data))

    def _read_index(self) -> dict[str, tuple[Path | str, int, int]]:
        """Every pack index in the directory, not just this process's — a
        reader is usually a different process than any of the writers — and
        every archived one, which a loose index of the same pack overrides."""
        if self._index is None:
            index: dict[str, tuple[Path | str, int, int]] = {}
            if self.archive is not None:
                for name in self.archive.names(f"{BLOCKS_DIR}/"):
                    if _PACK_NAME.fullmatch(name.rsplit("/", 1)[-1]) and name.endswith(".idx"):
                        raw = self.archive.read(name) or b""
                        _index_lines(index, raw.decode("utf-8"), name[: -len(".idx")] + ".bin")
            for idx in sorted(self.directory.glob("pack-*.idx")):
                _index_lines(index, idx.read_text(encoding="utf-8"), idx.with_suffix(".bin"))
            self._index = index
        return self._index

    def _archived_loose(self, key: str) -> bytes | None:
        """A loose block's bytes from the archive, or None."""
        if self.archive is None:
            return None
        return self.archive.read(f"{BLOCKS_DIR}/{key[:2]}/{key}")

    def _locate(self, key: str) -> tuple[Path | str, int, int]:
        """(pack, offset, length) for a packed block, or the loose file at
        offset 0. The single lookup path shared by every reader."""
        located = self._packed.get(key) or self._read_index().get(key)
        if located is None:
            # Re-read once: a live run may have packed this since we cached.
            self._index = None
            located = self._read_index().get(key)
        if located is None:
            # A missing payload, not a broken invariant: a partially synced or
            # trimmed run dir reaches here, and readers (the viewer) turn this
            # into a 404 rather than a crashed request.
            raise FileNotFoundError(f"block {key} is in neither a loose file nor any pack under {self.directory}")
        return located

    def raw_bytes(self, key: str) -> bytes:
        """The stored encoded bytes, loose or packed. Callers that need the
        payload rather than the raw bytes should use ``load``; this exists for
        the few that inspect bytes directly (e.g. PNG-magic sniffing)."""
        target = self.path(key)
        if target.exists():
            return target.read_bytes()
        archived = self._archived_loose(key)
        if archived is not None:
            return archived
        pack, offset, length = self._locate(key)
        if isinstance(pack, str):
            return self._archived_pack(pack)[offset : offset + length]
        with open(pack, "rb") as fh:
            fh.seek(offset)
            return fh.read(length)

    def _archived_pack(self, member: str) -> bytes:
        """An archived pack's bytes, decompressed on first read and kept while
        it is among the most recently read (``ARCHIVED_PACK_CACHE_BYTES``)."""
        with self._archived_packs_lock:
            data = self._archived_packs.get(member)
            if data is not None:
                self._archived_packs.move_to_end(member)
                return data
        assert self.archive is not None  # an archived pack is only indexed through one
        # Read outside the lock, so parallel readers of different packs do
        # not queue behind one decompression.
        data = self.archive.read(member) or b""
        with self._archived_packs_lock:
            if member not in self._archived_packs:
                self._archived_packs[member] = data
                self._archived_pack_bytes += len(data)
            self._archived_packs.move_to_end(member)
            while self._archived_pack_bytes > ARCHIVED_PACK_CACHE_BYTES and len(self._archived_packs) > 1:
                _, evicted = self._archived_packs.popitem(last=False)
                self._archived_pack_bytes -= len(evicted)
        return data

    def size(self, key: str) -> int:
        """Stored byte length — from stat for a loose file, from the index for
        a packed block (no read)."""
        target = self.path(key)
        if target.exists():
            return target.stat().st_size
        archived = self._archived_loose(key)
        if archived is not None:
            return len(archived)
        return self._locate(key)[2]

    def load(self, key: str, codec: str) -> Any:
        """A block's payload, decoded under ``codec``."""
        return _decode_block(self.raw_bytes(key), codec)


def _encode_block(payload: Any, codec: str) -> bytes:
    """Encode a payload under ``codec``; an unknown codec raises."""
    if codec == "json":
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if codec == "bytes":
        assert isinstance(payload, bytes), f"codec 'bytes' needs bytes, got {type(payload).__name__}"
        return payload
    if codec == "png-grid":
        return _grid_to_png(payload)
    if codec == "pickle":
        return pickle.dumps(payload)
    raise ValueError(f"unknown codec {codec!r}; valid: {CODECS}")


def _decode_block(data: bytes, codec: str) -> Any:
    """Inverse of ``_encode_block``."""
    if codec == "json":
        return json.loads(data)
    if codec == "bytes":
        return data
    if codec == "png-grid":
        return _png_to_grid(data)
    if codec == "pickle":
        return pickle.loads(data)
    raise ValueError(f"unknown codec {codec!r}; valid: {CODECS}")


def _grid_to_png(grid: Any) -> bytes:
    """2-D small-int grid -> indexed PNG (self-contained, ~30x smaller
    than a pickled ndarray; the palette rides in the file)."""
    from PIL import Image

    import taaf.grid

    array = np.asarray(grid, dtype=np.uint8)
    assert array.ndim == 2, f"png-grid needs a 2-D grid, got shape {array.shape}"
    image = Image.fromarray(array, mode="P")
    palette: list[int] = []
    for color in taaf.grid.ARC_COLORS:
        palette.extend(int(color[j : j + 2], 16) for j in (1, 3, 5))
    image.putpalette(palette)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_to_grid(data: bytes) -> np.ndarray:
    """An indexed PNG back to the small-int grid it encodes."""
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    return np.asarray(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


def _check_codec(codec: str) -> None:
    """Reject an unknown codec the way the encoder does — same exception
    type either side of the null stream, and not an assert (``-O`` would
    strip it)."""
    if codec not in CODECS:
        raise ValueError(f"unknown codec {codec!r}; valid: {CODECS}")


def _normalize_inline(key: str, value: Any) -> Any:
    """Coerce ``value`` to a JSON scalar, or raise. Structured data is
    rejected loudly — silent ``str()``-ing hides bugs and bloats the
    spine."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):  # numpy scalars are routine producer output
        coerced = cast(Any, value.item())
        if isinstance(coerced, (bool, int, float, str)):
            return coerced
    type_name = type(cast(object, value)).__name__
    raise TypeError(
        f"span value {key}={type_name} is not an inline scalar; "
        "attach structured data with span.artifact(name, payload, codec=...) instead"
    )


class Span:
    """One open span. Producers get these from ``SpanLogger.span(...)``
    / ``Span.span(...)`` context managers and never construct them."""

    __slots__ = (
        "_closed",
        "_incl_generated",
        "_incl_uncached",
        "_logger",
        "_outcome",
        "_parent",
        "_t_mono",
        "kind",
        "name",
        "self_generated",
        "self_uncached",
        "span_id",
    )

    def __init__(self, logger: SpanLogger, span_id: int, parent: Span | None, kind: str, name: str):
        """Open a span; ``_close`` writes the closing event."""
        self._logger = logger
        self.span_id = span_id
        self.kind = kind
        self.name = name
        self._parent = parent
        self._t_mono = time.monotonic()
        self._outcome: str | None = None
        self._closed = False
        self.self_generated = 0
        self.self_uncached = 0
        self._incl_generated = 0
        self._incl_uncached = 0

    # -- producer surface ---------------------------------------------------

    def _inline_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Normalize a value map for the spine: scalars inline, long strings
        auto-blocked, structured data rejected. Shared by ``kv`` and
        ``event`` so the spine stays scannable whichever one wrote a line."""
        out: dict[str, Any] = {}
        for key, value in values.items():
            normalized = _normalize_inline(key, value)
            if isinstance(normalized, str) and len(normalized) > _INLINE_STR_LIMIT:
                out[key] = {"$block": self._logger._store(normalized, "json"), "codec": "json"}
            else:
                out[key] = normalized
        return out

    def kv(self, **values: Any) -> None:
        """Attach small key-values. Scalars inline; long strings
        auto-block; structured data raises (use ``artifact``)."""
        self._logger._emit({"e": "kv", "span": self.span_id, "values": self._inline_values(values)})

    def artifact(self, name: str, payload: Any, *, codec: str = "json") -> None:
        """Attach a named out-of-line payload (block store)."""
        block = self._logger._store(payload, codec)
        self._logger._emit({"e": "artifact", "span": self.span_id, "name": name, "codec": codec, "block": block})

    def event(self, kind: str, *, t: float | None = None, **values: Any) -> None:
        """A point-in-time record attached to this span (no duration).
        ``t`` defaults to now; pass a wall timestamp for records drained
        after the fact (e.g. kernel executions collected at turn end)."""
        out = self._inline_values(values)
        self._logger._emit(
            {"e": "event", "span": self.span_id, "kind": kind, "t": time.time() if t is None else t, "values": out}
        )

    def span(self, kind: str, name: str = "") -> contextlib.AbstractContextManager[Span]:
        """Open a child span."""
        return self._logger._span_cm(kind, name, parent=self)

    def set_outcome(self, outcome: str) -> None:
        """Override the close outcome; an exception
        unwinding through the span still wins with ``"error"``."""
        self._outcome = outcome

    def add_usage(self, generated: int, uncached: int) -> None:
        """Attribute token spend to this span (self totals; inclusive
        totals aggregate through parents at close)."""
        self.self_generated += generated
        self.self_uncached += uncached

    def store_block(self, payload: Any, codec: str = "json") -> str:
        """Store a payload in the block store without attaching it as an
        artifact — for emitters that build structures referencing many
        blocks (e.g. a serialized conversation whose large texts store
        once each)."""
        return self._logger._store(payload, codec)

    # -- logger-internal ----------------------------------------------------

    def _close(self, outcome: str) -> None:
        """Write the close event, rolling self and inclusive totals into the parent."""
        if self._closed:  # defensive: a double close would corrupt totals
            return
        self._closed = True
        incl_generated = self._incl_generated + self.self_generated
        incl_uncached = self._incl_uncached + self.self_uncached
        if self._parent is not None:
            self._parent._incl_generated += incl_generated
            self._parent._incl_uncached += incl_uncached
        self._logger._emit(
            {
                "e": "close",
                "id": self.span_id,
                "outcome": self._outcome if self._outcome is not None and outcome == "ok" else outcome,
                "t": time.time(),
                "duration_s": round(time.monotonic() - self._t_mono, 6),
                "self_generated": self.self_generated,
                "self_uncached": self.self_uncached,
                "incl_generated": incl_generated,
                "incl_uncached": incl_uncached,
            }
        )


class _NullSpan(Span):
    """The span handed out when diagnostics is off: swallows everything
    (codec names still validated so misuse fails identically either
    way)."""

    def __init__(self) -> None:  # deliberately no super().__init__
        """The shared inert span; it holds no stream and no parent."""

    def kv(self, **values: Any) -> None:
        # Normalized and discarded: a structured value is a producer bug, and
        # it has to surface on the null stream too or it waits to be found
        # until someone runs the same code with diagnostics on.
        """Validate and discard (see the class docstring)."""
        for key, value in values.items():
            _normalize_inline(key, value)

    def artifact(self, name: str, payload: Any, *, codec: str = "json") -> None:
        """Validate the codec and discard."""
        _check_codec(codec)

    def event(self, kind: str, *, t: float | None = None, **values: Any) -> None:
        """Validate and discard."""
        for key, value in values.items():
            _normalize_inline(key, value)

    def span(self, kind: str, name: str = "") -> contextlib.AbstractContextManager[Span]:
        """A child span that is also inert."""
        return contextlib.nullcontext(self)

    def set_outcome(self, outcome: str) -> None:
        """No-op: nothing is recorded."""

    def add_usage(self, generated: int, uncached: int) -> None:
        """No-op: nothing is recorded."""

    def store_block(self, payload: Any, codec: str = "json") -> str:
        """Validate the codec and return an empty key."""
        _check_codec(codec)
        return ""

    def _close(self, outcome: str) -> None:
        """No-op: nothing is recorded."""


NULL_SPAN: Span = _NullSpan()

# Per-task ambient span: contextvars propagate across awaits within a
# task and copy into child tasks at creation — exactly the "attach at
# whatever span is current" semantics the contract asks for, with per-game
# isolation for free when games share one process.
_CURRENT: ContextVar[Span] = ContextVar("platypus_diag_current_span", default=NULL_SPAN)


def current_span() -> Span:
    """The ambient span (``NULL_SPAN`` when none is open in this task)."""
    return _CURRENT.get()


class SpanLogger:
    """One span stream (``ctx.diag``). Constructed with a stream path +
    block store, or with neither — the null stream, where every
    operation is a cheap no-op, so instrumented code never branches on
    "is diagnostics on".

    ``span(...)`` opens a child of the ambient current span (a root span
    when none is open in this task) — infra's move span and an agent's
    free-form spans nest without either knowing the other."""

    def __init__(self, stream_path: str | Path | None = None, blocks: BlockStore | None = None):
        """A logger writing to ``stream_path``, or an inert one with no path."""
        assert (stream_path is None) == (blocks is None), "stream_path and blocks come together (or neither)"
        self._stream_path = None if stream_path is None else Path(stream_path)
        self._blocks = blocks
        self._file: TextIO | None = None
        self._next_id = 0

    @property
    def active(self) -> bool:
        """Whether anything is actually recorded."""
        return self._stream_path is not None

    @property
    def stream_dir(self) -> Path | None:
        """The directory holding this stream (the game-run dir for game
        streams) — what the live-turn files derive from. None on the
        null stream."""
        return None if self._stream_path is None else self._stream_path.parent

    @property
    def block_store(self) -> BlockStore | None:
        """This stream's content-addressed block store (None on the null
        stream). Shared with capture-format serialization so live and
        recorded conversations reference the same dedup'd blocks."""
        return self._blocks

    def span(self, kind: str, name: str = "") -> contextlib.AbstractContextManager[Span]:
        """Open a span under the ambient current span."""
        parent = _CURRENT.get()
        # An ambient span from a DIFFERENT logger (e.g. the run-level
        # stream in the parent process) is not a parent here — span ids
        # only mean something within one stream.
        usable = parent is not NULL_SPAN and parent._logger is self
        return self._span_cm(kind, name, parent=parent if usable else None)

    def close(self) -> None:
        """Close the underlying file handle (idempotent). Spans still
        open write nothing further — their close events are lost, which
        is exactly the aborted-span signature."""
        if self._file is not None:
            with contextlib.suppress(Exception):
                self._file.close()
            self._file = None

    # -- internals ----------------------------------------------------------

    @contextlib.contextmanager
    def _span_cm(self, kind: str, name: str, parent: Span | None) -> Generator[Span, None, None]:
        """The span context manager: open, set current, close — marking the span
        aborted if an exception unwinds through it."""
        if self._stream_path is None:
            yield NULL_SPAN
            return
        span = Span(self, self._next_id, parent, kind, name)
        self._next_id += 1
        self._emit(
            {
                "e": "open",
                "id": span.span_id,
                "parent": parent.span_id if parent is not None else None,
                "kind": kind,
                "name": name,
                "t": time.time(),
            }
        )
        token = _CURRENT.set(span)
        outcome = "ok"
        try:
            yield span
        except BaseException as exc:
            # Cancellation is a routine control mechanism (wall budgets,
            # host-initiated move cancellation), not a failure — its own
            # outcome so the viewer can style it apart from errors.
            outcome = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            with contextlib.suppress(Exception):  # best-effort context on the failing span
                span.kv(error=f"{type(exc).__name__}: {exc}"[:_INLINE_STR_LIMIT])
            raise
        finally:
            _CURRENT.reset(token)
            span._close(outcome)

    def _emit(self, event: dict[str, Any]) -> None:
        """Append one event to the stream, line-buffered for live readers."""
        if self._file is None:
            assert self._stream_path is not None
            self._stream_path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered append: every event line is flushed as written,
            # so the stream is tail-readable in flight and survives an
            # untidy death up to the last completed event.
            self._file = open(self._stream_path, "a", buffering=1, encoding="utf-8")
        self._file.write(json.dumps(event, separators=(",", ":")) + "\n")

    def _store(self, payload: Any, codec: str) -> str:
        """Store a payload in the block store and return its key."""
        assert self._blocks is not None
        return self._blocks.store(payload, codec)


# ---------------------------------------------------------------------------
# Layout + reading
# ---------------------------------------------------------------------------


def resolve_blocks(store: BlockStore, node: Any) -> Any:
    """``node`` with every ``{"$block": key}`` reference replaced by the block.

    The capture format keeps large text and images out of line and refers
    to them by store key, so walking an artifact's JSON as written collects
    key strings where the content should be. This is the one reader to use
    for content: a copy of the structure with the references inlined, an
    image block left as its stored payload. A block the store no longer
    holds resolves to ``None`` rather than raising, so one lost block does
    not hide the rest of an artifact.
    """
    if isinstance(node, dict):
        mapping = cast("dict[str, Any]", node)
        key = mapping.get("$block")
        if isinstance(key, str):
            try:
                return resolve_blocks(store, store.load(key, str(mapping.get("codec", "json"))))
            except Exception:  # a codec this reader cannot decode, or a missing block
                return None
        return {name: resolve_blocks(store, value) for name, value in mapping.items()}
    if isinstance(node, list):
        return [resolve_blocks(store, value) for value in cast("list[Any]", node)]
    return node


def blocks_dir(job_dir: str | Path) -> Path:
    """The job dir's shared content-addressed block store."""
    return Path(job_dir) / BLOCKS_DIR


def game_stream_path(game_dir: str | Path) -> Path:
    """The per-(game, pass) stream, inside the game-run dir
    (``platypus.reference.game_run_dir``)."""
    return Path(game_dir) / "spans.jsonl"


def run_stream_path(job_dir: str | Path) -> Path:
    """The parent-written run-level stream."""
    return Path(job_dir) / "run_spans.jsonl"


#: Where the host monitors (``platypus.host_monitor``) write, under the
#: diagnostics root.
MONITOR_DIR = "monitor"
#: A monitor file still being written; renamed without it once final.
MONITOR_LIVE_SUFFIX = ".part"


def monitor_dir(root: str | Path) -> Path:
    """The host monitors' directory under a diagnostics root: the job dir,
    or its scratch dir."""
    return Path(root) / MONITOR_DIR


# ---------------------------------------------------------------------------
# Scratch and archive: where a run writes its diagnostics, and where they end up
# ---------------------------------------------------------------------------

# A run with a scratch dir (``RunConfig.diagnostics_scratch``: node-local disk)
# writes its streams and blocks there, in the job dir's own layout, and moves
# them into the job dir as numbered zip parts: a line-per-event stream and a
# file per block are cheap on a local disk and slow on a network filesystem,
# where a job's ~40k files also take the better part of an hour to delete.
# Readers resolve a job-dir-relative path the same way wherever the file is.

#: The job dir's archive parts, numbered in the order they were written. Their
#: members are job-dir-relative (``games/<run_id>/spans.jsonl``,
#: ``run_spans.jsonl``, ``diag_blocks/...``); a member in a later part
#: supersedes the same name in an earlier one.
ARCHIVE_PART_GLOB = "diagnostics-*.zip"


def archive_parts(job_dir: str | Path) -> list[Path]:
    """The job dir's archive parts, oldest first."""
    return sorted(Path(job_dir).glob(ARCHIVE_PART_GLOB))


class DiagnosticsArchive:
    """The read side of a job dir's archive parts: every member, resolved to
    the latest part that holds it. Holds no file open between reads: a part
    is opened through ``_open_part``, whose bound is process-wide."""

    def __init__(self, parts: list[Path]):
        """An archive over ``parts``, oldest first."""
        self._where: dict[str, Path] = {}
        for part in parts:
            for name in _open_part(part).namelist():
                self._where[name] = part

    def names(self, prefix: str = "") -> list[str]:
        """Every member name starting with ``prefix``, sorted."""
        return sorted(name for name in self._where if name.startswith(prefix))

    def read(self, name: str) -> bytes | None:
        """A member's bytes, or None when no part holds it."""
        part = self._where.get(name)
        return None if part is None else _open_part(part).read(name)

    def part(self, name: str) -> Path | None:
        """The part a member is read from, or None when no part holds it."""
        return self._where.get(name)


# The open parts, most recently used last, bounded so a viewer that browses
# many jobs holds a fixed number of descriptors rather than one per part it
# ever read. Keyed by the part's size as well as its path: a job rezipped in
# place gets fresh readers.
_OPEN_PARTS_MAX = 32
_OPEN_PARTS: collections.OrderedDict[tuple[Path, int], zipfile.ZipFile] = collections.OrderedDict()
_OPEN_PARTS_LOCK = threading.Lock()


def _open_part(part: Path) -> zipfile.ZipFile:
    """The part's reader, opened on demand and kept while it is among the
    most recently used."""
    key = (part, part.stat().st_size)
    with _OPEN_PARTS_LOCK:
        opened = _OPEN_PARTS.get(key)
        if opened is not None:
            _OPEN_PARTS.move_to_end(key)
            return opened
        opened = zipfile.ZipFile(part)
        _OPEN_PARTS[key] = opened
        while len(_OPEN_PARTS) > _OPEN_PARTS_MAX:
            _, evicted = _OPEN_PARTS.popitem(last=False)
            evicted.close()
        return opened


# One reader per job dir, rebuilt when its parts change (a live run adds
# them); the most recently used are kept, so a long-lived viewer's memory
# does not grow with every job it has shown.
_ARCHIVES_MAX = 64
_ARCHIVES: collections.OrderedDict[Path, tuple[tuple[tuple[str, int], ...], DiagnosticsArchive]] = (
    collections.OrderedDict()
)
_ARCHIVES_LOCK = threading.Lock()


def job_archive(job_dir: str | Path) -> DiagnosticsArchive | None:
    """The job dir's archive, or None when it has no parts."""
    parts = archive_parts(job_dir)
    if not parts:
        return None
    stamp = tuple((part.name, part.stat().st_size) for part in parts)
    key = Path(job_dir).resolve()
    with _ARCHIVES_LOCK:
        cached = _ARCHIVES.get(key)
        if cached is None or cached[0] != stamp:
            cached = (stamp, DiagnosticsArchive(parts))
            _ARCHIVES[key] = cached
        _ARCHIVES.move_to_end(key)
        while len(_ARCHIVES) > _ARCHIVES_MAX:
            _ARCHIVES.popitem(last=False)
        return cached[1]


# The scratch dir this process's run writes each job's diagnostics to, so a
# reader in the same process finds a stream written but not yet archived.
_SCRATCH: dict[Path, Path] = {}


def register_scratch(job_dir: str | Path, scratch: str | Path | None) -> None:
    """Record (or, with None, forget) where ``job_dir``'s run writes."""
    key = Path(job_dir).resolve()
    if scratch is None:
        _SCRATCH.pop(key, None)
    else:
        _SCRATCH[key] = Path(scratch)


def read_job_file(job_dir: str | Path, relpath: str) -> bytes | None:
    """One job-dir-relative diagnostics file, wherever it is: loose in the
    job dir (a run without scratch, or before archiving existed), in this
    process's scratch dir for it, or archived. None when it is nowhere."""
    loose = Path(job_dir) / relpath
    if loose.is_file():
        return loose.read_bytes()
    scratch = _SCRATCH.get(Path(job_dir).resolve())
    if scratch is not None and (scratch / relpath).is_file():
        return (scratch / relpath).read_bytes()
    archive = job_archive(job_dir)
    return None if archive is None else archive.read(relpath)


def job_file_version(job_dir: str | Path, relpath: str) -> tuple[str, int, int] | None:
    """What identifies the bytes :func:`read_job_file` would return, without
    reading them: the loose or scratch file's size and mtime, or else the
    archive part holding it and that part's size (a part rewritten in place
    changes size; a live run adds parts). None when the file is nowhere."""
    scratch = _SCRATCH.get(Path(job_dir).resolve())
    for path in (Path(job_dir) / relpath, None if scratch is None else scratch / relpath):
        if path is not None and path.is_file():
            stat = path.stat()
            return str(path), stat.st_size, stat.st_mtime_ns
    archive = job_archive(job_dir)
    part = None if archive is None else archive.part(relpath)
    return None if part is None else (str(part), part.stat().st_size, 0)


# One reading store per job dir, so its pack index — hundreds of thousands
# of entries in a long run — is parsed once, not on every block a viewer
# serves. Rebuilt with the archive when a live run adds parts; few are
# kept, since each holds its index and every archived pack it has read.
_STORES_MAX = 4
_STORES: collections.OrderedDict[tuple[Path, Path], tuple[BlockStore, threading.Lock]] = collections.OrderedDict()
_STORES_LOCK = threading.Lock()


def job_block_store(job_dir: str | Path) -> BlockStore:
    """A block store reading ``job_dir``'s blocks wherever they are, shared
    by every reader in the process; its index is read before it is handed
    out, once, however many readers ask for it at the same time."""
    scratch = _SCRATCH.get(Path(job_dir).resolve())
    directory = blocks_dir(job_dir)
    if scratch is not None and not directory.is_dir():
        directory = blocks_dir(scratch)
    archive = job_archive(job_dir)
    key = (Path(job_dir).resolve(), directory)
    with _STORES_LOCK:
        cached = _STORES.get(key)
        if cached is None or cached[0].archive is not archive:
            cached = (BlockStore(directory, archive=archive), threading.Lock())
            _STORES[key] = cached
        _STORES.move_to_end(key)
        while len(_STORES) > _STORES_MAX:
            _STORES.popitem(last=False)
    store, index_lock = cached
    with index_lock:
        store._read_index()
    return store


def game_run_ids(job_dir: str | Path) -> list[str]:
    """Every game run with a stream, loose or archived, sorted."""
    ids: set[str] = set()
    for root in (Path(job_dir), _SCRATCH.get(Path(job_dir).resolve())):
        games = None if root is None else root / "games"
        if games is not None and games.is_dir():
            ids.update(d.name for d in games.iterdir() if game_stream_path(d).is_file())
    archive = job_archive(job_dir)
    if archive is not None:
        ids.update(name.split("/")[1] for name in archive.names("games/") if name.endswith("/spans.jsonl"))
    return sorted(ids)


def monitor_files(job_dir: str | Path) -> list[str]:
    """Every host-monitor file's job-dir-relative path, loose, on this
    process's scratch, or archived; sorted."""
    names: set[str] = set()
    for root in (Path(job_dir), _SCRATCH.get(Path(job_dir).resolve())):
        directory = None if root is None else monitor_dir(root)
        if directory is not None and directory.is_dir():
            names.update(f"{MONITOR_DIR}/{entry.name}" for entry in os.scandir(directory))
    archive = job_archive(job_dir)
    if archive is not None:
        names.update(archive.names(f"{MONITOR_DIR}/"))
    return sorted(names)


def read_game_events(job_dir: str | Path, run_id: str) -> list[dict[str, Any]]:
    """One game's stream, wherever it is; empty when there is none."""
    return parse_events(read_job_file(job_dir, f"games/{run_id}/spans.jsonl") or b"")


def read_run_events(job_dir: str | Path) -> list[dict[str, Any]]:
    """The run-level stream, wherever it is; empty when there is none."""
    return parse_events(read_job_file(job_dir, run_stream_path(".").name) or b"")


def has_diagnostics(job_dir: str | Path) -> bool:
    """Whether a job dir holds any diagnostics, loose or archived."""
    job_dir = Path(job_dir)
    return run_stream_path(job_dir).exists() or (job_dir / "games").is_dir() or bool(archive_parts(job_dir))


class DiagnosticsArchiver:
    """Moves a run's diagnostics from the scratch dir they are written in to
    numbered zip parts in the job dir.

    A file goes in once it is final: a finished game's stream, a loose block
    (content-addressed, so never rewritten). ``final=True`` — the run's end —
    takes everything else. Two kinds of file go into every part as they
    stand, and again whenever they have grown, the later part winning: the
    run stream, a few hundred KB at most, so a viewer of a running job has
    the run's overview with the games archived so far; and every pack, live
    or not, since a finished game's stream may reference a small block a
    still-running process packed (dedup reaches across packs) and a part
    must hold what its streams reference. A pack is append-only and its
    index is written after its data, so a snapshot that takes the index
    before the data is consistent; packs are small (hundreds of KB), so the
    re-copies cost little. The host monitors' files go in once final, their
    stall log (small, and written only when a stall happens) whenever it
    grows. Thread-safe, so a periodic archive on a worker thread cannot
    interleave with the last."""

    def __init__(self, scratch: str | Path, job_dir: str | Path):
        """An archiver from ``scratch`` into ``job_dir``."""
        self.scratch = Path(scratch)
        self.job_dir = Path(job_dir)
        self._archived: dict[str, int] = {}  # scratch-relative path -> size when archived
        self._next = len(archive_parts(job_dir)) + 1
        self._lock = threading.Lock()

    def archive(self, finished_runs: Collection[str] = (), *, final: bool = False) -> Path | None:
        """Write one part holding every final file not yet in one, and
        return it; None when there is nothing to add."""
        with self._lock:
            members = self._final_files(finished_runs, final)
            if not members:
                return None
            part = self.job_dir / f"diagnostics-{self._next:04d}.zip"
            tmp = part.with_name(f".{part.name}.tmp")
            # Level 1: most of the bytes are PNGs and packs, which compress
            # little, and the streams compress well at any level.
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for relpath, path, _ in members:
                    archive.write(path, relpath)
            os.replace(tmp, part)
            for relpath, _, size in members:
                self._archived[relpath] = size
            self._next += 1
            return part

    def finish(self) -> Path | None:
        """Archive everything left, then delete the scratch dir."""
        part = self.archive(final=True)
        shutil.rmtree(self.scratch, ignore_errors=True)
        return part

    def _final_files(self, finished_runs: Collection[str], final: bool) -> list[tuple[str, Path, int]]:
        """(relpath, path, size) of every final file not yet archived at that size."""
        found: list[tuple[str, Path, int]] = []

        def consider(path: Path) -> None:
            relpath = path.relative_to(self.scratch).as_posix()
            size = path.stat().st_size
            if self._archived.get(relpath) != size:
                found.append((relpath, path, size))

        games = self.scratch / "games"
        if final and games.is_dir():
            finished_runs = [d.name for d in games.iterdir() if d.is_dir()]
        for run_id in finished_runs:
            stream = game_stream_path(games / run_id)
            if stream.is_file():
                consider(stream)
        if run_stream_path(self.scratch).is_file():
            consider(run_stream_path(self.scratch))  # a snapshot until the last part
        monitor = monitor_dir(self.scratch)
        if monitor.is_dir():
            # A closed pressure segment goes in once, the stall log again
            # whenever it grows, a segment still being written only in the
            # last part.
            for entry in os.scandir(monitor):
                if final or not entry.name.endswith(MONITOR_LIVE_SUFFIX):
                    consider(Path(entry.path))
        blocks = blocks_dir(self.scratch)
        if blocks.is_dir():
            packs: list[Path] = []
            for entry in os.scandir(blocks):
                if entry.is_dir():  # a prefix dir of loose blocks
                    for block in os.scandir(entry.path):
                        if not block.name.startswith(".tmp-"):
                            consider(Path(block.path))
                    continue
                if _PACK_NAME.fullmatch(entry.name):
                    packs.append(Path(entry.path))
            # The index before its data, so no entry points past the data
            # the same part holds.
            for pack in sorted(packs, key=lambda p: (p.stem, p.suffix != ".idx")):
                consider(pack)
        return found


def parse_events(raw: bytes) -> list[dict[str, Any]]:
    """One stream's bytes as events, tolerating a truncated last line (the
    crash-tail case). A malformed line anywhere ELSE is corruption and
    raises."""
    events: list[dict[str, Any]] = []
    lines = raw.split(b"\n")
    for i, line in enumerate(lines):
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:  # partial trailing write from an untidy death
                break
            raise
    return events


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read one stream file back (``parse_events``); empty when it is missing."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return []
    return parse_events(raw)


@dataclass
class SpanNode:
    """One span rehydrated from a stream, with children. ``closed`` is
    False for in-flight/aborted spans; their totals are summed from
    children (self spend of an unclosed span is unrecoverable)."""

    span_id: int
    kind: str
    name: str
    t_open: float
    parent_id: int | None = None
    closed: bool = False
    outcome: str | None = None
    duration_s: float | None = None
    self_generated: int = 0
    self_uncached: int = 0
    incl_generated: int = 0
    incl_uncached: int = 0
    kv: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    artifacts: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    events: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    children: list[SpanNode] = field(default_factory=lambda: list["SpanNode"]())


def build_span_tree(events: list[dict[str, Any]]) -> list[SpanNode]:
    """Events -> forest of roots (in open order). Unclosed spans get
    inclusive totals summed from their (recursively aggregated)
    children."""
    nodes: dict[int, SpanNode] = {}
    roots: list[SpanNode] = []
    for event in events:
        e = event["e"]
        if e == "open":
            node = SpanNode(
                span_id=event["id"],
                kind=event["kind"],
                name=event["name"],
                t_open=event["t"],
                parent_id=event["parent"],
            )
            nodes[node.span_id] = node
            parent = nodes.get(event["parent"]) if event["parent"] is not None else None
            (roots if parent is None else parent.children).append(node)
        elif e == "close":
            node = nodes[event["id"]]
            node.closed = True
            node.outcome = event["outcome"]
            node.duration_s = event["duration_s"]
            node.self_generated = event["self_generated"]
            node.self_uncached = event["self_uncached"]
            node.incl_generated = event["incl_generated"]
            node.incl_uncached = event["incl_uncached"]
        elif e == "kv":
            nodes[event["span"]].kv.update(event["values"])
        elif e == "artifact":
            nodes[event["span"]].artifacts.append(event)
        elif e == "event":
            nodes[event["span"]].events.append(event)
        else:
            raise ValueError(f"unknown span event type {e!r}")
    for root in roots:
        _aggregate_unclosed(root)
    return roots


def _aggregate_unclosed(node: SpanNode) -> None:
    """Roll child totals into spans that never closed, so an aborted tree
    still reports what happened underneath it."""
    for child in node.children:
        _aggregate_unclosed(child)
    if not node.closed:
        node.incl_generated = sum(child.incl_generated for child in node.children)
        node.incl_uncached = sum(child.incl_uncached for child in node.children)


# ---------------------------------------------------------------------------
# Token reconciliation (span sums vs TAAF ActionRecords; the
# host-backend leg lives in LLMAgentSolver, next to the backend)
# ---------------------------------------------------------------------------


def reconcile_game_tokens(job_dir: str | Path, run: Any) -> str | None:
    """Compare one game's span-stream inclusive totals against the tokens
    it reported into TAAF (``ActionRecord`` sums + finals). Returns a
    mismatch description, or ``None`` when they agree — or when no stream
    exists (diagnostics was off). ``run`` is a ``taaf.game.GameRun``."""
    events = read_game_events(job_dir, run.run_id)
    if not events:
        return None
    roots = build_span_tree(events)
    if not roots:
        return None
    root = roots[0]
    reported_generated = sum(rec.generated_tokens for rec in run.history) + run.final_generated_tokens
    reported_uncached = sum(rec.uncached_input_tokens for rec in run.history) + run.final_uncached_input_tokens
    if (root.incl_generated, root.incl_uncached) == (reported_generated, reported_uncached):
        return None
    return (
        f"game {run.game_id!r} ({run.run_id[:8]}): span stream totals "
        f"generated={root.incl_generated}, uncached_input={root.incl_uncached} vs "
        f"TAAF-reported generated={reported_generated}, uncached_input={reported_uncached}"
    )


def reconcile_job_dir(job_dir: str | Path) -> list[str]:
    """Standalone reconciliation over a finished job_dir: every game run
    in ``benchmark.json`` against its span stream. Returns mismatch
    descriptions (empty = all agree). Unclean runs (crashed/cancelled/
    in-flight) are skipped — both sides legitimately lose their tails."""
    import taaf.game

    benchmark = json.loads((Path(job_dir) / "benchmark.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for entry in benchmark["game_runs"]:
        run = taaf.game.GameRun.from_json_dict(entry)
        if run.state not in ("won", "gave_up"):
            continue
        message = reconcile_game_tokens(job_dir, run)
        if message is not None:
            mismatches.append(message)
    return mismatches
