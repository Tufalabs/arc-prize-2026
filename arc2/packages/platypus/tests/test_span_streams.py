"""Span-stream core (platypus.diagnostics): writer, nesting, totals,
normalization, block store, crash-tail reading."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from platypus.diagnostics import (
    NULL_SPAN,
    BlockStore,
    SpanLogger,
    build_span_tree,
    current_span,
    read_events,
)


def _logger(tmp_path: Path) -> SpanLogger:
    return SpanLogger(tmp_path / "spans.jsonl", BlockStore(tmp_path / "blocks"))


def test_nesting_totals_and_ambient_current(tmp_path: Path) -> None:
    logger = _logger(tmp_path)
    with logger.span("infra/game", "g1"):
        with logger.span("infra/move") as move:  # ambient parent = game
            assert current_span() is move
            move.add_usage(100, 10)
            with move.span("agent/search") as search:
                search.add_usage(7, 3)
                search.kv(nodes=41, exit_reason="win")
        with logger.span("infra/move") as move2:
            move2.add_usage(1, 1)
    assert current_span() is NULL_SPAN
    logger.close()

    roots = build_span_tree(read_events(tmp_path / "spans.jsonl"))
    assert len(roots) == 1
    game_node = roots[0]
    assert game_node.kind == "infra/game" and game_node.name == "g1"
    assert [c.kind for c in game_node.children] == ["infra/move", "infra/move"]
    move_node = game_node.children[0]
    search_node = move_node.children[0]
    assert search_node.kv == {"nodes": 41, "exit_reason": "win"}
    # Inclusive totals aggregate through parents; self stays own spend.
    assert (search_node.self_generated, search_node.incl_generated) == (7, 7)
    assert (move_node.self_generated, move_node.incl_generated) == (100, 107)
    assert (game_node.self_generated, game_node.incl_generated) == (0, 108)
    assert game_node.incl_uncached == 14
    assert all(node.closed and node.outcome == "ok" for node in (game_node, move_node, search_node))
    assert move_node.duration_s is not None


def test_cancellation_outcome(tmp_path: Path) -> None:
    """CancelledError unwinding through spans closes them as "cancelled"
    (routine control flow, not error) with exact totals."""
    logger = _logger(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        with logger.span("infra/game"):
            with logger.span("infra/move") as move:
                move.add_usage(9, 4)
                raise asyncio.CancelledError()
    logger.close()
    game_node = build_span_tree(read_events(tmp_path / "spans.jsonl"))[0]
    (move_node,) = game_node.children
    assert move_node.outcome == "cancelled" and game_node.outcome == "cancelled"
    assert "CancelledError" in move_node.kv["error"]
    assert game_node.incl_generated == 9  # totals still exact — closed, not aborted


def test_error_outcome_and_set_outcome(tmp_path: Path) -> None:
    logger = _logger(tmp_path)
    with pytest.raises(RuntimeError):
        with logger.span("infra/game"):
            with logger.span("infra/move") as move:
                move.set_outcome("executed:ACTION1")
            with logger.span("infra/move") as move2:
                move2.set_outcome("would-be-overridden")
                raise RuntimeError("agent crashed")
    logger.close()
    game_node = build_span_tree(read_events(tmp_path / "spans.jsonl"))[0]
    stamped, crashed = game_node.children
    assert stamped.outcome == "executed:ACTION1"  # I12 stamping
    assert crashed.outcome == "error"  # the exception wins over set_outcome
    assert "agent crashed" in crashed.kv["error"]
    assert game_node.outcome == "error"


def test_kv_normalization(tmp_path: Path) -> None:
    logger = _logger(tmp_path)
    with logger.span("x") as span:
        span.kv(count=np.int64(5), ratio=np.float32(0.5), flag=np.bool_(True), none=None)
        with pytest.raises(TypeError, match="artifact"):
            span.kv(structured={"a": 1})
        with pytest.raises(TypeError, match="artifact"):
            span.kv(array=np.zeros(3))
        span.kv(long="x" * 5000)  # auto-blocks instead of raising
    logger.close()
    node = build_span_tree(read_events(tmp_path / "spans.jsonl"))[0]
    assert node.kv["count"] == 5 and node.kv["ratio"] == 0.5 and node.kv["flag"] is True
    long_ref = node.kv["long"]
    assert set(long_ref) == {"$block", "codec"}
    assert BlockStore(tmp_path / "blocks").load(long_ref["$block"], long_ref["codec"]) == "x" * 5000


def test_artifact_codecs_roundtrip(tmp_path: Path) -> None:
    logger = _logger(tmp_path)
    grid = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64) % 16
    payload = {"frames": [1, 2], "note": "repro"}
    with logger.span("x") as span:
        span.artifact("grid", grid, codec="png-grid")
        span.artifact("data", payload)
        span.artifact("bundle", {"live": object}, codec="pickle")
        with pytest.raises(ValueError, match="unknown codec"):
            span.artifact("bad", 1, codec="msgpack")
    logger.close()
    node = build_span_tree(read_events(tmp_path / "spans.jsonl"))[0]
    blocks = BlockStore(tmp_path / "blocks")
    by_name = {a["name"]: a for a in node.artifacts}
    assert np.array_equal(blocks.load(by_name["grid"]["block"], "png-grid"), grid)
    assert blocks.load(by_name["data"]["block"], "json") == payload
    assert blocks.load(by_name["bundle"]["block"], "pickle") == {"live": object}
    # PNG is dramatically smaller than the raw grid pickle — and small enough
    # to be PACKED, so size() (loose or packed) is the storage-agnostic read.
    # A fresh BlockStore (different pid, no in-memory pack cache) proves the
    # packed grid resolves purely from the on-disk index.
    assert BlockStore(tmp_path / "blocks").size(by_name["grid"]["block"]) < 1000


def test_block_store_dedups(tmp_path: Path) -> None:
    """Storing identical bytes twice stores them once — for both a large
    payload (one loose file) and a small one (a single pack entry, not a
    second append)."""
    store = BlockStore(tmp_path / "blocks")
    big = {"pad": "x" * 8192}  # over the pack threshold -> loose file
    small = {"same": "payload"}  # under it -> packed
    assert store.store(big, "json") == store.store(big, "json")
    ksmall = store.store(small, "json")
    assert ksmall == store.store(small, "json")
    loose = [p for p in (tmp_path / "blocks").rglob("*") if p.is_file() and not p.name.startswith("pack-")]
    assert len(loose) == 1, "the large payload should dedup to a single loose file"
    # Dedup of the packed block: the second store did not re-append.
    idx = (tmp_path / "blocks").glob("pack-*.idx")
    entries = sum(len(p.read_text().splitlines()) for p in idx)
    assert entries == 1, "the small payload should be a single pack-index entry"
    # And it still loads back through the pack.
    assert store.load(ksmall, "json") == small


def test_crash_tail_tolerated_midfile_corruption_raises(tmp_path: Path) -> None:
    logger = _logger(tmp_path)
    with logger.span("infra/game"):
        with logger.span("infra/move") as move:
            move.add_usage(50, 5)
    logger.close()
    path = tmp_path / "spans.jsonl"
    complete = path.read_text().splitlines()

    # Drop the game's close line and append a torn partial write: the
    # untidy-death shape. Reading tolerates the tail; the tree marks the
    # game span unclosed and sums its totals from the closed child.
    path.write_text("\n".join(complete[:-1]) + '\n{"e": "clo')
    events = read_events(path)
    game_node = build_span_tree(events)[0]
    assert not game_node.closed and game_node.outcome is None
    assert game_node.incl_generated == 50  # aggregated from the closed move
    assert game_node.children[0].closed

    # The same malformed line ANYWHERE else is corruption, not a crash tail.
    path.write_text('{"e": "clo\n' + "\n".join(complete))
    with pytest.raises(json.JSONDecodeError):
        read_events(path)


def test_null_logger_is_inert(tmp_path: Path) -> None:
    logger = SpanLogger()
    assert not logger.active
    with logger.span("infra/game") as span:
        assert span is NULL_SPAN
        span.kv(anything=1)
        span.event("x", value=2)
        span.artifact("a", {"ok": True})
        with span.span("child") as child:
            child.add_usage(5, 5)
        # Misuse fails identically with diagnostics off, or it waits to be
        # found until someone runs the same code with diagnostics on.
        with pytest.raises(ValueError):
            span.artifact("bad", 1, codec="msgpack")
        with pytest.raises(TypeError):
            span.kv(structured={"not": "a scalar"})
        with pytest.raises(TypeError):
            span.event("x", structured=[1, 2, 3])
    assert list(tmp_path.iterdir()) == []


async def test_sibling_tasks_do_not_share_current_span(tmp_path: Path) -> None:
    """Two games hosted concurrently in one process (inprocess mode) must
    not attach spans to each other's trees: the ambient span is
    per-task."""
    logger_a = _logger(tmp_path / "a")
    logger_b = _logger(tmp_path / "b")

    async def game(logger: SpanLogger, tag: str, barrier: asyncio.Barrier) -> None:
        with logger.span("infra/game", tag):
            await barrier.wait()  # both games now have their game span open
            with logger.span("infra/move") as move:
                move.kv(tag=tag)
            await barrier.wait()

    barrier = asyncio.Barrier(2)
    await asyncio.gather(game(logger_a, "a", barrier), game(logger_b, "b", barrier))
    logger_a.close()
    logger_b.close()
    for tag, logger_path in (("a", tmp_path / "a"), ("b", tmp_path / "b")):
        roots = build_span_tree(read_events(logger_path / "spans.jsonl"))
        assert len(roots) == 1 and roots[0].name == tag
        (move,) = roots[0].children
        assert move.kv == {"tag": tag}


def test_stream_is_readable_in_flight(tmp_path: Path) -> None:
    """Every event is flushed as written: a reader sees all completed
    events while spans are still open (the live-viewing contract)."""
    logger = _logger(tmp_path)
    with logger.span("infra/game") as game:
        game.kv(status="warming")
        events = read_events(tmp_path / "spans.jsonl")
        assert [e["e"] for e in events] == ["open", "kv"]
        node = build_span_tree(events)[0]
        assert not node.closed and node.kv == {"status": "warming"}
    logger.close()


def test_pack_threshold_splits_small_from_large(tmp_path: Path) -> None:
    """Small payloads pack; large ones stay loose. The split is what turns a
    per-file explosion into a couple of pack files without losing the
    addressability that matters for big blocks (images)."""
    store = BlockStore(tmp_path / "b")
    small = store.store({"s": "x" * 100}, "json")
    large = store.store({"s": "x" * 8192}, "json")
    assert store.path(large).exists(), "a large payload keeps its own loose file"
    assert not store.path(small).exists(), "a small payload has no loose file"
    assert store.load(small, "json") == {"s": "x" * 100}
    assert store.load(large, "json") == {"s": "x" * 8192}


def test_packed_block_reads_from_a_fresh_store(tmp_path: Path) -> None:
    """A reader is a different process than the writer (the viewer vs the
    run), so a packed block must resolve from the on-disk index alone, with
    no in-memory pack cache."""
    writer = BlockStore(tmp_path / "b")
    key = writer.store({"only": "in a pack"}, "json")
    reader = BlockStore(tmp_path / "b")  # cold: _packed empty, index unread
    assert reader.load(key, "json") == {"only": "in a pack"}
    assert reader.size(key) == len(writer.raw_bytes(key))


def test_reader_merges_multiple_writers_packs(tmp_path: Path) -> None:
    """Concurrent game children each write their own pid-named pack; a reader
    must see blocks from all of them."""
    a = BlockStore(tmp_path / "b")
    a._pack_path = tmp_path / "b" / "pack-1.bin"
    a._index_path = tmp_path / "b" / "pack-1.idx"
    ka = a.store({"from": "writer-a"}, "json")
    b = BlockStore(tmp_path / "b")
    b._pack_path = tmp_path / "b" / "pack-2.bin"
    b._index_path = tmp_path / "b" / "pack-2.idx"
    kb = b.store({"from": "writer-b"}, "json")
    reader = BlockStore(tmp_path / "b")
    assert reader.load(ka, "json") == {"from": "writer-a"}
    assert reader.load(kb, "json") == {"from": "writer-b"}


def test_reader_tolerates_a_torn_final_index_line(tmp_path: Path) -> None:
    """A live run is mid-append: the index's last line may be partial. The
    reader must serve every complete entry and not choke on the torn one."""
    store = BlockStore(tmp_path / "b")
    good = store.store({"k": "complete"}, "json")
    # Simulate a half-written trailing index line, as a tail read might catch.
    with open(store._index_path, "a", encoding="utf-8") as fh:
        fh.write('{"k": "abc", "o": 999')  # no newline, unterminated JSON
    reader = BlockStore(tmp_path / "b")
    assert reader.load(good, "json") == {"k": "complete"}
