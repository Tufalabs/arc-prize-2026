"""On-demand live view of running turn managers, over the job dir.

While a turn runs, the game process writes its live snapshot to
``games/<run_id>/live_snapshot.json`` — but ONLY while a viewer is
asking: the viewer touches ``live_request`` whenever its live page is
open, and the writer skips writing whenever that flag is stale. So live
viewing is pure job-dir files: it works across hosts over NFS (one
viewer on the login node reaches a run on any compute node), and does
nothing when nobody watches.

Why files and not a socket: a Unix socket — even a pathname one on a
shared filesystem — is bound to the host whose process created it and
can't be reached from another host; the socket file is only a local
rendezvous inode. Files are the portable transport.

Freshness uses wall-clock timestamps in the file contents, compared
across the two processes — it assumes the nodes are NTP-synced (true on
a SLURM cluster) and uses generous windows. Reads open the file (NFS
close-to-open revalidates the cache) rather than stat-ing it, so a
compute node sees the login node's touch promptly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import ullm.conv
import ullm.manager

REQUEST_TTL_S = 20.0  # writer streams while the request flag is this fresh
WRITE_INTERVAL_S = 1.0  # snapshot cadence (below the viewer's 5 s refresh)
SNAPSHOT_MAX_AGE_S = 8.0  # viewer treats an older snapshot as "no active turn"


def request_path(game_dir: str | Path) -> Path:
    """The flag a viewer touches to ask this game to stream."""
    return Path(game_dir) / "live_request"


def snapshot_path(game_dir: str | Path) -> Path:
    """Where the running turn's periodic snapshot lands."""
    return Path(game_dir) / "live_snapshot.json"


def conversation_path(game_dir: str | Path) -> Path:
    """Where the running turn's conversation lands (written only as it grows)."""
    return Path(game_dir) / "live_conversation.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a same-directory temp file and rename, so a reader
    tailing a live run never sees a half-written file — the rename is what
    NFS makes atomic, which is why the temp file cannot live elsewhere."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp-{os.getpid()}-{path.name}")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)  # same-dir rename: atomic on NFS


def _read_json(path: Path) -> dict[str, Any] | None:
    """Parse a JSON file, or None if it is missing or mid-write."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- viewer side ----------------------------------------------------------


def touch_request(game_dir: str | Path) -> None:
    """Tell the game process a viewer is watching (called on every live
    page load). Cheap; the writer streams for ``REQUEST_TTL_S`` after."""
    with contextlib.suppress(OSError):
        _atomic_write_json(request_path(game_dir), {"t": time.time()})


def read_snapshot(game_dir: str | Path) -> dict[str, Any] | None:
    """The game's latest live snapshot, or ``None`` when there is no fresh
    one (no turn running, run finished, or nobody has been watching).
    Safe to call from the viewer's request thread — it is a plain file
    read."""
    data = _read_json(snapshot_path(game_dir))
    if data is None:
        return None
    if time.time() - float(data.get("t", 0.0)) > SNAPSHOT_MAX_AGE_S:
        return None
    return data


def read_conversation(game_dir: str | Path) -> dict[str, Any] | None:
    """The full serialized conversation of the running turn (capture
    format, block refs into the diagnostics store), or ``None``. Not
    freshness-gated — it is the committed history the ``read_snapshot``
    partial extends; the snapshot's freshness decides whether a turn is
    live at all. Rewritten only when the conversation grows, so it is far
    cheaper than the 1 Hz snapshot."""
    data = _read_json(conversation_path(game_dir))
    return data if isinstance(data, dict) else None


# --- game side ------------------------------------------------------------


def start_live_writer(
    game_dir: str | Path, *, span_id: int, label: str, manager: ullm.manager.TurnManager
) -> asyncio.Task[None]:
    """Spawn the per-turn snapshot writer. Cancel the returned task when
    the turn ends. It writes a snapshot only while the request flag is
    fresh, so an unwatched turn only ever reads one small file per
    ``WRITE_INTERVAL_S`` (both file ops run in a thread — NFS latency
    never blocks the game's event loop)."""
    return asyncio.create_task(_writer(Path(game_dir), span_id, label, manager, time.time()))


async def _writer(game_dir: Path, span_id: int, label: str, manager: ullm.manager.TurnManager, t_start: float) -> None:
    """Publish 1 Hz snapshots of the running turn for as long as a viewer's
    live page keeps the request flag fresh, then stop. Writes nothing while
    nobody is watching, which is what makes it safe to leave armed."""
    request = request_path(game_dir)
    snapshot = snapshot_path(game_dir)
    conversation = conversation_path(game_dir)
    last_version = -1
    while True:
        if await asyncio.to_thread(_requested, request):
            payload: dict[str, Any] = {
                "active": True,
                "span_id": span_id,
                "label": label,
                "elapsed_s": round(time.time() - t_start, 1),
                "t": time.time(),
            }
            try:
                payload.update(manager.live_snapshot())
            except Exception:
                pass  # a peek hiccup must never break the turn
            await asyncio.to_thread(_atomic_write_json, snapshot, payload)
            # Full conversation, only when it grew (round boundaries), so
            # the 1 Hz path stays tiny. Serialize ON the loop (the turn
            # mutates the conversation on this same loop — no concurrent
            # write); the file/block I/O goes to a thread.
            version = _conversation_version(manager)
            if version != last_version:
                structure = _serialize_conversation(game_dir, manager)
                if structure is not None:
                    await asyncio.to_thread(
                        _atomic_write_json, conversation, {"t": time.time(), "structure": structure}
                    )
                    last_version = version
        await asyncio.sleep(WRITE_INTERVAL_S)


def _conversation_version(manager: ullm.manager.TurnManager) -> int:
    """A cheap change-marker for the conversation: total element count,
    plus the last turn's element count (which grows within a turn as
    responses/tool results land)."""
    elements = manager.conversation.elements
    tail = elements[-1] if elements else None
    inner = len(tail.elements) if isinstance(tail, ullm.conv.ResponseTurn) else 0
    return len(elements) * 1000 + inner


def _serialize_conversation(game_dir: Path, manager: ullm.manager.TurnManager) -> Any | None:
    """Serialize the manager's conversation into the diagnostics block
    store (blocks shared with the recorded artifact). Returns the
    block-referencing structure, or None if the block store isn't
    reachable. Runs synchronously on the loop."""
    import platypus.diagnostics
    import platypus.llm

    diag = getattr(manager, "diag", None)
    store = diag.block_store if diag is not None else None
    if store is None:
        store = platypus.diagnostics.BlockStore(platypus.diagnostics.blocks_dir(game_dir.parent.parent))
    try:
        return platypus.llm.serialize_conversation_blocks(manager.conversation, store)
    except Exception:
        return None


def _requested(request: Path) -> bool:
    """Whether a viewer asked for this game recently enough to still be watching."""
    data = _read_json(request)
    if data is None:
        return False
    return time.time() - float(data.get("t", 0.0)) <= REQUEST_TTL_S
