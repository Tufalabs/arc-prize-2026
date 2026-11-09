"""Recorded LLM traffic, replayed against a live engine: the traffic benchmark.

A *workload* is what a recorded run asked of its engine, cut down to what a
replay needs -- token ids, images and reply lengths -- and organised by how the
run's slots were used:

- a *sequence* is one game's calls from a slot grant to the slot's next change
  of hands (a compaction or the game's end), each call normally extending the
  last;
- a *lane* is the chain of sequences one slot served in turn: a sequence
  follows the one whose end freed its slot.

``build_workload`` cuts a workload out of recorded runs at a snapshot moment of
the recording; ``Workload.save`` / ``Workload.load`` store it; ``replay`` runs
it against an engine that serves id prompts (``VLLM.serve_ids``).

A replay has no scheduler of its own. The recording's slot decisions are
replayed as they fell: a lane's next sequence starts when its previous one
ends. Each call's prompt is the lane's own history up to the prefix the
recorded prompt shared with the recorded previous prompt and reply, then the
recorded rest, and its reply is forced to the recorded length. So the prefix
cache sees the recording's structure, and every engine configuration does the
same work.
"""

import asyncio
import base64
import contextlib
import dataclasses
import io
import itertools
import json
import pathlib
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Mapping, Optional, Protocol, Sequence

import numpy as np

import ullm.backend

WORKLOAD_FORMAT = "ullm.traffic.workload.v2"


# ---------------------------------------------------------------------------
# What a recording provides
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedCall:
    """One model call a recorded run made, in the run's order."""

    input_ids: np.ndarray  # int32: the prompt as the engine consumed it, image placeholders expanded
    output_ids: np.ndarray  # int32: what the engine sampled
    images: tuple[str, ...]  # content keys of the prompt's images, in placeholder order
    finished_at: float  # wall-clock seconds at which the call returned
    continued: bool  # the prompt extended the run's previous prompt and reply by ids
    opens: bool  # the run was granted a slot since its previous call
    after: Optional[tuple[str, int]]  # when ``opens``: (run_id, call index) whose end freed that slot


@dataclass(frozen=True)
class RecordedRun:
    """One recorded game run's calls."""

    run_id: str
    calls: tuple[RecordedCall, ...]


# ---------------------------------------------------------------------------
# The workload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One call a lane replays."""

    sequence: int  # index of its sequence within the lane
    shared: int  # prompt tokens taken from the lane's own history; 0 opens a sequence
    suffix: np.ndarray  # int32: the recorded prompt after the shared part
    images: tuple[str, ...]  # content keys of the prompt's images, in placeholder order
    output_len: int  # the recorded reply's length, which the replay forces


@dataclass(frozen=True)
class Lane:
    """One slot's calls from the snapshot on, and the history it holds there."""

    history: Optional[np.ndarray]  # int32 held at the snapshot; None when the first call opens a sequence
    history_images: tuple[str, ...]  # that history's images
    calls: tuple[Call, ...]

    @property
    def output_tokens(self) -> int:
        """Reply tokens the lane's calls generate in all."""
        return sum(call.output_len for call in self.calls)


@dataclass
class Workload:
    """Lanes cut from a recording, with the images they carry."""

    lanes: tuple[Lane, ...]
    media: dict[str, bytes]  # content key -> encoded image
    meta: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())  # source, snapshot, horizon

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write the workload as one file at ``path`` and return it: a zip of
        ``workload.json``, one int32 id array ``ids.npy`` and the images under
        ``media/``, base64-encoded.

        One file, so a deploy carries it as it carries any other payload file.
        The encoding is load-bearing, as for the engine's overlay
        (``ullm.vllm_overlay``): Kaggle's dataset ingestion unpacks an archive
        it recognizes, so a plain zip reaches the kernel as a directory.
        """
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        chunks: list[np.ndarray] = []
        offset = 0

        # Every id run becomes an (offset, length) slice of one array.
        def slot(ids: np.ndarray) -> list[int]:
            nonlocal offset
            chunks.append(ids.astype(np.int32, copy=False))
            start, offset = offset, offset + len(ids)
            return [start, len(ids)]

        lanes = [
            {
                "history": None if lane.history is None else slot(lane.history),
                "history_images": list(lane.history_images),
                "calls": [
                    {
                        "sequence": call.sequence,
                        "shared": call.shared,
                        "suffix": slot(call.suffix),
                        "images": list(call.images),
                        "output_len": call.output_len,
                    }
                    for call in lane.calls
                ],
            }
            for lane in self.lanes
        ]
        ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
        ids_npy = io.BytesIO()
        np.save(ids_npy, ids)
        document = {"format": WORKLOAD_FORMAT, "meta": self.meta, "lanes": lanes}
        # The ids and the index compress; the images are compressed already.
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w") as archive:
            archive.writestr("workload.json", json.dumps(document, indent=1) + "\n", zipfile.ZIP_DEFLATED)
            archive.writestr("ids.npy", ids_npy.getvalue(), zipfile.ZIP_DEFLATED)
            for key, data in self.media.items():
                archive.writestr(f"media/{key}", data)
        target.write_bytes(base64.b64encode(packed.getvalue()))
        return target

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Workload":
        """Read a workload ``save`` wrote."""
        packed = io.BytesIO(base64.b64decode(pathlib.Path(path).read_bytes(), validate=True))
        with zipfile.ZipFile(packed) as archive:
            document = json.loads(archive.read("workload.json"))
            if document.get("format") != WORKLOAD_FORMAT:
                raise ValueError(f"{path}: not a {WORKLOAD_FORMAT} workload (format {document.get('format')!r})")
            ids = np.load(io.BytesIO(archive.read("ids.npy")))
            media = {
                name.removeprefix("media/"): archive.read(name)
                for name in archive.namelist()
                if name.startswith("media/")
            }

        # An (offset, length) slice back to its ids.
        def ids_at(slot: Sequence[int]) -> np.ndarray:
            return ids[slot[0] : slot[0] + slot[1]]

        lanes = tuple(
            Lane(
                history=None if lane["history"] is None else ids_at(lane["history"]),
                history_images=tuple(lane["history_images"]),
                calls=tuple(
                    Call(
                        sequence=call["sequence"],
                        shared=call["shared"],
                        suffix=ids_at(call["suffix"]),
                        images=tuple(call["images"]),
                        output_len=call["output_len"],
                    )
                    for call in lane["calls"]
                ),
            )
            for lane in document["lanes"]
        )
        keys = {key for lane in lanes for key in (*lane.history_images, *(k for c in lane.calls for k in c.images))}
        missing = keys - media.keys()
        if missing:
            raise ValueError(f"{path}: {len(missing)} images the lanes carry are not in the file")
        return cls(lanes=lanes, media=media, meta=dict(document.get("meta") or {}))


# ---------------------------------------------------------------------------
# Building one from a recording
# ---------------------------------------------------------------------------


@dataclass
class _Sequence:
    """A run's calls between two slot grants, as the builder tracks them."""

    run_id: str
    first: int  # the run's call index of its first call
    last: int  # ... and of its last
    after: Optional[tuple[str, int]]  # the call whose end freed its slot; None for a lane's first
    successor: Optional[int] = None  # index of the sequence that took its slot


def _shared_prefix(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the common prefix of two id arrays."""
    n = min(len(a), len(b))
    differ = np.flatnonzero(a[:n] != b[:n])
    return int(differ[0]) if differ.size else n


def _sequences(runs: Sequence[RecordedRun]) -> list[_Sequence]:
    """Split every run at the slot grants it recorded.

    A grant the run gave itself without its history breaking (a swap point it
    kept, the prompt still extending the last) is no boundary: nothing about
    the lane changed.
    """
    sequences: list[_Sequence] = []
    for run in runs:
        for index, call in enumerate(run.calls):
            kept_own_slot = call.after == (run.run_id, index - 1) and call.continued
            if index == 0 or (call.opens and not kept_own_slot):
                sequences.append(_Sequence(run.run_id, index, index, call.after if call.opens else None))
            else:
                sequences[-1].last = index
    return sequences


def _chain(sequences: list[_Sequence]) -> list[list[int]]:
    """The lanes: each sequence follows the one whose last call its ``after``
    names, and a sequence with nothing recorded to follow starts a lane."""
    ending_at = {(s.run_id, s.last): i for i, s in enumerate(sequences)}
    recorded = {(s.run_id, k) for s in sequences for k in range(s.first, s.last + 1)}
    heads: list[int] = []
    for i, sequence in enumerate(sequences):
        if sequence.after is None or sequence.after not in recorded:
            # A lane's first sequence, or one whose slot a run freed without
            # leaving a trace (it crashed before its trace was written): the
            # lane restarts here rather than splicing onto the wrong chain.
            heads.append(i)
            continue
        predecessor = ending_at.get(sequence.after)
        if predecessor is None:
            raise ValueError(f"run {sequence.run_id} opens after call {sequence.after}, which does not end a sequence")
        if sequences[predecessor].successor is not None:
            raise ValueError(f"two sequences follow the one ending at {sequence.after}")
        sequences[predecessor].successor = i
    lanes: list[list[int]] = []
    for head in heads:
        lane = [head]
        following = sequences[head].successor
        while following is not None:
            lane.append(following)
            following = sequences[following].successor
        lanes.append(lane)
    return lanes


def build_workload(
    runs: Sequence[RecordedRun],
    media: Mapping[str, bytes],
    *,
    horizon_tokens: int,
    snapshot: Optional[float] = None,
    settled_fraction: float = 0.9,
    meta: Optional[Mapping[str, Any]] = None,
    skip_in_flight: bool = False,
) -> Workload:
    """Cut a workload out of recorded runs at one moment of the recording.

    ``horizon_tokens`` is how many reply tokens every lane must at least still
    have to generate after the snapshot: it places the snapshot, and a lane
    keeps everything its slot recorded after it. How long a window that buys
    depends on the replaying engine's speed per lane; a replay that outruns it
    fails loudly. ``snapshot`` (wall-clock seconds) defaults to the
    middle of the span where the recording is settled -- ``settled_fraction``
    of its lanes past their first sequence, so the run's synchronised start
    has dispersed -- and every lane still holds the horizon.

    A lane's first call is the one in flight at the snapshot, which is
    length-biased: a moment falls inside a long call more often than a short
    one, so first calls are long reasoning turns far more often than calls
    are (on ARC2's 09-29 cut: a median reply of 7,244 tokens against 1,066
    for the calls after). A cold replay then spends its warm-up decoding them.
    ``skip_in_flight`` starts each lane at the call after it instead -- the
    first to start after the snapshot -- with the in-flight call's prompt
    and recorded reply as its history, or at its next sequence's opening
    call when the in-flight call ends its sequence.
    """
    by_run = {run.run_id: run for run in runs}
    sequences = _sequences(runs)
    lanes = _chain(sequences)

    # A sequence's calls, as recorded.
    def calls_of(i: int) -> tuple[RecordedCall, ...]:
        s = sequences[i]
        return by_run[s.run_id].calls[s.first : s.last + 1]

    def ends(i: int) -> float:
        return calls_of(i)[-1].finished_at

    # The settled span: late enough that most lanes have handed over at least
    # once, early enough that every lane still has the horizon left to play.
    first_ends = sorted(ends(lane[0]) for lane in lanes)
    settled_at = first_ends[min(len(first_ends) - 1, int(settled_fraction * len(first_ends)))]
    latest = float("inf")
    for lane in lanes:
        remaining = 0
        for call in reversed([c for i in lane for c in calls_of(i)]):
            remaining += len(call.output_ids)
            if remaining >= horizon_tokens:
                latest = min(latest, call.finished_at)
                break
        else:
            latest = float("-inf")  # this lane never holds the horizon
    if snapshot is None:
        if not settled_at < latest:
            raise ValueError(
                f"the recording has no settled moment with {horizon_tokens} reply tokens left on every lane "
                f"(settled at {settled_at:.0f}, horizon last held at {latest:.0f})"
            )
        snapshot = settled_at + (latest - settled_at) / 2

    cut: list[Lane] = []
    for lane in lanes:
        # The lane's sequence at the snapshot: the last that began by then.
        position = 0
        while position + 1 < len(lane) and ends(lane[position]) <= snapshot:
            position += 1
        s = sequences[lane[position]]
        run = by_run[s.run_id]
        first = next((k for k in range(s.first, s.last + 1) if run.calls[k].finished_at > snapshot), None)
        if first is None:
            raise ValueError(f"a lane's chain ends before the snapshot at {snapshot:.0f}")
        if skip_in_flight:
            if first < s.last:
                first += 1
            elif position + 1 < len(lane):
                position += 1
                s = sequences[lane[position]]
                run = by_run[s.run_id]
                first = s.first
            else:
                raise ValueError(f"a lane's chain ends with the call in flight at the snapshot {snapshot:.0f}")
        history: Optional[np.ndarray] = None
        history_images: tuple[str, ...] = ()
        if first > s.first:
            previous = run.calls[first - 1]
            history = np.concatenate([previous.input_ids, previous.output_ids]).astype(np.int32)
            history_images = previous.images
        calls: list[Call] = []
        for sequence_index, i in enumerate(lane[position:]):
            t = sequences[i]
            t_run = by_run[t.run_id]
            start = first if sequence_index == 0 else t.first
            for k in range(start, t.last + 1):
                call = t_run.calls[k]
                shared = 0
                if k > t.first:
                    previous = t_run.calls[k - 1]
                    shared = _shared_prefix(call.input_ids, np.concatenate([previous.input_ids, previous.output_ids]))
                calls.append(
                    Call(
                        sequence=sequence_index,
                        shared=shared,
                        suffix=call.input_ids[shared:].astype(np.int32),
                        images=call.images,
                        output_len=len(call.output_ids),
                    )
                )
        cut.append(Lane(history=history, history_images=history_images, calls=tuple(calls)))

    keys = {key for lane in cut for key in (*lane.history_images, *(k for c in lane.calls for k in c.images))}
    missing = sorted(keys - set(media))
    if missing:
        raise ValueError(f"the recording's calls cite images it does not hold: {missing[:3]}")
    return Workload(
        lanes=tuple(cut),
        media={key: media[key] for key in sorted(keys)},
        meta={
            **dict(meta or {}),
            "snapshot": snapshot,
            "settled_at": settled_at,
            "horizon_held_until": latest,
            "horizon_tokens": horizon_tokens,
            "recorded_lanes": len(lanes),
            "skip_in_flight": skip_in_flight,
        },
    )


# ---------------------------------------------------------------------------
# Replaying one
# ---------------------------------------------------------------------------


class ServesIds(Protocol):
    """An engine that serves prompts given as token ids (``VLLM.serve_ids``)."""

    def serve_ids(
        self,
        prompt_ids: Sequence[int],
        images: Sequence[Any],
        sampling_params: ullm.backend.SamplingParameters,
        request_id: str,
        prefix_cache_thread: Optional[tuple[str, int]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream the engine's ``RequestOutput``s for one prompt; closing the
        stream early aborts the request. ``prefix_cache_thread`` names the
        conversation and its history epoch to the engine's prefix cache."""
        ...


@dataclass
class _Tally:
    """Counters the sampler reads while the window runs."""

    generated: int = 0  # reply tokens streamed inside the window
    prompt: int = 0  # prompt tokens of the calls that started generating
    uncached: int = 0  # ... of which the engine computed rather than hit
    in_flight: int = 0  # calls submitted and not yet returned
    finished: int = 0  # calls returned inside the window


def _replay_params(base: ullm.backend.GenericSamplingParameters, length: int) -> ullm.backend.GenericSamplingParameters:
    """``base`` forced to exactly ``length`` reply tokens."""
    extra = {**base.extra_params, "min_tokens": length, "ignore_eos": True}
    return dataclasses.replace(base, max_tokens=length, extra_params=extra)


def _opening_head(workload: Workload) -> int:
    """Tokens every fresh prompt in the workload begins with -- the head all
    games share (system prompt, tool schema) -- which a restarted lane keeps,
    so it still hits there as a fresh game would."""
    heads = [call.suffix for lane in workload.lanes for call in lane.calls if call.shared == 0]
    heads += [lane.history for lane in workload.lanes if lane.history is not None]
    if len(heads) < 2:
        return 0
    return min(_shared_prefix(heads[0], head) for head in heads[1:])


def _with_nonce(prompt: np.ndarray, start: int, index: int) -> np.ndarray:
    """``prompt`` with one token at or past ``start`` replaced by its right
    neighbour: the ``index``-th position where that changes the prompt, so
    each index gives a prompt no other index gives.

    A position qualifies only where no two adjacent tokens around it are
    equal, which keeps the change out of an image's placeholder run and off
    the markers that bound it. The length is unchanged, so every later call
    that shares a prefix with this prompt still shares it at the same offset.
    """
    same = prompt[1:] == prompt[:-1]  # same[i]: prompt[i] == prompt[i + 1]
    n = len(prompt)
    if n < 5:
        raise ValueError(f"a {n}-token prompt has no position to carry a nonce")
    positions = np.arange(2, n - 2)
    ok = ~(same[positions - 2] | same[positions - 1] | same[positions] | same[positions + 1])
    candidates = positions[ok]
    past = candidates[candidates >= start]
    pool = past if len(past) > index else candidates
    if len(pool) <= index:
        raise ValueError(f"a {n}-token prompt has fewer than {index + 1} positions to carry a nonce")
    marked = prompt.copy()
    position = int(pool[index])
    marked[position] = prompt[position + 1]
    return marked


async def replay(
    engine: ServesIds,
    workload: Workload,
    *,
    lanes: int,
    window_s: float,
    gap_s: float,
    sampling_params: ullm.backend.GenericSamplingParameters,
    out_dir: Optional[str | pathlib.Path] = None,
    settle_s: float = 300.0,
    sample_s: float = 10.0,
    report_s: float = 60.0,
    restart: bool = True,
    tail_s: float = 0.0,
    placeholder_id: Optional[int] = None,
    record_outputs: bool = False,
    on_window: Optional[Callable[[dict[str, Any]], None]] = None,
    decode_gate: bool = False,
    on_gate_lifted: Optional[Callable[[float], None]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Replay ``workload``'s first ``lanes`` lanes and measure the window.

    Every lane fires its first call at once and plays on: each call waits
    ``gap_s`` after its lane's previous one returned, carries the lane's own
    history where the recording's prompt shared it, and is forced to its
    recorded reply length. At ``window_s`` the lanes are cancelled, which
    aborts whatever is in flight.

    Nothing is prefilled ahead of the window: the engine reaches the
    recording's state by serving the lanes, as it would in play. So the
    window's first ``settle_s`` is the run's warm-up -- every lane's first call
    is cold, and the pool and the host tier fill through it -- and the
    ``settled_`` rates, which are the reading, count from the first sample past
    it.

    A lane that runs out of calls before the window ends starts again, so the
    window keeps the asked concurrency however fast the engine is. It starts
    from its first call, its snapshot history cold again, so the window keeps
    the recording's balance of request sizes; the extra prefill is the price.
    Every fresh prompt of a restarted pass -- its first, and each sequence
    opening -- carries a nonce past the head all games share
    (``_with_nonce``), so it hits the cache where a new game would and no
    further, and differs from every earlier pass. The summary counts the
    restarts, and each row names its pass. With ``restart`` off, a lane that
    runs out raises ``RuntimeError`` instead, after the files are written.

    ``lanes`` may exceed the workload's: lane ``k`` then plays the workload's
    lane ``k`` mod its count, as a later copy (``copy`` in each row), so a
    replay runs at a column's own concurrency whatever the recording's. A
    copy's fresh prompts carry nonces as a restarted pass's do, one per copy
    and pass, so every copy is a game of its own to the cache.

    Each call names its conversation to the engine's prefix cache, as a
    game's calls do (``prefix_cache_thread``): a lane's sequence in one pass
    is one conversation, its history epoch steps where a call does not
    extend the lane's history (the recording rewrote it there), and its
    context is closed when the lane moves on. So an engine that keeps
    conversations live between turns (the host tier's park mode) sees the
    recording's threads and their ends.

    ``placeholder_id`` is the token that stands for an image in the engine's
    id prompts: a reply that samples it has it replaced by the token before
    it, so a later prompt carrying the reply stands for no image it does not
    carry. The reply's length, and the work, are unchanged.

    ``record_outputs`` keeps each returned call's reply ids, as served (before
    any placeholder is replaced), in its row as ``output_ids``: for reading
    what the engine wrote, at the price of a row file the size of the
    generated text.

    ``tail_s`` keeps the lanes playing that long past the window, counted in
    nothing (no rate, no restart): an engine at the same load for
    instrumentation that would disturb the reading, such as profiler windows.
    With a tail, the reading is taken and written as the window closes, so
    the tail cannot cost it: a profiler window that stalls or kills the
    engine leaves ``traffic_result.json`` in place. ``on_window`` is called
    with the reading there (at the end, without a tail), to report it.

    ``decode_gate`` opens the window warm, as play's pool stands rather than
    as a cold start leaves it: first, untimed, every lane's first prompt is
    prefilled on its own conversation thread by a one-token request, which
    ends at once, so the prompt's KV and states go where a parked turn's
    would. Then the window opens and each lane's first call continues that
    request -- its prompt plus the token it sampled, its reply one token
    short -- so it resumes at the parked conversation's end.
    An engine that times profiler windows from its start
    (``hold_profile_windows`` and ``delay_profile_windows``, which
    ``EngineFactory`` puts on its backend) has them held shut through the
    gate, however long it runs, and then moved past it, so a window meant for
    the tail stays there; ``on_gate_lifted`` is called with the gate's seconds
    as well.

    Every ``report_s`` a line on stdout gives the last stretch's rates, to
    read a live run by. Returns the summary; with ``out_dir`` it is also
    written there as ``traffic_result.json``, beside one row per call in
    ``traffic_calls.jsonl``.
    """
    recorded = len(workload.lanes)
    copies = -(-lanes // recorded)
    chosen = [workload.lanes[k % recorded] for k in range(lanes)]
    pictures: dict[str, Any] = {}
    request_ids = itertools.count()
    head = _opening_head(workload)
    restarts = [0] * lanes

    # Decoded once and shared: a lane's prompts carry the same images call
    # after call.
    def images_for(keys: Sequence[str]) -> list[Any]:
        from PIL import Image

        for key in keys:
            if key not in pictures:
                pictures[key] = Image.open(io.BytesIO(workload.media[key])).convert("RGB")
        return [pictures[key] for key in keys]

    def first_prompt(lane_index: int, lane: Lane) -> np.ndarray:
        """A lane's first call's prompt in its first pass, as ``play_call``
        builds it."""
        call = lane.calls[0]
        if call.shared:
            assert lane.history is not None  # a call sharing a prefix always has a history before it
            prompt = np.concatenate([lane.history[: call.shared], call.suffix])
        else:
            prompt = call.suffix
        copy = lane_index // recorded
        return _with_nonce(prompt, head, copy - 1) if copy else prompt

    # Per lane, under the decode gate: the first prompt prefilled and the
    # token its one-token request sampled, which the first call carries.
    gated: dict[int, tuple[np.ndarray, int]] = {}
    gate_s: Optional[float] = None
    gate_tokens = 0
    if decode_gate:

        async def prefill(lane_index: int, lane: Lane) -> None:
            call = lane.calls[0]
            prompt = first_prompt(lane_index, lane)
            stream = engine.serve_ids(
                prompt.tolist(),
                images_for(call.images),
                _replay_params(sampling_params, 1),
                f"traffic-gate-{lane_index}-{next(request_ids)}",
                prefix_cache_thread=(f"traffic-{lane_index}-0-{call.sequence}/thread", 0),
            )
            output = None
            async with contextlib.aclosing(stream):
                async for output in stream:
                    pass
            assert output is not None  # a returned stream produced at least one output
            token = int(output.outputs[0].token_ids[0])
            if placeholder_id is not None and token == placeholder_id:
                token = int(prompt[-1])
            gated[lane_index] = (prompt, token)

        gate_tokens = sum(len(first_prompt(k, lane)) for k, lane in enumerate(chosen))
        print(f"[traffic] decode gate: prefilling {lanes} lanes' first prompts, {gate_tokens} tokens", flush=True)
        hold = getattr(engine, "hold_profile_windows", None)
        if hold is not None:
            hold()
        gate_started = clock()
        await asyncio.gather(*(prefill(k, lane) for k, lane in enumerate(chosen)))
        gate_s = clock() - gate_started
        print(f"[traffic] decode gate lifted after {gate_s:.0f} s; the timed window opens", flush=True)
        delay = getattr(engine, "delay_profile_windows", None)
        if delay is not None:
            delay(gate_s)
        if on_gate_lifted is not None:
            on_gate_lifted(gate_s)

    print(
        f"[traffic] {lanes} lanes, {window_s:.0f} s, {gap_s:.1f} s between a lane's calls; "
        f"the rates are counted from {settle_s:.0f} s in",
        flush=True,
    )

    tally = _Tally()
    rows: list[dict[str, Any]] = []
    started = clock()
    window_end = started + window_s

    async def play(lane_index: int, lane: Lane) -> None:
        previous_end: Optional[float] = None
        for pass_index in itertools.count():
            if pass_index:
                if not restart:
                    return
                if clock() <= window_end:
                    restarts[lane_index] += 1
                print(
                    f"[traffic] {clock() - started:.0f}s: lane {lane_index} ran out of calls; "
                    f"pass {pass_index} starts from its first call",
                    flush=True,
                )
            history = lane.history
            for call_index in range(len(lane.calls)):
                await play_call(lane_index, lane, call_index, pass_index, history, previous_end)
                history, previous_end = last_history[lane_index], clock()

    last_history: dict[int, np.ndarray] = {}
    # Per lane: the conversation its calls name, and its history epoch.
    threads: dict[int, tuple[str, int]] = {}
    close_context = getattr(engine, "close_context", None)

    async def play_call(
        lane_index: int,
        lane: Lane,
        call_index: int,
        pass_index: int,
        history: Optional[np.ndarray],
        previous_end: Optional[float],
    ) -> None:
        call = lane.calls[call_index]
        if previous_end is not None:
            await asyncio.sleep(max(0.0, previous_end + gap_s - clock()))
        copy = lane_index // recorded
        gate = gated.pop(lane_index, None) if pass_index == 0 and call_index == 0 else None
        length = call.output_len
        if gate is not None:
            # The decode gate prefilled this prompt and sampled its first
            # reply token: the call continues from there.
            prompt = np.concatenate([gate[0], np.asarray([gate[1]], dtype=gate[0].dtype)])
            length -= 1
        elif call.shared:
            assert history is not None  # a call sharing a prefix always has a history before it
            prompt = np.concatenate([history[: call.shared], call.suffix])
        else:
            prompt = call.suffix
        if gate is None and (pass_index or copy) and (call.shared == 0 or call_index == 0):
            # A fresh prompt of a restarted pass or a later copy: marked so it
            # hits no further than the head all games share, and unlike any
            # other copy's or pass's (one nonce each).
            prompt = _with_nonce(prompt, head, pass_index * copies + copy - 1)
        thread = threads.get(lane_index)
        if thread is None or call_index == 0 or call.shared == 0:
            # A sequence opens: a conversation of its own. The one the lane
            # leaves has ended, as a game's does.
            if thread is not None and close_context is not None:
                await close_context(thread[0].rsplit("/", 1)[0])
            thread = (f"traffic-{lane_index}-{pass_index}-{call.sequence}/thread", 0)
        elif history is not None and call.shared < len(history):
            # The recording rewrote the history here (a compaction, a trim).
            thread = (thread[0], thread[1] + 1)
        threads[lane_index] = thread
        row: dict[str, Any] = {
            "lane": lane_index,
            "copy": copy,
            "pass": pass_index,
            "call": call_index,
            "sequence": call.sequence,
            "shared": call.shared,
            "prompt": len(prompt),
            "output_len": length,
            "submitted": clock() - started,
            "thread": thread[0],
            "epoch": thread[1],
        }
        if gate is not None:
            row["gated"] = True
        rows.append(row)
        if length == 0:
            # The gate's one token was the whole reply.
            row.update(first_token=row["submitted"], cached=len(prompt), generated=0, returned=row["submitted"])
            tally.finished += 1
            last_history[lane_index] = prompt
            return
        stream = engine.serve_ids(
            prompt.tolist(),
            images_for(call.images),
            _replay_params(sampling_params, length),
            f"traffic-{lane_index}-{next(request_ids)}",
            prefix_cache_thread=thread,
        )
        tally.in_flight += 1
        seen = 0
        output = None
        try:
            # Closed on the way out, so a call the window cuts is aborted
            # engine-side rather than left decoding for nobody.
            async with contextlib.aclosing(stream):
                async for output in stream:
                    if "cached" not in row:
                        # The first output carries the prefix-cache hit.
                        row["first_token"] = clock() - started
                        row["cached"] = int(output.num_cached_tokens or 0)
                        if clock() <= window_end:
                            tally.prompt += len(prompt)
                            tally.uncached += len(prompt) - row["cached"]
                    count = len(output.outputs[0].token_ids)
                    if clock() <= window_end:
                        tally.generated += count - seen
                    seen = count
        finally:
            tally.in_flight -= 1
            row["generated"] = seen
        row["returned"] = clock() - started
        tally.finished += 1
        assert output is not None  # a returned stream produced at least one output
        reply = np.asarray(output.outputs[0].token_ids, dtype=np.int32)
        if record_outputs:
            row["output_ids"] = reply.tolist()
        stray = np.flatnonzero(reply == placeholder_id) if placeholder_id is not None else ()
        if len(stray):
            reply = reply.copy()
            for i in stray:
                reply[i] = reply[i - 1] if i else prompt[-1]
            row["placeholders_replaced"] = len(stray)
        last_history[lane_index] = np.concatenate([prompt, reply])

    samples: list[dict[str, Any]] = []
    settled: dict[str, float] = {}  # the first sample past settle_s, where the settled rates count from

    async def sample() -> None:
        reported = {"t": 0.0, "generated": 0, "uncached": 0}
        while True:
            now = clock() - started
            samples.append(
                {
                    "t": now,
                    "generated": tally.generated,
                    "prompt": tally.prompt,
                    "uncached": tally.uncached,
                    "in_flight": tally.in_flight,
                    "finished": tally.finished,
                }
            )
            if not settled and now >= settle_s:
                settled.update(t=now, generated=tally.generated, uncached=tally.uncached)
                print(f"[traffic] {now:.0f}s: settled; the settled rates count from here", flush=True)
            span = now - reported["t"]
            if span >= report_s:
                # The stretch since the last line, not the window so far: a
                # live run is read by how its rate moves.
                print(
                    f"[traffic] {now:.0f}s: {tally.in_flight} in flight, {tally.finished} returned, "
                    f"{sum(restarts)} restarts; "
                    f"gen {(tally.generated - reported['generated']) / span:.0f} tok/s, uncached prompt "
                    f"{(tally.uncached - reported['uncached']) / span:.0f} tok/s over the last {span:.0f}s",
                    flush=True,
                )
                reported = {"t": now, "generated": tally.generated, "uncached": tally.uncached}
            await asyncio.sleep(sample_s)

    tasks = [asyncio.create_task(play(i, lane)) for i, lane in enumerate(chosen)]
    sampler = asyncio.create_task(sample())
    # A lane finishes only with restarts off. Its end would leave the rest of
    # the window at less than the asked concurrency: the reading is void, so
    # the replay stops there and says so rather than report it.
    ran_dry: Optional[str] = None

    def write(summary: Optional[dict[str, Any]]) -> None:
        if out_dir is None:
            return
        root = pathlib.Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        if summary is not None:
            (root / "traffic_result.json").write_text(json.dumps(summary, indent=1) + "\n")
        with open(root / "traffic_calls.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def close_window() -> dict[str, Any]:
        """The reading, as the window closes: every count stopped at
        ``window_end``, so it reads the same whenever it is taken."""
        elapsed = min(clock(), window_end) - started
        samples.append(
            {
                "t": elapsed,
                "generated": tally.generated,
                "prompt": tally.prompt,
                "uncached": tally.uncached,
                "in_flight": tally.in_flight,
                "finished": tally.finished,
            }
        )
        settled_s = elapsed - settled["t"] if settled else 0.0
        summary: dict[str, Any] = {
            "lanes": lanes,
            "recorded_lanes": recorded,
            "window_s": window_s,
            "elapsed_s": elapsed,
            "settled_at_s": settled.get("t"),
            "settled_s": settled_s,
            "settled_generated_tok_s": (tally.generated - settled["generated"]) / settled_s if settled_s > 0 else None,
            "settled_uncached_prompt_tok_s": (
                (tally.uncached - settled["uncached"]) / settled_s if settled_s > 0 else None
            ),
            "gap_s": gap_s,
            "generated_tokens": tally.generated,
            "generated_tok_s": tally.generated / elapsed if elapsed > 0 else 0.0,
            "prompt_tokens": tally.prompt,
            "uncached_prompt_tokens": tally.uncached,
            "uncached_prompt_tok_s": tally.uncached / elapsed if elapsed > 0 else 0.0,
            "calls_submitted": len(rows),
            "calls_returned": tally.finished,
            "ran_dry": ran_dry,
            "restarts": sum(restarts),
            "restarts_by_lane": restarts,
            "decode_gate": decode_gate,
            "gate_s": gate_s,
            "gate_tokens": gate_tokens,
            "workload": dict(workload.meta),
            "samples": list(samples),
        }
        write(summary)
        if on_window is not None:
            on_window(summary)
        return summary

    reading: Optional[dict[str, Any]] = None
    try:
        pending = set(tasks)
        while pending and ran_dry is None:
            now = clock()
            if reading is None and tail_s > 0 and now >= window_end:
                # The tail's instrumentation may stall or kill the engine:
                # the reading is taken and written before it starts.
                reading = close_window()
            remaining = (window_end + tail_s if reading is not None or tail_s <= 0 else window_end) - now
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                error = task.exception()
                if error is not None:
                    raise error
                if reading is not None:
                    # Out of calls in the tail: the reading is taken already.
                    continue
                lane_index = tasks.index(task)
                ran_dry = (
                    f"lane {lane_index} ran out of workload {clock() - started:.0f}s into a {window_s:.0f}s window, "
                    f"after all {len(chosen[lane_index].calls)} of its calls: the workload holds too little "
                    f"for this window at this speed"
                )
                break
    finally:
        for task in (*tasks, sampler):
            task.cancel()
        await asyncio.gather(*tasks, sampler, return_exceptions=True)
    if reading is None:
        # No tail, or it never began: the window's end is the run's end, and
        # the rows of calls the window cut carry what they streamed.
        reading = close_window()
    else:
        # The tail's calls, and the rows the window's close caught in flight,
        # completed; the reading stays as the window closed.
        write(None)
    if ran_dry is not None:
        raise RuntimeError(ran_dry)
    return reading
