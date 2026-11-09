"""``ullm.traffic``: cutting a workload out of a recording, and replaying it.

The properties here are the ones a wrong replay would get wrong silently:
lanes follow the recording's slot links, a lane starts from the history it
held at the snapshot, each call's prompt carries the lane's OWN replies where
the recording shared a prefix (so the prefix cache sees the recording's
structure), every reply is forced to its recorded length, and a window cut
aborts what is in flight while counting what it had streamed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Sequence

import numpy as np
import pytest
import ullm.backend
import ullm.traffic
from PIL import Image


def _ids(*values: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def _call(
    input_ids: np.ndarray,
    output_ids: np.ndarray,
    finished_at: float,
    *,
    after: tuple[str, int] | None | str = "none",
    images: tuple[str, ...] = (),
    continued: bool = True,
) -> ullm.traffic.RecordedCall:
    """A recorded call; ``after`` left at ``"none"`` means no grant preceded it."""
    opens = after != "none"
    return ullm.traffic.RecordedCall(
        input_ids=input_ids,
        output_ids=output_ids,
        images=images,
        finished_at=finished_at,
        continued=continued,
        opens=opens,
        after=after if opens and not isinstance(after, str) else None,
    )


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 10, 10)).save(buffer, format="PNG")
    return buffer.getvalue()


def _two_slot_recording() -> tuple[list[ullm.traffic.RecordedRun], dict[str, bytes]]:
    """Two slots. Slot one plays run a, then run c takes it when a ends.
    Slot two plays run b, which compacts at its third call and keeps its
    slot: the fresh render shares only b's head. Run a's second prompt
    carries an image."""
    a0_in, a0_out = _ids(1, 2, 3, 4), _ids(5, 6)
    a1_in = np.concatenate([a0_in, a0_out, _ids(7, 8)])
    a1_out = _ids(9, 10, 11)
    a2_in = np.concatenate([a1_in, a1_out, _ids(12)])
    run_a = ullm.traffic.RecordedRun(
        "a",
        (
            _call(a0_in, a0_out, 10.0, after=None, continued=False),
            _call(a1_in, a1_out, 20.0, images=("img",)),
            _call(a2_in, _ids(13, 14), 30.0, images=("img",)),
        ),
    )
    c0_in = _ids(1, 2, 50, 51)
    run_c = ullm.traffic.RecordedRun(
        "c",
        (
            _call(c0_in, _ids(52), 40.0, after=("a", 2), continued=False),
            _call(np.concatenate([c0_in, _ids(52, 53)]), _ids(54, 55), 50.0),
        ),
    )
    b0_in, b0_out = _ids(1, 2, 3, 60), _ids(61, 62)
    b1_in = np.concatenate([b0_in, b0_out, _ids(63)])
    b2_in = _ids(1, 2, 3, 60, 70, 71)  # the compacted render: b's head, then new text
    run_b = ullm.traffic.RecordedRun(
        "b",
        (
            _call(b0_in, b0_out, 12.0, after=None, continued=False),
            _call(b1_in, _ids(64), 22.0),
            _call(b2_in, _ids(72, 73), 32.0, after=("b", 1), continued=False),
            _call(np.concatenate([b2_in, _ids(72, 73, 74)]), _ids(75), 42.0),
        ),
    )
    return [run_a, run_b, run_c], {"img": _png()}


def test_a_workload_starts_each_lane_from_the_history_it_held_at_the_snapshot() -> None:
    runs, media = _two_slot_recording()
    workload = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0)
    assert workload.meta["recorded_lanes"] == 2
    lane_a, lane_b = workload.lanes

    # Slot one was inside run a's sequence: its warm-up is a's history before
    # the call in flight, and that call continues from it.
    a1_in, a1_out = runs[0].calls[1].input_ids, runs[0].calls[1].output_ids
    assert lane_a.history is not None and lane_a.history.tolist() == [*a1_in.tolist(), *a1_out.tolist()]
    assert lane_a.history_images == ("img",)
    first = lane_a.calls[0]
    assert first.shared == len(a1_in) + len(a1_out) and first.suffix.tolist() == [12]
    # Run c follows a on the slot a freed, from its whole recorded prompt.
    assert [(c.sequence, c.shared) for c in lane_a.calls[1:]] == [(1, 0), (1, 5)]
    assert lane_a.calls[1].suffix.tolist() == runs[2].calls[0].input_ids.tolist()

    # Slot two had just handed b to itself at a compaction: the compacted
    # render opens a sequence, so there is no history to warm.
    assert lane_b.history is None
    assert lane_b.calls[0].shared == 0 and lane_b.calls[0].suffix.tolist() == [1, 2, 3, 60, 70, 71]
    assert [c.output_len for c in lane_b.calls] == [2, 1]


def test_the_default_snapshot_is_settled_and_leaves_every_lane_its_horizon() -> None:
    """Three slots, each playing six runs of four calls in turn, the slots
    out of phase. The chosen moment lies after most slots have handed over
    once, and every lane still holds the horizon after it."""
    runs: list[ullm.traffic.RecordedRun] = []
    for slot in range(3):
        previous: tuple[str, int] | None = None
        clock = float(slot)
        for n in range(6):
            run_id = f"s{slot}r{n}"
            calls = []
            prompt = _ids(1, 2, 3)
            for k in range(4):
                clock += 10.0
                out = _ids(*range(100, 110))
                calls.append(_call(prompt, out, clock, after=previous if k == 0 else "none", continued=k > 0))
                prompt = np.concatenate([prompt, out, _ids(4)])
            runs.append(ullm.traffic.RecordedRun(run_id, tuple(calls)))
            previous = (run_id, 3)
    workload = ullm.traffic.build_workload(runs, {}, horizon_tokens=95)
    meta = workload.meta
    assert meta["settled_at"] < meta["snapshot"] < meta["horizon_held_until"]
    for lane in workload.lanes:
        assert lane.output_tokens >= 95


def test_a_workload_survives_save_and_load(tmp_path: Path) -> None:
    runs, media = _two_slot_recording()
    workload = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0)
    loaded = ullm.traffic.Workload.load(workload.save(tmp_path / "workload.b64"))
    assert loaded.meta == json.loads(json.dumps(workload.meta))
    assert loaded.media == workload.media
    for before, after in zip(workload.lanes, loaded.lanes, strict=True):
        assert (before.history is None) == (after.history is None)
        if before.history is not None and after.history is not None:
            assert before.history.tolist() == after.history.tolist()
        assert [(c.sequence, c.shared, c.suffix.tolist(), c.images, c.output_len) for c in before.calls] == [
            (c.sequence, c.shared, c.suffix.tolist(), c.images, c.output_len) for c in after.calls
        ]


class _FakeEngine:
    """Serves id prompts with distinct, recognisable reply ids, one token per
    ``delay_s``, recording what it was asked and whether a stream was cut."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.requests: list[dict[str, Any]] = []
        self.cut: list[str] = []
        self.closed: list[str] = []

    async def close_context(self, prefix: str) -> None:
        self.closed.append(prefix)

    async def serve_ids(
        self,
        prompt_ids: Sequence[int],
        images: Sequence[Any],
        sampling_params: ullm.backend.SamplingParameters,
        request_id: str,
        prefix_cache_thread: tuple[str, int] | None = None,
    ) -> AsyncGenerator[Any, None]:
        assert isinstance(sampling_params, ullm.backend.GenericSamplingParameters)
        n = sampling_params.max_tokens or 0
        reply = [1000 * (len(self.requests) + 1) + i for i in range(n)]
        self.requests.append(
            {
                "id": request_id,
                "prompt": list(prompt_ids),
                "images": len(images),
                "params": sampling_params,
                "thread": prefix_cache_thread,
            }
        )
        finished = False
        try:
            for i in range(1, n + 1):
                await asyncio.sleep(self.delay_s)
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=reply[:i])], num_cached_tokens=2)
            finished = True
        finally:
            if not finished:
                self.cut.append(request_id)


_PROFILE = ullm.backend.GenericSamplingParameters(temperature=0.6, top_p=0.9, extra_params={"top_k": 20})


async def test_a_replay_extends_each_lane_with_its_own_replies(tmp_path: Path) -> None:
    runs, media = _two_slot_recording()
    workload = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0)
    engine = _FakeEngine()
    # Restarts off, and the window outlasts the tiny workload: the first lane
    # to finish its calls stops the replay, loudly, with the files written.
    with pytest.raises(RuntimeError, match="ran out of workload"):
        await ullm.traffic.replay(
            engine,
            workload,
            lanes=2,
            window_s=30.0,
            gap_s=0.05,
            sampling_params=_PROFILE,
            out_dir=tmp_path,
            restart=False,
        )

    # Nothing is prefilled ahead of the window: every request the engine saw
    # is a lane's call, and the lane holding a history starts from it cold.
    timed = engine.requests
    assert all(not r["id"].startswith("traffic-warmup") for r in timed)
    first_of_lane_a = next(r for r in timed if r["id"].startswith("traffic-0-"))
    history = workload.lanes[0].history
    assert history is not None and first_of_lane_a["prompt"][: len(history)] == history.tolist()

    # Every timed reply is forced to its recorded length, the profile kept.
    for request, call in zip([r for r in timed if r["id"].startswith("traffic-0-")], workload.lanes[0].calls):
        params = request["params"]
        assert params.max_tokens == params.extra_params["min_tokens"] == call.output_len
        assert params.extra_params["ignore_eos"] is True and params.extra_params["top_k"] == 20
        assert params.temperature == 0.6

    # Run b's call after its compaction continues from the engine's own reply
    # to the compacted render, not from the recorded one.
    lane_b = [r for r in timed if r["id"].startswith("traffic-1-")]
    compacted, following = lane_b
    reply = [1000 * (engine.requests.index(compacted) + 1) + i for i in range(2)]
    assert following["prompt"] == [*compacted["prompt"], *reply, *workload.lanes[1].calls[1].suffix.tolist()]

    assert json.loads((tmp_path / "traffic_result.json").read_text())["ran_dry"].startswith("lane 1")
    # A lane's next call waits the gap after its previous one returned.
    rows = [json.loads(line) for line in (tmp_path / "traffic_calls.jsonl").read_text().splitlines()]
    for lane in (0, 1):
        mine = [row for row in rows if row["lane"] == lane]
        for before, after in zip(mine, mine[1:]):
            assert after["submitted"] - before["returned"] >= 0.05 - 1e-3

    # Each call names its conversation as a game's would: a sequence is one,
    # its epoch steps where a call does not extend the lane's history, and a
    # sequence the lane leaves is closed.
    assert [r["thread"] for r in timed] == [(row["thread"], row["epoch"]) for row in rows]
    for lane in (0, 1):
        mine = [row for row in rows if row["lane"] == lane]
        for before, after in zip(mine, mine[1:]):
            if after["sequence"] != before["sequence"]:
                assert after["thread"] != before["thread"]
                assert before["thread"].rsplit("/", 1)[0] in engine.closed
            else:
                extends = after["shared"] == before["prompt"] + before["generated"]
                assert after["thread"] == before["thread"]
                assert after["epoch"] == before["epoch"] + (0 if extends else 1)


async def test_the_decode_gate_prefills_each_first_prompt_and_the_first_call_continues_it(tmp_path: Path) -> None:
    runs, media = _two_slot_recording()
    workload = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0)
    engine = _FakeEngine()
    delays: list[float] = []
    engine.delay_profile_windows = delays.append  # type: ignore[attr-defined]
    held_at: list[int] = []
    engine.hold_profile_windows = lambda: held_at.append(len(engine.requests))  # type: ignore[attr-defined]
    lifted: list[float] = []
    with pytest.raises(RuntimeError, match="ran out of workload"):
        await ullm.traffic.replay(
            engine,
            workload,
            lanes=2,
            window_s=30.0,
            gap_s=0.05,
            sampling_params=_PROFILE,
            out_dir=tmp_path,
            restart=False,
            decode_gate=True,
            on_gate_lifted=lifted.append,
        )

    # Before the window: one one-token request per lane, on the thread its
    # first call names, carrying that call's whole prompt.
    gates = [r for r in engine.requests if r["id"].startswith("traffic-gate-")]
    assert len(gates) == 2 and engine.requests[:2] == gates
    rows = [json.loads(line) for line in (tmp_path / "traffic_calls.jsonl").read_text().splitlines()]
    for lane, gate in zip((0, 1), sorted(gates, key=lambda r: r["id"])):
        assert gate["params"].max_tokens == 1
        first = next(r for r in engine.requests if r["id"].startswith(f"traffic-{lane}-"))
        row = next(r for r in rows if r["lane"] == lane)
        token = 1000 * (engine.requests.index(gate) + 1)
        # The first call continues the gate's request: its prompt and the
        # token it sampled, the reply one token short, on the same thread.
        assert first["prompt"] == [*gate["prompt"], token]
        assert first["params"].max_tokens == workload.lanes[lane].calls[0].output_len - 1
        assert first["thread"] == gate["thread"] == (row["thread"], row["epoch"])
        assert row["gated"] is True
    result = json.loads((tmp_path / "traffic_result.json").read_text())
    assert result["decode_gate"] is True and lifted == [result["gate_s"]]
    # The engine's profiler windows are held from before the gate's first
    # request, and move past the gate once it lifts.
    assert held_at == [0]
    assert delays == [result["gate_s"]]


async def test_the_window_aborts_what_is_in_flight_and_counts_what_it_streamed() -> None:
    runs, media = _two_slot_recording()
    workload = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0)
    engine = _FakeEngine(delay_s=0.05)
    summary = await ullm.traffic.replay(engine, workload, lanes=2, window_s=0.12, gap_s=0.0, sampling_params=_PROFILE)
    assert engine.cut, "calls still decoding when the window ended were aborted"
    total = sum(call.output_len for lane in workload.lanes for call in lane.calls)
    assert 0 < summary["generated_tokens"] < total
    assert summary["calls_returned"] < summary["calls_submitted"]
    assert summary["ran_dry"] is None


async def test_with_a_tail_the_reading_is_written_as_the_window_closes(tmp_path: Path) -> None:
    """The tail plays on for instrumentation, counted in nothing; the reading
    is on disk and reported before it starts, and is what the replay returns."""
    runs, media = _two_slot_recording()
    workload = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0)
    # 0.05 s a token: lane 0's third call starts in the tail, and lane 1 runs
    # out there, which with restarts off voids nothing the window measured.
    engine = _FakeEngine(delay_s=0.05)
    reported: list[dict[str, Any]] = []

    def on_window(summary: dict[str, Any]) -> None:
        assert json.loads((tmp_path / "traffic_result.json").read_text()) == summary
        reported.append(summary)

    summary = await ullm.traffic.replay(
        engine,
        workload,
        lanes=2,
        window_s=0.125,
        gap_s=0.0,
        sampling_params=_PROFILE,
        out_dir=tmp_path,
        restart=False,
        tail_s=0.2,
        on_window=on_window,
    )
    assert reported == [summary]
    assert json.loads((tmp_path / "traffic_result.json").read_text()) == summary
    # The tail's calls are in the rows, past the ones the reading counted.
    rows = [json.loads(line) for line in (tmp_path / "traffic_calls.jsonl").read_text().splitlines()]
    assert len(rows) > summary["calls_submitted"]
    assert all("generated" in row for row in rows)


async def test_an_engine_dying_in_the_tail_leaves_the_reading(tmp_path: Path) -> None:
    """A profiler window in the tail that kills the engine costs nothing the
    window measured."""

    class _Dying(_FakeEngine):
        dead = False

        async def serve_ids(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
            if self.dead:
                raise RuntimeError("engine died")
            async for output in super().serve_ids(*args, **kwargs):
                yield output

    runs, media = _two_slot_recording()
    workload = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0)
    engine = _Dying(delay_s=0.05)
    reported: list[dict[str, Any]] = []

    def on_window(summary: dict[str, Any]) -> None:
        reported.append(summary)
        engine.dead = True

    with pytest.raises(RuntimeError, match="engine died"):
        await ullm.traffic.replay(
            engine,
            workload,
            lanes=2,
            window_s=0.125,
            gap_s=0.0,
            sampling_params=_PROFILE,
            out_dir=tmp_path,
            restart=False,
            tail_s=5.0,
            on_window=on_window,
        )
    assert json.loads((tmp_path / "traffic_result.json").read_text()) == reported[0]
    assert reported[0]["generated_tokens"] > 0


async def test_without_a_tail_the_reading_is_reported_once_at_the_end() -> None:
    runs, media = _two_slot_recording()
    workload = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0)
    reported: list[dict[str, Any]] = []
    summary = await ullm.traffic.replay(
        _FakeEngine(delay_s=0.05),
        workload,
        lanes=2,
        window_s=0.12,
        gap_s=0.0,
        sampling_params=_PROFILE,
        on_window=reported.append,
    )
    assert reported == [summary]


async def test_lanes_past_the_recording_replay_its_lanes_as_nonced_copies(tmp_path: Path) -> None:
    """Asked for more lanes than recorded, the replay plays each recorded lane
    again as a later copy: the same calls, every fresh prompt keeping the head
    all games share and carrying a nonce of its own copy, so no copy shares a
    cached prefix past the head with another."""
    head = list(range(1, 11))
    opening = np.asarray([*head, *range(100, 160)], dtype=np.int32)  # room for a nonce per copy and pass
    lane = ullm.traffic.Lane(
        history=None,
        history_images=(),
        calls=(
            ullm.traffic.Call(sequence=0, shared=0, suffix=opening, images=(), output_len=2),
            ullm.traffic.Call(
                sequence=0, shared=72, suffix=np.asarray([40, 41], dtype=np.int32), images=(), output_len=2
            ),
        ),
    )
    other = dataclasses.replace(
        lane, calls=(dataclasses.replace(lane.calls[0], suffix=np.asarray([*head, *range(300, 360)], dtype=np.int32)),)
    )
    workload = ullm.traffic.Workload(lanes=(lane, other), media={})
    engine = _FakeEngine(delay_s=0.01)
    summary = await ullm.traffic.replay(
        engine, workload, lanes=5, window_s=0.2, gap_s=0.0, sampling_params=_PROFILE, out_dir=tmp_path
    )
    assert summary["lanes"] == 5 and summary["recorded_lanes"] == 2
    rows = [json.loads(line) for line in (tmp_path / "traffic_calls.jsonl").read_text().splitlines()]
    assert sorted({(row["lane"], row["copy"]) for row in rows}) == [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)]
    firsts = {
        row["lane"]: engine.requests[i]["prompt"] for i, row in enumerate(rows) if row["call"] == 0 and row["pass"] == 0
    }
    # Lanes 0, 2 and 4 play the first recorded lane: its opening as recorded,
    # then marked once per copy past the shared head, each unlike the others.
    assert firsts[0] == opening.tolist()
    for k in (2, 4):
        assert firsts[k][: len(head)] == head and len(firsts[k]) == len(opening)
    assert len({tuple(firsts[k]) for k in (0, 2, 4)}) == 3
    # A copy's continuation extends its own marked opening.
    second = next(i for i, row in enumerate(rows) if row["lane"] == 2 and row["call"] == 1 and row["pass"] == 0)
    assert engine.requests[second]["prompt"][: len(firsts[2])] == firsts[2]


async def test_a_lane_that_runs_out_starts_again_from_its_first_call_with_nonces(tmp_path: Path) -> None:
    head = list(range(1, 11))  # the head every fresh prompt shares
    opening = np.asarray([*head, 20, 21, 22, 23, 24, 25], dtype=np.int32)
    lane = ullm.traffic.Lane(
        history=np.asarray([*head, 30, 31, 32, 33, 34], dtype=np.int32),
        history_images=(),
        calls=(
            ullm.traffic.Call(
                sequence=0, shared=15, suffix=np.asarray([35, 36], dtype=np.int32), images=(), output_len=2
            ),
            ullm.traffic.Call(sequence=1, shared=0, suffix=opening, images=(), output_len=2),
            ullm.traffic.Call(
                sequence=1, shared=18, suffix=np.asarray([40, 41], dtype=np.int32), images=(), output_len=2
            ),
        ),
    )
    workload = ullm.traffic.Workload(lanes=(lane,), media={})
    engine = _FakeEngine(delay_s=0.01)
    summary = await ullm.traffic.replay(
        engine, workload, lanes=1, window_s=0.2, gap_s=0.0, sampling_params=_PROFILE, out_dir=tmp_path
    )
    assert summary["restarts"] >= 2 and summary["ran_dry"] is None

    rows = [json.loads(line) for line in (tmp_path / "traffic_calls.jsonl").read_text().splitlines()]
    # Every pass plays every call, from the snapshot history again.
    assert [row["call"] for row in rows if row["pass"] == 0] == [0, 1, 2]
    assert [row["call"] for row in rows if row["pass"] == 1] == [0, 1, 2]
    prompts = [request["prompt"] for request in engine.requests]
    firsts = [prompts[i] for i, row in enumerate(rows) if row["call"] == 0]
    assert lane.history is not None
    history = lane.history.tolist()
    assert firsts[0][: len(history)] == history
    for i, marked in enumerate(firsts[1:3], start=1):
        assert marked[: len(head)] == head and len(marked) == len(firsts[0])
        assert all(marked != earlier for earlier in firsts[:i])

    openings = [prompts[i] for i, row in enumerate(rows) if row["call"] == 1]
    # Each pass's opening keeps the shared head, has the opening's length, and
    # differs from every earlier pass's.
    assert openings[0] == opening.tolist()
    for i, marked in enumerate(openings[1:3], start=1):
        assert marked[: len(head)] == head and len(marked) == len(opening)
        assert all(marked != earlier for earlier in openings[:i])
    # The call after a marked opening continues from the marked prompt.
    marked_opening = next(i for i, row in enumerate(rows) if row["pass"] == 1 and row["call"] == 1)
    assert prompts[marked_opening + 1][: len(prompts[marked_opening])] == prompts[marked_opening]


def test_skip_in_flight_starts_each_lane_after_the_call_in_flight() -> None:
    """The call in flight at the snapshot is length-biased; skipping it starts
    each lane at the next call, the in-flight call's prompt and reply its
    history, or at its next sequence's opening call."""
    runs, media = _two_slot_recording()
    plain = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0)
    skipped = ullm.traffic.build_workload(runs, media, horizon_tokens=100, snapshot=25.0, skip_in_flight=True)
    assert skipped.meta["skip_in_flight"] is True
    for plain_lane, skipped_lane in zip(plain.lanes, skipped.lanes):
        assert len(skipped_lane.calls) == len(plain_lane.calls) - 1
    lane_a, lane_b = skipped.lanes
    # Slot one's in-flight call ended run a's sequence: run c opens the lane.
    assert lane_a.history is None
    assert lane_a.calls[0].shared == 0
    assert lane_a.calls[0].suffix.tolist() == runs[2].calls[0].input_ids.tolist()
    # Slot two's in-flight call, the compacted render, is now its history.
    in_flight = next(c for c in runs[1].calls if c.input_ids.tolist() == [1, 2, 3, 60, 70, 71])
    assert lane_b.history is not None
    assert lane_b.history.tolist() == [*in_flight.input_ids.tolist(), *in_flight.output_ids.tolist()]
    assert [c.output_len for c in lane_b.calls] == [1]
