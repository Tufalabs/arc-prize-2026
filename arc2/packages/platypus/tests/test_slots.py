"""The run's queue tally, as the contract states it.

``ctx.slots`` is read on access rather than
pushed, never reporting a drained queue when it simply does not know, and
written by infra only. The whale reads it to decide whether a token cap still
buys anything, so a reader that guessed would change how a run spends.

These drive the file directly. What the solver publishes at each transition is
its own to test; what a reader does with what it finds is this.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from infra_fixtures import ProbeAgent

import taaf.game_examples
from platypus.contract import RunConfig
from platypus.priority import PrioritySlots
from platypus.slots import FILENAME, SlotView, slots_path, write_slots
from platypus.solver import AgentSolver, pace_while_busy


async def _settle() -> None:
    """Let every task that could run, run."""
    for _ in range(4):
        await asyncio.sleep(0)


def test_nothing_published_is_not_a_drained_queue(tmp_path: Path) -> None:
    """The direction that matters: absence must never read as "go ahead".

    An agent that spends more because nothing is queued has to have positively
    read that nothing is queued. A run with no writer therefore behaves exactly
    as one without the feature.
    """
    view = SlotView(job_dir=tmp_path)

    assert view.waiting is None
    assert view.running is None
    assert view.limit is None
    assert not view.drained


def test_no_job_dir_reads_as_unknown(tmp_path: Path) -> None:
    """A view with nowhere to look answers the same way as an empty one."""
    del tmp_path
    view = SlotView()

    assert view.waiting is None
    assert not view.drained


def test_a_published_tally_is_read_back(tmp_path: Path) -> None:
    """The happy path, and the one place `drained` may be true."""
    write_slots(tmp_path, waiting=3, running=2, limit=8)
    view = SlotView(job_dir=tmp_path, ttl_s=0.0)

    assert (view.waiting, view.running, view.limit) == (3, 2, 8)
    assert not view.drained

    write_slots(tmp_path, waiting=0, running=8, limit=8)
    assert view.drained, "a zero that was published is the one thing that means drained"


def test_a_torn_or_unreadable_file_keeps_the_last_good_answer(tmp_path: Path) -> None:
    """A half-written or corrupt tally must not read as unknown *or* as zero.

    The writer renames into place so a reader should never see one, but the
    fallback decides what happens if it does: the previous answer is stale by
    at most the write, where "unknown" would discard a fact we hold and a
    parse-as-empty would invent a drained queue.
    """
    write_slots(tmp_path, waiting=5, running=1, limit=8)
    view = SlotView(job_dir=tmp_path, ttl_s=0.0)
    assert view.waiting == 5

    slots_path(tmp_path).write_text('{"waiting": 0, "run')

    assert view.waiting == 5, "a torn read replaced a known tally"
    assert not view.drained


def test_a_tally_missing_its_numbers_reads_as_unknown(tmp_path: Path) -> None:
    """A well-formed file that says nothing useful is not a drained queue."""
    slots_path(tmp_path).write_text(json.dumps({"t": 1.0}))
    view = SlotView(job_dir=tmp_path, ttl_s=0.0)

    assert view.waiting is None
    assert not view.drained


def test_a_reader_is_not_pinned_to_its_first_answer(tmp_path: Path) -> None:
    """Read on access, so a move that occupies a game for minutes still sees
    the queue as it is when it asks — the property a pushed number cannot
    have, and the reason this is a file rather than a value on the context."""
    write_slots(tmp_path, waiting=2, running=6, limit=8)
    view = SlotView(job_dir=tmp_path, ttl_s=0.0)
    assert not view.drained

    write_slots(tmp_path, waiting=0, running=8, limit=8)

    assert view.drained


def test_a_read_is_reused_within_its_ttl(tmp_path: Path) -> None:
    """The tally moves only when a game starts or ends, so a second of
    staleness costs nothing and bounds the read rate whatever an agent does."""
    write_slots(tmp_path, waiting=4, running=4, limit=8)
    view = SlotView(job_dir=tmp_path, ttl_s=3600.0)
    assert view.waiting == 4

    write_slots(tmp_path, waiting=0, running=8, limit=8)

    assert view.waiting == 4, "the cached answer was not reused"
    assert not view.drained


def test_publishing_never_raises_on_a_bad_directory(tmp_path: Path) -> None:
    """Bookkeeping, not the run: a tally that cannot be written is not a
    reason to end a game."""
    write_slots(tmp_path / "does" / "not" / "exist", waiting=1, running=1, limit=2)


def test_an_unpublished_tally_is_not_re_read_every_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bound holds when there is nothing to read, which is the case it is for.

    A run whose solver publishes nothing has no cached value to short-circuit
    on, so a TTL that stamped only successful parses left every property
    access opening the file — and ``drained`` alone reads ``waiting``. The
    attempt is what the clock has to measure.
    """
    reads = 0
    real = Path.read_text

    def counting(self: Path, *args: object, **kwargs: object) -> str:
        nonlocal reads
        if self.name == FILENAME:
            reads += 1
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", counting)
    view = SlotView(job_dir=tmp_path, ttl_s=3600.0)

    assert not view.drained
    assert view.waiting is None
    assert view.running is None
    assert view.limit is None

    assert reads == 1, f"nothing published, but the file was opened {reads} times"


def _solver_with_holder_and_rival(tmp_path: Path, config: RunConfig) -> AgentSolver:
    """A solver publishing to ``tmp_path`` with ``0`` holding the one slot
    and ``1`` queued behind it."""
    solver = AgentSolver(agent=ProbeAgent())
    solver.job_dir = tmp_path
    solver._priority = PrioritySlots(config.max_concurrent)
    solver._slot_labels = {"0": "holder", "1": "rival"}
    for key, game_id in solver._slot_labels.items():
        solver._status.register(key, game_id)
    return solver


async def test_a_parked_game_keeps_the_queue_undrained(tmp_path: Path) -> None:
    """One paused game is enough to make `drained` false, through the
    published tally an agent actually reads.

    `drained` asks whether giving this game up would free a slot anything
    can use. A parked game can use it — it is a started game waiting for
    one, not a finished one — so an agent that lifted a spending cap here
    would be spending against a queue that is not empty. "Every game has
    started" is a different question, and answering it here would make
    this one wrong.
    """
    config = RunConfig(max_concurrent=1)
    solver = _solver_with_holder_and_rival(tmp_path, config)
    assert solver._priority is not None
    await solver._priority.acquire("0")
    rival = asyncio.create_task(solver._priority.acquire("1"))
    await _settle()

    # Nothing queued to start: the queue reads drained, and would still
    # read drained after a game parked if parking were not counted.
    solver._slots_waiting, solver._slots_running = 0, 1
    solver._publish_slots(config)
    view = SlotView(job_dir=tmp_path, ttl_s=0.0)
    assert view.drained

    yield_slot = solver._yield_handle(taaf.game_examples.flip_game("g"), "0", config)

    async def offer() -> None:
        """The holder offering its slot back, from inside its own move."""
        await yield_slot({"tokens": 10, "compacted": True})

    parking = asyncio.create_task(offer())
    await _settle()

    assert not view.drained, "a game is parked, and it can take the slot a give-up would free"
    assert view.waiting == 1

    solver._priority.release("1")
    await parking
    assert view.drained, "resumed: nothing wants a slot again"
    rival.cancel()


async def test_a_pause_line_that_cannot_be_printed_leaves_the_tally_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pause bookkeeping and the flag that undoes it must agree.

    The pause prints, so a closed stdout raises out of the callback and
    the game unwinds. Whatever the flag says, the undo in `finally` has to
    reverse exactly what was applied: crediting a transition that never
    happened moves the published tally the wrong way by two and it never
    recovers — `waiting` goes negative as soon as the game that took the
    slot starts, and every reader of `drained` is wrong for the rest of
    the run.
    """
    config = RunConfig(max_concurrent=1)
    solver = _solver_with_holder_and_rival(tmp_path, config)
    assert solver._priority is not None
    await solver._priority.acquire("0")
    rival = asyncio.create_task(solver._priority.acquire("1"))
    await _settle()
    solver._slots_waiting, solver._slots_running = 1, 1

    def closed_stdout(*args: object, **kwargs: object) -> None:
        raise ValueError("I/O operation on closed file")

    monkeypatch.setattr("builtins.print", closed_stdout)
    yield_slot = solver._yield_handle(taaf.game_examples.flip_game("g"), "0", config)

    with pytest.raises(ValueError, match="closed file"):
        await yield_slot({"tokens": 10, "compacted": True})

    assert (solver._slots_running, solver._slots_waiting) == (1, 1), "the failed pause left no trace"
    assert solver._status.detail("0") is None
    rival.cancel()


def test_requests_are_paced_only_while_the_run_is_busy(tmp_path: Path) -> None:
    """The backend's gap while at least the configured share of slots is held,
    none below it, and the gap again when the tally cannot be read."""
    from types import SimpleNamespace
    from typing import Any, cast

    config = RunConfig(max_concurrent=10)
    backend = SimpleNamespace(min_request_gap_s=5.0)
    ctx = cast(Any, SimpleNamespace(backend=backend, slots=SlotView(job_dir=tmp_path, ttl_s=0.0)))

    write_slots(tmp_path, waiting=0, running=5, limit=10)
    pace_while_busy(ctx, config, 5.0)
    assert backend.min_request_gap_s == 5.0, "half the slots held is still busy"

    write_slots(tmp_path, waiting=0, running=4, limit=10)
    pace_while_busy(ctx, config, 5.0)
    assert backend.min_request_gap_s == 0.0

    write_slots(tmp_path, waiting=0, running=4, limit=10)
    pace_while_busy(ctx, RunConfig(min_request_gap_running_fraction=0.0), 5.0)
    assert backend.min_request_gap_s == 5.0, "a zero share paces always"

    unknown = cast(Any, SimpleNamespace(backend=backend, slots=SlotView(job_dir=None)))
    backend.min_request_gap_s = 0.0
    pace_while_busy(unknown, config, 5.0)
    assert backend.min_request_gap_s == 5.0, "an unknown tally keeps the gap"
