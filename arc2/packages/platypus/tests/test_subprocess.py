"""Subprocess hosting: loud child-death handling, error policies across the
process boundary, buffered pipe ordering, the memory limit, and cancellation with reaping.

Game-side assertions only — an agent's own introspection cannot cross a
process boundary, which is exactly why these tests exist separately from the
in-process ones.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from infra_fixtures import (
    CrashingAgent,
    HungryAgent,
    OutcomeDumpingSlowAgent,
    PrintingAgent,
    ProbeAgent,
    SelfKillingAgent,
)

import taaf.benchmark
import taaf.game
import taaf.game_examples
import taaf.solver_examples
from platypus import AgentSolver, RunConfig
from platypus.subproc import reap_child
from taaf.game_examples import flip_game


def _config(**kwargs: object) -> RunConfig:
    """Subprocess hosting with the status line off (noise in a test log)."""
    return RunConfig(process_mode="subprocess", status_interval_s=None, **kwargs)  # type: ignore[arg-type]


def _bm(agent: ProbeAgent, n_games: int = 1, **kwargs: object) -> taaf.benchmark.Benchmark:
    """A benchmark of flip tasks hosted by ``agent`` in child processes."""
    return taaf.benchmark.Benchmark(
        label="t",
        games=[taaf.game_examples.flip_game(f"g{i}") for i in range(n_games)],
        solver=AgentSolver(agent=agent),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_answers_made_in_a_child_reach_the_parents_game():
    """The whole point of the one-way pipe: the child never learns what its
    answer scored, and the parent records it anyway."""
    agent = ProbeAgent(n_answers=2, run_config=_config())
    bm = _bm(agent)
    await bm.run()
    run = bm.game_runs[0]
    assert len(run.history) == 2
    assert run.state == "won" and run.final_score == 100.0


async def test_a_child_that_dies_without_the_protocol_is_a_loud_error():
    """A SIGKILLed child may never deliver pipe EOF, so silence has to become
    an error rather than a game that hangs holding a concurrency slot."""
    agent = SelfKillingAgent(run_config=_config())
    bm = _bm(agent)
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "died" in run.solver_note
    # The stderr tail is the only account of an OOM or segfault child.
    assert "child stderr tail" in run.solver_note


async def test_child_stdout_reaches_the_parent_and_never_the_inherited_descriptor(capfd: pytest.CaptureFixture[str]):
    """capfd stands in for the notebook's pipe: the parent's descriptor 1,
    which the child inherits. What the child prints arrives through the
    parent's own print, once; what it writes to descriptor 1 never does."""
    agent = PrintingAgent(run_config=_config())
    bm = _bm(agent)
    await bm.run()
    assert bm.game_runs[0].state == "won"
    out = capfd.readouterr().out
    assert out.count("a line the child prints\n") == 1
    assert out.count("a partial line the child flushes") == 1
    assert "a line native code writes" not in out


async def test_child_descriptor_1_lands_in_the_death_report():
    agent = PrintingAgent(die=True, run_config=_config())
    bm = _bm(agent)
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "a line native code writes" in run.solver_note


@pytest.mark.parametrize(("sends_done", "exitcode"), [(True, 0), (False, 0), (False, 7)])
@pytest.mark.parametrize("prequeue_first", [False, True])
async def test_watchdog_preserves_buffered_protocol_before_classifying_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sends_done: bool, exitcode: int, prequeue_first: bool
):
    from platypus import subproc

    game = flip_game("deadline-race")
    game.start_game()
    view = game.query()
    answers = [taaf.solver_examples.all_black(view), taaf.solver_examples.flip_lr(view)]
    messages: list[tuple[str, Any]] = [
        ("submit", (answers[0], 3, 5)),
        ("submit", (answers[1], 7, 11)),
        ("tokens", (13, 17)),
    ]
    if sends_done:
        messages.append(("done", "ok"))
    host = subproc.SubprocessGameHost(game, subproc._ChildPayload(b"", game.game_id, 0, view, 0, 0, None))
    parent, child = multiprocessing.Pipe(duplex=False)
    inbox: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    release = threading.Event()
    exited = threading.Event()
    death_checks: list[tuple[bool, int]] = []

    class Process:
        def __init__(self) -> None:
            self.exitcode = exitcode

        def is_alive(self) -> bool:
            death_checks.append((parent.poll(), inbox.qsize()))
            return not exited.is_set()

        def join(self, timeout: float | None = None) -> None:
            assert exited.is_set()

    def read_pending() -> None:
        try:
            if not parent.poll():
                return
            message = parent.recv()
        except EOFError:
            loop.remove_reader(parent.fileno())
            message = None
        inbox.put_nowait(message)

    def send_and_exit() -> None:
        assert release.wait(timeout=2.0)
        for message in messages:
            child.send(message)
        child.close()
        exited.set()

    producer = threading.Thread(target=send_and_exit, daemon=True)

    def block_loop_until_exit() -> None:
        release.set()
        producer.join(timeout=2.0)
        assert not producer.is_alive()
        if prequeue_first:
            read_pending()

    loop.add_reader(parent.fileno(), read_pending)
    monkeypatch.setattr(subproc, "_LIVENESS_TIMEOUT_S", 0.01)
    stderr = tmp_path / "stderr.log"
    stderr.write_text("child diagnostic evidence")
    process = Process()
    pending = asyncio.create_task(host._next_message(inbox, process, stderr, read_pending))
    producer.start()
    await asyncio.sleep(0)
    # Both callbacks become ready together, with the blocker first. The child
    # exits while the loop is blocked; the watchdog then precedes pipe delivery.
    loop.call_soon(block_loop_until_exit)
    time.sleep(0.02)  # noqa: ASYNC251 — deliberate watchdog/reader ordering
    received: list[str] = []
    error: subproc.SubprocessDiedError | None = None
    try:
        message = await pending
        while True:
            tag, value = message
            received.append(tag)
            if tag == "done":
                host._finish(None)
                break
            host._apply(tag, value)
            message = await host._next_message(inbox, process, stderr, read_pending)
    except subproc.SubprocessDiedError as exc:
        error = exc
    finally:
        release.set()
        producer.join(timeout=2.0)
        loop.remove_reader(parent.fileno())
        parent.close()

    assert death_checks[0] == (True, int(prequeue_first))
    assert received == [tag for tag, _value in messages]
    run = game.game_run
    assert run is not None
    assert [r.submission.to_json_list() for r in run.history] == [a.to_json_list() for a in answers]
    assert [(r.generated_tokens, r.uncached_input_tokens) for r in run.history] == [(3, 5), (7, 11)]
    if sends_done:
        assert error is None
        assert run.state == "won"
        assert (run.final_generated_tokens, run.final_uncached_input_tokens) == (13, 17)
    else:
        assert error is not None
        assert f"exit code {exitcode}" in str(error)
        assert "without completing the protocol" in str(error)
        assert "child diagnostic evidence" in str(error)
        assert run.state == "playing"


async def test_an_agent_crash_in_a_child_crosses_as_an_agent_error():
    """A crash inside the child is the agent's failure, not the child's
    death — the two get different reports because they need different fixes."""
    agent = CrashingAgent(n_answers=1, run_config=_config())
    bm = _bm(agent)
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "deliberate agent crash" in run.solver_note
    assert "died" not in run.solver_note
    # The answer it made before crashing still counts.
    assert len(run.history) == 1
    assert run.final_score == 100.0


@pytest.mark.local
async def test_the_memory_limit_applies_in_the_child():
    """RLIMIT_AS is run protection: a runaway agent must fail its own game
    rather than take the machine down. Local tier — it allocates for real."""
    agent = HungryAgent(run_config=_config(subprocess_memory_limit_bytes=512 * 2**20))
    bm = _bm(agent)
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None
    assert "MemoryError" in run.solver_note or "died" in run.solver_note


async def test_cancellation_reaps_the_child_and_keeps_its_answer(tmp_path: Path):
    """The relay: SIGTERM cancels the child's call, the parent finishes the
    game with whatever crossed, and nothing is left unreaped."""
    agent = OutcomeDumpingSlowAgent(n_answers=1, dump_dir=str(tmp_path), run_config=_config())
    bm = _bm(agent, n_games=2)
    task = asyncio.create_task(bm.run())
    # Child readiness precedes submission and parent-side pipe delivery.
    # Cancel only once both answers have crossed into the parent's games.
    for _ in range(600):
        if len(bm.game_runs) == 2 and all(len(run.history) == 1 for run in bm.game_runs):
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("children never delivered their answers")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for run in bm.game_runs:
        assert run.state == "cancelled"
        # It answered before the cut, so the answer stands and is scored.
        assert len(run.history) == 1
        assert run.final_score == 100.0


async def test_a_cancelled_child_is_told_it_was_cancelled(tmp_path: Path):
    """``on_finish`` runs in the child, shielded, and is told how the call
    ended — the only way to observe that across the boundary is a file."""
    agent = OutcomeDumpingSlowAgent(n_answers=1, dump_dir=str(tmp_path), run_config=_config())
    bm = _bm(agent)
    task = asyncio.create_task(bm.run())
    for _ in range(600):
        if list(tmp_path.glob("*.ready")):
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("the child never started playing")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(200):
        dumps = list(tmp_path.glob("*.outcome"))
        if dumps:
            assert dumps[0].read_text() == "cancelled 1"
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the child never wrote its outcome")


async def test_the_raise_policy_propagates_out_of_a_child():
    """``on_agent_error="raise"`` ends the run from a child too: the
    traceback crosses the pipe and surfaces where the run was awaited."""
    agent = CrashingAgent(n_answers=0, run_config=_config(on_agent_error="raise"))
    solver = AgentSolver(agent=agent)
    solver.setup()
    game = flip_game("g0")
    game.start_game()
    with pytest.raises(RuntimeError, match="deliberate agent crash"):
        await solver.run_games([game])


def _game_children() -> list[Any]:
    """Live child processes minus the shared-semaphore manager — a
    deliberate session-long singleton (spawned by whichever LLM
    subprocess-mode test ran first on this worker), not a leaked game child."""
    return [p for p in multiprocessing.active_children() if not p.name.startswith("_SemaphoreManager")]


class _FakeChild:
    """A child process as the reaper sees it: alive until some rung ends it,
    or never. Records which rungs were tried."""

    pid = 4242

    def __init__(self, *, dies_on: str | None) -> None:
        self._dies_on = dies_on
        self.alive = True
        self.calls: list[str] = []

    def is_alive(self) -> bool:
        return self.alive

    def kill(self) -> None:
        self.calls.append("kill")
        if self._dies_on == "kill":
            self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.calls.append("join")
        if self._dies_on == "join":
            self.alive = False

    def close(self) -> None:
        self.calls.append("close")
        if self.alive:
            raise ValueError("Cannot close a process while it is still running.")


async def test_reaper_is_silent_for_a_child_that_exits_on_its_own(capsys: pytest.CaptureFixture[str]) -> None:
    child = _FakeChild(dies_on=None)
    child.alive = False
    await reap_child(child, label="g pid 4242")
    assert child.calls == ["join", "close"]
    assert capsys.readouterr().out == ""


async def test_reaper_announces_the_sigkill_rung(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from platypus import subproc

    monkeypatch.setattr(subproc, "_KILL_AFTER_S", 0.0)
    child = _FakeChild(dies_on="kill")
    await reap_child(child, label="g pid 4242")
    assert child.calls == ["kill", "join", "close"]
    out = capsys.readouterr().out
    assert "[reaper] g pid 4242: still alive" in out and "SIGKILL" in out
    assert "exited after SIGKILL" in out


async def test_reaper_abandons_a_child_that_survives_sigkill(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child in uninterruptible wait outlives SIGKILL, and ``close()`` on
    it raises — out of a game's ``finally``, that raise would cancel every
    sibling in the run. The reaper stops short of ``close()``, says so, and
    the caller's game result stands."""
    from platypus import subproc

    monkeypatch.setattr(subproc, "_KILL_AFTER_S", 0.0)
    child = _FakeChild(dies_on=None)
    await reap_child(child, label="g pid 4242")  # must not raise
    assert child.calls == ["kill", "join"], "close() on a live process raises; it must not be reached"
    assert "SURVIVED SIGKILL" in capsys.readouterr().out


# A parent that abandons a real child and then exits. The child is a sleep
# standing in for uninterruptible wait, so its `kill` is made a no-op to
# reproduce "outlives SIGKILL"; nothing else about the ladder is faked.
# The parent's stdout must be a FILE: the orphan inherits it, and a pipe
# would not see EOF until the orphan died — the test would then wait on the
# child, which is exactly what it is checking the parent does not do.
_ABANDON_THEN_EXIT = """
import asyncio, multiprocessing, sys, time
import platypus.subproc as subproc

subproc._KILL_AFTER_S = 0.0
child = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(600,))
child.start()
child.kill = lambda: None
asyncio.run(subproc.reap_child(child, label=f"g pid {child.pid}"))
print("ORPHAN", child.pid, flush=True)
"""


def test_a_parent_that_abandons_a_child_can_still_exit() -> None:
    """The fake child above cannot show this: the interpreter's exit hook
    joins every process still in multiprocessing's active-children set,
    with no timeout. A parent that abandoned a live child and left it there
    would hang at shutdown on the child it gave up on — so this runs a real
    parent, which has to come back."""
    with tempfile.TemporaryFile("w+") as out:
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _ABANDON_THEN_EXIT], stdout=out, stderr=out, text=True, timeout=60, check=False
            )
        finally:
            out.seek(0)
            printed = out.read()
            orphan = next((int(line.split()[1]) for line in printed.splitlines() if line.startswith("ORPHAN ")), None)
            if orphan is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(orphan, signal.SIGKILL)
    assert proc.returncode == 0, printed
    assert "SURVIVED SIGKILL" in printed
    assert orphan is not None, "the parent never reached the line after the reap"


async def test_a_failing_game_winds_down_its_siblings_before_returning(tmp_path: Path):
    """One game's failure under ``raise`` must not hand control back while
    its siblings are still playing: they are cancelled, their children
    reaped, and their ``on_finish`` run — wound down, not abandoned."""
    agent = OutcomeDumpingSlowAgent(
        n_answers=1,
        dump_dir=str(tmp_path),
        crash_on_game_id="crasher",
        crash_after_ready=2,  # both games are inside their calls before the crash
        run_config=_config(on_agent_error="raise"),
    )
    solver = AgentSolver(agent=agent)
    solver.setup()
    games: list[taaf.game.Game] = [flip_game("crasher"), flip_game("sibling")]
    for game in games:
        game.start_game()
    with pytest.raises(RuntimeError, match="deliberate agent crash"):
        await solver.run_games(games)
    assert _game_children() == []
    assert (tmp_path / "sibling.outcome").read_text() == "cancelled 1"
