"""Subprocess hosting (I9): loud child-death handling, error policies
across the process boundary, memory limit, cancellation with reaping.
Engine-side assertions only — agent introspection can't cross processes."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from infra_fixtures import (
    CrashingAgent,
    HungryAgent,
    PrintingAgent,
    ProbeAgent,
    ResultDumpingSlowAgent,
    SelfKillingAgent,
)
from taaf.benchmark import Benchmark
from taaf.game import Game
from taaf.game_examples import ExampleGame

from platypus import AgentSolver, RunConfig
from platypus.subproc import reap_child

WIN_SCRIPT = ("ACTION1",) * 3 + ("ACTION2",) * 3


def _subprocess_config(**overrides: object) -> RunConfig:
    return RunConfig(process_mode="subprocess", **overrides)  # type: ignore[arg-type]


async def test_win_and_give_up_through_subprocess() -> None:
    # The child runs the same loop: a full scripted win (crossing a level
    # transition, with its delivery-only call) and a clean give-up.
    winner = ProbeAgent(script=WIN_SCRIPT, run_config=_subprocess_config())
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=winner))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "won"
    # The delivery-only RESET ack was discarded — the engine saw only the script.
    assert [record.action.id.name for record in run.history] == list(WIN_SCRIPT)

    quitter = ProbeAgent(script=(), give_up_reason="bored", run_config=_subprocess_config())
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=quitter))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "gave_up"
    assert run.solver_note is not None and "bored" in run.solver_note


async def test_agent_crash_in_child_is_contained_and_loud() -> None:
    agent = CrashingAgent(script=("ACTION1",), run_config=_subprocess_config())
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "deliberate agent crash" in run.solver_note


async def test_agent_crash_raise_policy_propagates_from_child() -> None:
    agent = CrashingAgent(script=(), run_config=_subprocess_config(on_agent_error="raise"))
    solver = AgentSolver(agent=agent)
    solver.setup()
    games: list[Game] = [ExampleGame()]
    for game in games:
        game.start_game()
    with pytest.raises(RuntimeError, match="deliberate agent crash"):
        await solver.run_games(games)


async def test_child_sigkill_is_a_loud_error_not_a_fallback() -> None:
    agent = SelfKillingAgent(run_config=_subprocess_config())
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None
    assert "subprocess died" in run.solver_note and "SIGKILL" in run.solver_note
    # No actions were silently replayed in-process.
    assert len(run.history) == 0


async def test_child_stdout_reaches_the_parent_and_never_the_inherited_descriptor(
    capfd: pytest.CaptureFixture[str],
) -> None:
    # capfd stands in for the notebook's pipe: the parent's descriptor 1,
    # which the child inherits. What the child prints arrives through the
    # parent's own print, once; what it writes to descriptor 1 never does.
    agent = PrintingAgent(script=("ACTION1",), run_config=_subprocess_config())
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    assert benchmark.game_runs[0].state == "gave_up"
    out = capfd.readouterr().out
    assert out.count("a line the child prints\n") == 1
    assert out.count("a partial line the child flushes") == 1
    assert "a line native code writes" not in out


async def test_child_descriptor_1_lands_in_the_death_report() -> None:
    agent = PrintingAgent(die=True, run_config=_subprocess_config())
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "a line native code writes" in run.solver_note


async def test_memory_limit_breach_is_loud() -> None:
    agent = HungryAgent(
        allocate_bytes=32 * 2**30,
        run_config=_subprocess_config(subprocess_memory_limit_bytes=8 * 2**30),
    )
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "MemoryError" in run.solver_note


async def test_failing_game_winds_down_its_siblings_before_returning(tmp_path: Path) -> None:
    # asyncio.gather resolves on the FIRST child's outcome without waiting
    # for — or cancelling — its siblings. Left unhandled, one game's failure
    # returns control to the benchmark while the other games are still
    # playing, their subprocesses alive past the end of the run.
    agent = ResultDumpingSlowAgent(
        dump_dir=str(tmp_path),
        crash_on_game_id="crasher",
        crash_after_ready=2,  # both games are in their move loops before the crash
        run_config=_subprocess_config(on_agent_error="raise"),
    )
    solver = AgentSolver(agent=agent)
    solver.setup()
    games: list[Game] = [ExampleGame(label="crasher"), ExampleGame(label="sibling")]
    for game in games:
        game.start_game()
    with pytest.raises(RuntimeError, match="deliberate agent crash"):
        await solver.run_games(games)
    # No survivors: the sibling was cancelled and its child reaped, and its
    # on_finish ran (it was wound down, not abandoned mid-move).
    assert _game_children() == []
    assert (tmp_path / "sibling.result").exists()


def _game_children() -> list[Any]:
    """Live child processes minus the shared-semaphore manager: that one
    is a deliberate session-long singleton (enable_multiprocess_semaphores,
    spawned by whichever subprocess-mode test ran first on this worker),
    not a leaked game child."""
    return [p for p in multiprocessing.active_children() if not p.name.startswith("_SemaphoreManager")]


async def test_cancellation_reaps_child_processes(tmp_path: Path) -> None:
    agent = ResultDumpingSlowAgent(dump_dir=str(tmp_path), run_config=_subprocess_config())
    solver = AgentSolver(agent=agent)
    solver.setup()
    games: list[Game] = [ExampleGame(label=f"game{i}") for i in range(2)]
    for game in games:
        game.start_game()
    task = asyncio.create_task(solver.run_games(games))
    # Wait until both children reached their move loops (ready markers) —
    # cancelling earlier can SIGTERM a child that is still importing.
    for _ in range(600):
        await asyncio.sleep(0.05)
        if len(list(tmp_path.glob("*.ready"))) >= 2:
            break
    else:
        raise AssertionError("children never became ready")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for game in games:
        assert game.game_run is not None
        assert game.game_run.state == "cancelled"
    # The reaper left no orphans.
    assert _game_children() == []
    # Each child's on_finish saw the REAL result: parent-computed, score
    # included (not a child-side score=None stand-in).
    results = sorted(tmp_path.glob("*.result"))
    assert len(results) == 2
    for result_file in results:
        state, _levels, score = result_file.read_text().split()
        assert state == "cancelled"
        assert score != "None"


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
    import platypus.subproc as subproc

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
    import platypus.subproc as subproc

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
