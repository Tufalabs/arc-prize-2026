# type: ignore
"""Tests for the solver's start-up step and the startup deadline that bounds it."""

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

import pytest

import taaf

#: Every event of a run in the order it happened; reset by each test.
EVENTS: list[str] = []


@dataclass
class LoggedSolver(taaf.solver_examples.SolverFlipLR):
    """The reference solver, saying when it starts up, plays and shuts down;
    ``start_up_s`` is how long its start-up takes, ``start_up_error`` what it
    raises at the end of it."""

    start_up_s: float = 0.0
    start_up_error: ClassVar[Exception | None] = None
    submission_seen: ClassVar[bool | None] = None

    async def _start_up(self):
        EVENTS.append("start_up")
        submission = self.job_dir / "submission.json" if self.job_dir is not None else None
        LoggedSolver.submission_seen = submission is not None and submission.is_file()
        await asyncio.sleep(self.start_up_s)
        if LoggedSolver.start_up_error is not None:
            raise LoggedSolver.start_up_error
        EVENTS.append("up")

    async def _run_games(self, games):
        EVENTS.append("run_games")
        await super()._run_games(games)

    async def _shut_down(self):
        EVENTS.append("shut_down")


@dataclass
class BlockingSolver(LoggedSolver):
    """Starts up by blocking the event loop, as a synchronous engine build does."""

    async def _start_up(self):
        EVENTS.append("start_up")
        time.sleep(self.start_up_s)  # noqa: ASYNC251 - the block is the point
        EVENTS.append("up")


@pytest.fixture(autouse=True)
def _fresh_events():
    EVENTS.clear()
    LoggedSolver.start_up_error = None
    LoggedSolver.submission_seen = None


@pytest.fixture
def exits(monkeypatch):
    """The killswitch's exits, recorded as (seconds after the fixture, status)
    instead of ending the test process."""
    t0 = time.monotonic()
    recorded: list[tuple[float, int]] = []
    monkeypatch.setattr(taaf.benchmark, "_exit_process", lambda code: recorded.append((time.monotonic() - t0, code)))
    return recorded


def _benchmark(solver=None, **fields) -> taaf.benchmark.Benchmark:
    return taaf.benchmark.Benchmark(
        games=[taaf.game_examples.flip_game("a"), taaf.game_examples.flip_game("b")],
        solver=solver or LoggedSolver(),
        **fields,
    )


async def test_the_solver_starts_up_after_the_placeholder_submission_and_before_it_plays(tmp_path):
    """Start-up comes in the solver task, after the placeholder submission is
    written, so a process that dies starting up still leaves one."""
    bm = _benchmark(job_dir=tmp_path, submission_path=tmp_path / "submission.json")
    await bm.run()
    assert EVENTS == ["start_up", "up", "run_games", "shut_down"]
    assert LoggedSolver.submission_seen is True
    assert [run.state for run in bm.game_runs] == ["won", "won"]


async def test_a_start_up_that_fails_is_a_solver_error_like_any_other(tmp_path):
    """As when the engine started inside ``run_games``: the run tears down and
    keeps its placeholder submission, and raises only under
    ``raise_solver_errors``."""
    LoggedSolver.start_up_error = RuntimeError("the engine did not come up")
    bm = _benchmark(submission_path=tmp_path / "submission.json")
    await bm.run()
    assert EVENTS == ["start_up", "shut_down"]
    assert [run.state for run in bm.game_runs] == ["crashed", "crashed"]
    assert set(json.loads((tmp_path / "submission.json").read_text())) == {"a", "b"}

    EVENTS.clear()
    bm = _benchmark(raise_solver_errors=True)
    with pytest.raises(RuntimeError, match="the engine did not come up"):
        await bm.run()
    assert EVENTS == ["start_up", "shut_down"]


async def test_the_soft_deadline_during_start_up_cancels_the_games():
    bm = _benchmark(solver=LoggedSolver(start_up_s=5.0))
    t0 = time.monotonic()
    await bm.run(soft_end_time=datetime.now() + timedelta(seconds=0.2))
    assert time.monotonic() - t0 < 3
    assert EVENTS == ["start_up", "shut_down"]
    assert [run.state for run in bm.game_runs] == ["cancelled", "cancelled"]


async def test_the_startup_killswitch_fires_on_time_with_the_event_loop_blocked(exits):
    """The deadline passes while the start-up holds the event loop: the thread
    exits the process then, not when the loop is next free."""
    bm = _benchmark(solver=BlockingSolver(start_up_s=1.0))
    await bm.run(startup_deadline=datetime.now() + timedelta(seconds=0.2))
    assert len(exits) == 1
    when, code = exits[0]
    assert code == taaf.benchmark.STARTUP_DEADLINE_EXIT_CODE
    assert 0.15 < when < 0.8, when


def test_the_startup_killswitch_ends_the_process_for_real(tmp_path):
    """Unfaked: a process whose solver is not up by the deadline dies with the
    killswitch's status, mid start-up."""
    script = tmp_path / "late_start.py"
    script.write_text(
        "import asyncio, time\n"
        "from dataclasses import dataclass\n"
        "from datetime import datetime, timedelta\n"
        "import taaf\n"
        "@dataclass\n"
        "class Slow(taaf.solver_examples.SolverFlipLR):\n"
        "    async def _start_up(self):\n"
        "        time.sleep(30)\n"
        "bm = taaf.benchmark.Benchmark(games=[taaf.game_examples.flip_game('a')], solver=Slow())\n"
        "asyncio.run(bm.run(startup_deadline=datetime.now() + timedelta(seconds=0.5)))\n"
        "print('the run returned')\n"
    )
    t0 = time.monotonic()
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == taaf.benchmark.STARTUP_DEADLINE_EXIT_CODE, done.stdout + done.stderr
    assert time.monotonic() - t0 < 20
    assert "the solver is not up by its startup deadline" in done.stdout
    assert "the run returned" not in done.stdout


def test_the_startup_killswitch_exits_while_nobody_drains_its_output(tmp_path):
    """Unfaked, with the start-up filling a stdout pipe nobody reads: the
    killswitch's exit line cannot be written, and the process exits anyway."""
    script = tmp_path / "stuck_output.py"
    script.write_text(
        "import asyncio, sys\n"
        "from dataclasses import dataclass\n"
        "from datetime import datetime, timedelta\n"
        "import taaf\n"
        "@dataclass\n"
        "class Chatty(taaf.solver_examples.SolverFlipLR):\n"
        "    async def _start_up(self):\n"
        "        while True:\n"
        "            sys.stdout.write('x' * 65536)\n"
        "            sys.stdout.flush()\n"
        "bm = taaf.benchmark.Benchmark(games=[taaf.game_examples.flip_game('a')], solver=Chatty())\n"
        "asyncio.run(bm.run(startup_deadline=datetime.now() + timedelta(seconds=0.5)))\n"
    )
    child = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Nothing reads the pipes until the child is gone.
        assert child.wait(timeout=30) == taaf.benchmark.STARTUP_DEADLINE_EXIT_CODE
    finally:
        child.kill()
        child.communicate()


async def test_a_solver_up_before_its_startup_deadline_disarms_the_killswitch(exits):
    """The deadline then passes during play, which it no longer bounds."""
    bm = _benchmark(solver=LoggedSolver(start_up_s=0.05, delay_s=0.1))
    await bm.run(startup_deadline=datetime.now() + timedelta(seconds=0.3))
    await asyncio.sleep(0.2)
    assert exits == []
    assert EVENTS == ["start_up", "up", "run_games", "shut_down"]


async def test_a_startup_deadline_already_past_fires_at_once(exits):
    """Setup commands took longer than the whole allowance: the killswitch,
    armed only as the solver starts up, fires as it is armed."""
    bm = _benchmark(solver=LoggedSolver(start_up_s=0.3))
    await bm.run(startup_deadline=datetime.now() - timedelta(seconds=1))
    assert len(exits) == 1 and exits[0][0] < 0.2
