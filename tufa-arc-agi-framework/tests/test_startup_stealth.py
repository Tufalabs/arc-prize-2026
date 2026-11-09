# type: ignore
"""Tests for starting the solver before any game is opened (R2.66)."""

import ast
import asyncio
import json
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

import arc_agi
import pytest

import taaf
import taaf.competition_arcade
import taaf.deploy_kaggle
from taaf.competition_arcade import CompetitionArcadeServer
from taaf.game_api import GameAPI

#: Every event of a run in the order it happened; reset by each test.
EVENTS: list[str] = []


@dataclass
class LoggedGame(taaf.game_examples.ExampleGame):
    """An example game that says when it is opened."""

    def _start_game(self, session):
        EVENTS.append(f"open {self.label}")
        return super()._start_game(session)


@dataclass
class LoggedSolver(taaf.solver_examples.SolverRandom):
    """A random solver that says when it starts up, plays and shuts down;
    ``start_up_s`` is how long its start-up takes, ``start_up_error`` what it
    raises at the end of it."""

    start_up_s: float = 0.0
    start_up_error: ClassVar[Exception | None] = None

    async def _start_up(self):
        EVENTS.append("start_up")
        await asyncio.sleep(self.start_up_s)
        if LoggedSolver.start_up_error is not None:
            raise LoggedSolver.start_up_error
        EVENTS.append("up")

    async def _run_games(self, games):
        EVENTS.append("run_games")
        await super()._run_games(games)

    async def _shut_down(self):
        EVENTS.append("shut_down")


@pytest.fixture(autouse=True)
def _fresh_events():
    EVENTS.clear()
    LoggedSolver.start_up_error = None


def _games(source_events: bool = False):
    """Two games, as the source a run lists them from."""
    if source_events:
        EVENTS.append("list")
    return [LoggedGame(label="a"), LoggedGame(label="b")]


def _benchmark(stealth: bool, **solver_fields) -> taaf.benchmark.Benchmark:
    return taaf.benchmark.Benchmark(
        games=_games(),
        solver=LoggedSolver(seed=1, max_actions_per_game=3, **solver_fields),
        startup_stealth=stealth,
    )


async def test_startup_stealth_opens_and_lists_no_game_before_the_solver_is_up():
    bm = _benchmark(stealth=True)
    await bm.run(games_source=lambda: _games(source_events=True))
    assert EVENTS == ["start_up", "up", "list", "open a", "open b", "run_games", "shut_down"]
    assert [run.state for run in bm.game_runs] == ["gave_up", "gave_up"]


async def test_without_startup_stealth_the_games_open_first_and_start_up_is_not_called():
    """The default order is the one before R2.66: games open, then the solver
    runs and brings up whatever it needs itself; ``start_up`` and so
    ``shut_down`` never run."""
    bm = _benchmark(stealth=False)
    await bm.run()
    assert EVENTS == ["open a", "open b", "run_games"]


async def test_a_start_up_that_fails_opens_no_game_and_raises():
    """Not swallowed as a solver error (R2.15): with no game opened there is
    nothing to score, and a Kaggle submission must end in an error."""
    LoggedSolver.start_up_error = RuntimeError("the engine did not load")
    bm = _benchmark(stealth=True)
    with pytest.raises(RuntimeError, match="the engine did not load"):
        await bm.run(games_source=lambda: _games(source_events=True))
    assert EVENTS == ["start_up", "shut_down"]
    assert bm.game_runs == []


async def test_the_soft_deadline_bounds_a_start_up_that_never_finishes():
    bm = _benchmark(stealth=True, start_up_s=30.0)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="soft deadline passed while the solver was starting up"):
        await bm.run(soft_end_time=datetime.now() + timedelta(seconds=0.2))
    assert time.monotonic() - t0 < 5.0
    assert EVENTS == ["start_up", "shut_down"]
    assert bm.game_runs == []


async def test_a_game_that_fails_to_open_after_start_up_still_shuts_the_solver_down():
    """``run_games`` never runs, so ``shut_down`` is what releases what
    ``start_up`` brought up."""

    def broken_source():
        raise ConnectionError("gateway unreachable")

    bm = _benchmark(stealth=True)
    with pytest.raises(ConnectionError, match="gateway unreachable"):
        await bm.run(games_source=broken_source)
    assert EVENTS == ["start_up", "up", "shut_down"]


async def test_games_source_is_checked_against_game_weights_once_it_has_listed():
    bm = _benchmark(stealth=True)
    bm.game_weights = [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="game_weights length 3 must equal games length 2"):
        await bm.run(games_source=lambda: _games(source_events=True))
    assert "open a" not in EVENTS


@pytest.mark.parametrize("listed", ["through games_source", "before run"])
async def test_startup_stealth_sends_the_gateway_nothing_until_the_solver_is_up(monkeypatch, tmp_path, listed):
    """Against the localhost competition gateway (R11.13), listed the way the
    Kaggle notebook lists a rerun's games: not one request reaches it while
    the solver starts up, and the run plays through it afterwards. Listed
    before ``run`` instead, as the notebook did before R2.66, the listing's
    requests are there to see: what the count would catch."""
    # A listing Arcade also scans its working directory for environments; a
    # rerun's is the kernel's empty working dir.
    monkeypatch.chdir(tmp_path)
    requests: list[str] = []

    def record(handler, fmt, *args):
        # Not the server's own readiness probe, which it logs once its reply
        # is out, so it can land after the server has started.
        if handler.path != "/api/healthcheck":
            requests.append(fmt % args)

    monkeypatch.setattr(taaf.competition_arcade._QuietWSGIRequestHandler, "log_message", record)
    seen_during_start_up: list[list[str]] = []

    @dataclass
    class WatchingSolver(taaf.solver_examples.SolverRandom):
        async def _start_up(self):
            await asyncio.sleep(0.2)
            seen_during_start_up.append(list(requests))

    with CompetitionArcadeServer(game_ids=("ls20",), total_runs=2) as server:

        def gateway_games():
            arcade = arc_agi.Arcade(
                operation_mode=arc_agi.OperationMode.COMPETITION, arc_base_url=server.base_url, environments_dir=""
            )
            ids = [env.game_id for env in arcade.available_environments]
            return [GameAPI(env_name=game_id, arcade_spec=server.arcade_spec) for game_id in ids]

        bm = taaf.benchmark.Benchmark(
            solver=WatchingSolver(seed=7, max_actions_per_game=1),
            startup_stealth=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="arcengine returned no baseline_actions")
            if listed == "before run":
                bm.games = gateway_games()
                await bm.run()
            else:
                await bm.run(games_source=gateway_games)

    if listed == "before run":
        assert seen_during_start_up[0], "the count saw no request even for a listing made before start-up"
    else:
        assert seen_during_start_up == [[]]
    assert requests, "the run never reached the gateway"
    assert [run.game_id for run in bm.game_runs] == ["k000", "k001"]


@dataclass
class BlockingSolver(LoggedSolver):
    """Starts up by blocking the event loop, as a synchronous engine build does."""

    async def _start_up(self):
        EVENTS.append("start_up")
        time.sleep(self.start_up_s)
        EVENTS.append("up")


@pytest.fixture
def exits(monkeypatch):
    """The killswitch's exits, recorded as (seconds after the fixture, status)
    instead of ending the test process."""
    t0 = time.monotonic()
    recorded: list[tuple[float, int]] = []
    monkeypatch.setattr(taaf.benchmark, "_exit_process", lambda code: recorded.append((time.monotonic() - t0, code)))
    return recorded


async def test_the_startup_killswitch_fires_on_time_with_the_event_loop_blocked(exits):
    """The deadline passes while the start-up holds the event loop: the thread
    exits the process then, not when the loop is next free (R2.67). With the
    exit faked, the late start-up is still refused before any game opens."""
    bm = taaf.benchmark.Benchmark(
        games=_games(), solver=BlockingSolver(seed=1, max_actions_per_game=3, start_up_s=1.0), startup_stealth=True
    )
    with pytest.raises(RuntimeError, match="came up after its startup deadline"):
        await bm.run(startup_deadline=datetime.now() + timedelta(seconds=0.2))
    assert len(exits) == 1
    when, code = exits[0]
    assert code == taaf.benchmark.STARTUP_DEADLINE_EXIT_CODE
    assert 0.15 < when < 0.8, when
    assert not any(event.startswith("open") for event in EVENTS)


def test_the_startup_killswitch_ends_the_process_for_real(tmp_path):
    """Unfaked: a process whose solver is not up by the deadline dies with the
    killswitch's status, mid start-up and before any game is opened."""
    script = tmp_path / "late_start.py"
    script.write_text(
        "import asyncio, time\n"
        "from dataclasses import dataclass\n"
        "from datetime import datetime, timedelta\n"
        "import taaf\n"
        "@dataclass\n"
        "class Slow(taaf.solver_examples.SolverRandom):\n"
        "    async def _start_up(self):\n"
        "        time.sleep(30)\n"
        "game = taaf.game_examples.ExampleGame(label='a')\n"
        "bm = taaf.benchmark.Benchmark(games=[game], solver=Slow(), startup_stealth=True)\n"
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
        "class Chatty(taaf.solver_examples.SolverRandom):\n"
        "    async def _start_up(self):\n"
        "        while True:\n"
        "            sys.stdout.write('x' * 65536)\n"
        "            sys.stdout.flush()\n"
        "game = taaf.game_examples.ExampleGame(label='a')\n"
        "bm = taaf.benchmark.Benchmark(games=[game], solver=Chatty(), startup_stealth=True)\n"
        "asyncio.run(bm.run(startup_deadline=datetime.now() + timedelta(seconds=0.5)))\n"
    )
    child = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        # Nothing reads the pipes until the child is gone.
        assert child.wait(timeout=30) == taaf.benchmark.STARTUP_DEADLINE_EXIT_CODE
    finally:
        child.kill()
        child.communicate()


async def test_a_start_up_that_blocks_past_the_soft_deadline_opens_no_game():
    """The soft deadline's cancellation cannot run while the start-up holds
    the event loop, so a start-up that returns past it still opens no game."""
    bm = taaf.benchmark.Benchmark(
        games=_games(), solver=BlockingSolver(seed=1, max_actions_per_game=3, start_up_s=0.1), startup_stealth=True
    )
    with pytest.raises(RuntimeError, match="soft deadline passed while the solver was starting up"):
        await bm.run(soft_end_time=datetime.now() + timedelta(seconds=0.02))
    assert EVENTS == ["start_up", "up", "shut_down"]
    assert bm.game_runs == []


async def test_a_solver_up_before_its_startup_deadline_disarms_the_killswitch(exits):
    """The deadline then passes during play, which it no longer bounds."""
    bm = taaf.benchmark.Benchmark(
        games=_games(),
        solver=LoggedSolver(seed=1, max_actions_per_game=8, delay_move=0.05, start_up_s=0.05),
        startup_stealth=True,
    )
    await bm.run(startup_deadline=datetime.now() + timedelta(seconds=0.3))
    await asyncio.sleep(0.2)
    assert exits == []
    assert EVENTS[:4] == ["start_up", "up", "open a", "open b"]


async def test_a_startup_deadline_already_past_fires_at_once(exits):
    """Package install took longer than the whole allowance: the killswitch,
    armed only as the solver starts, fires as it is armed."""
    bm = _benchmark(stealth=True, start_up_s=0.3)
    with pytest.raises(RuntimeError, match="came up after its startup deadline"):
        await bm.run(startup_deadline=datetime.now() - timedelta(seconds=1))
    assert len(exits) == 1 and exits[0][0] < 0.2


async def test_a_startup_deadline_without_startup_stealth_is_refused(exits):
    bm = _benchmark(stealth=False)
    with pytest.raises(ValueError, match="startup_deadline needs startup_stealth"):
        await bm.run(startup_deadline=datetime.now() + timedelta(minutes=5))
    assert EVENTS == [] and exits == []


async def test_a_kaggle_target_refuses_a_startup_deadline_without_startup_stealth(tmp_path):
    """Refused at deploy, before anything is staged, rather than by ``run``
    on Kaggle once the install has been paid for."""
    bm = _benchmark(stealth=False)
    bm.job_dir = tmp_path / "job"
    target = taaf.deploy_kaggle.KaggleTarget(
        dry_run=True, startup_stealth=False, startup_deadline_s=25 * 60, submission_startup_deadline_s=14 * 60
    )
    with pytest.raises(ValueError, match="startup_deadline_s, submission_startup_deadline_s needs startup_stealth"):
        await bm.deploy(target)
    assert not bm.job_dir.exists()


@pytest.mark.parametrize("share", [False, True])
def test_the_kaggle_notebooks_list_the_gateway_s_games_through_one_function(share):
    """Both runners take startup_stealth from the target, on unless it says
    otherwise, and the rerun's two gateway calls -- the readiness wait and
    the listing -- live in one function: handed to ``run`` as
    ``games_source`` under stealth, called before the run without it, as
    the notebooks did before R2.66."""
    template = taaf.deploy_kaggle._SHARE_NOTEBOOK_TEMPLATE if share else taaf.deploy_kaggle._NOTEBOOK_TEMPLATE
    notebook = json.loads(taaf.deploy_kaggle._render_kaggle_notebook(run_as_submission=False, template=template))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    assert 'bm.startup_stealth = getattr(target, "startup_stealth", True)' in code
    assert taaf.deploy_kaggle.KaggleTarget().startup_stealth
    assert "games_source=games_source" in code
    assert "startup_deadline=startup_deadline" in code
    assert '"submission_startup_deadline_s" if run_as_submission else "startup_deadline_s"' in code
    # Before the run: a version the killswitch ends never comes back from it.
    assert code.index('"submission.parquet"') < code.index("await bm.run(")

    def calls(node, name):
        return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]

    tree = ast.parse(code)
    (gateway_games,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_gateway_games"]
    for name in ("_wait_for_gateway", "_competition_games"):
        assert calls(tree, name), name
        assert len(calls(tree, name)) == len(calls(gateway_games, name)), f"{name} is called outside _gateway_games"
    (choice,) = [n for n in ast.walk(tree) if isinstance(n, ast.If) and ast.unparse(n.test) == "bm.startup_stealth"]
    assert [ast.unparse(n) for n in choice.body] == ["games_source = _gateway_games"]
    assert [ast.unparse(n) for n in choice.orelse] == ["bm.games = _gateway_games()"]


@pytest.mark.parametrize("share", [False, True])
def test_the_kaggle_notebooks_take_the_earlier_of_the_two_startup_deadlines(share):
    """One deadline counts from notebook start, one from machine boot: a
    session that sat on its machine for hours is past the second before its
    notebook starts, and with no reading from /proc only the first applies.
    In a submission both are the target's submission ones."""
    template = taaf.deploy_kaggle._SHARE_NOTEBOOK_TEMPLATE if share else taaf.deploy_kaggle._NOTEBOOK_TEMPLATE
    notebook = json.loads(taaf.deploy_kaggle._render_kaggle_notebook(run_as_submission=False, template=template))
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    # The notebook's own helper, executed as written rather than restated here.
    helper = ast.Module(
        body=[n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "_startup_deadline"],
        type_ignores=[],
    )
    namespace = {"datetime": datetime, "timedelta": timedelta}
    exec(compile(helper, template.name, "exec"), namespace)  # noqa: S102

    def deadline(machine_uptime_s, run_as_submission=False, **allowances):
        target = taaf.deploy_kaggle.KaggleTarget(dry_run=True, **allowances)
        return namespace["_startup_deadline"](
            target, run_as_submission=run_as_submission, notebook_start=start, machine_uptime_s=machine_uptime_s
        )

    start = 1_790_000_000.0
    both = {"startup_deadline_s": 14 * 60, "startup_deadline_after_boot_s": 20 * 60}
    from_start = datetime.fromtimestamp(start + 14 * 60)
    # A machine booted shortly before: notebook start's deadline comes first.
    assert deadline(120.0, **both) == from_start
    # One that sat for hours: boot's deadline, already past at notebook start.
    assert deadline(24071.0, **both) == datetime.fromtimestamp(start - 24071 + 20 * 60)
    assert deadline(24071.0, **both) < datetime.fromtimestamp(start)
    # Each alone, and neither.
    assert deadline(None, **both) == from_start
    assert deadline(24071.0, startup_deadline_s=14 * 60) == from_start
    assert deadline(600.0, startup_deadline_after_boot_s=20 * 60) == datetime.fromtimestamp(start + 10 * 60)
    assert deadline(None, startup_deadline_after_boot_s=20 * 60) is None
    assert deadline(600.0) is None
    # By mode: a machine up 40 minutes when a prompt session's notebook starts
    # passes the long allowance outside a submission, and is past the short one in it.
    modes = {**both, "startup_deadline_after_boot_s": 90 * 60, "submission_startup_deadline_after_boot_s": 20 * 60}
    assert deadline(2379.0, **modes) == from_start
    assert deadline(2379.0, run_as_submission=True, **modes) == datetime.fromtimestamp(start - 2379 + 20 * 60)
    assert deadline(2379.0, run_as_submission=True, startup_deadline_after_boot_s=20 * 60) is None
    # And from notebook start: the longer allowance outside a submission, the tighter one in it.
    starts = {"startup_deadline_s": 25 * 60, "submission_startup_deadline_s": 14 * 60}
    assert deadline(120.0, **starts) == datetime.fromtimestamp(start + 25 * 60)
    assert deadline(120.0, run_as_submission=True, **starts) == from_start
    assert deadline(120.0, run_as_submission=True, startup_deadline_s=25 * 60) is None
