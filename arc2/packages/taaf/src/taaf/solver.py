"""Abstract ``Solver`` base class. Concrete example
subclasses live in ``taaf.solver_examples``.

Contract:

- ``setup()`` runs once before ``run_games``; ``teardown()`` once after
  it finishes / raises / cancels.
- ``start_up()`` brings up what is slow before the solver can play (an
  engine, say). ``Benchmark`` calls it right before ``run_games``, in the
  same task, under the run's startup deadline when one is set.
  ``shut_down()`` releases it when ``run_games`` did not; a no-op unless
  ``start_up()`` was called.
- ``run_games(list[Game])`` plays each game via ``game.execute_action``.
  Order and concurrency are the implementation's choice.
- On ``asyncio.CancelledError`` the solver must call ``finish_game()`` on
  every still-``playing`` game and re-raise. ``Game.finish_game()`` reads
  the current task's cancelling state to pick ``cancelled`` vs ``gave_up``.
- The solver must yield at least every 5 seconds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import taaf.game

if TYPE_CHECKING:
    import taaf.deploy


@dataclass
class Solver:
    """Abstract solver.

    Fields (all populated by ``Benchmark`` before ``setup()``):

    - ``label``: short string used in plots and diagnostics.
    - ``runtime_environment``: the ``DeploymentTarget`` this benchmark is
      running under, or ``None`` for a direct ``Benchmark.run()`` call.
      Solvers dispatch on it by ``isinstance`` (or by reading target config
      fields like ``gpu`` / ``cpus_per_gpu``) rather than parsing a tag.
    - ``job_dir``: where solver-internal diagnostics may land. ``None`` when
      ``Benchmark.job_dir`` was not set.
    - ``soft_end_time``: indicative pacing hint. Cancellation is enforced
      by ``Benchmark`` via ``task.cancel()`` — solvers never need to check
      the clock themselves.
    - ``stop_run``: ends the run now, the way the soft deadline does —
      every game the solver still holds lands ``cancelled`` and teardown
      writes everything — with a reason recorded on the benchmark. Set only
      while ``Benchmark.run`` is awaiting the solver, for a solver's own
      watchdog to call; ``None`` outside a run.
    - ``minimal_diagnostics``: submission mode. Solvers should skip optional
      diagnostic capture / writes when True; ``Benchmark`` already skips its
      own JSON / HTML / sidecar saves.
    """

    label: str = ""
    runtime_environment: taaf.deploy.DeploymentTarget | None = field(default=None, init=False)
    job_dir: Path | None = field(default=None, init=False)
    soft_end_time: datetime | None = field(default=None, init=False)
    stop_run: Callable[[str], None] | None = field(default=None, init=False, repr=False, compare=False)
    minimal_diagnostics: bool = field(default=False, init=False)
    _setup_called: bool = field(default=False, init=False, repr=False)
    _teardown_called: bool = field(default=False, init=False, repr=False)
    _start_up_called: bool = field(default=False, init=False, repr=False)
    _shut_down_called: bool = field(default=False, init=False, repr=False)

    def setup(self) -> None:
        """Run game-independent setup, exactly once."""
        assert not self._setup_called, "setup() already called"
        self._setup_called = True
        self._setup()

    async def start_up(self) -> None:
        """Bring up what is slow before the solver can play, exactly once."""
        assert self._setup_called, "Call setup() before start_up()"
        assert not self._teardown_called, "Cannot call start_up() after teardown()"
        assert not self._start_up_called, "start_up() already called"
        self._start_up_called = True
        await self._start_up()

    async def shut_down(self) -> None:
        """Release what ``start_up()`` brought up; a no-op when it was never
        called, and after the first call."""
        if not self._start_up_called or self._shut_down_called:
            return
        self._shut_down_called = True
        await self._shut_down()

    def teardown(self) -> None:
        """Run game-independent teardown, exactly once, after the games."""
        assert self._setup_called, "Call setup() before teardown()"
        assert not self._teardown_called, "teardown() already called"
        self._teardown_called = True
        self._teardown()

    async def run_games(self, games: list[taaf.game.Game]) -> None:
        """Play every game. Order and concurrency are the subclass's choice."""
        assert self._setup_called, "Call setup() before run_games()"
        assert not self._teardown_called, "Cannot call run_games() after teardown()"
        await self._run_games(games)

    # --- subclass hooks -----------------------------------------------------

    def _setup(self) -> None:
        """Optional setup. Default: no-op."""

    def _teardown(self) -> None:
        """Optional teardown. Default: no-op."""

    async def _start_up(self) -> None:
        """Optional: bring up what is slow before the solver can play.
        Default: no-op."""

    async def _shut_down(self) -> None:
        """Optional: release what ``_start_up`` brought up, for a run whose
        ``run_games`` never ran or did not release it. Default: no-op."""

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        """The solver's play loop; implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run_games")
