"""``ctx.pass_index`` is TAAF's stamp on the copy being played, in either
process mode — never the copy's position in the (shuffled) play order."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import taaf.benchmark
import taaf.game
from platypus import Agent, AgentSolver, GiveUp, MoveContext, RunConfig
from taaf.game_examples import flip_game


@dataclass
class PassReporter(Agent):
    """Gives up at once, naming the pass it was told it is."""

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Report the pass through the give-up reason, which the run records."""
        del view
        return GiveUp(f"pass {ctx.pass_index}")


@pytest.mark.parametrize("mode", ["inprocess", "subprocess"])
async def test_the_context_carries_the_copys_own_pass(mode: str) -> None:
    """Shuffled play order, so a position in the list and a pass disagree
    for at least some copies; each copy must still be told its own."""
    config = RunConfig(process_mode=mode, status_interval_s=None, shuffle_games=True, shuffle_seed=1)  # type: ignore[arg-type]
    benchmark = taaf.benchmark.Benchmark(
        label="passes",
        games=[flip_game(f"g{i}") for i in range(3)],
        solver=AgentSolver(agent=PassReporter(run_config=config)),
        n_passes=3,
    )
    await benchmark.run()

    assert len(benchmark.game_runs) == 9
    assert {run.pass_index for run in benchmark.game_runs} == {0, 1, 2}
    for run in benchmark.game_runs:
        assert run.solver_note == f"gave up: pass {run.pass_index}"
