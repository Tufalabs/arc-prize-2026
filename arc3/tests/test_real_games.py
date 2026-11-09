"""Cross-member integration: the random agent on a real local game via
GameAPI (offline bundled env — no network, no key), multi-pass. n_passes
>= 2 against GameAPI is a known hidden-problem area — keep it exercised."""

from __future__ import annotations

from taaf.benchmark import Benchmark
from taaf.game_api import GameAPI

from platypus import AgentSolver, RunConfig
from reference_agents import RandomAgent


async def test_random_agent_multipass_on_gameapi() -> None:
    agent = RandomAgent(max_moves=8, run_config=RunConfig(rng_seed=0))
    benchmark = Benchmark(
        games=[GameAPI(env_name="ls20")],
        solver=AgentSolver(agent=agent),
        n_passes=2,
    )
    await benchmark.run()
    assert len(benchmark.game_runs) == 2
    for run in benchmark.game_runs:
        assert run.state in ("gave_up", "won")
        assert run.final_score is not None
        assert len(run.history) <= 8
    # Independent pass copies with distinct per-game seeds.
    histories = [[record.action.id.name for record in run.history] for run in benchmark.game_runs]
    assert histories[0] != histories[1]
