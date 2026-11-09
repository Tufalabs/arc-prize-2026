"""The seal's production engine comes up and plays, on a B200.

Nothing else covers the seal's production startup: the GPU matrix runs
backend mechanics on a small model, and the duck's engine columns boot the
duck. One column, one engine boot:

- ``seal_direct_boot``: what a bare ``SealAgent.build()`` boots, Flash-Next
  in-process on the B200 with the expert kernels ullm pins, started through the
  solver's normal path. The game must beat taps level 1: a smoke of the whole
  loop, since the kernel, the file suite and the per-level history all have
  to hold for a level to fall.

The boot is put through the backend matrix before its game
(``LLMConfig.verify_backend``), so an expert backend the engine refuses fails
here by name rather than as a lost game.

Deliberately absent: the duck's traced-nesting variants. Nesting is a
property of the backend both agents share, and the duck's columns hold it.
Also absent, as there: any assertion about the ``verify_serving_bug`` check,
whose verdict is a statistical read of boot-varying engine behavior (see
EngineFactory.verify_serving_bug).

Run via ``make test-seal-engine-srun`` (needs a B200), not the standard GPU
matrix columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _require_b200(request: pytest.FixtureRequest) -> None:
    pytest.importorskip("vllm")
    import torch

    if not torch.cuda.is_available():
        if "gpu" in (request.config.option.markexpr or ""):
            pytest.fail("gpu tests selected but no GPU is visible; run under `make test-seal-engine-srun`")
        pytest.skip("no GPU available")
    if torch.cuda.get_device_properties(0).total_memory < 100 * 1024**3:
        pytest.skip("the production engine needs a B200-class card")
    if "direct_boot" not in (request.config.option.keyword or ""):
        pytest.skip("the production engine needs a GPU to itself: its own column selects it")


async def _play_taps(agent: Any, job_dir: Path | None, *, keep_backend: Any = None) -> Any:
    """One taps game to level 1 under the solver; returns the run. A
    ``KeptBackend`` holder receives the engine when given."""
    import taaf.benchmark
    import taaf.game_api

    from platypus import LLMAgentSolver

    solver = LLMAgentSolver(agent=agent, keep_backend=keep_backend)
    benchmark = taaf.benchmark.Benchmark(
        # stop_at_level=1: the game counts as won the instant one level is
        # completed, so a win assertion is exactly "beat level 1".
        games=[taaf.game_api.GameAPI(env_name="taps-0001", stop_at_level=1)],
        solver=solver,
        job_dir=job_dir,
    )
    await benchmark.run()
    (run,) = benchmark.game_runs
    return run


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_seal_direct_boot_beats_taps(request: pytest.FixtureRequest) -> None:
    """The bare seal boots its default engine and beats taps level 1."""
    _require_b200(request)
    from seal import SealAgent

    agent = SealAgent.build()
    # The backend matrix against the engine this boot serves, before its game.
    agent.llm_config.verify_backend = True
    run = await _play_taps(agent, None)
    assert run.state == "won", f"expected a level-1 win on taps, got state={run.state!r} ({run.solver_note})"
    assert run.levels_completed >= 1
