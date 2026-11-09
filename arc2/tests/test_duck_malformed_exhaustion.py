"""Duck's rejected-reply budget propagates exhaustion; accepted replies keep
flowing through the ordinary game budget."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from duck import DuckAgent

import taaf.benchmark
from platypus import LLMAgentSolver, LLMConfig, RunConfig
from taaf.game_examples import flip_game


class _MalformedStub(ullm.backends.stub.StubBackend):
    """Emit empty model replies so the real response manager decides their fate."""

    draws: ClassVar[int] = 0

    def _check_constraints(self) -> None:
        """Allow empty script entries to reach the runtime rejection policy."""
        ullm.backend.Backend._check_constraints(self)

    def _select_response(self) -> ullm.conv.LLMResponse:
        """Record each model draw, including replies the manager rejects."""
        type(self).draws += 1
        return super()._select_response()


# A rejected reply is re-sampled ``malformed_retries`` times, then dropped and
# the game nudged once; the second exhaustion crashes it. So a game that only
# ever replies empty makes two rounds of 1 + retries draws.
@pytest.mark.parametrize(
    "retries,accept,expected_state,expected_draws",
    [(None, False, "crashed", 8), (2, False, "crashed", 6), (0, True, "gave_up", 1)],
)
async def test_rejected_reply_exhaustion_and_accepted_reply_continuation(
    tmp_path: Path,
    retries: int | None,
    accept: bool,
    expected_state: str,
    expected_draws: int,
) -> None:
    _MalformedStub.draws = 0
    backend = _MalformedStub(
        responses=[
            ullm.conv.LLMResponse(
                final_content=ullm.conv.EmptyContent(),
                stop_reason="stop",
                token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=100),
            )
        ],
        on_exhausted="loop",
        accept_malformed_output=accept,
    )
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=4096,
        max_tokens_per_game=100,
        images="off",
        show_changed_answers=False,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
    )
    if retries is not None:
        agent.malformed_retries = retries
    benchmark = taaf.benchmark.Benchmark(
        games=[flip_game("malformed")], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )
    await asyncio.wait_for(benchmark.run(), timeout=120)
    (run,) = benchmark.game_runs
    assert run.state == expected_state
    assert _MalformedStub.draws == expected_draws
    if accept:
        assert sum(record.generated_tokens for record in run.history) + run.final_generated_tokens == 100
