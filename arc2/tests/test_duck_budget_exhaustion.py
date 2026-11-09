"""A spent budget ends the game instead of spinning.

Three layers have to agree for the give-up to happen at all, and none of them
shows the whole rule on its own:

1. the turn's ceiling is what is left of the game allowance;
2. a turn truncated on that ceiling has no tool call, so the entry NUDGES —
   and the nudge loop is bounded only by ``max_llm_calls_per_move``, 0
   (unbounded) by default;
3. the manager's synthetic placeholder carries no ``token_usage``, so a turn
   that generates nothing does not advance the spend.

A ceiling of zero therefore has no bound of its own: only refusing to hand one
out ends the game. The scripted response below is that state exactly — stops on
``max_tokens``, calls no tool, reports usage equal to the whole allowance — and
the timeout is part of the assertion, since the failure mode is a run that
never terminates rather than one that terminates wrongly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from duck import DuckAgent

import taaf.benchmark
from platypus import LLMAgentSolver, LLMConfig, RunConfig
from taaf.game_examples import flip_game

ALLOWANCE = 100


def _truncated_no_tool_call() -> ullm.conv.LLMResponse:
    """One response that spends the allowance exactly and calls nothing — what
    a turn cut off on its ceiling looks like."""
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="Thinking about the rule"),
        stop_reason="max_tokens",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=ALLOWANCE),
    )


async def test_a_spent_game_budget_gives_up_instead_of_nudging_forever(tmp_path: Path) -> None:
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=None,
        max_tokens_per_game=ALLOWANCE,
        # The default that made the live loop unbounded, stated rather than
        # inherited: no per-entry call cap.
        max_llm_calls_per_move=0,
        llm_config=LLMConfig(
            # "loop": the model keeps answering, so nothing but the budget can
            # end this game. Exhausting a script would end it for the wrong
            # reason, and the test would pass while broken.
            backend_factory=ullm.factory.PrebuiltBackend(
                backend=ullm.backends.stub.StubBackend(responses=[_truncated_no_tool_call()], on_exhausted="loop")
            ),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[flip_game("budget")], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )

    # The bound is the assertion: unfixed, this never returns.
    await asyncio.wait_for(benchmark.run(), timeout=120)

    (run,) = benchmark.game_runs
    assert run.state == "gave_up", f"expected a give-up on the budget, got {run.state}"
    generated = sum(record.generated_tokens for record in run.history) + run.final_generated_tokens
    assert generated == ALLOWANCE
