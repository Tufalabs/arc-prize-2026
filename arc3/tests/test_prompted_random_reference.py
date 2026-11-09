"""The PromptedRandomAgent integration reference (I13) — the framework
put to USE, not under test.

``test_prompted_random_agent_matches_reference`` replays the committed
Haiku-generated archives in the cheap tier on every run: the agent code
in THIS worktree must reproduce the recorded behavior bit-exact. When the
agent's behavior changes on purpose, regenerate with

    ULLM_API_TESTS=1 PLATYPUS_REGENERATE_REFERENCES=1 \
        make test-api TEST_ARGS='-k regenerate_prompted_reference'

(spends a few Haiku calls) and commit the refreshed archives in
``tests/references/``. The extra opt-in keeps `make test-api` read-only:
regenerating re-baselines what "correct" means, so it must never happen
as a side effect of running the test tiers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import taaf.benchmark
import ullm.backend
from taaf.game_examples import ExampleGame

from platypus import LLMAgentSolver, RunConfig
from reference_agents import PromptedRandomAgent
from reference_support import ReferenceCase, ReferenceRun, regenerate_case, replay_case, require_regeneration

# The generation-time agent configuration. The replay test rebuilds the
# agent from THESE constants + current code — the reference is a claim
# about behavior, so a deliberate behavior change means regenerating.
N_GAMES = 2
MAX_MOVES = 4
MAX_LLM_CALLS = 2
RNG_SEED = 20260722


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    agent = PromptedRandomAgent(
        max_moves=MAX_MOVES,
        max_llm_calls=MAX_LLM_CALLS,
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
    )
    return taaf.benchmark.Benchmark(
        games=[ExampleGame(label=f"prompted_ref{i}") for i in range(N_GAMES)],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=run.job_dir,
    )


CASE = ReferenceCase(
    name="prompted_random_haiku",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=100)},
)


async def test_prompted_random_agent_matches_reference(tmp_path: Path) -> None:
    await replay_case(CASE, tmp_path)


@pytest.mark.api
async def test_regenerate_prompted_reference(tmp_path: Path) -> None:
    """Not a test of behavior — the sanctioned way to REGENERATE the
    committed reference after a deliberate agent change (I13: the fresh
    reference is validated by immediate replay before it lands)."""
    key = require_regeneration(spends_api=True)
    import ullm.backends.anthropic

    backend = ullm.backends.anthropic.Anthropic(api_key=key, model="claude-haiku-4-5", set_adaptive_thinking=False)
    await regenerate_case(CASE, backend=backend, tmp_path=tmp_path)
