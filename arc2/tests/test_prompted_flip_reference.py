"""The ``PromptedFlipAgent`` integration reference — the framework put to
USE, not under test.

``test_prompted_flip_agent_matches_reference`` replays the committed
archive in the cheap tier on every run: the agent code in THIS worktree
must reproduce the recorded behaviour bit-exact — every prompt, every
answer, every token count, and the viewer pages rendered from the run.
When the behaviour changes on purpose, regenerate with

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k regenerate_prompted_flip'

and commit the refreshed archive in ``tests/references/``. Free: the stub
answers the sums, so regeneration is deterministic and spends nothing. The
opt-in is still required — regenerating re-baselines what "correct" means,
and that must never happen as a side effect of running a tier.
"""

from __future__ import annotations

from pathlib import Path

import ullm.backend
import ullm.backends.stub

import taaf.benchmark
from platypus import LLMAgentSolver, RunConfig
from platypus.reference_agents import PromptedFlipAgent
from reference_support import ReferenceCase, ReferenceRun, regenerate_case, replay_case, require_regeneration
from taaf.game_examples import flip_game

# The generation-time agent configuration. The replay test rebuilds the
# agent from THESE constants + current code — the reference is a claim
# about behaviour, so a deliberate behaviour change means regenerating.
N_ANSWERS = 3
MAX_LLM_CALLS = 2
RNG_SEED = 20260822


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The one benchmark both capture and replay assemble."""
    agent = PromptedFlipAgent(
        answers=("flip",) * N_ANSWERS,
        max_llm_calls=MAX_LLM_CALLS,
        attach_grid_image=True,  # the multimodal path is part of what is pinned
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds, process_mode="inprocess"),
    )
    # One game: the reference carries a viewer slice, and the index page of
    # a multi-game run names sibling run ids no replay can reproduce.
    return taaf.benchmark.Benchmark(
        games=[flip_game("prompted_ref")], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir
    )


CASE = ReferenceCase(
    name="prompted_flip_stub",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
)


async def test_prompted_flip_agent_matches_reference(tmp_path: Path) -> None:
    await replay_case(CASE, tmp_path)


async def test_regenerate_prompted_flip_reference(tmp_path: Path) -> None:
    """Not a test of behaviour — the sanctioned way to REGENERATE the
    committed reference after a deliberate change (the fresh reference is
    validated by immediate replay before it lands)."""
    require_regeneration()
    await regenerate_case(CASE, backend=ullm.backends.stub.ArithmeticStubBackend(), tmp_path=tmp_path)
