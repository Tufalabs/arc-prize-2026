"""The DuckAgent integration reference (I13): pathfinder-0001
(1-indexed levels 2-3) played by sonnet-5, committed as a replayable
reference. Pathfinder exercises long planned sequences: whole arrow
paths executed from one decision inside a single snippet.

``test_duck_pathfinder_matches_reference`` replays the committed archive in the
cheap tier on every run: the duck code in THIS worktree must reproduce the
recorded behavior bit-exact (prompts, eviction, batch mechanics, kernel
environment — any behavioral drift shows up as a replay mismatch diff).
When duck behavior changes on purpose, regenerate with

    ULLM_API_TESTS=1 PLATYPUS_REGENERATE_REFERENCES=1 \
        make test-api TEST_ARGS='-k regenerate_duck_pathfinder'

(spends real sonnet-5 calls) and commit the refreshed archive in
``tests/references/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import taaf.benchmark
import ullm.backend
from taaf.game_api import GameAPI

from duck import DuckAgent
from platypus import LLMAgentSolver, RunConfig
from reference_support import ReferenceCase, ReferenceRun, regenerate_case, replay_case, require_regeneration

# The generation-time configuration. The replay test rebuilds the agent
# from THESE constants + current duck code — a deliberate behavior change
# means regenerating the reference. The agent is built bare: the shipped
# defaults ARE the configuration under test, so a default flip lands here
# as a replay mismatch that forces regeneration rather than as a silent
# repurposing. The sampling profile is provider-forced (Anthropic requires
# max_tokens; temperature 1.0 with thinking); its cap is the shipped reply
# cap by reference, so the reply reserve — and with it the prompt budget —
# runs the shipped coordinates. The knob-exercising counterpart is the
# most-values-different reference (test_duck_nondefault_reference.py).
GAME_ID = "pathfinder-0001"
START_AT_LEVEL = 1  # 0-indexed half-open window = 1-indexed levels 2-3
STOP_AT_LEVEL = 3
RNG_SEED = 20260722


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=32768,
        # sonnet-5 is neither the vllm fork nor a backend that can serve
        # text after the generation prompt, so both engine-side defaults are
        # turned off here; each is refused on this backend.
        zero_recurrent=False,
        inject_text_on_first_prompt="",
        inject_text_on_later_prompts="",
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
    )
    return taaf.benchmark.Benchmark(
        games=[GameAPI(env_name=GAME_ID, start_at_level=START_AT_LEVEL, stop_at_level=STOP_AT_LEVEL)],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=run.job_dir,
    )


CASE = ReferenceCase(
    name="duck_pathfinder_sonnet5",
    build=_build,
    # temperature 1.0: Anthropic requires it when (adaptive) thinking is on.
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=1.0, max_tokens=None)},
)


async def test_duck_pathfinder_matches_reference(tmp_path: Path) -> None:
    await replay_case(CASE, tmp_path)


@pytest.mark.api
async def test_regenerate_duck_pathfinder_reference(tmp_path: Path) -> None:
    """Not a test of behavior — the sanctioned way to REGENERATE the
    committed reference after a deliberate duck change (I13: the fresh
    reference is validated by immediate replay before it lands)."""
    key = require_regeneration(spends_api=True)
    import ullm.backends.anthropic

    backend = ullm.backends.anthropic.Anthropic(api_key=key, model="claude-sonnet-5")
    await regenerate_case(CASE, backend=backend, tmp_path=tmp_path)
