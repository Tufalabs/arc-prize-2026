"""The duck's integration reference — the framework put to USE.

``test_duck_matches_reference`` replays the committed archives in the cheap
tier on every run: the duck's code in THIS worktree must reproduce the
recorded behaviour bit-exact — every prompt, every image, every kernel
result, every answer, every token count. When the behaviour changes on
purpose, regenerate with

    ANTHROPIC_API_KEY=... PLATYPUS_REGENERATE_REFERENCES=1 \
        make test-api TEST_ARGS='-k regenerate_duck_reference'

(spends a few Haiku calls) and commit the refreshed archives in
``tests/references/``. The extra opt-in keeps the tiers read-only:
regenerating re-baselines what "correct" means, so it must never happen as a
side effect of running a tier.

**One game, and it is the three-question one.** Capture requires a
``job_dir`` — that is where the solver writes the per-game files before they
are collected — and a reference built with one carries deterministic viewer
HTML, whose index page names sibling run ids a replay cannot reproduce. So a
case is one game, and the one worth spending it on is the task with the most
questions: that is where the answer path does most of its work, and it is
covered nowhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import ullm.backend
from duck import DuckAgent

import taaf.benchmark
import taaf.datasets
import taaf.game
from platypus import LLMAgentSolver, RunConfig
from reference_support import ReferenceCase, ReferenceRun, regenerate_case, replay_case, require_regeneration

# The generation-time configuration. The replay rebuilds the agent from THESE
# constants plus current code — the reference is a claim about behaviour, so a
# deliberate behaviour change means regenerating.
GAME_ID = "3cd86f4f"  # three questions — the most the curated set offers
RNG_SEED = 20260822
MAX_TOKENS_PER_GAME = 8_000
REFERENCE_MODEL = "claude-haiku-4-5"


def _games() -> list[taaf.game.Game]:
    by_id = {spec.game_id: spec for spec in taaf.datasets.load("arc2-train")}
    return taaf.game.games_from_specs([by_id[GAME_ID]])


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    """The one benchmark both capture and replay assemble."""
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=4096,
        max_tokens_per_game=MAX_TOKENS_PER_GAME,
        # Pinned to the OFF value rather than away from the shipped default:
        # text after the generation prompt is served by the in-process vLLM
        # backend alone, so this column cannot record the register opening at
        # all — a live regeneration against the hosted model is refused, and a
        # replay would only pin what the recording happened to hold.
        inject_text_on_first_prompt="",
        inject_text_on_later_prompts="",
        llm_config=run.llm_config,
        # inprocess: the reference is about the duck's behaviour, and a
        # spawned child adds a process boundary that buys nothing here.
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds, process_mode="inprocess"),
    )
    return taaf.benchmark.Benchmark(games=_games(), solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=run.job_dir)


CASE = ReferenceCase(
    name="duck_haiku",
    build=_build,
    profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)},
)


async def test_duck_matches_reference(tmp_path: Path) -> None:
    await replay_case(CASE, tmp_path)


@pytest.mark.api
async def test_regenerate_duck_reference(tmp_path: Path) -> None:
    """Not a test of behaviour — the sanctioned way to REGENERATE the
    committed reference after a deliberate change. The fresh reference is
    validated by immediate replay before it lands."""
    key = require_regeneration(spends_api=True)
    import ullm.backends.anthropic

    backend = ullm.backends.anthropic.Anthropic(api_key=key, model=REFERENCE_MODEL, set_adaptive_thinking=False)
    await regenerate_case(CASE, backend=backend, tmp_path=tmp_path)


def test_the_reference_covers_a_multi_question_task() -> None:
    """The reason this case exists rather than reusing the flip one, and the
    reason it is this game. If the data ever changed under it, the reference
    would go on replaying green while pinning a path it no longer covers."""
    by_id = {spec.game_id: spec for spec in taaf.datasets.load("arc2-train")}
    assert sum(1 for p in by_id[GAME_ID].pairs if not p.given) > 1


def test_the_api_key_is_only_needed_to_regenerate() -> None:
    """The replay above must run in the cheap tier on every machine, so the
    committed archive is the only thing it may depend on."""
    assert not os.environ.get("PLATYPUS_REGENERATE_REFERENCES"), "regeneration must be opt-in, never ambient"
