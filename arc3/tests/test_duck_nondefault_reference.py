"""The most-values-different DuckAgent integration reference (I13):
taps-0001 levels 1-2 played by sonnet-5, every field it names set away from
what the duck ships. It exists for the paths a default-configuration run
never reaches — ``multimodal_context="level_start"`` is the only place the
once-a-level grid image is exercised end to end, and the image is what makes
this the live recording of the numeric arm's palette legend;
``compaction_strategy="prune"`` is the eviction the summary replaced — and
``exact_token_counts=False`` (inert: the decision path runs on
engine-billed usage plus a conservative bound in every configuration)
proves the flag stays accepted.

No field here is set to its default, with the two exemptions named in
``_FORK_ONLY_FIELDS``: a value that agrees with the shipped one would be a
second copy of that default, silently repurposed the next time it moves. Where a value needs a reason it carries one; none of them
records what the default is, because ``duck/agent.py`` is where that lives.

Complements the pathfinder reference, which plays its own game bare.

``test_duck_nondefault_matches_reference`` replays the committed archive
in the cheap tier on every run. When duck behavior changes on purpose,
regenerate with

    ULLM_API_TESTS=1 PLATYPUS_REGENERATE_REFERENCES=1 \
        make test-api TEST_ARGS='-k regenerate_duck_nondefault'

(spends real sonnet-5 calls) and commit the refreshed archive in
``tests/references/``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest
import taaf.benchmark
import ullm.backend
from taaf.game_api import GameAPI

from duck import DuckAgent
from duck.prompts import DuckPrompts
from platypus import LLMAgentSolver, LLMConfig, RunConfig
from reference_support import ReferenceCase, ReferenceRun, regenerate_case, replay_case, require_regeneration

GAME_ID = "taps-0001"
STOP_AT_LEVEL = 2  # levels 1-2 bound the generation cost
RNG_SEED = 20260723


def _build(run: ReferenceRun) -> taaf.benchmark.Benchmark:
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=6000,
        # The engine-side knobs are pinned to the state a cloud backend can
        # serve -- no blank, nothing served as the model's own opening --
        # which is why _FORK_ONLY_FIELDS exempts them: sonnet-5 is neither
        # the vllm fork nor a backend that can serve text after the
        # generation prompt, so the values that exercise them are not
        # recordable here at all.
        zero_recurrent=False,
        inject_text_on_first_prompt="",
        inject_text_on_later_prompts="",
        # Wide enough to hold one prompt: under the chars/3 estimator this case
        # runs, the palette legend the numeric arm attaches at each level start
        # weighs about 12k on its own, and a prompt that cannot fit raises
        # rather than degrading.
        context_window_tokens=32768,
        compaction_strategy="prune",  # eviction, which the two knobs around it steer
        compact_to_tokens=3072,  # far under the budget, so the sawtooth runs deep
        journal=True,  # and the journal beside it
        # The bare user prompt, no brevity ask, and neither overview line. The
        # summary's own switches (`thorough_summary`, the listing) are inert
        # under prune and left at what ships.
        prompts=DuckPrompts(brevity="off", winnable_line=False, retest_actions_line=False),
        max_llm_calls_per_move=3,
        auto_correct_tool_arguments=False,
        python_timeout_s=15,
        tool_output_tokens=512,
        evict_turns=1,
        # The grid as an image on each level's first prompt only. An image has
        # to travel for the palette legend below; the text-only board is
        # pinned by the scripted references and the `no_image` prompt golden.
        multimodal_context="level_start",
        multimodal_upscale=2,  # and at a rendering scale of its own
        profile="probe",
        max_moves=60,  # also bounds the generation run
        # A one-game run drains its queue at once, so the uncap lifts the
        # allowance here: the tuple is pinned but inert, and what this
        # reference covers is the LIFTED path (no per-turn ceiling on any
        # request). The allowance itself is unit-tested.
        max_tokens_per_level=(200_000, 300_000),
        uncap_when_queue_drained=True,
        exact_token_counts=False,  # inert flag, pinned to prove it stays accepted
        persistence="round",  # a fresh kernel process per call
        # Colors as indices, and — because an image travels here — the palette
        # legend that explains them. The scripted numeric reference steers the
        # same arm through a fixed script; this is a model meeting it.
        frame_api="numeric",
        color_name_legend=False,  # the index legend, not the colour names
        completion_cutoff_notice=True,  # the continuation message names the cutoff
        expose_undo=False,  # UNDO withheld
        wide_sandbox=False,  # the allowlist without the seven wider modules
        numpy_scalar_repr="np2",  # NumPy 2's own scalar repr, constructor wrapper and all
        # The file suite at its full five. `file_tools="none"` is the other
        # non-default state and cannot be covered in the same case; the absent
        # surface is pinned by the `no_file_tools` prompt golden, while the
        # WIDENED one is what wants a live model behind it -- the prompt
        # counts and names the tools it is given.
        expose_tool_search_files=True,
        expose_tool_undo_edit=True,
        reasoning_effort="low",  # serialized on every request; inert on 3.6, live on 3.8
        llm_config=run.llm_config,
        run_config=RunConfig(rng_seed=RNG_SEED, rng_seed_per_game=run.seeds),
    )
    return taaf.benchmark.Benchmark(
        games=[GameAPI(env_name=GAME_ID, stop_at_level=STOP_AT_LEVEL)],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=run.job_dir,
    )


#: The fields ``_build`` states, read off its own source so the guard below
#: cannot drift from the call it guards. ``llm_config`` and ``run_config``
#: are the harness's configuration, not the duck's, and have no shipped
#: default to differ from.
_HARNESS_FIELDS = {"llm_config", "run_config"}

#: Fields whose non-default value only exists on the in-process vLLM engine:
#: the blank needs the fork's sampling parameter, and the register opening
#: needs a backend that can serve ids after the generation prompt. A cloud
#: recording can pin them only AT the default, so the guard below cannot ask
#: them to differ from it. They are named here rather than dropped from
#: ``_build`` because leaving them unstated would let a later default change
#: alter this recording silently.
_FORK_ONLY_FIELDS = {"zero_recurrent", "inject_text_on_first_prompt", "inject_text_on_later_prompts"}


def _configured_fields() -> list[str]:
    """Every DuckAgent field ``_build`` names."""
    tree = ast.parse(Path(__file__).read_text())
    build = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_build")
    call = next(n for n in ast.walk(build) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "DuckAgent")
    return [kw.arg for kw in call.keywords if kw.arg and kw.arg not in _HARNESS_FIELDS | _FORK_ONLY_FIELDS]


CASE = ReferenceCase(
    name="duck_nondefault_sonnet5",
    build=_build,
    # Anthropic requires max_tokens; temperature 1.0 with (adaptive) thinking.
    profiles={"probe": ullm.backend.GenericSamplingParameters(temperature=1.0, max_tokens=None)},
)


def test_every_configured_field_differs_from_the_shipped_default() -> None:
    """The case earns its name, and keeps earning it.

    A default that moves onto a value pinned here costs a paid recording and
    buys nothing: the field stops being exercised and the replay still passes,
    because the recorded bytes are what the two agree on. Nothing else in the
    suite can notice that, so it is asserted rather than annotated."""
    run = ReferenceRun(llm_config=LLMConfig(), job_dir=None, seeds={})
    configured = cast(LLMAgentSolver, _build(run).solver).agent
    shipped = DuckAgent()
    names = _configured_fields()
    assert names, "the guard found no configured fields; _build no longer looks the way it parses"
    same = [name for name in names if getattr(configured, name) == getattr(shipped, name)]
    assert not same, f"set to the shipped default, so no longer covered here: {same}"


async def test_duck_nondefault_matches_reference(tmp_path: Path) -> None:
    await replay_case(CASE, tmp_path)


@pytest.mark.api
async def test_regenerate_duck_nondefault_reference(tmp_path: Path) -> None:
    """Not a test of behavior — the sanctioned way to REGENERATE the
    committed reference after a deliberate duck change (I13: the fresh
    reference is validated by immediate replay before it lands)."""
    key = require_regeneration(spends_api=True)
    import ullm.backends.anthropic

    backend = ullm.backends.anthropic.Anthropic(api_key=key, model="claude-sonnet-5")
    await regenerate_case(CASE, backend=backend, tmp_path=tmp_path)
