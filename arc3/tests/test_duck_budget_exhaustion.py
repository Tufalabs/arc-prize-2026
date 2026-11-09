"""A spent level budget ends the game instead of spinning
(``packages/duck/design.md``, "Run budgets").

Three layers have to agree for the give-up to happen at all, and none of them
shows the whole rule on its own:

1. the turn's ceiling is what is left of the level allowance;
2. a turn truncated on that ceiling has no tool call, so the entry NUDGES —
   and the nudge loop is bounded only by ``max_llm_calls_per_move``, 0
   (unbounded) by default;
3. the manager's synthetic placeholder carries no ``token_usage``, so a turn
   that generates nothing does not advance the level's spend.

A ceiling of zero therefore has no bound of its own: only refusing to hand one
out ends the game. The scripted response below is that state exactly — stops on
``max_tokens``, calls no tool, reports usage equal to the whole allowance — and
the timeout is part of the assertion, since the failure mode is a run that
never terminates rather than one that terminates wrongly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from taaf.game_examples import ExampleGame

from duck import DuckAgent
from platypus import LLMAgentSolver, LLMConfig, RunConfig

ALLOWANCE = 100


def _truncated_no_tool_call() -> ullm.conv.LLMResponse:
    """One response that spends the level exactly and calls nothing — what a
    turn cut off on its ceiling looks like."""
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="World model: thinking about it"),
        stop_reason="max_tokens",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=ALLOWANCE),
    )


async def test_a_spent_level_gives_up_instead_of_nudging_forever(tmp_path: Path) -> None:
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        max_tokens_per_level=(ALLOWANCE,),
        # The defaults that made the live loop unbounded, stated rather than
        # inherited: no per-move call cap, and the cap enforced (no queue tally
        # is published here, and an unknown queue is never a drained one).
        max_llm_calls_per_move=0,
        uncap_when_queue_drained=False,
        multimodal_context="",
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
        games=[ExampleGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )

    # The bound is the assertion: unfixed, this never returns.
    await asyncio.wait_for(benchmark.run(), timeout=120)

    (run,) = benchmark.game_runs
    assert run.state == "gave_up", f"expected a give-up on the budget, got {run.state}"
    generated = sum(record.generated_tokens for record in run.history) + run.final_generated_tokens
    assert generated == pytest.approx(ALLOWANCE, abs=ALLOWANCE), (
        f"the game should stop within about one allowance of the cap, spent {generated}"
    )


async def test_a_spent_game_budget_gives_up_without_a_level_budget(tmp_path: Path) -> None:
    """The parity-run control terminates through the real backend meter."""
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        max_tokens_per_level=None,
        max_tokens_per_game=ALLOWANCE,
        max_llm_calls_per_move=0,
        uncap_when_queue_drained=False,
        multimodal_context="",
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(
                backend=ullm.backends.stub.StubBackend(responses=[_truncated_no_tool_call()], on_exhausted="loop")
            ),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[ExampleGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )

    await asyncio.wait_for(benchmark.run(), timeout=120)

    (run,) = benchmark.game_runs
    assert run.state == "gave_up", f"expected a give-up on the game budget, got {run.state}"
    generated = sum(record.generated_tokens for record in run.history) + run.final_generated_tokens
    assert generated == ALLOWANCE


async def test_a_level_won_mid_turn_does_not_bill_the_next_level(tmp_path: Path) -> None:
    """In live mode ``submit_action(...)`` runs inside the turn, so a level can be won
    before the turn returns — and the enforcement counter the next level starts
    from must be zero regardless.

    The generation is not lost: it ran under the old level's ceiling, so the
    `budget` event records it against that level, and `level_crossed` marks the
    turn that spanned the boundary. What must never happen is the new level
    inheriting it — every level after the first would then start pre-spent by
    its predecessor's last turn, biasing against exactly the deep levels a
    per-level allowance exists to protect.
    """
    from platypus.diagnostics import build_span_tree, game_stream_path, read_events
    from platypus.reference import game_run_dir
    from reference_support import tool_call
    from test_duck_scripted_reference import CornerGame

    # CornerGame: UP,UP clears level 1; DOWN,DOWN clears level 2 and wins. Both
    # land from inside the snippet, so each win happens while its turn is still
    # running — which is the case under test.
    script = [
        tool_call("python", {"code": "submit_action(['UP','UP'])\n"}, "Plan: press UP twice."),
        tool_call("python", {"code": "submit_action(['DOWN','DOWN'])\n"}, "Plan: press DOWN twice."),
    ]
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        max_tokens_per_level=(50_000,),
        multimodal_context="",
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(
                backend=ullm.backends.stub.StubBackend(responses=script, on_exhausted="repeat_last")
            ),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[CornerGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )
    await asyncio.wait_for(benchmark.run(), timeout=180)
    (run,) = benchmark.game_runs
    assert run.state == "won", f"the scripted plays should win the game, got {run.state}"

    def spans(node: Any) -> list[Any]:
        return [node, *(d for child in node.children for d in spans(child))]

    (root,) = build_span_tree(read_events(game_stream_path(game_run_dir(tmp_path, run.run_id))))
    budgets = [e["values"] for span in spans(root) for e in span.events if e["kind"] == "budget"]
    assert budgets, "every analyzer turn reports its budget event"

    crossing = [b for b in budgets if b["level_crossed"]]
    assert crossing, "the turn that won a level should report itself as crossing"
    assert all(b["tokens_added"] > 0 for b in crossing), (
        "a crossing turn's generation is recorded against the level it ran under, not dropped"
    )

    # The decisive one: the first charge on each later level must be that
    # level's whole spend — nothing carried over from the level before it.
    first_on_level: dict[int, dict[str, Any]] = {}
    for event in budgets:
        first_on_level.setdefault(event["level"], event)
    for level, event in first_on_level.items():
        if level == 0:
            continue
        assert event["tokens_used_this_level"] == event["tokens_added"], (
            f"level {level} started pre-spent by the previous level's winning turn: {event}"
        )


async def test_the_crossing_budget_event_describes_one_level(tmp_path: Path) -> None:
    """The event a level-crossing turn emits describes one level, in all three
    of its fields: the level the turn started on, that level's cumulative spend
    including this turn, and that level's own allowance — all taken from the
    turn's start, while the live enforcement counter stays at the new level's
    zero.

    Both halves need their own condition to be observable, which is why this
    scenario is shaped the way it is: unequal caps are the only way a wrong
    allowance shows, and a charged turn before the win is the only way a wrong
    counter shows. A record whose fields disagree cannot be corrected after the
    fact, because nothing in it says which field to believe.
    """
    from platypus.diagnostics import build_span_tree, game_stream_path, read_events
    from platypus.reference import game_run_dir
    from reference_support import tool_call
    from test_duck_scripted_reference import CornerGame

    level_0_cap, level_1_cap = 40_000, 60_000
    script = [
        # Charged normally on level 0: no tool call, so the entry nudges and
        # this block closes as its own turn.
        ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="World model: two lanes. Plan: UP twice.")),
        tool_call("python", {"code": "submit_action(['UP','UP'])\n"}),  # wins level 0 mid-turn
        tool_call("python", {"code": "submit_action(['DOWN','DOWN'])\n"}),  # wins the game
    ]
    agent = DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        max_tokens_per_level=(level_0_cap, level_1_cap),
        multimodal_context="",
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(
                backend=ullm.backends.stub.StubBackend(responses=script, on_exhausted="repeat_last")
            ),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
        ),
        run_config=RunConfig(rng_seed=1, process_mode="inprocess"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[CornerGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )
    await asyncio.wait_for(benchmark.run(), timeout=180)
    (run,) = benchmark.game_runs

    def spans(node: Any) -> list[Any]:
        return [node, *(d for child in node.children for d in spans(child))]

    (root,) = build_span_tree(read_events(game_stream_path(game_run_dir(tmp_path, run.run_id))))
    budgets = [e["values"] for span in spans(root) for e in span.events if e["kind"] == "budget"]
    caps = {0: level_0_cap, 1: level_1_cap}

    charged_on_0 = [b for b in budgets if b["level"] == 0 and not b["level_crossed"]]
    assert charged_on_0, "the pre-win turn should be charged to level 0"
    spent_on_0 = max(b["tokens_used_this_level"] for b in charged_on_0)
    assert spent_on_0 > 0

    (crossing,) = [b for b in budgets if b["level_crossed"]]
    assert crossing["max_tokens_per_level"] == caps[crossing["level"]], (
        f"event names level {crossing['level']} but reports the allowance of another: {crossing}"
    )
    assert crossing["tokens_added"] > 0, "the winning response ran under this level's cap and is its spend"
    assert crossing["tokens_used_this_level"] == spent_on_0 + crossing["tokens_added"], (
        f"level {crossing['level']} had spent {spent_on_0} and its crossing turn added "
        f"{crossing['tokens_added']}, but the event reports {crossing['tokens_used_this_level']}: {crossing}"
    )
