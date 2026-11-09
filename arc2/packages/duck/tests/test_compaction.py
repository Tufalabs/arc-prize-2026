"""``compact_to_tokens``: trim to a LOWER target than the budget that fired it.

Without hysteresis a trim stops the moment it is under budget, leaving the
prompt at the edge — so the next round crosses it again and re-trims, busting
the KV prefix every round. With it the context runs as a sawtooth: fire at
the budget, fall back to the target, and pay for a trim once per many rounds.

Companion to test_context_pruning.py, which covers the in-turn pruning this
target drives.
"""

from __future__ import annotations

import pytest
import ullm.conv
from duck import DuckAgent, prompts
from duck_fixtures import TARGET, WINDOW, agent, many_blocks, single_block, turn

# --- the target ---------------------------------------------------------------


def test_target_outside_the_valid_band_raises() -> None:
    """Positive and below the budget is the whole rule: at or above the budget
    a trim reaches nothing (the trigger fires there), and a non-positive target
    has no sawtooth at all. Both raise; anything in between — including within
    a reply reserve of the budget — is a permitted preference, not an error."""
    target = agent()
    for invalid in (0, WINDOW * 4, target.context_budget_tokens()):
        target.compact_to_tokens = invalid
        with pytest.raises(ValueError, match="compact_to_tokens"):
            target.compact_target_tokens()


def test_shallow_target_within_a_reserve_of_the_budget_is_allowed() -> None:
    """Upstream's best swept arms sit here: shallower than budget - reserve. A
    worst-case reply can re-cross the budget within a round, but that is a
    slower configuration, not a broken one."""
    budget = agent().context_budget_tokens()
    assert agent(compact_to_tokens=budget - 512).compact_target_tokens() == budget - 512
    assert agent(compact_to_tokens=budget - 1).compact_target_tokens() == budget - 1


def test_target_is_used_when_lower() -> None:
    assert agent(compact_to_tokens=2048).compact_target_tokens() == 2048


# Building the column's engine config imports the GPU stack, which probes CUDA
# and warns (not errors) on a GPU-less or old-driver box; scoped to torch's
# probing so everything else in these constructions still errors.
@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_the_default_columns_shipped_defaults_cohere() -> None:
    """The shipped numbers must stay mutually coherent, not just individually
    declared: computing a target runs the target-vs-budget check and settling
    runs the window-vs-max_model_len check, so an incoherent edit to any one
    default fails here."""
    built = DuckAgent.build()
    # The column summarizes, and a summary's reset ceiling is the budget
    # itself; prune's target, still a field, has to sit under that budget.
    assert built.compaction_strategy == "summarize"
    assert built.compact_target_tokens() == built.context_budget_tokens() > 0
    assert 0 < built._compact_target_tokens() < built._context_budget_tokens()
    built.prepare_for_backend()


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_explicit_arguments_beat_the_column() -> None:
    """Every value the column supplies must stay overridable, or a field the
    agent declares is one the caller cannot actually reach."""
    built = DuckAgent.build(
        compaction_strategy="prune",
        context_window_tokens=32768,
        reply_cap_tokens=4096,
        compact_to_tokens=2048,
    )
    assert built.compaction_strategy == "prune"
    assert built.compact_to_tokens == 2048
    assert built.compact_target_tokens() == 2048


# --- the sawtooth -------------------------------------------------------------


async def test_compaction_undershoots_the_budget() -> None:
    """The point of hysteresis: land at the target, well UNDER the budget,
    rather than at the budget's edge."""
    compacting = agent()
    single_block(compacting)
    await compacting._trim_messages_for_context(compacting.messages, enforce=True)
    after = compacting._decision_prompt_tokens(compacting.messages)
    assert after <= compacting.compact_target_tokens(), (after, TARGET)
    assert compacting.compact_target_tokens() < compacting.context_budget_tokens()


async def test_an_under_budget_prompt_is_left_alone() -> None:
    """The TRIGGER stays the budget: a prompt already under it is not compacted
    down to the target for no reason."""
    idle = agent()
    single_block(idle, rounds=1)
    before = idle._decision_prompt_tokens(idle.messages)
    assert before <= idle.context_budget_tokens(), "fixture must start under budget"
    await idle._trim_messages_for_context(idle.messages, enforce=True)
    assert idle._decision_prompt_tokens(idle.messages) == before


async def test_a_trim_whose_eviction_gets_under_budget_still_reaches_the_target() -> None:
    """The case the single-block fixtures miss. Eviction lands the prompt
    between target and budget; pruning must finish the trim. It only does if it
    shares eviction's budget trigger instead of re-testing the budget itself —
    re-testing bails in exactly this band, which is where nearly every real
    trim ends up."""
    mixed = agent()
    many_blocks(mixed)
    before = mixed._decision_prompt_tokens(mixed.messages)
    assert before > mixed.context_budget_tokens(), "fixture must trigger a trim"

    await mixed._trim_messages_for_context(mixed.messages, enforce=True)

    after = mixed._decision_prompt_tokens(mixed.messages)
    assert after <= mixed.compact_target_tokens(), (after, mixed.compact_target_tokens())


async def test_compaction_still_keeps_rounds_whole() -> None:
    """Undershooting drops more rounds; they must still leave in pairs."""
    deep = agent()
    single_block(deep)
    await deep._trim_messages_for_context(deep.messages, enforce=True)
    tail = deep.messages[-1]
    assert isinstance(tail, ullm.conv.ResponseTurn)
    if tail.elements:
        assert isinstance(tail.elements[0], ullm.conv.LLMResponse)
        assert len(tail.elements) % 2 == 0, [type(e).__name__ for e in tail.elements]


# --- what the bound counts ----------------------------------------------------


def test_a_longer_system_prompt_moves_the_bound() -> None:
    """The system prompt is in the pseudo-request the bound weighs — it is
    sent on every call and cannot be evicted — so the prefix is measured,
    not a constant."""
    plain = agent()
    wordy = agent(prompts=prompts.DuckPrompts(system_prompt_opening="x" * 3000))
    assert wordy._bounded_prompt_tokens([]) > plain._bounded_prompt_tokens([])


# --- the brevity line ---------------------------------------------------------


def test_brevity_line_rides_the_system_prompt_once() -> None:
    with_line = agent(prompts=prompts.DuckPrompts(brevity_line=True))
    assert with_line._system_prompt_text().count(prompts.DuckPrompts().user_prompt_brevity_line) == 1


def test_brevity_line_off_removes_it() -> None:
    without = agent(prompts=prompts.DuckPrompts(brevity_line=False))
    assert prompts.DuckPrompts().user_prompt_brevity_line not in without._system_prompt_text()


def test_turn_fixture_builds_whole_rounds() -> None:
    """The fixtures' own invariant: every case above reasons about rounds
    leaving in pairs, which is only meaningful if they arrive in pairs."""
    built = turn(3)
    assert len(built.elements) == 6
    assert all(isinstance(e, ullm.conv.LLMResponse) for e in built.elements[::2])
    assert all(isinstance(e, ullm.conv.ToolResult) for e in built.elements[1::2])
