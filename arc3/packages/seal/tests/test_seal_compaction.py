"""``compact_to_tokens``: trim to a LOWER target than the budget that fired it.

Without hysteresis a trim stops the moment it is under budget, leaving the
prompt at the edge — so the next round crosses it again and re-trims, busting
the KV prefix every round. With it the context runs as a sawtooth: fire at the
budget, fall back to the target, and pay for a trim once per many rounds.

Companion to test_context_pruning.py, which covers the in-turn pruning this
target drives.
"""

from __future__ import annotations

from typing import Any

import pytest
import ullm.conv

from seal import SealAgent, prompts
from seal.state import MemoryEntry

WINDOW = 25856  # budget = 25856 - 512 (reply reserve) - 512 (margin) = 24832
# A target the multi-block fixture can actually reach: above the ~17.5k
# decision-token floor an emptied turn leaves (system prompt + the user
# prompt itself, text charged its character count) and below where block
# eviction alone stops (~23k). Ask for less than the floor and every trim
# "fails" for a reason that has nothing to do with the trigger.
REACHABLE_TARGET = 20876


def _turn(rounds: int) -> ullm.conv.ResponseTurn:
    elements: list[Any] = []
    for i in range(rounds):
        elements.append(ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text=f"thinking {i} " * 40)))
        elements.append(
            ullm.conv.ToolResult(
                call_id=f"c{i}", tool_name="python", content=ullm.conv.StringContent(text=f"result {i} " * 40)
            )
        )
    return ullm.conv.ResponseTurn(elements=elements, status="intermediate")


def _agent(**kwargs: Any) -> SealAgent:
    # The windows sit just above the system prompt, sized for the prompt
    # without the world-model section.
    kwargs.setdefault("reply_cap_tokens", None)
    kwargs.setdefault("compaction_strategy", "prune")
    kwargs.setdefault("journal", True)
    return SealAgent(context_window_tokens=WINDOW, use_world_model="none", **kwargs)


def _single_block(agent: SealAgent, rounds: int = 12) -> None:
    """ONE user-led block: the shape block eviction cannot touch."""
    agent.messages = [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")), _turn(rounds)]


def _many_blocks(agent: SealAgent, blocks: int = 4, rounds: int = 6) -> None:
    """SEVERAL user-led blocks, the ordinary shape: eviction has room to work,
    runs out of blocks before it runs out of tokens, and leaves the prompt
    parked between the target and the budget."""
    messages: list[Any] = []
    for b in range(blocks):
        messages.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=f"move {b}")))
        messages.append(_turn(rounds))
    agent.messages = messages


# --- the target ---------------------------------------------------------------


def test_target_outside_the_valid_band_raises() -> None:
    """Positive and below the budget is the whole rule: at or above the
    budget a trim reaches nothing (the trigger fires there), and a
    non-positive target has no sawtooth at all. Both raise; anything in
    between — including within a reply reserve of the budget — is a
    permitted preference, not an error."""
    agent = _agent(compact_to_tokens=WINDOW * 4)
    with pytest.raises(ValueError, match="compact_to_tokens"):
        agent.compact_target_tokens()
    at_zero = _agent(compact_to_tokens=0)
    with pytest.raises(ValueError, match="compact_to_tokens"):
        at_zero.compact_target_tokens()


def test_shallow_target_within_a_reserve_of_the_budget_is_allowed() -> None:
    """The sweeps' best arms sit here: shallower than budget − reserve. A
    worst-case reply can re-cross the budget within a round, but that is a
    slower configuration, not a broken one."""
    shallow = _agent(compact_to_tokens=24832 - 512)  # budget - reserve exactly
    assert shallow.compact_target_tokens() == 24832 - 512
    just_below_budget = _agent(compact_to_tokens=19455)
    assert just_below_budget.compact_target_tokens() == 19455


def test_the_raise_boundary_is_the_budget_itself() -> None:
    """The boundary, not merely either side of it. The test above pins one
    below the budget as allowed; this pins the budget itself as refused, so a
    `>=` loosening to `>` cannot make "target equals its own trigger" — the
    case with no hysteresis at all — quietly legal."""
    at_budget = _agent(compact_to_tokens=24832)
    assert at_budget.context_budget_tokens() == 24832, "fixture: that value IS the budget"
    with pytest.raises(ValueError, match="compact_to_tokens"):
        at_budget.compact_target_tokens()


def test_target_is_used_when_lower() -> None:
    agent = _agent(compact_to_tokens=2048)
    assert agent.compact_target_tokens() == 2048


# Building the preset's engine config imports the GPU stack, which probes
# CUDA and warns (not errors) on a GPU-less or old-driver box; scoped to
# torch's probing so everything else in these constructions still errors.
@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_qwen_vllm_standard_defaults_cohere() -> None:
    """The standard deployment's numbers must stay mutually coherent, not just
    individually declared: construction computes a target, which runs the
    target-vs-budget check, and settling runs the window-vs-engine one — so an
    incoherent edit to any one default fails here."""
    agent = SealAgent.build()
    assert 0 < agent._compact_target_tokens() < agent._context_budget_tokens()
    agent.prepare_for_backend()


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_qwen_vllm_explicit_arguments_beat_the_standard() -> None:
    """Every value a column supplies must stay overridable: ``build`` assigns
    the keyword arguments after the column, so an explicit one always wins."""
    agent = SealAgent.build(
        context_window_tokens=32768,
        reply_cap_tokens=4096,
        compact_to_tokens=2048,
    )
    assert agent.compact_to_tokens == 2048
    assert agent._compact_target_tokens() == 2048


# --- the sawtooth -------------------------------------------------------------


async def test_compaction_undershoots_the_budget() -> None:
    """The point of hysteresis: land at the target, well UNDER the budget,
    not at the budget's edge."""
    compacting = _agent(compact_to_tokens=REACHABLE_TARGET)
    _single_block(compacting)
    await compacting._trim_messages_for_context(compacting.messages, enforce=True)
    compacted = compacting._decision_prompt_tokens(compacting.messages)

    assert compacted <= compacting._compact_target_tokens(), (compacted, REACHABLE_TARGET)
    assert compacting._compact_target_tokens() < compacting._context_budget_tokens()


async def test_an_under_budget_prompt_is_left_alone() -> None:
    """The TRIGGER stays the budget: a prompt already under it is not
    compacted down to the target for no reason."""
    agent = _agent(compact_to_tokens=2048)
    agent.messages = [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")), _turn(1)]
    before = agent._decision_prompt_tokens(agent.messages)
    assert before <= agent._context_budget_tokens(), "fixture must start under budget"
    await agent._trim_messages_for_context(agent.messages, enforce=True)
    assert agent._decision_prompt_tokens(agent.messages) == before


async def test_a_trim_eviction_gets_under_budget_still_reaches_the_target() -> None:
    """The 20260731 regression, and the case the single-block fixtures above
    all miss. Eviction lands the prompt between target and budget; pruning must
    finish the trim. It only does if it shares eviction's budget trigger
    instead of re-testing the budget itself -- re-testing bails in exactly this
    band, which is where nearly every real trim ends up. Measured on that run:
    340 trims, 11 reached the target, 0 pruned a round."""
    agent = _agent(compact_to_tokens=REACHABLE_TARGET)
    _many_blocks(agent)
    before = agent._decision_prompt_tokens(agent.messages)
    assert before > agent._context_budget_tokens(), "fixture must trigger a trim"

    await agent._trim_messages_for_context(agent.messages, enforce=True)

    after = agent._decision_prompt_tokens(agent.messages)
    assert after <= agent._compact_target_tokens(), (after, agent._compact_target_tokens())


async def test_compaction_still_keeps_rounds_whole() -> None:
    """Undershooting drops more rounds; they must still leave in pairs."""
    agent = _agent(compact_to_tokens=2048)
    _single_block(agent)
    await agent._trim_messages_for_context(agent.messages, enforce=True)
    turn = agent.messages[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    if turn.elements:
        assert isinstance(turn.elements[0], ullm.conv.LLMResponse)
        assert len(turn.elements) % 2 == 0, [type(e).__name__ for e in turn.elements]


# --- the brevity line ---------------------------------------------------------


def test_brevity_line_rides_the_system_prompt_once() -> None:
    # Pinned on rather than taken from the default: this covers the mechanism
    # (once, in the system prompt, never in the user prompt), and a swept
    # default moving is not a reason for it to fail.
    for level in ("full", "half"):
        line = prompts.SealPrompts().brevity_lines[level]
        agent = _agent(prompts=prompts.SealPrompts(brevity=level))
        assert agent._system_prompt_text().count(line) == 1
        assert line not in agent._build_user_prompt_text()


def test_brevity_line_off_removes_it() -> None:
    agent = _agent(prompts=prompts.SealPrompts(brevity="off"))
    lines = prompts.SealPrompts().brevity_lines
    assert lines["full"] not in agent._system_prompt_text()
    assert lines["half"] not in agent._system_prompt_text()


def test_game_over_prompt_does_not_repeat_system_guidance() -> None:
    agent = _agent()
    agent.game_over_active = True
    prompt = agent._build_user_prompt_text()
    assert "RESET is the only action" in prompt
    assert prompts.SealPrompts().brevity_lines["full"] not in prompt


async def test_no_count_in_a_trim_renders_an_undeclared_prefix() -> None:
    """A firing trim rewrites the system prefix (it refreshes the journal
    snapshot) and then re-counts the prompt many times while evicting. Under
    ``exact_history`` the backend refuses any render whose served prefix
    changed without ``history_epoch`` moving — so the declaration has to come
    BEFORE the refresh, not after the trim.

    The window is narrow and self-hiding: once eviction has dropped the base
    witness, ``served_history`` returns None and the check stops running, so
    only the counts between the refresh and the first eviction can catch it.
    It opens whenever the snapshot SHRINKS — the journal was rendered whole
    while no ``memory`` result was retained, and a newly retained one then
    excludes its own entry from it.
    """
    seen: list[tuple[str, int]] = []

    class _Recorder:
        async def token_count(self, params: Any, conversation: Any) -> int:
            seen.append((conversation.system_prompt.content.to_string(), conversation.history_epoch))
            return 10**6 if len(seen) < 5 else 0

    agent = _agent(compact_to_tokens=REACHABLE_TARGET, token_native_context=True)
    agent.memory_entries = [
        MemoryEntry(index=i, response=i, step=i, level=1, last_action="UP", text=f"finding {i}") for i in range(1, 6)
    ]
    agent.memory_entry_count = 5
    # The stale snapshot: the whole journal, as rendered while nothing was
    # retained. The retained result below excludes entry 3 and later from it.
    agent.memory_prompt_snapshot = agent._render_memory(agent.memory_entries)  # pyright: ignore[reportPrivateUsage]

    elements: list[ullm.conv.ConversationElement] = []
    for index in range(6):
        elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=f"user turn {index}")))
        answered: list[Any] = []
        if index == 4:
            ack = agent.prompts.memory_append_acknowledgement(agent.memory_entries[2])
            answered = [
                ullm.conv.ToolResult(call_id="t", tool_name="memory", content=ullm.conv.StringContent(text=ack))
            ]
        elements.append(ullm.conv.ResponseTurn(elements=answered))
    agent.messages = elements

    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(
            content=ullm.conv.StringContent(
                text=agent._system_prompt_text()  # pyright: ignore[reportPrivateUsage]
            )
        ),
        elements=elements,
        status="intermediate",
        history_epoch=0,
    )
    served = conversation.system_prompt.content.to_string()
    agent._live_backend = _Recorder()  # pyright: ignore[reportPrivateUsage]
    agent._live_conversation = conversation  # pyright: ignore[reportPrivateUsage]
    agent._live_params = object()  # pyright: ignore[reportPrivateUsage]

    await agent._trim_messages_for_context(elements, enforce=False)  # pyright: ignore[reportPrivateUsage]

    assert seen, "the trim took no counts, so this proves nothing"
    assert any(text != served for text, _ in seen), "the snapshot never changed; the window was not opened"
    undeclared = [i for i, (text, epoch) in enumerate(seen) if text != served and epoch == 0]
    assert not undeclared, (
        f"counts {undeclared} rendered a rewritten system prefix while history_epoch still said it was "
        f"served unchanged — an exact_history backend raises HistoryDriftError there"
    )
