"""Engine-truth context accounting (U1): the compaction decision runs on
the engine's own billed usage plus a conservative bound for everything not
yet billed — never on a client-side prediction — and every billed usage is
held against the window invariant post-turn.

The failure class these tests pin shut: a client-side estimate cannot see
engine-side costs (vision tokens above all), so an estimate-driven trigger
lets histories outgrow the window unnoticed and hands oversized sequences
downstream. The decision ruler here can only fire the compaction trigger
EARLY: the billed part is engine truth, the unbilled part is charged the
character count of its text (a token spans at least one character) plus
framing and image constants, and any shortfall in those constants is
load-bearing for at most one round before the next billed usage replaces
it — with the window-invariant raise as the backstop.

Covers: anchor selection (newest billed usage; usage-less and synthetic
responses skipped over, not anchored on), the bound arithmetic (character
count, per-message overhead, per-image engine cost and its override), trigger
timing on engine truth where any text ruler would sleep through it, the
sawtooth settling after a trim retires pre-trim usage, re-anchoring on the
first post-trim response, and the post-turn checks (window invariant raise,
undercount report, pairing assert).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import ullm.backend
import ullm.continuation
import ullm.conv

from duck import DuckAgent
from duck.agent import _ANCHOR_RERENDER_OVERHEAD_TOKENS

WINDOW = 28672  # the training deployment's window: budget = 28672 - 512 - 512 = 27648
TARGET = 16384


def _agent(**kwargs: Any) -> DuckAgent:
    """The training-shaped agent: no image on the prompt (images enter these
    tests as hand-built message content), window/target as deployed."""
    kwargs.setdefault("context_window_tokens", WINDOW)
    kwargs.setdefault("reply_cap_tokens", None)
    kwargs.setdefault("compaction_strategy", "prune")
    kwargs.setdefault("compact_to_tokens", TARGET)
    return DuckAgent(multimodal_context="", **kwargs)


def _billed(prompt: int, completion: int, *, text: str = "answer") -> ullm.conv.LLMResponse:
    """A response the engine billed: usage is the engine's own account."""
    response = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text=text))
    response.token_usage = ullm.backend.TokenUsage(prompt_tokens=prompt, completion_tokens=completion)
    return response


def _user(text: str) -> ullm.conv.UserPrompt:
    return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))


def _tool_result(text: str, call_id: str = "c0") -> ullm.conv.ToolResult:
    return ullm.conv.ToolResult(call_id=call_id, tool_name="python", content=ullm.conv.StringContent(text=text))


def _image(side: int = 256) -> ullm.conv.ImageContent:
    return ullm.conv.ImageContent(data=np.zeros((side, side, 3), dtype=np.uint8))


# --- the anchor -----------------------------------------------------------


def test_decision_is_anchored_on_the_newest_billed_usage() -> None:
    """Engine truth dominates: the billed prompt+completion floor the
    prediction however small the text reads, and what was appended since
    rides on top."""
    agent = _agent()
    anchor = _billed(20000, 1000)
    messages: list[ullm.conv.ConversationElement] = [
        _user("go"),
        ullm.conv.ResponseTurn(elements=[anchor, _tool_result("result " * 50)], status="intermediate"),
        _user("next move"),
    ]
    decision = agent._decision_prompt_tokens(messages)
    assert decision > 21000 + _ANCHOR_RERENDER_OVERHEAD_TOKENS, decision
    # The pending tail (tool result + user prompt) is charged on top of it.
    known_only = agent._decision_prompt_tokens([_user("go"), ullm.conv.ResponseTurn(elements=[anchor])])
    assert decision > known_only


def test_pending_is_everything_after_the_anchor() -> None:
    """Appending content after the anchor moves the prediction; content
    already inside the billed prompt does not move it twice."""
    agent = _agent()
    anchor = _billed(10000, 500)
    base: list[ullm.conv.ConversationElement] = [
        _user("go"),
        ullm.conv.ResponseTurn(elements=[anchor], status="intermediate"),
    ]
    before = agent._decision_prompt_tokens(base)
    turn = base[1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    turn.elements.append(_tool_result("fresh output " * 100))
    after = agent._decision_prompt_tokens(base)
    assert after > before, "a pending tool result must weigh in"


def test_usage_less_and_synthetic_responses_do_not_anchor() -> None:
    """A response the backend returned without usage (a known vLLM lapse)
    and a synthetic placeholder are unbilled weight, not anchors: the
    decision anchors on the older billed response and bounds them on top."""
    agent = _agent()
    anchor = _billed(15000, 800)
    usage_less = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="lost usage " * 40))
    synthetic = ullm.conv.LLMResponse(do_not_send_to_agent=True)
    synthetic.token_usage = ullm.backend.TokenUsage(prompt_tokens=1, completion_tokens=1)
    messages: list[ullm.conv.ConversationElement] = [
        _user("go"),
        ullm.conv.ResponseTurn(elements=[anchor, _tool_result("r"), usage_less, synthetic], status="intermediate"),
    ]
    found, known = agent._usage_anchor(messages)
    assert found is anchor
    assert known == 15800
    assert agent._decision_prompt_tokens(messages) > 15800, "the usage-less tail must still be bounded"


def test_cold_start_bounds_the_whole_prompt() -> None:
    """No response yet: the whole request is bounded by the character
    count of its rendering (a token spans at least one character) plus the
    per-message framing constants — so the bound covers every character of
    prompt text, and appended text moves it character for character."""
    agent = _agent()
    text = "open " * 200
    messages: list[ullm.conv.ConversationElement] = [_user(text)]
    decision = agent._decision_prompt_tokens(messages)
    assert decision > len(text), "the bound must cover at least the user text's own characters"
    longer: list[ullm.conv.ConversationElement] = [_user(text + "x" * 100)]
    assert agent._decision_prompt_tokens(longer) == decision + 100, "character-denominated, exactly"


# --- the trigger ----------------------------------------------------------


async def test_engine_truth_fires_the_trigger_where_text_rulers_sleep() -> None:
    """The engine has billed more than any text ruler can see (vision
    tokens, dense reasoning): every text ruler reads under the budget while
    the decision ruler fires — the case the engine-truth anchor exists for."""
    agent = _agent()
    messages: list[ullm.conv.ConversationElement] = [
        _user("a"),
        ullm.conv.ResponseTurn(elements=[_billed(27500, 400)], status="intermediate"),
        _user("b"),
        ullm.conv.ResponseTurn(elements=[], status="intermediate"),
    ]
    budget = agent._context_budget_tokens()
    unbilled = agent._bounded_prompt_tokens(messages)
    assert unbilled < budget, "fixture: without the bill, the same conversation reads under budget"
    assert agent._decision_prompt_tokens(messages) > budget
    await agent._trim_messages_for_context(messages)
    assert agent._last_trim_fired, "engine-billed growth must fire the trigger"


async def test_a_fired_trim_retires_pre_trim_usage_and_settles() -> None:
    """The sawtooth: a trim rewrites the prompt the billed usage described,
    so the bill must stop steering — otherwise the stale 27.9k reading
    would re-fire every round until the conversation was gone."""
    agent = _agent()
    messages: list[ullm.conv.ConversationElement] = [
        _user("a"),
        ullm.conv.ResponseTurn(elements=[_billed(27500, 400)], status="intermediate"),
        _user("b"),
        ullm.conv.ResponseTurn(elements=[], status="intermediate"),
    ]
    await agent._trim_messages_for_context(messages)
    assert agent._last_trim_fired
    assert agent._stale_responses is not None
    settled = agent._decision_prompt_tokens(messages)
    assert settled <= agent._context_budget_tokens(), settled
    await agent._trim_messages_for_context(messages)
    assert not agent._last_trim_fired, "a settled prompt must not re-fire on the retired bill"


def test_a_fresh_billed_response_re_anchors_after_a_trim() -> None:
    """Trust after verify: the first response generated after the rewrite
    carries the engine's account of the REWRITTEN prompt, and the decision
    path returns to engine truth (clearing the staleness record)."""
    agent = _agent()
    stale_response = _billed(27500, 400)
    messages: list[ullm.conv.ConversationElement] = [
        _user("a"),
        ullm.conv.ResponseTurn(elements=[stale_response], status="intermediate"),
    ]
    agent._mark_usage_stale(messages)
    assert agent._decision_prompt_tokens(messages) < 27000, "the stale bill must not steer"
    fresh = _billed(6000, 300)
    turn = messages[1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    turn.elements.append(fresh)
    decision = agent._decision_prompt_tokens(messages)
    assert decision >= 6300
    assert decision < 27500, "the fresh bill, not the stale one, must anchor"
    assert agent._stale_responses is None, "fresh engine truth settles the staleness record"


def test_the_safety_margin_is_a_knob() -> None:
    """budget = window − reserve − margin; the margin is configuration."""
    default = _agent()
    widened = _agent(safety_margin_tokens=1024)
    assert default._context_budget_tokens() - widened._context_budget_tokens() == 512


def test_anti_conservative_knobs_are_refused() -> None:
    """A negative margin, framing allowance, or image cost would
    reintroduce undercounting by configuration; refused at construction."""
    with pytest.raises(ValueError, match="safety_margin_tokens"):
        _agent(safety_margin_tokens=-1)
    with pytest.raises(ValueError, match="per_message_overhead_tokens"):
        _agent(per_message_overhead_tokens=-1)
    with pytest.raises(ValueError, match="image_tokens_override"):
        _agent(image_tokens_override=-1)


# --- images ---------------------------------------------------------------


def test_image_cost_derivation() -> None:
    """One visual token per 28x28 cell for the densest Qwen-VL family, x2
    slack, +32 framing: 256x256 (the default 64x64 board at upscale 4) ->
    ceil(256/28)^2 = 100 cells -> 232; 64x64 -> 9 cells -> 50."""
    agent = _agent()
    assert agent._image_bound_tokens(_image(256)) == 2 * 100 + 32
    assert agent._image_bound_tokens(_image(64)) == 2 * 9 + 32
    assert _agent(image_tokens_override=999)._image_bound_tokens(_image(256)) == 999


def test_a_pending_image_is_charged_its_engine_cost() -> None:
    """The failure class behind the crashed runs: an image adds engine-side
    tokens no text ruler can see. A pending user prompt carrying one must
    weigh exactly its text plus the per-image bound."""
    agent = _agent()
    anchor = _billed(10000, 500)

    def prompt_with(elements: list[ullm.conv.Content]) -> list[ullm.conv.ConversationElement]:
        return [
            _user("go"),
            ullm.conv.ResponseTurn(elements=[anchor], status="intermediate"),
            ullm.conv.UserPrompt(content=ullm.conv.ListContent(elements=elements)),
        ]

    text_only = agent._decision_prompt_tokens(prompt_with([ullm.conv.StringContent(text="look")]))
    with_image = agent._decision_prompt_tokens(prompt_with([ullm.conv.StringContent(text="look"), _image(256)]))
    assert with_image - text_only == agent._image_bound_tokens(_image(256))


def test_cold_start_counts_images_too() -> None:
    """The whole-prompt bound charges images the same engine-side cost."""
    agent = _agent()
    plain: list[ullm.conv.ConversationElement] = [_user("look")]
    imaged: list[ullm.conv.ConversationElement] = [
        ullm.conv.UserPrompt(
            content=ullm.conv.ListContent(elements=[ullm.conv.StringContent(text="look"), _image(256)])
        )
    ]
    assert agent._decision_prompt_tokens(imaged) - agent._decision_prompt_tokens(plain) >= agent._image_bound_tokens(
        _image(256)
    )


# --- the post-turn checks -------------------------------------------------


def test_window_invariant_violation_raises() -> None:
    """The invariant the whole path serves: billed prompt+completion above
    the window is an oversized sequence in the making, and it fails the
    game loudly at the moment of the overrun instead of travelling on."""
    agent = _agent()
    agent._pending_bounds.extend([27000])
    turn = ullm.conv.ResponseTurn(elements=[_billed(28000, 1000)])
    with pytest.raises(RuntimeError, match="context window invariant violated"):
        agent._check_engine_usage(turn)


def test_billed_within_the_window_passes_and_counts() -> None:
    agent = _agent()
    agent._pending_bounds.extend([27000])
    assert agent._check_engine_usage(ullm.conv.ResponseTurn(elements=[_billed(26500, 400)])) == 1


def test_a_beaten_bound_is_reported_not_fatal() -> None:
    """An undercount of the conservative increment heals at the next
    response and lives inside the safety margin meanwhile; only the window
    invariant kills."""
    agent = _agent()
    agent._pending_bounds.extend([1000])
    assert agent._check_engine_usage(ullm.conv.ResponseTurn(elements=[_billed(1200, 100)])) == 1


def test_usage_less_and_synthetic_responses_skip_the_check() -> None:
    """Never sent, or billed without a usage block: nothing to hold against
    the invariant — skipped (and the usage-less case logged), not fatal."""
    agent = _agent()
    agent._pending_bounds.extend([1000, 1000])
    no_usage = ullm.conv.LLMResponse()
    synthetic = ullm.conv.LLMResponse(do_not_send_to_agent=True)
    assert agent._check_engine_usage(ullm.conv.ResponseTurn(elements=[no_usage, synthetic])) == 0


def test_broken_bound_pairing_is_a_hard_error() -> None:
    """before_round records one bound per LLM call; if the pairing breaks
    the billed usage cannot be held against what predicted it."""
    agent = _agent()
    agent._pending_bounds.extend([1000, 2000])
    with pytest.raises(AssertionError, match="bound pairing broke"):
        agent._check_engine_usage(ullm.conv.ResponseTurn(elements=[_billed(900, 10)]))


# --- token-native measurement ---------------------------------------------


class _ExactBackend:
    """A backend answering ``token_count`` with a chosen exact weight — the
    seam token-native mode measures through."""

    def __init__(self, value: int | None) -> None:
        self.value = value
        self.calls = 0

    async def token_count(self, params: Any, conversation: Any) -> int | None:
        self.calls += 1
        return self.value


def _wire_live(agent: DuckAgent, messages: list[ullm.conv.ConversationElement], backend: _ExactBackend) -> None:
    agent._live_conversation = ullm.conv.Conversation(elements=messages, history_epoch=agent.history_epoch)
    agent._live_backend = backend
    agent._live_params = object()


async def test_token_native_measures_with_the_backends_exact_weight() -> None:
    """In token-native mode the decision reading IS the backend's exact
    weight of the next request (the id-ledger length on continued calls) —
    no bills, no bounds."""
    agent = _agent(token_native_context=True)
    messages: list[ullm.conv.ConversationElement] = [_user("go")]
    backend = _ExactBackend(23456)
    _wire_live(agent, messages, backend)
    assert await agent._measured_prompt_tokens(messages) == 23456
    assert backend.calls == 1


class _DriftingBackend(_ExactBackend):
    """``token_count`` raising ``HistoryDriftError`` -- the win15 class
    surfacing on the count path."""

    def __init__(self) -> None:
        super().__init__(None)

    async def token_count(self, params: Any, conversation: Any) -> int | None:
        self.calls += 1
        raise ullm.continuation.HistoryDriftError("history no longer renders as served")


async def test_token_native_count_drift_degrades_to_the_ruler() -> None:
    """Drift raised on the COUNT path must not kill the episode: the
    measurement is advisory, so the reading degrades to the bill+bound
    ruler (which can only fire early). The respond path keeps its raises
    (U5) -- this catch is count-only."""
    agent = _agent(token_native_context=True)
    messages: list[ullm.conv.ConversationElement] = [_user("go")]
    backend = _DriftingBackend()
    _wire_live(agent, messages, backend)
    assert await agent._measured_prompt_tokens(messages) == agent._decision_prompt_tokens(messages)
    assert backend.calls == 1


async def test_token_native_falls_back_when_the_backend_cannot_weigh() -> None:
    """A backend answering None, a config without the mode, or a list that
    is not the live conversation's all measure with the bill+bound
    arithmetic — and the modeless agent never asks the backend at all
    (cloud recordings must not grow count calls)."""
    messages: list[ullm.conv.ConversationElement] = [_user("go")]

    native = _agent(token_native_context=True)
    unweighable = _ExactBackend(None)
    _wire_live(native, messages, unweighable)
    assert await native._measured_prompt_tokens(messages) == native._decision_prompt_tokens(messages)
    assert unweighable.calls == 1

    modeless = _agent()
    silent = _ExactBackend(31337)
    _wire_live(modeless, messages, silent)
    assert await modeless._measured_prompt_tokens(messages) == modeless._decision_prompt_tokens(messages)
    assert silent.calls == 0

    other_list: list[ullm.conv.ConversationElement] = [_user("not the live list")]
    counted = _ExactBackend(31337)
    _wire_live(native, messages, counted)
    assert await native._measured_prompt_tokens(other_list) == native._decision_prompt_tokens(other_list)
    assert counted.calls == 0


@pytest.mark.parametrize("growth", [1000, 512])
async def test_compaction_spacing_follows_the_window_arithmetic(growth: int) -> None:
    """The sawtooth's period, pinned: with window W, target C, reserve R,
    margin M and g tokens of growth per round, a compaction fires every
    floor((W−R−M−C)/g) + 1 rounds — the trigger squeezes all remaining
    slack out of the window, and any future change that quietly narrows the
    band shows up here as a shorter cycle."""
    window, target = 20480, 16384
    agent = _agent(token_native_context=True, context_window_tokens=window, compact_to_tokens=target)
    budget = agent._context_budget_tokens()  # W − 512 (reserve fallback) − 512 (margin)
    expected_cycle = (budget - target) // growth + 1

    messages: list[ullm.conv.ConversationElement] = [_user("go")]

    class _Sawtooth:
        """The exact weight of a run that grows and gets trimmed: the live
        value until a trim fires, the target once it has (a fired trim's
        own re-measures see the trimmed prompt)."""

        value = target

        async def token_count(self, params: Any, conversation: Any) -> int:
            return target if agent._last_trim_fired else self.value

    backend = _Sawtooth()
    agent._live_conversation = ullm.conv.Conversation(elements=messages, history_epoch=agent.history_epoch)
    agent._live_backend = backend
    agent._live_params = object()

    for cycle in range(2):
        rounds = 0
        baseline: int | None = None
        for step in range(1, expected_cycle + 3):
            backend.value = target + step * growth
            rounds += 1
            agent._rounds_since_compaction += 1  # as _trim_live_messages counts each LLM call
            baseline = agent._tokens_at_last_compaction
            await agent._trim_messages_for_context(messages)
            if agent._last_trim_fired:
                break
        assert rounds == expected_cycle, f"cycle {cycle}: fired after {rounds} rounds, expected {expected_cycle}"
        assert agent._rounds_since_compaction == 0, "the spacing counter resets at the trim"
        assert agent._tokens_at_last_compaction == target, "the post-trim reading is the next baseline"
        if cycle > 0:
            # The spacing the trim just logged: the fire-time reading minus
            # the previous post-trim baseline — one full band of growth.
            assert baseline == target
            assert (target + rounds * growth) - baseline == expected_cycle * growth


async def test_a_pruned_round_still_faces_the_window_invariant() -> None:
    """Round pruning removes a billed response from the live turn before the
    block's post-turn check runs; the response leaves the live pairing, not
    the invariant. An over-window bill in a pruned round must still raise."""
    agent = _agent()
    over = _billed(28000, 1000)  # 29000 > the 28672 window
    turn = ullm.conv.ResponseTurn(
        elements=[over, _tool_result("r0", "c0"), _billed(5000, 10), _tool_result("r1", "c1")],
        status="intermediate",
    )
    messages: list[ullm.conv.ConversationElement] = [_user("go"), turn]
    agent._pending_bounds.extend([100, 200])
    pruned = await agent._prune_current_turn(messages, 1)  # a trigger every prompt exceeds
    assert pruned >= 1 and agent._deferred_usage_checks, "fixture: pruning must defer at least one pair"
    with pytest.raises(RuntimeError, match="context window invariant violated"):
        agent._check_engine_usage(turn)


async def test_pruned_in_window_usage_is_checked_exactly_once() -> None:
    """Deferred pairs are drained by the next check and only by it: every
    billed response is counted once, pruned or not."""
    agent = _agent()
    turn = ullm.conv.ResponseTurn(
        elements=[_billed(4000, 10), _tool_result("r0", "c0"), _billed(5000, 10), _tool_result("r1", "c1")],
        status="intermediate",
    )
    messages: list[ullm.conv.ConversationElement] = [_user("go"), turn]
    agent._pending_bounds.extend([4500, 5500])
    await agent._prune_current_turn(messages, 1)  # empties the turn, deferring both pairs
    assert not agent._pending_bounds
    assert agent._check_engine_usage(turn) == 2
    assert agent._check_engine_usage(ullm.conv.ResponseTurn(elements=[])) == 0  # drained, never re-checked


if __name__ == "__main__":
    pytest.main([__file__])
