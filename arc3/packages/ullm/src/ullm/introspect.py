"""Read-only introspection helpers over finished conversation objects.

Hosts that record or analyze LLM traffic (platypus diagnostics, ad-hoc
notebooks) consume turns through these instead of walking ``ullm.conv``
internals — one stable surface, one place to update when the
conversation model changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import ullm.conv


@dataclass(frozen=True)
class ResponseSummary:
    """Bookkeeping for one ``LLMResponse`` within a turn. Token fields are
    zero for synthetic placeholders (prompt-too-long, per-turn-cap
    markers), which carry no backend usage."""

    completion_tokens: int
    uncached_input_tokens: int  # prompt minus cached — the I1 accounting convention
    cached_input_tokens: int  # prompt tokens served from prefix cache
    reasoning_tokens: int
    stop_reason: str | None
    malformed_retries: int
    n_tool_calls: int
    synthetic: bool  # do_not_send_to_agent placeholders
    truncated_by_per_turn_cap: bool
    response_injection: str | None  # text served after the generation prompt that the answer starts with


def turn_response_summaries(turn: ullm.conv.ResponseTurn) -> list[ResponseSummary]:
    """One summary per ``LLMResponse`` in the turn, in order. Works on
    partial (unprocessed) turns — a turn that errored mid-way summarizes
    the responses that completed."""
    summaries: list[ResponseSummary] = []
    for element in turn.elements:
        if not isinstance(element, ullm.conv.LLMResponse):
            continue
        usage = element.token_usage
        summaries.append(
            ResponseSummary(
                completion_tokens=usage.completion_tokens if usage else 0,
                uncached_input_tokens=(usage.prompt_tokens - usage.cached_tokens) if usage else 0,
                cached_input_tokens=usage.cached_tokens if usage else 0,
                reasoning_tokens=usage.reasoning_tokens if usage else 0,
                stop_reason=element.stop_reason,
                malformed_retries=element.malformed_retries,
                n_tool_calls=len(element.final_content.tool_calls()),
                synthetic=element.do_not_send_to_agent,
                truncated_by_per_turn_cap=element.truncated_by_per_turn_cap,
                response_injection=element.response_injection,
            )
        )
    return summaries
