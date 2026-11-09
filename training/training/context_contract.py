"""Trainer-owned context budgets and gateway lineage limits."""

from __future__ import annotations

from dataclasses import dataclass

from training.gateway_lineage import LineagePolicy


@dataclass(frozen=True)
class ContextContract:
    context_window_tokens: int
    max_completion_tokens: int
    max_summary_completion_tokens: int
    safety_margin_tokens: int
    compaction_trigger_tokens: int
    reset_prompt_ceiling_tokens: int
    max_served_sequence_tokens: int
    max_compaction_proof_tokens: int
    summary_forbidden_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        values = (
            self.context_window_tokens,
            self.max_completion_tokens,
            self.max_summary_completion_tokens,
            self.safety_margin_tokens,
            self.compaction_trigger_tokens,
            self.reset_prompt_ceiling_tokens,
            self.max_served_sequence_tokens,
            self.max_compaction_proof_tokens,
        )
        if any(type(value) is not int for value in values):
            raise ValueError("context contract budgets must be integers")
        if type(self.summary_forbidden_token_ids) is not tuple or any(
            type(token) is not int or token < 0
            for token in self.summary_forbidden_token_ids
        ):
            raise ValueError("summary forbidden token IDs must be nonnegative integers")
        if not (
            0
            < self.context_window_tokens
            <= self.max_served_sequence_tokens
            <= self.max_compaction_proof_tokens
            and 0 < self.max_completion_tokens < self.max_served_sequence_tokens
            and self.max_summary_completion_tokens >= 0
            and (
                self.max_summary_completion_tokens == 0
                or self.max_served_sequence_tokens > self.context_window_tokens
            )
            and (
                not self.summary_forbidden_token_ids
                or self.max_summary_completion_tokens > 0
            )
            and self.safety_margin_tokens >= 0
            and 0
            < self.reset_prompt_ceiling_tokens
            <= self.compaction_trigger_tokens
            < self.max_compaction_proof_tokens
        ):
            raise ValueError("context contract budgets must be positive and consistent")

    def lineage_policy(self, *, max_completion_tokens: int | None = None) -> LineagePolicy:
        """Bound resets using the reply budget actually available to the request."""
        reset_ceiling = self.reset_prompt_ceiling_tokens
        if self.max_summary_completion_tokens and max_completion_tokens is not None:
            reset_ceiling += max(0, self.max_completion_tokens - max_completion_tokens)
        return LineagePolicy(
            self.compaction_trigger_tokens,
            reset_ceiling,
            self.max_compaction_proof_tokens,
        )


def resolve_context_contract(
    window: int | None,
    reply: int | None,
    target: int | None,
    margin: int | None,
    summary_cap: int | None = None,
    served_ceiling: int | None = None,
    summary_forbidden_token_ids: tuple[int, ...] = (),
) -> ContextContract:
    """Resolve and validate full-game context budgets from numeric limits."""
    # Pin the ordinary Duck default independently of harness presets.
    margin = 512 if margin is None else margin
    if (
        type(window) is not int
        or window <= 0
        or type(reply) is not int
        or not 0 < reply < window
        or type(margin) is not int
        or margin < 0
    ):
        raise ValueError(
            "full-game serving requires explicit positive context/reply budgets and a nonnegative margin"
        )
    trigger = max(1024, window - reply - margin)
    summary_cap = 0 if summary_cap is None else summary_cap
    served_ceiling = window if served_ceiling is None else served_ceiling
    if type(summary_cap) is not int or summary_cap < 0:
        raise ValueError("summary completion cap must be a nonnegative integer")
    if type(target) is not int or not 0 < target <= trigger or (
        target == trigger and not summary_cap
    ):
        raise ValueError(
            "agent_compact_to_tokens must be positive and "
            + ("at or below" if summary_cap else "below")
            + " the context trigger"
        )
    if type(served_ceiling) is not int or served_ceiling < window:
        raise ValueError("served sequence ceiling must be at least the context window")
    if summary_cap > 0 and served_ceiling == window:
        raise ValueError(
            "a positive summary cap requires a served ceiling above the window"
        )
    if any(
        type(token) is not int or token < 0 for token in summary_forbidden_token_ids
    ):
        raise ValueError("summary forbidden token IDs must be nonnegative integers")
    if summary_forbidden_token_ids and summary_cap == 0:
        raise ValueError("summary forbidden token IDs require a positive summary cap")
    return ContextContract(
        context_window_tokens=window,
        max_completion_tokens=reply,
        max_summary_completion_tokens=summary_cap,
        safety_margin_tokens=margin,
        compaction_trigger_tokens=trigger,
        reset_prompt_ceiling_tokens=target,
        max_served_sequence_tokens=served_ceiling,
        max_compaction_proof_tokens=max(2 * window, served_ceiling),
        summary_forbidden_token_ids=tuple(summary_forbidden_token_ids),
    )
