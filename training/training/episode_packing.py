"""Episode packing: merge per-step samples into one sequence per episode.

Consecutive REPL steps of one rollout share their prompt prefix: with
token-continuation rollouts, step i+1's prompt IS step i's full sequence plus
the tokenized delta appended after it, so the token ids nest exactly. One
packed sequence — the LAST step's input_ids — contains every earlier step's
completion at unchanged positions, and causal attention evaluates every span under its original token
history in a single forward pass instead of one forward per step. Numerical
kernel differences remain subject to the sampled-logprob parity tolerances.

Nesting is required WITHIN a segment: a same-trajectory step that fails to
extend the previous sequence token-for-token raises, because it means the
rollout agent re-rendered or truncated history unexpectedly. The one allowed
break is a flagged compaction boundary (`sample.segment_start`): the rollout
compacted its context (e.g. 32k -> 16k), so the next step's prompt is the
fresh compacted prefix. There, packing closes the current segment and starts a
new one — an N-compaction episode becomes N+1 packed sequences, all sharing the
trajectory id. Token mass is additive across these sequences. The compacted
prefix is prompt, hence masked, so a new segment only trains its own
post-compaction generations. The runtime rollout-logprob parity check remains
the independent validator on a bounded set of those native target positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from training.dataset import TokenizedSample


log = logging.getLogger(__name__)


@dataclass(slots=True)
class PackingStats:
    input_samples: int = 0
    output_samples: int = 0
    packed_sequences: int = 0
    packed_spans: int = 0
    passthrough_samples: int = 0
    nesting_breaks: int = 0
    length_breaks: int = 0
    # Intended segment boundaries (rollout compacted its context): the trajectory
    # continues but as a fresh append-only segment / packed sequence.
    segment_boundaries: int = 0

    @property
    def coverage(self) -> float:
        if self.input_samples <= 0:
            return 0.0
        return self.packed_spans / self.input_samples


def _nests_after(previous: TokenizedSample, sample: TokenizedSample) -> bool:
    """True iff `sample`'s prompt starts with the whole previous sequence."""
    previous_length = len(previous.input_ids)
    if sample.prompt_token_count < previous_length:
        return False
    return sample.input_ids[:previous_length] == previous.input_ids and media_extends(
        previous.media, sample.media
    )


def media_extends(
    previous: dict[str, Any] | None, current: dict[str, Any] | None
) -> bool:
    """Require unchanged ordered pixels, layout and processing within a segment."""
    if previous is None:
        return True
    if current is None or previous["processing"] != current["processing"]:
        return False
    count = len(previous["images"])
    return (
        current["images"][:count] == previous["images"]
        and current["layout"][:count] == previous["layout"]
    )


def assign_loss_weights(
    samples: list[TokenizedSample], *, loss_aggregation: str
) -> None:
    """Assign generated-token mass for the global token-mean objective.

    PPO divides by a fixed window denominator, including across accumulation
    cycles and ranks. Packing, compaction and prompt length cannot change a
    target token's coefficient. The explicit span_mean alternative retains
    its historical equal-episode weighting; episode_mean also gives every
    episode unit mass but spreads it over the episode's generated tokens,
    so every token of one episode carries 1 / (its token count).
    """
    if loss_aggregation in ("token_mean", "sequence_sum_fixed", "token_sum_fixed"):
        for sample in samples:
            sample.sequence_weight = float(sum(sample.completion_mask))
        return
    episode_totals: dict[str, int] = {}

    def mass(sample: TokenizedSample) -> int:
        if loss_aggregation == "episode_mean":
            return sum(sample.completion_mask)
        return max(1, int(sample.span_count))

    for sample in samples:
        episode_totals[sample.trajectory_id] = episode_totals.get(
            sample.trajectory_id, 0
        ) + mass(sample)
    for sample in samples:
        total = episode_totals[sample.trajectory_id]
        if total > 0:
            sample.sequence_weight = mass(sample) / float(total)


def _log_nesting_divergence(
    previous: TokenizedSample,
    sample: TokenizedSample,
    tokenizer: Any,
) -> None:
    """Decode the first divergent stretch so the re-render normalization that
    breaks prefix stability can be identified from run logs."""
    previous_length = len(previous.input_ids)
    if sample.prompt_token_count < previous_length:
        log.info(
            "nesting diagnostic %s step %d->%d: next prompt SHORTER than previous "
            "sequence (%d < %d) — history was truncated/rewritten",
            sample.trajectory_id,
            previous.step_index,
            sample.step_index,
            sample.prompt_token_count,
            previous_length,
        )
        return
    divergence = next(
        (
            index
            for index in range(previous_length)
            if sample.input_ids[index] != previous.input_ids[index]
        ),
        previous_length,
    )
    window_start = max(0, divergence - 24)
    window_end = min(previous_length, divergence + 24)
    try:
        previous_text = tokenizer.decode(previous.input_ids[window_start:window_end])
        next_text = tokenizer.decode(sample.input_ids[window_start:window_end])
    except Exception:  # noqa: BLE001
        previous_text = next_text = "<decode failed>"
    in_completion = divergence >= previous.prompt_token_count
    log.info(
        "nesting diagnostic %s step %d->%d: first divergence at token %d/%d "
        "(inside previous %s)\n  prev: %r\n  next: %r",
        sample.trajectory_id,
        previous.step_index,
        sample.step_index,
        divergence,
        previous_length,
        "completion" if in_completion else "prompt",
        previous_text,
        next_text,
    )


def _pack_run(run: list[TokenizedSample]) -> TokenizedSample:
    if any(
        s.completion_topk_ids is not None or s.completion_topk_logprobs is not None
        for s in run
    ):
        if any(
            s.completion_topk_ids is None or s.completion_topk_logprobs is None
            for s in run
        ):
            raise ValueError("cannot pack incomplete sampler top-k evidence")
    last = run[-1]
    total_length = len(last.input_ids)
    target_length = max(0, total_length - 1)
    labels = [-100] * total_length
    completion_mask = [0] * total_length
    span_ids = [0] * target_length
    token_advantages = [0.0] * target_length
    token_levels = [0] * target_length
    old_token_logprobs = [0.0] * target_length
    token_logit_bias: dict[int, list[float]] = {}
    total_old_logprob = 0.0
    total_reward = 0.0

    for span_index, sample in enumerate(run, start=1):
        span_start = sample.prompt_token_count
        span_end = len(sample.input_ids)
        for position in range(span_start, span_end):
            completion_mask[position] = 1
            labels[position] = last.input_ids[position]
        for target_position in range(span_start - 1, span_end - 1):
            span_ids[target_position] = span_index
            token_advantages[target_position] = sample.advantage
            token_levels[target_position] = sample.level
            old_token_logprobs[target_position] = sample.old_token_logprobs[
                target_position
            ]
        for token, biases in sample.token_logit_bias.items():
            values = token_logit_bias.setdefault(token, [0.0] * target_length)
            values[span_start - 1 : span_end - 1] = biases[
                span_start - 1 : span_end - 1
            ]
        total_old_logprob += float(sample.old_logprob)
        total_reward += float(sample.reward)

    return TokenizedSample(
        trajectory_id=last.trajectory_id,
        step_index=run[0].step_index,
        input_ids=list(last.input_ids),
        labels=labels,
        attention_mask=[1] * total_length,
        completion_mask=completion_mask,
        reward=total_reward,
        return_to_go=run[0].return_to_go,
        # Keep a nonzero scalar when any span has signal. Losses and metrics
        # use the actual target-aligned advantages, not this filter sentinel.
        advantage=max(run, key=lambda item: abs(item.advantage)).advantage,
        task_id=run[0].task_id,
        policy_iteration=run[0].policy_iteration,
        level=run[0].level,
        prompt_message_count=last.prompt_message_count,
        prompt_token_count=run[0].prompt_token_count,
        old_logprob=total_old_logprob,
        old_token_logprobs=old_token_logprobs,
        token_logit_bias=token_logit_bias,
        completion_topk_ids=(
            np.concatenate([s.completion_topk_ids for s in run])
            if run[0].completion_topk_ids is not None
            else None
        ),
        completion_topk_logprobs=(
            np.concatenate([s.completion_topk_logprobs for s in run])
            if run[0].completion_topk_logprobs is not None
            else None
        ),
        span_ids=span_ids,
        token_advantages=token_advantages,
        token_levels=token_levels,
        span_count=len(run),
        media=last.media,
        media_evidence=last.media_evidence,
    )


def pack_episode_samples(
    samples: list[TokenizedSample],
    *,
    max_sequence_tokens: int,
    tokenizer: Any = None,
    loss_aggregation: str = "span_mean",
) -> tuple[list[TokenizedSample], PackingStats]:
    """Pack per-step samples into nested-episode sequences where possible.

    Samples must arrive grouped by trajectory in step order (build_samples
    emits them that way). Returns the new sample list plus coverage stats."""
    stats = PackingStats(input_samples=len(samples))
    output: list[TokenizedSample] = []

    def flush(run: list[TokenizedSample]) -> None:
        if not run:
            return
        if len(run) == 1:
            stats.passthrough_samples += 1
            output.append(run[0])
            return
        stats.packed_sequences += 1
        stats.packed_spans += len(run)
        output.append(_pack_run(run))

    run: list[TokenizedSample] = []
    for sample in samples:
        # Every packed sequence is bounded by the training budget. Compaction
        # keeps each segment under it; a single step over budget is fatal.
        if len(sample.input_ids) > max_sequence_tokens:
            stats.length_breaks += 1
            raise RuntimeError(
                "episode packing invariant violated: "
                f"{sample.trajectory_id} step {sample.step_index} sequence "
                f"({len(sample.input_ids)} tokens) exceeds max_sequence_tokens="
                f"{max_sequence_tokens}; the rollout context budget must not "
                "exceed the training budget."
            )
        same_trajectory = bool(run) and sample.trajectory_id == run[-1].trajectory_id
        if same_trajectory and getattr(sample, "segment_start", False):
            # Intended compaction boundary: the trajectory continues, but this
            # step's prompt is the fresh compacted prefix (does not nest). Close
            # the current segment and start a new one. The compacted prefix is
            # prompt -> masked, so training only sees post-compaction generations.
            stats.segment_boundaries += 1
            flush(run)
            run = []
        elif same_trajectory:
            # Same trajectory, no compaction flag: MUST nest token-for-token. A
            # break here is a real fork (re-render/truncation) -> fatal.
            if not _nests_after(run[-1], sample):
                stats.nesting_breaks += 1
                if tokenizer is not None:
                    _log_nesting_divergence(run[-1], sample, tokenizer)
                raise RuntimeError(
                    "episode packing invariant violated: "
                    f"{sample.trajectory_id} step {run[-1].step_index}->{sample.step_index} "
                    "does not extend the previous sequence token-for-token. "
                    "Token-continuation rollouts must nest exactly (or set "
                    "segment_start at a compaction); see the nesting diagnostic above."
                )
        elif run:
            flush(run)
            run = []
        run.append(sample)
    flush(run)

    assign_loss_weights(output, loss_aggregation=loss_aggregation)

    stats.output_samples = len(output)
    return output, stats
