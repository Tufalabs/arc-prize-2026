from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import torch

from training.contracts import RolloutTrajectory
from training.rewards import StepSignal

_SIGNAL_EPSILON = 1e-8


@dataclass(slots=True)
class TokenizedSample:
    trajectory_id: str
    step_index: int
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    completion_mask: list[int]
    reward: float
    return_to_go: float
    advantage: float
    # Game the episode was played on (RolloutTrajectory.task_id): the sample
    # budget stratifies its subsample per game to preserve cross-game
    # diversity. Empty only in legacy fixtures.
    task_id: str = ""
    # Policy iteration that generated the episode (stamped by the async
    # collector); the parity oracle checks the window's NEWEST-policy samples
    # so legitimate policy drift on stale episodes cannot masquerade as a
    # recompute-pipeline failure. -1 = unknown (windowed path/fixtures).
    policy_iteration: int = -1
    level: int = 0
    prompt_message_count: int = 0
    prompt_token_count: int = 0
    old_logprob: float = 0.0
    old_token_logprobs: list[float] = field(default_factory=list)
    # Only generated rows are stored; collation scatters them to causal targets.
    completion_topk_ids: np.ndarray | None = None
    completion_topk_logprobs: np.ndarray | None = None
    # Sparse vocabulary columns, each aligned to causal target positions.
    token_logit_bias: dict[int, list[float]] = field(default_factory=dict)
    # Episode packing (training/episode_packing.py): a packed sample carries
    # several completion spans, each with its own advantage. Both lists align
    # to causal TARGET positions (length len(input_ids) - 1). span_ids: 0 =
    # not a target, k >= 1 = k-th span; PPO losses average per span so a span
    # is weighted exactly like an unpacked sample. None -> single-span sample
    # derived from `advantage`/`completion_mask` at collate time.
    span_ids: list[int] | None = None
    token_advantages: list[float] | None = None
    # Reporting metadata for the same target positions. Packed spans can
    # belong to different levels; this does not affect the training loss.
    token_levels: list[int] | None = None
    span_count: int = 1
    # Global token_mean: generated-token count before inclusion correction.
    # The trainer uses one fixed window denominator across microbatches/ranks.
    # Explicit span_mean retains its historical episode-normalized span mass.
    sequence_weight: float = 1.0
    # True when the rollout compacted its context immediately before this step,
    # so this step's prompt is the fresh compacted prefix and does NOT nest into
    # the previous step. Episode packing starts a new segment here instead of
    # treating it as a nesting violation. The compacted prefix is prompt (and
    # therefore masked), so training only sees post-compaction generations.
    segment_start: bool = False
    media: dict[str, Any] | None = None
    media_evidence: dict[str, Any] | None = None


def build_samples(
    rollouts: list[RolloutTrajectory],
    *,
    per_rollout_signals: list[list[StepSignal]],
) -> list[TokenizedSample]:
    samples: list[TokenizedSample] = []

    for rollout, per_step_signals in zip(rollouts, per_rollout_signals, strict=True):
        rollout_policy_iteration = (
            int(rollout.metadata.get("policy_iteration", -1))
            if isinstance(rollout.metadata, dict)
            else -1
        )
        for step, signal in zip(rollout.steps, per_step_signals, strict=True):
            prompt_ids = [int(token_id) for token_id in step.prompt_token_ids]
            completion_ids = [int(token_id) for token_id in step.completion_token_ids]
            if not completion_ids:
                if _step_requires_completion_tokens(step.metadata, signal):
                    raise ValueError(
                        "Rollout step is missing completion token ids for a trainable action/reward signal "
                        f"(trajectory_id={rollout.trajectory_id}, step_index={step.step_index}, "
                        f"reward={signal.reward}, return_to_go={signal.return_to_go}, "
                        f"advantage={signal.advantage})."
                    )
                continue
            if not prompt_ids:
                raise ValueError(
                    "Rollout step is missing prompt token ids "
                    f"(trajectory_id={rollout.trajectory_id}, step_index={step.step_index})."
                )

            input_ids = prompt_ids + completion_ids
            prompt_token_count = len(prompt_ids)
            labels = ([-100] * prompt_token_count) + completion_ids
            attention_mask = [1] * len(input_ids)
            completion_mask = ([0] * prompt_token_count) + ([1] * len(completion_ids))
            old_token_logprobs = _target_aligned_logprobs(
                prompt_token_count=prompt_token_count,
                completion_token_ids=completion_ids,
                completion_logprobs=step.completion_logprobs,
            )
            if not (
                len(input_ids)
                == len(labels)
                == len(attention_mask)
                == len(completion_mask)
            ):
                raise ValueError(
                    "Tokenized rollout sample fields must have matching sequence lengths "
                    f"(trajectory_id={rollout.trajectory_id}, step_index={step.step_index}, "
                    f"input_ids={len(input_ids)}, labels={len(labels)}, "
                    f"attention_mask={len(attention_mask)}, completion_mask={len(completion_mask)})."
                )
            if len(old_token_logprobs) != len(input_ids) - 1:
                raise ValueError(
                    "Tokenized rollout sample old logprobs must align to causal target positions "
                    f"(trajectory_id={rollout.trajectory_id}, step_index={step.step_index}, "
                    f"old_token_logprobs={len(old_token_logprobs)}, expected={len(input_ids) - 1})."
                )

            samples.append(
                TokenizedSample(
                    trajectory_id=rollout.trajectory_id,
                    step_index=step.step_index,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    completion_mask=completion_mask,
                    prompt_message_count=len(step.prompt_messages),
                    prompt_token_count=prompt_token_count,
                    reward=signal.reward,
                    return_to_go=signal.return_to_go,
                    advantage=signal.advantage,
                    task_id=str(rollout.task_id or ""),
                    policy_iteration=rollout_policy_iteration,
                    level=_sample_level(
                        dict(step.metadata), rollout_metadata=dict(rollout.metadata)
                    ),
                    old_logprob=float(sum(old_token_logprobs)),
                    old_token_logprobs=old_token_logprobs,
                    completion_topk_ids=step.completion_topk_ids,
                    completion_topk_logprobs=step.completion_topk_logprobs,
                    token_logit_bias={
                        int(token): [0.0] * (prompt_token_count - 1)
                        + [float(bias)] * len(completion_ids)
                        for token, bias in step.metadata.get("logit_bias", {}).items()
                    },
                    segment_start=bool(
                        isinstance(step.metadata, dict)
                        and step.metadata.get("segment_start")
                    ),
                    media=step.media,
                    media_evidence=step.media_evidence,
                )
            )
    return samples


def _step_requires_completion_tokens(
    metadata: dict[str, Any],
    signal: StepSignal,
) -> bool:
    del metadata
    return any(
        abs(float(value)) > _SIGNAL_EPSILON
        for value in (signal.reward, signal.return_to_go, signal.advantage)
    )


def _target_aligned_logprobs(
    *,
    prompt_token_count: int,
    completion_token_ids: list[int],
    completion_logprobs: list[Any],
) -> list[float]:
    parsed_logprobs = [_logprob_value(item) for item in completion_logprobs]
    if len(parsed_logprobs) != len(completion_token_ids):
        raise ValueError(
            "Rollout step completion logprobs must match completion token ids "
            f"(completion_logprobs={len(parsed_logprobs)}, completion_token_ids={len(completion_token_ids)})."
        )
    for index, (token_id, logprob_item) in enumerate(
        zip(completion_token_ids, completion_logprobs, strict=True)
    ):
        logged_token_id = _logprob_token_id(logprob_item)
        if logged_token_id is None:
            continue
        if logged_token_id != int(token_id):
            raise ValueError(
                "Rollout step completion logprob token ids must match completion token ids "
                f"(index={index}, completion_token_id={int(token_id)}, logprob_token_id={logged_token_id})."
            )

    target_logprobs: list[float] = []
    for token_position in range(
        max(0, prompt_token_count + len(completion_token_ids) - 1)
    ):
        input_position = token_position + 1
        if input_position < prompt_token_count:
            target_logprobs.append(0.0)
            continue
        completion_index = input_position - prompt_token_count
        target_logprobs.append(float(parsed_logprobs[completion_index]))
    return target_logprobs


def _logprob_value(item: Any) -> float:
    if isinstance(item, dict):
        item = item.get("logprob")
    return float(item)


def _logprob_token_id(item: Any) -> int | None:
    if not isinstance(item, dict) or "token_id" not in item:
        return None
    try:
        return int(item["token_id"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Rollout step completion logprob token_id must be an integer, got {item['token_id']!r}."
        ) from exc


def _sample_level(
    step_metadata: dict[str, Any], *, rollout_metadata: dict[str, Any] | None = None
) -> int:
    stage = step_metadata.get("stage", {}) if isinstance(step_metadata, dict) else {}
    if not isinstance(stage, dict):
        return 0
    completed_before = stage.get("levels_completed_before_decision")
    if completed_before is not None:
        try:
            return max(1, int(completed_before) + 1)
        except (TypeError, ValueError):
            return 0
    explicit_level = stage.get("level")
    if explicit_level is not None:
        try:
            return max(1, int(explicit_level))
        except (TypeError, ValueError):
            return 0
    levels_remaining = stage.get("levels_remaining")
    if levels_remaining is not None and isinstance(rollout_metadata, dict):
        try:
            remaining = max(0, int(levels_remaining))
            total = int(rollout_metadata.get("game_total_levels"))
        except (TypeError, ValueError):
            return 0
        if total > 0:
            return max(1, total - remaining + 1)
    return 0


def _target_span_fields(sample: TokenizedSample) -> tuple[list[int], list[float]]:
    """Target-aligned (len(input_ids) - 1) span ids and advantages.

    Packed samples carry these explicitly; single-span samples derive them
    from `completion_mask` and the scalar `advantage`."""
    target_len = max(0, len(sample.input_ids) - 1)
    if sample.span_ids is not None and sample.token_advantages is not None:
        if (
            len(sample.span_ids) != target_len
            or len(sample.token_advantages) != target_len
        ):
            raise ValueError(
                "Packed sample span fields must align to causal target positions "
                f"(trajectory_id={sample.trajectory_id}, step_index={sample.step_index}, "
                f"span_ids={len(sample.span_ids)}, token_advantages={len(sample.token_advantages)}, "
                f"expected={target_len})."
            )
        return list(sample.span_ids), list(sample.token_advantages)
    span_ids = [
        1 if sample.completion_mask[position + 1] else 0
        for position in range(target_len)
    ]
    token_advantages = [sample.advantage if span_id else 0.0 for span_id in span_ids]
    return span_ids, token_advantages


def collate_samples(
    samples: list[TokenizedSample], *, pad_token_id: int, processor: Any = None
) -> dict[str, torch.Tensor]:
    max_len = max(len(sample.input_ids) for sample in samples)
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_completion_mask = []
    advantages = []
    old_logprobs = []
    old_token_logprobs = []
    batch_span_ids = []
    batch_token_advantages = []
    span_counts = []
    sequence_weights = []

    for sample in samples:
        pad = max_len - len(sample.input_ids)
        batch_input_ids.append(sample.input_ids + ([pad_token_id] * pad))
        batch_labels.append(sample.labels + ([-100] * pad))
        batch_attention_mask.append(sample.attention_mask + ([0] * pad))
        batch_completion_mask.append(sample.completion_mask + ([0] * pad))
        advantages.append(sample.advantage)
        old_logprobs.append(getattr(sample, "old_logprob", 0.0))
        target_len = max(0, len(sample.input_ids) - 1)
        old_token_values = list(getattr(sample, "old_token_logprobs", []))[:target_len]
        old_target_pad = max(0, (max_len - 1) - len(old_token_values))
        old_token_logprobs.append(old_token_values + ([0.0] * old_target_pad))
        span_ids, token_advantages = _target_span_fields(sample)
        span_pad = max(0, (max_len - 1) - len(span_ids))
        batch_span_ids.append(span_ids + ([0] * span_pad))
        batch_token_advantages.append(token_advantages + ([0.0] * span_pad))
        span_counts.append(max(1, int(getattr(sample, "span_count", 1))))
        sequence_weights.append(float(getattr(sample, "sequence_weight", 1.0)))

    batch = {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "completion_mask": torch.tensor(batch_completion_mask, dtype=torch.bool),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
        "old_token_logprobs": torch.tensor(old_token_logprobs, dtype=torch.float32),
        "span_ids": torch.tensor(batch_span_ids, dtype=torch.long),
        "token_advantages": torch.tensor(batch_token_advantages, dtype=torch.float32),
        "span_counts": torch.tensor(span_counts, dtype=torch.long),
        "sequence_weights": torch.tensor(sequence_weights, dtype=torch.float32),
    }
    if any(
        sample.completion_topk_ids is not None
        or sample.completion_topk_logprobs is not None
        for sample in samples
    ):
        k = next(
            s.completion_topk_ids.shape[1]
            for s in samples
            if s.completion_topk_ids is not None
        )
        ids = torch.zeros((len(samples), max_len - 1, k), dtype=torch.int64)
        logprobs = torch.zeros_like(ids, dtype=torch.float32)
        for row, sample in enumerate(samples):
            mask = batch["completion_mask"][row, 1:]
            shape = (int(mask.sum()), k)
            if (
                sample.completion_topk_ids is None
                or sample.completion_topk_logprobs is None
                or sample.completion_topk_ids.shape != shape
                or sample.completion_topk_logprobs.shape != shape
            ):
                raise ValueError(
                    "sampler top-k must align exactly with completion tokens"
                )
            ids[row, mask] = torch.tensor(sample.completion_topk_ids, dtype=torch.int64)
            logprobs[row, mask] = torch.tensor(
                sample.completion_topk_logprobs, dtype=torch.float32
            )
        batch["sampler_topk_ids"] = ids
        batch["sampler_topk_logprobs"] = logprobs
    if any(sample.token_logit_bias for sample in samples):
        tokens = sorted(
            {token for sample in samples for token in sample.token_logit_bias}
        )
        values = torch.zeros(
            (len(samples), max_len - 1, len(tokens)), dtype=torch.float32
        )
        for row, sample in enumerate(samples):
            for column, token in enumerate(tokens):
                biases = sample.token_logit_bias.get(token)
                if biases is not None:
                    if len(biases) != len(sample.input_ids) - 1:
                        raise ValueError(
                            "sample logit biases must align to causal target positions"
                        )
                    values[row, : len(biases), column] = torch.tensor(
                        biases, dtype=torch.float32
                    )
        batch["logit_bias_token_ids"] = torch.tensor(tokens, dtype=torch.long)
        batch["logit_bias_values"] = values
    if any(
        sample.media is None and sample.media_evidence is not None for sample in samples
    ):
        raise ValueError("Text-only sample carries unexpected media evidence")
    if any(sample.media is not None for sample in samples):
        if processor is None:
            raise ValueError("Multimodal samples require the pinned image processor")
        from training.multimodal import prepare_training_media

        batch.update(
            prepare_training_media(
                [sample.input_ids for sample in samples],
                [sample.media for sample in samples],
                [sample.media_evidence for sample in samples],
                processor,
            )
        )
    return batch
