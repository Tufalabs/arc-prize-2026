from __future__ import annotations

import copy
import datetime
import hashlib
import inspect
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.utils.checkpoint
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AttentionInterface,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from training.run_artifacts import resolve_runtime_adapter_root
from training.config import ParityThresholds, TrainingConfig
from training.contracts import RolloutTrajectory
from training.dataset import TokenizedSample, build_samples, collate_samples
from training.episode_packing import assign_loss_weights, pack_episode_samples
from training.policy_update import (
    dppo_binary_tv_objective,
    merge_policy_update_statistics,
    policy_update_statistics,
    summarize_policy_update_statistics,
)
from training.ddp_worker import (
    flatten_grads,
    flatten_tensors,
    round_robin_split,
)
from training.memory_profile import NullMemoryLogger
from training.metrics import episode_levels_completed
from training.rewards import AdvantageBatchStats, StepSignal
from training.grpo import compute_grpo_signal_batch
from training.runtime_lora import export_runtime_lora_adapter
from training.shutdown import raise_if_shutdown_requested, shutdown_at_safe_boundary
from training.baseline_batch import (
    BaselineSignalBatch,
    BaselineUpdateStats,
)

from training.state_critic import (
    StateCriticBaseline,
    StateCriticSignalBatch,
    StateCriticUpdateStats,
    compute_state_critic_step_signals,
)

from training.verify_fa4 import (
    register_fa4_contiguous_inputs,
    require_fa4_distribution,
)


log = logging.getLogger(__name__)
_TRAINER_STATE_VERSION = 1
_NONZERO_SIGNAL_EPSILON = 1e-8
# Frozen probe batch for the fixed-population entropy series (probe_entropy):
# the first K tokenized samples of the first window with retained samples,
# persisted so the population stays identical across the whole run (resumes
# included).
_PROBE_STATE_FILENAME = "probe_states.pt"
_PROBE_STATE_SAMPLE_COUNT = 8
_PROBE_STATE_VERSION = 1
# Adapter exports younger than this are never pruned (in-flight 30-min
# episodes + eval cycles pin adapters; see _cleanup_old_adapters).
_ADAPTER_RETENTION_MIN_AGE_SECONDS = 7200.0
_ADAPTER_RETENTION_HARD_CAP = 40
# Deliberate long-term snapshots (PINNED marker written at save time by
# training.pinned_adapter_interval) are exempt from every retention rule up
# to this many newest pinned dirs (~33 GiB at ~830 MB per 27B LoRA export);
# older pinned dirs fall back to the normal bounds so a long run cannot fill
# the checkpoints volume.
_PINNED_ADAPTER_CAP = 40
_PINNED_ADAPTER_MARKER = "PINNED"
# Check a bounded subset of real sampled tokens, preserving their complete
# causal prefixes. Prefer raw start/reset requests over long packed episodes.
# A smaller window still checks every available token up to the sample cap.
# "summary" is a draw sampled under a logit bias (the summary token ban): its
# targets are the only ones the learner rescores with a bias, so the rotation
# checks them explicitly rather than when a long continuation happens to be one.
_PARITY_MIN_CHECK_TOKENS = 128
_PARITY_MAX_CHECK_BATCHES = 8
_PARITY_CONTEXT_CYCLE = ("initial", "reset", "summary", "continuation")
_PPO_LOG_RATIO_CLAMP = 20.0
_DDP_SHUTDOWN_TIMEOUT_SECONDS = 90.0
_DDP_HEALTH_POLL_SECONDS = 1.0
# Tighter clamp for the k3 KL estimator: exp() of the log-ratio appears
# directly in the estimator, so the PPO clamp of 20 would overflow it.
_KL_LOG_RATIO_CLAMP = 5.0
# Window trust region (training.max_window_clip_fraction): learning-rate
# backoff applied after a rejected window, its recovery multiplier on
# accepted windows, the floor for the scale, and the consecutive-rejection
# abort threshold (a persistent rejection loop means the step size cannot
# be made safe and needs a human).
_TRUST_REGION_LR_BACKOFF = 0.5
_TRUST_REGION_LR_RECOVERY = 1.25
_TRUST_REGION_MIN_LR_SCALE = 1.0 / 16.0
_TRUST_REGION_MAX_CONSECUTIVE_REJECTIONS = 10
_EXTRA_CYCLE_SLACK_FACTOR = 1.15
# Leave room after the indivisible cycle for adapter export and handoff; live
# easy-run exports currently take roughly 7-15 seconds.
_EXTRA_CYCLE_MIN_SLACK_SECONDS = 20.0
_DROPOUT_MODULE_TYPES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)

PolicyHiddenPlan = tuple[int, list[int], list[int], list[int]]
PolicyHiddenBatch = list[PolicyHiddenPlan]
PolicyHiddenFeatures = tuple[list[tuple[int, int]], torch.Tensor]


def _extra_cycle_block_reason(
    *,
    next_window_ready: Callable[[], bool],
    next_window_seconds_remaining: Callable[[], float | None] | None,
    estimated_cycle_seconds: float,
) -> str:
    """Return why another indivisible optimizer cycle must not start."""
    if next_window_ready():
        return "window_ready"
    if next_window_seconds_remaining is None:
        return "rollout_eta_unavailable"
    remaining = next_window_seconds_remaining()
    if remaining is None:
        return "rollout_eta_unavailable"
    remaining = float(remaining)
    estimated_cycle_seconds = float(estimated_cycle_seconds)
    if (
        not math.isfinite(remaining)
        or not math.isfinite(estimated_cycle_seconds)
        or estimated_cycle_seconds <= 0.0
    ):
        return "rollout_eta_unavailable"
    required_slack = max(
        estimated_cycle_seconds * _EXTRA_CYCLE_SLACK_FACTOR,
        estimated_cycle_seconds + _EXTRA_CYCLE_MIN_SLACK_SECONDS,
    )
    if remaining < required_slack:
        return "insufficient_rollout_slack"
    return ""


def _disable_dropout_modules(module: torch.nn.Module) -> int:
    disabled = 0
    for child in module.modules():
        if isinstance(child, _DROPOUT_MODULE_TYPES):
            child.p = 0.0
            disabled += 1
    return disabled


def _enable_input_require_grads_for_checkpointing(model: Any) -> None:
    enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
    if not callable(enable_input_require_grads):
        raise RuntimeError(
            "gradient_checkpointing with LoRA requires enable_input_require_grads() "
            "on the loaded policy model."
        )
    enable_input_require_grads()


def _enable_activation_checkpointing(model: Any, *, preserve_rng_state: bool) -> None:
    enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable_checkpointing):
        raise RuntimeError(
            "gradient_checkpointing is enabled, but the loaded policy model does "
            "not expose gradient_checkpointing_enable()."
        )
    signature = inspect.signature(enable_checkpointing)
    parameters = signature.parameters
    accepts_checkpointing_kwargs = "gradient_checkpointing_kwargs" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if not accepts_checkpointing_kwargs:
        raise RuntimeError(
            "gradient_checkpointing requires a transformers model whose "
            "gradient_checkpointing_enable() accepts gradient_checkpointing_kwargs; "
            "non-reentrant checkpointing is required for this LoRA trainer."
        )
    enable_checkpointing(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "preserve_rng_state": preserve_rng_state,
        }
    )
    is_gradient_checkpointing = getattr(model, "is_gradient_checkpointing", None)
    if is_gradient_checkpointing is not True:
        raise RuntimeError(
            "gradient_checkpointing_enable() returned without enabling model "
            f"checkpointing (is_gradient_checkpointing={is_gradient_checkpointing!r})."
        )

    _enable_input_require_grads_for_checkpointing(model)


_LENGTH_PACKING_MAX_PADDING_WASTE_FRACTION = 0.25


def _save_peft_adapter_without_model_card(model: Any, output_dir: Path) -> None:
    # PEFT currently writes a README/model card during save_pretrained(). Some
    # cluster nodes have a broken huggingface_hub install missing that template,
    # which should not block checkpoint export.
    original_create_model_card = getattr(model, "create_or_update_model_card", None)
    had_instance_override = "create_or_update_model_card" in getattr(
        model, "__dict__", {}
    )
    if original_create_model_card is None:
        model.save_pretrained(output_dir, safe_serialization=True)
        return

    setattr(model, "create_or_update_model_card", lambda *_args, **_kwargs: None)
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    finally:
        if had_instance_override:
            setattr(model, "create_or_update_model_card", original_create_model_card)
        else:
            delattr(model, "create_or_update_model_card")


def _filter_zero_signal_samples(
    samples: list[TokenizedSample],
    *,
    previous_policy_kl_coef: float,
    reference_kl_coef: float,
    entropy_coef: float,
) -> tuple[list[TokenizedSample], int]:
    """Skip zero advantages only when the actor has no auxiliary loss.

    KL and entropy gradients do not depend on the advantage. Even a KL term
    that is zero before the first step can become nonzero during later
    accumulation cycles or extra passes, so retain its complete population.
    The return EMA is updated independently of this actor filter.
    """
    if any(
        coefficient > 0.0
        for coefficient in (
            previous_policy_kl_coef,
            reference_kl_coef,
            entropy_coef,
        )
    ):
        return samples, 0
    retained = [
        sample
        for sample in samples
        if abs(float(sample.advantage)) > _NONZERO_SIGNAL_EPSILON
    ]
    return retained, len(samples) - len(retained)


def _assert_packing_identity(stats: Any, episodes: int) -> None:
    """One packed sequence per episode plus one per segment boundary — the
    invariant behind "one datapoint per compaction segment". The harness
    marks every step continued-by-tokens or rendered-afresh, the assembly
    opens a boundary only on a fresh render, and packing refuses a nesting
    break that is not a boundary; anything else here is a bug in that
    chain, not a data property."""
    expected = int(episodes) + int(stats.segment_boundaries)
    if int(stats.output_samples) != expected:
        raise RuntimeError(
            "episode packing identity violated: "
            f"{stats.output_samples} sequences for {episodes} episodes with "
            f"{stats.segment_boundaries} segment boundaries (expected {expected})"
        )


def _zero_signal_denominator_scale(
    *, retained_count: float, filtered_count: float
) -> float:
    """Window-level loss-denominator factor undoing the zero-signal filter's
    effect on the effective step size.

    The PPO loss averages over RETAINED spans only, so a window with many
    filtered quiet turns would otherwise take a proportionally larger step —
    an unchosen coupling. Multiplying every cycle's denominator by
    (retained + filtered) / retained makes the aggregate update exactly as if
    the zero-gradient spans were still present in the average.
    """
    retained_count = max(0.0, float(retained_count))
    filtered_count = max(0.0, float(filtered_count))
    if retained_count <= 0 or filtered_count <= 0:
        return 1.0
    return float(retained_count + filtered_count) / float(retained_count)


def _fixed_token_sum_scale(
    *, rollout_count: int, normalization_tokens: int, window_token_mass: float
) -> float:
    """Denominator factor replacing the sampled token count with N * L_ref.

    token_mean divides the window by its generated-token mass T. Multiplying
    every cycle denominator by (N * L_ref) / T divides the same token sum by
    episodes * loss_normalization_tokens instead, for every accumulation
    cycle, extra cycle and gradient replica, independent of sampled lengths.
    Zero-advantage and empty episodes stay in N.
    """
    if rollout_count < 1 or normalization_tokens < 1:
        raise ValueError("token_sum_fixed requires episodes and a positive L_ref")
    if window_token_mass <= 0:
        return 1.0
    return float(rollout_count) * float(normalization_tokens) / float(window_token_mass)


def _assert_samples_fit_training_budgets(
    samples: list[TokenizedSample],
    *,
    max_sequence_tokens: int,
) -> None:
    # Generations are uncapped; the packed-sequence budget is the only bound.
    limit = max(1, int(max_sequence_tokens))
    over_sequence = [sample for sample in samples if len(sample.input_ids) > limit]
    if not over_sequence:
        return
    examples = [
        f"{sample.trajectory_id}:step={sample.step_index}:"
        f"input={len(sample.input_ids)}:completion={sum(sample.completion_mask)}"
        for sample in over_sequence[:5]
    ]
    raise ValueError(
        "Training samples must fit the configured token budget before PPO optimization. "
        f"Found {len(over_sequence)} samples over training.max_sequence_tokens={limit}. "
        f"Examples: {', '.join(examples)}. Lower inference.agent_context_window or "
        "increase training.max_sequence_tokens before collecting this rollout window."
    )


def _sample_budget_count(
    eligible_count: int,
    *,
    fraction: float,
    step_multiple: int,
) -> int:
    eligible_count = max(0, int(eligible_count))
    if eligible_count <= 0:
        return 0
    fraction = float(fraction)
    if fraction <= 0.0:
        return 0
    fraction = min(1.0, fraction)
    if fraction >= 1.0:
        return eligible_count

    requested = math.floor(eligible_count * fraction)
    step_multiple = max(1, int(step_multiple))
    if eligible_count >= step_multiple:
        requested = max(step_multiple, requested)
        return min(eligible_count, (requested // step_multiple) * step_multiple)
    return max(1, min(eligible_count, requested))


def _stratified_episode_sample(
    samples: list[TokenizedSample],
    budget: int,
) -> list[TokenizedSample]:
    """Uniform random sample stratified by EPISODE (trajectory_id).

    Every episode with eligible segments keeps at least one whenever the
    budget allows (the diversity floor -- an episode that vanishes entirely
    contributes zero gradient, and with one episode per game per window this
    is also the per-game floor). The remaining budget is spread by
    largest-remainder quotas proportional to each episode's eligible count,
    capped at the episode's capacity, with a uniform draw inside each
    episode. Inclusion probabilities are corrected by the caller.
    """
    if budget >= len(samples):
        return list(samples)
    if budget <= 0:
        return []
    by_episode: dict[str, list[TokenizedSample]] = {}
    for sample in samples:
        by_episode.setdefault(str(sample.trajectory_id), []).append(sample)
    episode_ids = list(by_episode)
    quotas: dict[str, int] = {episode_id: 0 for episode_id in episode_ids}
    if budget >= len(episode_ids):
        for episode_id in episode_ids:
            quotas[episode_id] = 1
    else:
        for episode_id in random.sample(episode_ids, budget):
            quotas[episode_id] = 1
    spare = budget - sum(quotas.values())
    if spare > 0:
        capacity = {
            episode_id: len(group) - quotas[episode_id]
            for episode_id, group in by_episode.items()
        }
        total_capacity = sum(capacity.values())
        remainders: list[tuple[float, float, str]] = []
        for episode_id, cap in capacity.items():
            if total_capacity <= 0:
                break
            exact = spare * cap / total_capacity
            take = min(cap, int(exact))
            quotas[episode_id] += take
            if take < cap:
                remainders.append((exact - take, random.random(), episode_id))
        leftover = budget - sum(quotas.values())
        for _fraction, _tie, episode_id in sorted(remainders, reverse=True):
            if leftover <= 0:
                break
            if quotas[episode_id] < len(by_episode[episode_id]):
                quotas[episode_id] += 1
                leftover -= 1
    retained: list[TokenizedSample] = []
    for episode_id, group in by_episode.items():
        retained.extend(random.sample(group, min(quotas[episode_id], len(group))))
    return retained


def _trainer_policy_iteration_at_window_start(iteration_index: int) -> int:
    """Adapter iteration the trainer weights correspond to when iteration N
    begins optimizing.

    Iteration N-1 exports adapter_iter_{N-1} at its END (training/cli.py
    publishes it before iteration N's train_window is entered), so at the
    start of iteration N the trainer already holds adapter_iter_{N-1}. Only
    the very first window (N=0) runs on the initialized LoRA policy (-1).
    """
    return max(-1, int(iteration_index) - 1)


def _policy_lag_for_samples(
    samples: list[TokenizedSample], *, trainer_policy_iteration: int
) -> int:
    """Return the trainer-to-rollout lag represented by a coherent sample set."""
    policy_iterations = {
        int(getattr(sample, "policy_iteration", -1)) for sample in samples
    }
    if len(policy_iterations) != 1:
        raise RuntimeError(
            "PPO samples contain mixed rollout policy iterations: "
            + ", ".join(str(value) for value in sorted(policy_iterations))
        )
    rollout_policy_iteration = next(iter(policy_iterations), -1)
    lag = int(trainer_policy_iteration) - rollout_policy_iteration
    if lag < 0:
        raise RuntimeError(
            "PPO samples were generated by a policy newer than the trainer "
            f"(trainer={trainer_policy_iteration}, rollout={rollout_policy_iteration})."
        )
    return lag


def _parity_context_kind(sample: TokenizedSample) -> str:
    if any(any(values) for values in sample.token_logit_bias.values()):
        return "summary"
    if sample.step_index == 0:
        return "initial"
    return "reset" if sample.segment_start else "continuation"


def _parity_check_pool(
    samples: list[TokenizedSample], *, policy_iteration: int = -1
) -> list[TokenizedSample]:
    """Rotate exact-prefix coverage without RNG or checkpoint state changes.

    Keep initial checks cheap, and check post-compaction requests, biased
    summary draws and a long continuation every fourth policy each. Missing
    kinds fall back to the cheapest available evidence, never skip validation.
    """
    newest_policy = max(
        (int(getattr(sample, "policy_iteration", -1)) for sample in samples),
        default=-1,
    )
    pool = [
        sample
        for sample in samples
        if int(getattr(sample, "policy_iteration", -1)) == newest_policy
    ] or list(samples)

    target = _PARITY_CONTEXT_CYCLE[(policy_iteration + 1) % len(_PARITY_CONTEXT_CYCLE)]

    def cost(sample: TokenizedSample) -> tuple:
        positions = [i for i, mask in enumerate(sample.completion_mask) if mask]
        checked = min(len(positions), _PARITY_MIN_CHECK_TOKENS)
        prefix_length = positions[checked - 1] + 1 if checked else len(sample.input_ids)
        kind = _parity_context_kind(sample)
        long_context = target == kind == "continuation"
        # Prefer enough evidence in one forward over a nearly empty deadline
        # reply. Among those requests, exercise the longest causal prefix.
        rotation = hashlib.sha256(
            f"{policy_iteration}:{sample.trajectory_id}:{sample.step_index}".encode()
        ).digest()
        return (
            kind != target,
            not checked,
            checked < _PARITY_MIN_CHECK_TOKENS if long_context else False,
            -prefix_length if long_context else prefix_length / max(1, checked),
            prefix_length,
            rotation,
        )

    pool.sort(key=cost)
    return pool


def _parity_prefix(sample: TokenizedSample, token_budget: int) -> TokenizedSample:
    """Bound sampled targets while preserving their context and media witness."""
    positions = [i for i, mask in enumerate(sample.completion_mask) if mask]
    if not positions or token_budget <= 0:
        raise ValueError("parity prefix requires at least one sampled target")
    target_end = positions[min(len(positions), token_budget) - 1] + 1
    # Packed samples carry the last request's media witness. Keep its whole
    # prompt reproducible, but do not score later targets retained as context.
    end = max(target_end, sample.media["prompt_length"]) if sample.media else target_end
    context_tail = end - target_end
    return replace(
        sample,
        input_ids=sample.input_ids[:end],
        labels=sample.labels[:target_end] + [-100] * context_tail,
        attention_mask=sample.attention_mask[:end],
        completion_mask=sample.completion_mask[:target_end] + [0] * context_tail,
        old_token_logprobs=(
            sample.old_token_logprobs[: target_end - 1] + [0.0] * context_tail
        ),
        old_logprob=sum(sample.old_token_logprobs[: target_end - 1]),
        completion_topk_ids=(
            None
            if sample.completion_topk_ids is None
            else sample.completion_topk_ids[:token_budget]
        ),
        completion_topk_logprobs=(
            None
            if sample.completion_topk_logprobs is None
            else sample.completion_topk_logprobs[:token_budget]
        ),
        token_logit_bias={
            token: values[: target_end - 1] + [0.0] * context_tail
            for token, values in sample.token_logit_bias.items()
        },
        span_ids=(
            None
            if sample.span_ids is None
            else sample.span_ids[: target_end - 1] + [0] * context_tail
        ),
        token_advantages=(
            None
            if sample.token_advantages is None
            else sample.token_advantages[: target_end - 1] + [0.0] * context_tail
        ),
        token_levels=(
            None
            if sample.token_levels is None
            else sample.token_levels[: target_end - 1] + [0] * context_tail
        ),
    )


def _apply_inclusion_weights(
    eligible: list[TokenizedSample],
    kept: list[TokenizedSample],
) -> None:
    """Horvitz-Thompson correction for the subsample's inclusion rates.

    Correct both uniform episode selection (when the budget is smaller than
    the episode count) and uniform segment selection within each episode.
    Keep these weights with the pre-subsample denominator: renormalizing the
    survivors would undo the correction for unequal segment token masses.
    """
    kept_by_episode: dict[str, int] = {}
    for sample in kept:
        kept_by_episode[sample.trajectory_id] = (
            kept_by_episode.get(sample.trajectory_id, 0) + 1
        )
    eligible_by_episode: dict[str, int] = {}
    for sample in eligible:
        eligible_by_episode[sample.trajectory_id] = (
            eligible_by_episode.get(sample.trajectory_id, 0) + 1
        )
    episode_factor = len(eligible_by_episode) / max(1, len(kept_by_episode))
    for sample in kept:
        kept_count = kept_by_episode[sample.trajectory_id]
        sample.sequence_weight = float(sample.sequence_weight) * (
            episode_factor * eligible_by_episode[sample.trajectory_id] / kept_count
        )


def _subsample_train_samples(
    samples: list[TokenizedSample],
    *,
    fraction: float,
    step_multiple: int,
) -> tuple[list[TokenizedSample], int]:
    budget_count = _sample_budget_count(
        len(samples),
        fraction=fraction,
        step_multiple=step_multiple,
    )
    if budget_count >= len(samples):
        return samples, 0
    if budget_count <= 0:
        return [], len(samples)

    # Sample independently of outcome. Always retaining reward-bearing
    # segments while dropping failures changes the episode distribution;
    # within-episode weights cannot correct that selection bias.
    retained = _stratified_episode_sample(samples, budget_count)
    _apply_inclusion_weights(samples, retained)
    random.shuffle(retained)
    return retained, len(samples) - len(retained)


def _balanced_micro_batch_cycles(
    micro_batches: list[list[TokenizedSample]],
    *,
    cycle_size: int,
) -> list[list[list[TokenizedSample]]]:
    """Partition micro-batches into optimizer cycles of near-equal size.

    A greedy `[i : i + cycle_size]` split leaves a short tail cycle (e.g.
    50 -> 48 + 2). Because Adam is approximately scale-invariant, that tail
    receives a full-magnitude optimizer step built from a couple of samples --
    loss-coefficient scaling cannot shrink it. Splitting 50 into 25 + 25 keeps
    every optimizer step aggregated over a comparable sample count.
    """
    total = len(micro_batches)
    if total == 0:
        return []
    cycle_size = max(1, int(cycle_size))
    cycle_count = max(1, math.ceil(total / cycle_size))
    base, remainder = divmod(total, cycle_count)
    cycles: list[list[list[TokenizedSample]]] = []
    start = 0
    for index in range(cycle_count):
        size = base + (1 if index < remainder else 0)
        cycles.append(micro_batches[start : start + size])
        start += size
    return cycles


def _window_loss_denominator(
    samples: list[TokenizedSample],
    *,
    micro_batch_count: int,
    cycle_size: int,
    expected_mass: float | None = None,
) -> float:
    """Fixed per-cycle denominator derived from the complete window loss mass.

    Normalizing each accumulation cycle by ONLY its own weight mass lets a
    short tail cycle (e.g. a single 0.5-weight compaction segment left over
    after full cycles) receive a full-scale optimizer step -- overweighting
    that episode by up to cycle_size x. Dividing every cycle by the same
    average mass per cycle preserves global token weighting independent of
    sequence lengths, compaction boundaries and cycle placement.
    """
    total_mass = (
        expected_mass
        if expected_mass is not None
        else sum(float(getattr(sample, "sequence_weight", 1.0)) for sample in samples)
    )
    cycles = max(1, math.ceil(max(1, int(micro_batch_count)) / max(1, int(cycle_size))))
    return max(1e-8, total_mass / cycles)


def _rollout_minibatches(
    samples: list[TokenizedSample],
    *,
    rollout_ids: Sequence[str],
    minibatch_rollouts: int,
) -> list[tuple[list[TokenizedSample], int]]:
    """Keep original rollouts intact, including empty/zero-signal rollouts.

    The count is independent of how many training segments an episode has.
    Input order is retained and the final minibatch uses its actual count.
    """
    if minibatch_rollouts <= 0:
        raise ValueError("optimizer_minibatch_rollouts must be positive")
    if len(set(rollout_ids)) != len(rollout_ids):
        raise ValueError("optimizer minibatches require unique rollout identities")
    grouped: dict[str, list[TokenizedSample]] = {key: [] for key in rollout_ids}
    for sample in samples:
        if sample.trajectory_id not in grouped:
            raise ValueError("training sample has no original rollout identity")
        grouped[sample.trajectory_id].append(sample)
    return [
        (
            [sample for key in keys for sample in grouped[key]],
            len(keys),
        )
        for start in range(0, len(rollout_ids), minibatch_rollouts)
        for keys in [rollout_ids[start : start + minibatch_rollouts]]
    ]


def _rollout_loss_batches(
    samples: list[TokenizedSample],
    *,
    rollout_ids: Sequence[str],
    minibatch_rollouts: int,
    aggregation: str,
    normalization_tokens: int,
) -> list[tuple[list[TokenizedSample], float]]:
    groups = _rollout_minibatches(
        samples,
        rollout_ids=rollout_ids,
        minibatch_rollouts=minibatch_rollouts,
    )
    if aggregation == "token_mean":
        # One denominator for the entire wave preserves equal token weight
        # across games, compactions and unequal minibatch lengths.
        denominator = max(1.0, sum(sum(s.completion_mask) for s in samples)) / max(
            1, len(groups)
        )
        return [(group, denominator) for group, _count in groups]
    if aggregation == "sequence_sum_fixed":
        return [(group, count * normalization_tokens) for group, count in groups]
    raise ValueError("rollout minibatching requires token_mean or sequence_sum_fixed")


def _validate_policy_resume_recipe(
    saved: dict[str, Any], config: TrainingConfig
) -> None:
    """Never import optimizer moments from a different DPPO objective."""
    saved_loss = saved.get("policy_loss", "ppo")
    current_loss = getattr(config, "policy_loss", "ppo")
    saved_baseline = saved.get("value_baseline")
    current_baseline = getattr(config, "value_baseline", None)
    if (
        "group_mean" in (saved_baseline, current_baseline)
        and saved_baseline != current_baseline
    ):
        raise RuntimeError(
            "Cannot resume optimizer state across a group_mean baseline change "
            "(different training recipe); start a fresh run or initialize from adapter weights only"
        )
    fixed_aggregations = ("sequence_sum_fixed", "token_sum_fixed")
    if (
        "dppo_binary_tv" not in (saved_loss, current_loss)
        and "score_centered_tis" not in (saved_loss, current_loss)
        and saved.get("loss_aggregation") not in fixed_aggregations
        and getattr(config, "loss_aggregation", None) not in fixed_aggregations
    ):
        return  # Preserve legacy PPO resume behavior.
    fields = (
        "policy_loss",
        "score_centering_top_k",
        "tis_ratio_cap",
        "value_baseline",
        "dppo_tv_threshold",
        "dppo_importance_ratio_cap",
        "loss_aggregation",
        "loss_normalization_tokens",
        "optimizer_minibatch_rollouts",
        "learning_rate",
        "weight_decay",
        "gae_gamma",
        "gae_lambda",
        "advantage_clip",
        "epochs_per_window",
        "previous_policy_kl_coef",
        "reference_kl_coef",
        "entropy_coef",
        "clip_ratio_low",
        "clip_ratio_high",
        "max_window_clip_fraction",
        "max_grad_norm",
        "warmup_steps",
        "rollout_schedule",
        "max_sequence_tokens",
        "disable_dropout",
        "extra_train_cycles_until_window",
        "extra_cycle_max_clip_fraction",
        "model_name",
        "model_revision",
        "multimodal",
    )
    defaults = {
        "policy_loss": "ppo",
        "dppo_tv_threshold": 0.15,
        "dppo_importance_ratio_cap": 5.0,
        "loss_normalization_tokens": 32768,
        "optimizer_minibatch_rollouts": 16,
        "rollout_schedule": "overlapped",
    }
    changed = [
        name
        for name in fields
        if saved.get(name, defaults.get(name)) != getattr(config, name, None)
        # The knob is null when the loss has no fixed denominator; older
        # checkpoints recorded its unused default there.
        and not (
            name == "loss_normalization_tokens" and getattr(config, name, None) is None
        )
    ]
    for name in ("lora", "sample_budget", "entropy_control"):
        setting = getattr(config, name, None)
        if setting is not None and saved.get(name) != asdict(setting):
            changed.append(name)
    if changed:
        raise RuntimeError(
            "Cannot resume group-mean/DPPO training with a different training recipe: "
            + ", ".join(changed)
            + ". Start a fresh run instead of restoring optimizer state."
        )


def _sample_count(
    samples: list[TokenizedSample], predicate: Callable[[TokenizedSample], bool]
) -> int:
    return sum(1 for sample in samples if predicate(sample))


def _sample_fraction(
    samples: list[TokenizedSample], predicate: Callable[[TokenizedSample], bool]
) -> float:
    if not samples:
        return 0.0
    return _sample_count(samples, predicate) / len(samples)


@dataclass(frozen=True)
class SampleAdvantageSummary:
    mean: float
    std: float
    mean_by_level: dict[int, float]
    nonzero_fraction: float
    positive_fraction: float


def _sample_advantage_summary(samples: list[TokenizedSample]) -> SampleAdvantageSummary:
    """Token-weighted supervised advantages, invariant to packing boundaries."""
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    count = nonzero = positive = 0
    mean = squared_deviations = 0.0
    for sample in samples:
        target_count = len(sample.input_ids) - 1
        for field_name in ("token_advantages", "token_levels"):
            values = getattr(sample, field_name)
            if values is not None and len(values) != target_count:
                raise ValueError(f"{field_name} must align to causal target positions")
        for index, supervised in enumerate(sample.completion_mask[1:]):
            if not supervised:
                continue
            advantage = float(
                sample.token_advantages[index]
                if sample.token_advantages is not None
                else sample.advantage
            )
            level = max(
                0,
                int(
                    sample.token_levels[index]
                    if sample.token_levels is not None
                    else sample.level
                ),
            )
            count += 1
            nonzero += abs(advantage) > _NONZERO_SIGNAL_EPSILON
            positive += advantage > 0.0
            delta = advantage - mean
            mean += delta / count
            squared_deviations += delta * (advantage - mean)
            totals[level] = totals.get(level, 0.0) + advantage
            counts[level] = counts.get(level, 0) + 1
    return SampleAdvantageSummary(
        mean=mean,
        std=math.sqrt(max(0.0, squared_deviations / count)) if count else 0.0,
        mean_by_level={
            level: totals[level] / counts[level] for level in sorted(totals)
        },
        nonzero_fraction=nonzero / count if count else 0.0,
        positive_fraction=positive / count if count else 0.0,
    )


def _mean_advantage_by_level(samples: list[TokenizedSample]) -> dict[int, float]:
    return _sample_advantage_summary(samples).mean_by_level


@contextmanager
def _optimizer_pass_timer(
    device: torch.device | str,
    timings: list[tuple[str, Any, Any]],
    name: str,
) -> Iterator[None]:
    """Time actual pass calls without introducing a CUDA synchronization.

    CUDA events are read only after the micro-batch's existing scalar CPU
    transfers have synchronized the current stream, including backward.
    CPU runs use perf_counter. CUDA intervals include kernels/collectives and
    launch gaps on that stream; reference forwards count as forward time and
    checkpoint recomputation counts as backward time.
    """
    if torch.device(device).type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        start.record(stream)
        yield
        end.record(stream)
        timings.append((name, start, end))
    else:
        start = time.perf_counter()
        yield
        timings.append((name, time.perf_counter() - start, None))


@dataclass(slots=True)
class TokenUsageSummary:
    total_completion_tokens: int
    total_env_steps: int
    # Completion tokens per rollout, like the per-step and per-turn averages.
    avg_tokens_per_rollout: float
    avg_tokens_per_step: float
    avg_tokens_per_model_turn: float


@dataclass(slots=True)
class SampleLengthSummary:
    sample_count: int
    min_input_tokens: int
    median_input_tokens: int
    p95_input_tokens: int
    max_input_tokens: int
    avg_prompt_tokens: float
    min_completion_tokens: int
    median_completion_tokens: int
    p95_completion_tokens: int
    max_completion_tokens: int


@dataclass(frozen=True, slots=True)
class WindowArtifacts:
    publication_path: Path | None = None
    checkpoint_path: Path | None = None
    adapter_dir: Path | None = None
    policy_export_seconds: float = 0.0


@dataclass(slots=True)
class WindowTrainResult:
    sample_count: int
    raw_sample_count: int
    eligible_sample_count: int
    filtered_zero_signal_sample_count: int
    subsampled_sample_count: int
    sample_budget_fraction: float
    sample_budget_step_multiple: int
    trainable_parameter_count: int
    mean_advantage: float
    std_advantage: float
    mean_return_to_go: float
    mean_advantage_by_level: dict[int, float]
    mean_old_logprob: float
    final_loss: float
    mean_entropy: float
    entropy_loss: float
    # Coefficient actually applied this window (fixed config value, or the
    # entropy-target controller's current output).
    entropy_coef_used: float
    final_learning_rate: float
    optimizer_steps: int
    mean_ratio: float
    mean_clip_fraction: float
    mean_grad_norm: float
    max_grad_norm: float
    transition_reward_nonzero_fraction_raw: float
    transition_reward_positive_fraction_raw: float
    transition_reward_sum_raw: float
    transition_reward_nonzero_fraction_kept: float
    transition_reward_positive_fraction_kept: float
    transition_reward_sum_kept: float
    return_to_go_nonzero_fraction_kept: float
    return_to_go_positive_fraction_kept: float
    advantage_nonzero_fraction_kept: float
    advantage_positive_fraction_kept: float
    positive_transition_reward_sample_count_raw: int
    positive_transition_reward_sample_count_kept: int
    trainable_param_norm_before: float
    trainable_param_norm_after: float
    advantage_group_mean_reward: float
    advantage_group_std_reward: float
    total_completion_tokens: int
    total_env_steps: int
    avg_actions_per_completed_level: float
    avg_tokens_per_rollout: float
    avg_tokens_per_step: float
    avg_tokens_per_model_turn: float
    avg_prompt_tokens_per_sample: float
    avg_history_messages_per_sample: float
    sample_build_seconds: float
    optimization_seconds: float
    policy_export_seconds: float
    total_train_window_seconds: float
    adapter_dir: Path | None
    runtime_adapter_dir: Path | None
    trainer_state_path: Path | None
    optimizer_cycle_seconds: float = 0.0
    processed_tokens: int = 0
    forward_time_s: float = 0.0
    backward_time_s: float = 0.0
    cuda_trainer_model_allocated_gb: float = 0.0
    cuda_optimization_peak_delta_gb: float = 0.0
    cuda_optimization_persistent_delta_gb: float = 0.0
    rollout_logprob_parity_token_count: int = 0
    rollout_logprob_parity_mean_abs: float = 0.0
    rollout_logprob_parity_p95_abs: float = 0.0
    rollout_logprob_parity_max_abs: float = 0.0
    rollout_logprob_parity_outlier_count: int = 0
    rollout_logprob_parity_outlier_fraction: float = 0.0
    rollout_logprob_parity_signed_mean: float = 0.0
    rollout_logprob_parity_biased_token_count: int = 0
    rollout_logprob_parity_biased_mean_abs: float = 0.0
    rollout_logprob_parity_sampler_policy_iteration: int | None = None
    rollout_logprob_parity_trainer_policy_iteration: int = -1
    rollout_logprob_parity_collection_lag: int | None = None
    rollout_logprob_parity_checked_policy_lag: int | None = None
    rollout_logprob_parity_seconds: float = 0.0
    rollout_logprob_parity_snapshot_seconds: float = 0.0
    rollout_logprob_parity_forward_seconds: float = 0.0
    rollout_logprob_parity_input_token_count: int = 0
    rollout_logprob_parity_sample_count: int = 0
    rollout_logprob_parity_batch_count: int = 0
    rollout_logprob_parity_startup: bool = False
    rollout_logprob_parity_expanded: bool = False
    baseline_feature_seconds: float = 0.0
    baseline_loss: float = 0.0
    baseline_update_steps: int = 0
    baseline_mean_value: float = 0.0
    baseline_mean_target: float = 0.0
    baseline_explained_variance: float = 0.0
    baseline_post_update_mean_value: float = 0.0
    baseline_post_update_mean_absolute_error: float = 0.0
    baseline_post_update_explained_variance: float = 0.0
    baseline_mean_raw_advantage: float = 0.0
    baseline_std_raw_advantage: float = 0.0
    baseline_update_seconds: float = 0.0
    # Episode-weighted statistics over the entire rollout window, independent
    # of the number of packed samples on each rank. Contains pre/post-clip mean.
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    critic_metrics: dict[str, float] = field(default_factory=dict)
    # Sampled-token forward statistics, pooled by token count across all ranks.
    policy_update_metrics: dict[str, float] = field(default_factory=dict)
    # Flash only: expert load balance of the learner's routing by token role
    # (train, rollout, eval), from training.expert_load.summarize_counts.
    expert_load: dict[str, dict[str, float]] = field(default_factory=dict)
    optimizer_skipped_reason: str = ""
    # Opportunistic extra accumulation cycles run after the configured epochs
    # while waiting for the next rollout window (see
    # training.extra_train_cycles_until_window).
    extra_cycles: int = 0
    extra_optimizer_steps: int = 0
    extra_phase_seconds: float = 0.0
    extra_stop_reason: str = ""
    post_update_clip_fraction: float = 0.0
    # Train-window seconds excluding the extra-cycle phase; the sample budget
    # controller must steer on this so free extra cycles do not read as
    # "trainer at capacity".
    budget_train_seconds: float = 0.0
    # Window-level factor multiplied into every cycle's loss denominator so
    # the aggregate update is exactly as if the zero-signal-filtered
    # (zero-gradient) spans were still present in the average (1.0 when
    # nothing was filtered).
    zero_signal_denominator_scale: float = 1.0
    # Episode packing accounting: with token continuation every episode is
    # exactly 1 + (fresh renders) sequences, asserted per window.
    packed_episodes: int = 0
    packed_sequences: int = 0
    segment_boundaries: int = 0
    # Mean token entropy of the frozen probe batch — a fixed-population
    # entropy series immune to the composition drift of train/mean_entropy.
    # 0.0 when no probe was captured yet.
    probe_entropy: float = 0.0
    # Mean k3 KL(rollout policy || current) across the window's micro-batch
    # updates. Measured even when the anchor coefficient is 0.
    mean_previous_policy_kl: float = 0.0
    mean_reference_kl: float = 0.0
    # Window trust region: True when this window's optimizer update was
    # rejected and rolled back (primary mean clip fraction over the bound).
    window_update_rejected: bool = False
    # Learning-rate scale currently applied by the trust-region backoff.
    trust_region_lr_scale: float = 1.0
    # ||grad(entropy_loss)|| / ||grad(policy_loss)|| measured once per window
    # on the first retained micro-batch (0.0 when not computed). Guidance:
    # this ratio should sit <=5-10%; measurement only, no control action.
    entropy_to_policy_grad_ratio: float = 0.0

    @property
    def artifacts(self) -> WindowArtifacts:
        return WindowArtifacts(
            publication_path=self.runtime_adapter_dir,
            checkpoint_path=self.trainer_state_path,
            adapter_dir=self.adapter_dir,
            policy_export_seconds=self.policy_export_seconds,
        )


# Masked target positions scored per vocab-projection chunk. Bounds the live
# full-vocab logits to [chunk, vocab] (~2.3 GiB fp32 at 4096 x 152k) instead of
# [num_targets, vocab]: a densely-trained 32k sequence (long completions, tiny
# observations) has ~30k targets, whose fp32 logits alone are ~17 GiB per copy
# -- that OOM'd single-GPU ranks in production (run 20260711_122810, rank 6).
_TOKEN_SCORE_CHUNK = 4096


def _masked_token_scores(
    masked_hidden: torch.Tensor,
    masked_ids: torch.Tensor,
    lm_head: torch.nn.Module,
    compute_entropies: bool,
    bias_token_ids: torch.Tensor | None = None,
    bias_values: torch.Tensor | None = None,
    head_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    logits = lm_head(masked_hidden).float()
    if bias_values is not None:
        logits.index_add_(
            -1, bias_token_ids.to(logits.device), bias_values.to(logits.device)
        )
    selected = logits.gather(
        dim=-1, index=masked_ids.to(logits.device).unsqueeze(-1)
    ).squeeze(-1)
    log_norm = torch.logsumexp(logits, dim=-1)
    logprobs = selected.to(dtype=log_norm.dtype) - log_norm
    if compute_entropies:
        probabilities = torch.softmax(logits, dim=-1)
        entropies = log_norm - (probabilities * logits).sum(dim=-1)
    else:
        entropies = torch.zeros_like(logprobs)
    if head_ids is not None:
        head_logprobs = logits.gather(
            -1, head_ids.to(logits.device)
        ) - log_norm.unsqueeze(-1)
        return logprobs, entropies, head_logprobs
    return logprobs, entropies


def token_scores_from_hidden_states(
    hidden_states: torch.Tensor,
    *,
    lm_head: torch.nn.Module,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    compute_entropies: bool,
    chunk_size: int = _TOKEN_SCORE_CHUNK,
    bias_token_ids: torch.Tensor | None = None,
    bias_values: torch.Tensor | None = None,
    head_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Per-target logprobs (and entropies) with logits only at masked positions.

    Applying the LM head to the full ``[batch, seq, vocab]`` sequence is the
    dominant training allocation (30+ GiB per micro-batch at 8k tokens); the
    loss only needs completion targets, so gather those hidden states first
    and keep the vocab projection at ``[num_targets, vocab]``. The projection
    itself runs in ``chunk_size`` slices of masked positions, gradient-
    checkpointed so backward recomputes one chunk's logits at a time — without
    this, autograd retains every chunk's fp32 logits until backward.
    """
    # Under device_map sharding the decoder output, the lm_head, and the batch
    # tensors can live on different GPUs; compute at the hidden/lm_head devices
    # and return on the batch device (all moves are no-ops on a single GPU).
    output_device = target_ids.device
    shifted_hidden = hidden_states[:, :-1, :]
    target_ids = target_ids.to(shifted_hidden.device)
    target_mask = target_mask.to(shifted_hidden.device)
    if not bool(target_mask.any()):
        zeros = torch.zeros(
            target_ids.shape,
            dtype=torch.float32,
            device=output_device,
        )
        if head_ids is not None:
            return (
                zeros,
                torch.zeros_like(zeros),
                torch.zeros_like(head_ids, dtype=torch.float32, device=output_device),
            )
        return zeros, torch.zeros_like(zeros)

    masked_hidden = shifted_hidden[target_mask]
    masked_ids = target_ids[target_mask]
    masked_bias = (
        bias_values.to(shifted_hidden.device)[target_mask]
        if bias_values is not None
        else None
    )
    masked_heads = (
        head_ids.to(shifted_hidden.device)[target_mask]
        if head_ids is not None
        else None
    )
    head_chunks = []
    checkpoint_chunks = torch.is_grad_enabled() and masked_hidden.requires_grad
    logprob_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    for start in range(0, masked_hidden.shape[0], chunk_size):
        hidden_chunk = masked_hidden[start : start + chunk_size]
        ids_chunk = masked_ids[start : start + chunk_size]
        bias_chunk = (
            masked_bias[start : start + chunk_size] if masked_bias is not None else None
        )
        head_chunk = (
            masked_heads[start : start + chunk_size]
            if masked_heads is not None
            else None
        )
        if checkpoint_chunks:
            scores = torch.utils.checkpoint.checkpoint(
                _masked_token_scores,
                hidden_chunk,
                ids_chunk,
                lm_head,
                compute_entropies,
                bias_token_ids,
                bias_chunk,
                head_chunk,
                use_reentrant=False,
            )
        else:
            scores = _masked_token_scores(
                hidden_chunk,
                ids_chunk,
                lm_head,
                compute_entropies,
                bias_token_ids,
                bias_chunk,
                head_chunk,
            )
        chunk_logprobs, chunk_entropies = scores[:2]
        if head_ids is not None:
            head_chunks.append(scores[2])
        logprob_chunks.append(chunk_logprobs)
        entropy_chunks.append(chunk_entropies)
    masked_logprobs = torch.cat(logprob_chunks)
    masked_entropies = torch.cat(entropy_chunks)

    score_device = masked_logprobs.device
    score_mask = target_mask.to(score_device)
    token_logprobs = torch.zeros(
        target_ids.shape,
        dtype=masked_logprobs.dtype,
        device=score_device,
    )
    token_logprobs[score_mask] = masked_logprobs
    token_entropies = torch.zeros_like(token_logprobs)
    if compute_entropies:
        token_entropies[score_mask] = masked_entropies
    if head_ids is not None:
        heads = torch.zeros(
            head_ids.shape, dtype=masked_logprobs.dtype, device=score_device
        )
        heads[score_mask] = torch.cat(head_chunks)
        return (
            token_logprobs.to(output_device),
            token_entropies.to(output_device),
            heads.to(output_device),
        )
    return token_logprobs.to(output_device), token_entropies.to(output_device)


def _metadata_nonnegative_int(metadata: dict[str, Any], key: str) -> int | None:
    if key not in metadata:
        return None
    try:
        return max(0, int(metadata.get(key)))
    except (TypeError, ValueError):
        return None


def _env_step_count(rollout: RolloutTrajectory) -> int:
    if isinstance(rollout.metadata, dict):
        parsed = _metadata_nonnegative_int(rollout.metadata, "run_total_actions")
        if parsed is not None:
            return parsed

    metadata_steps = 0
    saw_metadata_step_count = False
    for step in rollout.steps:
        env_metadata = (
            step.metadata.get("env", {}) if isinstance(step.metadata, dict) else {}
        )
        parsed = (
            _metadata_nonnegative_int(env_metadata, "executed_count")
            if isinstance(env_metadata, dict)
            else None
        )
        if parsed is not None:
            saw_metadata_step_count = True
            metadata_steps += parsed
    if saw_metadata_step_count:
        return metadata_steps
    return len(rollout.steps)


def _avg_actions_per_completed_level(rollouts: list[RolloutTrajectory]) -> float:
    """All episode actions per level solved."""
    completed_levels = sum(episode_levels_completed(rollout) for rollout in rollouts)
    if completed_levels <= 0:
        return 0.0
    return sum(_env_step_count(rollout) for rollout in rollouts) / completed_levels


def summarize_token_usage(
    rollouts: list[RolloutTrajectory], samples: list[TokenizedSample]
) -> TokenUsageSummary:
    total_completion_tokens = sum(sum(sample.completion_mask) for sample in samples)
    total_env_steps = sum(_env_step_count(rollout) for rollout in rollouts)
    total_model_turns = sum(len(rollout.steps) for rollout in rollouts)
    rollout_count = len(rollouts)
    return TokenUsageSummary(
        total_completion_tokens=total_completion_tokens,
        total_env_steps=total_env_steps,
        avg_tokens_per_rollout=(total_completion_tokens / rollout_count)
        if rollout_count
        else 0.0,
        avg_tokens_per_step=(total_completion_tokens / total_env_steps)
        if total_env_steps > 0
        else 0.0,
        avg_tokens_per_model_turn=(total_completion_tokens / total_model_turns)
        if total_model_turns > 0
        else 0.0,
    )


def _sample_token_length(sample: TokenizedSample) -> int:
    return max(1, len(sample.input_ids))


def _pack_micro_batches(
    samples: list[TokenizedSample], *, batch_size: int, token_budget: int
) -> list[list[TokenizedSample]]:
    micro_batches: list[list[TokenizedSample]] = []
    chunk: list[TokenizedSample] = []
    chunk_max_tokens = 0
    chunk_actual_tokens = 0
    min_chunk_size_for_waste_split = max(1, batch_size // 2)
    for sample in samples:
        sample_tokens = _sample_token_length(sample)
        next_max_tokens = max(chunk_max_tokens, sample_tokens)
        next_size = len(chunk) + 1
        next_actual_tokens = chunk_actual_tokens + sample_tokens
        next_padded_tokens = next_size * next_max_tokens
        exceeds_batch_size = next_size > batch_size
        exceeds_token_budget = token_budget > 0 and next_padded_tokens > token_budget
        exceeds_padding_waste = False
        if chunk and len(chunk) >= min_chunk_size_for_waste_split:
            next_padding_waste = 0.0
            if next_padded_tokens > 0:
                next_padding_waste = (
                    next_padded_tokens - next_actual_tokens
                ) / next_padded_tokens
            exceeds_padding_waste = (
                next_padding_waste > _LENGTH_PACKING_MAX_PADDING_WASTE_FRACTION
            )
        if chunk and (
            exceeds_batch_size or exceeds_token_budget or exceeds_padding_waste
        ):
            micro_batches.append(chunk)
            chunk = [sample]
            chunk_max_tokens = sample_tokens
            chunk_actual_tokens = sample_tokens
            continue
        chunk.append(sample)
        chunk_max_tokens = next_max_tokens
        chunk_actual_tokens = next_actual_tokens
    if chunk:
        micro_batches.append(chunk)
    return micro_batches


def _micro_batch_padded_tokens(micro_batch: list[TokenizedSample]) -> int:
    if not micro_batch:
        return 0
    return len(micro_batch) * max(
        _sample_token_length(sample) for sample in micro_batch
    )


def _length_bucketed_micro_batches(
    samples: list[TokenizedSample], *, batch_size: int, token_budget: int
) -> list[list[TokenizedSample]]:
    ordered = sorted(samples, key=_sample_token_length, reverse=True)
    micro_batches = _pack_micro_batches(
        ordered, batch_size=batch_size, token_budget=token_budget
    )
    for micro_batch in micro_batches:
        random.shuffle(micro_batch)
    if len(micro_batches) <= 1:
        return micro_batches

    largest_index = max(
        range(len(micro_batches)),
        key=lambda index: _micro_batch_padded_tokens(micro_batches[index]),
    )
    largest_micro_batch = micro_batches.pop(largest_index)
    random.shuffle(micro_batches)
    return [largest_micro_batch, *micro_batches]


def _micro_batch_length_stats(
    micro_batches: list[list[TokenizedSample]],
) -> dict[str, float | int]:
    non_empty = [micro_batch for micro_batch in micro_batches if micro_batch]
    if not non_empty:
        return {}

    total_actual_tokens = 0
    total_padded_tokens = 0
    max_length_spread = 0
    max_micro_batch_size = 0
    largest_padded_tokens = 0
    for micro_batch in non_empty:
        lengths = [_sample_token_length(sample) for sample in micro_batch]
        max_length = max(lengths)
        min_length = min(lengths)
        padded_tokens = len(micro_batch) * max_length
        total_actual_tokens += sum(lengths)
        total_padded_tokens += padded_tokens
        max_length_spread = max(max_length_spread, max_length - min_length)
        max_micro_batch_size = max(max_micro_batch_size, len(micro_batch))
        largest_padded_tokens = max(largest_padded_tokens, padded_tokens)

    padding_waste_fraction = 0.0
    if total_padded_tokens > 0:
        padding_waste_fraction = (
            total_padded_tokens - total_actual_tokens
        ) / total_padded_tokens

    return {
        "micro_batch_count": len(non_empty),
        "mean_micro_batch_size": sum(len(batch) for batch in non_empty)
        / len(non_empty),
        "max_micro_batch_size": max_micro_batch_size,
        "max_length_spread": max_length_spread,
        "padding_waste_fraction": padding_waste_fraction,
        "largest_padded_tokens": largest_padded_tokens,
    }


def summarize_sample_lengths(samples: list[TokenizedSample]) -> SampleLengthSummary:
    if not samples:
        return SampleLengthSummary(
            sample_count=0,
            min_input_tokens=0,
            median_input_tokens=0,
            p95_input_tokens=0,
            max_input_tokens=0,
            avg_prompt_tokens=0.0,
            min_completion_tokens=0,
            median_completion_tokens=0,
            p95_completion_tokens=0,
            max_completion_tokens=0,
        )

    def _percentile(sorted_values: list[int], *, fraction: float) -> int:
        if not sorted_values:
            return 0
        index = max(
            0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * fraction) - 1)
        )
        return int(sorted_values[index])

    input_lengths = sorted(len(sample.input_ids) for sample in samples)
    completion_lengths = sorted(sum(sample.completion_mask) for sample in samples)
    prompt_lengths = [
        len(sample.input_ids) - sum(sample.completion_mask) for sample in samples
    ]
    median_index = len(samples) // 2
    return SampleLengthSummary(
        sample_count=len(samples),
        min_input_tokens=input_lengths[0],
        median_input_tokens=input_lengths[median_index],
        p95_input_tokens=_percentile(input_lengths, fraction=0.95),
        max_input_tokens=input_lengths[-1],
        avg_prompt_tokens=sum(prompt_lengths) / len(prompt_lengths),
        min_completion_tokens=completion_lengths[0],
        median_completion_tokens=completion_lengths[median_index],
        p95_completion_tokens=_percentile(completion_lengths, fraction=0.95),
        max_completion_tokens=completion_lengths[-1],
    )


class _DdpWorkerGroup:
    """Rank-0 handle for the data-parallel gradient co-workers.

    Rank 0 keeps the only optimizer; workers (training/ddp_worker.py) run
    forward/backward on their share of each gradient-accumulation cycle,
    contribute LoRA grads to a NCCL reduce, and install the stepped weights
    rank 0 broadcasts back. All control flow is rank-0 initiated.
    """

    _CMD_TRAIN_CYCLE = "train_cycle"
    _CMD_SYNC_WEIGHTS = "sync_weights"
    _CMD_HIDDEN_FEATURES = "hidden_features"
    _CMD_EXIT = "exit"

    def __init__(
        self,
        *,
        config: TrainingConfig,
        parameters: list[torch.nn.Parameter],
        resume_adapter_dir: Path | None,
        grpo_group_size: int | None = None,
        grpo_puzzles_per_window: int | None = None,
        checkpoint_identity: dict[str, Any] | None = None,
        game_vocab: Sequence[str] | None = None,
    ) -> None:
        replicas = [
            [int(gpu) for gpu in replica]
            for replica in (config.worker_device_map_gpus or [])
        ]
        if not replicas:
            raise RuntimeError("_DdpWorkerGroup requires worker replicas.")
        self.world_size = 1 + len(replicas)
        self.parameters = parameters
        self.primary_device = torch.device(config.device)
        # Own the actual rendezvous socket before spawning clients. A
        # bind-close free-port probe races other processes for that address.
        store = dist.TCPStore(
            "127.0.0.1",
            0,
            world_size=self.world_size,
            is_master=True,
            wait_for_workers=False,
            timeout=datetime.timedelta(hours=6),
        )
        port = store.port
        run_root = str(os.environ.get("ARC3_RUN_DIR", "") or "").strip()
        spec_dir = Path(run_root) if run_root else Path.cwd()
        self._processes: list[subprocess.Popen[bytes]] = []
        self._health_stopped = threading.Event()
        worker_config = asdict(config)
        for worker_index, gpus in enumerate(replicas):
            rank = worker_index + 1
            log_path = spec_dir / f"ddp_worker_{rank}.log"
            spec = {
                "rank": rank,
                "world_size": self.world_size,
                "port": port,
                "seed": int(config.seed),
                "training_config": worker_config,
                "grpo_group_size": grpo_group_size,
                "grpo_puzzles_per_window": grpo_puzzles_per_window,
                "checkpoint_identity": checkpoint_identity,
                "game_vocab": list(game_vocab or ()),
                "device_map_gpus": gpus,
                "resume_adapter_dir": (
                    str(resume_adapter_dir) if resume_adapter_dir else None
                ),
                "parent_pid": os.getpid(),
                "log_path": str(log_path),
            }
            spec_path = spec_dir / f"ddp_worker_{rank}.spec.json"
            spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            with log_path.with_suffix(".stdout.log").open("ab") as stdout_log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "training.ddp_worker_entry",
                        "--spec",
                        str(spec_path),
                    ],
                    stdout=stdout_log,
                    stderr=subprocess.STDOUT,
                )
            self._processes.append(process)
            log.info(
                "spawned ddp worker rank %d (pid=%d, gpus=%s, log=%s)",
                rank,
                process.pid,
                gpus,
                log_path,
            )

        # Watch independently of the main thread: it can be blocked inside
        # process-group setup or an NCCL collective when a worker dies.
        self._health_thread = threading.Thread(
            target=self._watch_worker_health, name="ddp-worker-health", daemon=True
        )
        self._health_thread.start()
        dist.init_process_group(
            "gloo",
            store=store,
            rank=0,
            world_size=self.world_size,
            timeout=datetime.timedelta(hours=6),
        )
        self._nccl_group = dist.new_group(
            backend="nccl", timeout=datetime.timedelta(minutes=30)
        )
        self._flat_weights = flatten_tensors(
            [parameter.data for parameter in self.parameters],
            device=self.primary_device,
            dtype=torch.float32,
        )
        dist.broadcast(self._flat_weights, src=0, group=self._nccl_group)
        log.info(
            "ddp group ready: world_size=%d, %d trainable tensors, %.1fMB flat buffer",
            self.world_size,
            len(self.parameters),
            self._flat_weights.numel() * 4 / 1e6,
        )

    def check_health(self) -> None:
        for rank, process in enumerate(self._processes, start=1):
            code = process.poll()
            if code is not None:
                self._abort_shutdown(
                    f"worker rank {rank} pid={process.pid} exited unexpectedly with code {code}"
                )

    def _watch_worker_health(self) -> None:
        while not self._health_stopped.wait(_DDP_HEALTH_POLL_SECONDS):
            self.check_health()

    def send_cycle(
        self,
        worker_micro_batches: list[list[list[TokenizedSample]]],
        *,
        loss_denominator: float,
        entropy_coef: float,
    ) -> None:
        self.check_health()
        # One broadcast reaches every worker; the command carries every
        # worker's share and rank r trains shares[r-1].
        command = [
            {
                "type": self._CMD_TRAIN_CYCLE,
                "shares": worker_micro_batches,
                "loss_denominator": loss_denominator,
                "entropy_coef": float(entropy_coef),
            }
        ]
        dist.broadcast_object_list(command, src=0)

    def reduce_grads_into_params(self) -> None:
        """Sum every rank's LoRA grads (rank 0's included) into param.grad."""
        flat = flatten_grads(self.parameters, device=self.primary_device)
        dist.reduce(flat, dst=0, group=self._nccl_group)
        offset = 0
        for parameter in self.parameters:
            numel = parameter.numel()
            chunk = flat[offset : offset + numel].view_as(parameter)
            grad = chunk.to(device=parameter.device, dtype=parameter.dtype)
            parameter.grad = grad
            offset += numel

    def broadcast_weights(self) -> None:
        flat_weights = flatten_tensors(
            [parameter.data for parameter in self.parameters],
            device=self.primary_device,
            dtype=torch.float32,
        )
        self._flat_weights.copy_(flat_weights)
        dist.broadcast(self._flat_weights, src=0, group=self._nccl_group)

    @shutdown_at_safe_boundary
    def sync_weights_out_of_cycle(self) -> None:
        """Resync worker replicas to rank 0's parameters OUTSIDE the train
        cycle protocol (workers are idle in the command loop). Used after a
        window trust-region rejection restores rank 0's weights."""
        command = [{"type": self._CMD_SYNC_WEIGHTS}]
        dist.broadcast_object_list(command, src=0)
        self.broadcast_weights()

    def gather_worker_metrics(self) -> list[dict[str, Any]]:
        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.gather_object({}, gathered, dst=0)
        return [entry for entry in gathered[1:] if isinstance(entry, dict)]

    @shutdown_at_safe_boundary
    def extract_hidden_features(
        self,
        worker_batches: list[list[PolicyHiddenBatch]],
        local_extract: Callable[[], list[PolicyHiddenFeatures]],
    ) -> list[PolicyHiddenFeatures]:
        """Complete one read-only feature round before honoring termination.

        The caller sends at most one existing batch per rank in a round, so
        TERM waits for a single forward, rather than the entire critic pass.
        All replicas already hold the last broadcast policy weights.
        """
        self.check_health()
        try:
            dist.broadcast_object_list(
                [{"type": self._CMD_HIDDEN_FEATURES, "shares": worker_batches}],
                src=0,
            )
            local_features = local_extract()
            gathered: list[Any] = [None] * self.world_size
            # Only selected decision states travel back, as dense CPU tensors.
            # This never copies decoder activations or uses the filesystem.
            dist.gather_object(local_features, gathered, dst=0)
            if any(not isinstance(features, list) for features in gathered):
                raise RuntimeError("DDP hidden feature response is malformed")
            return [block for features in gathered for block in features]
        except Exception as error:
            # A failed rank-zero forward must not leave workers stuck in their
            # gather while cleanup tries to send another command.
            self._abort_shutdown(f"hidden feature extraction failed: {error!r}")
            raise

    def shutdown(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return
        if hasattr(self, "_health_stopped"):
            self._health_stopped.set()
            self._health_thread.join(timeout=2.0)
        completed = threading.Event()
        deadline = time.monotonic() + _DDP_SHUTDOWN_TIMEOUT_SECONDS

        def watch_shutdown() -> None:
            if not completed.wait(timeout=_DDP_SHUTDOWN_TIMEOUT_SECONDS):
                self._abort_shutdown(
                    f"timed out after {_DDP_SHUTDOWN_TIMEOUT_SECONDS:g}s"
                )

        # Keep distributed calls on the main thread. A daemon blocked in a
        # collective can still hang interpreter/C++ destruction after an error.
        # This watchdog only kills/reaps owned workers and hard-exits on failure.
        watchdog = threading.Thread(
            target=watch_shutdown, name="ddp-shutdown-watchdog", daemon=True
        )
        watchdog.start()
        try:
            dist.broadcast_object_list([{"type": self._CMD_EXIT}], src=0)
            # Every rank must destroy its groups before waiting for peers to
            # exit: NCCL destruction can itself require peer participation.
            # Both sides destroy all groups in the same reverse creation order.
            dist.destroy_process_group()
            for process in self._processes:
                returncode = process.wait(
                    timeout=max(0.001, deadline - time.monotonic())
                )
                if returncode != 0:
                    raise RuntimeError(
                        f"DDP worker pid={process.pid} exited with code {returncode}"
                    )
            self._shutdown_complete = True
            log.info("ddp process groups destroyed and all workers exited cleanly")
        except BaseException as error:
            self._abort_shutdown(f"failed: {error!r}")
        finally:
            completed.set()
            watchdog.join(timeout=1.0)

    def _abort_shutdown(self, reason: str) -> None:
        message = f"FATAL: DDP shutdown {reason}; killing owned workers and exiting"

        def write_stderr() -> None:
            # Bypass logging locks, but stderr itself can block on a full pipe
            # or slow filesystem. This thread must never delay worker cleanup.
            try:
                os.write(2, (message + "\n").encode())
            except OSError:
                pass

        def publish_failure() -> None:
            from training.run_failure import record_run_failure

            run_root = os.environ.get("ARC3_RUN_DIR")
            if run_root:
                try:
                    record_run_failure(Path(run_root), RuntimeError(message))
                except Exception:  # noqa: BLE001 - hard exit must still happen
                    pass

        # Keep both outputs independent: blocked stderr must not prevent the
        # durable marker, and slow NFS must not prevent the emergency message.
        # They share one deadline with worker reaping before the hard exit.
        deadline = time.monotonic() + 2.0
        publisher = threading.Thread(target=publish_failure, daemon=True)
        stderr_writer = threading.Thread(target=write_stderr, daemon=True)
        publisher.start()
        stderr_writer.start()
        for process in self._processes:
            try:
                process.kill()
            except OSError:
                pass
        for process in self._processes:
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except (OSError, subprocess.TimeoutExpired):
                pass
        publisher.join(timeout=max(0.0, deadline - time.monotonic()))
        stderr_writer.join(timeout=max(0.0, deadline - time.monotonic()))
        # Raising cannot guarantee exit with a damaged process group: Python
        # finalization can wait indefinitely for its C++ collective/destructor.
        os._exit(1)


def plan_policy_hidden_segments(
    rollouts: list[RolloutTrajectory],
) -> list[PolicyHiddenPlan]:
    """Share a decoder forward within each verified append-only segment.

    Values use the last PROMPT token before each model response. A compaction
    starts another forward, never another return/advantage chain.
    """
    plans: list[PolicyHiddenPlan] = []

    def flush(index, steps):
        if steps:
            plans.append(
                (
                    index,
                    [step.step_index for step in steps],
                    [len(step.prompt_token_ids) - 1 for step in steps],
                    list(steps[-1].prompt_token_ids),
                )
            )

    for index, rollout in enumerate(rollouts):
        steps = []
        for step in rollout.steps:
            if not step.prompt_token_ids:
                raise RuntimeError(
                    "hidden critic requires a nonempty exact prompt for every decision"
                )
            if steps and (step.metadata or {}).get("segment_start"):
                flush(index, steps)
                steps = []
            if steps:
                prefix = steps[-1].prompt_token_ids + steps[-1].completion_token_ids
                if step.prompt_token_ids[: len(prefix)] != prefix:
                    raise RuntimeError(
                        "hidden critic observed an undeclared prompt rewrite"
                    )
            steps.append(step)
        flush(index, steps)
    return plans


def batch_policy_hidden_segments(
    plans: list[PolicyHiddenPlan], *, max_tokens: int, token_budget: int
) -> list[PolicyHiddenBatch]:
    """Preserve the existing length-sorted, padded feature forward shapes."""
    ordered = sorted(plans, key=lambda plan: len(plan[3]), reverse=True)
    if any(not plan[3] or len(plan[3]) > max_tokens for plan in ordered):
        raise RuntimeError("hidden critic prompt exceeds training context budget")
    budget = max(token_budget, max((len(plan[3]) for plan in ordered), default=0))
    batches: list[PolicyHiddenBatch] = []
    current: PolicyHiddenBatch = []
    for plan in ordered:
        if current and (len(current) + 1) * len(current[0][3]) > budget:
            batches.append(current)
            current = []
        current.append(plan)
    if current:
        batches.append(current)
    return batches


def distribute_policy_hidden_batches(
    batches: list[PolicyHiddenBatch], world_size: int
) -> list[list[PolicyHiddenBatch]]:
    """Balance padded token work without changing any batch's rows or width."""
    if world_size < 1:
        raise ValueError("hidden feature world size must be positive")
    shares: list[list[PolicyHiddenBatch]] = [[] for _ in range(world_size)]
    costs = [0] * world_size
    for batch in sorted(
        batches, key=lambda rows: len(rows) * len(rows[0][3]), reverse=True
    ):
        rank = min(range(world_size), key=lambda index: costs[index])
        shares[rank].append(batch)
        costs[rank] += len(batch) * len(batch[0][3])
    return shares


def merge_policy_hidden_features(
    blocks: list[PolicyHiddenFeatures], expected: set[tuple[int, int]]
) -> dict[tuple[int, int], list[float]]:
    hidden_by_step: dict[tuple[int, int], list[float]] = {}
    hidden_size: int | None = None
    for keys, values in blocks:
        if (
            not isinstance(values, torch.Tensor)
            or values.device.type != "cpu"
            or values.ndim != 2
            or values.shape[0] != len(keys)
            or values.shape[1] < 1
            or (hidden_size is not None and values.shape[1] != hidden_size)
            or not bool(torch.isfinite(values).all())
        ):
            raise RuntimeError("hidden critic received invalid decision-state features")
        hidden_size = values.shape[1]
        for key, row in zip(keys, values.tolist(), strict=True):
            if key not in expected or key in hidden_by_step:
                raise RuntimeError(
                    "hidden critic received unexpected or duplicate decision state"
                )
            hidden_by_step[key] = row
    if set(hidden_by_step) != expected:
        raise RuntimeError("hidden critic did not extract every decision-state feature")
    return hidden_by_step


class PPOTrainer:
    # Class-level defaults so test doubles built without __init__ (and
    # checkpoints predating the field) still read consistent state.
    _dry_optimizer_step_count: int = 0
    _trust_region_lr_scale: float = 1.0
    _consecutive_window_rejections: int = 0

    # Class-level default so instances constructed without __init__ (tests
    # stub the heavyweight model load) behave as single-replica trainers.
    _ddp_group: _DdpWorkerGroup | None = None

    def __init__(
        self,
        config: TrainingConfig,
        *,
        resume_adapter_dir: Path | None = None,
        resume_state_path: Path | None = None,
        memory_logger: Any | None = None,
        game_vocab: Sequence[str] | None = None,
        grpo_group_size: int | None = None,
        grpo_puzzles_per_window: int | None = None,
        checkpoint_identity: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self._grpo_group_size = grpo_group_size
        self._grpo_puzzles_per_window = grpo_puzzles_per_window
        self._checkpoint_identity = copy.deepcopy(checkpoint_identity)
        if config.advantage_estimator in {"grpo", "dr_grpo"}:
            if (
                grpo_group_size is None
                or grpo_group_size < 2
                or grpo_puzzles_per_window is None
                or grpo_puzzles_per_window < 1
            ):
                raise ValueError(
                    "GRPO requires group counts derived from the full rollout window."
                )
            # GRPO binds the checkpoint to its task set; whether an image
            # stack is pinned is the harness's media declaration, not the
            # algorithm's (training/multimodal.py).
            if not checkpoint_identity or any(
                not checkpoint_identity.get(key)
                for key in ("environment", "advantage_estimator", "dataset_manifest")
            ):
                raise ValueError("GRPO requires a complete checkpoint identity.")
            if checkpoint_identity["advantage_estimator"] != config.advantage_estimator:
                raise ValueError(
                    "Checkpoint identity does not match the GRPO algorithm."
                )
        self.game_vocab = tuple(game_vocab or ())
        self.memory_logger = memory_logger or NullMemoryLogger()
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.device_index = self._resolve_device_index()
        if self.device.type == "cuda" and self.device_index is not None:
            torch.cuda.set_device(self.device_index)
        from training.multimodal import checkpoint_source

        tokenizer_source, tokenizer_revision, local_files_only = checkpoint_source(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=tokenizer_revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = None
        # Whichever harness pinned an image-processing identity gets the
        # matching processor; text-only identities (or none) stay text-only.
        if self._checkpoint_identity and self._checkpoint_identity.get(
            "processing_stack"
        ):
            from training.multimodal import load_processing_stack

            self.processor = load_processing_stack(
                self._checkpoint_identity["processing_stack"]
            )
        self.model = self._load_policy_model(resume_adapter_dir=resume_adapter_dir)
        from training.blackwell_gdn import configure_blackwell_gdn

        configure_blackwell_gdn("fla", device=self.device)
        self._place_policy_model()
        self.model.train()
        self.state_critic: StateCriticBaseline | None = None
        if (
            config.advantage_estimator == "state_critic"
            and self.config.value_baseline == "hidden_critic"
        ):
            self.state_critic = self._build_state_critic()
        self.optimizer = self._build_optimizer()
        self._optimizer_step_count = 0
        # Preserve historical dry-step offsets when resuming old checkpoints.
        # New training never performs dry optimizer steps.
        self._dry_optimizer_step_count = 0
        self._trainable_parameter_count = sum(
            parameter.numel() for parameter in self._trainable_parameters()
        )
        self._cuda_trainer_model_allocated_gb = self._cuda_memory_allocated_gb()
        log.info(
            "initialized trainer on %s with %d trainable adapter parameters, warmup_steps=%d, cuda_allocated=%.3fGiB",
            self.device,
            self._trainable_parameter_count,
            self.config.warmup_steps,
            self._cuda_trainer_model_allocated_gb,
        )
        if resume_state_path is not None:
            self.load_training_state(resume_state_path)
            log.info(
                "resumed trainer state from %s (optimizer_step_count=%d)",
                resume_state_path,
                self._optimizer_step_count,
            )
        # Data-parallel gradient workers attach last so their initial weight
        # broadcast reflects the fully initialized/resumed policy.
        self._ddp_group: _DdpWorkerGroup | None = None
        # No workers exist yet, so stop here if TERM arrived during model load.
        # After spawning, initialization must finish without a shutdown check;
        # the caller first needs ownership of this trainer for cleanup.
        raise_if_shutdown_requested()
        if self.config.worker_device_map_gpus:
            self._ddp_group = _DdpWorkerGroup(
                config=self.config,
                parameters=self._trainable_parameters(),
                resume_adapter_dir=resume_adapter_dir,
                grpo_group_size=self._grpo_group_size,
                grpo_puzzles_per_window=self._grpo_puzzles_per_window,
                checkpoint_identity=self._checkpoint_identity,
                game_vocab=self.game_vocab,
            )

    def _place_policy_model(self) -> None:
        """Place the loaded policy; distributed trainers override this hook."""
        model_is_dispatched = bool(getattr(self.model, "hf_device_map", None))
        if not model_is_dispatched:
            self.model.to(self.device)
        if model_is_dispatched:
            log.info(
                "policy model sharded across GPUs via device_map: %s",
                {
                    module: str(device)
                    for module, device in sorted(
                        self.model.hf_device_map.items(),
                        key=lambda item: str(item[0]),
                    )
                }
                if len(self.model.hf_device_map) <= 8
                else f"{len(set(self.model.hf_device_map.values()))} devices",
            )

    def _policy_hidden_size(self) -> int:
        config = self._base_policy_model().config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(config, "text_config", None)
            hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError(
                "cannot resolve the policy hidden size for the state critic "
                f"from {type(config).__name__}."
            )
        return int(hidden_size)

    def _build_state_critic(self) -> StateCriticBaseline:
        policy_hidden_dim = self._policy_hidden_size()
        if policy_hidden_dim <= 0:
            raise RuntimeError(
                "hidden critic requires a positive policy hidden dimension"
            )
        if policy_hidden_dim:
            log.info(
                "state critic conditions on the policy hidden state "
                "(dim=%d, projection=%d)",
                policy_hidden_dim,
                self.config.state_critic_policy_hidden_projection_dim,
            )
        return StateCriticBaseline(
            game_vocab=getattr(self, "game_vocab", ()),
            game_embedding_buckets=self.config.state_critic_game_embedding_buckets,
            game_embedding_dim=self.config.state_critic_game_embedding_dim,
            hidden_dim=self.config.state_critic_hidden_dim,
            learning_rate=self.config.state_critic_learning_rate,
            weight_decay=self.config.state_critic_weight_decay,
            device="cpu",
            seed=self.config.seed + 104729,
            initial_value=self.config.state_critic_initial_value,
            target_scale=self.config.state_critic_target_scale,
            target_normalization=self.config.state_critic_target_normalization,
            normalization_alpha=self.config.state_critic_normalization_alpha,
            normalization_min_rollouts=self.config.state_critic_normalization_min_rollouts,
            normalization_min_std=self.config.state_critic_normalization_min_std,
            policy_hidden_dim=policy_hidden_dim,
            policy_hidden_projection_dim=(
                self.config.state_critic_policy_hidden_projection_dim
            ),
        )

    def _resolve_device_index(self) -> int | None:
        if self.device.type != "cuda":
            return None
        if self.device.index is not None:
            return int(self.device.index)
        return torch.cuda.current_device()

    def _resolved_attn_implementation(self) -> str | None:
        requested = str(getattr(self.config, "attn_implementation", "sdpa")).strip()
        lowered = requested.lower()
        if lowered in {"", "auto"}:
            raise ValueError(
                "training.attn_implementation must be explicit; set "
                "'flash_attention_4', 'flash_attention_2', 'sdpa', 'eager', "
                "or 'none'."
            )
        if lowered in {"default", "none"}:
            return None
        if lowered == "flash_attention_4":
            try:
                require_fa4_distribution()
            except RuntimeError as exc:
                raise RuntimeError(
                    "training.attn_implementation='flash_attention_4' requires "
                    "the locked flash-attn-4 package."
                ) from exc
            if "flash_attention_4" not in AttentionInterface().valid_keys():
                raise RuntimeError(
                    "the installed Transformers build does not register "
                    "flash_attention_4."
                )
            register_fa4_contiguous_inputs()
        return requested

    def _checkpoint_preserve_rng_state(self) -> bool:
        # RNG state only matters when the recomputed forward is stochastic;
        # this trainer disables dropout for PPO ratio stability, so skipping
        # the per-module RNG stash saves sync overhead on every checkpoint.
        return not bool(self.config.disable_dropout)

    @property
    def optimizer_step_count(self) -> int:
        return self._optimizer_step_count

    def check_worker_health(self) -> None:
        if self._ddp_group is not None:
            self._ddp_group.check_health()

    def shutdown_workers(self) -> None:
        """Stop data-parallel gradient workers (no-op without replicas)."""
        if self._ddp_group is not None:
            self._ddp_group.shutdown()
            self._ddp_group = None

    def _sharded_device_map_gpus(self) -> list[int]:
        gpus = [int(gpu) for gpu in (self.config.device_map_gpus or [])]
        if not gpus:
            return []
        if not torch.cuda.is_available():
            return []
        if self.device_index != gpus[0]:
            raise RuntimeError(
                "training.device must be the first device_map_gpus entry; got "
                f"device={self.device} device_map_gpus={gpus}."
            )
        return gpus

    def _load_policy_model(
        self, *, resume_adapter_dir: Path | None = None
    ) -> PeftModel:
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": None,
        }
        if self.config.bf16 and self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
        sharded_gpus = self._sharded_device_map_gpus()
        if sharded_gpus:
            # Shard the policy across the listed GPUs (accelerate dispatch
            # hooks move activations between shards); no CPU offload -- a model
            # that does not fit the listed GPUs should fail loudly.
            model_kwargs["device_map"] = "balanced"
            model_kwargs["max_memory"] = {
                gpu_index: int(
                    torch.cuda.get_device_properties(gpu_index).total_memory * 0.92
                )
                for gpu_index in sharded_gpus
            }

        attn_implementation = self._resolved_attn_implementation()
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        log.info("loading policy with attention backend %s", attn_implementation)
        base_model = self._load_base_model(model_kwargs)
        # The trainer never generates, so per-forward KV/recurrent-state caches
        # are pure memory and latency waste; disable them for every mode (the
        # config default leaves use_cache=True).
        if hasattr(base_model, "config") and hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
        if self.config.gradient_checkpointing:
            _enable_activation_checkpointing(
                base_model,
                preserve_rng_state=self._checkpoint_preserve_rng_state(),
            )
        if resume_adapter_dir is not None:
            adapter_dir = Path(resume_adapter_dir)
            if not adapter_dir.exists():
                raise FileNotFoundError(
                    f"Resume adapter directory does not exist: {adapter_dir}"
                )
            model = PeftModel.from_pretrained(
                base_model, str(adapter_dir), is_trainable=True
            )
            if self.config.disable_dropout:
                disabled_dropouts = _disable_dropout_modules(model)
                log.info(
                    "disabled dropout by setting %d policy dropout modules to p=0.0",
                    disabled_dropouts,
                )
            model.print_trainable_parameters()
            return model
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=0.0
            if self.config.disable_dropout
            else self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
        )
        model = get_peft_model(base_model, lora_config)
        if self.config.disable_dropout:
            disabled_dropouts = _disable_dropout_modules(model)
            log.info(
                "disabled dropout by setting %d policy dropout modules to p=0.0",
                disabled_dropouts,
            )
        model.print_trainable_parameters()
        return model

    def _training_world_size(self) -> int:
        return self._ddp_group.world_size if self._ddp_group is not None else 1

    def _synchronize_cycle_seconds(self, seconds: float) -> float:
        return seconds

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

    def _autocast_context(self):
        if self.device.type != "cuda":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _cuda_memory_allocated_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.memory_allocated(self.device)) / (1024.0**3)

    def _cuda_trainer_model_memory_allocated_gb(self) -> float:
        return float(
            getattr(
                self,
                "_cuda_trainer_model_allocated_gb",
                self._cuda_memory_allocated_gb(),
            )
        )

    def _cuda_max_memory_allocated_gb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device)) / (1024.0**3)

    def _cuda_reset_peak_memory_stats(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        parameters = self._trainable_parameters()
        if not parameters:
            raise RuntimeError(
                "No trainable parameters were found for the LoRA policy."
            )
        return torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _learning_rate_for_step(self, step_index: int) -> float:
        trust_region_scale = float(getattr(self, "_trust_region_lr_scale", 1.0))
        if self.config.warmup_steps <= 0:
            return float(self.config.learning_rate) * trust_region_scale
        real_step = step_index - self._dry_optimizer_step_count
        scale = min(1.0, max(1, real_step) / float(self.config.warmup_steps))
        return float(self.config.learning_rate) * scale * trust_region_scale

    def _set_optimizer_learning_rate(self, step_index: int) -> float:
        learning_rate = self._learning_rate_for_step(step_index)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def _current_learning_rate(self) -> float:
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _window_clip_fraction_bound(self) -> float:
        return max(0.0, float(getattr(self.config, "max_window_clip_fraction", 0.0)))

    def _snapshot_update_state(self) -> dict[str, Any]:
        """CPU snapshot of the trainable parameters and optimizer state,
        taken before a window's policy update so a destructive update can be
        rejected wholesale (window trust region). Rank 0 only: rank 0 is the
        weight authority — workers receive its weights via broadcast every
        optimizer step, and their gradients are reduced into rank 0 before
        it steps, so restoring here restores the whole group."""

        def clone_to_cpu(obj: Any) -> Any:
            if torch.is_tensor(obj):
                return obj.detach().to("cpu", copy=True)
            if isinstance(obj, dict):
                return {key: clone_to_cpu(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [clone_to_cpu(value) for value in obj]
            return copy.deepcopy(obj)

        return {
            "parameters": [
                (param, param.detach().to("cpu", copy=True))
                for param in self._trainable_parameters()
            ],
            "optimizer_state": clone_to_cpu(self.optimizer.state_dict()),
            "optimizer_step_count": self._optimizer_step_count,
            "dry_optimizer_step_count": self._dry_optimizer_step_count,
        }

    def _restore_update_state(self, snapshot: dict[str, Any]) -> None:
        with torch.no_grad():
            for param, backup in snapshot["parameters"]:
                param.copy_(backup.to(param.device))
        # load_state_dict moves state tensors back onto the param device.
        self.optimizer.load_state_dict(snapshot["optimizer_state"])
        self._optimizer_step_count = int(snapshot["optimizer_step_count"])
        self._dry_optimizer_step_count = int(snapshot["dry_optimizer_step_count"])
        if self._ddp_group is not None:
            self._ddp_group.sync_weights_out_of_cycle()

    def _apply_window_trust_region(
        self,
        snapshot: dict[str, Any] | None,
        *,
        post_update_clip_fraction: float | None,
        iteration_index: int,
    ) -> bool:
        """Bound primary-update drift from the window-start policy.

        The fixed token probe is scored after the update, independently of
        sampler lag. Rejection restores weights, optimizer moments and counters.
        """
        bound = self._window_clip_fraction_bound()
        if bound <= 0.0:
            return False
        if (
            snapshot is None
            or post_update_clip_fraction is None
            or not math.isfinite(post_update_clip_fraction)
            or not 0.0 <= post_update_clip_fraction <= 1.0
        ):
            raise RuntimeError(
                "enabled window trust region requires a valid update probe"
            )
        primary_clip_fraction = post_update_clip_fraction
        if primary_clip_fraction <= bound:
            if int(getattr(self, "_consecutive_window_rejections", 0)):
                self._consecutive_window_rejections = 0
            scale = float(getattr(self, "_trust_region_lr_scale", 1.0))
            if scale < 1.0:
                recovered = min(1.0, scale * _TRUST_REGION_LR_RECOVERY)
                self._trust_region_lr_scale = recovered
                log.info(
                    "window trust region: accepted iteration %d update "
                    "(mean clip fraction %.4f <= %.4f); lr scale recovers "
                    "%.4f -> %.4f",
                    iteration_index,
                    primary_clip_fraction,
                    bound,
                    scale,
                    recovered,
                )
            return False
        scale = float(getattr(self, "_trust_region_lr_scale", 1.0))
        new_scale = max(_TRUST_REGION_MIN_LR_SCALE, scale * _TRUST_REGION_LR_BACKOFF)
        consecutive = int(getattr(self, "_consecutive_window_rejections", 0)) + 1
        log.error(
            "window trust region: REJECTING iteration %d update (primary mean "
            "clip fraction %.4f > %.4f); restoring pre-window weights and "
            "optimizer state, lr scale %.4f -> %.4f (consecutive rejections: %d)",
            iteration_index,
            primary_clip_fraction,
            bound,
            scale,
            new_scale,
            consecutive,
        )
        self._restore_update_state(snapshot)
        self._trust_region_lr_scale = new_scale
        self._consecutive_window_rejections = consecutive
        if consecutive >= _TRUST_REGION_MAX_CONSECUTIVE_REJECTIONS:
            raise RuntimeError(
                f"window trust region rejected {consecutive} consecutive "
                f"windows (mean clip fraction persistently above {bound}); "
                "learning-rate backoff cannot make the update step safe — "
                "aborting for human inspection."
            )
        return True

    def _trust_region_token_logprobs(
        self, samples: list[TokenizedSample]
    ) -> torch.Tensor:
        """Score a fixed sample without dropout noise or training side effects."""
        modes = [(module, module.training) for module in self.model.modules()]
        cuda_devices = sorted(
            {
                tensor.device.index
                for tensor in (*self.model.parameters(), *self.model.buffers())
                if tensor.is_cuda
            }
        )
        scores = []
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
                # This diagnostic does not backpropagate. Eval mode disables
                # both dropout modules and functional attention dropout even
                # when the training config explicitly allows dropout.
                self.model.eval()
                for sample in samples:
                    batch = collate_samples(
                        [sample],
                        pad_token_id=self.tokenizer.pad_token_id,
                        processor=getattr(self, "processor", None),
                    )
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    values = self._token_logprobs(batch)[
                        batch["completion_mask"][:, 1:]
                    ]
                    if not bool(torch.isfinite(values).all().item()):
                        raise RuntimeError(
                            "nonfinite post-update trust-region logprobs"
                        )
                    scores.append(values.detach().float().cpu())
        finally:
            # Restore mixed parent/child modes exactly, without recursively
            # overwriting a child's independently configured training flag.
            for module, training in modes:
                module.training = training
        return torch.cat(scores) if scores else torch.empty(0)

    def _trust_region_token_count(self, scores: torch.Tensor) -> int:
        """Tokens the trust-region probe scored (FSDP ranks pool their shards)."""
        return int(scores.numel())

    def _trust_region_clip_mask(
        self, samples: list[TokenizedSample], before: torch.Tensor
    ) -> torch.Tensor:
        """Rescore the probe; True marks a token whose ratio left the clip band.

        An empty rescore yields an empty mask, so the caller decides whether a
        missing probe is fatal before any shape check.
        """
        after = self._trust_region_token_logprobs(samples)
        if after.numel() == 0:
            return torch.zeros(0, dtype=torch.bool)
        if (
            before.shape != after.shape
            or not bool(torch.isfinite(before).all().item())
            or not bool(torch.isfinite(after).all().item())
        ):
            raise RuntimeError(
                "window trust-region probe must have matching finite token scores"
            )
        ratios = torch.exp(
            (after - before).clamp(-_PPO_LOG_RATIO_CLAMP, _PPO_LOG_RATIO_CLAMP)
        )
        high = self.config.clip_ratio_high
        high = self.config.clip_ratio_low if high is None else high
        return (ratios < 1 - self.config.clip_ratio_low) | (ratios > 1 + high)

    def _post_update_clip_fraction(
        self, samples: list[TokenizedSample], before: torch.Tensor
    ) -> float | None:
        clipped = self._trust_region_clip_mask(samples, before)
        if clipped.numel() == 0:
            if self._window_clip_fraction_bound() > 0.0:
                raise RuntimeError(
                    "enabled window trust region requires sampled probe tokens"
                )
            return None
        result = float(clipped.float().mean().item())
        log.info(
            "window trust region post-update clip fraction=%.6f tokens=%d",
            result,
            clipped.numel(),
        )
        return result

    def _model_revision(self) -> str | None:
        """Pinned HF revision for model_name (None = HEAD, previous behavior)."""
        revision = str(getattr(self.config, "model_revision", "") or "").strip()
        return revision or None

    def _load_base_model(self, model_kwargs: dict[str, Any]):
        """Load the base checkpoint with the right auto-class.

        training.multimodal=true loads through the image-text auto-class
        (Qwen3_5ForConditionalGeneration keeps its vision tower) so the
        trainer can reproduce the rollout servers' forward at image
        placeholder positions; otherwise the CausalLM path drops the tower
        (text-only harness, tiny test models).
        """
        if getattr(self.config, "multimodal", False):
            base_model = AutoModelForImageTextToText.from_pretrained(
                self.config.model_name,
                revision=self._model_revision(),
                **model_kwargs,
            )
            # The vision tower + projector stay FROZEN (owner decision
            # 2026-07-18: not training the encoder). LoRA only targets the
            # language decoder's projections; this guard makes any accidental
            # adapter placement inside the tower fail loudly.
            visual = getattr(base_model, "visual", None) or getattr(
                getattr(base_model, "model", base_model), "visual", None
            )
            if visual is not None:
                for parameter in visual.parameters():
                    parameter.requires_grad_(False)
            return base_model
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision=self._model_revision(),
            **model_kwargs,
        )

    def _model_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.bf16 else torch.float16

    def _base_policy_model(self) -> torch.nn.Module:
        """Return the causal policy beneath the LoRA wrapper."""
        return self.model.get_base_model()

    @property
    def live_entropy_coef(self) -> float:
        """Entropy coefficient actually applied in the loss.

        Falls back to the fixed ``config.entropy_coef`` until overridden --
        the entropy-target controller (cli-side) overrides it per window via
        ``train_window(entropy_coef=...)``, and the value rides the DDP cycle
        payload so every rank applies the same coefficient.
        """
        override = getattr(self, "_live_entropy_coef", None)
        if override is None:
            return max(0.0, float(self.config.entropy_coef))
        return override

    @live_entropy_coef.setter
    def live_entropy_coef(self, value: float) -> None:
        self._live_entropy_coef = max(0.0, float(value))

    def _reference_token_logprobs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Token logprobs of the frozen base policy: the same forward with the
        LoRA adapters disabled, no gradient. The reference-KL anchor's target."""
        with torch.no_grad(), self.model.disable_adapter():
            return self._token_logprobs(batch).detach()

    def _span_loss_terms(
        self,
        batch: dict[str, torch.Tensor],
        new_token_logprobs: torch.Tensor,
        token_entropies: torch.Tensor,
        reference_token_logprobs: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Span-level PPO loss terms shared by the training pass and the
        observability probes (probe entropy, entropy/policy grad-norm ratio).

        Losses average per SPAN (an unpacked sample is one span; an
        episode-packed sample carries one span per env step), so a span is
        weighted exactly like a legacy per-step sample. Policy, entropy and
        KL losses all use this weighting; their sum is the actor training loss.
        """
        span_ids = batch["span_ids"]
        target_mask_bool = span_ids > 0
        old_token_logprobs = batch["old_token_logprobs"]
        token_advantages = batch["token_advantages"]

        def require_finite_completion_tokens(
            values: torch.Tensor, *, label: str
        ) -> None:
            selected = values.masked_select(target_mask_bool)
            invalid_count = int((~torch.isfinite(selected)).sum().item())
            if invalid_count:
                raise RuntimeError(
                    f"PPO {label} contain {invalid_count} non-finite completion "
                    "token value(s); refusing to run an optimizer step."
                )

        require_finite_completion_tokens(old_token_logprobs, label="rollout logprobs")
        require_finite_completion_tokens(
            new_token_logprobs, label="current-policy logprobs"
        )
        require_finite_completion_tokens(token_advantages, label="advantages")
        require_finite_completion_tokens(token_entropies, label="entropies")

        raw_log_ratio = new_token_logprobs - old_token_logprobs
        if not bool(
            torch.isfinite(raw_log_ratio.masked_select(target_mask_bool)).all()
        ):
            raise RuntimeError(
                "PPO log-ratio computation produced non-finite values on completion "
                "tokens; refusing to run an optimizer step."
            )
        log_ratio = torch.clamp(
            raw_log_ratio,
            min=-_PPO_LOG_RATIO_CLAMP,
            max=_PPO_LOG_RATIO_CLAMP,
        )
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            min=1.0 - self.config.clip_ratio_low,
            max=(
                None
                if self.config.clip_ratio_high is None
                else 1.0 + self.config.clip_ratio_high
            ),
        )
        span_counts = batch["span_counts"]
        row_base = torch.cumsum(span_counts, dim=0) - span_counts
        total_spans = int(span_counts.sum().item())
        device = ratio.device

        def span_mean(
            values: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            selected_span_index = (row_base.unsqueeze(-1) + span_ids - 1)[mask].to(
                device
            )
            sums = torch.zeros(
                total_spans, dtype=values.dtype, device=device
            ).index_add_(0, selected_span_index, values[mask])
            counts = torch.zeros(
                total_spans, dtype=values.dtype, device=device
            ).index_add_(
                0,
                selected_span_index,
                torch.ones_like(values[mask]),
            )
            return sums / torch.clamp(counts, min=1.0), counts

        centering_stats = {}
        if getattr(self.config, "policy_loss", "ppo") == "score_centered_tis":
            from training.score_centering import score_centered_objective

            heads = batch["trainer_topk_logprobs"]
            sampler_heads = batch["sampler_topk_logprobs"]
            if heads.shape[-1] != self.config.score_centering_top_k:
                raise ValueError(
                    "sampler head width differs from score_centering_top_k"
                )
            objective, diagnostics = score_centered_objective(
                new_token_logprobs,
                old_token_logprobs,
                heads,
                sampler_heads,
                token_advantages,
                cap=self.config.tis_ratio_cap,
            )
            clipped_tokens = raw_log_ratio > math.log(self.config.tis_ratio_cap)
            centering_stats = {"score_centering/tokens": float(target_mask_bool.sum())}
            for name, values in diagnostics.items():
                centering_stats[f"score_centering/{name}_sum"] = float(
                    values[target_mask_bool].sum().detach()
                )
            centering_stats["score_centering/capped_sum"] = float(
                clipped_tokens[target_mask_bool].sum()
            )
        elif getattr(self.config, "policy_loss", "ppo") == "dppo_binary_tv":
            objective, policy_mask = dppo_binary_tv_objective(
                new_token_logprobs,
                old_token_logprobs,
                token_advantages,
                threshold=self.config.dppo_tv_threshold,
                ratio_cap=self.config.dppo_importance_ratio_cap,
            )
            clipped_tokens = ~policy_mask
        else:
            objective = torch.minimum(
                ratio * token_advantages, clipped_ratio * token_advantages
            )
            clipped_tokens = ratio != clipped_ratio
        span_mean_ratio, selected_counts = span_mean(ratio, target_mask_bool)
        span_clip_fraction, _ = span_mean(clipped_tokens.float(), target_mask_bool)
        span_policy_loss, _ = span_mean(-objective, target_mask_bool)
        span_entropy, _ = span_mean(token_entropies, target_mask_bool)
        span_entropy_loss = -self.live_entropy_coef * span_entropy
        # Low-variance non-negative k3 estimator for KL(old || current), where
        # the rollout actions are sampled from ``old``. Unlike the raw
        # log-ratio estimator this has exactly zero gradient when the policies
        # match, so an active coefficient injects no drift at the start.
        previous_kl_log_ratio = torch.clamp(
            raw_log_ratio,
            min=-_KL_LOG_RATIO_CLAMP,
            max=_KL_LOG_RATIO_CLAMP,
        )
        previous_kl_per_token = (
            torch.exp(previous_kl_log_ratio) - 1.0 - previous_kl_log_ratio
        )
        span_previous_policy_kl, _ = span_mean(previous_kl_per_token, target_mask_bool)
        span_previous_policy_kl_loss = (
            float(getattr(self.config, "previous_policy_kl_coef", 0.0))
            * span_previous_policy_kl
        )
        # Reference anchor: the same k3 estimator against the base policy's
        # logprobs on the sampled tokens. The gradient reaches the current
        # logprobs only; the reference is a constant.
        reference_kl_coef = float(getattr(self.config, "reference_kl_coef", 0.0))
        if reference_token_logprobs is not None:
            reference_log_ratio = torch.clamp(
                reference_token_logprobs.detach() - new_token_logprobs,
                min=-_KL_LOG_RATIO_CLAMP,
                max=_KL_LOG_RATIO_CLAMP,
            )
            reference_kl_per_token = (
                torch.exp(reference_log_ratio) - 1.0 - reference_log_ratio
            )
        else:
            reference_kl_per_token = torch.zeros_like(new_token_logprobs)
        span_reference_kl, _ = span_mean(reference_kl_per_token, target_mask_bool)
        span_reference_kl_loss = reference_kl_coef * span_reference_kl
        active_spans = selected_counts > 0
        if not bool(active_spans.any()):
            raise RuntimeError(
                "PPO micro-batch has no completion tokens; refusing to run an "
                "optimizer step."
            )
        sequence_weights = batch.get("sequence_weights")
        if sequence_weights is None:
            sequence_weights = torch.ones(
                len(span_counts), dtype=span_policy_loss.dtype, device=device
            )
        # One code path for every batch: each sequence contributes its fixed
        # mass spread over its ACTIVE spans (a span whose tokens were all
        # excluded redistributes its share to the sequence's other spans),
        # and the micro-batch loss is the SUM over sequences — the same
        # magnitude semantics the fixed window denominator assumes.
        # The old all-unit-weights fallback took a global span mean instead,
        # which silently changed the loss scale by ~batch size whenever a
        # micro-batch held more than one sequence.
        active_counts_per_sequence: list[int] = []
        offset = 0
        for count in span_counts.tolist():
            count = int(count)
            active_counts_per_sequence.append(
                int(active_spans[offset : offset + count].sum().item())
            )
            offset += count
        active_count_tensor = torch.tensor(
            active_counts_per_sequence,
            dtype=sequence_weights.dtype,
            device=device,
        ).clamp(min=1.0)
        aggregation = getattr(self.config, "loss_aggregation", "span_mean")
        if aggregation == "sequence_sum_fixed":
            # Undo the temporary span means: every captured completion token
            # contributes once. The optimizer cycle supplies N_rollouts * L_ref,
            # including zero-advantage rollouts and all compaction segments.
            span_weights = selected_counts
        elif aggregation in ("token_mean", "episode_mean", "token_sum_fixed"):
            # Every completion token of a sequence carries equal weight: a
            # span's share of the sequence mass is its token count over the
            # sequence's active-token total, so span_mean * share sums to
            # the token mean over the sequence (DAPO-style aggregation).
            # episode_mean sets the sequence mass to the sequence's share of
            # its episode's tokens, so this same spread gives each token of
            # the episode one equal coefficient.
            sequence_index = torch.repeat_interleave(
                torch.arange(len(span_counts), device=device),
                span_counts.to(device),
            )
            active_token_counts = selected_counts.to(sequence_weights.dtype) * (
                active_spans.to(sequence_weights.dtype)
            )
            sequence_token_totals = torch.zeros(
                len(span_counts), dtype=sequence_weights.dtype, device=device
            ).index_add_(0, sequence_index, active_token_counts)
            span_weights = (
                sequence_weights.to(device)[sequence_index]
                * active_token_counts
                / sequence_token_totals.clamp(min=1.0)[sequence_index]
            )
        else:
            span_weights = torch.repeat_interleave(
                sequence_weights.to(device) / active_count_tensor,
                span_counts.to(device),
            ) * active_spans.to(sequence_weights.dtype)
        policy_loss = (span_policy_loss * span_weights).sum()
        entropy_loss = (span_entropy_loss * span_weights).sum()
        previous_policy_kl_loss = (span_previous_policy_kl_loss * span_weights).sum()
        reference_kl_loss = (span_reference_kl_loss * span_weights).sum()
        if (
            not bool(torch.isfinite(policy_loss).item())
            or not bool(torch.isfinite(entropy_loss).item())
            or not bool(torch.isfinite(previous_policy_kl_loss).item())
            or not bool(torch.isfinite(reference_kl_loss).item())
        ):
            raise RuntimeError(
                "PPO loss computation produced a non-finite value; refusing to run "
                "an optimizer step."
            )
        return {
            "policy_loss": policy_loss,
            "centering_statistics": centering_stats,
            "entropy_loss": entropy_loss,
            "previous_policy_kl_loss": previous_policy_kl_loss,
            "reference_kl_loss": reference_kl_loss,
            "span_reference_kl": span_reference_kl[active_spans],
            "span_mean_ratio": span_mean_ratio[active_spans],
            "span_clip_fraction": span_clip_fraction[active_spans],
            "span_policy_loss": span_policy_loss,
            "span_entropy": span_entropy[active_spans],
            "span_entropy_loss": span_entropy_loss[active_spans],
            "span_previous_policy_kl": span_previous_policy_kl[active_spans],
            "total_spans": total_spans,
        }

    def _micro_batch_forward_backward(
        self,
        chunk: list[TokenizedSample],
        *,
        loss_denominator: float,
        backward_scale: float = 1.0,
        profiler_step: int | None = None,
        iteration_index: int | None = None,
        epoch_index: int | None = None,
    ) -> dict[str, Any]:
        """Collate one micro-batch, run forward + PPO loss + backward, and
        return the batch diagnostics. Shared by rank 0 and the data-parallel
        gradient workers; gradients accumulate on the module parameters."""
        mem = self._memory_logger()
        step = (
            profiler_step
            if profiler_step is not None
            else self._optimizer_step_count + 1
        )
        profiler_extra = {
            "phase": "ppo_optimization",
            "iteration": iteration_index,
            "epoch": epoch_index,
            "microbatch_size": len(chunk),
        }
        with mem.region("batch_to_gpu", step, extra=profiler_extra):
            batch = collate_samples(
                chunk,
                pad_token_id=self.tokenizer.pad_token_id,
                processor=getattr(self, "processor", None),
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}

        pass_timings = []
        with (
            mem.region("actor_forward", step, extra=profiler_extra),
            _optimizer_pass_timer(self.device, pass_timings, "forward_time_s"),
        ):
            # train/mean_entropy is a diagnostic, not the entropy bonus. Keep
            # measuring it even when entropy_coef is zero and the bonus is off.
            new_token_logprobs, token_entropies = self._token_logprobs_and_entropies(
                batch
            )

        reference_token_logprobs = None
        if float(getattr(self.config, "reference_kl_coef", 0.0)) > 0.0:
            with (
                mem.region("reference_forward", step, extra=profiler_extra),
                _optimizer_pass_timer(self.device, pass_timings, "forward_time_s"),
            ):
                reference_token_logprobs = self._reference_token_logprobs(batch)

        with mem.region("loss_compute", step, extra=profiler_extra):
            terms = self._span_loss_terms(
                batch,
                new_token_logprobs,
                token_entropies,
                reference_token_logprobs=reference_token_logprobs,
            )
            span_mean_ratio = terms["span_mean_ratio"]
            span_clip_fraction = terms["span_clip_fraction"]
            span_entropy = terms["span_entropy"]
            span_entropy_loss = terms["span_entropy_loss"]
            span_previous_policy_kl = terms["span_previous_policy_kl"]
            span_reference_kl = terms["span_reference_kl"]
            total_spans = terms["total_spans"]
            loss = (
                terms["policy_loss"]
                + terms["entropy_loss"]
                + terms["previous_policy_kl_loss"]
                + terms["reference_kl_loss"]
            )
            movement_stats = terms["centering_statistics"]
            if (
                getattr(self.config, "policy_loss", "ppo") == "dppo_binary_tv"
                or getattr(self.config, "loss_aggregation", "span_mean")
                == "sequence_sum_fixed"
            ):
                movement_stats.update(
                    policy_update_statistics(
                        new_token_logprobs,
                        batch["old_token_logprobs"],
                        batch["token_advantages"],
                        batch["span_ids"] > 0,
                        tv_threshold=self.config.dppo_tv_threshold,
                        ratio_cap=self.config.dppo_importance_ratio_cap,
                    )
                )

        if not loss.requires_grad:
            raise RuntimeError(
                "PPO optimization loss has no gradient path. "
                f"microbatch_samples={len(chunk)}. "
                "Refusing to export unchanged policy weights."
            )
        with mem.region("backward", step, extra=profiler_extra):
            backward_loss = loss / loss_denominator
            if backward_scale != 1.0:
                backward_loss = backward_loss * backward_scale
            with _optimizer_pass_timer(self.device, pass_timings, "backward_time_s"):
                backward_loss.backward()

        metrics = {
            "mean_ratio": float(span_mean_ratio.mean().detach().cpu().item()),
            "clip_fraction": (
                movement_stats["score_centering/capped_sum"]
                / movement_stats["score_centering/tokens"]
                if "score_centering/tokens" in movement_stats
                else float(span_clip_fraction.mean().detach().cpu().item())
            ),
            "entropy": float(span_entropy.mean().detach().cpu().item()),
            "entropy_loss": float(span_entropy_loss.mean().detach().cpu().item()),
            "previous_policy_kl": float(
                span_previous_policy_kl.mean().detach().cpu().item()
            ),
            "reference_kl": float(span_reference_kl.mean().detach().cpu().item()),
            "loss": float(loss.detach().cpu().item()),
            "spans": float(total_spans),
            "policy_update_statistics": movement_stats,
        }
        metrics["processed_tokens"] = sum(len(sample.input_ids) for sample in chunk)
        metrics["forward_time_s"] = 0.0
        metrics["backward_time_s"] = 0.0
        for name, start, end in pass_timings:
            metrics[name] += (
                start.elapsed_time(end) / 1000.0 if end is not None else start
            )
        return metrics

    def _probe_states_path(self, output_dir: Path) -> Path:
        return Path(output_dir) / _PROBE_STATE_FILENAME

    def _ensure_probe_states(
        self, samples: list[TokenizedSample], output_dir: Path
    ) -> None:
        """Load (resume) or capture (first window with retained samples) the
        frozen probe batch. Rank-0 only. Always active: the probe is the
        composition-stable entropy series, and entropy drift (in either
        direction) is a primary collapse signal regardless of whether the
        entropy controller is steering."""
        if getattr(self, "_probe_samples", None):
            return
        path = self._probe_states_path(output_dir)
        if path.exists():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                probe = list(payload.get("samples", []))
                for sample in probe:
                    # Slotted samples from older snapshots do not acquire new defaults.
                    for name in ("completion_topk_ids", "completion_topk_logprobs"):
                        if not hasattr(sample, name):
                            setattr(sample, name, None)
                    if not hasattr(sample, "token_logit_bias"):
                        sample.token_logit_bias = {}
                if probe:
                    self._probe_samples = probe
                    log.info("loaded %d probe states from %s", len(probe), path)
                    return
                raise ValueError("probe state file contains no samples")
            except Exception as exc:  # noqa: BLE001 - re-capture on any load fault
                log.warning(
                    "failed to load probe states from %s (%s); re-capturing",
                    path,
                    exc,
                )
        if not samples:
            return
        # Deep copy freezes the probe population (token ids and advantages)
        # against downstream mutation of the live batch.
        probe = copy.deepcopy(samples[:_PROBE_STATE_SAMPLE_COUNT])
        self._probe_samples = probe
        path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = path.with_name(f"{path.name}.tmp")
        torch.save({"version": _PROBE_STATE_VERSION, "samples": probe}, staging_path)
        staging_path.replace(path)
        log.info("captured %d probe states to %s", len(probe), path)

    def _compute_probe_entropy(self) -> float:
        """Mean span entropy of the frozen probe batch under torch.no_grad().

        A fixed-population entropy series immune to the composition drift of
        train/mean_entropy (which is computed on trained turns only). Runs on
        rank 0 only, no DDP fan-out; bounded cost (K=8 sequences, no grad).
        """
        probe = getattr(self, "_probe_samples", None)
        if not probe:
            return 0.0
        try:
            return self._compute_probe_entropy_forward(probe)
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            log.warning("probe entropy computation failed: %s", exc)
            return math.nan

    def _compute_probe_entropy_forward(self, probe: Sequence[TokenizedSample]) -> float:
        """Run the probe forward without deciding how failures are handled."""
        # Eval-consistent mode: the same forward path as the training
        # entropy computation, with dropout disabled for determinism.
        self._set_policy_mode_for_logprob_computation()
        entropy_sum, span_total = self._probe_entropy_sums(
            list(self._batch_iter(list(probe)))
        )
        if span_total <= 0:
            return 0.0
        return entropy_sum / span_total

    def _probe_entropy_sums(
        self, chunks: Sequence[list[TokenizedSample]]
    ) -> tuple[float, int]:
        """Span-entropy sum and span count over probe micro-batches."""
        entropy_sum = 0.0
        span_total = 0
        with torch.no_grad():
            for chunk in chunks:
                batch = collate_samples(
                    chunk,
                    pad_token_id=self.tokenizer.pad_token_id,
                    processor=getattr(self, "processor", None),
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                new_token_logprobs, token_entropies = (
                    self._token_logprobs_and_entropies(batch)
                )
                terms = self._span_loss_terms(
                    batch, new_token_logprobs, token_entropies
                )
                span_entropy = terms["span_entropy"]
                entropy_sum += float(span_entropy.sum().item())
                span_total += int(span_entropy.numel())
        return entropy_sum, span_total

    def _entropy_to_policy_grad_ratio(self, chunk: list[TokenizedSample]) -> float:
        """||grad(entropy_loss)|| / ||grad(policy_loss)|| on one micro-batch.

        Observability only (guidance: the ratio should sit <=5-10%). Computed
        once per window BEFORE the training cycles start, on rank 0 only, via
        two torch.autograd.grad calls that never touch param.grad or the
        optimizer state. Data/shape failures report an unavailable value;
        device and numerical runtime failures propagate.
        """
        if not chunk:
            return 0.0
        if self.live_entropy_coef <= 0.0:
            return 0.0
        parameters = self._trainable_parameters()
        if not parameters:
            return 0.0
        try:
            self._set_policy_mode_for_training_forward()
            batch = collate_samples(
                chunk,
                pad_token_id=self.tokenizer.pad_token_id,
                processor=getattr(self, "processor", None),
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            new_token_logprobs, token_entropies = self._token_logprobs_and_entropies(
                batch
            )
            terms = self._span_loss_terms(batch, new_token_logprobs, token_entropies)
            policy_grads = torch.autograd.grad(
                terms["policy_loss"],
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            entropy_grads = torch.autograd.grad(
                terms["entropy_loss"], parameters, allow_unused=True
            )

            def grad_norm(grads: tuple[torch.Tensor | None, ...]) -> float:
                total = 0.0
                for grad in grads:
                    if grad is not None:
                        total += float(grad.detach().float().pow(2).sum().item())
                return math.sqrt(total)

            policy_norm = grad_norm(policy_grads)
            entropy_norm = grad_norm(entropy_grads)
            if policy_norm <= 0.0 or not math.isfinite(policy_norm):
                return 0.0
            if not math.isfinite(entropy_norm):
                return 0.0
            return entropy_norm / policy_norm
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            log.warning("entropy-to-policy grad-ratio probe failed: %s", exc)
            return math.nan
        finally:
            # autograd.grad leaves param.grad untouched, but clear any stale
            # state defensively before the real accumulation pass begins.
            self.optimizer.zero_grad(set_to_none=True)

    @shutdown_at_safe_boundary
    def _execute_train_cycle(
        self,
        cycle: list[list[TokenizedSample]],
        *,
        cycle_index: int,
        epoch_label: str,
        epoch_number: int,
        iteration_index: int,
        loss_denominator: float,
        world_size: int,
        mem: Any,
        accumulators: dict[str, Any],
        expected_optimizer_steps_total: int,
        check_health: Callable[[], None] | None = None,
        max_clip_fraction: float | None = None,
    ) -> float:
        """Run one gradient-accumulation cycle: forward/backward on this
        rank's share, worker fan-out, grad reduce, optimizer step, weight
        broadcast. Returns the cycle's mean forward-phase clip fraction across
        ranks. An extra-cycle bound can reject before the optimizer step; the
        unchanged weights still complete every worker's collective protocol.
        """
        clip_sum_before = float(accumulators["clip_fraction_sum"])
        updates_before = int(accumulators["ratio_updates"])
        if check_health is not None:
            check_health()
        shares = round_robin_split(cycle, world_size)
        if self._ddp_group is not None:
            self._ddp_group.send_cycle(
                shares[1:],
                loss_denominator=loss_denominator,
                entropy_coef=self.live_entropy_coef,
            )
        profiler_step = self._optimizer_step_count + 1
        cycle_work = dict(processed_tokens=0, forward_time_s=0.0, backward_time_s=0.0)
        for chunk in shares[0]:
            # DDP peers must finish the current reduce/broadcast/gather
            # protocol before rank 0 can propagate a health failure.
            if check_health is not None and self._ddp_group is None:
                check_health()
            metrics = self._micro_batch_forward_backward(
                chunk,
                loss_denominator=loss_denominator,
                profiler_step=profiler_step,
                iteration_index=iteration_index,
                epoch_index=epoch_number,
            )
            for key in cycle_work:
                cycle_work[key] += metrics.get(key, 0)
            accumulators["mean_ratio_sum"] += metrics["mean_ratio"]
            accumulators["clip_fraction_sum"] += metrics["clip_fraction"]
            accumulators["entropy_sum"] += metrics["entropy"]
            accumulators["entropy_loss_sum"] += metrics["entropy_loss"]
            accumulators["previous_policy_kl_sum"] = float(
                accumulators.get("previous_policy_kl_sum", 0.0)
            ) + float(metrics.get("previous_policy_kl", 0.0))
            accumulators["reference_kl_sum"] = float(
                accumulators.get("reference_kl_sum", 0.0)
            ) + float(metrics.get("reference_kl", 0.0))
            accumulators["ratio_updates"] = int(accumulators["ratio_updates"]) + 1
            accumulators["final_loss"] = metrics["loss"]
            merge_policy_update_statistics(
                accumulators.setdefault("policy_update_statistics", {}),
                metrics.get("policy_update_statistics", {}),
            )
        if self._ddp_group is not None:
            self._ddp_group.reduce_grads_into_params()
            for worker_metrics in self._ddp_group.gather_worker_metrics():
                cycle_work["processed_tokens"] += worker_metrics.get(
                    "processed_tokens", 0
                )
                for key in ("forward_time_s", "backward_time_s"):
                    cycle_work[key] = max(
                        cycle_work[key], worker_metrics.get(key, 0.0)
                    )
                for key in (
                    "mean_ratio_sum",
                    "clip_fraction_sum",
                    "entropy_sum",
                    "entropy_loss_sum",
                    "previous_policy_kl_sum",
                    "reference_kl_sum",
                ):
                    accumulators[key] = float(
                        accumulators.get(key, 0.0)
                    ) + worker_metrics.get(key, 0.0)
                accumulators["ratio_updates"] = int(
                    accumulators["ratio_updates"]
                ) + int(worker_metrics.get("micro_batches", 0))
                merge_policy_update_statistics(
                    accumulators.setdefault("policy_update_statistics", {}),
                    worker_metrics.get("policy_update_statistics", {}),
                )
        for key, value in cycle_work.items():
            accumulators[key] = accumulators.get(key, 0) + value
        cycle_updates = int(accumulators["ratio_updates"]) - updates_before
        cycle_clip_fraction = (
            (float(accumulators["clip_fraction_sum"]) - clip_sum_before) / cycle_updates
            if cycle_updates
            else 0.0
        )
        if max_clip_fraction is not None and cycle_clip_fraction > max_clip_fraction:
            self.optimizer.zero_grad(set_to_none=True)
            if self._ddp_group is not None:
                # Complete the protocol with unchanged weights; every worker
                # already reduced and cleared this rejected cycle's gradients.
                self._ddp_group.broadcast_weights()
            return cycle_clip_fraction
        with mem.region(
            "optimizer_step",
            profiler_step,
            extra={
                "phase": "ppo_optimization",
                "iteration": iteration_index,
                "epoch": epoch_number,
                "cycle": cycle_index,
            },
        ):
            learning_rate = self._set_optimizer_learning_rate(
                self._optimizer_step_count + 1
            )
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_parameters(),
                    self.config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                .detach()
                .cpu()
                .item()
            )
            accumulators["grad_norm_sum"] += grad_norm
            accumulators["max_grad_norm"] = max(
                float(accumulators["max_grad_norm"]), grad_norm
            )
            accumulators["grad_norm_updates"] = (
                int(accumulators["grad_norm_updates"]) + 1
            )
            if check_health is not None:
                try:
                    check_health()
                except BaseException:
                    if self._ddp_group is not None:
                        # The reduce has completed and peers await weights.
                        # Finish with unchanged weights so shutdown finds
                        # every peer back in its command loop.
                        self._ddp_group.broadcast_weights()
                        self._ddp_group.gather_worker_metrics()
                    raise
            self.optimizer.step()
            if self.device.type == "cuda":
                # Backward already synchronized via metrics; finish the cheap
                # optimizer tail so cycle wall time includes its GPU work.
                torch.cuda.synchronize(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self._optimizer_step_count += 1
        accumulators["global_step"] = int(accumulators["global_step"]) + 1
        if self._ddp_group is not None:
            self._ddp_group.broadcast_weights()
        global_step = int(accumulators["global_step"])
        if (
            global_step == 1
            or global_step == expected_optimizer_steps_total
            or (global_step % 10) == 0
        ):
            log.info(
                "optimizer progress: step %d/%d (%s, lr=%.3e, grad_norm=%.4f, last_loss=%.4f)",
                global_step,
                expected_optimizer_steps_total,
                epoch_label,
                learning_rate,
                grad_norm,
                accumulators["final_loss"],
            )
        cycle_updates = int(accumulators["ratio_updates"]) - updates_before
        if cycle_updates <= 0:
            return 0.0
        return (
            float(accumulators["clip_fraction_sum"]) - clip_sum_before
        ) / cycle_updates

    def _memory_logger(self) -> Any:
        return getattr(self, "memory_logger", NullMemoryLogger())

    def _trainable_parameter_l2_norm(self) -> float:
        total = 0.0
        with torch.no_grad():
            for parameter in self._trainable_parameters():
                total += float(torch.sum(parameter.detach().float() ** 2).cpu().item())
        return math.sqrt(total)

    def _training_state_payload(
        self,
        *,
        iteration_index: int,
        adapter_dir: Path,
        runtime_adapter_dir: Path,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": _TRAINER_STATE_VERSION,
            "iteration": int(iteration_index),
            "next_iteration": int(iteration_index) + 1,
            "adapter_dir": str(adapter_dir),
            "runtime_adapter_dir": str(runtime_adapter_dir),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "optimizer_step_count": self._optimizer_step_count,
            "dry_optimizer_step_count": self._dry_optimizer_step_count,
            "trust_region_lr_scale": float(
                getattr(self, "_trust_region_lr_scale", 1.0)
            ),
            "consecutive_window_rejections": int(
                getattr(self, "_consecutive_window_rejections", 0)
            ),
            "training_config": asdict(self.config),
            "checkpoint_identity": copy.deepcopy(
                getattr(self, "_checkpoint_identity", None)
            ),
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if getattr(self, "state_critic", None) is not None:
            payload["state_critic_state"] = self.state_critic.state_dict()
        return payload

    def _save_training_state(
        self,
        output_dir: Path,
        *,
        iteration_index: int,
        adapter_dir: Path,
        runtime_adapter_dir: Path,
    ) -> Path:
        state_path = output_dir / f"trainer_state_iter_{iteration_index:06d}.pt"
        staging_path = output_dir / f"{state_path.name}.tmp"
        latest_link = output_dir / "latest_trainer_state.pt"
        torch.save(
            self._training_state_payload(
                iteration_index=iteration_index,
                adapter_dir=adapter_dir,
                runtime_adapter_dir=runtime_adapter_dir,
            ),
            staging_path,
        )
        staging_path.replace(state_path)

        if latest_link.is_symlink() or latest_link.exists():
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        try:
            latest_link.symlink_to(state_path.name)
        except OSError:
            shutil.copy2(state_path, latest_link)
        # The CLI publishes/evaluates before it advances training_state.json.
        # Keep the prior trainer state valid throughout that crash window.
        if self.config.checkpoint_keep_last is not None:
            self._cleanup_old_training_states(
                output_dir, keep=self.config.checkpoint_keep_last
            )
        return state_path

    @staticmethod
    def _cleanup_old_training_states(output_dir: Path, *, keep: int) -> None:
        if keep <= 0:
            return
        # Retain the initialized policy's state for later startup/eval audits,
        # independently of the rolling resume-checkpoint retention count.
        state_paths = sorted(
            (
                path
                for path in output_dir.glob("trainer_state_iter_*.pt")
                if path.name != "trainer_state_iter_-00001.pt"
            ),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if len(state_paths) <= keep:
            return
        for path in state_paths[:-keep]:
            path.unlink(missing_ok=True)

    def pin_checkpoint_identity(self, reported: dict[str, Any]) -> None:
        """Pin the harness agent settings a rollout window reported; refuse a
        window whose settings differ from the pinned ones."""
        from training.rollout_identity import pin_harness_agent

        self._checkpoint_identity = pin_harness_agent(
            getattr(self, "_checkpoint_identity", None),
            reported,
            source="this window's rollouts",
        )

    def load_training_state(self, state_path: Path) -> None:
        # Load onto CPU: optimizer.load_state_dict casts floating state to each
        # param's own device (the policy may be sharded across GPUs) while the
        # non-fused AdamW `step` scalars must STAY on CPU. map_location onto a
        # single CUDA device strands `step` tensors there, and any param living
        # on another shard then trips the foreach device-grouping assert at the
        # first optimizer step after a resume.
        try:
            state = torch.load(Path(state_path), map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(Path(state_path), map_location="cpu")
        self._load_training_state_payload(state, state_path=state_path)

    def _load_training_state_payload(self, state: dict[str, Any], *, state_path: Path) -> None:
        version = int(state.get("version", 0))
        if version != _TRAINER_STATE_VERSION:
            raise RuntimeError(
                f"Unsupported trainer checkpoint version {version}; expected {_TRAINER_STATE_VERSION}."
            )

        advantage_estimator = getattr(
            self.config, "advantage_estimator", "state_critic"
        )
        saved_algorithm = state.get("training_config", {}).get(
            "advantage_estimator", "state_critic"
        )
        if saved_algorithm != advantage_estimator:
            raise RuntimeError("Trainer checkpoint advantage estimator mismatch.")
        from training.rollout_identity import adopt_harness_agent, without_harness_agent

        expected_identity = without_harness_agent(
            getattr(self, "_checkpoint_identity", None)
        )
        saved_identity = without_harness_agent(state.get("checkpoint_identity"))
        if saved_identity != expected_identity:
            legacy_arc3 = (
                saved_identity is None
                and advantage_estimator == "state_critic"
                and (
                    expected_identity is None
                    or (
                        expected_identity.get("environment") == "arc3"
                        and "processing_stack" not in expected_identity
                        # An identity-less checkpoint was never augmented.
                        and "rollout_augmentation" not in expected_identity
                    )
                )
            )
            if not legacy_arc3:
                differing = sorted(
                    key
                    for key in set(saved_identity or {}) | set(expected_identity or {})
                    if (saved_identity or {}).get(key) != (expected_identity or {}).get(key)
                )
                raise RuntimeError(
                    "Trainer checkpoint environment, dataset manifest, reward settings, rollout augmentation, or processing stack mismatch: "
                    f"the checkpoint identity differs in {', '.join(differing)}."
                )
        # Harness settings the first window pinned (training/rollout_identity.py)
        # carry over; the next window must report them again.
        self._checkpoint_identity = adopt_harness_agent(
            getattr(self, "_checkpoint_identity", None), state.get("checkpoint_identity")
        )
        if advantage_estimator in {"grpo", "dr_grpo"} and "state_critic_state" in state:
            raise RuntimeError("GRPO checkpoints must not contain a state critic.")

        _validate_policy_resume_recipe(state.get("training_config", {}), self.config)
        if getattr(self.config, "policy_loss", "ppo") == "dppo_binary_tv" and (
            float(state.get("trust_region_lr_scale", 1.0)) != 1.0
            or int(state.get("consecutive_window_rejections", 0)) != 0
        ):
            raise RuntimeError("Cannot resume DPPO with PPO trust-region backoff state")
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._optimizer_step_count = int(state.get("optimizer_step_count", 0))
        self._dry_optimizer_step_count = int(state.get("dry_optimizer_step_count", 0))
        self._trust_region_lr_scale = float(state.get("trust_region_lr_scale", 1.0))
        self._consecutive_window_rejections = int(
            state.get("consecutive_window_rejections", 0)
        )
        if getattr(self, "state_critic", None) is not None:
            if "state_critic_state" not in state:
                raise RuntimeError(
                    "Trainer state lacks state_critic_state; refusing a fresh hidden critic on resume"
                )
            self.state_critic.load_state_dict(state["state_critic_state"])

        python_random_state = state.get("python_random_state")
        if python_random_state is not None:
            random.setstate(python_random_state)
        torch_rng_state = state.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.cpu())
        cuda_rng_state_all = state.get("torch_cuda_rng_state_all")
        if cuda_rng_state_all is not None:
            if not isinstance(cuda_rng_state_all, (list, tuple)):
                raise RuntimeError(f"Invalid CUDA RNG state list in {state_path}")
            for index, rng_state in enumerate(cuda_rng_state_all):
                if (
                    not isinstance(rng_state, torch.Tensor)
                    or rng_state.dtype != torch.uint8
                    or rng_state.layout != torch.strided
                    or rng_state.ndim != 1
                    or rng_state.numel() == 0
                ):
                    raise RuntimeError(
                        f"Invalid CUDA RNG state for saved device {index} in {state_path}"
                    )
            if torch.cuda.is_available():
                visible_devices = torch.cuda.device_count()
                # Validate every saved generator, including devices no longer
                # visible, before changing any live generator. Native format
                # errors are corruption, not a harmless topology difference.
                validator = torch.Generator(device="cuda:0")
                restored_states = []
                for index, rng_state in enumerate(cuda_rng_state_all):
                    restored = rng_state.detach().cpu().contiguous()
                    try:
                        validator.set_state(restored)
                    except (RuntimeError, TypeError) as error:
                        raise RuntimeError(
                            f"Invalid CUDA RNG state for saved device {index} in {state_path}"
                        ) from error
                    restored_states.append(restored)
                for index, rng_state in enumerate(restored_states[:visible_devices]):
                    torch.cuda.set_rng_state(rng_state, device=index)
                if len(restored_states) != visible_devices:
                    log.warning(
                        "CUDA RNG topology changed on resume from %s: saved=%d visible=%d; "
                        "restored %d matching visible-device ordinals, ignored %d absent "
                        "devices, kept initialized RNG states for %d added devices",
                        state_path,
                        len(restored_states),
                        visible_devices,
                        min(len(restored_states), visible_devices),
                        max(0, len(restored_states) - visible_devices),
                        max(0, visible_devices - len(restored_states)),
                    )
            elif cuda_rng_state_all:
                log.warning(
                    "CUDA is unavailable while resuming %s; preserving the saved "
                    "CPU RNG state and not restoring %d CUDA generator states",
                    state_path,
                    len(cuda_rng_state_all),
                )

    def _extract_policy_hidden_batches(
        self, batches: list[PolicyHiddenBatch], *, check_shutdown: bool = True
    ) -> list[PolicyHiddenFeatures]:
        """Read detached decision states using the unchanged policy forward.

        In a distributed round, shutdown checks belong outside the complete
        command/gather protocol. Single-rank extraction checks every batch.
        """
        if not batches:
            return []
        self._set_policy_mode_for_logprob_computation()
        decoder, _lm_head = self._decoder_and_lm_head()
        pad_token_id = int(self.tokenizer.pad_token_id or 0)
        blocks: list[PolicyHiddenFeatures] = []
        with torch.no_grad(), self._autocast_context():
            for batch in batches:
                if check_shutdown:
                    raise_if_shutdown_requested()
                width = len(batch[0][3])
                input_ids = torch.full(
                    (len(batch), width),
                    pad_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask = torch.zeros_like(input_ids)
                keys: list[tuple[int, int]] = []
                rows: list[int] = []
                positions: list[int] = []
                for row, (rollout_index, step_indices, anchors, token_ids) in enumerate(
                    batch
                ):
                    length = len(token_ids)
                    if not length or length > width:
                        raise RuntimeError(
                            "hidden critic received an invalid prompt batch"
                        )
                    input_ids[row, :length] = torch.tensor(
                        token_ids, dtype=torch.long, device=self.device
                    )
                    attention_mask[row, :length] = 1
                    for step_index, anchor in zip(step_indices, anchors, strict=True):
                        if anchor < 0 or anchor >= length:
                            raise RuntimeError(
                                "hidden critic decision anchor is outside its prompt"
                            )
                        keys.append((rollout_index, step_index))
                        rows.append(row)
                        positions.append(anchor)
                hidden = decoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                # One indexed copy per batch, instead of a GPU synchronization
                # and Python conversion for every individual action.
                selected = (
                    hidden[
                        torch.tensor(rows, dtype=torch.long, device=hidden.device),
                        torch.tensor(positions, dtype=torch.long, device=hidden.device),
                    ]
                    .detach()
                    .float()
                    .cpu()
                )
                blocks.append((keys, selected))
        if check_shutdown:
            raise_if_shutdown_requested()
        return blocks

    def _extract_policy_hidden_states(
        self, rollouts: list[RolloutTrajectory]
    ) -> dict[tuple[int, int], list[float]]:
        """Extract last-prompt-token states across existing policy replicas.

        Each verified append-only segment shares one no-grad decoder forward.
        Preserve the original padded batches, splitting whole batches across
        ranks; critic fitting, GAE, and policy gradients remain on their
        existing paths.
        """
        raise_if_shutdown_requested()
        started_at = time.perf_counter()
        plans = plan_policy_hidden_segments(rollouts)
        expected = {
            (i, step.step_index)
            for i, rollout in enumerate(rollouts)
            for step in rollout.steps
        }
        if not plans:
            if expected:
                raise RuntimeError(
                    "hidden critic could not prepare any decision-state features"
                )
            return {}
        max_tokens = max(1, int(self.config.max_sequence_tokens))
        batches = batch_policy_hidden_segments(
            plans,
            max_tokens=max_tokens,
            token_budget=int(self.config.max_micro_batch_tokens) or max_tokens,
        )
        group = self._ddp_group
        world_size = group.world_size if group is not None else 1
        if group is None:
            blocks = self._extract_policy_hidden_batches(batches)
        else:
            shares = distribute_policy_hidden_batches(batches, world_size)
            blocks = []
            for round_index in range(max(map(len, shares))):
                # Bound each collective round by one forward per rank, retaining
                # safe SIGTERM boundaries throughout a multi-minute feature pass.
                local_batches = shares[0][round_index : round_index + 1]
                worker_batches = [
                    share[round_index : round_index + 1] for share in shares[1:]
                ]
                blocks.extend(
                    group.extract_hidden_features(
                        worker_batches,
                        lambda: self._extract_policy_hidden_batches(
                            local_batches, check_shutdown=False
                        ),
                    )
                )
        hidden_by_step = merge_policy_hidden_features(blocks, expected)
        raise_if_shutdown_requested()
        log.info(
            "extracted policy hidden states for %d steps across %d segments "
            "(%d batches, %d replicas) in %.2fs",
            len(hidden_by_step),
            len(plans),
            len(batches),
            world_size,
            time.perf_counter() - started_at,
        )
        return hidden_by_step

    def _compute_step_signals(
        self,
        rollouts: list[RolloutTrajectory],
    ) -> tuple[
        list[list[StepSignal]],
        AdvantageBatchStats,
        BaselineSignalBatch | StateCriticSignalBatch,
    ]:
        self._baseline_feature_seconds = 0.0
        estimator = getattr(getattr(self, "config", None), "advantage_estimator", None)
        if estimator in {"grpo", "dr_grpo"}:
            # Dr. GRPO is GRPO without the group standard-deviation division.
            batch = compute_grpo_signal_batch(
                rollouts,
                group_size=self._grpo_group_size,
                puzzles_per_window=self._grpo_puzzles_per_window,
                normalize_std=estimator == "grpo",
            )
            log.info(
                "GRPO summary: puzzles=%d episodes_per_puzzle=%d reward_mean=%.4f reward_std=%.4f advantage_mean=%.4f advantage_std=%.4f",
                self._grpo_puzzles_per_window,
                self._grpo_group_size,
                batch.stats.mean_reward,
                batch.stats.std_reward,
                batch.stats.mean_advantage,
                batch.stats.std_advantage,
            )
            return batch.per_rollout_signals, batch.stats, batch
        if (
            getattr(getattr(self, "config", None), "value_baseline", None)
            == "group_mean"
        ):
            from training.group_baseline import build_group_mean_signals

            batch = build_group_mean_signals(rollouts)
            log.info(
                "group baseline: rollouts=%d groups=%d min_group=%d "
                "mean_arc_score=%.4f advantage_std=%.6f",
                len(rollouts),
                batch.diagnostics["group_count"],
                batch.diagnostics["group_size_min"],
                batch.mean_target,
                batch.stats.std_advantage,
            )
            return batch.per_rollout_signals, batch.stats, batch
        if getattr(self, "state_critic", None) is not None:
            feature_started = time.perf_counter()
            hidden = self._extract_policy_hidden_states(rollouts)
            self._baseline_feature_seconds = time.perf_counter() - feature_started
            batch = compute_state_critic_step_signals(
                rollouts,
                context_window_tokens=self.config.max_sequence_tokens,
                critic=self.state_critic,
                normalize_advantages=True,
                gae_gamma=self.config.gae_gamma,
                gae_lambda=self.config.gae_lambda,
                advantage_clip=self.config.advantage_clip,
                policy_hidden_by_step=hidden,
            )
            return batch.per_rollout_signals, batch.stats, batch
        raise RuntimeError("no value baseline initialized")

    def _update_baseline(
        self,
        signal_batch: BaselineSignalBatch | StateCriticSignalBatch,
    ) -> tuple[BaselineUpdateStats | StateCriticUpdateStats, float]:
        update_started_at = time.perf_counter()
        config = getattr(self, "config", None)
        if (
            getattr(config, "advantage_estimator", None) in {"grpo", "dr_grpo"}
            or getattr(config, "value_baseline", None) == "group_mean"
        ):
            return BaselineUpdateStats(
                update_steps=0,
                mean_prediction=signal_batch.mean_value,
                mean_target=signal_batch.mean_target,
                mean_absolute_error=signal_batch.diagnostics["mean_absolute_error"],
                explained_variance=signal_batch.explained_variance,
            ), 0.0
        if isinstance(signal_batch, StateCriticSignalBatch):
            if self.state_critic is None:
                raise RuntimeError("hidden critic is not initialized")
            stats = self.state_critic.update(
                signal_batch.observations,
                epochs=self.config.state_critic_epochs_per_window,
                batch_size=self.config.state_critic_batch_size,
                max_grad_norm=self.config.state_critic_max_grad_norm,
            )
            return stats, time.perf_counter() - update_started_at
        raise RuntimeError("no value baseline initialized")

    def _set_policy_mode_for_logprob_computation(self) -> None:
        # Ratio estimation is more stable when old and new logprobs are
        # computed without dropout noise. Keep the model in train mode so
        # Transformers gradient checkpointing layers remain active.
        self.model.train()
        if self.config.disable_dropout:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.eval()

    def _set_policy_mode_for_training_forward(self) -> None:
        self.model.train()

    def _decoder_and_lm_head(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        causal_lm = self._base_policy_model()
        decoder = getattr(causal_lm, "model", None)
        lm_head = getattr(causal_lm, "lm_head", None)
        if not isinstance(decoder, torch.nn.Module) or not isinstance(
            lm_head, torch.nn.Module
        ):
            raise RuntimeError(
                "PPO training requires a causal LM exposing .model (decoder) and "
                f".lm_head submodules; got {type(causal_lm).__name__}."
            )
        return decoder, lm_head

    def _token_scores(
        self,
        batch: dict[str, torch.Tensor],
        *,
        compute_entropies: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder, lm_head = self._decoder_and_lm_head()
        with self._autocast_context():
            # Calling the decoder directly (instead of the CausalLM forward)
            # skips the full-sequence [batch, seq, vocab] logits projection;
            # LoRA modules and checkpointing wrappers live inside the decoder,
            # so both stay active.
            hidden_states = decoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **{
                    key: batch[key]
                    for key in (
                        "pixel_values",
                        "image_grid_thw",
                        "mm_token_type_ids",
                        "position_ids",
                    )
                    if key in batch
                },
            ).last_hidden_state
            scores = token_scores_from_hidden_states(
                hidden_states,
                lm_head=lm_head,
                target_ids=batch["input_ids"][:, 1:],
                target_mask=batch["labels"][:, 1:] != -100,
                compute_entropies=compute_entropies,
                bias_token_ids=batch.get("logit_bias_token_ids"),
                bias_values=batch.get("logit_bias_values"),
                head_ids=(
                    batch["sampler_topk_ids"]
                    if self.config.policy_loss == "score_centered_tis"
                    and (torch.is_grad_enabled() or compute_entropies)
                    else None
                ),
            )
            if len(scores) == 3:
                batch["trainer_topk_logprobs"] = scores[2]
            return scores[:2]

    def _token_logprobs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        token_logprobs, _token_entropies = self._token_scores(
            batch, compute_entropies=False
        )
        return token_logprobs

    def _token_logprobs_and_entropies(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._token_scores(batch, compute_entropies=True)

    def _attach_logprobs(self, samples: list[TokenizedSample]) -> None:
        for sample in samples:
            token_count = max(0, len(sample.input_ids) - 1)
            if len(sample.old_token_logprobs) != token_count:
                raise ValueError(
                    "Training sample is missing rollout token logprobs "
                    f"(trajectory_id={sample.trajectory_id}, step_index={sample.step_index}, "
                    f"old_token_logprobs={len(sample.old_token_logprobs)}, expected={token_count})."
                )
            if len(sample.completion_mask) != len(sample.input_ids):
                raise ValueError(
                    "Training sample completion mask is misaligned "
                    f"(trajectory_id={sample.trajectory_id}, step_index={sample.step_index}, "
                    f"completion_mask={len(sample.completion_mask)}, "
                    f"input_ids={len(sample.input_ids)})."
                )
            for target_index, (old_logprob, is_completion) in enumerate(
                zip(
                    sample.old_token_logprobs,
                    sample.completion_mask[1:],
                    strict=True,
                )
            ):
                if is_completion and not math.isfinite(float(old_logprob)):
                    raise RuntimeError(
                        "Training sample contains a non-finite rollout token logprob; "
                        "refusing to optimize "
                        f"(trajectory_id={sample.trajectory_id}, "
                        f"step_index={sample.step_index}, target_index={target_index}, "
                        f"value={old_logprob!r})."
                    )
            sample.old_logprob = float(sum(sample.old_token_logprobs))

    def _validate_parity_snapshot(
        self,
        snapshot: dict[str, torch.Tensor],
        parameters: dict[str, torch.nn.Parameter],
    ) -> None:
        if not parameters or set(snapshot) != set(parameters):
            raise RuntimeError(
                "Same-checkpoint parity snapshot has different trainable parameter names."
            )
        for name, parameter in parameters.items():
            saved = snapshot[name]
            if (
                not isinstance(saved, torch.Tensor)
                or saved.device.type != "cpu"
                or saved.shape != parameter.shape
                or saved.dtype != parameter.dtype
                or not bool(torch.isfinite(saved).all().item())
            ):
                raise RuntimeError(
                    f"Same-checkpoint parity snapshot is invalid for parameter {name!r} "
                    "(CPU storage, matching shape/dtype and finite values are required)."
                )

    def _remember_parity_policy(self, policy_iteration: int) -> None:
        """Keep the current and preceding sampler policies in CPU memory.

        Snapshots describe weights at a window's start, before any optimizer
        step. Copies never alias CPU parameters. Resume needs no historical
        files: its first collected window uses the restored current policy.
        """
        parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        snapshot = {
            name: parameter.detach().to(device="cpu", copy=True)
            for name, parameter in parameters.items()
        }
        self._validate_parity_snapshot(snapshot, parameters)
        cache = getattr(self, "_parity_policy_snapshots", {})
        if cache and max(cache) > policy_iteration:
            raise RuntimeError(
                "Same-checkpoint parity policy iterations moved backwards."
            )
        if policy_iteration in cache:
            previous = cache[policy_iteration]
            self._validate_parity_snapshot(previous, parameters)
            if any(
                not torch.equal(previous[name], value)
                for name, value in snapshot.items()
            ):
                raise RuntimeError(
                    f"Same-checkpoint parity policy {policy_iteration} changed without "
                    "advancing its iteration."
                )
        cache[policy_iteration] = snapshot
        self._parity_policy_snapshots = {
            version: cache[version] for version in sorted(cache)[-2:]
        }

    def _check_window_logprob_parity(
        self,
        samples: list[TokenizedSample],
        *,
        trainer_policy_iteration: int,
        policy_lag_windows: int | None,
    ) -> dict[str, Any]:
        """Check real sampled tokens under their exact policy, restoring rank 0.

        DDP workers remain idle with the current policy. Only parameter VALUES
        change temporarily: objects, optimizer state, gradients and counters
        are untouched, and no collective broadcasts these temporary weights.
        """
        started_at = time.perf_counter()
        if not samples:
            return {
                "token_count": 0,
                "mean_abs": 0.0,
                "p95_abs": 0.0,
                "max_abs": 0.0,
                "outlier_count": 0,
                "outlier_fraction": 0.0,
                "signed_mean": 0.0,
                "checked_policy_lag": None,
                "sampler_policy_iteration": None,
                "trainer_policy_iteration": trainer_policy_iteration,
                "collection_lag": policy_lag_windows,
                "seconds": time.perf_counter() - started_at,
            }
        actual_lag = _policy_lag_for_samples(
            samples, trainer_policy_iteration=trainer_policy_iteration
        )
        if actual_lag > 1:
            raise RuntimeError(
                f"Same-checkpoint parity requires collection lag at most one; got {actual_lag}."
            )
        if policy_lag_windows is not None and int(policy_lag_windows) != actual_lag:
            raise RuntimeError(
                "Same-checkpoint parity collection lag disagrees with the raw samples: "
                f"reported={policy_lag_windows}, actual={actual_lag}."
            )
        sampler_policy_iteration = trainer_policy_iteration - actual_lag
        cache = getattr(self, "_parity_policy_snapshots", {})
        if (
            sampler_policy_iteration not in cache
            or trainer_policy_iteration not in cache
        ):
            raise RuntimeError(
                "Same-checkpoint parity has no cached weights for sampler policy "
                f"{sampler_policy_iteration} and trainer policy {trainer_policy_iteration}; "
                f"available={sorted(cache)}. Refusing to skip the check."
            )
        parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        current = cache[trainer_policy_iteration]
        sampled = cache[sampler_policy_iteration]
        self._validate_parity_snapshot(current, parameters)
        if actual_lag:
            self._validate_parity_snapshot(sampled, parameters)
        cuda_devices = sorted(
            {
                parameter.device.index
                for parameter in parameters.values()
                if parameter.is_cuda
            }
        )
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
                if actual_lag:
                    for name, parameter in parameters.items():
                        parameter.copy_(sampled[name])
                metrics = self._check_rollout_logprob_parity(
                    samples, trainer_policy_iteration=sampler_policy_iteration
                )
                failure = self._rollout_logprob_parity_failure(
                    metrics, sample_count=len(samples)
                )
                if failure:
                    raise RuntimeError(
                        "Same-checkpoint logprob parity check failed for sampler policy "
                        f"{sampler_policy_iteration} (active trainer policy "
                        f"{trainer_policy_iteration}, collection lag {actual_lag}): {failure}."
                    )
        finally:
            if actual_lag:
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        parameter.copy_(current[name])
            for module, training in modes:
                module.training = training
        log.info(
            "same-checkpoint parity passed: sampler_policy=%d trainer_policy=%d "
            "collection_lag=%d tokens=%d; current policy restored before optimization",
            sampler_policy_iteration,
            trainer_policy_iteration,
            actual_lag,
            metrics["token_count"],
        )
        return {
            **metrics,
            "sampler_policy_iteration": sampler_policy_iteration,
            "trainer_policy_iteration": trainer_policy_iteration,
            "collection_lag": actual_lag,
            "seconds": time.perf_counter() - started_at,
        }

    @staticmethod
    def _parity_result_fields(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            f"rollout_logprob_parity_{name}": metrics.get(name, 0)
            for name in (
                "token_count",
                "mean_abs",
                "p95_abs",
                "max_abs",
                "outlier_count",
                "outlier_fraction",
                "signed_mean",
                "biased_token_count",
                "biased_mean_abs",
                "sampler_policy_iteration",
                "trainer_policy_iteration",
                "collection_lag",
                "checked_policy_lag",
                "seconds",
                "snapshot_seconds",
                "forward_seconds",
                "input_token_count",
                "sample_count",
                "batch_count",
                "startup",
                "expanded",
            )
        }

    def _parity_thresholds(self) -> ParityThresholds:
        """The parity gate's tolerances: training.parity_thresholds for this
        weight mode, else the LoRA/full-weight defaults. FlashPPOTrainer
        reads training.flash.parity_thresholds instead."""
        config = getattr(self, "config", None)
        return getattr(config, "parity_thresholds", {}).get(
            getattr(config, "weight_mode", "lora"), ParityThresholds()
        )

    def _check_rollout_logprob_parity(
        self,
        samples: list[TokenizedSample],
        *,
        trainer_policy_iteration: int = -1,
        token_target: int | None = None,
        _expanded: bool = False,
    ) -> dict[str, float | int]:
        empty_metrics: dict[str, float | int] = {
            "token_count": 0,
            "mean_abs": 0.0,
            "p95_abs": 0.0,
            "max_abs": 0.0,
            "outlier_count": 0,
            "outlier_fraction": 0.0,
            "signed_mean": 0.0,
            "biased_token_count": 0,
            "biased_mean_abs": 0.0,
            "checked_policy_lag": 0,
            "forward_seconds": 0.0,
            "input_token_count": 0,
            "sample_count": 0,
            "batch_count": 0,
            "expanded": _expanded,
        }
        if not samples:
            return empty_metrics

        batch_size = max(1, self.config.micro_batch_size)
        max_batches = max(
            1, int(getattr(self.config, "rollout_logprob_check_batches", 1))
        )
        checked_batches = 0
        checked_samples = 0
        checked_tokens = 0
        input_token_count = 0
        forward_seconds = 0.0
        min_checked_policy_iteration: int | None = None
        abs_diff_chunks: list[torch.Tensor] = []
        biased_abs_diff_chunks: list[torch.Tensor] = []
        signed_diff_sum = 0.0
        nonfinite_count = 0
        worst_detail: dict[str, float | int | str] | None = None

        self._set_policy_mode_for_training_forward()
        # Preserve every token preceding the selected sampled outputs. A
        # right-truncated raw request scores the same causal targets without
        # prefilling later turns from a packed training sequence. The legacy
        # batch-count setting increases evidence, never disables the check.
        remaining_tokens = token_target or _PARITY_MIN_CHECK_TOKENS * max_batches
        selected = []
        for sample in _parity_check_pool(
            samples, policy_iteration=trainer_policy_iteration
        ):
            if remaining_tokens <= 0:
                break
            if not any(sample.completion_mask):
                continue
            prefix = _parity_prefix(sample, remaining_tokens)
            selected.append(prefix)
            log.info(
                "parity coverage: target=%s selected=%s policy=%d step=%d "
                "prompt_tokens=%d checked_tokens=%d",
                _PARITY_CONTEXT_CYCLE[
                    (trainer_policy_iteration + 1) % len(_PARITY_CONTEXT_CYCLE)
                ],
                _parity_context_kind(sample),
                trainer_policy_iteration,
                sample.step_index,
                next(i for i, mask in enumerate(prefix.completion_mask) if mask),
                sum(prefix.completion_mask),
            )
            remaining_tokens -= sum(prefix.completion_mask)
            if not _expanded and len(selected) >= batch_size * max(
                max_batches, _PARITY_MAX_CHECK_BATCHES
            ):
                break
        check_batches = _pack_micro_batches(
            selected,
            batch_size=batch_size,
            token_budget=max(0, int(self.config.max_micro_batch_tokens)),
        )
        with torch.no_grad():
            for chunk in check_batches:
                if not _expanded and checked_batches >= max(
                    max_batches, _PARITY_MAX_CHECK_BATCHES
                ):
                    break
                for sample in chunk:
                    sample_policy = int(getattr(sample, "policy_iteration", -1))
                    if (
                        min_checked_policy_iteration is None
                        or sample_policy < min_checked_policy_iteration
                    ):
                        min_checked_policy_iteration = sample_policy
                batch = collate_samples(
                    chunk,
                    pad_token_id=self.tokenizer.pad_token_id,
                    processor=getattr(self, "processor", None),
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                forward_started = time.perf_counter()
                recomputed_logprobs = self._token_logprobs(batch)
                completion_mask = batch["completion_mask"][:, 1:]
                if not bool(completion_mask.any()):
                    checked_batches += 1
                    checked_samples += len(chunk)
                    continue
                diff = recomputed_logprobs - batch["old_token_logprobs"]
                masked_diff = diff.masked_select(completion_mask)
                invalid_count = int((~torch.isfinite(masked_diff)).sum().item())
                if invalid_count:
                    # Fatal after the pooled evidence (_pool_parity_evidence):
                    # ranks that score in lockstep finish the same forwards
                    # and fail together.
                    nonfinite_count += invalid_count
                    checked_batches += 1
                    checked_samples += len(chunk)
                    continue
                completion_mask_cpu = completion_mask.detach().cpu().bool()
                diff_cpu = diff.detach().float().cpu()
                masked_abs = diff_cpu.abs().masked_fill(~completion_mask_cpu, -1.0)
                batch_max_abs = float(masked_abs.max().item())
                if worst_detail is None or batch_max_abs > float(
                    worst_detail["abs_diff"]
                ):
                    flat_index = int(masked_abs.reshape(-1).argmax().item())
                    target_width = int(masked_abs.shape[1])
                    row_index = flat_index // target_width
                    target_index = flat_index % target_width
                    sample = chunk[row_index]
                    input_index = target_index + 1
                    token_id = int(sample.input_ids[input_index])
                    completion_index = input_index - int(sample.prompt_token_count)
                    old_value = float(
                        batch["old_token_logprobs"][row_index, target_index]
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    )
                    new_value = (
                        float(diff_cpu[row_index, target_index].item()) + old_value
                    )
                    worst_detail = {
                        "trajectory_id": sample.trajectory_id,
                        "step_index": int(sample.step_index),
                        "target_index": int(target_index),
                        "completion_index": int(completion_index),
                        "token_id": token_id,
                        "old_logprob": old_value,
                        "new_logprob": new_value,
                        "signed_diff": float(diff_cpu[row_index, target_index].item()),
                        "abs_diff": batch_max_abs,
                        "prompt_tokens": int(sample.prompt_token_count),
                        "completion_tokens": int(
                            max(0, len(sample.input_ids) - sample.prompt_token_count)
                        ),
                    }
                if masked_diff.numel() == 0:
                    checked_batches += 1
                    checked_samples += len(chunk)
                    continue
                detached_diff = masked_diff.detach().float().cpu()
                bias_values = batch.get("logit_bias_values")
                if bias_values is not None:
                    # Targets the learner rescored under a nonzero bias.
                    biased = (bias_values != 0).any(dim=-1) & completion_mask
                    biased_abs_diff_chunks.append(
                        diff.masked_select(biased).detach().float().cpu().abs()
                    )
                forward_seconds += time.perf_counter() - forward_started
                input_token_count += sum(len(sample.input_ids) for sample in chunk)
                abs_diff_chunks.append(detached_diff.abs())
                signed_diff_sum += float(detached_diff.sum().item())
                checked_tokens += int(detached_diff.numel())
                checked_batches += 1
                checked_samples += len(chunk)

        thresholds = self._parity_thresholds()
        evidence: dict[str, float | int] = {
            "nonfinite_count": nonfinite_count,
            "token_count": checked_tokens,
            "mean_abs": 0.0,
            "p95_abs": 0.0,
            "max_abs": 0.0,
            "outlier_count": 0,
            "inlier_mean_abs": 0.0,
            "signed_mean": 0.0,
            "biased_token_count": 0,
            "biased_mean_abs": 0.0,
        }
        if checked_tokens > 0:
            abs_diffs = torch.cat(abs_diff_chunks)
            inliers = abs_diffs[abs_diffs <= thresholds.outlier_abs]
            biased_abs_diffs = (
                torch.cat(biased_abs_diff_chunks)
                if biased_abs_diff_chunks
                else torch.zeros(0)
            )
            evidence.update(
                mean_abs=float(abs_diffs.mean().item()),
                p95_abs=float(torch.quantile(abs_diffs, 0.95).item()),
                max_abs=float(abs_diffs.max().item()),
                outlier_count=int((abs_diffs > thresholds.outlier_abs).sum().item()),
                # Decides only whether to expand: in 128 tokens one many-nat
                # outlier can carry the mean past tolerance by itself.
                inlier_mean_abs=(
                    float(inliers.mean().item()) if inliers.numel() else 0.0
                ),
                signed_mean=signed_diff_sum / checked_tokens,
                biased_token_count=int(biased_abs_diffs.numel()),
                biased_mean_abs=(
                    float(biased_abs_diffs.mean().item())
                    if biased_abs_diffs.numel()
                    else 0.0
                ),
            )
            if biased_abs_diffs.numel():
                log.info(
                    "rollout logprob parity over %d biased (summary) targets: "
                    "abs_diff mean=%.6g max=%.6g",
                    int(biased_abs_diffs.numel()),
                    float(evidence["biased_mean_abs"]),
                    float(biased_abs_diffs.max().item()),
                )
        # Every decision below reads the pooled evidence, so ranks that score
        # in lockstep expand, pass and fail together.
        evidence = self._pool_parity_evidence(evidence)
        if evidence["nonfinite_count"]:
            raise RuntimeError(
                "Same-policy rollout logprob parity encountered "
                f"{int(evidence['nonfinite_count'])} non-finite completion token value(s); "
                "refusing to train."
            )
        checked_tokens = int(evidence["token_count"])
        if checked_tokens <= 0:
            log.info(
                "rollout logprob parity check skipped: no completion target tokens in first %d batches",
                checked_batches,
            )
            return empty_metrics
        mean_abs = float(evidence["mean_abs"])
        p95_abs = float(evidence["p95_abs"])
        max_abs = float(evidence["max_abs"])
        outlier_count = int(evidence["outlier_count"])
        outlier_fraction = outlier_count / max(1, checked_tokens)
        inlier_mean_abs = float(evidence["inlier_mean_abs"])
        mean_signed = float(evidence["signed_mean"])
        biased_token_count = int(evidence["biased_token_count"])
        biased_mean_abs = float(evidence["biased_mean_abs"])
        log.info(
            "rollout logprob parity check over %d tokens from %d samples/%d batches: "
            "abs_diff mean=%.6g p95=%.6g max=%.6g outliers=%d/%d signed_mean=%.6g",
            checked_tokens,
            checked_samples,
            checked_batches,
            mean_abs,
            p95_abs,
            max_abs,
            outlier_count,
            checked_tokens,
            mean_signed,
        )
        if worst_detail is not None:
            log.info(
                "rollout logprob parity worst token: trajectory_id=%s step_index=%d "
                "target_index=%d completion_index=%d token_id=%d old=%.6g new=%.6g "
                "diff=%.6g abs_diff=%.6g prompt_tokens=%d completion_tokens=%d",
                worst_detail["trajectory_id"],
                int(worst_detail["step_index"]),
                int(worst_detail["target_index"]),
                int(worst_detail["completion_index"]),
                int(worst_detail["token_id"]),
                float(worst_detail["old_logprob"]),
                float(worst_detail["new_logprob"]),
                float(worst_detail["signed_diff"]),
                float(worst_detail["abs_diff"]),
                int(worst_detail["prompt_tokens"]),
                int(worst_detail["completion_tokens"]),
            )
        checked_policy_lag = 0
        if min_checked_policy_iteration is not None:
            checked_policy_lag = max(
                0, int(trainer_policy_iteration) - min_checked_policy_iteration
            )
        metrics = {
            "token_count": checked_tokens,
            "mean_abs": mean_abs,
            "p95_abs": p95_abs,
            "max_abs": max_abs,
            "outlier_count": outlier_count,
            "outlier_fraction": outlier_fraction,
            "signed_mean": mean_signed,
            "biased_token_count": biased_token_count,
            "biased_mean_abs": biased_mean_abs,
            "checked_policy_lag": checked_policy_lag,
            "forward_seconds": forward_seconds,
            "input_token_count": input_token_count,
            "sample_count": checked_samples,
            "batch_count": checked_batches,
            "expanded": _expanded,
        }
        rare_outlier_failure = (
            inlier_mean_abs <= thresholds.mean_abs
            and p95_abs <= thresholds.p95_abs
            and outlier_fraction > thresholds.max_outlier_fraction
        )
        failing_statistics = [
            name
            for name, value, tolerance in (
                ("mean_abs", mean_abs, thresholds.mean_abs),
                ("p95_abs", p95_abs, thresholds.p95_abs),
                (
                    "outlier_fraction",
                    outlier_fraction,
                    thresholds.max_outlier_fraction,
                ),
            )
            if value > tolerance
        ]
        marginal_failure = (
            bool(failing_statistics)
            and inlier_mean_abs <= thresholds.marginal_band_factor * thresholds.mean_abs
            and p95_abs <= thresholds.marginal_band_factor * thresholds.p95_abs
        )
        if not _expanded and (rare_outlier_failure or failing_statistics):
            # Extend the SAME deterministic prefixes for more evidence;
            # all original targets (including the outlier) remain present.
            # Only existing captures are used, once. An insufficient captured
            # population still fails the final, unchanged tolerance gate.
            # Every failure expands, not only marginal ones: a 128-target spot
            # check of a single sample sits far from the floor often enough
            # that failing on it alone stopped healthy runs, while a systematic
            # mismatch cannot pass by adding evidence.
            target = max(
                1000,
                math.ceil(outlier_count / thresholds.max_outlier_fraction),
            )
            if rare_outlier_failure:
                log.info(
                    "expanding sampled-token parity evidence to %d targets for rare-outlier frequency",
                    target,
                )
            else:
                log.info(
                    "expanding sampled-token parity evidence to %d targets for %s failure: %s",
                    target,
                    "marginal" if marginal_failure else "spot-check",
                    ", ".join(failing_statistics),
                )
            expanded = self._check_rollout_logprob_parity(
                samples,
                trainer_policy_iteration=trainer_policy_iteration,
                token_target=target,
                _expanded=True,
            )
            for name in (
                "forward_seconds",
                "input_token_count",
                "sample_count",
                "batch_count",
            ):
                expanded[name] += metrics[name]
            return expanded
        return metrics

    def _pool_parity_evidence(
        self, evidence: dict[str, float | int]
    ) -> dict[str, float | int]:
        """The parity statistics every expand/pass/fail decision reads.

        One process scores alone, so its own statistics decide. Trainers whose
        ranks score the same targets in lockstep pool them across ranks.
        """
        return evidence

    def _rollout_logprob_parity_failure(
        self, metrics: dict[str, float | int], *, sample_count: int
    ) -> str:
        """Return a failure description when parity is outside tolerances."""
        token_count = int(metrics.get("token_count", 0))
        if sample_count > 0 and token_count <= 0:
            raise RuntimeError(
                "Rollout logprob parity check did not compare any completion tokens. "
                "Refusing to optimize because PPO old-policy ratios cannot be trusted."
            )
        mean_abs = float(metrics.get("mean_abs", 0.0))
        p95_abs = float(metrics.get("p95_abs", 0.0))
        max_abs = float(metrics.get("max_abs", 0.0))
        outlier_count = int(metrics.get("outlier_count", 0))
        outlier_fraction = float(metrics.get("outlier_fraction", 0.0))
        checked_policy_lag = int(metrics.get("checked_policy_lag", 0))
        if checked_policy_lag != 0:
            raise RuntimeError(
                "Rollout logprob parity is only valid when trainer and rollout "
                f"use the same policy; received lag {checked_policy_lag}."
            )
        thresholds = self._parity_thresholds()
        mean_tolerance = thresholds.mean_abs
        p95_tolerance = thresholds.p95_abs
        fraction_tolerance = thresholds.max_outlier_fraction
        # A SYSTEMATIC mismatch fails the gate (mean, p95, or the share of
        # tokens more than 2 nats off). One deep-tail token does not: at T=1
        # a draw with probability ~5e-5 lands once per ~20k tokens, and bf16
        # scoring of such a token can differ from the engine by many nats
        # without any template, identity, or adapter fault. max_abs stays in
        # the message as a diagnostic.
        if (
            mean_abs > mean_tolerance
            or p95_abs > p95_tolerance
            or outlier_fraction > fraction_tolerance
        ):
            return (
                "same-policy rollout logprob parity outside tolerances "
                f"(tokens={token_count}, mean_abs={mean_abs:.6g}, p95_abs={p95_abs:.6g}, "
                f"max_abs={max_abs:.6g}, outliers={outlier_count}/{token_count} "
                f"({outlier_fraction:.6g}), "
                "thresholds: "
                f"mean_abs<={mean_tolerance}, "
                f"p95_abs<={p95_tolerance}, "
                f"outlier_abs>{thresholds.outlier_abs} "
                f"fraction<={fraction_tolerance})"
            )
        return ""

    def _batch_iter(self, samples: list[TokenizedSample]):
        batch_size = max(1, self.config.micro_batch_size)
        token_budget = max(0, int(getattr(self.config, "max_micro_batch_tokens", 0)))
        yield from _length_bucketed_micro_batches(
            list(samples), batch_size=batch_size, token_budget=token_budget
        )

    def _extra_pass_cycles(self, samples, *, cycle_size):
        # Only primary epochs reserve a largest-first OOM probe. Every extra
        # pass gives all micro-batches a fresh chance to precede its deadline.
        micro_batches = list(self._batch_iter(samples))
        random.shuffle(micro_batches)
        return _balanced_micro_batch_cycles(micro_batches, cycle_size=cycle_size)

    def _write_policy_adapter(self, path: Path) -> None:
        _save_peft_adapter_without_model_card(self.model, path)

    def _save_latest_adapter(self, output_dir: Path, iteration_index: int) -> Path:
        adapter_dir = output_dir / f"adapter_iter_{iteration_index:06d}"
        staging_dir = output_dir / f"{adapter_dir.name}_tmp"
        latest_link = output_dir / "latest_adapter"

        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        self._write_policy_adapter(staging_dir)

        readme_path = staging_dir / "README.md"
        if readme_path.exists():
            readme_path.unlink()

        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        staging_dir.rename(adapter_dir)

        pinned_interval = self.config.pinned_adapter_interval
        if iteration_index == -1 or (
            pinned_interval > 0 and iteration_index % pinned_interval == 0
        ):
            # Deliberate long-term snapshot for post-hoc forensics; marked at
            # save time so retention stays name-agnostic (see
            # _cleanup_old_adapters).
            (adapter_dir / _PINNED_ADAPTER_MARKER).touch()

        if latest_link.is_symlink() or latest_link.exists():
            if latest_link.is_dir() and not latest_link.is_symlink():
                shutil.rmtree(latest_link)
            else:
                latest_link.unlink()
        try:
            latest_link.symlink_to(adapter_dir.name)
        except OSError:
            shutil.copytree(adapter_dir, latest_link)

        # Retention must outlive every in-flight episode's adapter pin, not
        # just the current window: vLLM keeps serving a request on the adapter
        # it started with and may RE-READ it from disk whenever the LRU cache
        # evicts it mid-request. keep=2 killed run 20260711_133455 — 30-minute
        # episodes span 2-3 of the ~29-minute iterations, a request pinned to
        # adapter_iter_000002 outlived its deletion, and the engine's
        # FileNotFoundError took down the whole node-2 fleet. keep=6 covers
        # ~3 hours of published adapters (~2 GiB) — far beyond any episode.
        if self.config.checkpoint_keep_last is not None:
            self._cleanup_old_adapters(
                output_dir, keep=self.config.checkpoint_keep_last
            )
        return adapter_dir

    def _save_runtime_adapter(
        self, adapter_dir: Path, *, publish_latest: bool = True
    ) -> Path:
        checkpoints_root = adapter_dir.parent
        runtime_root = resolve_runtime_adapter_root(checkpoints_root)
        export_parent = (
            runtime_root / adapter_dir.name if runtime_root is not None else adapter_dir
        )
        runtime_dir = export_parent / "runtime_vllm"
        staging_dir = export_parent / "runtime_vllm_tmp"
        latest_link = checkpoints_root / "latest_runtime_adapter"

        export_parent.mkdir(parents=True, exist_ok=True)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        if runtime_root is not None:
            # Mirror the NFS retention: node-local exports whose checkpoint
            # was pruned go too.
            for stale in runtime_root.iterdir():
                if stale.is_dir() and not (checkpoints_root / stale.name).exists():
                    shutil.rmtree(stale, ignore_errors=True)

        export = export_runtime_lora_adapter(
            source_adapter_dir=adapter_dir,
            output_dir=staging_dir,
        )
        self._complete_runtime_export(adapter_dir, staging_dir)
        staging_dir.rename(runtime_dir)

        if publish_latest:
            if latest_link.is_symlink() or latest_link.exists():
                if latest_link.is_dir() and not latest_link.is_symlink():
                    shutil.rmtree(latest_link)
                else:
                    latest_link.unlink()
            try:
                latest_link.symlink_to(
                    runtime_dir
                    if runtime_root is not None
                    else runtime_dir.relative_to(checkpoints_root)
                )
            except OSError:
                shutil.copytree(runtime_dir, latest_link)

        log.info(
            "exported runtime LoRA adapter %s from %s (rank %s -> %s, alpha %.1f -> %.1f, targets=%s, tensors=%s)",
            runtime_dir,
            adapter_dir,
            export.source_rank,
            export.runtime_rank,
            export.source_lora_alpha,
            export.runtime_lora_alpha,
            ",".join(export.runtime_target_modules),
            export.converted_tensor_count,
        )
        return runtime_dir

    def _complete_runtime_export(self, adapter_dir: Path, runtime_dir: Path) -> None:
        """Add trainer-specific files to a runtime export before it is published."""

    def _cleanup_old_adapters(
        self,
        output_dir: Path,
        *,
        keep: int,
        min_age_seconds: float = _ADAPTER_RETENTION_MIN_AGE_SECONDS,
        hard_cap: int = _ADAPTER_RETENTION_HARD_CAP,
    ) -> None:
        """Prune old adapter exports, bounded in COUNT and in TIME.

        A count-only bound is unsafe: in-flight episodes and eval cycles pin
        an adapter for up to ~35 minutes (vLLM re-reads it from disk on LRU
        eviction mid-request), and the iteration cadence is not bounded below
        — run 20260716 death-spiraled to ~30-second iterations on zero-action
        ERROR waves after a server crash, churning keep=6 in under 3 minutes
        and deleting adapters other servers still had pinned. Any adapter
        younger than ``min_age_seconds`` is therefore kept regardless of
        count; ``hard_cap`` bounds disk usage if a spiral ever outpaces the
        age window anyway.

        Snapshot dirs carrying the PINNED marker are exempt from every rule
        up to the ``_PINNED_ADAPTER_CAP`` newest pinned dirs; the count, age,
        and hard-cap bounds apply to the remaining (prunable) dirs only, so
        the spiral protection is unaffected by how many snapshots exist.
        """
        if keep <= 0:
            return
        adapters = sorted(
            (
                path
                for path in output_dir.glob("adapter_iter_*")
                if path.is_dir() and not path.name.endswith("_tmp")
            ),
            key=lambda path: path.name,
        )
        pinned = [path for path in adapters if (path / _PINNED_ADAPTER_MARKER).exists()]
        exempt = set(pinned[-_PINNED_ADAPTER_CAP:])
        prunable = [path for path in adapters if path not in exempt]
        if len(prunable) <= keep:
            return
        now = time.time()
        hard_excess = set(prunable[:-hard_cap]) if len(prunable) > hard_cap else set()
        for path in prunable[:-keep]:
            if path not in hard_excess:
                try:
                    if now - path.stat().st_mtime < min_age_seconds:
                        continue
                except OSError:
                    continue
            shutil.rmtree(path, ignore_errors=True)

    def _export_policy_snapshot(
        self, output_dir: Path, iteration_index: int
    ) -> tuple[Path, Path, float]:
        policy_export_started_at = time.perf_counter()
        adapter_dir = self._save_latest_adapter(output_dir, iteration_index)
        runtime_adapter_dir = self._save_runtime_adapter(adapter_dir)
        return (
            adapter_dir,
            runtime_adapter_dir,
            time.perf_counter() - policy_export_started_at,
        )

    def _finalize_window_artifacts(
        self,
        output_dir: Path,
        iteration_index: int,
        *,
        check_health: Callable[[], None] | None = None,
    ) -> WindowArtifacts:
        adapter_dir, runtime_adapter_dir, policy_export_seconds = (
            self._export_policy_snapshot(output_dir, iteration_index)
        )
        if check_health is not None:
            check_health()
        trainer_state_path = self._save_training_state(
            output_dir,
            iteration_index=iteration_index,
            adapter_dir=adapter_dir,
            runtime_adapter_dir=runtime_adapter_dir,
        )
        return WindowArtifacts(
            publication_path=runtime_adapter_dir,
            checkpoint_path=trainer_state_path,
            adapter_dir=adapter_dir,
            policy_export_seconds=policy_export_seconds,
        )

    def train_window(
        self,
        rollouts: list[RolloutTrajectory],
        *,
        output_dir: Path,
        iteration_index: int,
        sample_budget_fraction: float = 1.0,
        policy_lag_windows: int | None = None,
        entropy_coef: float | None = None,
        next_window_ready: Callable[[], bool] | None = None,
        next_window_seconds_remaining: Callable[[], float | None] | None = None,
        check_health: Callable[[], None] | None = None,
    ) -> WindowTrainResult:
        if check_health is not None:
            check_health()
        # Dry-run state is per-iteration; never let it leak across windows.
        train_window_started_at = time.perf_counter()
        if entropy_coef is not None:
            self.live_entropy_coef = max(0.0, float(entropy_coef))
        mem = self._memory_logger()
        trainer_policy_iteration = _trainer_policy_iteration_at_window_start(
            iteration_index
        )
        parity_startup = not bool(getattr(self, "_parity_policy_snapshots", {}))
        parity_snapshot_started = time.perf_counter()
        self._remember_parity_policy(trainer_policy_iteration)
        parity_snapshot_seconds = time.perf_counter() - parity_snapshot_started

        sample_build_started_at = train_window_started_at
        format_break_count = sum(
            1
            for rollout in rollouts
            for step in getattr(rollout, "steps", []) or []
            if isinstance(getattr(step, "metadata", None), dict)
            and step.metadata.get("format_break")
        )
        if format_break_count:
            log.info(
                "%d sampled replies were malformed; all retain their episode advantage",
                format_break_count,
            )
        per_rollout_signals, advantage_stats, baseline_signal_batch = (
            self._compute_step_signals(rollouts)
        )
        raw_samples = build_samples(rollouts, per_rollout_signals=per_rollout_signals)
        token_usage = summarize_token_usage(rollouts, raw_samples)
        max_sequence_tokens = max(1, int(self.config.max_sequence_tokens))
        _assert_samples_fit_training_budgets(
            raw_samples, max_sequence_tokens=max_sequence_tokens
        )
        # Select witnessed raw requests before packing can attach
        # later turns. Validate the complete population before filtering or
        # baseline updates, even when no episode has a learning signal.
        self._attach_logprobs(raw_samples)
        rollout_logprob_parity = self._check_window_logprob_parity(
            raw_samples,
            trainer_policy_iteration=trainer_policy_iteration,
            policy_lag_windows=policy_lag_windows,
        )
        rollout_logprob_parity["snapshot_seconds"] = parity_snapshot_seconds
        rollout_logprob_parity["seconds"] += parity_snapshot_seconds
        rollout_logprob_parity["startup"] = parity_startup
        policy_lag_windows = rollout_logprob_parity["collection_lag"]
        packed_episodes = 0
        packed_sequences = 0
        segment_boundaries = 0
        if getattr(self.config, "episode_packing", False):
            packed_episodes = len({sample.trajectory_id for sample in raw_samples})
            raw_samples, packing_stats = pack_episode_samples(
                raw_samples,
                max_sequence_tokens=max_sequence_tokens,
                tokenizer=self.tokenizer,
                loss_aggregation=self.config.loss_aggregation,
            )
            _assert_packing_identity(packing_stats, packed_episodes)
            packed_sequences = int(packing_stats.output_samples)
            segment_boundaries = int(packing_stats.segment_boundaries)
            log.info(
                "episode packing: %d step samples -> %d sequences "
                "(%d packed carrying %d spans, %d passthrough, coverage=%.1f%%, "
                "segment_boundaries=%d, nesting_breaks=%d, length_breaks=%d)",
                packing_stats.input_samples,
                packing_stats.output_samples,
                packing_stats.packed_sequences,
                packing_stats.packed_spans,
                packing_stats.passthrough_samples,
                100.0 * packing_stats.coverage,
                packing_stats.segment_boundaries,
                packing_stats.nesting_breaks,
                packing_stats.length_breaks,
            )
        else:
            assign_loss_weights(
                raw_samples, loss_aggregation=self.config.loss_aggregation
            )
        _assert_samples_fit_training_budgets(
            raw_samples,
            max_sequence_tokens=max_sequence_tokens,
        )
        # Packing may combine target arrays; validate their alignment too.
        self._attach_logprobs(raw_samples)
        raw_transition_reward_nonzero_fraction = _sample_fraction(
            raw_samples, lambda sample: abs(float(sample.reward)) > 1e-8
        )
        raw_transition_reward_positive_fraction = _sample_fraction(
            raw_samples, lambda sample: float(sample.reward) > 0.0
        )
        raw_transition_reward_sum = sum(float(sample.reward) for sample in raw_samples)
        raw_positive_transition_reward_sample_count = _sample_count(
            raw_samples, lambda sample: float(sample.reward) > 0.0
        )
        samples, skipped_zero_signal_samples = _filter_zero_signal_samples(
            raw_samples,
            previous_policy_kl_coef=self.config.previous_policy_kl_coef,
            reference_kl_coef=self.config.reference_kl_coef,
            entropy_coef=self.live_entropy_coef,
        )
        fixed_sequence_sum = self.config.loss_aggregation == "sequence_sum_fixed"
        rollout_minibatching = (
            fixed_sequence_sum or self.config.policy_loss == "dppo_binary_tv"
        )
        if rollout_minibatching and samples:
            # Retain quiet episodes alongside learning episodes; their count
            # belongs in the fixed denominator and their tokens in diagnostics.
            # A completely zero-signal window still skips optimizer mutation.
            samples = raw_samples
            skipped_zero_signal_samples = 0
        # Undo the filter's coupling of effective step size to the number of
        # quiet turns: the scale rides every cycle's loss_denominator (which
        # already reaches the DDP workers in the send_cycle payload).
        # Steps and packed segments both carry generated-token mass.
        # Counting sequences would couple the update to compaction frequency
        # instead of the intended token objective.
        raw_mass = sum(sample.sequence_weight for sample in raw_samples)
        retained_mass = sum(sample.sequence_weight for sample in samples)
        log.info(
            "PPO loss weighting=%s full_window_mass=%.0f retained_mass=%.0f",
            {
                "token_mean": "global_token_mean",
                "episode_mean": "episode_token_mean",
                "span_mean": "episode_span_mean",
                "sequence_sum_fixed": "original_rollout_token_sum_fixed",
                "token_sum_fixed": "window_token_sum_fixed",
            }[self.config.loss_aggregation],
            raw_mass,
            retained_mass,
        )
        zero_signal_denominator_scale = _zero_signal_denominator_scale(
            retained_count=retained_mass,
            filtered_count=raw_mass - retained_mass,
        )
        if skipped_zero_signal_samples > 0:
            log.info(
                "skipped %d/%d train samples with zero advantage and no active actor auxiliary loss "
                "(loss denominator scaled by %.4f)",
                skipped_zero_signal_samples,
                len(raw_samples),
                zero_signal_denominator_scale,
            )
        sample_build_seconds = time.perf_counter() - sample_build_started_at
        if check_health is not None and isinstance(
            baseline_signal_batch, StateCriticSignalBatch
        ):
            check_health()
        baseline_update, baseline_update_seconds = self._update_baseline(
            baseline_signal_batch
        )
        expected_loss_mass = sum(sample.sequence_weight for sample in samples)
        eligible_sample_count = len(samples)
        sample_budget_fraction = min(1.0, max(0.0, float(sample_budget_fraction)))
        sample_budget_step_multiple = max(
            1,
            int(self.config.micro_batch_size)
            * int(self.config.gradient_accumulation_steps)
            * self._training_world_size(),
        )
        samples, subsampled_sample_count = _subsample_train_samples(
            samples,
            fraction=sample_budget_fraction,
            step_multiple=sample_budget_step_multiple,
        )
        # Keep the inverse-inclusion weights from subsampling. Their total
        # equals the full loss mass in expectation, not in every draw.
        if subsampled_sample_count > 0:
            log.info(
                "subsampled %d/%d eligible train samples after cross-game shuffle "
                "(sample_budget_fraction=%.4f, step_multiple=%d)",
                subsampled_sample_count,
                eligible_sample_count,
                sample_budget_fraction,
                sample_budget_step_multiple,
            )
        # Load-or-capture the frozen probe batch (capture happens on the
        # first window that has retained samples).
        self._ensure_probe_states(samples, output_dir)
        if not samples:
            optimizer_skipped_reason = "zero_nonzero_advantage_samples"
            log.warning(
                "PPO training prepared zero nonzero-advantage samples; skipping policy optimizer and exporting unchanged adapter. "
                "raw_samples=%d zero_signal_samples=%d eligible_samples=%d rollouts=%d "
                "sample_budget_fraction=%.4f transition_reward_nonzero_fraction_raw=%.6f "
                "transition_reward_positive_fraction_raw=%.6f transition_reward_sum_raw=%.6f "
                "baseline_mean_target=%.6f baseline_mean_raw_advantage=%.6f "
                "baseline_std_raw_advantage=%.6f",
                len(raw_samples),
                skipped_zero_signal_samples,
                eligible_sample_count,
                len(rollouts),
                sample_budget_fraction,
                raw_transition_reward_nonzero_fraction,
                raw_transition_reward_positive_fraction,
                raw_transition_reward_sum,
                baseline_signal_batch.mean_target,
                baseline_signal_batch.mean_raw_advantage,
                baseline_signal_batch.std_raw_advantage,
            )
            optimization_started_at = time.perf_counter()
            cuda_optimization_start_allocated_gb = self._cuda_memory_allocated_gb()
            self._cuda_reset_peak_memory_stats()
            trainable_param_norm_before = self._trainable_parameter_l2_norm()
            optimization_seconds = time.perf_counter() - optimization_started_at
            cuda_optimization_peak_delta_gb = max(
                0.0,
                self._cuda_max_memory_allocated_gb()
                - cuda_optimization_start_allocated_gb,
            )
            cuda_optimization_end_allocated_gb = self._cuda_memory_allocated_gb()
            cuda_optimization_persistent_delta_gb = (
                cuda_optimization_end_allocated_gb
                - cuda_optimization_start_allocated_gb
            )
            with mem.region(
                "checkpoint_save",
                self._optimizer_step_count,
                extra={
                    "phase": "checkpoint_save",
                    "iteration": iteration_index,
                },
            ):
                if check_health is not None:
                    check_health()
                window_artifacts = self._finalize_window_artifacts(
                    output_dir, iteration_index, check_health=check_health
                )
                adapter_dir = window_artifacts.adapter_dir
                runtime_adapter_dir = window_artifacts.publication_path
                trainer_state_path = window_artifacts.checkpoint_path
                policy_export_seconds = window_artifacts.policy_export_seconds
            total_train_window_seconds = time.perf_counter() - train_window_started_at
            trainable_param_norm_after = self._trainable_parameter_l2_norm()
            return WindowTrainResult(
                sample_count=0,
                raw_sample_count=len(raw_samples),
                eligible_sample_count=eligible_sample_count,
                filtered_zero_signal_sample_count=skipped_zero_signal_samples,
                subsampled_sample_count=subsampled_sample_count,
                sample_budget_fraction=sample_budget_fraction,
                sample_budget_step_multiple=sample_budget_step_multiple,
                trainable_parameter_count=self._trainable_parameter_count,
                mean_advantage=0.0,
                std_advantage=0.0,
                mean_return_to_go=0.0,
                mean_advantage_by_level={},
                mean_old_logprob=0.0,
                final_loss=0.0,
                mean_entropy=0.0,
                entropy_loss=0.0,
                entropy_coef_used=self.live_entropy_coef,
                final_learning_rate=self._current_learning_rate(),
                optimizer_steps=0,
                mean_ratio=1.0,
                mean_clip_fraction=0.0,
                mean_grad_norm=0.0,
                max_grad_norm=0.0,
                transition_reward_nonzero_fraction_raw=raw_transition_reward_nonzero_fraction,
                transition_reward_positive_fraction_raw=raw_transition_reward_positive_fraction,
                transition_reward_sum_raw=raw_transition_reward_sum,
                transition_reward_nonzero_fraction_kept=0.0,
                transition_reward_positive_fraction_kept=0.0,
                transition_reward_sum_kept=0.0,
                return_to_go_nonzero_fraction_kept=0.0,
                return_to_go_positive_fraction_kept=0.0,
                advantage_nonzero_fraction_kept=0.0,
                advantage_positive_fraction_kept=0.0,
                positive_transition_reward_sample_count_raw=raw_positive_transition_reward_sample_count,
                positive_transition_reward_sample_count_kept=0,
                trainable_param_norm_before=trainable_param_norm_before,
                trainable_param_norm_after=trainable_param_norm_after,
                advantage_group_mean_reward=advantage_stats.mean_reward,
                advantage_group_std_reward=advantage_stats.std_reward,
                total_completion_tokens=token_usage.total_completion_tokens,
                total_env_steps=token_usage.total_env_steps,
                avg_actions_per_completed_level=_avg_actions_per_completed_level(
                    rollouts
                ),
                avg_tokens_per_rollout=token_usage.avg_tokens_per_rollout,
                avg_tokens_per_step=token_usage.avg_tokens_per_step,
                avg_tokens_per_model_turn=token_usage.avg_tokens_per_model_turn,
                avg_prompt_tokens_per_sample=0.0,
                avg_history_messages_per_sample=0.0,
                sample_build_seconds=sample_build_seconds,
                optimization_seconds=optimization_seconds,
                policy_export_seconds=policy_export_seconds,
                total_train_window_seconds=total_train_window_seconds,
                adapter_dir=adapter_dir,
                runtime_adapter_dir=runtime_adapter_dir,
                trainer_state_path=trainer_state_path,
                cuda_trainer_model_allocated_gb=self._cuda_trainer_model_memory_allocated_gb(),
                cuda_optimization_peak_delta_gb=cuda_optimization_peak_delta_gb,
                cuda_optimization_persistent_delta_gb=cuda_optimization_persistent_delta_gb,
                **self._parity_result_fields(rollout_logprob_parity),
                baseline_feature_seconds=self._baseline_feature_seconds,
                baseline_loss=baseline_update.loss,
                baseline_update_steps=baseline_update.update_steps,
                baseline_mean_value=baseline_signal_batch.mean_value,
                baseline_mean_target=baseline_signal_batch.mean_target,
                baseline_explained_variance=(baseline_signal_batch.explained_variance),
                baseline_post_update_mean_value=(baseline_update.mean_prediction),
                baseline_post_update_mean_absolute_error=(
                    baseline_update.mean_absolute_error
                ),
                baseline_post_update_explained_variance=(
                    baseline_update.explained_variance
                ),
                baseline_mean_raw_advantage=baseline_signal_batch.mean_raw_advantage,
                baseline_std_raw_advantage=baseline_signal_batch.std_raw_advantage,
                baseline_update_seconds=baseline_update_seconds,
                baseline_metrics=(
                    dict(baseline_signal_batch.diagnostics)
                    if not isinstance(baseline_signal_batch, StateCriticSignalBatch)
                    else {}
                ),
                critic_metrics=(
                    dict(baseline_signal_batch.diagnostics)
                    if isinstance(baseline_signal_batch, StateCriticSignalBatch)
                    else {}
                ),
                optimizer_skipped_reason=optimizer_skipped_reason,
                zero_signal_denominator_scale=zero_signal_denominator_scale,
                packed_episodes=packed_episodes,
                packed_sequences=packed_sequences,
                segment_boundaries=segment_boundaries,
                probe_entropy=self._compute_probe_entropy(),
                entropy_to_policy_grad_ratio=0.0,
            )
        sample_length_summary = summarize_sample_lengths(samples)
        log.info(
            "prepared %d train samples from %d rollouts on device %s in %.2fs",
            len(samples),
            len(rollouts),
            self.device,
            sample_build_seconds,
        )
        log.info(
            "train sample lengths: input_tokens min=%d p50=%d p95=%d max=%d; completion_tokens min=%d p50=%d p95=%d max=%d",
            sample_length_summary.min_input_tokens,
            sample_length_summary.median_input_tokens,
            sample_length_summary.p95_input_tokens,
            sample_length_summary.max_input_tokens,
            sample_length_summary.min_completion_tokens,
            sample_length_summary.median_completion_tokens,
            sample_length_summary.p95_completion_tokens,
            sample_length_summary.max_completion_tokens,
        )
        log.info(
            "train prompt context: avg_prompt_tokens=%.1f",
            sample_length_summary.avg_prompt_tokens,
        )
        avg_history_messages_per_sample = sum(
            max(0, sample.prompt_message_count - 2) for sample in samples
        ) / len(samples)
        log.info(
            "train prompt context: avg_history_messages_per_sample=%.2f",
            avg_history_messages_per_sample,
        )
        optimizer_skipped_reason = ""

        final_loss = 0.0
        global_step = 0
        grad_accum = max(1, self.config.gradient_accumulation_steps)
        world_size = self._training_world_size()
        # One optimizer step consumes grad_accum micro-batches PER REPLICA.
        cycle_size = grad_accum * world_size
        fixed_rollout_cycles = []
        if rollout_minibatching:
            if sample_budget_fraction != 1.0:
                raise ValueError(
                    "rollout minibatching requires the complete rollout batch"
                )
            fixed_rollout_cycles = [
                (list(self._batch_iter(group)), denominator)
                for group, denominator in _rollout_loss_batches(
                    samples,
                    rollout_ids=[rollout.trajectory_id for rollout in rollouts],
                    minibatch_rollouts=self.config.optimizer_minibatch_rollouts,
                    aggregation=self.config.loss_aggregation,
                    normalization_tokens=self.config.loss_normalization_tokens,
                )
                if group
            ]
            epoch_micro_batches = [
                [
                    chunk
                    for cycle, _denominator in fixed_rollout_cycles
                    for chunk in cycle
                ]
                for _ in range(max(0, int(self.config.epochs_per_window)))
            ]
            expected_optimizer_steps_total = len(fixed_rollout_cycles) * len(
                epoch_micro_batches
            )
        else:
            epoch_micro_batches = [
                list(self._batch_iter(samples))
                for _ in range(max(0, int(self.config.epochs_per_window)))
            ]
            expected_optimizer_steps_total = sum(
                math.ceil(len(micro_batches) / cycle_size)
                for micro_batches in epoch_micro_batches
            )
        cuda_optimization_start_allocated_gb = self._cuda_memory_allocated_gb()
        self._cuda_reset_peak_memory_stats()
        optimization_started_at = time.perf_counter()
        cycle_accumulators: dict[str, Any] = {
            "mean_ratio_sum": 0.0,
            "clip_fraction_sum": 0.0,
            "entropy_sum": 0.0,
            "entropy_loss_sum": 0.0,
            "previous_policy_kl_sum": 0.0,
            "reference_kl_sum": 0.0,
            "ratio_updates": 0,
            "grad_norm_sum": 0.0,
            "max_grad_norm": 0.0,
            "grad_norm_updates": 0,
            "final_loss": 0.0,
            "global_step": 0,
            "policy_update_statistics": {},
        }
        trainable_param_norm_before = self._trainable_parameter_l2_norm()
        self._set_policy_mode_for_logprob_computation()
        extra_cycles_run = 0
        extra_optimizer_steps = 0
        extra_phase_seconds = 0.0
        extra_stop_reason = ""
        primary_optimization_seconds: float | None = None
        primary_cycle_seconds: list[float] = []
        optimizer_cycle_seconds = 0.0

        window_denominator = _window_loss_denominator(
            samples,
            micro_batch_count=(
                len(epoch_micro_batches[0]) if epoch_micro_batches else 0
            ),
            cycle_size=cycle_size,
            expected_mass=expected_loss_mass,
        )

        # Use the same fixed token-mass denominator with or without packing.
        cycle_loss_denominator = window_denominator * zero_signal_denominator_scale
        if self.config.loss_aggregation == "token_sum_fixed":
            cycle_loss_denominator *= _fixed_token_sum_scale(
                rollout_count=len(rollouts),
                normalization_tokens=self.config.loss_normalization_tokens,
                window_token_mass=raw_mass,
            )

        entropy_to_policy_grad_ratio = 0.0
        window_update_rejected = False
        trust_region_snapshot: dict[str, Any] | None = None
        trust_samples: list[TokenizedSample] = []
        trust_before = torch.empty(0)
        post_update_clip_fraction = None
        try:
            if not optimizer_skipped_reason:
                self._set_policy_mode_for_training_forward()
                if self._window_clip_fraction_bound() > 0.0:
                    # Window trust region: hold a pre-update snapshot so a
                    # destructive primary update can be measured against its
                    # own starting policy and rejected before publication.
                    trust_region_snapshot = self._snapshot_update_state()
                    trust_samples = [
                        _parity_prefix(sample, 128) for sample in samples[:8]
                    ]
                    trust_before = self._trust_region_token_logprobs(trust_samples)
                    if self._trust_region_token_count(trust_before) == 0:
                        raise RuntimeError(
                            "enabled window trust region requires sampled probe tokens"
                        )
                # Once per window, BEFORE the training cycles: measure the
                # entropy-vs-policy gradient-norm ratio on the first retained
                # micro-batch (rank 0 only, optimizer state untouched; the
                # method itself gates on entropy control + a live coef > 0).
                if epoch_micro_batches and epoch_micro_batches[0]:
                    entropy_to_policy_grad_ratio = self._entropy_to_policy_grad_ratio(
                        epoch_micro_batches[0][0]
                    )
                for epoch_index, micro_batches in enumerate(epoch_micro_batches):
                    total_micro_batches = len(micro_batches)
                    expected_optimizer_steps = (
                        len(fixed_rollout_cycles)
                        if rollout_minibatching
                        else math.ceil(total_micro_batches / cycle_size)
                    )
                    length_batch_stats = _micro_batch_length_stats(micro_batches)
                    if length_batch_stats:
                        log.info(
                            "length-bucketed train micro-batches: count=%d, mean_size=%.2f, max_size=%d, max_length_spread=%d, padding_waste=%.2f%%, largest_padded_tokens=%d",
                            length_batch_stats["micro_batch_count"],
                            length_batch_stats["mean_micro_batch_size"],
                            length_batch_stats["max_micro_batch_size"],
                            length_batch_stats["max_length_spread"],
                            100.0 * length_batch_stats["padding_waste_fraction"],
                            length_batch_stats["largest_padded_tokens"],
                        )
                    if self.config.max_micro_batch_tokens > 0:
                        log.info(
                            "starting train epoch %d/%d with %d micro-batches and %d expected optimizer steps (micro_batch_size=%d, max_micro_batch_tokens=%d)",
                            epoch_index + 1,
                            self.config.epochs_per_window,
                            total_micro_batches,
                            expected_optimizer_steps,
                            self.config.micro_batch_size,
                            self.config.max_micro_batch_tokens,
                        )
                    else:
                        log.info(
                            "starting train epoch %d/%d with %d micro-batches and %d expected optimizer steps",
                            epoch_index + 1,
                            self.config.epochs_per_window,
                            total_micro_batches,
                            expected_optimizer_steps,
                        )
                    with mem.region(
                        "zero_grad",
                        self._optimizer_step_count,
                        extra={
                            "phase": "epoch_start",
                            "iteration": iteration_index,
                            "epoch": epoch_index + 1,
                        },
                    ):
                        self.optimizer.zero_grad(set_to_none=True)
                    cycle_specs = (
                        fixed_rollout_cycles
                        if rollout_minibatching
                        else [
                            (cycle, cycle_loss_denominator)
                            for cycle in _balanced_micro_batch_cycles(
                                micro_batches, cycle_size=cycle_size
                            )
                        ]
                    )
                    for cycle_index, (cycle, denominator) in enumerate(
                        cycle_specs, start=1
                    ):
                        cycle_started_at = time.perf_counter()
                        self._execute_train_cycle(
                            cycle,
                            cycle_index=cycle_index,
                            epoch_label=(
                                f"epoch {epoch_index + 1}/"
                                f"{self.config.epochs_per_window}"
                            ),
                            epoch_number=epoch_index + 1,
                            iteration_index=iteration_index,
                            loss_denominator=denominator,
                            world_size=world_size,
                            mem=mem,
                            accumulators=cycle_accumulators,
                            expected_optimizer_steps_total=expected_optimizer_steps_total,
                            check_health=check_health,
                        )
                        primary_cycle_seconds.append(
                            self._synchronize_cycle_seconds(
                                time.perf_counter() - cycle_started_at
                            )
                        )
                        optimizer_cycle_seconds += primary_cycle_seconds[-1]
                        global_step = cycle_accumulators["global_step"]
                primary_optimization_seconds = (
                    time.perf_counter() - optimization_started_at
                )
                post_update_clip_fraction = (
                    self._post_update_clip_fraction(trust_samples, trust_before)
                    if trust_region_snapshot is not None
                    else None
                )
                window_update_rejected = self._apply_window_trust_region(
                    trust_region_snapshot,
                    post_update_clip_fraction=post_update_clip_fraction,
                    iteration_index=iteration_index,
                )
                trust_region_snapshot = None
                extra_cycles_enabled = (
                    bool(getattr(self.config, "extra_train_cycles_until_window", False))
                    and not window_update_rejected
                )
                full_eligible_coverage = (
                    sample_budget_fraction >= 1.0
                    and subsampled_sample_count == 0
                    and len(samples) == eligible_sample_count
                )
                if window_update_rejected:
                    extra_stop_reason = "window_update_rejected"
                elif extra_cycles_enabled and not full_eligible_coverage:
                    extra_stop_reason = "not_full_coverage"
                elif extra_cycles_enabled and next_window_ready is None:
                    extra_stop_reason = "window_signal_unavailable"
                elif (
                    extra_cycles_enabled
                    and int(cycle_accumulators["global_step"]) > 0
                    and next_window_ready is not None
                ):
                    extra_started_at = time.perf_counter()
                    post_primary_snapshot = None
                    post_primary_accumulators = None
                    max_extra_clip_fraction = float(
                        getattr(self.config, "extra_cycle_max_clip_fraction", 0.10)
                    )
                    extra_micro_batches = list(self._batch_iter(samples))
                    extra_cycle_list = _balanced_micro_batch_cycles(
                        extra_micro_batches, cycle_size=cycle_size
                    )
                    if not extra_cycle_list or not primary_cycle_seconds:
                        extra_stop_reason = "cycle_time_unavailable"
                    else:
                        estimated_cycle_seconds = max(primary_cycle_seconds)
                        extra_stop_reason = ""
                        extra_pass_index = 1
                        while not extra_stop_reason:
                            extra_cycle_list = self._extra_pass_cycles(
                                samples, cycle_size=cycle_size
                            )
                            for extra_index, cycle in enumerate(
                                extra_cycle_list, start=1
                            ):
                                extra_stop_reason = _extra_cycle_block_reason(
                                    next_window_ready=next_window_ready,
                                    next_window_seconds_remaining=(
                                        next_window_seconds_remaining
                                    ),
                                    estimated_cycle_seconds=estimated_cycle_seconds,
                                )
                                if extra_stop_reason:
                                    break
                                if (
                                    self._window_clip_fraction_bound() > 0.0
                                    and post_primary_snapshot is None
                                ):
                                    # Retain the accepted primary update if the
                                    # complete extra phase exceeds the same
                                    # window-start bound. Snapshot lazily: a
                                    # ready next window needs no extra copy.
                                    post_primary_snapshot = (
                                        self._snapshot_update_state()
                                    )
                                    post_primary_accumulators = dict(cycle_accumulators)
                                    # The CPU snapshot can consume rollout
                                    # slack; recheck before starting a cycle.
                                    extra_stop_reason = _extra_cycle_block_reason(
                                        next_window_ready=next_window_ready,
                                        next_window_seconds_remaining=next_window_seconds_remaining,
                                        estimated_cycle_seconds=estimated_cycle_seconds,
                                    )
                                    if extra_stop_reason:
                                        break
                                cycle_started_at = time.perf_counter()
                                cycle_clip_fraction = self._execute_train_cycle(
                                    cycle,
                                    cycle_index=extra_index,
                                    epoch_label=f"extra pass {extra_pass_index}",
                                    epoch_number=(
                                        int(self.config.epochs_per_window)
                                        + extra_pass_index
                                    ),
                                    iteration_index=iteration_index,
                                    loss_denominator=cycle_loss_denominator,
                                    world_size=world_size,
                                    mem=mem,
                                    accumulators=cycle_accumulators,
                                    max_clip_fraction=max_extra_clip_fraction,
                                    expected_optimizer_steps_total=max(
                                        expected_optimizer_steps_total,
                                        int(cycle_accumulators["global_step"])
                                        + len(extra_cycle_list),
                                    ),
                                    check_health=check_health,
                                )
                                actual_cycle_seconds = self._synchronize_cycle_seconds(
                                    time.perf_counter() - cycle_started_at
                                )
                                optimizer_cycle_seconds += actual_cycle_seconds
                                estimated_cycle_seconds = max(
                                    estimated_cycle_seconds, actual_cycle_seconds
                                )
                                if cycle_clip_fraction > max_extra_clip_fraction:
                                    extra_stop_reason = "clip_guard"
                                    log.info(
                                        "extra-cycle clip guard tripped: cycle clip_fraction %.4f > %.4f",
                                        cycle_clip_fraction,
                                        max_extra_clip_fraction,
                                    )
                                    break
                                extra_cycles_run += 1
                                extra_optimizer_steps += 1
                            extra_pass_index += 1
                    if post_primary_snapshot is not None and extra_optimizer_steps:
                        final_clip_fraction = self._post_update_clip_fraction(
                            trust_samples, trust_before
                        )
                        if (
                            final_clip_fraction is None
                            or not math.isfinite(final_clip_fraction)
                            or not 0.0 <= final_clip_fraction <= 1.0
                        ):
                            raise RuntimeError(
                                "enabled window trust region requires a valid extra-phase probe"
                            )
                        if final_clip_fraction > self._window_clip_fraction_bound():
                            log.warning(
                                "window trust region: rejecting %d extra optimizer steps "
                                "at iteration %d (final clip fraction %.6f > %.6f); "
                                "restoring accepted post-primary weights and optimizer state",
                                extra_optimizer_steps,
                                iteration_index,
                                final_clip_fraction,
                                self._window_clip_fraction_bound(),
                            )
                            self._restore_update_state(post_primary_snapshot)
                            cycle_accumulators.clear()
                            cycle_accumulators.update(post_primary_accumulators)
                            extra_cycles_run = extra_optimizer_steps = 0
                            extra_stop_reason = "window_clip_guard"
                        else:
                            post_update_clip_fraction = final_clip_fraction
                    post_primary_snapshot = None
                    post_primary_accumulators = None
                    if not extra_stop_reason:
                        extra_stop_reason = "cycle_time_unavailable"
                    extra_phase_seconds = time.perf_counter() - extra_started_at
                if extra_cycles_enabled:
                    if extra_cycles_run:
                        log.info(
                            "opportunistic extra train cycles: %d (+%d optimizer steps in %.1fs, stop=%s)",
                            extra_cycles_run,
                            extra_optimizer_steps,
                            extra_phase_seconds,
                            extra_stop_reason,
                        )
                    else:
                        log.info(
                            "opportunistic extra train cycles skipped (stop=%s, coverage=%d/%d, sample_budget_fraction=%.4f)",
                            extra_stop_reason or "no_optimizer_steps",
                            len(samples),
                            eligible_sample_count,
                            sample_budget_fraction,
                        )
        except torch.OutOfMemoryError as exc:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "Trainer CUDA OOM during PPO optimization on "
                f"{self.device}. Current config: micro_batch_size={self.config.micro_batch_size}, "
                f"gradient_accumulation_steps={self.config.gradient_accumulation_steps}, "
                f"max_sequence_tokens={self.config.max_sequence_tokens}, "
                f"max_micro_batch_tokens={self.config.max_micro_batch_tokens}. "
                "Observed sample lengths: "
                f"input_tokens min/p50/p95/max="
                f"{sample_length_summary.min_input_tokens}/{sample_length_summary.median_input_tokens}/"
                f"{sample_length_summary.p95_input_tokens}/{sample_length_summary.max_input_tokens}, "
                f"completion_tokens min/p50/p95/max="
                f"{sample_length_summary.min_completion_tokens}/{sample_length_summary.median_completion_tokens}/"
                f"{sample_length_summary.p95_completion_tokens}/{sample_length_summary.max_completion_tokens}. "
                "Reduce training.micro_batch_size or training.max_sequence_tokens if this persists."
            ) from exc

        mean_ratio_sum = float(cycle_accumulators["mean_ratio_sum"])
        clip_fraction_sum = float(cycle_accumulators["clip_fraction_sum"])
        previous_policy_kl_sum = float(
            cycle_accumulators.get("previous_policy_kl_sum", 0.0)
        )
        reference_kl_sum = float(cycle_accumulators.get("reference_kl_sum", 0.0))
        entropy_sum = float(cycle_accumulators["entropy_sum"])
        entropy_loss_sum = float(cycle_accumulators["entropy_loss_sum"])
        ratio_updates = int(cycle_accumulators["ratio_updates"])
        grad_norm_sum = float(cycle_accumulators["grad_norm_sum"])
        max_grad_norm = float(cycle_accumulators["max_grad_norm"])
        grad_norm_updates = int(cycle_accumulators["grad_norm_updates"])
        final_loss = float(cycle_accumulators["final_loss"])
        global_step = int(cycle_accumulators["global_step"])
        optimization_seconds = time.perf_counter() - optimization_started_at
        if primary_optimization_seconds is None:
            primary_optimization_seconds = optimization_seconds
        cuda_optimization_peak_delta_gb = max(
            0.0,
            self._cuda_max_memory_allocated_gb() - cuda_optimization_start_allocated_gb,
        )
        cuda_optimization_end_allocated_gb = self._cuda_memory_allocated_gb()
        cuda_optimization_persistent_delta_gb = (
            cuda_optimization_end_allocated_gb - cuda_optimization_start_allocated_gb
        )
        # End-of-window probe entropy: the frozen batch through the current
        # policy (no-grad, rank 0 only; 0.0 unless entropy control is on).
        probe_entropy = self._compute_probe_entropy()
        with mem.region(
            "checkpoint_save",
            self._optimizer_step_count,
            extra={
                "phase": "checkpoint_save",
                "iteration": iteration_index,
            },
        ):
            if check_health is not None:
                check_health()
            window_artifacts = self._finalize_window_artifacts(
                output_dir, iteration_index, check_health=check_health
            )
            adapter_dir = window_artifacts.adapter_dir
            runtime_adapter_dir = window_artifacts.publication_path
            trainer_state_path = window_artifacts.checkpoint_path
            policy_export_seconds = window_artifacts.policy_export_seconds
        total_train_window_seconds = time.perf_counter() - train_window_started_at
        trainable_param_norm_after = self._trainable_parameter_l2_norm()
        log.info(
            "saved latest adapter to %s and runtime adapter to %s (sample_build=%.2fs optimize=%.2fs policy_export=%.2fs total=%.2fs)",
            adapter_dir,
            runtime_adapter_dir,
            sample_build_seconds,
            optimization_seconds,
            policy_export_seconds,
            total_train_window_seconds,
        )
        if self.device.type == "cuda":
            log.info(
                "trainer CUDA memory: model_allocated=%.3fGiB optimization_delta=%.3fGiB optimization_persistent_delta=%+.3fGiB",
                self._cuda_trainer_model_memory_allocated_gb(),
                cuda_optimization_peak_delta_gb,
                cuda_optimization_persistent_delta_gb,
            )
        if expected_optimizer_steps_total > 0:
            log.info(
                "optimizer coverage: %d/%d steps (%.1f%%)",
                global_step,
                expected_optimizer_steps_total,
                100.0 * global_step / expected_optimizer_steps_total,
            )
        if self.config.policy_loss == "score_centered_tis":
            log.info(
                "score centering: %s",
                summarize_policy_update_statistics(
                    cycle_accumulators["policy_update_statistics"]
                ),
            )
        advantage_summary = _sample_advantage_summary(samples)
        mean_return_to_go = sum(sample.return_to_go for sample in samples) / len(
            samples
        )
        completion_token_count = max(
            1, sum(sum(sample.completion_mask[1:]) for sample in samples)
        )
        mean_old_logprob = (
            sum(sum(getattr(sample, "old_token_logprobs", [])) for sample in samples)
            / completion_token_count
        )
        transition_reward_nonzero_fraction = sum(
            1.0 for sample in samples if abs(sample.reward) > 1e-8
        ) / len(samples)
        transition_reward_positive_fraction = sum(
            1.0 for sample in samples if sample.reward > 0.0
        ) / len(samples)
        transition_reward_sum = sum(float(sample.reward) for sample in samples)
        return_to_go_nonzero_fraction = _sample_fraction(
            samples, lambda sample: abs(float(sample.return_to_go)) > 1e-8
        )
        return_to_go_positive_fraction = _sample_fraction(
            samples, lambda sample: float(sample.return_to_go) > 0.0
        )
        kept_positive_transition_reward_sample_count = _sample_count(
            samples, lambda sample: float(sample.reward) > 0.0
        )
        return WindowTrainResult(
            sample_count=len(samples),
            raw_sample_count=len(raw_samples),
            eligible_sample_count=eligible_sample_count,
            filtered_zero_signal_sample_count=skipped_zero_signal_samples,
            subsampled_sample_count=subsampled_sample_count,
            sample_budget_fraction=sample_budget_fraction,
            sample_budget_step_multiple=sample_budget_step_multiple,
            trainable_parameter_count=self._trainable_parameter_count,
            mean_advantage=advantage_summary.mean,
            std_advantage=advantage_summary.std,
            mean_return_to_go=mean_return_to_go,
            mean_advantage_by_level=advantage_summary.mean_by_level,
            mean_old_logprob=mean_old_logprob,
            final_loss=final_loss,
            mean_entropy=(entropy_sum / ratio_updates) if ratio_updates > 0 else 0.0,
            entropy_loss=(entropy_loss_sum / ratio_updates)
            if ratio_updates > 0
            else 0.0,
            entropy_coef_used=self.live_entropy_coef,
            final_learning_rate=self._current_learning_rate(),
            optimizer_steps=global_step,
            mean_ratio=(mean_ratio_sum / ratio_updates) if ratio_updates > 0 else 1.0,
            mean_clip_fraction=(
                cycle_accumulators["policy_update_statistics"].get(
                    "score_centering/capped_sum", 0.0
                )
                / max(
                    1.0,
                    cycle_accumulators["policy_update_statistics"].get(
                        "score_centering/tokens", 0.0
                    ),
                )
                if self.config.policy_loss == "score_centered_tis"
                else (clip_fraction_sum / ratio_updates)
                if ratio_updates > 0
                else 0.0
            ),
            mean_previous_policy_kl=(previous_policy_kl_sum / ratio_updates)
            if ratio_updates > 0
            else 0.0,
            mean_reference_kl=(reference_kl_sum / ratio_updates)
            if ratio_updates > 0
            else 0.0,
            window_update_rejected=window_update_rejected,
            trust_region_lr_scale=float(getattr(self, "_trust_region_lr_scale", 1.0)),
            post_update_clip_fraction=post_update_clip_fraction or 0.0,
            mean_grad_norm=(grad_norm_sum / grad_norm_updates)
            if grad_norm_updates > 0
            else 0.0,
            max_grad_norm=max_grad_norm,
            transition_reward_nonzero_fraction_raw=raw_transition_reward_nonzero_fraction,
            transition_reward_positive_fraction_raw=raw_transition_reward_positive_fraction,
            transition_reward_sum_raw=raw_transition_reward_sum,
            transition_reward_nonzero_fraction_kept=transition_reward_nonzero_fraction,
            transition_reward_positive_fraction_kept=transition_reward_positive_fraction,
            transition_reward_sum_kept=transition_reward_sum,
            return_to_go_nonzero_fraction_kept=return_to_go_nonzero_fraction,
            return_to_go_positive_fraction_kept=return_to_go_positive_fraction,
            advantage_nonzero_fraction_kept=advantage_summary.nonzero_fraction,
            advantage_positive_fraction_kept=advantage_summary.positive_fraction,
            positive_transition_reward_sample_count_raw=raw_positive_transition_reward_sample_count,
            positive_transition_reward_sample_count_kept=kept_positive_transition_reward_sample_count,
            trainable_param_norm_before=trainable_param_norm_before,
            trainable_param_norm_after=trainable_param_norm_after,
            advantage_group_mean_reward=advantage_stats.mean_reward,
            advantage_group_std_reward=advantage_stats.std_reward,
            total_completion_tokens=token_usage.total_completion_tokens,
            total_env_steps=token_usage.total_env_steps,
            avg_actions_per_completed_level=_avg_actions_per_completed_level(rollouts),
            avg_tokens_per_rollout=token_usage.avg_tokens_per_rollout,
            avg_tokens_per_step=token_usage.avg_tokens_per_step,
            avg_tokens_per_model_turn=token_usage.avg_tokens_per_model_turn,
            avg_prompt_tokens_per_sample=sample_length_summary.avg_prompt_tokens,
            avg_history_messages_per_sample=avg_history_messages_per_sample,
            sample_build_seconds=sample_build_seconds,
            optimizer_cycle_seconds=optimizer_cycle_seconds,
            processed_tokens=int(cycle_accumulators.get("processed_tokens", 0)),
            forward_time_s=float(cycle_accumulators.get("forward_time_s", 0.0)),
            backward_time_s=float(cycle_accumulators.get("backward_time_s", 0.0)),
            optimization_seconds=optimization_seconds,
            policy_export_seconds=policy_export_seconds,
            total_train_window_seconds=total_train_window_seconds,
            adapter_dir=adapter_dir,
            runtime_adapter_dir=runtime_adapter_dir,
            trainer_state_path=trainer_state_path,
            cuda_trainer_model_allocated_gb=self._cuda_trainer_model_memory_allocated_gb(),
            cuda_optimization_peak_delta_gb=cuda_optimization_peak_delta_gb,
            cuda_optimization_persistent_delta_gb=cuda_optimization_persistent_delta_gb,
            **self._parity_result_fields(rollout_logprob_parity),
            baseline_feature_seconds=self._baseline_feature_seconds,
            baseline_loss=baseline_update.loss,
            baseline_update_steps=baseline_update.update_steps,
            baseline_mean_value=baseline_signal_batch.mean_value,
            baseline_mean_target=baseline_signal_batch.mean_target,
            baseline_explained_variance=(baseline_signal_batch.explained_variance),
            baseline_post_update_mean_value=(baseline_update.mean_prediction),
            baseline_post_update_mean_absolute_error=(
                baseline_update.mean_absolute_error
            ),
            baseline_post_update_explained_variance=(
                baseline_update.explained_variance
            ),
            baseline_mean_raw_advantage=baseline_signal_batch.mean_raw_advantage,
            baseline_std_raw_advantage=baseline_signal_batch.std_raw_advantage,
            baseline_update_seconds=baseline_update_seconds,
            baseline_metrics=(
                dict(baseline_signal_batch.diagnostics)
                if not isinstance(baseline_signal_batch, StateCriticSignalBatch)
                else {}
            ),
            critic_metrics=(
                dict(baseline_signal_batch.diagnostics)
                if isinstance(baseline_signal_batch, StateCriticSignalBatch)
                else {}
            ),
            optimizer_skipped_reason=optimizer_skipped_reason,
            extra_cycles=extra_cycles_run,
            extra_optimizer_steps=extra_optimizer_steps,
            extra_phase_seconds=extra_phase_seconds,
            extra_stop_reason=extra_stop_reason,
            budget_train_seconds=max(
                0.0, total_train_window_seconds - extra_phase_seconds
            ),
            zero_signal_denominator_scale=zero_signal_denominator_scale,
            packed_episodes=packed_episodes,
            packed_sequences=packed_sequences,
            segment_boundaries=segment_boundaries,
            probe_entropy=probe_entropy,
            entropy_to_policy_grad_ratio=entropy_to_policy_grad_ratio,
            policy_update_metrics=summarize_policy_update_statistics(
                cycle_accumulators["policy_update_statistics"]
            ),
        )
