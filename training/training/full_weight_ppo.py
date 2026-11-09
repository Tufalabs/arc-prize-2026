from __future__ import annotations

import copy
import logging
import math
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

from training.dataset import TokenizedSample
from training.ddp_worker import length_balanced_split
from training.ppo import (
    PPOTrainer,
    WindowArtifacts,
    _disable_dropout_modules,
    _micro_batch_padded_tokens,
)


log = logging.getLogger(__name__)
_TensorSignature = tuple[tuple[str, tuple[int, ...], str], ...]
_T = TypeVar("_T")
# The statistics PPOTrainer._check_rollout_logprob_parity decides on.
_PARITY_EVIDENCE = (
    "nonfinite_count",
    "token_count",
    "mean_abs",
    "p95_abs",
    "max_abs",
    "outlier_count",
    "inlier_mean_abs",
    "signed_mean",
    "biased_token_count",
    "biased_mean_abs",
)


def padded_rank_shard(
    items: Sequence[_T], *, rank: int, world_size: int
) -> tuple[list[_T], list[_T]]:
    """This rank's share ``items[rank::world_size]`` and its forward padding.

    Every rank runs ``ceil(len(items) / world_size)`` forwards so FSDP
    collectives stay aligned. Padding repeats the rank's first item, or
    ``items[0]`` when the share is empty; callers discard its results.
    """
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside a world of size {world_size}")
    share = list(items[rank::world_size])
    if not items:
        return share, []
    forwards = -(-len(items) // world_size)
    filler = share[0] if share else items[0]
    return share, [filler] * (forwards - len(share))


def _fsdp_cycle_plan(
    cycle: list[list[TokenizedSample]], world_size: int
) -> tuple[list[list[list[TokenizedSample]]], list[TokenizedSample]]:
    """Each rank's micro-batches for one FSDP cycle, and the padding micro-batch.

    Slot s runs on every rank in lockstep and costs the slowest rank's
    micro-batch, so each slot gets micro-batches of similar padded length;
    slot 0 keeps the largest (the OOM probe). A rank one slot short runs the
    cycle's smallest micro-batch at zero backward weight: the cheapest work
    that keeps its FSDP collectives in step."""
    shares = length_balanced_split(cycle, world_size, size=_micro_batch_padded_tokens)
    return shares, min(cycle, key=_micro_batch_padded_tokens)


def _slot_imbalance(shares: list[list[list[TokenizedSample]]]) -> float:
    """Sum over slots of (max - mean) padded tokens, over the sum of slot maxima.

    A rank without a real micro-batch in a slot counts zero tokens, so this is
    the fraction of lockstep time spent waiting or padding, by padded tokens."""
    longest = lost = 0.0
    for slot in range(max(len(share) for share in shares)):
        sizes = [
            _micro_batch_padded_tokens(share[slot]) if slot < len(share) else 0
            for share in shares
        ]
        longest += max(sizes)
        lost += max(sizes) - sum(sizes) / len(shares)
    return lost / longest if longest else 0.0


class DistributedCycleMixin:
    """Collective PPO cycles and diagnostics shared by FSDP policy trainers."""

    def _reshard_policy(self) -> None:
        from torch.distributed.fsdp import FSDPModule

        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

    def _compute_probe_entropy(self) -> float:
        """Fail-stop rank-local FSDP probe failures instead of desynchronizing."""
        probe = getattr(self, "_probe_samples", None)
        if not probe:
            return 0.0
        return self._compute_probe_entropy_forward(probe)

    def _diagnostic_shard(self, items: Sequence[_T]) -> tuple[list[_T], list[_T]]:
        # Diagnostic forwards also run in single-process unit tests.
        if dist.is_available() and dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = 0, 1
        return padded_rank_shard(items, rank=rank, world_size=world_size)

    def _probe_entropy_sums(
        self, chunks: Sequence[list[TokenizedSample]]
    ) -> tuple[float, int]:
        """Pool span-entropy sums of each rank's share of the probe forwards.

        Every rank received the same chunk list (the global Python RNG that
        shuffles micro-batches must advance identically everywhere), scores
        its round-robin share, and returns the world's totals.
        """
        share, padding = self._diagnostic_shard(chunks)
        entropy_sum, span_total = super()._probe_entropy_sums(share)
        super()._probe_entropy_sums(padding)
        totals = torch.tensor(
            [entropy_sum, float(span_total)], dtype=torch.float64, device=self.device
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        pooled_sum, pooled_spans = totals.tolist()
        return pooled_sum, int(pooled_spans)

    def _ensure_probe_states(
        self, samples: list[TokenizedSample], output_dir: Path
    ) -> None:
        """Let rank zero touch the probe file, then share its frozen samples.

        The probe never changes after its first capture or load, so once every
        rank holds it a window costs one all_reduce, not a broadcast of the
        samples. A rank-zero file error reaches every rank as an exception
        instead of leaving the peers in the broadcast.
        """
        if not dist.is_available() or not dist.is_initialized():
            super()._ensure_probe_states(samples, output_dir)
            return

        locally_ready = torch.tensor(
            1 if getattr(self, "_probe_samples", None) else 0,
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(locally_ready, op=dist.ReduceOp.MIN)
        if int(locally_ready.item()) == 1:
            return

        result: list[dict[str, Any] | None] = [None]
        if dist.get_rank() == 0:
            try:
                super()._ensure_probe_states(samples, output_dir)
                result[0] = {
                    "probe_samples": getattr(self, "_probe_samples", None),
                }
            except Exception as exc:  # noqa: BLE001 - relay rank-zero file failures
                result[0] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(result, src=0)
        payload = result[0]
        if payload is None:
            raise RuntimeError("Rank zero did not publish probe state setup.")
        error = payload.get("error")
        if error is not None:
            raise RuntimeError(f"Rank-zero probe state setup failed: {error}")
        self._probe_samples = payload.get("probe_samples")

        # The later entropy forward intentionally remains all-rank: FSDP
        # forwards contain collectives even though only rank zero owns files.

    def _pool_parity_evidence(
        self, evidence: dict[str, float | int]
    ) -> dict[str, float | int]:
        """Pool every rank's parity statistics into one verdict.

        All ranks score the same targets, each with its own forward, and each
        trains on its own scores, so the gate holds only if it holds on every
        rank. Each rank keeps its scores; the pooled statistics are the worst
        rank's (largest mean, p95, maximum, outlier count and inlier mean, the
        signed mean of the rank with the largest mean), non-finite counts add
        up, and every rank takes the same expand, pass or fail branch.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return evidence
        local = torch.tensor(
            [float(evidence[name]) for name in _PARITY_EVIDENCE],
            dtype=torch.float64,
            device=self.device,
        )
        rows = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(rows, local)
        table = dict(zip(_PARITY_EVIDENCE, torch.stack(rows).cpu().T, strict=True))
        nonfinite = int(table["nonfinite_count"].sum().item())
        token_counts = [int(value) for value in table["token_count"].tolist()]
        if not nonfinite and len(set(token_counts)) != 1:
            raise RuntimeError(
                "Same-policy rollout logprob parity scored different target "
                f"counts across ranks: {token_counts}."
            )
        worst = int(table["mean_abs"].argmax().item())
        if dist.get_rank() == 0 and max(token_counts) > 0:
            log.info(
                "rollout logprob parity by rank: mean_abs=%s p95_abs=%s "
                "max_abs=%s outliers=%s nonfinite=%s",
                *(
                    [round(value, 6) for value in table[name].tolist()]
                    for name in ("mean_abs", "p95_abs", "max_abs")
                ),
                [int(value) for value in table["outlier_count"].tolist()],
                [int(value) for value in table["nonfinite_count"].tolist()],
            )
        return {
            "nonfinite_count": nonfinite,
            "token_count": max(token_counts),
            **{
                name: float(table[name].max().item())
                for name in ("mean_abs", "p95_abs", "max_abs", "inlier_mean_abs", "biased_mean_abs")
            },
            "outlier_count": int(table["outlier_count"].max().item()),
            "signed_mean": float(table["signed_mean"][worst].item()),
            "biased_token_count": int(table["biased_token_count"].max().item()),
        }

    def _trust_region_token_logprobs(
        self, samples: list[TokenizedSample]
    ) -> torch.Tensor:
        """Score this rank's share of the trust-region probe.

        Before and after the update each rank scores the same share, so the
        clip verdict pools token counts in ``_post_update_clip_fraction``.
        """
        share, padding = self._diagnostic_shard(samples)
        scores = super()._trust_region_token_logprobs(share)
        super()._trust_region_token_logprobs(padding)
        return scores

    def _trust_region_token_count(self, scores: torch.Tensor) -> int:
        """Probe tokens scored by every rank's share together."""
        count = torch.tensor(
            float(scores.numel()), dtype=torch.float64, device=self.device
        )
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return int(count.item())

    def _post_update_clip_fraction(
        self, samples: list[TokenizedSample], before: torch.Tensor
    ) -> float | None:
        """Token-weighted post-update clip fraction over every rank's share.

        Each rank rescores its share of the probe; the window verdict must be
        the same everywhere, so clipped and scored token counts are pooled
        across the training world before the shared decision. A rank whose
        share scored tokens before the update but none after is missing.
        """
        clipped = self._trust_region_clip_mask(samples, before)
        lost = clipped.numel() == 0 and before.numel() > 0
        stats = torch.tensor(
            [float(clipped.sum().item()), float(clipped.numel()), float(lost)],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        clipped_tokens, total, missing = stats.tolist()
        if missing not in (0.0, float(self._training_world_size())):
            raise RuntimeError("FSDP trust-region probe differs across ranks.")
        if missing or total <= 0.0:
            if self._window_clip_fraction_bound() > 0.0:
                raise RuntimeError(
                    "enabled window trust region requires sampled probe tokens"
                )
            return None
        result = clipped_tokens / total
        log.info(
            "window trust region post-update clip fraction=%.6f tokens=%d",
            result,
            int(total),
        )
        return result

    def _apply_window_trust_region(
        self,
        snapshot: dict[str, Any] | None,
        *,
        post_update_clip_fraction: float | None,
        iteration_index: int,
    ) -> bool:
        fraction = (
            -1.0
            if post_update_clip_fraction is None
            else float(post_update_clip_fraction)
        )
        state = torch.tensor(
            [float(snapshot is not None), fraction],
            dtype=torch.float64,
            device=self.device,
        )
        highest = state.clone()
        lowest = state.clone()
        dist.all_reduce(highest, op=dist.ReduceOp.MAX)
        dist.all_reduce(lowest, op=dist.ReduceOp.MIN)
        if not torch.equal(highest, lowest):
            raise RuntimeError("FSDP trust-region decision differs across ranks.")
        return super()._apply_window_trust_region(
            snapshot,
            post_update_clip_fraction=post_update_clip_fraction,
            iteration_index=iteration_index,
        )

    def _synchronize_cycle_seconds(self, seconds: float) -> float:
        # Extra-cycle admission must use one duration on every FSDP rank.
        duration = torch.tensor(seconds, dtype=torch.float64, device=self.device)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        return float(duration.item())

    def _entropy_to_policy_grad_ratio(self, chunk: list[TokenizedSample]) -> float:
        # autograd.grad over composable FSDP parameters does not participate in
        # the normal synchronized backward protocol. Keep the training path
        # correct and leave this optional diagnostic disabled for full weights.
        del chunk
        return 0.0

    def _reduce_cycle_metrics(
        self, local_metrics: dict[str, float]
    ) -> dict[str, float]:
        names = (
            "mean_ratio_sum",
            "clip_fraction_sum",
            "entropy_sum",
            "entropy_loss_sum",
            "previous_policy_kl_sum",
            "reference_kl_sum",
            "loss_sum",
            "updates",
            "processed_tokens",
        )
        packed = torch.tensor(
            [local_metrics[name] for name in names],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        reduced = {name: float(value) for name, value in zip(names, packed.tolist())}
        timing_names = ("forward_time_s", "backward_time_s")
        timings = torch.tensor(
            [local_metrics[name] for name in timing_names],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(timings, op=dist.ReduceOp.MAX)
        reduced.update(zip(timing_names, timings.tolist()))
        return reduced

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
        accumulators: dict[str, float | int],
        expected_optimizer_steps_total: int,
        check_health: Callable[[], None] | None = None,
        max_clip_fraction: float | None = None,
    ) -> float:
        actual_world_size = self._training_world_size()
        if world_size != actual_world_size:
            raise RuntimeError(
                f"PPO cycle world_size={world_size} does not match distributed "
                f"world size {actual_world_size}."
            )
        if not cycle:
            raise ValueError("Full-weight PPO received an empty optimizer cycle.")

        # Cycle membership is unchanged: only which rank runs which
        # micro-batch, so the update differs only in summation order.
        shares, padding_chunk = _fsdp_cycle_plan(cycle, world_size)
        local_share = shares[dist.get_rank()]
        slot_count = max(len(share) for share in shares)
        if dist.get_rank() == 0 and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "cycle %d (%s): %d micro-batches in %d slots, %.1f%% of slot "
                "time waiting or padding (by padded tokens)",
                cycle_index,
                epoch_label,
                len(cycle),
                slot_count,
                100.0 * _slot_imbalance(shares),
            )
        profiler_step = self._optimizer_step_count + 1
        local_metrics = {
            "mean_ratio_sum": 0.0,
            "clip_fraction_sum": 0.0,
            "entropy_sum": 0.0,
            "entropy_loss_sum": 0.0,
            "previous_policy_kl_sum": 0.0,
            "reference_kl_sum": 0.0,
            "loss_sum": 0.0,
            "updates": 0.0,
            "processed_tokens": 0.0,
            "forward_time_s": 0.0,
            "backward_time_s": 0.0,
        }

        for slot_index in range(slot_count):
            if check_health is not None:
                check_health()
            is_final_slot = slot_index + 1 == slot_count
            self._set_fsdp_gradient_sync(is_final_slot=is_final_slot)
            is_real = slot_index < len(local_share)
            chunk = local_share[slot_index] if is_real else padding_chunk
            metrics = self._micro_batch_forward_backward(
                chunk,
                loss_denominator=loss_denominator,
                backward_scale=float(world_size) if is_real else 0.0,
                profiler_step=profiler_step,
                iteration_index=iteration_index,
                epoch_index=epoch_number,
            )
            # Padding slots execute real forward/backward work too.
            for key in ("processed_tokens", "forward_time_s", "backward_time_s"):
                local_metrics[key] += metrics.get(key, 0)
            if not is_real:
                continue
            local_metrics["mean_ratio_sum"] += metrics["mean_ratio"]
            local_metrics["clip_fraction_sum"] += metrics["clip_fraction"]
            local_metrics["entropy_sum"] += metrics["entropy"]
            local_metrics["entropy_loss_sum"] += metrics["entropy_loss"]
            local_metrics["previous_policy_kl_sum"] += metrics["previous_policy_kl"]
            local_metrics["reference_kl_sum"] += metrics["reference_kl"]
            local_metrics["loss_sum"] += metrics["loss"]
            local_metrics["updates"] += 1.0

        self._synchronize_replicated_gradients()
        reduced = self._reduce_cycle_metrics(local_metrics)
        for key in ("processed_tokens", "forward_time_s", "backward_time_s"):
            accumulators[key] = accumulators.get(key, 0) + reduced[key]
        updates = int(reduced["updates"])
        if updates <= 0:
            raise RuntimeError("Full-weight PPO cycle contained no real micro-batches.")
        cycle_clip_fraction = reduced["clip_fraction_sum"] / updates
        if max_clip_fraction is not None and cycle_clip_fraction > max_clip_fraction:
            # The pooled forward-phase clip fraction is identical on every
            # rank, so all ranks skip this extra cycle's optimizer step together.
            self.optimizer.zero_grad(set_to_none=True)
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
            grad_norm = self._global_grad_norm_and_clip(self.config.max_grad_norm)
            if check_health is not None:
                check_health()
            self.optimizer.step()
            if self.device.type == "cuda":
                # Metrics/gradient norms already synchronized backward. Include
                # the remaining optimizer kernels in the cycle wall time.
                torch.cuda.synchronize(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        self._optimizer_step_count += 1

        accumulators["mean_ratio_sum"] += reduced["mean_ratio_sum"]
        accumulators["clip_fraction_sum"] += reduced["clip_fraction_sum"]
        accumulators["entropy_sum"] += reduced["entropy_sum"]
        accumulators["entropy_loss_sum"] += reduced["entropy_loss_sum"]
        accumulators["previous_policy_kl_sum"] += reduced["previous_policy_kl_sum"]
        accumulators["reference_kl_sum"] += reduced["reference_kl_sum"]
        accumulators["ratio_updates"] = int(accumulators["ratio_updates"]) + updates
        accumulators["final_loss"] = reduced["loss_sum"] / updates
        accumulators["grad_norm_sum"] += grad_norm
        accumulators["max_grad_norm"] = max(
            float(accumulators["max_grad_norm"]), grad_norm
        )
        accumulators["grad_norm_updates"] = int(accumulators["grad_norm_updates"]) + 1
        accumulators["global_step"] = int(accumulators["global_step"]) + 1

        global_step = int(accumulators["global_step"])
        if dist.get_rank() == 0 and (
            global_step == 1
            or global_step == expected_optimizer_steps_total
            or (global_step % 10) == 0
        ):
            log.info(
                "optimizer progress: step %d/%d (%s, lr=%.3e, "
                "grad_norm=%.4f, last_loss=%.4f)",
                global_step,
                expected_optimizer_steps_total,
                epoch_label,
                learning_rate,
                grad_norm,
                accumulators["final_loss"],
            )
        return cycle_clip_fraction


class FullWeightPPOTrainer(DistributedCycleMixin, PPOTrainer):
    """FSDP2 policy trainer used by one torchrun process per training GPU.

    Every rank owns the same control flow. FSDP shards the language decoder,
    while the frozen vision tower and the comparatively small LM head stay
    replicated. Rollout orchestration, checkpointing, and policy publication
    are process-boundary concerns and intentionally live outside this class.
    """

    def _load_policy_model(
        self, *, resume_adapter_dir: Path | None = None
    ) -> torch.nn.Module:
        if resume_adapter_dir is not None:
            raise ValueError(
                "Full-weight training cannot resume from a LoRA adapter directory."
            )

        if (
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 10
        ):
            # Pinned FLA/TileLang crashes in GDN backward on B200, including
            # native Qwen head geometry. Its existing Triton backend passes the
            # multimodal FSDP update/rollback checks; Hopper requires TileLang.
            os.environ["FLA_TILELANG"] = "0"
            log.info(
                "selecting FLA Triton backend for Blackwell full-weight training "
                "to avoid the pinned TileLang GDN backward misaligned-address failure"
            )

        # Only rank zero reads the base checkpoint. Peers instantiate the same
        # architecture on meta; _place_policy_model shards that skeleton and
        # streams one parameter at a time from rank zero into its final layout.
        load_rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": torch.float32,
            "device_map": None,
            "low_cpu_mem_usage": True,
        }
        attn_implementation = self._resolved_attn_implementation()
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        log.info(
            "loading full-weight policy with attention backend %s on rank %d",
            attn_implementation,
            load_rank,
        )
        model = (
            self._load_base_model(model_kwargs)
            if load_rank == 0
            else self._load_empty_policy_model(attn_implementation=attn_implementation)
        )
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if self.config.gradient_checkpointing:
            self._enable_full_weight_activation_checkpointing(model)

        # Full-weight means every language-policy parameter is trainable. The
        # vision path remains a fixed feature extractor on every rank.
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        visual = self._visual_module(model)
        if visual is not None:
            for parameter in visual.parameters():
                parameter.requires_grad_(False)
        if self.config.disable_dropout:
            disabled_dropouts = _disable_dropout_modules(model)
            log.info(
                "disabled dropout by setting %d policy dropout modules to p=0.0",
                disabled_dropouts,
            )
        self._assert_fp32_trainable_parameters(model)
        return model

    def _load_empty_policy_model(
        self, *, attn_implementation: str | None
    ) -> torch.nn.Module:
        """Instantiate the configured policy architecture without reading weights."""
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            revision=self._model_revision(),
            trust_remote_code=True,
        )
        model_class = (
            AutoModelForImageTextToText
            if getattr(self.config, "multimodal", False)
            else AutoModelForCausalLM
        )
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        with torch.device("meta"):
            model = model_class.from_config(model_config, **kwargs)
        return model.to(dtype=torch.float32)

    @staticmethod
    def _assert_fp32_trainable_parameters(model: torch.nn.Module) -> None:
        non_fp32 = [
            f"{name}={parameter.dtype}"
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.dtype != torch.float32
        ]
        if non_fp32:
            preview = ", ".join(non_fp32[:4])
            raise RuntimeError(
                "Full-weight FSDP2 requires FP32 persistent trainable parameters; "
                f"found {preview}. BF16 is applied by MixedPrecisionPolicy at compute time."
            )

    def _enable_full_weight_activation_checkpointing(
        self, model: torch.nn.Module
    ) -> None:
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise RuntimeError(
                "gradient_checkpointing is enabled, but the loaded policy model "
                "does not expose gradient_checkpointing_enable()."
            )
        enable_checkpointing(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
                "preserve_rng_state": self._checkpoint_preserve_rng_state(),
            }
        )
        if getattr(model, "is_gradient_checkpointing", None) is not True:
            raise RuntimeError(
                "gradient_checkpointing_enable() returned without enabling model "
                "checkpointing."
            )

    @staticmethod
    def _visual_module(model: torch.nn.Module) -> torch.nn.Module | None:
        visual = getattr(model, "visual", None)
        if isinstance(visual, torch.nn.Module):
            return visual
        nested = getattr(model, "model", None)
        visual = getattr(nested, "visual", None)
        return visual if isinstance(visual, torch.nn.Module) else None

    @staticmethod
    def _language_layers(decoder: torch.nn.Module) -> torch.nn.ModuleList:
        language_decoder = getattr(decoder, "language_model", decoder)
        layers = getattr(language_decoder, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList) or not layers:
            raise RuntimeError(
                "Full-weight Qwen training requires decoder.layers or "
                "decoder.language_model.layers to be a non-empty ModuleList."
            )
        return layers

    @staticmethod
    def _parameters_share_storage(
        first: torch.nn.Parameter, second: torch.nn.Parameter
    ) -> bool:
        if first is second:
            return True
        if first.device.type == "meta" or second.device.type == "meta":
            return False
        try:
            return (
                first.untyped_storage().data_ptr()
                == second.untyped_storage().data_ptr()
            )
        except RuntimeError:
            return False

    def _assert_untied_lm_head(self, lm_head: torch.nn.Module) -> None:
        get_input_embeddings = getattr(self.model, "get_input_embeddings", None)
        input_embeddings = (
            get_input_embeddings() if callable(get_input_embeddings) else None
        )
        embedding_weight = getattr(input_embeddings, "weight", None)
        head_weight = getattr(lm_head, "weight", None)
        if not isinstance(embedding_weight, torch.nn.Parameter) or not isinstance(
            head_weight, torch.nn.Parameter
        ):
            raise RuntimeError(
                "Full-weight Qwen training requires input embeddings and lm_head "
                "to expose weight parameters."
            )
        if self._parameters_share_storage(embedding_weight, head_weight):
            raise RuntimeError(
                "Full-weight training cannot leave lm_head replicated when its "
                "weight is tied to the sharded input embedding. Load a checkpoint "
                "with untied word embeddings."
            )

    def _place_policy_model(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "FullWeightPPOTrainer must be constructed after torch.distributed "
                "initialization."
            )

        decoder: torch.nn.Module | None = None
        lm_head: torch.nn.Module | None = None
        layers = torch.nn.ModuleList()
        visual: torch.nn.Module | None = None
        local_preflight_error: str | None = None
        try:
            decoder, lm_head = self._decoder_and_lm_head()
            layers = self._language_layers(decoder)
            self._assert_untied_lm_head(lm_head)
            visual = self._visual_module(self.model)
            if visual is not None and any(
                parameter.requires_grad for parameter in visual.parameters()
            ):
                raise RuntimeError(
                    "Full-weight FSDP2 requires the replicated vision tower to be "
                    "frozen before policy placement."
                )
            self._assert_fp32_trainable_parameters(self.model)
        except Exception as exc:  # noqa: BLE001 - report preflight on every rank
            local_preflight_error = f"{type(exc).__name__}: {exc}"
        preflight_errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(preflight_errors, local_preflight_error)
        if any(error is not None for error in preflight_errors):
            raise RuntimeError(
                "Full-weight policy placement preflight failed: "
                + "; ".join(
                    f"rank {rank}: {error}"
                    for rank, error in enumerate(preflight_errors)
                    if error is not None
                )
            )
        if decoder is None or lm_head is None:
            raise RuntimeError("Full-weight policy placement produced no decoder/head.")

        full_state: dict[str, torch.Tensor] = {}
        nonpersistent_buffers: dict[str, torch.Tensor] = {}
        if dist.get_rank() == 0:
            full_state = dict(self.model.state_dict())
            persistent_names = set(full_state)
            nonpersistent_buffers = {
                name: buffer
                for name, buffer in self.model.named_buffers()
                if name not in persistent_names
            }
        self.model.to(torch.device("meta"))

        # Import composable FSDP lazily so importing CLI/config code stays safe
        # on hosts whose PyTorch build has no distributed accelerator support.
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        self._fsdp_mesh = init_device_mesh(
            self.device.type,
            (dist.get_world_size(),),
            mesh_dim_names=("fsdp",),
        )
        use_bf16 = self.config.bf16 and self.device.type == "cuda"
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if use_bf16 else None,
            reduce_dtype=torch.float32 if use_bf16 else None,
        )
        shard_kwargs = {"mesh": self._fsdp_mesh, "mp_policy": mp_policy}
        for layer in layers:
            fully_shard(layer, **shard_kwargs)

        ignored_params = set(visual.parameters()) if visual is not None else set()
        fully_shard(
            decoder,
            ignored_params=ignored_params or None,
            **shard_kwargs,
        )
        self._fsdp_root = decoder
        self._materialize_sharded_model(
            full_state=full_state,
            nonpersistent_buffers=nonpersistent_buffers,
        )
        self._assert_fp32_trainable_parameters(self.model)

        # The head is intentionally replicated and synchronized explicitly in
        # each optimizer cycle. This keeps vocabulary projection out of FSDP's
        # repeated all-gathers while the large decoder remains sharded.
        from torch.distributed.tensor import DTensor

        if any(isinstance(parameter, DTensor) for parameter in lm_head.parameters()):
            raise RuntimeError("lm_head was unexpectedly sharded by FSDP2.")

    def _materialize_sharded_model(
        self,
        *,
        full_state: dict[str, torch.Tensor],
        nonpersistent_buffers: dict[str, torch.Tensor],
    ) -> None:
        """Broadcast rank-zero base weights into their final FSDP2 layout."""
        from torch.distributed.tensor import DTensor, distribute_tensor

        rank = dist.get_rank()
        sharded_state = self.model.state_dict()
        live_buffers = dict(self.model.named_buffers())
        live_nonpersistent_buffers = {
            name: buffer
            for name, buffer in live_buffers.items()
            if name not in sharded_state
        }
        expected_signatures: list[dict[str, _TensorSignature] | None] = [None]
        if rank == 0:
            expected_signatures[0] = {
                "state": self._tensor_signature(full_state),
                "nonpersistent_buffers": self._tensor_signature(nonpersistent_buffers),
            }
        dist.broadcast_object_list(expected_signatures, src=0)
        expected = expected_signatures[0]
        if expected is None:
            raise RuntimeError("Rank zero did not publish model state metadata.")
        local_signatures = {
            "state": self._tensor_signature(sharded_state),
            "nonpersistent_buffers": self._tensor_signature(live_nonpersistent_buffers),
        }
        local_mismatch = local_signatures != expected
        any_mismatch = torch.tensor(
            int(local_mismatch),
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(any_mismatch, op=dist.ReduceOp.MAX)
        if int(any_mismatch.item()) != 0:
            differences = "; ".join(
                self._first_signature_difference(
                    section,
                    expected[section],
                    local_signatures[section],
                )
                for section in ("state", "nonpersistent_buffers")
                if local_signatures[section] != expected[section]
            )
            if not differences:
                differences = "another rank reported a structural mismatch"
            raise RuntimeError(
                "Full-weight model structure differs from rank zero before "
                f"materialization on rank {rank}: {differences}."
            )
        materialized: dict[str, torch.Tensor] = {}
        for name, target in sharded_state.items():
            if rank == 0:
                source = full_state.pop(name)
                full_tensor = source.detach().to(self.device)
                del source
            else:
                full_tensor = torch.empty(
                    tuple(target.shape),
                    dtype=target.dtype,
                    device=self.device,
                )
            if isinstance(target, DTensor):
                materialized[name] = distribute_tensor(
                    full_tensor,
                    target.device_mesh,
                    target.placements,
                )
            else:
                dist.broadcast(full_tensor, src=0)
                materialized[name] = full_tensor
        self.model.load_state_dict(materialized, assign=True)

        for name, target in live_nonpersistent_buffers.items():
            if rank == 0:
                source = nonpersistent_buffers.pop(name)
                buffer = source.detach().to(self.device)
                del source
            else:
                buffer = torch.empty(
                    tuple(target.shape),
                    dtype=target.dtype,
                    device=self.device,
                )
            dist.broadcast(buffer, src=0)
            if "." in name:
                parent_name, local_name = name.rsplit(".", 1)
                parent = self.model.get_submodule(parent_name)
            else:
                local_name = name
                parent = self.model
            parent.register_buffer(local_name, buffer, persistent=False)

        remaining_meta = [
            name
            for name, tensor in (
                *self.model.named_parameters(),
                *self.model.named_buffers(),
            )
            if tensor.device.type == "meta"
        ]
        if remaining_meta:
            raise RuntimeError(
                "Full-weight FSDP2 initialization left tensors on meta: "
                + ", ".join(remaining_meta[:4])
            )

    @staticmethod
    def _tensor_signature(
        tensors: Mapping[str, torch.Tensor],
    ) -> _TensorSignature:
        return tuple(
            (
                name,
                tuple(int(size) for size in tensor.shape),
                str(tensor.dtype),
            )
            for name, tensor in tensors.items()
        )

    @staticmethod
    def _first_signature_difference(
        section: str,
        expected: _TensorSignature,
        local: _TensorSignature,
    ) -> str:
        for index, (expected_item, local_item) in enumerate(
            zip(expected, local, strict=False)
        ):
            if expected_item != local_item:
                return f"{section}[{index}] expected={expected_item} local={local_item}"
        return f"{section} count expected={len(expected)} local={len(local)}"

    def _base_policy_model(self) -> torch.nn.Module:
        return self.model

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        # No-grad FSDP forwards may temporarily expose unsharded aliases through
        # model.parameters(). Optimizers retain the persistent shard objects.
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            return [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
        return super()._trainable_parameters()

    def _token_scores(
        self,
        batch: dict[str, torch.Tensor],
        *,
        compute_entropies: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return super()._token_scores(batch, compute_entropies=compute_entropies)
        finally:
            # Root modules may retain gathered weights after inference. Return
            # to persistent shards before snapshots, publication or checkpoints.
            if not torch.is_grad_enabled():
                self._reshard_policy()

    def _extract_policy_hidden_states(
        self, rollouts: list[Any]
    ) -> dict[tuple[int, int], list[float]]:
        try:
            return super()._extract_policy_hidden_states(rollouts)
        finally:
            self._reshard_policy()

    def _training_world_size(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Full-weight training requires torch.distributed.")
        return dist.get_world_size()

    def _finalize_window_artifacts(
        self,
        output_dir: Path,
        iteration_index: int,
        *,
        check_health: Callable[[], None] | None = None,
    ) -> WindowArtifacts:
        del output_dir, iteration_index, check_health
        # Durable FSDP checkpoints and live vLLM publication are collective
        # process-boundary operations. The inherited PPO window owns metrics
        # only and must not export PEFT artifacts on every rank.
        return WindowArtifacts()

    def _snapshot_update_state(self) -> dict[str, Any]:
        """Snapshot only this rank's shards on CPU, with no full-weight gather.

        Tensor templates retain the existing optimizer state objects and their
        DTensor layouts; only backup values allocate new storage, on CPU.
        """
        return {
            "parameters": [
                (parameter, self._local_tensor(parameter).detach().to("cpu", copy=True))
                for parameter in self._trainable_parameters()
            ],
            "optimizer_state": {
                parameter: {
                    name: (
                        value,
                        self._local_tensor(value).detach().to("cpu", copy=True),
                    )
                    if torch.is_tensor(value)
                    else copy.deepcopy(value)
                    for name, value in state.items()
                }
                for parameter, state in self.optimizer.state.items()
            },
            "optimizer_groups": [
                {
                    key: copy.deepcopy(value)
                    for key, value in group.items()
                    if key != "params"
                }
                for group in self.optimizer.param_groups
            ],
            "optimizer_step_count": self._optimizer_step_count,
            "dry_optimizer_step_count": self._dry_optimizer_step_count,
        }

    def _restore_update_state(self, snapshot: dict[str, Any]) -> None:
        with torch.no_grad():
            for parameter, backup in snapshot["parameters"]:
                local = self._local_tensor(parameter)
                local.copy_(backup.to(local.device))
            self.optimizer.state.clear()
            for parameter, saved_state in snapshot["optimizer_state"].items():
                state = {}
                for name, saved in saved_state.items():
                    if isinstance(saved, tuple) and torch.is_tensor(saved[0]):
                        value, backup = saved
                        local = self._local_tensor(value)
                        local.copy_(backup.to(local.device))
                        state[name] = value
                    else:
                        state[name] = copy.deepcopy(saved)
                self.optimizer.state[parameter] = state
            for group, saved in zip(
                self.optimizer.param_groups, snapshot["optimizer_groups"], strict=True
            ):
                group.update(saved)
        self._optimizer_step_count = int(snapshot["optimizer_step_count"])
        self._dry_optimizer_step_count = int(snapshot["dry_optimizer_step_count"])

    def _partition_trainable_parameters(
        self,
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        from torch.distributed.tensor import DTensor

        sharded: list[torch.nn.Parameter] = []
        replicated: list[torch.nn.Parameter] = []
        for parameter in self._trainable_parameters():
            if isinstance(parameter, DTensor):
                if (
                    len(parameter.placements) != 1
                    or not parameter.placements[0].is_shard()
                ):
                    raise RuntimeError(
                        "Full-weight PPO supports only one-dimensional sharded "
                        f"FSDP2 parameters, got placements={parameter.placements}."
                    )
                sharded.append(parameter)
            else:
                replicated.append(parameter)
        return sharded, replicated

    def _build_optimizer(self) -> torch.optim.Optimizer:
        sharded, replicated = self._partition_trainable_parameters()
        if not sharded:
            raise RuntimeError(
                "Full-weight FSDP2 policy has no sharded trainable parameters."
            )
        if not replicated:
            raise RuntimeError(
                "Full-weight FSDP2 policy has no replicated lm_head parameters."
            )
        return torch.optim.AdamW(
            [{"params": sharded}, {"params": replicated}],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            # DTensor and native tensors cannot share a foreach kernel group.
            foreach=False,
        )

    def _set_fsdp_gradient_sync(self, *, is_final_slot: bool) -> None:
        # Reduce-scatter after EVERY slot backward. Deferring sync to the final
        # slot keeps each parameter's unsharded gradient resident for the whole
        # cycle (~55 GiB bf16 for 27B on every rank, independent of world
        # size), which OOMs 32k-token windows on 178 GiB devices. Per-slot
        # reduce-scatter accumulates gradients in the sharded fp32 space
        # instead, trading one reduce-scatter per micro-batch for the ~33 GiB
        # per rank that the long-sequence backward actually needs.
        del is_final_slot
        self._fsdp_root.set_requires_gradient_sync(True)
        self._fsdp_root.set_is_last_backward(True)

    def _replicated_trainable_parameters(self) -> list[torch.nn.Parameter]:
        _, replicated = self._partition_trainable_parameters()
        return replicated

    def _synchronize_replicated_gradients(self) -> None:
        world_size = self._training_world_size()
        for parameter in self._replicated_trainable_parameters():
            present = torch.tensor(
                1 if parameter.grad is not None else 0,
                dtype=torch.int32,
                device=parameter.device,
            )
            dist.all_reduce(present, op=dist.ReduceOp.SUM)
            present_count = int(present.item())
            if present_count == 0:
                continue
            if present_count != world_size:
                raise RuntimeError(
                    "Replicated parameter gradient presence differs across "
                    f"FSDP ranks ({present_count}/{world_size})."
                )
            grad = parameter.grad
            if grad is None:  # All ranks took the synchronized error path above.
                raise AssertionError("replicated gradient unexpectedly missing")
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(world_size)

    @staticmethod
    def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def _global_squared_sum(self, *, gradients: bool) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        local_total = torch.zeros((), dtype=torch.float32, device=self.device)
        rank = dist.get_rank()
        for parameter in self._trainable_parameters():
            value = parameter.grad if gradients else parameter.detach()
            if value is None:
                continue
            # A DTensor holds a distinct shard on every rank. Native trainable
            # tensors are replicated, so count their identical value once.
            if isinstance(value, DTensor) or rank == 0:
                local_value = self._local_tensor(value).detach()
                local_total.add_(local_value.float().square().sum())
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)
        return local_total

    def _global_grad_norm_and_clip(self, max_norm: float) -> float:
        squared_norm = self._global_squared_sum(gradients=True)
        grad_norm = float(torch.sqrt(squared_norm).item())
        if not math.isfinite(grad_norm):
            raise RuntimeError(
                "Full-weight PPO produced a non-finite global gradient norm."
            )
        clip_coefficient = float(max_norm) / (grad_norm + 1e-6)
        if clip_coefficient < 1.0:
            for parameter in self._trainable_parameters():
                if parameter.grad is not None:
                    self._local_tensor(parameter.grad).mul_(clip_coefficient)
        return grad_norm

    def _trainable_parameter_l2_norm(self) -> float:
        squared_norm = self._global_squared_sum(gradients=False)
        return float(torch.sqrt(squared_norm).item())
