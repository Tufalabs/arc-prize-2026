"""Bounded two-GPU FSDP2 multimodal acceptance gate inside an existing allocation.

Run with torchrun --standalone --nproc-per-node=2 --max-restarts=0 -m
scripts.full_weight_gpu_preflight --output-dir "$SCRATCH_DIR/full-weight-preflight".
Only the pinned architecture, tokenizer and processor are read from the local HF
cache. Decoder/vision dimensions are reduced and weights are seeded diagnostic
initializations; this does not validate loading or serving the 27B checkpoint.
Each rank writes JSON evidence, and rank zero publishes report.json on success.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from training.config import ExperimentConfig
from training.contracts import RolloutStep, RolloutTrajectory
from training.dataset import build_samples, collate_samples
from training.episode_packing import assign_loss_weights
from training.full_weight_checkpoint import FullWeightCheckpointer
from training.full_weight_ppo import FullWeightPPOTrainer
from training.memory_profile import NullMemoryLogger


SEED = 1729
# Preserve the existing centralized-update oracle's tolerances. This comparison
# uses the same BF16 FSDP graph for both sample partitions, with FP32 reductions.
UPDATE_RTOL = 2e-5
UPDATE_ATOL = 2e-6


class PreflightTrainer(FullWeightPPOTrainer):
    """Use the production trainer with a reduced, randomly initialized policy."""

    def __init__(self, *, diagnostic_config, diagnostic_report, **kwargs):
        self.diagnostic_config = diagnostic_config
        self.diagnostic_report = diagnostic_report
        super().__init__(**kwargs)

    def _micro_batch_forward_backward(self, *args, **kwargs):
        if "first_backward" in self.diagnostic_report:
            return super()._micro_batch_forward_backward(*args, **kwargs)
        record = self.diagnostic_report["first_backward"] = {}
        cuda_marker(record, "before_forward_backward")
        result = super()._micro_batch_forward_backward(*args, **kwargs)
        cuda_marker(record, "after_forward_backward")
        return result

    def _load_base_model(self, model_kwargs):
        return AutoModelForImageTextToText.from_config(
            copy.deepcopy(self.diagnostic_config),
            attn_implementation=model_kwargs.get("attn_implementation"),
            trust_remote_code=True,
        ).float()

    def _load_empty_policy_model(self, *, attn_implementation):
        with torch.device("meta"):
            return self._load_base_model({"attn_implementation": attn_implementation})


def cuda_marker(record, phase):
    """Record both sides of a sync without reading possibly poisoned GPU values."""
    record["phase"] = phase
    record["synchronized"] = False
    print(json.dumps({"rank": dist.get_rank(), "cuda_marker": phase}), flush=True)
    torch.cuda.synchronize()
    record["synchronized"] = True


def tensor_metadata(value):
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "storage_offset": value.storage_offset(),
        "pointer_mod_16": value.data_ptr() % 16,
        "contiguous": value.is_contiguous(),
        "requires_grad": value.requires_grad,
    }


def instrument_first_linear_backward(trainer, report):
    """Observe the first real GDN training operands; never alter tensors/kernels."""
    records = report["linear_attention_kernel_trace"] = {}
    for name, module in trainer.model.named_modules():
        original = getattr(module, "chunk_gated_delta_rule", None)
        if original is None:
            continue

        def traced(*args, _name=name, _original=original, **kwargs):
            if not torch.is_grad_enabled() or _name in records:
                return _original(*args, **kwargs)
            operands = dict(zip(("q", "k", "v"), args, strict=False))
            operands.update(kwargs)
            record = records[_name] = {
                "implementation": f"{_original.__module__}.{_original.__qualname__}",
                "inputs": {
                    key: tensor_metadata(value)
                    for key, value in operands.items()
                    if isinstance(value, torch.Tensor)
                },
            }
            cuda_marker(record, "before_gated_delta_rule_forward")
            result = _original(*args, **kwargs)
            cuda_marker(record, "after_gated_delta_rule_forward")
            output = result[0]
            record["output"] = tensor_metadata(output)

            def observe_gradient(gradient):
                record["output_gradient"] = tensor_metadata(gradient)
                cuda_marker(record, "before_gated_delta_rule_backward")

            output.register_hook(observe_gradient)
            return result

        module.chunk_gated_delta_rule = traced
    if not any(
        getattr(module, "chunk_gated_delta_rule", None) is not None
        for module in trainer.model.modules()
    ):
        raise RuntimeError("Pinned hybrid decoder has no GDN kernel to instrument")


def diagnostic_architecture(model: str, revision: str):
    """Keep the pinned hybrid blocks, rotary geometry and image patch format."""
    config = AutoConfig.from_pretrained(
        model, revision=revision, local_files_only=True, trust_remote_code=True
    )
    text, vision = config.text_config, config.vision_config
    kinds = list(dict.fromkeys(text.layer_types))
    if set(kinds) != {"linear_attention", "full_attention"}:
        raise RuntimeError(f"Expected the pinned hybrid Qwen decoder, got {kinds}")
    text.hidden_size = 128
    text.intermediate_size = 256
    text.num_hidden_layers = len(kinds)
    text.layer_types = kinds
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.linear_num_key_heads = 2
    text.linear_num_value_heads = 4
    text.use_cache = False
    vision.depth = 2
    vision.hidden_size = 128
    vision.intermediate_size = 256
    vision.num_heads = 4
    vision.out_hidden_size = text.hidden_size
    config.tie_word_embeddings = text.tie_word_embeddings = False
    return config


def samples_for(trainer, rollouts):
    signals, _stats, critic = trainer._compute_step_signals(rollouts)
    if critic is not None:
        raise RuntimeError("GRPO preflight entered the state-critic path")
    samples = build_samples(rollouts, per_rollout_signals=signals)
    assign_loss_weights(samples, loss_aggregation="token_mean")
    return samples


def diagnostic_rollouts(trainer, config, processor, identity):
    """Generate actual processor pixels and witnessed native image-token spans."""
    from PIL import Image
    from ullm.media import build_media

    rollouts = []
    for member, (color, reward) in enumerate(
        zip(("red", "blue", "green", "yellow"), (0.0, 1.0, 0.5, -0.1), strict=True)
    ):
        image = Image.new("RGB", (128 + 32 * (member % 2), 128), color)
        image_inputs = processor.image_processor(images=[image], return_tensors="pt")
        grid = image_inputs["image_grid_thw"][0]
        image_tokens = (
            int(grid.prod()) // int(processor.image_processor.merge_size) ** 2
        )
        prefix = [config.vision_start_token_id]
        prefix += [config.image_token_id] * image_tokens
        prefix += [config.vision_end_token_id]
        prefix += trainer.tokenizer.encode(
            "Describe the colors:", add_special_tokens=False
        )
        completion = trainer.tokenizer.encode(
            " " + " ".join([color] * (member + 1)), add_special_tokens=False
        )
        media = build_media(prefix, [image], processor, identity)
        evidence = {
            key: value
            for key, value in media.items()
            if key not in {"version", "images"}
        }
        rollouts.append(
            RolloutTrajectory(
                trajectory_id=f"diagnostic-{member}",
                task_id="synthetic-colors",
                done=True,
                trajectory_reward=reward,
                metadata={
                    "rollout_slot_id": f"diagnostic-slot-{member}",
                    "rollout_seed": SEED + member,
                    "behavior_policy_snapshot": {
                        "iteration": -1,
                        "source": "seeded-diagnostic",
                    },
                },
                steps=[
                    RolloutStep(
                        step_index=0,
                        prompt_messages=[],
                        tool_schemas=[],
                        assistant_message={},
                        reward=reward,
                        done=True,
                        prompt_token_ids=prefix,
                        completion_token_ids=completion,
                        completion_logprobs=[0.0] * len(completion),
                        media=media,
                        media_evidence=evidence,
                    )
                ],
            )
        )
    for rollout, sample in zip(rollouts, samples_for(trainer, rollouts), strict=True):
        batch = collate_samples(
            [sample], pad_token_id=trainer.tokenizer.pad_token_id, processor=processor
        )
        batch = {key: value.to(trainer.device) for key, value in batch.items()}
        with torch.no_grad():
            scores = trainer._token_logprobs(batch)
        count = len(rollout.steps[0].completion_token_ids)
        rollout.steps[0].completion_logprobs = scores[0, -count:].cpu().tolist()
    return rollouts


def accumulators():
    return dict(
        mean_ratio_sum=0.0,
        clip_fraction_sum=0.0,
        entropy_sum=0.0,
        entropy_loss_sum=0.0,
        previous_policy_kl_sum=0.0,
        reference_kl_sum=0.0,
        ratio_updates=0,
        grad_norm_sum=0.0,
        max_grad_norm=0.0,
        grad_norm_updates=0,
        final_loss=0.0,
        global_step=0,
    )


def compare_snapshot(trainer, snapshot, *, exact: bool):
    """Compare local FP32 parameter shards and Adam states without a full gather."""
    rtol, atol = (0.0, 0.0) if exact else (UPDATE_RTOL, UPDATE_ATOL)
    maximum = 0.0
    for parameter, backup in snapshot["parameters"]:
        actual = trainer._local_tensor(parameter).detach().cpu()
        maximum = max(maximum, float((actual - backup).abs().max()))
        torch.testing.assert_close(actual, backup, rtol=rtol, atol=atol)
    if set(trainer.optimizer.state) != set(snapshot["optimizer_state"]):
        raise RuntimeError("Optimizer parameter-state membership changed")
    for parameter, saved in snapshot["optimizer_state"].items():
        for key, (_template, backup) in saved.items():
            actual = (
                trainer._local_tensor(trainer.optimizer.state[parameter][key])
                .detach()
                .cpu()
            )
            torch.testing.assert_close(actual, backup, rtol=rtol, atol=atol)
    return maximum


def weighted_update_oracle(trainer, rollouts):
    """Compare whole-window accumulation to an uneven 2:1 rank partition."""
    initial = trainer._snapshot_update_state()
    samples = samples_for(trainer, rollouts)
    cycle = [[samples[0], samples[1]], [samples[2]], [samples[3]]]
    mass = sum(sample.sequence_weight for sample in samples)
    if mass != sum(len(rollout.steps[0].completion_token_ids) for rollout in rollouts):
        raise RuntimeError("Token-mean denominator differs from generated token count")
    trainer.optimizer.zero_grad(set_to_none=True)
    for chunk in cycle:
        trainer._set_fsdp_gradient_sync(is_final_slot=True)
        trainer._micro_batch_forward_backward(chunk, loss_denominator=mass)
    trainer._synchronize_replicated_gradients()
    trainer._set_optimizer_learning_rate(1)
    expected_norm = trainer._global_grad_norm_and_clip(trainer.config.max_grad_norm)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    expected = trainer._snapshot_update_state()
    trainer._restore_update_state(initial)
    metrics = accumulators()
    trainer._execute_train_cycle(
        cycle,
        cycle_index=1,
        epoch_label="uneven oracle",
        epoch_number=1,
        iteration_index=0,
        loss_denominator=mass,
        world_size=dist.get_world_size(),
        mem=NullMemoryLogger(),
        accumulators=metrics,
        expected_optimizer_steps_total=1,
    )
    difference = compare_snapshot(trainer, expected, exact=False)
    torch.testing.assert_close(
        torch.tensor(metrics["grad_norm_sum"]),
        torch.tensor(expected_norm),
        rtol=UPDATE_RTOL,
        atol=UPDATE_ATOL,
    )
    if expected_norm <= 0.0:
        raise RuntimeError("Multimodal weighted update produced no gradient")
    trainer._restore_update_state(initial)
    return {
        "microbatch_sample_counts": [2, 1, 1],
        "rank_slot_counts": [2, 1],
        "generated_token_mass": mass,
        "reference_grad_norm": expected_norm,
        "distributed_grad_norm": metrics["grad_norm_sum"],
        "max_parameter_abs_difference": difference,
        "rtol": UPDATE_RTOL,
        "atol": UPDATE_ATOL,
    }


def window_checks(trainer, rollouts, output_dir):
    """Exercise accepted extra cycles and rejection of a complete later window."""
    readiness_calls = 0

    def ready():
        nonlocal readiness_calls
        # Calls occur in the same order on every rank. The production runtime
        # separately tests rank-zero callback result/error broadcasts.
        readiness_calls += 1
        return readiness_calls == 2

    random.seed(SEED)
    accepted = trainer.train_window(
        rollouts,
        output_dir=output_dir,
        iteration_index=0,
        policy_lag_windows=0,
        next_window_ready=ready,
        next_window_seconds_remaining=lambda: 600.0,
    )
    if accepted.window_update_rejected or accepted.extra_cycles != 1:
        raise RuntimeError(
            f"Expected one admitted extra cycle, got {accepted.extra_cycles}, rejected={accepted.window_update_rejected}"
        )
    saved = trainer._snapshot_update_state()
    altered = copy.deepcopy(rollouts)
    for rollout in altered:
        rollout.steps[0].completion_logprobs = [
            value - 4.0 for value in rollout.steps[0].completion_logprobs
        ]
    random.seed(SEED)
    rejected = trainer.train_window(
        altered,
        output_dir=output_dir,
        iteration_index=1,
        policy_lag_windows=1,
        next_window_ready=lambda: False,
        next_window_seconds_remaining=lambda: 600.0,
    )
    if not rejected.window_update_rejected or rejected.extra_cycles:
        raise RuntimeError(
            "Clipped window was not rolled back before extra-cycle admission"
        )
    compare_snapshot(trainer, saved, exact=True)
    if trainer.optimizer_step_count != saved["optimizer_step_count"]:
        raise RuntimeError("Rollback did not restore the optimizer step count")
    return {
        "accepted_optimizer_steps": accepted.optimizer_steps,
        "extra_cycles": accepted.extra_cycles,
        "startup_parity_mean_abs": accepted.rollout_logprob_parity_mean_abs,
        "startup_parity_p95_abs": accepted.rollout_logprob_parity_p95_abs,
        "rollback_clip_fraction": rejected.mean_clip_fraction,
        "rollback_threshold": trainer.config.max_window_clip_fraction,
        "rollback_lr_scale": rejected.trust_region_lr_scale,
        "restored_optimizer_step": trainer.optimizer_step_count,
    }


def checkpoint_check(trainer, output_dir, control_group):
    saved = trainer._snapshot_update_state()
    checkpointer = FullWeightCheckpointer(
        output_dir / "checkpoints", process_group=control_group
    )
    checkpoint = checkpointer.save(
        "diagnostic",
        model=trainer.model,
        optimizer=trainer.optimizer,
        optimizer_step=trainer.optimizer_step_count,
        trainer_payload={"checkpoint_identity": trainer._checkpoint_identity},
        metadata_fields={"purpose": "seeded-small-architecture-preflight"},
    )
    with torch.no_grad():
        for parameter in trainer._trainable_parameters():
            trainer._local_tensor(parameter).add_(0.125)
        for state in trainer.optimizer.state.values():
            for value in state.values():
                trainer._local_tensor(value).add_(1)
    metadata = checkpointer.load(
        checkpoint, model=trainer.model, optimizer=trainer.optimizer
    )
    compare_snapshot(trainer, saved, exact=True)
    if metadata.optimizer_step != trainer.optimizer_step_count:
        raise RuntimeError("Checkpoint optimizer count changed")
    return {
        "path": str(checkpoint),
        "optimizer_step": metadata.optimizer_step,
        "exact_roundtrip": True,
    }


def operator_probe(shape, report):
    """Isolate the same FLA operator with reduced or pinned native head counts."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    packed = shape == "native-packed"
    key_heads, value_heads = (2, 4) if shape == "reduced-padded" else (16, 48)
    batch, tokens = (1, 160) if packed else (2, 80)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    query = torch.randn(
        batch,
        tokens,
        key_heads,
        128,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn(
        batch,
        tokens,
        value_heads,
        128,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    gate = (
        -torch.nn.functional.softplus(
            torch.randn(batch, tokens, value_heads, device=device, dtype=torch.float32)
        )
    ).requires_grad_()
    beta = torch.rand(
        batch,
        tokens,
        value_heads,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    operands = {
        "q": query.repeat_interleave(value_heads // key_heads, dim=2),
        "k": key.repeat_interleave(value_heads // key_heads, dim=2),
        "v": value,
        "g": gate,
        "beta": beta,
    }
    if packed:
        operands["cu_seqlens"] = torch.tensor([0, 80, 160], device=device)
    record = report["operator_probe"] = {
        "shape": shape,
        "key_heads_before_repeat": key_heads,
        "inputs": {name: tensor_metadata(item) for name, item in operands.items()},
    }
    report["stage"] = "isolated_gdn_forward"
    cuda_marker(record, "before_isolated_gdn_forward")
    output, _state = chunk_gated_delta_rule(
        **operands, use_qk_l2norm_in_kernel=True, output_final_state=False
    )
    cuda_marker(record, "after_isolated_gdn_forward")
    gradient = torch.randn_like(output)
    record["output_gradient"] = tensor_metadata(gradient)
    report["stage"] = "isolated_gdn_backward"
    output.backward(gradient)
    cuda_marker(record, "after_isolated_gdn_backward")
    for item in (query, key, value, gate, beta):
        if item.grad is None or not bool(torch.isfinite(item.grad).all()):
            raise RuntimeError("Isolated GDN returned absent or nonfinite gradients")
    record["finite_gradients"] = True
    report["stage"] = "complete"


def run(args, report):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Run inside an existing Slurm allocation; this script never allocates GPUs"
        )
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError(
            "This bounded gate requires torchrun with exactly two CUDA ranks"
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError(
            "The pinned Blackwell compute stack requires B200-class GPUs"
        )
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))
    control_group = dist.new_group(
        backend="gloo", timeout=datetime.timedelta(minutes=5)
    )
    torch.manual_seed(SEED)
    if args.operator_probe is not None:
        operator_probe(args.operator_probe, report)
        return control_group
    base = ExperimentConfig.load(args.config).training
    if base.reference_kl_coef != 0 or base.loss_aggregation != "token_mean":
        raise RuntimeError("Use the agreed ARC2 token-mean recipe without reference KL")
    config = diagnostic_architecture(base.model_name, base.model_revision)
    processor = AutoProcessor.from_pretrained(
        base.model_name,
        revision=base.model_revision,
        local_files_only=True,
        trust_remote_code=True,
    )
    from ullm.media import processing_identity

    identity = processing_identity(
        processor,
        model=base.model_name,
        revision=base.model_revision,
        mm_processor_kwargs={},
    )
    training = replace(
        base,
        device=f"cuda:{local_rank}",
        weight_mode="full_weight",
        full_weight=replace(
            base.full_weight, device_ids=list(range(dist.get_world_size()))
        ),
        bf16=True,
        multimodal=True,
        device_map_gpus=[],
        worker_device_map_gpus=[],
        max_sequence_tokens=1024,
        max_micro_batch_tokens=2048,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        epochs_per_window=1,
        episode_packing=False,
        extra_train_cycles_until_window=True,
    )
    report.update(
        model=base.model_name,
        revision=base.model_revision,
        diagnostic_config=config.to_dict(),
        device=torch.cuda.get_device_name(),
        attention=training.attn_implementation,
        weight_mode=training.weight_mode,
        trainer_device_ids=training.full_weight.device_ids,
        processing_identity=identity,
    )
    report["stage"] = "trainer_initialization"
    trainer = PreflightTrainer(
        diagnostic_config=config,
        diagnostic_report=report,
        config=training,
        grpo_group_size=4,
        grpo_puzzles_per_window=1,
        checkpoint_identity={
            "environment": "arc2",
            "advantage_estimator": "grpo",
            "dataset_manifest": {"name": "synthetic-colors", "seed": SEED},
            "processing_stack": identity,
        },
    )
    instrument_first_linear_backward(trainer, report)
    visual = trainer._visual_module(trainer.model)
    if visual is None or any(
        parameter.requires_grad for parameter in visual.parameters()
    ):
        raise RuntimeError("The diagnostic vision encoder/projector is not frozen")
    vision_before = {
        name: value.detach().cpu().clone() for name, value in visual.named_parameters()
    }
    vision_dtypes = set()

    def observe_vision(_module, _inputs, output):
        value = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output[0]
        )
        vision_dtypes.add(str(value.dtype))

    hook = visual.register_forward_hook(observe_vision)
    report["stage"] = "multimodal_inputs"
    rollouts = diagnostic_rollouts(trainer, config, processor, identity)
    report["completion_tokens"] = [
        len(item.steps[0].completion_token_ids) for item in rollouts
    ]
    report["stage"] = "uneven_weighting"
    report["weighted_update"] = weighted_update_oracle(trainer, rollouts)
    report["stage"] = "extra_cycles_and_rollback"
    report["windows"] = window_checks(trainer, rollouts, args.output_dir)
    report["stage"] = "checkpoint_roundtrip"
    report["checkpoint"] = checkpoint_check(trainer, args.output_dir, control_group)
    hook.remove()
    for name, parameter in visual.named_parameters():
        torch.testing.assert_close(
            parameter.detach().cpu(), vision_before[name], rtol=0, atol=0
        )
        if parameter.grad is not None:
            raise RuntimeError(f"Frozen visual parameter acquired a gradient: {name}")
    if vision_dtypes != {"torch.bfloat16"}:
        raise RuntimeError(f"Vision compute was not BF16: {sorted(vision_dtypes)}")
    report["vision"] = {"unchanged": True, "compute_dtypes": sorted(vision_dtypes)}
    report["persistent_dtypes"] = sorted(
        {str(parameter.dtype) for parameter in trainer._trainable_parameters()}
    )
    if report["persistent_dtypes"] != ["torch.float32"]:
        raise RuntimeError("Persistent trainable parameters are not FP32")
    torch.cuda.synchronize()
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    report["stage"] = "complete"
    return control_group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/grpo_8gpu_arc2_full_weight_overfit.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--operator-probe",
        choices=("reduced-padded", "native-padded", "native-packed"),
        help="Run only one isolated FLA forward/backward shape, without FSDP or a model",
    )
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    report: dict[str, Any] = {
        "rank": rank,
        "seed": SEED,
        "stage": "startup",
        "mode": "isolated_gdn_operator"
        if args.operator_probe
        else "full_trainer_preflight",
        "weights": (
            "none; isolated random operator inputs"
            if args.operator_probe
            else "seeded reduced architecture, not the pretrained 27B checkpoint"
        ),
        "versions": {},
        "torch_cuda": torch.version.cuda,
        "kernel_environment": {
            name: os.environ.get(name)
            for name in ("CUDA_LAUNCH_BLOCKING", "FLA_TILELANG")
        },
    }
    for name in (
        "torch",
        "transformers",
        "peft",
        "triton",
        "flash-attn-4",
        "flash-linear-attention",
        "fla-core",
        "causal-conv1d",
        "tilelang",
        "apache-tvm-ffi",
    ):
        try:
            report["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][name] = None
    started = time.monotonic()
    try:
        control_group = run(args, report)
        report.update(ok=True, elapsed_seconds=time.monotonic() - started)
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, report, group=control_group)
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "report.json").write_text(
                json.dumps({"ok": True, "ranks": reports}, indent=2) + "\n"
            )
    except BaseException as error:
        report.update(
            ok=False,
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        # No exception-path collective: torchrun must fail-stop peer ranks even
        # if they are inside a different model/checkpoint collective.
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"report-rank-{rank:03d}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {"rank": rank, "ok": report.get("ok", False), "stage": report["stage"]}
            ),
            flush=True,
        )
    dist.destroy_process_group(control_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
