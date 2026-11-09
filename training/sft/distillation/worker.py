"""Distributed router-only distillation from an immutable teacher cache."""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from sft.distillation.accumulation import (
    RouterGradientAccumulator,
    checkpoint_cursor,
    microbatch_count,
    nvfp4_expert_digests,
    remaining_updates,
    validate_resume_fingerprint,
    validate_resume_manifest,
)
from sft.distillation.common import (
    TEACHER,
    Dataset,
    dump,
    row_key,
    sha,
    validate_cache,
    validate_student,
)
from sft.distillation.compute_metrics import useful_model_flops
from sft.distillation.diagnostics import (
    assert_tree_exact,
    atomic_save,
    cpu_tree,
    tree_digest,
)
from sft.distillation.flash_model import Reader, build_model, frozen_fingerprint
from sft.distillation.gradient_noise import GradientNoise, local_moments
from sft.distillation.host_memory import record_pinned
from sft.distillation.loss import completion_distillation, objective_weights
from sft.distillation.pruning_probe import PruningProbe, validate_probe
from sft.distillation.routing import RouteRecorder, attach
from sft.distillation.update_ratio import router_update_ratio, snapshot_routers
from training.blackwell_gdn import configure_blackwell_gdn
from sft.distillation.config import WORKER_KERNEL_FLAGS, worker_kernel_stack
from training.flash.kernel_mode import KERNEL_FIELDS, KERNEL_MODES, builder_kwargs
from training.flash.options import EXPERT_OPTIMIZATIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=0.0)
    ap.add_argument("--loss-weighting", choices=("token", "example"), default="example")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--probe-config", type=Path, required=True)
    # Execution stack: --kernel-mode picks the preset; the kernel flags below
    # and --kernel-options (JSON) override single fields of it.
    ap.add_argument("--kernel-mode", choices=KERNEL_MODES, default="reference")
    ap.add_argument("--kernel-options", default="{}")
    ap.add_argument("--attention-backend", choices=("sdpa", "flex_qsa"))
    ap.add_argument(
        "--qsa-kernel",
        choices=("flex", "indexed_backward", "physical_sparse", "group_sparse"),
        default=None,
    )
    ap.add_argument("--reuse-qsa-selection", action="store_true")
    ap.add_argument(
        "--retain-qsa-attention", action=argparse.BooleanOptionalAction, default=None
    )
    ap.add_argument("--expert-optimization", choices=EXPERT_OPTIMIZATIONS)
    ap.add_argument("--expert-lora", default="null")
    ap.add_argument(
        "--train-routers", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument(
        "--fuse-norm-tail", action=argparse.BooleanOptionalAction, default=None
    )
    ap.add_argument("--batching-config", default="{}")
    ap.add_argument("--offload-every", type=int, default=1)
    ap.add_argument("--gradient-accumulation", type=int, default=1)
    ap.add_argument("--data-microbatches", type=int)
    ap.add_argument("--allow-accumulation-change", action="store_true")
    ap.add_argument("--resume-sha256")
    args = ap.parse_args()
    from training.flash.expert_lora import normalize_lora

    stack = worker_kernel_stack(args)
    for name in WORKER_KERNEL_FLAGS:
        setattr(args, name, stack[name])
    args.expert_lora = normalize_lora(json.loads(args.expert_lora))
    if not args.train_routers and args.expert_lora is None:
        raise ValueError("training requires routers or expert LoRA")
    if args.expert_lora and args.expert_optimization == "none":
        raise ValueError("expert LoRA requires an optimized expert implementation")
    from training.flash.offload import resolve_qsa_retention

    args.retain_qsa_attention = resolve_qsa_retention(
        args.attention_backend, args.reuse_qsa_selection, args.retain_qsa_attention
    )
    microbatch_count(0, args.gradient_accumulation, args.data_microbatches)
    if args.retain_qsa_attention and (
        args.attention_backend != "flex_qsa" or not args.reuse_qsa_selection
    ):
        raise ValueError("QSA retention requires flex_qsa and selection reuse")
    if args.reuse_qsa_selection and args.attention_backend != "flex_qsa":
        raise ValueError("QSA selection reuse requires flex_qsa")
    if args.offload_every < 0:
        raise ValueError("offload_every must be nonnegative")
    if args.steps < 1 or args.save_every < 1 or args.lr <= 0:
        raise ValueError("invalid steps, checkpoint interval, or learning rate")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if stack["gdn_guard"]:
        # Before any FLA kernel runs: the guard refuses an autotuner that has
        # already picked a configuration.
        configure_blackwell_gdn("fla", device=torch.device("cuda", rank))
    dist.init_process_group(
        "nccl",
        timeout=datetime.timedelta(minutes=30),
        device_id=torch.device("cuda", rank),
    )
    world = dist.get_world_size()
    if world not in (2, 4):
        raise ValueError("The FSDP layout requires two or four ranks")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    probe_config = json.loads(args.probe_config.read_text())
    validate_probe({"probe": probe_config, "data": str(args.data)})
    dataset = Dataset(args.data)
    if len(dataset) < world:
        raise ValueError("need at least one example per rank")
    from sft.distillation.batching import BatchPlanner

    lengths = [
        len(row["input_token_ids"]) + len(row["output_token_ids"]) for row in dataset
    ]
    planner = BatchPlanner(lengths, world, args.seed, json.loads(args.batching_config))
    if (
        planner.config["policy"] == "length_within_update"
        and args.gradient_accumulation < 2
    ):
        raise ValueError("length_within_update requires gradient_accumulation >= 2")
    roles = validate_student(TEACHER, args.student)
    cache = validate_cache(args.cache, args.data, args.student)
    manifest = {
        "batching": planner.config,
        # The whole resolved stack; the selector keeps its historical key.
        "kernel_mode": args.kernel_mode,
        **{name: stack[name] for name in KERNEL_FIELDS if name != "selector_policy"},
        "qsa_selector_policy": stack["selector_policy"],
        "reuse_qsa_selection": args.reuse_qsa_selection,
        "retain_qsa_attention": args.retain_qsa_attention,
        "expert_lora": args.expert_lora,
        "train_routers": args.train_routers,
        "activation_offload": "cpu-pinned-checkpoint-inputs-v1",
        "offload_every": args.offload_every,
        "probe_config": probe_config,
        "probe_cache_sha256": sha(Path(probe_config["cache"]) / "manifest.json"),
        "objective": "T^2 KL(teacher||student) + ce_weight * completion_CE",
        "loss_weighting": args.loss_weighting,
        "temperature": args.temperature,
        "ce_weight": args.ce_weight,
        "lr": args.lr,
        "chunk": args.chunk,
        "seed": args.seed,
        "world_size": world,
        "gradient_accumulation": args.gradient_accumulation,
        "data_sha256": sha(args.data),
        "teacher_cache_sha256": sha(args.cache / "manifest.json"),
        "student_config_sha256": sha(args.student / "config.json"),
        "student_index_sha256": sha(args.student / "model.safetensors.index.json"),
        "pruning_sha256": sha(args.student / "distillation-parent.json"),
    }
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "run-lock").mkdir()
        dump(args.out / "manifest.json", manifest)
        dump(
            args.out / "expert-teacher-ids.json",
            roles["kept_teacher_expert_ids"],
        )
    dist.barrier()
    if rank == 0:
        print(json.dumps({"kernel_mode": args.kernel_mode, "kernel_stack": stack}), flush=True)
    model, head, routers = build_model(
        args.student,
        offload_every=args.offload_every,
        reuse_qsa_selection=args.reuse_qsa_selection,
        retain_qsa_attention=args.retain_qsa_attention,
        expert_lora=args.expert_lora,
        train_routers=args.train_routers,
        **builder_kwargs(stack),
    )
    teacher_head = load_file(args.cache / "head.safetensors", device="cpu")[
        "weight"
    ].cuda()
    teacher_head.requires_grad_(False)
    before = frozen_fingerprint(model, head)
    optimizer = torch.optim.AdamW(
        list(routers.values()), lr=args.lr, weight_decay=0, foreach=False
    )
    recorder = RouteRecorder(48, roles["experts"], "cuda")
    attach(model, recorder)
    probe = PruningProbe(
        probe_config,
        args.student,
        model,
        head,
        teacher_head,
        args.temperature,
        args.chunk,
        args.out,
        recorder,
    )
    gradient_noise = GradientNoise(routers)
    compute_config = json.loads((args.student / "config.json").read_text())[
        "text_config"
    ]
    start = 0
    data_cursor = 0
    if args.resume:
        if args.resume_sha256 and sha(args.resume) != args.resume_sha256:
            raise ValueError("resume checkpoint hash changed")
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        accumulation_changed = validate_resume_manifest(
            saved["manifest"], manifest, args.allow_accumulation_change
        )
        assert saved["router_names"] == list(routers)
        data_cursor = checkpoint_cursor(saved)
        # An NVFP4-era checkpoint hashed the packed experts the build now
        # expands to BF16: they are hashed again from the student, and the
        # next checkpoint records the BF16 fingerprint.
        validate_resume_fingerprint(
            saved["manifest"],
            saved["frozen"][rank],
            before,
            lambda parents: nvfp4_expert_digests(
                Reader(args.student),
                parents,
                compute_config["num_experts"],
                saved["manifest"].get("nvfp4_distribution", "replicated"),
                rank,
                world,
            ),
        )
        with torch.no_grad():
            for name, value in routers.items():
                value.copy_(saved["routers"][name])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"][rank]["cpu"])
        torch.cuda.set_rng_state(saved["rng"][rank]["cuda"])
        assert_tree_exact(
            {k: v.detach().cpu() for k, v in routers.items()}, saved["routers"]
        )
        assert_tree_exact(optimizer.state_dict(), saved["optimizer"])
        assert_tree_exact(
            {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
            saved["rng"][rank],
        )
        start = saved["step"]
        probe.restore(saved["probe"][rank])
        # Old checkpoints lack this new global-only diagnostic; start its EMA fresh.
        if "global_gradient_noise" in saved and not accumulation_changed:
            gradient_noise.restore(saved["global_gradient_noise"])
    if start >= args.steps:
        raise ValueError("requested total steps already completed")
    if args.data_microbatches is not None and args.steps - start > remaining_updates(
        data_cursor, args.data_microbatches, args.gradient_accumulation
    ):
        raise ValueError("optimizer update target exceeds remaining data budget")
    initial = {k: v.detach().cpu().clone() for k, v in routers.items()}
    if not args.resume:
        probe.evaluate(start)
    for step in range(start, args.steps):
        micros = microbatch_count(
            data_cursor, args.gradient_accumulation, args.data_microbatches
        )
        cohort = planner.update(data_cursor, micros)
        indices = [group[rank] for group in cohort]
        active_slots = [index >= 0 for index in indices]
        # All ranks execute identical FSDP collectives. Inactive slots replay a
        # real cached example, but contribute zero objective/statistics weight.
        load_indices = [
            index
            if index >= 0
            else min((i for i in group if i >= 0), key=lengths.__getitem__)
            for index, group in zip(indices, cohort, strict=True)
        ]
        rows = [dataset[index] for index in load_indices]
        teacher_hiddens = []
        for index, row in zip(load_indices, rows, strict=True):
            expected = cache["entries"][str(index)]
            path = args.cache / f"{index:08d}.safetensors"
            if (
                row_key(row) != expected["row_sha256"]
                or sha(path) != expected["file_sha256"]
            ):
                raise ValueError("teacher target/token mismatch or cache corruption")
            teacher_hiddens.append(load_file(path, device="cpu")["hidden"])
        sequence_lengths = [
            len(row["input_token_ids"]) + len(row["output_token_ids"]) for row in rows
        ]
        completion_lengths = [len(row["output_token_ids"]) for row in rows]
        from training.flash.flex_qsa import begin_metrics, finish_metrics

        begin_metrics()
        # Preload only CPU targets; each GPU graph is consumed and released before the next.
        count = torch.tensor(
            sum(
                n
                for n, active in zip(completion_lengths, active_slots, strict=True)
                if active
            ),
            device="cuda",
            dtype=torch.float32,
        )
        dist.all_reduce(count)
        global_tokens = count.item()
        global_examples = sum(index >= 0 for group in cohort for index in group)
        weights = [
            objective_weights(
                args.loss_weighting, n, global_tokens, global_examples, world
            )
            for n in completion_lengths
        ]
        optimizer.zero_grad(set_to_none=True)
        accumulated = RouterGradientAccumulator()
        moments = torch.zeros(2, device="cuda", dtype=torch.float64)
        loss_value = torch.zeros((), device="cuda", dtype=torch.float32)
        example_loss_value = torch.zeros_like(loss_value)
        usage_counts = torch.zeros_like(recorder.counts)
        usage_mass = torch.zeros_like(recorder.mass)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.monotonic()
        for row, teacher_hidden, local_tokens, (scale, weight), active in zip(
            rows,
            teacher_hiddens,
            completion_lengths,
            weights,
            active_slots,
            strict=True,
        ):
            prompt = len(row["input_token_ids"])
            ids = torch.tensor(
                [row["input_token_ids"] + row["output_token_ids"]],
                device="cuda",
            )
            record_pinned(args.out, rank, step + 1, "before_forward")
            recorder.begin(prompt)
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                try:
                    hidden = model(input_ids=ids, use_cache=False).last_hidden_state
                finally:
                    recorder.finish()
                loss_sum = completion_distillation(
                    hidden,
                    head.weight,
                    teacher_hidden,
                    teacher_head,
                    ids,
                    prompt,
                    args.temperature,
                    args.ce_weight,
                    args.chunk,
                )
                # All ranks AND actual microbatches share one objective denominator.
                if not active:
                    (loss_sum * 0.0).backward()
                elif args.loss_weighting == "token":
                    (loss_sum * world / count).backward()
                else:
                    (loss_sum * scale).backward()
            record_pinned(args.out, rank, step + 1, "after_backward")
            if active and global_examples > 1:
                moments.add_(
                    local_moments(
                        routers,
                        local_tokens,
                        global_tokens,
                        world,
                        sequence_weight=weight,
                    )
                )
            accumulated.add(routers)
            if active:
                loss_value.add_(loss_sum.detach())
                example_loss_value.add_(loss_sum.detach() / local_tokens)
                usage_counts.add_(recorder.counts)
                usage_mass.add_(recorder.mass)
            del hidden, loss_sum, ids
        accumulated.assign(routers)
        dist.all_reduce(moments)
        for name, p in routers.items():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError("Missing/nonfinite router gradient: " + name)
            dist.all_reduce(p.grad)
            p.grad.div_(world)
        for p in model.parameters():
            if not p.requires_grad and p.grad is not None:
                raise RuntimeError("frozen weight received gradient")
        noise_report = (
            gradient_noise.finish(routers, moments)
            if global_examples > 1
            else {
                "effective_sequences": 1.0,
                "observations": gradient_noise.state["observations"],
                "scopes": {},
                "reason": "single_example_no_variance_estimate",
            }
        )
        group_norms = {}
        if args.expert_lora:
            for group, adapter in (("router", False), ("expert_lora", True)):
                norms = [
                    torch.linalg.vector_norm(p.grad)
                    for name, p in routers.items()
                    if (".lora_" in name) == adapter
                ]
                group_norms[group + "_grad_norm"] = (
                    torch.linalg.vector_norm(torch.stack(norms)).item()
                    if norms
                    else 0.0
                )
        norm = torch.nn.utils.clip_grad_norm_(
            list(routers.values()), 1.0, error_if_nonfinite=True
        )
        # Routers are replicated: one rank measures the actual optimizer update.
        router_before = snapshot_routers(routers) if rank == 0 else None
        optimizer.step()
        if rank == 0:
            update_ratio = router_update_ratio(routers, router_before)
            del router_before
        dist.all_reduce(loss_value)
        loss_value /= count
        dist.all_reduce(example_loss_value)
        example_loss_value /= global_examples
        objective_loss = (
            loss_value if args.loss_weighting == "token" else example_loss_value
        )
        if not torch.isfinite(loss_value) or not torch.isfinite(objective_loss):
            raise RuntimeError("nonfinite distillation loss")
        recorder.counts.copy_(usage_counts)
        recorder.mass.copy_(usage_mass)
        routes = recorder.report()
        del usage_counts, usage_mass
        torch.cuda.synchronize()
        update_seconds = time.monotonic() - begin
        data_cursor += micros
        metrics = {
            "step": step + 1,
            "row_index": indices[0] if indices else -1,
            "row_indices": indices,
            "load_row_indices": load_indices,
            "active_slots": active_slots,
            "batching": planner.config,
            "cohort_sequence_lengths": [
                [lengths[i] if i >= 0 else 0 for i in group] for group in cohort
            ],
            "sequence_lengths": sequence_lengths,
            "sequence_length": sum(
                n
                for n, active in zip(sequence_lengths, active_slots, strict=True)
                if active
            ),
            "microbatches": micros,
            "data_microbatch_cursor": data_cursor,
            "global_completion_tokens": count.item(),
            "loss_per_token": loss_value.item(),
            "loss_per_example": example_loss_value.item(),
            "objective_loss": objective_loss.item(),
            "loss_weighting": args.loss_weighting,
            "global_examples": global_examples,
            "router_grad_norm": norm.item(),
            "global_gradient_noise": noise_report,
            "seconds": update_seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        if args.expert_lora:
            metrics["trainable_grad_norm"] = norm.item()
            metrics.update(group_norms)
        metrics["attention"] = finish_metrics()
        estimates = [
            useful_model_flops(compute_config, n, c)
            for n, c in zip(sequence_lengths, completion_lengths, strict=True)
        ]
        for estimate, active in zip(estimates, active_slots, strict=True):
            if not active:
                estimate["parts"] = dict.fromkeys(estimate["parts"], 0)
                estimate["useful_model_flops_estimate"] = 0
        compute = {
            "schema": estimates[0]["schema"],
            "parts": {
                key: sum(e["parts"][key] for e in estimates)
                for key in estimates[0]["parts"]
            },
            "useful_model_flops_estimate": sum(
                e["useful_model_flops_estimate"] for e in estimates
            ),
        }
        compute["accounting_seconds"] = update_seconds
        metrics["compute"] = compute
        if rank == 0:
            metrics[
                "trainable_update_ratio" if args.expert_lora else "router_update_ratio"
            ] = update_ratio
        with (args.out / f"metrics-rank{rank}.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        if rank == 0:
            dump(args.out / f"expert-usage-step{step + 1}.json", routes)
            print(json.dumps(metrics), flush=True)
        del teacher_hiddens, teacher_hidden, rows
        record_pinned(args.out, rank, step + 1, "after_cleanup")
        if (step + 1) % probe_config["every"] == 0 or step + 1 == args.steps:
            optimizer.zero_grad(set_to_none=True)
            probe.evaluate(
                step + 1,
                record_history=(step + 1) % probe_config["every"] == 0,
            )
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            rng, frozen = [None] * world, [None] * world
            dist.all_gather_object(
                rng, {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
            )
            dist.all_gather_object(frozen, before)
            probe_states = [None] * world
            dist.all_gather_object(probe_states, probe.state)
            if rank == 0:
                atomic_save(
                    {
                        "manifest": manifest,
                        "step": step + 1,
                        "data_microbatch_cursor": data_cursor,
                        "router_names": list(routers),
                        "routers": cpu_tree(routers),
                        "optimizer": cpu_tree(optimizer.state_dict()),
                        "rng": rng,
                        "frozen": frozen,
                        "probe": probe_states,
                        "global_gradient_noise": gradient_noise.state,
                    },
                    args.out / f"checkpoint-step{step + 1}.pt",
                )
            dist.barrier()
    final = cpu_tree(routers)
    assert any(not torch.equal(initial[k], final[k]) for k in final), (
        "No trainable weights changed"
    )
    if args.expert_lora:
        assert any(
            not torch.equal(initial[k], final[k]) for k in final if ".lora_" in k
        ), "No expert LoRA weights changed"
    assert before == frozen_fingerprint(model, head), "Frozen student changed"
    hashes = [None] * world
    dist.all_gather_object(hashes, tree_digest(final))
    assert len(set(hashes)) == 1, "Trainable replicas differ"
    dump(
        args.out / f"final-rank{rank}.json",
        {
            "status": "passed",
            "frozen_unchanged": True,
            "router_replicas_equal": True,
            "trainable_replicas_equal": True,
            "expert_lora_updated": bool(args.expert_lora),
            "completed_steps": args.steps,
            "data_microbatch_cursor": data_cursor,
            "gradient_accumulation": args.gradient_accumulation,
            "resume_exact_restore_checked": bool(args.resume),
        },
    )
    if rank == 0:
        # With LoRA, expose only a complete trainable bundle: an old router-only
        # exporter must not silently drop the learned expert corrections.
        filename = (
            "trainables-fp32.safetensors"
            if args.expert_lora
            else "routers-fp32.safetensors"
        )
        save_file(final, args.out / filename)
    dist.barrier()
    if rank == 0:
        assert all(
            json.loads((args.out / f"final-rank{i}.json").read_text())["status"]
            == "passed"
            for i in range(world)
        )
        dump(
            args.out / "TRAINING_PASSED.json",
            {
                "steps": args.steps,
                "qualification": "finite training/frozen-state/replica checks; not exact numerical replay or quality evaluation",
            },
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
