"""Data-parallel gradient co-worker for the PPO trainer.

Rank 0 (the main trainer process) owns the optimizer, LR schedule, checkpoints,
and all orchestration. Each worker replica is a pure forward/backward engine:
it receives micro-batches for one gradient-accumulation cycle, accumulates
LoRA gradients, contributes them to a NCCL reduce, then installs the freshly
stepped LoRA weights that rank 0 broadcasts back. Workers hold no optimizer
state and never write artifacts, so resume/checkpoint semantics are identical
to single-replica training.

The command protocol is strictly rank-0 driven (workers never initiate a
collective), which makes divergent control flow — time-budget breaks, skipped
windows, evals — impossible to deadlock on.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from training.policy_update import merge_policy_update_statistics
from training.shutdown import worker_sigterm_handler


log = logging.getLogger(__name__)

_CMD_TRAIN_CYCLE = "train_cycle"
_CMD_SYNC_WEIGHTS = "sync_weights"
_CMD_HIDDEN_FEATURES = "hidden_features"
_CMD_EXIT = "exit"
# The default-group (gloo) timeout must comfortably exceed the longest gap
# between commands: workers idle through rollout collection AND eval windows.
_CONTROL_TIMEOUT = datetime.timedelta(hours=6)
_NCCL_TIMEOUT = datetime.timedelta(minutes=30)
_PARENT_POLL_SECONDS = 10.0


def trainable_parameter_list(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Trainable params in deterministic module-traversal order (identical
    across ranks because every replica builds the same architecture)."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def flatten_tensors(
    tensors: list[torch.Tensor], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if not tensors:
        return torch.zeros(0, device=device, dtype=dtype)
    return torch.cat(
        [
            tensor.detach().reshape(-1).to(device=device, dtype=dtype)
            for tensor in tensors
        ]
    )


def unflatten_into(flat: torch.Tensor, targets: list[torch.Tensor]) -> None:
    offset = 0
    for target in targets:
        numel = target.numel()
        chunk = flat[offset : offset + numel].view_as(target)
        target.copy_(chunk.to(device=target.device, dtype=target.dtype))
        offset += numel
    if offset != flat.numel():
        raise RuntimeError(
            f"flat tensor has {flat.numel()} elements but targets consume {offset}."
        )


def flatten_grads(
    parameters: list[torch.nn.Parameter], *, device: torch.device
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for parameter in parameters:
        if parameter.grad is None:
            chunks.append(
                torch.zeros(parameter.numel(), device=device, dtype=torch.float32)
            )
        else:
            chunks.append(
                parameter.grad.detach()
                .reshape(-1)
                .to(device=device, dtype=torch.float32)
            )
    if not chunks:
        return torch.zeros(0, device=device, dtype=torch.float32)
    return torch.cat(chunks)


def round_robin_split(items: list[Any], world_size: int) -> list[list[Any]]:
    """Deal items across ranks preserving order within each rank's share.

    Index k goes to rank k % world_size, so the length difference between any
    two shares is at most one item."""
    shares: list[list[Any]] = [[] for _ in range(world_size)]
    for index, item in enumerate(items):
        shares[index % world_size].append(item)
    return shares


def length_balanced_split(
    items: list[Any], world_size: int, *, size: Callable[[Any], int]
) -> list[list[Any]]:
    """Deal items so that ranks running the same slot in lockstep get similar sizes.

    Items are sorted by `size`, largest first (ties keep their input order),
    and slot s takes sorted[s * world_size : (s + 1) * world_size], rank r
    the r-th item of each slot. Slot 0 holds the largest items, and the
    shares differ in length by at most one item, as in `round_robin_split`.
    Only the assignment changes: the shares hold exactly the input items."""
    return round_robin_split(sorted(items, key=size, reverse=True), world_size)


def parameter_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        total += float(parameter.detach().float().pow(2).sum().item())
    return total**0.5


def _watch_parent(parent_pid: int) -> None:
    """Exit hard if the rank-0 process dies; a worker blocked in a collective
    would otherwise linger until the gloo timeout."""
    while True:
        try:
            os.kill(parent_pid, 0)
        except OSError:
            log.error("parent process %d is gone; exiting worker", parent_pid)
            os._exit(1)
        time.sleep(_PARENT_POLL_SECONDS)


def run_worker(spec: dict[str, Any]) -> None:
    with worker_sigterm_handler(int(spec.get("parent_pid", os.getppid()))):
        _run_worker(spec)


def _run_worker(spec: dict[str, Any]) -> None:
    from training.config import TrainingConfig
    from training.ppo import PPOTrainer

    rank = int(spec["rank"])
    world_size = int(spec["world_size"])
    torch.manual_seed(int(spec["seed"]))

    config = TrainingConfig.from_dict(spec["training_config"])
    gpus = [int(gpu) for gpu in spec["device_map_gpus"]]
    config.device = f"cuda:{gpus[0]}"
    config.device_map_gpus = gpus
    config.worker_device_map_gpus = []

    resume_adapter_dir = spec.get("resume_adapter_dir") or None
    trainer = PPOTrainer(
        config,
        resume_adapter_dir=Path(resume_adapter_dir) if resume_adapter_dir else None,
        resume_state_path=None,
        grpo_group_size=spec.get("grpo_group_size"),
        grpo_puzzles_per_window=spec.get("grpo_puzzles_per_window"),
        checkpoint_identity=spec.get("checkpoint_identity"),
        game_vocab=spec.get("game_vocab"),
    )
    parameters = trainable_parameter_list(trainer.model)
    primary_device = torch.device(config.device)
    torch.cuda.set_device(primary_device)

    # Match rank zero's explicitly owned store. Using init_method here
    # would add a different rendezvous key prefix on the worker side.
    store = dist.TCPStore(
        "127.0.0.1",
        int(spec["port"]),
        world_size=world_size,
        is_master=False,
        timeout=_CONTROL_TIMEOUT,
    )
    dist.init_process_group(
        "gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=_CONTROL_TIMEOUT,
    )
    nccl_group = dist.new_group(backend="nccl", timeout=_NCCL_TIMEOUT)
    log.info(
        "ddp worker rank %d/%d ready on gpus %s (%d trainable tensors)",
        rank,
        world_size,
        gpus,
        len(parameters),
    )

    # Rank 0 broadcasts its initial LoRA weights so every replica starts from
    # bit-identical parameters regardless of init/resume path.
    flat_weights = flatten_tensors(
        [parameter.data for parameter in parameters],
        device=primary_device,
        dtype=torch.float32,
    )
    dist.broadcast(flat_weights, src=0, group=nccl_group)
    with torch.no_grad():
        unflatten_into(flat_weights, [parameter.data for parameter in parameters])
    log.info("installed initial weights (L2=%.6f)", parameter_l2_norm(parameters))

    trainer.model.train()
    step_index = 0
    while True:
        command: list[Any] = [None]
        dist.broadcast_object_list(command, src=0)
        payload = command[0]
        if payload["type"] == _CMD_EXIT:
            log.info("received exit command after %d cycles", step_index)
            break
        if payload["type"] == _CMD_SYNC_WEIGHTS:
            # Out-of-cycle resync: rank 0 restored its parameters (window
            # trust region rejected an update) and every replica must match
            # before the next forward computes ratios.
            dist.broadcast(flat_weights, src=0, group=nccl_group)
            with torch.no_grad():
                unflatten_into(
                    flat_weights, [parameter.data for parameter in parameters]
                )
            log.info(
                "resynced weights on command (L2=%.6f)",
                parameter_l2_norm(parameters),
            )
            continue
        if payload["type"] == _CMD_HIDDEN_FEATURES:
            # This is a read-only decoder pass under the synchronized policy.
            # Return only selected CPU hidden states; no gradients, critic fit,
            # adapter changes, or filesystem artifacts are produced here.
            features = trainer._extract_policy_hidden_batches(
                payload["shares"][rank - 1], check_shutdown=False
            )
            dist.gather_object(features, None, dst=0)
            continue
        if payload["type"] != _CMD_TRAIN_CYCLE:
            raise RuntimeError(f"unknown ddp worker command: {payload['type']!r}")

        micro_batches = payload["shares"][rank - 1]
        loss_denominator = float(payload["loss_denominator"])
        # The entropy-target controller varies the coefficient per window;
        # every rank must apply the value rank 0 used for this cycle.
        if "entropy_coef" in payload:
            trainer.live_entropy_coef = max(0.0, float(payload["entropy_coef"]))
        metrics_sums = {
            "mean_ratio_sum": 0.0,
            "clip_fraction_sum": 0.0,
            "entropy_sum": 0.0,
            "entropy_loss_sum": 0.0,
            "previous_policy_kl_sum": 0.0,
            "reference_kl_sum": 0.0,
            "micro_batches": 0,
            "last_loss": 0.0,
            "policy_update_statistics": {},
        }
        for chunk in micro_batches:
            metrics = trainer._micro_batch_forward_backward(
                chunk, loss_denominator=loss_denominator
            )
            for key in ("processed_tokens", "forward_time_s", "backward_time_s"):
                metrics_sums[key] = metrics_sums.get(key, 0) + metrics.get(key, 0)
            metrics_sums["mean_ratio_sum"] += metrics["mean_ratio"]
            metrics_sums["clip_fraction_sum"] += metrics["clip_fraction"]
            metrics_sums["entropy_sum"] += metrics["entropy"]
            metrics_sums["entropy_loss_sum"] += metrics["entropy_loss"]
            metrics_sums["previous_policy_kl_sum"] += metrics.get(
                "previous_policy_kl", 0.0
            )
            metrics_sums["reference_kl_sum"] += metrics.get("reference_kl", 0.0)
            metrics_sums["micro_batches"] += 1
            metrics_sums["last_loss"] = metrics["loss"]
            merge_policy_update_statistics(
                metrics_sums["policy_update_statistics"],
                metrics.get("policy_update_statistics", {}),
            )

        flat_grads = flatten_grads(parameters, device=primary_device)
        dist.reduce(flat_grads, dst=0, group=nccl_group)
        trainer.optimizer.zero_grad(set_to_none=True)

        # Rank 0 needs every forward-phase clip statistic before deciding
        # whether this cycle may mutate the policy.
        dist.gather_object(metrics_sums, None, dst=0)
        dist.broadcast(flat_weights, src=0, group=nccl_group)
        with torch.no_grad():
            unflatten_into(flat_weights, [parameter.data for parameter in parameters])
        step_index += 1
        if step_index % 10 == 1:
            log.info(
                "cycle %d: trained %d micro-batches, post-step weight L2=%.6f",
                step_index,
                metrics_sums["micro_batches"],
                parameter_l2_norm(parameters),
            )

    # Rank 0 destroys its groups concurrently, before waiting for this process.
    # NCCL teardown must not depend on a peer that is waiting for us to exit.
    dist.destroy_process_group()
    log.info("ddp worker rank=%d process groups destroyed; exiting cleanly", rank)


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO data-parallel gradient worker.")
    parser.add_argument("--spec", required=True, help="Path to the worker spec JSON.")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    log_path = spec.get("log_path")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )

    watchdog = threading.Thread(
        target=_watch_parent, args=(int(spec["parent_pid"]),), daemon=True
    )
    watchdog.start()
    run_worker(spec)


if __name__ == "__main__":
    main()
