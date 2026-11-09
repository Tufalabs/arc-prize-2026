from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from training.cli import run_training
from training.config import ExperimentConfig
from training.full_weight_distributed import FullWeightDistributedRuntime
from training.full_weight_launch import SEGMENT_RESULT_ENV
from training.policy_training_backend import FullWeightPolicyTrainingBackend
from training.shutdown import (
    SEGMENT_YIELD_EXIT_CODE,
    SegmentYield,
    TrainingShutdown,
    peer_sigterm_handler,
    training_sigterm_handler,
)


_NCCL_TIMEOUT = datetime.timedelta(minutes=30)
_CONTROL_TIMEOUT = datetime.timedelta(hours=6)


def _local_rank() -> int:
    try:
        return int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            "Distributed training must be launched by torchrun with LOCAL_RANK."
        ) from exc


def run(config_path: str, *, weight_mode: str = "full_weight") -> int:
    """Train on rank zero and serve its collectives elsewhere; return the exit code."""
    config = ExperimentConfig.load(config_path)
    if config.training.weight_mode != weight_mode:
        raise RuntimeError(
            f"training.full_weight_entrypoint requires training.weight_mode={weight_mode!r}."
        )

    local_rank = _local_rank()
    configured_ranks = len(getattr(config.training, weight_mode).device_ids)
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "Distributed training requires torchrun world-size metadata."
        ) from error
    if (
        world_size != configured_ranks
        or local_world_size != configured_ranks
        or not 0 <= local_rank < configured_ranks
    ):
        raise RuntimeError(
            f"Torchrun topology must match training.{weight_mode}.device_ids "
            "on one node."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=_NCCL_TIMEOUT)
    control_group = dist.new_group(backend="gloo", timeout=_CONTROL_TIMEOUT)
    if weight_mode == "flash":
        from training.flash_backend import FlashDistributedRuntime, FlashPolicyTrainingBackend

        runtime = FlashDistributedRuntime(control_group=control_group)
        backend_type = FlashPolicyTrainingBackend
    else:
        runtime = FullWeightDistributedRuntime(control_group=control_group)
        backend_type = FullWeightPolicyTrainingBackend

    exit_code = 0
    completed = False
    try:
        if dist.get_rank() == 0:
            # A scheduled TERM is deferred to an iteration boundary, where the
            # CLI drains checkpoint publication and closes the peers (EXIT).
            with training_sigterm_handler():
                try:
                    run_training(
                        config_path,
                        backend_factory=lambda loaded_config: (
                            backend_type(loaded_config, runtime)
                        ),
                    )
                except SegmentYield:
                    exit_code = SEGMENT_YIELD_EXIT_CODE
                except TrainingShutdown:
                    pass
                # Covers clean no-op paths before the CLI enters its owner finally.
                runtime.close()
        else:
            with peer_sigterm_handler():
                runtime.serve()
        dist.barrier(group=control_group)
        completed = True
    finally:
        # On failure, do not enter new collectives while a peer may be wedged in
        # FSDP/NCCL. torchrun observes the failed rank and terminates the job.
        if completed:
            dist.destroy_process_group()
    return exit_code


def main(*, weight_mode: str = "full_weight") -> None:
    parser = argparse.ArgumentParser(description=f"ARC3 {weight_mode} training pipeline")
    parser.add_argument(
        "--config", required=True, help="Path to the training config JSON file."
    )
    parser.add_argument(
        "--run-dir",
        help=(
            "Use this run directory instead of creating a timestamped one. "
            "If it contains training_state.json, resume its policy checkpoint."
        ),
    )
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.run_dir:
        os.environ["ARC3_RUN_DIR"] = args.run_dir
    exit_code = run(args.config, weight_mode=weight_mode)
    if exit_code == SEGMENT_YIELD_EXIT_CODE and os.environ.get(SEGMENT_RESULT_ENV):
        # run() has drained publication and completed the all-rank barrier.
        # Let torchrun exit successfully, then the parent reports the yield.
        Path(os.environ[SEGMENT_RESULT_ENV]).write_text(str(exit_code))
        return
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
