from __future__ import annotations

import datetime
import os
import random
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gloo_ranks import spawn_ranks
import training.full_weight_checkpoint as checkpoint_module
from training.full_weight_checkpoint import FullWeightCheckpointMetadata
from training.full_weight_distributed import FullWeightDistributedRuntime
from training.shutdown import (
    TrainingShutdown,
    peer_sigterm_handler,
    raise_if_shutdown_requested,
    shutdown_requested,
    training_sigterm_handler,
)


class _HealthCheckTrainer:
    def train_window(self, _rollouts, **kwargs):
        if dist.get_rank() != 0:
            os.kill(os.getpid(), signal.SIGTERM)
        kwargs["check_health"]()
        # A model collective must still be reached by both ranks after TERM.
        count = torch.tensor(1)
        dist.all_reduce(count)
        assert count.item() == 2
        kwargs["check_health"]()
        return "completed"


def _sigterm_callback_worker(rank, rendezvous, fatal):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=15),
    )
    try:
        runtime = FullWeightDistributedRuntime()
        runtime._trainer = _HealthCheckTrainer()
        if rank == 0:
            with training_sigterm_handler():

                def health():
                    os.kill(os.getpid(), signal.SIGTERM)
                    raise_if_shutdown_requested()
                    if fatal:
                        raise RuntimeError("real worker failure")

                def train():
                    return runtime.train_window(
                        [],
                        output_dir=Path(rendezvous).parent,
                        iteration_index=0,
                        sample_budget_fraction=1.0,
                        policy_lag_windows=0,
                        entropy_coef=None,
                        check_health=health,
                    )

                if fatal:
                    with pytest.raises(RuntimeError, match="real worker failure"):
                        train()
                else:
                    assert train() == "completed"
                    assert shutdown_requested()
                    with pytest.raises(TrainingShutdown):
                        raise_if_shutdown_requested()
                    runtime.close()
        else:
            with peer_sigterm_handler():
                if fatal:
                    with pytest.raises(RuntimeError, match="real worker failure"):
                        runtime.serve()
                else:
                    runtime.serve()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fatal", [False, True])
def test_sigterm_during_a_two_rank_callback_preserves_collectives_and_fatal_checks(
    tmp_path, fatal
):
    mp.spawn(_sigterm_callback_worker, args=(str(tmp_path / "init"), fatal), nprocs=2, join=True)


class _StateCritic:
    def __init__(self) -> None:
        self.value = torch.tensor(0.0)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.value = state["value"].clone()


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = torch.nn.Linear(3, 2)
        self.lm_head = torch.nn.Linear(2, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.decoder(inputs))


class _Trainer:
    def __init__(self, *, config, game_vocab, memory_logger, **identity_kwargs) -> None:
        del memory_logger
        self.identity_kwargs = identity_kwargs
        self.config = config
        self.game_vocab = list(game_vocab)
        torch.manual_seed(17)
        self.model = _Policy()

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.tensor import DTensor

        mesh = init_device_mesh("cpu", (dist.get_world_size(),))
        fully_shard(self.model.decoder, mesh=mesh)
        sharded = [
            parameter
            for parameter in self.model.parameters()
            if isinstance(parameter, DTensor)
        ]
        replicated = [
            parameter
            for parameter in self.model.parameters()
            if not isinstance(parameter, DTensor)
        ]
        self.optimizer = torch.optim.AdamW(
            [{"params": sharded}, {"params": replicated}],
            lr=0.01,
            foreach=False,
        )
        self.state_critic = _StateCritic()
        self._optimizer_step_count = 0
        self._dry_optimizer_step_count = 0
        self._trust_region_lr_scale = 0.5
        self._consecutive_window_rejections = 2
        self.last_random: tuple[float, float] | None = None

    @property
    def optimizer_step_count(self) -> int:
        return self._optimizer_step_count

    def train_window(self, rollouts, **kwargs):
        assert kwargs["next_window_ready"] is None
        assert kwargs["next_window_seconds_remaining"] is None
        inputs = rollouts[0]
        self.optimizer.zero_grad(set_to_none=True)
        self.model(inputs).square().sum().backward()
        for parameter in self.model.lm_head.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(dist.get_world_size())
        self.optimizer.step()
        self._optimizer_step_count += 1
        self.state_critic.value += 1
        self._trust_region_lr_scale = 0.25
        self._consecutive_window_rejections = 3
        self.last_random = (random.random(), float(torch.rand(()).item()))
        return {
            "iteration_index": kwargs["iteration_index"],
            "random": self.last_random,
        }


@pytest.mark.parametrize(
    ("payload_iteration", "metadata_fields", "message"),
    [
        (1, {"model_name": "tiny", "model_revision": "revision-a"}, "iteration"),
        (2, {"model_name": "other", "model_revision": "revision-a"}, "model_name"),
        (2, {"model_name": "tiny", "model_revision": "other"}, "model_revision"),
    ],
)
def test_checkpoint_identity_must_match_run_state(
    payload_iteration: int,
    metadata_fields: dict[str, object],
    message: str,
) -> None:
    runtime = object.__new__(FullWeightDistributedRuntime)
    metadata = FullWeightCheckpointMetadata(
        name="full_weight_iter_000002",
        optimizer_step=1,
        trainable_manifest=(),
        trainer_payload={"iteration_index": payload_iteration},
        metadata_fields=metadata_fields,
    )

    with pytest.raises(RuntimeError, match=message):
        runtime._restore_checkpoint_payload(
            metadata,
            expected={
                "iteration_index": 2,
                "model_name": "tiny",
                "model_revision": "revision-a",
            },
        )


def test_checkpoint_identity_is_validated_before_broadcast() -> None:
    runtime = object.__new__(FullWeightDistributedRuntime)
    runtime.rank = 0

    with pytest.raises(ValueError, match="must contain exactly"):
        runtime.create_trainer(
            SimpleNamespace(device="cpu"),
            game_vocab=["game-a"],
            checkpoint_dir=Path("checkpoint"),
            expected_checkpoint={"iteration_index": 2},
        )


def _runtime_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    checkpoint_root: str,
) -> None:
    os.environ["SCRATCH_DIR"] = f"{checkpoint_root}_scratch"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        runtime = FullWeightDistributedRuntime(trainer_factory=_Trainer)
        if rank == 0:
            write_root = runtime._checkpoint_write_root(Path(checkpoint_root))
            (write_root / "full_weight_iter_000002").mkdir(parents=True)
        dist.barrier()
        sent_names: list[str] = []
        if rank == 0:
            trainer = runtime.create_trainer(
                SimpleNamespace(device="cpu"),
                game_vocab=["game-a"],
                checkpoint_identity={
                    "environment": "arc2",
                    "processing_stack": "pinned",
                },
                grpo_group_size=8,
                grpo_puzzles_per_window=4,
            )
            assert trainer.game_vocab == ["game-a"]
            result = runtime.train_window(
                [torch.ones(2, 3)],
                output_dir=Path(checkpoint_root),
                iteration_index=2,
                sample_budget_fraction=0.75,
                policy_lag_windows=1,
                entropy_coef=0.01,
            )
            assert result["iteration_index"] == 2
            manifest = runtime.publication_manifest()
            assert manifest.names == (
                "decoder.bias",
                "decoder.weight",
                "lm_head.weight",
            )

            def send(parameters) -> None:
                for _name in manifest.names:
                    name, _tensor = next(parameters)
                    sent_names.append(name)

            runtime.publish(send)
            assert sent_names == [
                "decoder.bias",
                "decoder.weight",
                "lm_head.weight",
            ]
            checkpoint_dir = runtime.checkpoint(
                root=Path(checkpoint_root),
                name="full_weight_iter_000002",
                iteration_index=2,
                metadata_fields={
                    "model_name": "tiny",
                    "model_revision": "revision-a",
                },
            ).result()
            assert checkpoint_dir.is_dir()
            assert checkpoint_dir.parent == Path(checkpoint_root)
            runtime.close()
            with pytest.raises(RuntimeError, match="runtime is closed"):
                runtime.publication_manifest()
        else:
            runtime.serve()
            with pytest.raises(RuntimeError, match="runtime is closed"):
                runtime.serve()

        random_values: list[tuple[float, float] | None] = [None] * world_size
        dist.all_gather_object(random_values, runtime.trainer.last_random)
        assert all(value == random_values[0] for value in random_values)
        assert runtime.trainer.optimizer_step_count == 1
        assert runtime.trainer._trust_region_lr_scale == 0.25
        assert runtime.trainer._consecutive_window_rejections == 3
        assert torch.equal(runtime.trainer.state_critic.value, torch.tensor(1.0))

        checkpoint_rng = runtime._capture_rng_state()
        random.random()
        torch.rand(())
        original_trainer = runtime.trainer
        runtime = FullWeightDistributedRuntime(trainer_factory=_Trainer)
        checkpoint_dir = Path(checkpoint_root) / "full_weight_iter_000002"
        staged_reads = []
        original_load = checkpoint_module.dcp.load

        def local_load(*args, **kwargs):
            read_path = Path(kwargs["checkpoint_id"])
            assert read_path.is_relative_to(
                runtime._checkpoint_write_root(Path(checkpoint_root))
            )
            staged_reads.append(read_path)
            return original_load(*args, **kwargs)

        checkpoint_module.dcp.load = local_load
        if rank == 0:
            runtime.create_trainer(
                SimpleNamespace(device="cpu"),
                game_vocab=["game-a"],
                checkpoint_identity={
                    "environment": "arc2",
                    "processing_stack": "pinned",
                },
                grpo_group_size=8,
                grpo_puzzles_per_window=4,
                checkpoint_dir=checkpoint_dir,
                expected_checkpoint={
                    "iteration_index": 2,
                    "model_name": "tiny",
                    "model_revision": "revision-a",
                },
            )
            runtime.close()
        else:
            runtime.serve()
        checkpoint_module.dcp.load = original_load
        assert len(staged_reads) == 1
        assert not staged_reads[0].parent.parent.exists()
        assert checkpoint_dir.is_dir()
        assert runtime.trainer.optimizer_step_count == 1
        assert runtime.trainer._trust_region_lr_scale == 0.25
        assert runtime.trainer._consecutive_window_rejections == 3
        assert torch.equal(runtime.trainer.state_critic.value, torch.tensor(1.0))
        for original, restored in zip(
            original_trainer.model.parameters(),
            runtime.trainer.model.parameters(),
            strict=True,
        ):
            original_local = (
                original.to_local() if hasattr(original, "to_local") else original
            )
            restored_local = (
                restored.to_local() if hasattr(restored, "to_local") else restored
            )
            assert torch.equal(original_local, restored_local)

        restored_rng = runtime._capture_rng_state()
        assert restored_rng.keys() == checkpoint_rng.keys()
        assert restored_rng["python"] == checkpoint_rng["python"]
        assert torch.equal(restored_rng["torch_cpu"], checkpoint_rng["torch_cpu"])
        if "torch_cuda" in checkpoint_rng:
            assert torch.equal(restored_rng["torch_cuda"], checkpoint_rng["torch_cuda"])

        followup_kwargs = {
            "output_dir": Path(checkpoint_root),
            "iteration_index": 3,
            "sample_budget_fraction": 0.75,
            "policy_lag_windows": 1,
            "entropy_coef": 0.01,
            "next_window_ready": None,
            "next_window_seconds_remaining": None,
        }
        runtime._restore_rng_state(checkpoint_rng)
        original_trainer.train_window([torch.full((2, 3), 0.5)], **followup_kwargs)
        runtime._restore_rng_state(checkpoint_rng)
        runtime.trainer.train_window([torch.full((2, 3), 0.5)], **followup_kwargs)
        assert original_trainer.last_random == runtime.trainer.last_random
        for original, restored in zip(
            original_trainer.model.parameters(),
            runtime.trainer.model.parameters(),
            strict=True,
        ):
            original_local = (
                original.to_local() if hasattr(original, "to_local") else original
            )
            restored_local = (
                restored.to_local() if hasattr(restored, "to_local") else restored
            )
            torch.testing.assert_close(original_local, restored_local)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _incomplete_publication_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        runtime = FullWeightDistributedRuntime(trainer_factory=_Trainer)
        if rank == 0:
            runtime.create_trainer(
                SimpleNamespace(device="cpu"),
                game_vocab=["game-a"],
            )
            runtime.publication_manifest()

            def consume_only_one(parameters) -> None:
                next(parameters)

            with pytest.raises(RuntimeError, match="without exhausting"):
                runtime.publish(consume_only_one)
        else:
            with pytest.raises(RuntimeError, match="without exhausting"):
                runtime.serve()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_runtime_requires_distributed_initialization() -> None:
    if dist.is_initialized():
        pytest.skip("test requires no ambient process group")
    with pytest.raises(RuntimeError, match="requires torch.distributed"):
        FullWeightDistributedRuntime()


def test_checkpoint_write_root_uses_job_local_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable_root = tmp_path / "durable" / "run-a" / "checkpoints"
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("SCRATCH_DIR", str(scratch))

    write_root = FullWeightDistributedRuntime._checkpoint_write_root(durable_root)

    assert write_root.parent == scratch / "full_weight_checkpoints"
    assert write_root.name.startswith("run-a-")
    assert write_root != durable_root

    monkeypatch.delenv("SCRATCH_DIR")
    assert (
        FullWeightDistributedRuntime._checkpoint_write_root(durable_root)
        == durable_root
    )


def test_two_rank_command_stream_trains_publishes_checkpoints_and_resumes(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _runtime_worker,
        args=(
            2,
            str(tmp_path / "runtime_init"),
            str(tmp_path / "checkpoints"),
        ),
        nprocs=2,
        join=True,
    )


def test_incomplete_rank_zero_publication_is_drained_and_rejected(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _incomplete_publication_worker,
        args=(2, str(tmp_path / "incomplete_init")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    "field", ["checkpoint_identity", "grpo_group_size", "grpo_puzzles_per_window"]
)
def test_arc2_checkpoint_identity_must_match(field: str) -> None:
    runtime = object.__new__(FullWeightDistributedRuntime)
    expected = {
        "iteration_index": 2,
        "model_name": "tiny",
        "model_revision": "revision-a",
        "checkpoint_identity": {
            "dataset_manifest": "current",
            "processing_stack": "pinned",
        },
        "grpo_group_size": 8,
        "grpo_puzzles_per_window": 4,
    }
    fields = {
        name: value for name, value in expected.items() if name != "iteration_index"
    }
    fields[field] = None
    metadata = FullWeightCheckpointMetadata(
        name="full_weight_iter_000002",
        optimizer_step=1,
        trainable_manifest=(),
        trainer_payload={"iteration_index": 2},
        metadata_fields=fields,
    )
    with pytest.raises(RuntimeError, match=field):
        runtime._restore_checkpoint_payload(metadata, expected=expected)


class _CallbackTrainer:
    def __init__(self, *, config, game_vocab, memory_logger, **identity_kwargs):
        self.identity_kwargs = identity_kwargs
        self.result = None

    def train_window(self, rollouts, **kwargs):
        kwargs["check_health"]()
        self.result = (
            kwargs["next_window_ready"](),
            kwargs["next_window_seconds_remaining"](),
            kwargs["next_window_ready"](),
        )
        kwargs["check_health"]()
        return self.result


def _callback_worker(rank, world_size, rendezvous_path, fail):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        runtime = FullWeightDistributedRuntime(trainer_factory=_CallbackTrainer)
        identity = {"environment": "arc2", "advantage_estimator": "grpo"}
        if rank == 0:
            trainer = runtime.create_trainer(
                SimpleNamespace(device="cpu"),
                game_vocab=["puzzle"],
                checkpoint_identity=identity,
                grpo_group_size=8,
                grpo_puzzles_per_window=4,
            )
            assert trainer.identity_kwargs == {
                "checkpoint_identity": identity,
                "grpo_group_size": 8,
                "grpo_puzzles_per_window": 4,
            }
            calls = []

            def check_health():
                calls.append("health")
                if fail:
                    raise ValueError("collector failed")

            ready = iter((False, True))
            kwargs = dict(
                output_dir=Path("unused"),
                iteration_index=0,
                sample_budget_fraction=1.0,
                policy_lag_windows=0,
                entropy_coef=0.0,
                check_health=check_health,
                next_window_ready=lambda: next(ready),
                next_window_seconds_remaining=lambda: 42.5,
            )
            if fail:
                with pytest.raises(
                    RuntimeError,
                    match="check_health failed: ValueError: collector failed",
                ):
                    runtime.train_window([], **kwargs)
            else:
                assert runtime.train_window([], **kwargs) == (False, 42.5, True)
                assert calls == ["health", "health"]
                runtime.close()
        elif fail:
            with pytest.raises(
                RuntimeError, match="check_health failed: ValueError: collector failed"
            ):
                runtime.serve()
        else:
            runtime.serve()
            assert runtime.trainer.result == (False, 42.5, True)
        assert runtime.trainer.identity_kwargs["checkpoint_identity"] == identity
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail", [False, True])
def test_window_callbacks_run_on_rank_zero_and_broadcast_outcomes(tmp_path, fail):
    spawn_ranks(_callback_worker, 2, str(tmp_path / "callbacks"), fail)
