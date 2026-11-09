from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import training.full_weight_checkpoint as checkpoint_module
from training.full_weight_checkpoint import (
    FullWeightCheckpointer,
    TrainableParameter,
    trainable_parameter_manifest,
    upload_checkpoint_async,
)


def _model() -> torch.nn.Sequential:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 2),
    )
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    return model


def _optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.05,
    )


def _train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()


class _MixedPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 4),
        )
        self.lm_head = torch.nn.Linear(4, 2)
        self.vision = torch.nn.Linear(3, 3)
        for parameter in self.vision.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.decoder(inputs))


def _mixed_optimizer(model: _MixedPolicy) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": list(model.decoder.parameters())},
            {"params": list(model.lm_head.parameters())},
        ],
        lr=0.05,
        foreach=False,
    )


def _distributed_round_trip_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    checkpoint_root: str,
) -> None:
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
    )
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.tensor import DTensor

        def distributed_step(
            current_model: _MixedPolicy,
            current_optimizer: torch.optim.Optimizer,
            inputs: torch.Tensor,
        ) -> None:
            current_optimizer.zero_grad(set_to_none=True)
            current_model(inputs).square().sum().backward()
            for parameter in current_model.lm_head.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world_size)
            current_optimizer.step()

        def local_tensor(value: torch.Tensor) -> torch.Tensor:
            return value.to_local() if isinstance(value, DTensor) else value

        def assert_optimizer_state_equal(
            expected_optimizer: torch.optim.Optimizer,
            actual_optimizer: torch.optim.Optimizer,
        ) -> None:
            expected_state = expected_optimizer.state_dict()
            actual_state = actual_optimizer.state_dict()
            assert expected_state["param_groups"] == actual_state["param_groups"]
            assert expected_state["state"].keys() == actual_state["state"].keys()
            for parameter_id, parameter_state in expected_state["state"].items():
                actual_parameter_state = actual_state["state"][parameter_id]
                assert parameter_state.keys() == actual_parameter_state.keys()
                for name, expected_value in parameter_state.items():
                    actual_value = actual_parameter_state[name]
                    if isinstance(expected_value, torch.Tensor):
                        assert isinstance(actual_value, torch.Tensor)
                        assert torch.equal(
                            local_tensor(expected_value), local_tensor(actual_value)
                        )
                    else:
                        assert actual_value == expected_value

        torch.manual_seed(123)
        model = _MixedPolicy()
        mesh = init_device_mesh("cpu", (world_size,))
        fully_shard(model.decoder, mesh=mesh)
        optimizer = _mixed_optimizer(model)
        distributed_step(model, optimizer, torch.ones(2, 3))
        expected = {
            name: (
                parameter.to_local().detach().clone()
                if isinstance(parameter, DTensor)
                else parameter.detach().clone()
            )
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

        checkpointer = FullWeightCheckpointer(Path(checkpoint_root))
        with pytest.raises(RuntimeError, match="operation descriptor"):
            FullWeightCheckpointer(Path(checkpoint_root) / f"rank_{rank}").save(
                "divergent_root",
                model=model,
                optimizer=optimizer,
                optimizer_step=1,
                trainer_payload={},
            )
        with pytest.raises(RuntimeError, match="preflight failed"):
            checkpointer.save(
                "valid_name" if rank == 0 else "../invalid_name",
                model=model,
                optimizer=optimizer,
                optimizer_step=1,
                trainer_payload={},
            )

        original_get_state_dict = checkpoint_module.get_state_dict

        def fail_get_state_dict(*args, **kwargs) -> None:
            raise RuntimeError("injected get_state_dict failure")

        if rank == 1:
            checkpoint_module.get_state_dict = fail_get_state_dict
        try:
            with pytest.raises(
                RuntimeError, match="save state-dict preparation failed: rank 1"
            ):
                checkpointer.save(
                    "get_state_failure",
                    model=model,
                    optimizer=optimizer,
                    optimizer_step=1,
                    trainer_payload={},
                )
        finally:
            checkpoint_module.get_state_dict = original_get_state_dict

        checkpoint_dir = checkpointer.save(
            "full_weight_iter_000002",
            model=model,
            optimizer=optimizer,
            optimizer_step=2,
            trainer_payload={"saved_by_rank": rank},
        )

        torch.manual_seed(999)
        restored = _MixedPolicy()
        fully_shard(restored.decoder, mesh=mesh)
        restored_optimizer = _mixed_optimizer(restored)

        with pytest.raises(RuntimeError, match="operation descriptor"):
            checkpointer.load(
                checkpoint_dir
                if rank == 0
                else checkpoint_dir.parent / "different_checkpoint",
                model=restored,
                optimizer=restored_optimizer,
            )

        staging_root = Path(checkpoint_root).parent / "restore_scratch"
        with pytest.raises(RuntimeError, match="operation descriptor"):
            checkpointer.load(
                checkpoint_dir,
                model=restored,
                optimizer=restored_optimizer,
                staging_root=staging_root if rank == 0 else None,
            )
        assert not staging_root.exists()
        source_files = {
            path.relative_to(checkpoint_dir): path.read_bytes()
            for path in checkpoint_dir.rglob("*")
            if path.is_file()
        }
        original_copy_tree = checkpoint_module._copy_checkpoint_tree
        original_copy_file = checkpoint_module._copy_checkpoint_file
        original_dcp_load = checkpoint_module.dcp.load
        copies = []
        read_paths = []

        def observed_copy(*args, **kwargs):
            assert rank == 0
            copies.append(args[1])
            return original_copy_tree(*args, **kwargs)

        def fail_copy(*args, **kwargs):
            raise OSError("injected restore copy failure")

        def forbidden_load(*args, **kwargs):
            pytest.fail("DCP load entered after stage-in failure")

        def observed_load(*args, **kwargs):
            read_path = Path(kwargs["checkpoint_id"])
            assert read_path.is_relative_to(staging_root)
            assert read_path.parent.name == checkpoint_dir.name
            assert read_path.is_dir()
            read_paths.append(str(read_path))
            return original_dcp_load(*args, **kwargs)

        checkpoint_module._copy_checkpoint_tree = observed_copy
        checkpoint_module._copy_checkpoint_file = fail_copy
        checkpoint_module.dcp.load = forbidden_load
        try:
            with pytest.raises(RuntimeError, match="injected restore copy failure"):
                checkpointer.load(
                    checkpoint_dir,
                    model=restored,
                    optimizer=restored_optimizer,
                    staging_root=staging_root,
                )
        finally:
            checkpoint_module._copy_checkpoint_file = original_copy_file
            checkpoint_module.dcp.load = observed_load
        assert not list(staging_root.iterdir())

        original_set_state_dict = checkpoint_module.set_state_dict

        def fail_set_state_dict(*args, **kwargs) -> None:
            raise RuntimeError("injected set_state_dict failure")

        if rank == 1:
            checkpoint_module.set_state_dict = fail_set_state_dict
        try:
            with pytest.raises(
                RuntimeError, match="load state-dict application failed: rank 1"
            ):
                checkpointer.load(
                    checkpoint_dir,
                    model=restored,
                    optimizer=restored_optimizer,
                    staging_root=staging_root,
                )
        finally:
            checkpoint_module.set_state_dict = original_set_state_dict
        assert not list(staging_root.iterdir())

        metadata = checkpointer.load(
            checkpoint_dir,
            model=restored,
            optimizer=restored_optimizer,
            staging_root=staging_root,
        )
        checkpoint_module.dcp.load = original_dcp_load
        checkpoint_module._copy_checkpoint_tree = original_copy_tree
        assert len(copies) == (3 if rank == 0 else 0)
        paths_by_rank = [None] * world_size
        dist.all_gather_object(paths_by_rank, read_paths)
        assert all(paths == read_paths for paths in paths_by_rank)
        assert len(read_paths) == 2
        assert not list(staging_root.iterdir())
        assert source_files == {
            path.relative_to(checkpoint_dir): path.read_bytes()
            for path in checkpoint_dir.rglob("*")
            if path.is_file()
        }
        assert checkpointer.latest.resolve() == checkpoint_dir.resolve()
        assert metadata.name == checkpoint_dir.name
        for name, parameter in restored.named_parameters():
            if parameter.requires_grad:
                actual = (
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                )
                assert torch.equal(actual, expected[name])
        assert_optimizer_state_equal(optimizer, restored_optimizer)

        followup_inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 5.0
        distributed_step(model, optimizer, followup_inputs)
        distributed_step(restored, restored_optimizer, followup_inputs)
        for (source_name, source_parameter), (restored_name, restored_parameter) in zip(
            model.named_parameters(), restored.named_parameters(), strict=True
        ):
            assert source_name == restored_name
            if source_parameter.requires_grad:
                assert torch.equal(
                    local_tensor(source_parameter), local_tensor(restored_parameter)
                )
        assert metadata.optimizer_step == 2
        assert metadata.trainer_payload == {"saved_by_rank": 0}
    finally:
        dist.destroy_process_group()


def test_trainable_manifest_is_canonical_and_omits_frozen_parameters() -> None:
    manifest = trainable_parameter_manifest(_model())

    assert manifest == (
        TrainableParameter("1.bias", (2,), "float32"),
        TrainableParameter("1.weight", (2, 4), "float32"),
    )


def test_checkpoint_round_trip_restores_trainable_model_optimizer_and_payload(
    tmp_path: Path,
) -> None:
    torch.manual_seed(123)
    source = _model()
    source_optimizer = _optimizer(source)
    _train_step(source, source_optimizer)
    expected_trainable = {
        name: parameter.detach().clone()
        for name, parameter in source.named_parameters()
        if parameter.requires_grad
    }

    checkpointer = FullWeightCheckpointer(tmp_path)
    checkpoint_dir = checkpointer.save(
        "full_weight_iter_000007",
        model=source,
        optimizer=source_optimizer,
        optimizer_step=19,
        trainer_payload={
            "python_rng": (3, (1, 2, 3), None),
            "critic": {"weight": torch.tensor([2.0])},
        },
        metadata_fields={
            "iteration": 7,
            "model_name": "Qwen/test",
            "model_revision": "abc123",
        },
    )

    header = json.loads((checkpoint_dir / "metadata.json").read_text())
    assert header["optimizer_step"] == 19
    assert header["metadata"] == {
        "iteration": 7,
        "model_name": "Qwen/test",
        "model_revision": "abc123",
    }
    assert header["trainer_payload_file"] == "trainer_payload.pt"
    assert checkpointer.latest.is_symlink()
    assert checkpointer.latest.resolve() == checkpoint_dir.resolve()

    torch.manual_seed(999)
    restored = _model()
    restored_optimizer = _optimizer(restored)
    frozen_before_load = restored[0].weight.detach().clone()
    metadata = checkpointer.load(
        checkpoint_dir,
        model=restored,
        optimizer=restored_optimizer,
    )

    for name, parameter in restored.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter, expected_trainable[name])
    assert torch.equal(restored[0].weight, frozen_before_load)
    assert metadata.name == "full_weight_iter_000007"
    assert metadata.optimizer_step == 19
    assert metadata.metadata_fields["iteration"] == 7
    assert metadata.trainer_payload["python_rng"] == (3, (1, 2, 3), None)
    assert torch.equal(
        metadata.trainer_payload["critic"]["weight"], torch.tensor([2.0])
    )

    source_state = source_optimizer.state_dict()
    restored_state = restored_optimizer.state_dict()
    assert source_state["param_groups"] == restored_state["param_groups"]
    assert source_state["state"].keys() == restored_state["state"].keys()
    for parameter_id, state in source_state["state"].items():
        for key, value in state.items():
            assert torch.equal(value, restored_state["state"][parameter_id][key])


def test_checkpoint_uses_parallel_file_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_writer = checkpoint_module.dcp.FileSystemWriter
    writer_options: dict[str, object] = {}

    def capture_writer(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        writer_options.update(kwargs)
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.dcp, "FileSystemWriter", capture_writer)
    model = _model()
    FullWeightCheckpointer(tmp_path).save(
        "full_weight_iter_000001",
        model=model,
        optimizer=_optimizer(model),
        optimizer_step=0,
        trainer_payload={},
    )

    assert writer_options == {
        "single_file_per_rank": True,
        "sync_files": True,
        "thread_count": 4,
        "per_thread_copy_ahead": 64 * 1024 * 1024,
        "overwrite": True,
    }


def test_checkpoint_fsyncs_staged_directory_tree_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[Path] = []
    original_fsync_directory = checkpoint_module._fsync_directory

    def record_fsync(path: Path) -> None:
        fsync_calls.append(path)
        original_fsync_directory(path)

    monkeypatch.setattr(checkpoint_module, "_fsync_directory", record_fsync)
    model = _model()
    FullWeightCheckpointer(tmp_path).save(
        "full_weight_iter_000001",
        model=model,
        optimizer=_optimizer(model),
        optimizer_step=0,
        trainer_payload={},
    )

    staging = tmp_path / ".full_weight_iter_000001.staging"
    distributed_state = staging / "distributed_state"
    assert fsync_calls.index(distributed_state) < fsync_calls.index(staging)
    assert fsync_calls.index(staging) < fsync_calls.index(tmp_path)


def test_node_local_checkpoint_upload_is_atomic_and_parallel(tmp_path: Path) -> None:
    local_checkpoint = tmp_path / "scratch" / "full_weight_iter_000002"
    (local_checkpoint / "distributed_state").mkdir(parents=True)
    (local_checkpoint / "distributed_state" / "__0_0.distcp").write_bytes(b"a" * 64)
    (local_checkpoint / "distributed_state" / "__0_1.distcp").write_bytes(b"b" * 32)
    (local_checkpoint / "metadata.json").write_text("metadata")
    durable_root = tmp_path / "durable"

    checkpoint_dir = upload_checkpoint_async(
        local_checkpoint,
        durable_root=durable_root,
        checkpoint_name="full_weight_iter_000002",
    ).result(timeout=5)

    assert not local_checkpoint.exists()
    assert (
        checkpoint_dir / "distributed_state" / "__0_0.distcp"
    ).read_bytes() == b"a" * 64
    assert (
        checkpoint_dir / "distributed_state" / "__0_1.distcp"
    ).read_bytes() == b"b" * 32
    assert (checkpoint_dir / "metadata.json").read_text() == "metadata"
    assert (durable_root / "latest").resolve() == checkpoint_dir.resolve()
    assert not (durable_root / ".full_weight_iter_000002.uploading").exists()


def test_failed_node_local_upload_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable_root = tmp_path / "durable"
    previous = durable_root / "full_weight_iter_000000"
    previous.mkdir(parents=True)
    (durable_root / "latest").symlink_to(previous.name, target_is_directory=True)
    local_checkpoint = tmp_path / "scratch" / "full_weight_iter_000001"
    local_checkpoint.mkdir(parents=True)
    (local_checkpoint / "data").write_bytes(b"checkpoint")

    def fail_copy(_source: Path, _destination: Path) -> None:
        raise OSError("injected upload failure")

    monkeypatch.setattr(checkpoint_module, "_copy_checkpoint_file", fail_copy)
    future = upload_checkpoint_async(
        local_checkpoint,
        durable_root=durable_root,
        checkpoint_name="full_weight_iter_000001",
    )

    with pytest.raises(RuntimeError, match="injected upload failure"):
        future.result(timeout=5)
    assert local_checkpoint.is_dir()
    assert not (durable_root / "full_weight_iter_000001").exists()
    assert (durable_root / "latest").resolve() == previous.resolve()


def test_publish_fsync_failure_rolls_back_checkpoint_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable_root = tmp_path / "durable"
    previous = durable_root / "full_weight_iter_000000"
    previous.mkdir(parents=True)
    (durable_root / "latest").symlink_to(previous.name, target_is_directory=True)
    local_checkpoint = tmp_path / "scratch" / "full_weight_iter_000001"
    local_checkpoint.mkdir(parents=True)
    (local_checkpoint / "data").write_bytes(b"checkpoint")
    original_fsync_directory = checkpoint_module._fsync_directory
    durable_fsync_calls = 0

    def fail_first_publish_fsync(path: Path) -> None:
        nonlocal durable_fsync_calls
        if path == durable_root:
            durable_fsync_calls += 1
            if durable_fsync_calls == 1:
                raise OSError("injected publish fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        checkpoint_module,
        "_fsync_directory",
        fail_first_publish_fsync,
    )
    future = upload_checkpoint_async(
        local_checkpoint,
        durable_root=durable_root,
        checkpoint_name="full_weight_iter_000001",
    )

    with pytest.raises(OSError, match="injected publish fsync failure"):
        future.result(timeout=5)
    assert local_checkpoint.is_dir()
    assert not (durable_root / "full_weight_iter_000001").exists()
    assert (durable_root / "latest").resolve() == previous.resolve()


def test_latest_fsync_failure_restores_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    durable_root = tmp_path / "durable"
    previous = durable_root / "full_weight_iter_000000"
    previous.mkdir(parents=True)
    (durable_root / "latest").symlink_to(previous.name, target_is_directory=True)
    local_checkpoint = tmp_path / "scratch" / "full_weight_iter_000001"
    local_checkpoint.mkdir(parents=True)
    (local_checkpoint / "data").write_bytes(b"checkpoint")
    original_fsync_directory = checkpoint_module._fsync_directory
    durable_fsync_calls = 0

    def fail_after_latest_replace(path: Path) -> None:
        nonlocal durable_fsync_calls
        if path == durable_root:
            durable_fsync_calls += 1
            if durable_fsync_calls == 2:
                raise OSError("injected latest fsync failure")
        original_fsync_directory(path)

    monkeypatch.setattr(
        checkpoint_module,
        "_fsync_directory",
        fail_after_latest_replace,
    )
    future = upload_checkpoint_async(
        local_checkpoint,
        durable_root=durable_root,
        checkpoint_name="full_weight_iter_000001",
    )

    with pytest.raises(OSError, match="injected latest fsync failure"):
        future.result(timeout=5)
    assert local_checkpoint.is_dir()
    assert not (durable_root / "full_weight_iter_000001").exists()
    assert (durable_root / "latest").resolve() == previous.resolve()


def test_trainer_payload_uses_weights_only_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    checkpointer = FullWeightCheckpointer(tmp_path)
    checkpoint_dir = checkpointer.save(
        "full_weight_iter_000001",
        model=model,
        optimizer=_optimizer(model),
        optimizer_step=0,
        trainer_payload={"rng": (1, 2, 3), "tensor": torch.tensor([4.0])},
    )
    original_load = torch.load
    weights_only_values: list[object] = []

    def checked_load(*args, **kwargs) -> object:
        weights_only_values.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.torch, "load", checked_load)

    _, payload = checkpointer._read_metadata(checkpoint_dir)

    assert weights_only_values == [True]
    assert payload["rng"] == (1, 2, 3)
    assert torch.equal(payload["tensor"], torch.tensor([4.0]))


def test_unsafe_trainer_payload_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    model = _model()
    optimizer = _optimizer(model)
    checkpointer = FullWeightCheckpointer(tmp_path)
    prior_checkpoint = checkpointer.save(
        "full_weight_iter_000000",
        model=model,
        optimizer=optimizer,
        optimizer_step=0,
        trainer_payload={},
    )

    with pytest.raises(ValueError, match="weights_only=True"):
        checkpointer.save(
            "full_weight_iter_000001",
            model=model,
            optimizer=optimizer,
            optimizer_step=0,
            trainer_payload={"unsupported": tmp_path},
        )

    assert not (tmp_path / "full_weight_iter_000001").exists()
    assert checkpointer.latest.resolve() == prior_checkpoint.resolve()


def test_checkpoint_is_not_published_when_distributed_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    optimizer = _optimizer(model)
    checkpointer = FullWeightCheckpointer(tmp_path)

    def fail_save(*args, **kwargs) -> None:
        raise RuntimeError("injected DCP failure")

    monkeypatch.setattr(checkpoint_module.dcp, "save", fail_save)

    with pytest.raises(RuntimeError, match="injected DCP failure"):
        checkpointer.save(
            "full_weight_iter_000001",
            model=model,
            optimizer=optimizer,
            optimizer_step=0,
            trainer_payload={},
        )

    assert not (tmp_path / "full_weight_iter_000001").exists()
    assert not checkpointer.latest.exists()
    assert not (
        tmp_path / ".full_weight_iter_000001.staging" / "metadata.json"
    ).exists()


def test_checkpoint_is_not_published_when_metadata_cannot_commit(
    tmp_path: Path,
) -> None:
    model = _model()
    checkpointer = FullWeightCheckpointer(tmp_path)

    with pytest.raises(TypeError, match="JSON serializable"):
        checkpointer.save(
            "full_weight_iter_000001",
            model=model,
            optimizer=_optimizer(model),
            optimizer_step=0,
            trainer_payload={},
            metadata_fields={"invalid": object()},
        )

    assert not (tmp_path / "full_weight_iter_000001").exists()
    assert not checkpointer.latest.exists()
    assert not (
        tmp_path / ".full_weight_iter_000001.staging" / "metadata.json"
    ).exists()


def test_latest_link_failure_rolls_back_publication_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    optimizer = _optimizer(model)
    checkpointer = FullWeightCheckpointer(tmp_path)
    prior_checkpoint = checkpointer.save(
        "full_weight_iter_000000",
        model=model,
        optimizer=optimizer,
        optimizer_step=0,
        trainer_payload={},
    )
    original_replace_latest_link = checkpointer._replace_latest_link

    def fail_latest_link(_checkpoint_dir: Path) -> None:
        raise OSError("injected latest-link failure")

    monkeypatch.setattr(checkpointer, "_replace_latest_link", fail_latest_link)
    with pytest.raises(OSError, match="injected latest-link failure"):
        checkpointer.save(
            "full_weight_iter_000001",
            model=model,
            optimizer=optimizer,
            optimizer_step=0,
            trainer_payload={},
        )

    checkpoint_dir = tmp_path / "full_weight_iter_000001"
    assert not checkpoint_dir.exists()
    assert checkpointer.latest.resolve() == prior_checkpoint.resolve()

    monkeypatch.setattr(
        checkpointer, "_replace_latest_link", original_replace_latest_link
    )
    assert (
        checkpointer.save(
            "full_weight_iter_000001",
            model=model,
            optimizer=optimizer,
            optimizer_step=0,
            trainer_payload={},
        )
        == checkpoint_dir
    )
    assert checkpointer.latest.resolve() == checkpoint_dir.resolve()


def test_load_rejects_manifest_mismatch_before_reading_distributed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    optimizer = _optimizer(model)
    checkpointer = FullWeightCheckpointer(tmp_path)
    checkpoint_dir = checkpointer.save(
        "full_weight_iter_000001",
        model=model,
        optimizer=optimizer,
        optimizer_step=0,
        trainer_payload={},
    )

    mismatched_model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 3),
    )
    for parameter in mismatched_model[0].parameters():
        parameter.requires_grad_(False)
    mismatched_optimizer = _optimizer(mismatched_model)

    def unexpected_load(*args, **kwargs) -> None:
        raise AssertionError("DCP load must not run for a mismatched model")

    monkeypatch.setattr(checkpoint_module.dcp, "load", unexpected_load)
    with pytest.raises(RuntimeError, match="manifest does not match"):
        checkpointer.load(
            checkpoint_dir,
            model=mismatched_model,
            optimizer=mismatched_optimizer,
        )


def test_load_rejects_corrupted_manifest_hash(tmp_path: Path) -> None:
    model = _model()
    checkpointer = FullWeightCheckpointer(tmp_path)
    checkpoint_dir = checkpointer.save(
        "full_weight_iter_000001",
        model=model,
        optimizer=_optimizer(model),
        optimizer_step=0,
        trainer_payload={},
    )
    metadata_path = checkpoint_dir / "metadata.json"
    header = json.loads(metadata_path.read_text())
    header["trainable_manifest"][0]["dtype"] = "float16"
    metadata_path.write_text(json.dumps(header))

    restored = _model()
    with pytest.raises(RuntimeError, match="manifest hash does not match"):
        checkpointer.load(
            checkpoint_dir,
            model=restored,
            optimizer=_optimizer(restored),
        )


def test_load_rejects_checkpoint_without_committed_metadata(tmp_path: Path) -> None:
    incomplete = tmp_path / "full_weight_iter_000001"
    (incomplete / "distributed_state").mkdir(parents=True)
    model = _model()

    with pytest.raises(RuntimeError, match="incomplete"):
        FullWeightCheckpointer(tmp_path).load(
            incomplete,
            model=model,
            optimizer=_optimizer(model),
        )


def test_restore_rejects_staging_inside_source_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "full_weight_iter_000001"
    checkpoint_dir.mkdir()
    model = _model()
    with pytest.raises(ValueError, match="outside its checkpoint"):
        FullWeightCheckpointer(tmp_path).load(
            checkpoint_dir,
            model=model,
            optimizer=_optimizer(model),
            staging_root=checkpoint_dir / "scratch",
        )
    assert not list(checkpoint_dir.iterdir())


@pytest.mark.integration
def test_two_rank_mixed_fsdp2_gloo_checkpoint_round_trip(tmp_path: Path) -> None:
    world_size = 2
    torch.multiprocessing.spawn(
        _distributed_round_trip_worker,
        args=(
            world_size,
            str(tmp_path / "gloo-rendezvous"),
            str(tmp_path / "checkpoints"),
        ),
        nprocs=world_size,
        join=True,
    )
