from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)


_CHECKPOINT_FORMAT_VERSION = 1
_DISTRIBUTED_STATE_DIR = "distributed_state"
_METADATA_FILENAME = "metadata.json"
_TRAINER_PAYLOAD_FILENAME = "trainer_payload.pt"
_LATEST_LINK_NAME = "latest"
_WRITER_THREAD_COUNT = 4
_WRITER_COPY_AHEAD_BYTES = 64 * 1024 * 1024
_UPLOAD_THREAD_COUNT = 4
_T = TypeVar("_T")
log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainableParameter:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrainableParameter:
        raw_shape = payload.get("shape")
        if not isinstance(raw_shape, list):
            raise RuntimeError("Checkpoint manifest parameter shape must be a list.")
        return cls(
            name=str(payload.get("name", "")),
            shape=tuple(int(size) for size in raw_shape),
            dtype=str(payload.get("dtype", "")),
        )


@dataclass(frozen=True, slots=True)
class FullWeightCheckpointMetadata:
    name: str
    optimizer_step: int
    trainable_manifest: tuple[TrainableParameter, ...]
    trainer_payload: Any
    metadata_fields: dict[str, object]


@dataclass(frozen=True, slots=True)
class _OperationDescriptor:
    operation: str
    root: str
    checkpoint_path: str
    checkpoint_name: str
    staging_root: str | None = None


@dataclass(frozen=True, slots=True)
class _PreflightStatus:
    error: str | None
    descriptor: _OperationDescriptor | None = None
    manifest: tuple[TrainableParameter, ...] = ()


def trainable_parameter_manifest(
    model: torch.nn.Module,
) -> tuple[TrainableParameter, ...]:
    """Return the stable logical inventory persisted by FSDP state-dict APIs."""
    parameters = (
        TrainableParameter(
            name=name,
            shape=tuple(int(size) for size in parameter.shape),
            dtype=str(parameter.dtype).removeprefix("torch."),
        )
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    return tuple(sorted(parameters, key=lambda parameter: parameter.name))


def _manifest_hash(manifest: tuple[TrainableParameter, ...]) -> str:
    serialized = json.dumps(
        [parameter.to_dict() for parameter in manifest],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _state_dict_options() -> StateDictOptions:
    # Frozen tensors are intentionally absent, so model.load_state_dict must
    # tolerate exactly those missing keys when set_state_dict applies the state.
    return StateDictOptions(ignore_frozen_params=True, strict=False)


def _prepare_staging(
    root: Path,
    *,
    checkpoint_dir: Path,
    staging_dir: Path,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.exists() or checkpoint_dir.is_symlink():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_dir}")
    if staging_dir.exists() or staging_dir.is_symlink():
        if staging_dir.is_dir() and not staging_dir.is_symlink():
            shutil.rmtree(staging_dir)
        else:
            staging_dir.unlink()
    staging_dir.mkdir()


def _publish_staging(
    root: Path,
    *,
    checkpoint_dir: Path,
    staging_dir: Path,
    replace_latest: Callable[[Path], None] | None = None,
) -> None:
    update_latest = replace_latest or (
        lambda published: _replace_latest_link(root, published)
    )
    _fsync_directory_tree(staging_dir)
    staging_dir.replace(checkpoint_dir)
    try:
        _fsync_directory(root)
        update_latest(checkpoint_dir)
    except Exception as publish_error:
        try:
            checkpoint_dir.replace(staging_dir)
            _fsync_directory(root)
        except Exception as rollback_error:
            raise RuntimeError(
                "Checkpoint publication and rollback both failed: "
                f"publish={publish_error}; rollback={rollback_error}."
            ) from publish_error
        raise


def upload_checkpoint_async(
    local_checkpoint_dir: Path,
    *,
    durable_root: Path,
    checkpoint_name: str,
) -> Future[Path]:
    """Upload a complete node-local checkpoint and publish it atomically."""
    local_checkpoint_dir = Path(local_checkpoint_dir)
    durable_root = Path(durable_root)
    checkpoint_name = FullWeightCheckpointer._validate_name(checkpoint_name)
    if not local_checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Node-local checkpoint does not exist: {local_checkpoint_dir}"
        )

    checkpoint_dir = durable_root / checkpoint_name
    staging_dir = durable_root / f".{checkpoint_name}.uploading"
    _prepare_staging(
        durable_root,
        checkpoint_dir=checkpoint_dir,
        staging_dir=staging_dir,
    )

    future: Future[Path] = Future()
    log.info(
        "starting background full-weight checkpoint upload: local=%s durable=%s",
        local_checkpoint_dir,
        checkpoint_dir,
    )

    def upload() -> None:
        if not future.set_running_or_notify_cancel():
            return
        started_at = time.perf_counter()
        try:
            copied_bytes = _copy_checkpoint_tree(
                local_checkpoint_dir,
                staging_dir,
                thread_count=_UPLOAD_THREAD_COUNT,
            )
            _publish_staging(
                durable_root,
                checkpoint_dir=checkpoint_dir,
                staging_dir=staging_dir,
            )
            shutil.rmtree(local_checkpoint_dir, ignore_errors=True)
            log.info(
                "uploaded full-weight checkpoint %s to durable storage "
                "(%.1f GiB in %.2fs)",
                checkpoint_name,
                copied_bytes / (1024**3),
                time.perf_counter() - started_at,
            )
            future.set_result(checkpoint_dir)
        except BaseException as exc:  # noqa: BLE001 - relay background upload failure
            future.set_exception(exc)

    threading.Thread(
        target=upload,
        name=f"full-weight-checkpoint-upload-{checkpoint_name}",
        daemon=True,
    ).start()
    return future


def _copy_checkpoint_tree(
    source_root: Path,
    destination_root: Path,
    *,
    thread_count: int,
) -> int:
    files: list[tuple[Path, Path]] = []
    for source_dir, dirnames, filenames in os.walk(source_root):
        source_path = Path(source_dir)
        relative_dir = source_path.relative_to(source_root)
        destination_dir = destination_root / relative_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        for dirname in dirnames:
            (destination_dir / dirname).mkdir(exist_ok=True)
        files.extend(
            (source_path / filename, destination_dir / filename)
            for filename in filenames
        )

    work: queue.Queue[tuple[Path, Path] | None] = queue.Queue()
    for item in files:
        work.put(item)
    worker_count = min(max(1, int(thread_count)), max(1, len(files)))
    for _ in range(worker_count):
        work.put(None)

    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def copy_worker() -> None:
        while True:
            item = work.get()
            try:
                if item is None:
                    return
                source, destination = item
                _copy_checkpoint_file(source, destination)
            except BaseException as exc:  # noqa: BLE001 - join then relay first error
                with errors_lock:
                    errors.append(exc)
            finally:
                work.task_done()

    workers = [
        threading.Thread(
            target=copy_worker,
            name=f"full-weight-checkpoint-copy-{index}",
            daemon=True,
        )
        for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    work.join()
    for worker in workers:
        worker.join()
    if errors:
        raise RuntimeError(f"Checkpoint copy failed: {errors[0]}") from errors[0]
    _fsync_directory_tree(destination_root)
    return sum(source.stat().st_size for source, _destination in files)


def _copy_checkpoint_file(source: Path, destination: Path) -> None:
    with (
        source.open("rb") as source_handle,
        destination.open("wb") as destination_handle,
    ):
        shutil.copyfileobj(source_handle, destination_handle, length=16 * 1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    shutil.copystat(source, destination)


def _replace_latest_link(root: Path, checkpoint_dir: Path) -> None:
    latest = root / _LATEST_LINK_NAME
    staging_link = root / f".{_LATEST_LINK_NAME}.tmp"
    previous_target: str | None = None
    if latest.is_symlink():
        previous_target = os.readlink(latest)
    elif latest.exists():
        raise RuntimeError(f"Checkpoint latest path is not a symlink: {latest}")
    if staging_link.exists() or staging_link.is_symlink():
        staging_link.unlink()
    staging_link.symlink_to(checkpoint_dir.name, target_is_directory=True)
    replaced = False
    try:
        os.replace(staging_link, latest)
        replaced = True
        _fsync_directory(root)
    except Exception as update_error:
        if replaced:
            try:
                if previous_target is None:
                    latest.unlink()
                else:
                    rollback_link = root / f".{_LATEST_LINK_NAME}.rollback"
                    if rollback_link.exists() or rollback_link.is_symlink():
                        rollback_link.unlink()
                    rollback_link.symlink_to(previous_target, target_is_directory=True)
                    os.replace(rollback_link, latest)
                _fsync_directory(root)
            except Exception as rollback_error:
                raise RuntimeError(
                    "Updating the checkpoint latest link and restoring its previous "
                    f"target both failed: update={update_error}; "
                    f"rollback={rollback_error}."
                ) from update_error
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [Path(path) for path, _dirs, _files in os.walk(root)]
    for directory in reversed(directories):
        _fsync_directory(directory)


class FullWeightCheckpointer:
    """Save and restore FSDP2 model and optimizer state with atomic publication."""

    def __init__(self, root: Path, *, process_group: Any | None = None) -> None:
        self.root = Path(root)
        self.process_group = process_group

    def save(
        self,
        name: str,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_step: int,
        trainer_payload: Any,
        metadata_fields: Mapping[str, object] | None = None,
    ) -> Path:
        def prepare_save() -> tuple[
            _OperationDescriptor,
            tuple[TrainableParameter, ...],
            tuple[str, Path, Path, tuple[TrainableParameter, ...]],
        ]:
            checkpoint_name = self._validate_name(name)
            checkpoint_dir = self.root / checkpoint_name
            staging_dir = self.root / f".{checkpoint_name}.staging"
            manifest = trainable_parameter_manifest(model)
            if not manifest:
                raise ValueError(
                    "A full-weight checkpoint requires trainable parameters."
                )
            return (
                self._operation_descriptor(
                    "save",
                    checkpoint_path=checkpoint_dir,
                    checkpoint_name=checkpoint_name,
                ),
                manifest,
                (checkpoint_name, checkpoint_dir, staging_dir, manifest),
            )

        checkpoint_name, checkpoint_dir, staging_dir, manifest = (
            self._collective_preflight("save", prepare_save)
        )
        self._run_on_rank_zero(lambda: self._validate_trainer_payload(trainer_payload))
        self._run_on_rank_zero(
            lambda: _prepare_staging(
                self.root,
                checkpoint_dir=checkpoint_dir,
                staging_dir=staging_dir,
            )
        )

        options = _state_dict_options()
        model_state, optimizer_state = self._run_on_all_ranks(
            "save state-dict preparation",
            lambda: get_state_dict(
                model,
                optimizer,
                options=options,
            ),
        )
        storage_writer = dcp.FileSystemWriter(
            staging_dir / _DISTRIBUTED_STATE_DIR,
            single_file_per_rank=True,
            sync_files=True,
            thread_count=_WRITER_THREAD_COUNT,
            per_thread_copy_ahead=_WRITER_COPY_AHEAD_BYTES,
            overwrite=True,
        )
        save_started_at = time.perf_counter()
        dcp.save(
            {
                "model": model_state,
                "optimizer": optimizer_state,
            },
            storage_writer=storage_writer,
            process_group=self.process_group,
            no_dist=not self._distributed,
        )
        self._barrier()
        if self._rank == 0:
            log.info(
                "saved node-local full-weight checkpoint %s with %d writer "
                "threads per rank in %.2fs",
                checkpoint_name,
                _WRITER_THREAD_COUNT,
                time.perf_counter() - save_started_at,
            )

        fields = {} if metadata_fields is None else metadata_fields
        self._run_on_rank_zero(
            lambda: self._commit_checkpoint(
                checkpoint_dir=checkpoint_dir,
                staging_dir=staging_dir,
                checkpoint_name=checkpoint_name,
                optimizer_step=optimizer_step,
                manifest=manifest,
                trainer_payload=trainer_payload,
                metadata_fields=fields,
            )
        )
        return checkpoint_dir

    def load(
        self,
        checkpoint_dir: Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        staging_root: Path | None = None,
    ) -> FullWeightCheckpointMetadata:
        def prepare_load() -> tuple[
            _OperationDescriptor,
            tuple[TrainableParameter, ...],
            tuple[Path, tuple[TrainableParameter, ...]],
        ]:
            checkpoint_path = Path(checkpoint_dir)
            manifest = trainable_parameter_manifest(model)
            if not manifest:
                raise ValueError(
                    "A full-weight checkpoint requires trainable parameters."
                )
            resolved_path = checkpoint_path.resolve(strict=False)
            return (
                self._operation_descriptor(
                    "load",
                    checkpoint_path=checkpoint_path,
                    checkpoint_name=resolved_path.name,
                    staging_root=staging_root,
                ),
                manifest,
                (resolved_path, manifest),
            )

        checkpoint_path, current_manifest = self._collective_preflight(
            "load", prepare_load
        )
        local_checkpoint: Path | None = None
        if staging_root is not None:

            def stage() -> Path:
                destination_root = Path(staging_root).resolve()
                if destination_root.is_relative_to(checkpoint_path):
                    raise ValueError("Restore staging must be outside its checkpoint.")
                destination_root.mkdir(parents=True, exist_ok=True)
                temporary_root = Path(
                    tempfile.mkdtemp(prefix=".restore-", dir=destination_root)
                )
                destination = temporary_root / checkpoint_path.name
                started_at = time.perf_counter()
                log.info(
                    "staging full-weight checkpoint for restore: %s -> %s",
                    checkpoint_path,
                    destination,
                )
                try:
                    copied_bytes = _copy_checkpoint_tree(
                        checkpoint_path,
                        destination,
                        thread_count=_UPLOAD_THREAD_COUNT,
                    )
                except BaseException:
                    shutil.rmtree(temporary_root)
                    raise
                log.info(
                    "staged full-weight checkpoint for restore (%.1f GiB in %.2fs)",
                    copied_bytes / (1024**3),
                    time.perf_counter() - started_at,
                )
                return destination

            local_checkpoint = self._run_on_rank_zero(stage)
        read_path = local_checkpoint or checkpoint_path
        try:
            header, trainer_payload = self._run_on_rank_zero(
                lambda: self._read_metadata(read_path)
            )
            metadata = self._parse_metadata(header, trainer_payload=trainer_payload)

            if current_manifest != metadata.trainable_manifest:
                raise RuntimeError(
                    "Checkpoint trainable parameter manifest does not match the model: "
                    f"checkpoint={metadata.trainable_manifest!r}, current={current_manifest!r}."
                )

            options = _state_dict_options()
            model_state, optimizer_state = self._run_on_all_ranks(
                "load state-dict preparation",
                lambda: get_state_dict(
                    model,
                    optimizer,
                    options=options,
                ),
            )
            state = {
                "model": model_state,
                "optimizer": optimizer_state,
            }
            dcp.load(
                state,
                checkpoint_id=read_path / _DISTRIBUTED_STATE_DIR,
                process_group=self.process_group,
                no_dist=not self._distributed,
            )
            self._run_on_all_ranks(
                "load state-dict application",
                lambda: set_state_dict(
                    model,
                    optimizer,
                    model_state_dict=state["model"],
                    optim_state_dict=state["optimizer"],
                    options=options,
                ),
            )
            if self._rank == 0:
                log.info(
                    "restored full-weight checkpoint %s via %s",
                    checkpoint_path,
                    read_path,
                )
            return metadata
        finally:
            if local_checkpoint is not None:
                self._barrier()
                self._run_on_rank_zero(lambda: shutil.rmtree(local_checkpoint.parent))

    @property
    def latest(self) -> Path:
        return self.root / _LATEST_LINK_NAME

    @property
    def _distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    @property
    def _rank(self) -> int:
        if not self._distributed:
            return 0
        return dist.get_rank(self.process_group)

    @staticmethod
    def _validate_name(name: str) -> str:
        normalized = str(name).strip()
        if (
            not normalized
            or Path(normalized).name != normalized
            or normalized in {".", ".."}
        ):
            raise ValueError("Checkpoint name must be one non-empty path component.")
        if normalized == _LATEST_LINK_NAME or normalized.startswith("."):
            raise ValueError(f"Checkpoint name {normalized!r} is reserved.")
        return normalized

    def _operation_descriptor(
        self,
        operation: str,
        *,
        checkpoint_path: Path,
        checkpoint_name: str,
        staging_root: Path | None = None,
    ) -> _OperationDescriptor:
        return _OperationDescriptor(
            operation=operation,
            root=str(self.root.resolve(strict=False)),
            checkpoint_path=str(checkpoint_path.resolve(strict=False)),
            checkpoint_name=checkpoint_name,
            staging_root=None
            if staging_root is None
            else str(staging_root.resolve(strict=False)),
        )

    def _collective_preflight(
        self,
        operation: str,
        prepare: Callable[
            [], tuple[_OperationDescriptor, tuple[TrainableParameter, ...], _T]
        ],
    ) -> _T:
        prepared: (
            tuple[_OperationDescriptor, tuple[TrainableParameter, ...], _T] | None
        ) = None
        try:
            prepared = prepare()
            local_status = _PreflightStatus(
                error=None,
                descriptor=prepared[0],
                manifest=prepared[1],
            )
        except Exception as exc:  # noqa: BLE001 - relay local validation failures
            if not self._distributed:
                raise
            local_status = _PreflightStatus(error=f"{type(exc).__name__}: {exc}")

        if not self._distributed:
            if prepared is None:
                raise AssertionError("checkpoint preflight result is missing")
            return prepared[2]

        statuses: list[_PreflightStatus | None] = [None] * dist.get_world_size(
            self.process_group
        )
        dist.all_gather_object(statuses, local_status, group=self.process_group)
        failures = [
            f"rank {rank}: "
            f"{status.error if status is not None else 'missing preflight status'}"
            for rank, status in enumerate(statuses)
            if status is None or status.error is not None
        ]
        if failures:
            raise RuntimeError(
                f"Full-weight checkpoint {operation} preflight failed: "
                + "; ".join(failures)
            )
        first = statuses[0]
        if first is None or first.descriptor is None:
            raise RuntimeError("Checkpoint preflight did not return a descriptor.")
        if any(
            status is None or status.descriptor != first.descriptor
            for status in statuses
        ):
            raise RuntimeError(
                f"Full-weight checkpoint {operation} ranks do not share one "
                "operation descriptor."
            )
        if any(
            status is None or status.manifest != first.manifest for status in statuses
        ):
            raise RuntimeError(
                "Full-weight checkpoint ranks do not share the same trainable manifest."
            )
        if prepared is None:
            raise AssertionError("local checkpoint preflight result is missing")
        return prepared[2]

    def _run_on_all_ranks(self, operation: str, action: Callable[[], _T]) -> _T:
        try:
            result = action()
            local_error = None
        except Exception as exc:  # noqa: BLE001 - relay rank-local state failures
            if not self._distributed:
                raise
            result = None
            local_error = f"{type(exc).__name__}: {exc}"

        if not self._distributed:
            return result  # type: ignore[return-value]

        errors: list[str | None] = [None] * dist.get_world_size(self.process_group)
        dist.all_gather_object(errors, local_error, group=self.process_group)
        failures = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        ]
        if failures:
            raise RuntimeError(f"{operation} failed: " + "; ".join(failures))
        return result  # type: ignore[return-value]

    def _barrier(self) -> None:
        if self._distributed:
            dist.barrier(group=self.process_group)

    def _run_on_rank_zero(self, action: Callable[[], _T]) -> _T:
        if not self._distributed:
            return action()

        result: list[tuple[str, object] | None] = [None]
        if self._rank == 0:
            try:
                result[0] = ("ok", action())
            except Exception as exc:  # noqa: BLE001 - relay rank-zero filesystem failures
                result[0] = ("error", f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(
            result,
            group=self.process_group,
            group_src=0,
        )
        status = result[0]
        if status is None:
            raise RuntimeError(
                "Rank zero did not publish a checkpoint operation result."
            )
        if status[0] == "error":
            raise RuntimeError(f"Rank-zero checkpoint operation failed: {status[1]}")
        return status[1]  # type: ignore[return-value]

    def _commit_checkpoint(
        self,
        *,
        checkpoint_dir: Path,
        staging_dir: Path,
        checkpoint_name: str,
        optimizer_step: int,
        manifest: tuple[TrainableParameter, ...],
        trainer_payload: Any,
        metadata_fields: Mapping[str, object],
    ) -> None:
        payload_path = staging_dir / _TRAINER_PAYLOAD_FILENAME
        payload_staging_path = staging_dir / f".{_TRAINER_PAYLOAD_FILENAME}.tmp"
        torch.save(trainer_payload, payload_staging_path)
        self._fsync_file(payload_staging_path)
        payload_staging_path.replace(payload_path)

        metadata = {
            "format_version": _CHECKPOINT_FORMAT_VERSION,
            "name": checkpoint_name,
            "optimizer_step": int(optimizer_step),
            "trainable_manifest": [parameter.to_dict() for parameter in manifest],
            "manifest_sha256": _manifest_hash(manifest),
            "trainer_payload_file": _TRAINER_PAYLOAD_FILENAME,
            "metadata": dict(metadata_fields),
        }
        metadata_staging_path = staging_dir / f".{_METADATA_FILENAME}.tmp"
        metadata_path = staging_dir / _METADATA_FILENAME
        metadata_staging_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._fsync_file(metadata_staging_path)
        metadata_staging_path.replace(metadata_path)

        _publish_staging(
            self.root,
            checkpoint_dir=checkpoint_dir,
            staging_dir=staging_dir,
            replace_latest=self._replace_latest_link,
        )

    def _replace_latest_link(self, checkpoint_dir: Path) -> None:
        _replace_latest_link(self.root, checkpoint_dir)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _validate_trainer_payload(trainer_payload: Any) -> None:
        buffer = io.BytesIO()
        torch.save(trainer_payload, buffer)
        buffer.seek(0)
        try:
            torch.load(buffer, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                "Full-weight checkpoint trainer_payload must be loadable with "
                "torch.load(weights_only=True)."
            ) from exc

    @staticmethod
    def _read_metadata(checkpoint_dir: Path) -> tuple[dict[str, object], Any]:
        metadata_path = checkpoint_dir / _METADATA_FILENAME
        if not metadata_path.is_file():
            raise RuntimeError(
                f"Checkpoint is incomplete because {metadata_path} is missing."
            )
        raw_header = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw_header, dict):
            raise RuntimeError("Checkpoint metadata must be a JSON object.")
        payload_file = raw_header.get("trainer_payload_file")
        if payload_file != _TRAINER_PAYLOAD_FILENAME:
            raise RuntimeError(
                f"Checkpoint references unsupported trainer payload {payload_file!r}."
            )
        payload_path = checkpoint_dir / _TRAINER_PAYLOAD_FILENAME
        if not payload_path.is_file():
            raise RuntimeError(
                f"Checkpoint is incomplete because {payload_path} is missing."
            )
        trainer_payload = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=True,
        )
        return raw_header, trainer_payload

    @staticmethod
    def _parse_metadata(
        header: Mapping[str, object], *, trainer_payload: Any
    ) -> FullWeightCheckpointMetadata:
        version = int(header.get("format_version", 0))
        if version != _CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported full-weight checkpoint version {version}; "
                f"expected {_CHECKPOINT_FORMAT_VERSION}."
            )
        raw_manifest = header.get("trainable_manifest")
        if not isinstance(raw_manifest, list):
            raise RuntimeError("Checkpoint trainable manifest must be a list.")
        manifest = tuple(
            TrainableParameter.from_dict(parameter)
            for parameter in raw_manifest
            if isinstance(parameter, dict)
        )
        if len(manifest) != len(raw_manifest) or not manifest:
            raise RuntimeError("Checkpoint trainable manifest is invalid or empty.")
        expected_hash = _manifest_hash(manifest)
        if header.get("manifest_sha256") != expected_hash:
            raise RuntimeError("Checkpoint trainable manifest hash does not match.")
        raw_fields = header.get("metadata", {})
        if not isinstance(raw_fields, dict):
            raise RuntimeError("Checkpoint metadata fields must be a JSON object.")
        return FullWeightCheckpointMetadata(
            name=str(header.get("name", "")),
            optimizer_step=int(header.get("optimizer_step", 0)),
            trainable_manifest=manifest,
            trainer_payload=trainer_payload,
            metadata_fields=dict(raw_fields),
        )
