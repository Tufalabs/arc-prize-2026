from __future__ import annotations

import copy
import hashlib
import os
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from training.full_weight_checkpoint import (
    FullWeightCheckpointMetadata,
    FullWeightCheckpointer,
    upload_checkpoint_async,
)
from training.memory_profile import NullMemoryLogger
from training.shutdown import defer_training_shutdown
from training.policy_sync.full_weight import FullWeightParameterSource
from training.policy_sync.vllm_nccl import WeightManifest


_CMD_START = "start"
_CMD_TRAIN_WINDOW = "train_window"
_CMD_MANIFEST = "manifest"
_CMD_PUBLISH = "publish"
_CMD_CHECKPOINT = "checkpoint"
_CMD_EXIT = "exit"


class FullWeightDistributedRuntime:
    """Rank-zero facade over one rank-symmetric FSDP2 command stream."""

    def __init__(
        self,
        *,
        control_group: Any | None = None,
        trainer_factory: Callable[..., Any] | None = None,
        checkpointer_factory: Callable[[Path], FullWeightCheckpointer] | None = None,
        parameter_source_factory: Callable[
            [torch.nn.Module], FullWeightParameterSource
        ] = FullWeightParameterSource,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Full-weight distributed runtime requires torch.distributed."
            )
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        self.control_group = control_group
        if dist.get_backend(control_group) != "gloo":
            raise RuntimeError(
                "Full-weight command transport requires a full-WORLD Gloo "
                "control group."
            )
        self.rank = dist.get_rank(control_group)
        self.world_size = dist.get_world_size(control_group)
        if self.rank != dist.get_rank() or self.world_size != dist.get_world_size():
            raise RuntimeError(
                "Full-weight control group must contain the complete WORLD group "
                "with unchanged rank ordering."
            )
        self._trainer_factory = trainer_factory or self._default_trainer_factory
        self._checkpointer_factory = checkpointer_factory or (
            lambda root: FullWeightCheckpointer(
                root,
                process_group=self.control_group,
            )
        )
        self._parameter_source_factory = parameter_source_factory
        self._trainer: Any | None = None
        self._publication_manifest: WeightManifest | None = None
        self._rank_zero_memory_logger: Any = NullMemoryLogger()
        self._closed = False
        self._checkpoint_identity_fields: dict[str, Any] = {}
        self._rank_zero_window_callbacks: dict[str, Callable[[], Any]] = {}

    @staticmethod
    def _default_trainer_factory(**kwargs: Any) -> Any:
        from training.full_weight_ppo import FullWeightPPOTrainer

        return FullWeightPPOTrainer(**kwargs)

    @property
    def trainer(self) -> Any:
        if self._trainer is None:
            raise RuntimeError("Full-weight trainer has not been initialized.")
        return self._trainer

    def pin_checkpoint_identity(self, reported: dict[str, Any]) -> None:
        """Pin a window's harness agent settings where rank zero writes checkpoints."""
        from training.rollout_identity import pin_harness_agent

        self._require_rank_zero()
        if "checkpoint_identity" not in self._checkpoint_identity_fields:
            # Flash: the trainer's own state carries the identity.
            self.trainer.pin_checkpoint_identity(reported)
            return
        self._set_checkpoint_identity(
            pin_harness_agent(
                self._checkpoint_identity_fields["checkpoint_identity"],
                reported,
                source="this window's rollouts",
            )
        )

    def _set_checkpoint_identity(self, identity: dict[str, Any] | None) -> None:
        """The identity full-weight checkpoints record, kept on the trainer too."""
        self._checkpoint_identity_fields["checkpoint_identity"] = identity
        if hasattr(self.trainer, "_checkpoint_identity"):
            self.trainer._checkpoint_identity = copy.deepcopy(identity)

    def _require_rank_zero(self) -> None:
        if self.rank != 0:
            raise RuntimeError("Only rank zero may drive the distributed runtime.")

    def _broadcast(self, command: dict[str, Any]) -> None:
        self._require_rank_zero()
        if self._closed:
            raise RuntimeError("Full-weight distributed runtime is closed.")
        payload: list[dict[str, Any] | None] = [command]
        dist.broadcast_object_list(
            payload,
            src=0,
            group=self.control_group,
        )

    def _receive(self) -> dict[str, Any]:
        payload: list[dict[str, Any] | None] = [None]
        dist.broadcast_object_list(
            payload,
            src=0,
            group=self.control_group,
        )
        command = payload[0]
        if not isinstance(command, dict):
            raise RuntimeError("Rank zero broadcast an invalid full-weight command.")
        return command

    def _run_symmetric(self, action: Callable[[], Any]) -> Any:
        # Do not catch rank-local failures here. A rank can fail before its
        # peers enter an FSDP collective; trying to relay that exception with
        # another collective would deadlock. Let torchrun observe the exited
        # process and fail-stop the remaining ranks instead.
        result = action()
        dist.barrier(group=self.control_group)
        return result

    @staticmethod
    def _local_device(training_config: Any) -> str:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}"
        return str(training_config.device)

    def create_trainer(
        self,
        training_config: Any,
        *,
        game_vocab: Sequence[str],
        memory_logger: Any | None = None,
        checkpoint_dir: Path | None = None,
        expected_checkpoint: Mapping[str, object] | None = None,
        checkpoint_identity: dict[str, Any] | None = None,
        grpo_group_size: int | None = None,
        grpo_puzzles_per_window: int | None = None,
    ) -> Any:
        self._require_rank_zero()
        if (checkpoint_dir is None) != (expected_checkpoint is None):
            raise ValueError(
                "Checkpoint path and expected full-weight identity must be "
                "provided together."
            )
        required_identity = {"iteration_index", "model_name", "model_revision"}
        if (
            expected_checkpoint is not None
            and set(expected_checkpoint) != required_identity
        ):
            raise ValueError(
                "Expected full-weight checkpoint identity must contain exactly "
                "iteration_index, model_name, and model_revision."
            )
        self._rank_zero_memory_logger = memory_logger or NullMemoryLogger()
        identity_fields = {
            "checkpoint_identity": copy.deepcopy(checkpoint_identity),
            "grpo_group_size": grpo_group_size,
            "grpo_puzzles_per_window": grpo_puzzles_per_window,
        }
        command = {
            "type": _CMD_START,
            "training_config": training_config,
            "game_vocab": list(game_vocab),
            "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
            "expected_checkpoint": {
                **dict(expected_checkpoint or {}),
                **identity_fields,
            },
            "identity_fields": identity_fields,
        }
        self._broadcast(command)
        return self._run_symmetric(lambda: self._handle_start(command))

    def _handle_start(self, command: Mapping[str, Any]) -> Any:
        if self._trainer is not None:
            raise RuntimeError("Full-weight trainer was initialized more than once.")
        training_config = copy.deepcopy(command["training_config"])
        training_config.device = self._local_device(training_config)
        memory_logger = (
            self._rank_zero_memory_logger if self.rank == 0 else NullMemoryLogger()
        )
        self._checkpoint_identity_fields = copy.deepcopy(command["identity_fields"])
        trainer_identity_kwargs = {
            name: value
            for name, value in self._checkpoint_identity_fields.items()
            if value is not None
        }
        self._trainer = self._trainer_factory(
            config=training_config,
            **trainer_identity_kwargs,
            game_vocab=list(command["game_vocab"]),
            memory_logger=memory_logger,
        )
        checkpoint_dir = command.get("checkpoint_dir")
        if checkpoint_dir is not None:
            checkpoint_path = Path(str(checkpoint_dir))
            durable_root = checkpoint_path.parent
            staging_root = self._checkpoint_write_root(durable_root)
            metadata = self._checkpointer_factory(durable_root).load(
                checkpoint_path,
                model=self.trainer.model,
                optimizer=self.trainer.optimizer,
                staging_root=staging_root if staging_root != durable_root else None,
            )
            self._restore_checkpoint_payload(
                metadata,
                expected=command.get("expected_checkpoint", {}),
            )
        return self.trainer

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state()
        return state

    @staticmethod
    def _restore_rng_state(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_state = state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_state.cpu())

    @defer_training_shutdown()
    def train_window(
        self,
        rollouts: list[Any],
        *,
        output_dir: Path,
        iteration_index: int,
        sample_budget_fraction: float,
        policy_lag_windows: int,
        entropy_coef: float | None,
        check_health: Callable[[], None] | None = None,
        next_window_ready: Callable[[], bool] | None = None,
        next_window_seconds_remaining: Callable[[], float | None] | None = None,
    ) -> Any:
        self._require_rank_zero()
        self._rank_zero_window_callbacks = {
            name: callback
            for name, callback in {
                "check_health": check_health,
                "next_window_ready": next_window_ready,
                "next_window_seconds_remaining": next_window_seconds_remaining,
            }.items()
            if callback is not None
        }
        command = {
            "type": _CMD_TRAIN_WINDOW,
            "callback_names": list(self._rank_zero_window_callbacks),
            "rollouts": rollouts,
            "output_dir": str(output_dir),
            "iteration_index": int(iteration_index),
            "sample_budget_fraction": float(sample_budget_fraction),
            "policy_lag_windows": int(policy_lag_windows),
            "entropy_coef": entropy_coef,
            "rng_state": self._capture_rng_state(),
        }
        try:
            self._broadcast(command)
            return self._run_symmetric(lambda: self._handle_train_window(command))
        finally:
            self._rank_zero_window_callbacks = {}

    def _window_callback(self, name: str) -> Any:
        # All ranks reach these callsites together. Only rank zero touches the
        # collector/process handles; every rank receives the same decision or
        # failure before entering its next model collective.
        payload: list[Any] = [None]
        if self.rank == 0:
            try:
                payload[0] = (name, None, self._rank_zero_window_callbacks[name]())
            except Exception as error:
                payload[0] = (name, f"{type(error).__name__}: {error}", None)
        dist.broadcast_object_list(payload, src=0, group=self.control_group)
        callback_name, error, result = payload[0]
        if callback_name != name:
            raise RuntimeError(
                f"Full-weight window callbacks diverged: {name} != {callback_name}."
            )
        if error is not None:
            raise RuntimeError(f"Full-weight {name} failed: {error}")
        return result

    def _handle_train_window(self, command: Mapping[str, Any]) -> Any:
        self._restore_rng_state(command["rng_state"])
        callbacks = {
            name: (lambda name=name: self._window_callback(name))
            for name in command.get("callback_names", ())
        }
        return self.trainer.train_window(
            list(command["rollouts"]),
            output_dir=Path(str(command["output_dir"])),
            iteration_index=int(command["iteration_index"]),
            sample_budget_fraction=float(command["sample_budget_fraction"]),
            policy_lag_windows=int(command["policy_lag_windows"]),
            entropy_coef=command.get("entropy_coef"),
            check_health=callbacks.get("check_health"),
            next_window_ready=callbacks.get("next_window_ready"),
            next_window_seconds_remaining=callbacks.get(
                "next_window_seconds_remaining"
            ),
        )

    def publish(
        self,
        send_rank_zero: Callable[[Iterator[tuple[str, torch.Tensor]]], None],
    ) -> None:
        self._require_rank_zero()
        command = {"type": _CMD_PUBLISH}
        self._broadcast(command)
        self._handle_publish(send_rank_zero=send_rank_zero)

    def publication_manifest(self) -> WeightManifest:
        self._require_rank_zero()
        command = {"type": _CMD_MANIFEST}
        self._broadcast(command)
        return self._run_symmetric(self._handle_publication_manifest)

    def _handle_publication_manifest(self) -> WeightManifest:
        source = self._parameter_source_factory(self.trainer.model)
        source.validate_manifest()
        self._publication_manifest = source.manifest
        return source.manifest

    def _handle_publish(
        self,
        *,
        send_rank_zero: Callable[[Iterator[tuple[str, torch.Tensor]]], None] | None,
    ) -> None:
        source = self._parameter_source_factory(self.trainer.model)
        if self._publication_manifest is None:
            raise RuntimeError(
                "Full-weight publication manifest must be validated before transfer."
            )
        if source.manifest != self._publication_manifest:
            raise RuntimeError(
                "Full-weight parameter manifest changed after transfer initialization."
            )
        publication_error: str | None = None
        if self.rank == 0:
            if send_rank_zero is None:
                raise RuntimeError("Rank zero publication requires a sender.")
            parameters = source.rank_zero_iterator()
            send_rank_zero(parameters)
            try:
                next(parameters)
            except StopIteration:
                pass
            else:
                for _name, _tensor in parameters:
                    pass
                publication_error = (
                    "Full-weight sender returned without exhausting the parameter "
                    "stream. Remaining gathers were drained; publication is rejected."
                )
        else:
            source.participate()
        status = [publication_error]
        dist.broadcast_object_list(
            status,
            src=0,
            group=self.control_group,
        )
        if status[0] is not None:
            raise RuntimeError(status[0])

    def checkpoint(
        self,
        *,
        root: Path,
        name: str,
        iteration_index: int,
        metadata_fields: Mapping[str, object] | None = None,
    ) -> Future[Path]:
        self._require_rank_zero()
        durable_root = Path(root)
        write_root = self._checkpoint_write_root(durable_root)
        if write_root != durable_root:
            retry = 0
            base_write_root = write_root
            while (write_root / str(name)).exists() or (
                write_root / str(name)
            ).is_symlink():
                retry += 1
                write_root = base_write_root.with_name(
                    f"{base_write_root.name}-retry-{retry:03d}"
                )
        command = {
            "type": _CMD_CHECKPOINT,
            "root": str(write_root),
            "name": str(name),
            "iteration_index": int(iteration_index),
            "metadata_fields": {
                **dict(metadata_fields or {}),
                **self._checkpoint_identity_fields,
            },
        }
        self._broadcast(command)
        local_checkpoint = self._run_symmetric(lambda: self._handle_checkpoint(command))
        if write_root == durable_root:
            completed: Future[Path] = Future()
            completed.set_result(local_checkpoint)
            return completed
        return upload_checkpoint_async(
            local_checkpoint,
            durable_root=durable_root,
            checkpoint_name=str(name),
        )

    @staticmethod
    def _checkpoint_write_root(durable_root: Path) -> Path:
        scratch_dir = os.environ.get("SCRATCH_DIR", "").strip()
        if not scratch_dir:
            return durable_root
        resolved = str(durable_root.resolve(strict=False)).encode("utf-8")
        suffix = hashlib.sha256(resolved).hexdigest()[:12]
        return (
            Path(scratch_dir)
            / "full_weight_checkpoints"
            / f"{durable_root.parent.name}-{suffix}"
        )

    def _handle_checkpoint(self, command: Mapping[str, Any]) -> Path:
        rng_by_rank: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(
            rng_by_rank,
            self._capture_rng_state(),
            group=self.control_group,
        )
        if any(state is None for state in rng_by_rank):
            raise RuntimeError("Checkpoint did not collect RNG state from every rank.")
        state_critic = getattr(self.trainer, "state_critic", None)
        trainer_payload = {
            "iteration_index": int(command["iteration_index"]),
            "dry_optimizer_step_count": self.trainer._dry_optimizer_step_count,
            "trust_region_lr_scale": float(
                getattr(self.trainer, "_trust_region_lr_scale", 1.0)
            ),
            "consecutive_window_rejections": int(
                getattr(self.trainer, "_consecutive_window_rejections", 0)
            ),
            "rng_by_rank": rng_by_rank,
            "state_critic_state": None
            if state_critic is None
            else state_critic.state_dict(),
        }
        return self._checkpointer_factory(Path(str(command["root"]))).save(
            str(command["name"]),
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            optimizer_step=int(self.trainer.optimizer_step_count),
            trainer_payload=trainer_payload,
            metadata_fields=command.get("metadata_fields", {}),
        )

    def _restore_checkpoint_payload(
        self,
        metadata: FullWeightCheckpointMetadata,
        *,
        expected: Mapping[str, object],
    ) -> None:
        payload = metadata.trainer_payload
        if not isinstance(payload, dict):
            raise RuntimeError("Full-weight checkpoint trainer payload must be a dict.")
        expected_iteration = int(expected["iteration_index"])
        saved_iteration = int(payload.get("iteration_index", -1))
        if saved_iteration != expected_iteration:
            raise RuntimeError(
                "Full-weight checkpoint iteration does not match run state: "
                f"checkpoint={saved_iteration} expected={expected_iteration}."
            )
        from training.rollout_identity import adopt_harness_agent, without_harness_agent

        for field in (
            "model_name",
            "model_revision",
            "checkpoint_identity",
            "grpo_group_size",
            "grpo_puzzles_per_window",
        ):
            saved, configured = metadata.metadata_fields.get(field), expected.get(field)
            if field == "checkpoint_identity":
                # Harness settings pinned from a window are the checkpoint's
                # own (training/rollout_identity.py); they carry over below.
                saved, configured = without_harness_agent(saved), without_harness_agent(configured)
            if saved != configured:
                raise RuntimeError(
                    f"Full-weight checkpoint {field} does not match this run: "
                    f"checkpoint={metadata.metadata_fields.get(field)!r} "
                    f"expected={expected.get(field)!r}."
                )
        if "checkpoint_identity" in self._checkpoint_identity_fields:
            self._set_checkpoint_identity(
                adopt_harness_agent(
                    self._checkpoint_identity_fields["checkpoint_identity"],
                    metadata.metadata_fields.get("checkpoint_identity"),
                )
            )
        rng_by_rank = payload.get("rng_by_rank")
        if not isinstance(rng_by_rank, list) or len(rng_by_rank) != self.world_size:
            raise RuntimeError(
                "Full-weight checkpoint RNG topology does not match this run."
            )
        rank_rng = rng_by_rank[self.rank]
        if not isinstance(rank_rng, dict):
            raise RuntimeError("Full-weight checkpoint rank RNG state is invalid.")
        self._restore_rng_state(rank_rng)
        state_critic_state = payload.get("state_critic_state")
        state_critic = getattr(self.trainer, "state_critic", None)
        if state_critic is not None:
            if not isinstance(state_critic_state, dict):
                raise RuntimeError(
                    "Full-weight checkpoint is missing state critic state."
                )
            state_critic.load_state_dict(state_critic_state)
        elif state_critic_state is not None:
            raise RuntimeError("GRPO checkpoints must not contain a state critic.")
        self.trainer._trust_region_lr_scale = float(
            payload.get("trust_region_lr_scale", 1.0)
        )
        self.trainer._consecutive_window_rejections = int(
            payload.get("consecutive_window_rejections", 0)
        )
        self.trainer._optimizer_step_count = int(metadata.optimizer_step)
        self.trainer._dry_optimizer_step_count = int(
            payload.get("dry_optimizer_step_count", 0)
        )

    def serve(self) -> None:
        if self.rank == 0:
            raise RuntimeError("Rank zero drives commands instead of serving them.")
        if self._closed:
            raise RuntimeError("Full-weight distributed runtime is closed.")
        while True:
            command = self._receive()
            command_type = command.get("type")
            if command_type == _CMD_EXIT:
                self._closed = True
                return
            if command_type == _CMD_START:
                self._run_symmetric(lambda: self._handle_start(command))
            elif command_type == _CMD_TRAIN_WINDOW:
                self._run_symmetric(lambda: self._handle_train_window(command))
            elif command_type == _CMD_MANIFEST:
                self._run_symmetric(self._handle_publication_manifest)
            elif command_type == _CMD_PUBLISH:
                self._handle_publish(send_rank_zero=None)
            elif command_type == _CMD_CHECKPOINT:
                self._run_symmetric(lambda: self._handle_checkpoint(command))
            elif not self._serve_extension_command(command):
                raise RuntimeError(
                    f"Unknown full-weight command type {command_type!r}."
                )

    def _serve_extension_command(self, command: Mapping[str, Any]) -> bool:
        """Run a subclass's own command on a nonzero rank; False if unknown."""
        return False

    def close(self) -> None:
        self._require_rank_zero()
        if self._closed:
            return
        self._broadcast({"type": _CMD_EXIT})
        self._closed = True
