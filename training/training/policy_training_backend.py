from __future__ import annotations

import copy
import logging
import shutil
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from training.config import ExperimentConfig
from training.full_weight_distributed import FullWeightDistributedRuntime
from training.official_eval_policy import (
    FullWeightEvalRequest,
    FullWeightEvalRequestInbox,
    protected_full_weight_eval_iterations,
)
from training.policy_sync import (
    VllmControlClient,
    VllmNcclTrainerEndpoint,
    VllmNcclWeightTransfer,
    WeightManifest,
)
from training.run_artifacts import write_json_atomic


log = logging.getLogger(__name__)


def _prune_full_weight_checkpoints(
    checkpoints_root: Path,
    *,
    keep_last: int,
    protected_checkpoint: Path,
) -> None:
    """Remove superseded committed checkpoints after run state advances."""
    checkpoints = sorted(
        (
            path
            for path in checkpoints_root.glob("full_weight_iter_*")
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: path.name,
    )
    protected = {protected_checkpoint.resolve()}
    latest = checkpoints_root / "latest"
    if latest.is_symlink():
        protected.add(latest.resolve())
    retained = {path.resolve() for path in checkpoints[-keep_last:]}
    evaluated = protected_full_weight_eval_iterations(checkpoints_root.parent)
    for checkpoint in checkpoints:
        iteration = int(
            checkpoint.name.removeprefix("full_weight_iter_").split("_retry_", 1)[0]
        )
        if (
            checkpoint.resolve() in protected
            or checkpoint.resolve() in retained
            or iteration in evaluated
        ):
            continue
        try:
            shutil.rmtree(checkpoint)
        except Exception:  # noqa: BLE001 - retention must not fail durable training
            log.exception(
                "could not prune superseded full-weight checkpoint %s; "
                "retention will retry on a later checkpoint",
                checkpoint,
            )
        else:
            log.info("pruned superseded full-weight checkpoint %s", checkpoint)


@dataclass(frozen=True, slots=True)
class TrainingResume:
    start_iteration: int = 0
    last_completed_iteration: int = -1
    next_schedule_index: int = 0
    adapter_dir: Path | None = None
    runtime_adapter_dir: Path | None = None
    trainer_state_path: Path | None = None
    full_weight_checkpoint: Path | None = None


@dataclass(slots=True)
class _PendingFullWeightCheckpoint:
    future: Future[Path]
    run_root: Path
    state: dict[str, Any]


def _resolve_state_path(run_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else run_root / path


def _relative_state_path(run_root: Path, path: Path) -> str:
    try:
        return (
            path.resolve(strict=False)
            .relative_to(run_root.resolve(strict=False))
            .as_posix()
        )
    except ValueError:
        return str(path)


class FullWeightPolicyTrainingBackend:
    """FSDP2 trainer, DCP checkpointer, and persistent vLLM publisher."""

    refresh_strategy = "vllm_nccl"

    def __init__(
        self,
        config: ExperimentConfig,
        runtime: FullWeightDistributedRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self._manifest: WeightManifest | None = None
        self._endpoint: VllmNcclTrainerEndpoint | None = None
        self._transfer: VllmNcclWeightTransfer | None = None
        self._eval_endpoint: VllmNcclTrainerEndpoint | None = None
        self._eval_transfer: VllmNcclWeightTransfer | None = None
        self._eval_weight_version: str | None = None
        self._eval_server_urls: tuple[str, ...] | None = None
        self._eval_inbox: FullWeightEvalRequestInbox | None = None
        self._server_auth_headers: dict[str, str] = {}
        self._pending_checkpoint: _PendingFullWeightCheckpoint | None = None
        self._latest_completed_iteration: tuple[Path, int, dict[str, Any]] | None = None
        # Longest time a checkpoint took to become durable; the segment budget
        # reserves it before starting an iteration near the deadline.
        self.last_checkpoint_seconds: float | None = None
        self._last_checkpoint_iteration: int | None = None
        self._eval_checkpoint_iteration: int | None = None
        self._run_root: Path | None = None
        self._training_collectors: list[Any] = []
        self._training_clients: list[Any] = []
        self._eval_clients: list[Any] = []
        self._training_started = False
        self._weight_version = uuid.uuid4().hex

    @property
    def trainer(self) -> Any:
        return self.runtime.trainer

    @property
    def published_policy_path(self) -> None:
        return None

    def resolve_resume(
        self, run_root: Path, state: dict[str, Any] | None
    ) -> TrainingResume:
        if state is None:
            return TrainingResume()
        version = int(state.get("version", 0))
        if version != 2 or state.get("weight_mode") != "full_weight":
            raise RuntimeError(
                "Full-weight training requires a version 2 full_weight run state; "
                f"got version={version} weight_mode={state.get('weight_mode')!r}."
            )
        start_iteration = int(state.get("next_iteration", 0))
        last_completed_iteration = int(
            state.get("last_completed_iteration", start_iteration - 1)
        )
        policy_iteration = int(state.get("policy_iteration", -1))
        if (
            start_iteration != last_completed_iteration + 1
            or policy_iteration != last_completed_iteration
        ):
            raise RuntimeError(
                "Full-weight run state iteration fields are inconsistent: "
                f"next_iteration={start_iteration} "
                f"last_completed_iteration={last_completed_iteration} "
                f"policy_iteration={policy_iteration}."
            )
        checkpoint = _resolve_state_path(run_root, state["checkpoint_path"])
        if not checkpoint.is_dir():
            raise RuntimeError(
                f"Full-weight run state points at a missing checkpoint: {checkpoint}"
            )
        return TrainingResume(
            start_iteration=start_iteration,
            last_completed_iteration=last_completed_iteration,
            next_schedule_index=int(state.get("next_schedule_index", start_iteration)),
            full_weight_checkpoint=checkpoint,
        )

    def pin_checkpoint_identity(self, reported: dict[str, Any]) -> None:
        """Pin a window's harness agent settings (training/rollout_identity.py)."""
        self.runtime.pin_checkpoint_identity(reported)

    def create_trainer(
        self,
        *,
        game_vocab: list[str],
        memory_logger: Any,
        resume: TrainingResume,
        checkpoint_identity: dict[str, Any] | None = None,
        grpo_group_size: int | None = None,
        grpo_puzzles_per_window: int | None = None,
    ) -> Any:
        trainer = self.runtime.create_trainer(
            self.config.training,
            game_vocab=game_vocab,
            memory_logger=memory_logger,
            checkpoint_dir=resume.full_weight_checkpoint,
            checkpoint_identity=checkpoint_identity,
            grpo_group_size=grpo_group_size,
            grpo_puzzles_per_window=grpo_puzzles_per_window,
            expected_checkpoint=(
                None
                if resume.full_weight_checkpoint is None
                else {
                    "iteration_index": resume.last_completed_iteration,
                    "model_name": self.config.training.model_name,
                    "model_revision": self.config.training.model_revision,
                }
            ),
        )
        self._manifest = self.runtime.publication_manifest()
        return trainer

    def prepare_rollouts(
        self,
        collector: Any,
        *,
        resume: TrainingResume,
        run_root: Path,
        official_eval_process: Any | None = None,
    ) -> None:
        self._run_root = run_root
        collector.prepare_servers()
        self._endpoint = VllmNcclTrainerEndpoint()
        clients = [
            VllmControlClient(
                rollout_collector.config.inference.base_url,
                auth_headers=rollout_collector._server_auth_headers(),
            )
            for rollout_collector in collector.collectors
        ]
        self._training_collectors = list(collector.collectors)
        self._training_clients = clients
        self._transfer = VllmNcclWeightTransfer(
            clients,
            trainer=self._endpoint,
        )
        self._server_auth_headers = collector.collectors[0]._server_auth_headers()
        collector.activate_policy_at_boundary(
            policy_iteration=resume.last_completed_iteration,
            activate=self._publish_current_policy,
        )
        if getattr(getattr(self.config, "official_eval", None), "enabled", False):
            self._eval_inbox = FullWeightEvalRequestInbox(run_root=run_root)

            def require_eval_worker_alive() -> None:
                if official_eval_process is None:
                    return
                returncode = official_eval_process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        "full-weight official eval worker exited during startup "
                        f"with code {returncode}; see official_eval.log and "
                        "official_eval.stdout.log in the run directory."
                    )

            self._service_eval_request(
                self._eval_inbox.wait(
                    timeout_seconds=1800.0,
                    liveness_check=require_eval_worker_alive,
                ),
                policy_iteration=resume.last_completed_iteration,
            )
        collector.start()

    def _publish_current_policy(self) -> None:
        if self._manifest is None or self._transfer is None or self._endpoint is None:
            raise RuntimeError("Full-weight publication has not been initialized.")
        self._transfer.transfer(
            self._manifest,
            send_weights=lambda: self.runtime.publish(self._endpoint.send_weights),
            weight_version=self._weight_version,
        )
        for collector, client in zip(
            self._training_collectors, self._training_clients, strict=True
        ):
            collector.pin_full_weight_identity(client.adapter_identity)

    def _service_eval_request(
        self,
        request: FullWeightEvalRequest,
        *,
        policy_iteration: int,
    ) -> None:
        if self._manifest is None or self._eval_inbox is None:
            raise RuntimeError("Full-weight eval publication has not been initialized.")
        if self._eval_transfer is None:
            self._eval_server_urls = request.server_urls
            self._eval_endpoint = VllmNcclTrainerEndpoint()
            clients = [
                VllmControlClient(url, auth_headers=self._server_auth_headers)
                for url in request.server_urls
            ]
            self._eval_clients = clients
            self._eval_transfer = VllmNcclWeightTransfer(
                clients,
                trainer=self._eval_endpoint,
            )
        elif request.server_urls != self._eval_server_urls:
            raise RuntimeError(
                "Full-weight official-eval server URLs changed after the NCCL "
                "transfer group was initialized."
            )
        endpoint = self._eval_endpoint
        if endpoint is None:
            raise RuntimeError("Full-weight eval NCCL endpoint is missing.")
        if self._eval_weight_version != self._weight_version:
            self._eval_transfer.transfer(
                self._manifest,
                send_weights=lambda: self.runtime.publish(endpoint.send_weights),
                weight_version=self._weight_version,
            )
            self._eval_weight_version = self._weight_version
        self._eval_inbox.acknowledge(
            request,
            policy_iteration=policy_iteration,
            publication_identities=[
                client.adapter_identity for client in self._eval_clients
            ],
        )
        self._eval_checkpoint_iteration = policy_iteration

    def _service_pending_eval_request(self, *, policy_iteration: int) -> None:
        if self._eval_inbox is None:
            return
        request = self._eval_inbox.pending()
        if request is not None:
            self._service_eval_request(
                request,
                policy_iteration=policy_iteration,
            )

    def train_window(self, rollouts: list[Any], **kwargs: Any) -> Any:
        pending = self._pending_checkpoint
        if pending is not None and pending.future.done():
            pending.future.result()
        if not self._training_started and self._run_root is not None:
            marker = self._run_root / "checkpoints" / ".full_weight_in_progress"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch(exist_ok=True)
            self._training_started = True
        result = self.runtime.train_window(rollouts, **kwargs)
        self._weight_version = uuid.uuid4().hex
        return result

    def publish_after_train(
        self,
        collector: Any,
        *,
        train_result: Any,
        iteration_index: int,
        should_publish: bool,
        deadline_epoch: float | None = None,
    ) -> bool:
        del train_result
        self.publication_wait_seconds = 0.0
        if should_publish:
            publication_started = time.perf_counter()

            def activate() -> None:
                # Only the wait for the rollout boundary is idle. Weight
                # transfer and evaluation publication remain trainer work.
                self.publication_wait_seconds = (
                    time.perf_counter() - publication_started
                )
                self._publish_current_policy()

            if not collector.activate_policy_at_boundary(
                policy_iteration=iteration_index,
                activate=activate,
                deadline_epoch=deadline_epoch,
            ):
                return False
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            return False
        self._service_pending_eval_request(policy_iteration=iteration_index)
        return True

    @staticmethod
    def _commit_checkpoint_when_durable(
        checkpoint_future: Future[Path],
        *,
        run_root: Path,
        state: dict[str, Any],
        keep_last: int,
        started_at: float | None = None,
        report_seconds: Callable[[float], None] | None = None,
    ) -> Future[Path]:
        """Write run state as soon as its checkpoint upload becomes durable."""
        committed: Future[Path] = Future()

        def commit(source: Future[Path]) -> None:
            try:
                checkpoint_path = source.result()
                if started_at is not None and report_seconds is not None:
                    report_seconds(time.perf_counter() - started_at)
                payload = {
                    **state,
                    "checkpoint_path": _relative_state_path(
                        run_root,
                        checkpoint_path,
                    ),
                }
                write_json_atomic(run_root / "training_state.json", payload)
                try:
                    _prune_full_weight_checkpoints(
                        run_root / "checkpoints",
                        keep_last=keep_last,
                        protected_checkpoint=checkpoint_path,
                    )
                except Exception:  # noqa: BLE001 - retention is post-commit cleanup
                    log.exception(
                        "full-weight checkpoint retention pass failed; "
                        "retention will retry after a later checkpoint"
                    )
            except BaseException as exc:  # noqa: BLE001 - surface via committed future
                committed.set_exception(exc)
            else:
                committed.set_result(checkpoint_path)

        checkpoint_future.add_done_callback(commit)
        return committed

    def finalize_iteration(
        self,
        *,
        run_root: Path,
        iteration_index: int,
        train_result: Any,
        common_run_state: dict[str, Any],
        force_checkpoint: bool = False,
    ) -> dict[str, Any] | None:
        del train_result
        # Keep off-cadence progress available for a clock-triggered yield.
        self._latest_completed_iteration = (
            run_root,
            iteration_index,
            copy.deepcopy(common_run_state),
        )
        completed_state = self._finish_pending_checkpoint(wait=False)
        if force_checkpoint and self._last_checkpoint_iteration == iteration_index:
            if self._pending_checkpoint is not None:
                current_state = self._finish_pending_checkpoint(wait=True)
                return current_state or completed_state
            return completed_state
        completed_iterations = iteration_index + 1
        cadence = self.config.training.full_weight.checkpoint_every_iterations
        should_checkpoint = (
            force_checkpoint
            or iteration_index == self._eval_checkpoint_iteration
            or completed_iterations == 1
            or completed_iterations % cadence == 0
            or completed_iterations == self.config.num_iterations
        )
        if not should_checkpoint:
            return completed_state
        if self._pending_checkpoint is not None:
            completed_state = self._finish_pending_checkpoint(wait=True)
        checkpoints_root = run_root / "checkpoints"
        base_name = f"full_weight_iter_{iteration_index:06d}"
        checkpoint_name = base_name
        retry = 0
        while (checkpoints_root / checkpoint_name).exists() or (
            checkpoints_root / checkpoint_name
        ).is_symlink():
            retry += 1
            checkpoint_name = f"{base_name}_retry_{retry:03d}"
        checkpoint_started_at = time.perf_counter()
        checkpoint_future = self.runtime.checkpoint(
            root=checkpoints_root,
            name=checkpoint_name,
            iteration_index=iteration_index,
            metadata_fields={
                "model_name": self.config.training.model_name,
                "model_revision": self.config.training.model_revision,
            },
        )
        pending_state = {
            "version": 2,
            "weight_mode": "full_weight",
            **common_run_state,
            "policy_iteration": iteration_index,
        }
        committed_future = self._commit_checkpoint_when_durable(
            checkpoint_future,
            run_root=run_root,
            state=pending_state,
            keep_last=self.config.training.full_weight.checkpoint_keep_last,
            started_at=checkpoint_started_at,
            report_seconds=self._record_checkpoint_seconds,
        )
        self._pending_checkpoint = _PendingFullWeightCheckpoint(
            future=committed_future,
            run_root=run_root,
            state=pending_state,
        )
        self._last_checkpoint_iteration = iteration_index
        if force_checkpoint or completed_iterations == self.config.num_iterations:
            # Segment yield / final iteration: the caller is about to exit, so
            # the checkpoint must be durable before this returns.
            return self._finish_pending_checkpoint(wait=True)
        # The callback owns state publication while this checkpoint is pending.
        # Returning an older completed state here would let the caller overwrite
        # a newer callback commit if the upload finishes between this check and
        # the caller's write.
        return self._finish_pending_checkpoint(wait=False)

    def checkpoint_latest_iteration(self) -> None:
        """Make all completed work durable before reporting a segment yield."""
        if self._latest_completed_iteration is None:
            # No update in this process; a resumed run is already durable.
            return
        run_root, iteration_index, state = self._latest_completed_iteration
        self.finalize_iteration(
            run_root=run_root,
            iteration_index=iteration_index,
            train_result=None,
            common_run_state=state,
            force_checkpoint=True,
        )

    def _record_checkpoint_seconds(self, seconds: float) -> None:
        self.last_checkpoint_seconds = max(
            self.last_checkpoint_seconds or 0.0, float(seconds)
        )

    def _finish_pending_checkpoint(self, *, wait: bool) -> dict[str, Any] | None:
        pending = self._pending_checkpoint
        if pending is None or (not wait and not pending.future.done()):
            return None
        checkpoint_path = pending.future.result()
        self._pending_checkpoint = None
        return {
            **pending.state,
            "checkpoint_path": _relative_state_path(
                pending.run_root,
                checkpoint_path,
            ),
        }

    def shutdown(self, *, graceful: bool) -> None:
        if not graceful:
            return
        self._finish_pending_checkpoint(wait=True)
        self.runtime.close()
        if self._endpoint is not None:
            self._endpoint.close()
        if self._eval_endpoint is not None:
            self._eval_endpoint.close()
