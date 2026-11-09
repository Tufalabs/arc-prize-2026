"""Flash's distributed command runtime and merged-weight publication lifecycle.

Every publication goes from the learner's rank zero straight into the
servers' GPU memory over NCCL (training/policy_sync/stateless_nccl.py), at a
rollout boundary, through the servers' /weights/* routes
(training/policy_sync/vllm_nccl.py); the servers verify every tensor against
the manifest's checksums (training/flash_policy_weights.py). No publication
bytes touch disk: the adapter checkpoint stays the durable policy, with the
publication's manifest beside it.
"""

import copy
import logging
from pathlib import Path
import time

from training.flash_policy_weights import read_policy_manifest
from training.full_weight_distributed import FullWeightDistributedRuntime
from training.memory_profile import NullMemoryLogger
from training.policy_sync.stateless_nccl import StatelessNcclTrainerEndpoint
from training.policy_sync.vllm_nccl import VllmControlClient, VllmNcclWeightTransfer, WeightManifest
from training.policy_training_backend import TrainingResume, _relative_state_path, _resolve_state_path
from training.run_artifacts import write_json_atomic

log = logging.getLogger(__name__)

_CMD_PUBLISH_POLICY_WEIGHTS = "flash_publish_policy_weights"


class FlashDistributedRuntime(FullWeightDistributedRuntime):
    @staticmethod
    def _default_trainer_factory(**kwargs):
        from training.flash_ppo import FlashPPOTrainer

        return FlashPPOTrainer(**kwargs)

    def create_trainer(self, training_config, *, resume, run_root, memory_logger=None, **kwargs):
        self._rank_zero_memory_logger = memory_logger or NullMemoryLogger()
        command = {"type": "start", "training_config": training_config, "resume": resume,
                   "run_root": str(run_root), "trainer_kwargs": kwargs, "rng_state": self._capture_rng_state()}
        self._broadcast(command)
        return self._run_symmetric(lambda: self._handle_start(command))

    def _handle_start(self, command):
        self._restore_rng_state(command["rng_state"])
        config = copy.deepcopy(command["training_config"])
        config.device = self._local_device(config)
        resume = command["resume"]
        self._trainer = self._trainer_factory(
            config=config, resume_adapter_dir=resume.adapter_dir,
            resume_state_path=resume.trainer_state_path,
            memory_logger=self._rank_zero_memory_logger if self.rank == 0 else NullMemoryLogger(),
            **command["trainer_kwargs"],
        )
        if resume.start_iteration == 0 and resume.adapter_dir is None:
            self.initial_artifacts = self.trainer._finalize_window_artifacts(
                Path(command["run_root"]) / "checkpoints", -1,
            )
        return self.trainer

    def publish_policy_weights(self, adapter_dir, iteration):
        """Materialize the current (just restored) policy's merged weights on every rank; rank zero holds them."""
        command = {"type": _CMD_PUBLISH_POLICY_WEIGHTS, "adapter_dir": str(adapter_dir),
                   "iteration": int(iteration)}
        self._broadcast(command)
        return self._run_symmetric(lambda: self._handle_publish_policy_weights(command))

    def _handle_publish_policy_weights(self, command):
        return self.trainer.publish_policy_weights(Path(command["adapter_dir"]), command["iteration"])

    def _serve_extension_command(self, command):
        if command.get("type") != _CMD_PUBLISH_POLICY_WEIGHTS:
            return False
        self._run_symmetric(lambda: self._handle_publish_policy_weights(command))
        return True


class FlashPolicyTrainingBackend:
    refresh_strategy = "vllm_nccl"
    last_checkpoint_seconds = None
    publication_wait_seconds = 0.0

    def __init__(self, config, runtime):
        self.config = config
        self.runtime = runtime
        self.published_policy_path = None
        self._endpoint = None
        self._transfer = None
        self._clients = []
        self._collectors = []

    @property
    def trainer(self):
        return self.runtime.trainer

    def resolve_resume(self, run_root, state):
        self.run_root = run_root
        self._resume_state = copy.deepcopy(state)
        if state is None:
            if self.config.training.init_adapter_dir or self.config.training.init_trainer_state_path:
                raise ValueError("Flash warm starts must use a Flash run checkpoint (--run-dir)")
            return TrainingResume()
        if state.get("version") != 1 or state.get("weight_mode") != "flash":
            raise ValueError("Flash resume requires a Flash version 1 run state")
        paths = {key: _resolve_state_path(run_root, state[key])
                 for key in ("adapter_dir", "runtime_adapter_dir", "trainer_state_path")}
        if not paths["adapter_dir"].is_dir() or not paths["trainer_state_path"].is_file():
            raise ValueError("Flash resume checkpoint artifacts are missing")
        start = int(state["next_iteration"])
        last = int(state["last_completed_iteration"])
        if start != last + 1:
            raise ValueError("Flash resume iteration fields are inconsistent")
        return TrainingResume(start_iteration=start, last_completed_iteration=last,
                              next_schedule_index=int(state["next_schedule_index"]), **paths)

    def create_trainer(self, *, resume, **kwargs):
        return self.runtime.create_trainer(self.config.training, resume=resume,
                                           run_root=self.run_root, **kwargs)

    def pin_checkpoint_identity(self, reported):
        """Pin a window's harness agent settings (training/rollout_identity.py)."""
        self.runtime.pin_checkpoint_identity(reported)

    def prepare_rollouts(self, collector, *, resume, run_root, official_eval_process=None):
        if resume.adapter_dir is None:
            artifacts = self.runtime.initial_artifacts
            adapter_dir = artifacts.adapter_dir
            state = {
                "version": 1, "weight_mode": "flash", "next_iteration": 0,
                "last_completed_iteration": -1, "next_schedule_index": 0,
                "adapter_dir": _relative_state_path(run_root, adapter_dir),
                "runtime_adapter_dir": _relative_state_path(run_root, artifacts.publication_path),
                "trainer_state_path": _relative_state_path(run_root, artifacts.checkpoint_path),
                "policy_sha256": read_policy_manifest(adapter_dir)[1],
            }
            write_json_atomic(run_root / "training_state.json", state)
            write_json_atomic(run_root / "initial_policy.json", {
                "last_completed_iteration": -1, "initial_adapter_dir": str(adapter_dir.resolve()),
                "runtime_adapter_dir": str(artifacts.publication_path.resolve()), "resume_state": state,
            })
        else:
            # No publication outlives its job: every segment publishes the
            # restored policy (that adapter's) from the adapter checkpoint.
            adapter_dir, iteration = resume.adapter_dir, resume.last_completed_iteration
            recorded = self._resume_state.get("policy_sha256")
            self.runtime.publish_policy_weights(adapter_dir, iteration)
            _, digest = read_policy_manifest(adapter_dir)
            if recorded is None:
                log.info("republishing %s for iteration %s: identity %s (the run state records none)",
                         adapter_dir, iteration, digest)
            elif recorded == digest:
                log.info("republishing %s for iteration %s: identity %s, as recorded", adapter_dir, iteration, digest)
            else:
                log.warning("republishing %s for iteration %s: identity %s differs from the recorded %s (the "
                            "merge did not reproduce bit for bit); serving the new one", adapter_dir, iteration,
                            digest, recorded)
            write_json_atomic(run_root / "training_state.json", {
                **self._resume_state,
                "runtime_adapter_dir": _relative_state_path(run_root, adapter_dir),
                "policy_sha256": digest,
            })
        collector.prepare_servers()
        self._collectors = list(collector.collectors)
        self._clients = [
            VllmControlClient(server.config.inference.base_url, auth_headers=server._server_auth_headers())
            for server in self._collectors
        ]
        # One group for the run (this learner process): rank zero and every
        # server's workers; a server that restarts is fatal (its instance id
        # no longer matches), as is any failed publication.
        self._endpoint = StatelessNcclTrainerEndpoint(device=self.trainer.device)
        self._transfer = VllmNcclWeightTransfer(self._clients, trainer=self._endpoint, packed=False)
        collector.activate_policy_at_boundary(policy_iteration=resume.last_completed_iteration,
                                              activate=self._publish_held_policy)
        self.published_policy_path = adapter_dir
        collector.start()

    def _publish_held_policy(self):
        """Broadcast rank zero's held publication to every server; each verifies it, then serves it."""
        manifest, digest, tensors = self.trainer.take_policy_weights()
        label = {"kind": "policy_weights", "policy_sha256": digest, "policy_iteration": manifest["iteration"]}
        coordinates = WeightManifest(tuple(manifest["names"]), ("bfloat16",) * len(manifest["names"]),
                                     tuple(tuple(shape) for shape in manifest["shapes"]))
        started_at = time.perf_counter()
        initialized = self._endpoint.initialized
        sent = {}

        def send_weights():
            begun = time.perf_counter()
            self._endpoint.send_weights(tensors)
            sent["seconds"] = time.perf_counter() - begun

        self._transfer.transfer(coordinates, send_weights=send_weights, policy=(label, manifest))
        transferred_at = time.perf_counter()
        model_id = self.config.inference.model_id or self.config.training.model_name
        for server, client in zip(self._collectors, self._clients, strict=True):
            server.confirm_policy_weights(client.adapter_identity, label=label, model_id=model_id)
        log.info(
            "published merged policy weights of iteration %s (identity %s, %d tensors) to %d server(s): "
            "%stransfer %.2fs (broadcast %.2fs), confirm %.2fs", manifest["iteration"], digest,
            len(manifest["names"]), len(self._clients),
            "" if initialized else "group initialization included, ",
            transferred_at - started_at, sent.get("seconds", float("nan")), time.perf_counter() - transferred_at,
        )

    def train_window(self, rollouts, **kwargs):
        return self.runtime.train_window(rollouts, **kwargs)

    def publish_after_train(self, collector, *, train_result, iteration_index, should_publish,
                            deadline_epoch=None):
        self.publication_wait_seconds = 0.0
        if not should_publish:
            self.trainer.release_policy_weights()
            return deadline_epoch is None or time.time() < deadline_epoch
        started_at = time.perf_counter()

        def activate():
            # Only the wait for the rollout boundary is idle.
            self.publication_wait_seconds = time.perf_counter() - started_at
            self._publish_held_policy()

        if not collector.activate_policy_at_boundary(policy_iteration=iteration_index, activate=activate,
                                                     deadline_epoch=deadline_epoch):
            self.trainer.release_policy_weights()
            return False
        self.published_policy_path = train_result.adapter_dir
        return deadline_epoch is None or time.time() < deadline_epoch

    def finalize_iteration(self, *, run_root, train_result, common_run_state, **kwargs):
        state = {
            **common_run_state, "version": 1, "weight_mode": "flash",
            "adapter_dir": _relative_state_path(run_root, train_result.adapter_dir),
            "runtime_adapter_dir": _relative_state_path(run_root, train_result.runtime_adapter_dir),
            "trainer_state_path": _relative_state_path(run_root, train_result.trainer_state_path),
            "policy_sha256": read_policy_manifest(train_result.adapter_dir)[1],
        }
        write_json_atomic(run_root / "training_state.json", state)
        return state

    def checkpoint_latest_iteration(self):
        pass  # Each window synchronously commits its adapter and optimizer state.

    def shutdown(self, *, graceful=True):
        if self._endpoint is not None:
            self._endpoint = None
        if graceful:
            self.runtime.close()
        if self.runtime._trainer is not None:
            self.trainer.shutdown_workers()
