"""CPU hook contracts: no Flash kernels/checkpoint required."""

from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import Mock, patch as mock_patch

import pytest
import torch

pytest.importorskip("triton")

from training.config import TrainingConfig
from training.flash_config import FlashSettings
from training.flash.dense_lora import LoRALinear
from training.flash_ppo import FlashPPOTrainer
from training.flash_backend import FlashPolicyTrainingBackend, FlashDistributedRuntime
from training.flash_policy_weights import read_policy_manifest
from training.policy_training_backend import TrainingResume
from training.ppo import PPOTrainer


def config():
    return TrainingConfig.from_dict({
        "model_name": "Qwen/Qwen3.8-Flash-Next", "weight_mode": "flash", "device": "cuda", "output_dir": "runs",
        "flash": {"checkpoint_root": "/checkpoint", "replicate_frozen_routers": False},
        "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": ["q_proj"]},
    })


def toy_trainer():
    trainer = FlashPPOTrainer.__new__(FlashPPOTrainer)
    trainer.config = config()
    layer = LoRALinear(torch.nn.Linear(4, 4), 2, 4)
    trainer.bundle = SimpleNamespace(model=layer, head=torch.nn.Linear(4, 8), parameter_groups={
        "custom": {"layers.0.self_attn.q_proj.lora_A": layer.lora_A,
                   "layers.0.self_attn.q_proj.lora_B": layer.lora_B}, "routers": {},
    })
    trainer.model = layer
    trainer.device = torch.device("cpu")
    trainer.optimizer = trainer._build_optimizer()
    trainer._optimizer_step_count = 0
    trainer._gradient_reducer = None
    return trainer


def test_optimizer_snapshot_and_checkpoint_resume(tmp_path):
    trainer = toy_trainer()
    expected = {id(p) for p in trainer.model.parameters() if p.requires_grad}
    assert {id(p) for g in trainer.optimizer.param_groups for p in g["params"]} == expected
    trainer.model(torch.ones(1, 4)).sum().backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    trainer._optimizer_step_count = 1
    snap = trainer._snapshot_update_state()
    trainer.model.lora_A.data.zero_()
    trainer._restore_update_state(snap)
    artifacts = trainer._finalize_window_artifacts(tmp_path, 0)
    clone = toy_trainer()
    clone._load_adapter(artifacts.adapter_dir)
    with mock_patch("torch.load", wraps=torch.load) as load:
        clone.load_training_state(artifacts.checkpoint_path)
    assert load.call_count == 1
    again = clone._save_latest_adapter(tmp_path / "resaved", 0)
    assert (again / "adapter_model.safetensors").read_bytes() == (artifacts.adapter_dir / "adapter_model.safetensors").read_bytes()
    assert clone._optimizer_step_count == 1
    for original, restored in zip(trainer.optimizer.state.values(), clone.optimizer.state.values(), strict=True):
        for key in original:
            assert torch.equal(original[key], restored[key])
    # Exact continuation, including optimizer moments.
    for instance in (trainer, clone):
        instance.model(torch.ones(1, 4)).sum().backward()
        instance.optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(trainer._trainable_parameters(), clone._trainable_parameters(), strict=True))


class TinyDecoder(torch.nn.Module):
    """A one-adapter decoder: hidden state = q_proj(one-hot token)."""

    def __init__(self):
        super().__init__()
        self.q_proj = LoRALinear(torch.nn.Linear(4, 4), 2, 4)

    def forward(self, *, input_ids, attention_mask, **kwargs):
        return SimpleNamespace(last_hidden_state=self.q_proj(torch.nn.functional.one_hot(input_ids % 4, 4).float()))


def decoder_trainer():
    trainer = toy_trainer()
    decoder = TinyDecoder()
    layer = decoder.q_proj
    trainer.bundle = SimpleNamespace(model=decoder, head=torch.nn.Linear(4, 8), parameter_groups={
        "custom": {"layers.0.self_attn.q_proj.lora_A": layer.lora_A,
                   "layers.0.self_attn.q_proj.lora_B": layer.lora_B}, "routers": {},
    })
    trainer.model = decoder
    trainer.optimizer = trainer._build_optimizer()
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    return trainer


def test_diagnostic_settings_stay_out_of_the_resume_identity():
    from dataclasses import asdict

    from training.flash_ppo import _recipe_identity

    saved = asdict(config().flash)
    for key in ("load_balance_metrics", "load_balance_eval_episodes"):
        saved.pop(key)  # a checkpoint from before these settings existed
    changed = replace(config().flash, load_balance_metrics=False, load_balance_eval_episodes=0)
    assert _recipe_identity(saved) == _recipe_identity(asdict(changed))
    assert _recipe_identity(asdict(replace(changed, decision_replay=True))) != _recipe_identity(saved)


def test_placement_changes_resume(tmp_path):
    """The heterogeneous job renumbers device_ids; offload and the checkpoint's location are placement."""
    from dataclasses import asdict

    from training.flash_ppo import _recipe_identity

    shared = toy_trainer()
    shared.config.flash = replace(shared.config.flash, device_ids=[4, 5, 6, 7])
    artifacts = shared._finalize_window_artifacts(tmp_path, 0)
    for placement in ({"device_ids": [0, 1, 2, 3]}, {"offload_every": 3}, {"checkpoint_root": "/scratch/copy"},
                      {"device_ids": [0, 1], "offload_every": 2, "checkpoint_root": "/elsewhere"}):
        moved = replace(shared.config.flash, **placement)
        assert _recipe_identity(asdict(moved)) == _recipe_identity(asdict(shared.config.flash))
        clone = toy_trainer()
        clone.config.flash = moved
        clone.load_training_state(artifacts.checkpoint_path)
        assert clone._optimizer_step_count == 0


@pytest.mark.parametrize("saved_keep", ["absent", 6, None, 3])
@pytest.mark.parametrize("keep", [6, None, 3])
def test_checkpoint_retention_may_change_on_resume(tmp_path, saved_keep, keep):
    trainer = toy_trainer()
    if saved_keep != "absent":
        trainer.config.checkpoint_keep_last = saved_keep
    artifacts = trainer._finalize_window_artifacts(tmp_path, 0)
    if saved_keep == "absent":
        state = torch.load(artifacts.checkpoint_path, weights_only=False)
        del state["training_config"]["checkpoint_keep_last"]
        torch.save(state, artifacts.checkpoint_path)
    clone = toy_trainer()
    clone.config.checkpoint_keep_last = keep
    clone.load_training_state(artifacts.checkpoint_path)
    assert clone.config.checkpoint_keep_last == keep


def test_recipe_refusal_names_each_differing_field_with_both_values(tmp_path):
    artifacts = toy_trainer()._finalize_window_artifacts(tmp_path, 0)
    clone = toy_trainer()
    # Placement changes beside the recipe change are not reported.
    clone.config.flash = replace(clone.config.flash, decision_replay=True, group_learning_rates={"custom": 0.002},
                                 device_ids=[4, 5, 6, 7], offload_every=3)
    with pytest.raises(ValueError) as refused:
        clone.load_training_state(artifacts.checkpoint_path)
    assert str(refused.value) == (
        "Flash checkpoint recipe differs from the configured recipe: "
        "decision_replay: checkpoint False, configured True; "
        "group_learning_rates: checkpoint {}, configured {'custom': 0.002}"
    )
    # A field only one side records is named as absent on the other.
    state = torch.load(artifacts.checkpoint_path, weights_only=False)
    state["flash_recipe"]["future_field"] = 1
    future = tmp_path / "future.pt"
    torch.save(state, future)
    with pytest.raises(ValueError, match=r"recipe: future_field: checkpoint 1, configured absent$"):
        toy_trainer().load_training_state(future)
    del state["flash_recipe"]
    torch.save(state, future)
    with pytest.raises(ValueError, match=r"recipe: the checkpoint records no Flash recipe \(None\)$"):
        toy_trainer().load_training_state(future)


def test_policy_weights_rounding_stays_out_of_the_resume_identity(tmp_path):
    """A segment may switch the servers' merge rounding by editing its frozen config."""
    from dataclasses import asdict

    from training.flash_ppo import _recipe_identity

    saved = asdict(config().flash)
    assert saved["policy_weights_rounding"] == "dither"
    nearest = asdict(replace(config().flash, policy_weights_rounding="nearest"))
    legacy = {key: value for key, value in saved.items() if key != "policy_weights_rounding"}
    assert _recipe_identity(saved) == _recipe_identity(nearest) == _recipe_identity(legacy)
    # A dithered run's state resumes under nearest rounding, and back.
    artifacts = toy_trainer()._finalize_window_artifacts(tmp_path / "dither", 0)
    clone = toy_trainer()
    clone.config.flash = replace(clone.config.flash, policy_weights_rounding="nearest")
    clone.load_training_state(artifacts.checkpoint_path)
    switched = clone._finalize_window_artifacts(tmp_path / "nearest", 1)
    assert read_policy_manifest(switched.publication_path)[0]["rounding"] == "nearest"
    toy_trainer().load_training_state(switched.checkpoint_path)


def test_states_written_with_the_retired_probe_resume_with_one_warning(tmp_path, caplog, monkeypatch):
    import training.flash_config as flash_config

    monkeypatch.setattr(flash_config, "_warned_retired", set())
    artifacts = toy_trainer()._finalize_window_artifacts(tmp_path, 0)
    state = torch.load(artifacts.checkpoint_path, weights_only=False)
    assert "adapter_probe_scale" not in state and "adapter_probe" not in state["flash_recipe"]
    # What a state written while the amplified-adapter probe existed carries.
    state["adapter_probe_scale"] = 7.25
    state["flash_recipe"]["adapter_probe"] = {"initial_scale": 8.0, "interval": 1}
    old = tmp_path / "probe_era_state.pt"
    torch.save(state, old)
    with caplog.at_level(logging.WARNING, logger="training.flash_config"):
        for _ in range(2):
            clone = toy_trainer()
            clone.load_training_state(old)
            assert clone._optimizer_step_count == 0
    warnings = [r.getMessage() for r in caplog.records if "retired setting" in r.getMessage()]
    assert len(warnings) == 1 and "training.flash.adapter_probe" in warnings[0] and str(old) in warnings[0]


def test_states_saved_with_nvfp4_expert_storage_resume_with_one_warning(tmp_path, caplog, monkeypatch):
    """The live run's states record "nvfp4"; the BF16 experts compute bitwise the same."""
    import training.flash_config as flash_config

    monkeypatch.setattr(flash_config, "_warned_retired", set())
    artifacts = toy_trainer()._finalize_window_artifacts(tmp_path, 0)
    state = torch.load(artifacts.checkpoint_path, weights_only=False)
    assert "expert_storage" not in state["flash_recipe"]
    state["flash_recipe"]["expert_storage"] = "nvfp4"
    old = tmp_path / "nvfp4_era_state.pt"
    torch.save(state, old)
    with caplog.at_level(logging.WARNING, logger="training.flash_config"):
        for _ in range(2):
            clone = toy_trainer()
            clone.load_training_state(old)
            assert clone._optimizer_step_count == 0
    warnings = [r.getMessage() for r in caplog.records if "retired setting" in r.getMessage()]
    assert len(warnings) == 1 and "training.flash.expert_storage" in warnings[0] and str(old) in warnings[0]
    state["flash_recipe"]["expert_storage"] = "fp8"
    torch.save(state, old)
    with pytest.raises(ValueError, match="training.flash.expert_storage must be one of"):
        toy_trainer().load_training_state(old)


def test_group_learning_rates_survive_schedule():
    trainer = toy_trainer()
    trainer.config.flash.group_learning_rates = {"custom": 0.002}
    trainer.config.warmup_steps = 4
    assert trainer._set_optimizer_learning_rate(2) == pytest.approx(0.001)


def test_mode_specific_parity():
    trainer = toy_trainer()
    trainer.config.flash.parity_thresholds.mean_abs = 0.2
    metrics = dict(token_count=128, mean_abs=0.1, p95_abs=0.1)
    assert not trainer._rollout_logprob_parity_failure(metrics, sample_count=1)
    # The base trainer (LoRA, full weight) keeps training.parity_thresholds.
    base = PPOTrainer.__new__(PPOTrainer)
    base.config = replace(trainer.config, weight_mode="lora")
    assert base._rollout_logprob_parity_failure(metrics, sample_count=1)


def _runtime_with(trainer, tmp_path):
    """A distributed runtime faked around one real trainer: rank zero's materializations hold on it."""
    runtime = Mock()
    runtime.trainer = trainer
    runtime._trainer = trainer
    runtime.publish_policy_weights.side_effect = trainer.publish_policy_weights
    return runtime


def test_factory_and_backend_lifecycle(tmp_path, nccl_transfer):
    trainer, _ = merged_trainer()
    runtime = _runtime_with(trainer, tmp_path)
    backend = FlashPolicyTrainingBackend(SimpleNamespace(training=config(), inference=SimpleNamespace(model_id=None)),
                                         runtime)
    resume = backend.resolve_resume(tmp_path, None)
    backend.create_trainer(resume=resume, game_vocab=[])
    assert runtime.create_trainer.call_args.kwargs["resume"] == TrainingResume()
    adapter = tmp_path / "checkpoints" / "adapter_iter_-00001"
    adapter.mkdir(parents=True)
    trainer.publish_policy_weights(adapter, -1)  # the start's _finalize_window_artifacts(-1)
    artifacts = SimpleNamespace(adapter_dir=adapter, publication_path=adapter,
                                checkpoint_path=tmp_path / "checkpoints" / "state.pt")
    artifacts.checkpoint_path.touch()
    runtime.initial_artifacts = artifacts
    collector = _Collector()
    backend.prepare_rollouts(collector, resume=resume, run_root=tmp_path)
    # Servers first, then the initial policy over NCCL at the (empty) boundary, then collection.
    assert collector.events == ["prepare", ("boundary", -1), "start"]
    (initial,) = nccl_transfer
    manifest, digest = read_policy_manifest(adapter)
    assert initial["manifest"] == manifest
    assert initial["label"] == {"kind": "policy_weights", "policy_sha256": digest, "policy_iteration": -1}
    assert [name for name, _ in initial["tensors"]] == manifest["names"]
    # Every server confirms the identity it resumed with, under the base name.
    assert [(url, identity["policy_sha256"], model) for url, identity, _, model in collector.confirmed] == [
        ("http://server0:1", digest, "Qwen/Qwen3.8-Flash-Next"), ("http://server1:1", digest, "Qwen/Qwen3.8-Flash-Next")]
    state = json.loads((tmp_path / "training_state.json").read_text())
    assert state["policy_sha256"] == digest and state["runtime_adapter_dir"] == state["adapter_dir"]
    # After a window: the held policy goes out at the boundary; the final one is dropped.
    trained = tmp_path / "checkpoints" / "adapter_iter_000000"
    trained.mkdir()
    trainer.publish_policy_weights(trained, 0)
    result = SimpleNamespace(adapter_dir=trained, runtime_adapter_dir=trained,
                             trainer_state_path=artifacts.checkpoint_path)
    assert backend.publish_after_train(collector, train_result=result, iteration_index=0, should_publish=True)
    assert nccl_transfer[-1]["label"]["policy_iteration"] == 0 and backend.published_policy_path == trained
    assert backend.publication_wait_seconds >= 0.0
    trainer.publish_policy_weights(trained, 0)
    assert backend.publish_after_train(collector, train_result=result, iteration_index=0, should_publish=False)
    assert len(nccl_transfer) == 2
    with pytest.raises(RuntimeError, match="no merged policy weights are held"):
        trainer.take_policy_weights()
    # A deadline that passes at the boundary sends nothing and releases the tensors.
    trainer.publish_policy_weights(trained, 0)
    late = _Collector(boundary=False)
    assert not backend.publish_after_train(late, train_result=result, iteration_index=0, should_publish=True,
                                           deadline_epoch=0.0)
    assert len(nccl_transfer) == 2
    with pytest.raises(RuntimeError, match="no merged policy weights are held"):
        trainer.take_policy_weights()
    backend.finalize_iteration(run_root=tmp_path, train_result=result,
                               common_run_state={"next_iteration": 1, "last_completed_iteration": 0,
                                                 "next_schedule_index": 1})
    state = json.loads((tmp_path / "training_state.json").read_text())
    assert state["weight_mode"] == "flash" and state["policy_sha256"] == read_policy_manifest(trained)[1]
    trainer.shutdown_workers = Mock()
    backend.shutdown()
    runtime.close.assert_called_once()
    trainer.shutdown_workers.assert_called_once()


@pytest.mark.parametrize("recorded", ["same", "other", None])
def test_every_resume_republishes_the_restored_adapter(tmp_path, nccl_transfer, caplog, recorded):
    """No publication outlives its job: a resume merges the restored adapter again, sends it to
    this segment's servers and records its identity, whatever the previous segment recorded."""
    trainer, _ = merged_trainer()
    run = tmp_path / "run"
    adapter = run / "checkpoints" / "adapter_iter_000048"
    adapter.mkdir(parents=True)
    (run / "checkpoints" / "trainer_state_iter_000048.pt").touch()
    trainer.publish_policy_weights(adapter, 48)
    _, digest, _ = _held(trainer)
    gone = tmp_path / "scratch-of-job-122" / "runtime_adapters" / "run" / "adapter_iter_000048" / "policy_weights"
    state = {"version": 1, "weight_mode": "flash", "next_iteration": 49, "last_completed_iteration": 48,
             "next_schedule_index": 49, "adapter_dir": "checkpoints/adapter_iter_000048",
             "runtime_adapter_dir": str(gone), "trainer_state_path": "checkpoints/trainer_state_iter_000048.pt"}
    if recorded is not None:
        state["policy_sha256"] = digest if recorded == "same" else "0" * 64
    runtime = _runtime_with(trainer, tmp_path)
    backend = FlashPolicyTrainingBackend(SimpleNamespace(training=config(), inference=SimpleNamespace(model_id="m")),
                                         runtime)
    collector = _Collector(servers=1)
    with caplog.at_level(logging.INFO, logger="training.flash_backend"):
        backend.prepare_rollouts(collector, resume=backend.resolve_resume(run, state), run_root=run)
    runtime.publish_policy_weights.assert_called_once_with(adapter, 48)
    (sent,) = nccl_transfer
    assert sent["label"] == {"kind": "policy_weights", "policy_sha256": digest, "policy_iteration": 48}
    written = json.loads((run / "training_state.json").read_text())
    assert (run / written["runtime_adapter_dir"]).resolve() == adapter.resolve()
    assert written["policy_sha256"] == digest and written["next_iteration"] == 49
    messages = [record.getMessage() for record in caplog.records if record.name == "training.flash_backend"]
    expected = {"same": "as recorded", "other": "differs from the recorded", None: "records none"}[recorded]
    assert any(expected in message for message in messages)
    assert any(re.search(r"published merged policy weights of iteration 48 .* transfer \d+\.\d\ds "
                         r"\(broadcast \d+\.\d\ds\), confirm", message) for message in messages)


def test_flash_runtime_factory(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("training.flash_ppo.FlashPPOTrainer", factory)
    FlashDistributedRuntime._default_trainer_factory(config="config")
    factory.assert_called_once_with(config="config")


def test_launch_flash(monkeypatch, tmp_path):
    from training import launch_frozen

    cfg = SimpleNamespace(training=config(),
                          inference=SimpleNamespace(server_allocation="shared", base_urls=[]))
    cfg.training.flash.device_ids = [2, 5]
    monkeypatch.setattr(launch_frozen.ExperimentConfig, "load", lambda _: cfg)
    monkeypatch.setenv("ARC3_FLASH_PYTHON", "/flash/bin/python")
    execute = Mock()
    monkeypatch.setattr(launch_frozen.os, "execvpe", execute)
    monkeypatch.setattr(launch_frozen.os, "chdir", Mock())
    paths = SimpleNamespace(run_root=tmp_path, live_repo_root=tmp_path, source_root=tmp_path,
                            frozen_config_path=tmp_path / "config.json", runtime_fingerprint_path=tmp_path / "fingerprint")
    launch_frozen._exec_training_from_snapshot(paths=paths, run_dir=None, argv0_python="/launcher/python")
    interpreter, command, env = execute.call_args.args
    assert interpreter == command[0] == "/flash/bin/python"
    assert "training.flash_entrypoint" in command and "--nproc-per-node=2" in command
    assert env["CUDA_VISIBLE_DEVICES"] == "2,5"
    monkeypatch.delenv("ARC3_FLASH_PYTHON")
    with pytest.raises(ValueError, match="ARC3_FLASH_PYTHON"):
        launch_frozen._exec_training_from_snapshot(paths=paths, run_dir=None, argv0_python="/launcher/python")


def test_flash_training_config_validation():
    from dataclasses import asdict

    payload = asdict(config())
    assert TrainingConfig.from_dict(payload).flash == FlashSettings.from_dict(payload["flash"])
    assert TrainingConfig.from_dict({**payload, "multimodal": True}).multimodal
    for patch in ({"learning_rate": 0}, {"worker_device_map_gpus": [[0]]},
                  {"lora": {"dropout": 0.05, "target_modules": ["q_proj"]}},
                  {"flash": {"checkpoint_root": "", "unknown": 1}},
                  {"flash": {"checkpoint_root": "weights"}},
                  {"flash": {"checkpoint_root": "/checkpoint", "attention_backend": "sdpa", "qsa_kernel": "group_sparse"}}):
        with pytest.raises(ValueError):
            TrainingConfig.from_dict({**payload, **patch})


def test_adapter_save_hook_keeps_peft_default(monkeypatch, tmp_path):
    save = Mock()
    monkeypatch.setattr("training.ppo._save_peft_adapter_without_model_card", save)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = Mock()
    trainer._write_policy_adapter(Path(tmp_path))
    save.assert_called_once_with(trainer.model, tmp_path)


@pytest.mark.parametrize("rank", [0, 1])
def test_torchrun_flash_entrypoint(monkeypatch, rank):
    from training import full_weight_entrypoint as entry
    from training import flash_backend

    cfg = SimpleNamespace(training=config())
    cfg.training.flash.device_ids = [3, 4]
    monkeypatch.setattr(entry.ExperimentConfig, "load", lambda _: cfg)
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setattr(entry.torch.cuda, "set_device", Mock())
    for name in ("init_process_group", "new_group", "barrier", "destroy_process_group"):
        monkeypatch.setattr(entry.dist, name, Mock())
    monkeypatch.setattr(entry.dist, "get_rank", lambda: rank)
    runtime = Mock()
    monkeypatch.setattr(flash_backend, "FlashDistributedRuntime", Mock(return_value=runtime))

    def train(path, *, backend_factory):
        assert isinstance(backend_factory(cfg), FlashPolicyTrainingBackend)

    run = Mock(side_effect=train)
    monkeypatch.setattr(entry, "run_training", run)
    assert entry.run("flash.json", weight_mode="flash") == 0
    if rank == 0:
        run.assert_called_once()
        runtime.close.assert_called_once()
    else:
        run.assert_not_called()
        runtime.serve.assert_called_once()


def test_token_scores_bias_and_rank_local_parity_scores(monkeypatch):
    trainer = toy_trainer()

    class Decoder(torch.nn.Module):
        def forward(self, *, input_ids, attention_mask, **kwargs):
            if kwargs:
                assert torch.equal(kwargs["position_ids"], torch.arange(4)[None])
            return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, 4).float())

    trainer.bundle.model = Decoder()
    trainer.model = trainer.bundle.model
    data = {"input_ids": torch.tensor([[0, 1, 2, 3]]), "attention_mask": torch.ones(1, 4),
            "labels": torch.tensor([[-100, -100, 2, 3]]),
            "logit_bias_token_ids": torch.tensor([2]), "logit_bias_values": torch.tensor([[[0.], [1.], [2.]]])}
    # Parity scores are never replaced by another rank's: the gate pools the
    # ranks' error statistics instead (test_two_rank_parity_failure_on_one_rank_fails_every_rank).
    broadcast = Mock()
    monkeypatch.setattr("training.flash_ppo.dist.is_initialized", lambda: True)
    monkeypatch.setattr("training.flash_ppo.dist.broadcast", broadcast)
    with torch.no_grad():
        actual, entropy = trainer._token_scores(data, compute_entropies=True)
    broadcast.assert_not_called()
    mask = data["labels"][:, 1:] != -100
    logits = trainer.bundle.head(torch.eye(4)[None, :-1][mask]).float()
    logits[:, 2] += torch.tensor([1., 2.])
    expected = logits.log_softmax(-1).gather(-1, torch.tensor([[2], [3]])).squeeze(-1)
    torch.testing.assert_close(actual[mask], expected, rtol=0, atol=1e-6)
    assert torch.isfinite(entropy).all()
    assert actual[0, 0] == 0
    trainer._token_scores({**data, "position_ids": torch.arange(4)[None]}, compute_entropies=False)
    with pytest.raises(ValueError, match="multimodal=true"):
        trainer._token_scores({**data, "pixel_values": torch.zeros(1)}, compute_entropies=False)


def _flash_parity_worker(rank, world_size, rendezvous_path, output_dir):
    import datetime

    import torch.distributed as dist

    from training.dataset import TokenizedSample, collate_samples
    from training.full_weight_ppo import DistributedCycleMixin

    dist.init_process_group("gloo", init_method=f"file://{rendezvous_path}", rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=60))
    try:
        assert FlashPPOTrainer._pool_parity_evidence is DistributedCycleMixin._pool_parity_evidence
        verdicts = {}
        for fault in ("none", "wrong", "nonfinite"):
            with torch.random.fork_rng():
                torch.manual_seed(0)  # the same toy policy on every rank
                trainer = decoder_trainer()
            ids = [1 + index % 7 for index in range(4 + 1500)]
            sample = TokenizedSample(
                trajectory_id="parity", step_index=0, input_ids=ids, labels=[-100] * 4 + ids[4:],
                attention_mask=[1] * len(ids), completion_mask=[0] * 4 + [1] * 1500, reward=1.0,
                return_to_go=1.0, advantage=1.0, prompt_token_count=4, old_token_logprobs=[0.0] * (len(ids) - 1),
            )
            with torch.no_grad():
                # The sampler's logprobs are the policy's own.
                sample.old_token_logprobs = trainer._token_logprobs(collate_samples([sample], pad_token_id=0))[0].tolist()
                if rank == 1 and fault == "wrong":
                    trainer.bundle.head.bias.add_(torch.arange(8.0))  # finite, wrong scores on rank one only
                elif rank == 1 and fault == "nonfinite":
                    trainer.bundle.head.bias[3] = float("nan")
            trainer._remember_parity_policy(-1)
            try:
                trainer._check_window_logprob_parity([sample], trainer_policy_iteration=-1, policy_lag_windows=0)
                verdicts[fault] = "passed"
            except RuntimeError as error:
                verdicts[fault] = str(error)
        report = [None] * world_size
        dist.all_gather_object(report, verdicts)
        assert report[0] == report[1], report
        assert verdicts["none"] == "passed"
        assert "logprob parity check failed" in verdicts["wrong"] and "outside tolerances" in verdicts["wrong"]
        assert "non-finite completion token" in verdicts["nonfinite"]
        Path(output_dir, f"flash_parity_rank_{rank}").write_text("done")
    finally:
        dist.destroy_process_group()


def test_two_rank_parity_failure_on_one_rank_fails_every_rank(tmp_path):
    """Rank one scores wrong (finite or not) while rank zero is exact: both ranks fail the gate."""
    from gloo_ranks import spawn_ranks

    spawn_ranks(_flash_parity_worker, 2, str(tmp_path / "parity_init"), str(tmp_path))
    assert all((tmp_path / f"flash_parity_rank_{rank}").is_file() for rank in (0, 1))


def test_two_rank_probe_states_broadcast_once_and_relay_rank_zero_errors(tmp_path):
    """Flash shares the full-weight trainer's probe protocol (DistributedCycleMixin)."""
    from test_full_weight_ppo import run_probe_protocol
    from training.full_weight_ppo import DistributedCycleMixin

    assert FlashPPOTrainer._ensure_probe_states is DistributedCycleMixin._ensure_probe_states
    run_probe_protocol(tmp_path, "training.flash_ppo:FlashPPOTrainer")


def test_builder_receives_flash_recipe(monkeypatch):
    import sys
    from types import ModuleType

    trainer = toy_trainer()
    builder = Mock(return_value=trainer.bundle)
    module = ModuleType("training.flash.model")
    module.build_flash_model = builder
    monkeypatch.setitem(sys.modules, "training.flash.model", module)
    assert trainer._load_policy_model() is trainer.bundle.model
    args, kwargs = builder.call_args
    assert args == ("/checkpoint",)
    assert kwargs["trainability"].mode == "custom"
    assert "distribution" not in kwargs  # the builder's own: BF16 experts, FSDP
    assert kwargs["qsa_kernel"] == "group_sparse"
    assert kwargs["checkpoint_policy"] == "small_gemm"
    assert kwargs["offload_every"] == 0
    assert trainer._ddp_group is None


def test_builder_receives_the_reference_stack(monkeypatch):
    import sys
    from types import ModuleType

    from training.flash.kernel_mode import builder_kwargs, resolve_kernel_stack

    trainer = toy_trainer()
    trainer.config.flash = FlashSettings.from_dict({"checkpoint_root": "/checkpoint", "kernel_mode": "reference"})
    builder = Mock(return_value=trainer.bundle)
    module = ModuleType("training.flash.model")
    module.build_flash_model = builder
    monkeypatch.setitem(sys.modules, "training.flash.model", module)
    trainer._load_policy_model()
    kwargs = builder.call_args.kwargs
    expected = builder_kwargs(resolve_kernel_stack("reference", packed=True))
    assert {name: kwargs[name] for name in expected} == expected
    assert (kwargs["attention_backend"], kwargs["checkpoint_policy"]) == ("sdpa", "full")


def test_multimodal_loader_keeps_frozen_vision_outside_text_model(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    from training import flash_model

    trainer = toy_trainer()
    trainer.config.multimodal = True
    trainer.config.flash.checkpoint_root = str(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"image_token_id": 7}))
    trainer._checkpoint_identity = {"processing_stack": {"model": "flash"}}
    module = ModuleType("training.flash.model")
    module.build_flash_model = Mock(return_value=trainer.bundle)
    monkeypatch.setitem(sys.modules, "training.flash.model", module)
    visual = torch.nn.Linear(4, 4).requires_grad_(False).eval()
    loader = Mock(return_value=visual)
    monkeypatch.setattr(flash_model, "load_flash_vision_tower", loader)
    policy = trainer._load_policy_model()
    decoder, head = trainer._decoder_and_lm_head()
    assert policy is trainer.bundle.model and head is trainer.bundle.head
    assert decoder.visual is visual and decoder._text_model is policy
    policy.train()
    assert not visual.training
    assert not {id(p) for p in visual.parameters()} & {id(p) for p in policy.parameters()}
    loader.assert_called_once_with(str(tmp_path), device=trainer.device)


def test_gradient_buckets_follow_reverse_decoder_order():
    trainer = toy_trainer()
    # Deliberately interleave groups/layers; module insertion order is not the
    # collective schedule. Parameters from a layer must share one bucket.
    parameters = {index: torch.nn.Parameter(torch.ones(2, 2)) for index in range(4)}
    trainer.bundle.parameter_groups = {
        "custom": {"layers.2.self_attn.q_proj.lora_A": parameters[0],
                   "layers.10.self_attn.q_proj.lora_A": parameters[1],
                   "layers.0.self_attn.q_proj.lora_A": parameters[2]},
        "routers": {"layers.2.router.weight": parameters[3]},
    }
    assert [[id(p) for p in bucket] for bucket in trainer._gradient_buckets()] == [
        [id(parameters[1])], [id(parameters[0]), id(parameters[3])], [id(parameters[2])],
    ]


def router_config(**flash):
    return TrainingConfig.from_dict({
        "model_name": "Qwen/Qwen3.8-Flash-Next", "weight_mode": "flash", "device": "cuda", "output_dir": "runs",
        "flash": {"checkpoint_root": "/checkpoint", "train_routers": True, **flash},
        "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": ["q_proj"]},
    })


class RouterToy(torch.nn.Module):
    """Layer 0 of the learner: a LoRA projection and an FP32 router master."""

    def __init__(self):
        super().__init__()
        self.q_proj = LoRALinear(torch.nn.Linear(4, 4), 2, 4)
        self.gate = torch.nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.q_proj(x)[..., :3] * self.gate(x).softmax(-1)


def router_trainer(**flash):
    trainer = FlashPPOTrainer.__new__(FlashPPOTrainer)
    trainer.config = router_config(**flash)
    # Identical frozen base weights in every toy: the router gradient depends on them.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = RouterToy()
    trainer.bundle = SimpleNamespace(model=model, head=torch.nn.Linear(4, 8), parameter_groups={
        "routers": {"layers.0.mlp.gate.weight": model.gate.weight},
        "custom": {"layers.0.self_attn.q_proj.lora_A": model.q_proj.lora_A,
                   "layers.0.self_attn.q_proj.lora_B": model.q_proj.lora_B},
    })
    trainer.model = model
    trainer.device = torch.device("cpu")
    trainer.optimizer = trainer._build_optimizer()
    trainer._optimizer_step_count = 0
    trainer._gradient_reducer = None
    return trainer


def test_train_routers_config_and_recipe_identity():
    from dataclasses import asdict
    from training.flash_ppo import _recipe_identity

    assert FlashSettings().train_routers is False
    settings = router_config(group_learning_rates={"routers": 1e-5}).flash
    assert settings.train_routers and settings.group_learning_rates == {"routers": 1e-5}
    assert "train_routers" not in settings.model_kwargs()
    payload = asdict(router_config())
    for patch in ({"flash": {**payload["flash"], "train_routers": 1}},
                  {"flash": {**payload["flash"], "train_routers": False, "group_learning_rates": {"routers": 1e-5}}},
                  {"weight_mode": "lora"},
                  {"reference_kl_coef": 0.01}):
        with pytest.raises(ValueError, match="train_routers"):
            TrainingConfig.from_dict({**payload, **patch})
    # A resume cannot flip it; recipes saved before the option trained no routers.
    recipe = asdict(config().flash)
    legacy = {key: value for key, value in recipe.items() if key != "train_routers"}
    assert _recipe_identity(legacy) == _recipe_identity(recipe) != _recipe_identity(asdict(settings))


def test_router_policy_composes_with_dense_lora():
    from training.flash.dense_lora import dense_lora_policy
    from training.flash.policies import TrainabilityPolicy

    policy = dense_lora_policy(2, 4, ["q_proj"], train_routers=True)
    assert policy.mode == "custom" and policy.routers_trainable
    assert not dense_lora_policy(2, 4, ["q_proj"]).routers_trainable
    assert TrainabilityPolicy.router_only().routers_trainable
    for kwargs in ({"mode": "routers", "train_routers": True}, {"mode": "frozen", "train_routers": True},
                   {"mode": "custom", "configure_layer": lambda layer, index: {}, "train_routers": True,
                    "router_backend": "stock"}):
        with pytest.raises(ValueError):
            TrainabilityPolicy(**kwargs)


def _held(trainer):
    """The publication rank zero holds for the broadcast: (manifest, identity, {name: tensor}); releases it."""
    manifest, digest, pairs = trainer.take_policy_weights()
    assert [name for name, _ in pairs] == manifest["names"]
    return manifest, digest, dict(pairs)


def test_router_publication_peft_exclusion_and_exact_resume(tmp_path):
    from safetensors.torch import load_file

    torch.manual_seed(0)
    trainer = router_trainer(group_learning_rates={"routers": 0.01})
    groups = {group["name"]: group for group in trainer.optimizer.param_groups}
    assert (groups["routers"]["lr"], groups["routers"]["weight_decay"]) == (0.01, 0.0)
    assert groups["custom"]["lr"] == trainer.config.learning_rate
    # The initial publication carries the starting routers unchanged.
    initial = trainer._finalize_window_artifacts(tmp_path / "initial", -1)
    assert torch.equal(load_file(initial.adapter_dir / "routers.safetensors")[
        "model.language_model.layers.0.mlp.gate.weight"], trainer.model.gate.weight.detach().bfloat16())
    assert torch.equal(_held(trainer)[2]["model.language_model.layers.0.mlp.gate.weight"],
                       trainer.model.gate.weight.detach().bfloat16())
    trainer.model.q_proj.lora_B.data.fill_(0.1)
    trainer.model(torch.ones(1, 4)).sum().backward()
    assert trainer.model.gate.weight.grad.abs().sum() > 0
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    trainer._optimizer_step_count = 1
    with torch.no_grad():
        # A master BF16 cannot represent: the state keeps it, the files its cast.
        trainer.model.gate.weight[0, 0] = 1 + 2**-12
    artifacts = trainer._finalize_window_artifacts(tmp_path, 0)
    adapter, publication = artifacts.adapter_dir, artifacts.publication_path
    data = (adapter / "routers.safetensors").read_bytes()
    manifest = json.loads((adapter / "routers.json").read_text())
    assert manifest == {"file": "routers.safetensors", "sha256": hashlib.sha256(data).hexdigest(),
                        "iteration": 0, "tensors": 1}
    # The servers receive the routers inside the merged policy weights; only the manifest is written.
    assert publication == adapter
    published, digest, tensors = _held(trainer)
    assert (published["iteration"], published["routers"]) == (0, True)
    assert read_policy_manifest(adapter) == (published, digest)
    assert set(tensors) == {"model.language_model.layers.0.mlp.gate.weight",
                            "model.language_model.layers.0.self_attn.q_proj.weight"}
    assert torch.equal(tensors["model.language_model.layers.0.mlp.gate.weight"],
                       trainer.model.gate.weight.detach().bfloat16())
    assert not any(path.suffix == ".safetensors" and path.name != "routers.safetensors"
                   and path.name != "adapter_model.safetensors" for path in adapter.iterdir())
    assert not any("gate" in key for key in load_file(adapter / "adapter_model.safetensors"))
    # A resume reads the BF16 routers with the adapter, then the exact FP32 masters.
    clone = router_trainer(group_learning_rates={"routers": 0.01})
    clone._load_adapter(adapter)
    assert torch.equal(clone.model.gate.weight, trainer.model.gate.weight.bfloat16().float())
    clone.load_training_state(artifacts.checkpoint_path)
    assert torch.equal(clone.model.gate.weight, trainer.model.gate.weight)
    assert clone.model.gate.weight[0, 0].item() == 1 + 2**-12
    # The restored policy publishes the same bytes, so the same identity (every resume republishes).
    assert clone.publish_policy_weights(adapter, 0) == adapter
    assert _held(clone)[1] == digest
    for instance in (trainer, clone):
        instance.model(torch.ones(1, 4)).sum().backward()
        instance.optimizer.step()
    for a, b in zip(trainer._trainable_parameters(), clone._trainable_parameters(), strict=True):
        assert torch.equal(a, b)
    # Publishing without a collective materialization is refused.
    with pytest.raises(RuntimeError, match="collective materialization"):
        trainer._save_runtime_adapter(adapter)


class MergedToy(torch.nn.Module):
    """Two decoder layers with BF16 frozen bases: attention q_proj/o_proj and a shared-expert down_proj."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(3)

        def adapted(out_features, in_features):
            base = torch.nn.Linear(in_features, out_features, bias=False).bfloat16().requires_grad_(False)
            base.weight.copy_(torch.randn(out_features, in_features, generator=generator))
            module = LoRALinear(base, 2, 4)
            module.lora_A.data.copy_(torch.randn(module.lora_A.shape, generator=generator))
            module.lora_B.data.copy_(0.1 * torch.randn(module.lora_B.shape, generator=generator))
            return module

        self.q_proj = torch.nn.ModuleList([adapted(6, 4), adapted(6, 4)])
        self.o_proj = adapted(4, 6)
        self.down_proj = adapted(4, 5)


def merged_trainer(**flash):
    trainer = FlashPPOTrainer.__new__(FlashPPOTrainer)
    trainer.config = TrainingConfig.from_dict({
        "model_name": "Qwen/Qwen3.8-Flash-Next", "weight_mode": "flash", "device": "cuda", "output_dir": "runs",
        "flash": {"checkpoint_root": "/checkpoint", **flash},
        "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": ["q_proj", "o_proj", "down_proj"]},
    })
    model = MergedToy()
    modules = {"layers.0.self_attn.q_proj": model.q_proj[0], "layers.1.self_attn.q_proj": model.q_proj[1],
               "layers.0.self_attn.o_proj": model.o_proj, "layers.1.mlp.shared_expert.down_proj": model.down_proj}
    custom = {f"{name}.lora_{part}": getattr(module, f"lora_{part}") for name, module in modules.items()
              for part in "AB"}
    trainer.bundle = SimpleNamespace(model=model, head=torch.nn.Linear(4, 8),
                                     parameter_groups={"custom": custom, "routers": {}})
    trainer.model = model
    trainer.device = torch.device("cpu")
    return trainer, modules


def test_publication_merges_every_target_once_and_carries_its_checksums(tmp_path):
    from training.flash_policy_weights import manifest_sha256, tensor_checksums

    trainer, modules = merged_trainer()
    adapter_dir = tmp_path / "checkpoints" / "adapter_iter_000004"
    adapter_dir.mkdir(parents=True)
    assert trainer.publish_policy_weights(adapter_dir, 4) == adapter_dir
    # Only the manifest is written; the tensors wait on rank zero for the broadcast.
    assert sorted(path.name for path in adapter_dir.iterdir()) == ["policy_weights.json"]
    assert not (tmp_path / "checkpoints" / "latest_runtime_adapter").exists()
    manifest, digest, tensors = _held(trainer)
    assert read_policy_manifest(adapter_dir) == (manifest, digest) and digest == manifest_sha256(manifest)
    assert manifest["iteration"] == 4
    assert manifest["routers"] is False and manifest["lora_targets"] == ["q_proj", "o_proj", "down_proj"]
    assert manifest["rounding"] == "dither"
    # Broadcast order: layer, then module name.
    assert manifest["names"] == list(tensors) == [
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.1.mlp.shared_expert.down_proj.weight",
        "model.language_model.layers.1.self_attn.q_proj.weight",
    ]
    assert manifest["shapes"] == [list(value.shape) for value in tensors.values()]
    from training.flash_policy_weights import dither_seed, merge_lora_weight

    for name, module in modules.items():
        layer, _, path = name.removeprefix("layers.").partition(".")
        hf_name = f"model.language_model.layers.{layer}.{path}.weight"
        merged = tensors[hf_name]
        # Each target merged once, with its own fixed dither.
        expected = merge_lora_weight(module.base.weight, module.lora_A, module.lora_B, 2.0, seed=dither_seed(hf_name))
        assert torch.equal(merged, expected), name
        exact = module.base.weight.float() + 2.0 * (module.lora_B @ module.lora_A)
        assert float((merged.float() - exact).abs().max()) <= float(exact.abs().max()) * 2**-7
        assert not torch.equal(merged, module.base.weight)
    # The learner's device-side checksums travel in the manifest.
    assert manifest["checksums"] == tensor_checksums(tensors.values())
    # A held publication is sent once; a rewrite of the same policy has the same identity.
    with pytest.raises(RuntimeError, match="no merged policy weights are held"):
        trainer.take_policy_weights()
    trainer.publish_policy_weights(adapter_dir, 4)
    assert _held(trainer)[1] == digest
    trainer.publish_policy_weights(adapter_dir, 4)
    trainer.release_policy_weights()
    with pytest.raises(RuntimeError, match="no merged policy weights are held"):
        trainer.take_policy_weights()


def test_nearest_rounding_publishes_the_plain_cast_of_each_merge(tmp_path):
    trainer, modules = merged_trainer(policy_weights_rounding="nearest")
    adapter_dir = tmp_path / "checkpoints" / "adapter_iter_000004"
    adapter_dir.mkdir(parents=True)
    trainer.publish_policy_weights(adapter_dir, 4)
    manifest, _, tensors = _held(trainer)
    assert manifest["rounding"] == "nearest"
    with torch.no_grad():
        for name, module in modules.items():
            layer, _, path = name.removeprefix("layers.").partition(".")
            exact = torch.add(module.base.weight.float(), module.lora_B @ module.lora_A, alpha=2.0)
            assert torch.equal(tensors[f"model.language_model.layers.{layer}.{path}.weight"], exact.bfloat16()), name


def test_trainer_state_records_the_publication_identity(tmp_path):
    trainer = toy_trainer()
    artifacts = trainer._finalize_window_artifacts(tmp_path, 0)
    _, digest = read_policy_manifest(artifacts.adapter_dir)
    state = torch.load(artifacts.checkpoint_path, weights_only=False)
    assert state["flash_policy_sha256"] == digest
    assert state["runtime_adapter_dir"] == str(artifacts.adapter_dir)
    assert _held(trainer)[1] == digest


class _Server:
    """A rollout collector's server: its URL, auth and the confirmation after each publication."""

    def __init__(self, index, confirmed):
        self.config = SimpleNamespace(inference=SimpleNamespace(base_url=f"http://server{index}:1"))
        self.confirmed = confirmed

    def _server_auth_headers(self):
        return {"Authorization": "Bearer k"}

    def confirm_policy_weights(self, identity, *, label, model_id):
        self.confirmed.append((self.config.inference.base_url, identity, label, model_id))


class _Collector:
    def __init__(self, servers=2, boundary=True):
        self.confirmed, self.events = [], []
        self.collectors = [_Server(index, self.confirmed) for index in range(servers)]
        self.boundary = boundary

    def prepare_servers(self):
        self.events.append("prepare")

    def activate_policy_at_boundary(self, *, policy_iteration, activate, deadline_epoch=None):
        self.events.append(("boundary", policy_iteration))
        if not self.boundary:
            return False
        activate()
        return True

    def start(self):
        self.events.append("start")


@pytest.fixture
def nccl_transfer(monkeypatch):
    """The NCCL control plane, faked: what each transfer sent, and the identity each server resumed."""
    sent = []

    class Endpoint:
        initialized = False

        def __init__(self, *, device):
            assert device == torch.device("cpu")

        def send_weights(self, parameters):
            sent[-1]["tensors"] = [(name, tensor.clone()) for name, tensor in parameters]

    class Client:
        def __init__(self, url, *, auth_headers):
            self.url, self.adapter_identity = url, None

    class Transfer:
        def __init__(self, clients, *, trainer, packed):
            assert packed is False and isinstance(trainer, Endpoint)
            self.clients = clients

        def transfer(self, coordinates, *, send_weights, policy):
            label, manifest = policy
            assert list(coordinates.names) == manifest["names"]
            sent.append({"label": label, "manifest": manifest})
            send_weights()
            for client in self.clients:
                client.adapter_identity = {"name": "base", "id": 0, "path": None,
                                           "server_instance_id": client.url, **label}
            return label

    monkeypatch.setattr("training.flash_backend.StatelessNcclTrainerEndpoint", Endpoint)
    monkeypatch.setattr("training.flash_backend.VllmControlClient", Client)
    monkeypatch.setattr("training.flash_backend.VllmNcclWeightTransfer", Transfer)
    return sent


def test_router_recipe_flips_are_refused(tmp_path):
    routed = router_trainer()._finalize_window_artifacts(tmp_path / "routed", 0)
    plain = toy_trainer()
    with pytest.raises(ValueError, match="carries trained routers"):
        plain._load_adapter(routed.adapter_dir)
    with pytest.raises(ValueError, match="recipe differs"):
        plain.load_training_state(routed.checkpoint_path)
    unrouted = plain._finalize_window_artifacts(tmp_path / "plain", 0)
    assert not (unrouted.adapter_dir / "routers.json").exists()
    assert not (unrouted.publication_path / "routers.json").exists()
    # Without trained routers the servers keep the checkpoint's.
    assert not any(name.endswith("mlp.gate.weight") for name in read_policy_manifest(unrouted.publication_path)[0]["names"])
    with pytest.raises(ValueError, match="lacks routers.safetensors"):
        router_trainer()._load_adapter(unrouted.adapter_dir)
    with pytest.raises(ValueError, match="recipe differs"):
        router_trainer().load_training_state(unrouted.checkpoint_path)
    # States written before the option resume as LoRA-only runs.
    state = torch.load(unrouted.checkpoint_path, weights_only=False)
    del state["flash_recipe"]["train_routers"]
    legacy = tmp_path / "legacy.pt"
    torch.save(state, legacy)
    toy_trainer().load_training_state(legacy)


def test_lagged_parity_scores_with_the_sampler_routers():
    trainer = router_trainer()
    trainer._remember_parity_policy(0)
    assert "gate.weight" in trainer._parity_policy_snapshots[0]
    sampled = trainer.model.gate.weight.detach().clone()
    with torch.no_grad():
        trainer.model.gate.weight.add_(1.0)
    current = trainer.model.gate.weight.detach().clone()
    trainer._remember_parity_policy(1)
    seen = []

    def check(samples, *, trainer_policy_iteration):
        seen.append((trainer_policy_iteration, trainer.model.gate.weight.detach().clone()))
        return {"token_count": 1, "mean_abs": 0.0}

    trainer._check_rollout_logprob_parity = check
    trainer._rollout_logprob_parity_failure = lambda metrics, sample_count: None
    trainer._check_window_logprob_parity([SimpleNamespace(policy_iteration=0)],
                                         trainer_policy_iteration=1, policy_lag_windows=1)
    assert seen[0][0] == 0 and torch.equal(seen[0][1], sampled)
    assert torch.equal(trainer.model.gate.weight, current)


def test_recipe_identity_ignores_gate_settings_and_the_retired_settings():
    from training.flash_ppo import _recipe_identity

    saved = {"checkpoint_root": "/c", "parity_thresholds": {"mean_abs": 0.17}, "adapter_probe": {"interval": 1}}
    configured = {"checkpoint_root": "/c", "parity_thresholds": {"mean_abs": 0.25}}
    assert _recipe_identity(saved) == _recipe_identity(configured)
    for storage in ("nvfp4", "bf16"):
        assert _recipe_identity({**saved, "expert_storage": storage}) == _recipe_identity(configured)
    # Placement is not recipe (test_placement_changes_resume); the stack is.
    for placement in ({"offload_every": 2}, {"device_ids": [4, 5, 6, 7]}, {"checkpoint_root": "/d"}):
        assert _recipe_identity({**saved, **placement}) == _recipe_identity(configured)
    assert _recipe_identity({**saved, "kernel_mode": "reference"}) != _recipe_identity(configured)
    assert _recipe_identity(None) is None


class CountingRouter(torch.nn.Module):
    """Top-2 of 4 experts over one-hot token states; counts its own calls."""

    layer_idx, num_experts, top_k = 0, 4, 2

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        values, indices = (hidden.reshape(-1, 4).float() * 3).softmax(-1).topk(2, -1)
        recorder = getattr(self, "route_recorder", None)
        if recorder is not None:
            recorder.record(self.route_layer, indices, values)
        return values


class RoutedDecoder(TinyDecoder):
    """TinyDecoder with a router inside an activation-checkpointed layer."""

    def __init__(self):
        super().__init__()
        self.gate = CountingRouter()

    def _layer(self, hidden):
        self.gate(hidden)
        return hidden * hidden.sigmoid()

    def forward(self, *, input_ids, attention_mask, **kwargs):
        hidden = self.q_proj(torch.nn.functional.one_hot(input_ids % 4, 4).float())
        hidden = torch.utils.checkpoint.checkpoint(self._layer, hidden, use_reentrant=False)
        return SimpleNamespace(last_hidden_state=hidden)


def routed_sample(name, ids, completion):
    from training.dataset import TokenizedSample

    return TokenizedSample(
        trajectory_id=name, step_index=0, input_ids=ids,
        labels=[t if m else -100 for t, m in zip(ids, completion)], attention_mask=[1] * len(ids),
        completion_mask=completion, reward=1.0, return_to_go=1.0, advantage=0.5,
        old_token_logprobs=[-1.0] * (len(ids) - 1),
    )


def test_expert_load_counts_each_trained_token_once(tmp_path):
    from training.expert_load import EVAL_HANDOFF_DIR, install_expert_load_recorder

    trainer = decoder_trainer()
    decoder = RoutedDecoder()
    layer = decoder.q_proj
    trainer.bundle = SimpleNamespace(model=decoder, head=torch.nn.Linear(4, 8), parameter_groups={
        "custom": {"layers.0.self_attn.q_proj.lora_A": layer.lora_A,
                   "layers.0.self_attn.q_proj.lora_B": layer.lora_B}, "routers": {},
    })
    trainer.model = decoder
    trainer.optimizer = trainer._build_optimizer()
    trainer._expert_load = install_expert_load_recorder(decoder, router_type=CountingRouter, device="cpu")
    long = routed_sample("a", [1, 2, 3, 5, 6, 7], [0, 0, 1, 1, 0, 1])
    short = routed_sample("b", [1, 2, 3, 4], [0, 1, 1, 1])
    trainer._micro_batch_forward_backward([long, short], loss_denominator=8.0)
    # Backward recomputed the checkpointed router; only the forward counted.
    assert decoder.gate.calls == 2
    counts = trainer._expert_load.counts
    assert counts[0, 0].sum() == 2 * (6 + 4)  # train: every real token, top-2
    assert counts[1, 0].sum() == 2 * (3 + 3)  # rollout: positions deciding a sampled token
    # The same routing, sample by sample, without the padded batch.
    expected = torch.zeros(4, dtype=torch.float64)
    with torch.no_grad():
        for ids in (long.input_ids, short.input_ids):
            hidden = layer(torch.nn.functional.one_hot(torch.tensor(ids) % 4, 4).float())
            expected += torch.bincount((hidden * 3).softmax(-1).topk(2, -1).indices.flatten(), minlength=4)
    torch.testing.assert_close(counts[0, 0], expected)
    # An FSDP padding slot (zero backward weight) trains nothing and counts nothing.
    trainer._micro_batch_forward_backward([long], loss_denominator=8.0, backward_scale=0.0)
    assert counts[0, 0].sum() == 20 and counts[2].sum() == 0
    # The eval pass scores the handed-over episodes' sampled positions, no grad.
    (tmp_path / EVAL_HANDOFF_DIR).mkdir()
    torch.save({"version": 1, "cycle_index": 0, "policy_iteration": 0, "episodes": 2,
                "samples": [long, short]}, tmp_path / EVAL_HANDOFF_DIR / "cycle_00000.pt")
    trainer._window_expert_load = {}
    trainer._finalize_window_artifacts(tmp_path / "checkpoints", 0)
    summary = trainer._window_expert_load
    assert (summary["train"]["tokens"], summary["rollout"]["tokens"], summary["eval"]["tokens"]) == (10, 6, 6)
    assert summary["eval_pass"]["eval_episodes"] == 2
    assert not list((tmp_path / EVAL_HANDOFF_DIR).iterdir())
    assert trainer._expert_load.counts.sum() == 0
    # Without a handoff the eval role is simply absent.
    trainer._finalize_window_artifacts(tmp_path / "checkpoints", 1)
    assert trainer._window_expert_load == {}
