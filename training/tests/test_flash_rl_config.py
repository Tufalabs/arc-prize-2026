from dataclasses import asdict

import pytest

from training.flash_config import FlashSettings, ParityThresholds
from training.flash.lora_targets import validate_target_modules

# Flash defaults sit above the serving engine's own per-kind floor; see
# training/FLASH_RL.md "Parity thresholds".
FLASH_PARITY = {"mean_abs": 0.25, "p95_abs": 1.0, "outlier_abs": 2.0, "max_outlier_fraction": 0.01,
                "marginal_band_factor": 1.5}
LORA_PARITY = {"mean_abs": 0.05, "p95_abs": 0.25, "outlier_abs": 2.0, "max_outlier_fraction": 0.001,
               "marginal_band_factor": 1.5}


def test_recipe_defaults():
    config = FlashSettings.from_dict({})
    assert config.device_ids == [0, 1, 2, 3]
    assert config.qsa_kernel == "group_sparse"
    assert "expert_storage" not in asdict(config)  # experts are always BF16
    assert config.checkpoint_policy == "small_gemm"
    assert config.parity_thresholds == ParityThresholds.flash()
    assert config.policy_weights_rounding == "dither"


def test_policy_weights_rounding_choices():
    for rounding in ("dither", "nearest"):
        assert FlashSettings.from_dict({"policy_weights_rounding": rounding}).policy_weights_rounding == rounding


@pytest.mark.parametrize("payload", [
    {"typo": True}, {"offload_every": True}, {"offload_every": -1},
    {"qsa_kernel": "physical_sparse"}, {"compile_layers": 1}, {"expert_storage": "fp8"}, {"device_ids": [0, 0]},
    {"device_ids": []}, {"parity_thresholds": {"mean_abs": float("nan")}},
    {"parity_thresholds": {"max_outlier_fraction": 2}},
    {"decision_replay": True, "attention_backend": "sdpa"},
    {"attention_backend": "sdpa", "qsa_kernel": "group_sparse"},
    {"group_learning_rates": {"custom": -1}},
    {"load_balance_metrics": 1}, {"load_balance_eval_episodes": -1}, {"load_balance_eval_episodes": 2.0},
    {"policy_weights_rounding": "stochastic"}, {"policy_weights_rounding": None},
    {"policy_weights_rounding": "Nearest"},
])
def test_invalid_recipe(payload):
    with pytest.raises(ValueError):
        FlashSettings.from_dict(payload)


@pytest.mark.parametrize("targets", [[], ["gate"], ["q_proj", "q_proj"], ["experts.gate_proj"]])
def test_invalid_targets(targets):
    with pytest.raises(ValueError, match="target_modules"):
        validate_target_modules(targets)


def test_flash_config_accepts_optional_gdn_targets():
    # The runtime converter allowlists in_proj_b/in_proj_a and the E1 probe served them.
    from training.config import TrainingConfig

    targets = ["in_proj_qkv", "in_proj_b", "in_proj_a"]
    config = TrainingConfig.from_dict({
        "model_name": "Qwen/Qwen3.8-Flash-Next", "weight_mode": "flash", "device": "cuda", "output_dir": "runs",
        "flash": {"checkpoint_root": "/checkpoint"},
        "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": targets},
    })
    assert list(config.lora.target_modules) == targets


def test_qsa_kernel_follows_the_attention_backend():
    assert FlashSettings.from_dict({"attention_backend": "sdpa"}).qsa_kernel == "flex"
    assert FlashSettings(attention_backend="sdpa").qsa_kernel == "flex"
    assert FlashSettings.from_dict({"qsa_kernel": "flex"}).qsa_kernel == "flex"


def test_learner_and_helper_adapter_metadata_agree():
    from training.flash.lora_targets import ALL_TARGETS, peft_adapter_config
    from training.flash_model import write_flash_peft_adapter
    import json
    import tempfile
    from pathlib import Path

    text_config = {
        "hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "linear_num_key_heads": 1, "linear_num_value_heads": 2, "linear_key_head_dim": 4,
        "linear_value_head_dim": 4, "shared_expert_intermediate_size": 8,
        "layer_types": ["linear_attention", "full_attention"],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "adapter"
        write_flash_peft_adapter(path, checkpoint="Qwen/Qwen3.8-Flash-Next", text_config=text_config,
                                 rank=2, alpha=4.0, target_modules=ALL_TARGETS)
        written = json.loads((path / "adapter_config.json").read_text())
    assert written == peft_adapter_config(base_model_name_or_path="Qwen/Qwen3.8-Flash-Next", r=2, alpha=4.0,
                                          target_modules=ALL_TARGETS)


def _smoke_payload_for_flash():
    import json
    from pathlib import Path

    payload = json.loads((Path(__file__).resolve().parents[1] / "configs/ppo_2gpu_arc3_platypus_lora_smoke.json").read_text())
    payload["training"].update({
        "weight_mode": "flash", "device": "cuda", "multimodal": False, "device_map_gpus": [],
        "worker_device_map_gpus": [], "model_name": "Qwen/Qwen3.8-Flash-Next",
        "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": ["q_proj"]},
        "flash": {"checkpoint_root": "/checkpoint", "device_ids": [6, 7]},
    })
    return payload


def _load(tmp_path, payload):
    import json

    from training.config import ExperimentConfig

    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return ExperimentConfig.load(path)


def test_flash_gpu_claims_must_be_disjoint(tmp_path):
    payload = _smoke_payload_for_flash()
    payload["training"]["multimodal"] = True
    assert _load(tmp_path, payload).training.multimodal
    assert _load(tmp_path, payload).training.flash.device_ids == [6, 7]
    # The smoke config spells GPU ids as strings; claims compare integers.
    payload["training"]["flash"]["device_ids"] = [int(payload["inference"]["server_gpus"][0]), 7]
    with pytest.raises(ValueError, match="inference.server_gpus and training.flash.device_ids must claim disjoint GPUs"):
        _load(tmp_path, payload)


def test_flash_refuses_official_eval(tmp_path):
    """Flash policies reach servers only through the learner's NCCL broadcast to its rollout
    fleet; dedicated eval servers would need a group of their own."""
    payload = _smoke_payload_for_flash()
    payload["official_eval"] = {"enabled": True, "server_gpus": ["5"], "games": payload["inference"]["games"][:1],
                                "cycle_minutes": 5}
    with pytest.raises(ValueError, match="official_eval is not supported with training.weight_mode='flash'"):
        _load(tmp_path, payload)
    payload["official_eval"]["enabled"] = False
    assert not _load(tmp_path, payload).official_eval.enabled


def test_flash_parity_defaults_sit_above_the_serving_floor():
    for settings in (FlashSettings(), FlashSettings.from_dict({}), FlashSettings.from_dict({"parity_thresholds": {}})):
        assert asdict(settings.parity_thresholds) == FLASH_PARITY
    assert asdict(ParityThresholds.flash()) == FLASH_PARITY
    # Each recipe owns its thresholds; mutating one leaves the defaults intact.
    first, second = FlashSettings(), FlashSettings()
    first.parity_thresholds.mean_abs = 1.0
    assert second.parity_thresholds.mean_abs == ParityThresholds.flash().mean_abs == 0.25


def test_retired_adapter_probe_section_is_ignored_with_one_warning(caplog, monkeypatch):
    import training.flash_config as flash_config
    from training.config import ExperimentConfig

    monkeypatch.setattr(flash_config, "_warned_retired", set())
    recipe = ExperimentConfig.load("configs/flash_8gpu_arc2_grpo_tokmean.json").training.flash
    assert "adapter_probe" not in asdict(recipe)
    assert FlashSettings.from_dict(asdict(recipe)) == recipe
    # A config frozen while the amplified-adapter probe existed; any value is ignored.
    frozen = asdict(recipe) | {"adapter_probe": {"initial_scale": 8.0, "interval": 1}}
    with caplog.at_level("WARNING", logger="training.flash_config"):
        assert FlashSettings.from_dict(frozen) == recipe
        assert FlashSettings.from_dict({"adapter_probe": []}) == FlashSettings.from_dict({})
    warnings = [r.getMessage() for r in caplog.records if "retired setting" in r.getMessage()]
    assert len(warnings) == 1 and "training.flash.adapter_probe" in warnings[0]


def test_retired_expert_storage_is_ignored_with_one_warning(caplog, monkeypatch):
    import training.flash_config as flash_config
    from training.config import ExperimentConfig

    monkeypatch.setattr(flash_config, "_warned_retired", set())
    recipe = ExperimentConfig.load("configs/flash_8gpu_arc2_grpo_tokmean.json").training.flash
    assert "expert_storage" not in asdict(recipe)
    # Configs frozen while the setting existed (the live run's says "nvfp4")
    # load as the BF16 recipe; a value the setting never took is still refused.
    with caplog.at_level("WARNING", logger="training.flash_config"):
        for storage in ("nvfp4", "bf16"):
            assert FlashSettings.from_dict(asdict(recipe) | {"expert_storage": storage}) == recipe
    warnings = [r.getMessage() for r in caplog.records if "retired setting" in r.getMessage()]
    assert len(warnings) == 1 and "training.flash.expert_storage" in warnings[0]
    for storage in ("fp8", None, "NVFP4"):
        with pytest.raises(ValueError, match="training.flash.expert_storage must be one of"):
            FlashSettings.from_dict({"expert_storage": storage})


def test_load_balance_settings_are_diagnostics():
    defaults = FlashSettings.from_dict({})
    assert defaults.load_balance_metrics and defaults.load_balance_eval_episodes == 8
    off = FlashSettings.from_dict({"load_balance_metrics": False, "load_balance_eval_episodes": 0})
    # Never model arguments: they change what the learner measures, not the model.
    assert off.model_kwargs() == defaults.model_kwargs()
    assert not {"load_balance_metrics", "load_balance_eval_episodes"} & set(off.model_kwargs())


def test_flash_parity_override_keeps_unset_flash_defaults():
    override = FlashSettings.from_dict({"parity_thresholds": {"mean_abs": 0.3}}).parity_thresholds
    assert asdict(override) == FLASH_PARITY | {"mean_abs": 0.3}
    assert FlashSettings.from_dict({"parity_thresholds": LORA_PARITY}).parity_thresholds == ParityThresholds()


def test_lora_and_full_weight_parity_defaults_are_unchanged():
    from training.config import TrainingConfig

    assert asdict(ParityThresholds()) == LORA_PARITY
    assert asdict(ParityThresholds.from_dict(None)) == LORA_PARITY
    config = TrainingConfig.from_dict({
        "model_name": "Qwen/Qwen3.6-27B", "output_dir": "runs",
        "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": ["q_proj"]},
        "parity_thresholds": {"full_weight": {"mean_abs": 0.07}},
    })
    assert config.parity_thresholds == {"full_weight": ParityThresholds(mean_abs=0.07)}
    assert asdict(config.parity_thresholds["full_weight"]) == LORA_PARITY | {"mean_abs": 0.07}


def _trainer_for(weight_mode, trainer_class=None, **training):
    from training.config import TrainingConfig
    from training.ppo import PPOTrainer

    payload = {"model_name": "Qwen/Qwen3.8-Flash-Next", "output_dir": "runs", "weight_mode": weight_mode,
               "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": ["q_proj"]}, **training}
    if weight_mode == "flash":
        payload |= {"device": "cuda", "flash": {"checkpoint_root": "/checkpoint", **training.get("flash", {})}}
    trainer_class = trainer_class or PPOTrainer
    trainer = trainer_class.__new__(trainer_class)
    trainer.config = TrainingConfig.from_dict(payload)
    return trainer


# Worst engine-noise control on the distinct-file corpus (online LoRA vs the same
# adapter merged, B scale 0.02) and the same adapter dropped by the server.
SERVING_FLOOR = {"token_count": 94188, "mean_abs": 0.0791, "p95_abs": 0.426, "outlier_fraction": 0.00058}
DROPPED_ADAPTER = {"token_count": 94188, "mean_abs": 0.344, "p95_abs": 1.864, "outlier_fraction": 0.0443}


def test_trainer_resolves_parity_thresholds_per_weight_mode():
    from training.full_weight_ppo import FullWeightPPOTrainer

    lora = _trainer_for("lora")
    assert lora._parity_thresholds() == ParityThresholds()
    # The LoRA/full-weight gate sits below Flash's engine floor.
    assert lora._rollout_logprob_parity_failure(SERVING_FLOOR, sample_count=1)
    for trainer_class in (None, FullWeightPPOTrainer):
        full_weight = _trainer_for("full_weight", trainer_class,
                                   parity_thresholds={"full_weight": {"mean_abs": 0.07}})
        assert asdict(full_weight._parity_thresholds()) == LORA_PARITY | {"mean_abs": 0.07}
        assert _trainer_for("full_weight", trainer_class)._parity_thresholds() == ParityThresholds()


def test_flash_trainer_resolves_its_own_parity_thresholds():
    # FlashPPOTrainer overrides the base trainer's hook; the base never reads
    # training.flash.
    pytest.importorskip("triton")
    from training.flash_ppo import FlashPPOTrainer

    flash = _trainer_for("flash", FlashPPOTrainer)
    assert flash._parity_thresholds() == ParityThresholds.flash()
    assert not flash._rollout_logprob_parity_failure(SERVING_FLOOR, sample_count=1)
    assert flash._rollout_logprob_parity_failure(DROPPED_ADAPTER, sample_count=1)
    override = _trainer_for("flash", FlashPPOTrainer, flash={"parity_thresholds": {"p95_abs": 2.0}})
    assert asdict(override._parity_thresholds()) == FLASH_PARITY | {"p95_abs": 2.0}


def test_flash_loads_tokenizer_and_processor_from_the_local_checkpoint(monkeypatch):
    from training.config import ExperimentConfig
    from training.multimodal import build_checkpoint_identity, checkpoint_source
    import training.multimodal as multimodal

    config = ExperimentConfig.load("configs/flash_8gpu_arc2_grpo_tokmean.json")
    root = config.training.flash.checkpoint_root
    assert checkpoint_source(config.training) == (root, None, True)
    calls = []
    monkeypatch.setattr(multimodal, "resolve_processing_identity",
                        lambda source, revision, **kw: calls.append((source, revision, kw)) or {"model": source})
    support = type("Support", (), {"training_identity": staticmethod(lambda c, m: {"environment": "arc3"}),
                                   "media_requirements": staticmethod(lambda inference: {"images": True})})
    identity = build_checkpoint_identity(config, support=support, manifest=None)
    assert calls == [(root, None, {"local_files_only": True})]
    assert identity["processing_stack"] == {"model": root}
    # Other modes keep the Hub id and the pinned revision.
    lora = ExperimentConfig.load("configs/dppo_2gpu_arc3_multimodal_smoke.json").training
    assert checkpoint_source(lora) == (lora.model_name, lora.model_revision, False)


def test_flash_rollout_spec_uses_the_local_tokenizer_and_the_flash_engine():
    import inspect

    from training.config import ExperimentConfig
    from training.harness import harness_engine

    inference_module = pytest.importorskip("training.inference")

    flash = ExperimentConfig.load("configs/flash_8gpu_arc2_grpo_tokmean.json").training
    lora = ExperimentConfig.load("configs/dppo_2gpu_arc3_multimodal_smoke.json").training
    assert harness_engine(flash) == "qwen3.8-flash-next"
    assert harness_engine(lora) == "qwen3.8-27b"
    source = inspect.getsource(inference_module.RolloutCollector._collect_window_platypus)
    assert '"tokenizer_id": tokenizer_source' in source
    assert '"engine": harness_engine(self.config.training)' in source
    assert '"tokenizer_id": self.config.training.model_name' not in source
