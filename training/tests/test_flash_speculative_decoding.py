"""inference.speculative_decoding: the opt-in flag, its launch path, resume and the harness pin."""

import ast
from dataclasses import asdict
import json
import shlex
from types import SimpleNamespace as NS

import pytest

from flash_configs import RECIPE, ROOT
from training.config import ExperimentConfig, SpeculativeDecodingConfig
from training.flash_drafting import (
    EXACT_VERIFICATION,
    FLASH_E307_DRAFTER,
    HARNESS_SPECULATIVE,
)
from training.job_shape import job_shape, write_launch_config

HARNESS_DEPLOYMENT = ROOT / "platypus/packages/ullm/src/ullm/backends/deployment.py"


def _recipe(tmp_path, speculative=None, **training):
    payload = json.loads(RECIPE.read_text())
    if speculative is not None:
        payload["inference"]["speculative_decoding"] = speculative
    payload["training"].update(training)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_shipped_recipe_drafts_with_the_harness_settings():
    assert json.loads(RECIPE.read_text())["inference"]["speculative_decoding"] == {"enabled": True}
    cfg = ExperimentConfig.load(RECIPE)
    assert cfg.inference.speculative_decoding == SpeculativeDecodingConfig(enabled=True)
    assert json.loads(job_shape(cfg, {})["SERVER_SPECULATIVE_CONFIG"]) == (
        HARNESS_SPECULATIVE | {"num_speculative_tokens": 4, "model": FLASH_E307_DRAFTER})


def test_drafting_is_off_unless_a_config_asks(tmp_path):
    payload = json.loads(RECIPE.read_text())
    del payload["inference"]["speculative_decoding"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    cfg = ExperimentConfig.load(path)
    assert cfg.inference.speculative_decoding == SpeculativeDecodingConfig()
    assert cfg.inference.speculative_decoding.engine_config() is None
    assert cfg.inference.speculative_decoding.identity() is None
    assert job_shape(cfg, {})["SERVER_SPECULATIVE_CONFIG"] == ""


def test_enabling_fills_in_the_harness_drafting(tmp_path):
    cfg = ExperimentConfig.load(_recipe(tmp_path, {"enabled": True}))
    speculative = cfg.inference.speculative_decoding
    assert (speculative.drafter, speculative.num_speculative_tokens) == (FLASH_E307_DRAFTER, 4)
    assert speculative.engine_config() == {
        "method": "mtp", "num_speculative_tokens": 4, "quantization": "modelopt_fp4",
        "draft_sample_method": "probabilistic", "gdn_recoverssm": True,
        "disable_eagle_block_drop": True, "model": FLASH_E307_DRAFTER,
    }
    # The attested identity adds the fork's exact-verification defaults.
    assert speculative.identity() == speculative.engine_config() | {
        "rejection_sample_method": "standard", "accept_threshold_single": 1.0,
        "accept_threshold_acc": 1.0, "draft_top_k_top_p": False,
    }


def test_harness_deployment_drafts_with_the_copied_settings():
    """The settings are the harness deployment's (platypus submodule pin), copied."""
    if not HARNESS_DEPLOYMENT.is_file():
        pytest.skip("platypus submodule is not checked out")
    tree = ast.parse(HARNESS_DEPLOYMENT.read_text())
    drafters = [
        ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "FLASHNEXT_E307_DRAFTER" for t in node.targets)
    ]
    configs = [
        ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "speculative_config"
        and isinstance(node.value, ast.Dict)
    ]
    assert drafters == [FLASH_E307_DRAFTER]
    assert configs == [HARNESS_SPECULATIVE | {"num_speculative_tokens": 4}]
    # The harness never relaxes the fork's verification.
    assert not set(EXACT_VERIFICATION) & set(configs[0])


@pytest.mark.parametrize(("speculative", "error"), [
    ("on", "must be a JSON object"),
    ({"enabled": "true"}, "enabled must be true or false"),
    ({"enabled": True, "drafter": "relative/drafter"}, "absolute checkpoint path"),
    ({"enabled": True, "drafter": 7}, "absolute checkpoint path"),
    ({"enabled": True, "num_speculative_tokens": 0}, "integer in 1..8"),
    ({"enabled": True, "num_speculative_tokens": 9}, "integer in 1..8"),
    ({"enabled": True, "num_speculative_tokens": 4.0}, "integer in 1..8"),
    ({"enabled": True, "method": "eagle3"}, "inference.speculative_decoding.method"),
])
def test_invalid_settings_are_refused(tmp_path, speculative, error):
    with pytest.raises(ValueError, match=error):
        ExperimentConfig.load(_recipe(tmp_path, speculative))


def test_drafting_is_refused_outside_flash(tmp_path):
    payload = json.loads((ROOT / "configs/ppo_2gpu_arc3_platypus_lora_smoke.json").read_text())
    payload["inference"]["speculative_decoding"] = {"enabled": True}
    path = tmp_path / "lora.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="requires training.weight_mode='flash'"):
        ExperimentConfig.load(path)
    # Disabled is accepted anywhere.
    payload["inference"]["speculative_decoding"] = {"enabled": False}
    path.write_text(json.dumps(payload))
    assert ExperimentConfig.load(path).inference.speculative_decoding.enabled is False


def test_job_shape_hands_every_server_the_engine_config(tmp_path):
    drafter = "/data/shared/research/checkpoints/another-drafter"
    cfg = ExperimentConfig.load(_recipe(
        tmp_path, {"enabled": True, "drafter": drafter, "num_speculative_tokens": 3}))
    value = job_shape(cfg, {})["SERVER_SPECULATIVE_CONFIG"]
    assert json.loads(value) == cfg.inference.speculative_decoding.engine_config()
    assert json.loads(value)["model"] == drafter
    assert json.loads(value)["num_speculative_tokens"] == 3
    # training.job_shape prints NAME=<shell-quoted value> for the batch script to source.
    assert shlex.split(f"SERVER_SPECULATIVE_CONFIG={shlex.quote(value)}") == [
        f"SERVER_SPECULATIVE_CONFIG={value}"
    ]


def test_frozen_config_keeps_the_setting_across_a_resume(tmp_path):
    source = _recipe(tmp_path, {"enabled": True})
    shape = job_shape(ExperimentConfig.load(source), {})
    frozen = tmp_path / "frozen_config.json"
    write_launch_config(source, frozen, shape, 50123, "/cache/harness")
    # A resume rewrites the frozen config from itself, with new ports and venvs.
    write_launch_config(frozen, frozen, shape, 50999, "/cache/new-harness")
    resumed = ExperimentConfig.load(frozen)
    assert resumed.inference.speculative_decoding == SpeculativeDecodingConfig(enabled=True)
    assert job_shape(resumed, {})["SERVER_SPECULATIVE_CONFIG"] == shape["SERVER_SPECULATIVE_CONFIG"]
    # The effective config round-trips through the strict loader.
    effective = tmp_path / "effective_config.json"
    effective.write_text(json.dumps(resumed.to_dict()))
    assert ExperimentConfig.load(effective).inference.speculative_decoding.enabled is True


def test_setting_is_outside_the_flash_learner_recipe(tmp_path):
    flash_ppo = pytest.importorskip("training.flash_ppo")
    drafting = ExperimentConfig.load(_recipe(tmp_path, {"enabled": True}))
    plain = ExperimentConfig.load(_recipe(tmp_path, {"enabled": False}))
    # A resume may turn drafting on or off: the learner's recipe identity is the same.
    assert flash_ppo._recipe_identity(asdict(drafting.training.flash)) == flash_ppo._recipe_identity(
        asdict(plain.training.flash)
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_attests_the_drafter_its_config_asks_for(tmp_path, monkeypatch, enabled):
    from training.inference import RolloutCollector

    cfg = ExperimentConfig.load(_recipe(tmp_path, {"enabled": enabled}))
    calls = []

    def validate(base_url, **kwargs):
        calls.append((base_url, kwargs))
        return {"speculative_decoding": enabled}

    monkeypatch.setattr("training.vllm_runtime.validate_server_runtime", validate)
    collector = NS(config=cfg, _server_is_ready=lambda: True, _server_auth_headers=lambda: {})
    RolloutCollector.ensure_server(collector)
    assert calls[0][1]["speculative"] == (
        SpeculativeDecodingConfig(enabled=True).identity() if enabled else None
    )
    assert collector._runtime_config == {"speculative_decoding": enabled}
