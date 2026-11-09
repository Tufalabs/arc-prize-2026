"""The restored full-weight path preserves the existing LoRA smoke recipe."""

from pathlib import Path

from training.config import ExperimentConfig


def test_upstream_lora_smoke_uses_native_serving_and_current_schema():
    root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.load(root / "configs/ppo_2gpu_arc3_platypus_lora_smoke.json")
    assert config.inference.base_url == "http://127.0.0.1:58900"
    assert config.parallel_rollouts == config.num_iterations == 2
    assert not config.official_eval.enabled
    assert config.inference.agent_malformed_retries == 0
    assert config.inference.agent_accept_malformed_output
    assert config.inference.agent_temperature == config.inference.agent_top_p == 1
    assert config.training.device_map_gpus == [1]
    assert config.inference.server_gpus == ["0"]
