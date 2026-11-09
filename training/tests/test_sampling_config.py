"""Training and evaluation overrides cannot inherit deployment penalties."""

import dataclasses
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.config import (
    ExperimentConfig,
    InferenceConfig,
    validate_training_sampling,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("top_k", [None, 0, -1, 20])
def test_eval_sampling_neutralizes_every_deployment_profile(tmp_path, top_k):
    module = _load_script("eval_base_vs_lora")
    presets = pytest.importorskip("duck.presets")
    baseline = presets.analyzer_sampling_profiles()
    benchmark = module._build_episode_benchmark(
        game="ar25",
        seed=45,
        arm=module.ArmConfig(
            arm="base",
            base_url="http://127.0.0.1:1",
            served_model="policy",
            cap_seconds=120,
            max_completion_tokens=4096,
            reasoning_effort="low",
            context_window_tokens=32768,
            compact_to_tokens=16384,
            sampling={"temperature": 1.0, "top_p": 1.0, "top_k": top_k},
            api_key=None,
        ),
        job_dir=tmp_path,
    )
    agent = benchmark.solver.agent
    assert agent._sampling(SimpleNamespace(rng=random.Random(7))).max_tokens == 4096
    for profile in agent.llm_config.sampling_profiles.values():
        assert profile.temperature == 1.0
        assert profile.top_p == 1.0
        assert profile.extra_params == {
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
            "min_tokens": 0,
            "top_k": 20 if top_k == 20 else -1,
            "ignore_eos": False,
            "skip_special_tokens": False,
            "logprobs": 0,
        }
    assert baseline == presets.analyzer_sampling_profiles()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("agent_temperature", 0.6),
        ("agent_top_p", 0.95),
        ("agent_top_k", 20),
        ("agent_malformed_retries", 5),
        ("agent_accept_malformed_output", False),
    ],
)
def test_training_rejects_biased_sampling_before_launch(tmp_path, field, value):
    payload = json.loads((ROOT / "configs/ppo_2gpu_arc3_games25_platypus.json").read_text())
    payload["inference"][field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=field):
        ExperimentConfig.load(path)


def test_default_sampling_keeps_malformed_draws_and_allows_separate_eval():
    config = ExperimentConfig.load(ROOT / "configs/ppo_2gpu_arc3_games25_platypus.json")
    validate_training_sampling(config.inference)
    assert config.official_eval.temperature == 0.6
    payload = dataclasses.asdict(config.inference)
    del payload["game"]
    del payload["agent_malformed_retries"]
    del payload["agent_accept_malformed_output"]
    parsed = InferenceConfig.from_dict(payload)
    assert parsed.agent_malformed_retries == 0
    assert parsed.agent_accept_malformed_output is True


def test_worker_keeps_missing_continuation_evidence_visible(tmp_path):
    module = _load_script("platypus_rollout_worker")
    step = {
        "provider_validation": {"request_args": {"request_id": "run#nonce"}},
        "output_token_ids": [1, 2],
        "extra": {},
    }
    (tmp_path / "steps.jsonl").write_text(json.dumps(step) + "\n")
    (report,) = module._step_reports(tmp_path)
    assert report["continued"] is None
    assert report["history_epoch"] is None
    assert report["exact_history"] is None


@pytest.mark.parametrize("top_k", [None, 0, -1, 20])
def test_worker_uses_explicit_neutral_sampling(tmp_path, top_k):
    pytest.importorskip("duck.agent")
    module = _load_script("platypus_rollout_worker")
    benchmark = module._build_benchmark(
        {
            "games": ["ar25-0c556536"],
            "max_concurrent": 1,
            "max_runtime_s_per_game": 900,
            "context_window_tokens": 32768,
            "compact_to_tokens": 16384,
            "compaction_trigger_tokens": 28160,
            "safety_margin_tokens": 512,
            "malformed_retries": 0,
            "accept_malformed_output": True,
            "base_url": "http://127.0.0.1:1",
            "served_model": "policy",
            "max_completion_tokens": 4096,
            "max_summary_tokens": 0,
            "summary_forbidden_token_ids": [],
            "max_served_sequence_tokens": 32768,
            "job_dir": str(tmp_path),
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": top_k},
        }
    )
    agent = benchmark.solver.agent
    assert agent.llm_config.exact_history and agent.token_native_context
    assert agent.run_config.min_request_gap_s == 0
    assert agent._sampling(SimpleNamespace(rng=random.Random(7))).max_tokens == 4096
    for profile in agent.llm_config.sampling_profiles.values():
        assert profile.temperature == 1.0 and profile.top_p == 1.0
        assert profile.send_back_reasoning == "all"
        assert profile.extra_params == {
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
            "min_tokens": 0,
            "top_k": 20 if top_k == 20 else -1,
            "ignore_eos": False,
            "skip_special_tokens": False,
            "logprobs": 0,
        }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accept_malformed_output", True),
        ("require_explicit_sampling", True),
        ("verify_engine_output", True),
        ("token_count_evidence", True),
        ("replay_on_link_loss", False),
    ],
)
def test_training_requires_review_when_shared_backend_defaults_change(field, value):
    pytest.importorskip("duck.agent")
    from dataclasses import replace
    from duck import DuckAgent
    from training_backend import qwen_engine_factory, training_backend_factory
    from ullm.backends.engine_proxy import EngineProxy

    guarded = {"derive_max_pixels": False} if hasattr(DuckAgent, "derive_max_pixels") else {}
    agent = DuckAgent.build(
        **guarded,
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        backend_factory=qwen_engine_factory(
            EngineProxy(base_url="http://127.0.0.1:1", model="policy"),
            engine="qwen3.8-27b",
        ),
        engine="qwen3.8-27b",
    )
    factory = agent.llm_config.backend_factory
    changed = replace(factory, backend_kwargs={**factory.backend_kwargs, field: value})
    with pytest.raises(ValueError, match=f"shipped backend {field} changed"):
        training_backend_factory(source_factory=changed)
    assert factory.engine._http is None
