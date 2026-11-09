"""Attestation reads native engine state and never falls back to legacy APIs."""

import pytest

from training.vllm_runtime import native_url, validate_server_runtime


def config(**overrides):
    return {
        "transport": "engine-proxy-v1",
        "generation_config": "vllm",
        "override_generation_config": {},
        "logprobs_mode": "processed_logprobs",
        "logits_processors": [],
        "default_sampling_params": {},
        "kv_offloading_size": None,
        "kv_transfer_enabled": False,
        "cpu_offload_gb": 0,
        "offload_group_size": 0,
        "speculative_decoding": False,
        "engram_cpu_offload": False,
    } | overrides


class Response:
    def __init__(self, payload, code=200):
        self.payload, self.status_code = payload, code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.mark.parametrize(
    "override",
    [
        {"transport": "openai"},
        {"generation_config": "auto"},
        {"override_generation_config": {"top_k": 20}},
        {"logprobs_mode": "raw_logprobs"},
        {"logprobs_mode": "raw_logits"},
        {"logits_processors": ["custom.Mask"]},
        {"default_sampling_params": {"top_k": 20}},
        {"kv_offloading_size": 10},
        {"kv_transfer_enabled": True},
        {"cpu_offload_gb": 2},
        {"offload_group_size": 1},
        {"speculative_decoding": True},
        {"engram_cpu_offload": True},
        {"engram_cpu_offload": None},
    ],
)
def test_rejects_incompatible_live_engine_configuration(monkeypatch, override):
    monkeypatch.setattr(
        "training.vllm_runtime.requests.get",
        lambda *a, **k: Response(config(**override)),
    )
    with pytest.raises(RuntimeError, match=next(iter(override))):
        validate_server_runtime("http://localhost:8000")


def test_reads_exact_native_route_with_auth_and_no_fallback(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response(config())

    monkeypatch.setattr("training.vllm_runtime.requests.get", get)
    assert (
        validate_server_runtime(
            "http://localhost:8000", headers={"Authorization": "Bearer test"}
        )
        == config()
    )
    assert calls == [
        (
            "http://localhost:8000/runtime",
            {"headers": {"Authorization": "Bearer test"}, "timeout": 10.0},
        )
    ]
    monkeypatch.setattr(
        "training.vllm_runtime.requests.get", lambda *a, **k: Response({}, 404)
    )
    with pytest.raises(RuntimeError, match="404"):
        validate_server_runtime("http://localhost:8000")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/v1",
        "http://host/endpoint",
        "http://host?token=x",
        "http://host/#frag",
        "file:///tmp/server",
    ],
)
def test_rejects_legacy_or_ambiguous_base_urls(url):
    with pytest.raises(ValueError, match="HTTP origin"):
        native_url(url, "/generate")


@pytest.mark.parametrize("version,accepted", [
    ("0.29.1.dev166+g3bda1ba09", True),
    ("0.29.1.dev166+g3bda1ba09.precompiled", True),
    ("0.29.0", False), ("0.29.1.dev108+gc43da30f2", False),
    ("0.29.1.dev63+g077920f36.precompiled", False),
    ("0.31.0", False), ("0.25.1", False),
])
def test_flash_requires_exact_harness_fork(version, accepted):
    from training.vllm_runtime import is_flash_vllm_version

    assert is_flash_vllm_version(version) is accepted


@pytest.mark.parametrize("override", [
    {}, {"model_architectures": ["Qwen3VLForConditionalGeneration"]},
    {"engram_cpu_offload": 1},
    # The generic UVA offload, even of the PLE table alone (the 077920f3 route).
    {"cpu_offload_gb": 48, "cpu_offload_params": ["ngram_embedding"], "offload_backend": "uva"},
    {"kv_offloading_size": 1}, {"kv_transfer_enabled": True},
])
def test_flash_allows_only_ple_table_on_host(override):
    from training.vllm_runtime import _validate_runtime_config

    settings = config(engram_cpu_offload=True,
                      model_architectures=["Qwen4ExpForConditionalGeneration"]) | override
    if override:
        with pytest.raises(RuntimeError):
            _validate_runtime_config(settings)
    else:
        _validate_runtime_config(settings)


def test_trainer_pins_the_drafter_over_http(monkeypatch):
    from training.config import SpeculativeDecodingConfig

    expected = SpeculativeDecodingConfig(enabled=True).identity()
    drafting = config(speculative_decoding=True, speculative=expected,
                      engram_cpu_offload=True,
                      model_architectures=["Qwen4ExpForConditionalGeneration"])
    monkeypatch.setattr(
        "training.vllm_runtime.requests.get", lambda *a, **k: Response(drafting)
    )
    assert validate_server_runtime("http://localhost:8000", speculative=expected) == drafting
    with pytest.raises(RuntimeError, match="speculative_decoding must be False"):
        validate_server_runtime("http://localhost:8000")
    with pytest.raises(RuntimeError, match="speculative must be"):
        validate_server_runtime(
            "http://localhost:8000", speculative=expected | {"num_speculative_tokens": 2}
        )
    monkeypatch.setattr(
        "training.vllm_runtime.requests.get", lambda *a, **k: Response(config())
    )
    with pytest.raises(RuntimeError, match="speculative_decoding must be True"):
        validate_server_runtime("http://localhost:8000", speculative=expected)
