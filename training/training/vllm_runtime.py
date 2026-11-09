"""Attest the native engine's actual configuration before collecting tokens."""

from urllib.parse import urlsplit
import re

import requests

from training.flash_drafting import (
    EXACT_VERIFICATION,
    HARNESS_SPECULATIVE,
    SPECULATIVE_FIELDS,
)

TRANSPORT = "engine-proxy-v1"
RUNTIME_CONFIG_PATH = "/runtime"


def is_flash_vllm_version(version: str) -> bool:
    """Only the harness lock's fork, including its precompiled-wheel suffix."""
    return re.fullmatch(
        r"0\.29\.1\.dev\d+\+g3bda1ba09[0-9a-f]*(?:\.precompiled)?", version
    ) is not None


def native_url(base_url: str, path: str) -> str:
    """Native serving has one root URL, with no compatibility route aliases."""
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("native vLLM base_url must be an HTTP origin without /v1")
    return base_url.rstrip("/") + path


def runtime_config(engine) -> dict:
    """Expose an allowlist of live engine fields; never dump credentials/config."""
    config = engine.vllm_config
    model = config.model_config
    result = {
        "transport": TRANSPORT,
        "generation_config": model.generation_config,
        "override_generation_config": model.override_generation_config,
        "logprobs_mode": model.logprobs_mode,
        "max_logprobs": model.max_logprobs,
        "vocab_size": model.get_vocab_size(),
        "logits_processors": [
            item if isinstance(item, str) else f"{item.__module__}.{item.__qualname__}"
            for item in (model.logits_processors or [])
        ],
        "default_sampling_params": model.get_diff_sampling_param(),
        "kv_offloading_size": config.cache_config.kv_offloading_size,
        "kv_transfer_enabled": config.kv_transfer_config is not None,
        "cpu_offload_gb": config.offload_config.uva.cpu_offload_gb,
        "offload_group_size": config.offload_config.prefetch.offload_group_size,
        "speculative_decoding": config.speculative_config is not None,
        # The Flash fork's pinned-host PLE table (absent before vLLM 0.29).
        "engram_cpu_offload": bool(
            getattr(getattr(config, "engram_config", None), "cpu_offload", False)
        ),
    }
    if result["engram_cpu_offload"]:
        result["model_architectures"] = list(model.architectures)
    if config.speculative_config is not None:
        # The drafter's identity and the verification mode, as the engine
        # resolved them (the fork's SpeculativeConfig fields).
        result["model_architectures"] = list(model.architectures)
        result["speculative"] = {
            name: getattr(config.speculative_config, name, None)
            for name in SPECULATIVE_FIELDS
        }
    if result["cpu_offload_gb"]:
        result.update(
            model_architectures=list(model.architectures),
            cpu_offload_params=sorted(config.offload_config.uva.cpu_offload_params),
            offload_backend=config.offload_config.offload_backend,
        )
    return result


def _validate_speculative(config: dict, speculative: dict | None) -> None:
    """Drafting only when asked for, with exactly the drafter asked for, and
    only the harness's MTP drafting with exact verification on Qwen4Exp."""
    observed = config.get("speculative")
    if speculative is None:
        if observed is not None:
            raise RuntimeError(
                f"vLLM runtime speculative must be absent; got {observed!r}"
            )
        return
    if observed != speculative:
        raise RuntimeError(
            f"vLLM runtime speculative must be {speculative!r}; got {observed!r}"
        )
    qualified = {**HARNESS_SPECULATIVE, **EXACT_VERIFICATION}
    if (
        any(
            observed.get(name) != value or type(observed.get(name)) is not type(value)
            for name, value in qualified.items()
        )
        or not isinstance(observed.get("model"), str)
        or not observed["model"]
        or type(observed.get("num_speculative_tokens")) is not int
        or observed["num_speculative_tokens"] < 1
        or config.get("model_architectures") != ["Qwen4ExpForConditionalGeneration"]
    ):
        raise RuntimeError(
            "vLLM runtime speculative decoding must be the harness's MTP drafting "
            f"with exact verification on Qwen4Exp; got {observed!r} for "
            f"{config.get('model_architectures')!r}"
        )


def _validate_runtime_config(
    config: dict, *, score_centering_top_k: int = 0, speculative: dict | None = None
) -> None:
    """``speculative`` is the drafter identity the caller expects
    (``speculative_identity``), or None for an engine that does not draft."""
    if not isinstance(config, dict):
        raise RuntimeError("vLLM runtime sampling config must be an object")
    if score_centering_top_k and (
        type(config.get("max_logprobs")) is not int
        or config["max_logprobs"] < score_centering_top_k
        or type(config.get("vocab_size")) is not int
        or config["vocab_size"] < score_centering_top_k
    ):
        raise RuntimeError(
            "vLLM runtime max_logprobs and vocab_size must cover score_centering_top_k"
        )
    # Only the Flash fork's frozen PLE n-gram table may live in host memory
    # (its Engram config, pinned and read over UVA), never the KV/SSM cache
    # and no other weight: no generic weight offload.
    engram = config.get("engram_cpu_offload")
    if type(engram) is not bool or (
        engram and config.get("model_architectures") != ["Qwen4ExpForConditionalGeneration"]
    ):
        raise RuntimeError(
            "vLLM runtime engram_cpu_offload must be false, or true only for "
            f"Qwen4Exp; got {engram!r} for "
            f"{config.get('model_architectures')!r}"
        )
    expected = {
        "transport": TRANSPORT,
        "generation_config": "vllm",
        "override_generation_config": {},
        "logprobs_mode": "processed_logprobs",
        "logits_processors": [],
        "default_sampling_params": {},
        "kv_transfer_enabled": False,
        "cpu_offload_gb": 0,
        "offload_group_size": 0,
        "speculative_decoding": speculative is not None,
    }
    for name, value in expected.items():
        if name not in config or config[name] != value:
            raise RuntimeError(
                f"vLLM runtime {name} must be {value!r}; got {config.get(name)!r}"
            )
    _validate_speculative(config, speculative)
    if "kv_offloading_size" not in config or config["kv_offloading_size"] not in (
        None,
        0,
    ):
        raise RuntimeError("vLLM runtime kv_offloading_size must be disabled")


def validate_server_runtime(
    base_url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    score_centering_top_k: int = 0,
    speculative: dict | None = None,
) -> dict:
    """Fail closed using the native process's actual engine state, including
    the drafter the run's config asks for (``speculative``) or none."""
    response = requests.get(
        native_url(base_url, RUNTIME_CONFIG_PATH), headers=headers, timeout=timeout
    )
    response.raise_for_status()
    config = response.json()
    _validate_runtime_config(
        config, score_centering_top_k=score_centering_top_k, speculative=speculative
    )
    if (
        score_centering_top_k
        and config.get("score_centering_top_k") != score_centering_top_k
    ):
        raise RuntimeError("vLLM runtime score_centering_top_k differs from trainer")
    return config
