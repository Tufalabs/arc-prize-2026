"""Supplied factories build without resolving local engine weights."""

from typing import Any

import pytest
import ullm.backends.deployment as deployment
import ullm.backends.vllmx
import ullm.factory

from seal import presets
from seal.agent import SealAgent


@pytest.mark.parametrize("engine_key", ["qwen3.8-27b", "qwen3.6-27b"])
def test_supplied_engine_never_constructs_local_engine_arguments(
    monkeypatch: pytest.MonkeyPatch, engine_key: str
) -> None:
    import vllm
    from ullm.backends.engine_proxy import EngineProxy
    from ullm.backends.vllmx import VLLM

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a supplied engine must not resolve another model's weights")

    monkeypatch.setattr(vllm, "AsyncEngineArgs", forbidden)
    handle = EngineProxy(base_url="http://127.0.0.1:1", model="trained-policy")
    agent = SealAgent.build(
        engine=engine_key,
        backend_factory=ullm.factory.PrebuiltBackend(
            backend=ullm.backends.vllmx.VLLM(engine=handle, **deployment.backend_kwargs(engine_key)),
            close_backend=True,
        ),
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=4096,
        context_window_tokens=32768,
        compact_to_tokens=16384,
    )
    factory = agent.llm_config.backend_factory
    assert isinstance(factory, ullm.factory.PrebuiltBackend)
    assert isinstance(factory.backend, VLLM) and factory.backend.engine is handle
    assert handle.metadata == {} and handle._http is None
    assert factory.backend.native_reasoning_effort is (engine_key == "qwen3.8-27b")
    assert agent.llm_config.sampling_profiles == presets.analyzer_sampling_profiles()
