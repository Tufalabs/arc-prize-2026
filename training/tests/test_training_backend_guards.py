"""Startup guards reject inconsistent controls before opening an engine."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def backend_module(monkeypatch):
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    class EngineProxy:
        pass

    class EngineProxyFactory(SimpleNamespace):
        pass

    stub("platypus", LLMAgentSolver=object)
    stub("ullm.backends.proxy", ProxyBackend=object)
    stub(
        "ullm.backends.engine_proxy",
        EngineProxy=EngineProxy,
        EngineProxyFactory=EngineProxyFactory,
    )
    stub("ullm.backends.vllmx", VLLM=SimpleNamespace)
    path = Path(__file__).resolve().parents[1] / "scripts/training_backend.py"
    spec = importlib.util.spec_from_file_location("training_backend_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    factory = EngineProxyFactory(
        engine=EngineProxy(),
        started=False,
        backend_kwargs=dict(
            classify_tool_calls=False,
            accept_malformed_output=True,
            require_explicit_sampling=True,
            verify_engine_output=True,
            token_count_evidence=True,
            replay_on_link_loss=False,
        ),
    )
    agent = SimpleNamespace(
        classify_tool_calls=False,
        preserve_failed_draws=False,
        llm_config=SimpleNamespace(backend_factory=factory, sampling_profiles={}),
    )
    return module, agent


@pytest.mark.parametrize("backend_classifies", [False, True])
def test_rejects_classification_mismatch(backend_module, backend_classifies):
    module, agent = backend_module
    agent.llm_config.backend_factory.backend_kwargs["classify_tool_calls"] = backend_classifies
    agent.classify_tool_calls = not backend_classifies
    with pytest.raises(ValueError, match="matching backend and agent classify_tool_calls"):
        module.validate_training_backend(agent)


@pytest.mark.parametrize("classifies", [False, True])
@pytest.mark.parametrize("preserves", [False, True])
def test_accepts_matching_controls(backend_module, classifies, preserves):
    module, agent = backend_module
    agent.llm_config.backend_factory.backend_kwargs["classify_tool_calls"] = classifies
    agent.classify_tool_calls = classifies
    agent.preserve_failed_draws = preserves
    module.validate_training_backend(agent)


def test_retaining_failed_draws_requires_backend_acceptance(backend_module):
    module, agent = backend_module
    agent.preserve_failed_draws = True
    agent.llm_config.backend_factory.backend_kwargs["accept_malformed_output"] = False
    with pytest.raises(ValueError, match="preserve_failed_draws requires backend accept_malformed_output"):
        module.validate_training_backend(agent)


@pytest.mark.parametrize("configure", [False, True])
def test_rejects_started_factory_through_public_state(backend_module, configure):
    module, agent = backend_module
    factory = agent.llm_config.backend_factory
    factory.started = True
    with pytest.raises(RuntimeError, match="before.*startup"):
        if configure:
            module.training_backend_factory(source_factory=factory)
        else:
            module.validate_training_backend(agent)
