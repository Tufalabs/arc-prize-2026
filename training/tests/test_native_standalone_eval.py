"""Standalone evaluation uses the same native contract as training."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import requests
from aiohttp.test_utils import TestServer

pytest.importorskip("duck.agent")
pytest.importorskip("vllm")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_base_vs_lora as evaluation  # noqa: E402
from duck.agent import DuckAgent  # noqa: E402
from ullm.backends.engine_proxy import EngineProxy  # noqa: E402
from ullm.backends.vllmx import VLLM  # noqa: E402
from engine_wire import generation  # noqa: E402
from training.engine_protocol import unpack_output  # noqa: E402
from training.native_vllm_server import NativeService, create_app  # noqa: E402
from training.rollout_gateway import GatewayError  # noqa: E402


def arm(**changes):
    return evaluation.ArmConfig(
        **(
            {
                "arm": "base",
                "base_url": "http://unused.invalid",
                "served_model": "base",
                "cap_seconds": 120,
                "max_completion_tokens": 128,
                "reasoning_effort": "low",
                "context_window_tokens": 4096,
                "compact_to_tokens": 1500,
                "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
                "api_key": None,
            }
            | changes
        )
    )


def native_service():
    spec = importlib.util.spec_from_file_location(
        "standalone_engine_fixture", ROOT / "tests/test_native_vllm_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = module.Engine()
    engine.vllm_config.model_config.max_model_len = 4096
    return NativeService(
        engine,
        sampling_factory=lambda **kwargs: NS(
            **{
                "stop": None,
                "include_stop_str_in_output": False,
                "spaces_between_special_tokens": True,
                **kwargs,
            }
        ),
        lora_factory=lambda name, identifier, path: NS(
            lora_name=name, lora_int_id=identifier, lora_path=path
        ),
    )


@pytest.mark.parametrize("deployment", [False, True])
@pytest.mark.parametrize("arm_name", ["base", "lora"])
def test_standalone_build_preserves_full_game_and_explicit_neutral_sampling(
    tmp_path, deployment, arm_name
):
    config = arm(arm=arm_name, sampling=None) if deployment else arm(arm=arm_name)
    benchmark = evaluation._build_episode_benchmark(
        game="ar25", seed=45, arm=config, job_dir=tmp_path
    )
    agent = benchmark.solver.agent
    assert type(agent) is DuckAgent
    factory = agent.llm_config.backend_factory
    backend = VLLM(engine=factory.engine, **factory.backend_kwargs)
    assert isinstance(backend, VLLM)
    assert isinstance(backend.engine, EngineProxy)
    assert agent.run_config.rng_seed_per_game == {"ar25": 45}
    assert agent.run_config.max_runtime_s_per_game == 120
    assert agent.run_config.on_agent_error == "raise"
    assert agent.run_config.min_request_gap_s == 0.0
    # Eval disables the Duck's per-level give-up (owner ruling, 6aad3b0f);
    # training inherits the preset's value instead.
    assert agent.max_tokens_per_level is None
    assert agent.malformed_retries == 0
    assert not agent.zero_recurrent
    assert agent.llm_config.exact_history is True
    expected = evaluation._sampling(config)
    for profile in agent.llm_config.sampling_profiles.values():
        assert profile.temperature == expected["temperature"]
        assert profile.top_p == expected["top_p"]
        assert profile.extra_params["top_k"] == expected["top_k"]
        assert profile.extra_params["presence_penalty"] == 0
        assert profile.extra_params["repetition_penalty"] == 1


def test_standalone_native_gateway_pins_engine_and_captures_malformed_text(tmp_path):
    async def scenario():
        service = native_service()
        async with TestServer(create_app(service)) as server:
            config = arm(base_url=str(server.make_url("/")).rstrip("/"))
            benchmark = evaluation._build_episode_benchmark(
                game="ar25", seed=45, arm=config, job_dir=tmp_path
            )
            gateway = await asyncio.to_thread(
                evaluation._episode_gateway, benchmark, config
            )
            address = gateway.base_url
            try:
                assert gateway.port > 0
                factory = benchmark.solver.agent.llm_config.backend_factory
                assert isinstance(factory.engine, EngineProxy)
                assert factory.engine.base_url == address
                assert (
                    gateway._policy.expected_adapter
                    == service.metadata("base")["adapter"]
                )
                body = {
                    "model": "base",
                    "user": "eval-run#1",
                    "prompt": [1, 2],
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 4,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "min_p": 0.0,
                    "min_tokens": 0,
                    "skip_special_tokens": False,
                    "arc3_lineage": {
                        "version": 1,
                        "epoch": 0,
                        "kind": "start",
                        "compaction_prompt_token_ids": None,
                    },
                }
                response = await asyncio.to_thread(
                    requests.post,
                    address + "/generate",
                    json=generation(body, gateway._policy.expected_adapter),
                    timeout=5,
                )
                response.raise_for_status()
                assert unpack_output(response.json())["text"] == "!!!!! malformed <tool"
                assert len(service.engine.calls) == 1
                capture = (await asyncio.to_thread(gateway.drain_window)).captures[
                    "eval-run", "1"
                ]
                assert capture.prompt_token_ids == [1, 2]
                assert capture.completion_token_ids == [3, 4]
                assert capture.completion_logprobs == [-0.4, -0.4]
                assert capture.lineage["kind"] == "start"
            finally:
                await asyncio.to_thread(gateway.stop)
            with pytest.raises(requests.ConnectionError):
                await asyncio.to_thread(requests.get, address + "/health", timeout=1)

    asyncio.run(scenario())


def test_standalone_contract_latch_cancels_benchmark():
    async def scenario():
        cancelled = asyncio.Event()

        class Benchmark:
            async def run(self, **kwargs):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        class Gateway:
            def raise_if_invalid(self):
                raise GatewayError("prefix changed")

        with pytest.raises(GatewayError, match="prefix changed"):
            await asyncio.wait_for(
                evaluation._checked_benchmark(Benchmark(), Gateway()), timeout=3
            )
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_standalone_uses_native_adapter_load_and_warm(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "runtime_vllm_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").touch()
    identity = {
        "name": "policy",
        "id": 2,
        "path": str(adapter_dir),
        "server_instance_id": "native-server",
    }
    calls = []

    def post(url, *, json, **kwargs):
        calls.append((url, json))
        return NS(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"adapter": identity},
        )

    monkeypatch.setattr(evaluation.requests, "post", post)
    monkeypatch.setattr(evaluation, "_model_alias_available", lambda *args: True)
    result = evaluation._prepare_lora_arm(
        adapter_dir=tmp_path / "source",
        eval_dir=tmp_path,
        lora_base_url="http://native",
        alias="policy",
        api_key=None,
    )
    assert result == adapter_dir
    assert calls == [
        ("http://native/adapters/load", {"name": "policy", "path": str(adapter_dir)}),
        ("http://native/adapters/warm", {"name": "policy"}),
    ]
