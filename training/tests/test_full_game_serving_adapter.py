"""Real pinned-harness hooks preserve full-game behavior under strict serving."""

import asyncio
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("duck.agent")
pytest.importorskip("vllm")

import httpx  # noqa: E402
import ullm.backend  # noqa: E402
import ullm.continuation  # noqa: E402
import ullm.conv  # noqa: E402
from ullm.backends.proxy import ProxyBackend, start_backend_server, _HostLinkLost  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from serving_contract import (  # noqa: E402
    CompactionTokenCount,
    install_contract_backend,
)
from ullm.backends.vllmx import VLLM
from ullm.backends.engine_proxy import EngineProxy
from training_agent_fixture import agent_for_training, context_agent
from duck.agent import DuckAgent
from test_training_context_observer import _fixture


def _user(text):
    return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))


@pytest.mark.parametrize("transport", ["http", "proxy"])
def test_uncertain_generation_is_sent_exactly_once(transport: str, monkeypatch) -> None:
    async def scenario() -> None:
        attempts = []
        failure = (
            httpx.ConnectError("connection lost after sending")
            if transport == "http"
            else _HostLinkLost("connection lost after sending")
        )

        async def send_once(*args, **kwargs):
            attempts.append((args, kwargs))
            raise failure

        async def ready():
            pass

        if transport == "http":
            from vllm import SamplingParams

            backend = EngineProxy(base_url="http://127.0.0.1:1", model="test")
            backend._http = SimpleNamespace(post=send_once, aclose=ready)
            monkeypatch.setattr(backend, "connect", ready)
            iterator = backend.generate(
                {"prompt_token_ids": [1]}, SamplingParams(max_tokens=4), "r#first"
            )
        else:
            backend = ProxyBackend(
                socket_path="/unused/socket", replay_on_link_loss=False
            )

            async def generate_once(*args):
                yield await send_once(*args)

            monkeypatch.setattr(backend, "_generate_once", generate_once)
            iterator = backend._generate(ullm.backend.GenericSamplingParameters(), {})
        with pytest.raises(type(failure)):
            async for _ in iterator:
                pass
        assert len(attempts) == 1
        if transport == "http":
            with pytest.raises(type(failure)):
                await backend.aclose()
        else:
            await backend.close()

    asyncio.run(scenario())


def test_backend_preserves_native_preset_and_remains_disconnected_when_pickled():
    agent = agent_for_training(
        DuckAgent,
        model="model",
        traces=True,
        max_completion_tokens=4096,
        context_window_tokens=65536,
        compact_to_tokens=32768,
    )
    factory = agent.llm_config.backend_factory
    install_contract_backend(agent)
    assert agent.llm_config.backend_factory is factory
    restored_factory = pickle.loads(pickle.dumps(factory))
    restored = VLLM(engine=restored_factory.engine, **restored_factory.backend_kwargs)
    assert isinstance(restored, VLLM) and isinstance(restored.engine, EngineProxy)
    assert restored.engine._http is None and restored.engine.model == "model"
    assert restored.accept_malformed_output and restored.require_explicit_sampling
    assert restored.verify_engine_output and not restored.replay_on_link_loss
    assert agent.inject_text_on_first_prompt == agent.inject_text_on_later_prompts == ""
    assert not agent.zero_recurrent


def test_exact_compaction_prompt_survives_the_real_proxy_count_rpc(
    tmp_path, monkeypatch
):
    async def scenario():
        backend = VLLM(
            engine=EngineProxy(base_url="http://unused.invalid", model="model"),
            token_count_evidence=True,
        )

        async def ready():
            pass

        monkeypatch.setattr(backend, "probe_continuation", ready)
        monkeypatch.setattr(
            backend,
            "_prepare_input",
            lambda *args: {
                "engine": {"prompt_token_ids": [1, 2, 3, 4]},
                "trace_ctx": {"continued_prompt_ids": [1, 2, 3, 4]},
            },
        )
        host = await start_backend_server(
            backend=backend, socket_path=str(tmp_path / "backend.sock")
        )
        client = ProxyBackend(socket_path=host.socket_path, replay_on_link_loss=False)
        try:
            conversation = ullm.conv.Conversation(
                elements=[_user("play"), ullm.conv.ResponseTurn(elements=[])]
            )
            count = await client.token_count(
                ullm.backend.GenericSamplingParameters(), conversation
            )
            assert isinstance(count, CompactionTokenCount)
            assert int(count) == 4 and count.prompt_token_ids == (1, 2, 3, 4)
        finally:
            await client.close()
            await host.close()
            await backend.close()

    asyncio.run(scenario())


def test_real_turn_manager_wires_native_history_through_compaction(
    monkeypatch, tmp_path
):
    """Only the GPU engine is simulated; use both HTTP layers and real proxy RPC."""
    import importlib.util
    import ullm.manager
    import transformers
    from aiohttp.test_utils import TestServer
    from training.native_vllm_server import NativeService, create_app
    from training.rollout_gateway import CaptureGateway, SamplingPolicy
    from training.context_contract import resolve_context_contract
    import socket

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "platypus/packages/ullm/tests/test_token_native_serving.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pr76_tokenizer_fixture", fixture_path
    )
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)

    class Tokenizer(fixture_module._StubTokenizer):
        chat_template = "test-qwen-template"
        special_tokens_map = {"eos_token": "<|im_end|>"}
        bos_token_id, pad_token_id = None, None
        max_token_id = 0x10FFFF
        backend_tokenizer = SimpleNamespace(
            to_str=lambda: '{"model":{"type":"test-unicode"}}'
        )

        def __len__(self):
            return 0x110000

        def apply_chat_template(
            self, messages, *, add_generation_prompt=True, tokenize=True, **kwargs
        ):
            rendered = super().apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt, **kwargs
            )
            return self.encode(rendered) if tokenize else rendered

        def decode(self, ids, **kwargs):
            return super().decode(ids)

    tokenizer = Tokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )

    async def scenario():
        service_fixture_path = Path(__file__).with_name("test_native_vllm_server.py")
        service_spec = importlib.util.spec_from_file_location(
            "pr76_engine_fixture", service_fixture_path
        )
        service_fixture = importlib.util.module_from_spec(service_spec)
        service_spec.loader.exec_module(service_fixture)
        engine = service_fixture.Engine()
        engine.get_tokenizer = lambda: tokenizer
        engine.vllm_config.model_config.served_model_name = "model"
        engine.vllm_config.model_config.max_model_len = 40000
        engine.vllm_config.model_config.get_vocab_size = lambda: 0x110000
        sent = []
        outputs = ["", "x" * 17000, "reset done", "continue done"]

        async def generate(prompt, sampling, request_id, *, lora_request):
            ids = list(prompt["prompt_token_ids"])
            text = outputs[len(sent)]
            output = tokenizer.encode(text) + [tokenizer.eos_token_id]
            sent.append((request_id, ids, output))
            assert sampling.logprobs == 0 and sampling.skip_special_tokens is False
            assert sampling.presence_penalty == 0 and sampling.repetition_penalty == 1
            yield SimpleNamespace(
                request_id=request_id,
                prompt_token_ids=ids,
                finished=True,
                lora_request=lora_request,
                num_cached_tokens=0,
                outputs=[
                    SimpleNamespace(
                        index=0,
                        token_ids=output,
                        # vLLM retains the terminal stop ID but omits its text.
                        text=text,
                        finish_reason="stop",
                        stop_reason=None,
                        logprobs=[
                            {token: SimpleNamespace(logprob=-0.1, rank=None)}
                            for token in output
                        ],
                    )
                ],
            )

        engine.generate = generate
        service = NativeService(
            engine,
            sampling_factory=lambda **kwargs: SimpleNamespace(
                stop=None,
                include_stop_str_in_output=False,
                spaces_between_special_tokens=True,
                **kwargs,
            ),
            lora_factory=lambda *args, **kwargs: None,
        )
        native_http = TestServer(create_app(service))
        await native_http.start_server()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        gateway = CaptureGateway(
            upstream_base_url=str(native_http.make_url("/")).rstrip("/"), port=port
        )
        gateway.set_policy(
            SamplingPolicy(
                model="model",
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                context=resolve_context_contract(40000, 20000, 14000, 0),
                expected_adapter=service.adapter("model").identity(),
            )
        )
        gateway.start()
        backend = VLLM(
            engine=EngineProxy(base_url=gateway.base_url, model="model"),
            send_back_reasoning="all",
            exact_history=True,
            accept_malformed_output=True,
            require_explicit_sampling=True,
            verify_engine_output=True,
            token_count_evidence=True,
            replay_on_link_loss=False,
        )
        records = []
        backend.trace_recorder = records.append
        host = await start_backend_server(
            backend=backend, socket_path=str(tmp_path / "turns.sock")
        )
        client = ProxyBackend(
            socket_path=host.socket_path, run_id="r", replay_on_link_loss=False
        )
        agent = context_agent(
            reply_cap_tokens=20000,
            multimodal_context="",
            token_native_context=True,
            context_window_tokens=40000,
            inject_text_on_first_prompt="",
            inject_text_on_later_prompts="",
            compact_to_tokens=14000,
            safety_margin_tokens=0,
        )
        agent.messages = []
        conversation = ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(
                content=ullm.conv.StringContent(text=agent._system_prompt_text()),
                reasoning_effort=None,
                tools=[],
            ),
            elements=agent.messages,
            status="intermediate",
        )
        params = ullm.backend.GenericSamplingParameters(
            temperature=1.0,
            top_p=1.0,
            max_tokens=20000,
            send_back_reasoning="all",
            extra_params={
                "top_k": -1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "min_p": 0.0,
                "min_tokens": 0,
                "skip_special_tokens": False,
                "logprobs": 0,
            },
        )
        agent.llm_config.sampling_profiles = {agent.profile: params}
        agent._live_backend, agent._live_conversation, agent._live_params = (
            client,
            conversation,
            params,
        )
        answers = []
        try:
            for index in range(4):
                conversation.status = "intermediate"
                conversation.elements.append(_user(f"step {index}"))
                manager = ullm.manager.TurnManager(
                    conversation=conversation,
                    response_manager=ullm.manager.ResponseManager(
                        backend=client,
                        conversation=conversation,
                        sampling_params=params,
                        max_retries=0,
                        before_attempt=agent._observe_request_attempt,
                    ),
                    before_round=agent._trim_live_messages,
                )
                await manager.run()
                answers.append(conversation.last_turn.last_response)
                await agent._persist_history()
            captures = list(gateway.drain_window().captures.values())
            metadata = [capture.lineage for capture in captures]
            assert [item["kind"] for item in metadata] == [
                "start",
                "continue",
                "reset",
                "continue",
            ]
            assert [item["epoch"] for item in metadata] == [0, 0, 1, 1]
            assert answers[0].malformed_error and answers[0].malformed_retries == 0
            for previous, current in ((sent[0], sent[1]), (sent[2], sent[3])):
                expected = previous[1] + previous[2]
                assert current[1][: len(expected)] == expected and len(
                    current[1]
                ) > len(expected)
            proof = metadata[2]["compaction_prompt_token_ids"]
            prior = sent[1][1] + sent[1][2]
            assert proof[: len(prior)] == prior and len(proof) > 20000
            assert len(sent[2][1]) <= 14000 and len(sent[2][1]) < len(proof)
            assert all(capture.native_token_prompt for capture in captures)
            assert all(
                capture.adapter == service.adapter("model").identity()
                for capture in captures
            )
            assert len(records) == 4
            assert [capture.prompt_token_ids for capture in captures] == [
                row[1] for row in sent
            ]
            assert [capture.completion_token_ids for capture in captures] == [
                row[2] for row in sent
            ]
            assert all(
                len(c.completion_logprobs) == len(c.completion_token_ids)
                for c in captures
            )
        finally:
            await client.close()
            await host.close()
            await backend.close()
            gateway.stop()
            await native_http.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "context,reply,target,margin", [(32768, 4096, 16384, 0), (65536, 4096, 32768, None)]
)
def test_trainer_context_bounds_equal_the_actual_pinned_duck(
    context, reply, target, margin
):
    from training.config import InferenceConfig, full_game_context_contract

    config = InferenceConfig(
        base_url="http://unused/v1",
        model_id="model",
        game="ar25",
        games=["ar25"],
        max_runtime_minutes=15,
        agent_context_window=context,
        agent_max_completion_tokens=reply,
        agent_compact_to_tokens=target,
        agent_safety_margin_tokens=margin,
    )
    trigger, resolved_target, resolved_margin = full_game_context_contract(config)
    agent, _ = _fixture()
    agent.context_window_tokens = context
    agent.compact_to_tokens = target
    agent.safety_margin_tokens = resolved_margin
    agent.reply_cap_tokens = reply
    agent._live_params.max_tokens = reply
    agent.llm_config.sampling_profiles = {agent.profile: agent._live_params}
    assert trigger == agent._context_budget_tokens()
    assert resolved_target == agent._compact_target_tokens()
