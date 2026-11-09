"""Engine transport U5/U7: real socket drains, exact evidence and no replay."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import os
import pickle
import subprocess
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable

import httpx
import pytest
import ullm.backend
import ullm.backends.engine_proxy as proxy
import ullm.backends.proxy
import ullm.conv


class Tokenizer:
    chat_template = "template"
    special_tokens_map = {"eos_token": "<end>"}
    bos_token_id = None
    eos_token_id = 2
    pad_token_id = 2
    backend_tokenizer = types.SimpleNamespace(to_str=lambda: '{"model":{"vocab":{"a":1}}}')


ADAPTER = {"name": "policy", "id": 4, "path": "/adapter/4", "server_instance_id": "engine-1"}
META: dict[str, Any] = {
    "model_id": "test-model",
    "model_revision": "model-revision",
    "tokenizer_id": "test-tokenizer",
    "tokenizer_revision": "tokenizer-revision",
    "max_model_len": 1000,
    "tokenizer_fingerprint": proxy.tokenizer_fingerprint(Tokenizer()),
    "adapter": ADAPTER,
    **{
        key: getattr(Tokenizer, key)
        for key in ("chat_template", "special_tokens_map", "bos_token_id", "eos_token_id", "pad_token_id")
    },
}


class Engine:
    def __init__(self) -> None:
        self.requests = []
        self.accepted = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.completed = asyncio.Event()
        self.mutate: Callable[[Any], Any] = _identity

    async def generate(self, prompt: Any, params: Any, request_id: str, **kwargs: Any):
        self.requests.append((copy.deepcopy(prompt), copy.deepcopy(params), request_id, kwargs))
        self.accepted.set()
        output = types.SimpleNamespace(
            request_id=request_id,
            prompt_token_ids=prompt["prompt_token_ids"],
            finished=False,
            num_cached_tokens=1,
            lora_request=kwargs.get("lora_request"),
            adapter=dict(ADAPTER),
            outputs=[
                types.SimpleNamespace(
                    index=0,
                    text="",
                    token_ids=[],
                    logprobs=[],
                    cumulative_logprob=0.0,
                    finish_reason=None,
                    stop_reason=None,
                )
            ],
        )
        yield output
        await self.release.wait()
        output.finished = True
        output.outputs[0] = types.SimpleNamespace(
            index=0,
            text="engine text",
            token_ids=[17, 2],
            cumulative_logprob=-0.8,
            logprobs=[
                {17: types.SimpleNamespace(logprob=-0.3, rank=2, decoded_token="x")},
                {2: types.SimpleNamespace(logprob=-0.5, rank=1, decoded_token="<end>")},
            ],
            finish_reason="stop",
            stop_reason=2,
        )
        self.completed.set()
        yield self.mutate(output)


def _identity(output: Any) -> Any:
    return output


async def test_direct_engine_factory_constructs_and_owns_the_transport_backend():
    vllmx = pytest.importorskip("ullm.backends.vllmx")
    engine = proxy.EngineProxy(base_url="http://127.0.0.1:1", model="policy")
    factory = proxy.EngineProxyFactory(
        engine=engine, backend_kwargs={"accept_malformed_output": True}, status_interval_s=None
    )
    restored = pickle.loads(pickle.dumps(factory))
    backend = await restored.start()
    assert isinstance(backend, vllmx.VLLM)
    assert backend.engine is restored.engine
    assert backend.accept_malformed_output
    with pytest.raises(AssertionError, match="Cannot pickle a started"):
        pickle.dumps(restored)
    await restored.close()
    assert restored.engine._closing
    assert not engine._closing


async def test_direct_engine_factory_refuses_a_chat_backend():
    pytest.importorskip("ullm.backends.vllmx")
    import ullm.backends.oai

    chat = ullm.backends.oai.OpenAIServer(base_url="http://127.0.0.1:1/v1", model="policy")
    factory = proxy.EngineProxyFactory(engine=chat, status_interval_s=None)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="not a chat backend"):
        await factory.start()


async def _setup(monkeypatch: Any, tmp_path: Any, unix: bool = False, **server_options: Any) -> tuple[Any, Any, Any]:
    def load_tokenizer(self: Any, metadata: Any) -> Tokenizer:
        return Tokenizer()

    def sampling_factory(**fields: Any) -> dict[str, Any]:
        return fields

    def lora_factory(**fields: Any) -> Any:
        return types.SimpleNamespace(**fields)

    monkeypatch.setattr(proxy.EngineProxy, "_load_tokenizer", load_tokenizer)
    engine = Engine()
    handle = await proxy.start_engine_server(
        engine,
        META,
        unix_path=str(tmp_path / "engine.sock") if unix else None,
        sampling_factory=sampling_factory,
        lora_factory=lora_factory,
        **server_options,
    )
    client = proxy.EngineProxy(handle.base_url, "policy", unix_socket=handle.unix_path)
    await client.connect()
    return engine, handle, client


async def _draw(client: Any, request_id: str = "run#nonce", context: Any = None) -> list[Any]:
    return [
        output
        async for output in client.generate(
            {"type": "token", "prompt_token_ids": [1, 3]},
            {"temperature": 0.7, "max_tokens": 10},
            request_id,
            lora_request=client.lora_request,
            request_context=context,
        )
    ]


@pytest.mark.parametrize("unix", [False, True])
async def test_engine_socket_preserves_context_and_raw_final_output(monkeypatch: Any, tmp_path: Any, unix: bool):
    engine, handle, client = await _setup(monkeypatch, tmp_path, unix)
    context = {"lineage": {"epoch": 2}, "progress": {"after_nonce": "before", "decision_clock": 3.5}}
    try:
        outputs = await _draw(client, context=context)
        assert len(outputs) == len(engine.requests) == 1
        assert engine.requests[0][3]["request_context"] == context
        assert outputs[0].request_id == "run#nonce"
        assert outputs[0].outputs[0].text == "engine text"
        assert outputs[0].outputs[0].token_ids == [17, 2]
        assert outputs[0].outputs[0].logprobs[0][17].logprob == -0.3
        assert outputs[0].outputs[0].stop_reason == 2
        assert outputs[0].adapter == ADAPTER
        assert outputs[0].lora_request.lora_path == "/adapter/4"
        assert outputs[0].num_cached_tokens == 1
        assert await client.allows_generation_replay() is False
        restored = pickle.loads(pickle.dumps(client))
        assert restored.metadata == {} and restored._http is None and restored._tokenizer is None
    finally:
        await client.aclose()
        await handle.aclose()


async def test_model_metadata_is_resolved_before_admission_and_frozen_for_draw():
    import aiohttp.web

    metadata = copy.deepcopy(META)
    described = []
    engine = Engine()
    engine.release.clear()

    def describe(model: str | None) -> dict[str, Any]:
        described.append(model)
        if model != "policy":
            raise aiohttp.web.HTTPBadRequest(text="unknown model alias")
        return metadata

    def clear_adapter(output: Any) -> Any:
        output.adapter = None
        return output

    def sampling_factory(**fields: Any) -> Any:
        return fields

    def lora_factory(**fields: Any) -> Any:
        return types.SimpleNamespace(**fields)

    engine.mutate = clear_adapter
    handle = await proxy.start_engine_server(
        engine, describe, sampling_factory=sampling_factory, lora_factory=lora_factory
    )
    body = {
        "model": "unknown",
        "request_id": "run#same-nonce",
        "prompt": {"type": "token", "prompt_token_ids": [1, 3]},
        "sampling_params": {"temperature": 0.7, "max_tokens": 10},
        "lora_request": {"lora_name": "policy", "lora_int_id": 4, "lora_path": "/adapter/4"},
        "request_context": {},
    }
    task = None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            rejected = await client.post(f"{handle.base_url}/generate", json=body)
            assert rejected.status_code == 400
            assert engine.requests == []
            body["model"] = "policy"
            task = asyncio.create_task(client.post(f"{handle.base_url}/generate", json=body))
            await asyncio.wait_for(engine.accepted.wait(), 5)
            metadata["adapter"]["server_instance_id"] = "later-engine"
            engine.release.set()
            response = await task
            assert response.status_code == 200
            assert response.json()["adapter"] == ADAPTER
            assert described == ["unknown", "policy"]
            assert len(engine.requests) == 1
    finally:
        engine.release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await handle.aclose()


async def test_cancelled_draw_and_close_drain_the_same_response(monkeypatch: Any, tmp_path: Any):
    engine, handle, client = await _setup(monkeypatch, tmp_path)
    engine.release.clear()
    task = asyncio.create_task(_draw(client))
    closing = None
    try:
        await asyncio.wait_for(engine.accepted.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        closing = asyncio.create_task(client.aclose())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), 0.03)
        assert not task.done()
        engine.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        await asyncio.wait_for(closing, 5)
        assert engine.completed.is_set() and len(engine.requests) == 1
        assert client._requests == {}
    finally:
        engine.release.set()
        await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await client.aclose()
        await handle.aclose()


async def test_cancellation_during_upload_does_not_close_partial_request(monkeypatch: Any, tmp_path: Any):
    engine, handle, client = await _setup(monkeypatch, tmp_path)
    await client._http.aclose()
    started = asyncio.Event()
    release = asyncio.Event()

    class Upload(httpx.AsyncByteStream):
        def __init__(self, content: bytes) -> None:
            self.content = content

        async def __aiter__(self):
            yield self.content[:1]
            started.set()
            await release.wait()
            yield self.content[1:]

    class Transport(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            request.stream = Upload(await request.aread())
            return await super().handle_async_request(request)

    client._http = httpx.AsyncClient(transport=Transport(), timeout=None)
    task = asyncio.create_task(_draw(client))
    try:
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not engine.accepted.is_set() and not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert engine.completed.is_set() and len(engine.requests) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await client.aclose()
        await handle.aclose()


@pytest.mark.parametrize("mutation", ["request", "prompt", "adapter", "probability", "missing_probability", "aborted"])
async def test_corrupt_engine_evidence_fails_once(monkeypatch: Any, tmp_path: Any, mutation: str):
    engine, handle, client = await _setup(monkeypatch, tmp_path)

    def mutate(output: Any) -> Any:
        if mutation == "request":
            output.request_id = "different"
        elif mutation == "prompt":
            output.prompt_token_ids = [9]
        elif mutation == "adapter":
            output.adapter["server_instance_id"] = "restarted"
        elif mutation == "probability":
            output.outputs[0].logprobs[0][17].logprob = 0.5
        elif mutation == "missing_probability":
            output.outputs[0].logprobs[0] = {}
        else:
            output.outputs[0].finish_reason = "abort"
        return output

    engine.mutate = mutate
    try:
        with pytest.raises(ValueError):
            await _draw(client)
        with pytest.raises(RuntimeError, match="has failed"):
            await _draw(client, "run#another")
        assert len(engine.requests) == 1
    finally:
        with contextlib.suppress(ValueError):
            await client.aclose()
        await handle.aclose()


async def test_actual_server_disconnect_is_not_retried(monkeypatch: Any, tmp_path: Any):
    _, handle, client = await _setup(monkeypatch, tmp_path)
    attempts = []

    async def disconnect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        attempts.append(1)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(disconnect, "127.0.0.1", 0)
    client.base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        with pytest.raises(httpx.RemoteProtocolError):
            await _draw(client)
        assert attempts == [1]
        with pytest.raises(RuntimeError, match="has failed"):
            await _draw(client, "run#different")
        assert attempts == [1]
    finally:
        with contextlib.suppress(httpx.RemoteProtocolError):
            await client.aclose()
        server.close()
        await server.wait_closed()
        await handle.aclose()


@pytest.mark.parametrize("mismatch", ["chat_template", "fingerprint", "revision"])
def test_tokenizer_rendering_identity_is_checked(monkeypatch: Any, mismatch: str):
    tokenizer = Tokenizer()

    def load_tokenizer(*args: Any, **kwargs: Any) -> Any:
        return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=load_tokenizer)),
    )
    metadata = copy.deepcopy(META)
    client = proxy.EngineProxy("http://unused", "policy", tokenizer_revision="tokenizer-revision")
    if mismatch == "chat_template":
        metadata["chat_template"] = "different"
    elif mismatch == "fingerprint":
        metadata["tokenizer_fingerprint"]["sha256"] = "different"
    else:
        metadata["tokenizer_revision"] = "different"
    with pytest.raises(ValueError, match="differs"):
        client._load_tokenizer(metadata)


def test_import_and_unpickle_do_not_import_gpu_runtime():
    code = """
import pickle,sys
from ullm.backends.engine_proxy import EngineProxy
x=pickle.loads(pickle.dumps(EngineProxy('http://unused','policy')))
assert x.metadata == {}
assert not any(name in sys.modules for name in ('vllm','torch','transformers'))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20, env=os.environ)
    assert result.returncode == 0, result.stderr


def test_real_sampling_params_preserve_effective_fields_and_nonneutral_extensions():
    import vllm

    extensions: dict[str, Any] = (
        {"zero_recurrent_state": True, "zero_recurrent_at": 0}
        if "zero_recurrent_state" in vllm.SamplingParams.__struct_fields__
        else {}
    )
    params = vllm.SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_tokens=70,
        logprobs=0,
        skip_special_tokens=False,
        presence_penalty=0.2,
        seed=5,
        **extensions,
    )
    wire = proxy.sampling_to_wire(params)
    assert wire["temperature"] == 0.7 and wire["top_p"] == 0.9 and wire["top_k"] == 40
    assert wire["presence_penalty"] == 0.2 and wire["repetition_penalty"] == 1.0
    assert wire["seed"] == 5
    for name, value in extensions.items():
        assert wire[name] == value
    assert wire["max_tokens"] == 70 and wire["logprobs"] == 0 and wire["skip_special_tokens"] is False
    assert "output_text_buffer_length" not in wire and "_all_stop_token_ids" not in wire
    assert "stop" not in wire and "detokenize" not in wire
    stopped = proxy.sampling_to_wire(vllm.SamplingParams(stop=["stop string"], logit_bias={1: 0.2}))
    assert stopped["stop"] == ["stop string"] and stopped["logit_bias"] == {1: 0.2}


def test_nonreproducible_private_sampling_state_is_rejected():
    import vllm

    params = vllm.SamplingParams()
    params._all_stop_token_ids = {123}
    with pytest.raises(ValueError, match="_all_stop_token_ids"):
        proxy.sampling_to_wire(params)


async def test_server_shutdown_waits_for_engine_capture(monkeypatch: Any, tmp_path: Any):
    engine, handle, client = await _setup(monkeypatch, tmp_path)
    engine.release.clear()
    task = asyncio.create_task(_draw(client))
    closing = None
    try:
        await asyncio.wait_for(engine.accepted.wait(), 5)
        closing = asyncio.create_task(handle.aclose())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), 0.03)
        engine.release.set()
        assert len(await asyncio.wait_for(task, 5)) == 1
        await asyncio.wait_for(closing, 5)
        assert engine.completed.is_set() and len(engine.requests) == 1
    finally:
        engine.release.set()
        await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await client.aclose()
        await handle.aclose()


@dataclass
class RemoteBackend(ullm.backend.DummyBackend):
    engine: Any = None

    async def _generate(self, sampling_params: Any, llm_input: Any):
        async for output in self.engine.generate(
            {"type": "token", "prompt_token_ids": [1, 3]},
            {"max_tokens": 10},
            "run#nonce",
            lora_request=self.engine.lora_request,
        ):
            yield output.outputs[0].text

    async def allows_generation_replay(self) -> bool:
        return await self.engine.allows_generation_replay()


@pytest.mark.parametrize("hops", [1, 2])
async def test_game_proxy_deadline_returns_while_engine_draw_drains(monkeypatch: Any, tmp_path: Any, hops: int):
    engine, server, remote = await _setup(monkeypatch, tmp_path)
    engine.release.clear()
    backend = RemoteBackend(engine=remote)
    handles = []
    clients = []
    host: Any = backend
    for hop in range(hops):
        path = str(tmp_path / f"backend-{hop}.sock")
        handles.append(await ullm.backends.proxy.start_backend_server(host, path))
        host = ullm.backends.proxy.ProxyBackend(socket_path=path, run_id="run")
        clients.append(host)

    async def respond() -> None:
        conversation = ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys")),
            elements=[
                ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")),
                ullm.conv.ResponseTurn(status="intermediate"),
            ],
            status="intermediate",
        )
        async for _ in clients[-1].respond(ullm.backend.GenericSamplingParameters(), conversation, run_id="run"):
            pytest.fail("a cancelled game must not consume the terminal draw")

    task = asyncio.create_task(respond())
    closing = None
    try:
        await asyncio.wait_for(engine.accepted.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        for client in reversed(clients):
            await client.close()
        for handle in reversed(handles):
            await handle.close()
        closing = asyncio.create_task(remote.aclose())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), 0.03)
        engine.release.set()
        await asyncio.wait_for(closing, 5)
        assert engine.completed.is_set() and len(engine.requests) == 1
    finally:
        engine.release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for client in reversed(clients):
            await client.close()
        for handle in reversed(handles):
            await handle.close()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await remote.aclose()
        await server.aclose()


async def test_hung_draw_has_a_fatal_deadline_despite_repeated_cancellation(monkeypatch: Any, tmp_path: Any):
    engine, handle, client = await _setup(monkeypatch, tmp_path)
    client.generation_timeout_s = 0.06
    engine.release.clear()
    task = asyncio.create_task(_draw(client))
    closing = None
    try:
        await asyncio.wait_for(engine.accepted.wait(), 5)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        closing = asyncio.create_task(client.aclose())
        with pytest.raises(TimeoutError, match="deadline"):
            await asyncio.wait_for(task, 1)
        with pytest.raises(TimeoutError, match="deadline"):
            await asyncio.wait_for(closing, 1)
        with pytest.raises(RuntimeError, match="has failed"):
            await _draw(client, "run#another")
        assert len(engine.requests) == 1
    finally:
        engine.release.set()
        await asyncio.gather(task, *([closing] if closing else []), return_exceptions=True)
        with contextlib.suppress(TimeoutError):
            await client.aclose()
        await handle.aclose()


async def test_server_generation_deadline_bounds_shutdown(monkeypatch: Any, tmp_path: Any):
    engine, handle, client = await _setup(monkeypatch, tmp_path, max_generation_seconds=0.06)
    assert client.generation_timeout_s == 0.06
    engine.release.clear()
    task = asyncio.create_task(_draw(client))
    try:
        await asyncio.wait_for(engine.accepted.wait(), 5)
        with pytest.raises((TimeoutError, httpx.HTTPStatusError)):
            await asyncio.wait_for(task, 1)
        await asyncio.wait_for(handle.aclose(), 1)
        assert len(engine.requests) == 1
    finally:
        engine.release.set()
        await asyncio.gather(task, return_exceptions=True)
        with contextlib.suppress(TimeoutError, httpx.HTTPStatusError):
            await client.aclose()
        await handle.aclose()


async def test_request_cap_keeps_old_ids_reserved_without_new_engine_draws(monkeypatch: Any, tmp_path: Any):
    engine, handle, client = await _setup(monkeypatch, tmp_path, max_seen_requests=1)
    try:
        await _draw(client)
        async with httpx.AsyncClient() as http:
            body = {
                "model": "policy",
                "request_id": "run#nonce",
                "prompt": {"type": "token", "prompt_token_ids": [1, 3]},
                "sampling_params": {},
                "lora_request": None,
                "request_context": {},
            }
            duplicate = await http.post(f"{handle.base_url}/generate", json=body)
            assert duplicate.status_code == 400
            body["request_id"] = "run#new"
            capped = await http.post(f"{handle.base_url}/generate", json=body)
            assert capped.status_code == 503 and "admission cap" in capped.text
            body["request_id"] = "run#nonce"
            duplicate = await http.post(f"{handle.base_url}/generate", json=body)
            assert duplicate.status_code == 400
        assert len(engine.requests) == 1
    finally:
        await client.aclose()
        await handle.aclose()


@pytest.mark.parametrize(
    "explicit_path,opt_in", [(None, False), ("/chosen/tokenizer", False), ("/chosen/tokenizer", True)]
)
def test_tokenizer_code_trust_requires_explicit_caller_choice(
    monkeypatch: Any, explicit_path: str | None, opt_in: bool
):
    loaded = []

    def load_tokenizer(path: str, **kwargs: Any) -> Any:
        loaded.append((path, kwargs))
        return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=load_tokenizer)),
    )
    client = proxy.EngineProxy("http://unused", "policy", tokenizer_path=explicit_path, trust_remote_code=opt_in)
    client._load_tokenizer(META)
    assert loaded[0][0] == (explicit_path or META["tokenizer_id"])
    assert loaded[0][1]["trust_remote_code"] is opt_in
    client = proxy.EngineProxy("http://unused", "policy", trust_remote_code=True)
    with pytest.raises(ValueError, match="explicit tokenizer_path"):
        client._load_tokenizer(META)
    assert len(loaded) == 1
