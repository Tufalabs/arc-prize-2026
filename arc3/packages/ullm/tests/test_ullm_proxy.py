"""ProxyBackend: a whole backend served over a Unix socket (CPU-only —
the host side here is a scripted backend, which is exactly the point:
the proxy is backend-agnostic)."""

import asyncio
import contextlib
import pathlib
import pickle
import struct
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import pytest
import ullm.backend
import ullm.backends.engine_proxy
import ullm.backends.proxy
import ullm.backends.stub
import ullm.conv
import ullm.manager
import ullm.tool
import ullm.utils


def _make_tm(backend: ullm.backend.Backend, *, tools: list[ullm.tool.Tool] | None = None) -> ullm.manager.TurnManager:
    return ullm.utils.make_turn_manager(backend, tools=tools, reasoning_effort="low", max_tokens=512)


async def _run_prompt(tm: ullm.manager.TurnManager, text: str) -> str:
    tm.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text)))
    if tm.state != "ready":
        tm.rearm()
    await tm.run()
    return tm.conversation.last_turn.last_response.final_content.to_string()


def _fresh_conversation():
    """A minimal conversation in the shape respond() expects: one user
    prompt inside an intermediate turn."""
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="be brief"))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="hi")))
    conversation.elements.append(ullm.conv.ResponseTurn(status="intermediate"))
    conversation.status = "intermediate"
    return conversation


async def test_passthrough_and_separate_accounting(tmp_path: pathlib.Path):
    """Responses pass through unchanged; the proxy keeps its OWN request
    counter/usage while the host backend independently keeps its own."""
    host_backend = ullm.backends.stub.StubBackend.from_script("alpha", "beta")
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    try:
        tm = _make_tm(client)
        assert await _run_prompt(tm, "one") == "alpha"
        assert await _run_prompt(tm, "two") == "beta"
        # Both sides account for themselves.
        assert client.total_usage.num_requests == 2
        assert host_backend.total_usage.num_requests == 2
        assert client._request_counter == 2
        assert client.total_usage.completion_tokens > 0
    finally:
        await client.close()
        await handle.close()


async def test_default_backend_replays_after_real_socket_loss(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = ullm.backends.proxy
    host_backend = ullm.backends.stub.StubBackend.from_script("reply")
    accepted = asyncio.Event()
    release = asyncio.Event()
    original_generate = host_backend._generate
    attempts = 0

    async def generate(params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        nonlocal attempts
        attempts += 1
        async for item in original_generate(params, llm_input):
            if attempts == 1:
                accepted.set()
                await release.wait()
            yield item

    monkeypatch.setattr(host_backend, "_generate", generate)
    monkeypatch.setattr(proxy, "_RECONNECT_BASE_DELAY_S", 0.0)
    monkeypatch.setattr(proxy, "_RECONNECT_DELAY_CAP_S", 0.0)
    socket_path = str(tmp_path / "replay.sock")
    handle = await proxy.start_backend_server(host_backend, socket_path)
    client = proxy.ProxyBackend(socket_path=socket_path)
    task = asyncio.create_task(_run_prompt(_make_tm(client), "go"))
    try:
        await asyncio.wait_for(accepted.wait(), timeout=5)
        assert client._writer is not None
        client._writer.transport.abort()
        assert await asyncio.wait_for(task, timeout=5) == "reply"
        assert attempts == 2
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()
        await handle.close()


async def test_tool_call_loop_through_proxy(tmp_path: pathlib.Path):
    """Tool calls parsed host-side arrive as ToolCallContent; the tool runs
    CLIENT-side and its result ships back for the next round."""
    host_backend = ullm.backends.stub.StubBackend.from_script(
        ullm.conv.LLMResponse(
            final_content=ullm.conv.ToolCallContent(tool_name="dummy_compute", arguments={"x": 2, "y": 3})
        ),
        "result received",
    )
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    try:
        tm = _make_tm(client, tools=[ullm.tool.DummyTool()])
        answer = await _run_prompt(tm, "go")
        results = [e for e in tm.conversation.last_turn.elements if isinstance(e, ullm.conv.ToolResult)]
        assert results and "12" in results[0].content.to_string()
        assert answer == "result received"
    finally:
        await client.close()
        await handle.close()


async def test_client_pickles_and_reconnects(tmp_path: pathlib.Path):
    """A used client pickles (config only) and the restored copy opens its
    own fresh connection with zeroed accounting."""
    host_backend = ullm.backends.stub.StubBackend.from_script("alpha", "beta", "gamma")
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    try:
        await _run_prompt(_make_tm(client), "one")
        restored = pickle.loads(pickle.dumps(client))
        assert restored.total_usage.num_requests == 0  # own, fresh accounting
        answer = await _run_prompt(_make_tm(restored), "two")
        assert answer == "beta"  # same host script, new connection
        assert restored.total_usage.num_requests == 1
        assert client.total_usage.num_requests == 1  # original untouched
        await restored.close()
    finally:
        await client.close()
        await handle.close()


@dataclass
class _SbrRecordingBackend(ullm.backend.DummyBackend):
    """Records the resolved send_back_reasoning its _prepare_input receives."""

    seen: list = field(default_factory=list)

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        self.seen.append(send_back_reasoning)
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)


async def test_reasoning_passback_resolves_at_the_wrapped_backend(tmp_path: pathlib.Path):
    """The policy is per-call config: unset -> the WRAPPED backend's default
    applies, never the proxy client's own; set on sampling_params -> it
    rides the wire and overrides."""
    host_backend = _SbrRecordingBackend(send_back_reasoning="none")
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    try:
        tm = _make_tm(client)
        await _run_prompt(tm, "one")
        assert host_backend.seen == ["none"]  # wrapped default, not the client's "current_turn"
        tm.response_manager.sampling_params.send_back_reasoning = "all"
        await _run_prompt(tm, "two")
        assert host_backend.seen == ["none", "all"]  # per-call value crossed the wire
    finally:
        await client.close()
        await handle.close()


@dataclass
class _HangingBackend(ullm.backend.DummyBackend):
    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        yield "partial"
        # Outlasts every poll window here by a wide margin, so a test that
        # waits for this generation to disappear is watching a cancellation
        # and never a natural completion. Bounded so that a watchdog that
        # stops firing fails the suite instead of stalling it.
        await asyncio.sleep(30)


async def test_cancellation_aborts_host_side(tmp_path: pathlib.Path):
    """Cancelling the client turn unwinds through the socket: the host's
    request task is cancelled instead of running to the timeout."""
    host_backend = _HangingBackend()
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    try:
        tm = _make_tm(client)
        tm.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="hi")))
        task = asyncio.create_task(tm.run())
        for _ in range(200):
            if handle.state.tasks:
                break
            await asyncio.sleep(0.01)
        assert handle.state.tasks, "request never reached the host"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(200):
            if not handle.state.tasks:
                break
            await asyncio.sleep(0.01)
        assert not handle.state.tasks  # host-side generation cancelled too
        assert client._in_flight == 0
    finally:
        await client.close()
        await handle.close()


@dataclass
class _FlakyHostBackend(ullm.backend.DummyBackend):
    failures_left: int = 1

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        if self.failures_left > 0:
            self.failures_left -= 1
            raise ullm.backend.MalformedOutputError("injected host-side parse failure")
        super()._process_output(sampling_params, llm_output, conversation, turn, element)


async def test_host_malformed_error_retries_client_side(tmp_path: pathlib.Path):
    """A MalformedOutputError raised by the wrapped backend crosses the wire
    and drives the CLIENT's retry machinery; the recovery bookkeeping lands
    in the client's own warning log, not the host's."""
    host_backend = _FlakyHostBackend(failures_left=1)
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    try:
        tm = _make_tm(client)
        answer = await _run_prompt(tm, "hello")
        assert answer  # recovered on retry
        assert tm.conversation.last_turn.last_response.malformed_retries == 1
        client_types = [w["type"] for w in client.warning_log]
        assert "malformed_output_retry" in client_types
        assert "malformed_output_recovered" in client_types
        assert host_backend.warning_log == []  # manager-side bookkeeping is the client's own
    finally:
        await client.close()
        await handle.close()


async def test_count_hydration_is_memoized_per_payload(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Consecutive counts of an unchanged conversation (the duck's trim
    fired-check and its reconciliation count) ship byte-identical pickles,
    and counting never mutates the conversation — so the host must hydrate
    once and reuse, not re-run the O(context) unpickle + unwrap on its
    event loop for every count. A changed conversation is a different
    payload and must re-hydrate."""
    host_backend = ullm.backends.stub.StubBackend.from_script("alpha")
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    unwraps: list[int] = []
    real_unwrap = ullm.backends.proxy._unwrap_conversation

    def _counting_unwrap(conversation: ullm.conv.Conversation) -> None:
        unwraps.append(1)
        real_unwrap(conversation)

    monkeypatch.setattr(ullm.backends.proxy, "_unwrap_conversation", _counting_unwrap)
    try:
        conversation = _fresh_conversation()
        params = ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=16)
        await client.token_count(params, conversation)
        await client.token_count(params, conversation)
        assert len(unwraps) == 1, "byte-identical count payloads must share one hydration"
        assert len(handle.state.count_conv_cache) == 1
        assert (handle.state.count_cache_hits, handle.state.count_cache_misses) == (1, 1)
        conversation.elements.insert(
            1, ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="now the prompt differs"))
        )
        await client.token_count(params, conversation)
        assert len(unwraps) == 2, "a changed conversation is a new payload and must re-hydrate"
        assert len(handle.state.count_conv_cache) == 2
        assert handle.state.count_cache_payload_bytes == sum(
            size for size, _ in handle.state.count_conv_cache.values()
        ), "the byte accounting must track exactly what the cache retains"
    finally:
        await client.close()
        await handle.close()


async def test_count_cache_evicts_to_its_byte_budget(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """The cache is bounded by retained payload BYTES, not entry count —
    hydrated conversations are a small multiple of their pickle, so the
    budget is what bounds host memory. Oldest entries go first, and the
    newest always survives (its repeat count is the imminent one)."""
    host_backend = ullm.backends.stub.StubBackend.from_script("alpha")
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    monkeypatch.setattr(ullm.backends.proxy, "_COUNT_CACHE_MAX_PAYLOAD_BYTES", 1)
    try:
        params = ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=16)
        for text in ("first", "second"):
            conversation = _fresh_conversation()
            conversation.elements.insert(1, ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text)))
            await client.token_count(params, conversation)
        assert len(handle.state.count_conv_cache) == 1, "over budget, only the newest entry may survive"
        assert handle.state.count_cache_payload_bytes == next(
            size for size, _ in handle.state.count_conv_cache.values()
        )
    finally:
        await client.close()
        await handle.close()


async def test_proxy_client_status_labels_claim_only_serialization(tmp_path: pathlib.Path):
    """In the solver's proxy topology the client's prepare/parse hooks only
    (de)serialize — the host renders and parses inside the client's
    generation wait. The published labels must claim exactly that much, or
    the status line charges host CPU to phases that never saw it."""
    host_backend = ullm.backends.stub.StubBackend.from_script("alpha")
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    events: list[tuple[str, str]] = []

    @contextlib.contextmanager
    def hook(label: str):
        events.append(("enter", label))
        try:
            yield
        finally:
            events.append(("exit", label))

    client.status_hook = hook
    try:
        tm = _make_tm(client)
        assert await _run_prompt(tm, "one") == "alpha"
        labels = [label for kind, label in events if kind == "enter"]
        assert labels == [
            "serializing LLM request",
            "waiting for LLM response",
            "deserializing LLM response",
        ], labels
    finally:
        await client.close()
        await handle.close()


@pytest.mark.parametrize("watchdog_timeout_s", [None, 3600.0])
async def test_backend_generation_deadline_is_not_an_inactivity_timeout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, watchdog_timeout_s: float | None
) -> None:
    """An engine's own deadline crosses the socket with its actual cause,
    whether the proxy inactivity watchdog is enabled or disabled."""
    host_backend = ullm.backends.stub.StubBackend.from_script("unused")
    pending = asyncio.get_running_loop().create_future()

    async def generate(params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        await ullm.backends.engine_proxy._drain(pending, timeout_s=0.01)
        yield "unreachable"

    monkeypatch.setattr(host_backend, "_generate", generate)
    handle = await ullm.backends.proxy.start_backend_server(
        host_backend, str(tmp_path / "b.sock"), default_inactivity_timeout_s=watchdog_timeout_s
    )
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    try:
        with pytest.raises(TimeoutError, match="engine generation drain exceeded its deadline; outcome is uncertain"):
            await _run_prompt(_make_tm(client), "hi")
        assert pending.cancelled()
        assert host_backend._request_counter == 1
        assert not handle.state.tasks
    finally:
        await client.close()
        await handle.close()


async def test_inactivity_watchdog_surfaces_a_stalled_backend(tmp_path: pathlib.Path):
    """A backend that stops producing must become a client-visible error,
    not an eternal await: the whole point of the watchdog (a silently
    dropped vLLM request used to hang entire runs)."""
    host_backend = _HangingBackend()
    handle = await ullm.backends.proxy.start_backend_server(
        host_backend, str(tmp_path / "b.sock"), default_inactivity_timeout_s=0.3
    )
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"))
    try:
        tm = _make_tm(client)
        with pytest.raises(RuntimeError, match="no output for"):
            await _run_prompt(tm, "hi")
        # The host stays usable afterwards: only that request died.
        for _ in range(200):
            if not handle.state.tasks:
                break
            await asyncio.sleep(0.01)
        assert not handle.state.tasks
    finally:
        await client.close()
        await handle.close()


# --- host resilience: one bad client must not take the server down --------


async def _raw_connect(socket_path: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a raw framed connection and consume the HELLO."""
    reader, writer = await asyncio.open_unix_connection(socket_path)
    await ullm.backends.proxy._read_frame(reader)
    return reader, writer


async def _assert_host_still_serves(socket_path: str, expected: str) -> None:
    client = ullm.backends.proxy.ProxyBackend(socket_path=socket_path)
    try:
        assert await _run_prompt(_make_tm(client), "still there?") == expected
    finally:
        await client.close()


async def test_host_survives_client_disconnect_mid_request(tmp_path: pathlib.Path):
    """A worker that dies mid-generation (crash, SIGKILL) must leave the
    host serving everyone else — and not leak the abandoned task."""
    sock = str(tmp_path / "b.sock")
    handle = await ullm.backends.proxy.start_backend_server(_HangingBackend(), sock)
    try:
        reader, writer = await _raw_connect(sock)
        await ullm.backends.proxy._write_frame(
            writer,
            asyncio.Lock(),
            ullm.backends.proxy._Tag.RUN,
            {
                "req_id": "r0",
                "conversation_pickle": pickle.dumps(_fresh_conversation()),
                "sampling_params": ullm.backend.GenericSamplingParameters(max_tokens=16),
            },
        )
        for _ in range(200):  # wait until the host is generating
            if handle.state.tasks:
                break
            await asyncio.sleep(0.01)
        assert handle.state.tasks
        writer.close()  # the "worker crashed" moment
        for _ in range(200):
            if not handle.state.tasks:
                break
            await asyncio.sleep(0.01)
        assert not handle.state.tasks, "abandoned generation was not cancelled"
        del reader
        handle.state.backend = ullm.backends.stub.StubBackend.from_script("alive")
        await _assert_host_still_serves(sock, "alive")
    finally:
        await handle.close()


async def test_host_survives_garbage_frame(tmp_path: pathlib.Path):
    """Unparseable bytes on one connection drop that connection only."""
    sock = str(tmp_path / "b.sock")
    handle = await ullm.backends.proxy.start_backend_server(
        ullm.backends.stub.StubBackend.from_script("alive", on_exhausted="repeat_last"), sock
    )
    try:
        _reader, writer = await _raw_connect(sock)
        try:
            writer.write(struct.pack("!I", 5) + b"junk!")  # length-prefixed nonsense
            await writer.drain()
            await asyncio.sleep(0.05)
            await _assert_host_still_serves(sock, "alive")
        finally:
            writer.close()
    finally:
        await handle.close()


async def test_host_ignores_unknown_tag(tmp_path: pathlib.Path):
    """Forward compatibility: an unrecognized frame tag is skipped, and the
    same connection keeps working."""
    sock = str(tmp_path / "b.sock")
    handle = await ullm.backends.proxy.start_backend_server(
        ullm.backends.stub.StubBackend.from_script("alive", on_exhausted="repeat_last"), sock
    )
    try:
        _reader, writer = await _raw_connect(sock)
        try:
            await ullm.backends.proxy._write_frame(writer, asyncio.Lock(), "SOME_FUTURE_TAG", {"req_id": "r0"})
            await asyncio.sleep(0.05)
            await _assert_host_still_serves(sock, "alive")
        finally:
            writer.close()
    finally:
        await handle.close()


# --- concurrency: the multiplexed connection is the deployment shape ------


@dataclass
class _EchoBackend(ullm.backend.DummyBackend):
    """Echoes the last user prompt after a per-request delay taken from the
    prompt itself ("d<seconds*100>:<text>"), so completion order can be made
    to differ from submission order."""

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        delay_spec, _, text = llm_input.partition(":")
        await asyncio.sleep(int(delay_spec[1:]) / 100)
        yield text

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        prompts = [e for e in conversation.elements if isinstance(e, ullm.conv.UserPrompt)]
        return prompts[-1].content.to_string()

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        element.final_content = ullm.conv.StringContent(text=llm_output)
        element.token_usage = ullm.backend.TokenUsage(completion_tokens=1, num_requests=1)
        element.status = "processed"


async def test_concurrent_requests_do_not_cross_talk(tmp_path: pathlib.Path):
    """Many in-flight requests on ONE connection: every reply must reach the
    request that asked for it even when they finish out of order."""
    sock = str(tmp_path / "b.sock")
    host_backend = _EchoBackend()
    handle = await ullm.backends.proxy.start_backend_server(host_backend, sock)
    client = ullm.backends.proxy.ProxyBackend(socket_path=sock)
    try:
        n = 6
        # Request i sleeps (n-i)*10ms: submission order is the reverse of
        # completion order, so a reply-to-request mapping by arrival breaks.
        prompts = [f"d{(n - i) * 10}:answer-{i}" for i in range(n)]
        answers = await asyncio.gather(*(_run_prompt(_make_tm(client), p) for p in prompts))
        assert answers == [f"answer-{i}" for i in range(n)]
        assert client.total_usage.num_requests == n
    finally:
        await client.close()
        await handle.close()


async def test_two_clients_share_one_host(tmp_path: pathlib.Path):
    """Independent clients on one host: separate accounting client-side,
    one combined total host-side."""
    sock = str(tmp_path / "b.sock")
    host_backend = _EchoBackend()
    handle = await ullm.backends.proxy.start_backend_server(host_backend, sock)
    a = ullm.backends.proxy.ProxyBackend(socket_path=sock)
    b = ullm.backends.proxy.ProxyBackend(socket_path=sock)
    try:
        answers = await asyncio.gather(
            _run_prompt(_make_tm(a), "d5:from-a"),
            _run_prompt(_make_tm(b), "d1:from-b"),
            _run_prompt(_make_tm(a), "d3:from-a-again"),
        )
        assert answers == ["from-a", "from-b", "from-a-again"]
        assert a.total_usage.num_requests == 2
        assert b.total_usage.num_requests == 1
        assert host_backend.total_usage.num_requests == 3  # host sums all clients
    finally:
        await a.close()
        await b.close()
        await handle.close()


# --- the real deployment shape: pickled client, separate OS process -------

_CHILD_SCRIPT = """
import asyncio, pickle, sys
import ullm.conv, ullm.utils

async def main(path):
    with open(path, "rb") as f:
        backend = pickle.load(f)
    tm = ullm.utils.make_turn_manager(backend, max_tokens=64)
    tm.conversation.elements.append(
        ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="d1:from-a-subprocess")))
    await tm.run()
    print("ANSWER:" + tm.conversation.last_turn.last_response.final_content.to_string())
    print("REQUESTS:%d" % backend.total_usage.num_requests)
    await backend.close()

asyncio.run(main(sys.argv[1]))
"""


async def test_pickled_client_works_from_another_process(tmp_path: pathlib.Path):
    """What production actually does: the host serves in this process while a
    worker in a SEPARATE process unpickles a ProxyBackend and runs a turn.
    Catches loop-bound or process-local state hiding in the pickle path,
    which same-process tests cannot see."""
    import sys

    sock = str(tmp_path / "b.sock")
    handle = await ullm.backends.proxy.start_backend_server(_EchoBackend(), sock)
    script = tmp_path / "child.py"
    script.write_text(_CHILD_SCRIPT)
    pickle_path = tmp_path / "backend.pkl"
    pickle_path.write_bytes(pickle.dumps(ullm.backends.proxy.ProxyBackend(socket_path=sock)))
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            str(pickle_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        out = stdout.decode()
        assert proc.returncode == 0, f"child failed:\n{stderr.decode()}"
        assert "ANSWER:from-a-subprocess" in out, out
        assert "REQUESTS:1" in out  # the child accounts for its own traffic
    finally:
        await handle.close()


# --- the shipped-conversation contract (regression) -----------------------


@dataclass
class _StrictPrepareBackend(ullm.backend.DummyBackend):
    """Asserts, at prepare time, that the conversation looks exactly like it
    would for a direct backend. Both invariants were violated by earlier
    proxy versions and cost a full GPU debugging round each."""

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        for el in conversation.elements:
            if not isinstance(el, ullm.conv.ResponseTurn):
                continue
            for sub in el.elements:
                if not isinstance(sub, ullm.conv.LLMResponse):
                    continue
                assert sub.status == "processed", "prepare must not see the not-yet-generated output element"
                assert sub._model_contents is not None, (
                    "prior responses must arrive with model contents (history reconstruction)"
                )
                assert not isinstance(sub._model_contents, ullm.backends.proxy._OpaqueModelContents), (
                    "model contents must be unwrapped before the wrapped backend sees them"
                )
        return None

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        element._model_contents = {"raw": llm_output}  # backend-native payload


async def test_shipped_conversation_matches_direct_shape(tmp_path: pathlib.Path):
    """Multi-round: the second request's prepare sees the first response
    complete and unwrapped, and no placeholder for its own output."""
    sock = str(tmp_path / "b.sock")
    handle = await ullm.backends.proxy.start_backend_server(_StrictPrepareBackend(), sock)
    client = ullm.backends.proxy.ProxyBackend(socket_path=sock)
    try:
        tm = _make_tm(client)
        await _run_prompt(tm, "one")
        await _run_prompt(tm, "two")  # this is the round that used to fail
        assert len(tm.conversation.elements) == 4
        # The client keeps the opaque form it can't decode; it never needs to.
        first_turn = tm.conversation.elements[1]
        assert isinstance(first_turn, ullm.conv.ResponseTurn)
        assert isinstance(first_turn.elements[-1]._model_contents, ullm.backends.proxy._OpaqueModelContents)
    finally:
        await client.close()
        await handle.close()


async def test_a_host_that_forbids_replay_fails_the_call_on_link_loss(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Backend.replay_on_link_loss=False`` on the HOST reaches a default
    client through HELLO: the lost call fails instead of drawing again."""
    proxy = ullm.backends.proxy
    host_backend = ullm.backends.stub.StubBackend.from_script("reply", replay_on_link_loss=False)
    accepted = asyncio.Event()
    release = asyncio.Event()
    original_generate = host_backend._generate
    attempts = 0

    async def generate(params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        nonlocal attempts
        attempts += 1
        async for item in original_generate(params, llm_input):
            accepted.set()
            await release.wait()
            yield item

    monkeypatch.setattr(host_backend, "_generate", generate)
    monkeypatch.setattr(proxy, "_RECONNECT_BASE_DELAY_S", 0.0)
    monkeypatch.setattr(proxy, "_RECONNECT_DELAY_CAP_S", 0.0)
    socket_path = str(tmp_path / "once.sock")
    handle = await proxy.start_backend_server(host_backend, socket_path)
    client = proxy.ProxyBackend(socket_path=socket_path)
    task = asyncio.create_task(_run_prompt(_make_tm(client), "go"))
    try:
        await asyncio.wait_for(accepted.wait(), timeout=5)
        assert client._writer is not None
        client._writer.transport.abort()
        with pytest.raises(ConnectionError, match="replay is disabled"):
            await asyncio.wait_for(task, timeout=5)
        assert attempts == 1
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()
        await handle.close()


@pytest.mark.parametrize("host_allows", [False, True])
async def test_the_replay_policy_propagates_through_a_nested_proxy(tmp_path: pathlib.Path, host_allows: bool) -> None:
    """A proxy served on a second socket advertises its own host's answer:
    the solver's per-game clients see the trainer's policy two hops away."""
    proxy = ullm.backends.proxy
    host_backend = ullm.backends.stub.StubBackend.from_script("reply", replay_on_link_loss=host_allows)
    inner_handle = await proxy.start_backend_server(host_backend, str(tmp_path / "host.sock"))
    inner = proxy.ProxyBackend(socket_path=str(tmp_path / "host.sock"))
    outer_handle = await proxy.start_backend_server(inner, str(tmp_path / "outer.sock"))
    client = proxy.ProxyBackend(socket_path=str(tmp_path / "outer.sock"))
    try:
        assert await client.allows_generation_replay() is host_allows
        assert await _run_prompt(_make_tm(client), "go") == "reply"
    finally:
        await client.close()
        await outer_handle.close()
        await inner.close()
        await inner_handle.close()


@dataclass
class _ClosingBackend(ullm.backend.DummyBackend):
    """Records the contexts it is told have ended."""

    closed: list[str] = field(default_factory=list)

    async def close_context(self, prefix: str) -> None:
        self.closed.append(prefix)


async def test_a_client_closes_its_conversation_under_its_run(tmp_path: pathlib.Path) -> None:
    """A game names only its conversation; the host closes it under the
    client's run, the name that conversation's requests carry."""
    host_backend = _ClosingBackend()
    handle = await ullm.backends.proxy.start_backend_server(host_backend, str(tmp_path / "b.sock"))
    client = ullm.backends.proxy.ProxyBackend(socket_path=str(tmp_path / "b.sock"), run_id="run-7")
    try:
        await client.close_conversation("abc")
        assert host_backend.closed == ["run-7/abc"]
    finally:
        await client.close()
        await handle.close()
