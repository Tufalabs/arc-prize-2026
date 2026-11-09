"""Transient provider failures cost a retry, not a game.

A remote provider under load rate-limits, drops connections, and answers
HTTP 200 with an error body carrying no ``choices`` — which the SDK turns
into a validation error, so nothing keyed on status sees a rate limit. Any
of them ends the game that request was serving unless it is retried.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import openai
import openai.types.chat
import pydantic
import pytest
import ullm.backend
import ullm.backends.oai
import ullm.conv


def _request() -> httpx.Request:
    """The SDK's errors carry the request they failed on."""
    return httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")


def _timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=_request())


def _rate_limited(retry_after: str | None = None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=_request())
    return openai.RateLimitError("rate limited", response=response, body=None)


def _no_wait(error: BaseException, attempt: int = 0) -> float:
    """No sleeping in tests; the delay itself is asserted separately."""
    return 0.0


def _backend(**kwargs: Any) -> ullm.backends.oai.OpenAIServer:
    # reasoning_history_field: the default send_back_reasoning is
    # "current_turn", which this backend refuses without a passback vehicle.
    return ullm.backends.oai.OpenAIServer(
        base_url="http://127.0.0.1:1/v1", model="m", reasoning_history_field="reasoning", **kwargs
    )


def _conversation() -> ullm.conv.Conversation:
    conv = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="s"), tools=[]),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="u"))],
        status="intermediate",
    )
    conv.elements.append(ullm.conv.ResponseTurn(elements=[], status="intermediate"))
    return conv


def _completion(text: str) -> Any:
    return openai.types.chat.ChatCompletion.model_validate(
        {
            "id": "c",
            "created": 0,
            "model": "m",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


def _validation_error(message: str) -> pydantic.ValidationError:
    """A stand-in for what the SDK raises on an unusable response body."""

    class _Model(pydantic.BaseModel):
        field: int

    try:
        _Model(field=message)  # type: ignore[arg-type]
    except pydantic.ValidationError as e:
        return e
    raise AssertionError("expected a ValidationError")


class _Attempts:
    """Fails the first ``failures`` attempts, then succeeds."""

    def __init__(self, error: BaseException, failures: int) -> None:
        self.error, self.failures, self.calls = error, failures, 0

    async def __call__(self, sampling_params: Any, llm_input: Any):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        yield _completion("ok")


async def _run(backend: ullm.backends.oai.OpenAIServer) -> str:
    conv = _conversation()
    async for _ in backend.respond(ullm.backend.GenericSamplingParameters(max_tokens=8), conv):
        pass
    return conv.last_turn.last_response.final_content.to_string()


@pytest.mark.parametrize(
    "error",
    [
        _rate_limited(),
        _timeout(),
        _validation_error("no choices in a 200 body"),
    ],
    ids=["rate_limit", "timeout", "validation_from_200_body"],
)
async def test_a_transient_failure_is_retried(error: BaseException, monkeypatch: pytest.MonkeyPatch) -> None:
    """Including the 200-with-error-body shape, which is the one no
    status-keyed retry can catch."""
    backend = _backend(max_transient_retries=3)
    attempts = _Attempts(error, failures=2)
    monkeypatch.setattr(backend, "_generate_once", attempts)
    monkeypatch.setattr(ullm.backends.oai, "_retry_after_seconds", _no_wait)
    assert await _run(backend) == "ok"
    assert attempts.calls == 3, attempts.calls


async def test_a_retry_line_says_what_the_provider_said(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A 429 is the account's concurrency, its requests or tokens a minute,
    or the provider's own overload: only the body and the headers tell
    them apart, so the retry line carries both, and not the cookies."""
    response = httpx.Response(
        429, headers={"retry-after": "3", "x-request-id": "req-7", "set-cookie": "s=1"}, request=_request()
    )
    error = openai.RateLimitError(
        "Error code: 429 - request reached max organization concurrency: 50", response=response, body=None
    )
    backend = _backend(max_transient_retries=1)
    monkeypatch.setattr(backend, "_generate_once", _Attempts(error, failures=1))
    monkeypatch.setattr(ullm.backends.oai, "_retry_after_seconds", _no_wait)
    assert await _run(backend) == "ok"
    line = capsys.readouterr().out
    assert "transient RateLimitError on attempt 1/2, retrying in 0.0s | failed after " in line, line
    assert "max organization concurrency: 50" in line, line
    assert '"x-request-id": "req-7"' in line and '"retry-after": "3"' in line, line
    assert "set-cookie" not in line, line


async def test_a_retry_line_counts_the_requests_still_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One backend serves a whole host, so what it has open at the provider
    when a request is refused is the run's own concurrency, which is half
    of telling our limit from the provider's."""
    backend = _backend(max_transient_retries=1)
    holding, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def _one_open_one_refused(sampling_params: Any, llm_input: Any):
        nonlocal calls
        calls += 1
        if calls == 1:
            holding.set()
            await release.wait()
            yield _completion("held")
        elif calls == 2:
            raise _rate_limited()
        else:
            yield _completion("ok")

    monkeypatch.setattr(backend, "_generate_once", _one_open_one_refused)
    monkeypatch.setattr(ullm.backends.oai, "_retry_after_seconds", _no_wait)
    held = asyncio.create_task(_run(backend))
    await holding.wait()
    assert await _run(backend) == "ok"
    release.set()
    assert await held == "held"
    assert ", 1 other request open" in capsys.readouterr().out
    assert backend._open_requests == 0


async def test_a_connection_retry_line_names_the_transport_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The SDK's connection error says only "Connection error."; what
    actually failed is the httpx error underneath it."""
    try:
        raise openai.APIConnectionError(request=_request()) from httpx.ConnectError("name resolution failed")
    except openai.APIConnectionError as e:
        error = e
    backend = _backend(max_transient_retries=1)
    monkeypatch.setattr(backend, "_generate_once", _Attempts(error, failures=1))
    monkeypatch.setattr(ullm.backends.oai, "_retry_after_seconds", _no_wait)
    assert await _run(backend) == "ok"
    assert "| Connection error. | cause ConnectError: name resolution failed" in capsys.readouterr().out


async def test_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The budget is a cap, not a suggestion — an endpoint that is simply
    down must fail rather than hold a game open forever."""
    backend = _backend(max_transient_retries=2)
    attempts = _Attempts(_timeout(), failures=99)
    monkeypatch.setattr(backend, "_generate_once", attempts)
    monkeypatch.setattr(ullm.backends.oai, "_retry_after_seconds", _no_wait)
    with pytest.raises(openai.APITimeoutError):
        await _run(backend)
    assert attempts.calls == 3, attempts.calls


async def test_nothing_is_retried_once_output_was_yielded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The consumer has already seen a snapshot, so replaying the request
    would duplicate content. Failing is the honest answer."""
    backend = _backend(max_transient_retries=3)
    calls = 0

    async def _partial_then_fail(sampling_params: Any, llm_input: Any):
        nonlocal calls
        calls += 1
        yield _completion("partial")
        raise _timeout()

    monkeypatch.setattr(backend, "_generate_once", _partial_then_fail)
    with pytest.raises(openai.APITimeoutError):
        await _run(backend)
    assert calls == 1, calls


async def test_a_retry_window_outlasts_the_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A provider down for hours is waited out: inside its window a call
    retries however many times that takes, the count no longer binding."""
    backend = _backend(max_transient_retries=1, transient_retry_window_s=7200.0)
    attempts = _Attempts(_rate_limited(), failures=5)
    monkeypatch.setattr(backend, "_generate_once", attempts)
    monkeypatch.setattr(ullm.backends.oai, "_retry_after_seconds", _no_wait)
    assert await _run(backend) == "ok"
    assert attempts.calls == 6, attempts.calls
    assert "on attempt 5 of a 2 h window, retrying in" in capsys.readouterr().out


async def test_a_retry_window_still_ends_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the window has passed the call fails, however few retries it
    took: a window is a bound like the count it replaces."""
    backend = _backend(max_transient_retries=1000, transient_retry_window_s=0.05)
    attempts = _Attempts(_timeout(), failures=10**9)
    monkeypatch.setattr(backend, "_generate_once", attempts)

    def _short_wait(error: BaseException, attempt: int = 0) -> float:
        return 0.01

    monkeypatch.setattr(ullm.backends.oai, "_retry_after_seconds", _short_wait)
    with pytest.raises(openai.APITimeoutError):
        await _run(backend)
    assert 2 <= attempts.calls < 1000, attempts.calls


async def test_zero_disables_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opting out has to be possible: a caller that wants every failure
    surfaced immediately says 0."""
    backend = _backend(max_transient_retries=0)
    attempts = _Attempts(_timeout(), failures=1)
    monkeypatch.setattr(backend, "_generate_once", attempts)
    with pytest.raises(openai.APITimeoutError):
        await _run(backend)
    assert attempts.calls == 1, attempts.calls


async def test_a_bare_stream_read_error_is_retried_even_mid_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """``httpx.ReadError`` from a stream read escapes the SDK's
    ``APIConnectionError`` wrapping and used to kill the game — even after
    snapshots were yielded, where a restart is safe because snapshots are
    cumulative (183/500 runs of the 2026-08-10 muse-glimmer benchmark)."""
    backend = _backend(max_transient_retries=3)
    calls = 0

    async def _partial_then_drop(sampling_params: Any, llm_input: Any):
        nonlocal calls
        calls += 1
        yield _completion("partial")
        if calls == 1:
            raise httpx.ReadError("connection dropped mid-stream")

    monkeypatch.setattr(backend, "_generate_once", _partial_then_drop)
    monkeypatch.setattr(ullm.backends.oai, "_CONNECTION_RETRY_DELAYS_S", (0.0, 0.0))
    assert await _run(backend) == "partial"
    assert calls == 2, calls


async def test_bare_transport_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that is simply gone must still fail the request."""
    backend = _backend(max_transient_retries=3)
    attempts = _Attempts(httpx.ReadError("connection dropped"), failures=99)
    monkeypatch.setattr(backend, "_generate_once", attempts)
    monkeypatch.setattr(ullm.backends.oai, "_CONNECTION_RETRY_DELAYS_S", (0.0, 0.0))
    with pytest.raises(httpx.ReadError):
        await _run(backend)
    assert attempts.calls == 3, attempts.calls


class _RouteAttempts:
    """Fail server-control requests before returning a JSON response."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def request(self, method: str, url: str, json: Any = None) -> httpx.Response:
        self.calls += 1
        request = httpx.Request(method, url, json=json)
        if self.calls <= self.failures:
            raise httpx.RemoteProtocolError("server closed an idle connection", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)


async def test_server_control_route_transport_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace tokenisation must survive one stale pooled connection."""
    backend = _backend()
    attempts = _RouteAttempts(failures=1)
    monkeypatch.setattr(backend, "_server_client", lambda: attempts)
    monkeypatch.setattr(ullm.backends.oai, "_SERVER_ROUTE_RETRY_DELAYS_S", (0.0, 0.0))

    assert await backend._server_post("tokenize", {"messages": []}) == {"ok": True}
    assert attempts.calls == 2


async def test_server_control_get_uses_transport_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-step served-model identity check uses the same retry path."""
    backend = _backend()
    attempts = _RouteAttempts(failures=1)
    monkeypatch.setattr(backend, "_server_client", lambda: attempts)
    monkeypatch.setattr(ullm.backends.oai, "_SERVER_ROUTE_RETRY_DELAYS_S", (0.0, 0.0))

    assert await backend._server_get_json("v1/models") == {"ok": True}
    assert attempts.calls == 2


async def test_server_control_route_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permanently unavailable trace route must still fail loudly."""
    backend = _backend()
    attempts = _RouteAttempts(failures=99)
    monkeypatch.setattr(backend, "_server_client", lambda: attempts)
    monkeypatch.setattr(ullm.backends.oai, "_SERVER_ROUTE_RETRY_DELAYS_S", (0.0, 0.0))

    with pytest.raises(httpx.RemoteProtocolError):
        await backend._server_post("detokenize", {"tokens": [1]})
    assert attempts.calls == 3


def test_retry_after_is_honoured_and_capped() -> None:
    """A provider naming a delay longer than the run has left would
    otherwise spend the budget sleeping."""

    assert ullm.backends.oai._retry_after_seconds(_rate_limited("2")) == 2.0
    assert ullm.backends.oai._retry_after_seconds(_rate_limited("9999")) == ullm.backends.oai._RETRY_BACKOFF_CAP_S
    assert ullm.backends.oai._retry_after_seconds(_rate_limited("soon")) == 1.0
    assert ullm.backends.oai._retry_after_seconds(_timeout()) == 1.0


def test_the_backoff_holds_its_cap_however_many_attempts() -> None:
    """A retry window bounded by time counts as many attempts as fast
    refusals fit into it, past the thousand where ``2**attempt`` no longer
    converts to a float: the wait stays at the cap there, with or without a
    ``Retry-After``, and climbs as before up to it."""
    cap = ullm.backends.oai._RETRY_BACKOFF_CAP_S
    assert [ullm.backends.oai._retry_after_seconds(_timeout(), a) for a in range(6)] == [1, 2, 4, 8, 16, cap]
    for attempt in (1023, 1024, 100_000):
        assert ullm.backends.oai._retry_after_seconds(_timeout(), attempt) == cap
        assert ullm.backends.oai._retry_after_seconds(_rate_limited("2"), attempt) == cap
