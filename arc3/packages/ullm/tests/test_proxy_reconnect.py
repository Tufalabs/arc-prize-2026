"""ProxyBackend reconnects on a lost host connection (job 1831634).

One host-side socket event crashed all 18 in-flight games at once: the read
loop woke every pending request with ``ConnectionError`` and ``_generate``
propagated it straight into the solver. These tests pin the repaired
contract: reconnect and replay while nothing has been yielded, exponential
backoff between attempts, fail loudly once attempts are exhausted or after
the first yield (where a replay would duplicate content).
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

import pytest
from ullm.backends import proxy

pytestmark = pytest.mark.asyncio


class _FlakyHost:
    """Stand-in for ``_generate_once``: fail N times, then serve."""

    def __init__(self, failures: int, payload: str = "ok") -> None:
        self.failures = failures
        self.payload = payload
        self.attempts = 0

    async def __call__(self, sampling_params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise proxy._HostLinkLost("Backend host connection closed")
        yield self.payload


def _backend(monkeypatch: pytest.MonkeyPatch, host: Any) -> proxy.ProxyBackend:
    """A client whose single-attempt path is ``host`` and whose backoff is
    zeroed via the module constants (never ``asyncio.sleep`` — process-wide)."""
    monkeypatch.setattr(proxy, "_RECONNECT_BASE_DELAY_S", 0.0)
    monkeypatch.setattr(proxy, "_RECONNECT_DELAY_CAP_S", 0.0)
    backend = proxy.ProxyBackend(socket_path="/nonexistent/backend.sock")
    monkeypatch.setattr(backend, "_generate_once", host)
    return backend


_NO_PARAMS: Any = None


async def _drain(backend: proxy.ProxyBackend) -> list[Any]:
    """Collect one full generate stream; params/input are unused by the fake host."""
    return [item async for item in backend._generate(_NO_PARAMS, None)]


async def test_a_lost_connection_is_retried_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fix, in one test: what killed 18 games is now one reconnect."""
    host = _FlakyHost(failures=1)
    backend = _backend(monkeypatch, host)
    assert await _drain(backend) == ["ok"]
    assert host.attempts == 2


async def test_exhausted_reconnects_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying must not become hanging: a host that stays gone has to surface."""
    host = _FlakyHost(failures=proxy._RECONNECT_ATTEMPTS + 1)
    backend = _backend(monkeypatch, host)
    with pytest.raises(ConnectionError):
        await _drain(backend)
    assert host.attempts == proxy._RECONNECT_ATTEMPTS + 1


async def test_explicit_no_replay_survives_pickling(monkeypatch: pytest.MonkeyPatch) -> None:
    import pickle

    host = _FlakyHost(failures=1)
    backend = proxy.ProxyBackend(socket_path="/nonexistent/backend.sock", replay_on_link_loss=False)
    restored = pickle.loads(pickle.dumps(backend))
    monkeypatch.setattr(restored, "_generate_once", host)
    with pytest.raises(ConnectionError, match="generation replay is disabled"):
        await _drain(restored)
    assert host.attempts == 1


async def test_no_replay_after_the_first_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the first partial the consumer holds a snapshot: propagate, never
    replay (the ``OpenAIServer`` retry contract)."""

    async def _yields_then_dies(sampling_params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        yield "partial"
        raise ConnectionError("Backend host connection closed")

    backend = _backend(monkeypatch, _yields_then_dies)
    with pytest.raises(ConnectionError):
        await _drain(backend)


async def test_a_host_originated_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception relayed FROM the host's backend surfaces once, untouched.

    The retry loop keys on the proxy's own link sentinel, so a backend that
    raises ConnectionError or FileNotFoundError itself — a missing model
    artifact, say — must not be mistaken for a lost socket and replayed
    against the production schedule (that mistake cost seven attempts and two
    minutes per occurrence).
    """
    host = _FlakyHost(failures=0)

    class _MissingArtifactHost(_FlakyHost):
        async def __call__(self, sampling_params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
            self.attempts += 1
            raise FileNotFoundError("/models/adapter/does-not-exist")
            yield  # pragma: no cover - makes this an async generator

    host = _MissingArtifactHost(failures=0)
    backend = _backend(monkeypatch, host)
    with pytest.raises(FileNotFoundError):
        async for _ in backend._generate(_NO_PARAMS, {"conversation_pickle": b""}):
            pass
    assert host.attempts == 1, "a relayed backend error was retried as a link failure"


class _HelloHost:
    """Stand-in for ``asyncio.open_unix_connection``: the first ``failures``
    connects hand back a socket whose first read fails the way a connect
    queued behind a full listen backlog does (EINVAL); the next one serves
    a HELLO and then closes."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.attempts = 0

    async def __call__(self, path: str) -> tuple[Any, Any]:
        import errno
        import pickle
        import struct

        self.attempts += 1
        writer = _ClosableWriter()
        if self.attempts <= self.failures:
            return _EinvalReader(errno.EINVAL), writer
        reader = asyncio.StreamReader()
        payload = pickle.dumps((proxy._Tag.HELLO, {"client_id": "1", "replay_on_link_loss": True}))
        reader.feed_data(struct.pack("!I", len(payload)) + payload)
        reader.feed_eof()
        return reader, writer


class _EinvalReader:
    def __init__(self, code: int) -> None:
        self.code = code

    async def readexactly(self, n: int) -> bytes:
        raise OSError(self.code, "Invalid argument")


class _ClosableWriter:
    closed = False

    def close(self) -> None:
        self.closed = True


async def test_a_connect_queued_behind_a_full_backlog_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connect queued behind a full listen backlog fails its first read
    with EINVAL, the socket never having been established; the handshake
    retries on the reconnect schedule instead of failing the game."""
    monkeypatch.setattr(proxy, "_RECONNECT_BASE_DELAY_S", 0.0)
    monkeypatch.setattr(proxy, "_RECONNECT_DELAY_CAP_S", 0.0)
    host = _HelloHost(failures=2)
    monkeypatch.setattr(proxy.asyncio, "open_unix_connection", host)
    backend = proxy.ProxyBackend(socket_path="/nonexistent/backend.sock")
    await backend._ensure_connected()
    assert host.attempts == 3
    assert backend._writer is not None
    assert backend._reader_task is not None
    backend._reader_task.cancel()


async def test_a_backlog_that_never_drains_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proxy, "_RECONNECT_BASE_DELAY_S", 0.0)
    monkeypatch.setattr(proxy, "_RECONNECT_DELAY_CAP_S", 0.0)
    host = _HelloHost(failures=proxy._RECONNECT_ATTEMPTS + 5)
    monkeypatch.setattr(proxy.asyncio, "open_unix_connection", host)
    backend = proxy.ProxyBackend(socket_path="/nonexistent/backend.sock")
    with pytest.raises(proxy._HostLinkLost):
        await backend._ensure_connected()
    assert host.attempts == proxy._RECONNECT_ATTEMPTS + 1


async def test_an_unreachable_host_costs_one_connect_schedule_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The generation retry loop and the connect loop each have a schedule; a
    host that never answers must cost one of them, not their product. An
    exhausted connect loop is final, and a generation call does not start it
    over."""
    monkeypatch.setattr(proxy, "_RECONNECT_BASE_DELAY_S", 0.0)
    monkeypatch.setattr(proxy, "_RECONNECT_DELAY_CAP_S", 0.0)
    connects = {"n": 0}

    async def never(path: str) -> tuple[Any, Any]:
        connects["n"] += 1
        raise FileNotFoundError(path)

    monkeypatch.setattr(proxy.asyncio, "open_unix_connection", never)
    backend = proxy.ProxyBackend(socket_path="/nonexistent/backend.sock")
    with pytest.raises(proxy._HostLinkLost):
        async for _ in backend._generate(_NO_PARAMS, {"conversation_pickle": b""}):
            pass
    assert connects["n"] == proxy._RECONNECT_ATTEMPTS + 1
