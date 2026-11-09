"""A host-side error must arrive at the client as an error, not a dead socket.

Job 1860916: OpenRouter answered 402 on every call, the host pickled the
``openai.APIStatusError`` into an ERROR frame (dumps succeeds), and the
client's UNpickle blew up — openai's exception classes take required
keyword-only arguments their default ``__reduce__`` never replays. The
decode error killed the client's read loop, which woke every request with
``ConnectionError("Backend host connection closed")``: 35 games crashed on
a fake connection loss hiding a real billing error.

Pinned here, over a real socket pair: a roundtrippable exception arrives
typed; a non-roundtrippable one arrives as ``RuntimeError`` naming it; and
in both cases the connection survives — the next request on the same client
is served.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from typing import Any, AsyncGenerator, Optional

import pytest
import ullm.backend
import ullm.conv
from ullm.backends import proxy

pytestmark = pytest.mark.asyncio


class _KwOnlyError(Exception):
    """Pickles, then fails to unpickle — the ``APIStatusError`` shape."""

    def __init__(self, *, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


# The proxy forwards sampling params opaquely; these tests never read them.
_NO_PARAMS: Any = None


class _PlainError(Exception):
    """Round-trips cleanly."""


class _RaisingBackend(ullm.backend.Backend):
    """Backend whose generation raises the configured exception."""

    exc: BaseException

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        object.__setattr__(self, "exc", exc)  # the base freezes attribute creation

    async def _generate(self, sampling_params: Any, llm_input: Any) -> AsyncGenerator[Any, None]:
        raise self.exc
        yield  # pragma: no cover — makes this an async generator

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: Any) -> Any:
        return {}

    def _process_output(
        self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any
    ) -> None:
        """Unused: generation raises before any output exists."""

    def _process_partial_output(self, sampling_params: Any, llm_output: Any, intermediate: Any) -> None:
        """Unused: generation raises before any output exists."""

    async def _token_count(self, sampling_params: Any, conversation: Any) -> Optional[int]:
        return 7


def _intermediate_conversation() -> ullm.conv.Conversation:
    conv = ullm.conv.Conversation(status="intermediate")
    conv.elements.append(ullm.conv.ResponseTurn(elements=[], status="intermediate"))
    return conv


async def _one_call(client: proxy.ProxyBackend) -> None:
    gen = client._generate(_NO_PARAMS, {"conversation_pickle": pickle.dumps(_intermediate_conversation())})
    async for _ in gen:
        pass


async def _served_error(exc: BaseException) -> tuple[BaseException, Optional[int]]:
    """Run one failing call against a real host, then a token count on the
    SAME client; return (what the call raised, what the count returned)."""
    with tempfile.TemporaryDirectory() as d:
        sock = os.path.join(d, "backend.sock")
        handle = await proxy.start_backend_server(_RaisingBackend(exc), sock)
        client = proxy.ProxyBackend(socket_path=sock)
        try:
            with pytest.raises(BaseException) as info:
                await _one_call(client)
            count = await client._token_count(_NO_PARAMS, _intermediate_conversation())
            return info.value, count
        finally:
            await client.close()
            await handle.close()


async def test_a_kwonly_exception_arrives_as_a_named_error_not_a_dead_socket() -> None:
    raised, count = await _served_error(_KwOnlyError(detail="quota exhausted"))
    assert isinstance(raised, RuntimeError)
    assert "_KwOnlyError" in str(raised)
    assert not isinstance(raised, ConnectionError)
    assert count == 7  # the connection outlived the error


async def test_a_roundtrippable_exception_arrives_typed() -> None:
    raised, count = await _served_error(_PlainError("boom"))
    assert isinstance(raised, _PlainError)
    assert count == 7


class _CountDiscriminatingBackend(_RaisingBackend):
    """Counts raise for an empty conversation, answer 7 for a non-empty one —
    so two concurrent counts can take different outcomes deterministically."""

    async def _token_count(self, sampling_params: Any, conversation: Any) -> Optional[int]:
        if len(conversation.elements) >= 2:
            raise _KwOnlyError(detail="tokenizer exploded")
        return 7


async def test_a_count_error_reaches_only_its_own_request() -> None:
    """The COUNT error path round-trips like RUN's, and stays isolated.

    A keyword-only exception sent raw does not unpickle; the read loop then
    sees a frame whose req_id is inside the undecodable payload and can only
    broadcast a fallback error to every pending request — one bad count
    poisoning its concurrent neighbours. Through the safe serializer, the
    failing count arrives as a RuntimeError naming the original, and the
    other count still gets its number.
    """
    import asyncio

    with tempfile.TemporaryDirectory() as d:
        sock = os.path.join(d, "backend.sock")
        handle = await proxy.start_backend_server(_CountDiscriminatingBackend(_PlainError("unused")), sock)
        client = proxy.ProxyBackend(socket_path=sock)
        try:
            failing = _intermediate_conversation()  # two turns -> raises host-side
            failing.elements.append(ullm.conv.ResponseTurn(elements=[], status="intermediate"))
            fine = _intermediate_conversation()
            got = await asyncio.gather(
                client._token_count(_NO_PARAMS, failing),
                client._token_count(_NO_PARAMS, fine),
                return_exceptions=True,
            )
        finally:
            await client.close()
            await handle.close()

    assert isinstance(got[0], RuntimeError) and "_KwOnlyError" in str(got[0])
    assert got[1] == 7, f"the healthy count was poisoned: {got[1]!r}"
