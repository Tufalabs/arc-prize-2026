"""Proxy a whole ``Backend`` over a Unix domain socket.

The host process runs any real backend (a direct-engine ``VLLM``, an
``OpenAIServer``, ...) behind ``start_backend_server``; worker processes
use a ``ProxyBackend`` pointing at the socket. The proxy implements the
backend's *private* hooks by RPC and inherits the public wrappers, so all
per-process accounting state — request counter/log, ``warning_log``,
``total_usage``, ``diagnostics`` — is the proxy's own; the host backend
keeps its own totals across all clients.

Division of labour per request: the client ships its (user-format)
conversation + sampling params; the host runs the wrapped backend's full
pipeline (``prepare_input`` → ``generate`` → ``process_output``) on a
copy. Streamed partials and the final result cross the wire as
**user-format ``LLMResponse`` objects**, so the client needs none of the
host backend's libraries (no vllm/transformers import client-side).
``_model_contents`` payloads are backend-native objects the client may
not be able to unpickle — they cross the wire wrapped as opaque bytes
(`_OpaqueModelContents`) and are unwrapped host-side on the next request,
which keeps reasoning round-tripping working.

A per-request inactivity watchdog (``default_inactivity_timeout_s``)
turns a backend that stops producing into a client-visible error instead
of an eternal await.

Wire protocol: length-prefixed pickle frames carrying ``(tag, dict)``;
one connection multiplexes concurrent requests by client-assigned
``req_id``. Cumulative streaming snapshots are throttled host-side
(``stream_interval``, default 3 s) with the final snapshot always sent.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import hashlib
import itertools
import os
import pickle
import random
import struct
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import ullm.backend
import ullm.conv
import ullm.tool

DEFAULT_INACTIVITY_TIMEOUT_S = 3600.0

# Reconnect schedule for a lost host connection: 2, 4, 8, 16, 32, 60s
# (+jitter) — ~2 minutes of patience against a fixed per-game wall clock.
_RECONNECT_ATTEMPTS = 6
_RECONNECT_BASE_DELAY_S = 2.0
_RECONNECT_DELAY_CAP_S = 60.0

# Connections the host's socket queues while its loop is elsewhere. Every
# game connects once at its first call, and a run's games all start within
# a minute or two while the host is busy starting them; asyncio's default
# of 100 is below any concurrency we run. The kernel silently caps this at
# net.core.somaxconn, so it is a ceiling to ask for rather than a promise.
_HOST_LISTEN_BACKLOG = 4096


class _Tag:
    """Wire frame types, with the payload each carries. Every frame is
    (tag, payload) and every request-bearing one is keyed by ``req_id``, so one
    connection multiplexes concurrent calls."""

    HELLO = "HELLO"  # host -> client: {"client_id": str, "replay_on_link_loss": bool}
    RUN = "RUN"  # client -> host: {"req_id", "conversation", "sampling_params", "stream_interval"}
    ABORT = "ABORT"  # client -> host: {"req_id"}
    COUNT = "COUNT"  # client -> host: {"req_id", "conversation_pickle", "sampling_params"}
    TEXT_COUNT = "TEXT_COUNT"  # client -> host: {"req_id", "text"}; answered with COUNTED/ERROR
    COUNTED = "COUNTED"  # host -> client: {"req_id", "count": int | None}
    CLOSE = "CLOSE"  # client -> host: {"req_id", "conversation_id", "run_id"}; answered with CLOSED/ERROR
    CLOSED = "CLOSED"  # host -> client: {"req_id"}
    PARTIAL = "PARTIAL"  # host -> client: {"req_id", "element": LLMResponse (partial)}
    RESULT = "RESULT"  # host -> client: {"req_id", "element": LLMResponse (processed, opaque model contents)}
    ERROR = "ERROR"  # host -> client: {"req_id", "exception": BaseException}


def _picklable_error(e: BaseException) -> BaseException:
    """``e`` if it survives a pickle ROUND TRIP, else a stand-in with its text.

    Full round trip, not just dumps: openai's APIStatusError subclasses pickle
    fine and fail to UNpickle (their default __reduce__ omits required
    keyword-only arguments), and a payload that explodes client-side raises
    FrameDecodeError in the read loop — with the req_id inside the undecodable
    pickle, so the fallback error broadcasts to every pending request. Every
    ERROR frame goes through here, RUN and COUNT alike.
    """
    try:
        pickle.loads(pickle.dumps(e))
        return e
    except Exception:
        detail = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return RuntimeError(f"{type(e).__name__}: {e}\n{detail}")


class _HostLinkLost(ConnectionError):
    """The proxy's own signal that the host LINK failed (socket missing,
    refused, or closed mid-read). Private and only ever raised by the proxy
    machinery itself: exceptions relayed FROM the host's backend pass through
    unwrapped, so a backend's own ConnectionError or FileNotFoundError — a
    missing model file, say — is a real error, not a reason to reconnect.
    """


class _HostUnreachable(_HostLinkLost):
    """The connect-and-HELLO schedule ran out without a host answering.

    Raised only by ``_ensure_connected``, which has already spent the whole
    reconnect schedule; a caller that retries link loss must treat it as
    final rather than start the schedule over."""


class FrameDecodeError(Exception):
    """A frame's payload bytes were fully read (the stream is still framed)
    but did not unpickle. The known way to hit this: an exception class whose
    default ``__reduce__`` omits required keyword-only arguments — every
    ``openai.APIStatusError`` subclass pickles fine and then fails to
    UNpickle, which job 1860916 turned into 35 fake "connection closed"
    crashes hiding a real 402. The connection is intact; only this frame is
    undecodable."""


async def _read_frame(reader: asyncio.StreamReader) -> Optional[tuple[str, dict[str, Any]]]:
    """Read one length-prefixed pickle frame. Returns None on clean EOF."""
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError as e:
        if not e.partial:
            return None
        raise
    (size,) = struct.unpack("!I", header)
    payload = await reader.readexactly(size)
    try:
        tag, body = pickle.loads(payload)
    except Exception as e:
        raise FrameDecodeError(f"{type(e).__name__}: {e}") from e
    return tag, body


async def _write_frame(writer: asyncio.StreamWriter, lock: asyncio.Lock, tag: str, body: dict[str, Any]) -> None:
    """Write one length-prefixed pickle frame, serialized against `lock`."""
    payload = pickle.dumps((tag, body))
    async with lock:
        writer.write(struct.pack("!I", len(payload)) + payload)
        await writer.drain()


class _OpaqueModelContents:
    """Pickled-bytes wrapper for a backend-native ``_model_contents`` payload.
    The client stores it verbatim (it may not have the libraries to unpickle
    the inner object); the host unwraps it when the conversation comes back."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload


def _wrap_model_contents(element: ullm.conv.LLMResponse) -> None:
    """Replace a live ``_model_contents`` with its opaque form (in place)."""
    mc = element._model_contents
    if mc is not None and not isinstance(mc, _OpaqueModelContents):
        element._model_contents = _OpaqueModelContents(pickle.dumps(mc))


def _partial_has_content(partial: Any) -> bool:
    """Whether a streamed user-format partial carries any text yet."""
    try:
        return bool(partial.reasoning_content.to_string() or partial.final_content.to_string())
    except Exception:
        return True  # unknown shape: better to relay than to drop


#: Byte budget for the host's count-hydration cache, measured in PAYLOAD
#: bytes (the hydrated object graph is a small multiple of its pickle — a
#: near-budget duck conversation hydrates to ~200 MB — so this bounds
#: retained memory at a few times the figure). The budget can be tiny
#: because a hit arrives milliseconds after its insert: the duck's repeat
#: count is the next frame from the same client, so an entry only has to
#: survive the handful of other games' counts that interleave into that
#: gap, not a whole move. A missed hit costs one hydration — what every
#: count paid before the cache existed — never correctness.
_COUNT_CACHE_MAX_PAYLOAD_BYTES = 256 * 1024**2


def _unwrap_conversation(conversation: ullm.conv.Conversation) -> None:
    """Host side: restore opaque ``_model_contents`` back to live objects."""
    for el in conversation.elements:
        if isinstance(el, ullm.conv.ResponseTurn):
            for sub in el.elements:
                if isinstance(sub, ullm.conv.LLMResponse) and isinstance(sub._model_contents, _OpaqueModelContents):
                    sub._model_contents = pickle.loads(sub._model_contents.payload)


# ---------------------------------------------------------------------------
# Client side
# ---------------------------------------------------------------------------


@dataclass
class ProxyBackend(ullm.backend.Backend):
    """Backend whose pipeline runs on a remote host over a Unix socket.

    ``socket_path``, ``stream_interval`` and ``replay_on_link_loss`` are configuration; the
    model, parsers, sampling behaviour, and the reasoning-passback DEFAULT
    all live with the wrapped backend on the host (a per-call
    ``sampling_params.send_back_reasoning`` rides the wire and is resolved
    there). Construction is side-effect-free and the instance pickles like
    any backend (connection state is transient, accounting restarts at
    zero)."""

    socket_path: str = field(kw_only=True)
    stream_interval: float = 3.0  # host-side throttle for streamed partials
    # An explicit opt-out may tighten the host's advertised policy, never
    # loosen it. The wrapped backend owns the host policy.
    _host_replay_on_link_loss: bool = field(default=True, init=False, repr=False)
    # Solver-stamped run identity (platypus U5/I10): when set, every RUN
    # frame carries it and the host passes it to the real backend's
    # respond(run_id=...). The solver constructs each game's client with
    # that game's run_id, so attribution is structural — per call, on the
    # wire, with no turn-manager plumbing.
    run_id: Optional[str] = None
    # This client's _prepare_input / _process_output only swap tool stubs
    # and (de)serialize; the real rendering and parsing run on the HOST,
    # inside this client's "waiting for LLM response" — so the labels say
    # serialization, and nothing more. Propagating the host's own phases
    # over the protocol is roadmapped (ullm roadmap).
    _prepare_status_label = "serializing LLM request"
    _parse_status_label = "deserializing LLM response"
    _reader: Optional[asyncio.StreamReader] = field(default=None, init=False, repr=False)
    _writer: Optional[asyncio.StreamWriter] = field(default=None, init=False, repr=False)
    _write_lock: Optional[asyncio.Lock] = field(default=None, init=False, repr=False)
    _connect_lock: Optional[asyncio.Lock] = field(default=None, init=False, repr=False)
    _reader_task: Optional["asyncio.Task[None]"] = field(default=None, init=False, repr=False)
    _per_req_queues: dict[str, "asyncio.Queue[tuple[str, dict[str, Any]]]"] = field(
        default_factory=lambda: dict[str, "asyncio.Queue[tuple[str, dict[str, Any]]]"](), init=False, repr=False
    )
    _req_counter: Any = field(
        default_factory=itertools.count, init=False, repr=False
    )  # itertools.count (not runtime-subscriptable)

    def _transient_fields(self) -> tuple[str, ...]:
        """The socket and its bookkeeping are per-process; a copy reconnects."""
        return (
            "_reader",
            "_writer",
            "_write_lock",
            "_connect_lock",
            "_reader_task",
            "_per_req_queues",
            "_req_counter",
            "_host_replay_on_link_loss",
        )

    def _rebuild_transient(self) -> None:
        """Drop the connection state so the next call reconnects."""
        self._reader = None
        self._writer = None
        self._write_lock = None
        self._connect_lock = None
        self._reader_task = None
        self._per_req_queues = {}
        self._req_counter = itertools.count()
        self._host_replay_on_link_loss = True

    # --- connection management --------------------------------------------

    async def _connect_once(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, Any]:
        """One connect and HELLO read; the socket is closed on a failed read."""
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        try:
            hello = await _read_frame(reader)
        except OSError:
            # A full listen queue on the host: Linux answers the connect with
            # EAGAIN, asyncio takes that for a connection in progress, and the
            # first recv on the never-established socket fails with EINVAL.
            # The host's accept loop stalls exactly when every game starts at
            # once, so the caller treats this as a wait, not a failure.
            writer.close()
            raise
        return reader, writer, hello

    async def _ensure_connected(self) -> None:
        """Open the Unix socket and complete the HELLO handshake, once.

        Connect failures (missing socket file, refused, reset, a queued
        connect that never established) are the link, not the work: they are
        retried on the reconnect schedule before surfacing as ``_HostLinkLost``.
        """
        if self._writer is not None:
            return
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()
        async with self._connect_lock:
            if self._writer is not None:
                return
            for attempt in range(_RECONNECT_ATTEMPTS + 1):
                try:
                    reader, writer, hello = await self._connect_once()
                    break
                except OSError as e:
                    if attempt == _RECONNECT_ATTEMPTS:
                        raise _HostUnreachable(f"cannot reach backend host: {e}") from e
                await asyncio.sleep(min(_RECONNECT_DELAY_CAP_S, _RECONNECT_BASE_DELAY_S * 2**attempt))
            else:  # pragma: no cover - the loop always breaks or raises
                raise AssertionError("unreachable")
            self._write_lock = asyncio.Lock()
            if hello is None or hello[0] != _Tag.HELLO:
                writer.close()
                raise RuntimeError(f"Backend host did not send HELLO; got {hello!r}")
            replay_allowed = hello[1].get("replay_on_link_loss", True)
            if not isinstance(replay_allowed, bool):
                writer.close()
                raise RuntimeError("Backend host sent an invalid generation replay policy")
            self._host_replay_on_link_loss = replay_allowed
            self._reader = reader
            self._writer = writer
            self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        """Demultiplex frames arriving from the host onto their per-request queues."""
        assert self._reader is not None
        try:
            while True:
                try:
                    frame = await _read_frame(self._reader)
                except FrameDecodeError as e:
                    # The frame is lost but the stream is still framed: wake
                    # the pending requests with the truth and keep reading.
                    # Killing the loop here turns a typed upstream error into
                    # a fake "connection closed" for every in-flight request.
                    err = RuntimeError(f"host sent an undecodable frame: {e}")
                    for q in list(self._per_req_queues.values()):
                        await q.put((_Tag.ERROR, {"req_id": "", "exception": err}))
                    continue
                if frame is None:
                    break
                tag, body = frame
                q = self._per_req_queues.get(body.get("req_id") or "")
                if q is not None:
                    await q.put((tag, body))
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            # Wake pending requests so they don't await forever.
            for q in list(self._per_req_queues.values()):
                await q.put(
                    (
                        _Tag.ERROR,
                        {
                            "req_id": "",
                            "exception": _HostLinkLost("Backend host connection closed"),
                        },
                    )
                )

    async def close(self) -> None:
        """Close the socket connection (never the remote backend). Idempotent."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        self._rebuild_transient()

    async def engine_metrics(self) -> None:
        """None on purpose: the HOST holds the real backend and is the one
        process that should report the engine. A client per game subprocess
        answering here would scrape the same engine once per game."""
        return None

    async def allows_generation_replay(self) -> bool:
        """Resolve and propagate the wrapped host's policy, including nested proxies."""
        if not self.replay_on_link_loss:
            return False
        await self._ensure_connected()
        return self._host_replay_on_link_loss

    def status(self) -> str:
        """One-line liveness for status reporters."""
        connected = self._writer is not None
        return f"ProxyBackend: in_flight={self._in_flight}, connected={connected}"

    # --- Backend hooks (all RPC) -------------------------------------------

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        # send_back_reasoning is NOT shipped: resolution belongs to the host
        # (the wrapped backend's default; a per-call value travels inside
        # sampling_params).
        # Snapshot NOW, not at _generate time: ResponseManager appends the
        # intermediate output element between prepare_input and generate, and
        # the wrapped backend's prepare must see the conversation exactly as
        # a direct backend would (pre-append). The host appends its own
        # intermediate element after its prepare, mirroring the manager.
        #
        # Tools are swapped for inert metadata stubs for the trip: the host
        # only reads name/description/parameters for prompt building, and a
        # live tool may legitimately refuse to pickle (a PythonTool with a
        # warm kernel, a per-move bound view) — and shipping a tool's whole
        # tmpdir snapshot per request would be waste even when it could.
        """Ship the user-format conversation itself — rendering is the host's, which
        is what lets this client run without the host's libraries.
        """
        tools = conversation.system_prompt.tools
        conversation.system_prompt.tools = [
            ullm.tool.Tool(name=t.name, description=t.description, parameters=t.parameters) for t in tools
        ]
        try:
            return {"conversation_pickle": pickle.dumps(conversation)}
        finally:
            conversation.system_prompt.tools = tools

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """Run the call once unless both client and host permit link-loss replay.

        A closed host socket surfaces here as ``_HostLinkLost`` — the read
        loop wakes every pending request with it, and connect/send failures
        raise it directly. Only that sentinel retries: an exception relayed
        from the host's own backend, whatever its type, is the caller's. Job 1831634 lost 18 in-flight games to one such
        event: every affected run crashed on an error the host recovered
        from moments later. So: reconnect and replay, exponential backoff
        (the event that closes the socket is the event that needs time to
        pass), and only while nothing has been yielded — past the first
        partial the consumer holds a snapshot and a replay would duplicate
        content, exactly the ``OpenAIServer`` retry contract.

        One schedule per lost link, not two multiplied: connecting has its
        own retry loop, and a host that never answers exhausts it inside one
        attempt here. That exhaustion is final rather than one more link loss
        to retry, or a lost host would cost this loop's schedule times the
        connect loop's before a game reported it."""
        for attempt in range(_RECONNECT_ATTEMPTS + 1):
            yielded = False
            try:
                async for item in self._generate_once(sampling_params, llm_input):
                    yielded = True
                    yield item
                return
            except _HostLinkLost as e:
                if not self.replay_on_link_loss or not self._host_replay_on_link_loss:
                    raise _HostLinkLost(
                        "Backend host connection lost; generation replay is disabled for this backend"
                    ) from e
                if isinstance(e, _HostUnreachable) or yielded or attempt == _RECONNECT_ATTEMPTS:
                    raise
                delay = min(_RECONNECT_DELAY_CAP_S, _RECONNECT_BASE_DELAY_S * 2**attempt)
                # Jitter so concurrent games don't reconnect in lockstep
                # against a host that just came back.
                delay += random.uniform(0.0, delay / 4.0)
                print(
                    f"[ProxyBackend] host connection lost ({type(e).__name__}: {e}); "
                    f"reconnect {attempt + 1}/{_RECONNECT_ATTEMPTS} in {delay:.1f}s",
                    flush=True,
                )
                await self.close()
                self._rebuild_transient()
                await asyncio.sleep(delay)

    async def _generate_once(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """One attempt: run the call on the host, yielding the partials it
        streams back."""
        await self._ensure_connected()
        # Stable local refs: a concurrent close() nulls the attributes.
        writer, lock = self._writer, self._write_lock
        assert writer is not None and lock is not None
        req_id = f"r{next(self._req_counter)}"
        q: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._per_req_queues[req_id] = q
        try:
            try:
                await _write_frame(
                    writer,
                    lock,
                    _Tag.RUN,
                    {
                        "req_id": req_id,
                        "conversation_pickle": llm_input["conversation_pickle"],
                        "sampling_params": sampling_params,
                        "stream_interval": self.stream_interval,
                        "run_id": self.run_id,
                    },
                )
            except (ConnectionResetError, BrokenPipeError) as e:
                raise _HostLinkLost(f"host link died sending the request: {e}") from e
            while True:
                tag, body = await q.get()
                if tag == _Tag.PARTIAL:
                    yield body["element"]
                elif tag == _Tag.RESULT:
                    yield body["element"]
                    return
                elif tag == _Tag.ERROR:
                    exc = body["exception"]
                    raise exc if isinstance(exc, BaseException) else RuntimeError(str(exc))
                else:
                    raise RuntimeError(f"Unexpected frame tag for request {req_id}: {tag}")
        finally:
            self._per_req_queues.pop(req_id, None)
            # Unwind (cancellation, error, early close): tell the host so it
            # cancels the wrapped backend's generation. Best-effort — the
            # connection may already be closed, in which case the write fails.
            try:
                await _write_frame(writer, lock, _Tag.ABORT, {"req_id": req_id})
            except Exception:
                pass

    def supports_response_injection(self) -> bool:
        """The wrapped backend answers, host-side: the sampling params ship
        whole, and a host that cannot inject refuses the call there — so the
        capability follows the real backend rather than the wrapper."""
        return True

    def supports_after_reasoning(self) -> bool:
        """The wrapped backend answers, host-side, as with injection: a host
        that cannot sample the answer apart refuses the call there."""
        return True

    async def _token_count(
        self, sampling_params: ullm.backend.SamplingParameters, conversation: ullm.conv.Conversation
    ) -> Optional[int]:
        """Ask the host what the shipped conversation would weigh. The wrapped
        backend answers (and returns None if IT cannot tokenize), so the
        capability follows the real backend rather than the wrapper."""
        await self._ensure_connected()
        writer, lock = self._writer, self._write_lock
        assert writer is not None and lock is not None
        req_id = f"c{next(self._req_counter)}"
        q: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._per_req_queues[req_id] = q
        try:
            llm_input = self._prepare_input(sampling_params, conversation.last_turn, conversation, "")
            await _write_frame(
                writer,
                lock,
                _Tag.COUNT,
                {
                    "req_id": req_id,
                    "conversation_pickle": llm_input["conversation_pickle"],
                    "sampling_params": sampling_params,
                },
            )
            tag, body = await q.get()
            if tag == _Tag.COUNTED:
                return body["count"]
            if tag == _Tag.ERROR:
                exc = body["exception"]
                raise exc if isinstance(exc, BaseException) else RuntimeError(str(exc))
            raise RuntimeError(f"Unexpected frame tag for count request {req_id}: {tag}")
        finally:
            self._per_req_queues.pop(req_id, None)

    async def _text_token_count(self, text: str) -> Optional[int]:
        """Ask the host what raw ``text`` weighs. The wrapped backend answers
        (and returns None if IT cannot tokenize), so the capability follows
        the real backend rather than the wrapper -- same as ``_token_count``."""
        await self._ensure_connected()
        writer, lock = self._writer, self._write_lock
        assert writer is not None and lock is not None
        req_id = f"c{next(self._req_counter)}"
        q: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._per_req_queues[req_id] = q
        try:
            await _write_frame(writer, lock, _Tag.TEXT_COUNT, {"req_id": req_id, "text": text})
            tag, body = await q.get()
            if tag == _Tag.COUNTED:
                return body["count"]
            if tag == _Tag.ERROR:
                exc = body["exception"]
                raise exc if isinstance(exc, BaseException) else RuntimeError(str(exc))
            raise RuntimeError(f"Unexpected frame tag for text count request {req_id}: {tag}")
        finally:
            self._per_req_queues.pop(req_id, None)

    async def close_conversation(self, conversation_id: str, run_id: Optional[str] = None) -> None:
        """Close the thread on the host's backend, under this client's run
        (stamped as RUN frames are). Returns once the host's backend has
        closed it, so a request sent after cannot overtake the close."""
        await self._ensure_connected()
        writer, lock = self._writer, self._write_lock
        assert writer is not None and lock is not None
        req_id = f"x{next(self._req_counter)}"
        q: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._per_req_queues[req_id] = q
        try:
            await _write_frame(
                writer,
                lock,
                _Tag.CLOSE,
                {
                    "req_id": req_id,
                    "conversation_id": conversation_id,
                    "run_id": self.run_id if run_id is None else run_id,
                },
            )
            tag, body = await q.get()
            if tag == _Tag.CLOSED:
                return
            if tag == _Tag.ERROR:
                exc = body["exception"]
                raise exc if isinstance(exc, BaseException) else RuntimeError(str(exc))
            raise RuntimeError(f"Unexpected frame tag for close request {req_id}: {tag}")
        finally:
            self._per_req_queues.pop(req_id, None)

    def _copy_element(self, src: ullm.conv.LLMResponse, dst: ullm.conv.LLMResponse) -> None:
        """Copy a shipped (already processed, user-format) response into the
        live conversation element."""
        dst.reasoning_content = src.reasoning_content
        dst.final_content = src.final_content
        dst.token_usage = src.token_usage
        dst.stop_reason = src.stop_reason
        dst.malformed_error = src.malformed_error
        dst.malformed_call_count = src.malformed_call_count
        dst.generation_request_id = src.generation_request_id
        dst.truncation_cap_source = src.truncation_cap_source
        dst._model_contents = src._model_contents  # opaque bytes, kept verbatim
        # The continuation witness must reach the per-game conversation, or
        # every call falls back to a fresh render.
        dst._served = src._served
        dst.response_injection = src.response_injection
        dst.status = "processed"

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        # llm_output is the RESULT element: the host already ran the wrapped
        # backend's full process_output (raising MalformedOutputError host-side
        # if needed — that surfaces from _generate as a normal retryable error).
        """Adopt the processed element the host produced."""
        assert isinstance(llm_output, ullm.conv.LLMResponse), (
            f"ProxyBackend expected a processed LLMResponse, got {type(llm_output).__name__}"
        )
        self._copy_element(llm_output, element)

    def _process_partial_output(
        self, sampling_params: ullm.backend.SamplingParameters, llm_output: Any, intermediate: ullm.conv.LLMResponse
    ) -> None:
        # Partials are parsed host-side into user-format LLMResponses too.
        """Adopt a streamed partial, which the host has already parsed."""
        if isinstance(llm_output, ullm.conv.LLMResponse):
            intermediate.reasoning_content = llm_output.reasoning_content
            intermediate.final_content = llm_output.final_content


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@dataclass
class _HostState:
    """The serving side of the socket: the one real backend, the defaults new
    connections inherit, and the in-flight request tasks (keyed by
    ``client_id:req_id``) that an ABORT frame or a shutdown cancels."""

    backend: ullm.backend.Backend
    default_stream_interval: float
    # Watchdog: if the wrapped backend produces nothing for this many seconds,
    # abort the request and send an ERROR so the client surfaces a failure
    # instead of awaiting forever. Exists because a silently dropped vLLM
    # request (vllm-project/vllm#31404) otherwise hangs a whole run. ``None``
    # disables. Set high enough not to trip on legitimate long prefills.
    default_inactivity_timeout_s: Optional[float] = None
    next_client_id: int = 0
    tasks: dict[str, "asyncio.Task[None]"] = field(default_factory=lambda: dict[str, "asyncio.Task[None]"]())
    # Hydrating a shipped conversation (pickle.loads plus the per-response
    # _unwrap_conversation) is O(context) of GIL-bound work on the event
    # loop, and the duck counts a byte-identical conversation more than once
    # per move (the trim's fired-check and the reconciliation count). Counting
    # never mutates the conversation, so byte-equal COUNT payloads share one
    # hydrated object, keyed by payload digest and evicted LRU down to
    # _COUNT_CACHE_MAX_PAYLOAD_BYTES. Requests must not use this cache:
    # respond() appends to the conversation it is handed.
    count_conv_cache: "collections.OrderedDict[bytes, tuple[int, ullm.conv.Conversation]]" = field(
        default_factory=lambda: collections.OrderedDict[bytes, "tuple[int, ullm.conv.Conversation]"]()
    )
    count_cache_payload_bytes: int = 0  # sum of cached entries' payload sizes
    count_cache_hits: int = 0
    count_cache_misses: int = 0


async def _serve_connection(state: _HostState, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Serve one client for the life of its connection: HELLO, then a frame
    loop dispatching its requests onto the wrapped backend.
    """
    client_id = str(state.next_client_id)
    state.next_client_id += 1
    write_lock = asyncio.Lock()
    try:
        await _write_frame(
            writer,
            write_lock,
            _Tag.HELLO,
            {"client_id": client_id, "replay_on_link_loss": await state.backend.allows_generation_replay()},
        )
        while True:
            frame = await _read_frame(reader)
            if frame is None:
                break
            tag, body = frame
            ns_id = f"{client_id}:{body.get('req_id')}"
            if tag == _Tag.RUN:
                task = asyncio.create_task(_run_request(state, writer, write_lock, body))
                state.tasks[ns_id] = task
                task.add_done_callback(lambda _t, _k=ns_id: state.tasks.pop(_k, None))
            elif tag == _Tag.COUNT:
                task = asyncio.create_task(_run_count(state, writer, write_lock, body))
                state.tasks[ns_id] = task
                task.add_done_callback(lambda _t, _k=ns_id: state.tasks.pop(_k, None))
            elif tag == _Tag.TEXT_COUNT:
                task = asyncio.create_task(_run_text_count(state, writer, write_lock, body))
                state.tasks[ns_id] = task
                task.add_done_callback(lambda _t, _k=ns_id: state.tasks.pop(_k, None))
            elif tag == _Tag.CLOSE:
                task = asyncio.create_task(_run_close(state, writer, write_lock, body))
                state.tasks[ns_id] = task
                task.add_done_callback(lambda _t, _k=ns_id: state.tasks.pop(_k, None))
            elif tag == _Tag.ABORT:
                task = state.tasks.get(ns_id)
                if task is not None:
                    task.cancel()
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass  # client vanished: normal, drop the connection
    except asyncio.CancelledError:
        raise
    except BaseException:
        # A corrupt/unreadable frame kills THIS connection only — never the
        # server, and never as an unretrieved task exception.
        print(f"[proxy host] dropping client {client_id} after a protocol error:", flush=True)
        traceback.print_exc()
    finally:
        for ns_id, task in list(state.tasks.items()):
            if ns_id.startswith(f"{client_id}:"):
                task.cancel()
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def _run_count(
    state: _HostState, writer: asyncio.StreamWriter, write_lock: asyncio.Lock, body: dict[str, Any]
) -> None:
    """Answer one token_count for a shipped conversation.

    Hydration is memoized on the payload bytes (see ``count_conv_cache``):
    the duck ships the identical conversation for consecutive counts of one
    move, and counting is read-only, so the second hit skips the O(context)
    unpickle + unwrap entirely."""
    req_id = body["req_id"]
    try:
        payload: bytes = body["conversation_pickle"]
        cache_key = hashlib.sha256(payload).digest()
        cached = state.count_conv_cache.get(cache_key)
        if cached is None:
            state.count_cache_misses += 1
            conversation: ullm.conv.Conversation = pickle.loads(payload)
            _unwrap_conversation(conversation)
            state.count_conv_cache[cache_key] = (len(payload), conversation)
            state.count_cache_payload_bytes += len(payload)
            # Evict oldest-first down to the byte budget; the newest entry
            # always stays, since it is the one whose repeat is imminent.
            while state.count_cache_payload_bytes > _COUNT_CACHE_MAX_PAYLOAD_BYTES and len(state.count_conv_cache) > 1:
                _, (dropped_bytes, _dropped) = state.count_conv_cache.popitem(last=False)
                state.count_cache_payload_bytes -= dropped_bytes
        else:
            state.count_cache_hits += 1
            conversation = cached[1]
            state.count_conv_cache.move_to_end(cache_key)
        count = await state.backend.token_count(body["sampling_params"], conversation)
        await _write_frame(writer, write_lock, _Tag.COUNTED, {"req_id": req_id, "count": count})
    except asyncio.CancelledError:
        raise
    except BaseException as e:
        try:
            await _write_frame(writer, write_lock, _Tag.ERROR, {"req_id": req_id, "exception": _picklable_error(e)})
        except Exception:
            pass


async def _run_text_count(
    state: _HostState, writer: asyncio.StreamWriter, write_lock: asyncio.Lock, body: dict[str, Any]
) -> None:
    """Answer one raw-text count on the wrapped backend. No hydration and
    no cache: the payload is a short string, not a shipped conversation."""
    req_id = body["req_id"]
    try:
        count = await state.backend.text_token_count(body["text"])
        await _write_frame(writer, write_lock, _Tag.COUNTED, {"req_id": req_id, "count": count})
    except asyncio.CancelledError:
        raise
    except BaseException as e:
        try:
            await _write_frame(writer, write_lock, _Tag.ERROR, {"req_id": req_id, "exception": _picklable_error(e)})
        except Exception:
            pass


async def _run_close(
    state: _HostState, writer: asyncio.StreamWriter, write_lock: asyncio.Lock, body: dict[str, Any]
) -> None:
    """Close one conversation's thread on the wrapped backend, then answer."""
    req_id = body["req_id"]
    try:
        await state.backend.close_conversation(body["conversation_id"], run_id=body.get("run_id"))
        await _write_frame(writer, write_lock, _Tag.CLOSED, {"req_id": req_id})
    except asyncio.CancelledError:
        raise
    except BaseException as e:
        try:
            await _write_frame(writer, write_lock, _Tag.ERROR, {"req_id": req_id, "exception": _picklable_error(e)})
        except Exception:
            pass


async def _run_request(
    state: _HostState, writer: asyncio.StreamWriter, write_lock: asyncio.Lock, body: dict[str, Any]
) -> None:
    """Run one full backend pipeline for a shipped conversation and stream
    the user-format results back."""
    req_id = body["req_id"]
    backend = state.backend
    try:
        conversation: ullm.conv.Conversation = pickle.loads(body["conversation_pickle"])
        sampling_params = body["sampling_params"]
        stream_interval = body.get("stream_interval", state.default_stream_interval)
        _unwrap_conversation(conversation)

        # One call runs the wrapped backend's whole pipeline on the shipped
        # copy; partials arrive already parsed into user format.
        timeout_s = state.default_inactivity_timeout_s
        last_sent = 0.0
        agen = backend.respond(sampling_params, conversation, run_id=body.get("run_id"))
        try:
            while True:
                # Watchdog per step: covers time-to-first-partial (prefill),
                # gaps between partials, and the final step (which includes
                # the backend's output processing).
                watchdog = asyncio.timeout(timeout_s)
                try:
                    async with watchdog:
                        partial = await agen.__anext__()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as e:
                    # The backend may enforce a different deadline of its own.
                    if not watchdog.expired():
                        raise
                    raise RuntimeError(
                        f"proxy host: wrapped backend produced no output for "
                        f"{timeout_s:.0f}s; aborting request {req_id}. Usually means the "
                        f"backend silently dropped the request (e.g. "
                        f"vllm-project/vllm#31404); the caller should surface this "
                        f"rather than await forever."
                    ) from e
                now = time.monotonic()
                # Content-free snapshots (the stream's message_start etc.)
                # must not consume the throttle window: a generation that
                # completes within stream_interval of its first event would
                # otherwise never ship a content-bearing partial at all.
                if now - last_sent >= stream_interval and _partial_has_content(partial):
                    await _write_frame(
                        writer,
                        write_lock,
                        _Tag.PARTIAL,
                        {
                            "req_id": req_id,
                            "element": copy.deepcopy(partial),
                        },
                    )
                    last_sent = now
        finally:
            await agen.aclose()

        element = conversation.last_turn.last_response
        result = copy.deepcopy(element)
        _wrap_model_contents(result)
        await _write_frame(
            writer,
            write_lock,
            _Tag.RESULT,
            {
                "req_id": req_id,
                "element": result,
            },
        )
    except asyncio.CancelledError:
        # ABORT path — the client is unwinding its own generator.
        raise
    except BaseException as e:
        payload: Any = _picklable_error(e)
        try:
            await _write_frame(
                writer,
                write_lock,
                _Tag.ERROR,
                {
                    "req_id": req_id,
                    "exception": payload,
                },
            )
        except Exception:
            pass


async def start_backend_server(
    backend: ullm.backend.Backend,
    socket_path: str,
    *,
    default_stream_interval: float = 3.0,
    default_inactivity_timeout_s: Optional[float] = DEFAULT_INACTIVITY_TIMEOUT_S,
) -> "BackendHostHandle":
    """Serve ``backend`` over a Unix socket; returns a handle. The caller
    owns the backend's lifecycle (the handle never closes it).

    ``default_inactivity_timeout_s``: per-request watchdog (see
    ``_HostState``). The transport watchdog is a last-resort bound; ``None``
    disables it. Hosting applications pass their configured bound explicitly."""
    state = _HostState(
        backend=backend,
        default_stream_interval=default_stream_interval,
        default_inactivity_timeout_s=default_inactivity_timeout_s,
    )
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass

    async def _on_connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _serve_connection(state, reader, writer)

    server = await asyncio.start_unix_server(_on_connect, path=socket_path, backlog=_HOST_LISTEN_BACKLOG)
    return BackendHostHandle(server=server, state=state, socket_path=socket_path)


@dataclass
class BackendHostHandle:
    """Handle for a running backend server."""

    server: "asyncio.base_events.Server"
    state: _HostState
    socket_path: str
    _closed: bool = field(default=False, init=False)

    def status(self) -> str:
        """One-line status for the host side, wrapping the backend's own."""
        return f"backend host: {len(self.state.tasks)} in-flight; wrapped {self.state.backend.status()}"

    async def serve_forever(self) -> None:
        """Accept connections until cancelled."""
        await self.server.serve_forever()

    async def close(self) -> None:
        """Stop serving and release resources. Idempotent; never touches the
        wrapped backend."""
        if self._closed:
            return
        self._closed = True
        for task in list(self.state.tasks.values()):
            task.cancel()
        self.server.close()
        try:
            # Bounded: since Python 3.12.1 wait_closed() waits for live client
            # connections; a client that never disconnects would hang forever.
            await asyncio.wait_for(self.server.wait_closed(), timeout=10.0)
        except TimeoutError:
            print(
                "[BackendHostHandle] wait_closed timed out; continuing shutdown "
                "(a proxy client connection is still open)",
                flush=True,
            )
        except Exception:
            pass
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
