"""Cross-process concurrency primitive: a name-keyed semaphore that works
both as a local ``asyncio.Semaphore`` (default) and as a shared counter
across OS processes (when multi-process mode is enabled).

The core trick: ``SharedSemaphore``'s only pickled state is
``(name, limit)``. The live primitive is resolved lazily on first
acquire. That means the same object is safe to:

* Pickle for worker-handoff: workers spawned via ``_spawn_ctx.Process``
  inherit the manager's authkey/address through multiprocessing's spawn
  machinery; on first acquire they connect to the manager and look up
  the name, receiving the *same* server-side primitive the parent uses.

* Pickle for storage / checkpoints: the dump is a tiny dict; on load in
  any process, the name resolves to a local ``asyncio.Semaphore`` if no
  manager is running. That's fine — storage-loaded results are for
  analysis, no one's going to acquire on them.

Usage::

    sem = SharedSemaphore("kernel_exists", limit=64)
    async with sem:
        ...                 # same shape as asyncio.Semaphore

    # Hold across method boundaries:
    await sem.acquire()
    try:
        ...
    finally:
        sem.release()

To share across processes, call ``enable_multiprocess_semaphores()`` in
the parent *before* the first ``acquire()``. Children spawned via
``multiprocessing`` inherit the connection automatically.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import threading
from multiprocessing.managers import BaseManager, BaseProxy
from typing import Any, Optional

__all__ = [
    "SharedSemaphore",
    "enable_multiprocess_semaphores",
    "disable_multiprocess_semaphores",
    "multiprocess_semaphores_enabled",
]


# ---------------------------------------------------------------------------
# Manager-side registry
# ---------------------------------------------------------------------------


class _NamedSem:
    """Server-side shared semaphore with an explicit held-count tracker.

    The held counter is incremented on successful acquire and decremented
    on release. Lives in the manager subprocess; clients interact via
    ``_NamedSemProxy``. A threading.Lock protects the counter against
    concurrent Manager serve threads.
    """

    def __init__(self, limit: int) -> None:
        self._sem = multiprocessing.BoundedSemaphore(limit)
        self._limit = limit
        self._held = 0
        self._count_lock = threading.Lock()

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Take a permit, counting it. Returns False if ``timeout`` expired."""
        ok = self._sem.acquire(block, timeout)
        if ok:
            with self._count_lock:
                self._held += 1
        return ok

    def release(self) -> None:
        """Give a permit back."""
        # Decrement first so observers never see held > limit.
        with self._count_lock:
            if self._held > 0:
                self._held -= 1
        self._sem.release()

    def held(self) -> int:
        """How many permits are out right now."""
        with self._count_lock:
            return self._held

    def limit(self) -> int:
        """The cap this semaphore was created with."""
        return self._limit


class _NamedSemProxy(BaseProxy):
    """Client-side proxy for ``_NamedSem``. Exposes the same methods
    ``acquire`` / ``release`` / ``held`` / ``limit``."""

    _exposed_ = ("acquire", "release", "held", "limit")

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Take a permit on the manager-side semaphore."""
        return self._callmethod("acquire", (block, timeout))  # type: ignore[return-value]

    def release(self) -> None:
        """Give a permit back to the manager-side semaphore."""
        self._callmethod("release")

    def held(self) -> int:
        """Permits out, as the manager counts them."""
        return self._callmethod("held")  # type: ignore[return-value]

    def limit(self) -> int:
        """The manager-side cap."""
        return self._callmethod("limit")  # type: ignore[return-value]


# Lives in the manager subprocess. Keyed by name; value is a ``_NamedSem``.
_server_registry: dict[str, _NamedSem] = {}
_server_lock = threading.Lock()


def _server_get_named_sem(name: str, limit: int) -> _NamedSem:
    """Runs inside the manager subprocess. Returns the shared semaphore
    for ``name``, creating it on first call. ``limit`` from later callers
    is ignored — the first resolver wins."""
    with _server_lock:
        sem = _server_registry.get(name)
        if sem is None:
            sem = _NamedSem(limit)
            _server_registry[name] = sem
        return sem


class _SemaphoreManager(BaseManager):
    """Custom manager exposing the name-keyed semaphore registry."""

    pass


_SemaphoreManager.register("get_named_sem", _server_get_named_sem, _NamedSemProxy)


# ---------------------------------------------------------------------------
# Mode control: parent starts the manager and advertises its address via
# an env var; spawned children auto-detect and connect.
# ---------------------------------------------------------------------------

# Env var set by ``enable_multiprocess_semaphores()`` so spawn-inherited
# children can find the parent's manager without extra wiring.
_ENV_ADDR = "AAF_SHARED_SEM_ADDR"

# Per-process handle to the manager. In the parent this is the server
# the parent started; in children it's a client connection. Set lazily.
_manager: Optional[_SemaphoreManager] = None
_manager_is_server: bool = False
_manager_lock = threading.Lock()


def _encode_addr(addr: Any) -> str:
    """Serialize ``BaseManager.address`` for an env var. Address is
    either a str (Unix socket path) or a (host, port) tuple."""
    if isinstance(addr, str):
        return json.dumps({"kind": "unix", "path": addr})
    host, port = addr
    return json.dumps({"kind": "tcp", "host": host, "port": port})


def _decode_addr(s: str) -> Any:
    """Inverse of :func:`_encode_addr`."""
    d = json.loads(s)
    if d["kind"] == "unix":
        return d["path"]
    return (d["host"], d["port"])


def enable_multiprocess_semaphores() -> None:
    """Start the semaphore manager in this process unless a parent has
    already advertised one via ``AAF_SHARED_SEM_ADDR``. Workers spawned
    after this call (via ``multiprocessing.Process``) auto-detect the
    manager through that env var and connect as clients on first
    ``acquire()``.

    Idempotent. Safe to call in any process — if the env var is already
    set (e.g. we're a spawned child), this is a no-op; the child will
    connect as a client lazily on first resolve.

    NOT called automatically: without it, every process falls back to
    local per-process semaphores and any machine-wide cap is fiction.
    Hosts that spawn workers must call it before the first acquire AND
    before spawning (platypus's LLMAgentSolver does, in subprocess mode).
    """
    global _manager, _manager_is_server
    with _manager_lock:
        if _manager is not None:
            return
        if _ENV_ADDR in os.environ:
            # Parent has already advertised a manager; we'll connect as
            # a client on first resolve. Don't start our own server.
            return
        mgr = _SemaphoreManager()
        mgr.start()
        _manager = mgr
        _manager_is_server = True
        os.environ[_ENV_ADDR] = _encode_addr(mgr.address)


def disable_multiprocess_semaphores() -> None:
    """Shut down the manager. Subsequent ``SharedSemaphore`` resolves
    fall back to local ``asyncio.Semaphore``. Mostly for tests; in a
    normal run the manager dies with the parent process anyway."""
    global _manager, _manager_is_server
    with _manager_lock:
        if _manager is None:
            return
        if _manager_is_server:
            try:
                _manager.shutdown()
            except Exception:
                import traceback

                traceback.print_exc()
            os.environ.pop(_ENV_ADDR, None)
        _manager = None
        _manager_is_server = False


def multiprocess_semaphores_enabled() -> bool:
    """Return True if this process has a manager (server or client)
    or can connect to one advertised via the env var."""
    if _manager is not None:
        return True
    return _ENV_ADDR in os.environ


def _get_manager_for_resolve() -> Optional[_SemaphoreManager]:
    """Return the manager for this process, opening a client connection
    on first call if the env var is set and we haven't connected yet.
    Returns ``None`` if multi-process mode isn't active here."""
    global _manager
    if _manager is not None:
        return _manager
    addr_s = os.environ.get(_ENV_ADDR)
    if not addr_s:
        return None
    with _manager_lock:
        if _manager is not None:
            return _manager
        # Child auto-connect: authkey is inherited from the parent via
        # multiprocessing.spawn, so `current_process().authkey` matches
        # the server's. BaseManager picks that up automatically.
        mgr = _SemaphoreManager(address=_decode_addr(addr_s))
        mgr.connect()
        _manager = mgr
    return _manager


# ---------------------------------------------------------------------------
# SharedSemaphore
# ---------------------------------------------------------------------------


class SharedSemaphore:
    """Concurrency primitive whose live handle is resolved lazily.

    Pickle footprint: ``(name, limit)`` only. Unpickling in any process
    recreates the bare object; on first acquire the process resolves its
    own live impl (Manager proxy in multi-process mode, else local
    ``asyncio.Semaphore``).

    The bound primitive is cached per-process; ``reset_cached_impl()``
    clears it (mainly useful for tests).
    """

    __slots__ = ("name", "limit", "_impl", "_resolve_lock")

    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = limit
        self._impl: Any = None
        # Protect concurrent first-time resolution within the same
        # process. asyncio.Lock would be cleaner but only works on the
        # running loop — fine for the normal case but brittle if two
        # loops end up touching the same SharedSemaphore. A threading
        # Lock protects us from both.
        self._resolve_lock: Any = threading.Lock()

    # --- pickle ------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        return {"name": self.name, "limit": self.limit}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.name = state["name"]
        self.limit = state["limit"]
        self._impl = None
        self._resolve_lock = threading.Lock()

    # --- deepcopy ---------------------------------------------------------

    def __deepcopy__(self, memo: dict[int, Any]) -> "SharedSemaphore":
        # Deepcopy returns a fresh shell bound to the same name; they'll
        # resolve to the same shared primitive in multi-process mode.
        new = SharedSemaphore.__new__(SharedSemaphore)
        new.name = self.name
        new.limit = self.limit
        new._impl = None
        new._resolve_lock = threading.Lock()
        memo[id(self)] = new
        return new

    # --- resolution -------------------------------------------------------

    def _resolve(self) -> Any:
        """Return the live primitive, creating it on first call. Both
        the manager-proxy and asyncio branches expose ``acquire``,
        ``release``, and ``async with`` (we adapt manager proxies via
        thread-offload below)."""
        if self._impl is not None:
            return self._impl
        with self._resolve_lock:
            if self._impl is not None:
                return self._impl
            mgr = _get_manager_for_resolve()
            if mgr is not None:
                # get_named_sem is registered dynamically via BaseManager.register;
                # static type checkers can't see it.
                proxy = mgr.get_named_sem(self.name, self.limit)  # type: ignore[attr-defined]
                self._impl = _AsyncNamedSemProxy(proxy)
            else:
                self._impl = asyncio.Semaphore(self.limit)
            return self._impl

    def reset_cached_impl(self) -> None:
        """Drop the cached live primitive. Next acquire re-resolves."""
        self._impl = None

    # --- async semaphore API ---------------------------------------------

    async def acquire(self) -> None:
        """Take a permit, resolving the live primitive on first use."""
        await self._resolve().acquire()

    def release(self) -> None:
        """Give a permit back."""
        self._resolve().release()

    async def __aenter__(self) -> "SharedSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()

    # --- introspection ---------------------------------------------------

    def status(self) -> str:
        """One-line status string for monitor/reporter hooks. Resolves
        the live primitive on first call so that monitors can see the
        shared count even in processes that never acquire themselves
        (subprocess-dispatch parents)."""
        impl = self._resolve()
        if isinstance(impl, asyncio.Semaphore):
            used = self.limit - impl._value  # type: ignore[attr-defined]
            return f"{self.name}: {used}/{self.limit} held (local)"
        # Manager-proxy path: held() queries the server-side tracker.
        try:
            held = impl.held()
            return f"{self.name}: {held}/{self.limit} held (shared)"
        except Exception:
            return f"{self.name}: shared (cap {self.limit})"

    def __repr__(self) -> str:
        return f"SharedSemaphore(name={self.name!r}, limit={self.limit})"


# ---------------------------------------------------------------------------
# _AsyncNamedSemProxy: wraps a blocking ``_NamedSemProxy`` in an async API.
# ---------------------------------------------------------------------------

# How long one blocking acquire attempt waits before returning to the event
# loop. Only a *queued* acquirer pays the round-trip — an uncontended one
# returns on its first attempt — and it bounds how long a cancelled caller's
# late-won permit can stay out.
_ACQUIRE_POLL_S = 0.05


class _AsyncNamedSemProxy:
    """Adapter: exposes ``acquire``/``release``/``async with``/``held``
    around a blocking ``_NamedSemProxy``. ``acquire`` is offloaded to a
    thread so the event loop isn't blocked while waiting on the
    cross-process primitive."""

    __slots__ = ("_proxy",)

    def __init__(self, proxy: Any) -> None:
        self._proxy = proxy

    async def acquire(self) -> None:
        """Take a permit, blocking until one is free.

        Waits in short bounded attempts rather than one unbounded call: a
        thread inside a blocking acquire cannot be interrupted, so an
        unbounded wait that gets cancelled abandons a thread that later
        takes a permit with nobody left to release it — a machine-wide
        slot lost for the life of the manager. Cancellation lands between
        attempts instead, and the attempt still in flight when it arrives
        is awaited out so a permit won late is handed straight back.
        """
        while True:
            attempt = asyncio.ensure_future(asyncio.to_thread(self._proxy.acquire, True, _ACQUIRE_POLL_S))
            try:
                acquired = await asyncio.shield(attempt)
            except asyncio.CancelledError:
                attempt.add_done_callback(self._release_if_won)
                raise
            if acquired:
                return

    def _release_if_won(self, attempt: "asyncio.Future[bool]") -> None:
        """Give back a permit an abandoned attempt won after its caller
        was cancelled. Bounded by the poll interval, so the slot is never
        held longer than one attempt past the cancellation."""
        if not attempt.cancelled() and attempt.exception() is None and attempt.result():
            self._proxy.release()

    def release(self) -> None:
        """Give a permit back to the shared counter."""
        self._proxy.release()

    def held(self) -> int:
        """Permits out across every process sharing this name."""
        return self._proxy.held()

    def limit(self) -> int:
        """The shared cap."""
        return self._proxy.limit()

    async def __aenter__(self) -> "_AsyncNamedSemProxy":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()
