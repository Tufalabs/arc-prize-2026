"""Backend factories: picklable descriptions of how to stand up a backend.

A ``BackendFactory`` is the spec form of a backend: its fields describe what
to run, ``start()`` (host process, once) performs any host-side lifecycle
(building an engine, launching a server) and returns a ready ``Backend``,
and ``close()`` tears down whatever ``start()`` built. For factories that
launch host-side resources, the returned backend is picklable and connects
to that resource from any process; backends with no host-side lifecycle
(API clients, stubs) use the trivial ``PrebuiltBackend`` wrapper.

Factories are picklable *until started* — after ``start()`` they hold live
handles and pickling fails loudly. The vLLM-backed factories live in
``ullm.backends.vllm_factory`` (importing them requires the ``vllm``
extra).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import ullm.backend
import ullm.support as uls


@dataclass
class BackendFactory(uls.BaseClass):
    """Base class for backend factories. Subclasses implement ``_start`` and
    ``_close``; use as ``async with factory as backend: ...`` or call
    ``start()`` / ``close()`` explicitly."""

    # Cadence of the periodic engine-status line (running/waiting, KV usage,
    # prefix-hit rate — see ullm.engine_monitor), printed to the stdout of
    # whichever process starts the factory. None disables it. Honoured for
    # every factory whose backend has an engine to report, including one that
    # merely points at a server somebody else started.
    status_interval_s: Optional[float] = field(default=60.0, kw_only=True)
    # The run's job directory as TAAF defines it: the benchmark sets it on
    # the solver, and the solver sets it here before start(). A backend
    # that writes run artifacts (the KV offload connector's trace) puts
    # them in it, beside the platypus diagnostics. None means no run
    # directory, and nothing is written.
    job_dir: Optional[Path] = field(default=None, kw_only=True)
    _started: bool = field(default=False, init=False, repr=False)
    _status_monitor: Any = field(default=None, init=False, repr=False)
    #: Whether the backend this factory builds can continue, by tokens, a
    #: served history that carries an image: what exact history needs of an
    #: agent that sends one. A factory whose backend cannot says so here, so
    #: the agent refuses the pairing before the first request rather than on it.
    continues_images: ClassVar[bool] = True

    def use_minimal_diagnostics(self) -> None:
        """Write only what a run needs, as TAAF's minimal diagnostics (a
        Kaggle submission) ask: a factory with optional run artifacts turns
        them off. The solver calls it on its own copy, before ``start()``."""

    @property
    def started(self) -> bool:
        """Whether backend setup completed without a subsequent successful close.

        This exposes lifecycle state, not a concurrency guard or a readiness
        check for the status monitor.
        """
        return self._started

    async def start(self) -> ullm.backend.Backend:
        """Perform host-side setup and return the ready backend. Once only."""
        assert not self._started, f"{type(self).__name__} already started"
        backend = await self._start()
        self._started = True
        await self._start_status_monitor(self._monitored_backend(backend))
        return backend

    async def close(self) -> None:
        """Tear down what ``start()`` built. Safe to call once after start."""
        if not self._started:
            return
        if self._status_monitor is not None:
            await self._status_monitor.stop()
            self._status_monitor = None
        await self._close()
        self._started = False

    def _monitored_backend(self, backend: ullm.backend.Backend) -> ullm.backend.Backend:
        """Which backend the status monitor reads. The returned one by
        default; a factory that hands out a CLIENT while holding the real
        backend host-side overrides this to name the real one."""
        return backend

    async def _start_status_monitor(self, backend: ullm.backend.Backend) -> None:
        """Start the engine-status line when there is an engine to report.

        Owned here rather than per factory: what can be reported is a
        property of the BACKEND (``engine_metrics``), not of how it was
        built, and the lifetime is the factory's either way. The probe keeps
        a backend with no engine silent rather than reporting nothing every
        interval.
        """
        if self.status_interval_s is None:
            return
        import ullm.engine_monitor as engine_monitor

        if await backend.engine_metrics() is None:
            return

        async def source() -> engine_monitor.EngineMetrics:
            metrics = await backend.engine_metrics()
            if metrics is None:
                raise engine_monitor.MetricsUnavailable(f"{type(backend).__name__} stopped reporting metrics")
            return metrics

        self._status_monitor = engine_monitor.start_monitor(source, interval_s=self.status_interval_s)

    async def _start(self) -> ullm.backend.Backend:
        """Build and return the backend. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement _start()")

    async def _close(self) -> None:
        """Release whatever ``_start`` acquired; nothing by default."""
        pass

    async def __aenter__(self) -> ullm.backend.Backend:
        return await self.start()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    def __getstate__(self) -> dict[str, Any]:
        # The factory is the picklable spec; once started it holds live
        # host-side handles that must not silently cross a process boundary.
        assert not self._started, f"Cannot pickle a started {type(self).__name__}; pickle the factory before start()"
        return dict(self.__dict__)


@dataclass
class PrebuiltBackend(BackendFactory):
    """The trivial factory: wraps an already-constructed backend that needs
    no host-side lifecycle (API clients, stubs, an OpenAIServer pointing at
    an externally managed URL). Picklable iff the wrapped backend is."""

    backend: ullm.backend.Backend = field(kw_only=True)
    close_backend: bool = False  # if True, close() also closes the wrapped backend

    async def _start(self) -> ullm.backend.Backend:
        """Hand back the backend this factory was given."""
        return self.backend

    async def _close(self) -> None:
        """Close the wrapped backend, if this factory was told to own it."""
        if self.close_backend:
            await self.backend.close()
