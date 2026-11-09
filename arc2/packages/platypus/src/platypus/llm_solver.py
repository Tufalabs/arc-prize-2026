"""``LLMAgentSolver``: ``AgentSolver`` plus backend lifecycle.

The topology is fixed: ``llm_config.backend_factory`` (a picklable spec)
is deep-copied and started host-side in ``start_up``, which TAAF calls
right before ``run_games`` under its startup deadline, or inside
``run_games`` when nothing has; the resulting
backend is **always** wrapped in a Unix-socket server
(``ullm.backends.proxy``), even when it would be picklable itself — the
host-side backend is where run-wide token accounting lives. Every game
gets its own ``ProxyBackend`` client
(in-process and subprocess alike), so per-game accounting is structural.

Prefer factories whose backend stays host-side (e.g.
``EngineFactory(mode="direct")``): this solver's socket is then the only
process boundary. A proxy-mode factory still works but adds a second hop;
an agent may reject that realized topology when it needs request-local
validation at the concrete backend.

Token cross-check: at the end of a clean run, the sum of per-game
tokens reported into TAAF must equal what the host backend consumed. A
mismatch is a hard error when every game finished cleanly; otherwise the
discrepancy is printed (a crashed/died child legitimately loses its
unreported tail).
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import shutil
import sys
import tempfile
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ullm.backend
import ullm.backend_matrix
import ullm.backends.proxy
import ullm.capture
import ullm.factory

import platypus.diagnostics
import platypus.reference
import platypus.solver
import platypus.tracing
import platypus.watchdog
import taaf.game
from platypus.agent import Agent
from platypus.llm import LLMAgent
from platypus.reference import game_run_dir
from platypus.solver import AgentSolver


@dataclass
class KeptBackend:
    """An engine one run starts and later runs share.

    Handed to ``LLMAgentSolver.keep_backend``: empty, the solver starts its
    factory as usual and leaves the engine and factory here at teardown;
    filled, a later solver serves the games from this backend in-process
    instead of starting its own, and closes nothing. The caller closes it.

    The benchmark runs deep copies of the solver, so this copies as itself
    and travels with every copy; it pickles empty, since the handles it holds
    are live and must not cross a pickle (the run's ``solver.pkl``).
    """

    factory: ullm.factory.BackendFactory | None = None
    backend: ullm.backend.Backend | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> KeptBackend:
        """One holder across every copy of the solver."""
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Live handles stay in this process; a pickle carries an empty holder."""
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """The empty holder a pickle carries."""
        self.factory = None
        self.backend = None

    async def close(self) -> None:
        """Release the engine and the backend the runs left open."""
        if self.factory is not None:
            await self.factory.close()
        if self.backend is not None:
            await self.backend.close()


@dataclass
class LLMAgentSolver(AgentSolver):
    """Hosts ``LLMAgent``s: everything ``AgentSolver`` does, plus the
    backend lifecycle around the game fan-out. Holds no configuration —
    reads ``agent.llm_config`` off the template."""

    # One engine across runs: an empty holder receives the engine this run
    # starts; a filled one serves this run in its place (a test that boots
    # one engine for several games, say). The caller closes the holder.
    keep_backend: KeptBackend | None = None
    _host_backend: ullm.backend.Backend | None = field(default=None, init=False, repr=False)
    _server: ullm.backends.proxy.BackendHostHandle | None = field(default=None, init=False, repr=False)
    _factory: ullm.factory.BackendFactory | None = field(default=None, init=False, repr=False)
    # The engine is started, gated and served on its socket: by ``start_up``
    # under TAAF, else by ``_run_agent_games`` itself.
    _backend_up: bool = field(default=False, init=False, repr=False)
    # The host backend's counters as the games began; the reconciliation
    # reads the games' share of a lifetime total.
    _usage_before_games: ullm.backend.TokenUsage | None = field(default=None, init=False, repr=False)
    _socket_dir: str | None = field(default=None, init=False, repr=False)
    _socket_path: str | None = field(default=None, init=False, repr=False)
    _solver_blob: bytes | None = field(default=None, init=False, repr=False)  # capture provenance
    _trace_collector: platypus.tracing.TraceCollector | None = field(default=None, init=False, repr=False)
    _stall_watchdog: platypus.watchdog.EngineStallWatchdog | None = field(default=None, init=False, repr=False)
    # close_context calls in flight (_close_game_context), held until done.
    _context_closes: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]], init=False, repr=False)
    strict_teardown: bool = False
    teardown_errors: list[str] = field(default_factory=list[str], init=False)

    def _setup(self) -> None:
        """Base setup plus the LLM-layer preconditions: an ``LLMAgent``, and a
        trace/replay combination that can actually be honored."""
        super()._setup()
        if not isinstance(self.agent, LLMAgent):
            raise TypeError(f"LLMAgentSolver requires an LLMAgent; got {type(self.agent).__name__}")
        config = self.agent.llm_config
        if config.replay_dir is None and config.backend_factory is None:
            raise ValueError("LLMAgentSolver requires agent.llm_config.backend_factory (or replay_dir)")
        if config.traces and config.replay_dir is not None:
            raise ValueError(
                "llm_config.traces=True is incompatible with replay_dir: a replay backend has no "
                "real token ids, so it can never produce R5-grade traces"
            )
        if config.exact_history and config.replay_dir is not None:
            raise ValueError(
                "llm_config.exact_history=True is incompatible with replay_dir: a replay backend serves "
                "recorded answers, never token ids, so it cannot continue the served history"
            )
        # The Literal annotation is not a runtime check, and a typo here would
        # write full traces while the run believed it wrote ids-only ones.
        if config.trace_content not in ("full", "ids"):
            raise ValueError(f'trace_content={config.trace_content!r} is not a content; expected "full" or "ids"')

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        """Stand up the host backend (behind its socket) around the base
        fan-out, then reconcile token accounting and finalize traces."""
        assert isinstance(self.agent, LLMAgent)
        if self.agent.run_config.process_mode == "subprocess":
            # Kernel-pool caps are machine-wide only if every game child
            # resolves its semaphores against one manager. Must run
            # before the first child spawns — the manager address travels
            # to children via an env var they inherit at spawn. (Aliased
            # import: a bare `import ullm.x` here would shadow the global.)
            from ullm import shared_semaphore as _shared_semaphore

            _shared_semaphore.enable_multiprocess_semaphores()
        await self._run_agent_games(games)

    @property
    def kaggle_bundle_files(self) -> list[str]:
        """The base solver's payload files plus the engine's own: the vLLM
        fork's overlay when the linked engine ships it (``ullm.kaggle``).

        TAAF reads this once, on the launcher, while it assembles a
        deployment — which is why building the overlay here is affordable:
        it hashes the installed engine against the base wheel it was built
        on, fetching that wheel once per base commit. The deploy runs from
        the workspace, which is where the engine's pin is read.
        """
        paths = super().kaggle_bundle_files
        engine = self.agent.run_config.kaggle_engine
        if engine is not None:
            paths.extend(engine.bundle_files(Path.cwd()))
        return paths

    async def _verify_host_backend(self) -> None:
        """Put the started engine through the backend matrix before any game.

        The same rows ullm's own backend matrix test runs, against THIS
        backend rather than one a fixture stood up: two-turn memory, a tool
        call, the per-turn cap, a prefix-cache hit, an image. They live in
        ullm.backend_matrix precisely so a deployed run can call them —
        deployed jobs ship src and not tests.

        The failure this exists for is silent and total. A configuration the
        engine cannot serve kills every game on its first turn, leaving zero
        actions and zero tokens, and the run still exits COMPLETED. A single
        smoke prompt is not enough to catch it either: a model that reasons
        past a small token cap comes back truncated, and ullm treats empty
        content under truncation as legitimate rather than malformed.
        """
        assert isinstance(self.agent, LLMAgent)
        backend = self._host_backend
        assert backend is not None
        # The image row where the agent sends images, since a prompt carrying
        # pictures is what its games will send.
        images = self.agent.needs_images()
        # Only claim the rows this engine can actually serve: the prefix-cache
        # row asserts a nonzero cache hit, which is a failure rather than a
        # skip when caching is off.
        engine_args = getattr(self._factory, "engine_args", None)
        prefix_cache = bool(getattr(engine_args, "enable_prefix_caching", False))
        try:
            results = await ullm.backend_matrix.run_live_matrix(
                backend,
                name=type(backend).__name__,
                images=images,
                prefix_cache=prefix_cache,
            )
        except Exception as exc:
            raise RuntimeError(
                f"llm_config.verify_backend: the host backend "
                f"({type(backend).__name__}) failed the backend matrix. Every game would meet "
                f"the same engine, so the run stops here instead. {exc}"
            ) from exc
        print(f"verify_backend: {results}", flush=True)

    def _child_agent(self, agent: Agent) -> Agent:
        """The agent minus its engine spec, which never crosses into a game
        child (``LLMAgent.without_engine``)."""
        if not isinstance(agent, LLMAgent):
            return agent
        return agent.without_engine()

    async def _start_up(self) -> None:
        """TAAF's start-up step: the engine started, put through its gates and
        served on its socket before the games are played, under TAAF's
        startup deadline; ``_run_agent_games`` then serves the games from it.
        A replay starts no engine."""
        assert isinstance(self.agent, LLMAgent)
        if self.agent.llm_config.replay_dir is None:
            await self._bring_up_backend()

    async def _shut_down(self) -> None:
        """The engine ``_start_up`` brought up, for a run whose games never ran;
        after one that did, ``_run_agent_games`` has torn it down already and
        this finds nothing left to close."""
        await self._teardown_backend()

    async def _run_agent_games(self, games: list[taaf.game.Game]) -> None:
        """The backend lifecycle around the run: start the configured
        factory unless ``start_up`` has, host the games, tear it down."""
        assert isinstance(self.agent, LLMAgent)
        if self.agent.llm_config.replay_dir is not None:
            # Replay: per-game ReplayBackends over recorded streams —
            # no live backend, no socket (it exists to share ONE engine;
            # there is none), no token cross-check (the recorded usage IS
            # what the games report).
            await super()._run_games(games)
            return
        if not self._backend_up:
            await self._bring_up_backend()
        assert self._host_backend is not None
        try:
            # From here the engine owes tokens to whatever the run sends it,
            # games or a replay, so this is where the watchdog starts.
            self._stall_watchdog = await self._start_stall_watchdog()
            # The games' own spend is counted from here: the matrix and the
            # probes the gates ran went through the same backend, and a kept engine
            # arrives with earlier runs on its counters.
            self._usage_before_games = ullm.backend.TokenUsage() + self._host_backend.total_usage
            if await self.agent.run_on_host(self._host_backend, self.job_dir):
                # Nothing was played: the games end as given up, saying why,
                # rather than being left for teardown to call them crashed.
                for game in games:
                    run = game.game_run
                    if run is not None and run.state == "playing":
                        run.solver_note = "not played: the agent took the run over on the host"
                        game.finish_game()
            else:
                await super()._run_games(games)
        finally:
            # In the finally: an aborted run is where accounting is most at
            # risk (every in-flight call loses its tail), and it reaches here
            # only through the exception path. Raises only when nothing is
            # propagating — raising here would mask the abort.
            # Teardown in a finally of its own: a reconciliation that raises
            # would otherwise leave the engine up, and an engine that outlives
            # its run holds the GPU against whatever asks for it next.
            if self._stall_watchdog is not None:
                await self._stall_watchdog.stop_watching()
                self._stall_watchdog = None
            try:
                self._verify_token_accounting(games, hard=sys.exc_info()[0] is None)
            finally:
                await self._teardown_backend()

    async def _bring_up_backend(self) -> None:
        """Start the configured factory, or adopt a kept engine; put it
        through the run's gates; serve it on the socket the games reach it
        by. A gate that fails tears the started engine down before raising."""
        assert isinstance(self.agent, LLMAgent)
        if self.agent.llm_config.capture:
            # The reference files' provenance member: this solver template,
            # pickled BEFORE any live handles exist (picklable-until-setup),
            # with API keys scrubbed — reference files get committed.
            self._solver_blob = platypus.reference.solver_provenance_blob(self)
        # The last moment an engine argument can still change: every caller
        # override is in, and the factory spec has not been copied yet.
        self.agent.prepare_for_backend()
        factory_spec = self.agent.llm_config.backend_factory
        assert factory_spec is not None
        # Start a COPY: the template's own factory field stays an unstarted,
        # picklable spec, so the host's per-game agent copies keep pickling.
        # The child's copy drops the spec instead (_child_agent).
        if self.keep_backend is not None and self.keep_backend.backend is not None:
            # An earlier run's engine, in this process; the agent's own factory
            # spec stays what it is and is never started.
            self._factory = None
            self._host_backend = self.keep_backend.backend
            # The earlier run's per-run wiring comes off: its trace collector
            # is gone, and its history promise was its own. The gates below
            # set both for this run.
            self._host_backend.trace_recorder = None
            self._host_backend.trace_ids_only = False
            self._host_backend.exact_history = False
        else:
            self._factory = copy.deepcopy(factory_spec)
            # Applied to the COPY, so the run's cadence never edits the agent's
            # own spec: the same agent deployed twice must describe the same
            # experiment. None leaves whatever the factory was built with.
            if self.agent.run_config.engine_status_interval_s is not None:
                self._factory.status_interval_s = self.agent.run_config.engine_status_interval_s
            # The run's job directory, for a backend that writes run artifacts.
            if self.job_dir is not None:
                self._factory.job_dir = Path(self.job_dir)
            # Minimal diagnostics (a Kaggle submission) write no park log: the
            # host tier's connector writes one by default.
            if self.minimal_diagnostics:
                self._factory.use_minimal_diagnostics()
            self._host_backend = await self._factory.start()
        try:
            if self.agent.llm_config.verify_backend:
                await self._verify_host_backend()
            # Everything after the factory start sits inside this try: a
            # failed trace gate must tear the backend down too (on GPU
            # that's a whole started engine).
            if self.agent.llm_config.traces:
                # Trace gate: fail loud AT SETUP when the host backend
                # cannot produce R5-grade records — never a silent skip
                # (deliberate divergence from ARC3-Inference).
                capability = await self._host_backend.trace_capability()
                if capability is None:
                    raise RuntimeError(
                        f"llm_config.traces=True but the host backend "
                        f"({type(self._host_backend).__name__}) cannot produce R5-grade traces "
                        "(exact token ids + input-token roundtrip). Trace-capable backends: an "
                        "in-process vLLM engine (EngineFactory mode='direct') or OpenAIServer "
                        "against a vLLM server (streaming=False). Proxy-mode factories and replay "
                        "backends are never trace-capable."
                    )
                content = self.agent.llm_config.trace_content
                if content not in capability.contents:
                    # A backend that ignored the flag would hand the collector
                    # the portable side, and the first step's reconstruction
                    # check would fail the game rather than the setup.
                    raise RuntimeError(
                        f"llm_config.trace_content={content!r} but the host backend "
                        f"({type(self._host_backend).__name__}) produces only {sorted(capability.contents)} "
                        "traces; ids-only traces need the in-process vLLM engine"
                    )
                self._trace_collector = platypus.tracing.TraceCollector(capability, content=content)
                self._host_backend.trace_recorder = self._trace_collector.record
                self._host_backend.trace_ids_only = content == "ids"
            if self.agent.llm_config.exact_history:
                # The serving-surface probe resolves what continuation needs;
                # the trace gate above, when on, has already done that and more.
                if not self.agent.llm_config.traces:
                    await self._host_backend.probe_continuation()
                if not self._host_backend.supports_token_continuation():
                    raise RuntimeError(
                        "llm_config.exact_history=True but the host backend "
                        f"({type(self._host_backend).__name__}) cannot continue the served history by tokens. "
                        "Capable backends: an in-process vLLM engine (EngineFactory mode='direct'), or "
                        "OpenAIServer against a vLLM server with streaming=False (hosted APIs, replay and "
                        "stub backends never can). See 'Token continuation' in ullm's design.md."
                    )
                self._host_backend.exact_history = True
            # Supplied remote engines expose their window only after connecting,
            # even when no optional capability gate needs that connection.
            engine = getattr(self._host_backend, "engine", None)
            if engine is not None and getattr(engine, "model_config", None) is None:
                await self._host_backend.probe_continuation()
            self.agent._validate_host_backend(self._host_backend)
            self._socket_dir = tempfile.mkdtemp(prefix="platypus-llm-")
            self._socket_path = os.path.join(self._socket_dir, "backend.sock")
            self._server = await ullm.backends.proxy.start_backend_server(
                self._host_backend,
                self._socket_path,
                default_inactivity_timeout_s=self.agent.run_config.backend_inactivity_timeout_s,
            )
        except BaseException:
            await self._teardown_backend()
            raise
        self._backend_up = True

    async def _start_stall_watchdog(self) -> platypus.watchdog.EngineStallWatchdog | None:
        """The run's watchdog over the engine, where the run can end itself
        (``stop_run``, stamped by TAAF for the run's duration) and the host
        backend reports engine metrics; a stub or replay backend has no
        engine to watch, and a run outside TAAF nothing to stop."""
        assert isinstance(self.agent, LLMAgent)
        timeout_s = self.agent.run_config.engine_stall_timeout_s
        backend = self._host_backend
        if timeout_s is None or self.stop_run is None or backend is None:
            return None
        if await backend.engine_metrics() is None:
            return None
        # The in-process engine says when it has died; a served one does not,
        # and its metrics going unreadable is what the watchdog sees instead.
        engine = getattr(backend, "engine", None)
        watchdog = platypus.watchdog.EngineStallWatchdog(
            metrics=backend.engine_metrics,
            dead=lambda: bool(getattr(engine, "errored", False)),
            stop=self.stop_run,
            timeout_s=timeout_s,
        )
        watchdog.start()
        return watchdog

    def _on_slot_granted(self, run_id: str | None, freed_by: str | None) -> None:
        """Record the grant on the traces, when the run writes them."""
        if self._trace_collector is not None and run_id is not None:
            self._trace_collector.note_grant(run_id, freed_by)

    def _game_llm_backend(self, game: taaf.game.Game) -> Any | None:
        """This game's own client: a ``ReplayBackend`` over its recorded stream
        in replay mode, else a fresh ``ProxyBackend`` to the host socket,
        stamped with the run_id and wired to its capture sink."""
        assert isinstance(self.agent, LLMAgent)
        config = self.agent.llm_config
        if config.replay_dir is not None:
            game_dir = os.path.join(config.replay_dir, game.game_id or "")
            if not os.path.isdir(game_dir):
                raise FileNotFoundError(
                    f"replay_dir has no recorded stream for game {game.game_id!r} (looked in {game_dir})"
                )
            return ullm.capture.ReplayBackend(directory=game_dir)
        assert self._socket_path is not None
        # The per-client config is the socket plus the solver-stamped run_id
        # the client stamps it into every RUN frame, so host-side
        # attribution is structural. The reasoning-passback default lives
        # with the host backend; per-call values ride sampling_params.
        run = game.game_run
        client = ullm.backends.proxy.ProxyBackend(
            socket_path=self._socket_path,
            run_id=run.run_id if run is not None else None,
            # One client per game, so the floor is this game's own.
            min_request_gap_s=self.agent.run_config.min_request_gap_s,
        )
        if self._trace_collector is not None:
            assert run is not None, "trace wiring runs after start_game"
            self._trace_collector.register_run(run)
        if config.capture:
            if self.job_dir is None:
                raise ValueError("llm_config.capture=True requires the benchmark to have a job_dir")
            assert run is not None, "capture wiring runs after start_game"
            game_dir = game_run_dir(Path(self.job_dir), run.run_id)
            game_dir.mkdir(parents=True, exist_ok=True)
            (game_dir / "info.json").write_text(
                json.dumps({"run_id": run.run_id, "game_id": game.game_id}, indent=1) + "\n"
            )
            client.capture_sink = ullm.capture.CaptureSink(directory=str(game_dir / "capture"))
        assert self._host_backend is not None
        self.configure_game_backend(game, self._host_backend, client)
        return client

    def configure_game_backend(
        self,
        game: taaf.game.Game,
        host_backend: ullm.backend.Backend,
        client: ullm.backends.proxy.ProxyBackend,
    ) -> None:
        """Customize a live game's backend wiring before its first request.

        Runs on the host after trace registration and capture setup. The
        backend and client are borrowed; lifecycle and tracing stay with the
        solver. A consumer may attach request context or configure the client
        without accessing private solver state. Replay does not call this.
        """

    async def _teardown_backend(self) -> None:
        """Release every resource and optionally fail on teardown errors.

        Errors are retained for training callers even when a primary exception
        is already propagating and must remain the raised failure.
        """
        primary_error = sys.exc_info()[1]
        if self._host_backend is not None:
            usage = self._host_backend.total_usage
            cost = self._host_backend.pricing.cost(usage)
            cost_note = f"; cost ${cost:.4f}" if not math.isnan(cost) else ""
            print(f"[LLMAgentSolver] backend usage: {usage.to_string()}{cost_note}", flush=True)
        if self._trace_collector is not None:
            leftover = self._trace_collector.unfinalized_run_ids()
            if leftover:
                # Only crashed/cancelled games legitimately leave steps behind
                # (their _on_game_hosted never finalized a complete history).
                print(
                    f"[LLMAgentSolver] traces never finalized for runs: {leftover} "
                    "(expected only for crashed/cancelled games)",
                    flush=True,
                )
        if self.keep_backend is not None:
            if self.keep_backend.backend is None:
                self.keep_backend.factory = self._factory
                self.keep_backend.backend = self._host_backend
            steps: tuple[Callable[[], Awaitable[None]], ...] = (self._close_server,)
        else:
            steps = (self._close_server, self._close_factory, self._close_backend)
        for step in steps:
            try:
                await step()
            except Exception as exc:
                self.teardown_errors.append(f"{step.__name__}: {type(exc).__name__}: {exc}")
                print(f"[LLMAgentSolver] {step.__name__} failed:", flush=True)
                traceback.print_exc()
        if self._socket_dir is not None:
            shutil.rmtree(self._socket_dir, ignore_errors=True)
        self._server = None
        self._factory = None
        self._host_backend = None
        self._socket_dir = None
        self._socket_path = None
        self._trace_collector = None
        self._usage_before_games = None
        self._backend_up = False
        if self.strict_teardown and self.teardown_errors and primary_error is None:
            raise RuntimeError("backend teardown failed: " + "; ".join(self.teardown_errors))

    async def _close_server(self) -> None:
        """Shut down the in-process server, if one was started."""
        if self._server is not None:
            await self._server.close()

    async def _close_factory(self) -> None:
        """Release the backend factory's resources (engine, weights)."""
        if self._factory is not None:
            await self._factory.close()

    async def _close_backend(self) -> None:
        """Close the host-side backend the proxy served."""
        # Backend.close() is idempotent; PrebuiltBackend leaves the wrapped
        # backend open by default, so close it explicitly here.
        if self._host_backend is not None:
            await self._host_backend.close()

    def _verify_token_accounting(self, games: list[taaf.game.Game], *, hard: bool = True) -> None:
        """The three-way reconciliation: per-game span-stream totals ==
        per-game TAAF ``ActionRecord`` sums (when diagnostics ran), and the
        TAAF sum across games == what the host backend consumed during the
        games, counted from ``_usage_before_games``.

        Hard error on a clean run, print otherwise. ``hard=False`` is the
        aborted-run path, where the run is already unwinding and a mismatch
        is expected (an in-flight call loses its tail) -- it still reports,
        because silence there is what hid the loss in the first place."""
        if self._host_backend is None:
            return  # aborted before the backend stood up
        reported_generated = 0
        reported_uncached = 0
        clean = True
        mismatches: list[str] = []
        for game in games:
            run = game.game_run
            if run is None or run.state in ("not_started", "playing", "crashed"):
                clean = False  # a crashed/unfinished game loses its unreported tail
                continue
            if run.state == "cancelled" or platypus.solver.RUNTIME_LIMIT_REASON in (run.solver_note or ""):
                # A move cancelled under the agent can lose a token tail
                # legitimately: the host backend counts a generation as it
                # completes, while this game's client only counts once it has
                # consumed the reply. A cancellation landing between the two
                # is a real difference, not an accounting bug — so the exact
                # check drops to the printed warning for the run.
                clean = False
            reported_generated += sum(rec.generated_tokens for rec in run.history) + run.final_generated_tokens
            reported_uncached += sum(rec.uncached_input_tokens for rec in run.history) + run.final_uncached_input_tokens
            if self.job_dir is not None:
                span_mismatch = platypus.diagnostics.reconcile_game_tokens(self.job_dir, run)
                if span_mismatch is not None:
                    mismatches.append(span_mismatch)
        usage = self._host_backend.total_usage
        before = self._usage_before_games or ullm.backend.TokenUsage()
        host_generated = usage.completion_tokens - before.completion_tokens
        host_uncached = (usage.prompt_tokens - usage.cached_tokens) - (before.prompt_tokens - before.cached_tokens)
        if (host_generated, host_uncached) != (reported_generated, reported_uncached):
            mismatches.append(
                f"host backend consumed generated={host_generated}, uncached_input={host_uncached}; "
                f"games reported generated={reported_generated}, uncached_input={reported_uncached}"
            )
        if not mismatches:
            return
        message = "token accounting mismatch: " + "; ".join(mismatches)
        if clean and hard:
            raise RuntimeError(f"LLMAgentSolver {message}")
        why = "expected: not all games finished cleanly" if not clean else "run aborted"
        print(f"[LLMAgentSolver] {message} ({why})", flush=True)

    def _close_game_context(self, run_id: str) -> None:
        """Tell the host backend a game's contexts have ended, in the
        background (``ullm.backend.Backend.close_context``)."""
        backend = self._host_backend
        assert backend is not None

        async def close() -> None:
            try:
                await backend.close_context(run_id)
            except Exception as e:
                print(f"[LLMAgentSolver] close_context({run_id}) failed: {e!r}", flush=True)

        try:
            task = asyncio.get_running_loop().create_task(close())
        except RuntimeError:
            return  # no loop: nothing to tell
        self._context_closes.add(task)
        task.add_done_callback(self._context_closes.discard)

    def _on_game_hosted(self, game: taaf.game.Game) -> None:
        """Per-game end-of-hosting work in the parent: register the R5
        trace (trace mode) and package the reference file (capture mode).
        The SOLVER owns both — collectors (test code) only copy and judge
        finished files."""
        super()._on_game_hosted(game)
        assert isinstance(self.agent, LLMAgent)
        if self._host_backend is not None and game.game_run is not None:
            # The game's conversations are over: their cached prefixes go
            # before live games' when the engine needs room.
            self._close_game_context(game.game_run.run_id)
        if self._trace_collector is not None and game.game_run is not None:
            # History is final here (finish_game ran inside hosting), so the
            # snapshot linkage resolves against the complete GameRun. A
            # validation failure raises — a trace-enabled run must not
            # silently drop or mangle a trace.
            folder = self._trace_collector.finalize(game.game_run)
            if folder is not None:
                print(f"[LLMAgentSolver] trace registered: {folder}", flush=True)
        if not self.agent.llm_config.capture or self.job_dir is None:
            return
        run = game.game_run
        if run is None:
            return
        job_dir = Path(self.job_dir)
        if not (game_run_dir(job_dir, run.run_id) / "capture" / "records.jsonl").exists():
            return  # game ended before any LLM call; nothing to package
        assert self._solver_blob is not None
        git_status = job_dir / "git_status.txt"
        platypus.reference.write_game_reference(
            job_dir,
            run,
            self._game_seeds.get(run.run_id),
            self._solver_blob,
            git_status.read_text() if git_status.exists() else None,
        )
