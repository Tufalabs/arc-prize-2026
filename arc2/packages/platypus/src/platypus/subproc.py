"""Subprocess-per-game hosting.

The taaf ``Game`` (GameRun bookkeeping, scoring) stays in the parent; the
agent runs in a spawned child. Traffic runs **child to parent**: the child
gets everything it needs at spawn and the parent answers nothing about the
game, because nothing such an answer could carry is something the agent may
see — an answer's result is its score. The parent speaks back for exactly
two things, neither of them about the game: the re-grant of a concurrency
slot the child offered back (``ctx.yield_slot``, a request the child blocks
on), and the cancellation raised into a child parked in that wait. Anything
going wrong (child death, memory-limit breach, pickle failure) is a loud
error for that game, or the run under ``on_agent_error="raise"``; there is
deliberately no fallback to in-process hosting.

Because nothing about the game comes back, a mis-shaped answer has to fail
**in the child**:
``platypus.solver.Submitter`` validates against the task before sending, so
the agent gets a traceback in its own process. The parent's own check inside
``Game.execute_action`` remains as a backstop, and a failure there is an
infra bug reported as a dead game rather than a wrong answer.

Hardening choices follow aaf's ``solver_subprocess`` (each earned):

- **spawn** context: a fresh interpreter per child — no inherited CUDA
  contexts, threads, or half-initialized module state.
- **liveness watchdog**: the parent's wait on the child pipe times out
  every few seconds and checks the process, because a SIGKILLed/OOMed
  child may never deliver pipe EOF — without this the parent would hang
  forever while holding a concurrency slot.
- **cancellation relay**: parent sends SIGTERM, which cancels the child's
  hosting task — cancel at the next await, shielded ``on_finish`` — exactly
  like in-process cancellation. Grace period, then SIGKILL.
- **stderr capture**: the child's stderr goes to a file, line-buffered;
  on abnormal death the parent surfaces the tail in the error — the only
  way to see why an OOM/segfault child died.
- **stdout relay**: the child's ``sys.stdout`` sends its lines to the
  parent over the child's pipe, and the parent prints them as its own;
  descriptor 1, which native code and the child's own subprocesses write
  to, goes to the stderr file. The child never writes to the descriptor
  it inherited: in a notebook that is a pipe one ipykernel thread drains,
  and once that thread stops, every process writing into the pipe blocks
  when it fills (2026-10-29: a Kaggle run froze at 3 h 53 m).
- **RLIMIT_AS** in the child enforces
  ``run_config.subprocess_memory_limit_bytes`` pre-allocation; a breach
  is a MemoryError on the agent-error path, and a failure to *apply* the
  limit is itself fatal (no silent uncapped mode).
- **no engine in the child**: a spawned child re-runs the agent module's
  imports in its own heap, so what that module imports is a per-game
  cost. ``_child_main`` asserts vllm is not among them — the engine runs
  in the host and the child talks to it through a proxy client.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import multiprocessing
import multiprocessing.connection
import multiprocessing.process
import multiprocessing.resource_tracker
import os
import pickle
import random
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import taaf.game
from platypus.context import MoveContext, SubmissionView
from platypus.contract import GameOutcome, RunConfig
from platypus.slots import SlotView
from platypus.status import NULL_STATUS, StatusHandle

if TYPE_CHECKING:
    from platypus.agent import Agent

_SPAWN = multiprocessing.get_context("spawn")

# child -> parent. Every message is a push the parent never answers, except
# the one request: a yield is answered when the host re-grants the slot.
_SUBMIT = "submit"  # (submission, generated, uncached) — one answer
_YIELD = "yield"  # request-reply: answered when the host re-grants the slot
_GAVE_UP = "gave_up"  # reason string
_AGENT_ERROR = "agent_error"  # traceback text
_TOKENS = "tokens"  # (generated, uncached) spent after the last answer
_DONE = "done"  # terminal kind: "ok" / "cancelled" / "agent_error"
_STATUS = "status"  # one-way push (platypus.status)
_OUTPUT = "output"  # one-way push: the child's stdout
# parent -> child: the yield's reply, or an exception raised into it.
_STATE = "state"
_RAISE = "raise"

_LIVENESS_TIMEOUT_S = 5.0  # re-check the child process while waiting on its pipe
_CANCEL_GRACE_S = 300.0  # cooperative-unwind window after SIGTERM (generous: LLM flushes later)
_KILL_AFTER_S = 10.0  # wait after pipe close before SIGKILL in the reaper
_STDERR_TAIL_BYTES = 8192


class SubprocessDiedError(RuntimeError):
    """The per-game child vanished without completing its protocol."""


@dataclass(frozen=True)
class _ChildPayload:
    """Everything a game child needs to start playing, pickled to it once at
    spawn. Nothing live crosses: the backend is a client spec, diagnostics are
    paths the child opens itself."""

    agent_blob: bytes  # pickled separately so agent pickle errors surface in the parent
    game_id: str
    pass_index: int  # TAAF's stamp on the GameRun, so the child's context carries it
    view: taaf.game.GameView  # the task, answers already withheld
    rng_seed: int  # resolved by AgentSolver._game_seed; always concrete
    memory_limit_bytes: int
    max_runtime_s: float | None
    # LLM layer: this game's backend client (a picklable ProxyBackend that
    # connects to the host-side socket from the child), or None for
    # LLM-free agents. Typed Any: this module stays ullm-free.
    llm_backend: Any = None
    # Diagnostics: (stream path, block-store dir), or None when off.
    # The CHILD constructs the SpanLogger and writes the stream directly —
    # bulk never rides the parent control pipe.
    diag_paths: tuple[str, str] | None = None
    # Status reporting (platypus.status): whether this child publishes what
    # it is doing. Unlike diagnostics the labels DO ride the control pipe —
    # they are a handful of bytes each and the parent is the only consumer.
    status_enabled: bool = False
    # The run's job dir, so the child can read the queue tally
    # (``platypus.slots``). A path, not a number: nothing live crosses the
    # payload, and the child reads it when it asks rather than being told.
    job_dir: str | None = None


def _tail(path: Path, max_bytes: int = _STDERR_TAIL_BYTES) -> str:
    """Last bytes of the child's stderr — the only account of why an OOM or
    segfault child died."""
    try:
        data = path.read_bytes()
    except OSError:
        return "<no stderr captured>"
    return data[-max_bytes:].decode(errors="replace")


def _print_child_output(text: Any) -> None:
    """A child's relayed stdout, as this process's own: in a notebook,
    ipykernel's in-process stream, not the pipe the child would block on."""
    with contextlib.suppress(Exception):
        print(text, end="", flush=True)


def _signame(exitcode: int | None) -> str:
    """An exit code as the signal or status it represents."""
    if exitcode is None:
        return "still running"
    if exitcode < 0:
        try:
            return f"signal {signal.Signals(-exitcode).name}"
        except ValueError:
            return f"signal {-exitcode}"
    return f"exit code {exitcode}"


# --------------------------------------------------------------------------
# Child side
# --------------------------------------------------------------------------


def _child_entry(
    to_parent: multiprocessing.connection.Connection,
    from_parent: multiprocessing.connection.Connection,
    payload_blob: bytes,
    stderr_path: str,
) -> None:
    """Process entry: capture stderr, relay stdout, apply the memory limit,
    run the agent. A setup failure before that starts (unpicklable payload, a
    memory limit that will not apply) sends nothing — it prints to the
    captured stderr and kills the child, and the parent surfaces it through
    the death report, whose stderr tail is the only account of it."""
    fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(fd, 2)
    sys.stderr = os.fdopen(2, "w", buffering=1)
    # Before anything can print: the payload's unpickling runs imports.
    os.dup2(fd, 1)
    sender = _ToParent(to_parent)
    sys.stdout = _ParentStdout(sender)
    try:
        payload: _ChildPayload = pickle.loads(payload_blob)
        if payload.memory_limit_bytes:
            import resource

            # Failure to apply the cap is fatal — never run uncapped silently.
            resource.setrlimit(resource.RLIMIT_AS, (payload.memory_limit_bytes, payload.memory_limit_bytes))
        asyncio.run(_child_main(sender, from_parent, payload))
    except BaseException:
        traceback.print_exc()  # lands in the captured stderr file
        raise


class _ToParent:
    """The child's end of its pipe to the parent, shared by the protocol and
    the stdout relay: a ``Connection`` is not safe to send on from two
    threads at once, and a print can come from any thread."""

    def __init__(self, connection: multiprocessing.connection.Connection) -> None:
        """Wrap the child's send end."""
        self._connection = connection
        self._lock = threading.Lock()

    def send(self, tag: str, value: Any) -> None:
        """Push one message; a closed pipe is not an error."""
        with self._lock, contextlib.suppress(BrokenPipeError, OSError):
            self._connection.send((tag, value))


class _ParentStdout(io.TextIOBase):
    """The child's ``sys.stdout``: whole lines go to the parent as ``_OUTPUT``
    messages, and a trailing partial line goes on ``flush``."""

    def __init__(self, sender: _ToParent) -> None:
        """Relay through ``sender``."""
        super().__init__()
        self._sender = sender
        self._pending = ""
        self._lock = threading.Lock()

    # The base's is None, which code that reads sys.stdout.encoding trips on;
    # the relay carries str, pickled.
    encoding = "utf-8"

    def writable(self) -> bool:
        """Always: ``print`` checks."""
        return True

    def write(self, text: str) -> int:
        """Send every complete line now; keep the rest for the next write or a flush."""
        with self._lock:
            self._pending += text
            cut = self._pending.rfind("\n") + 1
            lines, self._pending = self._pending[:cut], self._pending[cut:]
        if lines:
            self._sender.send(_OUTPUT, lines)
        return len(text)

    def flush(self) -> None:
        """Send a partial line too."""
        with self._lock:
            text, self._pending = self._pending, ""
        if text:
            self._sender.send(_OUTPUT, text)

    def fileno(self) -> int:
        """Descriptor 1, which ``_child_entry`` pointed at the stderr file."""
        return 1


class _ParentGone(Exception):
    """The parent closed its send pipe — cancellation or parent death."""


class _ChildPort:
    """Synchronous RPC to the parent. Blocking ``recv`` is fine child-side:
    while parked here there is nothing else to run, and parent-initiated
    cancellation unblocks it by closing the pipe (EOF -> ``_ParentGone``)."""

    def __init__(
        self,
        to_parent: _ToParent,
        from_parent: multiprocessing.connection.Connection,
    ):
        """Bind the two one-directional pipes this side talks over."""
        self._to_parent = to_parent
        self._from_parent = from_parent

    def request(self, tag: str, value: Any) -> Any:
        """Send a request and block for the parent's reply, re-raising whatever
        the parent raises. Pipe EOF means the parent is gone."""
        self.send(tag, value)
        try:
            reply_tag, reply_value = self._from_parent.recv()
        except (EOFError, OSError) as exc:
            raise _ParentGone from exc
        if reply_tag == _RAISE:
            raise reply_value
        return reply_value

    def send(self, tag: str, value: Any) -> None:
        """Push a message the parent never answers; a closed pipe is not an error."""
        self._to_parent.send(tag, value)


async def _child_main(
    to_parent: _ToParent,
    from_parent: multiprocessing.connection.Connection,
    payload: _ChildPayload,
) -> None:
    """The child side: rebuild the context, run the shared play call, and
    report what happened over the pipe."""
    # Imported here, not at module top: solver.py imports this module.
    from platypus.solver import AgentSolver, Submitter, run_game

    agent: Agent = pickle.loads(payload.agent_blob)
    # Checked once the agent is in, because unpickling it is what runs its
    # module's imports. vllm (and the torch under it) is the host's engine,
    # never a game's: a child reaches inference through its proxy client, and
    # importing vllm here would cost ~365 MB of private heap PER GAME —
    # ~85 GB across a 240-task submission, for nothing the child calls. It is
    # a fixed cost with no symptom of its own, so it is asserted rather than
    # left to be noticed in a memory report. An agent module that needs vllm
    # host-side imports it inside the function that uses it (duck's
    # ``_installed_vllm``).
    assert "vllm" not in sys.modules, (
        f"a game child imported vllm (agent {type(agent).__name__}): ~365 MB of private heap "
        "per game, for an engine only the host process runs. Move the import inside the "
        "function that needs it."
    )
    port = _ChildPort(to_parent, from_parent)
    submissions: list[taaf.game.Submission] = []
    rng = random.Random(payload.rng_seed)
    diag = AgentSolver._diag_logger(payload.diag_paths)
    # Push-on-change, never answered: the parent applies the label as it
    # services the pipe. A child wedged inside a kernel call or an LLM
    # request stops pushing, and the parent keeps reporting the state it
    # last declared — which is the state it is genuinely stuck in.
    status = StatusHandle(lambda label: port.send(_STATUS, label)) if payload.status_enabled else NULL_STATUS
    slots = SlotView(job_dir=Path(payload.job_dir) if payload.job_dir else None)
    meter: Any = None

    def _tokens() -> tuple[int, int]:
        """Tokens spent since the last report; (0, 0) for LLM-free agents."""
        return meter.take() if meter is not None else (0, 0)

    def raw_submit(submission: taaf.game.Submission) -> None:
        """Push one answer up with what it cost. The parent executes it."""
        port.send(_SUBMIT, (submission, *_tokens()))

    submitter = Submitter(raw_submit, submissions, len(payload.view.questions), diag)
    if payload.llm_backend is None:
        ctx = MoveContext(
            game_id=payload.game_id,
            history=SubmissionView(submissions),
            pass_index=payload.pass_index,
            rng=rng,
            submit=submitter,
            diag=diag,
            status=status,
            slots=slots,
        )
    else:
        import platypus.llm

        ctx, meter = platypus.llm.build_llm_context(
            payload.game_id,
            submissions,
            rng,
            payload.llm_backend,
            submitter,
            diag,
            status,
            slots,
            pass_index=payload.pass_index,
        )

    async def yield_slot(hints: Any) -> None:
        """Offer this game's slot back and block until it is re-granted.

        Blocking the child's loop is the point: a parked game has nothing
        else to run, and holding it here means no request is built, no
        prompt serialized and no conversation kept alive host-side for
        however long the wait lasts. The parent unblocks it by answering,
        or raises cancellation into it.
        """
        port.request(_YIELD, dict(hints))

    ctx.yield_slot = yield_slot

    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    assert main_task is not None
    # SIGTERM from the parent = cancellation; lands at the agent's next
    # await, exactly like in-process cancellation.
    loop.add_signal_handler(signal.SIGTERM, main_task.cancel)

    try:
        try:
            give_up = await run_game(agent, ctx, payload.view, submitter, payload.max_runtime_s)
        finally:
            if payload.llm_backend is not None:
                import platypus.llm

                platypus.llm.record_replay_consumption(payload.llm_backend, diag)
            diag.close()
    except (asyncio.CancelledError, _ParentGone):
        # The parent initiated this and has already finished the game on its
        # side; the trailing token delta rides up for the run's final_*
        # fields. on_finish still runs, shielded.
        port.send(_TOKENS, _tokens())
        await _shielded_on_finish(agent, ctx, GameOutcome(state="cancelled", n_submitted=len(submissions)))
        port.send(_DONE, "cancelled")
        return
    except BaseException as exc:
        formatted = "".join(traceback.format_exception(exc))
        # Report first (parent records the note), then best-effort on_finish.
        port.send(_AGENT_ERROR, formatted)
        port.send(_TOKENS, _tokens())
        await _shielded_on_finish(agent, ctx, GameOutcome(state="crashed", n_submitted=len(submissions)))
        port.send(_DONE, "agent_error")
        return

    if give_up is not None:
        port.send(_GAVE_UP, give_up.reason)
    port.send(_TOKENS, _tokens())
    state = "gave_up" if give_up is not None else "finished"
    await _shielded_on_finish(agent, ctx, GameOutcome(state=state, n_submitted=len(submissions)))
    port.send(_DONE, "ok")


async def _shielded_on_finish(agent: Agent, ctx: MoveContext, outcome: GameOutcome) -> None:
    """Run ``on_finish`` so a cancellation in flight cannot cut it short;
    a failure inside it is logged to the captured stderr, never raised."""
    try:
        await asyncio.shield(agent.on_finish(ctx, outcome))
    except asyncio.CancelledError:
        pass
    except Exception:
        traceback.print_exc()  # captured stderr


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------


class SubprocessGameHost:
    """Hosts one game's child process and applies its messages to the real
    (parent-side) ``taaf.game.Game``.

    ``run()`` returns the child's terminal kind (``"ok"``, ``"cancelled"``,
    or ``"agent_error"`` with the traceback text); raises
    :class:`SubprocessDiedError` on an unexplained death, and relays
    cancellation before re-raising."""

    def __init__(
        self,
        game: taaf.game.Game,
        payload: _ChildPayload,
        on_status: Any = None,
        on_yield: Any = None,
    ):
        """Host one game: its child payload, where status labels go, and
        what to do while the child offers its slot back."""
        self._game = game
        self._payload = payload
        # Callable[[str], None] applying a child-pushed status label, or
        # None. Typed loosely so this module keeps no status import beyond
        # the child-side handle.
        self._on_status = on_status
        # Awaitable Callable[[Mapping], None] that returns once the child
        # may run again, or None to answer every yield at once. This module
        # knows nothing about priority: it relays the request and parks the
        # reply for as long as the host says.
        self._on_yield = on_yield
        # Tokens spent after the child's last answer, pushed just before it
        # finishes. Held so both the clean path and the cancel relay can
        # charge them to the run.
        self._trailing: tuple[int, int] = (0, 0)

    async def run(self) -> tuple[str, str | None]:
        """Spawn the child and service it until it completes the protocol.
        Returns its terminal kind; raises on an unexplained death, and relays
        cancellation before re-raising."""
        parent_recv, to_parent = _SPAWN.Pipe(duplex=False)
        from_parent, parent_send = _SPAWN.Pipe(duplex=False)
        stderr_file = tempfile.NamedTemporaryFile(prefix="platypus_child_stderr_", suffix=".log", delete=False)
        stderr_path = Path(stderr_file.name)
        stderr_file.close()
        payload_blob = pickle.dumps(self._payload)  # loud parent-side pickle errors

        process = _SPAWN.Process(
            target=_child_entry,
            args=(to_parent, from_parent, payload_blob, str(stderr_path)),
            name=f"platypus-game-{self._game.game_id}",
        )
        process.start()
        to_parent.close()
        from_parent.close()

        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

        def _on_readable() -> None:
            """Pipe-readable callback: queue one child message, or EOF."""
            try:
                if not parent_recv.poll():
                    return
                message = parent_recv.recv()
            except (EOFError, OSError):
                loop.remove_reader(parent_recv.fileno())
                inbox.put_nowait(None)
                return
            inbox.put_nowait(message)

        loop.add_reader(parent_recv.fileno(), _on_readable)
        agent_error: str | None = None
        gave_up: str | None = None
        try:
            while True:
                tag, value = await self._next_message(inbox, process, stderr_path, _on_readable)
                if tag == _DONE:
                    if value == "agent_error":
                        # Leave the run unfinished: Benchmark's teardown
                        # pre-stamps "crashed" and finishes it there, which
                        # is what keeps a crash distinguishable from a game
                        # that merely answered. Its trailing tokens still
                        # have to land, so they go on directly.
                        self._charge_trailing()
                        return "agent_error", agent_error
                    self._finish(gave_up)
                    return value, None
                if tag == _AGENT_ERROR:
                    agent_error = value
                    if self._game.game_run is not None:
                        self._game.game_run.solver_note = f"agent error:\n{value}"
                elif tag == _GAVE_UP:
                    gave_up = value
                elif tag == _YIELD:
                    await self._serve_yield(parent_send, value)
                else:
                    self._apply(tag, value)
        except asyncio.CancelledError:
            # Relay: SIGTERM cancels the child's play call. The send pipe
            # stays open — the grace loop below raises the cancellation into
            # a child parked in a yield, which is how a parked child unwinds
            # at all. finish_game() first: we are cancelling, so it stamps
            # "cancelled" and scores whatever the child had already
            # submitted.
            process.terminate()
            with contextlib.suppress(Exception):
                self._game.finish_game()
            # Shielded because this drain is itself the cleanup: `_fan_out`
            # cancels each task again while it runs, and an await in an
            # already-cancelled task raises at once, which would close the
            # pipe below before the child reports its trailing tokens.
            # `_CANCEL_GRACE_S` bounds the wait.
            drain = asyncio.ensure_future(self._drain_until_done(inbox, process, parent_send))
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    continue
            raise
        finally:
            with contextlib.suppress(OSError):
                loop.remove_reader(parent_recv.fileno())
            for connection in (parent_recv, parent_send):
                with contextlib.suppress(OSError):
                    connection.close()
            await self._reap(process)
            # Always removed, dead or alive: the death report already embeds
            # the stderr tail (SubprocessDiedError -> solver_note ->
            # benchmark.json), so the file has no remaining reader and would
            # otherwise accumulate one unlabeled /tmp entry per dead child.
            stderr_path.unlink(missing_ok=True)

    def _apply(self, tag: str, value: Any) -> None:
        """Apply one child message that is not a terminal report."""
        if tag == _OUTPUT:
            _print_child_output(value)
        elif tag == _STATUS:
            self._apply_status(value)
        elif tag == _SUBMIT:
            self._serve_submit(value)
        elif tag == _TOKENS:
            self._trailing = value
        else:  # pragma: no cover - protocol drift is a bug
            raise SubprocessDiedError(f"unknown message from child: {tag!r}")

    def _finish(self, gave_up: str | None) -> None:
        """Finish the game with the trailing token delta, once the child is
        done. Idempotent through TAAF, so the cancel relay having already
        finished it is harmless."""
        run = self._game.game_run
        if gave_up is not None and run is not None:
            run.solver_note = f"gave up: {gave_up}"
        generated, uncached = self._trailing
        with contextlib.suppress(Exception):
            self._game.finish_game(generated_tokens=generated, uncached_input_tokens=uncached)

    async def _next_message(
        self,
        inbox: asyncio.Queue[tuple[str, Any] | None],
        process: Any,
        stderr_path: Path,
        read_pending: Callable[[], None],
    ) -> tuple[str, Any]:
        """Wait for a child message with a liveness watchdog: a SIGKILLed
        child may never deliver EOF, and hanging here would hold a
        concurrency slot forever."""
        while True:
            try:
                message = await asyncio.wait_for(inbox.get(), timeout=_LIVENESS_TIMEOUT_S)
            except TimeoutError:
                if process.is_alive():
                    continue
                # The watchdog can run before the same loop's pipe reader.
                # Keep queued messages first, then read any buffered frame.
                if inbox.empty():
                    read_pending()
                message = None if inbox.empty() else inbox.get_nowait()
            if message is not None:
                return message
            # Collect the exit code before reporting: pipe EOF can arrive
            # before the dead child has been waited on (exitcode None
            # would misreport as "still running").
            process.join(timeout=2.0)
            raise SubprocessDiedError(
                f"game subprocess for {self._game.game_id!r} died ({_signame(process.exitcode)}) "
                f"without completing the protocol; child stderr tail:\n{_tail(stderr_path)}"
            )

    def _apply_status(self, label: Any) -> None:
        """Record what the child says it is doing. Best-effort by design:
        a reporting failure must never take down the game it describes."""
        if self._on_status is None:
            return
        with contextlib.suppress(Exception):
            self._on_status(str(label))

    async def _serve_yield(self, parent_send: multiprocessing.connection.Connection, hints: Any) -> None:
        """Hold the child's reply until the host says it may run again.

        Awaited inline in the message loop, which is what parks the child:
        it is blocked in ``recv`` until this answers, and while it is
        there it sends nothing this loop could miss. Cancellation is
        relayed as a raise rather than a grant, so a parked child unwinds
        through its yield instead of sitting out the whole grace window.
        """
        try:
            if self._on_yield is not None:
                await self._on_yield(hints)
        except asyncio.CancelledError:
            with contextlib.suppress(BrokenPipeError, OSError):
                parent_send.send((_RAISE, asyncio.CancelledError()))
            raise
        with contextlib.suppress(BrokenPipeError, OSError):
            parent_send.send((_STATE, None))

    def _serve_submit(self, value: Any) -> None:
        """Execute one of the child's answers against the real game.

        Nothing is sent back — the child validated the shape before sending,
        so a failure here is an infra bug rather than a wrong answer, and it
        is reported as a dead game rather than swallowed."""
        submission, generated, uncached = value
        try:
            self._game.execute_action(submission, generated_tokens=generated, uncached_input_tokens=uncached)
        except Exception as exc:
            raise SubprocessDiedError(
                f"the child's answer for {self._game.game_id!r} was refused by the game: {exc!r}"
            ) from exc

    async def _drain_until_done(
        self,
        inbox: asyncio.Queue[tuple[str, Any] | None],
        process: Any,
        parent_send: multiprocessing.connection.Connection,
    ) -> bool:
        """Grace window after the cancellation relay: keep servicing the
        child so it can unwind cleanly and report its trailing tokens — a
        yield that arrives now gets the cancellation raised into it, since
        a child blocked in that request would otherwise sit out the whole
        window. SIGKILL is the reaper's job. A single pending getter
        survives timeouts so no message is lost to an orphaned waiter."""
        deadline = time.monotonic() + _CANCEL_GRACE_S
        getter: asyncio.Task[tuple[str, Any] | None] = asyncio.ensure_future(inbox.get())
        try:
            while time.monotonic() < deadline:
                timeout = max(min(_LIVENESS_TIMEOUT_S, deadline - time.monotonic()), 0.1)
                done, _ = await asyncio.wait({getter}, timeout=timeout)
                if getter in done:
                    message = getter.result()
                    if message is None:
                        return False
                    tag, value = message
                    if tag == _DONE:
                        self._charge_trailing()
                        return True
                    if tag == _TOKENS:
                        self._trailing = value
                    elif tag == _OUTPUT:
                        _print_child_output(value)
                    elif tag == _STATUS:
                        self._apply_status(value)
                    elif tag == _YIELD:
                        with contextlib.suppress(BrokenPipeError, OSError):
                            parent_send.send((_RAISE, asyncio.CancelledError()))
                    # A _SUBMIT that crossed after the cancel landed is
                    # dropped: finish_game has already run, so recording it
                    # would raise. Whatever the child sent before the cut is
                    # already in the run.
                    getter = asyncio.ensure_future(inbox.get())
                elif not process.is_alive():
                    return False
            return False
        finally:
            getter.cancel()

    def _charge_trailing(self) -> None:
        """Stamp the child's trailing token delta onto a run that
        ``finish_game`` already closed at the cancel relay."""
        generated, uncached = self._trailing
        run = self._game.game_run
        if run is not None:
            run.final_generated_tokens = generated
            run.final_uncached_input_tokens = uncached

    async def _reap(self, process: Any) -> None:
        """Run the escalation ladder on this game's child, labelled for its log lines."""
        await reap_child(process, label=f"{self._game.game_id} pid {process.pid}")


def _detach_orphan(process: Any) -> None:
    """Forget a child that cannot be ended, so the interpreter's shutdown
    does not wait on it.

    Two pieces of multiprocessing's bookkeeping both block at exit on a
    child that never exits. The active-children set is joined member by
    member with no timeout by ``multiprocessing.util``'s exit hook, and a
    timed-out ``join`` leaves the child in it. The resource tracker — the
    helper process every spawned child inherits a pipe to — is waited on by
    its own finalizer, and it only exits once every holder of that pipe has
    closed it, which the child never will. So the child leaves the set, as
    ``join`` would have made it on exit, and the parent closes its end of
    the tracker's pipe and forgets the tracker: its finalizer then has
    nothing to wait for, and the next spawn starts a fresh one. The old
    tracker lives on until the orphan dies, and cleans up whatever was
    registered with it then; anything registered from this process after
    the detach goes to the fresh one.

    Both are private names, and the only handles there are.
    """
    multiprocessing.process._children.discard(process)  # type: ignore[attr-defined]
    tracker = multiprocessing.resource_tracker._resource_tracker
    with tracker._lock:  # type: ignore[attr-defined]
        if tracker._fd is not None:  # type: ignore[attr-defined]
            os.close(tracker._fd)  # type: ignore[attr-defined]
            tracker._fd = None  # type: ignore[attr-defined]
            tracker._pid = None  # type: ignore[attr-defined]


async def reap_child(process: Any, *, label: str) -> None:
    """Escalation ladder: pipes are already closed (unblocks the child);
    poll politely, then SIGKILL, then give up. The kill/join lives in a
    ``finally`` and is synchronous — a second cancellation landing during
    the polling awaits must not abort the reap (that leaves an unjoined
    orphan). ``join`` after SIGKILL returns in milliseconds.

    Every rung past the first is announced on stdout, because nothing else
    records it: a child that had to be killed leaves no trace in the game's
    result, and a child that outlives SIGKILL would otherwise be visible
    only through whatever its survival breaks next.

    That last rung is the point of the ladder having an end. A process that
    is still alive after SIGKILL and a join is in uninterruptible kernel
    wait — the only state a kill cannot end — and there is nothing further
    the host can do to it. It is abandoned, loudly, and this returns
    normally: ``close()`` raises on a live process, and a raise out of a
    game's ``finally`` reaches the solver's gather and cancels every
    sibling in the run's fan-out over one stuck child. ``close()`` runs
    only on a child that is actually dead.

    Abandoning also means detaching (``_detach_orphan``): the interpreter
    would otherwise wait on the child twice at shutdown, and the parent
    would hang on the very child it gave up on.
    """
    escalated = False
    try:
        deadline = time.monotonic() + _KILL_AFTER_S
        while process.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
    finally:
        if process.is_alive():
            escalated = True
            print(
                f"[reaper] {label}: still alive {_KILL_AFTER_S:.0f}s after its pipes closed, sending SIGKILL",
                flush=True,
            )
            process.kill()
        process.join(timeout=5.0)
        if process.is_alive():
            print(
                f"[reaper] {label}: SURVIVED SIGKILL and a 5s join (uninterruptible wait?); "
                "abandoning it as an orphan so the run can continue",
                flush=True,
            )
            _detach_orphan(process)
        else:
            if escalated:
                print(f"[reaper] {label}: exited after SIGKILL", flush=True)
            process.close()


async def host_game_in_subprocess(
    game: taaf.game.Game,
    agent: Agent,
    rng_seed: int,
    config: RunConfig,
    llm_backend: Any | None = None,
    diag_paths: tuple[str, str] | None = None,
    on_status: Any = None,
    job_dir: str | None = None,
    on_yield: Any = None,
) -> tuple[str, str | None]:
    """Run one game with the agent in a child process. Returns
    ``(kind, agent_error_traceback)`` where kind is ``"ok"`` /
    ``"cancelled"`` / ``"agent_error"``; raises loudly on child death.

    ``on_status`` is ``Callable[[str], None]`` receiving the labels the
    child publishes through ``ctx.status``, or None to leave the child
    silent (nothing is then computed or sent). ``on_yield`` is an async
    callable that returns once the child may run again after offering its
    slot back (``ctx.yield_slot``); None answers every offer at once."""
    payload = _ChildPayload(
        agent_blob=pickle.dumps(agent),
        game_id=game.game_id,
        pass_index=game.game_run.pass_index if game.game_run is not None else 0,
        view=game.query(),
        rng_seed=rng_seed,
        memory_limit_bytes=config.subprocess_memory_limit_bytes,
        max_runtime_s=config.max_runtime_s_per_game,
        llm_backend=llm_backend,
        diag_paths=diag_paths,
        status_enabled=on_status is not None,
        job_dir=job_dir,
    )
    return await SubprocessGameHost(game, payload, on_status, on_yield).run()
