"""Subprocess-per-game hosting (I9).

The taaf ``Game`` (arcade session, GameRun bookkeeping) stays in the
parent; the agent and its move loop run in a spawned child. Actions flow
child->parent, ``GameState``s parent->child over two one-directional
pipes. Anything going wrong — child death, memory-limit breach, pickle
failure — is a loud error for that game (or the run, under
``on_agent_error="raise"``); there is deliberately no fallback to
in-process hosting.

Hardening choices follow aaf's ``solver_subprocess`` (each earned):

- **spawn** context: a fresh interpreter per child — no inherited CUDA
  contexts, threads, or half-initialized module state.
- **liveness watchdog**: the parent's wait on the child pipe times out
  every few seconds and checks the process, because a SIGKILLed/OOMed
  child may never deliver pipe EOF — without this the parent would hang
  forever while holding a concurrency slot.
- **cancellation relay**: parent sends SIGTERM (the child's loop cancels
  its move-loop task, so I5 semantics — cancel at the next await,
  shielded ``on_finish`` — hold inside the child) and closes its send
  pipe (unblocks a child parked in a synchronous ``recv`` via EOF).
  Grace period, then SIGKILL.
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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import taaf.game

import platypus.memory
from platypus.checkpoints import GameCheckpoints
from platypus.context import HistoryView, MoveContext, new_move_record
from platypus.contract import GameInfo, GameResult, Move, RunConfig
from platypus.slots import SlotView
from platypus.status import NULL_STATUS, StatusHandle

if TYPE_CHECKING:
    from platypus.agent import Agent

_SPAWN = multiprocessing.get_context("spawn")

# child -> parent
_EXECUTE = "execute"
_YIELD = "yield"  # request-reply: answered when the host re-grants the slot
_FINISH = "finish"
_CANCELLED = "cancelled"
_AGENT_ERROR = "agent_error"
_DONE = "done"
_STATUS = "status"  # one-way push, never answered (platypus.status)
_OUTPUT = "output"  # one-way push, never answered: the child's stdout
# parent -> child
_STATE = "state"
_RAISE = "raise"
_RESULT = "result"

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

    agent_blob: bytes  # pickled separately so agent pickle errors surface in the parent (A1)
    game_info: GameInfo
    initial_state: taaf.game.GameState
    number_of_levels: int | None
    base_actions_per_level: list[int] | None
    rng_seed: int  # resolved by AgentSolver._game_seed; always concrete
    memory_limit_bytes: int
    # LLM layer: this game's backend client (a picklable ProxyBackend that
    # connects to the host-side socket from the child), or None for
    # LLM-free agents. Typed Any: this module stays ullm-free.
    llm_backend: Any = None
    # Diagnostics (I7): (stream path, block-store dir), or None when off.
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
    # Where the child writes its level checkpoints, where its observation
    # numbering starts, and the random state it resumes, if any
    # (``platypus.checkpoints``). Plain values, like the rest.
    checkpoints: GameCheckpoints = GameCheckpoints()


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


def _picklable_exception(exc: BaseException) -> BaseException:
    """Exceptions cross the pipe; replace unpicklable ones with a
    RuntimeError carrying the traceback text so the other side always has
    something raisable."""
    try:
        pickle.loads(pickle.dumps(exc))
        return exc
    except Exception:
        return RuntimeError("".join(traceback.format_exception(exc)))


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
    run the agent loop. A setup failure before the loop starts (unpicklable
    payload, a memory limit that will not apply) sends nothing — it prints to
    the captured stderr and kills the child, and the parent surfaces it
    through the death report, whose stderr tail is the only account of it."""
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
    # Imported here, not at module top: solver.py imports this module.
    """The child side: rebuild the context, run the shared move loop, and
    report the outcome back over the pipe."""
    from platypus.solver import AgentSolver, run_move_loop

    agent: Agent = pickle.loads(payload.agent_blob)
    # Checked once the agent is in, because unpickling it is what runs its
    # module's imports. vllm (and the torch under it) is the host's engine,
    # never a game's: a child reaches inference through its proxy client, and
    # importing vllm here would cost ~365 MB of private heap PER GAME —
    # ~40 GB across a 110-game submission, for nothing the child calls. It is
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
    moves = new_move_record()
    rng = random.Random(payload.rng_seed)
    diag = AgentSolver._diag_logger(payload.diag_paths)
    # Push-on-change, never answered: the parent applies the label as it
    # services the pipe. A child wedged inside a kernel call or an LLM
    # request stops pushing, and the parent keeps reporting the state it
    # last declared — which is the state it is genuinely stuck in.
    status = StatusHandle(lambda label: port.send(_STATUS, label)) if payload.status_enabled else NULL_STATUS
    slots = SlotView(job_dir=Path(payload.job_dir) if payload.job_dir else None)
    if payload.llm_backend is None:
        ctx: MoveContext = MoveContext(
            history=HistoryView(moves),
            game=payload.game_info,
            rng=rng,
            shared={},
            diag=diag,
            status=status,
            slots=slots,
        )
        meter = None
    else:
        import platypus.llm

        ctx, meter = platypus.llm.build_llm_context(
            moves, payload.game_info, rng, payload.llm_backend, diag, status, slots
        )

    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    assert main_task is not None
    # SIGTERM from the parent = cancellation; lands at the agent's next
    # await, exactly like in-process I5 semantics.
    loop.add_signal_handler(signal.SIGTERM, main_task.cancel)

    def _tokens() -> tuple[int, int]:
        # Tokens spent since the last report (I1); (0, 0) for LLM-free agents.
        """Tokens spent since the last report (I1)."""
        return meter.take() if meter is not None else (0, 0)

    async def execute(action: Any) -> tuple[taaf.game.GameState, str]:
        """Ask the parent to execute one action against the real game."""
        reply = port.request(_EXECUTE, (action, *_tokens()))
        return reply  # (GameState, run_state)

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

    try:
        try:
            give_up = await run_move_loop(
                agent,
                ctx,
                moves,
                execute,
                payload.initial_state,
                "playing",
                payload.number_of_levels,
                payload.base_actions_per_level,
                payload.checkpoints,
            )
        finally:
            if payload.llm_backend is not None:
                import platypus.llm

                platypus.llm.record_replay_consumption(payload.llm_backend, diag)
            diag.close()
    except (asyncio.CancelledError, _ParentGone):
        # The parent initiated this; it has finished the game and serves
        # the real result (score included) during its grace window. The
        # trailing token delta rides along for the run's final_* fields.
        try:
            result: GameResult = port.request(_CANCELLED, _tokens())
        except (_ParentGone, asyncio.CancelledError):
            result = GameResult(state="cancelled", levels_completed=_levels(moves, payload), score=None)
        await _shielded_on_finish(agent, ctx, result)
        port.send(_DONE, "cancelled")
        return
    except BaseException as exc:
        formatted = "".join(traceback.format_exception(exc))
        result = GameResult(state="crashed", levels_completed=_levels(moves, payload), score=None)
        # Report first (parent records the note), then best-effort on_finish.
        port.send(_AGENT_ERROR, formatted)
        await _shielded_on_finish(agent, ctx, result)
        port.send(_DONE, "agent_error")
        return

    reason = give_up.reason if give_up is not None else None
    try:
        result: GameResult = port.request(_FINISH, (reason, *_tokens()))
    except _ParentGone:
        result = GameResult(state="cancelled", levels_completed=_levels(moves, payload), score=None)
    await _shielded_on_finish(agent, ctx, result)
    port.send(_DONE, "ok")


def _levels(moves: Sequence[Move], payload: _ChildPayload) -> int:
    """Levels completed as the child last saw them, for a result it has to
    report without the parent answering."""
    if moves:
        return moves[-1].observations[-1].state.levels_completed
    return payload.initial_state.levels_completed


async def _shielded_on_finish(agent: Agent, ctx: MoveContext, result: GameResult) -> None:
    """Run ``on_finish`` so a cancellation in flight cannot cut it short (I5);
    a failure inside it is logged to the captured stderr, never raised."""
    try:
        await asyncio.shield(agent.on_finish(ctx, result))
    except asyncio.CancelledError:
        pass
    except Exception:
        traceback.print_exc()  # captured stderr


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------


class SubprocessGameHost:
    """Hosts one game's child process and services its requests against
    the real (parent-side) ``taaf.game.Game``.

    ``run()`` returns the child's terminal kind (``"ok"``, ``"agent_error"``
    with the traceback text, or raises: :class:`SubprocessDiedError` on an
    unexplained death, ``CancelledError`` after relaying cancellation)."""

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

    async def run(self) -> tuple[str, str | None]:
        """Spawn the child and service it until it completes the protocol.
        Returns its terminal kind; raises on an unexplained death, and relays
        cancellation before re-raising."""
        parent_recv, to_parent = _SPAWN.Pipe(duplex=False)
        from_parent, parent_send = _SPAWN.Pipe(duplex=False)
        stderr_file = tempfile.NamedTemporaryFile(prefix="platypus_child_stderr_", suffix=".log", delete=False)
        stderr_path = Path(stderr_file.name)
        stderr_file.close()
        payload_blob = pickle.dumps(self._payload)  # loud parent-side pickle errors (A1)

        process = _SPAWN.Process(
            target=_child_entry,
            args=(to_parent, from_parent, payload_blob, str(stderr_path)),
            name=f"platypus-game-{self._game.game_id}",
        )
        process.start()
        # Read now: the reaper closes the process object, after which even
        # its pid is refused.
        pid = process.pid
        # Named for the memory report: its REPLs are this pid's descendants,
        # and only this side knows which game a pid is playing.
        platypus.memory.register_game_process(pid, self._game.game_id)
        to_parent.close()
        from_parent.close()

        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

        def _on_readable() -> None:
            """Pipe-readable callback: queue one child message, or EOF."""
            try:
                message = parent_recv.recv()
            except (EOFError, OSError):
                loop.remove_reader(parent_recv.fileno())
                inbox.put_nowait(None)
                return
            inbox.put_nowait(message)

        loop.add_reader(parent_recv.fileno(), _on_readable)
        agent_error: str | None = None
        try:
            while True:
                message = await self._next_message(inbox, process, stderr_path)
                tag, value = message
                if tag == _OUTPUT:
                    _print_child_output(value)
                elif tag == _STATUS:
                    self._apply_status(value)
                elif tag == _EXECUTE:
                    self._serve_execute(parent_send, value)
                elif tag == _YIELD:
                    await self._serve_yield(parent_send, value)
                elif tag == _FINISH:
                    reason, generated, uncached = value
                    if reason is not None and self._game.game_run is not None:
                        self._game.game_run.solver_note = f"gave up: {reason}"
                    self._game.finish_game(generated_tokens=generated, uncached_input_tokens=uncached)
                    parent_send.send((_RESULT, self._result()))
                elif tag == _CANCELLED:
                    # The child cancelled itself — an external SIGTERM to the
                    # process group, or agent code letting CancelledError
                    # escape. Not our relay (that path runs the grace loop
                    # below), and not a dead child: finish the game and answer
                    # so it can unwind through on_finish like any other end.
                    generated, uncached = value
                    self._game.finish_game(generated_tokens=generated, uncached_input_tokens=uncached)
                    parent_send.send((_RESULT, self._result()))
                elif tag == _AGENT_ERROR:
                    agent_error = value
                    if self._game.game_run is not None:
                        self._game.game_run.solver_note = f"agent error:\n{value}"
                elif tag == _DONE:
                    return ("agent_error", agent_error) if value == "agent_error" else (value, None)
                else:  # pragma: no cover - protocol drift is a bug
                    raise SubprocessDiedError(f"unknown message from child: {tag!r}")
        except asyncio.CancelledError:
            # Relay: SIGTERM cancels the child's move loop. The send pipe
            # stays open — the grace loop below answers the child's
            # requests (an in-flight execute gets the cancellation raised
            # into it; the child then asks for the real, score-carrying
            # result). finish_game() first: we are cancelling, so it
            # stamps "cancelled" and computes the final score.
            process.terminate()
            self._game.finish_game()
            # Shielded because this drain is itself the cleanup: `_fan_out`
            # cancels each task again while it runs, and an await in an
            # already-cancelled task raises at once, which would close the
            # pipes below before the child reports its trailing tokens.
            # `_CANCEL_GRACE_S` bounds the wait.
            drain = asyncio.ensure_future(self._drain_until_cancelled_done(inbox, process, parent_send))
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
            platypus.memory.forget_game_process(pid)
            # Always removed, dead or alive: the death report already embeds
            # the stderr tail (SubprocessDiedError -> solver_note ->
            # benchmark.json), so the file has no remaining reader and would
            # otherwise accumulate one unlabeled /tmp entry per dead child.
            stderr_path.unlink(missing_ok=True)

    async def _next_message(
        self,
        inbox: asyncio.Queue[tuple[str, Any] | None],
        process: Any,
        stderr_path: Path,
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

    def _serve_execute(self, parent_send: multiprocessing.connection.Connection, value: Any) -> None:
        """Execute one action against the real game and answer the child —
        relaying an engine failure so it unwinds inside the agent."""
        action, generated, uncached = value
        try:
            game_state = self._game.execute_action(
                action.to_action_input(), generated_tokens=generated, uncached_input_tokens=uncached
            )
        except Exception as exc:  # engine/infra error: relay so the child unwinds
            parent_send.send((_RAISE, _picklable_exception(exc)))
            return
        run = self._game.game_run
        parent_send.send((_STATE, (game_state, run.state if run is not None else "playing")))

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

    def _result(self) -> GameResult:
        """The finished game as the agent's ``on_finish`` should see it."""
        run = self._game.game_run
        assert run is not None
        run_state = run.state
        if run_state in ("not_started", "playing"):
            return GameResult(state="crashed", levels_completed=self._game.current_state.levels_completed, score=None)
        return GameResult(
            state=run_state, levels_completed=self._game.current_state.levels_completed, score=run.final_score
        )

    async def _drain_until_cancelled_done(
        self,
        inbox: asyncio.Queue[tuple[str, Any] | None],
        process: Any,
        parent_send: multiprocessing.connection.Connection,
    ) -> bool:
        """Grace window after the cancellation relay: keep servicing the
        child so it can unwind cleanly — an in-flight execute gets the
        cancellation raised into it, and the child's result request is
        answered with the real (finished, score-carrying) result. SIGKILL
        is the reaper's job. A single pending getter survives timeouts so
        no message is lost to an orphaned waiter."""
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
                        return True
                    if tag == _OUTPUT:
                        _print_child_output(value)
                    if tag == _STATUS:
                        self._apply_status(value)
                    with contextlib.suppress(BrokenPipeError, OSError):
                        if tag in (_EXECUTE, _YIELD):
                            parent_send.send((_RAISE, asyncio.CancelledError()))
                        elif tag in (_CANCELLED, _FINISH):
                            # finish_game already ran (with zero tokens) at
                            # the cancel relay; stamp the child's trailing
                            # token delta onto the run directly (R11.03).
                            generated, uncached = value if tag == _CANCELLED else value[1:]
                            run = self._game.game_run
                            if run is not None:
                                run.final_generated_tokens = generated
                                run.final_uncached_input_tokens = uncached
                            parent_send.send((_RESULT, self._result()))
                    getter = asyncio.ensure_future(inbox.get())
                elif not process.is_alive():
                    return False
            return False
        finally:
            getter.cancel()

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
    multiprocessing.process._children.discard(process)  # type: ignore[attr-defined]  # noqa: SLF001
    tracker = multiprocessing.resource_tracker._resource_tracker  # noqa: SLF001
    with tracker._lock:  # type: ignore[attr-defined]  # noqa: SLF001
        if tracker._fd is not None:  # type: ignore[attr-defined]  # noqa: SLF001
            os.close(tracker._fd)  # type: ignore[attr-defined]  # noqa: SLF001
            tracker._fd = None  # type: ignore[attr-defined]  # noqa: SLF001
            tracker._pid = None  # type: ignore[attr-defined]  # noqa: SLF001


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
    game_info: GameInfo,
    rng_seed: int,
    config: RunConfig,
    llm_backend: Any | None = None,
    diag_paths: tuple[str, str] | None = None,
    on_status: Any = None,
    job_dir: str | None = None,
    on_yield: Any = None,
    checkpoints: GameCheckpoints = GameCheckpoints(),
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
        game_info=game_info,
        initial_state=game.current_state,
        number_of_levels=game.number_of_levels,
        base_actions_per_level=game.base_actions_per_level,
        rng_seed=rng_seed,
        memory_limit_bytes=config.subprocess_memory_limit_bytes,
        llm_backend=llm_backend,
        diag_paths=diag_paths,
        status_enabled=on_status is not None,
        job_dir=job_dir,
        checkpoints=checkpoints,
    )
    return await SubprocessGameHost(game, payload, on_status, on_yield).run()
