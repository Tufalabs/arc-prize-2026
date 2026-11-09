"""Run a search in its own process, and survive whatever it does.

Search runs model-written code — ``do_move`` and ``render`` — tens of
thousands of times per call. That code can loop forever, allocate without
bound, or segfault a C extension, and none of those may take the game down
with them. So the search runs in a child this module owns outright: it is
spawned per call, memory-capped, killed on a deadline, and reaped, and
what it wrote goes with it: every file it makes is under a scratch
directory the parent owns and removes, a kill or a cancellation included. It is
handed a job rather than a model, and the job builds the model inside the
child once the caps are on: loading a model runs its code too.

**Every outcome is a return value.** Nothing here raises to signal a
search outcome, including the failures. The agent reports a timeout or an
out-of-memory to the model like any other result (``seal.search``), and an
exception cannot carry the diagnostics back alongside it without being
decorated with them.

Two bounds, and they are not the same bound. The planner's own
``max_duration`` is checked between node expansions, so it cannot rescue a
single expansion that hangs; the hard timeout here can, because it belongs
to the parent. The planner's ``memory_limit_bytes`` self-monitor exits
gracefully with full diagnostics; ``RLIMIT_AS`` here is the backstop whose
death the parent reports.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import multiprocessing
import os
import queue
import resource
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.queues import Queue as MPQueue
from typing import Any, Callable, Generator, Literal, Protocol, Sequence

import numpy as np
import numpy.typing as npt
from ullm.shared_semaphore import SharedSemaphore
from ullm.tools.python import default_kernel_pool

from platypus.contract import Action
from platypus.status import NULL_STATUS, StatusHandle
from seal.model import WorldModel
from seal.planner import BFSPlanner, NoPlan, Plan
from seal.trace import LevelTrace

#: Spawn, not fork: the parent holds an event loop, kernel handles and
#: possibly a CUDA context, none of which survive a fork intact.
_CTX = multiprocessing.get_context("spawn")

#: How much of the child's stderr to keep on a failure path.
_STDERR_TAIL_CHARS = 8000

#: Room for the call a search child is inside (``do_move(state, Action(...))``):
#: a MOUSE move's repr is the longest a move gets.
_MARKER_BYTES = 256

#: Searches running at once on the machine. Each child may take up to its
#: memory cap, so this is what bounds the searches' memory together — the
#: figure that matters on a Kaggle box, where the kernels and the engine
#: share the host's memory with them. Every process must agree on it: the
#: named semaphore keeps whichever limit resolved first.
SEARCH_SLOTS = 12


def cpu_semaphore() -> SharedSemaphore:
    """The one CPU permit a search child and a running REPL cell both hold.

    They are two consumers of one machine, so a search contends on the kernel
    pool's permit instead of opening a second pool sized against the same
    cores. Taken outside any python call — a search runs after the call that
    asked for it — so no holder of a permit ever waits on a second one.
    """
    return default_kernel_pool().cpu_semaphore()


def search_semaphore() -> SharedSemaphore:
    """The machine-wide slot a search holds on top of its CPU permit, for
    memory (see :data:`SEARCH_SLOTS`)."""
    return SharedSemaphore("seal_search", SEARCH_SLOTS)


#: Set in the environment the child is spawned with, before anything in it
#: imports numpy: a search's work is single-threaded, and a BLAS thread pool
#: sized for the machine's cores would compete with the kernels for them.
_CHILD_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


Outcome = Literal["ok", "timeout", "died", "error", "refused"]

#: What the child puts on the queue, once: ``("ok", (Plan | NoPlan, predicted
#: turns, boundary reason))``, ``("refused", why)`` or ``("err", traceback
#: string)``. The queue starts a thread in the child at its first put, whose
#: stack and allocator arena would come out of the memory budget, so nothing
#: goes on it before the search is over.
Payload = tuple[str, Any]


class PreparationRefused(RuntimeError):
    """A job could not set its search up for a reason of the harness's rather
    than the model's: the search does not run, and the caller ends the game
    rather than tell the model something false."""


class SearchJob(Protocol):
    """What a search searches, as plain data the child builds it from."""

    def prepare(self, announce: Callable[[dict[str, str]], None]) -> tuple[WorldModel, LevelTrace]:
        """The model and the level's trace, built in the child under its
        caps. ``announce`` takes the digests of the files the job read, before
        any of their code runs, so a child killed later still says what it
        read. Raises :class:`PreparationRefused` for a harness fault."""
        ...


@dataclass(frozen=True)
class PredictedTurn:
    """One move of a plan as the model predicts it: the board its state draws
    (``None`` when ``render`` raised on it) and the outcome. No state: the
    kernel steps its own model to the same states, and a state holding the
    model's classes pickles only against the process that loaded them."""

    action: Action
    frame: npt.NDArray[np.int8] | None
    outcome: str


def predict_turns(
    model: WorldModel, start_state: dict[str, Any], actions: Sequence[Action], *, boundary: str | None = None
) -> tuple[list[PredictedTurn], str | None]:
    """Walk ``actions`` through ``model`` from a copy of ``start_state``.

    Stops at a move the model makes no claim for — ``Unknown``, or code that
    raised — which is where a boundary plan ends, and returns why, as the
    model reads an error. A ``render_error`` boundary ends one move later: the
    move is predicted, and the reason is what ``render`` raised on the state
    it leads to. Any other board that does not draw is ``None``.
    """
    import traceback

    from seal.errors import for_model

    state = copy.deepcopy(start_state)
    turns: list[PredictedTurn] = []
    with model.active():
        for index, action in enumerate(actions):
            prediction = model.step(state, action)
            if prediction.kind == "unknown":
                return turns, f"do_move raised Unknown({prediction.unknown_reason!r})."
            if prediction.kind == "error":
                return turns, "do_move raised:\n" + for_model(prediction.error or "", where="world model")
            try:
                frame: npt.NDArray[np.int8] | None = model.render(state)
            except Exception as error:
                if boundary == "render_error" and index == len(actions) - 1:
                    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                    return turns + [PredictedTurn(action, None, str(prediction.outcome))], (
                        "render raised:\n" + for_model(rendered, where="world model")
                    )
                frame = None
            turns.append(PredictedTurn(action, frame, str(prediction.outcome)))
    return turns, None


@dataclass(frozen=True)
class SearchResult:
    """What the child produced, or what happened to it instead.

    ``value`` is populated only when ``outcome == "ok"``. ``stderr`` is the
    tail of the child's stderr on every failure path — the only thing that
    survives a child that dies before it can report anything.
    """

    outcome: Outcome
    value: Plan | NoPlan | None = None
    #: For a plan, what the model predicts each of its moves leads to, from
    #: the trace's current state — up to the last move a boundary plan ends
    #: on, whose effect the model does not know. Computed in the child, with
    #: the model that searched.
    predicted: tuple[PredictedTurn, ...] = ()
    #: For a boundary plan, why the model made no claim for its last move:
    #: the ``Unknown`` reason, or the traceback its code raised.
    boundary_reason: str | None = None
    #: For a child that was killed or died: the call into the model's code it
    #: was inside, or None when it was in the search's own work.
    stuck_in: str | None = None
    #: ``{file name: sha256 hex}`` of what the job read; empty when the child
    #: never got that far.
    digests: dict[str, str] = field(default_factory=dict[str, str])
    stderr: str = ""
    error: str | None = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the search returned a value rather than dying or timing out."""
        return self.outcome == "ok"

    def describe(self) -> str:
        """One line naming the outcome and what it cost."""
        if self.ok:
            return f"search ok in {self.duration_s:.1f}s"
        detail = f": {self.error}" if self.error else ""
        return f"search {self.outcome} after {self.duration_s:.1f}s{detail}"


def _child(
    q: "MPQueue[Payload]",
    planner: BFSPlanner,
    job: SearchJob,
    available_actions: Sequence[str],
    seed: int,
    stderr_path: str,
    digests_path: str,
    memory_limit_bytes: int | None,
    marker: Any,
) -> None:
    """Prepare the job and search it, and put the outcome on the queue.

    The memory cap is on before the job runs any of the model's code, and the
    planner's time budget is what the preparation left of it.

    Stdout goes to ``/dev/null``: model-written ``do_move`` prints, and at
    search volume that floods the parent's log. Stderr goes to a file the
    parent owns, so a bootstrap traceback or a C-extension abort is still
    recoverable when the child dies before reaching the reporting path; the
    digests of the files the job read go to another, for the same reason.
    """

    def _announce(digests: dict[str, str]) -> None:
        """Leave what the job read where the parent reads it after a kill."""
        with open(digests_path, "w") as out:
            json.dump(digests, out)

    _die_with_parent()
    # The process settings the kernel runs the model's code under.
    from seal.harness import game as harness_game

    harness_game._configure_model_process()  # pyright: ignore[reportPrivateUsage]

    def _mark(call: str | None) -> None:
        """Leave the call in shared memory, where the parent reads it after a kill."""
        marker.value = (call or "").encode("utf-8", "replace")[: _MARKER_BYTES - 1]

    with open(os.devnull, "w") as devnull, open(stderr_path, "w") as errf:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(errf):
            try:
                # Inside the reporting block, so a machine that will not take
                # the cap says so through the same path a search failure uses
                # rather than dying before anything can read it.
                if memory_limit_bytes is not None:
                    _cap_address_space(_address_space_reserved() + memory_limit_bytes)
                started = time.monotonic()
                model, trace = job.prepare(_announce)
                model.call_marker = _mark
                planner.max_duration = max(0.0, planner.max_duration - (time.monotonic() - started))
                outcome = planner.plan(model, trace, available_actions, seed=seed)
                predicted, reason = (
                    predict_turns(
                        model,
                        trace.current_state(),
                        outcome.actions,
                        boundary=outcome.diagnostics.boundary_type if outcome.target == "boundary" else None,
                    )
                    if isinstance(outcome, Plan)
                    else ([], None)
                )
                q.put(("ok", (outcome, tuple(predicted), reason)))
            except PreparationRefused as refusal:
                q.put(("refused", str(refusal)))
            except BaseException as e:
                # Mirrored into stderr as well as shipped, so the parent can
                # surface it even if the queue path is what broke.
                traceback.print_exc(file=errf)
                try:
                    q.put(("err", "".join(traceback.format_exception(type(e), e, e.__traceback__))))
                except Exception:
                    q.put(("err", f"unreportable worker exception {type(e).__name__}"))
            finally:
                errf.flush()


def _address_space_reserved() -> int:
    """The address space this process has reserved so far, in bytes.

    Read once the child has unpickled its arguments — the interpreter, numpy,
    the harness and the model loaded — so the cap can be that plus the
    search's budget. Reserved is not used: the libraries map far more than
    they touch, and a job's memory limit counts only what is touched, so a
    cap that included the reservation would spend the budget on nothing.
    """
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("search worker could not read its address space from /proc/self/status")


def _cap_address_space(limit: int) -> None:
    """Cap this process's address space, or refuse to search without one.

    The cap is the backstop for model-written code: the planner's own memory
    check runs between expansions, so a single allocation inside one reaches
    the machine before anything looks. A search that runs without the cap has
    no second line, and the failure it produces lands somewhere else — an
    OOM-killed worker, or a node taken out from under every other game on it.

    So the limit is set and then read back. ``setrlimit`` reports success for
    a value the kernel has already clamped, which would leave the search
    running under a ceiling nobody chose while everything here claimed
    otherwise.
    """
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError) as error:
        raise RuntimeError(f"search worker could not cap its address space at {limit} bytes: {error}") from error
    soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft != limit:
        raise RuntimeError(f"search worker asked for an address-space cap of {limit} bytes; the process reports {soft}")


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this child if the parent dies.

    The infra's no-orphans machinery covers the children *it* spawns; a
    search is a grandchild, so an abrupt kill of the per-game process would
    otherwise leave a search running. Linux-only, and best-effort — a
    failure here costs orphan-safety, not correctness.
    """
    with contextlib.suppress(Exception):
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, 9, 0, 0, 0)


async def run_search(
    planner: BFSPlanner,
    job: SearchJob,
    available_actions: Sequence[str],
    *,
    seed: int,
    hard_timeout: float,
    memory_limit_bytes: int | None = None,
    status: StatusHandle = NULL_STATUS,
) -> SearchResult:
    """Search ``job`` in a child process; never raise for a search outcome
    (a refused preparation is outcome ``refused``).

    Cancellation is clean: the wait below is a poll loop, so a
    ``CancelledError`` lands in the ``await`` and the ``finally`` kills and
    reaps the child on the way out. That poll is also what keeps the caller
    yielding to the event loop while a search runs.

    ``status`` separates the two waits, which look identical from outside
    and mean opposite things: queueing behind the machine's other searches
    says the box is oversubscribed, while running one says this game is
    working. Defaults to the null handle, so a caller with no context — a
    test, an offline replay — needs no wiring.

    ``memory_limit_bytes`` is what the child may reserve beyond its address
    space once its imports and arguments are loaded; None leaves it uncapped.
    """
    # The search slot first, then the CPU permit: a search queued for memory
    # holds no core while it waits.
    slots, cpu = search_semaphore(), cpu_semaphore()
    with status("waiting for a search slot"):
        await slots.acquire()
    try:
        with status("waiting for a search slot"):
            await cpu.acquire()
        try:
            with status("searching"):
                return await _run(
                    planner,
                    job,
                    available_actions,
                    seed=seed,
                    hard_timeout=hard_timeout,
                    memory_limit_bytes=memory_limit_bytes,
                )
        finally:
            cpu.release()
    finally:
        slots.release()


async def _run(
    planner: BFSPlanner,
    job: SearchJob,
    available_actions: Sequence[str],
    *,
    seed: int,
    hard_timeout: float,
    memory_limit_bytes: int | None,
) -> SearchResult:
    """Run one search in a spawned child and collect its outcome. Every failure
    mode — the hard timeout, an RLIMIT_AS death, a crash — comes back as a
    value, because the agent reports it to the model.
    """
    # The child's files, the model's copy among them, all under one directory
    # the child takes as its TMPDIR: a killed child runs no cleanup of its own.
    scratch = tempfile.mkdtemp(prefix="seal-search-")
    try:
        return await _run_in(
            scratch,
            planner,
            job,
            available_actions,
            seed=seed,
            hard_timeout=hard_timeout,
            memory_limit_bytes=memory_limit_bytes,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


async def _run_in(
    scratch: str,
    planner: BFSPlanner,
    job: SearchJob,
    available_actions: Sequence[str],
    *,
    seed: int,
    hard_timeout: float,
    memory_limit_bytes: int | None,
) -> SearchResult:
    """:func:`_run`, with the child's files under ``scratch``; returns once
    the child is reaped, a cancellation included."""
    stderr_path = os.path.join(scratch, "stderr.txt")
    digests_path = os.path.join(scratch, "files.json")

    q: MPQueue[Payload] = _CTX.Queue()
    # Unlocked: one writer, and the parent reads it only once the child is gone.
    marker: Any = _CTX.Array("c", _MARKER_BYTES, lock=False)
    proc = _CTX.Process(
        target=_child,
        args=(q, planner, job, list(available_actions), seed, stderr_path, digests_path, memory_limit_bytes, marker),
    )
    started = time.monotonic()
    with _environment({**_CHILD_ENV, "TMPDIR": scratch}):
        proc.start()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + hard_timeout
    status: str | None = None
    payload: Any = None  # narrowed by `status` below
    died_early = False
    timed_out = False
    try:
        while True:
            try:
                status, payload = q.get_nowait()
                break
            except queue.Empty:
                pass
            if not proc.is_alive():
                # Checked after the queue, not before: a child that finished
                # cleanly can already be dead with its result still in
                # flight, and reporting that as a death would lose a
                # perfectly good search.
                try:
                    status, payload = q.get_nowait()
                    break
                except queue.Empty:
                    died_early = True
                    break
            if loop.time() > deadline:
                timed_out = True
                break
            await asyncio.sleep(0.1)
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        q.close()
        q.join_thread()

    elapsed = time.monotonic() - started
    stuck_in = marker.value.decode("utf-8", "replace") or None
    stderr = _read_stderr(stderr_path) if (timed_out or died_early or status == "err") else ""
    digests = _read_digests(digests_path)

    if timed_out:
        return SearchResult(
            outcome="timeout",
            stderr=stderr,
            error=f"search exceeded its hard timeout of {hard_timeout:.0f}s and was killed",
            stuck_in=stuck_in,
            digests=digests,
            duration_s=elapsed,
        )
    if died_early:
        # The usual cause is the OS OOM-killer, or RLIMIT_AS turning an
        # allocation into a MemoryError the child could not report.
        return SearchResult(
            outcome="died",
            stderr=stderr,
            error=f"search process exited unexpectedly (exitcode={proc.exitcode})",
            stuck_in=stuck_in,
            digests=digests,
            duration_s=elapsed,
        )
    if status == "err":
        return SearchResult(outcome="error", stderr=stderr, error=str(payload), digests=digests, duration_s=elapsed)
    if status == "refused":
        return SearchResult(outcome="refused", error=str(payload), digests=digests, duration_s=elapsed)
    value, predicted, reason = payload
    return SearchResult(
        outcome="ok", value=value, predicted=predicted, boundary_reason=reason, digests=digests, duration_s=elapsed
    )


@contextlib.contextmanager
def _environment(values: dict[str, str]) -> Generator[None]:
    """``os.environ`` with ``values`` set, for exactly the span of a spawn: a
    spawned child takes the environment it is started with. Synchronous, so
    no other coroutine runs while it is changed."""
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _read_digests(path: str) -> dict[str, str]:
    """What the job announced it read; empty when the child never got that
    far, or was killed while writing it."""
    try:
        with open(path) as fh:
            return {str(name): str(digest) for name, digest in json.load(fh).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def _read_stderr(path: str) -> str:
    """The tail of the child's captured stderr, for the death report. Read
    from the end: a child whose model logs without bound leaves a file the
    parent must not load whole to keep its last lines."""
    try:
        with open(path, "rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            # A character takes at most four bytes, so this holds the tail.
            start = max(0, size - 4 * _STDERR_TAIL_CHARS)
            fh.seek(start)
            text = fh.read().decode(errors="replace")
    except OSError:
        return ""
    if start == 0 and len(text) <= _STDERR_TAIL_CHARS:
        return text
    return f"...<truncated>...\n{text[-_STDERR_TAIL_CHARS:]}"
