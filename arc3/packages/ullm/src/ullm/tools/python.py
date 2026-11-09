"""PythonTool: executes Python code in a persistent REPL subprocess.

The subprocess maintains state across calls (like a Jupyter kernel).
Communication is via JSON lines over stdin/stdout. Basic sandboxing
is applied: resource limits, clean environment, isolated tmpdir."""

import asyncio
import base64
import builtins
import contextlib
import copy
import io
import json
import os
import pathlib
import pickle
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from ullm.backend import StatusHook

import ullm.conv
import ullm.support as uls
import ullm.tool
from ullm.shared_semaphore import SharedSemaphore

_unshare_usable_cache: Optional[bool] = None


def _unshare_usable() -> bool:
    """Whether ``unshare --user --net`` works here. Kaggle's container (and
    macOS) disallow user namespaces, killing the kernel at spawn; probe once
    and cache. The isolation decision must be made where the kernel SPAWNS —
    tool config travels in pickles from launchers with different rules."""
    global _unshare_usable_cache
    if _unshare_usable_cache is None:
        try:
            _unshare_usable_cache = (
                subprocess.run(
                    ["unshare", "--user", "--net", "true"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            _unshare_usable_cache = False
    return _unshare_usable_cache


# Path to the worker script (in the same package directory)
_WORKER_SCRIPT = str(pathlib.Path(__file__).parent / "python_worker.py")

# Verbose logging flag — when True, prints kernel lifecycle and execution events
verbose: bool = False

# Stable model-visible recovery notice shared by every persistent REPL.
KERNEL_MEMORY_RESTART_NOTICE = (
    "The Python kernel was restarted after exhausting its memory limit. "
    "All Python variables and definitions were cleared."
)


def kernel_memory_exhausted(error: str) -> bool:
    """Whether an error means a persistent worker is unsafe to reuse."""
    lines = {line.strip() for line in error.splitlines()}
    return any(
        line in {"MemoryError", "MemoryError:"}
        or line.startswith("MemoryError: ")
        or "._ArrayMemoryError:" in line
        or line == "SystemError: error return without exception set"
        for line in lines
    )


@contextlib.contextmanager
def _bypass_debugger_subprocess_patch():
    """Temporarily restore original subprocess fork/exec functions.

    debugpy/pydevd patches _posixsubprocess.fork_exec and subprocess._fork_exec
    to inject debugger bootstrap code into child processes.  For sandboxed worker
    subprocesses with minimal environments this causes hangs or crashes.
    We detect the patches (stored as original_* attributes) and temporarily
    restore them.
    """
    patched: list[tuple[object, str, object, object]] = []  # (module, name, patched_fn, original_fn)

    for mod_name, func_name in [
        ("_posixsubprocess", "fork_exec"),
        ("subprocess", "_fork_exec"),
    ]:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        original = getattr(mod, f"original_{func_name}", None)
        if original is not None:
            patched.append((mod, func_name, getattr(mod, func_name), original))

    if not patched:
        yield
        return

    for mod, func_name, _, original in patched:
        setattr(mod, func_name, original)
    try:
        yield
    finally:
        for mod, func_name, patched_fn, _ in patched:
            setattr(mod, func_name, patched_fn)


def _sem_status(pool: "KernelPool") -> str:
    """Format semaphore state as a human-readable string.

    Force-initializes the pool's SharedSemaphores so the shared
    (cross-process) counts are visible even when this process has no
    live PythonTool (subprocess-dispatch mode — children acquire
    kernel slots, parent just reports)."""
    pool._ensure_semaphores()
    return f"{pool._exists_semaphore.status()}; {pool._running_semaphore.status()}"  # type: ignore[union-attr]


def pool_status(pool: "KernelPool") -> str:
    """One-line status for ``monitor_async_tasks`` reporters.

    Bind to a specific pool via a lambda or ``functools.partial``:
        reporters=[lambda: ullm.tools.python.pool_status(my_pool)]
    """
    return f"kernel pool: {_sem_status(pool)}"


def _log(msg: str, pool: "KernelPool") -> None:
    """Print a verbose log message with timestamp and semaphore state."""
    if verbose:
        import time

        ts = time.strftime("%H:%M:%S", time.localtime())
        print(f"[PythonTool {ts}] {msg}  [{_sem_status(pool)}]", flush=True)


@dataclass
class KernelPool(uls.BaseClass):
    """Manages concurrency limits for PythonTool kernels via shared
    semaphores.

    Shared across multiple PythonTool instances in one process, and —
    when ``ullm.shared_semaphore.enable_multiprocess_semaphores()`` is
    active — across OS processes too. The underlying primitives are
    ``SharedSemaphore``s, so the pool itself is picklable: it carries
    just the semaphore names + limits, and each process resolves to the
    same cross-process counter by name on first acquire.

    Semaphores are created lazily on first use; max_kernels/max_running
    cannot be changed after that."""

    # Semaphore name prefix — the pool's cross-process identity. Every pool
    # sharing a prefix shares the same machine-wide counters; a distinct
    # prefix is an independent pool with its own caps.
    name_prefix: str = "python"
    # Max simultaneously ALIVE subprocesses — not concurrently executing ones,
    # which is ``_max_running``. A slot is taken on a tool's first cell and
    # held until teardown, so it spans every LLM wait and every search in
    # between: the count tracks how many games are in play, not how much
    # Python is running. Set far above any real game concurrency, so the
    # solver's own cap is what gates a run. A cap sized near the game count
    # gates instead, and waiting here is unbounded — no deadline, no give-up,
    # and a slot frees only when some other game finishes.
    _max_kernels: int = 1024
    _max_running: int = 24  # max concurrently executing (includes startup)
    _exists_semaphore: Optional["SharedSemaphore"] = field(default=None, init=False, repr=False)
    _running_semaphore: Optional["SharedSemaphore"] = field(default=None, init=False, repr=False)

    @property
    def max_kernels(self) -> int:
        """How many kernels may exist at once."""
        return self._max_kernels

    @max_kernels.setter
    def max_kernels(self, value: int) -> None:
        """Set the cap. Only before first use: once the semaphore exists the name is
        already bound to a machine-wide counter that other processes may hold.
        """
        if self._exists_semaphore is not None:
            raise AttributeError("Cannot change max_kernels after semaphores have been initialized")
        self._max_kernels = value

    @property
    def max_running(self) -> int:
        """How many kernels may execute at once."""
        return self._max_running

    @max_running.setter
    def max_running(self, value: int) -> None:
        """Set the cap, under the same before-first-use rule as ``max_kernels``."""
        if self._running_semaphore is not None:
            raise AttributeError("Cannot change max_running after semaphores have been initialized")
        self._max_running = value

    def cpu_semaphore(self) -> "SharedSemaphore":
        """The permit for work that occupies a CPU, which is what
        ``max_running`` counts.

        Public because it is not only ours to hold: an owner that runs other
        CPU-bound work beside its kernels — a search in a child process, say —
        should contend on this counter rather than open a second pool sized
        against the same machine, or the two together can exceed it. Size the
        cap for everything that will hold it, and set it before the first
        cell runs (the setter refuses a later change).
        """
        self._ensure_semaphores()
        assert self._running_semaphore is not None
        return self._running_semaphore

    def _ensure_semaphores(self) -> None:
        """Create semaphores on first use.

        Names derive from ``name_prefix`` ("<prefix>_kernel_exists" /
        "<prefix>_kernel_running") — every KernelPool instance in every
        process sharing the same manager AND prefix resolves to the same
        primitives: the limits are machine-wide per prefix, not
        per-pool-instance."""
        from ullm.shared_semaphore import SharedSemaphore

        if self._exists_semaphore is None:
            self._exists_semaphore = SharedSemaphore(f"{self.name_prefix}_kernel_exists", self._max_kernels)
        if self._running_semaphore is None:
            self._running_semaphore = SharedSemaphore(f"{self.name_prefix}_kernel_running", self._max_running)

    def __deepcopy__(self, memo: dict[int, Any]) -> "KernelPool":
        """KernelPool is a shared resource — deepcopy returns self."""
        memo[id(self)] = self
        return self


# Module-level default pool (created lazily). Semaphore identity is the
# NAME, machine-wide — every process using the default participates in the
# same cap once multiprocess mode is enabled — so the pool object itself is
# just a plain config value; there is nothing live to wire through.
_default_kernel_pool: Optional[KernelPool] = None


def default_kernel_pool() -> KernelPool:
    """The default ``KernelPool`` used by tools constructed without one.
    Override caps early (before the first execute) via
    ``default_kernel_pool().max_kernels = ...``, or pass an explicit pool
    to the tool."""
    global _default_kernel_pool
    if _default_kernel_pool is None:
        _default_kernel_pool = KernelPool()
    return _default_kernel_pool


def _close_process_pipes(proc: "subprocess.Popen[bytes]") -> None:
    """Close a killed kernel's pipe file objects. Popen only closes them at
    __del__, so every killed kernel would otherwise leak three fds (and
    trip ResourceWarning under warnings-as-errors test runs)."""
    for f in (proc.stdin, proc.stdout, proc.stderr):
        if f is not None:
            try:
                f.close()
            except OSError:
                pass


class InitialCodeError(RuntimeError):
    """``initial_code`` failed (error or timeout) while bootstrapping a fresh
    kernel. Named so the failure is attributable at a glance — initial_code
    runs invisibly before the first user cell, which made its failures easy
    to misread as generic tool trouble."""


# Stand-in written into pickled tmpdir files where the dir's own absolute
# path appeared; rewritten to the new path at restore.
_PATH_PLACEHOLDER = b"__ULLM_TOOL_DIR_d41c8f26a9e3__"


def _kernel_home() -> str:
    """A scratch HOME for this user's kernel subprocesses, outside their tmpdirs.

    Created lazily under the system temp directory; never cleaned up (it holds
    only library caches, shared between this user's kernels by design).

    Per-uid and ``0700``, because the system temp directory is shared and
    model-written code runs with this HOME. A name without the uid is
    first-creator-wins: whoever makes it sets the mode, and every other user
    inherits a HOME they cannot write. The mode is equally load-bearing —
    these caches are read back by later kernels, so anything with write
    access to them can choose what a model's imports return.

    An existing entry is checked rather than trusted: a path under a
    world-writable directory can be planted, and a symlink there would put
    every kernel's cache somewhere of the planter's choosing. Refused loudly,
    since the alternative is running the model's code against it."""
    home = os.path.join(tempfile.gettempdir(), f"ullm-kernel-home-{os.getuid()}")
    try:
        os.mkdir(home, 0o700)
    except FileExistsError:
        info = os.lstat(home)
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"kernel HOME {home!r} exists and is not a directory") from None
        if info.st_uid != os.getuid():
            raise RuntimeError(f"kernel HOME {home!r} is owned by uid {info.st_uid}, not {os.getuid()}") from None
        # Tighten a directory left readable by an older build rather than
        # refusing it: it is ours, and the caches inside are the only thing
        # the mode protects.
        if info.st_mode & 0o077:
            os.chmod(home, 0o700)
    return home


def _tar_tmpdir_healing_paths(tmpdir: str) -> bytes:
    """Tar a tmpdir's contents; regular files that mention the dir's own
    absolute path get it swapped for ``_PATH_PLACEHOLDER``. Symlinks and
    non-regular entries are archived untouched."""
    old = tmpdir.rstrip("/").encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for root, dirs, files in os.walk(tmpdir):
            for name in dirs:
                path = os.path.join(root, name)
                tar.add(path, arcname=os.path.relpath(path, tmpdir), recursive=False)
            for name in files:
                path = os.path.join(root, name)
                arcname = os.path.relpath(path, tmpdir)
                if os.path.islink(path) or not os.path.isfile(path):
                    tar.add(path, arcname=arcname, recursive=False)
                    continue
                with open(path, "rb") as f:
                    data = f.read()
                if old in data:
                    data = data.replace(old, _PATH_PLACEHOLDER)
                info = tar.gettarinfo(path, arcname=arcname)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _heal_placeholders(tmpdir: str) -> None:
    """Rewrite ``_PATH_PLACEHOLDER`` occurrences in a restored tmpdir's
    regular files to the dir's new absolute path."""
    new = tmpdir.rstrip("/").encode()
    for root, _dirs, files in os.walk(tmpdir):
        for name in files:
            path = os.path.join(root, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                data = f.read()
            if _PATH_PLACEHOLDER in data:
                with open(path, "wb") as f:
                    f.write(data.replace(_PATH_PLACEHOLDER, new))


def _kill_kernel_process(proc: "subprocess.Popen[bytes]") -> None:
    """Kill + reap a kernel subprocess and close its pipes. NOT for call
    sites with a reader still blocked on the process's stdout — those must
    resolve the read between the reap and the close (see the timeout path
    in ``_execute``)."""
    try:
        proc.kill()
        proc.wait()
    except OSError:
        pass
    _close_process_pipes(proc)


#: The text a refused host call returns to the cell and logs to stderr. One
#: sentence, stable: a model that trips a refusal reads the same thing whatever
#: route it took, and downstream tests match on it.
HOST_CALL_REFUSED_NOTICE = "host call refused: the harness will not act on this frame"

#: Builtin container classes a restricted host-call payload may reconstruct.
#: None of them run code on unpickling; anything else — including builtins.exec
#: and any user-defined class with a __reduce__ / __setstate__ — is refused.
#: Each admitted class needs a real iterable to grow, so a large result needs a
#: correspondingly large pickle (bounded by transport size). ``bytearray`` is
#: deliberately absent: ``bytearray(N)`` allocates N bytes from a tiny
#: ``(bytearray, (N,))`` reducer, a compact parent-process allocation bomb with
#: no legitimate payload use here.
_HOST_PAYLOAD_SAFE_BUILTINS = frozenset({"list", "dict", "tuple", "set", "frozenset"})


class HostPayloadRefused(Exception):
    """A host-call payload asked to reconstruct something the restricted
    decoder does not permit. Recoverable: the cell gets a refusal, the game
    goes on."""


class _RestrictedUnpickler(pickle.Unpickler):
    """Decode a host-call payload without letting it run code.

    The payload is authored in the model's own cell — even a well-formed
    ``submit_action([...])`` pickles whatever object the model handed the
    adapter, and the model can reach the trusted primitive by object
    traversal. So the decoder must reconstruct plain data and nothing else: a
    payload whose ``__reduce__`` or ``__setstate__`` would execute is refused
    rather than run. Only the builtin containers in `_HOST_PAYLOAD_SAFE_BUILTINS`
    are constructible; every other class is a `HostPayloadRefused`.
    """

    def find_class(self, module: str, name: str) -> Any:
        """Admit a builtin container, refuse everything else."""
        if module == "builtins" and name in _HOST_PAYLOAD_SAFE_BUILTINS:
            return getattr(builtins, name)
        raise HostPayloadRefused(f"payload may not construct {module}.{name}")


@dataclass
class PythonTool(ullm.tool.Tool):
    """Executes Python code in a persistent REPL subprocess.

    Lifecycle (deliberate, four-tier):
      - __post_init__       creates the tmpdir; tmpdir belongs to the tool,
                            not the kernel, and lives until clear_state().
      - reset(keep_tmpdir_contents=True)  sync; kills the kernel subprocess
                            and (optionally) wipes tmpdir contents. Keeps the
                            tmpdir directory and the semaphore slot, so the
                            tool stays usable — the next call starts a fresh
                            kernel. This is the right hook for between-turn
                            recycling.
      - clear_state()       async (forced by parent); FINAL teardown — kills
                            the kernel, removes the tmpdir, releases the
                            semaphore slot. After this the tool asserts on use.
      - __del__             warns if clear_state() was never called and does
                            best-effort sync cleanup as a fallback.

    Note: the inherited `clear_state_on` ("run"/"turn"/"never") is awkward for
    this tool because clear_state means final teardown here — anything other
    than 'never' would destroy the tool. Use `reset()` for between-call /
    between-turn kernel recycling. Default is 'never'.

    Deepcopy and pickle preserve the on-disk tmpdir contents but not the
    in-memory kernel state. `kernel_pool` travels with both (it is a plain
    config value — names + limits; caps are shared machine-wide by
    semaphore name). Pickling is the snapshot path (A4): it FAILS LOUDLY
    while a kernel is alive — call reset() first; kernel contents are not
    state. Deepcopy of a warm tool is permitted (an in-process copy is not
    a snapshot); the copy starts kernel-less either way."""

    name: str = "python"
    description: str = "Stateful REPL kernel. State persists between executions. "
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        }
    )
    clear_state_on: Literal["run", "turn", "never"] = "never"

    # Configuration
    timeout: float = 10.0  # seconds before SIGINT for normal execute calls
    initial_code_timeout: float = (
        300.0  # seconds for the one-shot initial_code run (heavier: matplotlib font cache, slow shared disks)
    )
    interrupt_timeout: float = 5.0  # seconds after SIGINT before SIGKILL
    # Grace given to the kernel to CONSUME a host-call reply after the deadline
    # has already passed. Servicing a host call is never bounded, and delivering
    # its result is part of servicing: without this, a callback that overruns
    # the budget returns to remaining=0, so SIGINT lands inside the worker's
    # readline and the snippet never receives the value the callback produced.
    # Bounded ONCE per execution, not per call: a snippet looping on host calls
    # must not be able to buy a fresh window each iteration, which is the
    # deadline-dodge the wall-clock design exists to prevent.
    host_reply_grace: float = 2.0
    memory_limit_mb: int = 2048  # memory limit for the worker subprocess
    initial_code: Optional[str] = None  # code to execute on kernel startup; error raises RuntimeError
    prepend_code: Optional[str] = None  # code silently prepended to every user execution (e.g. module reloads)
    # Max characters returned from a single call; longer outputs are truncated
    # to truncate_head + marker + truncate_tail.
    max_output_chars: int = int(64 * 64 * 1.1)
    truncate_head: int = 600  # chars to keep from the start when truncating
    truncate_tail: int = 600  # chars to keep from the end when truncating
    nice: int = 10  # nice increment for the worker subprocess (0 = no change)
    # Drop network access for the worker by spawning it in a fresh user+network
    # namespace via ``unshare -rn``. Defaults to True: harmless if the LLM code
    # doesn't need network (the expected case), and blocks egress — including
    # credential exfil — if it does. Set to False on systems without usable
    # user namespaces, or when the worker legitimately needs the network.
    isolate_network: bool = True
    # Parent-side functions the kernel can invoke synchronously via the injected
    # ``host_call(name, *args, **kwargs)`` builtin. The function runs in THIS
    # (parent) process while the kernel blocks; its return value is pickled back.
    # If a host function raises, the kernel call is aborted and the exception is
    # treated as FATAL: it is stored on ``_pending_host_exception`` and re-raised
    # from ``_execute``, from where it propagates all the way out (TurnManager
    # does NOT swallow it — Processor.run re-raises). Callers running their own
    # catch-all layers can read ``pending_host_exception`` afterwards.
    # Not serialised: callables are runtime wiring the owner re-establishes.
    #
    # TRUST CONTRACT: anything registered here is MODEL-REACHABLE. The token
    # handshake stops executed code from forging host-call frames, but function
    # traversal (e.g. ``fn.__globals__``) reaches the trusted primitive past any
    # in-kernel wrapper, so wrapper-side argument validation protects nothing.
    # Host functions must validate their arguments HOST-SIDE and must not close
    # over secrets. For per-use bindings use ``bound()`` — swapping closures on
    # a long-lived tool risks a stale binding.
    host_functions: dict[str, Callable[..., Any]] = field(default_factory=dict)
    # Decode host-call payloads with a full ``pickle.loads`` instead of the
    # restricted decoder. Payloads are model-authored, so a full unpickle runs
    # model-chosen ``__reduce__`` in this process — hence off by default and
    # named ``trusted`` to make an owner choosing it say what it is. Only an
    # owner whose host functions genuinely need model-built OBJECTS (not plain
    # data) may set it, and only until a structural codec replaces the object
    # transfer; a data-only surface (the duck) must never set it.
    host_payload_trusted: bool = False
    # Shared pool; None = the module-level default_kernel_pool(). A plain
    # config value (names + limits): copies are equivalent, caps are shared
    # machine-wide by semaphore NAME.
    kernel_pool: Optional[KernelPool] = None
    # Status reporting: a host's "say what you are waiting on" hook (see
    # ullm.backend.StatusHook). The queue for a pool slot and the execution
    # itself are only distinguishable in here, so an owner that wants them
    # in its run report wires this. Process-local: dropped on pickle and
    # deepcopy, like host_functions, and re-wired by the new owner.
    status_hook: Optional["StatusHook"] = None

    # Internal state
    # One dict per completed _execute ({"t": wall start, "duration_s", "is_error",
    # "fresh_kernel"}), drained by pop_execution_records() — the hook hosts use
    # to attribute kernel activity (e.g. diagnostics events) without wrapping
    # the tool. Mirrors the pop_pending_warnings pattern.
    _execution_records: list[dict[str, Any]] = field(
        default_factory=lambda: list[dict[str, Any]](), init=False, repr=False
    )
    _pending_host_exception: Optional[BaseException] = field(
        default=None, init=False, repr=False
    )  # set when a host_call raised
    _host_call_token: Optional[str] = field(
        default=None, init=False, repr=False
    )  # per-kernel secret authorising trusted host_calls
    _process: Optional[subprocess.Popen] = field(default=None, init=False, repr=False)  # type: ignore[type-arg]
    _tmpdir: Optional[str] = field(default=None, init=False, repr=False)
    _owns_semaphore_slot: bool = field(default=False, init=False, repr=False)  # whether we hold an exists slot
    _cleared: bool = field(default=False, init=False, repr=False)  # True after clear_state — final teardown done

    def __post_init__(self) -> None:
        super().__post_init__()
        # tmpdir belongs to the tool instance — created here, lives across reset()s,
        # removed only by clear_state() (or best-effort by __del__).
        self._tmpdir = tempfile.mkdtemp(prefix="ullm_python_")

    @property
    def pool(self) -> KernelPool:
        """The pool in effect: the explicit ``kernel_pool`` or the module default."""
        return self.kernel_pool if self.kernel_pool is not None else default_kernel_pool()

    @property
    def is_fresh(self) -> bool:
        """True when the next execute will start a fresh kernel — first use,
        after ``reset()``, after restore from pickle/deepcopy, or after a
        kernel death. The signal agents key their bootstrap injection on."""
        return self._process is None or self._process.poll() is not None

    @property
    def pending_host_exception(self) -> Optional[BaseException]:
        """The exception a host function raised during the last execute, if
        any. ``_execute`` re-raises it (and TurnManager propagates it), so
        this read-only view exists for callers running their own catch-all
        layers around tool execution."""
        return self._pending_host_exception

    def _status_scope(self, label: str) -> contextlib.AbstractContextManager[None]:
        """Scope ``label`` as what this tool is doing, for whatever owner
        wired ``status_hook`` (no-op when none did)."""
        return contextlib.nullcontext() if self.status_hook is None else self.status_hook(label)

    def pop_execution_records(self) -> list[dict[str, Any]]:
        """Drain the per-execute records accumulated since the last drain
        (wall start ``t``, ``duration_s``, ``is_error``, ``fresh_kernel``).
        For hosts attributing kernel activity after the fact."""
        records = self._execution_records
        self._execution_records = []
        return records

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Execute code in the REPL subprocess, respecting pool semaphores."""
        assert not self._cleared, "PythonTool has been cleared and is no longer usable"
        import ullm.tool_helpers as _th

        # The repair policy also covers source changes made before execution.
        code = arguments["code"]
        if self.auto_correct_arguments:
            code = _th.fix_double_escapes_if_python_valid(code)
        if self.prepend_code is not None:
            code = self.prepend_code + "\n" + code
        self.pool._ensure_semaphores()
        fresh_kernel = self.is_fresh
        t_start, mono_start = time.time(), time.monotonic()

        # Acquire exists slot BEFORE running slot to avoid deadlock.
        # If we acquired running first, callers holding exists slots (already
        # started kernels) couldn't get running slots when those slots were
        # held by callers waiting for exists slots — classic lock inversion.
        #
        # Both acquisitions are queueing on a pool shared across every game on
        # the machine, and are reported apart from the execution itself: only
        # the waiter knows which slot it is queued for, and a host counting
        # the semaphores could not attribute the wait to a game.
        if not self._owns_semaphore_slot:
            with self._status_scope("waiting for kernel slot"):
                await self.pool._exists_semaphore.acquire()  # type: ignore[union-attr]
            self._owns_semaphore_slot = True

        # Acquire running slot (blocks if too many concurrent executions).
        # Explicit acquire/release rather than `async with`, so the label can
        # change exactly at the acquisition rather than around it.
        with self._status_scope("waiting for kernel slot"):
            await self.pool._running_semaphore.acquire()  # type: ignore[union-attr]
        try:
            with self._status_scope("running kernel"):
                # Ensure we have a live kernel
                await self._ensure_kernel()

                # Send code and read response with timeout escalation
                output_text, is_error = await self._send_and_read(code)
                if is_error:
                    output_text = self._restart_if_memory_exhausted(output_text)
        finally:
            self.pool._running_semaphore.release()  # type: ignore[union-attr]

        self.is_error = is_error
        self._execution_records.append(
            {
                "t": t_start,
                "duration_s": round(time.monotonic() - mono_start, 6),
                "is_error": is_error,
                "fresh_kernel": fresh_kernel,
            }
        )
        return ullm.conv.StringContent(text=output_text)

    def _restart_if_memory_exhausted(self, error: str) -> str:
        """Recycle an exhausted worker and append the model-visible notice."""
        if not kernel_memory_exhausted(error):
            return error
        # Keep files: the whale's level-persistent kernel stores its world
        # model in this directory. Only interpreter state is unsafe to retain.
        self._reset_impl(keep_tmpdir_contents=True)
        return f"{error.rstrip()}\n\n{KERNEL_MEMORY_RESTART_NOTICE}"

    def reset(self, keep_tmpdir_contents: bool = True) -> None:
        """Shut down the kernel subprocess (if running). Optionally wipe tmpdir
        contents. Keeps the tmpdir directory itself and the semaphore slot, so
        the tool remains usable — the next call will start a fresh kernel.

        This is the right hook for "between-prompt" or "between-turn" resets;
        clear_state() is the final teardown."""
        assert not self._cleared, "PythonTool has been cleared and is no longer usable"
        assert self.state != "running", "Cannot reset() while execution is in flight"
        self._reset_impl(keep_tmpdir_contents)

    def _reset_impl(self, keep_tmpdir_contents: bool) -> None:
        """reset() minus the Tool-state guard. Subclasses that recycle the
        kernel from INSIDE their own _execute (where Tool.state is
        "running" but no kernel operation is in flight) call this
        directly."""
        if self._process is not None:
            _kill_kernel_process(self._process)
            self._process = None
        if not keep_tmpdir_contents and self._tmpdir is not None and os.path.isdir(self._tmpdir):
            for entry in os.listdir(self._tmpdir):
                path = os.path.join(self._tmpdir, entry)
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    async def _ensure_kernel(self) -> None:
        """Start a kernel subprocess if one isn't already running."""
        # Check if existing process died unexpectedly — keep tmpdir for recovery
        if self._process is not None and self._process.poll() is not None:
            _close_process_pipes(self._process)
            self._process = None
            # Don't release kernel / delete tmpdir — reuse it for the new process

        if self._process is not None:
            return  # kernel already alive

        # Exists slot is already owned (acquired in _execute before running slot)
        assert self._owns_semaphore_slot
        # tmpdir is created in __post_init__ and lives the tool's lifetime
        assert self._tmpdir is not None

        # Launch worker with clean environment.
        # Temporarily bypass debugpy/pydevd's subprocess monkey-patch: it injects
        # debugger bootstrap code into child processes, which hangs or crashes
        # when the child has a sandboxed environment without access to debugpy.
        nice_val = self.nice
        cmd = [sys.executable, _WORKER_SCRIPT, str(self.memory_limit_mb)]
        if self.isolate_network and not _unshare_usable():
            # Running unisolated is acceptable ONLY on Kaggle (its container
            # disallows user namespaces and has no egress to protect anyway).
            # Anywhere else, silently dropping isolation would be a security
            # downgrade — fail loudly; opting out is an explicit config act.
            if not os.path.isdir("/kaggle/working"):
                raise RuntimeError(
                    "isolate_network=True but `unshare --user --net` is not usable on this "
                    "host; set isolate_network=False explicitly if running the kernel "
                    "without network isolation is acceptable here"
                )
            warnings.warn(
                "Kaggle container: unshare unavailable, kernel runs WITHOUT network isolation",
                stacklevel=2,
            )
        elif self.isolate_network:
            # --user: fresh user namespace so we can create the net ns without root
            # --net:  fresh network namespace (only has loopback, no egress)
            # Note: not using -r (uid map) — it requires uid_map write permission
            # that isn't always granted; --user alone is enough to unlock --net.
            cmd = ["unshare", "--user", "--net", *cmd]
        with _bypass_debugger_subprocess_patch():
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # HOME deliberately NOT the tmpdir: XDG-honoring libraries write
                # ~/.cache and friends, and pointing HOME at the working directory
                # plants system directories where the tool layer lists the model's
                # own files. A shared scratch home keeps those writes out of view;
                # caches shared between kernels are still just caches.
                # PYTHONDONTWRITEBYTECODE keeps __pycache__ out of the working
                # directory for the same reason.
                # PYTHONHASHSEED is pinned because kernel output must be
                # reproducible across processes: a replay re-executes the
                # model's code in a fresh kernel, and anything that iterates a
                # set of strings (e.g. printing one) otherwise renders in a
                # per-process random order.
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": _kernel_home(),
                    "LANG": "en_US.UTF-8",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                },
                cwd=self._tmpdir,
                preexec_fn=(lambda: os.nice(nice_val)) if nice_val else None,
            )
        _log(f"kernel started (pid={self._process.pid})", self.pool)

        # Hand the kernel a fresh secret token over stdin, before any cell runs.
        # Not via env/argv: code in the worker can read os.environ / sys.argv, and
        # the token is what distinguishes a trusted (bootstrap) host_call from one
        # the executed code issued. The worker reads it as its first stdin line and
        # keeps it out of the cell namespace.
        assert self._process.stdin is not None
        self._host_call_token = secrets.token_hex(16)
        config_frame = json.dumps({"config": {"host_call_token": self._host_call_token}}) + "\n"
        self._process.stdin.write(config_frame.encode())
        self._process.stdin.flush()

        # Run initial code if configured. This is where slow shared disks and
        # heavy import chains stall with nothing on screen, so always leave a
        # breadcrumb once it drags — not gated on `verbose`.
        if self.initial_code is not None:
            _log("running initial_code", self.pool)

            async def _nag() -> None:
                await asyncio.sleep(30.0)
                print(
                    f"[PythonTool] initial_code still running after 30s "
                    f"(timeout {self.initial_code_timeout}s) — heavy imports inside "
                    f"the isolated worker are the usual cause",
                    flush=True,
                )

            nag_task = asyncio.create_task(_nag())
            try:
                output_text, is_error = await self._send_and_read(
                    self.initial_code,
                    timeout=self.initial_code_timeout,
                )
            finally:
                nag_task.cancel()
            if is_error:
                raise InitialCodeError(f"initial_code failed:\n{output_text}")

    async def _send_and_read(self, code: str, timeout: Optional[float] = None) -> tuple[str, bool]:
        """Send code to worker and read response with SIGINT/SIGKILL escalation.
        Returns (formatted_output, is_error)."""
        assert self._process is not None and self._process.stdin is not None and self._process.stdout is not None

        effective_timeout = self.timeout if timeout is None else timeout

        pid = self._process.pid
        _log(f"run start (pid={pid}): {code[:80]!r}", self.pool)

        # Send the request
        request = json.dumps({"code": code}) + "\n"
        self._process.stdin.write(request.encode())
        self._process.stdin.flush()

        # Start a single readline in a thread
        # Read frames until the worker's terminal response. Host-call interludes
        # (the kernel invoking a parent-side function via host_call) are handled
        # inline and the loop continues. The timeout is a WALL-CLOCK deadline
        # over the whole call: the clock keeps running through host-call
        # interludes (so code that loops on host calls cannot dodge it by
        # resetting a per-frame window), and servicing a host call is never
        # bounded or interrupted once begun — enforcement fires only while
        # waiting on worker frames, via SIGINT-then-SIGKILL escalation once the
        # deadline has passed. Past that point host calls are refused rather
        # than serviced (below), so the escalation is bounded in total.
        loop = asyncio.get_event_loop()
        deadline = time.monotonic() + effective_timeout
        sigint_sent = False
        kill_deadline = 0.0  # absolute SIGKILL point, set when SIGINT goes out
        serviced_host_call = False  # last frame was a host call: its reply is in flight
        grace_deadline = deadline + self.host_reply_grace  # hard ceiling on that grace
        self._pending_host_exception = None
        while True:
            read_future = loop.run_in_executor(None, self._process.stdout.readline)

            # Phase 1: wait for the next frame within the remaining budget.
            # After SIGINT the interrupt grace is a single absolute window, not
            # a per-frame one: frames keep arriving while the cell unwinds, and
            # a fresh window per frame would put the SIGKILL out of reach.
            if sigint_sent:
                remaining = max(0.0, kill_deadline - time.monotonic())
            else:
                remaining = max(0.0, deadline - time.monotonic())
                if serviced_host_call:
                    # The reply we just wrote still has to be read. Enforcing a
                    # spent deadline here would interrupt the delivery rather
                    # than the snippet, losing a result the callback already
                    # produced. Costs nothing when the kernel is prompt, and is
                    # capped at grace_deadline so a host-call loop cannot renew
                    # it per iteration.
                    remaining = max(remaining, max(0.0, grace_deadline - time.monotonic()))
            done, _ = await asyncio.wait({read_future}, timeout=remaining)
            if not done and not sigint_sent:
                # Phase 2: send SIGINT, wait for graceful stop
                self._process.send_signal(signal.SIGINT)
                sigint_sent = True
                kill_deadline = time.monotonic() + self.interrupt_timeout
                done, _ = await asyncio.wait({read_future}, timeout=self.interrupt_timeout)

            if not done:
                # Phase 3: SIGKILL — force kill, keep tmpdir for recovery
                self._process.kill()
                self._process.wait()
                await read_future  # unblocks when pipe closes; must resolve before we close our end
                _close_process_pipes(self._process)
                self._process = None
                # Don't release kernel — tmpdir and semaphore slot are preserved
                # so _ensure_kernel can restart in the same tmpdir.
                _log(f"kernel killed after timeout (pid={pid})", self.pool)
                return (
                    f"Execution timed out after {effective_timeout}s and process was killed because SIGINT interrupt failed.",
                    True,
                )

            # Got a frame
            response_line = read_future.result()
            if not response_line:
                # Process died during execution — keep tmpdir for recovery
                self._process = None
                _log(f"kernel died unexpectedly (pid={pid})", self.pool)
                return "Kernel process died unexpectedly.", True

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as e:
                # Worker was alive (response_line is non-empty) but produced
                # a non-JSON line. Most often this means something wrote to
                # the worker's fd=1 outside execute_code's redirect_stdout
                # (e.g. a C-level print from a subprocess the user's code
                # spawned, or a library that bypasses sys.stdout). Drain
                # stderr non-blockingly so the worker's own stderr doesn't
                # hide the cause, then kill the kernel (protocol is
                # irrecoverable). _ensure_kernel will restart it fresh on
                # the next call.
                stderr_buf = b""
                try:
                    import fcntl

                    assert self._process is not None and self._process.stderr is not None
                    fd = self._process.stderr.fileno()
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                    stderr_buf = os.read(fd, 4096)
                except (BlockingIOError, OSError, Exception):
                    pass
                assert self._process is not None
                _kill_kernel_process(self._process)
                self._process = None
                _log(
                    f"worker non-JSON output (pid={pid}): "
                    f"stdout[:200]={response_line[:200]!r}, stderr[:200]={stderr_buf[:200]!r}",
                    self.pool,
                )
                return (
                    "Worker produced non-JSON output (protocol corrupted, "
                    "killing kernel).\n"
                    f"  stdout line ({len(response_line)} bytes): "
                    f"{response_line[:200]!r}{'...' if len(response_line) > 200 else ''}\n"
                    f"  stderr peek ({len(stderr_buf)} bytes): "
                    f"{stderr_buf[:200]!r}{'...' if len(stderr_buf) > 200 else ''}\n"
                    f"  json error: {e}",
                    True,
                )

            # Host-call interlude: run the parent-side function, reply to the
            # kernel, then keep reading for the eventual terminal response.
            # Past the deadline the call is refused instead: its callback acts
            # on the world (executing a game action, say), and the caller has
            # already been told this execution timed out. The refusal raises
            # _HostAborted in the cell, which user code cannot swallow, so a
            # cell that unwinds returns a terminal frame and the kernel
            # survives; one that does not is SIGKILLed at kill_deadline.
            if isinstance(response, dict) and "host_call" in response:
                if sigint_sent:
                    self._write_host_abort("execution timed out; host calls are no longer served")
                    continue
                await self._handle_host_call(response["host_call"])
                serviced_host_call = True
                continue
            serviced_host_call = False

            break

        output_text, is_error = self._format_response(
            response,
            timed_out=sigint_sent,
            timeout=effective_timeout,
        )
        _log(f"run done (pid={pid}): {output_text[:80]!r} error={is_error}", self.pool)
        # A host function raised mid-cell (the kernel was told to abort and has
        # since returned its terminal frame). Surface it as fatal — propagate it
        # rather than returning the cell's error text.
        if self._pending_host_exception is not None:
            raise self._pending_host_exception
        return output_text, is_error

    def _refuse_host_call(self, reason: str) -> None:
        """Refuse a model-authored host call: recoverable, and loud on stderr.

        A refusal is never expected traffic — a legitimate call always decodes
        and names a real function — so it goes to stderr (the job's stderr.log)
        where it is visible without parsing diagnostics. Systematic across every
        game is our own protocol drift; confined to a few is a cell reaching
        past the surface. Either way the game continues.
        """
        print(f"{HOST_CALL_REFUSED_NOTICE}: {reason}", file=sys.stderr, flush=True)
        self._write_host_abort(f"{HOST_CALL_REFUSED_NOTICE}: {reason}")

    def _write_host_abort(self, message: str) -> None:
        """Refuse a host call: the worker turns ``host_abort`` into
        ``_HostAborted`` in the cell. Recoverable by contract — it never
        touches ``_pending_host_exception``, which is reserved for the
        harness's own host calls failing."""
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write((json.dumps({"host_abort": message}) + "\n").encode())
        self._process.stdin.flush()

    async def _handle_host_call(self, call: dict[str, Any]) -> None:
        """Run a parent-side host function for the kernel and write the reply.

        ``call`` is ``{"name": str, "payload": b64(pickle((args, kwargs))), "token": str}``.

        The frame is model-authored: the payload rides out through the model's
        own cell, and the trusted primitive is reachable from that cell by
        object traversal, so nothing decoded here is trusted. Three layers, in
        order of what each rules out:

        - The token authenticates the CHANNEL. A stray or forged frame is
          refused; a valid token confers no trust on the contents.
        - The payload decodes through `_RestrictedUnpickler` (unless the owner
          set ``host_payload_trusted``), so a payload whose ``__reduce__``
          would execute is refused before it runs rather than executing in
          this process.
        - Everything derived from the frame — bad token, undecodable payload,
          unknown name — is refused with a recoverable ``host_abort``, logged
          to stderr, and never sets ``_pending_host_exception``.

        ``_pending_host_exception`` (fatal, re-raised as the cell unwinds) is
        reserved for a RESOLVED host function raising: that is our own code, and
        we want it surfaced rather than swallowed.
        """
        assert self._process is not None and self._process.stdin is not None
        token = call.get("token") if isinstance(call, dict) else None
        if self._host_call_token is None or token != self._host_call_token:
            self._refuse_host_call("frame carried no valid host-call token")
            return
        try:
            name = call["name"]
            raw = base64.b64decode(call["payload"])
            if self.host_payload_trusted:
                args, kwargs = pickle.loads(raw)
            else:
                args, kwargs = _RestrictedUnpickler(io.BytesIO(raw)).load()
            if not isinstance(args, tuple) or not isinstance(kwargs, dict):
                raise HostPayloadRefused("payload is not an (args, kwargs) pair")
        except BaseException as e:  # noqa: BLE001 — model-authored frame: refused, not escalated
            self._refuse_host_call(repr(e))
            return
        fn = self.host_functions.get(name) if isinstance(name, str) else None
        if fn is None:
            self._refuse_host_call(f"unknown host function {name!r}")
            return
        try:
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            reply = {"host_result": base64.b64encode(pickle.dumps(result)).decode("ascii")}
        except BaseException as e:  # noqa: BLE001 — any host failure is fatal, captured and re-raised later
            self._pending_host_exception = e
            reply = {"host_abort": repr(e)}
        self._process.stdin.write((json.dumps(reply) + "\n").encode())
        self._process.stdin.flush()

    def _format_response(
        self, response: dict[str, Any], timed_out: bool = False, timeout: Optional[float] = None
    ) -> tuple[str, bool]:
        """Format worker response into (output_text, is_error) matching Jupyter conventions.
        Truncates head+tail if the joined output exceeds `max_output_chars`, appending
        a 'tool_output_truncated' warning to `_pending_warnings` in that case."""
        parts = list[str]()

        # If we interrupted due to timeout, prepend a clear message
        if timed_out:
            effective_timeout = self.timeout if timeout is None else timeout
            parts.append(f"Execution timed out after {effective_timeout}s and was interrupted.")

        # Stdout first
        if response.get("stdout"):
            parts.append(response["stdout"].rstrip("\n"))

        # Stderr (informational, not treated as error by itself)
        if response.get("stderr"):
            parts.append(response["stderr"].rstrip("\n"))

        # Expression result
        if response.get("result") is not None:
            parts.append(response["result"])

        # Error/traceback
        is_error = False
        if response.get("error"):
            parts.append(response["error"].rstrip("\n"))
            is_error = True
        elif timed_out:
            # Timed-out responses are always errors even if the traceback is missing
            is_error = True

        text = "\n".join(parts) if parts else ""
        text = self._maybe_truncate_output(text, is_error=is_error)
        return text, is_error

    def _maybe_truncate_output(self, text: str, *, is_error: bool) -> str:
        """If `text` is longer than max_output_chars, replace its middle with
        a truncation marker and record a warning. Applies uniformly to both
        success and error output."""
        original_length = len(text)
        if original_length <= self.max_output_chars:
            return text
        head = text[: self.truncate_head]
        tail = text[-self.truncate_tail :] if self.truncate_tail > 0 else ""
        omitted = original_length - len(head) - len(tail)
        truncated = f"{head}\n... [truncated {omitted} chars of tool output] ...\n{tail}"
        self._pending_warnings.append(
            {
                "type": "tool_output_truncated",
                "tool_name": self.name,
                "original_length": original_length,
                "max_chars": self.max_output_chars,
                "is_error": is_error,
            }
        )
        return truncated

    async def _clear_state(self) -> None:
        """Final teardown: kill kernel, remove tmpdir, release semaphore slot.
        After this the instance is no longer usable — _execute will assert."""
        if self._cleared:
            return
        pid = self._process.pid if self._process is not None else None

        # Kill the kernel + wipe tmpdir contents (reset doesn't remove the dir)
        self.reset(keep_tmpdir_contents=False)

        # Remove the tmpdir directory itself
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

        # Release the exists-slot
        if self._owns_semaphore_slot:
            self.pool._exists_semaphore.release()  # type: ignore[union-attr]
            self._owns_semaphore_slot = False

        self._cleared = True
        _log(f"kernel shut down (pid={pid})", self.pool)

    def __del__(self) -> None:
        """Best-effort cleanup if clear_state() wasn't called.

        Warns loudly so callers fix their lifecycle, then attempts the same
        teardown clear_state would have done. Defensive against interpreter
        shutdown where modules may already be torn down."""
        if getattr(self, "_cleared", False):
            return
        # Warn — but be defensive: warnings module may already be GC'd
        try:
            warnings.warn(
                "PythonTool was garbage-collected without clear_state() being called; "
                "performing best-effort cleanup. Always call clear_state() explicitly.",
                ResourceWarning,
                stacklevel=2,
            )
        except Exception:
            pass
        # Kill the subprocess
        try:
            proc = getattr(self, "_process", None)
            if proc is not None:
                _kill_kernel_process(proc)
        except Exception:
            pass
        # Remove tmpdir
        try:
            tmpdir = getattr(self, "_tmpdir", None)
            if tmpdir is not None and shutil is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        # Release semaphore slot
        try:
            if getattr(self, "_owns_semaphore_slot", False):
                pool = getattr(self, "kernel_pool", None) or _default_kernel_pool
                if pool is not None and pool._exists_semaphore is not None:
                    pool._exists_semaphore.release()
        except Exception:
            pass

    # ---- copy / pickle ----
    #
    # Both deepcopy and pickle preserve the on-disk tmpdir contents but NOT
    # the in-memory kernel state. The restored instance gets:
    #   - a fresh tmpdir (different path) populated with the source's contents
    #   - _process = None (next call starts a fresh kernel; initial_code re-runs)
    #   - _owns_semaphore_slot = False (the copy must acquire its own slot)
    #   - _cleared = False
    #
    # Deepcopy/pickle is illegal while a call is mid-execution (state == "running");
    # idle kernels are fine.

    def __deepcopy__(self, memo: dict[int, Any]) -> "PythonTool":
        assert self.state != "running", "Cannot deepcopy PythonTool while execution is in flight"
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        # Copy fields, with overrides for transient state
        for k, v in self.__dict__.items():
            if k == "kernel_pool":
                # Shared resource — pass by reference (KernelPool.__deepcopy__ also returns self)
                new.__dict__[k] = v
                continue
            if k == "_tmpdir":
                continue  # set below from a fresh mkdtemp + copytree
            if k in (
                "_running_task",
                "_process",
                "_owns_semaphore_slot",
                "_cleared",
                "host_functions",
                "_pending_host_exception",
                "_host_call_token",
                "status_hook",
            ):
                continue  # reset below
            new.__dict__[k] = copy.deepcopy(v, memo)
        new.__dict__["_running_task"] = None  # a live task handle never travels
        new.__dict__["_process"] = None
        new.__dict__["_owns_semaphore_slot"] = False
        new.__dict__["_cleared"] = False
        # Host-call wiring is parent-process-local; the owner re-establishes it.
        new.__dict__["host_functions"] = {}
        new.__dict__["_pending_host_exception"] = None
        new.__dict__["_host_call_token"] = None  # regenerated when the next kernel starts
        new.__dict__["status_hook"] = None  # reporting is the new owner's to wire
        # Fresh tmpdir, populated from source
        new_tmpdir = tempfile.mkdtemp(prefix="ullm_python_")
        if self._tmpdir is not None and os.path.isdir(self._tmpdir):
            shutil.copytree(self._tmpdir, new_tmpdir, dirs_exist_ok=True, symlinks=True)
        new.__dict__["_tmpdir"] = new_tmpdir
        return new

    def __getstate__(self) -> dict[str, Any]:
        assert self.state != "running", "Cannot pickle PythonTool while execution is in flight"
        # A4: the pickle path IS the snapshot path — a live kernel here means
        # a cadence violation, and it must never yield a silent snapshot.
        assert self._process is None or self._process.poll() is not None, (
            "Cannot pickle PythonTool with a live kernel: kernel contents are not "
            "state (A4) — call reset() before pickling"
        )
        state = dict(self.__dict__)
        # Drop / reset transient runtime state
        state["_running_task"] = None  # a live task handle never travels (Tool.__getstate__)
        state["_process"] = None
        state["_owns_semaphore_slot"] = False
        state["_cleared"] = False
        # Host-call wiring is parent-process-local; the owner re-establishes it.
        state["host_functions"] = {}
        state["_pending_host_exception"] = None
        state["_host_call_token"] = None  # regenerated when the next kernel starts
        state["status_hook"] = None  # a live hook closes over the owning process; never travels
        # Replace tmpdir path with a tarball of its contents; regular files
        # mentioning the tmpdir's own absolute path get it swapped for a
        # placeholder (healed back to the new path at restore). Best-effort
        # mitigation only — A4's relative-path rule stands.
        tmpdir = state.pop("_tmpdir", None)
        if tmpdir is not None and os.path.isdir(tmpdir):
            state["_tmpdir_tar"] = _tar_tmpdir_healing_paths(tmpdir)
        else:
            state["_tmpdir_tar"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        tar_bytes = state.pop("_tmpdir_tar", None)
        self.__dict__.update(state)
        # Fresh tmpdir, populated from the tarball
        new_tmpdir = tempfile.mkdtemp(prefix="ullm_python_")
        if tar_bytes is not None:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
                # filter="data": strip absolute paths/links/devices — the tar
                # is our own tmpdir snapshot, and this is the 3.14 default.
                tar.extractall(new_tmpdir, filter="data")
            _heal_placeholders(new_tmpdir)
        self.__dict__["_tmpdir"] = new_tmpdir

    def linked_file_tools(
        self,
        *,
        allowed_files: Optional[list[str]] = None,
        allowed_suffixes: Optional[tuple[str, ...]] = None,
    ) -> tuple[Any, Any, Any]:
        """Read/Write/Edit file tools scoped to this tool's working
        directory — the one-tmpdir pattern (kernel + file tools working the
        same files, e.g. the whale's ``world_model.py`` setup). The scope is
        read off this tool at call time, so kernel resets and
        pickle-restores of the whole (kernel, file tools) graph never need
        path re-pointing."""
        import ullm.tools.file as _file

        kwargs: dict[str, Any] = {
            "allowed_files": list(allowed_files) if allowed_files is not None else [],
            "allowed_suffixes": allowed_suffixes,
            "linked_tool": self,
            "auto_correct_arguments": self.auto_correct_arguments,
        }
        return (
            _file.ReadFileTool(**kwargs),
            _file.WriteFileTool(**kwargs),
            _file.EditFileTool(**kwargs),
        )

    def bound(
        self, *, host_functions: dict[str, Callable[..., Any]], host_payload_trusted: bool = False
    ) -> "BoundPythonTool":
        """A per-use view over this tool with its own host-function binding.

        Build a fresh one for each turn-runner you construct and pass IT
        into the tool list; it
        shares this tool's kernel, tmpdir, and pool, but carries its own
        ``host_functions`` for exactly the calls made through it. This is
        the sanctioned alternative to swapping closures on a long-lived
        tool, where one forgotten re-swap leaves a stale binding silently
        validating against old state. The same trust contract applies:
        everything in the binding is model-reachable and must validate
        host-side."""
        return BoundPythonTool(target=self, host_functions=host_functions, host_payload_trusted=host_payload_trusted)


@dataclass
class BoundPythonTool(ullm.tool.Tool):
    """Per-use view over a ``PythonTool`` (see ``PythonTool.bound``). Not
    picklable by design: views are built per use and die with the turn
    manager they were handed to — snapshot the underlying tool instead."""

    target: PythonTool = field(kw_only=True)
    host_functions: dict[str, Callable[..., Any]] = field(default_factory=dict)
    host_payload_trusted: bool = False

    def __post_init__(self) -> None:
        # Mirror the LLM-facing metadata; execution state stays our own.
        self.name = self.target.name
        self.description = self.target.description
        self.parameters = self.target.parameters
        self.auto_correct_arguments = self.target.auto_correct_arguments
        self.clear_state_on = "never"  # lifecycle belongs to the target
        super().__post_init__()

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Execute against the target tool with this view's host functions in
        place, restoring the previous set afterwards — which is what keeps the
        binding per-use rather than mutated onto a long-lived tool.
        """
        previous = self.target.host_functions
        previous_trusted = self.target.host_payload_trusted
        self.target.host_functions = self.host_functions
        self.target.host_payload_trusted = self.host_payload_trusted
        try:
            content = await self.target._execute(arguments)
        finally:
            self.target.host_functions = previous
            self.target.host_payload_trusted = previous_trusted
        self.is_error = self.target.is_error
        self._pending_warnings.extend(self.target.pop_pending_warnings())
        return content

    def pop_execution_records(self) -> list[dict[str, Any]]:
        """Drain the TARGET's execution records — executions through any
        view land on the underlying tool."""
        return self.target.pop_execution_records()

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("BoundPythonTool is a per-use view and must not be pickled; snapshot the underlying PythonTool")

    def __deepcopy__(self, memo: dict[str, Any]) -> "BoundPythonTool":
        raise TypeError(
            "BoundPythonTool is a per-use view and must not be deepcopied; "
            "copy the underlying PythonTool and build a fresh view"
        )
