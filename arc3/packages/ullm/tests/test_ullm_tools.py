"""Unit tests for ullm's file tools, argument-healing helpers, and the
vendored shared semaphore (in-process mode)."""

import asyncio
import json
import pathlib
import pickle
from typing import Any, Callable, Sequence

import pytest
import ullm.backends.stub
import ullm.conv
import ullm.shared_semaphore
import ullm.tool
import ullm.tool_helpers
import ullm.tools.file
import ullm.tools.python
import ullm.utils


async def _run_tool(tool: ullm.tool.Tool, arguments: dict[str, Any]) -> tuple[str, bool]:
    tool.arguments = arguments
    if tool.state != "ready":
        tool.rearm()
    await tool.run()
    return tool.result.to_string(), tool.is_error


# --- file tools ---


async def test_file_tools_roundtrip_with_path_mapping(tmp_path: pathlib.Path):
    real = str(tmp_path / "actual_notes.txt")
    kwargs: dict[str, Any] = {"allowed_files": ["notes.txt"], "path_mapping": {"notes.txt": real}}
    write = ullm.tools.file.WriteFileTool(**kwargs)
    read = ullm.tools.file.ReadFileTool(**kwargs)
    edit = ullm.tools.file.EditFileTool(**kwargs)

    _, err = await _run_tool(write, {"path": "notes.txt", "content": "alpha beta gamma"})
    assert not err
    _, err = await _run_tool(edit, {"path": "notes.txt", "old_string": "beta", "new_string": "BETA"})
    assert not err
    text, err = await _run_tool(read, {"path": "notes.txt"})
    assert not err and "alpha BETA gamma" in text
    # The model-visible path never touches the real location's name.
    with open(real) as f:
        assert f.read() == "alpha BETA gamma"


async def test_file_tool_denies_unlisted_path(tmp_path: pathlib.Path):
    read = ullm.tools.file.ReadFileTool(allowed_files=["notes.txt"])
    text, err = await _run_tool(read, {"path": "/etc/passwd"})
    assert err and "notes.txt" in text  # denial names the allowed files


async def test_edit_rejects_ambiguous_match(tmp_path: pathlib.Path):
    real = str(tmp_path / "f.txt")
    with open(real, "w") as f:
        f.write("dup dup")
    edit = ullm.tools.file.EditFileTool(allowed_files=["f.txt"], path_mapping={"f.txt": real})
    _, err = await _run_tool(edit, {"path": "f.txt", "old_string": "dup", "new_string": "x"})
    assert err  # two occurrences: refuse rather than guess
    with open(real) as f:
        assert f.read() == "dup dup"  # untouched


async def test_write_heals_double_escaped_newlines(tmp_path: pathlib.Path):
    real = str(tmp_path / "f.txt")
    write = ullm.tools.file.WriteFileTool(allowed_files=["f.txt"], path_mapping={"f.txt": real})
    # Literal backslash-n with no real whitespace: the model double-escaped.
    _, err = await _run_tool(write, {"path": "f.txt", "content": "line1\\nline2"})
    assert not err
    with open(real) as f:
        assert f.read() == "line1\nline2"


async def test_single_argument_remap(tmp_path: pathlib.Path):
    real = str(tmp_path / "f.txt")
    with open(real, "w") as f:
        f.write("payload")
    read = ullm.tools.file.ReadFileTool(allowed_files=["f.txt"], path_mapping={"f.txt": real})
    # Model used the wrong key for the sole required property.
    text, err = await _run_tool(read, {"filename": "f.txt"})
    assert not err and "payload" in text


# --- tool_helpers ---


def test_heuristic_fix_double_escapes():
    fix = ullm.tool_helpers.heuristic_fix_double_escapes
    assert fix("a\\nb") == "a\nb"  # no real whitespace: unescape
    assert fix("a\\nb\nc") == "a\\nb\nc"  # real newline present: leave alone


def test_fix_double_escapes_if_python_valid():
    fix = ullm.tool_helpers.fix_double_escapes_if_python_valid
    broken = "x = 1\\ny = 2"  # SyntaxError as-is, valid unescaped
    assert fix(broken) == "x = 1\ny = 2"
    valid = 'text = "a\\nb"'  # already-valid python: must NOT be touched
    assert fix(valid) == valid


# --- shared semaphore (in-process mode) ---


async def test_shared_semaphore_local_limit(monkeypatch: pytest.MonkeyPatch):
    """Without the multiprocess manager, a SharedSemaphore degrades to a
    plain per-instance asyncio semaphore that still enforces its limit.
    Force local mode: another test in this worker (e.g. a subprocess-mode
    LLMAgentSolver run) may have left the manager enabled."""
    ss = ullm.shared_semaphore
    monkeypatch.setattr(ss, "_manager", None)
    monkeypatch.setattr(ss, "_manager_is_server", False)
    monkeypatch.delenv(ss._ENV_ADDR, raising=False)
    sem = ullm.shared_semaphore.SharedSemaphore("test-sem-local", 1)
    await sem.acquire()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(sem.acquire(), timeout=0.2)
    sem.release()
    await asyncio.wait_for(sem.acquire(), timeout=1.0)
    sem.release()


async def test_shared_semaphore_multiprocess_name_resolution():
    """With the manager enabled, two handles carrying the same name resolve
    to one machine-wide counter (checked via held() through the *other*
    handle, which keeps this test about name resolution alone)."""
    ss = ullm.shared_semaphore
    ss.enable_multiprocess_semaphores()
    try:
        sem_a = ss.SharedSemaphore("test-sem-shared", 2)
        sem_b = ss.SharedSemaphore("test-sem-shared", 2)
        other = ss.SharedSemaphore("test-sem-other", 2)
        await sem_a.acquire()
        assert sem_b._resolve().held() == 1  # visible through the sibling handle
        assert other._resolve().held() == 0  # different name: independent counter
        # A pickled handle reattaches to the same counter by name.
        restored = pickle.loads(pickle.dumps(sem_a))
        assert restored._resolve().held() == 1
        sem_a.release()
        assert sem_b._resolve().held() == 0
    finally:
        ss.disable_multiprocess_semaphores()


async def test_shared_semaphore_cancelled_waiter_leaks_no_permit():
    """Cancelling a queued acquirer must not consume a machine-wide permit.

    The caps this guards (the kernel pool's, the whale's search cap) are
    held for the life of the manager, so one stranded permit is a
    permanent loss of capacity — and cancellation while queued is
    routine: it is what a per-game runtime limit or a soft-end does to
    whichever game is waiting for a slot.
    """
    ss = ullm.shared_semaphore
    ss.enable_multiprocess_semaphores()
    try:
        sem = ss.SharedSemaphore("test-sem-cancel", 1)
        await sem.acquire()  # holder takes the only permit
        waiter = asyncio.create_task(sem.acquire())
        await asyncio.sleep(0.2)  # let it queue on the manager
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        sem.release()  # the holder gives its permit back
        await asyncio.sleep(0.3)  # any in-flight attempt resolves and hands back
        assert sem._resolve().held() == 0
        # The permit is genuinely reusable, not just reported free.
        await asyncio.wait_for(sem.acquire(), timeout=2.0)
        sem.release()
    finally:
        ss.disable_multiprocess_semaphores()


def test_shared_semaphore_pickles_by_name():
    sem = ullm.shared_semaphore.SharedSemaphore("test-sem-pickle", 3)
    restored = pickle.loads(pickle.dumps(sem))
    assert (restored.name, restored.limit) == ("test-sem-pickle", 3)


# --- PythonTool cells: what an earlier cell's code reads back as ------------


async def test_an_earlier_cells_function_keeps_its_own_source(kernel_pool: ullm.tools.python.KernelPool):
    """Each cell compiles under a name of its own, so a function defined in one
    cell reads its own lines from every later one: ``inspect.getsource``
    returns it, and a traceback through it quotes its line rather than
    whatever sits at that line number in the cell that called it. The name
    shown is still ``<repl>``."""
    tool = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, timeout=60.0)
    try:
        await _run_tool(tool, {"code": "def boom():\n    raise ValueError('kept')\n"})
        out, err = await _run_tool(tool, {"code": "import inspect\nprint(inspect.getsource(boom))"})
        assert not err and "def boom():\n    raise ValueError('kept')" in out, out
        out, err = await _run_tool(tool, {"code": "x = 1\ny = 2\nboom()\n"})
        assert err, out
        assert "in boom\n    raise ValueError('kept')" in out, out
        assert 'File "<repl>"' in out and "<repl-" not in out, out
    finally:
        await tool.clear_state()


@pytest.mark.parametrize("stop", ["raise SystemExit", "exit()", "import sys\nsys.exit(3)"])
async def test_a_cell_that_exits_ends_itself_not_the_kernel(kernel_pool: ullm.tools.python.KernelPool, stop: str):
    """Stopping a cell with ``raise SystemExit``, ``exit()`` or ``sys.exit()``
    ends that cell, as in a notebook: what it printed comes back, marked as an
    error, and the session -- its variables -- is there for the next cell."""
    tool = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, timeout=60.0)
    try:
        out, err = await _run_tool(tool, {"code": f"kept = 7\nprint('before')\n{stop}\nprint('after')\n"})
        assert err and "before" in out and "after" not in out and "SystemExit" in out, out
        assert not tool.is_fresh, out
        out, err = await _run_tool(tool, {"code": "print(kept)"})
        assert not err and out.strip() == "7", out
    finally:
        await tool.clear_state()


async def test_a_cells_data_and_source_pass_through_verbatim(kernel_pool: ullm.tools.python.KernelPool):
    """A cell's file is shown as ``<repl>`` only by the formatters that name a
    location in it: a traceback's frame header, a warning's location, a
    SyntaxError's file. Output is never rewritten, so a string that looks like
    one of those is the model's own data and reaches it unchanged -- in what
    it prints, in ``inspect.getsource``, and in a source line a traceback
    quotes."""
    look_alike = 'File "<repl-7>", line 2 <repl-7>:2: file "<repl-7>"'
    tool = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, timeout=60.0)
    try:
        await _run_tool(tool, {"code": f"def label():\n    return {look_alike!r}\n"})
        out, err = await _run_tool(tool, {"code": "import inspect\nprint(inspect.getsource(label))\nprint(label())"})
        assert not err and out.count(look_alike) == 2, out
        await _run_tool(tool, {"code": f"def boom():\n    raise ValueError({look_alike!r})\n"})
        out, err = await _run_tool(tool, {"code": "boom()"})
        assert err and 'File "<repl>", line 2, in boom' in out and f"raise ValueError({look_alike!r})" in out, out
        out, _ = await _run_tool(tool, {"code": "import warnings\nwarnings.warn('careful')"})
        assert "<repl>:2: UserWarning: careful\n  warnings.warn('careful')" in out, out
        out, err = await _run_tool(tool, {"code": "return 1"})
        assert err and 'File "<repl>", line 1' in out and "<repl-" not in out, out
    finally:
        await tool.clear_state()


# --- PythonTool host calls: the kernel invoking parent-side functions -----
# This is the mechanism agent runtimes build their in-kernel APIs on (the
# whale's `action(...)`), so both the happy path and the "LLM-written code
# cannot forge one" property matter.

_BOOTSTRAP = """
_hc = __host_call_trusted__
del __host_call_trusted__
def remember(value):
    return _hc("remember", value)
"""


async def test_host_call_round_trip(kernel_pool: ullm.tools.python.KernelPool):
    """Bootstrap captures the trusted primitive; a cell calls it; the parent
    function runs in THIS process and its return value comes back."""
    seen = list[str]()

    def remember(value: str) -> str:
        seen.append(value)
        return f"stored:{value}"

    tool = ullm.tools.python.PythonTool(
        kernel_pool=kernel_pool,
        initial_code=_BOOTSTRAP,
        host_functions={"remember": remember},
        timeout=60.0,
    )
    try:
        out, err = await _run_tool(tool, {"code": "print(remember('alpha'))"})
        assert not err, out
        assert seen == ["alpha"]  # ran parent-side
        assert "stored:alpha" in out  # return value crossed back
    finally:
        await tool.clear_state()


async def test_host_function_failure_is_fatal_not_a_tool_error(kernel_pool: ullm.tools.python.KernelPool):
    """A parent-side host function that raises must NOT degrade into an
    ordinary LLM-visible tool error: the harness's own callback failing
    (e.g. an agent's repair hook) has to surface to the caller."""

    def explode(_value: str) -> str:
        raise RuntimeError("host side blew up")

    tool = ullm.tools.python.PythonTool(
        kernel_pool=kernel_pool,
        initial_code=_BOOTSTRAP,
        host_functions={"remember": explode},
        timeout=60.0,
    )
    try:
        with pytest.raises(RuntimeError, match="host side blew up"):
            await _run_tool(tool, {"code": "remember('boom')"})
    finally:
        await tool.clear_state()


async def test_unknown_trusted_host_target_is_recoverable(kernel_pool: ullm.tools.python.KernelPool):
    """Model traversal can reach the primitive, so an unknown target is input, not a harness failure."""
    tool = ullm.tools.python.PythonTool(
        kernel_pool=kernel_pool,
        initial_code=(
            "_hc = __host_call_trusted__\ndel __host_call_trusted__\ndef probe():\n    return _hc('not_registered')"
        ),
        timeout=60.0,
    )
    try:
        out, err = await _run_tool(
            tool,
            {"code": "try:\n    probe()\nexcept BaseException as exc:\n    print(type(exc).__name__, exc)"},
        )
        assert not err, out
        assert "unknown host function" in out
        assert tool.pending_host_exception is None
    finally:
        await tool.clear_state()


async def test_forged_host_call_token_is_refused_through_the_shared_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A frame whose token does not match this kernel's secret came from the
    executed cell, not the trusted primitive. It must be refused on the SAME
    path as every other frame-derived failure: a recoverable ``host_abort``
    carrying the stable notice, a stderr line, and no fatal host exception —
    so a forged frame is never mistaken for the harness's own code failing."""

    class _FakeStdin:
        """Captures the abort frame the parent writes back to the worker."""

        def __init__(self) -> None:
            self.written = bytearray()

        def write(self, data: bytes) -> None:
            self.written += data

        def flush(self) -> None:
            pass

    class _FakeProc:
        """Minimal stand-in: `_handle_host_call` only needs `stdin`."""

        def __init__(self) -> None:
            self.stdin = _FakeStdin()

    tool = ullm.tools.python.PythonTool()
    tool._process = _FakeProc()  # type: ignore[assignment]
    tool._host_call_token = "the-real-secret"

    await tool._handle_host_call({"name": "remember", "payload": "", "token": "forged"})

    assert tool.pending_host_exception is None  # forged frame is never fatal
    assert ullm.tools.python.HOST_CALL_REFUSED_NOTICE in capsys.readouterr().err  # logged to stderr
    frame = json.loads(bytes(tool._process.stdin.written).decode())  # type: ignore[attr-defined]
    assert ullm.tools.python.HOST_CALL_REFUSED_NOTICE in frame["host_abort"]  # stable notice to the cell


def _restricted_decode(obj: object) -> object:
    """Round-trip ``obj`` through a full pickle and the restricted decoder, the
    exact path a non-trusted host-call payload takes on the parent side."""
    import io

    return ullm.tools.python._RestrictedUnpickler(io.BytesIO(pickle.dumps(obj))).load()


def test_restricted_decoder_admits_plain_container_data() -> None:
    """The data a data-only agent actually sends — nested builtin containers —
    survives the restricted decode unchanged; refusing it would break the Duck."""
    payload = (["UP", "DOWN"], {"note": "text", "counts": (1, 2, 3)}, {"a", "b"})
    assert _restricted_decode(payload) == payload


def test_restricted_decoder_refuses_a_bytearray_allocation_bomb() -> None:
    """``bytearray(N)`` allocates N bytes parent-side from a tiny reducer, so
    ``bytearray`` is not on the allowlist: a payload naming it is refused before
    the constructor runs, not sized down. Guards the removal of that entry."""

    class _Bomb:
        """Its reducer asks for a 4 GiB bytearray; the decode must refuse the
        request rather than ever calling ``bytearray(1 << 32)``."""

        def __reduce__(self) -> tuple[type, tuple[int]]:
            return (bytearray, (1 << 32,))

    with pytest.raises(ullm.tools.python.HostPayloadRefused, match="builtins.bytearray"):
        _restricted_decode(_Bomb())


def test_restricted_decoder_refuses_an_executing_reducer() -> None:
    """The A1 vector at the unit level: a payload whose reducer would run
    ``builtins.exec`` is refused at ``find_class`` before any code executes."""

    class _Exec:
        """Reducer resolves ``exec`` and would run arbitrary source parent-side."""

        def __reduce__(self) -> tuple[Any, tuple[str]]:
            return (exec, ("pass",))

    with pytest.raises(ullm.tools.python.HostPayloadRefused, match="builtins.exec"):
        _restricted_decode(_Exec())


# --- U9 surface: default pool, freshness, snapshot discipline ----------


async def test_default_pool_and_is_fresh_lifecycle():
    tool = ullm.tools.python.PythonTool(timeout=60.0)  # no pool: module default
    try:
        assert tool.pool is ullm.tools.python.default_kernel_pool()
        assert tool.is_fresh
        out, err = await _run_tool(tool, {"code": "x = 21; print(x * 2)"})
        assert not err and "42" in out
        assert not tool.is_fresh  # kernel warm now
        out, err = await _run_tool(tool, {"code": "print(x)"})
        assert not err and "21" in out  # state persisted across calls
        tool.reset()
        assert tool.is_fresh  # the bootstrap-injection signal
    finally:
        await tool.clear_state()


async def test_execution_records_drain(kernel_pool: ullm.tools.python.KernelPool):
    # The host-attribution hook (platypus kernel/exec events): one record
    # per execute, drained once, marking which call started a kernel.
    tool = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, timeout=60.0)
    try:
        await _run_tool(tool, {"code": "print('a')"})
        await _run_tool(tool, {"code": "this is not python"})
        records = tool.pop_execution_records()
        assert [r["fresh_kernel"] for r in records] == [True, False]
        assert [r["is_error"] for r in records] == [False, True]
        assert all(r["duration_s"] > 0 and r["t"] > 0 for r in records)
        assert tool.pop_execution_records() == []  # drained
    finally:
        await tool.clear_state()


async def test_memory_exhaustion_restarts_kernel_but_preserves_files(
    kernel_pool: ullm.tools.python.KernelPool,
) -> None:
    """A hard-cap failure clears poisoned interpreter state, not workspace files."""
    tool = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, memory_limit_mb=256, timeout=60.0)
    try:
        out, err = await _run_tool(
            tool,
            {"code": ("open('survives.txt', 'w').write('yes')\ncarried = 1234\njunk = [None] * 300_000_000")},
        )
        assert err and "MemoryError" in out
        assert "kernel was restarted" in out
        assert tool.is_fresh

        out, err = await _run_tool(
            tool,
            {
                "code": (
                    "import os\n"
                    "print('file', open('survives.txt').read())\n"
                    "try:\n"
                    "    print(carried)\n"
                    "except NameError:\n"
                    "    print('variable cleared')"
                )
            },
        )
        assert not err and "file yes" in out and "variable cleared" in out
    finally:
        await tool.clear_state()


async def test_live_kernel_pickle_fails_loudly(kernel_pool: ullm.tools.python.KernelPool):
    # A4: the pickle path is the snapshot path — a warm kernel must never
    # yield a silent snapshot.
    tool = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, timeout=60.0)
    try:
        await _run_tool(tool, {"code": "pass"})
        with pytest.raises(AssertionError, match="live kernel"):
            pickle.dumps(tool)
        tool.reset()
        pickle.dumps(tool)  # idle tool: snapshots fine
    finally:
        await tool.clear_state()


async def test_pickle_roundtrip_heals_absolute_paths(kernel_pool: ullm.tools.python.KernelPool):
    # A relative-path rule violation (absolute tmpdir path stored inside a
    # file) is healed best-effort across the pickle boundary: the restored
    # file points at the NEW tmpdir.
    tool = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, timeout=60.0)
    try:
        _, err = await _run_tool(tool, {"code": "import os\nopen('ptr.txt', 'w').write(os.getcwd() + '/data.bin')"})
        assert not err
        tool.reset()
        restored = pickle.loads(pickle.dumps(tool))
        try:
            out, err = await _run_tool(
                restored,
                {"code": "import os\nprint(open('ptr.txt').read() == os.getcwd() + '/data.bin')"},
            )
            assert not err and "True" in out
        finally:
            await restored.clear_state()
    finally:
        await tool.clear_state()


async def test_bound_views_carry_their_own_host_functions(kernel_pool: ullm.tools.python.KernelPool):
    # Per-use views: same kernel, per-view host functions, target untouched
    # — the structural fix for the stale-swap failure mode.
    calls = list[tuple[str, str]]()

    def make(tag: str) -> Callable[[Any], str]:
        def fn(value: Any) -> str:
            calls.append((tag, value))
            return f"{tag}:{value}"

        return fn

    tool = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, initial_code=_BOOTSTRAP, timeout=60.0)
    try:
        view1 = tool.bound(host_functions={"remember": make("move1")})
        out, err = await _run_tool(view1, {"code": "print(remember('a'))"})
        assert not err and "move1:a" in out
        view2 = tool.bound(host_functions={"remember": make("move2")})
        out, err = await _run_tool(view2, {"code": "print(remember('b'))"})
        assert not err and "move2:b" in out
        assert calls == [("move1", "a"), ("move2", "b")]
        assert tool.host_functions == {}  # the long-lived tool holds no binding
        with pytest.raises(TypeError, match="per-use view"):
            pickle.dumps(view2)
    finally:
        await tool.clear_state()


async def test_host_call_loop_counts_against_the_deadline(kernel_pool: ullm.tools.python.KernelPool):
    """The timeout is a wall-clock deadline over the whole call: code that
    loops on host calls cannot dodge it by resetting a per-frame window —
    each frame's wait uses the REMAINING budget."""
    import time as _time

    def ping() -> str:
        return "pong"

    tool = ullm.tools.python.PythonTool(
        kernel_pool=kernel_pool,
        initial_code="_hc = __host_call_trusted__\ndel __host_call_trusted__\ndef ping():\n    return _hc('ping')",
        host_functions={"ping": ping},
        timeout=2.0,
    )
    try:
        start = _time.monotonic()
        out, err = await _run_tool(tool, {"code": "while True:\n    ping()"})
        elapsed = _time.monotonic() - start
        assert err, out
        assert "timed out" in out.lower(), out
        # ~timeout (+ SIGINT grace), nowhere near unbounded.
        assert elapsed < 2.0 + tool.interrupt_timeout + 5.0, elapsed
    finally:
        await tool.clear_state()


async def test_host_calls_are_refused_after_the_deadline(kernel_pool: ullm.tools.python.KernelPool):
    """A cell that swallows the interrupt and keeps calling host functions must
    still terminate, and must not get its callbacks run past the deadline: a
    host call acts on the world, and the caller has already been told the
    execution timed out."""
    import asyncio as _asyncio
    import time as _time

    served = list[float]()

    def ping() -> str:
        served.append(_time.monotonic())
        return "pong"

    tool = ullm.tools.python.PythonTool(
        kernel_pool=kernel_pool,
        initial_code="_hc = __host_call_trusted__\ndel __host_call_trusted__\ndef ping():\n    return _hc('ping')",
        host_functions={"ping": ping},
        timeout=2.0,
    )
    try:
        start = _time.monotonic()
        # `except BaseException` swallows both the SIGINT and the refusal, so
        # nothing but the harness can end this cell.
        code = "while True:\n    try:\n        ping()\n    except BaseException:\n        pass"
        # Bounded await: a regression here is unbounded, and must fail rather
        # than hang the suite.
        out, err = await _asyncio.wait_for(_run_tool(tool, {"code": code}), timeout=60.0)
        elapsed = _time.monotonic() - start
        assert err and "timed out" in out.lower(), out
        assert elapsed < 2.0 + tool.interrupt_timeout + 5.0, elapsed
        # Callbacks stop well before the kill: the refusal is what ends them,
        # so the last one lands near the deadline rather than running the
        # whole execution out. The margin is loose on purpose — this is a
        # wall-clock proxy for an ordering property, and a loaded runner can
        # slip the event loop by seconds. A regression that keeps serving
        # calls shows up here as a last call near `elapsed`, not near the
        # deadline.
        assert served, "the callback never ran at all"
        assert served[-1] - start < 2.0 + tool.host_reply_grace + 5.0, served[-1] - start
    finally:
        await tool.clear_state()


async def test_host_callback_is_never_interrupted(kernel_pool: ullm.tools.python.KernelPool):
    """The deadline keeps counting during a host callback, but the callback
    itself must ALWAYS run to completion — enforcement only fires while
    waiting on worker frames. A callback that overruns the entire budget
    finishes; the snippet is interrupted afterwards."""
    import asyncio as _asyncio

    finished = list[str]()

    async def slow() -> str:
        await _asyncio.sleep(3.0)  # 1.5x the whole budget
        finished.append("done")
        return "slow-result"

    tool = ullm.tools.python.PythonTool(
        kernel_pool=kernel_pool,
        initial_code="_hc = __host_call_trusted__\ndel __host_call_trusted__\ndef slow():\n    return _hc('slow')",
        host_functions={"slow": slow},
        timeout=2.0,
    )
    try:
        out, err = await _run_tool(tool, {"code": "r = slow()\nprint('got', r)\nwhile True:\n    pass"})
        assert finished == ["done"]  # the callback ran to completion, uninterrupted
        assert err and "timed out" in out.lower(), out  # the snippet then hit the deadline
        assert "got slow-result" in out, out  # and had received the full result first
    finally:
        await tool.clear_state()


async def test_initial_code_failure_is_named(kernel_pool: ullm.tools.python.KernelPool):
    tool = ullm.tools.python.PythonTool(
        kernel_pool=kernel_pool, initial_code="raise ValueError('bootstrap boom')", timeout=60.0
    )
    try:
        tool.arguments = {"code": "pass"}
        with pytest.raises(ullm.tools.python.InitialCodeError, match="bootstrap boom"):
            await tool.run()
    finally:
        await tool.clear_state()


async def test_host_exception_propagates_through_turn_manager(kernel_pool: ullm.tools.python.KernelPool):
    # Regression pin for the stale "TurnManager swallows tool exceptions"
    # belief: a host function raising must surface from turn_manager.run(),
    # not degrade into an LLM-visible tool error.
    def explode(_value: Any) -> None:
        raise RuntimeError("host side blew up")

    backend = ullm.backends.stub.StubBackend.from_script(
        ullm.conv.LLMResponse(
            final_content=ullm.conv.ToolCallContent(tool_name="python", arguments={"code": "remember('x')"})
        ),
        "never reached",
    )
    tool = ullm.tools.python.PythonTool(
        kernel_pool=kernel_pool,
        initial_code=_BOOTSTRAP,
        host_functions={"remember": explode},
        timeout=60.0,
    )
    try:
        tm = ullm.utils.make_turn_manager(backend, tools=[tool], temperature=0.0, max_tokens=100)
        tm.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
        with pytest.raises(RuntimeError, match="host side blew up"):
            await tm.run()
        assert tool.pending_host_exception is not None  # public read for catch-all layers
    finally:
        await tool.clear_state()


# --- file tools: linked-tool auto-scoping ---------------------------------


async def test_file_tools_follow_linked_tool_directory(kernel_pool: ullm.tools.python.KernelPool):
    # The link IS the directory scope, read at resolve time: after a pickle
    # roundtrip of the (python tool, file tool) graph the file tool writes
    # into the NEW tmpdir with zero re-pointing — the property that kills
    # manual path_mapping surgery.
    py = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, timeout=60.0)
    write = ullm.tools.file.WriteFileTool(allowed_files=["notes.py"], linked_tool=py)
    try:
        _, err = await _run_tool(write, {"path": "notes.py", "content": "VALUE = 7"})
        assert not err
        out, err = await _run_tool(py, {"code": "import notes; print(notes.VALUE)"})
        assert not err and "7" in out

        py.reset()
        py2, write2 = pickle.loads(pickle.dumps((py, write)))
        try:
            assert write2.linked_tool is py2  # one graph: link survives by identity
            # Different LENGTH on purpose: the tmpdir snapshot carries the old
            # __pycache__, and a same-size same-mtime-second rewrite would
            # validate the stale pyc (Python's (mtime, size) check).
            _, err = await _run_tool(write2, {"path": "notes.py", "content": "VALUE = 8888"})
            assert not err
            out, err = await _run_tool(py2, {"code": "import notes; print(notes.VALUE)"})
            assert not err and "8888" in out
        finally:
            await py2.clear_state()
    finally:
        await py.clear_state()


async def test_whale_pattern_one_tmpdir_quartet(kernel_pool: ullm.tools.python.KernelPool):
    # The whale's setup: kernel + read/write/edit file tools all working
    # the same file in ONE directory. Written by a file tool, executed by
    # the kernel, edited, re-executed — then the whole quartet pickled in
    # one graph and still coherent in the new directory, zero re-pointing.
    py = ullm.tools.python.PythonTool(kernel_pool=kernel_pool, timeout=60.0)
    read, write, edit = py.linked_file_tools(allowed_files=["model.py"], allowed_suffixes=(".py",))
    try:
        _, err = await _run_tool(write, {"path": "model.py", "content": "def f():\n    return 10\n"})
        assert not err
        out, err = await _run_tool(py, {"code": "import model; print(model.f())"})
        assert not err and "10" in out
        # Different LENGTH on purpose, here and below: Python validates pycs
        # by (mtime, size), so a same-size edit within the same second would
        # execute the stale __pycache__ entry. Real hazard for edit-then-
        # import loops — the whale sidestepped it by exec'ing source.
        _, err = await _run_tool(edit, {"path": "model.py", "old_string": "return 10", "new_string": "return 2500"})
        assert not err
        # Fresh kernel so the edit is what executes (module already imported).
        py.reset()
        out, err = await _run_tool(py, {"code": "import model; print(model.f())"})
        assert not err and "2500" in out

        py.reset()
        py2, read2, write2, edit2 = pickle.loads(pickle.dumps((py, read, write, edit)))
        try:
            assert read2.linked_tool is py2 and write2.linked_tool is py2 and edit2.linked_tool is py2
            text, err = await _run_tool(read2, {"path": "model.py"})
            assert not err and "return 2500" in text
            _, err = await _run_tool(edit2, {"path": "model.py", "old_string": "return 2500", "new_string": "return 9"})
            assert not err
            out, err = await _run_tool(py2, {"code": "import model; print(model.f())"})
            assert not err and "9" in out
        finally:
            await py2.clear_state()
    finally:
        await py.clear_state()


# --- cross-process pool caps (I2b) ----------------------------------------


def _xproc_probe(q: Any) -> None:
    """Spawned-child probe: resolve the same-named semaphore and report
    (saw_shared_counter, nonblocking_acquire_result)."""
    import asyncio as _asyncio

    import ullm.shared_semaphore as ss

    sem = ss.SharedSemaphore("xproc-cap-test", 1)
    impl = sem._resolve()
    shared = not isinstance(impl, _asyncio.Semaphore)
    acquired = impl._proxy.acquire(False) if shared else None
    if acquired:
        impl._proxy.release()
    q.put((shared, acquired))


async def test_semaphore_cap_shared_with_spawned_child():
    # The I2b property under subprocess-per-game: a child process resolves
    # the SAME machine-wide counter by name (via the env-advertised manager)
    # and finds the parent's acquisition already holding the only slot.
    import multiprocessing

    ss = ullm.shared_semaphore
    ss.enable_multiprocess_semaphores()
    try:
        sem = ss.SharedSemaphore("xproc-cap-test", 1)
        await sem.acquire()
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(target=_xproc_probe, args=(q,))
        proc.start()
        try:
            shared, acquired = await asyncio.to_thread(q.get, True, 60)
        finally:
            proc.join(30)
        assert shared, "child fell back to a local semaphore — no cross-process cap"
        assert acquired is False, "child acquired a slot the parent already holds"
        sem.release()
    finally:
        ss.disable_multiprocess_semaphores()


def _pool_wave_child(
    prefix: str, max_kernels: int, max_running: int, barrier: Any, sleep_s: float, hold_s: float, q: Any
) -> None:
    """Spawned-child worker for the wave-timing tests: build an own-process
    pool+tool against the named counters, sync on the barrier so all
    children contend at once, run one timestamped cell, optionally hold the
    kernel alive idle, tear down, report timings."""
    import asyncio
    import time

    import ullm.tools.python as tp

    async def main():
        pool = tp.KernelPool(name_prefix=prefix)
        pool.max_kernels = max_kernels
        pool.max_running = max_running
        tool = tp.PythonTool(kernel_pool=pool, timeout=60.0)
        barrier.wait(60)
        tool.arguments = {"code": f"import time\ns = time.time()\ntime.sleep({sleep_s})\nprint(s, time.time())"}
        await tool.run()
        run_start, run_end = (float(x) for x in tool.result.to_string().split())
        time.sleep(hold_s)  # kernel alive but idle: occupies exists, not running
        await tool.clear_state()
        q.put((run_start, run_end, time.time()))

    asyncio.run(main())


def _max_overlap(intervals: Sequence[tuple[float, float]]) -> int:
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])
    current = best = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return best


async def _run_wave_children(
    prefix: str, *, max_kernels: int, max_running: int, sleep_s: float, hold_s: float, n: int = 4
) -> list[tuple[float, float, float]]:
    import multiprocessing

    ss = ullm.shared_semaphore
    # Left enabled on purpose: disabling would kill the manager under any
    # SharedSemaphore impl other tests in this worker already cached.
    ss.enable_multiprocess_semaphores()
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(n)
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_pool_wave_child, args=(prefix, max_kernels, max_running, barrier, sleep_s, hold_s, q))
        for _ in range(n)
    ]
    for p in procs:
        p.start()
    try:
        results = [await asyncio.to_thread(q.get, True, 120) for _ in range(n)]
    finally:
        for p in procs:
            p.join(30)
    return results


async def test_running_cap_serializes_execution_across_processes():
    # 4 children in 4 OS processes, released simultaneously, each running a
    # 0.4s cell against max_running=2 (exists cap slack): in-kernel run
    # windows overlap at most 2 deep — but DO reach 2 — and the total span
    # covers at least two waves. With per-process local semaphores this
    # would finish in one 4-wide wave.
    #
    # The equality is BOTH halves of the cap and neither is redundant:
    # `<= 2` is the cap holding, `>= 2` is the cap not being tighter than it
    # was asked for. Relaxing either direction to an inequality would leave
    # a per-process local semaphore (overlap 4) or an accidentally-serial
    # pool (overlap 1) passing. The span check below cannot stand in for the
    # `>= 2` half: a fully serial run has an even longer span.
    sleep_s = 0.4
    results = await _run_wave_children("xprocruncap", max_kernels=8, max_running=2, sleep_s=sleep_s, hold_s=0.0)
    runs = [(start, end) for start, end, _ in results]
    assert _max_overlap(runs) == 2, runs
    span = max(end for _, end in runs) - min(start for start, _ in runs)
    assert span >= 2 * sleep_s * 0.85, (span, runs)


async def test_exists_cap_serializes_kernel_lifetimes_across_processes():
    # Same shape, other cap: cells are instant but each child keeps its
    # kernel ALIVE (idle) for 0.4s before teardown, against max_kernels=2
    # (running cap slack). Alive windows — run start to post-clear_state —
    # overlap at most 2 deep with a two-wave span: the exists cap bites on
    # kernel lifetime, independently of execution concurrency.
    hold_s = 0.4
    results = await _run_wave_children("xprocexistscap", max_kernels=2, max_running=8, sleep_s=0.01, hold_s=hold_s)
    alive = [(start, cleared) for start, _, cleared in results]
    assert _max_overlap(alive) == 2, alive
    span = max(cleared for _, cleared in alive) - min(start for start, _ in alive)
    assert span >= 2 * hold_s * 0.85, (span, alive)
