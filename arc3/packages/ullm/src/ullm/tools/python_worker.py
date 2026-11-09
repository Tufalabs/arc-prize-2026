#!/usr/bin/env python3
"""REPL worker for PythonTool. Runs as a subprocess, communicates via JSON lines
over stdin/stdout. Executes Python code Jupyter-style: stdout is captured, and
if the last statement is an expression, its repr() is returned as the result.

Usage: python python_worker.py [memory_limit_mb]
"""

import ast
import base64
import contextlib
import io
import itertools
import json
import linecache
import os
import pickle
import re
import resource
import signal
import sys
import traceback
import warnings
from typing import Any, TextIO

#: What a cell's file is called where Python formats a location in it: a
#: traceback's frame header, a warning's location, a SyntaxError's file.
REPL_FILENAME = "<repl>"
#: What a cell's code is compiled as: a name of its own per cell. linecache
#: holds source by filename, so cells sharing one name would replace each
#: other's source, and a function defined in an earlier cell would read the
#: current cell's lines -- ``inspect.getsource`` would return whatever sits
#: at its line numbers there, and a traceback through it would quote lines it
#: does not have. Shown as ``REPL_FILENAME`` by the formatters alone
#: (``_show_cells_as_repl``): a cell's output is never rewritten, so the
#: model's own data and source reach it verbatim.
_CELL_FILENAME = re.compile(r"<repl-\d+>")
_cell_numbers = itertools.count(1)


class _HostAborted(BaseException):
    """Raised in the cell when a parent-side ``host_call`` failed, unwinding the
    cell so the failure is fatal. Subclasses ``BaseException`` so a user
    ``except Exception`` cannot swallow it."""


def _make_trusted_host_call(token: object, channel: TextIO):
    """Build the trusted ``host_call`` primitive for one kernel, closing over the
    parent's secret ``token`` (delivered via the stdin handshake in ``main``).

    Every frame carries the token; the parent dispatches only token-bearing calls
    (and only those can escalate a host failure to fatal). The token is never put
    in the cell namespace, ``os.environ``, or ``sys.argv`` — so LLM code in the
    kernel can't read it and thus can't forge a trusted call. The owner installs
    this under a private name that bootstrap code captures into a closure and then
    deletes, so it never reaches an LLM-visible name.

    Writes the request frame to the real fd 1 — bypassing ``execute_code``'s
    ``redirect_stdout`` capture — then blocks on ``channel`` (the protocol's
    stdin) for the reply. Returns the
    function's result, or raises ``_HostAborted`` if the parent reported failure
    or closed the pipe (the parent has, by then, recorded the real exception and
    surfaces it as fatal)."""

    def host_call(name: str, *args: object, **kwargs: object) -> object:
        payload = base64.b64encode(pickle.dumps((args, kwargs))).decode("ascii")
        frame = json.dumps({"host_call": {"name": name, "payload": payload, "token": token}}) + "\n"
        # A timeout may interrupt user code, but must not leave half a reply
        # for the next execution request to consume. SIGKILL still bounds a
        # blocked transport when the parent's interrupt grace expires.
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            os.write(1, frame.encode())
            line = channel.readline()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        if not line:
            raise _HostAborted(f"host_call({name!r}): parent closed the pipe")
        reply = json.loads(line)
        if "host_result" in reply:
            return pickle.loads(base64.b64decode(reply["host_result"]))
        raise _HostAborted(f"host_call({name!r}) failed in parent: {reply.get('host_abort')}")

    return host_call


def _register_source(filename: str, source: str) -> None:
    """Make ``source`` resolvable by ``linecache.getline(filename, lineno)``.

    Python's traceback formatter uses linecache to fetch source lines; for
    anonymous compilation targets like ``"<repl>"`` this normally comes up
    empty, so tracebacks display only filename/line/function and skip the
    code body. Injecting the mapping here restores the full traceback.
    """
    lines = [line + "\n" for line in source.splitlines()]
    linecache.cache[filename] = (len(source), None, lines, filename)


def _next_cell_filename() -> str:
    """The compile name for the next cell."""
    return f"<repl-{next(_cell_numbers)}>"


def _is_cell_filename(filename: str) -> bool:
    """Whether code compiled under ``filename`` is a cell's."""
    return filename == REPL_FILENAME or _CELL_FILENAME.fullmatch(filename) is not None


def _show_cells_as_repl() -> None:
    """Name a cell's file ``REPL_FILENAME`` where Python formats a location
    in it: a traceback's frame header, and a warning's location.

    Done inside the formatters, from the frame's or the warning's own
    filename, after its source line has been read under the cell's real
    name -- never by rewriting finished text, which cannot tell a location
    from the model's own data that looks like one. Installed once, by the
    worker's entry point; every traceback the cell's code formats itself
    (``traceback.print_exc``, ``logging``) goes through them too."""
    format_frame = traceback.StackSummary.format_frame_summary

    def format_frame_summary(self: traceback.StackSummary, frame_summary: traceback.FrameSummary) -> str:
        # The row's first line is the header Python builds from the frame's
        # filename; the source line under it is the cell's own, left as read.
        row = format_frame(self, frame_summary)
        filename = frame_summary.filename
        if filename == REPL_FILENAME or not _is_cell_filename(filename):
            return row
        header, newline, rest = row.partition("\n")
        return header.replace(f'File "{filename}"', f'File "{REPL_FILENAME}"', 1) + newline + rest

    traceback.StackSummary.format_frame_summary = format_frame_summary  # pyright: ignore[reportAttributeAccessIssue]

    format_warning = warnings.formatwarning

    def formatwarning(
        message: Warning | str, category: type[Warning], filename: str, lineno: int, line: str | None = None
    ) -> str:
        # The line is read under the cell's real name before the name is shown.
        if _is_cell_filename(filename):
            if line is None:
                line = linecache.getline(filename, lineno)
            filename = REPL_FILENAME
        return format_warning(message, category, filename, lineno, line)

    warnings.formatwarning = formatwarning


def _format_user_traceback(
    exc_type: "type[BaseException] | None", exc_value: "BaseException | None", exc_tb: Any
) -> str:
    """Format an exception's traceback with framework frames trimmed.

    The worker wraps user code in ``exec(...)`` / ``eval(...)`` calls that
    would otherwise show up as uninformative top frames pointing at this
    file's ``execute_code``. We walk the traceback until the first frame
    compiled from a cell (``<repl-N>``, set by our ``compile()`` calls) and
    format from there, so the LLM sees only its own code path.

    Edge cases handled:
    - ``SyntaxError`` / ``IndentationError`` have no meaningful traceback
      into user code (the error is raised by ``ast.parse`` / ``compile``
      before anything ran). We show just the ``format_exception_only``
      output, which already carries filename, line, offset, and caret.
    - If no cell frame is found (shouldn't happen for user code,
      but guard anyway), fall back to the full traceback.
    """
    if isinstance(exc_value, SyntaxError):
        # One raised compiling a cell (``'return' outside function``) names the
        # cell's file on the exception, where the formatter reads it.
        if exc_value.filename is not None and _is_cell_filename(exc_value.filename):
            exc_value.filename = REPL_FILENAME
        return "".join(traceback.format_exception_only(exc_type, exc_value))
    if isinstance(exc_value, _HostAborted):
        # A refused host call carries its reason in the message; its traceback
        # runs through this worker's own RPC frames, whose absolute paths are
        # noise to the model and would expose harness source. Show the message
        # only, as for a SyntaxError.
        return "".join(traceback.format_exception_only(exc_type, exc_value))
    tb = exc_tb
    while tb is not None:
        if _is_cell_filename(tb.tb_frame.f_code.co_filename):
            break
        tb = tb.tb_next
    return "".join(traceback.format_exception(exc_type, exc_value, tb or exc_tb))


def apply_resource_limits(memory_limit_mb: int) -> None:
    """Apply memory/fs/fd limits via setrlimit. CPU is not limited — the
    parent enforces its own wall-clock timeout via SIGINT."""
    memory_bytes = memory_limit_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except ValueError:
        assert sys.platform == "darwin", "RLIMIT_AS should only fail on macOS"
    # Cap file-size writes at 256 MB so a runaway write can't fill shared disk.
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024, 256 * 1024 * 1024))
    # Cap open file descriptors to prevent fd-exhaustion.
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))


def execute_code(code: str, namespace: dict) -> dict:
    """Execute code Jupyter-style. Returns dict with stdout, stderr, result, error."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    result = None
    error = None

    # Register the source with linecache under the filename we pass to
    # ``compile``, so traceback.format_exception can render the source line
    # (e.g. the caret-annotated `history = _rs.history`) instead of showing
    # just `File "<repl>", line 1, in <module>` with no body. The entry stays
    # for the life of the kernel: a function this cell defines is read back
    # from it by every later traceback and ``inspect.getsource``.
    filename = _next_cell_filename()
    _register_source(filename, code)

    try:
        tree = ast.parse(code)

        # If the last statement is an expression, split it off for eval
        last_expr = tree.body[-1] if tree.body else None
        if isinstance(last_expr, ast.Expr):
            tree.body.pop()

            # Exec everything except the last statement
            if tree.body:
                exec_code = compile(ast.Module(body=tree.body, type_ignores=[]), filename, "exec")
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    exec(exec_code, namespace)

            # Eval the last expression and capture its repr
            eval_code = compile(ast.Expression(body=last_expr.value), filename, "eval")
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                eval_result = eval(eval_code, namespace)
            # Only include result if non-None (matches Jupyter/IPython behavior)
            if eval_result is not None:
                result = repr(eval_result)
        else:
            # No trailing expression — just exec everything
            exec_code = compile(tree, filename, "exec")
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                exec(exec_code, namespace)

    except (KeyboardInterrupt, SystemExit, Exception, _HostAborted):
        # KeyboardInterrupt is how the parent's SIGINT (timeout) lands inside
        # user code. _HostAborted (a BaseException, so user `except Exception`
        # can't swallow it) is how a failed parent-side host_call unwinds the
        # cell. Catching it here returns a clean terminal frame — keeping the
        # worker alive — so the parent can surface the real host exception as
        # fatal. SystemExit (`exit()`, `sys.exit()`, `raise SystemExit`) ends
        # the cell, not the kernel, as in a notebook: code that stops itself
        # early keeps its session and the output it printed. Formatting it
        # surfaces the line that was running.
        error = _format_user_traceback(*sys.exc_info())

    return {
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "result": result,
        "error": error,
    }


def main() -> None:
    # Apply resource limits if specified
    """Worker entry point: apply the resource limits this kernel was given, then
    serve execution requests until stdin closes.
    """
    if len(sys.argv) > 1:
        apply_resource_limits(int(sys.argv[1]))

    # Put the current working directory on sys.path so `import foo` finds
    # files in the REPL's tmpdir (e.g. world_model.py). Python normally only
    # adds cwd to sys.path in interactive mode or with `python -m`; running
    # this worker as a script file would otherwise put the worker's own
    # directory on sys.path instead, leaving cwd-based imports broken.
    sys.path.insert(0, "")

    # A cell's file reads as REPL_FILENAME in every traceback and warning.
    _show_cells_as_repl()

    # Mask SIGINT during I/O to prevent protocol corruption
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # The protocol's own handle on stdin: requests and host-call replies are read
    # from it. Cells get an empty stdin in its place, so the code they run cannot
    # consume or close the protocol: `input()` reads end of file, and `exit()`,
    # which closes `sys.stdin` before it raises, closes that one.
    channel = sys.stdin
    sys.stdin = io.StringIO()

    # The parent sends a config handshake as the very first stdin line, carrying the
    # secret host_call token. Read it before any cell runs and keep the token out of
    # the cell namespace / env / argv. If the first line isn't a config frame (e.g. a
    # caller that doesn't handshake), treat it as the first request instead.
    host_call_token = None
    pending_request_line = None
    first_line = channel.readline()
    if first_line:
        try:
            _obj = json.loads(first_line)
        except json.JSONDecodeError:
            _obj = None
        if isinstance(_obj, dict) and "config" in _obj:
            host_call_token = (_obj.get("config") or {}).get("host_call_token")
        else:
            pending_request_line = first_line

    # Install the parent-RPC primitive under a private name so initial_code/bootstrap
    # can capture it into a closure (and delete the name); it carries the token, so
    # cells can't forge a trusted call even if they reach it.
    namespace: dict = {"__host_call_trusted__": _make_trusted_host_call(host_call_token, channel)}

    while True:
        # Read request (SIGINT masked); the first iteration may reuse a line already
        # read during the handshake.
        if pending_request_line is not None:
            line, pending_request_line = pending_request_line, None
        else:
            line = channel.readline()
        if not line:
            break  # EOF — parent closed the pipe

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response = {"stdout": "", "stderr": "", "result": None, "error": f"Invalid JSON request: {line.strip()}"}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue

        if "code" not in request:
            # Not a request: a host-call reply the cell abandoned when an
            # interrupt unwound it, still sitting in the pipe. Drop it —
            # reading it as a request would kill the worker on the way in,
            # and the parent is not waiting for an answer to it.
            continue

        # Unmask SIGINT during code execution
        signal.signal(signal.SIGINT, signal.default_int_handler)
        response = execute_code(request["code"], namespace)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Write response (SIGINT masked)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
