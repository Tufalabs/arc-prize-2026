"""The PythonTool kernel prelude: the model-facing python environment.

Runs once per kernel start (``PythonTool.initial_code``). The state payload
installs the duck's compact timeline and ``submit_action(...)``, inside the
restricted, hardened ullm kernel.

Per call the agent sends ``_duck_run('''<code>''')``; ``_duck_run`` pulls
the current game state from the host (``get_state`` host call), executes
the snippet, and prints exactly one JSON payload line
(``{"stdout", "result", "error", "action_results"}``) that the agent
parses back into the original tool-result rendering.

``submit_action`` normalizes its argument here and hands it to the
fixed-target ``action`` host call, which executes it against the live game.
The reply carries the real result AND a refreshed state payload, which is
re-installed into the snippet's globals before the next statement runs — so
a loop can act, look, and act again."""

import pathlib

# Spliced in verbatim (as the original sandbox bootstrap did): importing
# `duck.segmentation` in the kernel would execute duck/__init__ and drag the
# whole numpy/ullm/platypus import chain into the isolated worker.
#
# Verbatim means its SOURCE TEXT, comments and all, is part of the prelude:
# any edit to segmentation.py — even a reworded comment — changes what the
# kernel runs, so treat it as prelude code rather than as an ordinary module.
_SEGMENTATION_SOURCE = (pathlib.Path(__file__).parent / "segmentation.py").read_text(encoding="utf-8")

_PRELUDE_TEMPLATE = r"""
import array
import base64
import builtins
import contextlib
import importlib
import io
import json
import linecache
import os as _duck_os
import sys
import traceback

# The kernel's working directory is where the model's files live, so a file
# it wrote is importable. It goes LAST on the import path: the allowlist
# decides an import before the import runs, so a workspace file named after
# an allowlisted module would otherwise be reached under that module's
# permission — the model asks for `random` and gets its own draft.
# The relative entry the worker puts first is what would do that, so it is
# replaced rather than merely followed.
_DUCK_WORKSPACE = _duck_os.path.realpath(_duck_os.getcwd())
sys.path[:] = [entry for entry in sys.path if entry not in ('', '.', _DUCK_WORKSPACE)]
sys.path.append(_DUCK_WORKSPACE)

COLOR_CHARS = "WwgGcBMPRbSYOrNp"
# Numeric mode prints the board as hex digits so a glyph and an array value
# name the same colour below 10; the legend states where that stops.
HEX_CHARS = "0123456789abcdef"

__SEGMENTATION_SOURCE__

def _make_host_adapters(host_call):
    # Remove the accidental direct dispatcher path from model-callable globals.
    # Python callable traversal can still reach captured state, so these use
    # fixed targets and every parent callback validates model-authored input.
    def get_state():
        return host_call("get_state")

    def submit_actions(normalized_actions):
        return host_call("action", normalized_actions)

    return get_state, submit_actions


_DUCK_GET_STATE, _DUCK_SUBMIT_ACTIONS = _make_host_adapters(  # noqa: F821
    __host_call_trusted__
)
del _make_host_adapters, __host_call_trusted__  # noqa: F821

# What a snippet may import. Not a boundary (design.md, "The configuration
# surface"): a file of the model's own imports anything. What it keeps out of
# easy reach is what breaks the run by accident rather than by design -- `os`,
# `sys`, `subprocess`, `signal`, threads and processes, which act as the
# harness's own user or unseat the kernel's result line.
SAFE_MODULES = {
    "bisect", "collections", "copy", "difflib", "fractions", "functools",
    "heapq", "itertools", "json", "math", "numpy", "operator", "random",
    "re", "statistics", "string", "textwrap",
}
SAFE_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "complex", "dict", "dir", "divmod", "enumerate",
    "Exception", "filter", "float", "format", "frozenset", "getattr",
    "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "oct", "ord", "pow",
    "print", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "TypeError", "type", "ValueError",
    "RuntimeError", "zip",
}
# The wide sandbox (`DuckAgent.wide_sandbox`; 2026-09-22): what
# models reached for and were refused in sweep 12's pilot.
# `importlib`, `pickle` and `inspect` each reach past the list from a snippet;
# `scipy` costs a kernel ~30 MB for `ndimage`, ~70 MB with `signal`, and only
# where it is imported. `open` lets a snippet keep data in a file (a json dump,
# a pickle) as it keeps code in one; the path is not confined to the working
# directory, which a file of the model's own could never be kept from either.
# And every exception class Python has: an `except KeyError:` names its class
# only when an exception reaches it, so a missing one raises NameError exactly
# then, in place of the error the clause was written to handle.
WIDE_MODULES = {"hashlib", "importlib", "inspect", "pickle", "scipy", "time", "traceback"}
WIDE_BUILTINS = {"open"} | {
    name for name, value in vars(builtins).items() if isinstance(value, type) and issubclass(value, Exception)
}


_duck_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_duck_os.environ["OMP_NUM_THREADS"] = "1"
import numpy as _DUCK_NUMPY

# Every module a snippet may import, bound in a new session as `np` is: models
# write `collections.Counter` as they write `np.where`, without an import (29
# NameErrors in sweep 12's first full run, 30 minutes in; 2026-09-22).
# About 5 MB and 20 ms a kernel, since scipy's submodules still load on first
# use; imported here, under the bootstrap deadline, whichever sandbox the
# session gets. Bound when a session starts and not again, so a variable the
# model names `time` or `copy` stays its own.
_DUCK_MODULES = {name: importlib.import_module(name) for name in sorted(SAFE_MODULES | WIDE_MODULES)}


def _duck_namespace(wide):
    allowed_builtins = SAFE_BUILTINS | WIDE_BUILTINS if wide else SAFE_BUILTINS
    allowed_modules = SAFE_MODULES | WIDE_MODULES if wide else SAFE_MODULES
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in allowed_builtins}}
    namespace.update((name, module) for name, module in _DUCK_MODULES.items() if name in allowed_modules)
    return namespace


# The two questions the workspace raises, bound to the path functions that
# answer them. `os` is deleted from this namespace below and stays deleted:
# keeping even `os.path` around would put `os` back within reach of anything
# that walks a callable's globals.
def _make_workspace_probes(realpath, abspath, dirname, join, isfile):
    def loaded_from_workspace(origin):
        # Imports use the canonical workspace entry in sys.path. Reject
        # installed modules lexically before touching the filesystem; resolving
        # their library paths here would walk NFS metadata on every snippet.
        # Local symlinks still need the resolved-target containment check.
        return (dirname(abspath(origin)) == _DUCK_WORKSPACE
                and dirname(realpath(origin)) == _DUCK_WORKSPACE)

    def workspace_module(root):
        return isfile(join(_DUCK_WORKSPACE, root + '.py'))

    return loaded_from_workspace, workspace_module


_DUCK_FROM_WORKSPACE, _DUCK_WORKSPACE_MODULE = _make_workspace_probes(
    _duck_os.path.realpath, _duck_os.path.abspath, _duck_os.path.dirname,
    _duck_os.path.join, _duck_os.path.isfile
)
del _make_workspace_probes, _duck_os


# `importlib.reload` of a module the model wrote. Before every snippet the
# kernel drops the model's modules from `sys.modules`, so that an import sees
# an edit (`_refresh_workspace_modules`), and a reload then finds nothing to
# reload. It puts the module back first and reloads as Python does: in place,
# so the name the session already holds sees the edit too.
def _make_workspace_reload(reload, from_workspace):
    def reload_workspace_too(module):
        name = getattr(module, "__name__", None)
        origin = getattr(module, "__file__", None)
        if isinstance(name, str) and origin and name not in sys.modules and from_workspace(origin):
            sys.modules[name] = module
        return reload(module)

    reload_workspace_too.__name__ = "reload"
    return reload_workspace_too


importlib.reload = _make_workspace_reload(importlib.reload, _DUCK_FROM_WORKSPACE)
del _make_workspace_reload


def _numpy_module():
    # Loaded by initial_code under its bootstrap deadline, rather than charging
    # the first model snippet's much shorter execution deadline.
    return _DUCK_NUMPY


def _format_grid_ascii(grid, glyphs=COLOR_CHARS):
    if not grid:
        return "(empty grid)"
    out = []
    for row in grid:
        chars = []
        for v in row:
            i = int(v)
            if i < 0: i = 0
            elif i > 15: i = 15
            chars.append(glyphs[i])
        out.append("".join(chars))
    return "\n".join(out)


class FrameView:
    # `numeric` swaps what a colour LOOKS like, not what it is: the board
    # prints as hex, and the array and the segment nodes carry the index
    # itself rather than a one-character stand-in for it.
    def __init__(self, *, grid_bytes, shape, step, level, numeric=False):
        self.step = step
        self.level = level
        self.shape = tuple(shape)
        self._grid_bytes = grid_bytes
        self._grid_cache = None
        self._array_cache = None
        self._ascii_cache = None
        self._segments = None
        self._numeric = numeric
        self._glyphs = HEX_CHARS if numeric else COLOR_CHARS
        # segment_layer indexes this per component, so a range of ints yields
        # integer colours without touching the spliced segmentation source.
        self._colors = list(range(16)) if numeric else COLOR_CHARS

    @property
    def _grid(self):
        if self._grid_cache is None:
            rows, cols = self.shape
            flat = array.array("B", self._grid_bytes)
            self._grid_cache = [list(flat[r * cols:(r + 1) * cols]) for r in range(rows)]
        return self._grid_cache

    @property
    def ascii(self):
        if self._ascii_cache is None:
            self._ascii_cache = _format_grid_ascii(self._grid, self._glyphs)
        return self._ascii_cache

    @property
    def array(self):
        if self._array_cache is None:
            if self._numeric:
                self._array_cache = _numpy_module().array(
                    [[max(0, min(15, int(value))) for value in row] for row in self._grid],
                    dtype="int8",
                )
            else:
                self._array_cache = _numpy_module().array(
                    [[COLOR_CHARS[max(0, min(15, int(value)))] for value in row] for row in self._grid],
                    dtype="<U1",
                )
        return self._array_cache

    def segment(self):
        if self._segments is None:
            self._segments = segment_layer(self._grid, self._colors)
        return self._segments

    def __str__(self):
        rows, cols = self.shape
        return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

    __repr__ = __str__


def _frame_from_payload(payload, frame_api):
    if not isinstance(payload, dict):
        return None
    b64 = payload.get("grid_b64") or ""
    return FrameView(
        grid_bytes=base64.b64decode(b64) if b64 else b"",
        shape=payload.get("shape", [0, 0]),
        step=int(payload.get("step", 0)),
        level=int(payload.get("level", 0)),
        numeric=frame_api == "numeric",
    )


def _timeline_from_payload(payload, frame_api):
    frames = []
    animations = []
    actions = []
    level_transitions = []
    for entry in payload or []:
        if not isinstance(entry, dict):
            continue
        animation = [
            frame
            for frame in (_frame_from_payload(item, frame_api) for item in entry.get("animation", []))
            if frame is not None
        ]
        if not animation:
            continue
        animations.append(animation)
        frames.append(animation[-1])
        if len(frames) > 1:
            actions.append(str(entry.get("action", "")))
            level_transitions.append(frames[-2].level != frames[-1].level)
    return frames, animations, actions, level_transitions


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


# Every snippet compiles under a name of its own, and its lines are kept under
# that name for the life of the kernel. A function defined in one snippet and
# called from a later one raises with a line number counted in the snippet that
# defined it, so a traceback can only quote the right line if it knows which.
_DUCK_SNIPPETS = {}
_DUCK_SNIPPET_PREFIX = "<python_tool#"
# A frame's source line past this many characters is cut: a literal the model
# pasted into a line would otherwise fill the error field on its own.
_DUCK_SOURCE_LINE_CHARS = 200
# Identical consecutive frames shown before the rest are counted instead, as
# Python's own traceback does: a runaway recursion is a thousand of them, and
# the exception itself would be cut off the end of the tool result.
_DUCK_REPEATS_SHOWN = 3


# The model's own code is its snippets and the files it wrote and imported, and
# a traceback shows exactly those frames, each with its source line: the
# prelude's and the libraries' are nothing the model can act on, and their
# paths are the host's. The snippet running now is `<python_tool>`, one run by
# an earlier call `<python_tool, earlier call>`, and a file is named by itself
# alone, since the directory is the host's too.
def _model_frame(filename, lineno, current):
    if filename.startswith(_DUCK_SNIPPET_PREFIX):
        lines = _DUCK_SNIPPETS.get(filename) or []
        text = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        return ("<python_tool>" if filename == current else "<python_tool, earlier call>"), text
    # Pseudo-files (`<frozen importlib._bootstrap>`) have no directory, so the
    # workspace probe would resolve them INTO the workspace.
    if not filename.startswith("<") and _DUCK_FROM_WORKSPACE(filename):
        # The file as it is now: an edit since the import shows here, as it
        # does in Python's own traceback.
        linecache.checkcache(filename)
        return filename.rsplit("/", 1)[-1], linecache.getline(filename, lineno)
    return None


def _source_line(text):
    text = text.strip()
    if len(text) > _DUCK_SOURCE_LINE_CHARS:
        text = text[:_DUCK_SOURCE_LINE_CHARS] + " ..."
    return [f"    {text}"] if text else []


def _sanitize_exception(exc, current):
    extracted = traceback.extract_tb(exc.__traceback__)
    frames = []
    for frame in extracted:
        model = _model_frame(frame.filename, frame.lineno, current)
        if model is not None:
            frames.append((model[0], frame.lineno, frame.name, model[1]))
    if not frames:
        # Nothing of the model's on the stack: the innermost frame, unnamed.
        frames = [("<python_tool>", frame.lineno, frame.name, "") for frame in extracted[-1:]]
    lines = ["Traceback (most recent call last):"]
    previous, repeats = None, 0
    for entry in frames + [None]:
        if entry is not None and entry == previous:
            repeats += 1
            if repeats > _DUCK_REPEATS_SHOWN:
                continue
        else:
            if repeats > _DUCK_REPEATS_SHOWN:
                lines.append(f"  [Previous line repeated {repeats - _DUCK_REPEATS_SHOWN} more times]")
            previous, repeats = entry, 1
        if entry is None:
            break
        name, lineno, function, text = entry
        lines.append(f'  File "{name}", line {lineno}, in {function}')
        lines.extend(_source_line(text))
    # A file of the model's that does not parse raises with no frame of its
    # own, and Python's message carries the file's full path: shown instead as
    # Python shows it, at the file and line, with the line itself.
    model = (
        _model_frame(exc.filename, exc.lineno or 0, current)
        if isinstance(exc, SyntaxError) and exc.filename
        else None
    )
    if model is not None:
        lines.append(f'  File "{model[0]}", line {exc.lineno}')
        lines.extend(_source_line(exc.text or model[1]))
        lines.append(f"SyntaxError: {exc.msg}")
    else:
        lines.append(f"{exc.__class__.__name__}: {exc}")
    return "\n".join(lines)


def _safe_import(
    name,
    globals=None,
    locals=None,
    fromlist=(),
    level=0,
    *,
    file_tools=False,
    wide=False,
):
    root = str(name or "").split(".", 1)[0]
    if root == "submit_action" and str(name) == root and not fromlist and level == 0:
        # Compatibility for the common model mistake ``import submit_action``.
        # The import resolves to the already-installed callable rather than
        # spending a tool round on a sandbox error.
        if isinstance(globals, dict) and callable(globals.get("submit_action")):
            return globals["submit_action"]
    if root == "memory":
        raise ImportError("memory is a native tool, not a Python module or global.")
    if root not in SAFE_MODULES and not (wide and root in WIDE_MODULES):
        # A file the model wrote in its working directory is importable. The
        # module then executes with its own globals and the REAL builtins, so
        # it can reach what a snippet cannot: a deliberate widening, made
        # here rather than arrived at. The restricted namespace was never a
        # security boundary (design.md, "The configuration surface"); what it
        # buys is a model that stays on the intended surface, and a helper
        # the model itself wrote is on it.
        if file_tools and root.isidentifier() and _DUCK_WORKSPACE_MODULE(root):
            return builtins.__import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
    return builtins.__import__(name, globals, locals, fromlist, level)


# A coordinate read off an array is a numpy scalar, which the action host call
# cannot carry: its payload may rebuild plain containers and nothing else, so
# the call is refused before the host sees it. It goes as the Python number.
def _plain_scalar(value):
    return value.item() if isinstance(value, _DUCK_NUMPY.generic) else value


def _normalize_actions(actions, function_name):
    if isinstance(actions, str):
        items = [actions]
    elif isinstance(actions, dict):
        items = [actions]
    elif isinstance(actions, (list, tuple)):
        items = list(actions)
    else:
        raise TypeError(
            f"{function_name}(actions) expects a string, an action object, or a list of action strings/objects."
        )
    if not items:
        raise ValueError(f"{function_name}(actions) requires at least one action.")

    normalized = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            action_name = item.strip()
            if not action_name:
                raise ValueError(f"Action {index} is empty.")
            normalized.append({"action": action_name})
            continue
        if isinstance(item, dict):
            action_name = str(item.get("action", "")).strip()
            if not action_name:
                raise ValueError(f"Action {index} is missing an `action` field.")
            entry = {"action": action_name}
            if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                raise ValueError(
                    f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                )
            if "row" in item:
                entry["row"] = _plain_scalar(item.get("row"))
            if "col" in item:
                entry["col"] = _plain_scalar(item.get("col"))
            normalized.append(entry)
            continue
        raise TypeError(f"Action {index} must be a string or a dict.")
    return normalized


# Namespace carried between snippets when the agent runs a persistent mode.
# None until the first persistent call; dropped whenever the kernel is reset,
# because the reset takes the whole process with it.
_DUCK_SESSION = None


# Drop every module loaded from the working directory, and forget the import
# system's picture of what is in it. Run before each snippet: without it the
# model edits a file, imports it again, gets the cached module and reasons
# about code it has just replaced. The cache invalidation is the other half —
# importlib remembers a directory's contents by mtime at one-second
# granularity, so a file created just after the last look would be invisible.
def _refresh_workspace_modules():
    for name, module in list(sys.modules.items()):
        origin = getattr(module, '__file__', None)
        if not origin:
            continue
        if _DUCK_FROM_WORKSPACE(origin):
            del sys.modules[name]
    importlib.invalidate_caches()


def _state_modes(payload):
    if not isinstance(payload, dict):
        raise TypeError("State payload must be an object.")
    if "frame_api" not in payload:
        raise ValueError("State payload is missing required mode 'frame_api'.")
    frame_api = payload["frame_api"]
    if frame_api not in {"ergonomic", "numeric"}:
        raise ValueError(f"State payload has unknown frame_api {frame_api!r}.")
    # Unlike the mode above, this one may be absent: a mode decides what the
    # model is shown and so must always be stated, while file access is a
    # permission, and a permission nobody granted is not held.
    file_tools = payload.get("file_tools", False)
    if type(file_tools) is not bool:
        raise ValueError(f"State payload has non-boolean file_tools {file_tools!r}.")
    # Absent means the shipped repr, so a default run's payload -- which the
    # model can print -- is exactly the bytes it was before this setting
    # existed. Only a run that asks for the other one says so.
    scalar_repr = payload.get("numpy_scalar_repr", "np2")
    if scalar_repr not in {"np2", "legacy"}:
        raise ValueError(f"State payload has unknown numpy_scalar_repr {scalar_repr!r}.")
    # A permission too, absent unless granted, and so absent from a default
    # run's payload.
    wide = payload.get("wide_sandbox", False)
    if type(wide) is not bool:
        raise ValueError(f"State payload has non-boolean wide_sandbox {wide!r}.")
    return frame_api, scalar_repr, file_tools, wide


def _duck_run(code, persist=False):
    global _DUCK_SESSION
    state_payload = _DUCK_GET_STATE()
    action_results = []
    stdout = io.StringIO()
    frame_api, numpy_scalar_repr, file_tools, wide = _state_modes(state_payload)
    configured_modes = (frame_api, numpy_scalar_repr, file_tools, wide)
    if file_tools:
        # Without file permission, there are no workspace modules to evict.
        _refresh_workspace_modules()
    if numpy_scalar_repr == "legacy":
        # numpy 2 reprs a scalar as np.int8(5), so a printed coordinate list
        # reads [(np.int64(37), np.int64(7)), ...] rather than [(37, 7), ...].
        # Independent of frame_api: argwhere returns int64 indices whatever
        # the board's dtype, so the char arms print constructor calls too.
        _DUCK_NUMPY.set_printoptions(legacy='1.25')

    def _runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
        return _safe_import(
            name,
            globals,
            locals,
            fromlist,
            level,
            file_tools=file_tools,
            wide=wide,
        )

    if persist:
        if _DUCK_SESSION is None:
            _DUCK_SESSION = _duck_namespace(wide)
        runtime_globals = _DUCK_SESSION
        # `result` is a per-call output slot, not carried state: leaving the
        # previous call's value would silently re-report it as this call's.
        runtime_globals["result"] = None
    else:
        runtime_globals = _duck_namespace(wide)
        runtime_globals["result"] = None
    runtime_globals["__builtins__"]["__import__"] = _runtime_import

    def _install_state(payload):
        if _state_modes(payload) != configured_modes:
            raise ValueError("State payload modes changed within one configured Python session.")
        frames, animations, actions, level_transitions = _timeline_from_payload(
            payload.get("timeline"), frame_api
        )
        last_action_result = payload.get("last_action_result")
        action_result = dict(last_action_result) if isinstance(last_action_result, dict) else {}

        current_frame = frames[-1] if frames else None
        runtime_globals["current_frame"] = current_frame
        runtime_globals["frames"] = frames
        runtime_globals["animations"] = animations
        runtime_globals["actions"] = actions
        runtime_globals["level_transitions"] = level_transitions
        runtime_globals["last_action"] = actions[-1] if actions else None
        runtime_globals["valid_actions"] = [str(item) for item in payload.get("valid_actions", [])]
        runtime_globals["last_action_result"] = action_result

    def execute_actions(actions):
        normalized_actions = _normalize_actions(actions, "submit_action")
        reply = _DUCK_SUBMIT_ACTIONS(normalized_actions)
        if not isinstance(reply, dict):
            raise RuntimeError("Invalid action response from host.")
        if reply.get("error"):
            raise ValueError(str(reply["error"]))
        if isinstance(reply.get("state"), dict):
            # Live execution: the host ran the actions and returned the
            # refreshed runtime state — re-install it so current_frame /
            # animations / valid_actions are live for the rest of the snippet.
            _install_state(reply["state"])
        res = reply.get("result") if isinstance(reply.get("result"), dict) else (reply.get("ack") or {})
        action_results.append(res)
        return dict(res)

    execute_actions.__name__ = "submit_action"
    execute_actions.__qualname__ = "_duck_run.<locals>.submit_action"
    if runtime_globals.get("np") is None or runtime_globals.get("np") is _DUCK_NUMPY:
        runtime_globals["np"] = _DUCK_NUMPY
    runtime_globals.pop("memory", None)
    runtime_globals["submit_action"] = execute_actions
    _install_state(state_payload)

    snippet = f"{_DUCK_SNIPPET_PREFIX}{len(_DUCK_SNIPPETS)}>"
    _DUCK_SNIPPETS[snippet] = str(code).splitlines()
    # In linecache too, as IPython keeps its cells: `inspect.getsource` reads a
    # function's source from there, and a snippet has no file for it to read.
    # No mtime, so `linecache.checkcache` leaves the entry alone.
    linecache.cache[snippet] = (len(str(code)), None, str(code).splitlines(True), snippet)
    try:
        compiled = compile(str(code), snippet, "exec")
        with contextlib.redirect_stdout(stdout):
            exec(compiled, runtime_globals, runtime_globals)
        payload = {
            "stdout": stdout.getvalue(),
            "result": _json_safe(runtime_globals.get("result")),
            "action_results": _json_safe(action_results),
        }
    except Exception as exc:
        payload = {
            "error": _sanitize_exception(exc, snippet),
            "stdout": stdout.getvalue(),
            "action_results": _json_safe(action_results),
        }
    print("__DUCK_PAYLOAD__" + json.dumps(payload, ensure_ascii=False))
"""

PRELUDE = _PRELUDE_TEMPLATE.replace("__SEGMENTATION_SOURCE__", _SEGMENTATION_SOURCE)

#: Printed before the session listing's JSON line.
SESSION_LISTING_MARKER = "__DUCK_SESSION_LISTING__"

#: A host call, not a snippet: run in the kernel's own globals between rounds,
#: it lists what the model bound in its persistent session as
#: ``[name, description, is_function]``, in the order they were bound. Left
#: out: the names ``_duck_run`` installs on every call (the harness's), and
#: what is an import or a leftover rather than the model's work -- modules,
#: types other than a class from one of the model's own files, and ``None``.
SESSION_LISTING_CODE = (
    r"""
def _duck_session_listing():
    harness = {
        "result", "current_frame", "frames", "animations", "actions", "level_transitions",
        "last_action", "valid_actions", "last_action_result", "submit_action",
    }
    function_type, module_type = type(_duck_session_listing), type(json)

    def own_class(value):
        origin = getattr(sys.modules.get(getattr(value, "__module__", "")), "__file__", None)
        return bool(origin) and _DUCK_FROM_WORKSPACE(origin)

    listing = []
    for name, value in (_DUCK_SESSION or {}).items():
        if name.startswith("__") or name in harness or value is None or isinstance(value, module_type):
            continue
        if isinstance(value, type) and not own_class(value):
            continue
        kind = type(value).__name__
        if isinstance(value, type):
            text = "class"
        elif isinstance(value, _DUCK_NUMPY.ndarray):
            text = f"ndarray {value.dtype} {tuple(value.shape)}"
        elif isinstance(value, dict):
            text = f"dict, {len(value)} {'entry' if len(value) == 1 else 'entries'}"
        elif isinstance(value, (list, tuple, set, frozenset)):
            text = f"{kind}, {len(value)} {'item' if len(value) == 1 else 'items'}"
        elif isinstance(value, str):
            text = f"str, {len(value)} {'char' if len(value) == 1 else 'chars'}"
        else:
            text = kind
        listing.append([name, text, isinstance(value, function_type)])
    print("__MARKER__" + json.dumps(listing))


_duck_session_listing()
del _duck_session_listing
"""
).replace("__MARKER__", SESSION_LISTING_MARKER)
