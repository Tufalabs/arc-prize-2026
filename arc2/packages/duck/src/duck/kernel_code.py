"""What the model's kernel runs before and around its own code.

Three pieces of source text. :func:`prelude` is the one-time startup: the
sandbox patches, the imports a model would otherwise spend a tool call on,
and the numpy print settings. :func:`task_setup_code` builds the task into
the namespace as bare arrays. :data:`READ_ANSWER_CODE` reads the model's
answer back out.

Everything here is code the model's kernel runs and whose output the model
reads, so it is part of the prompt surface in all but name: a change to it
changes what the model sees.
"""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import numpy.typing as npt

import duck.sandbox
import taaf.game
import taaf.grid
from duck.errors import HARNESS_SETUP_FILE

#: Prefix of the line :data:`READ_ANSWER_CODE` prints. Chosen to be
#: implausible in ordinary output — a collision would be read as an answer.
ANSWER_MARKER = "__DUCK_ANSWER__"

#: Trusted worker-to-host callback names. Payloads remain model-authored and
#: are validated by :class:`duck.python_tool.DuckPythonTool` before use.
CHECK_FAILURE_HOST_FUNCTION = "duck_check_failure"
CHECK_SUCCESS_HOST_FUNCTION = "duck_check_success"
PYTHON_CALL_INDEX_HOST_FUNCTION = "duck_python_call_index"
SHOW_HOST_FUNCTION = "duck_show"

#: The names the task occupies, in the order the prompt introduces them.
#:
#: ``in``/``out`` rather than ``question``/``answer``: the model writes these
#: names in every line of code it produces, so short pays twice, and the
#: analogy the task *is* — ``test_out`` follows from ``test_in`` the way
#: ``train_out`` follows from ``train_in`` — is carried by the names
#: themselves.
TASK_NAMES = ("train_in", "train_out", "test_in", "test_out")

#: Substituted with ``True``/``False`` when the kernel source is built, so the
#: layout is a literal in the function rather than a global the model could
#: read or a parameter it could pass.
_ROW_NUMBERS_MARK = "__DUCK_ASCII_ROW_NUMBERS__"
#: The sentence the docstring carries only when rows are numbered — the model
#: reads it through ``help(ascii_grid)``, so it has to describe what prints.
_ROW_NUMBERS_DOC_MARK = "__DUCK_ASCII_ROW_NUMBERS_DOC__"
_ROW_NUMBERS_DOC = (
    "\n\n    Each row is prefixed with its index in the array you passed and a\n"
    "    tab, so on a crop add the slice's start."
)

#: Substituted with ``True``/``False`` when the kernel source is built: whether
#: a passing ``check`` prints its report and the one-line reminder to record
#: the answer. A bare ``check(fn)`` statement in exec mode otherwise echoes
#: nothing, and a model that sees no output reads the pass as a failure.
_CHECK_PRINTS_MARK = "__DUCK_CHECK_PRINTS_ON_PASS__"
_CHECK_PASS_HINT_MARK = "__DUCK_CHECK_PASS_HINT__"
#: What a printed pass is followed by, outside mandatory mode. Mandatory mode
#: submits on its own, so there the reminder would be wrong.
_CHECK_PASS_HINT = (
    "record it: apply the function you just checked to each test_in[i], and assign its result to test_out[i][0]."
)

#: ``ascii_grid``, defined in the kernel so the model can call it. Digits
#: only: ARC-AGI-2 has ten colours, so every value is its own character and
#: the whale's letters for 10-15 have nothing to name here.
_ASCII_GRID_SOURCE = '''
def ascii_grid(arr):
    """Render a 2-D array of colour values as text, one character per cell.

    Values 0-9 print as their own digit, so a row is a string of digits and
    the grid is those rows joined by newlines. Takes an ndarray, a list of
    lists, or anything np.array() accepts.__DUCK_ASCII_ROW_NUMBERS_DOC__
    """
    import numpy as _np
    a = _np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("ascii_grid expects a 2-D array, got %dD." % a.ndim)
    if a.size and (a.min() < 0 or a.max() > 9):
        raise ValueError(
            "ascii_grid expects colour values 0-9, got values from %d to %d." % (a.min(), a.max())
        )
    _lut = _np.array(list("0123456789"))
    _rows = ["".join(row) for row in _lut[a.astype(_np.int64)]]
    if __DUCK_ASCII_ROW_NUMBERS__:
        _rows = ["%d\\t%s" % (_i, _r) for _i, _r in enumerate(_rows)]
    return "\\n".join(_rows)
'''


def _ascii_grid_source(*, row_numbers: bool) -> str:
    """The kernel's ``ascii_grid``, with this agent's layout baked in."""
    return _ASCII_GRID_SOURCE.replace(_ROW_NUMBERS_MARK, repr(row_numbers)).replace(
        _ROW_NUMBERS_DOC_MARK, _ROW_NUMBERS_DOC if row_numbers else ""
    )


def grid_text(grid: taaf.grid.Grid, *, row_numbers: bool) -> str:
    """One ARC grid as text rows, the way the kernel's ``ascii_grid`` renders it.

    The host's twin of :data:`_ASCII_GRID_SOURCE`, and the only renderer the
    host has: the task prompt's digit rows and the failed-check feedback both
    come through here, so a layout the kernel prints and a layout the prompt
    shows cannot drift apart. ``row_numbers`` prefixes each row with its index
    and a tab — the shape every other numbered text a model reads has.
    """
    rows = ["".join(str(int(value)) for value in row) for row in grid.data]
    return "\n".join(f"{index}\t{row}" for index, row in enumerate(rows)) if row_numbers else "\n".join(rows)


_GRID_VALIDATION_SOURCE = '''
def _duck_make_grid_validator():
    """Build a validator over the trusted NumPy module."""
    _np = _duck_np

    def checked_grid(value, label, max_side, n_colors):
        """Validate one model-produced ARC grid and return int64 values."""
        try:
            array = _np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("%s must be a rectangular 2-D grid" % label) from exc
        if _np.issubdtype(array.dtype, _np.floating):
            if not _np.all(_np.isfinite(array)) or not _np.all(array == _np.round(array)):
                raise ValueError("%s contains non-integer values" % label)
        elif not _np.issubdtype(array.dtype, _np.integer):
            raise ValueError("%s must contain integers" % label)
        if array.ndim != 2 or array.size == 0:
            raise ValueError("%s must be a non-empty 2-D grid" % label)
        array = array.astype(_np.int64)
        if array.shape[0] > max_side or array.shape[1] > max_side:
            raise ValueError("%s exceeds %dx%d" % (label, max_side, max_side))
        if array.min() < 0 or array.max() >= n_colors:
            raise ValueError("%s has colours outside 0..%d" % (label, n_colors - 1))
        return array

    return checked_grid


_duck_checked_grid = _duck_make_grid_validator()
del _duck_make_grid_validator
'''

# Optional assignment leaves hand-built answers available; mandatory mode
# submits validated predictions through the host instead.
_CHECK_SOURCE = '''
class PairCheck:
    """The result of applying a candidate function to one training pair."""

    __slots__ = ("index", "ok", "error", "expected_shape", "got_shape",
                 "mismatch_count", "first_mismatches")

    def __init__(self, **values):
        for name in self.__slots__:
            setattr(self, name, values.get(name))

    def __repr__(self):
        if self.error is not None:
            return "pair %d: ERROR %s" % (self.index, self.error)
        if self.ok:
            return "pair %d: OK" % self.index
        if self.expected_shape != self.got_shape:
            return "pair %d: shape %r != expected %r" % (
                self.index, self.got_shape, self.expected_shape
            )
        return "pair %d: %d mismatched cells, first: %r" % (
            self.index, self.mismatch_count, self.first_mismatches
        )


class CheckReport:
    """The complete, programmatically inspectable result of ``check(fn)``."""

    __slots__ = ("ok", "pairs", "test_error")

    def __init__(self, pairs):
        self.pairs = pairs
        self.ok = all(pair.ok for pair in pairs)
        self.test_error = None

    @property
    def passed(self):
        return self.ok and self.test_error is None

    def __repr__(self):
        if self.test_error is not None:
            header = "check: TRAIN PAIRS OK, fn FAILED ON A TEST INPUT - nothing submitted"
        else:
            header = "check: ALL TRAIN PAIRS OK" if self.ok else "check: FAILED"
        lines = [header, *(repr(pair) for pair in self.pairs)]
        if self.test_error is not None:
            lines.append(self.test_error)
        return "\\n".join(lines)


class CheckError(Exception):
    """A failed candidate check; ``report`` holds the full ``CheckReport``."""

    def __init__(self, report):
        super().__init__(repr(report))
        self.report = report


def _duck_make_check(
    inputs, outputs, test_inputs, max_side, n_colors, submit_on_pass,
    host_call, failure_host_function, success_host_function, call_index_host_function,
    check_and_submit=False, n_attempts=2
):
    """Build ``check`` over private, immutable snapshots of the task."""
    _np = _duck_np
    _checked_grid = _duck_checked_grid
    _inputs = tuple(_np.array(grid, dtype=_np.int8, copy=True) for grid in inputs)
    _outputs = tuple(_np.array(grid, dtype=_np.int8, copy=True) for grid in outputs)
    _test_inputs = tuple(_np.array(grid, dtype=_np.int8, copy=True) for grid in test_inputs)
    for _grid in (*_inputs, *_outputs, *_test_inputs):
        _grid.flags.writeable = False
    active_call_index = None
    checked = False
    def check(fn):
        """Run ``fn`` over every training pair and report exact differences.

        A failure raises ``CheckError``. Its ``report.pairs`` remains
        available to caught exceptions, including mismatch counts and the
        first five ``(row, col, expected, got)`` tuples. A pass returns the
        report. In mandatory mode, a pass submits ``fn`` applied to every
        test input. One check is allowed per Python tool call.
        """
        nonlocal active_call_index, checked
        if not callable(fn):
            raise TypeError("check(fn) expects a callable taking an input grid array.")
        call_index = host_call(call_index_host_function) if host_call is not None else 0
        if call_index != active_call_index:
            active_call_index = call_index
            checked = False
        if checked:
            raise RuntimeError("check() may be called only once per Python tool call.")
        checked = True
        pairs = []
        got_by_index = {}
        for index, (candidate_input, expected) in enumerate(zip(_inputs, _outputs)):
            entry = dict(
                index=index,
                ok=False,
                error=None,
                expected_shape=tuple(int(v) for v in expected.shape),
                got_shape=None,
                mismatch_count=None,
                first_mismatches=None,
            )
            try:
                got = _checked_grid(
                    fn(candidate_input.copy()),
                    "fn(train_in[%d])" % index,
                    max_side,
                    n_colors,
                )
            except Exception as exc:
                entry["error"] = "%s: %s" % (exc.__class__.__name__, exc)
                pairs.append(PairCheck(**entry))
                continue
            got_by_index[index] = got
            entry["got_shape"] = tuple(int(v) for v in got.shape)
            if got.shape != expected.shape:
                pairs.append(PairCheck(**entry))
                continue
            mismatches = _np.argwhere(got != expected)
            entry["mismatch_count"] = int(len(mismatches))
            entry["first_mismatches"] = [
                (int(row), int(col), int(expected[row, col]), int(got[row, col]))
                for row, col in mismatches[:5]
            ]
            entry["ok"] = entry["mismatch_count"] == 0
            pairs.append(PairCheck(**entry))
        report = CheckReport(pairs)
        if report.ok:
            if submit_on_pass or check_and_submit:
                predictions = []
                for index, test_input in enumerate(_test_inputs):
                    try:
                        prediction = _checked_grid(
                            fn(test_input.copy()),
                            "fn(test_in[%d])" % index,
                            max_side,
                            n_colors,
                        )
                    except Exception as exc:
                        report.test_error = "test input %d: %s: %s" % (
                            index, exc.__class__.__name__, exc
                        )
                        raise CheckError(report) from None
                    predictions.append(prediction.tolist())
                if submit_on_pass and host_call is not None:
                    host_call(success_host_function, {"grids": predictions})
                elif check_and_submit:
                    answers = globals().get("test_out")
                    rows = answers if isinstance(answers, (list, tuple)) else []
                    # Stage every replacement before touching saved answers.
                    replacements = []
                    for i, prediction in enumerate(predictions):
                        row = rows[i] if i < len(rows) else None
                        alternatives = list(row[1:n_attempts]) if isinstance(row, (list, tuple)) else []
                        alternatives += [None] * (n_attempts - 1 - len(alternatives))
                        replacements.append([prediction, *alternatives])
                    if isinstance(answers, list):
                        answers[:] = replacements
                    else:
                        globals()["test_out"] = replacements
            if __DUCK_CHECK_PRINTS_ON_PASS__:
                print(repr(report))
                if not submit_on_pass and not check_and_submit:
                    print(__DUCK_CHECK_PASS_HINT__)
            return report
        failed = [pair for pair in pairs if not pair.ok]
        payload = {
            "total_failed": len(failed),
            "pairs": [
                {
                    "index": pair.index,
                    "got": got_by_index[pair.index].tolist() if pair.index in got_by_index else None,
                }
                for pair in failed
            ],
        }
        if host_call is not None:
            host_call(failure_host_function, payload)
        raise CheckError(report)

    if check_and_submit and not submit_on_pass:
        check.__doc__ += (
            "\\nWith check_and_submit, a pass saves validated test predictions in test_out's first slots. "
            "Missing or malformed answer rows are rebuilt, preserving available alternative slots. "
            "An unusable outer container is replaced. Failed validation leaves saved answers unchanged."
        )
    return check
'''

_SHOW_SOURCE = '''
def _duck_make_show(
    max_shows, max_side, n_colors, host_call, host_function, call_index_host_function
):
    """Build host-bounded ``show`` over one captured grid validator."""
    _checked_grid = _duck_checked_grid
    active_call_index = None
    shown = 0

    def show(grid, label=""):
        """Queue a validated grid for host-side rendering after this call."""
        nonlocal active_call_index, shown
        call_index = host_call(call_index_host_function) if host_call is not None else 0
        if call_index != active_call_index:
            active_call_index = call_index
            shown = 0
        if shown >= max_shows:
            raise RuntimeError("show() limit of %d per tool call reached." % max_shows)
        array = _checked_grid(grid, "show(grid)", max_side, n_colors)
        payload = {"grid": array.tolist(), "label": str(label)}
        accepted = host_call(host_function, payload) if host_call is not None else True
        if not accepted:
            raise RuntimeError("show() limit of %d per tool call reached." % max_shows)
        shown += 1

    return show
'''


#: Before numpy is imported anywhere in the kernel: OpenBLAS sizes its thread
#: pool to the machine's cores, and one thread is what a kernel sharing the box
#: with hundreds of others should have.
_ONE_BLAS_THREAD = (
    "import os as _duck_os\n"
    "_duck_os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
    "_duck_os.environ['OMP_NUM_THREADS'] = '1'\n"
    "del _duck_os\n"
)
#: The working directory last on ``sys.path`` before anything is imported, not
#: first where the worker puts it: a file of the model's named like a module
#: the prelude imports would otherwise be imported in its place, and a kernel
#: restarted in that directory would fail before the model's code ran.
#: ``kernel_workspace.install`` keeps it there by its real path.
_WORKSPACE_PATH_LAST = (
    "import sys as _duck_sys\n"
    "_duck_sys.path[:] = [_p for _p in _duck_sys.path if _p not in ('', '.')] + ['']\n"
    "del _duck_sys\n"
)
#: The kernel-side calls into ``duck.kernel_workspace``, each reached through
#: the import system so that it binds nothing in the cells' globals.
_WORKSPACE_INSTALL = '__import__("duck.kernel_workspace", fromlist=["install"]).install()\n'
_WORKSPACE_CAPTURE = '__import__("duck.kernel_workspace", fromlist=["capture_surface"]).capture_surface(globals())\n'
#: Run privately before every model cell (``kernel_workspace.housekeeping``).
HOUSEKEEPING_CODE = '__import__("duck.kernel_workspace", fromlist=["housekeeping"]).housekeeping(globals())\n'
#: Run privately between rounds; prints the session listing's JSON as its last line.
SESSION_LISTING_CODE = (
    'print(__import__("duck.kernel_workspace", fromlist=["session_listing"]).session_listing(globals()))\n'
)


def prelude(
    blocked_dirs: list[str] | None = None,
    *,
    include_check: bool = False,
    include_show: bool = False,
    row_numbers: bool = False,
    check_prints_on_pass: bool = False,
    numpy_scalar_repr: str = "legacy",
    file_tools: bool = False,
) -> str:
    """One-time kernel setup, as source text.

    The sandbox goes in first, before any name the model might use exists:
    the patches are what keep a later ``open()`` out of our source, so
    installing them after building a namespace would leave a window.

    **Nothing here may import the agent stack.** The kernel is an isolated
    worker with a startup deadline, and pulling ``platypus`` / ``taaf`` /
    ``ullm`` into it blows straight through that — which is why the blocked
    directories are resolved in the host and baked in as literals rather
    than discovered by importing the packages they belong to. ``duck.sandbox``
    itself imports only the standard library, and ``duck/__init__`` defers
    everything, so ``import duck.sandbox`` stays cheap.

    ``row_numbers`` is baked into ``ascii_grid`` as a literal, and into its
    docstring, which the model reads through ``help()``. Not a parameter the
    model may pass: it is the agent's layout for every render in this kernel,
    so a sweep over it changes what the model sees without the model choosing.

    ``check_prints_on_pass`` is baked into ``check`` the same way: whether a
    pass prints its report and the reminder to record the answer, so a bare
    ``check(fn)`` statement is never silent.

    ``numpy_scalar_repr`` is how the kernel reprs a NumPy scalar inside a
    container — what a printed row or coordinate list is made of. ``legacy``
    sets ``legacy='1.25'``, so ``list(grid[0])`` prints ``[5, 5, 5]``;
    ``np2`` leaves NumPy 2's own ``[np.int8(5), np.int8(5), np.int8(5)]``.
    The model reads this output; the modern repr triples its cost and reads
    as noise. The bulk imports only save the model tool calls — imports are
    cached in ``sys.modules``, so they cost nothing at run time.

    ``file_tools`` makes the working directory's files importable as the
    model's modules (``duck.kernel_workspace``), reached through the import
    system so that it binds no name a snippet could list or shadow.
    """
    if numpy_scalar_repr not in {"np2", "legacy"}:
        raise ValueError(f"unknown numpy_scalar_repr {numpy_scalar_repr!r}")
    dirs = duck.sandbox.host_blocked_dirs() if blocked_dirs is None else blocked_dirs
    source = (
        _ONE_BLAS_THREAD + (_WORKSPACE_PATH_LAST if file_tools else "") + "import duck.sandbox as _duck_sandbox\n"
        f"_duck_sandbox.install({dirs!r})\n"
        "del _duck_sandbox\n"
        + (_WORKSPACE_INSTALL if file_tools else "")
        + "import numpy as np\n"
        + ("np.set_printoptions(legacy='1.25')\n" if numpy_scalar_repr == "legacy" else "")
        + "import collections, copy, functools, itertools, json, math, os, re, sys\n"
        "from collections import Counter, defaultdict, deque\n"
    )
    harness = _ascii_grid_source(row_numbers=row_numbers)
    if include_check or include_show:
        harness += "import numpy as _duck_np\n" + _GRID_VALIDATION_SOURCE
    if include_check:
        harness += _CHECK_SOURCE.replace(_CHECK_PRINTS_MARK, repr(check_prints_on_pass)).replace(
            _CHECK_PASS_HINT_MARK, repr(_CHECK_PASS_HINT)
        )
    if include_show:
        harness += _SHOW_SOURCE
    # Compiled under a name of its own rather than run as the cell it sits
    # in: a frame of ``check``, ``show`` or ``ascii_grid`` then reads as ours
    # (``duck.errors``), not as a ``<repl>`` frame of the model's quoting a
    # line of this setup, and no source of it is there for inspect to read.
    return source + f"exec(compile({harness!r}, {HARNESS_SETUP_FILE!r}, 'exec'), globals())\n"


def _grid_literal(grid: taaf.game.Grid) -> str:
    """One grid as a numpy constructor call, values inline.

    Nested lists rather than a pickle or a base64 buffer: a grid is at most
    30x30, so the text is small, and the setup code stays something a person
    can read in a trace when an answer comes back wrong.
    """
    return f"np.array({grid.to_lists()!r}, dtype=np.int8)"


def task_setup_code(
    view: taaf.game.GameView,
    *,
    include_check: bool = False,
    check_mandatory: bool = False,
    check_and_submit: bool = False,
    include_show: bool = False,
    max_shows: int = 6,
) -> str:
    """Bind the task into the kernel namespace as bare arrays.

    The four names in :data:`TASK_NAMES` give the model the task the way it
    would get it in a notebook. ``test_out`` is pre-built with a slot per
    question so the model can assign into it without first constructing the
    shape — the two attempts per question are the harness's business, not a
    structure the model should have to remember to build. The task's arrays
    are read-only: a function that writes into its input, called on one of
    them as ``test_out[i][0] = fn(test_in[i])`` calls it, raises on the line
    that writes, where it would otherwise change the task under every later
    read. The lists that hold them are the model's to rebind. The optional check
    runtime closes over immutable copies of the training pairs and test
    inputs. ``check_mandatory`` makes a passing check send validated test
    predictions to the host. ``check_and_submit`` instead updates the first
    ``test_out`` slots, preserving the remaining attempts. The setup ends by
    recording what it bound (``duck.kernel_workspace.capture_surface``): the
    names the session listing leaves out as the harness's, with the file tools
    on or off, and under them what every module the model writes starts with.
    """
    given_in = ", ".join(_grid_literal(q) for q, _ in view.given)
    given_out = ", ".join(_grid_literal(a) for _, a in view.given)
    questions = ", ".join(_grid_literal(q) for q in view.questions)
    source = ""
    if include_check or include_show:
        source += "_duck_host_call = globals().pop('__host_call_trusted__', None)\n"
    source += (
        f"train_in = [{given_in}]\n"
        f"train_out = [{given_out}]\n"
        f"test_in = [{questions}]\n"
        f"test_out = [[None] * {taaf.game.N_ATTEMPTS} for _ in test_in]\n"
        "for _duck_grid in (*train_in, *train_out, *test_in):\n"
        "    _duck_grid.flags.writeable = False\n"
        "_duck_grid = None\n"
        "del _duck_grid\n"
    )
    if include_check:
        source += (
            "check = _duck_make_check("
            f"train_in, train_out, test_in, {taaf.grid.MAX_SIDE}, {taaf.grid.N_COLORS}, "
            f"{check_mandatory!r}, _duck_host_call, {CHECK_FAILURE_HOST_FUNCTION!r}, "
            f"{CHECK_SUCCESS_HOST_FUNCTION!r}, {PYTHON_CALL_INDEX_HOST_FUNCTION!r}, "
            f"{check_and_submit!r}, {taaf.game.N_ATTEMPTS})\n"
        )
    if include_show:
        source += (
            "show = _duck_make_show("
            f"{max_shows}, {taaf.grid.MAX_SIDE}, {taaf.grid.N_COLORS}, "
            f"_duck_host_call, {SHOW_HOST_FUNCTION!r}, {PYTHON_CALL_INDEX_HOST_FUNCTION!r})\n"
        )
    if include_check or include_show:
        source += "del _duck_host_call\n"
    return source + _WORKSPACE_CAPTURE


READ_ANSWER_CODE = f"""\
def _duck_read_answer():
    import json as _json
    import numpy as _np

    def _one(_value):
        if _value is None:
            return {{"state": "missing"}}
        try:
            _arr = _np.asarray(_value)
        except Exception as _exc:
            return {{"state": "bad", "why": "is not array-like (%s)" % type(_value).__name__}}
        if _arr.dtype == object or not _np.issubdtype(_arr.dtype, _np.number):
            return {{"state": "bad", "why": "holds %s, not numbers" % _arr.dtype}}
        if _arr.ndim != 2:
            return {{"state": "bad", "why": "has shape %r; a grid is 2-D" % (_arr.shape,)}}
        if _np.issubdtype(_arr.dtype, _np.floating):
            if not _np.all(_np.isfinite(_arr)):
                return {{"state": "bad", "why": "holds non-finite values"}}
            if not _np.all(_arr == _np.round(_arr)):
                return {{"state": "bad", "why": "holds non-integer values"}}
        return {{"state": "ok", "grid": _arr.astype("int64").tolist()}}

    _answers = globals().get("test_out")
    if not isinstance(_answers, (list, tuple)):
        _payload = {{"bound": False, "why": "test_out is %s, not a list" % type(_answers).__name__}}
    else:
        _rows = []
        for _entry in _answers:
            if isinstance(_entry, (list, tuple)):
                _rows.append([_one(_slot) for _slot in _entry])
            else:
                _rows.append([_one(_entry)])
        _payload = {{"bound": True, "rows": _rows}}
    print("{ANSWER_MARKER}" + _json.dumps(_payload))

_duck_read_answer()
del _duck_read_answer
"""
"""Read ``test_out`` back out, as one marker line of JSON.

Runs as a private execution after every python call, so it must survive
whatever the model has done to the namespace: ``test_out`` may be
rebound to any object, deleted, or filled with anything at all. Every
failure is reported rather than raised — a model that put a string in a
slot should get told so, not crash its own kernel.

Judgement is split deliberately. Everything decidable from the value alone
lives here (is it array-like, 2-D, integral); everything that is an
ARC-AGI-2 rule — sides within 30, values 0..9 — is checked host-side in
``agent.py``, next to the wording that explains it. The float check is what
"cast to int8 without rounding" means in practice: 3.0 is a 3, 3.5 is a
mistake worth naming.
"""


def parse_answer_payload(output: str) -> dict[str, object] | None:
    """Recover the marker line :data:`READ_ANSWER_CODE` printed.

    ``None`` when there is no such line — the kernel died, timed out, or was
    interrupted — which the caller treats as "no answer this round" rather
    than as a malformed one.
    """
    for line in reversed(output.splitlines()):
        if not line.startswith(ANSWER_MARKER):
            continue
        try:
            parsed: Any = json.loads(line[len(ANSWER_MARKER) :])
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return {str(key): value for key, value in cast("dict[Any, Any]", parsed).items()}
    return None


def grid_from_rows(rows: object) -> npt.NDArray[np.int64] | None:
    """The kernel's nested-list grid as an array, or ``None`` if it is not
    one. Defensive because the payload crossed a JSON boundary from a
    namespace the model controls."""
    if not isinstance(rows, list) or not rows:
        return None
    try:
        arr = np.array(cast("list[Any]", rows))
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.integer):
        return None
    return arr.astype(np.int64)


__all__ = [
    "ANSWER_MARKER",
    "CHECK_FAILURE_HOST_FUNCTION",
    "HOUSEKEEPING_CODE",
    "PYTHON_CALL_INDEX_HOST_FUNCTION",
    "READ_ANSWER_CODE",
    "SESSION_LISTING_CODE",
    "SHOW_HOST_FUNCTION",
    "TASK_NAMES",
    "grid_from_rows",
    "parse_answer_payload",
    "prelude",
    "task_setup_code",
]
