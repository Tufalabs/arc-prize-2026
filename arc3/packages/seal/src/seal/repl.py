"""GameREPL: the persistent Python kernel the model plays the game from.

A thin subclass of ullm's ``PythonTool`` that owns everything the model's
kernel sees before and around its own code: the bootstrap that builds the
``__main__`` namespace and defines ``submit_action``, and the per-call
prepend that restores the canonical ``history`` / ``available_actions`` /
cwd and pulls this level's state from the agent.

Every line of ``initial_code`` and ``prepend_code`` is code the model's
kernel runs and whose output the model reads, so a change to either is a
prompt change.

The whale pushes state into the kernel (``inject_game_state``) after each
action; the seal pulls it at every cell start and inside ``submit_action``,
which is what lets a cell act, look and act again. The host calls behind
that (``get_state``, ``action``) are the agent's, bound per move.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import re
import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import ullm.conv
import ullm.tools.python as python_tool

import seal.errors
import seal.harness.game
import seal.workspace
from framediff import Decomposer
from framediff.animation import MAX_CHARS
from seal import prompts
from seal.frame_delta import (
    ANIMATION_CUT_CHARS,
    MAX_BITS_UNDERSTOOD,
    UNCAPPED_END,
    UNCAPPED_START,
    AnimationDescription,
    default_decomposer,
)
from seal.palette import AsciiSymbols

#: The longest ``plan`` a call keeps (``SealAgent.plan_argument``); the prompt
#: states the same limit.
PLAN_MAX_CHARS = 500

#: The most of a failed listing's last line the model is shown. The line is an
#: exception's, which can carry a whole value, and it reaches the rebuilt
#: conversation past any tool-output cap, where an unbounded one could fill
#: the room the summary was installed to make. Its head is kept: the type.
LISTING_ERROR_MAX_CHARS = 500


class SessionInspectionError(RuntimeError):
    """The kernel did not produce its optional compaction inventory.

    The message is what the model is shown: the cell's last line of output,
    a traceback's exception line, cut at ``LISTING_ERROR_MAX_CHARS``.
    ``output`` keeps the cell's output whole for
    the record; the harness's is not rewritten for the model, so its frames
    and paths stay off the prompt."""

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


def plan_only(parameters: dict[str, Any], arguments: dict[str, Any]) -> bool:
    """Whether a call gives its ``plan`` and nothing else.

    ullm's lone-argument remap would run that plan as the call's code; the
    seal skips the remap for it, so the schema refuses the call for its
    missing ``code`` and the model hears so."""
    return list(arguments) == ["plan"] and "plan" in parameters.get("properties", {})


# Tripwire for source-access detection
_SOURCE_TRIPWIRE = "__SANDBOX_SRC_TRIPWIRE_a4c1f70d__"


def _seal_tool_parameters() -> dict[str, Any]:
    """The seal's ``python`` tool schema (one ``code`` string)."""
    return {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": prompts.PYTHON_TOOL_CODE_PARAM_DESCRIPTION},
        },
        "required": ["code"],
    }


def cap_around_uncapped(text: str, cap: Callable[[str], str], *, head: int, tail: int) -> str:
    """``cap`` applied to the model's own output alone, with what the harness marked uncapped kept whole.

    ``cap`` returns its text unchanged, or its first ``head`` and last
    ``tail`` characters around a note of what it removed. A marked text
    (between ``frame_delta.UNCAPPED_START`` and ``UNCAPPED_END``) stays where
    it was printed when that is in what the cap keeps, and follows the note,
    in order, when it was printed in what the cap removed; the marks go
    either way. Output the cap leaves alone comes back as if nothing had been
    marked.
    """
    pieces = re.split(f"{UNCAPPED_START}(.*?){UNCAPPED_END}", text, flags=re.DOTALL)
    own_pieces = [piece.replace(UNCAPPED_START, "").replace(UNCAPPED_END, "") for piece in pieces[0::2]]
    marked = pieces[1::2]
    own = "".join(own_pieces)
    capped = cap(own)
    at = list(zip(itertools.accumulate(len(piece) for piece in own_pieces[:-1]), marked))
    if capped == own:
        return _inserted(own, at)
    cut_from, cut_to = head, len(own) - tail
    assert capped.startswith(own[:cut_from]) and capped.endswith(own[cut_to:]), "the cap keeps a head and a tail"
    note = capped[cut_from : len(capped) - tail]
    before = _inserted(own[:cut_from], [(offset, piece) for offset, piece in at if offset <= cut_from])
    removed = "".join(f"{piece}\n" for offset, piece in at if cut_from < offset < cut_to)
    after = _inserted(own[cut_to:], [(offset - cut_to, piece) for offset, piece in at if offset >= cut_to])
    return before + note + removed + after


def _inserted(base: str, at: list[tuple[int, str]]) -> str:
    """``base`` with each piece inserted at its offset, the offsets ascending."""
    out: list[str] = []
    last = 0
    for offset, piece in at:
        out += [base[last:offset], piece]
        last = offset
    return "".join(out) + base[last:]


#: How ullm's tool opens the result of a cell it killed because the interrupt
#: at its time limit did not stop it.
_KILLED_AT_LIMIT = "Execution timed out after"


#: How many times the descendants are read and killed: a process can start
#: another between the read and the kill, and the next round finds it.
_KILL_ROUNDS = 3


def _descendants(pid: int) -> list[int]:
    """Every process descended from ``pid`` that has not exited, read from
    ``/proc``. A zombie is left out: it is already dead, and stays listed
    under its parent until the parent reaps it."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as stat:
                fields = stat.read()
        except OSError:
            continue
        # The command name sits in parentheses and may hold spaces or ")";
        # the state and the parent pid are the two fields after its closing one.
        state, parent = fields.rsplit(")", 1)[1].split()[:2]
        if state != "Z":
            children.setdefault(int(parent), []).append(int(entry))
    found: list[int] = []
    frontier = [pid]
    while frontier:
        frontier = [child for parent in frontier for child in children.get(parent, [])]
        found += frontier
    return found


def kill_descendants(pid: int) -> int:
    """SIGKILL every process descended from ``pid``, ``pid`` itself spared, and
    return how many were signalled."""
    killed: set[int] = set()
    for _ in range(_KILL_ROUNDS):
        strays = _descendants(pid)
        if not strays:
            break
        for stray in strays:
            try:
                os.kill(stray, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                continue
            killed.add(stray)
    return len(killed)


@dataclass
class GameREPL(python_tool.PythonTool):
    """PythonTool subclass with the runtime surface pre-loaded and the seal's state pull."""

    name: str = "python"
    description: str = prompts.PYTHON_TOOL_DESCRIPTION
    parameters: dict[str, Any] = field(default_factory=_seal_tool_parameters)
    clear_state_on: Literal["run", "turn", "never"] = "never"

    timeout: float = 30.0

    #: The response budget the model is told about, in tokens; the output
    #: cap is this many characters times four, split evenly between the head
    #: and the tail so what survives is about what the prompt promises. Text
    #: the harness prints marked uncapped does not count against it
    #: (:func:`cap_around_uncapped`).
    tool_output_tokens: int = 1024
    #: How ``ascii_grid`` lays a board out, decided by the agent and installed
    #: into the kernel at start (``SealAgent.ascii_row_numbers`` and
    #: ``ascii_column_group`` say why). Changed through :meth:`configure_ascii`
    #: so the bootstrap text moves with them.
    ascii_row_numbers: bool = True
    ascii_column_group: int | None = None
    #: The alphabet a board prints and reads back in
    #: (``SealAgent.ascii_symbols``), installed with the layout.
    ascii_symbols: AsciiSymbols = "hex"
    # True = a fresh process per call: module state, globals and tmpdir are
    # dropped before every snippet. False keeps them, and the
    # AGENT owns the reset cadence instead (SealAgent.persistence).
    reset_before_each_call: bool = True

    #: The kernel half of ``SealAgent.use_world_model``: what the bootstrap
    #: tells the gate, and whether ``test_world_model`` is a name at all.
    use_world_model: Literal["none", "optional", "mandatory"] = "mandatory"
    #: The kernel halves of ``SealAgent.settled_delta``,
    #: ``animation_description``, ``animation_max_chars`` and
    #: ``animation_cut_chars``: what is described to the model as a block of
    #: code, and how long an animation's block may run.
    settled_delta: bool = True
    animation_description: AnimationDescription = "available"
    animation_max_chars: int = MAX_CHARS
    animation_cut_chars: int = ANIMATION_CUT_CHARS
    #: The kernel half of ``SealAgent.diff_image``: whether the call's newest
    #: frame is compared with its baseline for the image after it.
    diff_image: bool = False
    #: The kernel halves of ``SealAgent.decomposer`` and
    #: ``max_bits_understood``: the search every block comes from, and its
    #: budget once the level is understood.
    decomposer: Decomposer = field(default_factory=default_decomposer)
    max_bits_understood: float = MAX_BITS_UNDERSTOOD
    #: The kernel half of ``SealAgent.crash_on_framediff_timeout``.
    crash_on_framediff_timeout: bool = False
    #: The kernel half of ``SealAgent.segmentation``: whether ``segment()``,
    #: ``Segmentation`` and ``Node`` are on the surface.
    segmentation: bool = True
    #: The kernel half of ``SealAgent.confine_filesystem``.
    confine_filesystem: bool = True
    #: The kernel half of ``SealAgent.expose_pixel_tolerance``: whether the
    #: model gets ``set_pixel_tolerance``.
    pixel_tolerance_offered: bool = False
    #: The kernel half of ``SealAgent.plan_argument``: whether a ``Turn`` has
    #: a ``plan``, and whether a call's ``plan`` is read.
    plan_argument: prompts.PlanArgument = "none"
    #: The running call's ``plan``, cut to :data:`PLAN_MAX_CHARS`, for the
    #: turns it plays; None between calls and without one.
    call_plan: str | None = field(default=None, init=False, repr=False)
    #: Whether the running call wrote its ``plan`` ahead of its ``code``,
    #: which the schema only hints at; None beside ``call_plan``.
    call_plan_first: bool | None = field(default=None, init=False, repr=False)
    #: How many python calls this tool has started. Counted host-side, so
    #: nothing the model runs can move it: the host keys its one exploratory
    #: move per call on it.
    cells_started: int = 0
    #: ``cells_started`` when the kernel was last reset (:attr:`session_used`).
    cells_at_reset: int = field(default=0, init=False)
    #: Set for the span of a harness cell: its output is read by the agent, not
    #: the model, so neither the model's output cap nor its rewriting applies.
    _harness_output: bool = field(default=False, init=False, repr=False)
    #: Whether the last cell ran into its time limit: interrupted there, or
    #: killed when the interrupt did not stop it. Set by every cell, so the
    #: agent reads the model's right after its call, before a harness cell.
    timed_out: bool = field(default=False, init=False, repr=False)
    #: Whether the last cell ended with its kernel gone: killed when the
    #: interrupt at its limit did not stop it, or dead. The next cell starts
    #: on a fresh session.
    kernel_lost: bool = field(default=False, init=False, repr=False)
    #: How many processes the kernel had started and left running when the
    #: last cell ran into its limit; all of them were killed then.
    strays_killed: int = field(default=0, init=False, repr=False)

    # The `str` (not `str | None`) narrowing on initial_code / prepend_code is
    # the point: this REPL always has both, and every caller reads them as text.
    # Both are assembled in __post_init__.
    initial_code: str = field(init=False, default="")  # pyright: ignore[reportIncompatibleVariableOverride]

    def _build_initial_code(self) -> str:
        """The kernel's one-time setup, as source text."""
        return (
            # Before numpy is imported anywhere below: OpenBLAS sizes its
            # thread pool to the machine's cores, and under the worker's
            # address-space limit that allocation hangs the import on a box
            # with enough of them. One thread is also what a kernel sharing
            # the box with dozens of others should have.
            "import os as _ullm_os\n"
            "_ullm_os.environ['OPENBLAS_NUM_THREADS'] = '1'\n"
            "_ullm_os.environ['OMP_NUM_THREADS'] = '1'\n"
            "del _ullm_os\n"
            # The working directory goes last on sys.path, not first where the
            # worker puts it: a file of the model's named like a module Python
            # has would otherwise be imported in its place — by the harness's
            # own imports just below, too.
            "import sys as _ullm_sys\n"
            "_ullm_sys.path[:] = [_p for _p in _ullm_sys.path if _p != ''] + ['']\n"
            "del _ullm_sys\n"
            # Set up the REPL __main__ namespace from one source of truth in
            # seal.harness.game. Imported for its side effect: the model calls
            # help() on these objects, and pydoc names the module only when it
            # can find it in sys.modules under the name the object claims.
            "import seal.harness as _harness\n"
            "_harness.install()\n"
            "del _harness\n"
            "import runtime.game\n"
            "import runtime.game as _sg\n"
            # _rs is the private canonical store for REPL history /
            # available_actions — see seal/harness/_repl_state.py. Deliberately
            # kept off seal.harness.game so the LLM doesn't stumble on it.
            "import runtime._repl_state as _rs\n"
            # Stash the subprocess's spawn cwd (= the model's working
            # directory) so prepend_code can restore it if the LLM drifts
            # away with os.chdir in a cell.
            "import os as _ullm_os\n"
            "_rs.tmpdir = _ullm_os.getcwd()\n"
            "del _ullm_os\n"
            # The process settings every process the model's code runs in
            # shares (numpy's 1.x scalar repr, sorted directory listings), so
            # its code behaves in the search child as it does here.
            "_sg._configure_model_process()\n"
            # The names every snippet, every module the model writes and
            # world_model.py start with, from one place: the modules the model
            # would otherwise re-import in every cell, numpy, the surface.
            # Off only: the line is not there at all where segmentation is on.
            + ("_sg._configure_segmentation(enabled=False)\n" if not self.segmentation else "")
            + f"for _n, _v in _sg._preloaded_names(world_model={self.use_world_model != 'none'!r}, "
            f"required={self.use_world_model == 'mandatory'!r}).items():\n"
            "    globals()[_n] = _v\n"
            "del _n, _v\n"
            # history and its companions are set by the first prepend_code run.
            "history = None\n"
            "available_actions = []\n"
            # ``bind_host`` captures the trusted host_call primitive (installed
            # by the worker under a private name) in closure cells, so it is
            # never an LLM-visible global. The private name is deleted right
            # after, so executed cells cannot reach it by name.
            f"_sg._configure_ascii_grid(row_numbers={self.ascii_row_numbers!r}, "
            f"column_group={self.ascii_column_group!r}, symbols={self.ascii_symbols!r})\n"
            f"_sg._configure_world_model(enabled={self.use_world_model != 'none'!r})\n"
            f"_sg._configure_plan(enabled={self.plan_argument != 'none'!r})\n"
            "_rs.bind_host(__host_call_trusted__)\n"
            "del __host_call_trusted__\n"
            "submit_action = _sg._make_submit_action(_rs._submit, _rs.installer(globals()), _rs.wm_gate, _rs.game_actions)\n"
            "_rs.current_submit_action = submit_action\n"
            # A module the model writes gets the cell's preloaded names, and
            # the moving ones again after every action (see _repl_state).
            "_rs.install_import_hook()\n"
            # The world-model surface: the gate's mode, and — unless the mode
            # removes the surface entirely — the model's test_world_model().
            f"_rs.wm_gate.mode = {self.use_world_model!r}\n"
            f"_rs.settled_delta = {self.settled_delta!r}\n"
            f"_rs.diff_image = {self.diff_image!r}\n"
            f"_rs.animation_description = {self.animation_description!r}\n"
            f"_rs.animation_max_chars = {self.animation_max_chars!r}\n"
            f"_rs.animation_cut_chars = {self.animation_cut_chars!r}\n"
            f"_rs.decomposer = __import__('framediff').Decomposer(**{dataclasses.asdict(self.decomposer)!r})\n"
            f"_rs.max_bits_understood = {self.max_bits_understood!r}\n"
            f"_rs.crash_on_framediff_timeout = {self.crash_on_framediff_timeout!r}\n"
            f"_rs.python_timeout_s = {float(self.timeout)!r}\n"
            + ("describe_animation = _rs.describe_animation\n" if self.animation_description != "none" else "")
            + (
                "test_world_model = _rs.make_test_world_model()\n_rs.current_test_world_model = test_world_model\n"
                f"globals().update(_rs.make_setters(tolerance_offered={self.pixel_tolerance_offered!r}))\n"
                if self.use_world_model != "none"
                else ""
            )
            + ("_sg._confine_filesystem()\n" if self.confine_filesystem else "")
            + "_sg._block_source_access()\n"
            # The harness's own handles leave the namespace: a snippet that
            # lists its globals sees the surface, not the plumbing.
            "del runtime, _sg, _rs\n"
        )

    # Before every REPL call, ``_repl_state.housekeeping`` (its docstring
    # lists the steps). One call, reached through the import system rather
    # than a name: a name would sit in the cell's globals, where the model can
    # list it or shadow it (``_rs = 0``), and the call binds nothing.
    _prepend_housekeeping: str = (
        '__import__("runtime._repl_state", fromlist=["housekeeping"]).housekeeping(globals())\n'
    )

    # Assembled in __post_init__ from the housekeeping above; the "" default is
    # only there to give the field one, and is never observable.
    prepend_code: str = field(init=False, default="")  # pyright: ignore[reportIncompatibleVariableOverride]

    #: The pseudo-file the housekeeping compiles under. Not a real path, so
    #: ``linecache`` finds no source for it and CPython prints the frame
    #: without echoing our line — which is the point: the source it would
    #: echo names the private store and the adapters behind it.
    HARNESS_SETUP_FILE = "<harness-setup>"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.initial_code = self._build_initial_code()  # pyright: ignore[reportIncompatibleVariableOverride]
        # Compiled under its own name rather than prepended as bare lines.
        # Two things follow, both about what the model reads when its own
        # code raises. Our frames stop saying ``<repl>``, so they can be told
        # apart from the model's and dropped (``seal.errors``); and the
        # model's code starts at line 2 of the cell instead of line ~10, so
        # the line numbers in its traceback are (nearly) its own. No trailing
        # newline: the tool joins prepend and cell with one already.
        self.prepend_code = f"exec(compile({self._prepend_housekeeping!r}, {self.HARNESS_SETUP_FILE!r}, 'exec'))"  # pyright: ignore[reportIncompatibleVariableOverride]  # same `str` narrowing as the field
        # The cap the prompt states, kept rather than ullm's default: its
        # head and tail are sized for a 4.5k-character cap, and would keep a
        # quarter of what the model was told it may read.
        self.max_output_chars = max(256, self.tool_output_tokens * 4)
        self.truncate_head = self.max_output_chars // 2
        self.truncate_tail = self.max_output_chars - self.truncate_head

    def configure_ascii(self, *, row_numbers: bool, column_group: int | None, symbols: AsciiSymbols = "hex") -> None:
        """Set the ``ascii_grid`` layout and alphabet and rebuild the bootstrap
        that carries them. Takes effect at the next kernel start: a running
        kernel keeps the layout it was started with, which is what a
        level-long kernel needs."""
        self.ascii_row_numbers = row_numbers
        self.ascii_column_group = column_group
        self.ascii_symbols = symbols
        self.initial_code = self._build_initial_code()  # pyright: ignore[reportIncompatibleVariableOverride]

    def _maybe_truncate_output(self, text: str, *, is_error: bool) -> str:
        """Strip host paths BEFORE the output cap decides where to cut.

        The cap keeps a head and a tail measured in raw characters, and a raw
        traceback still carries absolute paths whose length is the machine's
        — the tmpdir under whatever $TMPDIR is, the checkout wherever it was
        cloned. Cut first and cleaned after, the surviving content and the
        marker's omitted-count both vary by machine, and a committed
        reference then replays everywhere except on a box whose paths have a
        different length — which is exactly what a CI runner is. Cleaning
        first makes the cut a function of the text the model was always going
        to see. The later pass in :meth:`_execute` still runs and is
        idempotent; it covers what this hook never sees.

        The cap measures and cuts the model's own output only: a text the
        harness marked uncapped is kept whole (:func:`cap_around_uncapped`).
        """
        if self._harness_output:
            # The agent parses a harness cell's output: cut or rewritten, a
            # large JSON line stops being JSON.
            return text
        cap = super()._maybe_truncate_output
        return cap_around_uncapped(
            seal.errors.for_model(text, model_dir=self._tmpdir, world_model=self.use_world_model != "none"),
            lambda own: cap(own, is_error=is_error),
            head=self.truncate_head,
            tail=self.truncate_tail,
        )

    def _reset_impl(self, keep_tmpdir_contents: bool) -> None:
        """The tool's reset, noting how many calls had started by then."""
        super()._reset_impl(keep_tmpdir_contents)
        self.cells_at_reset = self.cells_started

    @property
    def session_used(self) -> bool:
        """Whether the model has run a call since the kernel was last reset: a
        session it built and would lose, whether or not the kernel holding it
        is still alive. Counted here rather than read off the kernel, which a
        death between calls has already taken."""
        return self.cells_started > self.cells_at_reset

    async def _send_and_read(self, code: str, timeout: float | None = None) -> tuple[str, bool]:
        """One cell, noting whether it ran into its time limit. A cell that did
        takes the kernel's child processes with it: one the model started (a
        shell-out, a worker pool) outlives the interrupt, and can leave the
        kernel unable to run the next cell. A kernel killed outright leaves
        its children to the system, unreachable from here."""
        self.timed_out = self.kernel_lost = False
        self.strays_killed = 0
        output, is_error = await super()._send_and_read(code, timeout)
        self.kernel_lost = self._process is None
        if self.kernel_lost and output.startswith(_KILLED_AT_LIMIT):
            self.timed_out = True
        if self.timed_out and self._process is not None:
            self.strays_killed = kill_descendants(self._process.pid)
        return output, is_error

    def _format_response(
        self, response: dict[str, Any], timed_out: bool = False, timeout: float | None = None
    ) -> tuple[str, bool]:
        """The tool's own formatting, noting a cell interrupted at its limit."""
        self.timed_out = self.timed_out or timed_out
        return super()._format_response(response, timed_out, timeout)

    async def run_harness(self, code: str, *, timeout: float) -> tuple[str, bool]:
        """Run the harness's own code in the kernel: no housekeeping prepended,
        nothing counted as a call, and nothing of it shown to the model — so
        its output comes back whole, without the model's cap or rewriting.
        Returns ``(output, is_error)``.

        Takes the CPU permit a call takes, and only between calls — never from
        inside one, which already holds a permit."""
        semaphore = self.pool.cpu_semaphore()
        await semaphore.acquire()
        try:
            await self._ensure_kernel()
            self._harness_output = True
            try:
                return await self._send_and_read(code, timeout=timeout)
            finally:
                self._harness_output = False
        finally:
            semaphore.release()

    def working_files(self) -> list[tuple[str, int]]:
        """The kernel's working directory, as ``(name, bytes)`` pairs sorted
        by name; a subdirectory is listed by its name with a trailing ``/``
        and a size of -1. Read host-side: the directory is this tool's. The
        import system's ``__pycache__`` is not the model's and is left out."""
        tmpdir = self._tmpdir
        if tmpdir is None or not os.path.isdir(tmpdir):
            return []
        entries: list[tuple[str, int]] = []
        for name in sorted(os.listdir(tmpdir)):
            if name == "__pycache__":
                continue
            path = os.path.join(tmpdir, name)
            if os.path.isdir(path):
                entries.append((f"{name}/", -1))
            elif os.path.isfile(path):
                entries.append((name, os.path.getsize(path)))
        return entries

    async def session_objects(self, *, timeout: float) -> list[tuple[str, str, bool]]:
        """What the model bound in its persistent ``python`` session, as
        ``(name, description, is_function)`` in binding order; empty when no
        session outlives a call or none has started.

        A harness cell between rounds, while the kernel is idle: it is not a
        tool call and reaches no transcript. It reaches the store through the
        import system rather than a name, so it binds nothing in the session
        it lists.

        Raises :class:`SessionInspectionError` when the cell gives no listing:
        it failed (an object whose description raised, a timeout that took
        the kernel with it) or printed something else last."""
        if self.reset_before_each_call or self.is_fresh:
            return []
        output, failed = await self.run_harness(
            'print(__import__("runtime._repl_state", fromlist=["session_listing"]).session_listing(globals()))',
            timeout=timeout,
        )
        lines = output.strip().splitlines()
        parsed: list[list[Any]] | None = None
        if not failed and lines:
            try:
                parsed = cast("list[list[Any]]", json.loads(lines[-1]))
            except ValueError:
                parsed = None
        if parsed is None:
            last = lines[-1].strip() if lines else "no output"
            if len(last) > LISTING_ERROR_MAX_CHARS:
                last = last[:LISTING_ERROR_MAX_CHARS] + "…"
            raise SessionInspectionError(f"the session listing returned no listing: {last}", output)
        return [(str(name), str(text), bool(is_function)) for name, text, is_function in parsed]

    def _maybe_remap_single_arg(self) -> None:
        """ullm's lone-argument remap, minus a call that gives only its
        ``plan`` (:func:`plan_only`)."""
        if not plan_only(self.parameters, self.arguments):
            super()._maybe_remap_single_arg()

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """One cell: reset first in round mode, run it, then the tripwire,
        the flat-directory sweep and the traceback normalisation over what
        the model will read."""
        self.cells_started += 1
        # A call that runs no cell leaves nothing of the last one's to report.
        self.timed_out = self.kernel_lost = False
        self.strays_killed = 0
        code = str(arguments.get("code", ""))
        if not code.strip():
            self.is_error = True
            return ullm.conv.StringContent(text="python requires a non-empty `code` string.")
        # Fresh kernel per call (~50 ms): module state and globals go, not
        # just the namespace. (_reset_impl: Tool.state is "running" here,
        # but no kernel operation is in flight.) Skipped when the agent
        # keeps state for longer than one snippet; the reset then happens
        # at its boundary. The working directory's CONTENTS survive every
        # reset — see SealAgent._reset_repl for why.
        if self.reset_before_each_call:
            self._reset_impl(keep_tmpdir_contents=True)
        plan = arguments.get("plan") if self.plan_argument != "none" else None
        self.call_plan = str(plan)[:PLAN_MAX_CHARS] if plan is not None and str(plan).strip() else None
        names = list(arguments)
        self.call_plan_first = None if self.call_plan is None else names.index("plan") < names.index("code")
        try:
            content = await super()._execute(arguments)
        finally:
            self.call_plan = None
            self.call_plan_first = None
        if isinstance(content, ullm.conv.StringContent) and _SOURCE_TRIPWIRE in content.text:
            # Defense-in-depth fallback: the in-process patches in
            # _block_source_access() should have caught this earlier and
            # returned the same notice. If the tripwire fires here, an
            # unknown bypass got through — discard the entire output and
            # surface the same notice the LLM would have seen from the
            # patch path, so the experience is uniform.
            self.is_error = True
            return ullm.conv.StringContent(
                text=seal.harness.game._BLOCKED_NOTICE + "\n\n" + seal.harness.game._BLOCKED_DENIAL
            )
        if isinstance(content, ullm.conv.StringContent) and self._tmpdir is not None:
            # The flat contract's enforcement half: a cell can mkdir where the
            # file tools cannot, so every execution ends with the sweep, and
            # the note lands in this cell's own output — teaching at the act.
            removed = seal.workspace.sweep_directory(self._tmpdir)
            if removed:
                content = ullm.conv.StringContent(text=content.text + seal.workspace.flat_note(removed))
        if isinstance(content, ullm.conv.StringContent):
            # Strip absolute paths from ``File "..."`` lines (tracebacks).
            # The REPL's tmpdir lives under a host-specific deploy path
            # that's noise to the LLM and burns tokens; keep filename only.
            content = ullm.conv.StringContent(
                text=seal.errors.for_model(
                    content.text, model_dir=self._tmpdir, world_model=self.use_world_model != "none"
                )
            )
        return content
