"""The duck's ``python`` tool: a stateful kernel with the task in it.

A thin subclass of ullm's ``PythonTool``. The kernel is the model's whole
interface to the task — the grids arrive in its namespace and the configured
answer channel leaves through it — so this class owns three things beyond
what ullm gives: the task setup, the answer harvest that follows every call,
and the rendering of kernel output into something worth putting in a prompt.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import ullm.conv
import ullm.tools.python

import taaf.game
import taaf.grid
from duck import errors, images, prompts
from duck.config import CheckMode
from duck.kernel_code import (
    CHECK_FAILURE_HOST_FUNCTION,
    CHECK_SUCCESS_HOST_FUNCTION,
    HOUSEKEEPING_CODE,
    PYTHON_CALL_INDEX_HOST_FUNCTION,
    READ_ANSWER_CODE,
    SESSION_LISTING_CODE,
    SHOW_HOST_FUNCTION,
    grid_text,
    parse_answer_payload,
    prelude,
    task_setup_code,
)
from duck.sandbox import BLOCKED_DENIAL, BLOCKED_NOTICE, SOURCE_TRIPWIRE
from duck.workspace import flat_note, sweep_directory

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
    the record; the listing's is raw kernel output, not rewritten for the
    model, so its frames and paths stay off the prompt."""

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


def _tool_parameters() -> dict[str, Any]:
    """The tool schema: one ephemeral ``code`` string."""
    return {
        "type": "object",
        "properties": {"code": {"type": "string", "description": prompts.PYTHON_TOOL_CODE_PARAM_DESCRIPTION}},
        "required": ["code"],
    }


def _grid_from_untyped_rows(rows: object) -> taaf.grid.Grid | None:
    """Validate one private JSON grid before constructing a TAAF grid."""
    if not isinstance(rows, list) or not rows:
        return None
    normalized: list[list[int]] = []
    for raw_row in cast(list[object], rows):
        if not isinstance(raw_row, list) or not raw_row:
            return None
        row: list[int] = []
        for value in cast(list[object], raw_row):
            if not isinstance(value, int) or isinstance(value, bool):
                return None
            row.append(value)
        normalized.append(row)
    try:
        return taaf.grid.Grid.from_lists(normalized)
    except (TypeError, ValueError):
        return None


def _rendered_grid(
    grid: taaf.grid.Grid,
    *,
    scale: int,
    gridlines: bool,
    alt_text: str,
) -> ullm.conv.ImageContent:
    """One validated grid as model-facing image content."""
    return ullm.conv.ImageContent(
        data=images.render(grid, scale=scale, gridlines=gridlines),
        alt_text=alt_text,
    )


def _show_content(
    payload: list[object] | None,
    *,
    image_scale: int,
    image_gridlines: bool,
) -> list[ullm.conv.Content]:
    """Validated ``show`` grids, in queue order, with optional labels."""
    if payload is None:
        return []
    rendered: list[ullm.conv.Content] = []
    for raw_item in payload:
        if not isinstance(raw_item, dict):
            continue
        item = cast(dict[object, object], raw_item)
        grid = _grid_from_untyped_rows(item.get("grid"))
        if grid is None:
            continue
        raw_label = item.get("label")
        label = raw_label.strip() if isinstance(raw_label, str) else ""
        if label:
            rendered.append(ullm.conv.StringContent(text=label))
        rendered.append(
            _rendered_grid(
                grid,
                scale=image_scale,
                gridlines=image_gridlines,
                alt_text=label or "grid queued by show()",
            )
        )
    if not rendered:
        return []
    return [ullm.conv.StringContent(text="Rendered grids from show():"), *rendered]


def _check_failure_content(
    payload: dict[str, object] | None,
    *,
    include_images: bool,
    image_scale: int,
    image_gridlines: bool,
    row_numbers: bool,
) -> list[ullm.conv.Content]:
    """Every failed candidate output as uncapped host-rendered text."""
    if payload is None:
        return []
    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        return []
    rendered: list[ullm.conv.Content] = []
    for raw_pair in cast(list[object], pairs):
        if not isinstance(raw_pair, dict):
            continue
        raw = cast(dict[object, object], raw_pair)
        index = raw.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            continue
        rows = raw.get("got")
        grid: taaf.grid.Grid | None = None
        # Numbered from 0, as `CheckError`, `err.report.pairs` and `train_in`
        # are, so the pair this names is the one those name.
        if rows is None:
            text = f"Train pair {index} — your fn raised an error; no output to render."
        else:
            grid = _grid_from_untyped_rows(rows)
            if grid is None:
                text = f"Train pair {index} — your output could not be rendered as a grid."
            else:
                rendered_rows = grid_text(grid, row_numbers=row_numbers)
                text = f"Train pair {index} — your output:\n{rendered_rows}"
        rendered.append(ullm.conv.StringContent(text=text))
        if rows is not None and grid is not None and include_images:
            rendered.append(
                _rendered_grid(
                    grid,
                    scale=image_scale,
                    gridlines=image_gridlines,
                    alt_text=f"failed check output for train pair {index}",
                )
            )
    if not rendered:
        return []
    header = ullm.conv.StringContent(
        text="check(fn) FAILED. Your fn's output for each failed train pair is rendered below:"
    )
    return [header, *rendered]


@dataclass
class DuckPythonTool(ullm.tools.python.PythonTool):
    """The model's kernel, for one game.

    Stateful by construction (``clear_state_on="never"``): the model builds
    up helpers and intermediate results across calls, and the answer it is
    accumulating lives in the same namespace. Nothing resets it inside a
    game — there is one game per task and one call into the agent, so a
    reset would only ever destroy work.
    """

    name: str = "python"
    description: str = prompts.PYTHON_TOOL_DESCRIPTION
    parameters: dict[str, Any] = field(default_factory=_tool_parameters)
    initial_code: str | None = None
    clear_state_on: str = "never"  # pyright: ignore[reportIncompatibleVariableOverride]
    # RLIMIT_AS for the kernel subprocess, raised from ullm's 2048. Not a
    # comfort margin: at 2048 `import numpy` never completes — OpenBLAS
    # reserves per-thread arenas out of address space and the import wedges
    # rather than failing, so the kernel dies at its startup deadline with
    # nothing to show for it. The whale ships this same number.
    memory_limit_mb: int = 8192
    # ullm's own output cap is disabled in favour of the duck's below: this
    # class truncates once, at a token budget, and reports that it did.
    max_output_chars: int = 50_000_000
    #: Cap on kernel output carried back into the conversation (x4 = chars).
    #: Upstream's tool default; DuckAgent passes its own.
    tool_output_tokens: int = 1024
    #: Whether and how this arm exposes the training-pair oracle.
    check_mode: CheckMode = CheckMode.DISABLED
    #: Whether model code can queue grids through ``show(grid, label)``.
    show_enabled: bool = False
    #: Maximum grids ``show`` accepts during one model tool call.
    max_shows: int = 6
    #: Whether failed ``check`` outputs include images as well as full text.
    check_images_enabled: bool = False
    #: Nearest-neighbour pixels per grid cell for tool-returned images.
    image_scale: int = 4
    #: Whether tool-returned grid images carry a one-pixel cell lattice.
    image_gridlines: bool = False
    #: Whether every rendered row carries its index and a tab — the kernel's
    #: ``ascii_grid`` and this tool's own feedback both, so what the model
    #: reads back looks the way what it renders itself does.
    ascii_row_numbers: bool = False
    #: Whether a passing ``check`` prints its report and the reminder to record
    #: the answer. Baked into the prelude; changes only take on a fresh kernel.
    check_prints_on_pass: bool = False
    #: Save validated predictions into the first answer slots after a passing
    #: check in enabled mode; mandatory mode keeps its host submission path.
    check_and_submit: bool = False
    #: Whether the working directory is the model's, as the file tools make it:
    #: its files import as modules (``duck.kernel_workspace``), every cell
    #: starts from the housekeeping there, and a directory a cell makes is
    #: swept, since the working directory is flat. Baked into the prelude;
    #: changes only take on a fresh kernel.
    file_tools: bool = False
    #: Asked after every model call for budget status and a one-shot warning
    #: to append to the tool result. ``None`` (the callable's answer or
    #: the field) appends nothing. Set per game by the agent.
    budget_note: Callable[[], str | None] | None = None
    #: Called with the raw kernel output of the answer read, after every
    #: model call. Set per game by the agent; unset in tests that only want
    #: the kernel.
    on_answer: Callable[[dict[str, object] | None], None] | None = None
    #: Asked right after ``on_answer`` for the answers the call changed, as a
    #: following user message of their own; empty where none changed. Set per
    #: game by the agent when it shows changed answers.
    answer_feedback: Callable[[], list[ullm.conv.Content]] | None = None
    #: Called with a passing ``fn``'s validated test predictions in mandatory
    #: mode. Set per game by the agent.
    on_checked_answer: Callable[[list[list[list[int]]]], None] | None = None
    # Side feedback must be a following UserPrompt: backend tool-result
    # schemas are text-only even when the model itself is multimodal.
    _pending_feedback: list[list[ullm.conv.Content]] = field(
        default_factory=lambda: list[list[ullm.conv.Content]](),
        init=False,
        repr=False,
    )
    _received_shows: list[object] = field(
        default_factory=lambda: list[object](),
        init=False,
        repr=False,
    )
    _received_show_label_chars: int = field(default=0, init=False, repr=False)
    _received_check_failure: dict[str, object] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _received_checked_answer: list[list[list[int]]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _n_train: int = field(default=0, init=False, repr=False)
    _n_test: int = field(default=0, init=False, repr=False)
    _python_call_index: int = field(default=0, init=False, repr=False)
    _active_python_call_index: int | None = field(default=None, init=False, repr=False)
    _feedback_tripwire: bool = field(default=False, init=False, repr=False)
    _task_setup_source: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the workspace and derive standalone defaults from the gate."""
        CheckMode.validated(cast(object, self.check_mode))
        super().__post_init__()
        if self.initial_code is None:
            self.initial_code = prelude(
                include_check=self.check_mode.enabled,
                include_show=self.show_enabled,
                check_prints_on_pass=self.check_prints_on_pass,
                file_tools=self.file_tools,
            )
        if self.description == prompts.PYTHON_TOOL_DESCRIPTION:
            self.description = prompts.DuckPrompts().python_tool_description(
                prompts.PromptModes(
                    check_mode=self.check_mode,
                    check_and_submit=self.check_and_submit,
                    show_enabled=self.show_enabled,
                    check_images_enabled=self.check_images_enabled,
                )
            )
        self.wire_feedback_hosts()

    def wire_feedback_hosts(self) -> None:
        """Bind the trusted callbacks that preserve feedback mid-snippet."""
        functions = dict(self.host_functions)
        functions.pop(CHECK_FAILURE_HOST_FUNCTION, None)
        functions.pop(CHECK_SUCCESS_HOST_FUNCTION, None)
        functions.pop(PYTHON_CALL_INDEX_HOST_FUNCTION, None)
        functions.pop(SHOW_HOST_FUNCTION, None)
        if self.check_mode.enabled:
            functions[CHECK_FAILURE_HOST_FUNCTION] = self._receive_check_failure
        if self.check_mode is CheckMode.MANDATORY:
            functions[CHECK_SUCCESS_HOST_FUNCTION] = self._receive_checked_answer
        if self.check_mode.enabled or self.show_enabled:
            functions[PYTHON_CALL_INDEX_HOST_FUNCTION] = self._current_python_call_index
        if self.show_enabled:
            functions[SHOW_HOST_FUNCTION] = self._receive_show
        self.host_functions = functions

    def _receive_show(self, payload: object) -> bool:
        """Validate and retain one grid, returning whether it was accepted."""
        if len(self._received_shows) >= self.max_shows or not isinstance(payload, dict):
            return False
        item = cast(dict[object, object], payload)
        grid = _grid_from_untyped_rows(item.get("grid"))
        if grid is None:
            return False
        raw_label = item.get("label")
        if not isinstance(raw_label, str):
            return False
        if SOURCE_TRIPWIRE in raw_label:
            self._feedback_tripwire = True
            return False
        label = raw_label.strip()
        max_label_chars = max(256, self.tool_output_tokens * 4)
        remaining_label_chars = max(0, max_label_chars - self._received_show_label_chars)
        if len(label) > remaining_label_chars:
            label = f"{label[: remaining_label_chars - 1]}…" if remaining_label_chars else ""
        self._received_show_label_chars += len(label)
        self._received_shows.append({"grid": grid.to_lists(), "label": label})
        return True

    def _receive_check_failure(self, payload: object) -> None:
        """Validate and retain one task-bounded failed-check payload."""
        if not isinstance(payload, dict):
            return
        raw_pairs = cast(dict[object, object], payload).get("pairs")
        if not isinstance(raw_pairs, list):
            return
        pairs_payload = cast(list[object], raw_pairs)
        if not 0 < len(pairs_payload) <= self._n_train:
            return
        pairs: list[dict[str, object]] = []
        seen: set[int] = set()
        for raw_pair in pairs_payload:
            if not isinstance(raw_pair, dict):
                return
            item = cast(dict[object, object], raw_pair)
            index = item.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < self._n_train or index in seen:
                return
            seen.add(index)
            rows = item.get("got")
            if rows is None:
                pairs.append({"index": index, "got": None})
                continue
            grid = _grid_from_untyped_rows(rows)
            if grid is None:
                return
            pairs.append({"index": index, "got": grid.to_lists()})
        self._received_check_failure = {"pairs": pairs}

    def _receive_checked_answer(self, payload: object) -> None:
        """Retain one task-sized set of validated mandatory predictions."""
        if self._received_checked_answer is not None:
            raise RuntimeError(f"{CHECK_SUCCESS_HOST_FUNCTION} callback received more than one answer")
        if not isinstance(payload, dict):
            raise RuntimeError(f"{CHECK_SUCCESS_HOST_FUNCTION} callback payload is not a dictionary")
        raw_grids = cast(dict[object, object], payload).get("grids")
        if not isinstance(raw_grids, list):
            raise RuntimeError(f"{CHECK_SUCCESS_HOST_FUNCTION} callback payload has no grid list")
        grids_payload = cast(list[object], raw_grids)
        if len(grids_payload) != self._n_test:
            raise RuntimeError(
                f"{CHECK_SUCCESS_HOST_FUNCTION} callback supplied {len(grids_payload)} grids, expected {self._n_test}"
            )
        grids: list[list[list[int]]] = []
        for index, raw_grid in enumerate(grids_payload):
            grid = _grid_from_untyped_rows(raw_grid)
            if grid is None:
                raise RuntimeError(f"{CHECK_SUCCESS_HOST_FUNCTION} callback grid {index} is invalid")
            grids.append(grid.to_lists())
        self._received_checked_answer = grids

    def _current_python_call_index(self) -> int:
        """Return the valid model snippet currently running."""
        if self._active_python_call_index is None:
            raise RuntimeError("no model Python call is active")
        return self._active_python_call_index

    async def _kernel_output(self, code: str) -> tuple[str, bool]:
        """Run code in the kernel and return ``(output, errored)`` raw.

        The one place kernel execution happens, so the private answer read
        and the model's own call cannot diverge in how they are run.
        """
        process = cast("subprocess.Popen[bytes] | None", self._process)
        try:
            content = await super()._execute({"code": code})
        finally:
            # Base PythonTool drops its Popen reference when a worker exits
            # unexpectedly, without reaping it. Task setup means model calls
            # already have a live process, so retain that local handle and
            # reap it here. The death is not always visible to poll() yet — a
            # worker that raised SystemExit can still be tearing down when
            # its stdout closes — and an unreaped Popen warns from its
            # destructor, which the test gate promotes to an error. The
            # worker is garbage either way, so a kill is the reliable route
            # to the wait.
            if process is not None and self._process is None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        return content.to_string(), bool(self.is_error)

    async def _restore_task_if_needed(self) -> None:
        """Rebind the task before using a worker restarted mid-game."""
        if not self.is_fresh or self._task_setup_source is None:
            return
        output, errored = await self._kernel_output(self._task_setup_source)
        if errored:
            raise RuntimeError(f"kernel task restore failed: {output.strip()}")

    @property
    def has_feedback(self) -> bool:
        """Whether the last model call queued a follow-up user message."""
        return bool(self._pending_feedback)

    @property
    def has_checked_answer(self) -> bool:
        """Whether the last model call produced a mandatory checked answer."""
        return self._received_checked_answer is not None

    def take_feedback(self) -> list[list[ullm.conv.Content]]:
        """Take each following user-message group queued by the last call."""
        feedback = self._pending_feedback
        self._pending_feedback = []
        return feedback

    async def setup_task(self, view: taaf.game.GameView) -> None:
        """Bind this game's grids into the kernel namespace.

        Raises rather than reporting: the model never sees this code, so a
        failure here is a duck bug, and a game that started without its task
        in the kernel would look to the model like an empty namespace it
        should work around.
        """
        self.wire_feedback_hosts()
        self._n_train = len(view.given)
        self._n_test = len(view.questions)
        self._python_call_index = 0
        self._task_setup_source = task_setup_code(
            view,
            include_check=self.check_mode.enabled,
            check_mandatory=self.check_mode is CheckMode.MANDATORY,
            check_and_submit=self.check_and_submit and self.check_mode is CheckMode.ENABLED,
            include_show=self.show_enabled,
            max_shows=self.max_shows,
        )
        output, errored = await self._kernel_output(self._task_setup_source)
        if errored:
            raise RuntimeError(f"kernel task setup failed: {output.strip()}")

    async def read_answer(self) -> dict[str, object] | None:
        """Read ``test_out`` out of the namespace.

        A private execution: the model never sees this call or its output.
        ``None`` where the kernel produced no marker line — dead, timed out,
        or interrupted — which the caller reads as "no answer this round"
        rather than as a malformed one.
        """
        # is_error belongs to the model's call, which has already been
        # rendered but whose flag the turn manager reads after we return.
        # A private run must not overwrite the verdict on the model's code.
        errored = self.is_error
        try:
            await self._restore_task_if_needed()
            output, _ = await self._kernel_output(READ_ANSWER_CODE)
        finally:
            self.is_error = errored
        return parse_answer_payload(output)

    async def _housekeeping(self) -> str:
        """Run ``kernel_workspace.housekeeping`` before a model cell, and
        return what it printed, which leads that cell's output.

        Raises rather than reporting, as the task setup does: the model never
        sees this code, so a failure here is a duck bug."""
        output, errored = await self._kernel_output(HOUSEKEEPING_CODE)
        if errored:
            raise RuntimeError(f"kernel housekeeping failed: {output.strip()}")
        return output

    def working_files(self) -> list[tuple[str, int]]:
        """The kernel's working directory, as ``(name, bytes)`` pairs sorted
        by name; a subdirectory is listed by its name with a trailing ``/``
        and a size of -1. Read host-side: the directory is this tool's."""
        tmpdir = self._tmpdir
        if tmpdir is None or not os.path.isdir(tmpdir):
            return []
        entries: list[tuple[str, int]] = []
        for name in sorted(os.listdir(tmpdir)):
            path = os.path.join(tmpdir, name)
            if os.path.isdir(path):
                entries.append((f"{name}/", -1))
            elif os.path.isfile(path):
                entries.append((name, os.path.getsize(path)))
        return entries

    async def session_objects(self) -> list[tuple[str, str, bool]]:
        """What the model bound in its ``python`` session, as ``(name,
        description, is_function)`` in binding order; empty before the kernel
        has started.

        A private call between rounds, while the kernel is idle: it reaches
        no transcript, and leaves the verdict on the model's last call as it
        was."""
        if self.is_fresh:
            return []
        errored = self.is_error
        try:
            output, failed = await self._kernel_output(SESSION_LISTING_CODE)
        finally:
            self.is_error = errored
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

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """One model call: run the snippet, harvest the answer, render."""
        self._pending_feedback = []
        self._received_shows = []
        self._received_show_label_chars = 0
        self._received_check_failure = None
        self._received_checked_answer = None
        self._feedback_tripwire = False
        code = str(arguments.get("code", "")).rstrip()
        if not code:
            self.is_error = True
            return self._result_with_budget_note("python requires a non-empty `code` string.")
        try:
            compile(code, "<repl>", "exec")
        except SyntaxError as exc:
            self.is_error = True
            return self._result_with_budget_note(f"Python syntax error: {exc}")

        await self._restore_task_if_needed()
        notices = await self._housekeeping() if self.file_tools else ""
        self._active_python_call_index = self._python_call_index
        self._python_call_index += 1
        try:
            output, _ = await self._kernel_output(code)
        finally:
            self._active_python_call_index = None
        if notices.strip():
            output = f"{notices.rstrip()}\n{output}"
        check_failure = self._received_check_failure
        checked_answer = self._received_checked_answer
        shows = self._received_shows
        answer_content: list[ullm.conv.Content] = []

        if SOURCE_TRIPWIRE in output or self._feedback_tripwire:
            # Defence in depth: the in-kernel patches should have caught this
            # and returned the same notice. If the tripwire fires here an
            # unknown bypass got through, so the whole output goes — a
            # partial one could still carry source.
            self.is_error = True
            output = f"{BLOCKED_NOTICE}\n\n{BLOCKED_DENIAL}"
            check_failure = None
            checked_answer = None
            self._received_checked_answer = None
            shows = []
        else:
            output = errors.for_model(output, model_dir=self._tmpdir)
            if checked_answer is not None and self.on_checked_answer is not None:
                self.on_checked_answer(checked_answer)
            elif self.on_answer is not None and not self.is_fresh:
                self.on_answer(await self.read_answer())
                if self.answer_feedback is not None:
                    answer_content = self.answer_feedback()

        text = self._render(output)
        if self.file_tools and self._tmpdir is not None:
            # The flat contract's enforcement half: a cell can make a directory
            # where the file tools cannot, so every cell ends with the sweep,
            # and the note lands in that cell's own output.
            removed = sweep_directory(self._tmpdir)
            if removed:
                text += flat_note(removed)
        rendered = self._result_with_budget_note(text)
        show_content = _show_content(
            shows,
            image_scale=self.image_scale,
            image_gridlines=self.image_gridlines,
        )
        feedback = _check_failure_content(
            check_failure,
            include_images=self.check_images_enabled,
            image_scale=self.image_scale,
            image_gridlines=self.image_gridlines,
            row_numbers=self.ascii_row_numbers,
        )
        self._pending_feedback = [group for group in (show_content, answer_content, feedback) if group]
        return rendered

    def _result_with_budget_note(self, text: str) -> ullm.conv.StringContent:
        """Append current budget feedback, including when Python cannot run."""
        note = self.budget_note() if self.budget_note is not None else None
        return ullm.conv.StringContent(text=text + (f"\n\n{note}" if note else ""))

    def _render(self, output: str) -> str:
        """Kernel output as the model reads it: empty runs say so, long ones
        are cut at the token budget with the cut named.

        A silent empty result reads as a broken tool; a silent truncation
        reads as the model's own code having produced less than it did. The
        cut takes the middle, keeping the budget's first half and its last:
        output ends where the cell ended, and when it ended on an error, the
        error is the part the model most needs to read.
        """
        text = output.strip()
        if not text:
            return prompts.EMPTY_OUTPUT_NOTE
        max_chars = max(256, self.tool_output_tokens * 4)
        if len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head
        note = prompts.truncation_note(len(text) - max_chars, self.tool_output_tokens)
        return f"{text[:head]}\n{note}\n{text[-tail:]}"


__all__ = ["DuckPythonTool"]
