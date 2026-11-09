"""The kernel: what it holds, what it survives, and how its output is
rendered.

Upstream's equivalent is `test_persistence.py`, which is mostly about
choosing between kernel lifetimes. There is one lifetime here — the kernel
lives as long as the game — so what is left is the claims that lifetime
makes, plus the rendering the model actually reads.
"""

from __future__ import annotations

import copy
import json
import pickle
import random
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
from duck import CheckMode, DuckAgent, prompts
from duck.kernel_code import (
    ANSWER_MARKER,
    CHECK_FAILURE_HOST_FUNCTION,
    CHECK_SUCCESS_HOST_FUNCTION,
    PYTHON_CALL_INDEX_HOST_FUNCTION,
    SHOW_HOST_FUNCTION,
    TASK_NAMES,
    parse_answer_payload,
    prelude,
    task_setup_code,
)
from duck.python_tool import DuckPythonTool
from duck.sandbox import SOURCE_TRIPWIRE

import taaf.game
import taaf.grid
from platypus.context import Produced, SubmitAnswer
from platypus.contract import GameOutcome
from platypus.llm import LLMConfig, build_llm_context


def _view(n_questions: int = 1, n_train: int = 2) -> taaf.game.GameView:
    """A tiny task: the mirror, so a kernel test can also be an answer test."""
    rng = np.random.default_rng(0)

    def grid() -> taaf.grid.Grid:
        return taaf.grid.Grid(rng.integers(0, 10, size=(3, 4), dtype=np.int8))

    given = tuple((g, taaf.grid.Grid(np.fliplr(g.data))) for g in (grid() for _ in range(n_train)))
    return taaf.game.GameView(game_id="kernel", given=given, questions=tuple(grid() for _ in range(n_questions)))


@pytest.fixture
async def kernel():
    tool = DuckPythonTool(timeout=60.0, check_mode=CheckMode.ENABLED, initial_code=prelude(include_check=True))
    try:
        yield tool
    finally:
        await tool.clear_state()


@pytest.fixture
async def show_kernel():
    tool = DuckPythonTool(
        timeout=60.0,
        check_mode=CheckMode.ENABLED,
        show_enabled=True,
        image_scale=4,
        image_gridlines=True,
    )
    try:
        yield tool
    finally:
        await tool.clear_state()


async def _run_snippets(tool: DuckPythonTool, *snippets: str) -> list[str]:
    return [(await tool._execute({"code": code})).to_string() for code in snippets]


def _feedback_text(tool: DuckPythonTool) -> str:
    """The side feedback as the following user message will render it."""
    return "\n".join(ullm.conv.ListContent(elements=group).to_string() for group in tool.take_feedback())


def _pickle_roundtrip(tool: DuckPythonTool) -> DuckPythonTool:
    """Restore one fresh tool through its snapshot representation."""
    return pickle.loads(pickle.dumps(tool))


class _RecordingStubBackend(ullm.backends.stub.StubBackend):
    """Stub backend retaining the user contents present on each request."""

    requests: list[list[ullm.conv.Content]]

    def __post_init__(self) -> None:
        """Initialize the request log alongside the scripted backend."""
        self.requests = []
        super().__post_init__()

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> object:
        """Record user contents before selecting the next response."""
        self.requests.append(
            [
                copy.deepcopy(element.content)
                for element in conversation.elements
                if isinstance(element, ullm.conv.UserPrompt)
            ]
        )
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)


def _collect_into(submissions: list[taaf.game.Submission]) -> SubmitAnswer:
    """A submit channel that keeps every answer, in order."""

    def submit(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        del produced
        submissions.append(submission)

    return submit


async def test_state_carries_between_calls(kernel: DuckPythonTool) -> None:
    """The point of one kernel per game: what the model defines stays
    defined, so it builds up helpers instead of rewriting them."""
    _, second = await _run_snippets(kernel, "carried = 1234", "print(carried)")
    assert "NameError" not in second, f"the kernel must carry state; got {second[:300]}"
    assert "1234" in second


async def test_the_kernel_restarts_after_memory_exhaustion(kernel: DuckPythonTool) -> None:
    """A caught MemoryError must not leave later snippets in a poisoned
    worker: the kernel is recycled and the model is told its state is gone."""
    first, second = await _run_snippets(
        kernel,
        "carried = 1234\njunk = [None] * 3_000_000_000",
        "print(carried)",
    )
    assert "MemoryError" in first
    assert "kernel was restarted" in first
    assert "variables and definitions were cleared" in first
    assert "NameError" in second


async def test_the_task_arrives_under_the_advertised_names(kernel: DuckPythonTool) -> None:
    """The system prompt names these; a kernel that bound anything else would
    leave the model working from a namespace it was told to expect."""
    await kernel.setup_task(_view())
    listing = await kernel._execute({"code": "print(sorted(n for n in globals() if not n.startswith('_')))"})
    for name in TASK_NAMES:
        assert name in listing.to_string(), name


async def test_train_pairs_are_arrays_the_model_can_work_with(kernel: DuckPythonTool) -> None:
    view = _view()
    await kernel.setup_task(view)
    out = await kernel._execute({"code": "print(type(train_in[0]).__name__, train_in[0].shape, train_in[0].dtype)"})
    assert "ndarray" in out.to_string()
    assert str(view.given[0][0].shape[0]) in out.to_string()


async def test_ascii_grid_renders_one_digit_per_cell(kernel: DuckPythonTool) -> None:
    """Ten colours, so every value is its own character and there are no
    letters to explain."""
    out = await kernel._execute({"code": "print(ascii_grid(np.array([[0,1,2],[7,8,9]])))"})
    assert out.to_string().strip() == "012\n789"


async def test_ascii_grid_refuses_what_is_not_a_grid(kernel: DuckPythonTool) -> None:
    """It raises in the model's own kernel, where the traceback is something
    the model can act on."""
    out = await kernel._execute({"code": "ascii_grid(np.zeros((2,2,2)))"})
    assert "2-D" in out.to_string()
    out = await kernel._execute({"code": "ascii_grid(np.array([[0, 42]]))"})
    assert "0-9" in out.to_string()


async def test_check_reports_a_passing_rule_without_answering(kernel: DuckPythonTool) -> None:
    """Checking is an oracle, not a second answer channel: a pass leaves the
    model's explicit ``test_out`` assignment entirely in its hands."""
    await kernel.setup_task(_view())
    out = await kernel._execute(
        {"code": ("def fn(grid):\n    return np.fliplr(grid)\nreport = check(fn)\nprint(report)\nprint(test_out)")}
    )
    text = out.to_string()
    assert "check: ALL TRAIN PAIRS OK" in text
    assert "pair 0: OK" in text and "pair 1: OK" in text
    assert "[[None, None]]" in text


async def test_mandatory_check_emits_validated_test_predictions() -> None:
    """A passing check, not ``test_out``, is the mandatory answer channel."""
    tool = DuckPythonTool(timeout=60.0, check_mode=CheckMode.MANDATORY, isolate_network=False)
    received: list[list[list[list[int]]]] = []
    tool.on_checked_answer = received.append
    view = _view(n_questions=2)
    try:
        await tool.setup_task(view)
        out = await tool._execute(
            {
                "code": (
                    "test_out[0][0] = np.zeros((1, 1), dtype=np.int8)\n"
                    "print(check(lambda grid: np.fliplr(grid)))\n"
                    "test_out[0][0] = np.full((1, 1), 9, dtype=np.int8)"
                )
            }
        )
        assert "check: ALL TRAIN PAIRS OK" in out.to_string()
        assert received == [[np.fliplr(question.data).tolist() for question in view.questions]]
    finally:
        await tool.clear_state()


@pytest.mark.parametrize("prints", [False, True])
async def test_check_and_submit_updates_first_slots_and_preserves_alternatives(prints: bool) -> None:
    tool = DuckPythonTool(check_mode=CheckMode.ENABLED, check_and_submit=True, check_prints_on_pass=prints)
    view = _view(n_questions=2)
    try:
        await tool.setup_task(view)
        out, saved = await _run_snippets(
            tool,
            "test_out = [[[[0]], [[9]]] for _ in test_in]\ncheck(lambda grid: np.fliplr(grid))\nready = True",
            "print(json.dumps(test_out))",
        )
        assert json.loads(saved) == [[np.fliplr(question.data).tolist(), [[9]]] for question in view.questions]
        assert "record it" not in out
        assert ("check: ALL TRAIN PAIRS OK" in out) == prints
        assert not tool.has_checked_answer
        await _run_snippets(tool, "test_out[0][0] = [[3]]\ncheck(lambda grid: np.fliplr(grid))")
        (again,) = await _run_snippets(tool, "print(json.dumps(test_out))")
        assert again == saved
    finally:
        await tool.clear_state()


@pytest.mark.parametrize("failure", ["training", "test_exception", "invalid_test_grid"])
async def test_check_and_submit_preserves_all_answers_on_failure(failure: str) -> None:
    tool = DuckPythonTool(check_mode=CheckMode.ENABLED, check_and_submit=True, check_prints_on_pass=True)
    try:
        await tool.setup_task(_view(n_questions=2))
        await _run_snippets(tool, "test_out = [[[[3]], [[9]]] for _ in test_in]")
        action = "raise ValueError('test failed')" if failure == "test_exception" else "return [[99]]"
        source = (
            "calls = 0\n"
            "def candidate(grid):\n"
            "    global calls\n"
            "    calls += 1\n"
            "    if calls == len(train_in) + 2:\n"
            f"        {action}\n"
            "    return np.fliplr(grid)\n"
            "check(candidate)"
        )
        if failure == "training":
            source = "check(lambda grid: grid)"
        failed, saved = await _run_snippets(tool, source, "print(json.dumps(test_out))")
        assert "CheckError" in failed
        assert "check: ALL TRAIN PAIRS OK" not in failed
        assert json.loads(saved) == [[[[3]], [[9]]], [[[3]], [[9]]]]
    finally:
        await tool.clear_state()


@pytest.mark.parametrize(
    "initial,alternatives",
    [
        ("[]", [None, None]),
        ("[None, [[[0]], [[9]]]]", [None, [[9]]]),
        ("([[[0]], [[8]]], ([[1]], [[9]]))", [[[8]], [[9]]]),
        ("None", [None, None]),
        ("[[[[0]]]]", [None, None]),
    ],
)
async def test_check_and_submit_repairs_answer_containers(initial: str, alternatives: list[object]) -> None:
    tool = DuckPythonTool(check_mode=CheckMode.ENABLED, check_and_submit=True)
    view = _view(n_questions=2)
    try:
        await tool.setup_task(view)
        out, saved = await _run_snippets(
            tool,
            f"test_out = {initial}\nprint(check(lambda grid: np.fliplr(grid)))",
            "print(json.dumps(test_out))",
        )
        assert "check: ALL TRAIN PAIRS OK" in out
        assert json.loads(saved) == [
            [np.fliplr(question.data).tolist(), alternative]
            for question, alternative in zip(view.questions, alternatives, strict=True)
        ]
    finally:
        await tool.clear_state()


@pytest.mark.parametrize("check_mode", [CheckMode.ENABLED, CheckMode.MANDATORY])
@pytest.mark.parametrize("automatic", [False, True])
async def test_check_help_describes_automatic_saving_only_when_active(check_mode: CheckMode, automatic: bool) -> None:
    tool = DuckPythonTool(check_mode=check_mode, check_and_submit=automatic)
    try:
        await tool.setup_task(_view())
        (doc,) = await _run_snippets(tool, "print(check.__doc__)")
        assert ("With check_and_submit" in doc) == (automatic and check_mode is CheckMode.ENABLED)
        if automatic and check_mode is CheckMode.ENABLED:
            assert "malformed answer rows are rebuilt" in doc
        else:
            assert "test input. One check is allowed per Python tool call." in doc
    finally:
        await tool.clear_state()


async def test_mandatory_check_rejects_a_function_that_fails_on_test() -> None:
    """Train consistency alone cannot submit invalid or failing test output."""
    tool = DuckPythonTool(timeout=60.0, check_mode=CheckMode.MANDATORY, isolate_network=False)
    received: list[list[list[list[int]]]] = []
    tool.on_checked_answer = received.append
    try:
        await tool.setup_task(_view())
        out = await tool._execute(
            {
                "code": (
                    "calls = 0\n"
                    "def fn(grid):\n"
                    "    global calls\n"
                    "    calls += 1\n"
                    "    if calls > len(train_in):\n"
                    "        raise ValueError('test failed')\n"
                    "    return np.fliplr(grid)\n"
                    "try:\n"
                    "    check(fn)\n"
                    "except CheckError as err:\n"
                    "    print('pairs:', [pair.ok for pair in err.report.pairs])\n"
                    "    print('passed:', err.report.passed)\n"
                    "    raise\n"
                )
            }
        )
        # A rule the training pairs accepted is not reported as a train
        # failure: the header names where it actually broke, and every pair
        # still reads OK beneath it.
        assert "check: TRAIN PAIRS OK, fn FAILED ON A TEST INPUT" in out.to_string()
        assert "check: FAILED" not in out.to_string()
        assert "test input 0: ValueError: test failed" in out.to_string()
        # The per-pair verdicts a model ranks candidates on stay true to the
        # training pairs; only `passed` — did this submit? — goes false.
        assert "pairs: [True, True]" in out.to_string()
        assert "passed: False" in out.to_string()
        assert not received
    finally:
        await tool.clear_state()


async def test_check_failure_is_catchable_and_inspectable(kernel: DuckPythonTool) -> None:
    """The compact report stays in the kernel for candidate ranking while
    the host separately renders the model's own failed outputs."""
    await kernel.setup_task(_view())
    out = await kernel._execute(
        {
            "code": (
                "try:\n"
                "    check(lambda grid: grid)\n"
                "except CheckError as err:\n"
                "    print(err.report.pairs[0].mismatch_count)\n"
                "    print(err.report.pairs[0].first_mismatches)"
            )
        }
    )
    feedback = _feedback_text(kernel)
    assert "Train pair 0 — your output:" in feedback
    assert "Train pair 1 — your output:" in feedback
    assert "expected output:" not in feedback and "— input:" not in feedback
    assert "(" in out.to_string(), "the first mismatch tuples must be inspectable"


async def test_check_feedback_includes_every_failed_pair_outside_the_text_cap(kernel: DuckPythonTool) -> None:
    """The failure grids are host feedback, not ordinary stdout, so a small
    tool-output cap cannot silently turn complete feedback into first-N
    feedback."""
    await kernel.setup_task(_view(n_train=5))
    kernel.tool_output_tokens = 64
    out = await kernel._execute({"code": "check(lambda grid: grid)"})
    assert "cut to stay within" in out.to_string()
    feedback = _feedback_text(kernel)
    assert all(f"Train pair {index} — your output:" in feedback for index in range(5))


async def test_check_names_candidate_exceptions_without_inventing_a_grid(kernel: DuckPythonTool) -> None:
    """An exception still gets one report entry per pair, but no placeholder
    grid that could be mistaken for the function's output."""
    await kernel.setup_task(_view())
    out = await kernel._execute({"code": "check(lambda grid: grid[missing_name])"})
    feedback = _feedback_text(kernel)
    assert feedback.count("no output to render") == 2
    assert "NameError" in out.to_string()


async def test_the_task_arrays_cannot_be_written_in_place(kernel: DuckPythonTool) -> None:
    """A write into a task array raises where it is made, and the task the
    next read sees is the task."""
    view = _view()
    await kernel.setup_task(view)
    first, second = await _run_snippets(
        kernel,
        "test_in[0][0, 0] = (test_in[0][0, 0] + 1) % 10",
        "print(test_in[0][0, 0], train_in[0].flags.writeable, train_out[0].flags.writeable)",
    )
    assert "read-only" in first, first
    assert f"{int(view.questions[0].data[0, 0])} False False" in second, second


async def test_a_function_that_writes_into_its_input_raises_on_a_test_input_and_checks_on_copies(
    kernel: DuckPythonTool,
) -> None:
    """``check`` hands ``fn`` a copy, so the oracle is unchanged; applied to
    ``test_in[i]`` itself, the same function fails loudly rather than
    answering from a test input it has just changed."""
    await kernel.setup_task(_view())
    out = await kernel._execute(
        {
            "code": (
                "def fn(grid):\n    grid[:] = grid[:, ::-1]\n    return grid\n"
                "check(fn)\nprint('checked')\ntest_out[0][0] = fn(test_in[0])"
            )
        }
    )
    assert "checked" in out.to_string() and "read-only" in out.to_string(), out.to_string()


async def test_check_uses_hidden_task_snapshots(
    kernel: DuckPythonTool,
) -> None:
    """Public task arrays remain notebook-like, but rebinding them cannot
    corrupt the oracle."""
    await kernel.setup_task(_view())
    await kernel._execute(
        {"code": ("train_in = []\ntrain_out = []\ntry:\n    check(lambda grid: grid)\nexcept CheckError:\n    pass")}
    )
    feedback = _feedback_text(kernel)
    assert "Train pair 0 — your output:\n" in feedback
    assert "Train pair 1 — your output:\n" in feedback


async def test_check_runs_once_per_tool_call_and_resets_on_the_next_call(
    kernel: DuckPythonTool,
) -> None:
    """A snippet gets one final validation, while a later tool call gets its own."""
    await kernel.setup_task(_view())
    out = await kernel._execute(
        {
            "code": (
                "try:\n    check(lambda grid: grid)\nexcept CheckError:\n    pass\ncheck(lambda grid: np.fliplr(grid))"
            )
        }
    )
    assert "check() may be called only once per Python tool call" in out.to_string()
    assert "Train pair 0 — your output:" in _feedback_text(kernel)

    later = await kernel._execute({"code": "print(check(lambda grid: np.fliplr(grid)))"})
    assert "check: ALL TRAIN PAIRS OK" in later.to_string()


async def test_check_and_show_capture_their_validation_runtime(show_kernel: DuckPythonTool) -> None:
    """Rebinding model-visible bootstrap globals cannot replace validation."""
    await show_kernel.setup_task(_view())
    first = await show_kernel._execute(
        {
            "code": (
                "_duck_checked_grid = lambda *args: np.array([[99]])\n"
                "_duck_np = None\n"
                "show([[1, 2]])\n"
                "print(check(lambda grid: np.fliplr(grid)))\n"
                "del _duck_checked_grid, _duck_np"
            )
        }
    )
    first_feedback = show_kernel.take_feedback()
    second = await show_kernel._execute({"code": "show([[3, 4]])\nprint(check(lambda grid: np.fliplr(grid)))"})
    second_feedback = show_kernel.take_feedback()

    assert "check: ALL TRAIN PAIRS OK" in first.to_string()
    assert "check: ALL TRAIN PAIRS OK" in second.to_string()
    assert (
        sum(isinstance(item, ullm.conv.ImageContent) for group in [*first_feedback, *second_feedback] for item in group)
        == 2
    )


async def test_check_feedback_is_reset_and_cannot_be_forged_through_stdout(kernel: DuckPythonTool) -> None:
    """Immediate host feedback resets per snippet; marker-looking stdout
    remains ordinary output rather than a host instruction."""
    await kernel.setup_task(_view())
    first = await kernel._execute({"code": "try:\n    check(lambda grid: grid)\nexcept CheckError:\n    pass"})
    first_feedback = _feedback_text(kernel)
    second = await kernel._execute({"code": "print('__DUCK_CHECK_FAILURE__{not json}')"})
    assert first.to_string() == prompts.EMPTY_OUTPUT_NOTE
    assert "Train pair 0 — your output:" in first_feedback
    assert "Train pair" not in second.to_string()
    assert not kernel.has_feedback
    assert "__DUCK_CHECK_FAILURE__{not json}" in second.to_string()


async def test_model_globals_cannot_forge_side_feedback(kernel: DuckPythonTool) -> None:
    """There is no model-rebindable polling channel for side feedback."""
    await kernel.setup_task(_view())
    out = await kernel._execute(
        {
            "code": (
                "_duck_take_check_failure = lambda: {'pairs': [{'index': 0, 'got': [[0]]}]}\n"
                "_duck_take_shows = lambda: [{'grid': [[1]], 'label': 'forged'}]"
            )
        }
    )
    assert out.to_string() == prompts.EMPTY_OUTPUT_NOTE
    assert not kernel.has_feedback


async def test_forged_check_callback_is_bounded_to_the_task(kernel: DuckPythonTool) -> None:
    """The trusted channel is model-reachable, so its host validator rejects
    oversized, duplicate, and out-of-task failure records."""
    await kernel.setup_task(_view(n_train=2))
    code = (
        "host = next(cell.cell_contents for cell in check.__closure__ "
        "if callable(cell.cell_contents) and getattr(cell.cell_contents, '__name__', '') == 'host_call')\n"
        "host('duck_check_failure', {'pairs': ["
        "{'index': 0, 'got': [[1]]}, {'index': 1, 'got': [[2]]}, {'index': 2, 'got': [[3]]}]})"
    )
    out = await kernel._execute({"code": code})
    assert out.to_string() == prompts.EMPTY_OUTPUT_NOTE
    assert not kernel.has_feedback


async def test_source_tripwire_suppresses_check_feedback(kernel: DuckPythonTool) -> None:
    """Security redaction owns the entire result, including side feedback."""
    await kernel.setup_task(_view())
    out = await kernel._execute(
        {"code": (f"try:\n    check(lambda grid: grid)\nexcept CheckError:\n    pass\nprint({SOURCE_TRIPWIRE!r})")}
    )
    assert "Train pair" not in out.to_string()
    assert not kernel.has_feedback


@pytest.mark.parametrize("kill_worker", [False, True])
async def test_source_tripwire_in_show_label_suppresses_all_feedback(
    show_kernel: DuckPythonTool,
    kill_worker: bool,
) -> None:
    """A label is model-facing output too, including when its worker dies
    after the callback has already reached the host."""
    await show_kernel.setup_task(_view())
    suffix = (
        "\nexcept RuntimeError:\n    import os\n    os._exit(1)" if kill_worker else "\nexcept RuntimeError:\n    pass"
    )
    out = await show_kernel._execute(
        {
            "code": (
                f"show([[1]], label='safe before tripwire')\ntry:\n    show([[2]], label={SOURCE_TRIPWIRE!r}){suffix}"
            )
        }
    )

    assert "Source code access denied" in out.to_string()
    assert SOURCE_TRIPWIRE not in out.to_string()
    assert not show_kernel.has_feedback


async def test_show_queues_validated_grids_in_order(show_kernel: DuckPythonTool) -> None:
    """Labels and grids become one grouped follow-up prompt, not text-only
    placeholders inside the tool result."""
    await show_kernel.setup_task(_view())
    out = await show_kernel._execute({"code": ("show(test_in[0], label='the test input')\nshow([[1, 0], [0, 1]])")})
    assert isinstance(out, ullm.conv.StringContent)
    groups = show_kernel.take_feedback()
    assert len(groups) == 1
    feedback = groups[0]
    assert isinstance(feedback[0], ullm.conv.StringContent)
    assert feedback[0].text == "Rendered grids from show():"
    assert isinstance(feedback[1], ullm.conv.StringContent) and feedback[1].text == "the test input"
    rendered = [item for item in feedback if isinstance(item, ullm.conv.ImageContent)]
    assert len(rendered) == 2
    assert rendered[1].data.shape == (11, 11, 3), "scale-four 2x2 grid plus the opt-in lattice"


async def test_show_label_is_bounded_with_the_uncapped_image_feedback(show_kernel: DuckPythonTool) -> None:
    """The image channel cannot be used to bypass the tool's text cap."""
    show_kernel.tool_output_tokens = 64
    await show_kernel.setup_task(_view())
    await show_kernel._execute({"code": "show([[1]], label='x' * 5000)"})
    groups = show_kernel.take_feedback()
    assert len(groups) == 1
    label = groups[0][1]
    assert isinstance(label, ullm.conv.StringContent)
    assert len(label.text) == 256
    assert label.text.endswith("…")


async def test_show_labels_share_the_text_budget(show_kernel: DuckPythonTool) -> None:
    """Multiple labels share one per-call character allowance."""
    show_kernel.tool_output_tokens = 64
    await show_kernel.setup_task(_view())
    await show_kernel._execute({"code": "show([[1]], label='a' * 200)\nshow([[2]], label='b' * 200)"})
    group = show_kernel.take_feedback()[0]
    labels = [item.text for item in group[1:] if isinstance(item, ullm.conv.StringContent)]
    assert [len(label) for label in labels] == [200, 56]
    assert labels[1].endswith("…")


async def test_show_limit_and_queue_reset_per_model_call(show_kernel: DuckPythonTool) -> None:
    """A later invalid/over-limit show does not erase earlier grids, and the
    per-call allowance is restored before the next model snippet."""
    show_kernel.max_shows = 1
    await show_kernel.setup_task(_view())
    first = await show_kernel._execute(
        {
            "code": (
                "try:\n"
                "    show([[10]])\n"
                "except ValueError as exc:\n"
                "    print(exc)\n"
                "show([[1]])\n"
                "try:\n"
                "    show([[2]])\n"
                "except RuntimeError as exc:\n"
                "    print(exc)"
            )
        }
    )
    assert "colours outside" in first.to_string()
    assert "show() limit of 1 per tool call reached" in first.to_string()
    assert sum(isinstance(item, ullm.conv.ImageContent) for group in show_kernel.take_feedback() for item in group) == 1

    await show_kernel._execute({"code": "show([[3]])"})
    assert sum(isinstance(item, ullm.conv.ImageContent) for group in show_kernel.take_feedback() for item in group) == 1


async def test_show_grids_survive_a_later_snippet_exception(show_kernel: DuckPythonTool) -> None:
    """Already queued views remain useful even when later model code fails."""
    await show_kernel.setup_task(_view())
    out = await show_kernel._execute({"code": "show([[1, 2]])\nraise ValueError('later')"})
    assert "ValueError" in out.to_string()
    assert any(isinstance(item, ullm.conv.ImageContent) for group in show_kernel.take_feedback() for item in group)


async def test_show_and_check_feedback_survive_a_later_timeout(show_kernel: DuckPythonTool) -> None:
    """Immediate host callbacks retain both feedback groups when a later
    infinite loop is interrupted before the snippet can return."""
    await show_kernel.setup_task(_view())
    show_kernel.timeout = 0.2
    out = await show_kernel._execute(
        {
            "code": (
                "show([[1, 2]])\n"
                "try:\n"
                "    check(lambda grid: grid)\n"
                "except CheckError:\n"
                "    pass\n"
                "while True:\n"
                "    pass"
            )
        }
    )
    assert "timed out" in out.to_string()
    groups = show_kernel.take_feedback()
    assert len(groups) == 2
    assert any(isinstance(item, ullm.conv.ImageContent) for item in groups[0])
    assert "Train pair 0 — your output:" in ullm.conv.ListContent(elements=groups[1]).to_string()


async def test_check_does_not_swallow_the_kernel_timeout(kernel: DuckPythonTool) -> None:
    """SIGINT escapes a hanging candidate so the worker and its state survive."""
    await kernel.setup_task(_view())
    kernel.timeout = 0.2
    kernel.interrupt_timeout = 0.2
    source = """sentinel = 'still here'
def hang(_grid):
    while True:
        pass
check(hang)"""
    out = await kernel._execute({"code": source})
    assert "timed out" in out.to_string()
    assert "process was killed" not in out.to_string()
    resumed = await kernel._execute({"code": "print(sentinel)"})
    assert "still here" in resumed.to_string()


async def test_show_and_check_feedback_survive_worker_death(show_kernel: DuckPythonTool) -> None:
    """Feedback reaches the host when queued, so a later kernel crash cannot
    erase grids that were already produced."""
    await show_kernel.setup_task(_view())
    out = await show_kernel._execute(
        {
            "code": (
                "show([[1, 2]])\ntry:\n    check(lambda grid: grid)\nexcept CheckError:\n    import os\n    os._exit(1)"
            )
        }
    )

    assert "Kernel process died unexpectedly" in out.to_string()
    feedback = show_kernel.take_feedback()
    assert len(feedback) == 2, "show and failed check are distinct consecutive user messages"
    assert any(isinstance(item, ullm.conv.ImageContent) for group in feedback for item in group)
    assert "Train pair 0 — your output:" in "\n".join(
        ullm.conv.ListContent(elements=group).to_string() for group in feedback
    )
    restored = await show_kernel._execute(
        {
            "code": "print([name for name in ('train_in', 'train_out', 'test_in', 'test_out', 'check', 'show') if name in globals()])"
        }
    )
    assert all(
        name in restored.to_string() for name in ("train_in", "train_out", "test_in", "test_out", "check", "show")
    )


async def test_agent_delivers_tool_grid_feedback_as_the_next_user_prompt() -> None:
    """Multimodal feedback stays out of the text-only tool-result transport
    and reaches the backend through an ordinary following user message."""
    agent = DuckAgent(reply_cap_tokens=None, images="task+show")
    tool = agent._ensure_repl()
    view = _view()
    try:
        await tool.setup_task(view)
        result_content = await tool._execute({"code": "show([[1, 2]])"})
        result = ullm.conv.ToolResult(
            call_id="show-call",
            tool_name="python",
            content=result_content,
        )
        response = ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text=""))

        assert isinstance(result.content, ullm.conv.StringContent)
        assert agent._stop_after_tool_feedback(response, result) is not None
        agent._pending_tool_feedback.extend(tool.take_feedback())
        prompts_for_entry = agent._build_state_prompts(view)

        assert len(prompts_for_entry) == 1
        prompt = prompts_for_entry[0]
        assert isinstance(prompt.content, ullm.conv.ListContent)
        assert any(isinstance(item, ullm.conv.ImageContent) for item in prompt.content.elements)
        assert not agent._pending_tool_feedback
    finally:
        await tool.clear_state()


async def test_completed_answer_still_gets_its_queued_feedback_turn() -> None:
    """The real TurnManager loop sends side feedback once even when the same
    Python call made the answer complete."""
    backend = cast(
        _RecordingStubBackend,
        _RecordingStubBackend.from_script(
            ullm.conv.LLMResponse(
                final_content=ullm.conv.ToolCallContent(
                    tool_name="python",
                    arguments={
                        "code": (
                            "show([[1, 2]], label='candidate')\n"
                            "try:\n"
                            "    check(lambda grid: grid)\n"
                            "except CheckError:\n"
                            "    pass\n"
                            "test_out[0][0] = test_in[0]"
                        )
                    },
                )
            ),
            "feedback reviewed",
        ),
    )
    submissions: list[taaf.game.Submission] = []
    ctx, _meter = build_llm_context(
        game_id="feedback-after-answer",
        submissions=submissions,
        rng=random.Random(7),
        backend=backend,
        submit=_collect_into(submissions),
    )
    agent = DuckAgent(
        reply_cap_tokens=64,
        check_mode=CheckMode.ENABLED,
        images="task+show",
        exact_token_counts=False,
        max_tokens_per_game=None,
        # The game ends at its stop, unaudited.
        audit_turn=False,
        llm_config=LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    )
    try:
        await agent._do_move(ctx, _view())
        assert backend._call_index == 2
        assert len(backend.requests) == 2
        # The call's show grids, then the answer it changed, then its failed check.
        sent_show, sent_changed, sent_check = backend.requests[1][-3:]
        assert isinstance(sent_show, ullm.conv.ListContent)
        assert isinstance(sent_changed, ullm.conv.ListContent)
        assert isinstance(sent_check, ullm.conv.ListContent)
        assert any(isinstance(item, ullm.conv.ImageContent) for item in sent_show.elements)
        assert sent_changed.elements[0].to_string().startswith("You changed test_out[0][0]")
        assert sent_check.elements[0].to_string().startswith("check(fn) FAILED")
        user_prompts = [element for element in agent.messages if isinstance(element, ullm.conv.UserPrompt)]
        assert len(user_prompts) == 4
        assert len(submissions) == 1
    finally:
        if agent.repl is not None:
            await agent.repl.clear_state()


async def test_deepcopied_agent_rewires_feedback_callbacks() -> None:
    """PythonTool intentionally drops process-local host functions on copy;
    the owning agent restores all enabled callbacks before use."""
    original = DuckAgent(reply_cap_tokens=None, check_mode=CheckMode.ENABLED, images="task+show")
    original_tool = original._ensure_repl()
    clone = copy.deepcopy(original)
    assert clone.repl is not None and clone.repl.host_functions == {}
    clone_tool = clone._ensure_repl()
    try:
        assert set(clone_tool.host_functions) == {
            CHECK_FAILURE_HOST_FUNCTION,
            PYTHON_CALL_INDEX_HOST_FUNCTION,
            SHOW_HOST_FUNCTION,
        }
        await clone_tool.setup_task(_view())
        await clone_tool._execute({"code": "show([[1]])"})
        assert clone_tool.has_feedback
    finally:
        await original_tool.clear_state()
        await clone_tool.clear_state()


@pytest.mark.parametrize("check_and_submit", [False, True])
async def test_mandatory_agent_ignores_direct_answers_then_submits_a_passing_check(check_and_submit: bool) -> None:
    """Mandatory mode keeps playing after ``test_out`` edits and ends on check."""
    backend = ullm.backends.stub.StubBackend.from_script(
        ullm.conv.LLMResponse(
            final_content=ullm.conv.ToolCallContent(
                tool_name="python",
                arguments={"code": "test_out[0][0] = np.zeros((1, 1), dtype=np.int8)"},
            )
        ),
        ullm.conv.LLMResponse(
            final_content=ullm.conv.ToolCallContent(
                tool_name="python",
                arguments={"code": "check(lambda grid: np.fliplr(grid))"},
            )
        ),
    )
    submissions: list[taaf.game.Submission] = []
    ctx, _meter = build_llm_context(
        game_id="mandatory-check",
        submissions=submissions,
        rng=random.Random(7),
        backend=backend,
        submit=_collect_into(submissions),
    )
    agent = DuckAgent(
        reply_cap_tokens=64,
        check_mode=CheckMode.MANDATORY,
        check_and_submit=check_and_submit,
        repl=DuckPythonTool(check_mode=CheckMode.MANDATORY, isolate_network=False),
        exact_token_counts=False,
        max_tokens_per_game=None,
        # Nothing for either to act on: a passing mandatory check ends the game.
        audit_turn=False,
        show_changed_answers=False,
        llm_config=LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    )
    view = _view()
    try:
        await agent._do_move(ctx, view)
        assert len(submissions) == 1
        expected = taaf.grid.Grid(np.fliplr(view.questions[0].data))
        assert submissions[0].attempts[0] == (expected, expected)
        assert agent.submissions_made == 1
        assert backend._call_index == 2
    finally:
        if agent.repl is not None:
            await agent.repl.clear_state()


async def test_check_and_submit_agent_saves_answers_and_keeps_refining() -> None:
    backend = ullm.backends.stub.StubBackend.from_script(
        ullm.conv.LLMResponse(
            final_content=ullm.conv.ToolCallContent(
                tool_name="python", arguments={"code": "check(lambda grid: np.fliplr(grid))"}
            )
        ),
        ullm.conv.LLMResponse(
            final_content=ullm.conv.ToolCallContent(tool_name="python", arguments={"code": "test_out[0][1] = [[7]]"})
        ),
        ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="Done.")),
    )
    submissions: list[taaf.game.Submission] = []
    ctx, _meter = build_llm_context(
        game_id="check-and-submit",
        submissions=submissions,
        rng=random.Random(7),
        backend=backend,
        submit=_collect_into(submissions),
    )
    agent = DuckAgent(
        reply_cap_tokens=64,
        check_mode=CheckMode.ENABLED,
        check_and_submit=True,
        repl=DuckPythonTool(check_mode=CheckMode.ENABLED, isolate_network=False),
        exact_token_counts=False,
        max_tokens_per_game=None,
        # The game ends at its stop, unaudited.
        audit_turn=False,
        llm_config=LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    )
    view = _view()
    try:
        await agent._do_move(ctx, view)
        assert backend._call_index == 3
        expected = taaf.grid.Grid(np.fliplr(view.questions[0].data))
        assert submissions[-1].attempts[0] == (expected, taaf.grid.Grid.from_lists([[7]]))
    finally:
        if agent.repl is not None:
            await agent.repl.clear_state()


@pytest.mark.parametrize("restore", [copy.deepcopy, _pickle_roundtrip])
async def test_restored_standalone_tool_rewires_on_task_setup(
    restore: Callable[[DuckPythonTool], DuckPythonTool],
) -> None:
    """The tool's own setup seam restores callbacks dropped by its base
    class's copy and pickle lifecycle, without requiring an agent owner."""
    original = DuckPythonTool(check_mode=CheckMode.ENABLED, show_enabled=True)
    restored = restore(original)
    assert restored.host_functions == {}
    try:
        await restored.setup_task(_view())
        assert set(restored.host_functions) == {
            CHECK_FAILURE_HOST_FUNCTION,
            PYTHON_CALL_INDEX_HOST_FUNCTION,
            SHOW_HOST_FUNCTION,
        }
        await restored._execute({"code": "show([[1]])"})
        assert restored.has_feedback
    finally:
        await original.clear_state()
        await restored.clear_state()


async def test_mandatory_tool_wires_the_success_callback() -> None:
    """Only mandatory mode exposes the checked-candidate host receiver."""
    tool = DuckPythonTool(check_mode=CheckMode.MANDATORY, isolate_network=False)
    try:
        await tool.setup_task(_view())
        assert CHECK_SUCCESS_HOST_FUNCTION in tool.host_functions
    finally:
        await tool.clear_state()


async def test_malformed_mandatory_success_callback_crashes_the_game() -> None:
    """A broken trusted submission callback is fatal, not model feedback."""
    tool = DuckPythonTool(check_mode=CheckMode.MANDATORY, isolate_network=False)
    try:
        await tool.setup_task(_view())
        code = (
            "host = next(cell.cell_contents for cell in check.__closure__ "
            "if callable(cell.cell_contents) and getattr(cell.cell_contents, '__name__', '') == 'host_call')\n"
            f"host({CHECK_SUCCESS_HOST_FUNCTION!r}, {{'grids': []}})"
        )
        with pytest.raises(RuntimeError, match="duck_check_success callback supplied 0 grids, expected 1"):
            await tool._execute({"code": code})
    finally:
        await tool.clear_state()


async def test_check_images_are_an_independent_feedback_mode(kernel: DuckPythonTool) -> None:
    """Only the check-image arm adds pixels, and error-only pairs never do."""
    kernel.check_images_enabled = True
    await kernel.setup_task(_view())
    await kernel._execute({"code": "check(lambda grid: grid)"})
    feedback = kernel.take_feedback()
    assert sum(isinstance(item, ullm.conv.ImageContent) for group in feedback for item in group) == 2

    await kernel._execute({"code": "check(lambda grid: grid[missing_name])"})
    feedback = kernel.take_feedback()
    assert sum(isinstance(item, ullm.conv.ImageContent) for group in feedback for item in group) == 0


def test_check_runtime_is_feature_gated() -> None:
    """The default namespace omits the oracle until its gate opens."""
    view = _view()
    assert "_duck_make_check" not in prelude()
    assert "check =" not in task_setup_code(view)
    assert "_duck_make_check" in prelude(include_check=True)
    assert "check = _duck_make_check" in task_setup_code(view, include_check=True)
    disabled = task_setup_code(view)
    assert "check =" not in disabled

    def grid(value: taaf.grid.Grid) -> str:
        return f"np.array({value.to_lists()!r}, dtype=np.int8)"

    expected = (
        f"train_in = [{', '.join(grid(candidate) for candidate, _ in view.given)}]\n"
        f"train_out = [{', '.join(grid(answer) for _, answer in view.given)}]\n"
        f"test_in = [{', '.join(grid(question) for question in view.questions)}]\n"
        f"test_out = [[None] * {taaf.game.N_ATTEMPTS} for _ in test_in]\n"
        "for _duck_grid in (*train_in, *train_out, *test_in):\n"
        "    _duck_grid.flags.writeable = False\n"
        "_duck_grid = None\n"
        "del _duck_grid\n"
        '__import__("duck.kernel_workspace", fromlist=["capture_surface"]).capture_surface(globals())\n'
    )
    assert disabled == expected


def test_show_runtime_only_exists_in_show_modes() -> None:
    """The compatibility namespace stays unchanged until the show gate opens."""
    view = _view()
    assert "_duck_make_show" not in prelude()
    assert "show =" not in task_setup_code(view)
    assert "_duck_make_show" in prelude(include_show=True)
    assert "show = _duck_make_show" in task_setup_code(view, include_show=True)


async def test_agent_stamps_check_gate_onto_an_injected_fresh_kernel() -> None:
    """An injected tool cannot leave the prompt and runtime in opposite modes."""
    tool = DuckPythonTool(check_mode=CheckMode.ENABLED)
    agent = DuckAgent(reply_cap_tokens=None, check_mode=CheckMode.DISABLED, repl=tool)
    try:
        assert agent._ensure_repl() is tool
        assert tool.check_mode is CheckMode.DISABLED
        assert tool.initial_code is not None and "_duck_make_check" not in tool.initial_code
    finally:
        await tool.clear_state()


async def test_default_tool_construction_uses_the_disabled_prelude() -> None:
    """The standalone tool preserves the default compatibility namespace."""
    tool = DuckPythonTool()
    try:
        assert tool.initial_code is not None and "_duck_make_check" not in tool.initial_code
        assert "check(fn)" not in tool.description
        await tool.setup_task(_view())
        out = await tool._execute({"code": "print('check' in globals(), 'CheckError' in globals())"})
        assert out.to_string() == "False False"
    finally:
        await tool.clear_state()


async def test_empty_output_is_named_rather_than_blank(kernel: DuckPythonTool) -> None:
    """A silent empty result reads as a broken tool."""
    out = await kernel._execute({"code": "x = 1"})
    assert out.to_string() == prompts.EMPTY_OUTPUT_NOTE


async def test_long_output_is_cut_and_says_so(kernel: DuckPythonTool) -> None:
    """A silent truncation reads as the model's own code having produced less
    than it did."""
    kernel.tool_output_tokens = 64
    out = await kernel._execute({"code": "print('x' * 5000)"})
    text = out.to_string()
    assert len(text) < 5000
    assert "cut to stay within" in text


async def test_a_cut_keeps_the_end_where_the_error_is(kernel: DuckPythonTool) -> None:
    """Output ends where the cell ended: a cell that printed a lot and then
    raised is cut in the middle, and the error stays in front of the model."""
    kernel.tool_output_tokens = 64
    out = (
        await kernel._execute({"code": "for i in range(500):\n    print('row', i)\nraise ValueError('at the end')"})
    ).to_string()
    assert "row 0" in out and "cut to stay within" in out and "ValueError: at the end" in out, out


async def test_the_kernel_runs_one_blas_thread(kernel: DuckPythonTool) -> None:
    out = (
        await kernel._execute(
            {"code": "import os\nprint(os.environ['OPENBLAS_NUM_THREADS'], os.environ['OMP_NUM_THREADS'])"}
        )
    ).to_string()
    assert out.strip() == "1 1", out


async def test_a_failing_check_names_check_but_not_our_code(kernel: DuckPythonTool) -> None:
    """The frame of a harness function the model called keeps the name it
    called and nothing else: no file, no line in our setup, no source."""
    await kernel.setup_task(_view())
    out = (await kernel._execute({"code": "check(lambda grid: grid)"})).to_string()
    assert 'File "<harness>", in check' in out and "CheckError" in out, out
    assert "<harness-setup>" not in out and 'File "<harness>", line' not in out, out
    source = (await kernel._execute({"code": "import inspect\nprint(inspect.getsource(check))"})).to_string()
    assert "_duck_" not in source and "OSError" in source, source


async def test_a_traceback_carries_no_absolute_paths(kernel: DuckPythonTool) -> None:
    """Cluster paths are tokens the model cannot act on, and they differ
    between machines, which makes an otherwise identical run's output
    differ."""
    out = await kernel._execute({"code": "raise ValueError('boom')"})
    text = out.to_string()
    assert "boom" in text
    assert "/scratch/" not in text and "/tmp/" not in text, text


async def test_syntax_errors_are_reported_without_reaching_the_kernel(kernel: DuckPythonTool) -> None:
    out = await kernel._execute({"code": "def broken(:"})
    assert "syntax error" in out.to_string().lower()
    assert kernel.is_error


async def test_an_empty_snippet_is_refused(kernel: DuckPythonTool) -> None:
    out = await kernel._execute({"code": "   "})
    assert "non-empty" in out.to_string()
    assert kernel.is_error


async def test_the_answer_read_survives_a_wrecked_namespace(kernel: DuckPythonTool) -> None:
    """It runs after every call the model makes, so it must survive whatever
    the model did — including deleting the name it reads."""
    await kernel.setup_task(_view())
    await kernel._execute({"code": "del test_out"})
    payload = await kernel.read_answer()
    assert payload is not None and payload.get("bound") is False

    await kernel._execute({"code": "test_out = 'not a list'"})
    payload = await kernel.read_answer()
    assert payload is not None and payload.get("bound") is False


async def test_the_answer_read_does_not_disturb_the_model_s_verdict(kernel: DuckPythonTool) -> None:
    """``is_error`` belongs to the model's call; the private read happens
    after it has been rendered but before the turn manager reads the flag."""
    await kernel.setup_task(_view())
    await kernel._execute({"code": "raise RuntimeError('the model broke it')"})
    assert kernel.is_error
    await kernel.read_answer()
    assert kernel.is_error, "the private read overwrote the verdict on the model's own code"


def test_setup_code_binds_one_slot_per_question_and_attempt() -> None:
    """The model assigns into the shape rather than building it, so the two
    attempts per question are never something it has to remember."""
    code = task_setup_code(_view(n_questions=3))
    assert f"[[None] * {taaf.game.N_ATTEMPTS} for _ in test_in]" in code


def test_a_missing_marker_reads_as_no_answer_not_a_bad_one() -> None:
    """No marker means the kernel died, timed out or was interrupted —
    nothing was learned about what the model had written, and reporting it as
    malformed would send a repair prompt about a slot that may be fine."""
    assert parse_answer_payload("no marker here") is None
    assert parse_answer_payload(f"{ANSWER_MARKER}not json") is None
    assert parse_answer_payload(f'{ANSWER_MARKER}{{"bound": true, "rows": []}}') == {"bound": True, "rows": []}


async def test_the_kernel_gets_enough_memory_to_import_numpy(kernel: DuckPythonTool) -> None:
    """At ullm's default limit this never completes: OpenBLAS reserves
    per-thread arenas out of address space and the import wedges rather than
    failing, so the kernel dies at its startup deadline with nothing to show
    for it. The prelude imports numpy, so this is the whole kernel."""
    assert kernel.memory_limit_mb >= 4096
    out = await kernel._execute({"code": "print(np.__name__)"})
    assert out.to_string().strip() == "numpy"


def test_system_prompt_states_the_actual_lifetime() -> None:
    """Telling the model its code is not saved when it IS saved wastes every
    snippet rewriting helpers; the reverse loses work it expected to keep."""
    text = DuckAgent(
        reply_cap_tokens=None,
    )._system_prompt_text()
    assert "not saved between calls" not in text
    assert "keeps your definitions" in text


def test_tool_description_is_not_contradicted_by_the_system_prompt() -> None:
    """The tool's own description and parameter help are in the same context
    window as the system prompt; one of them claiming ephemerality would
    contradict the other."""
    tool = DuckAgent(
        reply_cap_tokens=None,
    )._ensure_repl()
    blob = json.dumps({"description": tool.description, "parameters": tool.parameters})
    assert "ephemeral" not in blob and "not saved across tool calls" not in blob
    assert "persist" in blob


def test_the_tool_description_is_stamped_from_the_prompt_surface() -> None:
    """Re-stamped on every use, so a swapped prompt implementation reaches
    the tool schema too — which is also what the token estimate counts."""
    # Both texts, because which one the surface reads follows the oracle;
    # the shipped duck also binds show(), whose clause the surface appends.
    custom = prompts.DuckPrompts(
        python_tool_description_text="RUN CODE HERE.", check_python_tool_description_text="RUN CODE HERE."
    )
    tool = DuckAgent(reply_cap_tokens=None, prompts=custom)._ensure_repl()
    assert tool.description == "RUN CODE HERE." + custom.show_tool_description_clause


def test_a_sub_second_timeout_is_a_config_error() -> None:
    """Raised rather than quietly repaired: what the model is told and what
    the kernel enforces both come from one place."""
    with pytest.raises(ValueError, match="python_timeout_s"):
        DuckAgent(reply_cap_tokens=None, python_timeout_s=0)._python_timeout_s()


async def test_finish_tears_the_kernel_down() -> None:
    """``clear_state`` is a coroutine; firing it without awaiting leaves the
    subprocess alive for the rest of the run while raising nothing."""
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    tool = agent._ensure_repl()
    await tool._execute({"code": "x = 1"})
    await agent._on_finish(None, GameOutcome(state="finished", n_submitted=0))  # type: ignore[arg-type]
    assert agent.repl is None


async def test_a_passing_check_is_silent_unless_asked_to_print(kernel: DuckPythonTool) -> None:
    """A ``check(fn)`` that is not the snippet's final expression echoes
    nothing, and a model that sees no output reads the pass as a failure. The
    gate makes the pass print its report and the one-line reminder to record
    the answer wherever the call sits."""
    await kernel.setup_task(_view())
    snippet = "def fn(g):\n    return g[:, ::-1]\ncheck(fn)\nready = True"
    (silent,) = await _run_snippets(kernel, snippet)
    assert silent == prompts.EMPTY_OUTPUT_NOTE
    printing = DuckPythonTool(
        timeout=60.0,
        check_mode=CheckMode.ENABLED,
        check_prints_on_pass=True,
        initial_code=prelude(include_check=True, check_prints_on_pass=True),
    )
    try:
        await printing.setup_task(_view())
        (out,) = await _run_snippets(printing, snippet)
        assert "check: ALL TRAIN PAIRS OK" in out
        assert "record it: apply the function you just checked to each test_in[i]" in out
        assert "assign its result to test_out[i][0]" in out
        assert "fn(test_in[i])" not in out
        # A failure is a failure either way.
        (failed,) = await _run_snippets(printing, "def bad(g):\n    return g\ncheck(bad)")
        assert "CheckError" in failed and "record it" not in failed
    finally:
        await printing.clear_state()


async def test_the_mandatory_kernel_prints_a_pass_without_the_recording_hint() -> None:
    """Mandatory mode submits on its own, so telling the model to record the
    answer would be wrong there."""
    tool = DuckPythonTool(
        timeout=60.0,
        check_mode=CheckMode.MANDATORY,
        check_prints_on_pass=True,
        initial_code=prelude(include_check=True, check_prints_on_pass=True),
    )
    try:
        await tool.setup_task(_view())
        (out,) = await _run_snippets(tool, "def fn(g):\n    return g[:, ::-1]\ncheck(fn)\nready = True")
        assert "check: ALL TRAIN PAIRS OK" in out
        assert "record it" not in out
    finally:
        await tool.clear_state()


async def test_a_budget_note_rides_the_tool_result_when_the_agent_supplies_one(kernel: DuckPythonTool) -> None:
    await kernel.setup_task(_view())
    notes: list[str | None] = ["[budget] About 9,000 generated tokens remain for this puzzle.", None]
    kernel.budget_note = lambda: notes.pop(0)
    first, second = await _run_snippets(kernel, "print('a')", "print('b')")
    assert first == "a\n\n[budget] About 9,000 generated tokens remain for this puzzle."
    assert second == "b"


@pytest.mark.parametrize("code", ["", "if", "print('ok')"])
async def test_budget_status_reaches_successful_and_rejected_tool_calls(kernel: DuckPythonTool, code: str) -> None:
    agent = DuckAgent(prompts=prompts.DuckPrompts(budget_line=True), max_tokens_per_game=10_000)
    backend = ullm.backend.DummyBackend()

    def ignore(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        del submission, produced

    ctx, _meter = build_llm_context(
        game_id="budget", submissions=[], rng=random.Random(0), backend=backend, submit=ignore
    )
    kernel.budget_note = lambda: agent._budget_note(ctx)
    await kernel.setup_task(_view())
    backend.total_usage.completion_tokens = 2_000
    first = await kernel._execute({"code": code})
    assert first.to_string().endswith("\n\n[budget] Tokens left on this puzzle: about 8k of 10k.")
    backend.total_usage.completion_tokens = 3_000
    second = await kernel._execute({"code": code})
    assert second.to_string().endswith("\n\n[budget] Tokens left on this puzzle: about 7k of 10k.")


@pytest.mark.parametrize("prints", [False, True])
async def test_the_agent_stamps_the_pass_print_gate_onto_its_kernel(prints: bool) -> None:
    agent = DuckAgent(reply_cap_tokens=None, check_prints_on_pass=prints)
    repl = agent._ensure_repl()
    try:
        await repl.setup_task(_view())
        (out,) = await _run_snippets(repl, "check(lambda grid: np.fliplr(grid))\nready = True")
        assert ("check: ALL TRAIN PAIRS OK" in out) == prints
        assert ("record it" in out) == prints
    finally:
        await repl.clear_state()
