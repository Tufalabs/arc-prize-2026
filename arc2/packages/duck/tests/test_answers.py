"""The answer path: what the model writes into ``test_out``, and what stands.

Upstream's counterpart is `test_live_actions.py` — a model acting on a live
game through a host call. There is no live game here: the answer is a kernel
variable, read after every call the model makes and submitted whenever it is
readable. So the claims are different ones, and this file is written rather
than ported.
"""

from __future__ import annotations

import numpy as np
import pytest
from duck.python_tool import DuckPythonTool
from duck.state import AnswerReading, reading_from_payload

import taaf.game
import taaf.grid


def _view(n_questions: int = 2) -> taaf.game.GameView:
    rng = np.random.default_rng(1)

    def grid(rows: int = 3, cols: int = 4) -> taaf.grid.Grid:
        return taaf.grid.Grid(rng.integers(0, 10, size=(rows, cols), dtype=np.int8))

    given = tuple((g, taaf.grid.Grid(np.fliplr(g.data))) for g in (grid(), grid()))
    return taaf.game.GameView(game_id="answers", given=given, questions=tuple(grid() for _ in range(n_questions)))


@pytest.fixture
async def kernel():
    tool = DuckPythonTool(timeout=60.0)
    try:
        yield tool
    finally:
        await tool.clear_state()


async def _read(tool: DuckPythonTool) -> AnswerReading:
    return reading_from_payload(await tool.read_answer())


async def test_nothing_written_is_not_an_answer(kernel: DuckPythonTool) -> None:
    view = _view()
    await kernel.setup_task(view)
    reading = await _read(kernel)
    assert not reading.is_complete(2)
    assert reading.submission(view.questions) is None, "with nothing usable there is nothing to salvage"
    assert reading.problems(2) == ["test_out[0] is still unset", "test_out[1] is still unset"]


async def test_a_partial_answer_submits_but_does_not_end_the_game(kernel: DuckPythonTool) -> None:
    """The distinction the whole design turns on: placeholders let partial
    work score, and must never be mistaken for the model having finished."""
    view = _view()
    await kernel.setup_task(view)
    await kernel._execute({"code": "test_out[0][0] = np.fliplr(test_in[0])"})
    reading = await _read(kernel)

    assert not reading.is_complete(2), "one of two answered is not finished"
    submission = reading.submission(view.questions)
    assert submission is not None, "the answered question must still be able to score"
    assert reading.n_placeholders(2) == 1
    assert reading.problems(2) == ["test_out[1] is still unset"]


async def test_the_placeholder_is_the_question_itself(kernel: DuckPythonTool) -> None:
    """Always a valid grid, because it came from the task, and wrong in the
    ordinary way rather than malformed."""
    view = _view()
    await kernel.setup_task(view)
    await kernel._execute({"code": "test_out[0][0] = np.fliplr(test_in[0])"})
    submission = (await _read(kernel)).submission(view.questions)
    assert submission is not None
    assert np.array_equal(submission.attempts[1][0].data, view.questions[1].data)


async def test_one_attempt_is_duplicated_into_both(kernel: DuckPythonTool) -> None:
    """A model that offers one answer is not made to invent another."""
    view = _view(n_questions=1)
    await kernel.setup_task(view)
    await kernel._execute({"code": "test_out[0][0] = np.fliplr(test_in[0])"})
    submission = (await _read(kernel)).submission(view.questions)
    assert submission is not None
    assert len(submission.attempts[0]) == taaf.game.N_ATTEMPTS
    assert submission.attempts[0][0] == submission.attempts[0][1]


async def test_two_distinct_attempts_survive(kernel: DuckPythonTool) -> None:
    """The second attempt is free score, so it must not be quietly collapsed."""
    view = _view(n_questions=1)
    await kernel.setup_task(view)
    await kernel._execute({"code": "test_out[0][0] = np.fliplr(test_in[0])\ntest_out[0][1] = np.flipud(test_in[0])"})
    submission = (await _read(kernel)).submission(view.questions)
    assert submission is not None
    assert submission.attempts[0][0] != submission.attempts[0][1]


async def test_a_row_level_assignment_is_absorbed(kernel: DuckPythonTool) -> None:
    """``test_out[0] = arr`` skips the attempt index. It is a slip, not a
    different intention, and costing the model a repair round for it would be
    the harness being pedantic."""
    view = _view(n_questions=1)
    await kernel.setup_task(view)
    await kernel._execute({"code": "test_out[0] = np.fliplr(test_in[0])"})
    reading = await _read(kernel)
    assert reading.is_complete(1)
    assert reading.submission(view.questions) is not None


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("test_out[0][0] = 'nope'", "not numbers"),
        ("test_out[0][0] = np.zeros((40, 40), dtype=np.int8)", "each side must be 1..30"),
        ("test_out[0][0] = np.full((3, 3), 1.5)", "non-integer values"),
        ("test_out[0][0] = np.full((3, 3), 99)", "outside 0..9"),
        ("test_out[0][0] = np.zeros((2, 2, 2), dtype=np.int8)", "a grid is 2-D"),
    ],
)
async def test_a_malformed_answer_is_named_rather_than_raised(kernel: DuckPythonTool, code: str, expected: str) -> None:
    """Every one of these comes back to the model as something it can fix, in
    its own index vocabulary — none of them crashes the game."""
    view = _view(n_questions=1)
    await kernel.setup_task(view)
    await kernel._execute({"code": code})
    problems = (await _read(kernel)).problems(1)
    assert problems and expected in problems[0], problems


async def test_a_whole_float_answer_is_accepted(kernel: DuckPythonTool) -> None:
    """Cast without rounding: 3.0 is a 3, and refusing it would punish the
    model for numpy's default dtype."""
    view = _view(n_questions=1)
    await kernel.setup_task(view)
    await kernel._execute({"code": "test_out[0][0] = np.fliplr(test_in[0]).astype(float)"})
    reading = await _read(kernel)
    assert reading.problems(1) == []
    assert reading.is_complete(1)


async def test_the_wrong_number_of_entries_is_reported(kernel: DuckPythonTool) -> None:
    view = _view(n_questions=2)
    await kernel.setup_task(view)
    await kernel._execute({"code": "test_out.append([None, None])"})
    reading = await _read(kernel)
    assert reading.problems(2) == ["test_out has 3 entries, not 2"]
    assert reading.submission(view.questions) is None


async def test_a_spare_broken_attempt_is_not_nagged_about(kernel: DuckPythonTool) -> None:
    """The model has answered; spending a round on the unused slot buys
    nothing."""
    view = _view(n_questions=1)
    await kernel.setup_task(view)
    await kernel._execute({"code": "test_out[0][0] = np.fliplr(test_in[0])\ntest_out[0][1] = 'rubbish'"})
    reading = await _read(kernel)
    assert reading.problems(1) == []
    assert reading.is_complete(1)


def test_a_dead_kernel_reads_as_no_answer_not_a_bad_one() -> None:
    reading = reading_from_payload(None)
    assert not reading.bound
    assert reading.submission(()) is None
    assert "no answer" in reading.problems(1)[0]
