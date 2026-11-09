"""``show_changed_answers``: a ``python`` call that changes ``test_out`` is
followed by each changed answer as an image, beside its test input.

The tool asks the agent for that content right after it reads the answer, and
queues it as a user message of its own, after any ``show`` grids. These tests
wire a real kernel to an agent the way ``play`` does, so the comparison, the
order and the wording are each read off what the model would be sent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
import pytest
import ullm.conv
from duck import CheckMode, DuckAgent
from duck.python_tool import DuckPythonTool
from duck.state import reading_from_payload

import taaf.game
import taaf.grid


def _view(n_questions: int = 2) -> taaf.game.GameView:
    """A mirror task with 3x4 grids, so a label's size is known."""
    rng = np.random.default_rng(0)

    def grid() -> taaf.grid.Grid:
        return taaf.grid.Grid(rng.integers(0, 10, size=(3, 4), dtype=np.int8))

    given = tuple((g, taaf.grid.Grid(np.fliplr(g.data))) for g in (grid() for _ in range(2)))
    return taaf.game.GameView(game_id="changed", given=given, questions=tuple(grid() for _ in range(n_questions)))


def _wired(agent: DuckAgent, view: taaf.game.GameView) -> DuckPythonTool:
    """A kernel with the agent's answer hooks, as ``play`` sets them."""
    tool = DuckPythonTool(timeout=60.0, check_mode=CheckMode.ENABLED, show_enabled=True, image_scale=4)

    def absorb(payload: dict[str, object] | None) -> None:
        agent.reading = reading_from_payload(payload)

    tool.on_answer = absorb
    tool.answer_feedback = lambda: agent._changed_answers_content(view)  # pyright: ignore[reportPrivateUsage]
    return tool


@pytest.fixture
async def setup() -> AsyncIterator[tuple[DuckPythonTool, DuckAgent]]:
    agent = DuckAgent(show_changed_answers=True)
    view = _view()
    tool = _wired(agent, view)
    await tool.setup_task(view)
    try:
        yield tool, agent
    finally:
        await tool.clear_state()


def _groups(tool: DuckPythonTool) -> list[list[str]]:
    """The queued user messages, an image as its alt text in angle brackets."""
    return [
        [
            f"<{element.alt_text}>" if isinstance(element, ullm.conv.ImageContent) else element.to_string()
            for element in group
        ]
        for group in tool.take_feedback()
    ]


async def _run(tool: DuckPythonTool, code: str) -> list[list[str]]:
    await tool._execute({"code": code})  # pyright: ignore[reportPrivateUsage]
    return _groups(tool)


async def test_an_assigned_answer_is_shown_after_its_test_input(setup: tuple[DuckPythonTool, DuckAgent]) -> None:
    tool, _agent = setup

    groups = await _run(tool, "test_out[1][0] = np.fliplr(test_in[1])")

    assert groups == [
        [
            "You changed test_out[1][0] to this:",
            "test_in[1]  (3x4)",
            "<test_in[1]  (3x4)>",
            "test_out[1][0]  (3x4)",
            "<test_out[1][0]  (3x4)>",
        ]
    ]


async def test_an_answer_left_as_it_was_is_not_shown_again(setup: tuple[DuckPythonTool, DuckAgent]) -> None:
    tool, _agent = setup
    await _run(tool, "test_out[0][0] = np.fliplr(test_in[0])")

    assert await _run(tool, "print(test_out[0][0].shape)") == []
    # Rebound to an equal array: the same answer, so nothing to show.
    assert await _run(tool, "test_out[0][0] = np.fliplr(test_in[0]).copy()") == []
    # One cell edited in place (on a copy: ``fliplr`` is a view of a read-only
    # task array).
    changed = await _run(
        tool, "test_out[0][0] = np.fliplr(test_in[0]).copy(); test_out[0][0][0, 0] = (test_out[0][0][0, 0] + 1) % 10"
    )
    assert changed and changed[0][0] == "You changed test_out[0][0] to this:"


async def test_both_attempts_changed_share_one_test_input(setup: tuple[DuckPythonTool, DuckAgent]) -> None:
    tool, _agent = setup

    groups = await _run(tool, "test_out[0][0] = np.fliplr(test_in[0]); test_out[0][1] = test_in[0]")

    assert groups == [
        [
            "You changed test_out[0][0] and test_out[0][1] to this:",
            "test_in[0]  (3x4)",
            "<test_in[0]  (3x4)>",
            "test_out[0][0]  (3x4)",
            "<test_out[0][0]  (3x4)>",
            "test_out[0][1]  (3x4)",
            "<test_out[0][1]  (3x4)>",
        ]
    ]


async def test_the_changed_answers_follow_the_shown_grids(setup: tuple[DuckPythonTool, DuckAgent]) -> None:
    tool, _agent = setup

    groups = await _run(tool, "show(test_in[0], 'mine'); test_out[0][0] = np.fliplr(test_in[0])")

    assert [group[0] for group in groups] == ["Rendered grids from show():", "You changed test_out[0][0] to this:"]


async def test_an_unusable_answer_is_not_shown(setup: tuple[DuckPythonTool, DuckAgent]) -> None:
    tool, _agent = setup

    assert await _run(tool, "test_out[0][0] = np.full((2, 2), 11)") == []


def test_the_system_prompt_says_so_only_with_the_flag() -> None:
    line = "the next user message shows each changed answer as an image, beside its test input"

    assert line in DuckAgent(show_changed_answers=True)._system_prompt_text()  # pyright: ignore[reportPrivateUsage]
    assert line not in DuckAgent(show_changed_answers=False)._system_prompt_text()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"images": "off", "text_grids": True}, "needs images on"),
        ({"check_mode": CheckMode.MANDATORY}, "mandatory check mode"),
    ],
)
def test_configurations_it_cannot_serve_are_refused(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DuckAgent(show_changed_answers=True, **fields)._system_prompt_text()  # pyright: ignore[reportPrivateUsage, reportArgumentType]
