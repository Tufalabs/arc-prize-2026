"""How an ASCII grid is laid out is the agent's choice, not the model's.

One sweep axis — each row prefixed with its index and a tab, the shape every
other numbered text a model reads has — and the thing worth testing is its
REACH: the task prompt's digit rows, the failed-check feedback and the
kernel's own ``ascii_grid`` all render through one function, so no surface
can be left behind when the flag moves. The prompt's description travels with
it, since a numbered row the prompt calls unnumbered is worse than either.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import ullm.conv
from duck import CheckMode, kernel_code
from duck.agent import DuckAgent
from duck.python_tool import _check_failure_content

import taaf.game
import taaf.grid

BOARD = taaf.grid.Grid.from_lists([[0, 1, 2], [3, 4, 5]])


def _kernel_ascii_grid(*, row_numbers: bool) -> Callable[[Any], str]:
    """The kernel's own ``ascii_grid``, built and executed as the kernel gets it."""
    namespace: dict[str, Any] = {}
    exec(kernel_code._ascii_grid_source(row_numbers=row_numbers), namespace)  # noqa: S102
    return cast(Callable[[Any], str], namespace["ascii_grid"])


def _view() -> taaf.game.GameView:
    """A one-pair task, enough for a prompt that shows digit rows."""
    spec = taaf.game.GameSpec(
        game_id="rows",
        pairs=(
            taaf.game.Pair(question=BOARD, answer=BOARD, given=True),
            taaf.game.Pair(question=BOARD, answer=None, given=False),
        ),
    )
    return taaf.game.ArcGame(spec_data=spec).start_game()


def test_off_is_one_character_per_cell_and_on_prefixes_index_and_tab() -> None:
    assert kernel_code.grid_text(BOARD, row_numbers=False) == "012\n345"
    assert kernel_code.grid_text(BOARD, row_numbers=True) == "0\t012\n1\t345"


def test_the_kernels_own_render_agrees_with_the_hosts() -> None:
    """Two implementations of one layout, in two languages of the same
    program: the model calls one and is shown the other, and a disagreement
    would read as the board having changed."""
    for row_numbers in (False, True):
        rendered = _kernel_ascii_grid(row_numbers=row_numbers)(np.array(BOARD.data))
        assert rendered == kernel_code.grid_text(BOARD, row_numbers=row_numbers)


def _harness_source(initial_code: str | None) -> str:
    """What the prelude compiles under the harness's own name: the kernel's
    functions, ``ascii_grid`` among them."""
    for node in ast.walk(ast.parse(initial_code or "")):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "compile":
            source = node.args[0]
            assert isinstance(source, ast.Constant) and isinstance(source.value, str)
            return source.value
    raise AssertionError("the prelude compiles no harness source")


def test_the_kernels_docstring_describes_the_layout_it_prints() -> None:
    """``help(ascii_grid)`` is a surface the model reads."""
    assert "prefixed with its index" in (_kernel_ascii_grid(row_numbers=True).__doc__ or "")
    assert "prefixed with its index" not in (_kernel_ascii_grid(row_numbers=False).__doc__ or "")


def test_every_surface_the_model_reads_a_grid_on_moves_together() -> None:
    """The reach the flag is for. A row rendered numbered in one place and
    bare in another teaches the model that the index is sometimes a cell."""
    for row_numbers in (False, True):
        agent = DuckAgent(
            reply_cap_tokens=None,
            ascii_row_numbers=row_numbers,
            images="off",
            text_grids=True,
            check_mode=CheckMode.ENABLED,
            show_changed_answers=False,
        )
        expected = kernel_code.grid_text(BOARD, row_numbers=row_numbers)

        task = agent._task_prompt(_view())
        content = task.content
        assert isinstance(content, ullm.conv.ListContent)
        task_text = "\n".join(
            element.text for element in content.elements if isinstance(element, ullm.conv.StringContent)
        )
        assert expected in task_text, "the task prompt's digit rows"

        feedback = _check_failure_content(
            {"pairs": [{"index": 0, "got": [[0, 1, 2], [3, 4, 5]]}]},
            include_images=False,
            image_scale=4,
            image_gridlines=False,
            row_numbers=row_numbers,
        )
        last = feedback[-1]
        assert isinstance(last, ullm.conv.StringContent)
        assert expected in last.text, "the failed-check feedback"

        kernel_source = kernel_code._ascii_grid_source(row_numbers=row_numbers)
        assert kernel_source in _harness_source(agent._ensure_repl().initial_code), "the kernel"


def test_the_prompt_describes_whichever_layout_is_rendered() -> None:
    """One line either way rather than a base line plus a qualifier, and the
    task prompt's own rows are called out where they carry an index."""
    numbered = DuckAgent(
        reply_cap_tokens=None, ascii_row_numbers=True, images="off", text_grids=True, show_changed_answers=False
    )._system_prompt_text()
    assert "Each row is prefixed with its index" in numbered
    assert "rows on separate lines" not in numbered
    assert "the index is a label, not a cell" in numbered

    plain = DuckAgent(
        reply_cap_tokens=None, ascii_row_numbers=False, images="off", text_grids=True, show_changed_answers=False
    )._system_prompt_text()
    assert "rows on separate lines" in plain
    assert "prefixed with its index" not in plain
    assert "not a cell" not in plain


def test_the_shipped_duck_numbers_every_row() -> None:
    """The duck as it deploys. Asserted as an outcome across the surfaces
    rather than as the field's value, so flipping one default and not the
    others is caught."""
    agent = DuckAgent(
        reply_cap_tokens=None, images="off", text_grids=True, check_mode=CheckMode.ENABLED, show_changed_answers=False
    )
    assert kernel_code._ascii_grid_source(row_numbers=True) in _harness_source(agent._ensure_repl().initial_code)
    assert "prefixed with its index" in agent._system_prompt_text()

    content = agent._task_prompt(_view()).content
    assert isinstance(content, ullm.conv.ListContent)
    task_text = "\n".join(element.text for element in content.elements if isinstance(element, ullm.conv.StringContent))
    assert kernel_code.grid_text(BOARD, row_numbers=True) in task_text
    assert "0\t012" in task_text
