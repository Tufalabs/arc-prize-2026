"""How ``ascii_grid`` lays a board out is the agent's choice, not the model's.

Two sweep axes: row numbers (on by default — every other numbered text the
model reads has them, and the audit run's hand-rolled renderers added
exactly this) and a tab every N cells (unset until swept). Both travel from
the agent's fields into the kernel's bootstrap and into the prompt's
description of a render, so neither can disagree with what the model sees.
"""

from __future__ import annotations

import numpy as np
import pytest

from seal import SealAgent
from seal.harness import game

BOARD = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 0, 0, 0, 0]], dtype=np.int8)


@pytest.fixture
def restore_layout():
    """The layout is module state in the kernel; put the default back."""
    yield
    game._configure_ascii_grid(row_numbers=True, column_group=None)


def test_the_default_render_numbers_rows_like_the_file_tools(restore_layout: None) -> None:
    game._configure_ascii_grid(row_numbers=True, column_group=None)
    assert game.ascii_grid(BOARD) == "0\t0123456789\n1\tabcdef0000"


def test_row_numbers_off_is_the_whales_render(restore_layout: None) -> None:
    game._configure_ascii_grid(row_numbers=False, column_group=None)
    assert game.ascii_grid(BOARD) == "0123456789\nabcdef0000"


def test_a_column_group_is_a_tab_every_n_cells(restore_layout: None) -> None:
    game._configure_ascii_grid(row_numbers=True, column_group=4)
    assert game.ascii_grid(BOARD) == "0\t0123\t4567\t89\n1\tabcd\tef00\t00"
    with pytest.raises(ValueError):
        game._configure_ascii_grid(row_numbers=True, column_group=0)


def test_the_flags_reach_the_kernel_bootstrap_and_the_prompt() -> None:
    """One agent field, three readers: the kernel's initial_code, the system
    prompt, and nothing the model can pass."""
    agent = SealAgent(ascii_row_numbers=False, ascii_column_group=8, ascii_symbols="letters")
    repl = agent._ensure_repl()
    assert "_sg._configure_ascii_grid(row_numbers=False, column_group=8, symbols='letters')" in repl.initial_code
    prompt = agent._system_prompt_text()
    assert "puts a tab after every 8 cells" in prompt
    assert "prefixes each row with its index" not in prompt
    assert '9="b" blue' in prompt

    default = SealAgent()
    assert (
        "_configure_ascii_grid(row_numbers=True, column_group=None, symbols='letters')"
        in default._ensure_repl().initial_code
    )
    assert "prefixes each row with its index" in default._system_prompt_text()
    assert "tab after every" not in default._system_prompt_text()
    # Reconfiguring an existing tool moves its bootstrap with it.
    default.ascii_column_group = 16
    assert "column_group=16" in default._ensure_repl().initial_code


@pytest.mark.parametrize(
    ("row_numbers", "column_group"),
    [(True, None), (False, None), (True, 4), (False, 3)],
    ids=["numbered", "plain", "numbered_grouped", "plain_grouped"],
)
def test_from_ascii_grid_reads_back_what_ascii_grid_printed(
    restore_layout: None, row_numbers: bool, column_group: int | None
) -> None:
    """Whatever layout the agent chose, a render pasted back is the board: the
    prompt tells the model it can, so no layout may break it."""
    game._configure_ascii_grid(row_numbers=row_numbers, column_group=column_group)
    board = np.random.default_rng(0).integers(0, 16, (12, 17)).astype(np.int8)
    restored = game.from_ascii_grid(game.ascii_grid(board))
    assert restored.dtype == np.int8
    assert np.array_equal(restored, board)
