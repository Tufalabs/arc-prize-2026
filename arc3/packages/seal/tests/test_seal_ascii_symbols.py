"""The alphabet a board prints in (``SealAgent.ascii_symbols``): hex digits by
default, or the duck's letters. It reaches everything that writes or reads a
board as text -- the kernel's ``ascii_grid`` and ``from_ascii_grid`` and what
``help()`` says of them, the world model wherever it runs, the prompt's legend
and the index legend image -- and stays with its own agent where two share a
process, as two arms of a sweep can."""

from __future__ import annotations

import base64
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from framediff import grid as framediff_grid
from seal import SealAgent
from seal.palette import ARC_HEX_CHARS, ARC_LETTER_CHARS, AsciiLayout
from seal.repl import GameREPL

#: A world model that reads a sprite in letters as it loads.
LETTERS_MODEL = """\
SPRITE = from_ascii_grid('Wb')


def initial_state():
    return {'outcome': 'playing', 'energy': 2000}


def do_move(state, action):
    state['energy'] -= 1


def render(state):
    board = np.zeros((8, 8), dtype=np.int8)
    board[0, :2] = SPRITE[0]
    return board
"""


def _turn() -> dict[str, Any]:
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[0, 1] = 9
    frame = {"grid_b64": base64.b64encode(grid.tobytes()).decode("ascii"), "shape": [8, 8], "step": 0, "level": 1}
    return {"action": "", "outcome": "playing", "animation": [frame]}


async def test_the_kernel_prints_and_reads_boards_in_letters() -> None:
    tool = GameREPL(timeout=30.0, reset_before_each_call=False, use_world_model="none")
    tool.configure_ascii(row_numbers=False, column_group=None, symbols="letters")

    def action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        return {"ack": {"queued": len(normalized)}}

    tool.host_functions = {"get_state": lambda: {"level": 1, "turns": [], "available_actions": []}, "action": action}
    try:
        out = (
            await tool._execute(  # pyright: ignore[reportPrivateUsage]
                {
                    "code": "print(ascii_grid(np.array([[0, 9], [4, 15]])))\n"
                    "print(from_ascii_grid('Wb\\ncp').tolist())\n"
                    "help(ascii_grid)\n"
                }
            )
        ).to_string()
    finally:
        await tool.clear_state()
    assert "Wb\ncp" in out and "[[0, 9], [4, 15]]" in out, out
    assert "its letter from the color legend" in out, out


def test_the_world_model_reads_its_sprite_in_the_kernels_alphabet_here_and_in_the_search() -> None:
    """Loaded in the agent's process and again on unpickling in the search's,
    before that process configures anything; the process's own alphabet is
    back after either."""
    from seal.py_model import PyWorldModel

    model = PyWorldModel(
        {"world_model.py": LETTERS_MODEL.encode()}, [_turn()], ["UP"], ascii_layout=AsciiLayout(symbols="letters")
    )
    assert framediff_grid.alphabet() == ARC_HEX_CHARS
    copy = pickle.loads(pickle.dumps(model))
    for loaded in (model, copy):
        with loaded.active():
            assert loaded.render(loaded.initial_state())[0, :2].tolist() == [0, 9]
    assert framediff_grid.alphabet() == ARC_HEX_CHARS


async def test_the_search_tests_the_model_in_the_kernels_alphabet(tmp_path: Path) -> None:
    """Before it searches, the agent's process tests the model again on the
    files the REPL passed, and that test loads ``world_model.py`` afresh: a
    sprite it reads in letters reads in letters there too, rather than ending
    the game as a disagreement between two harnesses."""
    from seal.search import SearchReport, search_level
    from seal.state import FramePayload, TimelineEntry

    (tmp_path / "world_model.py").write_text(LETTERS_MODEL)
    seed = _turn()["animation"][0]
    frame = FramePayload(grid_b64=seed["grid_b64"], shape=seed["shape"], step=0, level=1)
    report = await search_level(
        directory=str(tmp_path),
        entries=[TimelineEntry(action=None, frame=frame, animation=[frame], level=1)],
        turns=[_turn()],
        model_actions=["UP"],
        game_actions=["UP"],
        level=1,
        seconds=5.0,
        memory_bytes=2 * 2**30,
        ascii_layout=AsciiLayout(symbols="letters"),
    )
    assert isinstance(report, SearchReport)
    assert framediff_grid.alphabet() == ARC_HEX_CHARS


def test_the_prompt_legend_follows_the_alphabet() -> None:
    hex_prompt = SealAgent(ascii_symbols="hex")._system_prompt_text()  # pyright: ignore[reportPrivateUsage]
    letters_prompt = SealAgent(ascii_symbols="letters")._system_prompt_text()  # pyright: ignore[reportPrivateUsage]
    assert '9="9" blue' in hex_prompt and '9="b" blue' in letters_prompt
    assert "digits for 0-9" not in letters_prompt


def test_two_agents_in_one_process_draw_their_own_index_legend() -> None:
    """The legend image is drawn for the alphabet it is asked for, not from a
    process setting another agent could have changed."""
    from seal.palette_legend import render_palette_legend

    hex_image = np.asarray(render_palette_legend(ARC_HEX_CHARS))
    letters_image = np.asarray(render_palette_legend(ARC_LETTER_CHARS))
    assert not np.array_equal(hex_image, letters_image)
    assert np.array_equal(hex_image, np.asarray(render_palette_legend()))


def test_an_unknown_alphabet_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown ascii_symbols"):
        SealAgent(ascii_symbols="runes")  # pyright: ignore[reportArgumentType]
