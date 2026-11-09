"""Axis labels on the board images (``SealAgent.axis_labels``).

Labelled, a board image is the unlabelled one inside a slate margin on its top
and left edges, with a label and a tick over every tenth row and column; the
diff image and the audit's boards carry the same margin, so every image of a
board lines up cell for cell. Off, the board image is the unlabelled one byte
for byte. Refused without images, and below the upscale the labels fit at.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
import taaf.game
import ullm.conv

from platypus.llm import frame_image
from seal import SealAgent
from seal.board_image import AXIS_TICK_STEP
from seal.state import FramePayload, TimelineEntry

#: The margin the labels sit in, in pixels, and its colour: the whale's.
PAD = 32
SLATE = (30, 30, 35)


def _grid() -> np.ndarray:
    """A 64 x 64 board with a few colours, so the grid is not one flat block."""
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10:20, 30:40] = 9
    grid[50:, :8] = 3
    return grid


def _entry(grid: np.ndarray, level: int = 1) -> FramePayload:
    return FramePayload(grid_b64=base64.b64encode(grid.tobytes()).decode(), shape=list(grid.shape), step=3, level=level)


def _images(content: ullm.conv.Content) -> list[np.ndarray]:
    assert isinstance(content, ullm.conv.ListContent)
    return [np.asarray(e.data) for e in content.elements if isinstance(e, ullm.conv.ImageContent)]


def _board_image(agent: SealAgent, grid: np.ndarray) -> np.ndarray:
    """The board image on a prompt (the legend, sent once a level, spent first)."""
    agent.palette_level_sent = 1
    return _images(agent._user_prompt_element("text", _entry(grid)).content)[-1]  # pyright: ignore[reportPrivateUsage]


def _drawn(pixels: np.ndarray) -> bool:
    """Whether anything is drawn on the margin here (glyphs are antialiased)."""
    return bool((pixels.reshape(-1, 3) != SLATE).any(axis=1).any())


def test_off_the_board_image_is_the_unlabelled_one() -> None:
    agent = SealAgent()
    plain = frame_image(taaf.game.Frame(data=_grid()), scale=agent.multimodal_upscale).data
    assert np.asarray(plain).tobytes() == _board_image(agent, _grid()).tobytes()


def test_labelled_the_board_sits_inside_a_margin_with_every_tenth_row_and_column_labelled() -> None:
    agent = SealAgent(axis_labels=True)
    scale = agent.multimodal_upscale
    plain = np.asarray(frame_image(taaf.game.Frame(data=_grid()), scale=scale).data)
    labelled = _board_image(agent, _grid())
    assert labelled.shape == (64 * scale + PAD, 64 * scale + PAD, 3)
    assert np.array_equal(labelled[PAD:, PAD:], plain), "the board itself is untouched"
    for cell in range(64):
        centre = PAD + cell * scale + scale // 2
        ticked = cell % AXIS_TICK_STEP == 0
        # The tick between label and board, on the margin's last pixels, over
        # every labelled column and beside every labelled row, and no other.
        assert _drawn(labelled[PAD - 3 : PAD, centre]) == ticked, f"column tick {cell}"
        assert _drawn(labelled[centre, PAD - 3 : PAD]) == ticked, f"row tick {cell}"
        # The label band above a column and the one left of a row, short of
        # the ticks: drawn on at a label, clear halfway between two.
        if ticked or cell % AXIS_TICK_STEP == AXIS_TICK_STEP // 2:
            top = labelled[: PAD - 6, centre - 1 : centre + 2]
            left = labelled[centre - 1 : centre + 2, : PAD - 6]
            assert _drawn(top) == ticked and _drawn(left) == ticked, f"label at {cell}"


def test_the_diff_image_carries_the_same_margin() -> None:
    """Changed cells match the labelled board's pixels, and so does the
    margin: the two line up cell for cell."""
    agent = SealAgent(axis_labels=True, diff_image=True)
    before, after = _grid(), _grid()
    after[30:34, 30:34] = 14
    changed = before != after
    agent.diff_image_pending = {
        "kind": "previous",
        "turn_index": 1,
        "step": 3,
        "move": "Action('UP')",
        "won": False,
        "shape": [64, 64],
        "frame_b64": base64.b64encode(after.tobytes()).decode(),
        "changed_b64": base64.b64encode(changed.astype(np.uint8).tobytes()).decode(),
    }
    agent.palette_level_sent = 1
    board, diff = _images(agent._user_prompt_element("text", _entry(after)).content)  # pyright: ignore[reportPrivateUsage]
    assert diff.shape == board.shape
    assert np.array_equal(diff[:PAD], board[:PAD]) and np.array_equal(diff[:, :PAD], board[:, :PAD])
    scale = agent.multimodal_upscale
    mask = changed.repeat(scale, axis=0).repeat(scale, axis=1)
    assert np.array_equal(diff[PAD:, PAD:][mask], board[PAD:, PAD:][mask])


def test_the_audits_boards_are_labelled_too() -> None:
    agent = SealAgent(axis_labels=True, strategy_audit="verbose")
    agent.game_action_names = ["RESET", "UP"]
    agent.history_entries = [
        TimelineEntry(action=None, frame=_entry(_grid()), animation=[], level=1),
        TimelineEntry(action=None, frame=_entry(_grid()), animation=[], level=1),
    ]
    images = _images(agent._build_audit_prompt_element().content)  # pyright: ignore[reportPrivateUsage]
    assert [image.shape for image in images] == [_board_image(SealAgent(axis_labels=True), _grid()).shape] * 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"multimodal_context": ""}, "requires images"),
        ({"multimodal_upscale": 3}, "requires multimodal_upscale >= 4"),
    ],
)
def test_refused_where_labels_cannot_be_drawn(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SealAgent(axis_labels=True, **kwargs).prepare_for_backend()  # pyright: ignore[reportArgumentType]
