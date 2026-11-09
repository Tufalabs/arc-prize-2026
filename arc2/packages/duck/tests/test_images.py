"""Pixel-exact grid rendering, composition, and optional gridlines."""

from __future__ import annotations

import hashlib

import numpy as np
from duck import images

import taaf.grid


def _grids() -> tuple[taaf.grid.Grid, taaf.grid.Grid]:
    left = taaf.grid.Grid.from_lists([[0, 1], [2, 3]])
    right = taaf.grid.Grid.from_lists([[4], [5], [6]])
    return left, right


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def test_gridlines_off_preserves_existing_image_bytes() -> None:
    left, right = _grids()
    render = images.render(left, scale=4)
    side_by_side = images.side_by_side([left, right], scale=4, gap=3, margin=2)
    pair = images.pair_image(left, right, scale=4, alt_text="pair")
    question = images.question_image(left, scale=4, alt_text="question")

    assert _digest(render) == "883a9826b0f51d44cd942ce8246f2475907408d60a601154f868e94935d4898a"
    assert _digest(side_by_side) == "291faa8ca8015968b6a83d9fde2c04e69eb08b314d71b0fcd6150946bf762e6e"
    assert _digest(pair.data) == "68344dc5b67c9b93c1a2b0ef46ef045224319f4a23aaa4eb60a27dccf067b190"
    assert _digest(question.data) == "4be2572d1557e59121afb46b5803b5a3dd5f87a61c12f3c221097513f9a5dc00"
    assert np.array_equal(render, images.render(left, scale=4, gridlines=False))
    assert np.array_equal(
        side_by_side,
        images.side_by_side([left, right], scale=4, gap=3, margin=2, gridlines=False),
    )
    assert np.array_equal(pair.data, images.pair_image(left, right, scale=4, gridlines=False).data)
    assert np.array_equal(question.data, images.question_image(left, scale=4, gridlines=False).data)


def test_gridlines_form_a_thin_lattice_without_changing_cell_colours() -> None:
    left, _ = _grids()
    rendered = images.render(left, scale=4, gridlines=True)

    assert rendered.shape == (11, 11, 3)
    line_positions = (0, 5, 10)
    for position in line_positions:
        assert np.all(rendered[position, :, :] == images.GRIDLINE_COLOR)
        assert np.all(rendered[:, position, :] == images.GRIDLINE_COLOR)
    assert np.all(rendered[1:5, 1:5, :] == images.PALETTE[0])
    assert np.all(rendered[1:5, 6:10, :] == images.PALETTE[1])
    assert np.all(rendered[6:10, 1:5, :] == images.PALETTE[2])
    assert np.all(rendered[6:10, 6:10, :] == images.PALETTE[3])


def test_gridlines_start_at_scale_four_and_reach_composed_images() -> None:
    left, right = _grids()
    assert np.array_equal(
        images.render(left, scale=3, gridlines=True),
        images.render(left, scale=3, gridlines=False),
    )

    expected_pair = images.side_by_side([left, right], scale=4, gridlines=True)
    expected_question = images.side_by_side([left], scale=4, gridlines=True)
    pair = images.pair_image(left, right, scale=4, alt_text="pair", gridlines=True)
    question = images.question_image(left, scale=4, alt_text="question", gridlines=True)

    assert pair.alt_text == "pair"
    assert question.alt_text == "question"
    assert np.array_equal(pair.data, expected_pair)
    assert np.array_equal(question.data, expected_question)
    assert not np.array_equal(pair.data, images.pair_image(left, right, scale=4).data)
    assert not np.array_equal(question.data, images.question_image(left, scale=4).data)
