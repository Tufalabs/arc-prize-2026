"""The world-model test harness: the render contract, the full-history
namespace, and the replay's verdicts — including the seal's loosenings
(wildcard cells, flexible energy) and refusals (Unknown on observed
moves)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from seal.harness import game as harness_game
from seal.harness import test as harness_test
from seal.harness.test import _render_grid, _RenderContractError

PASSING_MODEL = """
import numpy as np

start = int(history[0].frame.grid[0, 0])


def initial_state():
    return {'outcome': 'playing', 'energy': 1.0, 'colour': start}


def do_move(state, move):
    pass


def render(state):
    return np.full((8, 8), state['colour'], dtype=np.int8)
"""


def _payload(grid: np.ndarray, action: str = "", outcome: str = "playing") -> dict[str, Any]:
    """One host payload entry whose settled board is ``grid``."""
    return {
        "action": action,
        "outcome": outcome,
        "animation": [
            {
                "grid_b64": base64.b64encode(grid.astype(np.uint8).tobytes()).decode(),
                "shape": list(grid.shape),
                "step": 0,
                "level": 1,
            }
        ],
    }


def _history(*grids: np.ndarray, outcomes: tuple[str, ...] | None = None) -> harness_game.History:
    """A history whose turns settle on ``grids`` — the first is the seed turn."""
    history = harness_game.History()
    for index, grid in enumerate(grids):
        outcome = outcomes[index] if outcomes else "playing"
        history._append_turn(_payload(grid, action="" if index == 0 else "ACTION1", outcome=outcome))
    return history


def _two_same(fill: int = 3, size: int = 8) -> harness_game.History:
    """Two identical frames — enough to load, render and replay one turn."""
    frame = np.full((size, size), fill, dtype=np.int8)
    return _history(frame, frame.copy())


def _validate(path: Path, history: harness_game.History, **kwargs: Any) -> harness_test.TestResult:
    return harness_test.validate_world_model(str(path), history, available_actions=["ACTION1"], **kwargs)


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "world_model.py"
    path.write_text(source)
    return path


def _ns(render: "Callable[[dict[str, Any]], Any]") -> dict[str, Any]:
    return {"render": render}


def _returns(grid: Any) -> "Callable[[dict[str, Any]], Any]":
    """A ``render`` that hands back exactly this, whatever it is."""

    def render(_state: dict[str, Any]) -> Any:
        return grid

    return render


# --- the render contract ------------------------------------------------


def test_a_grid_render_comes_back_as_the_board() -> None:
    grid = _render_grid(_ns(_returns(np.full((8, 8), 4, dtype=np.int8))), {}, 8, 8)
    assert grid.shape == (8, 8)
    assert grid.dtype == np.int8
    assert int(grid[0, 0]) == 4


def test_a_list_of_lists_is_accepted() -> None:
    """numpy reads it as an array of whole numbers, which is the contract."""
    grid = _render_grid(_ns(_returns([[1] * 8 for _ in range(8)])), {}, 8, 8)
    assert np.array_equal(grid, np.ones((8, 8), dtype=np.int8))


def test_a_float_array_of_whole_numbers_is_accepted() -> None:
    """`np.zeros((64, 64))` is float64 and is what a model naturally writes,
    so refusing it would fail a model that is behaving."""
    grid = _render_grid(_ns(_returns(np.zeros((8, 8)) + 4.0)), {}, 8, 8)
    assert np.array_equal(grid, np.full((8, 8), 4, dtype=np.int8))


def test_the_wrong_shape_names_the_shape_it_got() -> None:
    with pytest.raises(_RenderContractError) as caught:
        _render_grid(_ns(_returns(np.zeros((4, 8), dtype=np.int8))), {}, 8, 8)
    assert "8x8" in str(caught.value)
    assert "(4, 8)" in str(caught.value)


def test_something_numpy_cannot_read_names_the_type() -> None:
    with pytest.raises(_RenderContractError) as caught:
        _render_grid(_ns(_returns(object())), {}, 8, 8)
    assert "object" in str(caught.value)


def test_every_colour_and_the_wildcard_come_back_unchanged() -> None:
    """The colour check refuses nothing a frame or a wildcard can hold: the
    bounds, -1 and 15, are inside it."""
    board = np.arange(-1, 63, dtype=np.int8).reshape(8, 8) % 17 - 1
    assert board.min() == -1 and board.max() == 15
    assert np.array_equal(_render_grid(_ns(_returns(board)), {}, 8, 8), board)


@pytest.mark.parametrize("value", [16, 40, -2])
def test_a_value_no_frame_holds_is_a_contract_error_naming_its_first_cell(value: int) -> None:
    """The frame delta would refuse such a board as a baseline and end the
    game; refused here, it is a message the model can act on."""
    board = np.zeros((8, 8), dtype=np.int8)
    board[2, 5] = value
    board[6, 0] = 99
    with pytest.raises(_RenderContractError) as caught:
        _render_grid(_ns(_returns(board)), {}, 8, 8)
    assert str(caught.value) == (
        "render(state) must return colour indices 0-15, or -1 for a cell it does not predict; "
        f"got {value} at row 2, column 5."
    )


def test_colours_are_judged_on_the_board_the_cast_makes() -> None:
    """A wider integer array wraps into int8 (200 -> -56), and the wrapped
    board is the one every caller keeps, so that is the one checked."""
    board = np.zeros((8, 8), dtype=np.int64)
    board[0, 1] = 200
    with pytest.raises(_RenderContractError, match=r"got -56 at row 0, column 1\.$"):
        _render_grid(_ns(_returns(board)), {}, 8, 8)


def test_the_models_own_exception_is_not_dressed_as_a_contract_error() -> None:
    """A bug inside ``render`` is the model's to debug, with its traceback."""

    def render(_state: dict[str, Any]) -> Any:
        raise ZeroDivisionError("division by zero")

    with pytest.raises(ZeroDivisionError):
        _render_grid(_ns(render), {}, 8, 8)


# --- the namespace: the whole level, as bare names -----------------------


def _wm_namespace(history: harness_game.History) -> dict[str, Any]:
    return harness_game._build_wm_namespace(["ACTION1"], file="world_model.py", history=history)


def test_module_level_code_reads_the_full_history() -> None:
    """The whole level is readable — the whale's history[0]-only cap is not
    carried: an optional model has no incentive to cheat by copying turns."""
    ns = _wm_namespace(_two_same())
    history = ns["history"]
    assert len(history) == 2
    assert int(history[-1].frame.grid[0, 0]) == 3
    assert [turn.outcome for turn in history] == ["playing", "playing"]


def test_unknown_has_one_name_and_a_home_made_one_declines_too() -> None:
    """``Unknown`` is the one spelling; a world model that defines its own
    class by that name declines with it all the same, rather than crashing."""
    ns = _wm_namespace(_two_same())
    assert "UnknownException" not in ns
    assert issubclass(ns["Unknown"], Exception)

    class Unknown(Exception):
        """A world model's own."""

    assert harness_game._is_decline(ns["Unknown"]("not seen"))
    assert harness_game._is_decline(Unknown("not seen"))
    assert not harness_game._is_decline(ValueError("not seen"))


def test_available_actions_is_a_defensive_copy() -> None:
    actions = ["ACTION1"]
    ns = harness_game._build_wm_namespace(actions, file="world_model.py", history=_two_same())
    ns["available_actions"].append("ACTION2")
    assert actions == ["ACTION1"]


def test_a_model_may_calibrate_from_a_later_turn(tmp_path: Path) -> None:
    path = _write(tmp_path, PASSING_MODEL.replace("history[0]", "history[-1]"))
    assert not _validate(path, _two_same())


# --- the replay's verdicts ----------------------------------------------


def test_a_correct_model_passes_and_says_so(tmp_path: Path) -> None:
    path = _write(tmp_path, PASSING_MODEL)
    result = _validate(path, _two_same())
    assert not result
    assert "ALL PASSED" in repr(result)
    assert result.turn_states is not None and len(result.turn_states) == 2
    assert result.turn_states[1] is not None and result.turn_states[1]["colour"] == 3


def test_a_missing_render_is_named(tmp_path: Path) -> None:
    path = _write(tmp_path, "def initial_state():\n    return {}\n\n\ndef do_move(state, move):\n    pass\n")
    result = _validate(path, _two_same())
    assert "render" in str(result)


def test_a_wrong_initial_frame_counts_its_pixels(tmp_path: Path) -> None:
    path = _write(tmp_path, PASSING_MODEL.replace("state['colour']", "state['colour'] + 1"))
    result = _validate(path, _two_same())
    assert "does not match initial frame" in str(result)
    assert "64 pixels differ" in str(result)


def test_a_render_that_raises_is_reported_against_render(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "def initial_state():\n    return {'outcome': 'playing', 'energy': 1.0}\n"
        "def do_move(state, move):\n    pass\n"
        "def render(state):\n    raise ZeroDivisionError('boom')\n",
    )
    result = _validate(path, _two_same())
    assert "render() raised" in str(result), str(result)


def test_a_missed_win_names_the_outcomes(tmp_path: Path) -> None:
    frame = np.full((8, 8), 3, dtype=np.int8)
    history = _history(frame, np.full((8, 8), 7, dtype=np.int8), outcomes=("playing", "won"))
    path = _write(tmp_path, PASSING_MODEL)
    result = _validate(path, history)
    assert "predicted outcome='playing', expected 'won'" in str(result)
    assert "don't try to make render reproduce that frame" in str(result)


def test_unknown_on_an_observed_move_fails(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        PASSING_MODEL.replace(
            "def do_move(state, move):\n    pass", "def do_move(state, move):\n    raise Unknown('never saw this')"
        ),
    )
    result = _validate(path, _two_same())
    assert "IS\nin history".replace("\n", " ") in str(result).replace("\n", " ")
    assert result.turn_index == 1


# --- the loosenings, and their notes -------------------------------------


ENERGY_MODEL = """
import numpy as np

BASE = 3


def initial_state():
    return {'outcome': 'playing', 'energy': 5}


def do_move(state, move):
    pass


def render(state):
    grid = np.full((8, 8), BASE, dtype=np.int8)
    grid[0, 0] = int(state['energy'])
    return grid
"""


def _energy_history() -> harness_game.History:
    """The energy cell drains from 5 to 4; the model never decrements."""
    first = np.full((8, 8), 3, dtype=np.int8)
    first[0, 0] = 5
    second = first.copy()
    second[0, 0] = 4
    return _history(first, second)


def test_an_all_wildcard_model_passes_loosely(tmp_path: Path) -> None:
    """The model may decline to render — mandatory mode is satisfiable by an
    all-wildcard model, which is allowed and labelled."""
    path = _write(tmp_path, PASSING_MODEL.replace("state['colour']", "-1"))
    result = _validate(path, _two_same())
    assert not result
    assert repr(result).endswith(
        "Passed loosely on:\n  wildcards: render declined cells with -1 on 2 turns (up to 64 per turn, first history[0])."
    )


def test_a_long_loose_pass_is_one_line_per_kind(tmp_path: Path) -> None:
    """A drifting energy and a declined region on every frame of a long level
    summarise to two lines, not two per turn."""
    frames: list[np.ndarray] = []
    for energy in range(15, 0, -1):
        frame = np.full((8, 8), 3, dtype=np.int8)
        frame[0, 0] = energy
        frames.append(frame)
    model = ENERGY_MODEL.replace("'energy': 5", "'energy': 15").replace(
        "    grid[0, 0] = int(state['energy'])\n", "    grid[0, 0] = int(state['energy'])\n    grid[7, 7] = -1\n"
    )
    result = _validate(_write(tmp_path, model), _history(*frames))
    assert not result, str(result)
    loose = repr(result).split("Passed loosely on:\n", 1)[1].splitlines()
    assert len(loose) == 2, loose
    assert loose[0].startswith(
        "  energy: render matched only under another state['energy'] on 14 turns (first history[1]: 14, yours was 15)"
    )
    assert loose[1] == "  wildcards: render declined cells with -1 on 15 turns (up to 1 per turn, first history[0])."


def test_energy_is_held_flexibly_by_default(tmp_path: Path) -> None:
    path = _write(tmp_path, ENERGY_MODEL)
    result = _validate(path, _energy_history())
    assert not result, str(result)
    assert "(first history[1]: 4, yours was 5)" in repr(result)


def test_strict_energy_holds_the_frame_to_the_models_value(tmp_path: Path) -> None:
    path = _write(tmp_path, ENERGY_MODEL)
    result = _validate(path, _energy_history(), rules=harness_test.MatchRules(strict_energy=True))
    assert result
    assert "Energy fix" in str(result)
    # -1 is in the list too: at energy -1 this model renders the wildcard
    # cell, which always matches — the loosenings compose.
    assert "were any of [-1, 4] instead of 5" in str(result)


def _sliced_bar_history() -> harness_game.History:
    """A bar five cells long drains to four; the model never decrements."""
    first = np.full((8, 8), 3, dtype=np.int8)
    first[1, :5] = 9
    second = np.full((8, 8), 3, dtype=np.int8)
    second[1, :4] = 9
    return _history(first, second)


def test_an_energy_used_as_a_slice_bound_is_held_flexibly_too(tmp_path: Path) -> None:
    """A bar drawn by slicing with the energy itself — a float there raises —
    passes one unit off as any other render does, and names the value under
    strict energy."""
    path = _write(tmp_path, ENERGY_MODEL.replace("grid[0, 0] = int(state['energy'])", "grid[1, :state['energy']] = 9"))

    result = _validate(path, _sliced_bar_history())
    assert not result, str(result)
    assert "(first history[1]: 4, yours was 5)" in repr(result)

    strict = _validate(path, _sliced_bar_history(), rules=harness_test.MatchRules(strict_energy=True))
    assert strict
    # -4 is in the list too: a negative slice bound counts from the end, and
    # on this 8-wide board it draws the same four cells.
    assert "were any of [-4, 4] instead of 5" in str(strict)


# --- the guards around the model's module --------------------------------


def test_a_broken_distance_to_win_is_named(tmp_path: Path) -> None:
    path = _write(tmp_path, PASSING_MODEL + "\n\ndef distance_to_win(state):\n    return 'far'\n")
    result = _validate(path, _two_same())
    assert "distance_to_win" in str(result)
    assert "not numeric" in str(result)


def test_a_leftover_print_is_the_last_thing_flagged(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        PASSING_MODEL.replace("def do_move(state, move):\n    pass", "def do_move(state, move):\n    print(move)"),
    )
    result = _validate(path, _two_same())
    assert "must not print" in str(result)
    assert "line" in str(result)


def test_module_level_data_is_frozen_against_do_move(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        PASSING_MODEL.replace("start = ", "CACHE = {}\nstart = ").replace(
            "def do_move(state, move):\n    pass",
            "def do_move(state, move):\n    CACHE['x'] = 1",
        ),
    )
    result = _validate(path, _two_same())
    assert "read-only after load" in str(result)


#: A model that counts its own calls in a file and loses on a count: su15 in
#: the first A/B did this, through an alias of ``open``.
COUNTING_MODEL = """\
import numpy as np

_op = open


def initial_state():
    return {'outcome': 'playing', 'energy': 1.0}


def do_move(state, move):
    try:
        with _op('calls.txt') as f:
            n = int(f.read())
    except OSError:
        n = 0
    with _op('calls.txt', 'w') as f:
        f.write(str(n + 1))


def render(state):
    return np.full((8, 8), 3, dtype=np.int8)
"""


def test_a_model_that_writes_files_fails_the_test(tmp_path: Path) -> None:
    """A replay that let the model write would judge a different model each
    time it ran; the write fails the test, however the code reached ``open``."""
    path = tmp_path / "world_model.py"
    path.write_text(COUNTING_MODEL)
    result = _validate(path, _two_same())
    assert result, "a model that writes files passed"
    assert "world_model.py wrote to your working directory while it was tested (calls.txt)" in repr(result)


def _flecked(cells: int) -> harness_game.History:
    """Two frames of colour 3, the second with ``cells`` cells of colour 9 on
    its first row: what a model that renders a plain board misses."""
    first = np.full((8, 8), 3, dtype=np.int8)
    second = first.copy()
    second[0, :cells] = 9
    return _history(first, second)


def _tolerant(n: int, *, offered: bool = True, strict: bool = False) -> harness_test.MatchRules:
    return harness_test.MatchRules(strict_energy=strict, pixel_tolerance=n, tolerance_offered=offered)


def test_a_render_within_the_pixel_tolerance_passes_and_says_so(tmp_path: Path) -> None:
    path = _write(tmp_path, PASSING_MODEL)
    result = _validate(path, _flecked(2), rules=_tolerant(2))
    assert not result, str(result)
    assert repr(result).endswith(
        "pixels: render missed up to 2 pixels on 1 turn (first history[1]), within your tolerance of 2; "
        "set_pixel_tolerance(0) requires every pixel."
    )
    assert result._pixels_tolerated == 2  # pyright: ignore[reportPrivateUsage]


def test_a_small_miss_past_the_tolerance_suggests_it_only_where_offered(tmp_path: Path) -> None:
    path = _write(tmp_path, PASSING_MODEL)
    suggestion = "set_pixel_tolerance(3) lets such a frame pass"
    assert suggestion in str(_validate(path, _flecked(3), rules=_tolerant(0)))
    assert suggestion in str(_validate(path, _flecked(3), rules=_tolerant(2)))
    assert "set_pixel_tolerance" not in str(_validate(path, _flecked(3), rules=_tolerant(0, offered=False)))
    # Past the suggestion's reach, a miss is not called a detail.
    big = harness_test.SUGGEST_TOLERANCE_PIXELS + 1
    assert "set_pixel_tolerance" not in str(_validate(path, _flecked(big), rules=_tolerant(0)))


def test_energy_and_pixels_pass_together(tmp_path: Path) -> None:
    """The energy cell drained, and one more cell the model does not render:
    an energy match within the tolerance passes, and both are said."""
    first = np.full((8, 8), 3, dtype=np.int8)
    first[0, 0] = 5
    second = first.copy()
    second[0, 0] = 4
    second[7, 7] = 9
    path = _write(tmp_path, ENERGY_MODEL)
    history = _history(first, second)

    assert _validate(path, history), "no tolerance: the extra cell fails it"
    result = _validate(path, history, rules=_tolerant(1))
    assert not result, str(result)
    assert "energy: render matched only under another state['energy']" in repr(result)
    assert "pixels: render missed up to 1 pixel on 1 turn" in repr(result)
    strict = _validate(path, history, rules=_tolerant(1, strict=True))
    assert strict and "Energy fix" in str(strict), "strict energy holds the energy cell whatever the tolerance"
