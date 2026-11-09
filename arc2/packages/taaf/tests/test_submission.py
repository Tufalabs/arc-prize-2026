"""Tests for taaf.submission — the file that IS the Kaggle deliverable, and
the properties it needs because a scoring rerun returns no logs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import taaf.submission
from taaf.game import ArcGame, GameSpec, Pair, Submission
from taaf.grid import Grid


def _spec(game_id: str, n_to_answer: int = 1) -> GameSpec:
    """A task with a known answer, so a filled submission is checkable."""
    pairs = [Pair(question=Grid(np.array([[1]], np.int8)), answer=Grid(np.array([[2]], np.int8)), given=True)]
    pairs += [
        Pair(question=Grid(np.array([[3]], np.int8)), answer=Grid(np.array([[4]], np.int8)), given=False)
        for _ in range(n_to_answer)
    ]
    return GameSpec(game_id=game_id, pairs=tuple(pairs))


def _answer(spec: GameSpec, value: int) -> Submission:
    """A submission of constant grids, shaped for ``spec``."""
    g = Grid(np.array([[value]], np.int8))
    return Submission(attempts=tuple((g, g) for _ in spec.to_answer))


def test_every_task_appears_even_unanswered():
    """A missing id scores 0 at best and may invalidate the submission, so
    coverage cannot depend on the solver having got there."""
    specs = [_spec("a"), _spec("b", n_to_answer=2)]
    built = taaf.submission.build(specs, runs=[])
    assert set(built) == {"a", "b"}
    assert len(built["b"]) == 2
    assert built["a"][0]["attempt_1"] == taaf.submission.PLACEHOLDER.to_lists()


def test_answered_runs_replace_their_placeholders():
    """The seeded file is progressively filled, which is what makes a kill at
    any moment leave something scorable."""
    specs = [_spec("a"), _spec("b")]
    game = ArcGame(spec_data=specs[0])
    game.start_game()
    game.execute_action(_answer(specs[0], 4))
    game.finish_game()
    assert game.game_run is not None
    built = taaf.submission.build(specs, [game.game_run])
    assert built["a"][0]["attempt_1"] == [[4]]
    assert built["b"][0]["attempt_1"] == taaf.submission.PLACEHOLDER.to_lists()


def test_the_standing_answer_is_the_one_submitted():
    """A solver that replaces its answer must not have the file keep the one
    it abandoned."""
    spec = _spec("a")
    game = ArcGame(spec_data=spec)
    game.start_game()
    game.execute_action(_answer(spec, 1))
    game.execute_action(_answer(spec, 7))
    game.finish_game()
    assert game.game_run is not None
    built = taaf.submission.build([spec], [game.game_run])
    assert built["a"][0]["attempt_1"] == [[7]]


def test_shape_is_the_competition_shape():
    """One dict per test input, in test order, keyed attempt_1..attempt_N."""
    spec = _spec("a", n_to_answer=2)
    built = taaf.submission.build([spec], runs=[])
    assert list(built["a"][0]) == ["attempt_1", "attempt_2"]
    assert isinstance(built["a"][0]["attempt_1"], list)
    assert isinstance(built["a"][0]["attempt_1"][0], list)


def test_validate_catches_missing_wrong_shape_and_bad_grids():
    """Failing here is recoverable; failing in the scoring rerun is a wasted
    submission we would never see a log for."""
    specs = [_spec("a"), _spec("b")]
    good = taaf.submission.build(specs, runs=[])
    taaf.submission.validate(good, specs)

    missing = {k: v for k, v in good.items() if k != "b"}
    with pytest.raises(ValueError, match="missing 1 task"):
        taaf.submission.validate(missing, specs)

    unknown = {**good, "zz": good["a"]}
    with pytest.raises(ValueError, match="unknown task"):
        taaf.submission.validate(unknown, specs)

    wrong_count = {**good, "a": good["a"] + good["a"]}
    with pytest.raises(ValueError, match="2 answer group"):
        taaf.submission.validate(wrong_count, specs)

    one_attempt = {**good, "a": [{"attempt_1": [[0]]}]}
    with pytest.raises(ValueError, match="expected"):
        taaf.submission.validate(one_attempt, specs)

    bad_grid = {**good, "a": [{"attempt_1": [[99]], "attempt_2": [[0]]}]}
    with pytest.raises(ValueError, match="attempt_1"):
        taaf.submission.validate(bad_grid, specs)


def test_write_is_atomic_and_leaves_no_partial_file(tmp_path: Path):
    """A kill mid-write would leave truncated non-JSON — strictly worse than
    the complete placeholder file it replaced."""
    specs = [_spec("a")]
    path = tmp_path / "submission.json"
    taaf.submission.write(path, taaf.submission.build(specs, runs=[]), specs)
    assert json.loads(path.read_text())
    # The temp must be a sibling: os.replace is only atomic within one
    # filesystem, so a temp elsewhere would silently stop being atomic.
    assert not list(tmp_path.glob("*.tmp"))
    assert taaf.submission.read(path) == taaf.submission.build(specs, runs=[])


def test_write_refuses_an_invalid_submission(tmp_path: Path):
    """Validation runs before the write, so a bad submission never reaches
    disk to be scored."""
    specs = [_spec("a"), _spec("b")]
    path = tmp_path / "submission.json"
    with pytest.raises(ValueError):
        taaf.submission.write(path, {"a": taaf.submission.build(specs, runs=[])["a"]}, specs)
    assert not path.exists()
