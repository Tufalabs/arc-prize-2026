"""``seal.frame_delta``: which baseline a frame is explained from, when the block is shown, and what the model reads.

The kernel-side wiring (the hook after a call, the test result's text) is
exercised by the scripted reference ``seal_scripted_frame_delta``; these
tests pin the choice and the wording in process.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import framediff.decompose as decompose_module
from framediff import GRID_SIZE, WILDCARD, BlockCopy, Decomposer, Grid, Source, Transform, block_namespace
from seal import frame_delta as fd


def _blank(color: int = 0) -> Grid:
    """A frame of one colour."""
    return np.full((GRID_SIZE, GRID_SIZE), color, dtype=np.int8)


def _moved_sprite() -> tuple[Grid, Grid]:
    """A 3x3 sprite before and after a move of four columns."""
    before = _blank()
    before[10:13, 10:13] = 9
    after = _blank()
    after[10:13, 14:17] = 9
    return before, after


#: What the sprite's move costs: a copy, priced by the four columns it travels.
MOVE = BlockCopy.COST + 4 * Decomposer().move_distance_cost


def _describe(target: Grid, baselines: dict[str, Source], *, understood: bool = False) -> fd.Delta:
    """One decomposition, the block run against the names the baselines use."""
    namespace = block_namespace(cur=target, **{source.name: np.asarray(source.frame) for source in baselines.values()})
    return fd.describe(
        target,
        "cur",
        baselines,
        understood=understood,
        decomposer=Decomposer(),
        max_bits_understood=fd.MAX_BITS_UNDERSTOOD,
        namespace=namespace,
        event="frame",
        reason="unpredicted",
        turn_index=1,
        subject="The current frame",
    )


def test_the_prediction_wins_when_it_is_close() -> None:
    """A render one cell off explains the frame in one line; the handicaps are not even needed."""
    before, after = _moved_sprite()
    predicted = after.copy()
    predicted[11, 15] = 4
    delta = _describe(
        after,
        {
            "prediction": fd.baseline("prediction", "pred", predicted),
            "previous": fd.baseline("previous", "prev", before),
        },
    )
    assert delta.chosen == "prediction" and delta.shown and delta.cost == 4
    assert delta.text.startswith(
        "The current frame can be most easily understood as a delta from your mispredicted frame"
    )
    assert "```python\ng = pred.copy()\ng[11, 15] = 9\nassert (g == cur).all()\n```" in delta.text
    assert [(entry["cost"], entry["handicap"]) for entry in delta.baselines] == [
        (4, 0),
        (MOVE, fd.HANDICAPS["previous"]),
    ]


def test_the_handicap_decides_a_near_tie() -> None:
    """The previous frame's move beats the level start at 6 once each carries its handicap; the record keeps both
    costs."""
    before, after = _moved_sprite()
    level_start = after.copy()
    level_start[20:23, 20:23] = 3
    delta = _describe(
        after,
        {
            "previous": fd.baseline("previous", "prev", before),
            "level_start": fd.baseline("level_start", "start", level_start),
        },
    )
    assert delta.chosen == "previous" and delta.cost == MOVE
    assert delta.baselines[1]["cost"] == 6 and delta.baselines[1]["handicap"] == fd.HANDICAPS["level_start"]


def test_the_limit_drops_once_the_level_is_understood() -> None:
    """A block between the two limits is shown before a world model passes on the level and skipped after."""
    before = _blank()
    after = _blank(8)
    after[27:36, 27:36] = np.random.default_rng(1).integers(0, 16, (9, 9))
    baselines = {"previous": fd.baseline("previous", "prev", before)}
    shown = _describe(after, baselines, understood=False)
    skipped = _describe(after, baselines, understood=True)
    assert shown.cost is not None and fd.MAX_BITS_UNDERSTOOD < shown.cost <= Decomposer().max_bits
    assert shown.shown and shown.limit == Decomposer().max_bits
    assert not skipped.shown and skipped.limit == fd.MAX_BITS_UNDERSTOOD
    assert skipped.text == (
        "The current frame has no concise description as a delta from the previous frame "
        "(this is based on visual decomposition, without understanding the game)."
    )
    assert skipped.code == shown.code, "the record keeps the block that would have been shown"
    assert (shown.understood, skipped.understood) == (False, True) and shown.reason == "unpredicted"


def test_the_skipped_wording_names_every_baseline() -> None:
    """With three baselines the note lists all three."""
    before, after = _moved_sprite()
    scrambled = np.random.default_rng(2).integers(0, 16, (GRID_SIZE, GRID_SIZE)).astype(np.int8)
    delta = _describe(
        scrambled,
        {
            "prediction": fd.baseline("prediction", "pred", after),
            "previous": fd.baseline("previous", "prev", before),
            "level_start": fd.baseline("level_start", "start", before),
        },
    )
    assert not delta.shown
    assert "from your mispredicted frame, the previous frame or the level start frame" in delta.text


def test_a_wildcard_prediction_is_resolved_in_the_block() -> None:
    """A render that declined the bar gets the resolving line, and the block still runs."""
    _, after = _moved_sprite()
    after[63, :20] = 11
    predicted = after.copy()
    predicted[63, :] = WILDCARD
    predicted[11, 15] = 4
    delta = _describe(after, {"prediction": fd.baseline("prediction", "pred", predicted)})
    assert delta.shown and delta.code is not None
    assert "g[pred == -1] = cur[pred == -1]  # resolve wildcards" in delta.code


def test_a_block_that_does_not_reach_the_frame_ends_the_game() -> None:
    """The block is run before it is shown; a namespace whose names hold other frames fails the assert."""
    before, after = _moved_sprite()
    with pytest.raises(fd.FrameDeltaError, match="did not run: AssertionError"):
        fd.describe(
            after,
            "cur",
            {"previous": fd.baseline("previous", "prev", before)},
            understood=False,
            decomposer=Decomposer(),
            max_bits_understood=fd.MAX_BITS_UNDERSTOOD,
            namespace={"np": np, "prev": before, "cur": before},
            event="frame",
            reason="unpredicted",
            turn_index=1,
            subject="The current frame",
        )


def test_out_of_time_a_frame_reads_as_undescribed_or_ends_the_game() -> None:
    """A search still running at its deadline chooses nothing: the frame reads as one with no concise
    description, and with ``crash_on_timeout`` the game ends instead."""
    before, after = _moved_sprite()
    baselines = {"previous": fd.baseline("previous", "prev", before)}

    def describe(deadline: float | None, crash: bool = False) -> fd.Delta:
        """The move's frame from the previous one, under ``deadline``."""
        return fd.describe(
            after,
            "cur",
            baselines,
            understood=False,
            decomposer=Decomposer(),
            max_bits_understood=fd.MAX_BITS_UNDERSTOOD,
            namespace=block_namespace(cur=after, prev=before),
            event="frame",
            reason="unpredicted",
            turn_index=1,
            subject="The current frame",
            deadline=deadline,
            crash_on_timeout=crash,
        )

    late = describe(time.monotonic())
    assert late.timed_out and not late.shown and late.code is None and late.chosen is None
    assert late.text == fd.skipped_text("The current frame", ["previous"])
    in_time = describe(None)
    assert not in_time.timed_out and in_time.shown and in_time.duration_s >= 0
    with pytest.raises(fd.FrameDeltaError, match="framediff ran out of time on cur"):
        describe(time.monotonic(), crash=True)


def test_out_of_time_the_best_block_found_reads_as_any_other(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search the clock passes once a baseline's script is back keeps that script: the model reads the block
    as it would any other, and only the record says the search timed out."""
    before, after = _moved_sprite()
    baselines = {"previous": fd.baseline("previous", "prev", before)}
    whole = _describe(after, baselines)
    now = [0.0]
    monkeypatch.setattr(decompose_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    search = Decomposer.decompose_within

    def then_late(self: Decomposer, a: Grid, b: Grid, give_up: Callable[[float], bool]) -> Transform | None:
        found = search(self, a, b, give_up)
        now[0] = 1e9
        return found

    monkeypatch.setattr(Decomposer, "decompose_within", then_late)
    late = _describe(after, baselines)
    assert late.timed_out and late.shown and not whole.timed_out
    assert late.text == whole.text and late.code == whole.code and late.cost == whole.cost


def test_an_animation_record_says_which_runs_were_written_as_loops() -> None:
    """With animation patterns on, a bar growing a row a frame is one loop, and the record counts the frames each
    kind of run covers, a scalar field per kind as a span takes it; off, the record carries no runs at all."""
    frames = []
    for rows in range(1, 9):
        frame = _blank()
        frame[40 : 40 + rows, 10:20] = 7
        frames.append(SimpleNamespace(grid=frame))
    history = [
        SimpleNamespace(frame=SimpleNamespace(grid=_blank()), animation_frames=[]),
        SimpleNamespace(frame=SimpleNamespace(grid=frames[-1].grid), animation_frames=frames),
    ]

    def describe(decomposer: Decomposer) -> dict[str, Any]:
        """The move's animation, described with ``decomposer``."""
        _, record = fd.describe_animation(
            history,
            1,
            None,
            decomposer=decomposer,
            max_chars=5000,
            cut_chars=10000,
            namespace={"np": np, "history": history},
            reason="requested",
        )
        return record

    on = describe(Decomposer())
    assert on["runs_stepped"] == 7 and on["built"] == 8
    assert not any(key.startswith("runs") for key in describe(Decomposer(animation_patterns=False)))
    assert all(isinstance(value, (str, int, float, bool, type(None))) for value in on.values()), "a span's values"


def test_out_of_time_an_animation_reads_as_undescribed_or_ends_the_game() -> None:
    """The same for an animation: every frame taken as it is, so nothing concise to show."""
    before, after = _moved_sprite()
    middle = _blank()
    middle[10:13, 12:15] = 9
    history = [
        SimpleNamespace(frame=SimpleNamespace(grid=before), animation_frames=[]),
        SimpleNamespace(frame=SimpleNamespace(grid=after), animation_frames=[SimpleNamespace(grid=middle)]),
    ]

    def describe(deadline: float | None, crash: bool = False) -> tuple[str, dict[str, Any]]:
        """The move's animation, under ``deadline``."""
        return fd.describe_animation(
            history,
            1,
            None,
            decomposer=Decomposer(),
            max_chars=5000,
            cut_chars=10000,
            namespace={"np": np, "history": history},
            reason="requested",
            deadline=deadline,
            crash_on_timeout=crash,
        )

    text, record = describe(time.monotonic())
    assert record["timed_out"] and not record["shown"]
    assert text.startswith("The animation of history[1] (1 frames) has no concise description")
    _, in_time = describe(None)
    assert not in_time["timed_out"] and in_time["shown"]
    with pytest.raises(fd.FrameDeltaError, match="framediff ran out of time on history\\[1\\]'s animation"):
        describe(time.monotonic(), crash=True)


def test_an_unchanged_frame_is_a_no_op() -> None:
    """A move that changed nothing is the comparison alone, at no cost."""
    before, _ = _moved_sprite()
    delta = _describe(before.copy(), {"previous": fd.baseline("previous", "prev", before)})
    assert delta.shown and delta.cost == 0 and delta.code == "assert (prev == cur).all()"


def test_matches_ignores_wildcards() -> None:
    """A render matches on the cells it predicts."""
    before, after = _moved_sprite()
    predicted = after.copy()
    predicted[63, :] = WILDCARD
    after[63, :20] = 11
    assert fd.matches(predicted, after)
    assert not fd.matches(before, after)
