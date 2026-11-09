"""``platypus.multi_take.TieredTakeAgent``: up to ten takes, queued by how far
they agree.

What is claimed: a game's category follows its ended takes — its first and
second take, then no grid 1 that two takes produced first, then none that
three did unless the first two agreed on it, then a category a take — reading
only the grid a take produced first, on every test output. The vote ranks
grids by how many takes produced them first, then second, puts the runner-up
in attempt 2, fills as the two-take rule does, counts a take once per grid,
and breaks remaining ties by the producing takes' confidence (a take without
one below any), then take 1's produced grids in order, then a stable hash;
the runner-up takes attempt 2 whether or not takes 1 and 2 agree. Reordering
later takes does not change the vote. Each take's confidence is logged as it
ends and reaches the vote. Driven through ten
takes: every take boundary carries its category as a handicap the ranking
serves before any spend; takes 1 and 2 count from their first answer and a
later take only once it ends; fewer than three counted answers compose as two
takes do and are named for their own takes; and the take log says where the
game queues next. The ten-take agent refuses any other number of takes, as the
two-take one still does.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import ullm.backends.stub as ullm_stub

import taaf.game
from platypus import GameOutcome, GiveUp, LLMAgent, LLMMoveContext, RunConfig
from platypus.context import Produced
from platypus.contract import HANDICAP_HINT, TAKE_BOUNDARY_HINT
from platypus.llm import build_llm_context
from platypus.multi_take import (
    CATEGORY_HANDICAP_TOKENS,
    N_TIERED_TAKES,
    MultiTakeAgent,
    TakeAnswer,
    TieredTakeAgent,
    category,
    vote,
)
from platypus.solver import _rank_key
from taaf.game_examples import flip_task
from taaf.grid import Grid


def _grid(value: int) -> Grid:
    return Grid(np.full((2, 2), value, dtype=np.int8))


#: One question's two slots as a take submitted them: a value is a grid the
#: take produced, None a filler (the grid then holds 0).
Slots = tuple[int | None, int | None]


def _answer(*questions: Slots) -> TakeAnswer:
    attempts = tuple(tuple(_grid(0 if value is None else value) for value in slots) for slots in questions)
    produced = tuple(tuple(value is not None for value in slots) for slots in questions)
    return TakeAnswer(submission=taaf.game.Submission(attempts=attempts), produced=produced)


# --------------------------------------------------------------------------
# the category
# --------------------------------------------------------------------------

# the ended takes' slots on one question (None: ended without an answer), and
# where the game queues for its next take.
CATEGORIES: dict[str, tuple[list[Slots | None], int]] = {
    "no take has ended": ([], 0),
    "one take has ended": ([(1, None)], 1),
    "two takes disagree": ([(1, None), (2, None)], 2),
    "agreeing second grids are no repeat": ([(1, 3), (2, 3)], 2),
    "a take that ended without an answer votes for nothing": ([(1, None), None], 2),
    "the first two agreeing count as three": ([(1, None), (1, None)], 4),
    "a repeat from a later take still needs a third vote": ([(1, None), (2, None), (1, None)], 3),
    "three takes producing it first": ([(1, None), (2, None), (1, None), (1, None)], 6),
    "past both, a category a take": ([(1, None), (1, None), (5, None), (6, None), (7, None)], 7),
}


@pytest.mark.parametrize(("ended", "expected"), CATEGORIES.values(), ids=list(CATEGORIES))
def test_the_category_follows_the_ended_takes(ended: list[Slots | None], expected: int) -> None:
    answers = [None if slots is None else _answer(slots) for slots in ended]
    assert category(answers, 1) == expected


def test_every_test_output_must_settle_before_a_game_moves_on() -> None:
    agreed_then_split = [_answer((1, None), (1, None)), _answer((1, None), (2, None))]
    assert category(agreed_then_split, 2) == 2
    agreed_on_both = [_answer((1, None), (3, None)), _answer((1, None), (3, None))]
    assert category(agreed_on_both, 2) == 4


# --------------------------------------------------------------------------
# the vote
# --------------------------------------------------------------------------


def _voted(*answers: TakeAnswer, questions: int = 1, confidences: dict[int, int] | None = None) -> taaf.game.Submission:
    composition = vote([_grid(9)] * questions, list(enumerate(answers)), confidences)
    assert composition is not None
    return composition.submission


def test_the_most_takes_producing_a_grid_first_decide_attempt_1_and_the_runner_up_attempt_2() -> None:
    voted = _voted(_answer((1, 2)), _answer((2, 1)), _answer((2, 3)))
    assert voted.attempts[0] == (_grid(2), _grid(1))


def test_second_productions_break_a_tie_in_first_ones() -> None:
    voted = _voted(_answer((1, 3)), _answer((2, 3)), _answer((3, None)), _answer((4, 1)))
    assert voted.attempts[0] == (_grid(3), _grid(1)), "every grid is some take's first; 3 is two more takes' second"


def test_a_grid_only_ever_produced_second_can_be_attempt_2() -> None:
    voted = _voted(_answer((1, 5)), _answer((1, 5)), _answer((1, None)))
    assert voted.attempts[0] == (_grid(1), _grid(5))


def test_a_take_votes_once_for_a_grid_in_both_its_slots() -> None:
    voted = _voted(_answer((1, 1)), _answer((2, None)), _answer((2, None)))
    assert voted.attempts[0] == (_grid(2), _grid(1))


def test_a_lone_candidate_is_copied_and_a_question_without_one_gets_its_input() -> None:
    answers = [_answer((1, None), (None, None)) for _ in range(3)]
    composition = vote([_grid(8), _grid(9)], list(enumerate(answers)))
    assert composition is not None
    assert composition.submission.attempts == ((_grid(1), _grid(1)), (_grid(9), _grid(9)))
    assert composition.produced == ((True, False), (False, False))
    assert composition.sources[1] == ("placeholder", "copy of attempt 1")
    assert vote([_grid(9)], list(enumerate([_answer((None, None))] * 3))) is None


def test_tied_votes_prefer_take_ones_grids_per_question_regardless_of_later_take_order() -> None:
    first = _answer((1, 4), (4, 1))
    later = [_answer((2, 3), (3, 2)), _answer((3, 2), (2, 3)), _answer((4, 1), (1, 4))]
    for order in itertools.permutations(later):
        assert _voted(first, *order, questions=2).attempts == ((_grid(1), _grid(4)), (_grid(4), _grid(1)))


def test_take_one_is_identified_by_its_index_not_its_position_in_the_counted_answers() -> None:
    answers = list(enumerate([_answer((1, 4)), _answer((2, 3)), _answer((3, 2)), _answer((4, 1))]))
    for order in itertools.permutations(answers):
        composition = vote([_grid(9)], order)
        assert composition is not None
        assert composition.submission.attempts == ((_grid(1), _grid(4)),)


def test_take_ones_only_production_in_slot_two_still_breaks_a_tie() -> None:
    voted = _voted(_answer((None, 1)), _answer((2, None)), _answer((3, None)))
    assert voted.attempts == ((_grid(1), _grid(2)),)


def test_confidence_breaks_a_tie_ahead_of_take_ones_grids() -> None:
    answers = (_answer((1, None)), _answer((2, None)), _answer((3, None)))
    assert _voted(*answers).attempts[0][0] == _grid(1), "unconfident, take 1's grid leads the tie"
    voted = _voted(*answers, confidences={0: 50, 1: 70, 2: 90})
    assert voted.attempts[0] == (_grid(3), _grid(2))


def test_a_take_that_stated_no_confidence_ranks_below_one_that_did() -> None:
    voted = _voted(_answer((1, None)), _answer((2, None)), _answer((3, None)), confidences={2: 10})
    assert voted.attempts[0] == (_grid(3), _grid(1))


def test_confidence_only_breaks_ties_in_the_counts() -> None:
    voted = _voted(_answer((1, None)), _answer((2, None)), _answer((1, None)), confidences={1: 99})
    assert voted.attempts[0] == (_grid(1), _grid(2)), "two takes' first votes outrank one confident take"


def test_the_runner_up_takes_attempt_2_where_the_first_two_takes_agree_too() -> None:
    voted = _voted(_answer((1, 5)), _answer((1, 6)), _answer((1, 6)), confidences={0: 99, 1: 10, 2: 10})
    assert voted.attempts[0] == (_grid(1), _grid(6)), "two takes' second votes outrank take 1's own second grid"
    voted = _voted(_answer((1, 5)), _answer((1, None)), _answer((6, None)))
    assert voted.attempts[0] == (_grid(1), _grid(6)), "a later take's first vote outranks a second one"


@pytest.mark.parametrize("first_answer", [None, _answer((None, None))])
def test_without_a_take_one_production_the_hash_orders_ties(first_answer: TakeAnswer | None) -> None:
    later = [(1, _answer((0, None))), (2, _answer((8, None))), (3, _answer((9, None)))]
    first = [] if first_answer is None else [(0, first_answer)]
    for order in itertools.permutations(later):
        composition = vote([_grid(9)], [*first, *order])
        assert composition is not None
        # 0 is also take 1's placeholder; it must not outrank either production.
        assert composition.submission.attempts == ((_grid(8), _grid(9)),)


# --------------------------------------------------------------------------
# ten takes, driven
# --------------------------------------------------------------------------


@dataclass(kw_only=True)
class _Answering(LLMAgent):
    """A take that answers every question with each of ``grids`` in turn, as
    its grid 1 alone."""

    grids: tuple[int, ...] = ()
    confidence: int | None = None

    def answer_confidence(self) -> int | None:
        return self.confidence

    async def _do_move(self, ctx: LLMMoveContext, view: taaf.game.GameView) -> GiveUp | None:
        for value in self.grids:
            grid = Grid(np.full(view.questions[0].shape, value, dtype=np.int8))
            produced: Produced = tuple((True, False) for _ in view.questions)
            ctx.submit(taaf.game.Submission(attempts=tuple((grid, grid) for _ in view.questions)), produced=produced)
        return None


async def _drive(
    *takes: tuple[int, ...], confidences: tuple[int | None, ...] = ()
) -> tuple[list[taaf.game.Submission], list[dict[str, Any]]]:
    """Play ten takes answering ``takes``, each stating its ``confidences``
    (none past them), and return what the game was sent and every slot offer
    it made."""
    view = flip_task("tiered").view()
    submissions: list[taaf.game.Submission] = []
    offers: list[dict[str, Any]] = []

    def submit(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        del produced
        submissions.append(submission)

    async def offer(hints: Any) -> None:
        offers.append(dict(hints))

    backend = ullm_stub.StubBackend.from_script("8", on_exhausted="loop")
    ctx, _meter = build_llm_context("tiered", submissions, random.Random(0), backend, submit, yield_slot=offer)
    stated = [*confidences, *[None] * (len(takes) - len(confidences))]
    agent = TieredTakeAgent(
        takes=tuple(_Answering(grids=grids, confidence=c) for grids, c in zip(takes, stated, strict=True))
    )
    decision = await agent.do_move(ctx, view)
    assert decision is None
    await agent.on_finish(ctx, GameOutcome(state="finished", n_submitted=len(submissions)))
    return submissions, offers


def _first_grids(submissions: list[taaf.game.Submission]) -> list[tuple[int, int]]:
    """Each submission's two attempts on the first question, as their values."""
    return [(int(s.attempts[0][0].data.flat[0]), int(s.attempts[0][1].data.flat[0])) for s in submissions]


#: Takes 1 and 2 disagree; take 3 first answers 7, then sides with take 2;
#: take 4 makes three votes; the rest answer something new.
TAKES: tuple[tuple[int, ...], ...] = ((1,), (2,), (7, 2), (2,), *[(3,)] * 6)


async def test_every_take_boundary_carries_its_category_ahead_of_any_spend() -> None:
    _, offers = await _drive(*TAKES)
    boundaries = [offer for offer in offers if offer.get(TAKE_BOUNDARY_HINT)]
    assert [offer["take_index"] for offer in boundaries] == list(range(1, N_TIERED_TAKES))
    categories = [offer[HANDICAP_HINT] // CATEGORY_HANDICAP_TOKENS for offer in boundaries]
    assert categories == [1, 2, 3, 6, 7, 8, 9, 10, 11]

    key = _rank_key(RunConfig(slot_policy="least_tokens"), [])
    hints = {"0": boundaries[0], "1": {"tokens": 50_000_000}}  # in its second take; deep in its first
    assert sorted(hints, key=lambda game: key(game, hints[game])) == ["1", "0"]


async def test_the_first_two_takes_count_from_their_first_answer_and_a_later_one_once_it_ends() -> None:
    submissions, _ = await _drive(*TAKES)
    sent = _first_grids(submissions)
    assert sent[:3] == [(1, 1), (1, 2), (2, 1)], "take 2 counts at once; take 3 only as it ended, siding with take 2"
    assert all(7 not in grids for grids in sent), "take 3's earlier answer never counted"
    assert sent[-1] == (3, 2), "six takes for 3 outvote three for 2"


async def test_fewer_than_three_counted_answers_compose_as_two_takes_named_for_their_own(
    capfd: pytest.CaptureFixture[str],
) -> None:
    await _drive((), (), (4,), *[()] * 7)
    out = capfd.readouterr().out
    final = [line for line in out.splitlines() if line.startswith("[take] tiered pass 0 answer question 1:")][-1]
    assert "attempt 1 = take 3 grid 1" in final and "attempt 2 = copy of attempt 1" in final
    assert "[take] ==== tiered pass 0: take 10 of 10 finished (finished) ====" in out


async def test_the_take_log_says_where_the_game_queues_next(capfd: pytest.CaptureFixture[str]) -> None:
    await _drive(*TAKES)
    out = capfd.readouterr().out
    assert "[take] tiered pass 0 take 3 queues in category 2: no grid 1 two takes produced yet" in out
    assert "[take] tiered pass 0 take 5 queues in category 6: take 5, a category a take" in out


async def test_each_takes_confidence_is_logged_and_breaks_the_votes_ties(capfd: pytest.CaptureFixture[str]) -> None:
    submissions, _ = await _drive((1,), (2,), (3,), *[()] * 7, confidences=(10, 20, 90))
    assert _first_grids(submissions)[2] == (3, 2), "three takes apart: the most confident first, then the next"
    out = capfd.readouterr().out
    assert (
        "[take] tiered pass 0 take 3: 0 tokens generated in the take, 0 in the game; 1 answer(s), confidence 90" in out
    )
    assert (
        "[take] tiered pass 0 take 4: 0 tokens generated in the take, 0 in the game; 0 answer(s), no confidence" in out
    )


def test_the_ten_take_agent_plays_ten_and_the_two_take_one_still_two() -> None:
    with pytest.raises(ValueError, match="plays 10 takes; got 2"):
        TieredTakeAgent(takes=(_Answering(), _Answering()))
    with pytest.raises(ValueError, match="plays 2 takes; got 10"):
        MultiTakeAgent(takes=tuple(_Answering() for _ in range(N_TIERED_TAKES)))
