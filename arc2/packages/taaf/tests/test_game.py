"""Tests for taaf.game — the task types, the one-move lifecycle, the
GameRun state machine, and scoring."""

from __future__ import annotations

import asyncio
import copy
import pickle

import numpy as np
import pytest

import taaf.game
import taaf.game_examples
from taaf.game import ArcGame, GameSpec, Pair, RunSession, Submission, score_submission
from taaf.grid import Grid


def _g(*rows: list[int]) -> Grid:
    """Shorthand for a small grid."""
    return Grid(np.array(rows, dtype=np.int8))


def _spec(n_to_answer: int = 1) -> GameSpec:
    """A task whose answer is the question doubled, so a wrong submission is
    easy to construct."""
    pairs = [Pair(question=_g([1, 2]), answer=_g([2, 4]), given=True)]
    pairs += [Pair(question=_g([3, 4]), answer=_g([6, 8]), given=False) for _ in range(n_to_answer)]
    return GameSpec(game_id="t", pairs=tuple(pairs))


def test_spec_requires_both_kinds_of_pair():
    """A task with nothing to demonstrate, or nothing to answer, is not a
    task — catching it at construction keeps every consumer simpler."""
    only_given = (Pair(question=_g([1]), answer=_g([1]), given=True),)
    with pytest.raises(ValueError, match="at least one pair to answer"):
        GameSpec(game_id="t", pairs=only_given)
    only_test = (Pair(question=_g([1]), answer=_g([1]), given=False),)
    with pytest.raises(ValueError, match="at least one given pair"):
        GameSpec(game_id="t", pairs=only_test)


def test_given_pair_must_carry_its_answer():
    """A demonstration with no answer demonstrates nothing."""
    with pytest.raises(ValueError, match="given pair must carry"):
        Pair(question=_g([1]), answer=None, given=True)


def test_view_never_exposes_a_to_answer_answer():
    """The whole point of the view: a solver cannot reach the answer key,
    whether or not the ground truth exists."""
    view = _spec(n_to_answer=2).view()
    assert len(view.given) == 1
    assert len(view.questions) == 2
    assert not hasattr(view, "answer")
    # The only grids reachable are the demonstrations and the questions.
    reachable = {g for pair in view.given for g in pair} | set(view.questions)
    assert _g([6, 8]) not in reachable


def test_withhold_answers_keeps_questions_and_demonstrations():
    """simulate_submission's data half: the task still plays, it just cannot
    be scored."""
    withheld = _spec(n_to_answer=2).withhold_answers()
    assert withheld.answers_known is False
    assert all(p.answer is None for p in withheld.to_answer)
    assert all(p.answer is not None for p in withheld.given)
    assert [p.question for p in withheld.to_answer] == [p.question for p in _spec(2).to_answer]


def test_scoring_takes_each_pair_and_either_attempt_independently():
    """The competition's rule: the attempt index carries no meaning across
    pairs, so attempt 1 on one pair and attempt 2 on another both count."""
    spec = _spec(n_to_answer=2)
    right, wrong = _g([6, 8]), _g([0, 0])
    submission = Submission(attempts=((right, wrong), (wrong, right)))
    result = score_submission(spec, submission)
    assert [p.correct for p in result.pairs] == [True, True]
    assert [p.matched_attempt for p in result.pairs] == [0, 1]
    assert result.score == 100.0


def test_partial_score_is_the_fraction_of_pairs():
    """Score is per test output, which is what makes a multi-test task
    scoreable between 0 and 100."""
    spec = _spec(n_to_answer=2)
    right, wrong = _g([6, 8]), _g([0, 0])
    result = score_submission(spec, Submission(attempts=((right, right), (wrong, wrong))))
    assert result.n_correct == 1
    assert result.score == 50.0


def test_unscored_result_is_none_not_false():
    """Withheld ground truth must not read as a wrong answer anywhere."""
    result = score_submission(_spec().withhold_answers(), Submission(attempts=((_g([6, 8]), _g([0, 0])),)))
    assert result.scored is False
    assert result.score is None
    assert [p.correct for p in result.pairs] == [None]


def test_lifecycle_won_when_any_pair_right():
    """number_of_levels is 1 and the level is beaten by a single correct
    pair, so `won` means 'got at least one right' — not 'solved'."""
    game = ArcGame(spec_data=_spec(n_to_answer=2))
    game.start_game()
    right, wrong = _g([6, 8]), _g([0, 0])
    game.execute_action(Submission(attempts=((right, right), (wrong, wrong))))
    game.finish_game()
    run = game.game_run
    assert run is not None
    assert run.state == "won"
    assert run.levels_completed == 1
    assert run.final_score == 50.0


def test_lifecycle_submitted_when_nothing_right():
    """Answered and wrong is `submitted`, never `gave_up`: the solver did
    answer, and give-up statistics have to stay meaningful."""
    game = ArcGame(spec_data=_spec())
    game.start_game()
    game.execute_action(Submission(attempts=((_g([0, 0]), _g([1, 1])),)))
    game.finish_game()
    assert game.game_run is not None
    assert game.game_run.state == "submitted"
    assert game.game_run.final_score == 0.0


def test_lifecycle_submitted_when_unscored():
    """Unscored is `submitted` with score None — distinct from answered-and-
    wrong, which scores 0.0."""
    game = ArcGame(spec_data=_spec())
    game.start_game(RunSession(withhold_answers=True))
    game.execute_action(Submission(attempts=((_g([6, 8]), _g([0, 0])),)))
    game.finish_game()
    assert game.game_run is not None
    assert game.game_run.state == "submitted"
    assert game.game_run.final_score is None


def test_gave_up_when_never_answered_and_tokens_still_counted():
    """A give-up still spent tokens; a total that omitted them would report
    less than the run paid for."""
    game = ArcGame(spec_data=_spec())
    game.start_game()
    game.finish_game(generated_tokens=77, uncached_input_tokens=5)
    run = game.game_run
    assert run is not None
    assert run.state == "gave_up"
    assert run.final_generated_tokens == 77
    assert run.history == []
    assert run.final_score == 0.0


async def test_cancelled_when_a_cancel_is_outstanding():
    """The same no-answer path, distinguished by whether the framework asked
    the solver to stop."""
    game = ArcGame(spec_data=_spec())
    game.start_game()

    async def finish_under_cancel() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            game.finish_game(generated_tokens=3)
            raise

    task = asyncio.create_task(finish_under_cancel())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert game.game_run is not None
    assert game.game_run.state == "cancelled"


def test_finish_game_is_idempotent_after_execute_action():
    """The idempotence guard cannot key on state or on final_score — an
    unscored run finishes with neither."""
    game = ArcGame(spec_data=_spec().withhold_answers())
    game.start_game()
    game.execute_action(Submission(attempts=((_g([6, 8]), _g([0, 0])),)))
    game.finish_game(generated_tokens=10)
    game.finish_game(generated_tokens=999)
    assert game.game_run is not None
    assert game.game_run.final_generated_tokens == 10


def test_a_later_answer_replaces_an_earlier_one():
    """A solver may change its mind; the last answer is the one that counts,
    and every earlier one is kept with what it cost."""
    game = ArcGame(spec_data=_spec())
    game.start_game()
    game.execute_action(Submission(attempts=((_g([0, 0]), _g([0, 0])),)), generated_tokens=10)
    game.execute_action(Submission(attempts=((_g([6, 8]), _g([0, 0])),)), generated_tokens=25)
    game.finish_game()
    run = game.game_run
    assert run is not None
    assert [rec.generated_tokens for rec in run.history] == [10, 25]
    assert run.standing_submission == run.history[-1].submission
    assert run.final_score == 100.0
    assert run.state == "won"


def test_answering_reveals_nothing_until_the_game_finishes():
    """Handing back per-pair correctness for an answer the solver can still
    replace would be a scoring oracle, so execute_action returns nothing and
    the run records no score until it ends."""
    game = ArcGame(spec_data=_spec())
    game.start_game()
    assert game.execute_action(Submission(attempts=((_g([6, 8]), _g([0, 0])),))) is None
    run = game.game_run
    assert run is not None
    assert run.state == "playing"
    assert run.result is None and run.final_score is None


def test_a_run_in_play_reads_the_score_of_the_answer_it_holds():
    """Whoever watches a run sees each game at its standing answer: zero before
    one, and down again when a right answer is replaced by a wrong one."""
    game = ArcGame(spec_data=_spec())
    game.start_game()
    run = game.game_run
    assert run is not None
    assert run.current_score() == 0.0
    game.execute_action(Submission(attempts=((_g([6, 8]), _g([0, 0])),)))
    assert run.current_score() == 100.0
    game.execute_action(Submission(attempts=((_g([0, 0]), _g([0, 0])),)))
    assert run.current_score() == 0.0
    result = run.current_result()
    assert result is not None and result.n_correct == 0


def test_a_run_in_play_with_no_answer_key_reads_as_unscored():
    """A withheld key cannot score an answer mid-play any more than at the
    finish, and reading it as a zero would report a run that got it wrong."""
    game = ArcGame(spec_data=_spec().withhold_answers())
    game.start_game()
    run = game.game_run
    assert run is not None
    game.execute_action(Submission(attempts=((_g([6, 8]), _g([0, 0])),)))
    assert run.current_score() is None


def test_the_finish_scores_a_run_whose_spec_was_not_recorded():
    """minimal_diagnostics records no spec on the run, so a run in play has
    nothing to be read against, but its finish still scores against the game's
    own answer key."""
    game = ArcGame(spec_data=_spec())
    game.start_game(RunSession(record_specs=False))
    run = game.game_run
    assert run is not None
    game.execute_action(Submission(attempts=((_g([6, 8]), _g([0, 0])),)))
    assert run.current_score() is None
    game.finish_game()
    assert run.final_score == 100.0
    assert run.current_score() == 100.0


def test_a_saved_give_up_with_no_result_reads_its_recorded_score():
    """benchmark.json can hold a finished give-up whose result is null; its
    score of record is the zero it finished with, not an unknown."""
    game = ArcGame(spec_data=_spec())
    game.start_game()
    game.finish_game()
    assert game.game_run is not None
    saved = game.game_run.to_json_dict()
    saved["result"] = None
    assert taaf.game.GameRun.from_json_dict(saved).current_score() == 0.0


async def test_a_cancelled_game_is_scored_on_the_answer_that_stood():
    """The reason a solver may answer more than once: a run cut off part-way
    submits its best so far. The state still says it was stopped from
    outside, which nothing else records."""
    game = ArcGame(spec_data=_spec())
    game.start_game()
    game.execute_action(Submission(attempts=((_g([6, 8]), _g([0, 0])),)), generated_tokens=4)

    async def finish_under_cancel() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            game.finish_game()
            raise

    task = asyncio.create_task(finish_under_cancel())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    run = game.game_run
    assert run is not None
    assert run.state == "cancelled"
    assert run.final_score == 100.0
    assert run.levels_completed == 1


def test_execute_action_rejects_a_mis_shaped_submission():
    """Loudly, rather than letting a shape bug read as a wrong answer."""
    game = ArcGame(spec_data=_spec(n_to_answer=2))
    game.start_game()
    with pytest.raises(ValueError, match="covers 1 pairs"):
        game.execute_action(Submission(attempts=((_g([1]), _g([1])),)))


def test_submission_requires_exactly_two_attempts():
    """The competition takes two per output; one or three is a bug we should
    see here, not in a rejected submission file."""
    with pytest.raises(ValueError, match="expected 2 attempts"):
        Submission(attempts=((_g([1]),),))


def test_pickle_and_deepcopy_contract():
    """Templates copy per pass; a started game does not, so a live run cannot
    be duplicated by accident."""
    game = ArcGame(spec_data=_spec())
    assert copy.deepcopy(game).spec_data == game.spec_data
    assert pickle.loads(pickle.dumps(game)).spec_data == game.spec_data
    game.start_game()
    with pytest.raises(RuntimeError, match="after start_game"):
        pickle.dumps(game)
    with pytest.raises(RuntimeError, match="after start_game"):
        copy.deepcopy(game)


def test_game_run_json_round_trip():
    """benchmark.json has to carry back everything the summary reads."""
    game = ArcGame(spec_data=_spec(n_to_answer=2))
    game.start_game()
    right, wrong = _g([6, 8]), _g([0, 0])
    game.execute_action(Submission(attempts=((right, wrong), (wrong, wrong))), generated_tokens=12)
    game.finish_game()
    run = game.game_run
    assert run is not None
    back = taaf.game.GameRun.from_json_dict(run.to_json_dict())
    assert back.state == run.state
    assert back.final_score == run.final_score
    assert back.history[0].submission == run.history[0].submission
    assert back.result is not None and back.result.n_correct == 1


def test_example_counting_game_tracks_its_hooks():
    """The lifecycle runs each hook exactly once."""
    game = taaf.game_examples.CountingGame(spec_data=_spec())
    game.start_game()
    game.execute_action(Submission(attempts=((_g([6, 8]), _g([0, 0])),)))
    game.finish_game()
    assert (game.starts, game.executes, game.finishes) == (1, 1, 1)
