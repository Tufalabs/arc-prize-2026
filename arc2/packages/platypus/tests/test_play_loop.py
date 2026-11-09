"""The play call and its contract: what reaches the agent, what it may
submit, and what the answer channel refuses."""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass
from pathlib import Path

import pytest
from infra_fixtures import (
    BadReturnAgent,
    CrashingAgent,
    LateSubmitAgent,
    MisShapedAgent,
    ProbeAgent,
    SlowAgent,
    mirror,
)

import taaf.benchmark
import taaf.game
import taaf.game_examples
from platypus import AgentSolver, GiveUp, GiveUpError, MoveContext, RunConfig, Submission
from taaf.game_examples import constant_game, flip_game


def _bm(agent: ProbeAgent, n_games: int = 1, **kwargs: object) -> taaf.benchmark.Benchmark:
    """A benchmark of flip tasks hosted by ``agent``."""
    return taaf.benchmark.Benchmark(
        label="t",
        games=[taaf.game_examples.flip_game(f"g{i}") for i in range(n_games)],
        solver=AgentSolver(agent=agent),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_the_agent_is_asked_once_and_sees_the_task_without_its_answers():
    """One call per game, and the view carries the demonstrations in full
    and the questions bare — the answers it must produce are not in it."""
    ProbeAgent.reset_log()
    bm = _bm(ProbeAgent())
    await bm.run()
    assert len(ProbeAgent.seen) == 1
    view = ProbeAgent.seen[0]
    assert view.game_id == "g0"
    assert len(view.given) == 3 and len(view.questions) == 1
    assert not hasattr(view, "answers")


async def test_every_submission_is_recorded_and_the_last_one_scores():
    """The channel is the record: each call appends to history, and TAAF
    scores whichever answer stood at the end."""
    ProbeAgent.reset_log()
    bm = _bm(ProbeAgent(n_answers=3))
    await bm.run()
    run = bm.game_runs[0]
    assert len(run.history) == 3
    assert run.final_score == 100.0
    assert run.state == "won"


@dataclass
class SilentAgent(ProbeAgent):
    """Returns without ever submitting."""

    async def _do_move(self, ctx: MoveContext, view: object) -> GiveUp | None:
        """Do nothing at all."""
        return None


@dataclass
class QuitAfterAnsweringAgent(ProbeAgent):
    """Answers, then gives up."""

    async def _do_move(self, ctx: MoveContext, view: object) -> GiveUp | None:
        """Submit the mirror, then stop early."""
        ctx.submit(mirror(view))  # type: ignore[arg-type]
        return GiveUp("nothing more to try")


@dataclass
class RaisesDeepAgent(ProbeAgent):
    """Raises GiveUpError from a nested call."""

    async def _do_move(self, ctx: MoveContext, view: object) -> GiveUp | None:
        """Bail from inside a helper."""

        def inner() -> None:
            raise GiveUpError("no idea")

        inner()
        return None


@dataclass
class AnswerThenHangAgent(ProbeAgent):
    """Submits the mirror, then blocks past any deadline."""

    async def _do_move(self, ctx: MoveContext, view: object) -> GiveUp | None:
        """Answer, then wait to be cut off."""
        ctx.submit(mirror(view))  # type: ignore[arg-type]
        await asyncio.sleep(3600)
        return None


@dataclass
class CountingHistoryAgent(ProbeAgent):
    """Records how long its history is before each answer, into a file."""

    dump: str = ""

    async def _do_move(self, ctx: MoveContext, view: object) -> GiveUp | None:
        """Submit three times, watching history grow."""
        lengths = [len(ctx.history)]
        for _ in range(3):
            ctx.submit(mirror(view))  # type: ignore[arg-type]
            lengths.append(len(ctx.history))
        assert isinstance(ctx.history[-1], Submission)
        Path(self.dump).write_text(",".join(str(n) for n in lengths))
        return None


async def test_an_agent_that_never_answers_gives_up():
    """No answer is a legitimate outcome, distinct from a wrong one: the
    submission file gets a placeholder and the state says why."""
    bm = _bm(SilentAgent())
    await bm.run()
    run = bm.game_runs[0]
    assert run.history == [] and run.state == "gave_up"
    # Nothing correct, and that needs no answer key: a zero, not an unknown.
    assert run.final_score == 0.0


async def test_give_up_keeps_the_answer_it_already_made():
    """Giving up frees the slot; it does not withdraw the answer. The
    distinction matters under a deadline, where giving up early is how an
    agent hands its slot to the queue."""
    bm = _bm(QuitAfterAnsweringAgent())
    await bm.run()
    run = bm.game_runs[0]
    assert len(run.history) == 1
    assert run.final_score == 100.0  # scored, despite the give-up
    assert run.solver_note is not None and "nothing more to try" in run.solver_note


async def test_give_up_error_from_any_depth_is_a_give_up():
    """Raising is equivalent to returning, so an agent deep in a helper can
    bail without threading a return value back up."""
    bm = _bm(RaisesDeepAgent())
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "gave_up"
    assert run.solver_note is not None and "no idea" in run.solver_note


async def test_returning_a_submission_is_a_contract_violation():
    """There is one answer channel. Returning an answer instead of
    submitting it would silently discard it, so it fails loudly."""
    bm = _bm(BadReturnAgent())
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "expected GiveUp | None" in run.solver_note


async def test_a_mis_shaped_answer_fails_at_the_submit_call():
    """A shape bug is the agent's, and it should read that way — not as a
    wrong answer discovered by the scorer."""
    bm = _bm(MisShapedAgent())
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "this task has 1 to answer" in run.solver_note


async def test_submitting_after_the_call_is_refused():
    """``ctx.submit`` is armed only while ``do_move`` runs. An answer from
    ``on_finish`` would arrive after the game was scored."""
    bm = _bm(LateSubmitAgent())
    await bm.run()
    run = bm.game_runs[0]
    # The late attempt is inside on_finish, whose failures are recorded
    # rather than raised — the game itself already ended cleanly.
    assert len(run.history) == 1
    assert run.solver_note is not None and "only available while do_move is running" in run.solver_note


async def test_a_crash_keeps_whatever_was_already_answered():
    """An agent that dies mid-thought still submits what it had, which is
    the whole reason answering is a channel rather than a return value."""
    bm = _bm(CrashingAgent(n_answers=2))
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "crashed"
    assert len(run.history) == 2
    assert run.final_score == 100.0


async def test_on_agent_error_decides_whether_one_failure_ends_the_run():
    """ "crash_game" contains the failure — the game is marked crashed and the
    run finishes; "raise" propagates out of the solver so a debugger lands on
    the original frame. Asserted against the solver rather than the
    benchmark, which deliberately swallows solver exceptions."""
    contained = _bm(CrashingAgent(), n_games=2)
    await contained.run()
    assert [r.state for r in contained.game_runs] == ["crashed"] * 2

    agent = CrashingAgent(
        run_config=RunConfig(process_mode="inprocess", status_interval_s=None, on_agent_error="raise")
    )
    game = taaf.game_examples.flip_game("g0")
    game.start_game()
    solver = AgentSolver(agent=agent)
    solver.setup()
    with pytest.raises(RuntimeError, match="deliberate agent crash"):
        await solver.run_games([game])


async def test_the_runtime_limit_cancels_the_call_and_scores_what_stood(tmp_path: Path):
    """The budget's whole point here: a task cut off part-way submits its
    best so far rather than a placeholder."""
    agent = AnswerThenHangAgent(
        run_config=RunConfig(process_mode="inprocess", status_interval_s=None, max_runtime_s_per_game=0.2)
    )
    bm = _bm(agent, job_dir=tmp_path)
    bm.submission_path = tmp_path / "submission.json"
    await bm.run()
    run = bm.game_runs[0]
    # It answered, so the state says so — "gave_up" is reserved for a game
    # that never answered at all. What the budget did is in the note.
    assert run.state == "won"
    assert run.solver_note is not None and "per-game runtime limit" in run.solver_note
    assert run.final_score == 100.0
    import taaf.submission

    written = taaf.submission.read(tmp_path / "submission.json")
    assert written["g0"][0]["attempt_1"] != taaf.submission.PLACEHOLDER.to_lists()


@dataclass
class ParkThenAnswerAgent(ProbeAgent):
    """Works for ``work_s``, offers its slot back, works ``work_s`` again and
    answers — the same spend either side of a park whose length the host
    decides."""

    work_s: float = 0.25

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Two equal stints of work around one free offer."""
        await asyncio.sleep(self.work_s)
        await ctx.yield_slot({"tokens": 1, "compacted": True})
        await asyncio.sleep(self.work_s)
        ctx.submit(mirror(view))
        return None


@pytest.mark.parametrize("process_mode", ["inprocess", "subprocess"])
async def test_parked_time_is_not_charged_to_the_runtime_limit(process_mode: str):
    """Three games on one ranked slot: each works 0.5 s in all and is parked
    for the other two's stints in between, since a game that has spent
    outranks none that has not. Under a 0.8 s net every game finishes — the
    net measures a game's own tenure, and a game waiting for the run to
    hand its slot back is not the kernel-spinner the net exists to catch.
    Counted by wall clock, the first game would be cut at 1.0 s. Both
    process modes, because a child measures its park inside a blocking
    wait on the pipe."""
    agent = ParkThenAnswerAgent(
        run_config=RunConfig(
            process_mode=process_mode,  # type: ignore[arg-type]
            status_interval_s=None,
            max_concurrent=1,
            slot_policy="least_tokens",
            max_runtime_s_per_game=0.8,
        )
    )
    bm = _bm(agent, n_games=3)
    await bm.run()
    assert [r.state for r in bm.game_runs] == ["won"] * 3
    assert all(r.solver_note is None or "per-game runtime limit" not in r.solver_note for r in bm.game_runs)


@pytest.mark.parametrize("process_mode", ["inprocess", "subprocess"])
async def test_the_runtime_limit_resumes_with_what_was_left_after_a_park(process_mode: str):
    """The pause is not a reset. Two games on one ranked slot, each working
    0.5 s in all under a 0.4 s net, are both cut in their second stint — at
    the 0.15 s the first stint left, whatever the park in between."""
    agent = ParkThenAnswerAgent(
        run_config=RunConfig(
            process_mode=process_mode,  # type: ignore[arg-type]
            status_interval_s=None,
            max_concurrent=1,
            slot_policy="least_tokens",
            max_runtime_s_per_game=0.4,
        )
    )
    bm = _bm(agent, n_games=2)
    await bm.run()
    assert [r.state for r in bm.game_runs] == ["gave_up"] * 2
    assert all(r.solver_note is not None and "per-game runtime limit" in r.solver_note for r in bm.game_runs)


async def test_on_finish_is_told_how_the_call_ended_and_nothing_more():
    """The outcome an agent sees names how its own call ended. It never
    distinguishes won from submitted — that is the scorer's answer, and an
    agent that could read it would have an oracle."""
    ProbeAgent.reset_log()
    bm = _bm(ProbeAgent(), n_games=2)
    await bm.run()
    assert {game_id for game_id, _, _ in ProbeAgent.finishes} == {"g0", "g1"}
    for _, outcome, _ in ProbeAgent.finishes:
        assert outcome.state == "finished"
        assert outcome.n_submitted == 1
        assert not hasattr(outcome, "score")


async def test_history_is_the_agents_own_answers(tmp_path: Path):
    """An agent can read back what it has already submitted, in order,
    without keeping its own copy."""
    dump = tmp_path / "lengths.txt"
    bm = _bm(CountingHistoryAgent(dump=str(dump)))
    await bm.run()
    assert dump.read_text() == "0,1,2,3"


async def test_every_pass_gets_a_fresh_copy_and_the_template_stays_untouched():
    """The solver plays deep copies of the template, one per (game, pass):
    state advances inside the run and never on the instance the caller
    holds, so one configured agent can describe a whole experiment and be
    reused for the next."""
    ProbeAgent.reset_log()
    template = ProbeAgent(n_answers=2)
    bm = _bm(template, n_passes=2)
    await bm.run()
    assert [run.state for run in bm.game_runs] == ["won", "won"]
    assert template.calls == 0
    # Each pass copy was asked exactly once, from a fresh state.
    assert [calls for _, _, calls in ProbeAgent.finishes] == [1, 1]


async def test_games_play_concurrently_on_isolated_copies():
    """Two tasks, one template: distinct outcomes prove each game got its
    own copy and its own history rather than sharing one."""
    ProbeAgent.reset_log()
    bm = taaf.benchmark.Benchmark(
        label="t", games=[flip_game("mirror"), constant_game("constant")], solver=AgentSolver(agent=ProbeAgent())
    )
    await bm.run()
    assert {run.game_id: run.state for run in bm.game_runs} == {"mirror": "won", "constant": "submitted"}
    assert sorted((game_id, calls) for game_id, _, calls in ProbeAgent.finishes) == [("constant", 1), ("mirror", 1)]


async def test_cancellation_lands_mid_call_and_on_finish_still_runs():
    """Cancelling the run cuts every in-flight call at its next await; the
    games finish ``cancelled`` with what they had answered, and ``on_finish``
    runs for each, told so."""
    ProbeAgent.reset_log()
    solver = AgentSolver(agent=SlowAgent(n_answers=1))
    solver.setup()
    games: list[taaf.game.Game] = [flip_game(f"game{i}") for i in range(2)]
    for game in games:
        game.start_game()
    task = asyncio.create_task(solver.run_games(games))
    # Both games in their call: starts are paced (``RunConfig.start_gap_s``),
    # so a fixed number of loop ticks does not reach the second one's call.
    while len(ProbeAgent.seen) < len(games):
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for game in games:
        assert game.game_run is not None
        assert game.game_run.state == "cancelled" and len(game.game_run.history) == 1
    assert sorted(game_id for game_id, _, _ in ProbeAgent.finishes) == ["game0", "game1"]
    assert all(outcome.state == "cancelled" for _, outcome, _ in ProbeAgent.finishes)


async def test_a_zero_budget_gives_up_before_the_first_answer():
    """The limit is a deadline, not a between-answers check: a budget that
    is already spent cuts the call at its first await, before any answer."""
    agent = ProbeAgent(
        run_config=RunConfig(process_mode="inprocess", status_interval_s=None, max_runtime_s_per_game=0.0)
    )
    bm = _bm(agent)
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "gave_up" and run.history == []
    assert run.solver_note is not None and "per-game runtime limit" in run.solver_note


async def test_no_budget_plays_to_completion():
    """``None`` is no deadline at all — the call runs until it returns."""
    agent = ProbeAgent(
        n_answers=2, run_config=RunConfig(process_mode="inprocess", status_interval_s=None, max_runtime_s_per_game=None)
    )
    bm = _bm(agent)
    await bm.run()
    run = bm.game_runs[0]
    assert run.state == "won" and len(run.history) == 2


def test_the_agent_pickles_with_its_config_and_state():
    """Everything on an agent must pickle — that is how it reaches a game
    child, and how a deploy bundle carries it."""
    agent = ProbeAgent(n_answers=3, run_config=RunConfig(rng_seed=7, on_agent_error="raise"))
    agent.calls = 1
    assert pickle.loads(pickle.dumps(agent)) == agent


def test_the_solver_pickles_before_setup():
    """A deploy bundle pickles the whole solver before it has run; live
    handles only appear at setup."""
    solver = AgentSolver(agent=ProbeAgent(n_answers=2))
    restored = pickle.loads(pickle.dumps(solver))
    assert isinstance(restored.agent, ProbeAgent) and restored.agent.n_answers == 2
