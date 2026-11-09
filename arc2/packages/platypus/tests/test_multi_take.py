"""``platypus.multi_take``: the composition rule, a take's endings, and a
two-take agent hosted end to end.

What is claimed: the composed answer follows the rule for every combination
of what the takes produced, per question; a take that gives up, crashes or
outruns its own time hands on to the next without giving the game up, time
the game spends parked is not charged to a take, and a take cannot answer
once its play is over; construction copies the takes and takes over the
first one's configuration, and refuses configuration handed to the wrapper;
the child copy carries no engine anywhere; the ranking places a game in its
second take behind its spend, and a take boundary hands the slot to a game
that has not started. The hosted runs play the real solver over the stub
backend, in both process modes, and read the take log off stdout.
"""

from __future__ import annotations

import asyncio
import pickle
import random
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pytest
import ullm.backend
import ullm.backends.stub as ullm_stub
import ullm.factory

import taaf.benchmark as taaf_benchmark
import taaf.game
from platypus import GameOutcome, GiveUp, GiveUpError, LLMAgent, LLMAgentSolver, LLMConfig, LLMMoveContext, RunConfig
from platypus.context import Produced
from platypus.contract import ContractViolationError
from platypus.llm import build_llm_context
from platypus.multi_take import MultiTakeAgent, TakeAnswer, compose
from platypus.reference_agents import ANSWERS, PromptedFlipAgent
from platypus.solver import _rank_key
from taaf.game_examples import flip_game, flip_task
from taaf.grid import Grid

# --------------------------------------------------------------------------
# the composition rule
# --------------------------------------------------------------------------


def _grid(value: int) -> Grid:
    return Grid(np.full((2, 2), value, dtype=np.int8))


#: One question's two slots as a take submitted them: a value is a grid the
#: take produced, None a filler (the grid then holds 0).
Slots = tuple[int | None, int | None]


def _answer(*questions: Slots) -> TakeAnswer:
    attempts = tuple(tuple(_grid(0 if value is None else value) for value in slots) for slots in questions)
    produced = tuple(tuple(value is not None for value in slots) for slots in questions)
    return TakeAnswer(submission=taaf.game.Submission(attempts=attempts), produced=produced)


# take 1's slots, take 2's slots (None: the take has not answered), and the
# composed (grid, source) per attempt.
CASES: dict[str, tuple[Slots | None, Slots | None, tuple[tuple[int, str], tuple[int, str]]]] = {
    "take 2 disagrees": ((1, None), (2, None), ((1, "take 1 grid 1"), (2, "take 2 grid 1"))),
    "take 2 agrees and take 1 has an alternative": ((1, 3), (1, None), ((1, "take 1 grid 1"), (3, "take 1 grid 2"))),
    "take 2's grid comes before take 1's alternative": (
        (1, 3),
        (2, 4),
        ((1, "take 1 grid 1"), (2, "take 2 grid 1")),
    ),
    "only take 2's second grid differs": ((1, None), (1, 4), ((1, "take 1 grid 1"), (4, "take 2 grid 2"))),
    "a second grid identical to attempt 1 is passed over": (
        (1, 1),
        (1, 4),
        ((1, "take 1 grid 1"), (4, "take 2 grid 2")),
    ),
    "nothing differs": ((1, 1), (1, 1), ((1, "take 1 grid 1"), (1, "copy of attempt 1"))),
    "take 2 has not answered yet": ((1, 3), None, ((1, "take 1 grid 1"), (3, "take 1 grid 2"))),
    "take 1 never answered": (None, (2, 4), ((2, "take 2 grid 1"), (4, "take 2 grid 2"))),
    "take 1 produced nothing for the question": (
        (None, None),
        (2, None),
        ((2, "take 2 grid 1"), (2, "copy of attempt 1")),
    ),
    "take 1's only grid sits in its second slot": (
        (None, 3),
        (2, None),
        ((3, "take 1 grid 2"), (2, "take 2 grid 1")),
    ),
}


@pytest.mark.parametrize(("take1", "take2", "expected"), CASES.values(), ids=list(CASES))
def test_the_composition_follows_the_rule(
    take1: Slots | None, take2: Slots | None, expected: tuple[tuple[int, str], tuple[int, str]]
) -> None:
    answers = [None if slots is None else _answer(slots) for slots in (take1, take2)]
    composition = compose([_grid(9)], answers)
    assert composition is not None
    assert composition.submission.attempts[0] == tuple(_grid(value) for value, _ in expected)
    assert composition.sources[0] == tuple(source for _, source in expected)
    assert composition.produced[0] == tuple(source.startswith("take") for _, source in expected)


def test_questions_compose_independently_and_an_unanswered_one_gets_its_input() -> None:
    first_question, second_question = _grid(8), _grid(9)
    composition = compose([first_question, second_question], [_answer((1, None), (None, None)), None])
    assert composition is not None
    assert composition.submission.attempts == ((_grid(1), _grid(1)), (second_question, second_question))
    assert composition.sources[1] == ("placeholder", "copy of attempt 1")
    assert composition.produced == ((True, False), (False, False))


def test_nothing_produced_composes_to_nothing() -> None:
    assert compose([_grid(9)], [_answer((None, None)), None]) is None
    assert compose([_grid(9)], [None, None]) is None


# --------------------------------------------------------------------------
# the ranking
# --------------------------------------------------------------------------


def test_the_ranking_places_a_later_take_behind_its_spend() -> None:
    key = _rank_key(RunConfig(slot_policy="least_tokens", take_handicap_tokens=30_000), [])
    hints: dict[str, dict[str, Any]] = {
        "0": {"tokens": 25_000, "take_index": 1},  # ranks at 55k
        "1": {"tokens": 50_000},
        "2": {"tokens": 60_000},
        "3": {},
    }
    assert sorted(hints, key=lambda game: key(game, hints[game])) == ["3", "1", "0", "2"]


# --------------------------------------------------------------------------
# construction and the child crossing
# --------------------------------------------------------------------------


def _stub_config() -> LLMConfig:
    backend = ullm_stub.StubBackend.from_script("8", on_exhausted="loop")
    return LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
    )


def _flip(answer: str, run_config: RunConfig | None = None) -> PromptedFlipAgent:
    return PromptedFlipAgent(
        answers=(answer,),
        randomize_sum=False,  # the scripted stub only ever answers "8"
        llm_config=_stub_config(),
        run_config=run_config or RunConfig(),
    )


def test_construction_copies_the_takes_and_takes_over_the_first_ones_configuration() -> None:
    first = _flip("flip", RunConfig(max_runtime_s_per_game=60.0))
    agent = MultiTakeAgent(takes=(first, first))
    one, two = agent.takes
    assert one is not first and two is not one, "the same agent twice must be two independent takes"
    assert first.run_config.max_runtime_s_per_game == 60.0, "the caller's agent is never modified"
    assert agent.run_config is one.run_config and agent.llm_config is one.llm_config
    assert two.llm_config.backend_factory is one.llm_config.backend_factory
    agent.run_config.min_request_gap_s = 1.25
    assert two.run_config.min_request_gap_s == 1.25, "a hosting override reaches every take"
    assert agent.run_config.max_runtime_s_per_game is None, "a one-take net would cut the pair short"
    assert agent.max_runtime_s_per_take == 60.0


@dataclass(kw_only=True)
class _Picturing(PromptedFlipAgent):
    """A take whose requests carry images."""

    def needs_images(self) -> bool:
        return True


def test_the_wrapper_needs_images_when_any_take_does() -> None:
    assert not MultiTakeAgent(takes=(_flip("flip"), _flip("flip"))).needs_images()
    picturing = _Picturing(answers=("flip",), randomize_sum=False, llm_config=_stub_config(), run_config=RunConfig())
    assert MultiTakeAgent(takes=(_flip("flip"), picturing)).needs_images()
    assert MultiTakeAgent(takes=(picturing, _flip("flip"))).needs_images()


def test_configuration_handed_to_the_wrapper_is_refused() -> None:
    with pytest.raises(ValueError, match="first take"):
        MultiTakeAgent(takes=(_flip("flip"), _flip("flip")), run_config=RunConfig(max_concurrent=3))


def test_the_child_copy_carries_no_engine_anywhere() -> None:
    agent = MultiTakeAgent(takes=(_flip("flip"), _flip("black")))
    child = cast(MultiTakeAgent, pickle.loads(pickle.dumps(LLMAgentSolver(agent=agent)._child_agent(agent))))
    assert child.llm_config.backend_factory is None
    assert [take.llm_config.backend_factory for take in child.takes] == [None, None]
    assert agent.llm_config.backend_factory is not None, "the template keeps the engine the host starts"
    assert all(take.llm_config.backend_factory is not None for take in agent.takes)


# --------------------------------------------------------------------------
# a take's endings, driven directly
# --------------------------------------------------------------------------

#: Every take's answer channel, kept past its play so a late answer can be tried.
_CHANNELS: list[Any] = []


@dataclass(kw_only=True)
class _Scripted(LLMAgent):
    """A take that answers from ``ANSWERS`` and then ends as told."""

    answer: str | None = None
    ending: Literal["return", "give_up", "raise", "overrun"] = "return"
    park: bool = False
    cleanup_fails: bool = False

    async def _do_move(self, ctx: LLMMoveContext, view: taaf.game.GameView) -> GiveUp | None:
        _CHANNELS.append(ctx.submit)
        if self.park:
            await ctx.yield_slot({"tokens": 0})
        if self.answer is not None:
            ctx.submit(ANSWERS[self.answer](view))
        if self.ending == "give_up":
            raise GiveUpError("scripted give-up")
        if self.ending == "raise":
            raise RuntimeError("scripted crash")
        if self.ending == "overrun":
            await asyncio.sleep(3600)
        return None

    async def _on_finish(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, ctx: LLMMoveContext, outcome: GameOutcome
    ) -> None:
        del ctx, outcome
        if self.cleanup_fails:
            raise OSError("scripted cleanup failure")


async def _drive(agent: MultiTakeAgent, *, park_s: float = 0.0) -> tuple[list[taaf.game.Submission], GiveUp | None]:
    _CHANNELS.clear()
    view = flip_task("scripted").view()
    submissions: list[taaf.game.Submission] = []

    def submit(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        del produced
        submissions.append(submission)

    async def park(hints: Any) -> None:
        del hints
        await asyncio.sleep(park_s)

    backend = ullm_stub.StubBackend.from_script("8", on_exhausted="loop")
    ctx, _meter = build_llm_context("scripted", submissions, random.Random(0), backend, submit, yield_slot=park)
    decision = await agent.do_move(ctx, view)
    await agent.on_finish(ctx, GameOutcome(state="finished", n_submitted=len(submissions)))
    return submissions, decision


def _mirror_and_black() -> tuple[Grid, Grid]:
    question = flip_task("scripted").view().questions[0]
    return Grid(np.fliplr(question.data)), Grid(np.zeros_like(question.data))


async def test_a_take_that_gives_up_hands_on_and_the_game_does_not_give_up() -> None:
    agent = MultiTakeAgent(takes=(_Scripted(ending="give_up"), _Scripted(answer="flip")))
    submissions, decision = await _drive(agent)
    mirror, _ = _mirror_and_black()
    assert decision is None
    assert submissions[-1].attempts[0][0] == mirror


async def test_the_game_gives_up_only_when_every_take_did() -> None:
    agent = MultiTakeAgent(takes=(_Scripted(ending="give_up"), _Scripted(ending="give_up")))
    _, decision = await _drive(agent)
    assert isinstance(decision, GiveUp)
    assert "take 1 gave_up" in decision.reason and "take 2 gave_up" in decision.reason


async def test_a_crashing_take_hands_on_unless_crashes_are_raised(capfd: pytest.CaptureFixture[str]) -> None:
    agent = MultiTakeAgent(takes=(_Scripted(answer="black", ending="raise"), _Scripted(answer="flip")))
    submissions, decision = await _drive(agent)
    mirror, black = _mirror_and_black()
    assert decision is None
    assert submissions[-1].attempts[0] == (black, mirror)
    assert "take 1 of 2 finished (crashed: RuntimeError: scripted crash)" in capfd.readouterr().out

    raising = MultiTakeAgent(
        takes=(_Scripted(ending="raise", run_config=RunConfig(on_agent_error="raise")), _Scripted(answer="flip"))
    )
    with pytest.raises(RuntimeError, match="scripted crash"):
        await _drive(raising)


@pytest.mark.parametrize("on_agent_error", ["crash_game", "raise"])
async def test_a_take_whose_cleanup_fails_still_hands_on(
    on_agent_error: str, capfd: pytest.CaptureFixture[str]
) -> None:
    """A take's failing ``on_finish`` is contained as the host contains an
    agent's, under either error policy: the take has already played, and
    its cleanup must not cost the next take its answer."""
    run_config = RunConfig(on_agent_error=on_agent_error)  # type: ignore[arg-type]
    agent = MultiTakeAgent(
        takes=(_Scripted(answer="black", cleanup_fails=True, run_config=run_config), _Scripted(answer="flip"))
    )
    submissions, decision = await _drive(agent)
    mirror, black = _mirror_and_black()
    assert decision is None
    assert submissions[-1].attempts[0] == (black, mirror)
    out = capfd.readouterr().out
    assert "[take] take 1 on_finish failed; the game plays on:" in out
    assert "OSError: scripted cleanup failure" in out


async def test_a_take_cannot_answer_once_its_play_is_over() -> None:
    agent = MultiTakeAgent(takes=(_Scripted(answer="black"), _Scripted(answer="flip")))
    await _drive(agent)
    late = _CHANNELS[0]
    with pytest.raises(ContractViolationError, match="take 1 submitted after its play ended"):
        late(ANSWERS["flip"](flip_task("scripted").view()))


async def test_a_take_that_outruns_its_net_is_cut_and_the_next_one_plays(capfd: pytest.CaptureFixture[str]) -> None:
    agent = MultiTakeAgent(takes=(_Scripted(answer="black", ending="overrun"), _Scripted(answer="flip")))
    agent.max_runtime_s_per_take = 0.2
    submissions, decision = await _drive(agent)
    mirror, black = _mirror_and_black()
    assert decision is None
    assert submissions[-1].attempts[0] == (black, mirror)
    assert "take 1 of 2 finished (cancelled: per-take runtime limit (0.2 s))" in capfd.readouterr().out


async def test_time_parked_is_not_charged_to_the_take(capfd: pytest.CaptureFixture[str]) -> None:
    agent = MultiTakeAgent(takes=(_Scripted(answer="flip", park=True), _Scripted(answer="flip")))
    agent.max_runtime_s_per_take = 0.5
    _, decision = await _drive(agent, park_s=1.0)
    assert decision is None
    assert "take 1 of 2 finished (finished)" in capfd.readouterr().out


# --------------------------------------------------------------------------
# hosted end to end
# --------------------------------------------------------------------------


async def _play(agent: MultiTakeAgent, games: list[taaf.game.Game]) -> list[taaf.game.GameRun]:
    benchmark = taaf_benchmark.Benchmark(games=games, solver=LLMAgentSolver(agent=agent))
    await benchmark.run()
    return benchmark.game_runs


@pytest.mark.parametrize("mode", ["inprocess", "subprocess"])
async def test_a_wrong_first_take_and_a_right_second_one_score_together(
    mode: str, capfd: pytest.CaptureFixture[str]
) -> None:
    run_config = RunConfig(process_mode=mode, rng_seed=7)  # type: ignore[arg-type]
    agent = MultiTakeAgent(takes=(_flip("black", run_config), _flip("flip")))
    (run,) = await _play(agent, [flip_game("g")])
    mirror = Grid(np.fliplr(flip_task("g").view().questions[0].data))
    black = Grid(np.zeros_like(mirror.data))
    assert run.history[-1].submission.attempts[0] == (black, mirror)
    assert run.result is not None and run.result.score == 100.0
    out = capfd.readouterr().out
    assert "[take] ==== g pass 0: take 1 of 2 finished (finished) ====" in out
    assert "[take] ==== g pass 0: take 2 of 2 finished (finished) ====" in out
    assert "g pass 0 take 1 question 1: produced take 1 grid 1" in out
    final = [line for line in out.splitlines() if line.startswith("[take] g pass 0 answer question 1:")][-1]
    assert "attempt 1 = take 1 grid 1" in final and "attempt 2 = take 2 grid 1" in final


async def test_a_take_boundary_hands_the_slot_to_a_game_that_has_not_started(
    capfd: pytest.CaptureFixture[str],
) -> None:
    run_config = RunConfig(
        process_mode="inprocess",
        max_concurrent=1,
        slot_policy="least_tokens",
        slot_hold_s=None,
        take_handicap_tokens=30_000,
        rng_seed=7,
    )
    agent = MultiTakeAgent(takes=(_flip("flip", run_config), _flip("flip")))
    await _play(agent, [flip_game("a"), flip_game("b")])
    events = [line for line in capfd.readouterr().out.splitlines() if line.startswith(("a has been", "b has been"))]
    assert [event.split(",")[0].rstrip(".") for event in events] == [
        "a has been started",
        "a has been paused",
        "b has been started",
        "b has been paused",
        "a has been resumed",
        "b has been resumed",
    ]
    assert events[1].endswith("take 1 finished, take 2 not yet started.")
