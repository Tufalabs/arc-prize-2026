"""The duck in two takes (``duck.multi_take``) and the provenance a duck
submits.

What is claimed: a duck's submission says which grids the model wrote, and a
change in that alone is a new answer; the second take starts with its whole
token allowance although the game's meter already holds the first take's
spend, and opens a history epoch past the first take's; the build asks both
takes for their likeliest answer first, in the prompt they actually render,
and the line stands alone where the second-attempt line is left out; the
second take differs from the first in the things chosen for it and in
nothing else; a later
take that reads the game's token meter as its own is refused, at
construction and after it, and so is a per-game observer on any take; and
the engine follows the first take, while a
duck that does not own the engine writes nothing onto it and is refused if
it does not fit. The ten-take duck is the two-take build's first take ten
times over, on the same run configuration and net, and a later take of it
starts as the second take of two does.
"""

from __future__ import annotations

import dataclasses
import pickle
import random
from typing import Any

import numpy as np
import pytest
import ullm.backend
from duck import DuckAgent, MultiTakeDuck, TieredTakeDuck
from duck.multi_take import SECOND_TAKE_AUDIT_HANDICAP_TOKENS, SECOND_TAKE_TEMPERATURE
from duck.prompts import DuckPrompts
from duck.state import reading_from_payload

import taaf.game
import taaf.grid
from platypus import GiveUpError
from platypus.context import Produced
from platypus.llm import build_llm_context
from platypus.multi_take import N_TIERED_TAKES


def _context(spent: int = 0, submissions: list[tuple[taaf.game.Submission, Produced | None]] | None = None) -> Any:
    """A context whose backend has been billed ``spent`` tokens, recording
    every answer with its provenance."""
    kept = submissions if submissions is not None else []

    def submit(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        kept.append((submission, produced))

    backend = ullm.backend.DummyBackend()
    backend.total_usage.completion_tokens = spent
    ctx, _meter = build_llm_context(
        game_id="takes", submissions=[], rng=random.Random(0), backend=backend, submit=submit
    )
    return ctx


def _view() -> taaf.game.GameView:
    questions = tuple(taaf.grid.Grid(np.full((2, 3), value, dtype=np.int8)) for value in (1, 2))
    given = ((questions[0], questions[1]),)
    return taaf.game.GameView(game_id="takes", given=given, questions=questions)


def _ok(grid: taaf.grid.Grid) -> dict[str, Any]:
    return {"state": "ok", "grid": grid.to_lists()}


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_the_reading_says_which_grids_the_model_wrote() -> None:
    view = _view()
    first, second = view.questions
    other = taaf.grid.Grid(np.full((2, 3), 7, dtype=np.int8))
    reading = reading_from_payload(
        {"bound": True, "rows": [[_ok(other), {"state": "missing"}], [{"state": "missing"}, _ok(second)]]}
    )
    submission = reading.submission(view.questions)
    assert submission is not None
    assert submission.attempts == ((other, other), (second, second)), "one grid is copied into both slots"
    assert reading.produced() == ((True, False), (True, False)), "the copy is a filler, wherever the grid sat"

    both = reading_from_payload({"bound": True, "rows": [[_ok(other), _ok(first)], [{"state": "missing"}] * 2]})
    assert both.produced() == ((True, True), (False, False)), "an unanswered question is a placeholder"


def test_a_grid_the_model_writes_equal_to_the_placeholder_is_a_new_answer() -> None:
    """The input grid is also what fills an unanswered question. Once the
    model writes that same grid, the grids do not change and the provenance
    does — which an agent composing from this duck must be told."""
    view = _view()
    first, second = view.questions
    agent = DuckAgent()
    sent: list[tuple[taaf.game.Submission, Produced | None]] = []
    ctx = _context(submissions=sent)
    agent._absorb_answer(ctx, view, {"bound": True, "rows": [[_ok(first)], [{"state": "missing"}]]})
    agent._absorb_answer(ctx, view, {"bound": True, "rows": [[_ok(first)], [_ok(second)]]})
    assert len(sent) == 2
    assert sent[0][0] == sent[1][0], "the same grids"
    assert [produced for _, produced in sent] == [((True, False), (False, False)), ((True, False), (True, False))]


# --------------------------------------------------------------------------
# the take boundary
# --------------------------------------------------------------------------


def test_the_second_take_starts_with_its_whole_allowance_and_a_new_epoch() -> None:
    agent = MultiTakeDuck(takes=(DuckAgent(max_tokens_per_game=3_000), DuckAgent(max_tokens_per_game=3_000)))
    first, second = (take for take in agent.takes if isinstance(take, DuckAgent))
    first.history_epoch = 4
    after_first = _context(spent=3_000)
    with pytest.raises(GiveUpError):
        second._per_turn_cap(after_first)  # the game's meter, as the second take would find it unaided
    agent._start_take(second, previous=first, spent_tokens=3_000)
    assert second._per_turn_cap(after_first) == 3_000
    assert second.history_epoch == 5


def test_a_later_tiered_take_starts_with_its_whole_allowance_and_a_new_epoch() -> None:
    agent = TieredTakeDuck(takes=tuple(DuckAgent(max_tokens_per_game=3_000) for _ in range(N_TIERED_TAKES)))
    ducks = [take for take in agent.takes if isinstance(take, DuckAgent)]
    fourth, fifth = ducks[3], ducks[4]
    fourth.history_epoch = 7
    after_four = _context(spent=12_000)
    with pytest.raises(GiveUpError):
        fifth._per_turn_cap(after_four)
    agent._start_take(fifth, previous=fourth, spent_tokens=12_000)
    assert fifth._per_turn_cap(after_four) == 3_000
    assert fifth.history_epoch == 8


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


@pytest.fixture
def built() -> MultiTakeDuck:
    pytest.importorskip("vllm", reason="the build composes real engine args")
    return MultiTakeDuck.build()


def test_both_takes_render_the_likeliest_first_line(built: MultiTakeDuck) -> None:
    for take in built.takes:
        assert isinstance(take, DuckAgent)
        assert "leave the second alone. Put your likeliest answer in `test_out[i][0]`.\n" in take._system_prompt_text()


def test_the_second_take_differs_from_the_first_in_what_was_chosen_for_it(built: MultiTakeDuck) -> None:
    """Digit rows, the shorter brevity ask, the full tail after a summary
    (sweep 7's picks for moving which tasks it solves), no file tools and a
    cooler play profile (sweep 9's), and its audit sent behind every game that
    has not deferred -- and nothing else."""
    first, second = built.takes
    assert isinstance(first, DuckAgent) and isinstance(second, DuckAgent)
    assert not first.text_grids and second.text_grids
    assert first.summary_tail == "without_reasoning" and second.summary_tail == "full"
    assert isinstance(first.prompts, DuckPrompts) and isinstance(second.prompts, DuckPrompts)
    assert not first.prompts.brevity_line and second.prompts.brevity_line and second.prompts.brevity == "half"
    assert dataclasses.replace(second.prompts, brevity_line=False, brevity="full") == first.prompts
    assert first.audit_handicap_tokens == 0
    assert second.audit_handicap_tokens == SECOND_TAKE_AUDIT_HANDICAP_TOKENS == 10_000_000
    assert first.file_tools == "full" and second.file_tools == "none"
    # The play profile, which the audit and summaries sample on as the duck
    # ships: the second take's at its own temperature, the first's untouched,
    # and every other sampling setting and the engine shared.
    assert first.audit_profile is first.summary_profile is second.audit_profile is second.summary_profile is None
    play, cooler = first.llm_config.sampling_profiles["play"], second.llm_config.sampling_profiles["play"]
    assert isinstance(play, ullm.backend.GenericSamplingParameters)
    assert isinstance(cooler, ullm.backend.GenericSamplingParameters)
    assert play.temperature == 0.8 and cooler.temperature == SECOND_TAKE_TEMPERATURE == 0.5
    assert dataclasses.replace(cooler, temperature=play.temperature) == play
    assert second.llm_config.backend_factory is first.llm_config.backend_factory
    shared = {f.name for f in dataclasses.fields(DuckAgent)} - {
        "text_grids",
        "summary_tail",
        "prompts",
        "audit_handicap_tokens",
        "file_tools",
        "llm_config",
    }
    differing = {name for name in shared if getattr(first, name) != getattr(second, name)}
    assert differing == set(), sorted(differing)
    # In what the model reads: the caveat over the rows each take is shown or
    # prints, and the short ask.
    first_text, second_text = first._system_prompt_text(), second._system_prompt_text()
    assert "spells every grid out as digit rows" in second_text
    assert "spells every grid out" not in first_text
    assert DuckPrompts().printed_grid_no_hand_arithmetic_line.strip() in first_text
    assert DuckPrompts().half_brevity_line in second_text
    assert DuckPrompts().half_brevity_line not in first_text
    assert DuckPrompts().user_prompt_brevity_line not in second_text


REFUSED_IN_A_LATER_TAKE: dict[str, dict[str, Any]] = {
    "first_response_max_tokens": {"first_response_max_tokens": 1_000},
    "budget_line": {"prompts": DuckPrompts(budget_line=True)},
    "cost_model": {"prompts": DuckPrompts(cost_model="explicit")},
}


@pytest.mark.parametrize("name", sorted(REFUSED_IN_A_LATER_TAKE))
def test_a_later_take_that_reads_the_game_meter_as_its_own_is_refused(name: str) -> None:
    setting = REFUSED_IN_A_LATER_TAKE[name]
    MultiTakeDuck(takes=(DuckAgent(**setting), DuckAgent()))  # the first take's meter is its own
    with pytest.raises(ValueError, match=name):
        MultiTakeDuck(takes=(DuckAgent(), DuckAgent(**setting)))
    agent = MultiTakeDuck(takes=(DuckAgent(), DuckAgent()))
    later = agent.takes[1]
    for field_name, value in setting.items():
        setattr(later, field_name, value)
    with pytest.raises(ValueError, match=name):
        agent.prepare_for_backend()


class _Observer:
    """A context observer that records nothing: its presence is the subject."""

    def prepare_request(self, params: ullm.backend.GenericSamplingParameters) -> None:
        del params

    def prompt_measured(self, count: int, *, epoch: int, trigger: int, compacting: bool) -> None:
        del count, epoch, trigger, compacting

    def history_rewritten(self, *, epoch: int) -> None:
        del epoch

    def request_prepared(
        self, params: ullm.backend.GenericSamplingParameters, *, epoch: int, prompt_tokens: int, target: int
    ) -> None:
        del params, epoch, prompt_tokens, target


def _ignore_response(response: Any) -> None:
    del response


PER_GAME_OBSERVERS: dict[str, dict[str, Any]] = {
    "context_observer": {"context_observer": _Observer()},
    "response_observer": {"response_observer": _ignore_response},
}


@pytest.mark.parametrize("name", sorted(PER_GAME_OBSERVERS))
@pytest.mark.parametrize("take", [0, 1])
def test_a_per_game_observer_on_any_take_is_refused(name: str, take: int) -> None:
    setting = PER_GAME_OBSERVERS[name]
    takes = [DuckAgent(), DuckAgent()]
    takes[take] = DuckAgent(**setting)
    with pytest.raises(ValueError, match=name):
        MultiTakeDuck(takes=(takes[0], takes[1]))


def test_the_likeliest_first_line_stands_alone_where_the_second_attempt_line_is_left_out() -> None:
    duck = DuckAgent(prompts=DuckPrompts(likeliest_first_line=True, second_attempt_mode="none"))
    text = duck._system_prompt_text()
    assert "\n- Put your likeliest answer in `test_out[i][0]`.\n" in text
    assert "You may also assign `test_out[i][1]`" not in text


def test_a_duck_that_does_not_own_the_engine_writes_nothing_onto_it() -> None:
    pytest.importorskip("vllm", reason="the build composes real engine args")
    duck = DuckAgent.build()
    before = pickle.dumps(duck.llm_config.backend_factory)
    duck.context_window_tokens -= 8192
    duck.run_config.max_concurrent += 5
    duck.image_scale *= 2
    duck.prepare_for_backend(settle_engine=False)
    assert pickle.dumps(duck.llm_config.backend_factory) == before
    duck.prepare_for_backend()
    assert pickle.dumps(duck.llm_config.backend_factory) != before, "the same changes do move an engine it owns"


def test_the_engine_follows_the_first_take_and_the_second_must_fit(built: MultiTakeDuck) -> None:
    factory: Any = built.llm_config.backend_factory
    engine_args = factory.engine_args
    built.run_config.max_concurrent = 7
    built.prepare_for_backend()
    assert engine_args.max_num_seqs == 7
    second = built.takes[1]
    assert isinstance(second, DuckAgent)
    assert second.derive_max_model_len and second.derive_max_num_seqs and second.derive_max_pixels, (
        "the second take's derive flags are left as built: it is told not to settle, not reconfigured"
    )
    second.context_window_tokens = engine_args.max_model_len * 2
    with pytest.raises(ValueError, match="exceeds the engine's"):
        built.prepare_for_backend()
    assert engine_args.max_model_len * 2 == second.context_window_tokens, "the second take never moves the engine"


def test_every_tiered_take_is_the_two_take_build_s_first_take(built: MultiTakeDuck) -> None:
    """No take after the first changes its sampling or its prompts."""
    tiered = TieredTakeDuck.build()
    takes = [take for take in tiered.takes if isinstance(take, DuckAgent)]
    assert len(takes) == N_TIERED_TAKES and len({id(take) for take in takes}) == len(takes)
    built_first = built.takes[0]
    assert isinstance(built_first, DuckAgent)
    compared = [f.name for f in dataclasses.fields(DuckAgent) if f.name not in ("llm_config", "run_config")]
    for take in takes:
        assert {name: getattr(take, name) for name in compared} == {
            name: getattr(built_first, name) for name in compared
        }
        assert take.llm_config.sampling_profiles == built_first.llm_config.sampling_profiles
        assert take.llm_config.backend_factory is tiered.llm_config.backend_factory
    assert tiered.run_config == built.run_config
    assert tiered.max_runtime_s_per_take == built.max_runtime_s_per_take
