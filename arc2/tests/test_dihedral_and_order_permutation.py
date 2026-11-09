"""Duck augmentation: solo LR reflection, distinct multi-take dihedral layouts,
original answer/provenance order on both channels, seeded replay, and opt-in.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
from duck import DuckAgent, MultiTakeDuck, TieredTakeDuck
from duck.config import CheckMode
from duck.python_tool import DuckPythonTool

from platypus import LLMMoveContext, RunConfig
from platypus.context import Produced
from platypus.llm import LLMConfig, build_llm_context
from sweeper import Choice, Design, Field, Knob, LLMSweepAgent
from taaf.game import GameView, Submission
from taaf.grid import Grid

ORDER = (0, 1, 2)


def _view() -> GameView:
    a = Grid.from_lists([[1, 2, 3], [4, 5, 6]])
    b = Grid.from_lists([[7, 8], [9, 0], [1, 2]])
    c = Grid.from_lists([[2, 0, 4, 1]])
    return GameView(game_id="augmentation", given=((a, b), (b, c), (c, a)), questions=(a, b, c))


def _oriented(grid: Grid, transform: int) -> Grid:
    rows = grid.to_lists()
    if transform >= 4:
        rows = [row[::-1] for row in rows]
    for _ in range(transform % 4):
        rows = [list(row) for row in zip(*rows, strict=True)][::-1]
    return Grid.from_lists(rows)


def _expected(train_order: tuple[int, ...], test_order: tuple[int, ...]) -> GameView:
    view = _view()
    return GameView(
        game_id=view.game_id,
        given=tuple((_oriented(view.given[i][0], 4), _oriented(view.given[i][1], 4)) for i in train_order),
        questions=tuple(_oriented(view.questions[i], 4) for i in test_order),
    )


def _force_orders(monkeypatch: pytest.MonkeyPatch, train: tuple[int, ...], test: tuple[int, ...]) -> Mock:
    orders = iter((train, test))

    def shuffle(indices: list[int]) -> None:
        indices[:] = next(orders)

    factory = Mock(return_value=Mock(shuffle=shuffle))
    # Replace this module's binding, leaving the sweep's and context's RNGs real.
    monkeypatch.setattr("duck.agent.random", SimpleNamespace(Random=factory))
    return factory


@pytest.mark.parametrize(
    "enabled,train_order,test_order",
    [
        (False, ORDER, ORDER),
        (True, ORDER, ORDER),
        (True, (2, 0, 1), ORDER),
        (True, ORDER, (1, 2, 0)),
        (True, (2, 0, 1), (1, 2, 0)),
    ],
)
@pytest.mark.parametrize("mandatory", [False, True])
async def test_sweep_restores_answer_and_provenance_order(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    train_order: tuple[int, ...],
    test_order: tuple[int, ...],
    mandatory: bool,
) -> None:
    draw = _force_orders(monkeypatch, train_order, test_order)
    original = _view()
    expected = _expected(train_order, test_order) if enabled else original
    a, b, c = original.questions
    answers = (b, c, a)
    model_answers = tuple(_oriented(answers[i], 4) if enabled else answers[i] for i in test_order)
    inner = DuckAgent(
        use_dihedral_and_order_permutation=not enabled,
        check_mode=CheckMode.MANDATORY if mandatory else CheckMode.ENABLED,
        audit_turn=False,
        show_changed_answers=False,
        text_grids=True,
    )
    agent = LLMSweepAgent(
        inner=inner,
        llm_config=inner.llm_config,
        design=Design(
            arm={
                "augment": Knob(
                    draw=Choice((enabled,)),
                    apply=Field("inner.use_dihedral_and_order_permutation"),
                    missing=False,
                )
            }
        ),
        run_seed=0,
        arm_seeds=(0,),
        announce=False,
    )
    repl = Mock(setup_task=AsyncMock())
    monkeypatch.setattr(inner, "_ensure_repl", Mock(return_value=repl))
    submissions: list[Submission] = []
    provenance: list[Produced | None] = []

    def submit(answer: Submission, *, produced: Produced | None = None) -> None:
        submissions.append(answer)
        provenance.append(produced)

    async def analyze(ctx: LLMMoveContext, view: GameView, **_kwargs: Any) -> tuple[int, int]:
        assert view == expected
        repl.setup_task.assert_awaited_once_with(expected)
        text = inner._task_prompt(view).content.to_string()
        for q, answer in expected.given:
            assert inner._grid_rows(q) in text
            assert inner._grid_rows(answer) in text
        if mandatory:
            assert repl.on_answer is None
            repl.on_checked_answer([g.to_lists() for g in model_answers])
            repl.on_checked_answer([g.to_lists() for g in model_answers])
        else:
            assert repl.on_checked_answer is None
            rows = [
                [{"state": "ok", "grid": g.to_lists()} for g in pair]
                for pair in zip(model_answers, view.questions, strict=True)
            ]
            partial = {"bound": True, "rows": [rows[0], [{"state": "missing"}] * 2, [{"state": "missing"}] * 2]}
            repl.on_answer(partial)
            repl.on_answer(partial)
            answered = test_order[0]
            assert submissions == [
                Submission(
                    attempts=tuple(
                        (answers[i], q) if i == answered else (q, q) for i, q in enumerate(original.questions)
                    )
                )
            ]
            assert provenance == [tuple((i == answered, i == answered) for i in ORDER)]
            repl.on_answer({"bound": True, "rows": rows})
        # The duck's own reading stays in the model's shuffled/reflected space.
        second = model_answers if mandatory else view.questions
        assert inner.reading.submission(view.questions) == Submission(
            attempts=tuple(zip(model_answers, second, strict=True))
        )
        assert list(ctx.history) == submissions
        return 1, 0

    monkeypatch.setattr(inner, "_run_analyzer_entry", analyze)
    rng = random.Random(0)
    before = rng.getstate()
    ctx, _ = build_llm_context(original.game_id, submissions, rng, ullm.backend.DummyBackend(), submit)
    assert await agent.do_move(ctx, original) is None
    assert submissions[-1] == Submission(
        attempts=tuple(zip(answers, answers if mandatory else original.questions, strict=True))
    )
    assert provenance[-1] == ((True, not mandatory),) * 3
    assert len(submissions) == (1 if mandatory else 2)
    assert original == _view()
    assert rng.getstate() == before
    assert draw.call_count == int(enabled)


async def test_only_the_opted_in_take_is_augmented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DuckAgent, "_ensure_repl", Mock(return_value=Mock(setup_task=AsyncMock())))
    seen: list[GameView] = []

    async def analyze(self: DuckAgent, ctx: LLMMoveContext, view: GameView, **_kwargs: Any) -> tuple[int, int]:
        seen.append(view)
        self._absorb_checked_answer(ctx, view, [q.to_lists() for q in view.questions])
        return 1, 0

    monkeypatch.setattr(DuckAgent, "_run_analyzer_entry", analyze)
    duck = DuckAgent(audit_turn=False, show_changed_answers=False, run_config=RunConfig(on_agent_error="raise"))
    agent = MultiTakeDuck(takes=(duck, duck))
    second = agent.takes[1]
    assert isinstance(second, DuckAgent)
    second.use_dihedral_and_order_permutation = True
    submissions: list[Submission] = []

    def submit(answer: Submission, *, produced: Produced | None = None) -> None:
        submissions.append(answer)

    ctx, _ = build_llm_context("augmentation", submissions, random.Random(0), ullm.backend.DummyBackend(), submit)
    assert await agent.do_move(ctx, _view()) is None
    assert seen[0] == _view()
    assert set(seen[1].given) == set(_expected(ORDER, ORDER).given)
    assert set(seen[1].questions) == set(_expected(ORDER, ORDER).questions)
    assert seen[1].given != _expected(ORDER, ORDER).given
    assert seen[1].questions != _expected(ORDER, ORDER).questions
    assert submissions == [Submission(attempts=tuple((q, q) for q in _view().questions))]


@pytest.mark.parametrize("n_questions", [1, 2, 3])
@pytest.mark.parametrize("mandatory", [False, True])
async def test_tiered_layouts_exhaust_independently_and_restore_answers(
    monkeypatch: pytest.MonkeyPatch, n_questions: int, mandatory: bool
) -> None:
    monkeypatch.setattr(DuckAgent, "_ensure_repl", Mock(return_value=Mock(setup_task=AsyncMock())))
    original = _view()
    original = GameView(game_id=original.game_id, given=original.given, questions=original.questions[:n_questions])
    answers = tuple(answer for _, answer in original.given)[:n_questions]
    layouts: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []

    async def analyze(self: DuckAgent, ctx: LLMMoveContext, view: GameView, **_kwargs: Any) -> tuple[int, int]:
        # Identify the layout from model-visible grids, independently of the plan.
        candidates = [
            (
                transform,
                tuple((_oriented(q, transform), _oriented(a, transform)) for q, a in original.given),
                tuple(_oriented(q, transform) for q in original.questions),
            )
            for transform in range(8)
        ]
        matches = [(t, train, test) for t, train, test in candidates if set(train) == set(view.given)]
        assert len(matches) == 1
        transform, train, test = matches[0]
        train_order = tuple(train.index(pair) for pair in view.given)
        test_order = tuple(test.index(q) for q in view.questions)
        layouts.append((transform, train_order, test_order))
        submit = ctx.submit
        captured: list[tuple[Submission, Produced | None]] = []

        def capture(answer: Submission, *, produced: Produced | None = None) -> None:
            captured.append((answer, produced))
            submit(answer, produced=produced)

        ctx.submit = capture
        model_answers = [_oriented(answers[i], transform).to_lists() for i in test_order]
        if mandatory:
            self._absorb_checked_answer(ctx, view, model_answers)
        else:
            rows = [
                [{"state": "ok", "grid": answer}, {"state": "ok", "grid": q.to_lists()}]
                for answer, q in zip(model_answers, view.questions, strict=True)
            ]
            self._absorb_answer(
                ctx, view, {"bound": True, "rows": [rows[0], *[[{"state": "missing"}] * 2] * (n_questions - 1)]}
            )
            assert captured[0] == (
                Submission(
                    attempts=tuple(
                        (answers[i], q) if i == test_order[0] else (q, q) for i, q in enumerate(original.questions)
                    )
                ),
                tuple((i == test_order[0], i == test_order[0]) for i in range(n_questions)),
            )
            self._absorb_answer(ctx, view, {"bound": True, "rows": rows})
        assert captured[-1] == (
            Submission(attempts=tuple(zip(answers, answers if mandatory else original.questions, strict=True))),
            ((True, not mandatory),) * n_questions,
        )
        return 1, 0

    monkeypatch.setattr(DuckAgent, "_run_analyzer_entry", analyze)
    for seed in (0, 0, 1):
        duck = DuckAgent(audit_turn=False, show_changed_answers=False, run_config=RunConfig(on_agent_error="raise"))
        agent = TieredTakeDuck(takes=(duck,) * 10)
        # Even opting in the first take must leave its model-visible puzzle alone.
        for take in agent.takes:
            assert isinstance(take, DuckAgent)
            take.use_dihedral_and_order_permutation = True
        rng = random.Random(seed)
        before = rng.getstate()
        ctx, _ = build_llm_context(original.game_id, [], rng, ullm.backend.DummyBackend(), Mock())
        assert await agent.do_move(ctx, original) is None
        assert rng.getstate() == before
        played = layouts[-10:]
        assert played[0] == (0, ORDER, tuple(range(n_questions)))
        assert played[1][0] == 4
        for dimension, period in enumerate((8, math.factorial(3), math.factorial(n_questions))):
            for start in range(0, 10, period):
                cycle = [layout[dimension] for layout in played[start : start + period]]
                assert len(set(cycle)) == len(cycle)
    assert layouts[:10] == layouts[10:20]
    assert layouts[:10] != layouts[20:]


async def test_disabled_takes_keep_original_views_without_consuming_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DuckAgent, "_ensure_repl", Mock(return_value=Mock(setup_task=AsyncMock())))
    seen: list[GameView] = []

    async def analyze(self: DuckAgent, ctx: LLMMoveContext, view: GameView, **_kwargs: Any) -> tuple[int, int]:
        seen.append(view)
        self._absorb_checked_answer(ctx, view, [q.to_lists() for q in view.questions])
        return 1, 0

    monkeypatch.setattr(DuckAgent, "_run_analyzer_entry", analyze)
    for enabled in ((), tuple(range(1, 10)), (1, 3, 5, 7, 9)):
        duck = DuckAgent(audit_turn=False, show_changed_answers=False, run_config=RunConfig(on_agent_error="raise"))
        agent = TieredTakeDuck(takes=(duck,) * 10)
        for index in enabled:
            take = agent.takes[index]
            assert isinstance(take, DuckAgent)
            take.use_dihedral_and_order_permutation = True
        ctx, _ = build_llm_context("augmentation", [], random.Random(0), ullm.backend.DummyBackend(), Mock())
        assert await agent.do_move(ctx, _view()) is None
    assert seen[:10] == [_view()] * 10
    assert seen[20::2] == [_view()] * 5
    assert seen[21::2] == seen[11:16]


async def test_game_seeds_replay_both_permutations_without_advancing_play(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DuckAgent, "_ensure_repl", Mock(return_value=Mock(setup_task=AsyncMock())))
    seen: list[GameView] = []

    async def analyze(self: DuckAgent, ctx: LLMMoveContext, view: GameView, **_kwargs: Any) -> tuple[int, int]:
        seen.append(view)
        self._absorb_checked_answer(ctx, view, [q.to_lists() for q in view.questions])
        return 1, 0

    monkeypatch.setattr(DuckAgent, "_run_analyzer_entry", analyze)
    for seed in list(range(12)) * 2:
        agent = DuckAgent(use_dihedral_and_order_permutation=True, audit_turn=False, show_changed_answers=False)
        rng = random.Random(seed)
        before = rng.getstate()
        ctx, _ = build_llm_context("augmentation", [], rng, ullm.backend.DummyBackend(), Mock())
        await agent.do_move(ctx, _view())
        assert rng.getstate() == before
    assert seen[:12] == seen[12:]
    assert len({v.given for v in seen}) > 1
    assert len({v.questions for v in seen}) > 1
    for view in seen:
        assert set(view.given) == set(_expected(ORDER, ORDER).given)
        assert set(view.questions) == set(_expected(ORDER, ORDER).questions)


async def test_real_kernel_answers_return_in_original_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_orders(monkeypatch, (2, 0, 1), (1, 2, 0))
    expected = _expected((2, 0, 1), (1, 2, 0))
    code = (
        f"assert [g.tolist() for g in train_in] == {[q.to_lists() for q, _ in expected.given]!r}\n"
        f"assert [g.tolist() for g in train_out] == {[a.to_lists() for _, a in expected.given]!r}\n"
        f"assert [g.tolist() for g in test_in] == {[q.to_lists() for q in expected.questions]!r}\n"
        "for i, q in enumerate(test_in): test_out[i] = [q.copy(), train_out[0].copy()]"
    )
    backend = ullm.backends.stub.StubBackend.from_script(
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code})),
        ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="Done.")),
    )
    agent = DuckAgent(
        use_dihedral_and_order_permutation=True,
        audit_turn=False,
        show_changed_answers=False,
        reply_cap_tokens=64,
        exact_token_counts=False,
        repl=DuckPythonTool(isolate_network=False),
        llm_config=LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    )
    submissions: list[Submission] = []

    def submit(answer: Submission, *, produced: Produced | None = None) -> None:
        submissions.append(answer)

    ctx, _ = build_llm_context("augmentation", submissions, random.Random(0), backend, submit)
    try:
        assert await agent.do_move(ctx, _view()) is None
    finally:
        if agent.repl is not None:
            await agent.repl.clear_state()
    a = _view().questions[0]
    assert submissions == [Submission(attempts=tuple((q, a) for q in _view().questions))]
