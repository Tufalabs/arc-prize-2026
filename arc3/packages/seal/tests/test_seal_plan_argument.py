"""``plan_argument``: the python tool's optional ``plan``, kept on every turn the
call plays as ``history[i].plan`` (the competitor study's H42), asked for ahead
of ``code`` or after it.

Played in a real game through a scripted stub, so the plan travels the whole
way: the tool's arguments, the host's turns, the kernel's ``history``. Off, the
schema has no ``plan`` and a turn no attribute of that name.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest
import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from taaf.game_examples import ExampleGame

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from platypus.diagnostics import SpanNode, build_span_tree, game_run_ids, game_stream_path, read_events
from platypus.reference import game_run_dir
from seal import SealAgent
from seal.prompts import SealPrompts
from seal.repl import PLAN_MAX_CHARS

PLAN = "press UP once and watch the counter"


def _python(code: str, plan: str | None = None, *, plan_first: bool = False) -> ullm.conv.LLMResponse:
    arguments: dict[str, Any] = {"code": code}
    if plan is not None:
        arguments = {"plan": plan, **arguments} if plan_first else {**arguments, "plan": plan}
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[ullm.conv.ToolCallContent(tool_name="python", arguments=arguments)]
        ),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=5),
    )


class _Script(ullm.backends.stub.StubBackend):
    """Scripted replies; records each request's conversation text, class-level
    since the game plays on a copy of the backend the test built."""

    texts: ClassVar[list[str]] = []

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        """Record what the request carries, then pick the scripted reply."""
        type(self).texts.append(conversation.to_string())
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)


def _agent(**fields: Any) -> SealAgent:
    return SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=4096,
        max_tokens_per_game=200,
        max_tokens_per_level=None,
        use_world_model="none",
        multimodal_context="",
        inject_text_on_first_prompt="",
        inject_text_on_later_prompts="",
        **fields,
    )


async def _play(tmp_path: Path, replies: list[ullm.conv.LLMResponse], **fields: Any) -> list[str]:
    """One game on the scripted replies; the request texts it sent."""
    _Script.texts = []
    agent = _agent(**fields)
    agent.llm_config = LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=_Script(responses=replies, on_exhausted="repeat_last")),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)},
    )
    agent.run_config = RunConfig(rng_seed=1, process_mode="inprocess")
    benchmark = taaf.benchmark.Benchmark(
        games=[ExampleGame()], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=tmp_path
    )
    await asyncio.wait_for(benchmark.run(), timeout=120)
    return _Script.texts


async def test_every_turn_of_a_call_keeps_its_plan(tmp_path: Path) -> None:
    # ExampleGame's first level is won by the third UP, so all three calls
    # play on it and the third prints the level's turns before it wins.
    replies = [
        _python("submit_action([Action('UP')])", plan=PLAN),
        _python("submit_action([Action('UP')])"),
        _python("print('plans', [t.plan for t in history])\nsubmit_action([Action('UP')])", plan="win"),
    ]
    texts = await _play(tmp_path, replies, plan_argument="second")

    # The seed turn, the planned call's move, the call without one.
    assert any(f"plans [None, {PLAN!r}, None]" in text for text in texts)


async def test_off_the_turn_has_no_plan(tmp_path: Path) -> None:
    replies = [_python("submit_action([Action('UP')])", plan=PLAN), _python("print(history[-1].plan)")]
    texts = await _play(tmp_path, replies)

    assert any("has no attribute 'plan'" in text for text in texts)


def test_the_schema_offers_plan_only_when_on() -> None:
    off = _agent()._ensure_repl()  # pyright: ignore[reportPrivateUsage]

    assert "plan" not in off.parameters["properties"]
    for mode in ("first", "second"):
        on = _agent(plan_argument=mode)._ensure_repl()  # pyright: ignore[reportPrivateUsage]
        assert on.parameters["properties"]["plan"]["type"] == "string"
        assert on.parameters["required"] == ["code"], "plan is optional"


def test_first_puts_plan_ahead_of_code_and_asks_for_it_there() -> None:
    surface = SealPrompts()
    first = _agent(plan_argument="first")._ensure_repl()  # pyright: ignore[reportPrivateUsage]
    second = _agent(plan_argument="second")._ensure_repl()  # pyright: ignore[reportPrivateUsage]

    assert list(first.parameters["properties"]) == ["plan", "code"]
    assert list(second.parameters["properties"]) == ["code", "plan"]
    assert first.parameters["properties"]["plan"]["description"].endswith(surface.plan_param_first_line)
    assert second.parameters["properties"]["plan"]["description"] == surface.plan_param_description


def test_the_order_follows_the_config_on_a_reused_repl() -> None:
    agent = _agent(plan_argument="second")
    repl = agent._ensure_repl()  # pyright: ignore[reportPrivateUsage]
    agent.plan_argument = "first"

    assert agent._ensure_repl() is repl  # pyright: ignore[reportPrivateUsage]
    assert list(repl.parameters["properties"]) == ["plan", "code"]


def test_a_bool_is_refused() -> None:
    # The option was a bool before it took an order; a config written then
    # must fail rather than read True as some mode.
    with pytest.raises(ValueError, match="unknown plan_argument True"):
        _agent(plan_argument=True)


@pytest.mark.parametrize("fields", [{}, {"settled_delta": True}], ids=["bare_repl", "searching_view"])
async def test_a_call_with_only_a_plan_is_refused(tmp_path: Path, fields: dict[str, Any]) -> None:
    # Without the seal's guard, ullm's lone-argument remap would run the plan
    # as code. Both tool objects a turn can offer carry the guard.
    replies = [
        ullm.conv.LLMResponse(
            final_content=ullm.conv.ListContent(
                elements=[ullm.conv.ToolCallContent(tool_name="python", arguments={"plan": "print('ran')"})]
            ),
            stop_reason="stop",
            token_usage=ullm.backend.TokenUsage(prompt_tokens=50, completion_tokens=5),
        ),
        _python("print('next')"),
    ]
    texts = await _play(tmp_path, replies, plan_argument="first", **fields)

    assert any("'code' is a required property" in text for text in texts)
    assert not any("\nran" in text for text in texts)


async def test_a_long_plan_is_cut(tmp_path: Path) -> None:
    long = "x" * (PLAN_MAX_CHARS + 50)
    replies = [
        _python("submit_action([Action('UP')])", plan=long),
        _python("print('plan length', len(history[-1].plan))\nsubmit_action([Action('UP')])"),
    ]
    texts = await _play(tmp_path, replies, plan_argument="second")

    assert any(f"plan length {PLAN_MAX_CHARS}" in text for text in texts)


def _moves(job_dir: Path) -> list[dict[str, Any]]:
    """The one game's ``seal/moves`` spans' kv, oldest first."""

    def spans(node: SpanNode) -> list[SpanNode]:
        return [node, *(d for child in node.children for d in spans(child))]

    (run_id,) = game_run_ids(job_dir)
    (root,) = build_span_tree(read_events(game_stream_path(game_run_dir(job_dir, run_id))))
    return [span.kv for span in spans(root) if span.kind == "seal/moves"]


async def test_the_moves_span_says_whether_the_plan_came_first(tmp_path: Path) -> None:
    # The schema only hints at the order, so the span records the one the
    # call was written in. (A reference cannot pin this: the capture sorts a
    # call's arguments, and a replay hands them over code first.)
    replies = [
        _python("submit_action([Action('UP')])", plan=PLAN, plan_first=True),
        _python("submit_action([Action('UP')])", plan=PLAN),
        _python("submit_action([Action('UP')])"),
    ]
    await _play(tmp_path, replies, plan_argument="first")

    # The stub repeats its last reply past the first level; these are the
    # three that won it.
    assert [(kv["plan_chars"], kv["plan_first"]) for kv in _moves(tmp_path)[:3]] == [
        (len(PLAN), True),
        (len(PLAN), False),
        (None, None),
    ]
