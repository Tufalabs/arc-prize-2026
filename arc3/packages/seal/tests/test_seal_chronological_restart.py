"""The order a summary's rebuilt history tells its story in (``SealAgent.chronological_restart``).

Off, it opens on the summary, then the prompt that started the current turn and the kept call; on, a
preamble says the game is under way and the summary follows, then the prompt and the kept call come, each
where it happened, then the summary and the closing message. The
kept call sits after the prompt when this turn made it and before it when an earlier turn did, either way;
with no call kept, the prompt alone. The live turn, emptied, ends the history in both.
"""

from __future__ import annotations

from typing import Any

import pytest
import ullm.backend
import ullm.conv

from seal import SealAgent

SUMMARY = "SUMMARY TEXT"


def _call(text: str) -> list[ullm.conv.ConversationElement]:
    """A response that called a tool, and the call's result."""
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[
                ullm.conv.StringContent(text=text),
                ullm.conv.ToolCallContent(call_id=text, tool_name="python", arguments={"code": "print(1)"}),
            ]
        )
    )
    result = ullm.conv.ToolResult(call_id=text, content=ullm.conv.StringContent(text=f"{text} RESULT"))
    return [response, result]


def _agent(*, chronological: bool, tail: str) -> SealAgent:
    agent = SealAgent(
        use_world_model="none",
        multimodal_context="",
        journal=False,
        add_files_and_repl_to_summary=False,
        compaction_strategy="summarize",
        summary_tail=tail,  # pyright: ignore[reportArgumentType]
        chronological_restart=chronological,
    )
    agent._live_params = ullm.backend.GenericSamplingParameters(temperature=0.0)  # pyright: ignore[reportPrivateUsage]
    return agent


def _kinds(agent: SealAgent, earlier: ullm.conv.UserPrompt, status: ullm.conv.UserPrompt) -> list[str]:
    """The rebuilt history as names: the summary, the closing message, the turn's prompt, a kept call by its
    text, and the live turn."""
    names: list[str] = []
    for element in agent.messages:
        text = element.to_string()
        if element is status:
            names.append("prompt")
        elif element is earlier:
            names.append("earlier prompt")
        elif isinstance(element, ullm.conv.UserPrompt):
            names.append("summary" if SUMMARY in text else "preamble" if "already under way" in text else "closing")
        elif element is agent.messages[-1]:
            names.append("live")
        else:
            names.append("call " + ("THIS" if "THIS TURN" in text else "EARLIER"))
    return names


async def _install(agent: SealAgent, *, this_turn: bool) -> list[str]:
    """Rebuild around a summary, the latest call made by this turn or by an earlier one."""
    earlier = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="EARLIER PROMPT"))
    status = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="STATUS"))
    old = ullm.conv.ResponseTurn(elements=_call("EARLIER TURN"), status="processed")
    live = ullm.conv.ResponseTurn(elements=_call("THIS TURN") if this_turn else [], status="intermediate")
    agent.messages = [earlier, old, status, live]
    agent._turn_prompt = status  # pyright: ignore[reportPrivateUsage]

    async def tokens(*_: Any) -> int:
        return 100

    agent._request_prompt_tokens = tokens  # type: ignore[method-assign]
    await agent._install_summary(SUMMARY, "", 0, record_pruned_round=lambda _round: None)  # pyright: ignore[reportPrivateUsage]
    return _kinds(agent, earlier, status)


@pytest.mark.parametrize(
    ("chronological", "this_turn", "tail", "expected"),
    [
        (False, True, "full", ["summary", "prompt", "call THIS", "closing", "live"]),
        (True, True, "full", ["preamble", "prompt", "call THIS", "summary", "closing", "live"]),
        (False, False, "full", ["summary", "call EARLIER", "prompt", "closing", "live"]),
        (True, False, "full", ["preamble", "call EARLIER", "prompt", "summary", "closing", "live"]),
        (False, True, "none", ["summary", "prompt", "closing", "live"]),
        (True, True, "none", ["preamble", "prompt", "summary", "closing", "live"]),
    ],
)
async def test_the_rebuilt_history_keeps_the_order_things_happened_in(
    chronological: bool, this_turn: bool, tail: str, expected: list[str]
) -> None:
    assert await _install(_agent(chronological=chronological, tail=tail), this_turn=this_turn) == expected
