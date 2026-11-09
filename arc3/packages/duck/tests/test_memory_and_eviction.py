"""Duck context eviction, persistent memory, action exposure, and tool rendering."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import taaf.game
import ullm.backend
import ullm.conv

from duck import DuckAgent, prompts
from duck.memory_tool import DuckMemoryTool
from duck.python_tool import DuckPythonTool
from duck.state import FramePayload, StepSummary, TimelineEntry
from platypus.contract import Observation
from platypus.llm import frame_image


def _block(user_chars: int, reply_chars: int, tag: str) -> list[ullm.conv.ConversationElement]:
    turn = ullm.conv.ResponseTurn(status="processed")
    turn.elements.append(
        ullm.conv.LLMResponse(
            final_content=ullm.conv.StringContent(text="r" * reply_chars), status="processed", stop_reason="stop"
        )
    )
    return [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=tag + "u" * user_chars)), turn]


async def test_eviction_drops_whole_oldest_blocks_and_keeps_last() -> None:
    agent = DuckAgent(
        journal=True,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        context_window_tokens=6000,
        evict_turns=2,
        multimodal_context="",
        compact_to_tokens=2048,
    )
    messages: list[ullm.conv.ConversationElement] = []
    for i in range(6):
        messages.extend(_block(2000, 1000, f"tag{i}-"))
    trimmed = await agent._trim_messages_for_context(list(messages))
    # Whole user-led blocks dropped from the oldest end, newest retained.
    kept_tags = [e.content.to_string()[:5] for e in trimmed if isinstance(e, ullm.conv.UserPrompt)]
    assert kept_tags == sorted(kept_tags)  # order preserved
    assert kept_tags[-1] == "tag5-"[:5]
    assert len(kept_tags) < 6  # something was evicted
    assert isinstance(trimmed[0], ullm.conv.UserPrompt)  # always starts at a user block
    # The last block is never evicted even under an impossible budget (the
    # system-prompt floor alone exceeds this window's budget).
    tiny = DuckAgent(
        journal=True,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        context_window_tokens=5000,
        multimodal_context="",
        compact_to_tokens=1024,
    )
    trimmed_tiny = await tiny._trim_messages_for_context(list(messages))
    assert sum(1 for e in trimmed_tiny if isinstance(e, ullm.conv.UserPrompt)) == 1


async def test_eviction_chunk_leaves_headroom() -> None:
    # evict_turns forces extra blocks out once eviction starts (KV-prefix
    # stability): with evict_turns=3, one over-budget trim drops >= 3 blocks.
    agent = DuckAgent(
        journal=True,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        context_window_tokens=20000,
        evict_turns=3,
        multimodal_context="",
        compact_to_tokens=8192,
    )
    messages: list[ullm.conv.ConversationElement] = []
    for i in range(8):
        messages.extend(_block(6000, 2000, f"tag{i}-"))
    trimmed = await agent._trim_messages_for_context(messages)
    dropped = 8 - sum(1 for e in trimmed if isinstance(e, ullm.conv.UserPrompt))
    assert dropped >= 3


def _level_frame(level: int) -> FramePayload:
    """A minimal staged frame: the journal reads only its level."""
    return FramePayload(grid_b64="", shape=[0, 0], step=0, level=level)


def test_an_image_carries_no_text_into_the_bound() -> None:
    # Images render as nothing in the bound payload — their engine-side
    # cost is charged per image, not per pixel of encoding.
    agent = DuckAgent(
        journal=True,
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    frame = taaf.game.Frame(data=np.arange(64 * 64, dtype=np.int8).reshape(64, 64) % 16)
    image = frame_image(frame, scale=4)
    assert agent._content_request_text(image) == ""
    assert agent._image_bound_tokens(image) > 0


def test_reasoning_counts_in_the_bound() -> None:
    agent = DuckAgent(
        journal=True,
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    turn = ullm.conv.ResponseTurn(status="processed")
    turn.elements.append(
        ullm.conv.LLMResponse(
            reasoning_content=ullm.conv.StringContent(text="THINKING-" * 100),
            final_content=ullm.conv.StringContent(text="short answer"),
            status="processed",
        )
    )
    with_reasoning = agent._bound_request_payload([turn])
    assert "THINKING-" in str(with_reasoning)


async def test_memory_is_a_native_host_tool() -> None:
    stored: list[str] = []

    def append_memory(text: str) -> str:
        stored.append(text)
        return text

    tool = DuckMemoryTool(appender=append_memory)
    tool.arguments = {"important": "cursor moves up; test right"}
    await tool.run()
    assert not tool.is_error
    assert "cursor moves up; test right" in tool.result.to_string()
    assert stored == ["cursor moves up; test right"]
    tool.rearm()
    tool.arguments = {}
    await tool.run()
    assert tool.is_error
    assert "'text' is a required property" in tool.result.to_string()


def _queue_obs() -> Observation:
    frame = taaf.game.Frame(data=np.zeros((64, 64), dtype=np.int8))
    state: Any = SimpleNamespace(
        levels_completed=0,
        available_actions=[1, 6, 7, 0],
        raw=SimpleNamespace(state=SimpleNamespace(name="NOT_FINISHED")),
        frame=frame,
        animation_frames=[frame],
    )
    return Observation(
        state=state, outcome="playing", level=1, index=0, caused_by=None, frame=frame, animation_frames=(frame,)
    )


def test_valid_actions_expose_undo_and_reset() -> None:
    agent = DuckAgent(journal=True, reply_cap_tokens=None, compaction_strategy="prune", expose_undo=True)
    obs = _queue_obs()

    assert agent._model_valid_actions(obs) == ["UP", "MOUSE", "UNDO", "RESET"]
    assert DuckAgent(
        journal=True, reply_cap_tokens=None, compaction_strategy="prune", expose_undo=False
    )._model_valid_actions(obs) == ["UP", "MOUSE", "RESET"]
    agent.current_valid_actions = agent._model_valid_actions(obs)
    assert "Valid actions right now: UP, MOUSE, UNDO, RESET." in agent._build_user_prompt_text()


async def test_user_prompt_is_dynamic_state_not_repeated_system_guidance() -> None:
    agent = DuckAgent(journal=True, reply_cap_tokens=None, compaction_strategy="prune", multimodal_context="")
    agent.current_valid_actions = ["UP", "UNDO", "RESET"]

    agent.last_step_summary = StepSummary(
        executed_count=2,
        executed_actions=["UP", "LEFT"],
        level=1,
        level_transition=False,
        run_complete=False,
        game_over=False,
    )
    prompt = agent._build_user_prompt_text()

    assert prompt == (
        "The code executed 2 actions in the previous sequence.\n"
        "Executed actions: UP, LEFT.\nYou are still on the same level.\n"
        "Current state: step 1, level 1.\nValid actions right now: UP, UNDO, RESET."
    )
    assert "Only tool:" not in prompt
    assert "end of world model" not in prompt

    # The one standing exception: the first move of a level opens with the
    # turn order. Every later turn in the level is state alone, above.
    agent.last_step_summary = dataclasses.replace(agent.last_step_summary, level_transition=True)
    opening_prompt = agent._build_user_prompt_text()
    assert opening_prompt.endswith(
        "\nThis turn:\n"
        "1. Orient on the newest frame: `current_frame.segment()` for structure.\n"
        "2. Decide the next move or short sequence — explore, or advance the level. "
        "Write a search if the right order is not obvious.\n"
        "3. Append anything new and durable with `memory`, before you act: the turn ends as soon as you do.\n"
        "4. Act with `submit_action([...])`."
    )
    agent.last_step_summary = dataclasses.replace(agent.last_step_summary, level_transition=False)

    initial_system = agent._system_prompt_text()
    assert "Persistent memory snapshot" in initial_system
    assert "(empty)" in initial_system
    agent.response_step = 4
    await agent._append_memory("player moves one tile; goal is green")
    assert agent._build_user_prompt_text() == prompt
    assert "player moves one tile; goal is green" not in agent._system_prompt_text()

    agent._refresh_memory_prompt_snapshot()
    refreshed_system = agent._system_prompt_text()
    assert "[#1 | response=4 | step=0 | level=1 | last_action=NONE]" in refreshed_system
    assert "player moves one tile; goal is green" in refreshed_system


async def test_context_compaction_refreshes_the_frozen_memory_snapshot() -> None:
    agent = DuckAgent(
        journal=True,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        context_window_tokens=6000,
        compact_to_tokens=2048,
        multimodal_context="",
    )
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text=agent._system_prompt_text())),
        elements=[],
    )
    agent._live_conversation = conversation
    assert "new route" not in conversation.system_prompt.content.to_string()
    await agent._append_memory("new route: go left around the wall")
    messages: list[ullm.conv.ConversationElement] = []
    for i in range(6):
        messages.extend(_block(2000, 1000, f"tag{i}-"))

    await agent._trim_messages_for_context(messages)

    assert agent._last_trim_fired
    assert "new route: go left around the wall" in agent._system_prompt_text()
    assert "new route: go left around the wall" in conversation.system_prompt.content.to_string()


async def test_memory_snapshot_excludes_entries_whose_tool_calls_are_retained() -> None:
    """Compaction carries only journal entries absent from immutable history."""
    agent = DuckAgent(journal=True, reply_cap_tokens=None, compaction_strategy="prune", multimodal_context="")
    for response, finding in enumerate(("old route", "middle route", "new route"), start=1):
        agent.response_step = response
        await agent._append_memory(finding)

    retained_turn = ullm.conv.ResponseTurn(
        elements=[
            ullm.conv.ToolResult(
                call_id="memory-2",
                tool_name="memory",
                content=ullm.conv.StringContent(
                    text="Persistent memory appended successfully [#2 | response=2 | step=0 | level=1 | last_action=NONE]."
                ),
            ),
            ullm.conv.ToolResult(
                call_id="memory-3",
                tool_name="memory",
                content=ullm.conv.StringContent(
                    text="Persistent memory appended successfully [#3 | response=3 | step=0 | level=1 | last_action=NONE]."
                ),
            ),
        ],
        status="processed",
    )

    agent._refresh_memory_prompt_snapshot([retained_turn])

    assert "old route" in (agent.memory_prompt_snapshot or "")
    assert "middle route" not in (agent.memory_prompt_snapshot or "")
    assert "new route" not in (agent.memory_prompt_snapshot or "")


def test_system_prompt_advertises_only_the_unified_runtime_surface() -> None:
    # Built without the file suite: this is about the unified python/memory
    # claim, so it asks for that surface rather than inheriting whichever
    # one happens to ship.
    prompt = DuckAgent(
        journal=True, reply_cap_tokens=None, compaction_strategy="prune", file_tools="none"
    )._system_prompt_text()
    for declaration in (
        "`frames: list[FrameView]`",
        "`animations: list[list[FrameView]]`",
        "`actions: list[str]`",
        "`level_transitions: list[bool]`",
        "`current_frame is frames[-1]`",
        "`last_action == actions[-1]`",
    ):
        assert declaration in prompt
    assert "exactly two native tools: `python` and `memory`" in prompt
    assert "one required `text` argument" in prompt
    assert "Never call or import it inside Python" in prompt
    assert "most recent 1,000 tokens" in prompt
    assert "`submit_action(actions)`" in prompt
    assert "`.segment()`" in prompt
    assert "`FrameView.array`" in prompt
    assert "2-D NumPy array of one-character strings" in prompt
    assert "`pixels: list[list[int]]` as `[row, col]` cells" in prompt
    assert "`pixel_count: int`" in prompt
    for legacy in (
        "`history`",
        "`transitions`",
        "`previous_frame`",
        "`last_transition`",
        "`.segmentation`",
        "`action(actions)`",
        "World model:",
        "Cross-level notes:",
    ):
        assert legacy not in prompt
    assert (
        "most recent 250 tokens"
        in DuckAgent(
            journal=True, reply_cap_tokens=None, compaction_strategy="prune", memory_context_tokens=250
        )._system_prompt_text()
    )


async def test_memory_journal_appends_timestamps_and_evicts_oldest_whole_entries() -> None:
    # Outside token-native mode the ruler is the nominal 3 chars/token
    # calibration.
    agent = DuckAgent(journal=True, reply_cap_tokens=None, compaction_strategy="prune", memory_context_tokens=90)
    agent.history_entries = [
        TimelineEntry(action="", frame=_level_frame(2), animation=[]),
        TimelineEntry(action="LEFT", frame=_level_frame(2), animation=[]),
    ]
    agent.action_count = 9
    agent.response_step = 12

    acknowledgement = await agent._append_memory("first finding " + "a" * 80)
    assert acknowledgement == (
        "Persistent memory appended successfully [#1 | response=12 | step=9 | level=2 | last_action=LEFT]."
    )
    assert "first finding" not in acknowledgement
    agent.response_step = 13
    await agent._append_memory("second finding " + "b" * 80)

    assert [entry.index for entry in agent.memory_entries] == [2]
    rendered = agent._render_memory()
    assert "first finding" not in rendered
    assert "second finding" in rendered
    assert "[#2 | response=13 | step=9 | level=2 | last_action=LEFT]" in rendered


async def test_memory_journal_rejects_one_oversized_entry_without_mutating_state() -> None:
    agent = DuckAgent(journal=True, reply_cap_tokens=None, compaction_strategy="prune", memory_context_tokens=20)
    with pytest.raises(ValueError, match="exceeding the 20-token memory limit"):
        await agent._append_memory("x" * 200)
    assert agent.memory_entries == []
    assert agent.memory_entry_count == 0


def test_completion_limit_nudge_is_a_new_message_and_preserves_generated_history() -> None:
    # The notice is off by default; this test exercises the notice path.
    agent = DuckAgent(journal=True, reply_cap_tokens=None, compaction_strategy="prune", completion_cutoff_notice=True)
    opening = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="original state prompt"))
    response = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text="partial route reasoning"),
        final_content=ullm.conv.StringContent(text="partial final text"),
        token_usage=ullm.backend.TokenUsage(prompt_tokens=100, completion_tokens=8000),
        stop_reason="max_tokens",
        status="processed",
    )
    turn = ullm.conv.ResponseTurn(elements=[response], status="processed")
    agent.messages = [opening, turn]
    before = (opening.content.to_string(), response.reasoning_content.to_string(), response.final_content.to_string())

    nudge = agent._followup_prompt()
    agent.messages.append(nudge)

    assert "You hit the 8,000-token limit, so your previous reasoning was cut off" in nudge.content.to_string()
    assert "Continue from where it stopped without restarting" in nudge.content.to_string()
    assert before == (
        opening.content.to_string(),
        response.reasoning_content.to_string(),
        response.final_content.to_string(),
    )
    assert agent.messages[:2] == [opening, turn]
    assert nudge.content.to_string() == prompts.DuckPrompts().followup_nudge(8000)
    # The cutoff nudge is the ordinary one behind its own explanation, not
    # a second instruction stacked on it.
    assert nudge.content.to_string() != prompts.FOLLOWUP_NUDGE
    assert nudge.content.to_string().endswith(prompts.FOLLOWUP_NUDGE)


async def test_tool_payload_truncation_and_planned_action_compaction() -> None:
    tool = DuckPythonTool(tool_output_tokens=64)  # 256-char floor applies
    try:
        rendered = tool._render_tool_payload(
            {"tool": "python", "stdout": "x" * 5000, "result": {"planned_actions": [{"action": "UP"}] * 40}},
            truncate_fields=("stdout",),
        )
    finally:
        await tool.clear_state()
    assert "truncated 4744 chars" in rendered  # 5000 - max(256, 64*4)
    assert "truncation_note" in rendered
    assert '"planned_action_count": 40' in rendered
    assert '"planned_actions"' not in rendered


async def test_kernel_payload_parse_marker_and_fallbacks() -> None:
    tool = DuckPythonTool(timeout=30.0)
    try:
        parsed = tool._parse_kernel_payload(
            'noise line\n__DUCK_PAYLOAD__{"stdout": "hi", "action_results": []}', tool_errored=False
        )
        assert parsed["stdout"] == "hi"
        parsed = tool._parse_kernel_payload("Execution timeout: interrupted", tool_errored=True)
        assert parsed["error"] == "Tool timed out after 30s"
        parsed = tool._parse_kernel_payload("kernel exploded somehow", tool_errored=True)
        assert parsed["error"] == "kernel exploded somehow"
    finally:
        await tool.clear_state()


class _TextCountingBackend:
    """A backend that tokenizes raw text exactly (4 chars per token here),
    standing in for vLLM's ``/tokenize``."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def text_token_count(self, text: str) -> int | None:
        self.calls.append(text)
        return len(text) // 4


async def test_memory_journal_budget_is_true_tokens_when_the_backend_tokenizes() -> None:
    """With a tokenizing backend the retention ruler is exact: the same
    entries that the character bound would evict are retained, because their
    true token cost fits the budget."""
    agent = DuckAgent(
        journal=True,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        memory_context_tokens=100,
        token_native_context=True,
    )
    agent._live_backend = _TextCountingBackend()  # type: ignore[assignment]
    agent.history_entries = [TimelineEntry(action="", frame=_level_frame(2), animation=[])]

    await agent._append_memory("first finding " + "a" * 80)
    await agent._append_memory("second finding " + "b" * 80)
    # Two rendered entries are ~300 chars = ~75 exact tokens: both fit a
    # 100-token budget that the 300-char bound alone would have evicted for.
    assert [entry.index for entry in agent.memory_entries] == [1, 2]

    await agent._append_memory("third finding " + "c" * 80)
    # ~112 exact tokens now exceed the budget: the oldest whole entry goes.
    assert [entry.index for entry in agent.memory_entries] == [2, 3]
    rendered = agent._render_memory()
    assert "first finding" not in rendered and "third finding" in rendered


class _FailingCountBackend:
    """``text_token_count`` raising -- a transport failure after retries."""

    async def text_token_count(self, text: str) -> int | None:
        raise RuntimeError("transport down")


class _NoneCountBackend:
    """``text_token_count`` answering ``None`` -- a failed ``/tokenize`` probe."""

    async def text_token_count(self, text: str) -> int | None:
        return None


async def test_memory_journal_append_fails_atomically_when_the_count_fails() -> None:
    """A failed count fails the append loudly and leaves the journal exactly
    as it was: no half-evicted state, and a retried call appends once."""
    agent = DuckAgent(
        journal=True,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        memory_context_tokens=100,
        token_native_context=True,
    )
    agent._live_backend = _TextCountingBackend()  # type: ignore[assignment]
    agent.history_entries = [TimelineEntry(action="", frame=_level_frame(2), animation=[])]
    await agent._append_memory("first finding " + "a" * 80)
    entries_before = list(agent.memory_entries)
    count_before = agent.memory_entry_count

    agent._live_backend = _FailingCountBackend()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="transport down"):
        await agent._append_memory("second finding " + "b" * 80)
    assert agent.memory_entries == entries_before
    assert agent.memory_entry_count == count_before

    agent._live_backend = _TextCountingBackend()  # type: ignore[assignment]
    await agent._append_memory("second finding " + "b" * 80)
    assert [entry.index for entry in agent.memory_entries] == [1, 2]


async def test_memory_journal_token_native_refuses_to_change_ruler() -> None:
    """A token-native backend answering ``None`` raises instead of quietly
    degrading the exact budget to the calibration ruler mid-run."""
    agent = DuckAgent(
        journal=True,
        reply_cap_tokens=None,
        compaction_strategy="prune",
        memory_context_tokens=100,
        token_native_context=True,
    )
    agent._live_backend = _NoneCountBackend()  # type: ignore[assignment]
    agent.history_entries = [TimelineEntry(action="", frame=_level_frame(2), animation=[])]
    with pytest.raises(RuntimeError, match="refusing to change ruler"):
        await agent._append_memory("finding one")
    assert agent.memory_entries == []
