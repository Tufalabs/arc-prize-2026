"""Seal feature-mode defaults, paired compatibility surfaces, and invariants."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import ullm.backend
import ullm.conv

from platypus import Observation
from seal import SealAgent, prompts
from seal.action_names import to_engine_action, to_model_action
from seal.repl import GameREPL


def _frame(fill: int, *, step: int) -> dict[str, Any]:
    grid = np.full((2, 2), fill, dtype=np.uint8)
    return {
        "grid_b64": base64.b64encode(grid.tobytes()).decode("ascii"),
        "shape": [2, 2],
        "step": step,
        "level": 1,
    }


def _state(
    *,
    turns: list[dict[str, Any]],
    available_actions: list[str],
) -> dict[str, Any]:
    return {
        "level": 1,
        "turns": turns,
        "available_actions": available_actions,
    }


def _ack_actions(normalized: list[dict[str, Any]]) -> dict[str, Any]:
    """Echo normalized actions through the host callback boundary."""
    return {"ack": normalized}


async def test_current_persistent_surface_preserves_inactive_user_globals() -> None:
    def get_state() -> dict[str, Any]:
        return _state(
            turns=[{"action": "", "outcome": "playing", "animation": [_frame(0, step=0)]}],
            available_actions=["UP"],
        )

    def execute(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {}

    tool = GameREPL(reset_before_each_call=False)
    tool.host_functions = {"get_state": get_state, "action": execute}
    try:
        first = (await tool._execute({"code": "keep = 123\nnp = np.zeros(2)"})).to_string()
        first_error = tool.is_error
        second = (await tool._execute({"code": "print(keep, type(np).__name__)"})).to_string()
        second_error = tool.is_error
    finally:
        await tool.clear_state()

    assert not first_error, first
    assert not second_error, second
    # `np` is the model's to rebind: the runtime refreshes its own names
    # (history, submit_action), not the model's.
    assert second == "123 ndarray", second


async def test_kernel_skips_malformed_turns_rather_than_failing_the_snippet() -> None:
    """A turn the host could not serialise is dropped, not raised on.

    The model is mid-analysis when this would fire, and a history one turn
    short is far less damaging than a snippet that dies before it reads any
    of it. An entry with no frames has nothing to show either way."""
    malformed: Any = [
        None,
        {"action": "UP", "outcome": "playing", "animation": [_frame(0, step=1)]},
        {"action": "DOWN", "outcome": "playing", "animation": []},
    ]

    def get_state() -> dict[str, Any]:
        return _state(turns=malformed, available_actions=["UP"])

    tool = GameREPL()
    tool.host_functions = {"get_state": get_state, "action": _ack_actions}
    try:
        text = (
            await tool._execute({"code": "print(len(history), history[-1].action.name, history[-1].frame.step)"})
        ).to_string()
    finally:
        await tool.clear_state()

    assert text == "1 UP 1", text


def test_cutoff_notice_off_adds_a_generic_nudge_without_rewriting_history() -> None:
    agent = SealAgent(reply_cap_tokens=None, compaction_strategy="prune", journal=True, completion_cutoff_notice=False)
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
    before = (
        opening.content.to_string(),
        response.reasoning_content.to_string(),
        response.final_content.to_string(),
    )

    nudge = agent._followup_prompt()
    text = nudge.content.to_string()
    agent.messages.append(nudge)

    assert text == prompts.FOLLOWUP_NUDGE
    assert "token limit" not in text and "cut off" not in text
    assert before == (
        opening.content.to_string(),
        response.reasoning_content.to_string(),
        response.final_content.to_string(),
    )
    assert agent.messages[0] is opening
    assert agent.messages[1] is turn
    assert turn.elements[0] is response


def test_action_mapping_and_host_validation() -> None:
    agent = SealAgent()
    playing: Any = SimpleNamespace(outcome="playing", available_actions=["ACTION7", "RESET"])
    lost: Any = SimpleNamespace(outcome="lost", available_actions=["ACTION7", "RESET"])

    # The translation stays complete and correct even though the seal never
    # offers UNDO: it is what recognises an engine ACTION7 in order to hide it.
    assert to_model_action("ACTION7") == "UNDO"
    assert to_engine_action("UNDO") == "ACTION7"
    assert to_model_action("RESET") == "RESET"
    assert to_engine_action("RESET") == "RESET"
    assert agent._model_valid_actions(playing) == ["RESET"]
    assert agent._model_valid_actions(lost) == ["RESET"]

    staged, error = agent._stage_actions([{"action": "RESET"}])
    assert error is None
    assert [action.name for action in staged] == ["RESET"]

    normalized, error = agent._validate_host_action_arguments(([], {}), {})
    assert normalized is None
    assert error == "submit_action() requires exactly one positional actions argument."
    normalized, error = agent._validate_host_action_arguments((["UP"],), {})
    assert normalized is None
    assert error == "submit_action() received an invalid normalized action batch."


def test_undo_is_never_offered_or_accepted() -> None:
    """UNDO is out of the seal's action space entirely: one more transition a
    world model would have to account for, and the only one that runs time
    backwards. The engine may still offer ACTION7; the seal hides it and
    refuses it by either name."""
    playing: Any = SimpleNamespace(outcome="playing", available_actions=["ACTION1", "ACTION7", "RESET"])

    no_undo = SealAgent()
    assert no_undo._model_valid_actions(playing) == ["UP", "RESET"]
    staged, error = no_undo._stage_actions([{"action": "UNDO"}])
    assert staged == []
    assert error == "Action 1: UNDO is disabled."
    staged, error = no_undo._stage_actions([{"action": "ACTION7"}])
    assert staged == []
    assert error == "Action 1: UNDO is disabled."
    staged, error = no_undo._stage_actions([{"action": "RESET"}])
    assert error is None
    assert [action.name for action in staged] == ["RESET"]

    hidden = SealAgent()
    only_hidden: Any = SimpleNamespace(outcome="playing", available_actions=["ACTION7"])
    hidden.current_valid_actions = hidden._model_valid_actions(only_hidden)
    hidden.valid_actions_observed = True
    assert "UNDO" not in hidden._system_prompt_text()
    # Hiding UNDO is a staging rule, not prompt text: the static prompt is untouched.
    assert hidden._system_prompt_text() == SealAgent()._system_prompt_text()


async def test_the_kernel_is_real_python_with_numpy_preloaded() -> None:
    """No allowlist: the standard library imports, numpy is preloaded as
    `np`, and so are the modules a board analysis reaches for first. The
    audit run counted 565 refused imports under the duck's allowlist; this
    is the inverse of that kernel's test."""

    def get_state() -> dict[str, Any]:
        return _state(
            turns=[{"action": "", "outcome": "playing", "animation": [_frame(0, step=0)]}],
            available_actions=["UP"],
        )

    def execute(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {}

    tool = GameREPL()
    tool.host_functions = {"get_state": get_state, "action": execute}
    code = (
        "import difflib, numpy, textwrap, time, importlib, sys\n"
        "print(np is numpy, os.getcwd() == os.path.realpath(os.getcwd()) or True, "
        "Counter('aab')['a'], bool(difflib and textwrap and time and importlib and sys))\n"
    )
    try:
        content = await tool._execute({"code": code})
        text = content.to_string()
    finally:
        await tool.clear_state()

    assert not tool.is_error, text
    assert "True True 2 True" in text, text


def test_the_system_prompt_advertises_that_surface() -> None:
    system_prompt = SealAgent()._system_prompt_text()
    assert "`numpy` is preloaded as `np`" in system_prompt
    assert "The whole standard library is importable" in system_prompt


def test_the_request_names_exactly_the_tools_the_agent_built() -> None:
    """The schema the context bound measures and the tools the agent hands
    the model are one list."""
    agent = SealAgent(journal=True)
    system_prompt = agent._system_prompt_text()
    payload = agent._bound_request_payload([])

    # The ORDER is the claim -- python first, the file suite in the order the
    # model is shown it, the journal last -- and it holds however the suite is
    # configured, so the membership comes from the agent rather than a literal.
    tool_names = [tool["function"]["name"] for tool in payload["tools"]]
    assert tool_names[0] == "python"
    assert tool_names[-1] == "memory"
    assert tool_names[1:-1] == list(agent._prompt_modes().file_tool_names)  # pyright: ignore[reportPrivateUsage]
    actual_tools = agent._turn_tools()
    assert [tool["function"] for tool in payload["tools"]] == [
        {"name": tool.name, "description": tool.description, "parameters": tool.parameters} for tool in actual_tools
    ]
    assert "Persistent memory snapshot" in system_prompt


def test_the_shipped_opening_is_in_the_register_it_is_meant_to_set() -> None:
    """The injected opening is not an arbitrary string: it is the register the
    seal scores in, and the classifier that established that is the thing that
    can still say so. Asserting the bytes would only copy the default; asking
    the classifier catches a reworded opening that no longer sets the register,
    which is the way this default can actually break.

    The blank is the other route to the same register and rides along here so
    the pair is stated in one place: it is OFF, and the tokens are what the
    shipped configuration uses.
    """
    import ullm.reasoning_register

    shipped = SealAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    assert shipped.zero_recurrent is False
    for served in (shipped.inject_text_on_first_prompt, shipped.inject_text_on_later_prompts):
        assert ullm.reasoning_register.opening_blanked_score(served) > 0.5
    # The every-call text is the opening's own first words rather than a
    # second string that happens to read the same: one register, written once.
    assert shipped.inject_text_on_first_prompt.startswith(shipped.inject_text_on_later_prompts)


#: `_build_user_prompt_element` reads the agent's own history, never its
#: argument, so the tests below have nothing meaningful to pass it.
_NO_OBS = cast(Observation, None)


def _agent_at_level(mode: str, level: int) -> SealAgent:
    """A seal with one history entry, on ``level``, in image mode ``mode``."""
    agent = SealAgent(multimodal_context=mode)
    grid = np.full((2, 2), 3, dtype=np.uint8)
    frame = SimpleNamespace(
        grid_b64=base64.b64encode(grid.tobytes()).decode("ascii"), shape=[2, 2], step=0, level=level
    )
    # A stand-in for the one attribute the image path reads.
    agent.history_entries = cast(Any, [SimpleNamespace(frame=frame)])
    return agent


def _carries_image(prompt: ullm.conv.UserPrompt) -> bool:
    content = prompt.content
    return isinstance(content, ullm.conv.ListContent) and any(
        isinstance(element, ullm.conv.ImageContent) for element in content.elements
    )


def test_level_start_images_ride_the_first_prompt_of_each_level_only() -> None:
    """The mode's whole claim. An image every turn is paid every turn and moves
    with the board, so each one lands past the cached prefix; one per level is
    paid once and then sits inside it. That only holds if later turns in the
    level carry no image, and if a new level starts one again."""
    agent = _agent_at_level("level_start", level=1)

    assert _carries_image(agent._build_user_prompt_element(_NO_OBS))  # noqa: SLF001
    assert not _carries_image(agent._build_user_prompt_element(_NO_OBS))  # noqa: SLF001

    agent.history_entries[-1].frame.level = 2
    assert _carries_image(agent._build_user_prompt_element(_NO_OBS))  # noqa: SLF001
    assert not _carries_image(agent._build_user_prompt_element(_NO_OBS))  # noqa: SLF001


def test_current_grid_images_ride_every_prompt() -> None:
    """The mode it is not: the contrast is the point of having both."""
    agent = _agent_at_level("current_grid", level=1)
    assert all(_carries_image(agent._build_user_prompt_element(_NO_OBS)) for _ in range(3))  # noqa: SLF001


def test_an_unknown_image_mode_is_refused_at_construction() -> None:
    """A misspelled mode would otherwise read as off, and an arm that meant to
    show images would quietly show none."""
    with pytest.raises(ValueError, match="unknown multimodal_context"):
        SealAgent(multimodal_context="every_level")


def test_without_a_world_model_the_prompt_keeps_the_ducks_guidance() -> None:
    """The prompt review cut these as lines a world model makes wrong or
    redundant; with none in use they are the duck's again."""
    lines = (
        "Re-ground on the newest frame after any score increase",
        "`WIN` means the whole game is solved.",
        "- IMPORTANT: Especially when the game",
        "Do not treat HUD-only changes as evidence",
        "DON'T DO THIS!",
    )
    with_model = SealAgent(use_world_model="optional")._system_prompt_text()  # pyright: ignore[reportPrivateUsage]
    without = SealAgent(use_world_model="none")._system_prompt_text()  # pyright: ignore[reportPrivateUsage]
    for line in lines:
        assert line in without and line not in with_model, line
