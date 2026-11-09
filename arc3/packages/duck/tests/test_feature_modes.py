"""Duck feature-mode defaults, paired compatibility surfaces, and invariants."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import ullm.backend
import ullm.conv

from duck import DuckAgent, prompts
from duck.action_names import to_engine_action, to_model_action
from duck.python_tool import DuckPythonTool
from platypus import Observation


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
    frame_api: str,
    timeline: list[dict[str, Any]],
    valid_actions: list[str],
    last_action_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "frame_api": frame_api,
        "timeline": timeline,
        "valid_actions": valid_actions,
        "last_action_result": last_action_result or {},
    }


def _ack_actions(normalized: list[dict[str, Any]]) -> dict[str, Any]:
    """Echo normalized actions through the host callback boundary."""
    return {"ack": normalized}


def test_unknown_frame_api_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown frame_api"):
        DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", frame_api="typo")  # type: ignore[arg-type]


def test_frame_toggle_changes_only_frame_interface_lines() -> None:
    """The numeric rung replaces how a board is described and nothing else —
    so the solving guidance around it cannot drift on one arm only."""
    current = (
        DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", multimodal_context="")
        ._system_prompt_text()
        .splitlines()
    )
    numeric = (
        DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", frame_api="numeric", multimodal_context="")
        ._system_prompt_text()
        .splitlines()
    )
    changed = (set(current) - set(numeric)) | (set(numeric) - set(current))

    assert len(changed) == 10
    assert all(any(term in line for term in ("FrameView", "boards are", "Color legend")) for line in changed)


async def test_the_import_surface_matches_what_the_prompt_advertises() -> None:
    """The allowlist, the preloaded ``np`` and the ``import submit_action``
    compatibility alias are one claim the system prompt makes and the kernel
    has to honour."""
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", multimodal_context="")
    system_prompt = agent._system_prompt_text()
    assert "`numpy` is preloaded as `np`" in system_prompt
    assert "difflib" in system_prompt

    def get_state() -> dict[str, Any]:
        return _state(
            frame_api="ergonomic",
            timeline=[{"action": "", "animation": [_frame(0, step=0)]}],
            valid_actions=["UP"],
        )

    def execute(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {}

    tool = DuckPythonTool()
    tool.host_functions = {"get_state": get_state, "action": execute}
    code = (
        "try:\n"
        "    np\n"
        "except Exception:\n"
        "    has_np = False\n"
        "else:\n"
        "    has_np = True\n"
        "try:\n"
        "    import numpy as imported_numpy\n"
        "except Exception:\n"
        "    has_numpy_import = False\n"
        "else:\n"
        "    has_numpy_import = imported_numpy is not None\n"
        "try:\n"
        "    import difflib\n"
        "    import textwrap\n"
        "except Exception:\n"
        "    has_extra_stdlib = False\n"
        "else:\n"
        "    has_extra_stdlib = bool(difflib and textwrap)\n"
        "action_import_error = ''\n"
        "try:\n"
        "    import submit_action as imported_submit_action\n"
        "except Exception as exc:\n"
        "    has_action_import = False\n"
        "    action_import_error = str(exc)\n"
        "else:\n"
        "    has_action_import = imported_submit_action is submit_action\n"
        "import math\n"
        "print(has_np, has_numpy_import, has_extra_stdlib, has_action_import, math.sqrt(4) == 2)\n"
        "print(action_import_error)\n"
    )

    try:
        content = await tool._execute({"code": code})
        payload = json.loads(content.to_string())
    finally:
        await tool.clear_state()

    assert payload.get("returncode") == 0, payload
    assert "True True True True True" in payload.get("stdout", ""), payload


async def test_current_persistent_surface_preserves_inactive_user_globals() -> None:
    def get_state() -> dict[str, Any]:
        return _state(
            frame_api="ergonomic",
            timeline=[{"action": "", "animation": [_frame(0, step=0)]}],
            valid_actions=["UP"],
        )

    def execute(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        del normalized
        return {}

    tool = DuckPythonTool(reset_before_each_call=False)
    tool.host_functions = {"get_state": get_state, "action": execute}
    try:
        first = json.loads((await tool._execute({"code": "history = 123\nnp = np.zeros(2)"})).to_string())
        second = json.loads(
            (
                await tool._execute(
                    {"code": "print(history, [name for name in dir() if name.startswith('_duck_')], type(np).__name__)"}
                )
            ).to_string()
        )
    finally:
        await tool.clear_state()

    assert first.get("returncode") == 0, first
    assert second.get("returncode") == 0, second
    assert second.get("stdout") == "123 [] ndarray\n"


async def test_kernel_skips_malformed_timeline_entries_without_losing_alignment() -> None:
    """A skipped seed entry cannot turn the first valid frame into a transition."""
    malformed_timeline: Any = [None, {"action": "UP", "animation": [_frame(0, step=1)]}]

    def get_state() -> dict[str, Any]:
        return _state(
            frame_api="ergonomic",
            timeline=malformed_timeline,
            valid_actions=["UP"],
        )

    tool = DuckPythonTool()
    tool.host_functions = {"get_state": get_state, "action": _ack_actions}
    try:
        payload = json.loads(
            (await tool._execute({"code": "print(len(frames), actions, level_transitions)"})).to_string()
        )
    finally:
        await tool.clear_state()

    assert payload.get("stdout") == "1 [] []\n", payload


async def test_kernel_requires_every_host_selected_mode() -> None:
    """Host/kernel configuration drift is a loud boundary error."""
    state = _state(frame_api="ergonomic", timeline=[], valid_actions=[])
    del state["frame_api"]
    tool = DuckPythonTool()
    tool.host_functions = {"get_state": lambda: state, "action": _ack_actions}
    try:
        payload = json.loads((await tool._execute({"code": "pass"})).to_string())
    finally:
        await tool.clear_state()

    assert "missing required mode 'frame_api'" in payload.get("error", "")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frame_api", "typo", "unknown frame_api"),
        ("file_tools", "yes", "non-boolean file_tools"),
    ],
)
async def test_kernel_rejects_unknown_host_selected_modes(field: str, value: object, message: str) -> None:
    """Unknown host modes cannot silently fall through to another surface."""
    state = _state(frame_api="ergonomic", timeline=[], valid_actions=[])
    state[field] = value
    tool = DuckPythonTool()
    tool.host_functions = {"get_state": lambda: state, "action": _ack_actions}
    try:
        payload = json.loads((await tool._execute({"code": "pass"})).to_string())
    finally:
        await tool.clear_state()

    assert message in payload.get("error", "")


def test_cutoff_notice_off_adds_a_generic_nudge_without_rewriting_history() -> None:
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", journal=True, completion_cutoff_notice=False)
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


@pytest.mark.parametrize("frame_api", ["ergonomic", "numeric"])
def test_every_frame_api_keeps_prompt_and_tool_schema_consistent(frame_api: str) -> None:
    """The system prompt, the tool description the model is handed, and the
    schema the context bound measures all describe one runtime."""
    agent = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
        journal=True,
        frame_api=frame_api,  # type: ignore[arg-type]
        multimodal_context="",
    )
    system_prompt = agent._system_prompt_text()
    tool_description = agent._ensure_repl().description
    payload = agent._bound_request_payload([])

    for rendered in (system_prompt, tool_description):
        assert "`submit_action(actions)`" in rendered
        assert "`frames`" in rendered
        assert ".segment()" in rendered
        assert ".segmentation" not in rendered

    # Whether colors are addressable is the whole of the numeric rung, and
    # the tool description has to make the same claim the prompt does.
    numeric = frame_api == "numeric"
    assert ("Colors are indices 0-15." in tool_description) is numeric
    assert ("Raw numeric colors are unavailable." in tool_description) is not numeric
    assert ("`array == 5` selects every cell of color 5" in system_prompt) is numeric
    assert ("Raw numeric color ids are unavailable." in system_prompt) is not numeric

    # The ORDER is the claim -- python first, the file suite in the order the
    # model is shown it, the journal last -- and it holds however the suite is
    # configured, so the membership comes from the agent rather than a literal.
    tool_names = [tool["function"]["name"] for tool in payload["tools"]]
    assert tool_names[0] == "python"
    assert tool_names[-1] == "memory"
    assert tool_names[1:-1] == list(agent._file_tool_names())  # pyright: ignore[reportPrivateUsage]
    actual_tools = agent._turn_tools()
    assert [tool["function"] for tool in payload["tools"]] == [
        {"name": tool.name, "description": tool.description, "parameters": tool.parameters} for tool in actual_tools
    ]
    assert "Persistent memory snapshot" in system_prompt


def test_default_action_mapping_and_host_validation() -> None:
    # The mapping under test needs UNDO offered, whatever ships.
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", expose_undo=True)
    playing: Any = SimpleNamespace(outcome="playing", available_actions=["ACTION7", "RESET"])
    lost: Any = SimpleNamespace(outcome="lost", available_actions=["ACTION7", "RESET"])

    assert to_model_action("ACTION7") == "UNDO"
    assert to_engine_action("UNDO") == "ACTION7"
    assert to_model_action("RESET") == "RESET"
    assert to_engine_action("RESET") == "RESET"
    assert agent._model_valid_actions(playing) == ["UNDO", "RESET"]
    assert agent._model_valid_actions(lost) == ["RESET"]

    staged, error = agent._stage_actions([{"action": "UNDO"}, {"action": "RESET"}])
    assert error is None
    assert [action.name for action, _display in staged] == ["ACTION7", "RESET"]

    normalized, error = agent._validate_host_action_arguments(([], {}), {})
    assert normalized is None
    assert error == "submit_action() requires exactly one positional actions argument."
    normalized, error = agent._validate_host_action_arguments((["UP"],), {})
    assert normalized is None
    assert error == "submit_action() received an invalid normalized action batch."


def test_undo_visibility_can_be_disabled() -> None:
    playing: Any = SimpleNamespace(outcome="playing", available_actions=["ACTION1", "ACTION7", "RESET"])

    no_undo = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", expose_undo=False)
    assert no_undo._model_valid_actions(playing) == ["UP", "RESET"]
    staged, error = no_undo._stage_actions([{"action": "UNDO"}])
    assert staged == []
    assert error == "Action 1: UNDO is disabled by expose_undo=False."
    staged, error = no_undo._stage_actions([{"action": "ACTION7"}])
    assert staged == []
    assert error == "Action 1: UNDO is disabled by expose_undo=False."
    staged, error = no_undo._stage_actions([{"action": "RESET"}])
    assert error is None
    assert [action.name for action, _display in staged] == ["RESET"]

    hidden = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", expose_undo=False, multimodal_context="")
    only_hidden: Any = SimpleNamespace(outcome="playing", available_actions=["ACTION7"])
    hidden.current_valid_actions = hidden._model_valid_actions(only_hidden)
    hidden.valid_actions_observed = True
    assert "Valid actions right now: none." in hidden._build_user_prompt_text()
    # The ablation is a staging rule, not prompt text: the static prompt is untouched.
    assert (
        hidden._system_prompt_text()
        == DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", multimodal_context="")._system_prompt_text()
    )


#: `_build_user_prompt_element` reads the agent's own history, never its
#: argument, so the tests below have nothing meaningful to pass it.
_NO_OBS = cast(Observation, None)


def _agent_at_level(mode: str, level: int) -> DuckAgent:
    """A duck with one history entry, on ``level``, in image mode ``mode``."""
    agent = DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", multimodal_context=mode)
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
        DuckAgent(reply_cap_tokens=None, compaction_strategy="prune", multimodal_context="every_level")


def _legend_and_images(agent: DuckAgent, level: int) -> tuple[int, bool]:
    """Images and colour-legend captions on one turn at ``level``."""
    agent.history_entries[-1].frame.level = level
    content = agent._build_user_prompt_element(_NO_OBS).content  # noqa: SLF001
    elements = content.elements if isinstance(content, ullm.conv.ListContent) else []
    images = sum(1 for element in elements if isinstance(element, ullm.conv.ImageContent))
    legend = any(
        isinstance(element, ullm.conv.StringContent) and "Color legend image" in element.text for element in elements
    )
    return images, legend


def test_under_level_start_the_legend_rides_exactly_when_an_image_does() -> None:
    """The pairing the sweep runs. Both are reference material for the level,
    both are sent once, and both are dropped together when the block holding
    them is evicted — so an arm never has a board picture whose colours are
    unexplained, nor a legend for a picture it cannot see."""
    agent = _agent_at_level("level_start", level=1)
    agent.color_name_legend = True

    for level in (1, 1, 1, 2, 2):
        images, legend = _legend_and_images(agent, level)
        assert legend == (images > 0), (level, images, legend)


def test_under_current_grid_the_board_outlives_its_legend() -> None:
    """The asymmetry between the two image modes, which is not a bug but is
    worth pinning: the board arrives every turn and the legend explaining its
    colours arrives once per level, so the two arms differ in how long a board
    goes unexplained as well as in how often it appears."""
    agent = _agent_at_level("current_grid", level=1)
    agent.color_name_legend = True

    assert _legend_and_images(agent, 1) == (2, True)
    assert _legend_and_images(agent, 1) == (1, False)
    assert _legend_and_images(agent, 2) == (2, True)


def test_the_colour_legend_is_not_gated_on_the_frame_api() -> None:
    """The whale's terms, and the reason this is a separate switch: a name ties
    a swatch to wording the prompt already uses, which an ergonomic board needs
    exactly as much as a numeric one."""
    agent = _agent_at_level("level_start", level=1)
    agent.color_name_legend = True
    assert agent.frame_api == "ergonomic"
    assert _legend_and_images(agent, 1)[1]


def test_the_shipped_opening_is_in_the_register_it_is_meant_to_set() -> None:
    """The injected opening is not an arbitrary string: it is the register the
    duck scores in, and the classifier that established that is the thing that
    can still say so. Asserting the bytes would only copy the default; asking
    the classifier catches a reworded opening that no longer sets the register,
    which is the way this default can actually break.

    The blank is the other route to the same register and rides along here so
    the pair is stated in one place: it is OFF, and the tokens are what the
    shipped configuration uses.
    """
    import ullm.reasoning_register

    shipped = DuckAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    assert shipped.zero_recurrent is False
    for served in (shipped.inject_text_on_first_prompt, shipped.inject_text_on_later_prompts):
        assert ullm.reasoning_register.opening_blanked_score(served) > 0.5
    # The every-call text is the opening's own first words rather than a
    # second string that happens to read the same: one register, written once.
    assert shipped.inject_text_on_first_prompt.startswith(shipped.inject_text_on_later_prompts)
