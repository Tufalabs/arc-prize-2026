"""The prompt surface: what the model is told, and what it is told twice.

Wording is configuration here — every fragment is a field on `DuckPrompts` —
so these pin the claims the prompt makes rather than its exact sentences. The
exception is the names: the prompt tells the model which variables exist, and
a prompt naming something the kernel does not bind is a lie the model cannot
recover from.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any, cast

import numpy as np
import pytest
import ullm.backend
import ullm.conv
from duck import CheckMode, DuckAgent, kernel_code, prompts
from duck.kernel_code import TASK_NAMES
from duck.python_tool import DuckPythonTool
from duck.state import reading_from_payload

import taaf.game
import taaf.grid

#: The shipped features a mandatory check leaves nothing to act on: the audit
#: turn fires at a stop with an answer standing, which a passing mandatory check
#: never leaves, and the changed answers read assignments the mode ignores.
_MANDATORY_OFF: dict[str, Any] = {"audit_turn": False, "show_changed_answers": False}


def _playable(check: CheckMode, images: str = "task+show") -> dict[str, Any]:
    """The shipped features this arm can play, the rest turned off: nothing
    under a mandatory check (``_MANDATORY_OFF``), and no changed answers
    without images to show them as."""
    if check is CheckMode.MANDATORY:
        return dict(_MANDATORY_OFF)
    return {"show_changed_answers": images != "off"}


def _view(n_questions: int = 1, given: int = 2) -> taaf.game.GameView:
    rng = np.random.default_rng(2)

    def grid(rows: int, cols: int) -> taaf.grid.Grid:
        return taaf.grid.Grid(rng.integers(0, 10, size=(rows, cols), dtype=np.int8))

    pairs = tuple((grid(3, 4), grid(5, 9)) for _ in range(given))
    return taaf.game.GameView(game_id="p", given=pairs, questions=tuple(grid(7, 7) for _ in range(n_questions)))


def _texts(prompt: ullm.conv.UserPrompt) -> list[str]:
    content = prompt.content
    assert isinstance(content, ullm.conv.ListContent)
    return [e.text for e in content.elements if isinstance(e, ullm.conv.StringContent)]


def _images(prompt: ullm.conv.UserPrompt) -> list[ullm.conv.ImageContent]:
    content = prompt.content
    assert isinstance(content, ullm.conv.ListContent)
    return [e for e in content.elements if isinstance(e, ullm.conv.ImageContent)]


def _cut_off_turn() -> tuple[ullm.conv.UserPrompt, ullm.conv.ResponseTurn, ullm.conv.LLMResponse]:
    opening = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="original state prompt"))
    response = ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text="partial route reasoning"),
        final_content=ullm.conv.StringContent(text="partial final text"),
        token_usage=ullm.backend.TokenUsage(prompt_tokens=100, completion_tokens=8000),
        stop_reason="max_tokens",
        status="processed",
    )
    return opening, ullm.conv.ResponseTurn(elements=[response], status="processed"), response


def test_the_system_prompt_names_every_bound_variable() -> None:
    """The one thing here that is not wording: a name the prompt advertises
    and the kernel does not bind sends the model looking for something that
    is not there."""
    text = DuckAgent(
        reply_cap_tokens=None,
    )._system_prompt_text()
    for name in TASK_NAMES:
        assert name in text, name


def test_check_claims_follow_the_runtime_gate() -> None:
    """The compatibility arm must not advertise a name its kernel omits.
    The nudge claim is the legacy nudge's: the causal one names no function."""
    surface = prompts.DuckPrompts(nudge_mode="legacy")
    enabled = DuckAgent(reply_cap_tokens=None, check_mode=CheckMode.ENABLED)._prompt_modes()
    disabled = DuckAgent(reply_cap_tokens=None, check_mode=CheckMode.DISABLED)._prompt_modes()
    assert "check(fn)" in surface.system_prompt(enabled)
    assert "check(fn)" in surface.python_tool_description(enabled)
    assert "check(fn)" in surface.followup_nudge(enabled)
    assert "check(fn)" not in surface.system_prompt(disabled)
    assert "check(fn)" not in surface.python_tool_description(disabled)
    assert "check(fn)" not in surface.followup_nudge(disabled)


def test_the_kernel_only_arm_withholds_every_cell_and_says_where_the_puzzle_is() -> None:
    """Pictures off and digit rows off is a legal arm, not a broken one: the
    kernel binds the whole task either way, so what the prompt withholds is
    reachable. What it must not do is promise what it is not showing — the
    text-only wording claims every grid is spelled out, and rendering that
    here would send the model looking for rows that are not there.
    """
    view = _view()
    agent = DuckAgent(reply_cap_tokens=None, images="off", text_grids=False, show_changed_answers=False)
    prompt = agent._task_prompt(view)

    assert not _images(prompt), "no pictures in this arm"
    text = "\n".join(_texts(prompt))
    for grid in (*(g for pair in view.given for g in pair), *view.questions):
        rows = "".join(str(int(value)) for value in grid.data[0])
        assert rows not in text, "a cell value reached the prompt"

    # What survives is the inventory: which arrays exist and what shape they
    # are, which is what makes the kernel worth opening.
    assert "train_in[0] -> train_out[0]  (3x4 -> 5x9)" in text
    assert "test_in[0]  (7x7)" in text
    assert "ascii_grid" in agent._system_prompt_text()
    assert "shown as exact digit rows" not in text

    # And the arrays really are there, so the withholding costs one tool call
    # rather than the puzzle. The kernel binds the task whatever the prompt
    # shows, which is the whole reason this arm is legal.
    setup = kernel_code.task_setup_code(view)
    for grid in (*(g for pair in view.given for g in pair), *view.questions):
        assert ", ".join(str(int(value)) for value in grid.data[0]) in setup


def test_mandatory_check_is_the_only_prompted_submission_path() -> None:
    """Every mandatory prompt branch points to the checked function."""
    agent = DuckAgent(reply_cap_tokens=None, check_mode=CheckMode.MANDATORY, **_MANDATORY_OFF)
    surface = prompts.DuckPrompts(nudge_mode="legacy")
    modes = agent._prompt_modes()
    system = surface.system_prompt(modes)
    nudge = surface.followup_nudge(modes)

    assert "only way to answer" in system
    assert "editing `test_out` does not submit anything" in system
    assert "submitted immediately" in system
    assert "no confirmation phase" in system
    assert "output grids are sent" not in system
    assert surface.second_attempt_line not in system
    assert "assign your answers" not in nudge
    assert "passing `check(fn)`" in nudge
    assert "editing `test_out` has no effect" in surface.mandatory_answer_repair()

    image_system = DuckAgent(
        reply_cap_tokens=None, check_mode=CheckMode.MANDATORY, images="task+show+check", **_MANDATORY_OFF
    )._system_prompt_text()
    assert "grid picture" not in image_system


def test_check_mode_requires_the_enum() -> None:
    """A boolean is not a mode, and must not silently select one."""
    with pytest.raises(ValueError, match="check_mode must be a CheckMode"):
        DuckAgent(reply_cap_tokens=None, check_mode=cast(Any, True))
    # The tool is exported too, and takes the same field; it refuses the same
    # way rather than failing later on an attribute the value does not have.
    with pytest.raises(ValueError, match="check_mode must be a CheckMode"):
        DuckPythonTool(check_mode=cast(Any, True))


def test_image_mode_claims_follow_the_runtime_and_task_gates() -> None:
    """Each arm advertises only media the model will actually receive."""
    task = DuckAgent(reply_cap_tokens=None, images="task")
    show = DuckAgent(reply_cap_tokens=None, images="task+show")
    check_images = DuckAgent(reply_cap_tokens=None, check_mode=CheckMode.ENABLED, images="task+show+check")
    text_only = DuckAgent(reply_cap_tokens=None, images="off", text_grids=True, show_changed_answers=False)

    assert "show(grid" not in task._system_prompt_text()
    assert "show(grid" in show._system_prompt_text()
    assert "show(grid" in show._ensure_repl().description
    assert "grid picture" not in show._system_prompt_text()
    assert "grid picture" in check_images._system_prompt_text()
    assert "image" not in text_only._system_prompt_text().lower()
    assert "picture" not in text_only._system_prompt_text().lower()

    view = _view()
    assert _images(task._task_prompt(view))
    assert not _images(text_only._task_prompt(view))
    rows = "".join(str(int(value)) for value in view.questions[0].data[0])
    assert rows in "\n".join(_texts(text_only._task_prompt(view)))


def test_check_and_text_grids_compose_with_the_standard_platypus_surface() -> None:
    """The two opt-ins add their surfaces without enabling v0.6 media modes."""
    agent = DuckAgent(reply_cap_tokens=None, check_mode=CheckMode.ENABLED, text_grids=True, images="task")
    view = _view()
    system = agent._system_prompt_text()
    task = agent._task_prompt(view)

    assert "check(fn)" in system
    assert "show(grid" not in system
    assert "grid picture" not in system
    assert _images(task)
    rows = "".join(str(int(value)) for value in view.questions[0].data[0])
    assert rows in "\n".join(_texts(task))
    assert agent.images == "task"
    assert agent.compaction_strategy == "prune"


def test_the_system_prompt_states_the_grid_rules_from_taaf() -> None:
    """Derived, not copied: a prompt holding its own literals would go on
    telling the model the old numbers if TAAF's ever moved."""
    text = DuckAgent(
        reply_cap_tokens=None,
    )._system_prompt_text()
    assert str(taaf.grid.MAX_SIDE) in text
    assert f"0..{taaf.grid.N_COLORS - 1}" in text


def test_the_second_attempt_is_only_offered_when_there_is_one() -> None:
    surface = prompts.DuckPrompts()
    assert surface.second_attempt_line in surface.system_prompt(prompts.PromptModes())
    assert surface.second_attempt_line not in surface.system_prompt(prompts.PromptModes(n_attempts=1))


def test_the_task_prompt_opens_with_the_legend() -> None:
    """It is byte-identical every game, so leading with it extends the shared
    prefix and every game after the first reads it from the KV cache."""
    prompt = DuckAgent(image_legend=True)._task_prompt(_view())
    content = prompt.content
    assert isinstance(content, ullm.conv.ListContent)
    first_image = next(i for i, e in enumerate(content.elements) if isinstance(e, ullm.conv.ImageContent))
    assert first_image == 1, "the legend must come before anything task-specific"


def test_every_grid_is_captioned_with_its_names_and_size() -> None:
    """The caption is what lets the model connect a picture to a variable
    without measuring anything, and a size change is often the
    transformation's single most important fact."""
    texts = _texts(
        DuckAgent(
            reply_cap_tokens=None,
        )._task_prompt(_view(n_questions=2))
    )
    joined = "\n".join(texts)
    # Separate layout: every grid carries its own name and size, rather
    # than a pair sharing one caption and one arrow.
    assert "train_in[0]  (3x4)" in joined
    assert "train_out[0]  (5x9)" in joined
    assert "train_in[1]  (3x4)" in joined
    assert "train_out[1]  (5x9)" in joined
    assert "test_in[0]  (7x7)" in joined
    assert "test_in[1]  (7x7)" in joined


def test_sizes_are_always_rows_by_cols() -> None:
    """Stated once, up front, so the model never has to guess which number is
    which — and the captions must agree with the statement."""
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    assert "rows x cols" in agent._system_prompt_text()
    captions = "\n".join(_texts(agent._task_prompt(_view())))
    assert "train_in[0]  (3x4)" in captions, "a 3-row, 4-column input rendered the other way round"
    assert "train_out[0]  (5x9)" in captions


def test_one_image_per_pair_and_one_per_question_plus_the_legend() -> None:
    prompt = DuckAgent(image_layout="paired", image_legend=True)._task_prompt(_view(n_questions=2, given=3))
    assert len(_images(prompt)) == 3 + 2 + 1


def test_the_task_prompt_carries_no_text_grids() -> None:
    """With the digit rows off, images and the kernel are the whole channel:
    the grids are in the pictures approximately and in the kernel exactly,
    and spelling them out would both cost the budget and give the model a
    reason not to touch the REPL."""
    view = _view()
    joined = "\n".join(_texts(DuckAgent(reply_cap_tokens=None, text_grids=False)._task_prompt(view)))
    row = "".join(str(int(v)) for v in view.questions[0].data[0])
    assert row not in joined


def test_text_grid_gate_adds_exact_rows_without_removing_images() -> None:
    """The dual-presentation arm is independently sweepable from image mode."""
    view = _view()
    prompt = DuckAgent(reply_cap_tokens=None, text_grids=True)._task_prompt(view)
    rows = "".join(str(int(value)) for value in view.given[0][0].data[0])
    assert rows in "\n".join(_texts(prompt))
    assert _images(prompt), "text_grids is not an implicit images-off switch"


def test_agent_wires_scale_and_gridline_gates_into_task_images() -> None:
    """The source-sized lattice is opt-in; the target image remains smaller."""
    view = _view(given=1)
    compact = _images(DuckAgent(reply_cap_tokens=None, image_scale=4)._task_prompt(view))[1]
    source_style = _images(DuckAgent(reply_cap_tokens=None, image_scale=16, image_gridlines=True)._task_prompt(view))[1]
    assert source_style.data.shape[0] > compact.data.shape[0]
    assert source_style.data.shape[1] > compact.data.shape[1]


def test_task_image_layout_and_legend_are_independent_gates() -> None:
    """The compact pair composite and palette prefix remain available while
    a separate per-grid presentation can omit both."""
    view = _view(given=1)
    paired = DuckAgent(image_layout="paired", image_legend=True)._task_prompt(view)
    separate = DuckAgent(image_layout="separate", image_legend=True)._task_prompt(view)
    unlegended = DuckAgent(image_legend=False, image_layout="paired")._task_prompt(view)

    assert len(_images(paired)) == 3, "legend, one pair composite, one test input"
    assert len(_images(separate)) == 4, "legend, train input, train output, test input"
    assert len(_images(unlegended)) == 2
    assert "colour legend" not in DuckAgent(reply_cap_tokens=None, image_legend=False)._system_prompt_text().lower()
    assert "0 black" in DuckAgent(reply_cap_tokens=None, image_legend=False)._system_prompt_text().lower()

    dual = DuckAgent(
        reply_cap_tokens=None,
        image_layout="separate",
        image_legend=False,
        text_grids=True,
    )._task_prompt(view)
    assert isinstance(dual.content, ullm.conv.ListContent)
    train_input_text = dual.content.elements[1]
    train_input_image = dual.content.elements[2]
    train_output_text = dual.content.elements[3]
    train_output_image = dual.content.elements[4]
    assert isinstance(train_input_text, ullm.conv.StringContent)
    assert isinstance(train_input_image, ullm.conv.ImageContent)
    assert isinstance(train_output_text, ullm.conv.StringContent)
    assert isinstance(train_output_image, ullm.conv.ImageContent)
    assert "train_in[0]" in train_input_text.text and "\n" in train_input_text.text
    assert "train_out[0]" in train_output_text.text and "\n" in train_output_text.text


def test_media_gates_reject_invalid_values_at_the_prompt_runtime_seam() -> None:
    """A misspelled arm or unusable render limit fails before model work."""
    with pytest.raises(ValueError, match="images must be one of"):
        DuckAgent(reply_cap_tokens=None, images=cast(Any, "task+shows"))._system_prompt_text()
    with pytest.raises(ValueError, match="max_shows must be non-negative"):
        DuckAgent(reply_cap_tokens=None, max_shows=-1)._system_prompt_text()
    with pytest.raises(ValueError, match="image_scale must be at least 1"):
        DuckAgent(reply_cap_tokens=None, image_scale=0)._task_prompt(_view())
    with pytest.raises(ValueError, match="image_layout must be"):
        DuckAgent(reply_cap_tokens=None, image_layout=cast(Any, "tiles"))._task_prompt(_view())


def test_a_re_entry_gets_the_answer_status_not_the_pictures_again() -> None:
    """The kernel holds the puzzle, so a fresh state prompt names what is
    still wrong with the answer rather than re-sending every image."""
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    view = _view(n_questions=2)
    first = agent._build_state_prompt(view)
    assert _images(first), "the first entry carries the puzzle"
    agent.messages.append(first)
    before_any_call = agent._build_state_prompt(view)
    assert "has not been written to yet" in before_any_call.content.to_string()
    # As the kernel would report it after a call that answered only one.
    agent.reading = reading_from_payload(
        {"bound": True, "rows": [[{"state": "ok", "grid": [[1, 2]]}, {"state": "missing"}], [{"state": "missing"}] * 2]}
    )
    again = agent._build_state_prompt(view)
    assert isinstance(again.content, ullm.conv.StringContent)
    assert "test_out[1] is still unset" in again.content.to_string()
    assert "test_out[0]" not in again.content.to_string(), "an answered question is not nagged about"


def test_completion_limit_nudge_is_a_new_message_and_preserves_generated_history() -> None:
    # The notice is off by default; this test exercises the notice path, in
    # the legacy nudge's wording.
    legacy = prompts.DuckPrompts(nudge_mode="legacy")
    agent = DuckAgent(reply_cap_tokens=None, completion_cutoff_notice=True, prompts=legacy)
    opening, turn, response = _cut_off_turn()
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
    assert nudge.content.to_string() == legacy.followup_nudge(agent._prompt_modes(), 8000)
    assert legacy.followup_nudge(prompts.PromptModes()) not in nudge.content.to_string()


def test_cutoff_notice_off_adds_a_generic_nudge_without_rewriting_history() -> None:
    legacy = prompts.DuckPrompts(nudge_mode="legacy")
    agent = DuckAgent(reply_cap_tokens=None, completion_cutoff_notice=False, prompts=legacy)
    opening, turn, response = _cut_off_turn()
    agent.messages = [opening, turn]
    before = (opening.content.to_string(), response.reasoning_content.to_string(), response.final_content.to_string())

    nudge = agent._followup_prompt()
    text = nudge.content.to_string()
    agent.messages.append(nudge)

    assert text == legacy.followup_nudge(agent._prompt_modes())
    assert "token limit" not in text and "cut off" not in text
    assert before == (
        opening.content.to_string(),
        response.reasoning_content.to_string(),
        response.final_content.to_string(),
    )
    assert agent.messages[0] is opening
    assert agent.messages[1] is turn
    assert turn.elements[0] is response


def test_the_nudge_carries_the_tool_call_format_guidance_under_its_flag() -> None:
    """Upstream's, kept verbatim: a Qwen failure mode where the model writes
    the tool-call markup inside prose or a markdown fence. Off in the shipped
    nudge (sweep 7), reachable under ``nudge_format_guidance``."""
    guidance = prompts.DuckPrompts().tool_call_format_guidance
    assert guidance not in prompts.FOLLOWUP_NUDGE
    assert guidance in prompts.DuckPrompts(nudge_format_guidance=True).followup_nudge(prompts.PromptModes())


def test_the_repair_prompt_speaks_the_model_s_own_indices() -> None:
    """So the fix is an assignment it can make without working out which
    question is meant."""
    text = prompts.DuckPrompts().answer_repair(["test_out[1] is still unset"])
    assert "test_out[1]" in text


def test_wording_is_configuration() -> None:
    """An experiment overrides one sentence without subclassing anything, and
    the override must reach wherever that sentence is composed."""
    custom = prompts.DuckPrompts(system_prompt_opening="SOLVE THE PUZZLE.")
    modes = prompts.PromptModes()
    assert "SOLVE THE PUZZLE." in custom.system_prompt(modes)
    assert prompts.DuckPrompts().system_prompt_opening not in custom.system_prompt(modes)


def test_legacy_wording_overrides_still_configure_the_default_surface() -> None:
    """Established no-check field names keep their pre-gate meanings."""
    custom = prompts.DuckPrompts(
        checking_line="LEGACY CHECKING LINE\n",
        python_tool_description_text="LEGACY TOOL DESCRIPTION",
        answer_nudge_guidance="LEGACY NUDGE ",
        nudge_mode="legacy",
    )
    modes = prompts.PromptModes()
    assert "LEGACY CHECKING LINE" in custom.system_prompt(modes)
    assert custom.python_tool_description(modes) == "LEGACY TOOL DESCRIPTION"
    assert "LEGACY NUDGE" in custom.followup_nudge(modes)


def test_check_wording_overrides_are_separate_from_the_legacy_surface() -> None:
    """The enabled oracle selects only its explicitly named wording fields."""
    custom = prompts.DuckPrompts(
        check_checking_line="ORACLE CHECKING LINE\n",
        check_python_tool_description_text="ORACLE TOOL DESCRIPTION",
        check_answer_nudge_guidance="ORACLE NUDGE ",
        nudge_mode="legacy",
    )
    modes = prompts.PromptModes(check_mode=CheckMode.ENABLED)
    assert "ORACLE CHECKING LINE" in custom.system_prompt(modes)
    assert custom.python_tool_description(modes) == "ORACLE TOOL DESCRIPTION"
    assert "ORACLE NUDGE" in custom.followup_nudge(modes)


def test_the_agent_stamps_its_own_limits_into_the_prompt() -> None:
    """The two must not be settable apart: a prompt naming a timeout the
    kernel does not enforce is worse than naming none."""
    agent = DuckAgent(reply_cap_tokens=None, python_timeout_s=11, tool_output_tokens=222)
    text = agent._system_prompt_text()
    assert "11 seconds" in text
    assert "222 tokens" in text


def test_the_prompt_surface_has_no_field_that_could_contradict_the_agent() -> None:
    """The hazard this shape exists to remove is the NAME, not the value: a
    mirror of an agent field reads exactly like configuration that works, so
    a sweep arm setting one ran as a silent no-op with nothing to notice. The
    numbers arrive as PromptModes now, and there is no field left to set."""
    settable = {f.name for f in dataclasses.fields(prompts.DuckPrompts)}
    enforced_elsewhere = {"python_timeout_s", "tool_output_tokens", "n_attempts", "max_side", "n_colors"}
    assert settable.isdisjoint(enforced_elsewhere), sorted(settable & enforced_elsewhere)
    # The same hazard for the flags: none of them is a mirror of something the
    # agent enforces or a mode it decides.
    agent_fields = {f.name for f in dataclasses.fields(DuckAgent)}
    modes = {f.name for f in dataclasses.fields(prompts.PromptModes)}
    assert settable.isdisjoint(agent_fields), sorted(settable & agent_fields)
    assert settable.isdisjoint(modes), sorted(settable & modes)


def test_an_override_cannot_drop_the_marker_the_agent_scans_for() -> None:
    """The note is rewritten in place by finding its own prefix, so the
    prefix must survive any override of the wording — a note without it
    STACKS instead of replacing, precisely when context pressure is highest.
    The marker is prepended by the surface, and the surface that writes it
    is the one that reads it."""
    swept = prompts.DuckPrompts(rounds_pruned_note_template="{n} rounds gone.")
    note = swept.rounds_pruned_note(3)
    assert note.endswith("3 rounds gone.")
    assert swept.is_rounds_pruned_note(note)


# --- the configurable surface --------------------------------------------------
#
# Flags must compose with the modes an experiment already varies, while
# describing only the tools and answer channel that mode provides.

_DEFAULT_MODES = DuckAgent(
    reply_cap_tokens=None,
)._prompt_modes()


def _system(surface: prompts.DuckPrompts) -> str:
    return surface.system_prompt(_DEFAULT_MODES)


@pytest.mark.parametrize("check", tuple(CheckMode))
@pytest.mark.parametrize("images,text", [("off", False), ("off", True), ("task", False), ("task+show", True)])
def test_combined_flags_respect_the_tools_and_answer_channel(check: CheckMode, images: str, text: bool) -> None:
    surface = prompts.DuckPrompts(
        investigation_lines=False,
        work_policy="candidate_first",
        generalization_audit_line=True,
        second_attempt_mode="audit",
        assign_early_line=True,
        multi_test_line=True,
        no_hand_arithmetic_line=True,
        test_out_structure_explicit=True,
        nudge_mode="causal",
        nudge_format_guidance=False,
    )
    agent = DuckAgent(
        reply_cap_tokens=None,
        prompts=surface,
        check_mode=check,
        images=images,  # type: ignore[arg-type]
        text_grids=text,
        completion_cutoff_notice=True,
        **_playable(check, images),
    )  # type: ignore[arg-type]
    modes = agent._prompt_modes()
    system = agent._system_prompt_text()
    cut = surface.followup_nudge(modes, 8_000)
    stopped = surface.followup_nudge(modes, problems=["test_out[0][0] is not set"])
    assert "How to work:" in system
    assert ("not to count, index or diff cells by hand" in system) == text
    assert ("`check(fn)`" in system) == check.enabled
    assert ("`check(fn)`" in cut) == check.enabled
    if check is CheckMode.MANDATORY:
        for absent in (
            "never copy attempt 0",
            "generalization audit",
            "an unset question scores nothing",
            "never append",
        ):
            assert absent not in system
        assert "test_out[0][0] is not set" not in stopped
        assert "`check(fn)`" in stopped and "Only a passing check submits" in stopped
    else:
        assert "never copy attempt 0" in system
        assert "generalization audit" in system
        assert "test_out[0][0] is not set" in stopped
        assert "never append" in system


def test_prompt_flags_use_the_existing_sweeper_without_state_leaking_between_arms() -> None:
    from sweeper import Choice, Design, Field, Knob, LLMSweepAgent

    design = Design(
        arm={
            "work_policy": Knob(Choice(("none", "candidate_first")), Field("inner.prompts.work_policy"), "none"),
            "check_mode": Knob(Choice(tuple(CheckMode)), Field("inner.check_mode"), CheckMode.ENABLED),
            "check_and_submit": Knob(Choice((False, True)), Field("inner.check_and_submit"), False),
        }
    )
    inner = DuckAgent(prompts=prompts.DuckPrompts(), **_MANDATORY_OFF)
    sweep = LLMSweepAgent(inner=inner, design=design, announce=False)
    restored: LLMSweepAgent = pickle.loads(pickle.dumps(sweep))
    restored_inner = cast(DuckAgent, restored.inner)
    for check in CheckMode:
        restored._configure("arm", {"work_policy": "candidate_first", "check_mode": check, "check_and_submit": True})
        assert "How to work:" in restored_inner._system_prompt_text()
        assert ("automatically saves" in restored_inner._system_prompt_text()) == (check is CheckMode.ENABLED)
        restored._configure("arm", {"work_policy": "none", "check_mode": check, "check_and_submit": False})
        assert (
            restored_inner._system_prompt_text()
            == DuckAgent(reply_cap_tokens=None, check_mode=check, **_MANDATORY_OFF)._system_prompt_text()
        )
    assert "How to work:" not in inner._system_prompt_text()


@pytest.mark.parametrize(
    "surface",
    [
        prompts.DuckPrompts(nudge_mode="legacy"),
        prompts.DuckPrompts(assign_early_line=True, generalization_audit_line=True, nudge_mode="legacy"),
    ],
)
@pytest.mark.parametrize("check", list(CheckMode))
def test_check_and_submit_prompts_describe_the_active_answer_path(
    surface: prompts.DuckPrompts, check: CheckMode
) -> None:
    agent = DuckAgent(
        reply_cap_tokens=None, check_mode=check, check_and_submit=True, prompts=surface, **_playable(check)
    )
    ordinary = DuckAgent(reply_cap_tokens=None, check_mode=check, prompts=surface, **_playable(check))
    system = agent._system_prompt_text()
    tool = agent._ensure_repl()
    nudge = agent._followup_prompt().content.to_string()
    if check is not CheckMode.ENABLED:
        assert system == ordinary._system_prompt_text()
        assert tool.description == ordinary._ensure_repl().description
        assert nudge == ordinary._followup_prompt().content.to_string()
        return
    for claim in (
        "automatically replace `test_out[i][0]`",
        "other attempt slots are preserved",
        "leaves the saved answers unchanged",
        "This does not end the task",
        "each later passing check overwrites the first slots",
        "You may also assign or edit `test_out` directly",
    ):
        assert claim in system
    assert "does not change `test_out`" not in system
    assert "That is the only way to answer" not in system
    assert "In the same call that" not in system
    assert "re-assign" not in system
    assert "automatically saves" in tool.description
    assert "automatically saves" in nudge
    if surface.generalization_audit_line:
        assert "re-run `check(fn)` to update `test_out[i][0]` automatically" in system


@pytest.mark.parametrize("enabled", [False, True])
def test_half_brevity_replaces_the_full_instruction_only_when_enabled(enabled: bool) -> None:
    surface = prompts.DuckPrompts(brevity_line=enabled, brevity="half")
    agent = DuckAgent(prompts=surface)
    text = agent._system_prompt_text()
    assert (surface.half_brevity_line in text) == enabled
    for absent in ("Keep your reasoning proportionate", "Long deliberation is worth it", "Every turn you spend"):
        assert absent not in text
    surface.brevity = "full"
    assert (surface.user_prompt_brevity_line in agent._system_prompt_text()) == enabled


@pytest.mark.parametrize("check", tuple(CheckMode))
@pytest.mark.parametrize("allowance", [None, 12_345])
@pytest.mark.parametrize("investigation", [False, True])
def test_cost_model_quotes_the_enforced_budget_and_submission_channel(
    check: CheckMode, allowance: int | None, investigation: bool
) -> None:
    surface = prompts.DuckPrompts(
        cost_model="explicit", investigation_lines=investigation, work_policy="candidate_first"
    )
    agent = DuckAgent(
        reply_cap_tokens=None, prompts=surface, check_mode=check, max_tokens_per_game=allowance, **_playable(check)
    )
    text = agent._system_prompt_text()
    assert text.count("What costs what:") == 1
    assert text.index("How to work:") < text.index("Tool limits and session lifetime:") < text.index("What costs what:")
    if allowance is None:
        assert "There is no per-puzzle generated-token cap" in text
        assert "fixed budget" not in text
    else:
        assert "fixed budget of about 12,345 generated tokens" in text
    assert "Tool output occupies context but is not charged" in text
    if check is CheckMode.MANDATORY:
        assert "Nothing is submitted unless a `check(fn)` passes" in text
        assert "Saved answers in `test_out`" not in text
    else:
        assert "Saved answers in `test_out` are scored" in text
    surface.cost_model = "none"
    assert "What costs what:" not in agent._system_prompt_text()


def test_the_investigation_lines_are_a_deletion_and_the_work_policy_an_addition() -> None:
    """Deletion and addition are independent so a sweep can measure each."""
    deleted = _system(prompts.DuckPrompts(investigation_lines=False))
    assert "Investigate until your code has a clear probe or plan" not in deleted
    assert "Build up the tools you want" not in deleted
    assert "do not build a general-purpose framework" in deleted
    assert "How to work:" not in deleted
    added = _system(prompts.DuckPrompts(work_policy="candidate_first"))
    assert "How to work:" in added
    assert "After at most two inspection-only calls" in added
    assert "run `check(fn)`" in added
    assert "Investigate until your code has a clear probe or plan" in added
    # The policy block precedes the session limits it operates within.
    both = _system(prompts.DuckPrompts(investigation_lines=False, work_policy="candidate_first"))
    assert both.index("How to work:") < both.index("Tool limits and session lifetime:")


def test_the_audit_and_the_second_attempt_modes_follow_their_flags() -> None:
    audited = _system(prompts.DuckPrompts(generalization_audit_line=True, second_attempt_mode="audit"))
    assert "generalization audit" in audited
    assert "never copy attempt 0" in audited
    assert "a bad one costs nothing" not in audited
    # The audit line precedes the slot that refers to it.
    assert audited.index("generalization audit") < audited.index("never copy attempt 0")
    # Either flag alone does only its own thing (the slot alone is the slot
    # with the audit line off, as it ships).
    audit_only = _system(prompts.DuckPrompts(generalization_audit_line=True))
    assert "generalization audit" in audit_only and "a bad one costs nothing" in audit_only
    slot_only = _system(prompts.DuckPrompts(generalization_audit_line=False, second_attempt_mode="audit"))
    assert "never copy attempt 0" in slot_only and "generalization audit" not in slot_only
    # ``none`` deletes the slot from the prompt altogether.
    silent = _system(prompts.DuckPrompts(second_attempt_mode="none"))
    assert "test_out[i][1]" not in silent and "a bad one costs nothing" not in silent


def test_assign_early_and_multi_test_are_separate_additions() -> None:
    early = _system(prompts.DuckPrompts(assign_early_line=True))
    assert "In the same call that `check(fn)` passes" in early
    assert "an unset question scores nothing" not in early
    multi = _system(prompts.DuckPrompts(multi_test_line=True))
    assert "an unset question scores nothing" in multi
    assert "In the same call that `check(fn)` passes" not in multi


def test_no_hand_arithmetic_swaps_the_digit_rows_line() -> None:
    rows = DuckAgent(reply_cap_tokens=None, text_grids=True)._prompt_modes()
    text = prompts.DuckPrompts(no_hand_arithmetic_line=True).system_prompt(rows)
    assert "not to count, index or diff cells by hand" in text
    assert "use the pictures for global structure" not in text
    # The row-index qualifier still follows the line it qualifies.
    assert text.index("by hand") < text.index("the index is a label, not a cell")


@pytest.mark.parametrize("row_numbers", [False, True])
def test_no_hand_arithmetic_without_digit_rows_rides_the_ascii_grid_line(row_numbers: bool) -> None:
    """Pictures alone: the caveat is about the rows the MODEL prints, so it
    follows the runtime line that describes that render -- and the digit-rows
    wording, which would claim rows this prompt does not show, stays away."""
    agent = DuckAgent(
        prompts=prompts.DuckPrompts(no_hand_arithmetic_line=True),
        text_grids=False,
        ascii_row_numbers=row_numbers,
    )
    text = agent._system_prompt_text()
    assert "Grids you print with `ascii_grid` are for orienting yourself" in text
    assert "The task prompt also spells every grid out as digit rows" not in text
    render = "Each row is prefixed with its" if row_numbers else "rows on separate lines"
    assert text.index(render) < text.index("Grids you print")


def test_no_hand_arithmetic_says_nothing_twice() -> None:
    """One caveat per render, never both: the two wordings disagree about
    where the rows come from, and a prompt carrying both is wrong either way."""
    for text_grids in (False, True):
        text = DuckAgent(
            prompts=prompts.DuckPrompts(no_hand_arithmetic_line=True), text_grids=text_grids
        )._system_prompt_text()
        assert text.count("hand-transcribing rows is where miscounts come from") == 1


def test_explicit_test_out_wording_says_what_not_to_do_with_it() -> None:
    text = _system(prompts.DuckPrompts(test_out_structure_explicit=True))
    assert "never append to, replace or reshape `test_out`" in text
    assert "starts as `[None, None]`" not in text


def test_the_causal_nudge_after_a_cut_names_the_limit_and_asks_for_a_checked_candidate() -> None:
    """A disclosed cutoff asks for an executable candidate."""
    agent = DuckAgent(prompts=prompts.DuckPrompts(nudge_mode="causal"), completion_cutoff_notice=True)
    opening, turn, _ = _cut_off_turn()
    agent.messages.extend([opening, turn])
    text = agent._followup_prompt().content.to_string()
    assert "cut off at the 8,000-token reply limit" in text
    assert "run `check(fn)`" in text
    assert "Investigate first" not in text


@pytest.mark.parametrize("causal", [False, True])
def test_the_agent_gates_cutoff_disclosure_to_custom_surfaces(causal: bool) -> None:
    received: list[int | None] = []

    class InspectingPrompts(prompts.DuckPrompts):
        def followup_nudge(
            self, modes: prompts.PromptModes, completion_limit: int | None = None, problems: list[str] | None = None
        ) -> str:
            received.append(completion_limit)
            return super().followup_nudge(modes, completion_limit, problems)

    agent = DuckAgent(reply_cap_tokens=None, prompts=InspectingPrompts(nudge_mode="causal" if causal else "legacy"))
    opening, turn, _ = _cut_off_turn()
    agent.messages.extend([opening, turn])
    hidden = agent._followup_prompt().content.to_string()
    assert received == [None]
    assert "8,000" not in hidden and "cut off" not in hidden
    agent.completion_cutoff_notice = True
    shown = agent._followup_prompt().content.to_string()
    assert received == [None, 8_000]
    assert "8,000" in shown and "cut off" in shown


def test_the_causal_nudge_after_a_finished_reply_quotes_the_reading_s_problems() -> None:
    """A reply without a tool call gets the answer reading's specific problems."""
    agent = DuckAgent(prompts=prompts.DuckPrompts(nudge_mode="causal"))
    agent._n_questions = 1
    agent.reading = reading_from_payload(
        {"bound": True, "rows": [[{"state": "ok", "grid": [[1]]}, {"state": "missing"}], [{"state": "missing"}]]}
    )
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="Done."),
        token_usage=ullm.backend.TokenUsage(prompt_tokens=100, completion_tokens=300),
        stop_reason="stop",
        status="processed",
    )
    agent.messages.append(ullm.conv.ResponseTurn(elements=[response], status="processed"))
    text = agent._followup_prompt().content.to_string()
    assert "made no `python` call" in text
    assert "test_out has 2 entries, not 1" in text
    assert "Investigate first" not in text


def test_the_legacy_nudge_hands_the_reading_s_problems_to_a_surface_that_ignores_them() -> None:
    """The agent computes both facts every time; the baseline surface uses
    neither the problems nor, without the notice flag, the cap."""
    legacy = prompts.DuckPrompts(nudge_mode="legacy")
    agent = DuckAgent(reply_cap_tokens=None, prompts=legacy)
    opening, turn, _ = _cut_off_turn()
    agent.messages.extend([opening, turn])
    assert agent._followup_prompt().content.to_string() == legacy.followup_nudge(prompts.PromptModes()).replace(
        legacy.answer_nudge_guidance, legacy.check_answer_nudge_guidance
    )


def test_the_format_guidance_is_its_own_flag_in_both_nudge_modes() -> None:
    guidance = prompts.DuckPrompts().tool_call_format_guidance
    without = prompts.DuckPrompts(nudge_format_guidance=False)
    with_guidance = prompts.DuckPrompts(nudge_format_guidance=True)
    causal_with = prompts.DuckPrompts(nudge_mode="causal", nudge_format_guidance=True)
    causal_without = prompts.DuckPrompts(nudge_mode="causal", nudge_format_guidance=False)
    assert guidance not in without.followup_nudge(_DEFAULT_MODES)
    assert guidance not in causal_without.followup_nudge(_DEFAULT_MODES, 16384)
    assert guidance not in causal_without.followup_nudge(_DEFAULT_MODES, None, ["test_out[0][0] is not set"])
    assert guidance in causal_with.followup_nudge(_DEFAULT_MODES, 16384)
    assert guidance in with_guidance.followup_nudge(_DEFAULT_MODES)
    assert guidance not in prompts.DuckPrompts().followup_nudge(_DEFAULT_MODES), "off as shipped"


def test_the_budget_warning_quotes_what_is_left() -> None:
    text = prompts.DuckPrompts().budget_warning(11_500, _DEFAULT_MODES)
    assert "11,500" in text
    assert "test_out" in text
