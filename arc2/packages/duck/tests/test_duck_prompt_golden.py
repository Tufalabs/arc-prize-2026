"""Every prompt the duck can render, as readable committed text.

The replay reference answers "did this change?" and cannot answer "is this
right?": a prompt already malformed when the reference was recorded has the
defect baked into the expected bytes and replays green forever. And a
reference holds only what one recorded run produced — the repair prompt, the
cutoff nudge and the pruned-rounds note are in no reference unless a run took
that branch, and the runs that take them are the expensive ones. So this
renders the surface exhaustively, in configurations chosen rather than
sampled, and writes it where it can be *read*.

Rendering goes through a configured ``DuckAgent`` — its ``_prompts()``
derivation and the composed system/state prompts over a staged view — rather
than through a bare ``DuckPrompts``, so what is pinned is what the agent
actually sends, the derivation seam included. Direct prompt-class calls
appear only for surfaces the agent exposes verbatim (tool descriptions,
nudges, notices).

Images are pinned by digest rather than by bytes: a picture is not readable
as text, but a silent change to what the model SEES is exactly the class of
defect this file exists to catch, so each attachment contributes its shape
and a hash to the rendering.

Regenerate (free, no model involved) after a deliberate prompt change::

    DUCK_REGENERATE_PROMPT_GOLDEN=1 make test TEST_ARGS='-k prompt_golden'

Then **read the diff** — that is the point of the file, not the fact that it
matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import ullm.backend
import ullm.conv
from duck import CheckMode, DuckAgent, kernel_workspace
from duck.file_suite import file_tools
from duck.prompts import DuckPrompts
from duck.python_tool import _check_failure_content, _show_content
from duck.state import AnswerReading, AnswerSlot
from duck.workspace import Workspace, flat_note

import taaf.game
import taaf.grid
from platypus.context import Produced
from platypus.llm import LLMConfig, build_llm_context

GOLDEN_DIR = Path(__file__).parent / "prompt_golden"

#: Set to rewrite the committed files instead of comparing against them.
REGENERATE = os.environ.get("DUCK_REGENERATE_PROMPT_GOLDEN") == "1"

#: The configurations to render under. Most isolate one prompt branch; the
#: final integrated arm pins the large, latticed dual-presentation surface.
#: Values are ``DuckAgent`` constructor kwargs.
CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "context_summary": {
        "compaction_strategy": "summarize",
        "check_mode": CheckMode.DISABLED,
        "llm_config": LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    },
    # The same request with the oracle bound. The arm above turns the check
    # off, so the pair is what pins that the summary request asks each arm
    # only for what that arm has.
    "context_summary_check": {
        "compaction_strategy": "summarize",
        "check_mode": CheckMode.ENABLED,
        "llm_config": LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    },
    "cutoff_notice": {"completion_cutoff_notice": True},
    # Without images the changed answers have nothing to be shown as.
    "images_off": {"images": "off", "text_grids": True, "show_changed_answers": False},
    "no_show": {"images": "task"},
    # The prompt with no cell values at all: names and shapes only, the
    # puzzle reachable through the kernel. Its wording is the arm's point.
    "kernel_only": {"images": "off", "show_changed_answers": False},
    "text_grids": {"text_grids": True},
    "paired_images": {"image_layout": "paired"},
    "legend": {"image_legend": True},
    "check_images": {"check_mode": CheckMode.ENABLED, "images": "task+show+check"},
    "gridlines": {"image_gridlines": True},
    "brevity": {"prompts": DuckPrompts(brevity_line=True)},
    "likeliest_first": {"prompts": DuckPrompts(likeliest_first_line=True)},
    "brevity_half": {"prompts": DuckPrompts(brevity_line=True, brevity="half")},
    "budget_line": {"prompts": DuckPrompts(budget_line=True)},
    "budget_line_warning": {
        "prompts": DuckPrompts(budget_line=True),
        "budget_warning_tokens": 12_000,
    },
    "cost_model": {"prompts": DuckPrompts(cost_model="explicit")},
    "cost_model_uncapped": {
        "prompts": DuckPrompts(cost_model="explicit"),
        "max_tokens_per_game": None,
    },
    "small_images": {"image_scale": 1},
    "tight_tool_output": {"tool_output_tokens": 500},
    "long_timeout": {"python_timeout_s": 600},
    "no_check": {"check_mode": CheckMode.DISABLED},
    # A passing mandatory check ends the game and takes no assigned answer, so
    # neither the audit turn nor the changed answers have anything to act on.
    "mandatory_check": {"check_mode": CheckMode.MANDATORY, "audit_turn": False, "show_changed_answers": False},
    "check_and_submit": {"check_mode": CheckMode.ENABLED, "check_and_submit": True},
    "flags_check_and_submit": {
        "check_mode": CheckMode.ENABLED,
        "check_and_submit": True,
        "prompts": DuckPrompts(assign_early_line=True),
    },
    # Numbered rows reach three surfaces; the default renders them on all
    # three, so the arm is the one that turns them OFF, over the two the
    # golden can show at once: the task prompt's own digit rows and the
    # failed-check feedback. The kernel's render is the third and is pinned
    # by `test_ascii_row_numbers`, which can execute it.
    "no_row_numbers": {"ascii_row_numbers": False},
    "large_dual_visuals": {
        "check_mode": CheckMode.ENABLED,
        "images": "task+show",
        "text_grids": True,
        "image_layout": "separate",
        "image_legend": False,
        "image_scale": 16,
        "image_gridlines": True,
    },
    # Flag wording and representative combinations with other solver modes.
    "audit": {"prompts": DuckPrompts(second_attempt_mode="audit")},
    "generalization_line": {"prompts": DuckPrompts(generalization_audit_line=True)},
    "no_second_attempt": {"prompts": DuckPrompts(second_attempt_mode="none")},
    "assign_early": {"prompts": DuckPrompts(assign_early_line=True, multi_test_line=True)},
    "work_policy": {"prompts": DuckPrompts(work_policy="candidate_first")},
    "no_investigation_lines": {"prompts": DuckPrompts(investigation_lines=False)},
    "hand_arithmetic_unguarded": {"prompts": DuckPrompts(no_hand_arithmetic_line=False)},
    "legacy_nudge_cutoff": {
        "prompts": DuckPrompts(nudge_mode="legacy"),
        "completion_cutoff_notice": True,
    },
    "nudge_format_guidance": {"prompts": DuckPrompts(nudge_format_guidance=True)},
    "legacy_nudge": {"prompts": DuckPrompts(nudge_mode="legacy")},
    "explicit_test_out": {
        "prompts": DuckPrompts(test_out_structure_explicit=True),
        "budget_warning_tokens": 12_000,
    },
    "flags_no_check": {
        "completion_cutoff_notice": True,
        "check_mode": CheckMode.DISABLED,
        "images": "off",
        "show_changed_answers": False,
        "prompts": DuckPrompts(work_policy="candidate_first", assign_early_line=True, nudge_mode="causal"),
    },
    "flags_mandatory": {
        "completion_cutoff_notice": True,
        "check_mode": CheckMode.MANDATORY,
        "audit_turn": False,
        "show_changed_answers": False,
        "budget_warning_tokens": 12_000,
        "prompts": DuckPrompts(work_policy="candidate_first", nudge_mode="causal", cost_model="explicit"),
    },
    # The file tools are under every answer channel above (the candidate's
    # example follows the channel); this is the surface without them.
    "no_file_tools": {"file_tools": "none"},
    # Both gated tools offered: the tool line counts and names all five.
    "file_tools_full_suite": {"expose_tool_search_files": True, "expose_tool_undo_edit": True},
    # The summary's three switches, each taken away from the summarize arm:
    # the listing without the file tools (no candidate under it), no listing,
    # and the compact request in place of the thorough one.
    "summary_without_file_tools": {
        "compaction_strategy": "summarize",
        "file_tools": "none",
        "llm_config": LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    },
    "summary_no_listing": {
        "compaction_strategy": "summarize",
        "add_files_and_repl_to_summary": False,
        "llm_config": LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    },
    "compact_summary": {
        "compaction_strategy": "summarize",
        "prompts": DuckPrompts(thorough_summary=False),
        "llm_config": LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    },
    # The play without the audit turn; then the audit in the paired layout with
    # digit rows, where its answer slots are drawn the other way.
    "no_audit_turn": {"audit_turn": False},
    "audit_turn_dual": {"image_layout": "paired", "text_grids": True},
    # The audit without its confidence question, its last.
    "no_audit_confidence": {"prompts": DuckPrompts(audit_confidence_line=False)},
    # The groundwork reaches the stopping line, the task footer and the
    # summary request, so its absence renders under the summarize arm.
    "no_audit_groundwork": {
        "compaction_strategy": "summarize",
        "prompts": DuckPrompts(audit_groundwork=False),
        "llm_config": LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    },
    # A call that changes test_out is followed by nothing more.
    "no_show_changed_answers": {"show_changed_answers": False},
    # Summaries that leave the history whole: their request and the message
    # after it, and the compaction request once one is in the history.
    "intermediate_summaries": {
        "compaction_strategy": "summarize",
        "intermediate_summary_prompt_lengths": (20_000,),
        "llm_config": LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    },
    # Every one of the audit and file features off together, under the
    # summarize arm: the play they were added to.
    "without_audit_and_files": {
        "compaction_strategy": "summarize",
        "audit_turn": False,
        "show_changed_answers": False,
        "file_tools": "none",
        "add_files_and_repl_to_summary": False,
        "prompts": DuckPrompts(
            thorough_summary=False,
            audit_groundwork=False,
            audit_confidence_line=False,
            generalization_audit_line=True,
        ),
        "llm_config": LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    },
}

#: What the listing is rendered over: the host's reading of the working
#: directory (a size of -1 is a directory) and of the model's session.
_LISTING_FILES = [("candidate.py", 2150), ("notes.md", 340), ("runs/", -1)]
#: In binding order, as the kernel reports them; the listing reorders them
#: (functions first, then by decreasing name length) and caps them, so there
#: are enough here for every part of the cap to render.
_LISTING_OBJECTS = [
    ("diffs", "ndarray int8 (9, 9)", False),
    ("colours", "dict, 4 entries", False),
    ("find_frame", "function", True),
    ("i", "int", False),
    ("solve", "function", True),
    *[(f"scratch{index}", "int", False) for index in range(130)],
]
#: ``candidate.py`` as the rebuilt conversation shows it, with a run of
#: backticks of its own so the fence it gets is the longer one.
_CANDIDATE = "def fn(grid):\n    # mirror each row; ``` marks nothing here\n    return grid[:, ::-1]\n"


def _view(n_questions: int = 1, given: int = 2) -> taaf.game.GameView:
    """A staged puzzle. Fixed seed and fixed shapes, so the images are
    deterministic and the labels' sizes are stable."""
    rng = np.random.default_rng(2)

    def grid(rows: int, cols: int) -> taaf.grid.Grid:
        return taaf.grid.Grid(rng.integers(0, taaf.grid.N_COLORS, size=(rows, cols), dtype=np.int8))

    pairs = tuple((grid(3, 4), grid(5, 9)) for _ in range(given))
    return taaf.game.GameView(game_id="p", given=pairs, questions=tuple(grid(7, 7) for _ in range(n_questions)))


def _render_element(element: ullm.conv.Content) -> str:
    """One content element as golden text.

    An image cannot be read, so it contributes its shape and a digest —
    enough that a changed picture is a changed file, which is the point.
    """
    if isinstance(element, ullm.conv.ImageContent):
        data = np.asarray(element.data)
        digest = hashlib.sha256(data.tobytes()).hexdigest()[:16]
        return f"<image {data.shape[1]}x{data.shape[0]} sha256:{digest} alt={element.alt_text!r}>"
    return element.to_string()


def _prompt_text(prompt: ullm.conv.UserPrompt) -> str:
    """A user prompt as the model receives it, images stood in for."""
    content = prompt.content
    elements = content.elements if isinstance(content, ullm.conv.ListContent) else [content]
    return "\n".join(_render_element(element) for element in elements)


def _staged(config: dict[str, Any], view: taaf.game.GameView) -> DuckAgent:
    """An agent that has already sent its task prompt, so the next state
    prompt is a re-entry rather than the puzzle again."""
    agent = DuckAgent(**config)
    agent.messages.append(agent._task_prompt(view))  # pyright: ignore[reportPrivateUsage]
    return agent


def _answered(agent: DuckAgent, *, stop_reason: ullm.conv.StopReason) -> DuckAgent:
    """``agent`` with one finished turn behind it, ended by ``stop_reason``.

    The nudge is rendered through the AGENT rather than the prompt class
    because that is the only place ``completion_cutoff_notice`` is read: a
    golden that called ``followup_nudge(n)`` directly would render the
    cutoff wording whether the flag was set or not, and pin nothing.
    """
    response = ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="some working, and no tool call"),
        token_usage=ullm.backend.TokenUsage(prompt_tokens=100, completion_tokens=16384),
        stop_reason=stop_reason,
        status="processed",
    )
    elements: list[ullm.conv.ConversationElement] = [response]
    agent.messages.append(ullm.conv.ResponseTurn(elements=elements, status="processed"))
    return agent


def _with_standing_answer(agent: DuckAgent) -> DuckAgent:
    """``agent`` with a usable first attempt read for its one question."""
    grid = taaf.grid.Grid(np.zeros((1, 1), dtype=np.int8))
    agent.reading = AnswerReading(bound=True, rows=((AnswerSlot(state="ok", grid=grid),),))
    agent._n_questions = 1  # pyright: ignore[reportPrivateUsage]
    return agent


def _audit_prompt(config: dict[str, Any], view: taaf.game.GameView) -> ullm.conv.UserPrompt:
    """The audit prompt of a game that stopped on a two-line reply. Its
    answer covers every way a slot can stand: the first question's second
    slot a copy, the second's a different grid, the third's unusable, and any
    further question's second slot empty."""
    agent = _answered(_staged(config, view), stop_reason="stop")
    rows: list[tuple[AnswerSlot, ...]] = []
    for index, question in enumerate(view.questions):
        first = AnswerSlot(state="ok", grid=question)
        second = (
            first,
            AnswerSlot(state="ok", grid=taaf.grid.Grid(np.fliplr(question.data))),
            AnswerSlot(state="bad", why="is 31x2; each side must be 1..30"),
        )
        rows.append((first, second[index]) if index < len(second) else (first,))
    agent.reading = AnswerReading(bound=True, rows=tuple(rows))
    agent._n_questions = len(view.questions)  # pyright: ignore[reportPrivateUsage]
    return agent._audit_prompt(view, "The rule: every row is mirrored.\n\n- Colours are kept.")  # pyright: ignore[reportPrivateUsage]


def _tool_feedback_prompts(config: dict[str, Any], view: taaf.game.GameView) -> list[ullm.conv.UserPrompt]:
    """Representative post-Python feedback through the configured agent.

    This covers a prompt that cannot be rendered by ``DuckPrompts``: it is
    composed from inspected kernel payloads and injected on the following
    agent entry so image content never passes through a text-only ToolResult.
    """
    agent = _staged(config, view)
    feedback: list[list[ullm.conv.Content]] = []
    if agent._show_enabled():  # pyright: ignore[reportPrivateUsage]
        feedback.append(
            _show_content(
                [{"grid": [[1, 2], [3, 4]], "label": "intermediate candidate"}],
                image_scale=agent._image_scale(),  # pyright: ignore[reportPrivateUsage]
                image_gridlines=agent.image_gridlines,
            )
        )
    if agent.show_changed_answers:
        # Both attempts of the one question written by the call: the question
        # shown once, then each slot.
        rows = ((AnswerSlot(state="ok", grid=view.questions[0]), AnswerSlot(state="ok", grid=view.questions[0])),)
        agent.reading = AnswerReading(bound=True, rows=rows)
        feedback.append(agent._changed_answers_content(view))  # pyright: ignore[reportPrivateUsage]
    if agent.check_mode.enabled:
        feedback.append(
            _check_failure_content(
                {
                    "pairs": [
                        {"index": 0, "got": [[1, 0], [0, 1]]},
                        {"index": 1, "got": None},
                    ]
                },
                include_images=agent._check_images_enabled(),  # pyright: ignore[reportPrivateUsage]
                image_scale=agent._image_scale(),  # pyright: ignore[reportPrivateUsage]
                image_gridlines=agent.image_gridlines,
                row_numbers=agent.ascii_row_numbers,
            )
        )
    feedback = [group for group in feedback if group]
    if not feedback:
        return []
    agent._pending_tool_feedback.extend(feedback)  # pyright: ignore[reportPrivateUsage]
    return agent._build_state_prompts(view)  # pyright: ignore[reportPrivateUsage]


def _renderings(config: dict[str, Any]) -> dict[str, str]:
    """Every prompt this configuration can produce, by name.

    Names are filenames, so the directory listing reads as an inventory of
    the surface. Each rendering gets its own agent, so no scenario leaks
    state into another.
    """
    base = DuckAgent(**config)
    surface = base._prompts()  # pyright: ignore[reportPrivateUsage]

    single = _view(n_questions=1)
    multi = _view(n_questions=3)

    # A re-entry over a view whose answers are all still missing: the
    # repair prompt is what the model is handed when it stops without a
    # readable answer, and it is the one prompt a happy-path run never
    # produces.
    reentry_agent = _staged(config, multi)
    reentry = reentry_agent._build_state_prompt(multi)  # pyright: ignore[reportPrivateUsage]

    renderings = {
        "system": base._system_prompt_text(),  # pyright: ignore[reportPrivateUsage]
        "tool_python": surface.python_tool_description(base._prompt_modes()),  # pyright: ignore[reportPrivateUsage]
        "tool_python_code_param": surface.python_tool_code_param_description(),
        "user_task_single_question": _prompt_text(
            DuckAgent(**config)._task_prompt(single)  # pyright: ignore[reportPrivateUsage]
        ),
        "user_task_multi_question": _prompt_text(
            DuckAgent(**config)._task_prompt(multi)  # pyright: ignore[reportPrivateUsage]
        ),
        "user_answer_repair": _prompt_text(reentry),
        # Through the agent: the nudge after an ordinary stop, and the one
        # after a generation cut off at the cap.
        "nudge": _prompt_text(
            _answered(_staged(config, single), stop_reason="stop")._followup_prompt()  # pyright: ignore[reportPrivateUsage]
        ),
        "nudge_after_cutoff": _prompt_text(
            _answered(_staged(config, single), stop_reason="max_tokens")._followup_prompt()  # pyright: ignore[reportPrivateUsage]
        ),
        "rounds_pruned_note": surface.rounds_pruned_note(7),
    }
    if base.check_mode is not CheckMode.MANDATORY:
        # A cut reply with the answer standing is nudged rather than ending the
        # game; a passing mandatory check ends the game before any such reply.
        renderings["nudge_after_cutoff_answered"] = _prompt_text(
            _answered(_with_standing_answer(_staged(config, single)), stop_reason="max_tokens")._followup_prompt()  # pyright: ignore[reportPrivateUsage]
        )
        renderings["answer_repair_one_problem"] = surface.answer_repair(["test_out[0][0] is not set"])
        renderings["answer_repair_many_problems"] = surface.answer_repair(
            [
                "test_out[0][0] is not set",
                "test_out[1][0] has values outside 0..9",
                "test_out[2][1] is 31 rows, over the 30 limit",
            ]
        )
    backend = ullm.backend.DummyBackend()
    backend.total_usage.completion_tokens = max(0, (base.max_tokens_per_game or 60_000) - 11_000)

    def ignore(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        del submission, produced

    ctx, _meter = build_llm_context(
        game_id="golden", submissions=[], rng=random.Random(0), backend=backend, submit=ignore
    )
    if note := base._budget_note(ctx):  # pyright: ignore[reportPrivateUsage]
        name = "budget_warning" if base.budget_warning_tokens is not None else "budget_status"
        renderings[name] = note
        backend.total_usage.completion_tokens += 1_000
        if repeated := base._budget_note(ctx):  # pyright: ignore[reportPrivateUsage]
            renderings["budget_status_later"] = repeated
    modes = base._prompt_modes()  # pyright: ignore[reportPrivateUsage]
    # After a reply refused on every re-sample and dropped (``malformed_nudges``).
    renderings["nudge_after_malformed"] = surface.malformed_nudge(modes, base.reading.problems(1))
    if base.check_mode is not CheckMode.MANDATORY:
        renderings["nudge_after_malformed_answered"] = surface.malformed_nudge(modes, [])
    if base.audit_turn:
        renderings["user_audit_single_question"] = _prompt_text(_audit_prompt(config, single))
        renderings["user_audit_multi_question"] = _prompt_text(_audit_prompt(config, _view(n_questions=4)))
    if base._uses_context_summary():  # pyright: ignore[reportPrivateUsage]
        target = base._summary_target_tokens()  # pyright: ignore[reportPrivateUsage]
        renderings["compaction_request"] = surface.compaction_prompt(modes, target)
        renderings["compaction_wrapper"] = surface.compaction_wrapper(
            "Representative compact summary with the current candidate source and next step."
        )
        later = surface.compaction_prompt(modes, target, previous_summary=True)
        if later != renderings["compaction_request"]:
            renderings["compaction_request_later"] = later
        if base.intermediate_summary_prompt_lengths:
            renderings["intermediate_summary_request"] = surface.intermediate_summary_prompt(modes)
            renderings["intermediate_summary_continue"] = surface.intermediate_summary_continue
            renderings["compaction_request_after_intermediate"] = surface.compaction_prompt(
                modes, target, earlier_summaries=True
            )
        if modes.add_files_and_repl_to_summary:
            listing = surface.compaction_listing(modes, _LISTING_FILES, _LISTING_OBJECTS)
            if modes.file_tools:
                renderings["compaction_request_candidate_shown"] = surface.compaction_prompt(
                    modes, target, candidate_shown=True
                )
                listing += "\n\n" + surface.compaction_candidate(_CANDIDATE)
            renderings["compaction_closing"] = surface.compaction_closing(modes, listing)
    if modes.file_tools:
        for tool in file_tools(
            # What a tool says of itself does not depend on the directory.
            Workspace(repl=cast(Any, None)),
            search_files=base.expose_tool_search_files,
            undo_edit=base.expose_tool_undo_edit,
        ):
            renderings[f"tool_{tool.name}"] = f"{tool.description}\n\n{json.dumps(tool.parameters, indent=2)}"
        renderings["kernel_notices"] = "\n\n".join(
            [
                kernel_workspace.CWD_NOTICE,
                kernel_workspace.TOOL_NOTICE.format(name="check"),
                flat_note(["sub/"]).strip(),
            ]
        )
    if feedback := _tool_feedback_prompts(config, single):
        renderings["user_python_feedback"] = "\n\n<next user message>\n\n".join(
            _prompt_text(prompt) for prompt in feedback
        )
    return renderings


def _golden_path(config: str, name: str) -> Path:
    """Where one rendering is committed."""
    return GOLDEN_DIR / config / f"{name}.txt"


def _all_renderings() -> Iterator[tuple[Path, str]]:
    """Every (path, text) pair the golden should contain."""
    for config_name, config in CONFIGS.items():
        for name, text in _renderings(config).items():
            yield _golden_path(config_name, name), text


def test_the_golden_matches_what_the_prompts_render() -> None:
    """Every committed rendering still renders that way.

    A failure here is not necessarily a bug — a deliberate prompt change
    lands as a diff. It is a prompt to go and read the diff, and to
    regenerate once it says what was meant.
    """
    rendered = dict(_all_renderings())

    if REGENERATE:
        if GOLDEN_DIR.exists():
            for stale in GOLDEN_DIR.rglob("*.txt"):
                stale.unlink()
            # And the directories they were the last of, or a renamed
            # configuration leaves an empty one behind for good.
            for directory in sorted(GOLDEN_DIR.rglob("*"), reverse=True):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
        for path, text in rendered.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        pytest.skip(f"regenerated {len(rendered)} renderings under {GOLDEN_DIR.name}/")

    missing = [path for path in rendered if not path.exists()]
    assert not missing, (
        f"{len(missing)} rendering(s) have no committed golden, first: "
        f"{missing[0].relative_to(GOLDEN_DIR)}. Regenerate: "
        f"DUCK_REGENERATE_PROMPT_GOLDEN=1 make test TEST_ARGS='-k prompt_golden'"
    )
    for path, text in rendered.items():
        assert path.read_text() == text, (
            f"{path.relative_to(GOLDEN_DIR)} renders differently than committed. Read the diff; "
            f"if it is what you meant, regenerate with DUCK_REGENERATE_PROMPT_GOLDEN=1."
        )


def test_the_golden_holds_nothing_the_prompts_no_longer_render() -> None:
    """No orphans: a rendering that was removed leaves its file behind, and
    a stale file is worse than a missing one — it reads as current."""
    if REGENERATE:
        pytest.skip("regenerating")
    expected = {path for path, _ in _all_renderings()}
    orphans = sorted(path.relative_to(GOLDEN_DIR) for path in GOLDEN_DIR.rglob("*.txt") if path not in expected)

    assert not orphans, f"committed goldens nothing renders any more: {orphans}"


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_every_configuration_changes_the_surface(config_name: str) -> None:
    """The configurations differ in what they produce.

    Compared over the whole surface rather than the system prompt alone —
    some fields (the image scale) reach the task prompt but no other. Without
    this, a field that stopped being read would leave a configuration's
    golden identical to the default's and the check would still pass: the
    surface would look covered while nothing exercised it.
    """
    default = _renderings(CONFIGS["default"])
    other = _renderings(CONFIGS[config_name])

    assert (other == default) == (config_name == "default"), (
        f"configuration {config_name!r} renders the same surface as 'default', "
        f"so it is not exercising the branch it was added for"
    )


def test_no_rendering_is_empty() -> None:
    """An empty prompt is a template that lost its content, which is the kind
    of defect a byte-comparison against an equally empty reference misses."""
    empty = [path.relative_to(GOLDEN_DIR) for path, text in _all_renderings() if not text.strip()]

    assert not empty, f"renderings came back empty: {empty}"


def test_prompt_state_is_not_mutated_by_rendering() -> None:
    """Rendering must not change what a later rendering sees: the same staged
    scenario rendered twice is byte-identical, so the golden's own order
    cannot become load-bearing."""
    assert _renderings(CONFIGS["default"]) == _renderings(CONFIGS["default"])


def test_the_task_prompt_pins_its_pictures() -> None:
    """The images are the half of the prompt that text cannot show, so the
    golden has to fail when a picture changes. Rendering the same view at a
    different scale must produce a different digest — otherwise the image
    lines are decoration and a changed picture would slip through."""
    view = _view()
    at_four = _prompt_text(DuckAgent(reply_cap_tokens=None, image_scale=4)._task_prompt(view))  # pyright: ignore[reportPrivateUsage]
    at_one = _prompt_text(DuckAgent(reply_cap_tokens=None, image_scale=1)._task_prompt(view))  # pyright: ignore[reportPrivateUsage]

    assert "sha256:" in at_four
    assert at_four != at_one
