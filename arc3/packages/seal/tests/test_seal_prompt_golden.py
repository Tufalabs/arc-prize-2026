"""Every prompt the seal can render, as readable committed text.

The committed references answer "did this change?" and cannot answer "is
this right?": a prompt already malformed when a reference was recorded has
the defect baked into the expected bytes and replays green forever. And a
reference holds only what one recorded run produced — the game-over lines,
the reset notice, the legacy runtime surface are in no reference unless a
run took that branch. So this renders the surface exhaustively, in
configurations chosen rather than sampled, and writes it where it can be
*read*.

Rendering goes through a configured ``SealAgent`` — its ``_prompts()``
derivation and the composed system/user prompts over staged state — rather
than through a bare ``SealPrompts``, so what is pinned is what the agent
actually sends: the derivation seam included. Direct prompt-class calls
appear only for surfaces the agent exposes verbatim (tool descriptions,
nudges, notices).

Regenerate (free, no model involved) after a deliberate prompt change::

    PLATYPUS_REGENERATE_PROMPT_GOLDEN=1 make test TEST_ARGS='-k prompt_golden'

Then **read the diff** — that is the point of the file, not the fact that
it matches.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import pytest

from platypus import Action, Observation
from seal.agent import DUCK_LIKE, KIMI_STUDENT, SealAgent
from seal.memory_tool import SealMemoryTool
from seal.prompts import SealPrompts
from seal.state import FramePayload, MemoryEntry, StepSummary, TimelineEntry

GOLDEN_DIR = Path(__file__).parent / "prompt_golden"

#: Set to rewrite the committed files instead of comparing against them.
REGENERATE = os.environ.get("PLATYPUS_REGENERATE_PROMPT_GOLDEN") == "1"

#: The configurations to render under. Chosen, not combinatorial: each
#: differs from the default in exactly one field the prompts branch on, so a
#: diff points at a cause. Values are ``SealAgent`` constructor kwargs.
CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    # The board as text alone: every sentence about the image goes.
    "no_image": {"multimodal_context": ""},
    # The third image mode. It differs from the default in one line — how
    # often an image arrives — which is exactly the claim a golden should pin,
    # since the schedule is the thing the model cannot check for itself.
    "multimodal_level_start": {"multimodal_context": "level_start"},
    "no_brevity": {"prompts": SealPrompts(brevity="off")},
    "persistence_round": {"persistence": "round"},
    # The default clears the session at a level, so the lifetime that never
    # clears it is the variant worth pinning.
    "persistence_always": {"persistence": "always"},
    # The coordinate convention is spent only where MOUSE exists, so the
    # game that offers it is the one configuration that states it.
    "mouse_game": {"game_action_names": ["RESET", "UP", "DOWN", "LEFT", "RIGHT", "MOUSE"]},
    # Every world-model sentence is spent only when one is in use, so the
    # seal without one is pinned as its own surface.
    "no_world_model": {"use_world_model": "none"},
    # The world model required: the exploration budget, and a batch that
    # cannot run until a model passes.
    "mandatory_world_model": {"use_world_model": "mandatory"},
    # UNDO offered, which only a seal without a world model may: named among
    # the game's actions.
    "undo_game": {
        "use_world_model": "none",
        "expose_undo": True,
        "game_action_names": ["RESET", "UP", "DOWN", "LEFT", "RIGHT", "UNDO"],
    },
    # The pixel tolerance offered, starting above 0: its line, with the value;
    # and not offered, where the default offers it.
    "pixel_tolerance": {"expose_pixel_tolerance": True, "pixel_tolerance": 2},
    "no_pixel_tolerance": {"expose_pixel_tolerance": False},
    # The python tool's optional plan: its parameter, and where it is kept;
    # asked for ahead of the code, and after it.
    "plan_first": {"plan_argument": "first"},
    "plan_second": {"plan_argument": "second"},
    # The prompt switches, each at its one non-default level beside
    # `no_brevity`.
    "brevity_full": {"prompts": SealPrompts(brevity="full")},
    "no_cost_model": {"prompts": SealPrompts(cost_model="none")},
    "no_nudge_format_guidance": {"prompts": SealPrompts(nudge_format_guidance=False)},
    "no_hand_arithmetic": {"prompts": SealPrompts(no_hand_arithmetic_line=True)},
    "no_winnable_line": {"prompts": SealPrompts(winnable_line=False)},
    "no_retest_actions_line": {"prompts": SealPrompts(retest_actions_line=False)},
    # What files are for besides code, at each of its two non-default levels.
    "notes_encourage": {"prompts": SealPrompts(notes_guidance="encourage")},
    "notes_discourage": {"prompts": SealPrompts(notes_guidance="discourage")},
    # The compact summary request, which states a size, in place of the
    # thorough one.
    "compact_summary": {"prompts": SealPrompts(thorough_summary=False)},
    # The journal: its tool, its sentences, and its line in the summary
    # request. Without the level reflection, which the journal refuses.
    "journal": {"journal": True, "level_reflection": "none"},
    # Eviction with the journal in place of the live summary, whose own
    # renderings (the request, the wrapper, the closing message) go with it.
    # Two fields, not one: eviction without the journal is a seal with no
    # memory, legal and run by nothing.
    "prune": {"compaction_strategy": "prune", "journal": True, "level_reflection": "none"},
    # The other setting the summary's wording reads; the journal is `journal`.
    "no_summary_listing": {"add_files_and_repl_to_summary": False},
    # The world model's file left out of the rebuilt conversation, and out of
    # what the summary request promises.
    "no_summary_world_model": {"add_world_model_to_summary": False},
    # The image's two settings besides the schedule, each off its default.
    "colour_legend": {"color_name_legend": True},
    "image_scale_4": {"multimodal_upscale": 4},
    # The board images labelled: one line on the image's coordinates.
    "axis_labels": {"axis_labels": True},
    # The board in hex digits, where the default prints the duck's letters:
    # the legend, and the index legend's glyphs.
    "hex": {"ascii_symbols": "hex"},
    # Every setting at its duck-like value (``SealAgent.build(duck_like=True)``).
    "duck_like": dict(DUCK_LIKE),
    # What the Kimi teacher traces were played with
    # (``SealAgent.build(kimi_student=True)``), so a student's prompts can be
    # read against them.
    "kimi_student": dict(KIMI_STUDENT),
    # The turn after a won level: its prompt, and the next level's after it,
    # concise here and verbose in the default; and no turn at all.
    "concise_level_reflection": {"level_reflection": "concise"},
    "no_level_reflection": {"level_reflection": "none"},
    # The strategy audit's prompt, in a game with MOUSE so the colours no click
    # has hit render beside the untried actions, in each of its two modes.
    "strategy_audit": {
        "audit_level_tokens": (75_000,),
        "strategy_audit": "verbose",
        "game_action_names": ["RESET", "UP", "DOWN", "LEFT", "RIGHT", "MOUSE"],
    },
    "strategy_audit_concise": {
        "audit_level_tokens": (75_000,),
        "strategy_audit": "concise",
        "game_action_names": ["RESET", "UP", "DOWN", "LEFT", "RIGHT", "MOUSE"],
    },
    # No audit, where the default runs one.
    "no_strategy_audit": {"strategy_audit": "none"},
    # The system prompt without the replay reminder's line, which the default
    # carries.
    "no_replay_reminder": {"replay_reminder": False},
}

#: What the listing is rendered over: the host's reading of the working
#: directory (a size of -1 is a directory) and of the model's session.
_LISTING_FILES = [("board_tools.py", 2150), ("notes.md", 340), ("runs/", -1)]
#: In binding order, as the kernel reports them; the listing reorders them
#: (functions first, then by decreasing name length) and caps them, so there
#: are enough here for every part of the cap to render.
_LISTING_OBJECTS = [
    ("walls", "ndarray uint8 (64, 64)", False),
    ("paths", "dict, 12 entries", False),
    ("find_path", "function", True),
    ("i", "int", False),
    ("solve", "function", True),
    *[(f"scratch{index}", "int", False) for index in range(130)],
]
#: A world model as the closing message shows it: the model's own, with a
#: fence inside it, which the rendering must not let close its own.
_WORLD_MODEL_SOURCE = '"""Keys open doors."""\n\nDOC = \'```\'\n\n\ndef do_move(state, action):\n    return state\n'


#: A stand-in action list, fixed rather than drawn from a game, so the
#: golden does not move when a game's action set does.
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

#: The game's action set, which a real agent freezes from its first
#: observation before the first system prompt is built. Staged here for the
#: same reason: a surface that names the actions must be rendered with some.
GAME_ACTIONS = ["RESET", "UP", "DOWN", "LEFT", "RIGHT"]


def _agent(config: dict[str, Any]) -> SealAgent:
    """A fresh agent in one golden configuration, before any staged state.

    The action set is staged at construction rather than by an observation:
    a real agent freezes it from its first observation, which lands before
    the first system prompt is built, so every rendering here sees one too.
    """
    kwargs: dict[str, Any] = {"game_action_names": list(GAME_ACTIONS)}
    kwargs.update(config)
    return SealAgent(**kwargs)


def _frame(step: int, level: int, *, fill: int = 0) -> FramePayload:
    """One staged timeline frame, as ``_frame_payload`` builds it, over a
    fixed small grid."""
    grid = np.full((4, 4), fill, dtype=np.uint8)
    grid[0, 0] = 5
    grid[3, 2] = 9
    return FramePayload(
        grid_b64=base64.b64encode(grid.tobytes()).decode("ascii"),
        shape=[4, 4],
        step=step,
        level=level,
    )


def _summary(**overrides: Any) -> StepSummary:
    """A staged executed-batch summary as ``_take_step_summary`` builds it,
    defaulting to an ordinary two-action same-level batch."""
    summary = StepSummary(
        executed_count=2,
        executed_actions=["UP", "LEFT"],
        level=1,
        level_transition=False,
        run_complete=False,
        game_over=False,
    )
    return dataclasses.replace(summary, **overrides)


def _stage_opening(agent: SealAgent) -> None:
    """The state every real prompt is built over: the game-initial
    observation has landed on the timeline and named the valid actions."""
    agent.history_entries.append(TimelineEntry(action=None, frame=_frame(0, 1), animation=[_frame(0, 1)]))
    agent.current_valid_actions = list(ACTIONS)
    agent.valid_actions_observed = True
    agent.exploration_moves_left = agent.exploration_moves_new_game


def _stage_batch(agent: SealAgent, summary: StepSummary) -> None:
    """State after one executed two-action batch reported by ``summary``."""
    _stage_opening(agent)
    level_after = (summary.level or 1) + (1 if summary.level_transition else 0)
    agent.history_entries.append(TimelineEntry(action=Action("ACTION1"), frame=_frame(1, 1), animation=[_frame(1, 1)]))
    agent.history_entries.append(
        TimelineEntry(
            action=Action("ACTION3"), frame=_frame(2, level_after, fill=1), animation=[_frame(2, level_after, fill=1)]
        )
    )
    agent.action_count = 2
    agent.last_step_summary = summary


def _stage_reflection(agent: SealAgent, *, win_missed: bool) -> None:
    """State in the turn after a move that won level 1: the winning turn at
    the end of level 1's history, its animation running on to level 2's
    opening board, and the move's report owed. The win screen's delta is a
    stand-in: the kernel writes it."""
    _stage_opening(agent)
    agent.history_entries[0].level = 1
    agent.history_entries.append(
        TimelineEntry(action=Action("ACTION1"), frame=_frame(1, 1), animation=[_frame(1, 1)], level=1)
    )
    win_screen, opening = _frame(2, 1, fill=1), _frame(2, 2, fill=2)
    agent.history_entries.append(
        TimelineEntry(action=Action("ACTION3"), frame=opening, animation=[win_screen, opening], outcome="won", level=1)
    )
    agent.history_entries.append(TimelineEntry(action=None, frame=opening, animation=[opening], level=2))
    agent.action_count = 2
    agent.terminal_this_move = {"level_completed": True}
    agent.last_step_summary = _summary(level_transition=True, level=2)
    agent.win_missed = win_missed
    agent.frame_delta_text = "<the win screen as a delta from the frame before the move>"


def _stage_audit(agent: SealAgent) -> None:
    """State on a level that has run long: a click on colour 5 and an UP
    played, so DOWN, LEFT and RIGHT are untried, and the board's colours 0 and
    9 no click has hit."""
    _stage_opening(agent)
    agent.history_entries[0].level = 1
    agent.history_entries.append(
        TimelineEntry(action=Action("ACTION6", row=0, col=0), frame=_frame(1, 1), animation=[_frame(1, 1)], level=1)
    )
    agent.history_entries.append(
        TimelineEntry(action=Action("ACTION1"), frame=_frame(2, 1), animation=[_frame(2, 1)], level=1)
    )
    agent.action_count = 2


#: Journal entries for the with-journal system prompt, in the shape
#: ``_append_memory`` records.
_JOURNAL = [
    MemoryEntry(
        index=1,
        response=2,
        step=0,
        level=1,
        last_action=None,
        text="Walls are colour 5; the avatar is the 2x2 block.",
    ),
    MemoryEntry(
        index=2,
        response=5,
        step=3,
        level=1,
        last_action="UP",
        text="UP moves the avatar one row; it does not wrap.",
    ),
]


def _user_prompt(agent: SealAgent) -> str:
    """The per-turn user prompt as the model receives it, rendered to text.

    Built through ``_build_user_prompt_element``, the path a turn takes, so
    the image configurations show their attachments. The observation argument is
    unused by the current composition (everything comes from staged state),
    so the golden passes a placeholder rather than fabricating an engine
    state.
    """
    element = agent._build_user_prompt_element(cast(Observation, None))  # pyright: ignore[reportPrivateUsage]
    return element.content.to_string()


def _reflection_prompt(agent: SealAgent) -> str:
    """The reflection turn's prompt as the model receives it, rendered to
    text, through the path the turn takes."""
    element = agent._build_reflection_prompt_element()  # pyright: ignore[reportPrivateUsage]
    return element.content.to_string()


def _audit_prompt(agent: SealAgent) -> str:
    """The strategy audit's prompt as the model receives it, rendered to text,
    through the path the audit takes."""
    element = agent._build_audit_prompt_element()  # pyright: ignore[reportPrivateUsage]
    return element.content.to_string()


class _Placeholders(dict[str, object]):
    """Placeholder values for the side strings, which are templates.

    Anything a new string interpolates gets a legible stand-in rather than
    a KeyError, so adding one needs no edit here.
    """

    _KNOWN = {"action": "UP", "verb": "executed", "index": 1, "name": "WARP", "budget": 1000}

    def __missing__(self, key: str) -> object:
        return self._KNOWN.get(key, f"<{key}>")


def _agent_read_strings(surface: Any) -> str:
    """Every string the agent reads at its own call sites, named and rendered.

    Derived from the `Prompts` protocol rather than listed, because a listed
    bundle covers only what someone remembered to add to it, and a string
    missing from it is one no golden pins.

    Named as well as rendered: these are one-line strings with no context of
    their own, and a diff of bare sentences says nothing about which one
    moved.
    """
    from seal.prompts import Prompts, SealPrompts

    reference = SealPrompts()
    lines: list[str] = []
    for field_name in Prompts.__annotations__:
        value = getattr(reference, field_name, None)
        if not isinstance(value, str):
            continue  # a method or a mapping; the composed renderings cover those
        rendered = str(getattr(surface, field_name)).format_map(_Placeholders())
        lines.append(f"{field_name}: {rendered}")
    return "\n\n".join(lines)


def _renderings(config: dict[str, Any]) -> dict[str, str]:
    """Every prompt this configuration can produce, by name.

    Names are filenames, so the directory listing reads as an inventory of
    the surface. Each rendering gets its own freshly staged agent, so no
    scenario leaks state into another.
    """
    base = _agent(config)

    with_journal = _agent(config)
    with_journal.memory_entries = [dataclasses.replace(entry) for entry in _JOURNAL]

    first_turn = _agent(config)
    _stage_opening(first_turn)

    after_batch = _agent(config)
    _stage_batch(after_batch, _summary())

    level_transition = _agent(config)
    _stage_batch(level_transition, _summary(level_transition=True, level=2))
    level_transition.exploration_moves_left = level_transition.exploration_moves_new_level

    # Under a world model: exploration over, and the two ways a model stops
    # a batch.
    required = _agent(config)
    _stage_batch(required, _summary())
    required.exploration_over = True

    mismatch = _agent(config)
    _stage_batch(mismatch, _summary(world_model_stop="mismatch"))
    mismatch.exploration_over = True

    declined = _agent(config)
    _stage_batch(declined, _summary(world_model_stop="unknown"))
    declined.exploration_over = True

    # Under an optional world model, what the last batch ran under.
    checked = _agent(config)
    _stage_batch(checked, _summary(checked=True))

    unchecked = _agent(config)
    _stage_batch(unchecked, _summary(checked=False))

    game_over_required = _agent(config)
    _stage_batch(game_over_required, _summary(game_over=True))
    game_over_required.game_over_active = True
    game_over_required.exploration_over = True

    run_complete = _agent(config)
    _stage_batch(run_complete, _summary(run_complete=True))

    game_over = _agent(config)
    _stage_batch(game_over, _summary(game_over=True))
    game_over.game_over_active = True

    reset_notice = _agent(config)
    _stage_opening(reset_notice)
    reset_notice.repl_reset_pending = "a new level started"

    reflection = _agent(config)
    _stage_reflection(reflection, win_missed=False)

    reflection_win_missed = _agent(config)
    _stage_reflection(reflection_win_missed, win_missed=True)

    audit = _agent(config)
    _stage_audit(audit)

    after_reflection = _agent(config)
    _stage_batch(after_reflection, _summary(level_transition=True, level=2, reported=True))
    after_reflection.exploration_moves_left = after_reflection.exploration_moves_new_level

    surface = base.prompts
    memory_tool = SealMemoryTool()
    modes = base._prompt_modes()  # pyright: ignore[reportPrivateUsage]

    renderings = {
        "system": base._system_prompt_text(),  # pyright: ignore[reportPrivateUsage]
        "system_with_journal": with_journal._system_prompt_text(),  # pyright: ignore[reportPrivateUsage]
        "tool_python": surface.python_tool_description(modes),
        "tool_python_code_param": surface.python_tool_code_param_description(modes),
        **(
            {"tool_python_plan_param": surface.python_tool_plan_param_description(modes)}
            if modes.plan_argument != "none"
            else {}
        ),
        "tool_memory": f"{memory_tool.description}\n\nparameters:\n{json.dumps(memory_tool.parameters, indent=2)}",
        "user_first_turn": _user_prompt(first_turn),
        "user_after_batch": _user_prompt(after_batch),
        "user_level_transition": _user_prompt(level_transition),
        "user_run_complete": _user_prompt(run_complete),
        "user_game_over": _user_prompt(game_over),
        "user_reset_notice": _user_prompt(reset_notice),
        "nudge": surface.followup_nudge(modes=modes),
        "nudge_completion_cutoff": surface.followup_nudge(16384, modes),
        "nudge_after_malformed": surface.malformed_nudge(modes),
        "rounds_pruned_note": surface.rounds_pruned_note(2),
        # In-kernel result strings: what acting past a terminal state, or a
        # second acting call in one response, answers with.
        "action_result_notices": _agent_read_strings(surface),
        # The journal's own model-visible text, including what the agent
        # scans back to decide which entries the conversation still carries.
        "memory_notices": "\n\n".join(
            [
                surface.memory_append_acknowledgement(_JOURNAL[1]),
                surface.memory_budget_not_positive,
                surface.memory_empty_text,
                surface.memory_entry_too_large.format(entry_tokens=412, budget=1000),
            ]
        ),
    }
    if base._uses_context_summary():  # pyright: ignore[reportPrivateUsage]
        # Rendered as a summary with the model's own world model in hand does.
        world_model = modes.world_model and base.add_world_model_to_summary
        listing = [
            surface.compaction_listing(modes, _LISTING_FILES, _LISTING_OBJECTS)
            if modes.add_files_and_repl_to_summary
            else "",
            surface.compaction_world_model(_WORLD_MODEL_SOURCE) if world_model else "",
        ]
        renderings.update(
            {
                "context_summary_request": surface.compaction_prompt(
                    modes, 5632, current_level=1, world_model_shown=world_model
                ),
                "context_summary_request_later_level": surface.compaction_prompt(
                    modes, 5632, current_level=3, world_model_shown=world_model
                ),
                "context_summary_wrapper": surface.compaction_wrapper("SUMMARY TEXT"),
                "context_summary_restart_preamble": surface.compaction_restart_preamble,
                "context_summary_closing": surface.compaction_closing(
                    modes, "\n\n".join(part for part in listing if part)
                ),
                # A session reset by the time the summary is installed, after a
                # listing that took the kernel with it.
                "context_summary_closing_reset": surface.compaction_closing(
                    modes,
                    surface.compaction_listing(
                        modes,
                        _LISTING_FILES,
                        [],
                        objects_unavailable=surface.compaction_objects_unavailable(
                            kernel_lost=True, error="<the listing's error>"
                        ),
                    )
                    if modes.add_files_and_repl_to_summary
                    else "",
                    session_intact=False,
                ),
                "context_summary_omitted_note": surface.tool_calls_omitted_note(3),
            }
        )
    # States only some modes reach: elsewhere they render as the ordinary
    # prompts, and pinning them twice says nothing.
    if base.use_world_model != "none":
        renderings.update(
            {
                "user_world_model_mismatch": _user_prompt(mismatch),
                "user_world_model_unknown": _user_prompt(declined),
            }
        )
    if base.use_world_model == "mandatory":
        renderings.update(
            {
                "user_world_model_required": _user_prompt(required),
                "user_game_over_world_model_required": _user_prompt(game_over_required),
            }
        )
    if base.use_world_model == "optional":
        renderings.update(
            {
                "user_after_checked_batch": _user_prompt(checked),
                "user_after_unchecked_batch": _user_prompt(unchecked),
            }
        )
    if base.strategy_audit != "none":
        renderings["user_strategy_audit"] = _audit_prompt(audit)
    if base.level_reflection != "none":
        renderings.update(
            {
                "user_reflection": _reflection_prompt(reflection),
                "user_reflection_win_missed": _reflection_prompt(reflection_win_missed),
                "user_level_transition_after_reflection": _user_prompt(after_reflection),
            }
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

    missing = [p for p in rendered if not p.exists()]
    assert not missing, (
        f"{len(missing)} rendering(s) have no committed golden, first: "
        f"{missing[0].relative_to(GOLDEN_DIR)}. Regenerate: "
        f"PLATYPUS_REGENERATE_PROMPT_GOLDEN=1 make test TEST_ARGS='-k prompt_golden'"
    )
    for path, text in rendered.items():
        assert path.read_text() == text, (
            f"{path.relative_to(GOLDEN_DIR)} renders differently than committed. Read the diff; "
            f"if it is what you meant, regenerate with PLATYPUS_REGENERATE_PROMPT_GOLDEN=1."
        )


def test_the_golden_holds_nothing_the_prompts_no_longer_render() -> None:
    """No orphans: a rendering that was removed leaves its file behind, and
    a stale file is worse than a missing one — it reads as current."""
    if REGENERATE:
        pytest.skip("regenerating")
    expected = {path for path, _ in _all_renderings()}
    orphans = sorted(p.relative_to(GOLDEN_DIR) for p in GOLDEN_DIR.rglob("*.txt") if p not in expected)

    assert not orphans, f"committed goldens nothing renders any more: {orphans}"


@pytest.mark.parametrize("config_name", sorted(CONFIGS))
def test_every_configuration_changes_the_surface(config_name: str) -> None:
    """The configurations differ in what they produce.

    Compared over the whole surface rather than the system prompt alone —
    some fields (the frame API) reach the tool description and the nudge
    but not every prompt. Without this, a field that stopped being read
    would leave a configuration's golden identical to the default's and the
    check would still pass — the surface would look covered while nothing
    exercised it.
    """
    default = _renderings(CONFIGS["default"])
    other = _renderings(CONFIGS[config_name])

    assert (other == default) == (config_name == "default"), (
        f"configuration {config_name!r} renders the same surface as 'default', "
        f"so it is not exercising the branch it was added for"
    )


def test_no_rendering_is_empty() -> None:
    """An empty prompt is a template that lost its content, which is the
    kind of defect a byte-comparison against an equally empty reference
    misses."""
    empty = [path.relative_to(GOLDEN_DIR) for path, text in _all_renderings() if not text.strip()]

    assert not empty, f"renderings came back empty: {empty}"


def test_the_inventory_is_not_silently_shrinking() -> None:
    """A floor on coverage, so deleting a rendering from the list is a
    decision rather than an omission nobody sees."""
    per_config = {name: len(_renderings(config)) for name, config in CONFIGS.items()}

    thin = {name: n for name, n in per_config.items() if n < 16}
    assert not thin, f"configurations rendering too few prompts: {thin}"


def test_prompt_state_is_not_mutated_by_rendering() -> None:
    """Rendering the golden must not change what a later rendering sees:
    the same staged scenario rendered twice is byte-identical, so the
    golden's own order cannot become load-bearing."""
    first = _renderings(CONFIGS["default"])
    second = _renderings(CONFIGS["default"])

    assert first == second
