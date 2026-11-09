"""Prompt templates for the analyzer agent.

Every fixed string the model reads lives on :class:`SealPrompts`: the
literal fragments as dataclass fields — constructor-settable, so a sweep
or an experiment overrides any one of them without a code edit (A6) —
and the composition as methods, which read the fields at render time so
an overridden fragment propagates everywhere it appears. The
:class:`Prompts` protocol is the seam a genuinely different wording
implementation drops in through (`SealAgent.prompts`).

What the agent decides — the runtime surface, the frame API, kernel
persistence, the numeric limits — is NOT on this class. It arrives per
render as a :class:`PromptModes`, built by ``SealAgent._prompt_modes()``
from the agent's own fields, so a prompt that describes the runtime and
the runtime it describes cannot disagree, and a mode cannot be set here
by mistake: there is no field to set.

The committed references and ``tests/prompt_golden/`` are what prove a
change to any of this deliberate: an edit that was not meant to reach the
model shows up as a prompt diff in the gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol

from seal.frame_delta import AnimationDescription
from seal.harness.game import VALID_ACTIONS_NOTE
from seal.palette import ARC_LETTER_CHARS, numeric_legend
from seal.state import MemoryEntry, MoveChange, StepSummary

#: How a side turn (the reflection, the strategy audit) runs: not at all, with
#: its full prompt and explicit questions, or with a lighter prompt asking for
#: a brief reflection.
SideTurnMode = Literal["none", "verbose", "concise"]
SIDE_TURN_MODES: tuple[str, ...] = ("none", "verbose", "concise")
#: Whether the ``python`` tool takes a ``plan``, and whether its schema and
#: description put it ahead of ``code`` or after it. The schema accepts
#: either order; only the hint differs.
PlanArgument = Literal["none", "first", "second"]
PLAN_ARGUMENT_MODES: tuple[str, ...] = ("none", "first", "second")


@dataclass(frozen=True)
class PromptModes:
    """What the AGENT decides about a render, passed per call.

    Separate from the prompt surface on purpose: these describe machinery
    the agent owns (which globals the kernel installs, how long its session
    lives, what the tool actually truncates at), so a prompt surface that
    could hold its own copy could hold a wrong one. Frozen, and built in
    one place — ``SealAgent._prompt_modes()``.

    No field carries a default, deliberately: a default here would be a
    second copy of a ``SealAgent`` default, and a copy is stale from the
    next sweep that ships one.
    """

    persistence: Literal["round", "move", "level", "always"]
    #: The file tools this agent offers, model-facing names in the order the
    #: suite is shown. The NAMES rather than a count, because the prompt
    #: states them: a surface that named tools the agent did not build would
    #: promise the model a tool it was never given.
    file_tool_names: tuple[str, ...]
    #: The image mode, not a flag: "" (off), "current_grid" (every user
    #: prompt) or "level_start" (the first of each level). Empty is falsy, so
    #: a surface that only asks whether images ride at all still reads it as
    #: a boolean — but one whose wording CLAIMS how often they ride can now
    #: tell the two on-modes apart, and one that does not implement a mode can
    #: refuse it by name.
    multimodal: str
    memory_context_tokens: int
    tool_output_tokens: int
    python_timeout_s: int
    #: The game's own action set, model-facing names, frozen at the first
    #: observation. Not the currently-legal subset: a prompt that named that
    #: would rewrite its own stable prefix every time the game refused one.
    action_names: tuple[str, ...]
    #: How the kernel's ``ascii_grid`` lays a board out — the prompt's
    #: description of a render has to match what the kernel prints.
    ascii_row_numbers: bool
    ascii_column_group: int | None
    #: Which alphabet a board prints in (``SealAgent.ascii_symbols``): the
    #: legend states the one the model will read.
    ascii_symbols: str
    #: Whether a world model is in use — every world-model sentence is spent
    #: only then, so a seal without one reads as it did before it existed.
    world_model: bool
    #: Whether the world model is the model's choice (``"optional"``) rather
    #: than required to act; meaningful only with ``world_model``.
    world_model_optional: bool
    #: Whether a passing ``test_world_model()`` is followed by a search.
    search: bool
    #: Whether a new settled frame is explained to the model
    #: (``seal.frame_delta``).
    settled_delta: bool
    #: How animations are described (``seal.frame_delta``): whether
    #: ``describe_animation`` is named, and what a move's full mark says.
    animation_description: AnimationDescription
    #: Whether executed moves are marked with their animation frames; without
    #: marks, an animation described with no concise description says so in
    #: a line of its own.
    animation_marks: bool
    #: Whether the agent keeps the journal: the `memory` tool and the
    #: snapshot at the system prompt's tail. Off, no sentence may speak of
    #: either — a prompt that named a tool the model was not given would
    #: have it calling one.
    journal: bool
    #: Whether a full context is compacted by a summary the model writes
    #: (``compaction_strategy="summarize"``) rather than by eviction.
    context_summary_enabled: bool
    #: Whether the conversation rebuilt around a summary lists the model's
    #: files and ``python`` objects, which the summary request then says.
    add_files_and_repl_to_summary: bool
    #: Whether the ``python`` tool takes a ``plan``, kept as ``Turn.plan``,
    #: and whether it is asked for ahead of ``code``.
    plan_argument: PlanArgument
    #: Whether the model is offered ``set_pixel_tolerance``, and the
    #: tolerance a game starts with (``SealAgent.expose_pixel_tolerance``,
    #: ``pixel_tolerance``).
    pixel_tolerance_offered: bool
    pixel_tolerance_start: int
    #: How many cells apart the board images label their rows and columns
    #: (``SealAgent.axis_labels``), None where they carry no labels.
    axis_label_step: int | None
    #: Whether the kernel has ``segment()`` (``SealAgent.segmentation``). Off,
    #: no sentence names it, and none names another first view in its place.
    segmentation: bool
    #: Whether the prompt says ``history`` holds what a replay would show,
    #: and a repeated attempt is pointed out (``SealAgent.replay_reminder``).
    replay_reminder: bool
    #: Whether an optional world model is stated as the default to attempt
    #: (``SealAgent.world_model_default``); meaningful only when optional.
    world_model_default: bool = False

    @property
    def file_tools(self) -> bool:
        """Whether the file surface is offered at all — what most of the
        prompt asks. Derived rather than carried: a stored flag beside the
        names is a second copy of the same fact, free to disagree with it."""
        return bool(self.file_tool_names)

    @property
    def mouse_available(self) -> bool:
        """Whether this game offers MOUSE, which is what decides if a prompt
        owes the model a coordinate convention at all."""
        return "MOUSE" in self.action_names


class Prompts(Protocol):
    """Every prompt the seal sends, and the configuration it renders under.

    An interface so the agent depends on the calls it makes rather than on
    one wording of them. The variation that belongs behind this protocol is
    a genuinely different way of conducting the conversation; a wording
    tweak is a field override on :class:`SealPrompts` instead.

    The split between a field and an argument is the split between what
    the SURFACE owns and what the AGENT decides: wording is a field,
    :class:`PromptModes` is an argument, and per-turn facts are arguments.
    An implementation therefore cannot hold a stale mode — it is handed
    the modes at every render.

    Where a string is written here and read back by the agent (the
    pruned-note marker, the memory acknowledgement's index), the reader is
    on this protocol too: one owner for both halves, so an override cannot
    silently break the scan.
    """

    # Strings the agent reads directly (formatted or returned at its own
    # call sites, outside any composed prompt).
    palette_legend_caption: str
    color_name_legend_caption: str
    diff_image_caption: str
    diff_image_caption_won: str
    compaction_restart_preamble: str
    game_over_refusal: str
    already_acted_skip: str
    suggest_moves_note: str
    stop_detail_run_complete: str
    stop_detail_game_over: str
    stop_detail_level_completed: str
    stop_detail_done: str
    stop_detail_max_moves: str
    stop_detail_terminal: str
    stop_detail_invalid_action: str
    invalid_action_error: str
    valid_actions_note: str
    stage_error_unknown_action: str
    stage_error_undo_disabled: str
    stage_error_mouse_coords: str
    world_model_required: str
    exploration_single_action_refusal: str
    exploration_one_per_call_refusal: str
    memory_budget_not_positive: str
    memory_empty_text: str
    memory_entry_too_large: str
    # The notice for a session cleared mid-turn, and the reasons the harness
    # gives when it restarts a kernel it could not use.
    repl_reset_notice_with_files: str
    kernel_stopped_responding_reason: str
    kernel_out_of_memory_reason: str
    # The message that ends a reflection turn cut short by its token limits.
    reflection_cut_note: str
    # What submit_action answers in a turn that is not for acting.
    no_action_turn_refusal: str
    # What a turn the strategy audit interrupted goes on with: the audit done,
    # or cut short by its token limits.
    audit_continue_line: str
    audit_cut_note: str

    def system_prompt(self, modes: PromptModes, memory_snapshot: str) -> str:
        """The full system prompt over the frozen journal ``memory_snapshot``."""
        ...

    def python_tool_description(self, modes: PromptModes) -> str:
        """The `python` tool's description."""
        ...

    def python_tool_code_param_description(self, modes: PromptModes) -> str:
        """The `python` tool's `code` parameter description."""
        ...

    def python_tool_plan_param_description(self, modes: PromptModes) -> str:
        """The `python` tool's optional `plan` parameter description, under
        ``plan_argument``."""
        ...

    def memory_entry_header(self, entry: MemoryEntry) -> str:
        """One journal entry's compact host timestamp."""
        ...

    def memory_append_acknowledgement(self, entry: MemoryEntry) -> str:
        """What the memory tool answers with after appending ``entry``."""
        ...

    def parse_memory_append_index(self, text: str) -> int | None:
        """The journal index in an acknowledgement this surface rendered, or
        ``None`` when ``text`` is not one — the reader paired with
        :meth:`memory_append_acknowledgement`."""
        ...

    def is_rounds_pruned_note(self, text: str) -> bool:
        """Whether ``text`` is a note this surface rendered — the reader
        paired with :meth:`rounds_pruned_note`."""
        ...

    def user_prompt_text(
        self,
        modes: PromptModes,
        *,
        reset_reason: str | None,
        summary: StepSummary | None,
        any_actions_executed: bool,
        current_step: int,
        current_level: int,
        observed_max_level: int,
        valid_actions: list[str],
        valid_actions_observed: bool,
        game_over_active: bool,
        exploration_moves_left: int = 0,
        exploration_over: bool = False,
        level_tokens_used: int = 0,
        level_token_allowance: int | None = None,
        frame_delta: str | None = None,
        animation_text: str | None = None,
        animation_described: int | None = None,
        animation_concise: bool = False,
    ) -> str:
        """The per-turn user prompt over the agent-computed facts.

        ``level_tokens_used`` / ``level_token_allowance`` are the level's
        completion-token spend and cap (``None`` when uncapped) — a
        per-turn fact a surface MAY render; the shipped ones do not.
        ``frame_delta`` is the newest frame explained as a block of code
        from a frame the model knew, or the note that none is concise.
        ``animation_text`` is a move's animation explained the same way,
        after it, and ``animation_described`` the turn it is
        (its move's mark says the description follows, or with
        ``animation_concise`` false that there is none).
        """
        ...

    def reflection_prompt_text(
        self,
        modes: PromptModes,
        *,
        concise: bool,
        reset_reason: str | None,
        summary: StepSummary | None,
        level: int,
        win_missed: bool,
        frame_delta: str | None,
        animation_text: str | None,
        animation_described: int | None,
        animation_concise: bool,
    ) -> str:
        """The prompt of the turn after a move that won ``level``
        (``SealAgent.level_reflection``): the batch reported as any turn's
        prompt reports it, then the reflection asked for. ``win_missed`` is a
        world model that passed before the call and did not predict the win.
        ``concise`` asks for a brief reflection in place of the questions. The
        other arguments are :meth:`user_prompt_text`'s."""
        ...

    def audit_prompt_text(
        self,
        modes: PromptModes,
        *,
        concise: bool,
        level: int,
        untried_actions: list[str],
        untried_actions_game: list[str],
        unclicked_colors: list[int],
        unclicked_colors_game: list[int],
    ) -> str:
        """The strategy audit's prompt (``SealAgent.audit_level_tokens``) on
        ``level``: the actions no move has played on the level and in the
        game, and in a game with MOUSE the colours on the board no click has
        hit, on the level and in the game. ``concise`` asks for a brief
        reflection in place of the questions."""
        ...

    def user_prompt_game_over_lines(self, modes: PromptModes | None = None) -> list[str]:
        """Loss guidance replacing the act/grounding lines on a lost frame."""
        ...

    def followup_nudge(self, completion_limit: int | None = None, modes: PromptModes | None = None) -> str:
        """The no-action nudge, cutoff-aware when a limit is known."""
        ...

    def malformed_nudge(self, modes: PromptModes | None = None) -> str:
        """The continuation prompt after a reply refused as malformed on every
        retry and dropped (``SealAgent.malformed_nudges``): in practice a reply
        that went in circles inside its reasoning and ended with no text and no
        tool call. The model never sees the dropped reply."""
        ...

    def rounds_pruned_note(self, n: int) -> str:
        """The note that rounds of the current turn were dropped."""
        ...

    def compaction_prompt(
        self,
        modes: PromptModes,
        summary_tokens: int,
        *,
        current_level: int,
        previous_summary: bool = False,
        world_model_shown: bool = False,
    ) -> str:
        """The request for a summary that replaces the history."""
        ...

    def compaction_world_model(self, source: str) -> str:
        """``world_model.py``'s full text, as the rebuilt conversation's
        closing message shows it."""
        ...

    def compaction_wrapper(self, summary: str) -> str:
        """The user message carrying one accepted summary."""
        ...

    def compaction_objects_unavailable(self, *, kernel_lost: bool, error: str) -> str:
        """What the objects section says when the session listing failed:
        whether the kernel went with it, and the error."""
        ...

    def compaction_listing(
        self,
        modes: PromptModes,
        files: list[tuple[str, int]],
        objects: list[tuple[str, str, bool]],
        *,
        objects_unavailable: str | None = None,
    ) -> str:
        """The listing of the model's files (``(name, bytes)``, a directory at
        -1) and ``python`` objects (``(name, description, is_function)``, in
        binding order)."""
        ...

    def compaction_closing(self, modes: PromptModes, listing: str, *, session_intact: bool = True) -> str:
        """The user message closing a rebuilt conversation, over the listing
        (empty when none is given)."""
        ...

    def tool_calls_omitted_note(self, n: int) -> str:
        """The note that ``n`` tool calls of the current turn were summarized away."""
        ...

    def is_tool_calls_omitted_note(self, text: str) -> bool:
        """Whether ``text`` is a note :meth:`tool_calls_omitted_note` rendered."""
        ...


@dataclass
class SealPrompts:
    """The seal's prompt surface: every field is yours to set.

    Set it on the instance the agent carries —
    ``SealAgent(prompts=SealPrompts(...))`` — and a sweep overrides any one
    fragment without a code edit (A6). What the AGENT decides is not here
    at all: it arrives per render as a :class:`PromptModes`, so there is no
    mode field to set by mistake and no stale copy to diverge.

    Two blocks: prompt-only switches, then the wording, composed by the
    methods at render time so an overridden fragment propagates everywhere
    it appears.
    """

    # -- prompt-only switches ---------------------------------------------

    # The proportionate-reasoning instruction at the system prompt's tail,
    # at one of ``brevity_lines``' levels. Steering, not enforcement: the
    # hard ceiling is the reply cap, which exists to stop a pathological
    # ramble, not to shape ordinary turns. ``half`` is the one-sentence form,
    # the default on weak evidence (2026-09-27; the duck's
    # ``evidence.md``, "Sweeps 12 and 13").
    brevity: Literal["full", "half", "off"] = "half"

    # Whether the three budgets are stated outright: reasoning and code cost
    # tokens, running code is free, actions cost score only on levels you
    # clear. Stating the economics makes no procedural demand at all, so it
    # separates "the model was optimising the wrong resource" from "the
    # model needed to be told to write a BFS".
    cost_model: Literal["none", "explicit"] = "explicit"

    # Whether the no-action nudge ends with ``tool_call_format_guidance``.
    # The system prompt states it either way.
    nudge_format_guidance: bool = True

    # A caveat that a printed board is for orienting, not for counting cells
    # by hand (``printed_board_no_hand_arithmetic_line``).
    no_hand_arithmetic_line: bool = False

    # Whether the summary request asks for a thorough summary
    # (``compaction_request_thorough_line``, and from the second summary on
    # ``compaction_request_previous_summary_line``). Sweep 12's summaries ran
    # ~1.5-2k tokens whatever their target -- the request's "At most ~N
    # tokens" reads as a ceiling -- and `brevity` did not move them either, so
    # wording is the lever left (2026-09-23). On, on no real evidence,
    # as the stronger choice for the longer plays ahead (2026-09-27;
    # the duck's ``evidence.md``, "Sweeps 12 and 13").
    thorough_summary: bool = True

    # State that every level can be won, and what a proof of the opposite
    # means. Passes that concluded a level was impossible stopped
    # experimenting and idled out its budget, where passes that kept testing
    # the rule behind the contradiction won it, and an A/B on the game where
    # that happened bore the line out (``evidence.md``, "``winnable_line``"). On
    # ahead of a sweep that prices it across the games.
    winnable_line: bool = True

    # What the solving guidance says files are for besides code: keeping notes
    # in ``notes.md`` (``encourage``), or not keeping notes in them
    # (``discourage``); ``none`` says neither. Models given the reflection turn
    # began notes files of their own and rarely read them back, so both
    # directions are worth pricing against saying nothing
    # (2026-10-07). Unset until a sweep does. The journal refuses it: its own
    # lines say where findings go.
    notes_guidance: Literal["none", "encourage", "discourage"] = "none"

    # Say that an action which did nothing once may do something later, in
    # the game overview and again on each new level. Losing passes on the
    # sweep 13 critical levels tried an action a few times, wrote it into
    # their notes as inert, and never tried it again after new objects or a
    # new mode appeared (glass_orchard level 3: every winner clicked the new
    # piece, two thirds of the losers never did). (2026-09-25.) On, on
    # no evidence, as the more powerful choice going forward
    # (2026-09-27; the duck's ``evidence.md``, "Sweeps 12 and 13").
    retest_actions_line: bool = True

    # -- user configuration: wording, sweepable leaf fragments -------------

    tool_call_format_guidance: str = (
        "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
        "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
        "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
    )

    # Everything above the board sentences, which follow the palette the
    # frames render under and so are appended per render.
    game_overview_addendum: str = (
        "\n\nGame overview:\n"
        "- You are solving a multi-level grid puzzle game.\n"
        "- You are called repeatedly over the course of a run.\n"
        "- Your job is to solve the entire game by clearing every level, not just the current screen.\n"
        "- Levels often build on earlier mechanics, but layouts and interactions can still change between levels.\n"
        "- Your goal is to be both token-efficient and action-efficient.\n"
        "- `RESET` will always restore the game to the start of the current level; however, you will still be "
        "charged for actions made before then.\n"
    )
    #: The three budgets, under ``cost_model="explicit"``: stated once where
    #: the game is described, as a nested list under the economy line. Two
    #: facts are load-bearing and checked against the harness: the token
    #: budget is per level and renewed on clearing one, and exhausting it ends
    #: the GAME -- scored as it stands -- not the level.
    cost_model_line: str = (
        "- There are three costs you should be aware of:\n"
        "  - Tokens: your reasoning and the code you write cost tokens, and a fixed token budget applies to "
        "each level, renewed when you clear one. When it is spent, the game ends with the score earned so far.\n"
        "  - Actions: each action you take above a reasonable baseline reduces the score for winning a level. "
        "(But winning with many actions is better than not winning at all.)\n"
        "  - Code runtime: this is not charged.\n"
        "  So, prefer doing search and analysis by running code, instead of reasoning in prose or searching by "
        "doing real actions.\n"
    )
    #: Rendered under ``winnable_line``, closing the game overview.
    game_overview_winnable_line: str = (
        "- Every level can be won with the objects and actions it gives you. If your reasoning concludes that a "
        "level is impossible, one of your assumptions about the rules is wrong: list the assumptions that "
        "conclusion rests on and test the ones you have not directly observed — above all, anything the game "
        "did that your model did not predict.\n"
    )
    #: Rendered under ``retest_actions_line``, after the winnable line.
    game_overview_retest_actions_line: str = (
        "- An action that seems inert in one mode or level may not be later; if you get stuck, check your "
        "assumptions on what actions can do.\n"
    )
    #: Rendered under ``retest_actions_line`` on the first turn of every level after the first.
    level_start_retest_actions_line: str = (
        "Actions that seemed inert on previous levels may not be now; retest your assumptions."
    )

    #: How the board is described, and the legend for reading it, by the
    #: alphabet a board prints in (``ascii_symbols``). The legend states the
    #: glyph for all sixteen indices, in hex including where glyph and index
    #: coincide: a legend covering only 10-15 would leave the model to work
    #: out where the correspondence stops.
    board_legend_lines: dict[str, str] = field(
        default_factory=lambda: {
            "hex": (
                "- In this environment, boards are 64 x 64 grids of color indices 0-15. `ascii_grid(grid)` renders "
                "each index as one character: digits for 0-9, letters a-f above that.\n"
                f"- Color legend (index, `ascii_grid` character, name): {numeric_legend()}. The same mapping is "
                "available in Python as `color_names`.\n"
            ),
            "letters": (
                "- In this environment, boards are 64 x 64 grids of color indices 0-15. `ascii_grid(grid)` renders "
                "each index as one letter, the symbol of its color.\n"
                f"- Color legend (index, `ascii_grid` character, name): {numeric_legend(ARC_LETTER_CHARS)}. The same "
                "mapping is available in Python as `color_names`.\n"
            ),
        }
    )
    #: When ``no_hand_arithmetic_line`` is on, after the frame API lines, whose
    #: last one is where printing a board is advised: the caveat has to
    #: follow the board's introduction to read. The status prompt never
    #: carries a board, so what it speaks of is a board the model printed.
    printed_board_no_hand_arithmetic_line: str = (
        "- Boards you print with `ascii_grid` are for orienting yourself, not for counting, indexing or diffing "
        "cells by hand: the kernel does that work exactly, and hand-transcribing rows is where miscounts come from.\n"
    )
    #: How a rendered row is laid out, only where the render differs from
    #: one character per cell: the legend line has just said it is.
    ascii_row_numbers_line: str = (
        "- `ascii_grid` prefixes each row with its index in the array you passed and a tab; on a crop, add the "
        "slice's start.\n"
    )
    ascii_column_group_line: str = (
        "- `ascii_grid` puts a tab after every {group} cells; it is a separator, not a cell.\n"
    )

    visual_game_addendum: str = (
        "\n\nVisual-game guidance:\n"
        "- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.\n"
        "- Game entities are usually rendered as connected multi-tile shapes such as 2×2, 2×3, 3×3, or longer patterned structures. Sometimes they might also be 1x1 tokens.\n"
        "- Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board configuration.\n"
        "- Background colors are often white or gray/black-ish large regions, but not always. Verify background hypotheses by area, stability, and object boundaries rather than assuming them.\n"
        "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often shrinks or changes each step.\n"
        "- Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a specific absolute row or column.\n"
        "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n"
    )
    # The duck's visual guidance, whole, for a seal with no world model: the
    # prompt review (2026-09-16) cut the timer-bar and segmented-bar warnings,
    # re-grounding after a score increase and the WIN bullet as lines a world
    # model makes wrong or redundant, which only holds where there is one
    # (2026-09-28).
    visual_game_addendum_without_world_model: str = (
        "\n\nVisual-game guidance:\n"
        "- Treat each board as a scene with objects, blockers, targets, adjacency, containment, motion, and symmetry.\n"
        "- Game entities are usually rendered as connected multi-tile shapes such as 2×2, 2×3, 3×3, or longer patterned structures. Sometimes they might also be 1x1 tokens.\n"
        "- Some games are logic or layout puzzles with no explicit player avatar or controllable sprite on the board. Do not assume a player exists; the relevant state may be an object, region, cursor, selector, or whole-board configuration.\n"
        "- Background colors are often white or gray/black-ish large regions, but not always. Verify background hypotheses by area, stability, and object boundaries rather than assuming them.\n"
        "- In many games, a long horizontal or vertical line near an edge is a timer or remaining-steps bar. It often shrinks or changes each step. If you identify such a bar, do not get distracted by it or treat it as core gameplay state unless there is concrete evidence that it interacts with the puzzle mechanics.\n"
        "- A common failure mode is to mistake a segmented edge bar for clickable puzzle pieces. If a repeated strip of small blocks sits flush against the top, bottom, left, or right border and actions only change that strip while the interior board stays the same, classify it as HUD/timer state, not as an object to click through segment by segment. DON'T DO THIS!\n"
        "- Use coordinates only to target actions or describe local evidence. Do not frame the objective as reaching a specific absolute row or column.\n"
        "- Re-ground on the newest frame after any score increase or abrupt scene change; the returned board may already be the next level.\n"
        "- `WIN` means the whole game is solved. Mid-run level completion is more likely to appear as a score increase while play continues.\n"
        "- Strategies may transfer loosely across levels, but layouts and mechanics can change. Re-check the new board before repeating a plan.\n"
    )
    #: After the visual guidance, with no world model, on a game whose engine
    #: offers MOUSE: the duck's line, in the seal's ``Action`` form.
    prior_mouse: str = "- A `MOUSE` action takes integer `row` and `col`: `Action('MOUSE', row=r, col=c)`. `row` is vertical position, `col` is horizontal position.\n"

    #: The game's own action set, which the system prompt can name because it
    #: is frozen for the game — unlike the currently-legal subset, which the
    #: per-turn prompt carries.
    available_actions_line: str = "- The available actions for this game are: {actions}.\n"

    runtime_header: str = "\n\nTool and Python interface:\n"
    # A template rather than a literal: `{count}` is the total spelled out and
    # `{names}` the file tools as the model will be given them, both filled
    # from the same suite the agent builds.
    runtime_tool_line_with_files: str = (
        "- You have {count} native tools. `python` runs code; `memory` appends durable game knowledge; "
        "{names} read and write files in your working directory.\n"
    )
    # The sentence above for an agent without the journal, which has no
    # `memory` tool to name.
    runtime_tool_line_with_files_without_journal: str = (
        "- You have {count} native tools. `python` runs code; {names} read and write files in your working directory.\n"
    )
    # What the working directory IS, told once where the rest of the
    # runtime is described. The import claim is the load-bearing one: it is
    # the difference between a place to park text and a place to build.
    runtime_file_lines: str = (
        "- The file tools act on one flat working directory — no subdirectories, and any file name is allowed. "
        "It is the same directory your `python` snippets run in, so a file you write there is a module you can import by its name without the `.py`.\n"
        "- `python` runs ordinary Python in that directory: `open()` reads and writes files there, and a "
        "write anywhere else is refused.\n"
        "- A module you import from there sees the preloaded names a snippet does — `history`, `submit_action`, "
        "`np`{segment_name} and the rest — and `history` and `available_actions` are refreshed inside it after "
        "every action.\n"
        "- A module runs from disk the first time a `python` call imports it, into the same module object every "
        "time: any name bound to it, in a snippet or in another module, sees the latest version.\n"
        "- Module-level data in every module you write{world_model_note} is read-only once the module has loaded: "
        "its dicts, lists and arrays cannot be changed by any code, the module's own functions included, and a "
        "snippet cannot assign to the module's names (attempts raise). To keep something between calls, keep it "
        "in a snippet variable, or write it to a file — JSON, or `np.save`/`pickle` for arrays — and read it back "
        "at module level.\n"
        "- Prefer files you can read back: `view_file` shows text.\n"
    )
    # Named only where the file exists.
    runtime_world_model_module_note: str = ", `world_model.py` included,"
    # Named only where the kernel has it.
    runtime_segment_module_name: str = ", `segment`"
    runtime_action_global_line: str = (
        "- Inside `python`, `submit_action(actions)`, {segment_call}`ascii_grid(grid)` and `from_ascii_grid(text)` "
        "are preloaded functions, "
        "`Action` a preloaded class and `color_names` a preloaded dict, not native tools or importable modules. Use "
        "them directly.\n"
    )
    # Named only where the kernel has it.
    runtime_segment_call: str = "`segment(grid)`, "
    runtime_import_line: str = (
        "- `numpy` is preloaded as `np`, along with `os`, `re`, `json`, `math`, `copy`, `itertools`, "
        "`functools`, `collections`, `Counter`, `defaultdict` and `deque`. The whole standard library is "
        "importable.\n"
    )
    runtime_projection_lines: str = (
        "- `history` is this level's play so far: index it (`history[-1]`), iterate it, `len()` it. Each item is a `Turn`.\n"
        "- `Turn.action: Action | None` (None on the level's opening turn; `.name`, and `.row`/`.col` for MOUSE), `.outcome: 'playing' | 'won' | 'lost'`, `.step: int` and `.level: int` (when it was played), `.frame: Frame` (the settled board after it), `.animation_frames: list[Frame]` (the frames it passed through, which may be empty).\n"
        "- `history` starts again at every new level. What you want to keep across one goes in a file{or_journal}.\n"
        "- `available_actions: list[str]` are the legal action names, fixed for the whole game.\n"
        "- `help(obj)` prints what a preloaded object holds: try `help(Turn)`, `help(Frame)`, `help(Node)`.\n"
    )
    #: Under ``replay_reminder``, after the history lines.
    runtime_replay_hint_line: str = (
        "- `history` keeps every turn of this level, including those before a RESET, with the frames each one "
        "produced. To see again what a sequence did, read its turns there: the game is deterministic, so playing "
        "it again shows nothing new.\n"
    )
    #: With animations described (``animation_description`` other than
    #: ``none``): the REPL function that describes one. Whether one is also
    #: described unasked (``auto``) is not announced: the model sees it
    #: happen.
    runtime_animation_line: str = (
        "- `describe_animation(i)` prints a concise code description of `history[i].animation_frames`: a block "
        "of numpy that rebuilds them from the board before the move. `describe_animation(-1)` is the newest "
        "turn's. With `stride=k` it builds every k-th frame.\n"
    )
    #: A move with animation frames, as the executed-actions line names it:
    #: the first such move in a prompt carries where its frames are, and the
    #: tool or the description that reads them; every other one its count.
    #: ``{label}`` is the move, ``{frames}`` how many animation frames its
    #: turn keeps, ``{index}`` the turn's index in ``history``.
    animation_mark: str = "{label} ({frames} animation frames, history[{index}].animation_frames)"
    animation_mark_ask: str = (
        "{label} ({frames} animation frames, history[{index}].animation_frames, use "
        "`describe_animation({index})` for a concise code description)"
    )
    animation_mark_described: str = (
        "{label} ({frames} animation frames, history[{index}].animation_frames, concise code description below)"
    )
    animation_mark_not_concise: str = (
        "{label} ({frames} animation frames, history[{index}].animation_frames, no concise code description "
        "because the visuals are too complex)"
    )
    animation_mark_short: str = "{label} ({frames} anim frames)"
    #: A batch's moves one per line (``SealAgent.move_changes``), each with
    #: what it changed on the board: how many cells, and the colours they
    #: held before and after. An animation mark follows in the same brackets.
    executed_move_line: str = "- {move}"
    move_change_note: str = "{count} pix changed, colors {before}->{after}"
    move_unchanged_note: str = "no change"
    #: The moves a long batch's line leaves out, when some had animation
    #: frames; without any it is ``...`` alone.
    elided_animations: str = "... ({count} more, {animated} with animation frames) ..."
    #: A long batch's line: the first and last five moves, and the move
    #: whose animation the prompt points at when it falls between them.
    executed_elided_header: str = "Executed actions (first 5 and last 5)"
    executed_elided_header_with_mark: str = "Executed actions (first 5, the move with its animation named, and last 5)"

    frame_common_lines: str = (
        "- `Frame.grid` is the board as a 2-D NumPy array of color indices, dtype int8: `grid == 5` selects "
        "every cell of color 5.\n"
    )
    #: Frames and turns refuse assignment, so a cell cannot edit the record
    #: it is reasoning from. Said plainly here because the alternative is
    #: the model discovering it through an AttributeError mid-analysis.
    runtime_readonly_line: str = (
        "- Frames and turns are read-only; `copy.deepcopy(obj)` gives an editable copy if you want to alter one.\n"
    )
    frame_api_lines: str = (
        "- `Frame.diff_sparse` maps each `(old_color, new_color)` change to its `[(row, col), ...]` cells "
        "since the previous frame (`None` on the level's first frame); `.color_counts` is a dict from each "
        "color present to its cell count (absent colors are left out).\n"
    )
    #: Where the kernel has ``segment()``.
    segment_api_line: str = (
        "- `segment(grid)` returns a `Segmentation` of any 2-D board — a frame's `.grid`, a crop, a mask, or a "
        "board you predicted. Each `seg.nodes` item is a `Node`, one 4-connected same-color object: `.id: int`, "
        "`.color: int`, `.hash: str` (translation-invariant, so the same shape matches itself anywhere), "
        "`.pixels` as an `(N, 2)` int array of `[row, col]` cells, `.pixel_count: int`, `.bbox` as "
        "`(row0, row1, col0, col1)` (`grid[row0:row1, col0:col1]` is its box), `.boundary` as an "
        "`(M, 2)` int array of clockwise corner points, and `.children: list[int]`. `seg.adjacency_list` pairs "
        "ids that touch; `seg.by_color(c)` filters.\n"
    )
    ascii_api_lines: str = (
        "- `ascii_grid(grid)` renders any 2-D board as one string, a line per row and one character per cell.\n"
        "- `from_ascii_grid(text)` reads one back into an int8 array: what `ascii_grid` printed, row numbers and "
        "tabs included, or a board or sprite you write out as a picture.\n"
    )
    #: How to look at a board: segmentation first, where the kernel has it.
    structural_view_line: str = (
        "- Use `segment(...)` as the primary structural view. Print only compact summaries — "
        "`color_counts`, `diff_sparse`, object lists, coordinates — or small `ascii_grid(...)` regions, "
        "never a full board.\n"
    )
    #: Without segmentation: how much to print, and no view to start from.
    compact_output_line: str = (
        "- Print only compact summaries — `color_counts`, `diff_sparse`, coordinates — or small "
        "`ascii_grid(...)` regions, never a full board.\n"
    )

    structured_runtime_state_live_lines: str = (
        "- Call `submit_action(actions)` to execute one or more real environment actions from Python.\n"
        "- Every move is an `Action`: `Action('LEFT')`, or `Action('MOUSE', row=4, col=7)` for a click. The same "
        "type is `history[i].action` and what `submit_action` takes.\n"
        "- Pass `submit_action(actions)` a list of them, like `[Action('LEFT')]` or `[Action('UP')] * 3`.\n"
        "- One action usually returns one frame, but a single action can result in a short multi-frame animation.\n"
        "- After `submit_action(actions)` returns, the new turns are already appended to `history`.\n"
    )

    # === The world model ===
    #
    # Spent only when a world model is in use. The contract follows the
    # whale's coder prompt, reworded where the seal differs: no confidence
    # (a prediction is a claim or `Unknown`), a -1 wildcard, loose energy,
    # the per-action check, and exploration in place of premoves.
    world_model_header: str = "\n\nWorld model:\n"
    world_model_intro_line: str = (
        "- The mechanics of the game are not arbitrary; mechanics are represented by visible objects. You write "
        "the rules you infer as code in `world_model.py` in your working directory, and your moves are held to "
        "it.\n"
    )
    #: The optional mode's opening: the same claim about the mechanics, with
    #: the choice where the mandatory one has the obligation.
    world_model_intro_line_optional: str = (
        "- The mechanics of the game are not arbitrary; mechanics are represented by visible objects. You can "
        "write the rules you infer as code in `world_model.py` in your working directory. You decide per game "
        "and per level; the lines below say what one gives you and what it costs.\n"
    )
    #: After the optional opening, with ``world_model_default``: attempting a
    #: model is the default, given up only on evidence (2026-10-27).
    world_model_default_line: str = (
        "- Your default mode should be to use a world model; in general, at least attempt to write one, and only "
        "stop if you are convinced it's not useful on this game.\n"
    )
    world_model_tool_lines: str = (
        "- `test_world_model()` replays this level's `history` through `world_model.py` and returns a result: falsy "
        "when every turn is predicted, otherwise it names the first failing turn and why. Call it on its own line "
        "(or `print()` it) to display the result.\n"
        "- It is also how you get the state of a past turn: it sets `history[i].state` to your predicted state "
        "after each turn, so `history[-1].state` is where the game stands under your own model. Do not write a "
        "replay loop of your own over `history` — this is the replay, it runs your code against every turn, and "
        "it reports what a hand-rolled one would have to check for itself.\n"
        "- `world_model` is imported for you at the start of every `python` call: its module-level code re-runs "
        "over the history so far, and your edits to it — or to a module it imports — are live. Call "
        "`world_model.do_move(...)` and the rest directly.\n"
        "- To look ahead, step a copy of a state through your model with the moves you mean to play: "
        "`state = copy.deepcopy(history[-1].state)`, then `for move in [Action('UP')] * 8: "
        "world_model.do_move(state, move)`. If `state` ends where you want, `submit_action([Action('UP')] * 8)` "
        "plays those same moves.\n"
        "- `test_world_model()` and the check before a move load the file themselves, and first re-run any module "
        "of yours that you have edited, so an edit counts at once there too, in the same call that made it.\n"
    )
    world_model_seed_line: str = (
        "- At the start of the game, `world_model.py` holds a complete, working world model of a different, "
        "simpler game. It fails on this one. Read it to see the contract below in use, then replace it with the "
        "rules of this game.\n"
    )
    world_model_seed_line_optional: str = (
        "- At the start of the game, `world_model.py` holds a complete, working world model of a different, "
        "simpler game. It fails on this one. If you write one, read it first to see the contract below in use, "
        "then replace it with the rules of this game.\n"
    )
    #: What a passing model buys, and what the moves before one cost. How
    #: many of those a level starts with, and what is left, belongs to the
    #: per-turn prompt: here it would be a rule to plan around.
    world_model_check_lines: str = (
        "- A passing `test_world_model()` is what lets `submit_action` run a sequence; until then it runs one "
        "action per `python` call, and only while the moves for that are left.\n"
        "- Under a passing world model, `submit_action` checks each action as it runs: your model predicts the "
        "new frame and outcome, and the first action it gets wrong stops the sequence and raises `WorldModelStop`. "
        "The executed actions stay in `history`, where `test_world_model()` then fails on the mispredicted turn "
        "until you fix the rules.\n"
    )
    #: The check without the obligation: what a passing model does to a
    #: batch, what happens without one, and the choice after a stop.
    world_model_check_lines_optional: str = (
        "- While `world_model.py` passes on this level, `submit_action` checks each action as it runs: your model "
        "predicts the new frame and outcome, and the first action it gets wrong stops the sequence and raises "
        "`WorldModelStop`. The executed actions stay in `history`, where `test_world_model()` then fails on the "
        "mispredicted turn.\n"
        "- While it does not pass — or before you write one — actions run unchecked. The check is made again at "
        "the first `submit_action` of every `python` call, so a model you fix checks again from then on.\n"
        "- After a `WorldModelStop`, nothing more runs in that `python` call. Then either fix the rule "
        "`test_world_model()` names, or play on without the model.\n"
    )
    #: When a world model is the model's choice: the default, what one gives
    #: and costs, and the games that favour or argue against one. What it
    #: gives is only what this configuration delivers (``{gains}``, from the
    #: ``world_model_gain_*`` fields). The advice for unchecked batches lives
    #: here rather than in the solving guidance, beside what it is the
    #: alternative to.
    world_model_pays_lines: str = (
        "\nWhen a world model pays:\n"
        "- As a default, you should operate with a working world model; deviate only if you find good arguments "
        "not to.\n"
        "- You can use a world model even if you don't understand the game fully yet, for example if the win "
        "condition isn't clear yet.\n"
        "- What it gives: {gains}.\n"
        "- What it costs: tokens to write it, and to keep it passing as each level adds mechanics. A model that "
        "does not pass gives you none of the above.\n"
        "- It tends to pay when the rules are deterministic and local (objects move, push, collide or change color "
        "by fixed rules), when a level needs a long or exact sequence (navigation, ordering, packing), and when "
        "the same mechanics carry across levels, so the model is written once and extended.\n"
        "- It tends not to pay when a level can be solved by inspection in a few actions, when the dynamics are "
        "irregular or visually complex and expensive to render, when every frame reveals new information (hidden "
        "areas, objects not yet seen), or when allies or enemies move in ways you cannot predict.\n"
        "- Without a passing world model, a sequence of actions can waste many actions if it does not do what you "
        "expect: add intermediate `assert`s that check the state is changing as expected, or play shorter "
        "sequences.\n"
    )
    world_model_gain_check: str = (
        "a sequence of actions is checked as it runs, so a wrong idea costs one action instead of the rest of the "
        "sequence"
    )
    world_model_gain_search: str = (
        "a passing model gets a shortest-path search after a `python` call that tests it and plays nothing"
    )
    world_model_gain_frame: str = "a new frame it predicted is confirmed in one line instead of described"
    #: Said only when the search is on: what a test buys beyond the result it
    #: prints, and that the path arrives after the call — models otherwise look
    #: it up in the call that ran the test.
    world_model_search_line: str = (
        "- When a `python` call ends whose last `test_world_model()` passed and which played no moves, a short "
        "BFS search runs; its path is in `suggested_actions` / `suggested_history` from your next call on.\n"
    )
    #: Said only when the search is on, at the ``Unknown`` contract: what
    #: declining a move is worth beyond honesty.
    world_model_search_unknown_line: str = (
        "  - The automatic BFS search will guide you to an `Unknown` exception if no path to a win is found, so "
        "you can use this to mark states where you will learn something useful.\n"
    )
    #: ``{mouse_note}`` is the MOUSE coordinate note, on a game that has it;
    #: ``{search_note}`` the ``Unknown`` line above, when a search is on.
    world_model_contract_lines: str = (
        "\n`world_model.py` contract:\n"
        "- Module-level code: anything that does not change between moves goes here (e.g. shapes, palettes, "
        "layouts), computed once at module load. It starts with a snippet's preloaded names, `history` and "
        "`available_actions` among them — use them directly, not through an import — except `submit_action` "
        "and `test_world_model`: the replay runs this file too, where those would act. Its module-level data "
        "is read-only after load, as in every module of yours: dicts, lists, and ndarrays defined at module level "
        "cannot be mutated from `do_move`/`render` or anywhere else (attempts raise loudly). Anything that changes "
        "during a game goes in `state`.\n"
        "- `initial_state()`: returns the starting state dict. Must include `'outcome': 'playing'` and "
        "`'energy': <numeric>`. Keep it as small as possible — store the abstract variables, not the rendered "
        "frame.\n"
        "- `do_move(state, move)`: updates `state` in place based on the move. The `move` argument is an Action "
        "object — read `move.name` (str) for the action name (e.g. 'LEFT'){mouse_note}. The same `Action` shape "
        "appears in `history[i].action` for past turns, and is what `submit_action` takes.\n"
        "  - Update `state['outcome']` to 'won' or 'lost' on transitions (leave 'playing' otherwise).\n"
        '  - `raise Unknown("reason")`: you have no prediction at all for this case. Under a passing world model '
        "the action is still executed, but nothing after it runs in that `python` call.\n"
        "  - `Unknown` is only valid for a move you have NOT yet seen played. The move you just declined is in "
        "`history` now, where declining it fails `test_world_model()` — so the next move you make is the one that "
        "models it. That is the pace of learning a part no rule predicts: one new transition per `python` call, "
        "each folded into the model before the next.\n"
        "{search_note}"
        "  - `state['energy']` is required to be numeric. Do not put move counters in any other key of the state.\n"
        "- `render(state)->np.ndarray`: returns the 64x64 board as a 2D array of colour indices (`np.int8`), which "
        "is compared to the real frame. Only needs to be accurate while the game is in the `playing` outcome — for "
        "`'won'`/`'lost'` states it is sufficient that you predict the outcome correctly.\n"
        "  - A cell you render as -1 is not predicted, and matches whatever the frame shows. Use it for a region "
        "you cannot model yet, not to skip one you can.\n"
        "- `distance_to_win(state)->numeric`: optional. Gives a rough metric (not formal A*) of how close the state "
        "is to the win — smaller values mean closer to winning (for example, if N objects need to be brought into "
        "some state, how many are left?). Keep it simple, it's only a rough guide; it should not do any search "
        "itself.\n"
        "- RESET will never be given as a move input to `do_move`; this move is handled by the game engine "
        "directly.\n"
        "- `world_model.py` must not print, log, warn or write files.\n"
        "- A world model that simply replays the history without implementing the rules behind them is "
        "unacceptable.\n"
    )
    world_model_mouse_note: str = ", and `move.row` / `move.col` for MOUSE"
    world_model_energy_lines: str = (
        "- Most games have an energy mechanic, with 0 energy causing a loss. Track this in `state['energy']`. If "
        "there is no visible energy tracking, set starting energy to 2000 and decrement by 1 per move.\n"
        "- Energy usually decreases by 1, but the number of pixels in the bar may change differently due to "
        "discretization. Typical equation is `round(64 * energy / capacity)`. Occasionally the width may be "
        "different, or floor/ceil may be used.\n"
        "- An energy/timer bar is a filled bar whose length is computed from the remaining amount, drawn over a "
        "full-length background track — less energy is a shorter bar, never a full bar with a growing 'spent' "
        "cover.\n"
        "- Energy is checked loosely, by `test_world_model()` and by the check on every move: a frame that would "
        "match under some other `state['energy']` passes, and the result says so. `set_strict_energy(True)` holds "
        "every frame to your own value, for the rest of the game, and `set_strict_energy(False)` goes back.\n"
        "- Most games provide more than enough energy, while the bar itself is very difficult to model. Feel free "
        "to predict wildcards (-1) unless a clear role for energy is revealed.\n"
        "- Every element that is not part of the background has some role to play, though you may need to render "
        "it -1 or raise `Unknown` until it has come into play.\n"
    )
    #: Where the pixel tolerance is offered, after the energy lines.
    world_model_pixel_tolerance_line: str = (
        "- `set_pixel_tolerance(n)` lets a frame pass when your render misses at most `n` of its cells (outside the "
        "cells you render -1), in `test_world_model()` and in the check on every move, for the rest of the game; it "
        "starts at {start}. Keep it low: one pixel can be the only sign that something changed.\n"
    )
    world_model_workflow_line: str = (
        "- Your workflow: write `world_model.py` from what you have seen → run `test_world_model()` → if it "
        "fails, fix and repeat → act under it. After a `WorldModelStop`, run `test_world_model()`, fix the rule "
        "it names, and act again.\n"
    )
    world_model_workflow_line_optional: str = (
        "- If you use one, the workflow is: write `world_model.py` from what you have seen → run "
        "`test_world_model()` → if it fails, fix and repeat → act under it.\n"
    )
    #: The `Turn` attribute a world model adds, beside the others.
    turn_state_line: str = (
        "- `Turn.state: dict | None` is your world model's predicted state after that turn, set by "
        "`test_world_model()` and by the check on every move under a passing world model (`None` until then, and "
        "past a turn your model did not predict). `Turn.predicted_frame: ndarray | None` is the board your model "
        "rendered for that turn, set alongside it.\n"
    )
    #: Under ``plan_argument``: where a call's plan is kept.
    turn_plan_line: str = (
        "- `Turn.plan: str | None` is the `plan` you gave the `python` call that played that turn (None if you "
        "gave none).\n"
    )
    #: Under ``plan_argument``: the ``plan`` parameter's description.
    plan_param_description: str = (
        "Optional. One line, at most 500 characters: what this call's actions should achieve, and what result "
        "would confirm it. Kept on each action the call plays, as `history[i].plan`."
    )
    #: Under ``plan_argument="first"``: what ends the description, so the
    #: plan is written before the code it plans.
    plan_param_first_line: str = " If you give one, write it before `code`."
    #: Added to the live acting lines: a world model is the other thing that
    #: stops a batch.
    world_model_batch_stop_line: str = (
        "- Under a passing world model, a batch also stops after the first action your world model mispredicts "
        "or raises `Unknown` for.\n"
    )
    #: Added to the python tool description.
    tool_world_model_sentence: str = (
        " `test_world_model()` tests your `world_model.py` against `history`; until it passes on a level, "
        "`submit_action` runs one action per call."
    )
    tool_world_model_sentence_optional: str = (
        " `test_world_model()` tests your `world_model.py` against `history`; while it passes, `submit_action` "
        "checks each action as it runs."
    )

    # Per-turn world-model lines. The status line follows the state line on
    # every turn; the rest are spent at their moment.
    exploration_status_line: str = (
        "You may explore for up to {n} more {moves}, but write a world model as soon as feasible. Until you do, "
        "you may only perform one action per tool call."
    )
    world_model_required: str = "A passing test_world_model() is required to make moves."
    world_model_required_line: str = "A passing `test_world_model()` is required to make moves."
    summary_world_model_mismatch: str = (
        "Your world model mispredicted the last executed action, so the rest of the sequence was not executed. "
        "`test_world_model()` now fails on that turn."
    )
    summary_world_model_unknown: str = (
        "Your world model raised `Unknown` for the last executed action, so nothing after it was executed. That "
        "move is in `history` now, where `Unknown` fails `test_world_model()` until `do_move` models it."
    )
    #: After either stop, when the world model is optional: the choice.
    summary_world_model_stop_choice: str = (
        "Fix `world_model.py`, or play on without it: your actions run unchecked until it passes again."
    )
    #: When the world model is optional, what the last batch ran under. Said
    #: only when no stop summary says it already, and not across a level.
    summary_checked_line: str = "Your actions were checked by `world_model.py`."
    summary_unchecked_line: str = "Your actions ran unchecked: `world_model.py` does not pass on this level."
    #: On the first turn of every new level, before its world-model lines.
    new_level_history_line: str = (
        "`history` has started over for this level: `history[0]` is its starting state, and the previous level's "
        "turns are gone."
    )
    new_level_world_model_lines: str = (
        "Your files carried over, `world_model.py` among them; `test_world_model()` now checks this level's turns "
        "only, and it runs at your first move. New levels will generally change the "
        "layout and introduce new mechanics and elements; existing mechanics are usually preserved, and elements "
        "you've seen before will behave similarly. Read `world_model.py` against the new board: keep what still "
        "applies, add what is new, and recheck anything calibrated from the previous level. Each new level starts "
        "with fresh energy, usually more than the last. Try to get your world model passing before doing any move, "
        "using the new frame in `history[0].frame`. If you really need to learn more first, use the "
        "exploration moves."
    )
    new_level_world_model_lines_optional: str = (
        "Your files carried over, `world_model.py` among them; `test_world_model()` now checks this level's turns "
        "only, and it runs at your first move. New levels will generally change the "
        "layout and introduce new mechanics and elements; existing mechanics are usually preserved, and elements "
        "you've seen before will behave similarly. If you keep one, read `world_model.py` against the new board: "
        "keep what still applies, add what is new, and recheck anything calibrated from the previous level. Each "
        "new level starts with fresh energy, usually more than the last. Update it for this level unless you have "
        "good reason not to."
    )
    #: The system prompt's ``world_model_default_line`` again, in the user
    #: prompt of every level's first turn (the game's first included), with
    #: ``world_model_default`` (2026-10-27).
    level_start_world_model_default_line: str = (
        "Your default mode should be to use a world model; in general, at least attempt to write one, and only "
        "stop if you are convinced it's not useful on this game."
    )
    checklist_decide_world_model: str = (
        "2. Decide what to learn next — a move to learn from, or rules to write into `world_model.py` — or, once "
        "`test_world_model()` passes, the next move or sequence."
    )
    #: Step 2 on the game's first move, the one time world_model.py still holds
    #: the example: reading it belongs to writing the model, not to orienting.
    checklist_decide_world_model_game_start: str = (
        "2. Read `world_model.py`, a working example of the contract for another game, then decide what to learn "
        "next — a move to learn from, or rules to write into `world_model.py` in its place — or, once "
        "`test_world_model()` passes, the next move or sequence."
    )
    checklist_decide_world_model_optional: str = (
        "2. Decide the next move or short sequence — explore, advance the level, or write or update "
        "`world_model.py`, unless it does not pay here. Write a search if the right order is not obvious."
    )
    #: Added to step 2 on the game's first move, when world_model.py is the
    #: example.
    checklist_decide_world_model_optional_game_start: str = (
        " `world_model.py` holds a working example of the contract for another game; read it if you decide to "
        "write one."
    )
    checklist_act_world_model: str = (
        "4. Act with `submit_action([...])`: one action until your world model passes, any sequence after it does."
    )
    # Refusals while exploring: exploration's own rules, said without the
    # world model, which exploring does not need.
    exploration_single_action_refusal: str = (
        "While exploring, an action batch holds one action; this one has {count}. Nothing was executed."
    )
    exploration_one_per_call_refusal: str = "While exploring, one action runs per python call. Nothing was executed."

    #: Introduces the palette legend image, which leads the prompt that
    #: carries it. In the user prompt rather than the system prompt because
    #: the image rides one turn per level: a standing claim that it is
    #: attached would be false on every other turn.
    palette_legend_caption: str = (
        "Palette legend image: one swatch per color index, labelled with the index "
        "(top-left), the character `ascii_grid` renders it as (top-right), and its name."
    )

    #: The name-only legend's caption. It names what the image shows and
    #: nothing else: the swatch-to-name mapping is the whole content, and a
    #: caption promising indices would describe the other legend.
    color_name_legend_caption: str = (
        "Color legend image: one swatch per color, labelled with the color name used in this prompt."
    )

    #: Introduces the diff image (``SealAgent.diff_image``), which follows the
    #: board image on the prompt after a move: no other prompt says what it
    #: is, so this says which move, against what, and what the navy means.
    #: ``{baseline}`` is the comparison's phrase (``frame_delta.PHRASES``).
    diff_image_caption: str = (
        "Diff image of your last move, `{move}` (step {step}): the cells that differ from {baseline}, in their new "
        "color; every other cell is dark navy, which is not a game color."
    )
    #: The same for the move that won a level, on the reflection's prompt
    #: (``level_reflection``): the frame is the win screen, which the board
    #: image there shows too.
    diff_image_caption_won: str = (
        "Diff image of the move that won the level, `{move}` (step {step}): the cells of the frame it ended on that "
        "differ from {baseline}, in their new color; every other cell is dark navy, which is not a game color."
    )

    #: The per-mode lines: how often an image arrives, and what it is a
    #: picture OF — the two claims that differ between the modes. The rest of
    #: the addendum is shared; a prompt that overstated either claim would be
    #: describing a conversation the model is not having.
    multimodal_mode_lines: dict[str, str] = field(
        default_factory=lambda: {
            "current_grid": (
                "- User turns include an attached image of the current ARC grid.\n"
                "- The image and `history[-1].frame` are two representations of the same current frame."
            ),
            "level_start": (
                "- The first user turn of each level includes an attached image of the grid as that "
                "level began. Later turns in the level do not; `history[-1].frame` is the current "
                "board.\n"
                "- The image is a fixed reference for the level, not a view of the current frame: "
                "after any action it shows where the level started."
            ),
        }
    )
    #: Follows the mode lines where the board images carry axis labels
    #: (``SealAgent.axis_labels``): the whale's sentence.
    axis_labels_line: str = (
        "\n- Grid images also carry row/column axis labels at the top and left edges (every {step} cells, matching "
        "`frame.grid[row, col]` indexing) so you can map an action's `row` / `col` directly onto the image."
    )
    multimodal_context_addendum: str = (
        "\n\nMultimodal context:\n"
        "{mode_lines}\n"
        "- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n"
    )

    python_addendum_live_lines: str = (
        "- You can call `submit_action(...)` multiple times in one Python snippet, including inside loops. Each call updates the preloaded variables before execution continues.\n"
        "- A batch of actions stops executing automatically when a level is completed or the game ends; re-ground on the refreshed state before acting again.\n"
    )

    compact_tool_session_head: str = (
        "\n\nTool limits and session lifetime:\n"
        "{persistence_line}"
        "\n- You can call the `python` tool as many times as you want per step. Investigate until your code has a clear probe or plan.\n"
    )
    # Appended after the persistence line, which describes what a reset
    # takes. Two versions, because the durability argument only lands where
    # a reset actually happens: under "always" nothing clears the session on
    # a schedule, so "not affected by that" would point at an event the line
    # above just ruled out, and the reason to use a file there is that it can
    # be read back and edited, not that it survives.
    compact_tool_session_files_line: str = (
        "- Files are not affected by that. What you write with the file tools stays for the whole game, so "
        "anything you want to survive a cleared session belongs in a file rather than in a variable.\n"
    )
    compact_tool_session_files_line_persistent: str = (
        "- Files sit alongside the session rather than inside it: they stay for the whole game either way, "
        "and unlike a definition in memory you can read one back and edit it in place.\n"
    )
    compact_tool_session_tail: str = (
        "- Each `python` tool call has a hard time limit of {python_timeout_s} seconds.\n"
        "- Tool responses are capped to about {tool_output_tokens} tokens.\n"
        "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
    )

    #: How long kernel state lives, in the words the tool description uses.
    python_lifetimes: dict[str, str] = field(
        default_factory=lambda: {
            "round": "ephemeral",
            "move": "session-scoped",
            "level": "session-scoped",
            "always": "game-scoped",
        }
    )
    code_param_ephemeral: str = "Python code to run. The snippet is ephemeral and is not saved across tool calls."
    code_param_persistent: str = (
        "Python code to run. Definitions and variables persist across tool calls in this session; you are told "
        "when the session is cleared."
    )
    #: Added to either of the two above. The `code` parameter is where the
    #: model is told what survives, and a sentence about what is cleared is
    #: misleading while files are not.
    code_param_files_suffix: str = " Files in your working directory are not cleared with it."

    # The no-tool-call nudge, as ONE template for every configuration: the
    # slots carry what the runtime surface changes, so an override lands on
    # every mode rather than on whichever branch happened to be reachable.
    # followup_nudge appends tool_call_format_guidance at render time.
    action_nudge_guidance: str = "Investigate further if you need to, then act: append any genuinely new finding with the native `memory` tool, and call `submit_action(actions)` inside Python. "
    action_nudge_guidance_without_journal: str = (
        "Investigate further if you need to, then act: call `submit_action(actions)` inside Python. "
    )
    #: The nudge after a reply refused as malformed on every retry and dropped
    #: (``SealAgent.malformed_nudges``), which the model never sees.
    #: ``malformed_nudge`` appends tool_call_format_guidance at render time,
    #: as ``followup_nudge`` does.
    malformed_nudge_text: str = (
        "Your last reply was discarded: it went in circles and ended with no text and no tool call. "
        "Take a different approach, and make your next step a `python` call."
    )

    # Included once in the system prompt at the level ``brevity`` names.
    # Asks for proportionate reasoning rather than a fixed number: a token
    # budget the model cannot measure invites it to guess, and the turns that
    # genuinely need long reasoning are the ones worth keeping. The hard
    # ceiling elsewhere is a backstop against a runaway, not the instrument
    # that shapes ordinary turns.
    brevity_lines: dict[str, str] = field(
        default_factory=lambda: {
            "full": (
                "Keep your reasoning proportionate to the decision: think as long as the move genuinely needs, "
                "then act. Long deliberation is worth it for a new mechanic or a level you have not solved; it is "
                "waste when you already know what to try. Every turn you spend re-deriving what you established "
                "earlier is a turn you are not playing."
            ),
            "half": "Think as long as the move genuinely needs, then act.",
            "off": "",
        }
    )

    # Under ``compaction_strategy="summarize"``, where the tool-session
    # limits end: the model is told a summary is coming, so conclusions it
    # states along the way are worth stating clearly.
    context_summary_line: str = (
        "- Your context is finite. When it fills, you will be asked to write a compact summary that "
        "replaces older turns, so investing in clear intermediate conclusions pays off.\n"
    )

    persistence_lines: dict[str, str] = field(
        default_factory=lambda: {
            "round": (
                "- The `python` tool code is not saved between calls, so rewrite any custom utility logic you still need."
            ),
            "move": (
                "- The `python` session keeps your definitions and variables across tool calls within this turn, so "
                "helpers you write stay available and results you compute can be reused instead of recomputed. It is "
                "cleared when you next act on the game."
            ),
            "level": (
                "- The `python` session keeps your definitions and variables across tool calls and across moves, so "
                "helpers you write stay available and results you compute can be reused instead of recomputed. It is "
                "cleared when a new level starts."
            ),
            "always": (
                "- The `python` session keeps your definitions and variables for the whole game, so helpers you write "
                "stay available and results you compute can be reused instead of recomputed. Build up the tools you "
                "want rather than rewriting them each time."
            ),
        }
    )

    # Appended to the CURRENT turn's user prompt when rounds of that turn
    # were pruned to fit the context window. The marker is what makes an
    # earlier note findable, so repeated pruning rewrites it in place
    # instead of stacking; ``rounds_pruned_note`` prepends it and
    # ``is_rounds_pruned_note`` looks for it, so the two cannot disagree
    # however either is overridden.
    # What the files are FOR. The journal is always in the prompt and small;
    # files are large, runnable, and invisible until read, so a file worth
    # keeping is named in the journal. What to BUILD in them stops at reusable
    # analysis — a model that wants a simulator can see for itself that it
    # has somewhere to put one.
    solving_file_lines: str = (
        "- When you write a file worth keeping, say so in the journal in one line. Old conversation is dropped "
        "as the game goes on, and the journal is then the only record that your files exist at all.\n"
        "- Keep your analysis in files rather than in one-off snippets: a board summariser, a frame diff, a scorer. Import them where you need them, and fix them in place with `str_replace` instead of re-sending the whole thing.\n"
    )
    # Ahead of the lines above, with the journal and no world model: the
    # duck's division of the two channels, which the prompt review cut beside
    # a world model (2026-09-28).
    solving_file_know_line: str = (
        "- The journal is what you KNOW; files are what you BUILT. Findings, corrections and plans belong in "
        "the journal, which is in front of you every turn. Code and computed data belong in files, which are "
        "there only when you look.\n"
    )
    # The lines above without the two that set files against the journal.
    solving_file_lines_without_journal: str = "- Keep your analysis in files rather than in one-off snippets: a board summariser, a frame diff, a scorer. Import them where you need them, and fix them in place with `str_replace` instead of re-sending the whole thing.\n"
    #: After the file lines, by ``notes_guidance``.
    notes_guidance_lines: dict[str, str] = field(
        default_factory=lambda: {
            "none": "",
            "encourage": (
                "- You can use your file tools to write, maintain, and read notes.md, including anything you want "
                "to remember.\n"
            ),
            "discourage": "- Use your file tools for code, not for notes.\n",
        }
    )

    rounds_pruned_marker: str = "[context] "
    rounds_pruned_note_template: str = (
        "{n} earlier round(s) of this turn were dropped to fit the context window. "
        "Anything you established in them is gone from this conversation — re-derive what you still need, "
        "and prefer acting on what you already know over re-investigating."
    )

    # Prepended to the user prompt on the first turn after the session was
    # cleared, so scratch state disappearing is never silent. Reachable only
    # under "move" and "level" persistence: "always" never resets mid-game,
    # and "round" resets every call, where a notice each time would be noise.
    repl_reset_notice_with_files: str = (
        "Your `python` session was cleared because {reason}: definitions and variables from earlier snippets "
        "are gone. Your files are still there — re-create anything else you need."
    )
    #: The notice's reason when the harness restarts a kernel it could not use
    #: after a call: the kernel hung past its limit, or a library it loaded
    #: could not get the memory it needed. Appended to that call's output.
    kernel_stopped_responding_reason: str = "it stopped responding"
    kernel_out_of_memory_reason: str = "it ran out of memory"

    # Asked on the move after a level transition, ahead of the checklist. A
    # surface whose host collects the level's notes itself leaves it empty.
    level_summary_request: str = "You have just completed a level. Before anything else, append one journal entry summarising what you learned on it: the objects that mattered, what each action did, and what won."

    # The reflection turn after a won level (``SealAgent.level_reflection``):
    # after the batch's report, the win, then the reflection asked for, its
    # world-model sentence spent only where a world model is in use.
    reflection_level_won_line: str = "You have won level {level}!"
    reflection_win_missed_line: str = (
        "Your world model did not predict this win: `test_world_model()` now fails on `history[-1]`."
    )
    reflection_intro: str = (
        "Reflect on what you learned from this level and how you beat it. Challenge your assumptions, and try to "
        "see if a simpler model explains what you saw."
    )
    reflection_world_model_sentence: str = " You may also write/update and clean up your world model along the way."
    reflection_questions: str = (
        " Then answer these questions:\n\n"
        "(1) Which mechanics are proven, based on what evidence?\n"
        "(2) Which mechanics are hypothesized (i.e. holding up so far but not yet proven)?\n"
        "(3) What learnings do you want to keep in mind for future levels? Consider how to read the board, how to "
        "find mechanics, and how to find winning strategies."
    )
    reflection_answer_line: str = (
        "When you are done, answer the questions in plain text, with no tool call: that reply ends this turn."
    )
    #: ``concise``: the brief reflection asked for in place of the statement
    #: and its questions, the world-model sentence after it as there.
    reflection_concise: str = "Reflect briefly on what you learned from this level and how you beat it."
    #: How a concise side turn ends, for the reflection and the audit alike.
    concise_answer_line: str = "When you are done, reply in plain text, with no tool call: that reply ends this turn."
    #: Closes the prompt of a turn that is not for acting (the reflection, the audit).
    no_action_turn_line: str = "`submit_action` is not available in this turn."
    #: Raised by ``submit_action`` in such a turn; nothing is executed.
    no_action_turn_refusal: str = (
        "submit_action is not available in this turn. Nothing was executed. Answer the questions in plain text, "
        "with no tool call, to end the turn."
    )
    reflection_cut_note: str = (
        "The reflection turn was ended due to token limits. Stop reflecting and continue with the next level."
    )

    # The strategy audit (``SealAgent.audit_level_tokens``): a step back from a
    # level that has run long, with advice rather than procedure
    # (2026-10-07: "the model is quite capable on its own"), the facts of what
    # is untried, and two questions. The advice is a list, so a sweep can vary
    # the set.
    audit_intro: str = (
        "You have been on level {level} for a while without clearing it. Take a step back and reconsider your "
        "approach before playing on. You may use Python and your files along the way."
    )
    audit_advice_header: str = "Some things that may help:"
    audit_advice: list[str] = field(
        default_factory=lambda: [
            "If the win condition is unclear or mechanics seem arbitrary, consider if any static board elements "
            "give a clue.",
            "If an action seems to be inert so far, consider if it may do something in some other game state.",
            "If the rules are clear but your plans keep failing, search over your model instead of planning by hand.",
            "The level is winnable by humans, so if it seems impossible reconsider what you may have missed.",
        ]
    )
    #: The untried facts, one advice line, each part left out when empty.
    audit_untried_actions: str = "Not yet tried on this level: {actions}"
    audit_untried_actions_game: str = "; {actions} {verb} not been tried in this game at all"
    audit_unclicked_colors: str = "Colors on the board that no click has hit on this level: {colors}"
    audit_unclicked_colors_game: str = "; {colors} {verb} never been clicked in this game"
    audit_questions: str = (
        "Then answer briefly:\n"
        "(1) What do you know for sure about this level, and what are you assuming?\n"
        "(2) What will you try next, and why?"
    )
    audit_answer_line: str = "When you are done, answer in plain text, with no tool call: that reply ends this turn."
    #: ``concise``: the step back without the questions; the advice and the
    #: untried facts stay.
    audit_concise_intro: str = (
        "You have been on level {level} for a while without clearing it. Take a step back and reflect briefly on "
        "your approach before playing on."
    )
    audit_continue_line: str = "Continue playing the level. `submit_action` is available again now."
    audit_cut_note: str = (
        "The audit turn was ended because it went on too long. Stop auditing and continue playing the level. "
        "`submit_action` is available again now."
    )
    # The turn order, rendered on the first move of a level only.
    checklist_header: str = "This turn:"
    checklist_orient_fresh: str = "1. Orient on the newest frame: `segment(history[-1].frame.grid)` for structure."
    # Without segmentation, no view to start from.
    checklist_orient_fresh_without_segment: str = "1. Orient on the newest frame."
    checklist_decide: str = "2. Decide the next move or short sequence — explore, or advance the level. Write a search if the right order is not obvious."
    checklist_memory: str = (
        "3. Append anything new and durable with `memory`, before you act: the turn ends as soon as you do."
    )
    checklist_act: str = "4. Act with `submit_action([...])`."
    # Without the journal the memory step goes and the act step moves up.
    checklist_act_without_journal: str = "3. Act with `submit_action([...])`."
    checklist_act_world_model_without_journal: str = (
        "3. Act with `submit_action([...])`: one action until your world model passes, any sequence after it does."
    )

    # Replace the act/grounding lines on a lost frame. One field per line, so
    # a surface that rejects one of these claims overrides that line rather
    # than restating the other two.
    game_over_study_line: str = (
        "The frame above is the state you lost in — study it, since it shows exactly what went wrong."
    )
    #: The turn after a RESET that closed an attempt repeating an earlier one
    #: on the level (``SealAgent.replay_reminder``); ``{moves}`` is the
    #: attempt, RESET included. Not an audit: the model plays on.
    #: The turn after a RESET that did not restore the board its level opened
    #: on, with a world model in use; ``{cells}`` how many cells differ.
    irregular_reset_line: str = (
        "That RESET led to a different frame than the one this level started on ({cells} cells differ). This is "
        "purely cosmetic: the gameplay state should still be assumed to be fully reset. But it is impossible to "
        "deal with in the world model framework, which replays every RESET as `initial_state()`: render those "
        "cells as -1 (wildcards), so the replay does not check them."
    )
    replay_reminder_line: str = (
        "You have now repeated {moves} from the start of the level exactly as before. The game is deterministic, "
        "so this cost you score while you could have just gotten everything from `history`.\n\n"
        "If you're just repeating actions because you've given up, keep in mind that the level is absolutely "
        "winnable. Review your assumptions and strategy."
    )
    game_over_reset_line: str = (
        "RESET is the only action that can be executed now. Append what caused the loss with the native "
        "`memory` tool if it is new, then call `submit_action([Action('RESET')])` inside Python; any other action is refused."
    )
    game_over_reset_line_without_journal: str = (
        "RESET is the only action that can be executed now. Call `submit_action([Action('RESET')])` inside Python; "
        "any other action is refused."
    )
    game_over_reset_scope_line: str = (
        "RESET restores the current level to its starting state; the level you had reached is not lost."
    )
    #: With the journal and no world model, after the three above, as the duck
    #: says it.
    game_over_carry_line: str = "Carry any new cause of the loss into persistent memory so the next attempt avoids it."

    system_prompt_opening: str = "You are a coding agent solving a grid-based puzzle game."

    # Refusal for acting past a loss, and the skip text put on tool calls
    # issued after an earlier call in the same response already acted.
    game_over_refusal: str = "The game is over: {action} cannot be executed. RESET is the only valid action."
    already_acted_skip: str = (
        "not executed: an earlier tool call in this response {verb} actions; re-issue next turn if still needed"
    )
    #: Ends every tool result once ``SealAgent.suggest_moves_after_tokens``
    #: have passed without a move; ``tokens`` is how many have, with
    #: thousands separators.
    suggest_moves_note: str = (
        "You have not done any move for {tokens} tokens. Consider experimenting rather than thinking."
    )

    # Why a batch stopped short, mid-snippet. Each names what to do next,
    # since the model reads this and then has to decide whether to keep going.
    stop_detail_run_complete: str = "No further actions were executed because the run is already complete."
    stop_detail_game_over: str = (
        "No further actions were executed because the previous action reached GAME_OVER; "
        "RESET on your next turn to restart the level."
    )
    stop_detail_level_completed: str = (
        "No further actions were executed because the previous action completed a level; "
        "re-ground on the new scene before acting again."
    )
    stop_detail_done: str = "No further actions were executed because the environment reported done."
    stop_detail_max_moves: str = (
        "No further actions were executed because the run's action limit was reached; the run ends here."
    )
    stop_detail_terminal: str = "No further actions were executed because the previous action reached a terminal state."

    # Refusals a batch answers with before or instead of executing: an
    # action the model spelled wrong or may not use, a MOUSE without
    # coordinates, one no longer legal in the current state. Model-visible
    # like everything else here — they land in the tool result the model
    # reads next.
    stage_error_unknown_action: str = "Action {index}: unknown action {name!r}."
    stage_error_undo_disabled: str = "Action {index}: UNDO is disabled."
    stage_error_mouse_coords: str = "Action {index}: MOUSE requires integer row and col."
    invalid_action_error: str = "Action 1: {action} is not currently valid."
    #: Ends every refusal of a batch the model wrote wrong: the game's actions.
    #: The kernel's own refusals end with the same (``seal.harness.game``).
    valid_actions_note: str = VALID_ACTIONS_NOTE
    stop_detail_invalid_action: str = "{action} is not valid in the current state; remaining actions were not executed."

    # The journal's own model-visible text: what the `memory` tool answers
    # with, what one entry's host timestamp looks like, and what it refuses.
    # The acknowledgement is scanned back by the agent to decide which
    # entries the conversation still carries, so the header template and the
    # reader (``parse_memory_append_index``) travel with it.
    memory_entry_header_template: str = (
        "[#{index} | response={response} | step={step} | level={level} | last_action={last_action}]"
    )
    memory_entry_no_action: str = "NONE"
    memory_append_ack: str = "Persistent memory appended successfully {header}."
    memory_budget_not_positive: str = "memory_context_tokens must be positive to append memory."
    memory_empty_text: str = "The memory tool requires a non-empty `text` entry."
    memory_entry_too_large: str = (
        "Memory entry is about {entry_tokens} tokens, exceeding the "
        "{budget}-token memory limit. Append a shorter finding."
    )

    # The summary compaction's own text (``compaction_strategy="summarize"``):
    # the request that asks for a summary, the line that carries it into the
    # rebuilt conversation, and the message that closes that conversation.
    # The request is composed by ``compaction_prompt`` from these pieces, one
    # per sentence a render fact can add or remove. ``{and_files}`` is filled
    # only where the model has file tools: without them it has no way to make
    # a file, and a sentence about its files describes nothing it has.
    compaction_request_head: str = (
        "Your context window is nearly full. Write a {kind} summary of your work so far to carry forward. "
        "Everything before this message will be discarded; your `python` session (its variables, functions "
        "and `history`){and_files} will not."
    )
    # What the head asks the summary to be: ``thorough`` under
    # ``thorough_summary`` (2026-09-23).
    compaction_request_kind: str = "compact"
    compaction_request_kind_thorough: str = "thorough"
    # Names the notes appended since the last compaction too: they sit in the
    # history's tool calls rather than in the snapshot, and the install is
    # what moves them into it.
    compaction_request_journal_line: str = (
        " Your journal notes are kept too, including those you appended since the last compaction; do not repeat them."
    )
    # Under ``thorough_summary``, after the head and the journal line. Not
    # quite true with the journal kept -- its notes survive -- and left so
    # (2026-09-23). The size sentence goes under it: the target read
    # as a ceiling, and the switch exists to lift the summary off it.
    compaction_request_thorough_line: str = (
        " Whatever you don't write in the summary is lost, so be thorough. Discuss everything you learned and "
        "the evidence behind it."
    )
    # Under ``thorough_summary``, once a previous summary is in the history.
    compaction_request_previous_summary_line: str = (
        " Make sure to also take over anything that is still relevant from the previous summary."
    )
    compaction_request_size: str = " At most ~{summary_tokens:,d} tokens."
    compaction_request_items: str = (
        " In this order:\n"
        "1. Known rules of the game: the hypotheses you tested and the outcome of each.\n"
        "2. Key observations about the frames on the current level that informed your thinking.\n"
        "3. The current state of the board: the objects that matter and where they are now.\n"
        "4. How you used `python`{and_files} for this game: the {kinds} worth knowing about, and what each is for.\n"
        "5. Open questions: the hypotheses you are working on now.\n"
        "6. Your immediate next step.\n"
    )
    compaction_request_earlier_levels_line: str = (
        "Include mechanics that were in play on earlier levels even if you have not seen them on this one yet; "
        "they may resurface.\n"
    )
    compaction_request_listing_line: str = (
        "A listing of your `python` session's objects{and_files} will be provided after compaction, so you "
        "need not repeat it, but do explain the most important ones.\n"
    )
    # Where the rebuilt conversation shows ``world_model.py`` in full: the
    # summary need not restate it, and what the file cannot say about itself
    # is where it is wrong or unfinished.
    compaction_request_world_model_line: str = (
        "`world_model.py` will be shown in full after compaction, so do not restate it; say instead what it "
        "still gets wrong or leaves open.\n"
    )
    compaction_request_tail: str = (
        "Reply in plain text only — do NOT call any tools (any earlier instruction to make a tool call is "
        "superseded for this reply). Do not add anything else."
    )
    compaction_wrapper_prefix: str = "[context compacted — the following is your own summary of your work so far]\n\n"
    #: Opens a history rebuilt in the order it happened
    #: (``SealAgent.chronological_restart``), whose first messages are the
    #: middle of a game: without it a model reading them in order learns why
    #: only at the summary, after them.
    compaction_restart_preamble: str = (
        "[context compacted — this game is already under way: its earlier turns are left out here, and your own "
        "summary of them follows the latest prompt and tool call below]"
    )
    compaction_closing_line: str = (
        "[context compacted — your `python` session and files are intact; continue from here]"
    )
    compaction_closing_line_without_files: str = (
        "[context compacted — your `python` session is intact; continue from here]"
    )
    #: The closing line when the ``python`` session was reset by the time the
    #: summary was installed: the duck's, with the files the seal's
    #: workspace keeps through a reset named where the model has them.
    compaction_closing_line_reset: str = (
        "[context compacted — your `python` session was reset but your files are intact; continue from the summary]"
    )
    compaction_closing_line_reset_without_files: str = (
        "[context compacted — your `python` session was reset; continue from the summary]"
    )
    #: The objects section when the session could not be listed
    #: (``compaction_objects_unavailable``): the duck's. The kernel
    #: either went with the listing, so the next call starts afresh, or is
    #: still there with nothing known of its state.
    compaction_objects_unavailable_head: str = "Python session listing unavailable."
    compaction_objects_unavailable_kernel_lost: str = (
        "The Python kernel is unavailable; its next call starts a fresh session. Earlier variables may have been lost."
    )
    compaction_objects_unavailable_session_kept: str = (
        "Your Python session was not cleared. Inspect its variables before relying on them."
    )
    # The listing under the closing line. Each section shows its first
    # ``compaction_listing_lines`` entries one to a line, then the next
    # ``compaction_listing_names`` by name alone on one line, then a count of
    # the rest (2026-09-22).
    compaction_listing_files_header: str = "Your files:"
    compaction_listing_objects_header: str = "Your `python` session, objects you defined:"
    compaction_listing_empty: str = "(none)"
    compaction_listing_more_names: str = "and more: {names}"
    compaction_listing_more: str = "and {n} more not named here."
    compaction_world_model_header: str = "Your `world_model.py`:"
    compaction_listing_lines: int = 20
    compaction_listing_names: int = 100
    # Appended to the kept status prompt when the summary replaced tool calls
    # of the turn it opened; rewritten in place, as the pruned-rounds note is.
    tool_calls_omitted_marker: str = "<"
    tool_calls_omitted_template: str = "{n} tool call{s} omitted>"

    # -- the system prompt and its blocks --------------------------------

    def system_prompt(self, modes: PromptModes, memory_snapshot: str) -> str:
        """Assemble a prompt whose tools and runtime claims match every mode.

        ``memory_snapshot`` is the frozen journal rendering the agent owns;
        empty means an empty journal, rendered as such rather than omitted.
        """
        text = self.system_prompt_opening
        text += self._game_overview()
        text += self.board_legend_lines[modes.ascii_symbols]
        if modes.ascii_row_numbers:
            text += self.ascii_row_numbers_line
        if modes.ascii_column_group is not None:
            text += self.ascii_column_group_line.format(group=modes.ascii_column_group)
        text += self._runtime_state_addendum(modes)
        if modes.journal:
            text += self._memory_runtime_addendum(modes)
        text += self.structured_runtime_state_live_lines
        if modes.multimodal:
            mode_lines = self.multimodal_mode_lines.get(modes.multimodal)
            if mode_lines is None:
                raise ValueError(
                    f"{type(self).__name__} has no wording for multimodal={modes.multimodal!r}; "
                    "a prompt that cannot describe the image schedule would misdescribe it"
                )
            if modes.axis_label_step is not None:
                mode_lines += self.axis_labels_line.format(step=modes.axis_label_step)
            text += self.multimodal_context_addendum.format(mode_lines=mode_lines)
        text += self._world_model_addendum(modes)
        if modes.world_model:
            text += self.visual_game_addendum
        else:
            text += self.visual_game_addendum_without_world_model + (self.prior_mouse if modes.mouse_available else "")
        text += self._python_addendum(modes)
        text += self.python_addendum_live_lines
        if modes.world_model and not modes.world_model_optional:
            text += self.world_model_batch_stop_line
        text += self._compact_tool_session_addendum(modes)
        if modes.context_summary_enabled:
            text += self.context_summary_line
        brevity = self.brevity_lines[self.brevity]
        if brevity:
            text += f"\n{brevity}"
        if modes.journal:
            text += self._memory_snapshot_addendum(memory_snapshot)
        return text

    def _world_model_addendum(self, modes: PromptModes) -> str:
        """The world model: what it is for, exploration, the contract, and
        the workflow — nothing when none is in use. When it is optional, the
        obligation gives way to the choice and what bears on it."""
        if not modes.world_model:
            return ""
        optional = modes.world_model_optional
        return "".join(
            (
                self.world_model_header,
                self.world_model_intro_line_optional if optional else self.world_model_intro_line,
                self.world_model_default_line if optional and modes.world_model_default else "",
                self.world_model_tool_lines,
                self.world_model_seed_line_optional if optional else self.world_model_seed_line,
                self.world_model_check_lines_optional if optional else self.world_model_check_lines,
                self.world_model_search_line if modes.search else "",
                self._world_model_pays_lines(modes) if optional else "",
                self.world_model_contract_lines.format(
                    mouse_note=self.world_model_mouse_note if modes.mouse_available else "",
                    search_note=self.world_model_search_unknown_line if modes.search else "",
                ),
                self.world_model_energy_lines,
                self.world_model_pixel_tolerance_line.format(start=modes.pixel_tolerance_start)
                if modes.pixel_tolerance_offered
                else "",
                self.world_model_workflow_line_optional if optional else self.world_model_workflow_line,
            )
        )

    def _world_model_pays_lines(self, modes: PromptModes) -> str:
        """When a world model pays, with what it gives limited to what this
        configuration delivers: the search and the settled delta may be off."""
        gains = [self.world_model_gain_check]
        if modes.search:
            gains.append(self.world_model_gain_search)
        if modes.settled_delta:
            gains.append(self.world_model_gain_frame)
        joined = gains[0] if len(gains) == 1 else "; ".join(gains[:-1]) + "; and " + gains[-1]
        return self.world_model_pays_lines.format(gains=joined)

    def _game_overview(self) -> str:
        """The game overview, with the three costs where ``cost_model`` states them
        and, under ``winnable_line`` and ``retest_actions_line``, closing on
        those lines."""
        text = self.game_overview_addendum + (self.cost_model_line if self.cost_model == "explicit" else "")
        if self.winnable_line:
            text += self.game_overview_winnable_line
        if self.retest_actions_line:
            text += self.game_overview_retest_actions_line
        return text

    #: How the tool count is written out. Only the reachable totals: one or
    #: two native tools plus three to five file tools.
    _TOOL_COUNT_WORDS: ClassVar[dict[int, str]] = {4: "four", 5: "five", 6: "six", 7: "seven"}

    def _runtime_tool_line(self, modes: PromptModes) -> str:
        """The sentence naming the native tools: the file suite among them, and
        the journal's `memory` where it is kept."""
        # `python`, and `memory` with the journal, are always there; the rest
        # is what was built.
        count = (2 if modes.journal else 1) + len(modes.file_tool_names)
        quoted = [f"`{name}`" for name in modes.file_tool_names]
        names = ", ".join(quoted[:-1]) + " and " + quoted[-1]
        template = (
            self.runtime_tool_line_with_files if modes.journal else self.runtime_tool_line_with_files_without_journal
        )
        return template.format(count=self._TOOL_COUNT_WORDS[count], names=names)

    def _runtime_state_addendum(self, modes: PromptModes) -> str:
        """Describe exactly the configured Python globals, frame schema and tools."""
        return "".join(
            (
                self.runtime_header,
                self.available_actions_line.format(actions=", ".join(modes.action_names)),
                self._runtime_tool_line(modes),
                self.runtime_action_global_line.format(
                    segment_call=self.runtime_segment_call if modes.segmentation else ""
                ),
                self.runtime_import_line,
                self.runtime_file_lines.format(
                    world_model_note=self.runtime_world_model_module_note if modes.world_model else "",
                    segment_name=self.runtime_segment_module_name if modes.segmentation else "",
                ),
                self.runtime_projection_lines.format(or_journal=" or the journal" if modes.journal else ""),
                self.runtime_replay_hint_line if modes.replay_reminder else "",
                self.runtime_animation_line if modes.animation_description != "none" else "",
                self.turn_state_line if modes.world_model else "",
                self.turn_plan_line if modes.plan_argument != "none" else "",
                self.frame_common_lines,
                self.frame_api_lines,
                self.segment_api_line if modes.segmentation else "",
                self.ascii_api_lines,
                self.structural_view_line if modes.segmentation else self.compact_output_line,
                self.printed_board_no_hand_arithmetic_line if self.no_hand_arithmetic_line else "",
                self.runtime_readonly_line,
            )
        )

    def _memory_runtime_addendum(self, modes: PromptModes) -> str:
        """Model-facing native journal semantics and retention budget."""
        return (
            "- Call the native `memory` tool with one required `text` argument to append an immutable finding. Never call or import it inside Python.\n"
            "- The host timestamps appended memory with memory index, response, game step, level, and last action.\n"
            f"- The journal keeps about the most recent {modes.memory_context_tokens:,} tokens by evicting oldest whole "
            "entries. Newer entries supersede older conflicting findings, and the journal survives conversation "
            "eviction, Python-session resets, moves, and level transitions.\n"
            "- A frozen journal snapshot appears at the end of this system prompt. New entry text remains in its memory tool call; the compact tool result only confirms its host timestamp. The snapshot is replaced only when conversation compaction occurs, from token pressure or a configured history cap.\n"
        )

    def _memory_snapshot_addendum(self, snapshot: str) -> str:
        """Render the frozen journal snapshot at the system prompt's stable tail."""
        body = snapshot or "(empty)"
        return f"\n\nPersistent memory snapshot (oldest to newest, refreshed only at compaction):\n{body}"

    def _python_addendum(self, modes: PromptModes) -> str:
        """Return current guidance with only API-owned terms substituted.

        Without the journal its three clauses go and nothing takes their
        place: what carries work across a compaction is then the summary."""
        if modes.journal:
            memory_line = "- Use the persistent journal for durable world-model facts, goals, action effects, corrections, uncertainties, cross-level facts, and plans. Append only genuinely new or corrected information with the native `memory` tool; do not repeat unchanged entries.\n"
            optimize_line = "- Optimize for the shortest reliable sequence that advances the current goal. If confidence is low, program a discriminating probe and append the durable result to memory.\n"
            memory_step = "save any new durable finding with `memory`, "
            file_lines = (
                self.solving_file_lines if modes.world_model else self.solving_file_know_line + self.solving_file_lines
            )
        else:
            memory_line = ""
            optimize_line = "- Optimize for the shortest reliable sequence that advances the current goal. If confidence is low, program a discriminating probe.\n"
            memory_step = ""
            file_lines = self.solving_file_lines_without_journal
        world_model_step = (
            ""
            if not modes.world_model
            else "write or update the world model if you keep one, "
            if modes.world_model_optional
            else "write or update the world model, "
        )
        # A world model stops a batch at the first move it mispredicts, which is
        # the check this asks a model without one to write by hand. An
        # optional one says it beside the world model's own lines.
        batch_line = (
            ""
            if modes.world_model
            else "- If you do multiple actions in one tool call, be aware that this can waste many actions if it "
            "doesn't have the expected outcome. Consider adding intermediate `assert`s in the code to verify "
            "the state is changing as expected.\n"
        )
        # The duck's emphasis and its HUD check, which the prompt review cut
        # beside a world model: that one predicts the bar and checks the move.
        important = "" if modes.world_model else "IMPORTANT: "
        hud_line = (
            ""
            if modes.world_model
            else "- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, "
            "or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n"
        )
        return (
            "\n\nSolving guidance:\n"
            + memory_line
            + f"- {important}Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n"
            + optimize_line
            + "- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n"
            + file_lines
            + self.notes_guidance_lines[self.notes_guidance]
            + "- Keep tool-output context size minimal and decision-oriented so you can quickly compare before/after state. It's fine to write a lot of python code, just make the output short and interpretable.\n"
            f"- A strong default loop is: summarize the board, {world_model_step}infer the desired change, write a small scorer or search, {memory_step}execute with `submit_action(...)`, then inspect again.\n"
            "- For object tracking, match objects by color, overlap, bounding box proximity, area change, and edge contact rather than by exact coordinates alone.\n"
            "- For frame diffs, summarize changed cells, color transitions, appearing/disappearing components, movement candidates, and small local row slices around the changed region.\n"
            + hud_line
            + "- Call `submit_action(...)` inside Python rather than returning action text in chat.\n"
            + batch_line
        )

    def _compact_tool_session_addendum(self, modes: PromptModes) -> str:
        """Return the session-limit text with its measurements filled in.

        The fixed fragments are filled first; the user-settable guidance is
        appended afterwards, untouched — an override is prose that may
        legitimately contain a literal brace, which would otherwise be read
        as a format field and raise.
        """
        fixed = self.compact_tool_session_head + self.compact_tool_session_tail
        return (
            fixed.format(
                tool_output_tokens=modes.tool_output_tokens,
                python_timeout_s=modes.python_timeout_s,
                persistence_line=(
                    self.persistence_lines[modes.persistence] + "\n" + self._files_lifetime_line(modes).rstrip("\n")
                ),
            )
            + f"- {self.tool_call_format_guidance}\n"
        )

    def _files_lifetime_line(self, modes: PromptModes) -> str:
        """What the files do when the session is cleared — or, under
        "always", what they offer instead, since nothing clears it."""
        if modes.persistence == "always":
            return self.compact_tool_session_files_line_persistent
        return self.compact_tool_session_files_line

    # -- the tool surface -------------------------------------------------

    def python_tool_description(self, modes: PromptModes) -> str:
        """The `python` tool's description, which has to agree with how the
        tool is actually configured: which globals exist, and how long kernel
        state survives."""
        return (
            f"Run one {self.python_lifetimes[modes.persistence]} Python snippet against preloaded game state. Available globals: "
            "`history` (this level's Turns), `available_actions`, and "
            "`submit_action(actions)` for executing one or more real environment actions, each an `Action`. "
            "Frames expose `.grid`, `.diff_sparse` and `.color_counts`"
            + ("; `segment(frame.grid)` gives their objects" if modes.segmentation else "")
            + ". For the latest before/after diff, read `history[-1].frame.diff_sparse`. "
            "A MOUSE action is `Action('MOUSE', row=r, col=c)`, with integer `row` and `col`. "
            "Colors are indices 0-15. "
            + (
                "Use `segment(...)` as the primary view and `ascii_grid(...)`"
                if modes.segmentation
                else "Use `ascii_grid(...)`"
            )
            + " only for a small specific region. "
            "Use `print(...)` for compact output; a trailing expression is echoed like a notebook cell."
            + (
                ""
                if not modes.world_model
                else self.tool_world_model_sentence_optional
                if modes.world_model_optional
                else self.tool_world_model_sentence
            )
        )

    def python_tool_code_param_description(self, modes: PromptModes) -> str:
        """The `code` parameter's description. Separate from the tool's own
        because it is where the model is told whether names it binds will
        still exist on the next call."""
        base = self.code_param_ephemeral if modes.persistence == "round" else self.code_param_persistent
        return base + self.code_param_files_suffix

    def python_tool_plan_param_description(self, modes: PromptModes) -> str:
        """The ``plan`` parameter's description, asking for it ahead of the
        code under ``plan_argument="first"``."""
        first = self.plan_param_first_line if modes.plan_argument == "first" else ""
        return self.plan_param_description + first

    def memory_entry_header(self, entry: MemoryEntry) -> str:
        """One journal entry's compact host timestamp."""
        return self.memory_entry_header_template.format(
            index=entry.index,
            response=entry.response,
            step=entry.step,
            level=entry.level,
            last_action=entry.last_action or self.memory_entry_no_action,
        )

    def memory_append_acknowledgement(self, entry: MemoryEntry) -> str:
        """What the memory tool answers with: the timestamp, not the finding —
        the text itself is already in the model's own tool call."""
        return self.memory_append_ack.format(header=self.memory_entry_header(entry))

    def parse_memory_append_index(self, text: str) -> int | None:
        """The journal index in an acknowledgement, or ``None`` when ``text``
        is not one.

        Paired with :meth:`memory_append_acknowledgement` so a reworded
        acknowledgement stays readable: the prefix is derived from the same
        two templates rather than spelled out a second time.
        """
        prefix = self.memory_append_ack.partition("{header}")[0]
        prefix += self.memory_entry_header_template.partition("{index}")[0]
        if not text.startswith(prefix):
            return None
        try:
            return int(text[len(prefix) :].partition(" ")[0])
        except ValueError:
            return None

    # -- per-turn prompts -------------------------------------------------

    def user_prompt_text(
        self,
        modes: PromptModes,
        *,
        reset_reason: str | None,
        summary: StepSummary | None,
        any_actions_executed: bool,
        current_step: int,
        current_level: int,
        observed_max_level: int,
        valid_actions: list[str],
        valid_actions_observed: bool,
        game_over_active: bool,
        exploration_moves_left: int = 0,
        exploration_over: bool = False,
        level_tokens_used: int = 0,
        level_token_allowance: int | None = None,
        frame_delta: str | None = None,
        animation_text: str | None = None,
        animation_described: int | None = None,
        animation_concise: bool = False,
    ) -> str:
        """Compose the per-turn user prompt, in order: a kernel-reset notice
        if one is owed, what the previous batch of actions did, where the
        game stands, the newest frame as a delta from one the model knew,
        when there is one to give, and last a move's animation as a block
        of code, when one was described. A lost frame adds what is legal now
        and its reset guidance; otherwise the legal actions are the game's
        whole set, which the system prompt names once.

        Every argument is a fact the agent computed from its state — this
        method turns facts into text and holds no state of its own.
        """
        lines: list[str] = []
        if reset_reason:
            # First turn after a reset: say so before anything else, so the
            # model does not plan around helpers that no longer exist.
            lines.append(self.repl_reset_notice_with_files.format(reason=reset_reason))
        if summary is not None:
            if not summary.reported:
                lines.extend(self._batch_report_lines(summary, modes, animation_described, animation_concise))
            if summary.run_complete:
                lines.append("You have completed the run!")
            elif summary.level_transition:
                lines.append("You have progressed to a new level!")
                if self.retest_actions_line:
                    lines.append(self.level_start_retest_actions_line)
            else:
                lines.append("You are still on the same level.")
                if modes.world_model and summary.irregular_reset:
                    lines.append(self.irregular_reset_line.format(cells=summary.irregular_reset))
                if modes.replay_reminder and summary.repeated:
                    lines.append(self.replay_reminder_line.format(moves=self._repeated_moves(summary.repeated)))
            if modes.world_model and summary.world_model_stop == "mismatch":
                lines.append(self.summary_world_model_mismatch)
            elif modes.world_model and summary.world_model_stop == "unknown":
                lines.append(self.summary_world_model_unknown)
            if modes.world_model_optional and summary.world_model_stop is not None:
                lines.append(self.summary_world_model_stop_choice)
            elif modes.world_model_optional and summary.checked is not None and not summary.level_transition:
                # Across a level the line would describe the previous level's
                # model as this one's; the new level's lines say where it stands.
                lines.append(self.summary_checked_line if summary.checked else self.summary_unchecked_line)
            if summary.game_over and game_over_active:
                # Guarded on the active flag so the line only ever describes
                # the board in front of the model — the lost one, never a
                # level the model's own RESET has already restarted.
                lines.append("The game is over.")
        elif any_actions_executed:
            lines.append("No previous action sequence was captured.")
        else:
            lines.append("No actions have been executed yet.")
        state_line = f"Current state: step {current_step}, level {current_level}"
        if observed_max_level > current_level:
            state_line += f" out of observed max level {observed_max_level} so far"
        state_line += "."
        if valid_actions:
            valid_line = ", ".join(valid_actions)
        else:
            valid_line = "none" if valid_actions_observed else "unknown"
        lines.append(state_line)
        exploring = exploration_moves_left > 0 and not exploration_over
        if modes.world_model and not modes.world_model_optional:
            if exploring:
                moves = "move" if exploration_moves_left == 1 else "moves"
                lines.append(self.exploration_status_line.format(n=exploration_moves_left, moves=moves))
            else:
                lines.append(self.world_model_required_line)
        if game_over_active:
            lines.append(f"Valid actions right now: {valid_line}.")
            lines.extend(self.user_prompt_game_over_lines(modes))
        else:
            if summary is not None and summary.level_transition:
                if self.level_summary_request and modes.journal:
                    lines.append(self.level_summary_request)
                lines.append(self.new_level_history_line)
                if modes.world_model_optional:
                    lines.append(self.new_level_world_model_lines_optional)
                elif modes.world_model:
                    lines.append(self.new_level_world_model_lines)
            level_start = (summary is not None and summary.level_transition) or (
                summary is None and not any_actions_executed
            )
            if level_start and modes.world_model_optional and modes.world_model_default:
                lines.append(self.level_start_world_model_default_line)
            lines.extend(self._checklist(summary, modes))
        if frame_delta:
            lines.append(frame_delta)
        # Without a concise description a move's mark says so, and the line would repeat it; without marks
        # the line is what says it.
        if animation_text and (animation_concise or not modes.animation_marks):
            lines.append(animation_text)
        return "\n".join(lines)

    def reflection_prompt_text(
        self,
        modes: PromptModes,
        *,
        concise: bool,
        reset_reason: str | None,
        summary: StepSummary | None,
        level: int,
        win_missed: bool,
        frame_delta: str | None,
        animation_text: str | None,
        animation_described: int | None,
        animation_concise: bool,
    ) -> str:
        """Compose the reflection turn's prompt, in order: a kernel-reset
        notice if one is owed, the batch's report, the win, a world model that
        did not predict it, the win screen as a delta and the winning
        animation as a block where :meth:`user_prompt_text` would put them,
        then the reflection asked for. No state or status line: nothing is
        played in the turn."""
        lines: list[str] = []
        if reset_reason:
            lines.append(self.repl_reset_notice_with_files.format(reason=reset_reason))
        if summary is not None:
            lines.extend(self._batch_report_lines(summary, modes, animation_described, animation_concise))
        lines.append(self.reflection_level_won_line.format(level=level))
        if win_missed and modes.world_model:
            lines.append(self.reflection_win_missed_line)
        if frame_delta:
            lines.append(frame_delta)
        if animation_text and (animation_concise or not modes.animation_marks):
            lines.append(animation_text)
        world_model = self.reflection_world_model_sentence if modes.world_model else ""
        if concise:
            lines.append(f"\n{self.reflection_concise}{world_model}\n")
            lines.append(self.concise_answer_line)
        else:
            lines.append(f"\n{self.reflection_intro}{world_model}{self.reflection_questions}\n")
            lines.append(self.reflection_answer_line)
        lines.append(self.no_action_turn_line)
        return "\n".join(lines)

    def audit_prompt_text(
        self,
        modes: PromptModes,
        *,
        concise: bool,
        level: int,
        untried_actions: list[str],
        untried_actions_game: list[str],
        unclicked_colors: list[int],
        unclicked_colors_game: list[int],
    ) -> str:
        """Compose the strategy audit's prompt: the step back, the advice with
        the untried facts last, the questions, and how to end the turn."""
        advice = [*self.audit_advice]
        untried = self._untried_line(untried_actions, untried_actions_game, unclicked_colors, unclicked_colors_game)
        if untried:
            advice.append(untried)
        intro = self.audit_concise_intro if concise else self.audit_intro
        ask = [] if concise else ["", self.audit_questions]
        return "\n".join(
            [
                intro.format(level=level),
                "",
                self.audit_advice_header,
                *(f"- {line}" for line in advice),
                *ask,
                "",
                self.concise_answer_line if concise else self.audit_answer_line,
                self.no_action_turn_line,
            ]
        )

    def _untried_line(
        self, actions: list[str], actions_game: list[str], colors: list[int], colors_game: list[int]
    ) -> str:
        """The untried facts as one line, or empty when nothing is untried."""

        def listed(items: list[str]) -> tuple[str, str]:
            text = items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"
            return text, "has" if len(items) == 1 else "have"

        parts: list[str] = []
        if actions:
            part = self.audit_untried_actions.format(actions=", ".join(actions))
            if actions_game:
                names, verb = listed(actions_game)
                part += self.audit_untried_actions_game.format(actions=names, verb=verb)
            parts.append(part + ".")
        if colors:
            part = self.audit_unclicked_colors.format(colors=", ".join(str(color) for color in colors))
            if colors_game:
                names, verb = listed([str(color) for color in colors_game])
                part += self.audit_unclicked_colors_game.format(colors=names, verb=verb)
            parts.append(part + ".")
        return " ".join(parts)

    def _batch_report_lines(
        self, summary: StepSummary, modes: PromptModes, animation_described: int | None, animation_concise: bool
    ) -> list[str]:
        """How many moves the previous batch played, and which."""
        count = summary.executed_count
        action_label = "action" if count == 1 else "actions"
        return [
            f"The code executed {count or 0} {action_label} in the previous sequence.",
            self._executed_actions_line(summary, modes, animation_described, animation_concise),
        ]

    def _executed_actions_line(
        self, summary: StepSummary, modes: PromptModes, animation_described: int | None, animation_concise: bool
    ) -> str:
        """The moves a batch played, each with its animation frames marked.

        One move is marked in full, where its frames are: the move whose
        animation is described below the prompt, else the first whose turn is
        in ``history``; with animations on, the full mark says how to have
        them described. Every other animated move gets its count. Past ten
        moves the line keeps the first and last five, and the fully marked
        move when it falls between; a gap that hid animated moves says how
        many.

        With what each move changed in the summary (``move_changes``) the
        moves go one per line, each with its change ahead of its mark in
        the same brackets.
        """
        animations = [*summary.animations, *[None] * (len(summary.executed_actions) - len(summary.animations))]
        changes = [*summary.changes, *[None] * (len(summary.executed_actions) - len(summary.changes))]
        # (label, animation frames or 0, history index or None), for the moves with a name.
        moves = [
            (str(name).strip(), animation[0] if animation else 0, animation[1] if animation else None)
            for name, animation in zip(summary.executed_actions, animations)
            if str(name).strip()
        ]
        notes = [
            self._move_change_note(change)
            for name, change in zip(summary.executed_actions, changes)
            if str(name).strip()
        ]
        if not moves:
            return "Executed actions: none."
        indexed = [position for position, (_, frames, index) in enumerate(moves) if frames and index is not None]
        described = [position for position in indexed if moves[position][2] == animation_described]
        marked = (described or indexed or [-1])[0]
        if described:
            full = self.animation_mark_described if animation_concise else self.animation_mark_not_concise
        elif modes.animation_description != "none":
            full = self.animation_mark_ask
        else:
            full = self.animation_mark
        rendered = [
            label
            if not frames
            else full.format(label=label, frames=frames, index=index)
            if position == marked
            else self.animation_mark_short.format(label=label, frames=frames)
            for position, (label, frames, index) in enumerate(moves)
        ]
        by_line = bool(summary.changes)
        if by_line:
            rendered = [
                self.executed_move_line.format(move=self._with_note(move, label, note))
                for move, (label, _, _), note in zip(rendered, moves, notes)
            ]
        if len(moves) <= 10:
            return (
                "\n".join(["Executed actions:", *rendered]) if by_line else f"Executed actions: {', '.join(rendered)}."
            )
        between = 5 <= marked < len(moves) - 5
        gaps = [(5, marked), (marked + 1, len(moves) - 5)] if between else [(5, len(moves) - 5)]
        shown = rendered[:5]
        for number, (first, last) in enumerate(gaps):
            animated = sum(1 for _, frames, _ in moves[first:last] if frames)
            if last > first:
                shown.append(
                    self.elided_animations.format(count=last - first, animated=animated) if animated else "..."
                )
            if between and number == 0:
                shown.append(rendered[marked])
        shown += rendered[-5:]
        header = self.executed_elided_header_with_mark if between else self.executed_elided_header
        return "\n".join([f"{header}:", *shown]) if by_line else f"{header}: {', '.join(shown)}."

    @staticmethod
    def _repeated_moves(labels: list[str]) -> str:
        """A repeated attempt's moves for the reminder: past ten, the first
        and last five."""
        if len(labels) <= 10:
            return ", ".join(labels)
        return ", ".join([*labels[:5], f"... ({len(labels) - 10} more) ...", *labels[-5:]])

    def _move_change_note(self, change: MoveChange | None) -> str | None:
        """What one move changed, as its line says it; None where unknown (a
        level win, whose settled frame is the next level's)."""
        if change is None:
            return None
        count, before, after = change
        if not count:
            return self.move_unchanged_note
        return self.move_change_note.format(
            count=count, before=",".join(map(str, before)), after=",".join(map(str, after))
        )

    @staticmethod
    def _with_note(move: str, label: str, note: str | None) -> str:
        """``move`` (``label``, perhaps with its animation mark in brackets)
        with ``note`` first inside the brackets."""
        if note is None:
            return move
        if move == label:
            return f"{label} ({note})"
        assert move.startswith(f"{label} ("), move
        return f"{label} ({note}; {move[len(label) + 2 :]}"

    def _checklist(self, summary: StepSummary | None, modes: PromptModes) -> list[str]:
        """The turn order, on the first move of a level only.

        That move is the one with nothing to diff against: no batch has run
        yet, or the last one crossed into a new level. Every later turn in
        the level is state alone. Without the journal there is no memory
        step, and the act step takes its number.
        """
        if summary is not None and not summary.level_transition:
            return []
        if not modes.journal:
            return [
                self.checklist_header,
                self._checklist_orient(modes),
                self._checklist_decide(summary, modes),
                self.checklist_act_world_model_without_journal
                if modes.world_model and not modes.world_model_optional
                else self.checklist_act_without_journal,
            ]
        return [
            self.checklist_header,
            self._checklist_orient(modes),
            self._checklist_decide(summary, modes),
            self.checklist_memory,
            self.checklist_act_world_model
            if modes.world_model and not modes.world_model_optional
            else self.checklist_act,
        ]

    def _checklist_orient(self, modes: PromptModes) -> str:
        """Step 1: on the newest frame, through segmentation where the kernel has it."""
        return self.checklist_orient_fresh if modes.segmentation else self.checklist_orient_fresh_without_segment

    def _checklist_decide(self, summary: StepSummary | None, modes: PromptModes) -> str:
        """Step 2: pointed at the example on the game's first move, the only
        move on which ``world_model.py`` is still the example."""
        if not modes.world_model:
            return self.checklist_decide
        if modes.world_model_optional:
            start = self.checklist_decide_world_model_optional_game_start if summary is None else ""
            return self.checklist_decide_world_model_optional + start
        return self.checklist_decide_world_model_game_start if summary is None else self.checklist_decide_world_model

    def user_prompt_game_over_lines(self, modes: PromptModes | None = None) -> list[str]:
        """Return the loss guidance that replaces the act/grounding lines.

        ``modes`` absent renders the journal's wording. Without the journal
        the reset line loses its memory clause and the carry line, which is
        nothing but one, goes."""
        if modes is not None and not modes.journal:
            return [
                self.game_over_study_line,
                self.game_over_reset_line_without_journal,
                self.game_over_reset_scope_line,
            ]
        # With the journal and no world model, the duck's carry line too; a
        # surface asked without modes renders the journal's wording, which is
        # the duck's.
        carry = [self.game_over_carry_line] if modes is None or not modes.world_model else []
        return [
            self.game_over_study_line,
            self.game_over_reset_line,
            self.game_over_reset_scope_line,
            *carry,
        ]

    def followup_nudge(self, completion_limit: int | None = None, modes: PromptModes | None = None) -> str:
        """Return the ordinary nudge, identifying a completion-limit cutoff when known.

        ``modes`` absent renders the journal's wording."""
        prefix = "" if completion_limit is None else self._completion_cutoff_prefix(completion_limit)
        journal = modes is None or modes.journal
        guidance = self.action_nudge_guidance if journal else self.action_nudge_guidance_without_journal
        text = prefix + "You have not acted yet. " + guidance
        if not self.nudge_format_guidance:
            return text.rstrip()
        return text + self.tool_call_format_guidance

    def malformed_nudge(self, modes: PromptModes | None = None) -> str:
        """The nudge after a dropped malformed reply; the format guidance rides
        under the same switch as the ordinary nudge's."""
        del modes  # one wording for every runtime surface
        if not self.nudge_format_guidance:
            return self.malformed_nudge_text
        return f"{self.malformed_nudge_text} {self.tool_call_format_guidance}"

    def _completion_cutoff_prefix(self, completion_limit: int) -> str:
        """Return the branch-current completion-cutoff explanation."""
        return (
            f"You hit the {completion_limit:,}-token limit, so your previous reasoning was cut off. "
            "Continue from where it stopped without restarting or altering it. "
        )

    def rounds_pruned_note(self, n: int) -> str:
        """The note that ``n`` rounds of the current turn were dropped."""
        return self.rounds_pruned_marker + self.rounds_pruned_note_template.format(n=n)

    def is_rounds_pruned_note(self, text: str) -> bool:
        """Whether ``text`` is a note this surface rendered.

        Paired with :meth:`rounds_pruned_note` — the agent rewrites an
        earlier note in place rather than stacking, and it finds it through
        here, so a reworded marker moves both halves at once.
        """
        return text.startswith(self.rounds_pruned_marker)

    def compaction_prompt(
        self,
        modes: PromptModes,
        summary_tokens: int,
        *,
        current_level: int,
        previous_summary: bool = False,
        world_model_shown: bool = False,
    ) -> str:
        """Ask for the state that must survive the history's replacement.

        What the request claims survives follows the agent: the journal's
        notes are named only where they exist, files only where the model has
        file tools, the earlier levels once there are any, and the listing
        only where the rebuilt conversation carries one. Under
        ``thorough_summary`` it asks for a thorough summary rather than a
        compact one, states no size, and asks for what still holds of the
        ``previous_summary`` where the history carries one. Where
        ``world_model_shown``, it says the file will be shown in full."""
        and_files = " and your files" if modes.file_tools else ""
        kind = self.compaction_request_kind_thorough if self.thorough_summary else self.compaction_request_kind
        text = self.compaction_request_head.format(kind=kind, and_files=and_files)
        if modes.journal:
            text += self.compaction_request_journal_line
        if self.thorough_summary:
            text += self.compaction_request_thorough_line
            if previous_summary:
                text += self.compaction_request_previous_summary_line
        else:
            text += self.compaction_request_size.format(summary_tokens=summary_tokens)
        text += self.compaction_request_items.format(
            and_files=and_files,
            kinds="variables, functions and files" if modes.file_tools else "variables and functions",
        )
        if current_level > 1:
            text += self.compaction_request_earlier_levels_line
        if modes.add_files_and_repl_to_summary:
            text += self.compaction_request_listing_line.format(and_files=and_files)
        if world_model_shown:
            text += self.compaction_request_world_model_line
        return text + self.compaction_request_tail

    def compaction_world_model(self, source: str) -> str:
        """``world_model.py`` under its header, in a code fence longer than any
        run of backticks inside it, so the file cannot close its own fence."""
        longest = max((len(run) for run in re.findall(r"`+", source)), default=0)
        fence = "`" * max(3, longest + 1)
        return f"{self.compaction_world_model_header}\n{fence}python\n{source.rstrip()}\n{fence}"

    def compaction_wrapper(self, summary: str) -> str:
        """Carry one accepted summary into the rebuilt conversation."""
        return self.compaction_wrapper_prefix + summary

    def compaction_objects_unavailable(self, *, kernel_lost: bool, error: str) -> str:
        """The objects section's text when the session listing failed: whether
        the kernel went with it, then the error."""
        state = (
            self.compaction_objects_unavailable_kernel_lost
            if kernel_lost
            else self.compaction_objects_unavailable_session_kept
        )
        return f"{self.compaction_objects_unavailable_head} {state}\n{error}"

    def compaction_listing(
        self,
        modes: PromptModes,
        files: list[tuple[str, int]],
        objects: list[tuple[str, str, bool]],
        *,
        objects_unavailable: str | None = None,
    ) -> str:
        """The files section (only where the model has file tools) and the
        ``python`` objects section, or in its place ``objects_unavailable``
        when the session could not be listed.

        Objects are ``(name, description, is_function)`` in the order they were
        bound, and are shown functions first, then by decreasing name length
        (binding order breaking ties): what the model built comes before its
        scratch, and a cap falls on names like ``i`` rather than on a solver
        defined late in the game."""

        def section(header: str, entries: list[tuple[str, str]]) -> str:
            if not entries:
                return f"{header}\n{self.compaction_listing_empty}"
            lines = [line for _, line in entries[: self.compaction_listing_lines]]
            rest = [name for name, _ in entries[self.compaction_listing_lines :]]
            named = rest[: self.compaction_listing_names]
            if named:
                lines.append(self.compaction_listing_more_names.format(names=", ".join(named)))
            if len(rest) > len(named):
                lines.append(self.compaction_listing_more.format(n=len(rest) - len(named)))
            return header + "\n" + "\n".join(lines)

        sections: list[str] = []
        if modes.file_tools:
            file_entries = [
                (name, f"- `{name}`" if size < 0 else f"- `{name}` ({_format_bytes(size)})") for name, size in files
            ]
            sections.append(section(self.compaction_listing_files_header, file_entries))
        if objects_unavailable is not None:
            sections.append(self.compaction_listing_objects_header + "\n" + objects_unavailable)
        else:
            ordered = sorted(objects, key=lambda entry: (not entry[2], -len(entry[0])))
            object_entries = [(name, f"- `{name}`: {text}") for name, text, _ in ordered]
            sections.append(section(self.compaction_listing_objects_header, object_entries))
        return "\n\n".join(sections)

    def compaction_closing(self, modes: PromptModes, listing: str, *, session_intact: bool = True) -> str:
        """Close the rebuilt conversation, over the listing when there is one;
        ``session_intact`` False says the ``python`` session was reset."""
        if session_intact:
            line = self.compaction_closing_line if modes.file_tools else self.compaction_closing_line_without_files
        else:
            line = (
                self.compaction_closing_line_reset
                if modes.file_tools
                else self.compaction_closing_line_reset_without_files
            )
        return line + (f"\n\n{listing}" if listing else "")

    def tool_calls_omitted_note(self, n: int) -> str:
        """The note that ``n`` tool calls of the current turn were summarized away."""
        return self.tool_calls_omitted_marker + self.tool_calls_omitted_template.format(n=n, s="" if n == 1 else "s")

    def is_tool_calls_omitted_note(self, text: str) -> bool:
        """Whether ``text`` is a note this surface rendered — the reader
        paired with :meth:`tool_calls_omitted_note`, so a summary that fires
        twice in one turn rewrites the note rather than stacking a second."""
        suffix = self.tool_calls_omitted_template.rpartition("}")[2]
        return text.startswith(self.tool_calls_omitted_marker) and text.endswith(suffix)


def _format_bytes(size: int) -> str:
    """A file size as the listing shows it: bytes, then KB and MB to one decimal."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


#: A stand-in for the module-level renderings below — NOT the shipped seal's
#: configuration, which ``SealAgent`` supplies at every render. The numeric
#: fields are zeroed rather than plausible: nothing here reads them, and a
#: real-looking value would be the copy of a default this class exists to
#: avoid.
STANDALONE_MODES = PromptModes(
    ascii_symbols="hex",
    persistence="round",
    file_tool_names=(),
    multimodal="",
    memory_context_tokens=0,
    tool_output_tokens=0,
    python_timeout_s=0,
    action_names=(),
    ascii_row_numbers=True,
    ascii_column_group=None,
    world_model=False,
    world_model_optional=False,
    search=False,
    settled_delta=False,
    animation_description="none",
    animation_marks=False,
    journal=True,
    context_summary_enabled=False,
    add_files_and_repl_to_summary=False,
    plan_argument="none",
    pixel_tolerance_offered=False,
    pixel_tolerance_start=0,
    axis_label_step=None,
    segmentation=True,
    replay_reminder=False,
)

#: The default nudge, for tests that pin the exact no-cutoff wording.
FOLLOWUP_NUDGE = SealPrompts().followup_nudge()

# Defaults for a GameREPL constructed on its own; SealAgent overwrites
# both from its own configuration before the tool is ever used.
PYTHON_TOOL_DESCRIPTION = SealPrompts().python_tool_description(STANDALONE_MODES)
PYTHON_TOOL_CODE_PARAM_DESCRIPTION = SealPrompts().python_tool_code_param_description(STANDALONE_MODES)
