"""Prompt templates for the analyzer agent.

Every fixed string the model reads lives on :class:`DuckPrompts`: the
literal fragments as dataclass fields — constructor-settable, so a sweep
or an experiment overrides any one of them without a code edit (A6) —
and the composition as methods, which read the fields at render time so
an overridden fragment propagates everywhere it appears. The
:class:`Prompts` protocol is the seam a genuinely different wording
implementation drops in through (`DuckAgent.prompts`).

What the agent decides — the runtime surface, the frame API, kernel
persistence, the numeric limits — is NOT on this class. It arrives per
render as a :class:`PromptModes`, built by ``DuckAgent._prompt_modes()``
from the agent's own fields, so a prompt that describes the runtime and
the runtime it describes cannot disagree, and a mode cannot be set here
by mistake: there is no field to set.

The committed references and ``tests/prompt_golden/`` are what prove a
change to any of this deliberate: an edit that was not meant to reach the
model shows up as a prompt diff in the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, Protocol

from duck.grid_utils import ARC_COLOR_LEGEND
from duck.palette import numeric_legend
from duck.state import MemoryEntry, StepSummary


@dataclass(frozen=True)
class PromptModes:
    """What the AGENT decides about a render, passed per call.

    Separate from the prompt surface on purpose: these describe machinery
    the agent owns (which globals the kernel installs, how long its session
    lives, what the tool actually truncates at), so a prompt surface that
    could hold its own copy could hold a wrong one. Frozen, and built in
    one place — ``DuckAgent._prompt_modes()``.

    No field carries a default, deliberately: a default here would be a
    second copy of a ``DuckAgent`` default, and a copy is stale from the
    next sweep that ships one.
    """

    frame_api: Literal["ergonomic", "numeric"]
    persistence: Literal["round", "move", "level", "always"]
    #: The file tools this agent offers, model-facing names in the order the
    #: suite is shown. Empty when the surface is off. The NAMES rather than a
    #: flag, because the prompt states them: two of the five are gated
    #: separately, and a surface that named all five regardless would promise
    #: the model a tool it was never given.
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
    #: Whether the agent keeps the journal: the `memory` tool and the
    #: snapshot at the system prompt's tail. Off, no sentence may speak of
    #: either — a prompt that named a tool the model was not given would
    #: have it calling one.
    journal: bool
    #: Whether the kernel grants the wide sandbox, whose modules the prompt's
    #: importable-modules line then names.
    wide_sandbox: bool
    #: Whether a full context is compacted by a summary the model writes
    #: (``compaction_strategy="summarize"``) rather than by eviction.
    context_summary_enabled: bool
    #: Whether the conversation rebuilt around a summary lists the model's
    #: files and ``python`` objects, which the summary request then says.
    add_files_and_repl_to_summary: bool

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
    """Every prompt the duck sends, and the configuration it renders under.

    An interface so the agent depends on the calls it makes rather than on
    one wording of them. The variation that belongs behind this protocol is
    a genuinely different way of conducting the conversation; a wording
    tweak is a field override on :class:`DuckPrompts` instead.

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
    game_over_refusal: str
    already_acted_skip: str
    stop_detail_run_complete: str
    stop_detail_game_over: str
    stop_detail_level_completed: str
    stop_detail_done: str
    stop_detail_max_moves: str
    stop_detail_terminal: str
    stop_detail_invalid_action: str
    invalid_action_error: str
    stage_error_unknown_action: str
    stage_error_undo_disabled: str
    stage_error_mouse_coords: str
    memory_budget_not_positive: str
    memory_empty_text: str
    memory_entry_too_large: str

    def system_prompt(self, modes: PromptModes, memory_snapshot: str) -> str:
        """The full system prompt over the frozen journal ``memory_snapshot``."""
        ...

    def python_tool_description(self, modes: PromptModes) -> str:
        """The `python` tool's description."""
        ...

    def python_tool_code_param_description(self, modes: PromptModes) -> str:
        """The `python` tool's `code` parameter description."""
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
        level_tokens_used: int = 0,
        level_token_allowance: int | None = None,
    ) -> str:
        """The per-turn user prompt over the agent-computed facts.

        ``level_tokens_used`` / ``level_token_allowance`` are the level's
        completion-token spend and cap (``None`` when uncapped) — a
        per-turn fact a surface MAY render; the shipped ones do not.
        """
        ...

    def user_prompt_game_over_lines(self, modes: PromptModes | None = None) -> list[str]:
        """Loss guidance replacing the act/grounding lines on a lost frame."""
        ...

    def followup_nudge(self, completion_limit: int | None = None, modes: PromptModes | None = None) -> str:
        """The no-action nudge, cutoff-aware when a limit is known."""
        ...

    def rounds_pruned_note(self, n: int) -> str:
        """The note that rounds of the current turn were dropped."""
        ...

    def compaction_prompt(
        self, modes: PromptModes, summary_tokens: int, *, current_level: int, previous_summary: bool = False
    ) -> str:
        """The request for a summary that replaces the history."""
        ...

    def compaction_wrapper(self, summary: str) -> str:
        """The user message carrying one accepted summary."""
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
class DuckPrompts:
    """The duck's prompt surface: every field is yours to set.

    Set it on the instance the agent carries —
    ``DuckAgent(prompts=DuckPrompts(...))`` — and a sweep overrides any one
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
    # the default on weak evidence (2026-09-27; ``evidence.md``,
    # "Sweeps 12 and 13").
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
    # ``evidence.md``, "Sweeps 12 and 13").
    thorough_summary: bool = True

    # State that every level can be won, and what a proof of the opposite
    # means. Passes that concluded a level was impossible stopped
    # experimenting and idled out its budget, where passes that kept testing
    # the rule behind the contradiction won it. On, on no evidence, as the
    # more powerful choice going forward (2026-09-27; ``evidence.md``,
    # "Sweeps 12 and 13").
    winnable_line: bool = True

    # Say that an action which did nothing once may do something later, in
    # the game overview and again on each new level. Losing passes on the
    # sweep 13 critical levels tried an action a few times, wrote it into
    # their notes as inert, and never tried it again after new objects or a
    # new mode appeared (glass_orchard level 3: every winner clicked the new
    # piece, two thirds of the losers never did). (2026-09-25.) On, on
    # no evidence, as the more powerful choice going forward
    # (2026-09-27; ``evidence.md``, "Sweeps 12 and 13").
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

    #: How the board is described, and the legend for reading it. Numeric
    #: mode states the glyph for all sixteen indices including where glyph
    #: and index coincide: a legend covering only 10-15 would leave the
    #: model to work out where the correspondence stops.
    board_legend_lines: dict[str, str] = field(
        default_factory=lambda: {
            "chars": (
                "- In this environment, boards are presented as 64 x 64 color grids rendered with ARC color symbols.\n"
                f"- Color legend: {ARC_COLOR_LEGEND}.\n"
            ),
            "numeric": (
                "- In this environment, boards are 64 x 64 grids of color indices 0-15. `.ascii` renders each "
                "index as one character: digits for 0-9, letters a-f above that.\n"
                f"- Color legend (index, `.ascii` character, name): {numeric_legend()}.\n"
            ),
        }
    )
    #: When ``no_hand_arithmetic_line`` is on, after the frame API lines, whose
    #: last one is where ``.ascii`` printing is advised: the caveat has to
    #: follow the board's introduction to read. The status prompt never
    #: carries a board, so what it speaks of is a board the model printed.
    printed_board_no_hand_arithmetic_line: str = (
        "- Boards you print with `.ascii` are for orienting yourself, not for counting, indexing or diffing cells "
        "by hand: the kernel does that work exactly, and hand-transcribing rows is where miscounts come from.\n"
    )

    #: Introduces the palette legend image, which leads the prompt that
    #: carries it. In the user prompt rather than the system prompt because
    #: the image rides one turn per level: a standing claim that it is
    #: attached would be false on every other turn.
    palette_legend_caption: str = (
        "Palette legend image: one swatch per color index, labelled with the index "
        "(top-left), the character `.ascii` renders it as (top-right), and its name."
    )

    #: The name-only legend's caption. It names what the image shows and
    #: nothing else: the swatch-to-name mapping is the whole content, and a
    #: caption promising indices would describe the other legend.
    color_name_legend_caption: str = (
        "Color legend image: one swatch per color, labelled with the color name used in this prompt."
    )
    #: Spent only on a game whose engine offers MOUSE.
    prior_mouse: str = "- For `MOUSE`, pass `row` and `col` integer arguments. `row` is vertical position, `col` is horizontal position.\n"
    visual_game_addendum: str = (
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

    runtime_header: str = "\n\nTool and Python interface:\n"
    runtime_tool_line: str = "- You have exactly two native tools: `python` and `memory`. `python` runs code; `memory` appends durable game knowledge.\n"
    # The same sentence with the file suite attached. A separate field rather
    # than a suffix: the count is in the middle of the sentence, so the two
    # cannot be one string with a piece added.
    #
    # A template rather than a literal, because two of the file tools are
    # gated: `{count}` is the total spelled out and `{names}` the file tools
    # as the model will be given them, both filled from the same suite the
    # agent builds. The plural verb needs no branch — three of the five are
    # not optional, so the list is never shorter than three.
    runtime_tool_line_with_files: str = (
        "- You have {count} native tools. `python` runs code; `memory` appends durable game knowledge; "
        "{names} read and write files in your working directory.\n"
    )
    # The two sentences above for an agent without the journal, which has no
    # `memory` tool to name.
    runtime_tool_line_without_journal: str = "- You have exactly one native tool: `python`, which runs code.\n"
    runtime_tool_line_with_files_without_journal: str = (
        "- You have {count} native tools. `python` runs code; {names} read and write files in your working directory.\n"
    )
    # What the working directory IS, told once where the rest of the
    # runtime is described. The import claim is the load-bearing one: it is
    # the difference between a place to park text and a place to build.
    runtime_file_lines: str = (
        "- The file tools act on one flat working directory — no subdirectories, and any file name is allowed. "
        "It is the same directory your `python` snippets run in, so a file you write there is a module you can import by its name without the `.py`.\n"
        "- An imported module runs with ordinary Python: the whole standard library, and `class` definitions, "
        "which a snippet itself cannot use. Anything that wants to be a class belongs in a file.\n"
        "- A module does not see `frames`, `current_frame`, `submit_action` or the other preloaded names: they exist "
        "only in your snippets, so pass what a function needs in as arguments.\n"
        "- Prefer files you can read back. `view_file` shows text; anything it cannot decode it reports by name and size only, so a file you cannot read is a file you cannot check.\n"
    )
    runtime_action_global_line: str = "- Inside `python`, `submit_action(actions)` is a preloaded function, not a native tool or an importable module. Call it directly.\n"
    runtime_import_line: str = "- `numpy` is preloaded as `np`. Importable modules: bisect, collections, copy, difflib, fractions, functools, heapq, itertools, json, math, numpy, operator, random, re, statistics, string, textwrap.\n"
    runtime_import_line_wide_sandbox: str = "- `numpy` is preloaded as `np`. Importable modules: bisect, collections, copy, difflib, fractions, functools, hashlib, heapq, importlib, inspect, itertools, json, math, numpy, operator, pickle, random, re, scipy, statistics, string, textwrap, time, traceback.\n"
    runtime_projection_lines: str = (
        "- Preloaded timeline types: `frames: list[FrameView]`, `animations: list[list[FrameView]]`, `actions: list[str]`, and `level_transitions: list[bool]`.\n"
        "- `frames[0]` is the initial settled frame. Action `actions[t]` is chosen from `frames[t]`; `animations[t+1]` contains the one or more frames it produced and ends at settled `frames[t+1]`.\n"
        "- Aliases: `current_frame is frames[-1]`; `last_action == actions[-1]` when any action exists, otherwise `None`. `valid_actions: list[str]` contains the currently legal names; `last_action_result: dict` is the latest submitted result and survives inspection-only calls.\n"
    )
    frame_common_lines: dict[str, str] = field(
        default_factory=lambda: {
            "chars": "- `FrameView.step: int`, `.level: int`, and `.shape: tuple[int, int]`. `.ascii: str` is one newline-delimited letter-coded board; call `.splitlines()` before indexing it by row and column.\n",
            "numeric": "- `FrameView.step: int`, `.level: int`, and `.shape: tuple[int, int]`. `.ascii: str` is one newline-delimited board of index characters; call `.splitlines()` before indexing it by row and column.\n",
        }
    )
    frame_api_lines: dict[str, str] = field(
        default_factory=lambda: {
            "ergonomic": (
                "- `FrameView.array` is the same board as a cached 2-D NumPy array of one-character strings. Prefer it for indexing, slicing, masks, and diffs. Raw numeric color ids are unavailable.\n"
                "- `FrameView.segment()` lazily returns and caches `{'nodes': list[dict], 'adjacency_list': list[list[int]]}`. Each node is one 4-connected same-color object: `id: int`, `color: str`, `hash: str`, `pixels: list[list[int]]` as `[row, col]` cells, `pixel_count: int`, `boundary: list[list[int]]`, and `children: list[int]`.\n"
                "- Use `.segment()` as the primary structural view. Print only compact summaries or small `.ascii` regions, never a full board.\n"
            ),
            "numeric": (
                "- `FrameView.array` is the same board as a cached 2-D NumPy array of color indices, dtype int8. Prefer it for indexing, slicing, masks, arithmetic and diffs: `array == 5` selects every cell of color 5.\n"
                "- `FrameView.segment()` lazily returns and caches `{'nodes': list[dict], 'adjacency_list': list[list[int]]}`. Each node is one 4-connected same-color object: `id: int`, `color: int`, `hash: str`, `pixels: list[list[int]]` as `[row, col]` cells, `pixel_count: int`, `boundary: list[list[int]]`, and `children: list[int]`.\n"
                "- Use `.segment()` as the primary structural view. Print only compact summaries or small `.ascii` regions, never a full board.\n"
            ),
        }
    )

    structured_runtime_state_live_lines: str = (
        "- Call `submit_action(actions)` to execute one or more real environment actions from Python.\n"
        "- Pass `submit_action(actions)` a list like `['LEFT']` or `[{'action': 'MOUSE', 'row': 4, 'col': 7}]`.\n"
        "- One action usually returns one frame, but a single action can result in a short multi-frame animation.\n"
        "- `submit_action(actions)` returns the executed result. After it returns, `frames`, `animations`, `actions`, `current_frame`, `last_action`, `valid_actions`, and `last_action_result` are refreshed.\n"
    )

    #: The per-mode lines: how often an image arrives, and what it is a
    #: picture OF — the two claims that differ between the modes. The rest of
    #: the addendum is shared; a prompt that overstated either claim would be
    #: describing a conversation the model is not having.
    multimodal_mode_lines: dict[str, str] = field(
        default_factory=lambda: {
            "current_grid": (
                "- User turns include an attached image of the current ARC grid.\n"
                "- The image and `current_frame.ascii` are two representations of the same current frame."
            ),
            "level_start": (
                "- The first user turn of each level includes an attached image of the grid as that "
                "level began. Later turns in the level do not; `current_frame.ascii` is the current "
                "board.\n"
                "- The image is a fixed reference for the level, not a view of the current frame: "
                "after any action it shows where the level started."
            ),
        }
    )
    multimodal_context_addendum: str = (
        "\n\nMultimodal context:\n"
        "{mode_lines}\n"
        "- You can use images and other tools to understand the game state and guide your strategy, each may be useful depending on the current uncertainty.\n"
    )

    python_addendum_live_lines: str = (
        "- You can call `submit_action(...)` multiple times in one Python snippet, including inside loops. Each call updates the preloaded variables before execution continues.\n"
        "- A batch of actions stops executing automatically when the game reports `game_over`, `run_complete`, or `level_completed`; re-ground on the refreshed state before acting again.\n"
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
        "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool result will tell you that.\n"
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
    #: Added to either of the two above when the file tools are attached.
    #: The `code` parameter is where the model is told what survives, and a
    #: sentence about what is cleared is misleading while files are not.
    code_param_files_suffix: str = " Files in your working directory are not cleared with it."

    # The no-tool-call nudge, as ONE template for every configuration: the
    # slots carry what the runtime surface changes, so an override lands on
    # every mode rather than on whichever branch happened to be reachable.
    # followup_nudge appends tool_call_format_guidance at render time.
    action_nudge_guidance: str = "Investigate further if you need to, then act: append any genuinely new finding with the native `memory` tool, and call `submit_action(actions)` inside Python. "
    action_nudge_guidance_without_journal: str = (
        "Investigate further if you need to, then act: call `submit_action(actions)` inside Python. "
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
                "cleared when you next act on the game; you are told when that has happened."
            ),
            "level": (
                "- The `python` session keeps your definitions and variables across tool calls and across moves, so "
                "helpers you write stay available and results you compute can be reused instead of recomputed. It is "
                "cleared when a new level starts; you are told when that has happened."
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
    # What the files are FOR. Two channels now carry work between turns and
    # they are good at opposite things, so the division is stated before any
    # advice about either: the journal is always in the prompt and small,
    # files are large, runnable, and invisible until read. What to BUILD in
    # them stops at reusable analysis — a model that wants a simulator can
    # see for itself that it has somewhere to put one.
    solving_file_lines: str = (
        "- The journal is what you KNOW; files are what you BUILT. Findings, corrections and plans belong in "
        "the journal, which is in front of you every turn. Code and computed data belong in files, which are "
        "there only when you look.\n"
        "- When you write a file worth keeping, say so in the journal in one line. Old conversation is dropped "
        "as the game goes on, and the journal is then the only record that your files exist at all.\n"
        "- Keep your analysis in files rather than in one-off snippets: a board summariser, a frame diff, a scorer. Import them where you need them, and fix them in place with `str_replace` instead of re-sending the whole thing.\n"
    )
    # The lines above without the two that set files against the journal.
    solving_file_lines_without_journal: str = "- Keep your analysis in files rather than in one-off snippets: a board summariser, a frame diff, a scorer. Import them where you need them, and fix them in place with `str_replace` instead of re-sending the whole thing.\n"

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
    #
    # KNOWINGLY INACCURATE, and left that way on purpose. Files outlive every
    # reset for every duck, so naming them among what is gone is wrong here
    # too — not only when the file tools are attached. It stays because this
    # is the flag-off arm: correcting it would change what an unrelated sweep
    # reads, and no committed reference renders this text, so nothing would
    # catch the change. What it misdescribes is a snippet that wrote files
    # through numpy and then hit a level or move boundary, which is rare
    # enough to carry. With the tools on, files are the point rather than an
    # accident, and the notice below is used instead.
    repl_reset_notice: str = (
        "Your `python` session was cleared because {reason}: definitions, variables and files from earlier "
        "snippets are gone. Re-create anything you still need."
    )
    repl_reset_notice_with_files: str = (
        "Your `python` session was cleared because {reason}: definitions and variables from earlier snippets "
        "are gone. Your files are still there — re-create anything else you need."
    )

    # Asked on the move after a level transition, ahead of the checklist. A
    # surface whose host collects the level's notes itself leaves it empty.
    level_summary_request: str = "You have just completed a level. Before anything else, append one journal entry summarising what you learned on it: the objects that mattered, what each action did, and what won."
    # The turn order, rendered on the first move of a level only.
    checklist_header: str = "This turn:"
    checklist_orient_fresh: str = "1. Orient on the newest frame: `current_frame.segment()` for structure."
    checklist_decide: str = "2. Decide the next move or short sequence — explore, or advance the level. Write a search if the right order is not obvious."
    checklist_memory: str = (
        "3. Append anything new and durable with `memory`, before you act: the turn ends as soon as you do."
    )
    checklist_act: str = "4. Act with `submit_action([...])`."
    # Without the journal the memory step goes and the act step moves up.
    checklist_act_without_journal: str = "3. Act with `submit_action([...])`."

    # Replace the act/grounding lines on a lost frame. One field per line, so
    # a surface that rejects one of these claims overrides that line rather
    # than restating the other three.
    game_over_study_line: str = (
        "The frame above is the state you lost in — study it, since it shows exactly what went wrong."
    )
    game_over_reset_line: str = (
        "RESET is the only action that can be executed now. Append what caused the loss with the native "
        '`memory` tool if it is new, then call `submit_action(["RESET"])` inside Python; any other action is refused.'
    )
    game_over_reset_line_without_journal: str = (
        'RESET is the only action that can be executed now. Call `submit_action(["RESET"])` inside Python; '
        "any other action is refused."
    )
    game_over_reset_scope_line: str = (
        "RESET restores the current level to its starting state; the level you had reached is not lost."
    )
    game_over_carry_line: str = "Carry any new cause of the loss into persistent memory so the next attempt avoids it."

    system_prompt_opening: str = "You are a coding agent solving a grid-based puzzle game."

    # Refusal for acting past a loss, and the skip text put on tool calls
    # issued after an earlier call in the same response already acted.
    game_over_refusal: str = "The game is over: {action} cannot be executed. RESET is the only valid action."
    already_acted_skip: str = (
        "not executed: an earlier tool call in this response {verb} actions; re-issue next turn if still needed"
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
    stage_error_undo_disabled: str = "Action {index}: UNDO is disabled by expose_undo=False."
    stage_error_mouse_coords: str = "Action {index}: MOUSE requires integer row and col."
    invalid_action_error: str = "Action 1: {action} is not currently valid."
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
        "and `frames`/`actions` history){and_files} will not."
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
    compaction_request_tail: str = (
        "Reply in plain text only — do NOT call any tools (any earlier instruction to make a tool call is "
        "superseded for this reply). Do not add anything else."
    )
    compaction_wrapper_prefix: str = "[context compacted — the following is your own summary of your work so far]\n\n"
    compaction_closing_line: str = (
        "[context compacted — your `python` session and files are intact; continue from here]"
    )
    compaction_closing_line_without_files: str = (
        "[context compacted — your `python` session is intact; continue from here]"
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
        text += self._board_legend(modes)
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
            text += self.multimodal_context_addendum.format(mode_lines=mode_lines)
        text += self.visual_game_addendum + (self.prior_mouse if modes.mouse_available else "")
        text += self._python_addendum(modes)
        text += self.python_addendum_live_lines
        text += self._compact_tool_session_addendum(modes)
        if modes.context_summary_enabled:
            text += self.context_summary_line
        brevity = self.brevity_lines[self.brevity]
        if brevity:
            text += f"\n{brevity}"
        if modes.journal:
            text += self._memory_snapshot_addendum(memory_snapshot)
        return text

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
        """The sentence naming the native tools, file suite and journal included or not."""
        if not modes.file_tools:
            return self.runtime_tool_line if modes.journal else self.runtime_tool_line_without_journal
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
                self._runtime_tool_line(modes),
                self.runtime_action_global_line,
                self.runtime_import_line_wide_sandbox if modes.wide_sandbox else self.runtime_import_line,
                self.runtime_file_lines if modes.file_tools else "",
                self.runtime_projection_lines,
                self.frame_common_lines[self._palette_key(modes)],
                self.frame_api_lines[modes.frame_api],
                self.printed_board_no_hand_arithmetic_line if self.no_hand_arithmetic_line else "",
            )
        )

    def _palette_key(self, modes: PromptModes) -> str:
        """Which of the two board renderings the frames use."""
        return modes.frame_api if modes.frame_api == "numeric" else "chars"

    def _board_legend(self, modes: PromptModes) -> str:
        """How boards are rendered, and the legend for reading them."""
        return self.board_legend_lines[self._palette_key(modes)]

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
            file_lines = self.solving_file_lines if modes.file_tools else ""
        else:
            memory_line = ""
            optimize_line = "- Optimize for the shortest reliable sequence that advances the current goal. If confidence is low, program a discriminating probe.\n"
            memory_step = ""
            file_lines = self.solving_file_lines_without_journal if modes.file_tools else ""
        return (
            "\n\nSolving guidance:\n"
            + memory_line
            + "- IMPORTANT: Especially when the game is about making an agent navigate to a target, it is usually safer to write an explicit search algorithm such as BFS. More generally, when the objective is understood but the best action order is unclear, pathfinding, flood fill, BFS, DFS, beam search, shortest-path search, limited action-sequence search, or custom heuristics are all valid.\n"
            + optimize_line
            + "- Once the important state variables and action effects are sufficiently understood, stop probing and search in the inferred state space.\n"
            + file_lines
            + "- Keep tool-output context size minimal and decision-oriented so you can quickly compare before/after state. It's fine to write a lot of python code, just make the output short and interpretable.\n"
            f"- A strong default loop is: summarize the board, infer the desired change, write a small scorer or search, {memory_step}execute with `submit_action(...)`, then inspect again.\n"
            "- For object tracking, match objects by color, overlap, bounding box proximity, area change, and edge contact rather than by exact coordinates alone.\n"
            "- For frame diffs, summarize changed cells, color transitions, appearing/disappearing components, movement candidates, and small local row slices around the changed region.\n"
            "- After every action, verify whether gameplay objects changed or whether only a timer, progress bar, or remaining-step bar moved. Do not treat HUD-only changes as evidence that the move worked.\n"
            "- Call `submit_action(...)` inside Python rather than returning action text in chat.\n"
            "- If you do multiple actions in one tool call, be aware that this can waste many actions if it "
            "doesn't have the expected outcome. Consider adding intermediate `assert`s in the code to verify "
            "the state is changing as expected.\n"
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
                    self.persistence_lines[modes.persistence]
                    + ("\n" + self._files_lifetime_line(modes).rstrip("\n") if modes.file_tools else "")
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
        colors_note = (
            "Colors are indices 0-15. " if modes.frame_api == "numeric" else "Raw numeric colors are unavailable. "
        )
        return (
            f"Run one {self.python_lifetimes[modes.persistence]} Python snippet against preloaded ASCII game state. Available globals: "
            "`frames`, `animations`, `actions`, `level_transitions`, `current_frame`, `last_action`, `valid_actions`, "
            "`last_action_result`, and `submit_action(actions)` for executing one or more real environment actions. "
            "Frames expose `.ascii`, `.array`, `.segment()`, `.step`, `.level`, and `.shape`. "
            "For the latest before/after diff, compare `frames[-2]` and `frames[-1]`. "
            "For MOUSE, pass `row` and `col` integer fields; legacy x/y fields are rejected. "
            f"{colors_note}Use `.segment()` as the primary view and `.ascii` only for a small specific region. "
            "Use `print(...)` for compact output or assign final data to `result`."
        )

    def python_tool_code_param_description(self, modes: PromptModes) -> str:
        """The `code` parameter's description. Separate from the tool's own
        because it is where the model is told whether names it binds will
        still exist on the next call."""
        base = self.code_param_ephemeral if modes.persistence == "round" else self.code_param_persistent
        return base + self.code_param_files_suffix if modes.file_tools else base

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
        level_tokens_used: int = 0,
        level_token_allowance: int | None = None,
    ) -> str:
        """Compose the per-turn user prompt, in order: a kernel-reset notice
        if one is owed, what the previous batch of actions did, where the
        game stands and what is legal right now. A lost frame adds its reset
        guidance after those dynamic facts.

        Every argument is a fact the agent computed from its state — this
        method turns facts into text and holds no state of its own.
        """
        lines: list[str] = []
        if reset_reason:
            # First turn after a reset: say so before anything else, so the
            # model does not plan around helpers that no longer exist.
            notice = self.repl_reset_notice_with_files if modes.file_tools else self.repl_reset_notice
            lines.append(notice.format(reason=reset_reason))
        if summary is not None:
            count = summary.executed_count
            action_label = "action" if count == 1 else "actions"
            lines.append(f"The code executed {count or 0} {action_label} in the previous sequence.")
            rendered_actions = [str(name).strip() for name in summary.executed_actions if str(name).strip()]
            if rendered_actions:
                action_prefix = "Executed actions (first 10):" if len(rendered_actions) > 10 else "Executed actions:"
                lines.append(f"{action_prefix} {', '.join(rendered_actions[:10])}.")
            else:
                lines.append("Executed actions: none.")
            if summary.run_complete:
                lines.append("You have completed the run!")
            elif summary.level_transition:
                lines.append("You have progressed to a new level!")
                if self.retest_actions_line:
                    lines.append(self.level_start_retest_actions_line)
            else:
                lines.append("You are still on the same level.")
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
        lines.append(f"Valid actions right now: {valid_line}.")
        if game_over_active:
            lines.extend(self.user_prompt_game_over_lines(modes))
        else:
            if summary is not None and summary.level_transition and self.level_summary_request and modes.journal:
                lines.append(self.level_summary_request)
            lines.extend(self._checklist(summary, modes))
        return "\n".join(lines)

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
                self.checklist_orient_fresh,
                self.checklist_decide,
                self.checklist_act_without_journal,
            ]
        return [
            self.checklist_header,
            self.checklist_orient_fresh,
            self.checklist_decide,
            self.checklist_memory,
            self.checklist_act,
        ]

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
        return [
            self.game_over_study_line,
            self.game_over_reset_line,
            self.game_over_reset_scope_line,
            self.game_over_carry_line,
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
        self, modes: PromptModes, summary_tokens: int, *, current_level: int, previous_summary: bool = False
    ) -> str:
        """Ask for the state that must survive the history's replacement.

        What the request claims survives follows the agent: the journal's
        notes are named only where they exist, files only where the model has
        file tools, the earlier levels once there are any, and the listing
        only where the rebuilt conversation carries one. Under
        ``thorough_summary`` it asks for a thorough summary rather than a
        compact one, states no size, and asks for what still holds of the
        ``previous_summary`` where the history carries one."""
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
        return text + self.compaction_request_tail

    def compaction_wrapper(self, summary: str) -> str:
        """Carry one accepted summary into the rebuilt conversation."""
        return self.compaction_wrapper_prefix + summary

    def compaction_listing(
        self,
        modes: PromptModes,
        files: list[tuple[str, int]],
        objects: list[tuple[str, str, bool]],
        *,
        objects_unavailable: str | None = None,
    ) -> str:
        """The files section (only where the model has file tools) and the
        ``python`` objects section.

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
        """Close the rebuilt conversation, over the listing when there is one."""
        line = self.compaction_closing_line if modes.file_tools else self.compaction_closing_line_without_files
        if not session_intact:
            line = "[context compacted — your `python` session was reset; continue from the summary]"
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


#: A stand-in for the module-level renderings below — NOT the shipped duck's
#: configuration, which ``DuckAgent`` supplies at every render. The numeric
#: fields are zeroed rather than plausible: nothing here reads them, and a
#: real-looking value would be the copy of a default this class exists to
#: avoid.
STANDALONE_MODES = PromptModes(
    frame_api="ergonomic",
    persistence="round",
    file_tool_names=(),
    multimodal="",
    memory_context_tokens=0,
    tool_output_tokens=0,
    python_timeout_s=0,
    action_names=(),
    journal=True,
    wide_sandbox=False,
    context_summary_enabled=False,
    add_files_and_repl_to_summary=False,
)

#: The default nudge, for tests that pin the exact no-cutoff wording.
FOLLOWUP_NUDGE = DuckPrompts().followup_nudge()

# Defaults for a DuckPythonTool constructed on its own; DuckAgent overwrites
# both from its own configuration before the tool is ever used.
PYTHON_TOOL_DESCRIPTION = DuckPrompts().python_tool_description(STANDALONE_MODES)
PYTHON_TOOL_CODE_PARAM_DESCRIPTION = DuckPrompts().python_tool_code_param_description(STANDALONE_MODES)
