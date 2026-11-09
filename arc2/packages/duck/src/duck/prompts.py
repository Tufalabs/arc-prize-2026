"""Everything the model reads that is not its own output.

`DuckPrompts` follows the shape upstream's world-model prep settled on:
**fields are what holds for a whole game** (the wording itself, and the mode
switches that select between wordings), **method arguments are what changes
per turn**. Every fragment is a constructor-settable field, so an experiment
overrides one sentence without subclassing anything and the override reaches
wherever that sentence is composed.

The agent computes facts and never composes text; this module composes text
and never computes facts. What the AGENT decides — the numbers its tool and
its kernel actually enforce, and the feature gates that decide which runtime
names and media the model will really get — is not on the prompt class at
all: it arrives per render as a :class:`PromptModes`, so a prompt describing
a limit or a tool and the code enforcing it cannot disagree, and there is no
field here to set by mistake.

The CONTENT is new: upstream's duck plays a live multi-level game and its
prompt describes frames, actions and a journal, none of which exist here.
What is kept from upstream is everything that is not about the game — the
class shape, the tool-call format guidance, the brevity ask, the cutoff
prefix, the pruned-rounds note, the tool-limit lines — because those were
swept and this branch has measured nothing.

The wording flags -- each adding, changing or deleting one thing against the
wording as shipped -- are fields beside the fragments they select, so a sweep
over them changes only the selected wording. `PromptsConfigurable` is the name
they were reached through before they moved here, kept for the callers and
pickled runs that use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, Protocol

from duck.config import CheckMode

EMPTY_OUTPUT_NOTE = "(no output — the code ran without printing anything)"

#: A number in ``\boxed{}``, as a confidence is asked for; a percent sign
#: after it, escaped or not, is allowed.
_BOXED_NUMBER = re.compile(r"\\boxed\{\s*(\d+(?:\.\d+)?)\s*\\?%?\s*\}")


def _shape(shape: tuple[int, int]) -> str:
    """One grid size, always rows x cols. Stated in that order everywhere so
    the model never has to guess which number is which."""
    return f"{shape[0]}x{shape[1]}"


def truncation_note(omitted: int, budget_tokens: int) -> str:
    """Told, not silent: a model that cannot see the cut reads short output
    as its own code having produced less than it did."""
    return f"... [{omitted} more characters cut to stay within the ~{budget_tokens}-token output budget]"


@dataclass(frozen=True)
class PromptModes:
    """What the AGENT decides about a render, handed over per call.

    Deliberately not fields on the prompt surface. Each of these is something
    else's enforcement — the tool's truncation, the kernel's timeout, TAAF's
    grid rules and attempt count, and the feature gates that decide which
    names the kernel binds and which media the model is sent — so a surface
    able to hold its own copy is a surface able to hold a wrong one, and a
    wrong one reads exactly like configuration that works. Frozen, and built
    in one place: ``DuckAgent._prompt_modes()``.
    """

    tool_output_tokens: int = 2425
    python_timeout_s: int = 78
    n_attempts: int = 2
    max_side: int = 30
    n_colors: int = 10
    # Feature gates. Each names something the kernel or the agent really
    # provides; the defaults are the standard Platypus surface.
    check_mode: CheckMode = CheckMode.DISABLED
    check_and_submit: bool = False
    images_enabled: bool = True
    text_grids: bool = False
    image_layout: Literal["paired", "separate"] = "paired"
    image_legend: bool = True
    show_enabled: bool = False
    max_shows: int = 6
    check_images_enabled: bool = False
    context_summary_enabled: bool = False
    #: Whether the continuation nudge may name the reply cap the last reply
    #: hit. The agent's flag (``DuckAgent.completion_cutoff_notice``), handed
    #: over like the rest so the surface reads it rather than holding a copy.
    completion_cutoff_notice: bool = False
    #: Generated-token cap for the puzzle; ``None`` means no game token cap.
    game_token_allowance: int | None = None
    #: The file tools the agent offers, in the order it offers them; empty
    #: without them. The prompt names exactly this set.
    file_tool_names: tuple[str, ...] = ()
    #: Whether the conversation rebuilt around a summary closes on the
    #: listing of the model's files and ``python`` objects.
    add_files_and_repl_to_summary: bool = False
    #: Whether a `python` call that changes ``test_out`` is followed by each
    #: changed answer as an image, beside its test input.
    show_changed_answers: bool = False

    @property
    def file_tools(self) -> bool:
        """Whether the model has the file tools, and with them a working
        directory to keep ``candidate.py`` in."""
        return bool(self.file_tool_names)

    @property
    def kernel_only(self) -> bool:
        """Whether the prompt shows no grid contents at all — no pictures and
        no digit rows — so the puzzle exists for the model only as the arrays
        the kernel binds. The names and shapes are still listed; what is
        withheld is every cell value."""
        return not self.images_enabled and not self.text_grids

    #: How a rendered row is laid out. The prompt's description of a render
    #: has to match what the kernel prints and what the task prompt shows.
    ascii_row_numbers: bool = False


class Prompts(Protocol):
    """What the agent needs from a prompt implementation.

    A seam, not a base class: a variant surface is a different object carried
    in ``DuckAgent.prompts``. The split between a field and an argument is
    the split between what the SURFACE owns and what the AGENT decides:
    wording is a field, :class:`PromptModes` is an argument, and per-turn
    facts are arguments. So an implementation cannot hold a stale mode —
    there is no field to hold it in.

    Where a string is written here and read back by the agent — the
    pruned-rounds note — the reader is on this protocol too, so one object
    owns both halves and an override cannot silently break the scan.
    """

    def system_prompt(self, modes: PromptModes) -> str:
        """The standing instructions."""
        ...

    def python_tool_description(self, modes: PromptModes) -> str:
        """The `python` tool's description."""
        ...

    def python_tool_code_param_description(self) -> str:
        """The `code` parameter's description."""
        ...

    def task_intro(self, modes: PromptModes, *, n_train: int) -> str:
        """The line introducing the example pairs."""
        ...

    def pair_label(self, index: int, *, in_shape: tuple[int, int], out_shape: tuple[int, int]) -> str:
        """The caption above one example pair's image."""
        ...

    def train_input_label(self, index: int, *, shape: tuple[int, int]) -> str:
        """The caption above one separately rendered training input."""
        ...

    def train_output_label(self, index: int, *, shape: tuple[int, int]) -> str:
        """The caption above one separately rendered training output."""
        ...

    def question_label(self, index: int, *, shape: tuple[int, int]) -> str:
        """The caption above one test input's image."""
        ...

    def test_intro_text(self, modes: PromptModes, *, n_test: int) -> str:
        """The line introducing the test inputs."""
        ...

    def followup_nudge(
        self, modes: PromptModes, completion_limit: int | None = None, problems: list[str] | None = None
    ) -> str:
        """The continuation prompt after a reply that did not finish.

        ``completion_limit`` is the cap the reply hit, when the agent's
        cutoff-notice gate permits naming it;
        ``problems`` is what is still wrong with the answer, in the model's
        own index vocabulary. An empty list means the answer stands, which
        the agent nudges only after a reply the cap cut off. A surface may
        ignore either."""
        ...

    def budget_status(self, used: int, allowance: int | None, modes: PromptModes) -> str:
        """The standing budget note, empty when the surface does not request it."""
        ...

    def budget_warning(self, remaining: int, modes: PromptModes) -> str:
        """The one-off note that the game budget is nearly spent."""
        ...

    def answer_repair(self, problems: list[str]) -> str:
        """What is wrong with the answer as it currently stands."""
        ...

    def mandatory_answer_repair(self) -> str:
        """The re-entry after a tool call that did not pass ``check(fn)``."""
        ...

    def malformed_nudge(self, modes: PromptModes, problems: list[str] | None = None) -> str:
        """The continuation prompt after a reply refused as malformed on every
        retry and dropped (``DuckAgent.malformed_nudges``): in practice a reply
        that went in circles inside its reasoning and ended with no text and no
        tool call. The model never sees the dropped reply. ``problems`` as for
        :meth:`followup_nudge`; an empty list means the answer stands."""
        ...

    def rounds_pruned_note(self, n: int) -> str:
        """The note that ``n`` rounds of the current turn were dropped."""
        ...

    def is_rounds_pruned_note(self, text: str) -> bool:
        """Whether ``text`` is a note this surface rendered — the reader
        paired with :meth:`rounds_pruned_note`, so a refreshed note replaces
        the previous one instead of stacking beside it."""
        ...

    def compaction_prompt(
        self,
        modes: PromptModes,
        summary_tokens: int,
        *,
        previous_summary: bool = False,
        candidate_shown: bool = False,
        earlier_summaries: bool = False,
    ) -> str:
        """The one-off request for a model-written context summary.
        ``previous_summary`` is whether the history carries one already;
        ``candidate_shown`` whether the rebuilt conversation shows
        ``candidate.py`` in full; ``earlier_summaries`` whether the history
        carries intermediate summaries, which go with it."""
        ...

    def intermediate_summary_prompt(self, modes: PromptModes) -> str:
        """The request for a summary that leaves the history whole
        (``DuckAgent.intermediate_summary_prompt_lengths``)."""
        ...

    @property
    def intermediate_summary_continue(self) -> str:
        """The message after an intermediate summary."""
        ...

    def compaction_wrapper(self, summary: str) -> str:
        """The summary as a user message in the rebuilt conversation."""
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

    def compaction_candidate(self, source: str) -> str:
        """``candidate.py``'s full text, as the rebuilt conversation shows it."""
        ...

    def compaction_closing(self, modes: PromptModes, listing: str, *, session_intact: bool = True) -> str:
        """The user message closing a rebuilt conversation, over the listing."""
        ...

    @property
    def palette_caption(self) -> str:
        """The caption above the colour legend."""
        ...

    def task_footer_text(self, modes: PromptModes) -> str:
        """The line closing the task prompt, in the configured media."""
        ...

    def audit_intro(self, modes: PromptModes, *, n_train: int) -> str:
        """The audit prompt's opening and the line introducing the example pairs."""
        ...

    def audit_test_intro(self, *, n_test: int) -> str:
        """The line introducing the test inputs and the answers standing."""
        ...

    def answer_slot_label(self, index: int, attempt: int, *, shape: tuple[int, int]) -> str:
        """The caption above one answer slot shown as a grid."""
        ...

    def answer_slot_note(self, index: int, attempt: int, *, same_as: int | None = None, why: str = "") -> str:
        """One answer slot not shown as a grid: a copy of slot ``same_as``,
        unusable for the reason ``why``, or else empty."""
        ...

    def audit_close(self, reply: str) -> str:
        """The audit prompt after the grids: the last ``reply`` quoted, and the questions."""
        ...

    def read_audit_confidence(self, text: str) -> int | None:
        """The confidence a reply to the audit gave, or None where it gave
        none or was not asked for one: the reader paired with the question
        asking for it."""
        ...

    def changed_answers_intro(self, slots: list[str]) -> str:
        """The line heading one test input's changed answers, ``slots``
        naming each changed slot as the kernel does."""
        ...


@dataclass
class DuckPrompts:
    """The duck's prompt surface for ARC-AGI-2: wording, all of it settable.

    Every field here is user configuration — ``DuckAgent(prompts=DuckPrompts(
    ...))`` — prompt-only switches first, then the wording fragments the
    methods compose at render time. Setting any of them does something.

    The switches are flags, each an ordinary ``Field("inner.prompts.<flag>")``
    sweep target that adds, changes or deletes one thing. Flags leave the
    agent's tools and answer channel unchanged: assignment guidance is absent
    in mandatory-check mode, second-attempt wording needs a second slot, and
    digit-row guidance follows where the rows are. Analyses should condition
    those effects on the modes where they apply.

    The numbers the prompt quotes and the gates that select between wordings
    are NOT here. They belong to whatever enforces them and arrive per render
    as a :class:`PromptModes`, so this class has no mode to set by mistake
    and no copy that can go stale.
    """

    # Ask the model to keep its reasoning proportionate. Steering, not
    # enforcement: the hard ceiling is the sampling profile's max_tokens,
    # which exists to stop a pathological ramble, not to shape ordinary
    # turns. The instruction lives once in the system prompt. Off by
    # default — it buys nothing measurable, and a prompt earns its place.
    brevity_line: bool = False
    # Ask for the likeliest answer in attempt 0. An agent that composes
    # answers from several takes (``platypus.multi_take``) reads a take's
    # attempt 0 as its first choice; a single play has no reader for the
    # order, so it is off by default.
    likeliest_first_line: bool = False

    # -- flags: each adds, changes or deletes one thing against the wording
    # -- below, and the defaults are what the duck ships ----------------------

    # DELETE the two sentences that license open-ended investigation —
    # "Build up the tools you want" and "Investigate until your code has a
    # clear probe or plan" — leaving the session block with its one fact.
    investigation_lines: bool = True
    # ADD a "How to work" block: facts from the kernel first, a checked
    # candidate after at most two inspection calls, pure candidates.
    work_policy: Literal["none", "candidate_first"] = "none"
    # ADD one generalization audit after a passing check: what the test has
    # that no training input has, and whether a different rule also fits.
    # Off beside the audit turn, which asks the same at the stop: with the
    # line on, the play does part of the audit's work for 5.6k more effective
    # tokens a game and the audit gains 1.1 points less, the score level
    # (sweep 9, +0.71 throughput-adjusted with it off, p=0.03; ``evidence.md``,
    # "Duck sweeps 8 and 9").
    generalization_audit_line: bool = False
    # CHANGE the second-attempt line. ``optional``: the slot is offered and
    # "a bad one costs nothing". ``audit``: the slot is the audit's strongest
    # alternative, never a copy or a tweak of attempt 0. ``none``: the slot
    # is not mentioned.
    second_attempt_mode: Literal["optional", "audit", "none"] = "optional"
    # ADD: record ``fn(test_in[i])`` in the call the check passes, then keep
    # refining — a passing candidate left in a variable scores nothing.
    assign_early_line: bool = False
    # ADD: with several test inputs, answer each as soon as it is known.
    multi_test_line: bool = False
    # CHANGE the digit-rows line: the rows are for orientation, not for
    # counting, indexing or diffing cells by hand. Where the prompt shows no
    # digit rows, the same caveat rides the ``ascii_grid`` runtime line
    # instead, over the rows the model prints for itself. On: sweep 7 read
    # the digit rows with the caveat +2.8 over the rows alone (p=2e-4).
    no_hand_arithmetic_line: bool = True
    # CHANGE the ``test_out`` description so it cannot be read as
    # ``[None, None]`` itself: one entry per test input, a pair of slots
    # per entry, never appended to or reshaped.
    test_out_structure_explicit: bool = False
    # CHANGE the no-tool-call nudge. ``legacy`` is the baseline's one text;
    # ``causal`` says what happened — the cap that cut the reply, named
    # under ``completion_cutoff_notice``, or the answer's own problems as
    # the repair prompt phrases them. Causal: the same score, and with the
    # throughput priced at its Kaggle value sweep 7 read it +1.3 over legacy
    # (p=0.018; ``packages/duck/evidence.md``, "The calls at the Kaggle weight").
    nudge_mode: Literal["legacy", "causal"] = "causal"
    # DELETE the tool-call-format sentences from every nudge. Off: sweep 7
    # read the sentences at -0.6 (p=0.18), and nothing argued for them.
    nudge_format_guidance: bool = False
    # ADD a remaining-budget line to every Python result while capped.
    budget_line: bool = False
    # CHANGE the inherited ``brevity_line`` text; that boolean still gates it.
    brevity: Literal["full", "half"] = "full"
    # ADD the enforced token costs and answer channel, without a work policy.
    cost_model: Literal["none", "explicit"] = "none"
    # CHANGE the summary request: a thorough summary rather than a compact
    # one, at no stated size, and from the second summary on one that also
    # carries over what still holds of the previous one. On, a judgment call:
    # level on the score in sweeps 8 and 9, 1.6k fewer effective tokens a game
    # in sweep 9 (``evidence.md``).
    thorough_summary: bool = True
    # ADD what the audit turn (``DuckAgent.audit_turn``) builds on, without
    # naming it: the stopping line and the task prompt ask for the rule in
    # full in the final reply, which the audit quotes, and the summary
    # request asks for the doubts and inconsistencies noticed so far. A flag
    # of its own, so the audit's flag changes nothing before its turn fires
    # and a game cut there is a game without it. Where a passing mandatory
    # check ends the game there is no final reply, and the rule is not asked
    # for. On, a judgment call: +0.4 in sweep 9 and +0.6 in sweep 8, short of
    # significance in each (``evidence.md``).
    audit_groundwork: bool = True
    # ADD the audit's last question: how confident the model is, from 0 to 100,
    # that its answer is right, in `\boxed{}`, which the agent records on the
    # game's span as ``audit_confidence``. On: asking costs nothing measurable
    # on the audit's gain, the one outcome it can act through (sweep 9, +0.07,
    # p=0.86), and every sweep 9 arm but four jobs' asked it. The number is a
    # weak signal: it ranks tasks by difficulty, and separates a task's won
    # games from its lost ones only 59% of the time (``evidence.md``).
    audit_confidence_line: bool = True

    # -- user configuration: wording, sweepable leaf fragments -------------

    # Upstream's, verbatim: a Qwen-shaped failure mode, where the model
    # writes the tool-call markup inside prose or a markdown fence.
    tool_call_format_guidance: str = (
        "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt for this model. "
        "Use only that format; do not add markdown fences, prose wrappers, or alternate tool-call syntax. "
        "Do not quote or place tool-call markup inside explanatory text; when you decide to call the tool, emit the tool call itself."
    )

    system_prompt_opening: str = "You are a coding agent solving an ARC-AGI-2 puzzle."

    puzzle_overview_addendum: str = (
        "\n\nPuzzle overview:\n"
        "- An ARC-AGI-2 puzzle is a small number of example input/output pairs that share one hidden "
        "transformation rule, and one or more test inputs you must apply that rule to.\n"
        "- Grids are 2-D numpy arrays of small integers, one integer per colour. You are shown them as "
        "pictures; the exact values are in the kernel.\n"
        "- A colour legend opens the puzzle: each swatch shows the integer the kernel holds for that colour "
        "and the colour's name. Those integers are the values in the arrays, and `ascii_grid` prints each "
        "as its own digit.\n"
    )
    unlegended_image_overview_addendum: str = (
        "\n\nPuzzle overview:\n"
        "- An ARC-AGI-2 puzzle is a small number of example input/output pairs that share one hidden "
        "transformation rule, and one or more test inputs you must apply that rule to.\n"
        "- Grids are 2-D numpy arrays of small integers, one integer per colour. You are shown them as "
        "pictures; the exact values are in the kernel.\n"
        "- Pictures use the standard ARC colours: 0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, "
        "6 fuchsia, 7 orange, 8 teal and 9 brown.\n"
    )
    text_grid_overview_line: str = (
        "- The task prompt also spells every grid out as digit rows, so use the pictures for global "
        "structure and the rows for exact cell values.\n"
    )
    #: Added wherever the prompt shows digit rows and the rows are numbered.
    #: Without it the leading index reads as a cell, which is the one way a
    #: numbered row can be worse than an unnumbered one.
    text_grid_row_numbers_line: str = (
        "- Each of those rows begins with its row index and a tab; the index is a label, not a cell.\n"
    )
    kernel_only_puzzle_overview_addendum: str = (
        "\n\nPuzzle overview:\n"
        "- An ARC-AGI-2 puzzle is a small number of example input/output pairs that share one hidden "
        "transformation rule, and one or more test inputs you must apply that rule to.\n"
        "- Grids are 2-D numpy arrays of small integers, one integer per colour. This prompt names the "
        "pairs and their sizes and shows no cell values: the arrays are in the kernel, and "
        "`ascii_grid(arr)` prints one as text.\n"
        "- Colour values are 0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 fuchsia, 7 orange, "
        "8 teal and 9 brown.\n"
    )
    text_only_puzzle_overview_addendum: str = (
        "\n\nPuzzle overview:\n"
        "- An ARC-AGI-2 puzzle is a small number of example input/output pairs that share one hidden "
        "transformation rule, and one or more test inputs you must apply that rule to.\n"
        "- Grids are 2-D numpy arrays of small integers, one integer per colour. Every grid is shown as "
        "digit rows and is also available exactly in the kernel.\n"
        "- Colour values are 0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 fuchsia, 7 orange, "
        "8 teal and 9 brown.\n"
    )

    runtime_header: str = "\n\nTool and Python interface:\n"
    runtime_tool_line: str = "- You have exactly one native tool: `python`. It runs code in a stateful kernel.\n"
    # The tool line where the file tools are attached. A template: `{count}`
    # is the total spelled out and `{names}` the file tools as the model will
    # be given them, both filled from the suite the agent builds.
    runtime_tool_line_with_files: str = (
        "- You have {count} native tools. `python` runs code in a stateful kernel; {names} read and write files in "
        "your working directory.\n"
    )
    # What the working directory IS, told once, at the end of the runtime
    # block, after the names it refers to. The import claim is the
    # load-bearing one: it is the difference between a place to park text
    # and a place to build.
    runtime_file_lines: str = (
        "- The file tools act on one flat working directory — no subdirectories, and any file name is allowed. "
        "It is the same directory your `python` snippets run in, so a file you write there is a module you can "
        "import by its name without the `.py`.\n"
        "- `python` runs ordinary Python in that directory: `open()` reads and writes files there, and a write "
        "anywhere else is refused.\n"
        "- A module you import from there sees the preloaded names a snippet does — `train_in`, `test_in`, `check`, "
        "`np` and the rest.\n"
        "- A module runs from disk the first time a `python` call imports it, as a new module each time: a name "
        "bound to it in an earlier call keeps the version it was bound to until you import it again.\n"
        "- Prefer files you can read back: `view_file` shows text.\n"
    )
    # Where the candidate lives under the file tools, and the example of a
    # call using it, which follows the answer channel. The example imports
    # the module rather than the function: importing the module runs the
    # file as it now is, while a function imported in an earlier call keeps
    # the version from before an edit.
    candidate_file_line: str = (
        "- Keep your candidate `fn` in `candidate.py`: write it with `create_file`, fix it in place with "
        "`str_replace` instead of re-sending the whole thing, and import the module in each `python` call that "
        "uses it, which runs the file as it is now:\n"
    )
    candidate_example_check: str = (
        "```python\nimport candidate\ncheck(candidate.fn)\nfor i, grid in enumerate(test_in):\n"
        "    test_out[i][0] = candidate.fn(grid)\n```\n"
    )
    # Where a passing check submits or records the answer itself.
    candidate_example_check_answers: str = "```python\nimport candidate\ncheck(candidate.fn)\n```\n"
    candidate_example_no_check: str = "```python\nimport candidate\nfor i, grid in enumerate(test_in):\n    test_out[i][0] = candidate.fn(grid)\n```\n"
    runtime_import_line: str = (
        "- `numpy` is preloaded as `np`; `collections`, `copy`, `functools`, `itertools`, `json`, `math`, "
        "`os`, `re` and `sys` are imported. Anything else you can import yourself.\n"
    )
    runtime_names_lines: str = (
        "- Preloaded names: `train_in[i]` and `train_out[i]` are the example pairs as arrays; `test_in[i]` "
        "are the inputs you must answer; `test_out[i]` is where your answers go and starts as `[None, None]`.\n"
        "- Sizes are rows x cols throughout, and arrays index as `[row, col]`.\n"
    )
    mandatory_runtime_names_lines: str = (
        "- Preloaded names: `train_in[i]` and `train_out[i]` are the example pairs as arrays; `test_in[i]` "
        "are the inputs your function must transform. `test_out` is available as scratch space, but changing "
        "it does not submit an answer.\n"
        "- Sizes are rows x cols throughout, and arrays index as `[row, col]`.\n"
    )
    #: How a render is laid out, as one line either way rather than a base
    #: line plus a qualifier: two lines describing one render is how they
    #: come to disagree.
    ascii_grid_runtime_line: str = (
        "- `ascii_grid(arr)` renders a 2-D array as text, one digit per cell, rows on separate lines.\n"
    )
    ascii_row_numbers_runtime_line: str = (
        "- `ascii_grid(arr)` renders a 2-D array as text, one digit per cell. Each row is prefixed with its "
        "index in the array you passed and a tab; on a crop, add the slice's start.\n"
    )
    check_runtime_line: str = (
        "- `check(fn)` runs `fn(grid)` against every training pair. A failure raises `CheckError`; "
        "`err.report.pairs` gives exact per-pair shapes, mismatch counts, and the first five "
        "`(row, col, expected, got)` mismatches. A pass returns the report and does not change `test_out`; "
        "call it once per Python call, after choosing your candidate.\n"
        "- Failed `check` output grids are sent in full in the next user message; the ordinary stdout cap "
        "does not truncate them.\n"
    )
    mandatory_check_runtime_line: str = (
        "- `check(fn)` runs `fn(grid)` against every training pair. A failure raises `CheckError`; "
        "`err.report.pairs` gives exact per-pair shapes, mismatch counts, and the first five "
        "`(row, col, expected, got)` mismatches. When every training pair passes, `fn` is applied to every "
        "test input; an error or invalid output raises `CheckError` and submits nothing, otherwise those outputs "
        "are submitted immediately. Call it once per Python call, after choosing your candidate.\n"
    )
    check_and_submit_runtime_line: str = (
        "- `check(fn)` runs `fn(grid)` against every training pair. A failure raises `CheckError`; "
        "`err.report.pairs` gives exact per-pair shapes, mismatch counts, and the first five "
        "`(row, col, expected, got)` mismatches. When every training pair passes, `fn` is applied to every "
        "test input. If all predictions are valid, they automatically replace `test_out[i][0]` for every "
        "test input; other attempt slots are preserved. A training failure, test-input error or invalid "
        "prediction raises `CheckError` and leaves the saved answers unchanged. A pass returns the report. "
        "Call it once per Python call, after choosing your candidate.\n"
        "- Failed `check` output grids are sent in full in the next user message; the ordinary stdout cap "
        "does not truncate them.\n"
    )
    check_images_runtime_line: str = (
        "- Each renderable failed `check` output is sent both as digit rows and as a grid picture.\n"
    )
    show_runtime_line: str = (
        "- `show(grid, label='')` renders a grid in the next user message; at most {max_shows} grids may be "
        "queued per call.\n"
    )
    changed_answers_runtime_line: str = (
        "- Whenever a `python` call changes `test_out`, the next user message shows each changed answer as an "
        "image, beside its test input.\n"
    )

    answer_header: str = "\n\nHow to answer:\n"
    answer_line: str = (
        "- Assign a 2-D array of integers to `test_out[i][0]` for each test input `i` — `test_out` follows "
        "from `test_in` the way `train_out` follows from `train_in`. That is the only way to answer; there "
        "is nothing to submit and no other tool to call.\n"
    )
    mandatory_answer_line: str = (
        "- Implement your rule as `fn(grid) -> grid`, then call `check(fn)`. That is the only way to answer: "
        "assigning or editing `test_out` does not submit anything.\n"
    )
    check_and_submit_answer_line: str = (
        "- Implement your rule as `fn(grid) -> grid` and call `check(fn)`: a pass automatically saves "
        "its test predictions in `test_out[i][0]`. No separate assignment is needed. This does not end "
        "the task: you can keep refining, and each later passing check overwrites the first slots. "
        "You may also assign or edit `test_out` directly.\n"
    )
    second_attempt_line: str = (
        "- You may also assign `test_out[i][1]`. The two attempts are scored independently and the question "
        "counts as solved if either is right, so a second genuinely different hypothesis is worth having — "
        "and a bad one costs nothing. If you only have one answer, leave the second alone.\n"
    )
    #: Ends ``second_attempt_line`` under ``likeliest_first_line``.
    likeliest_first_text: str = "Put your likeliest answer in `test_out[i][0]`."
    checking_line: str = (
        "- Check yourself before you stop: apply your rule to every `train_in[i]` and compare against "
        "`train_out[i]`. A rule that does not reproduce the examples will not solve the test.\n"
    )
    check_checking_line: str = (
        "- Check yourself before you stop: choose your best rule, implement it as `fn(grid) -> grid`, then "
        "run `check(fn)` once. Study any reported differences on the next call. A rule that does not "
        "reproduce every example will not solve the test.\n"
    )
    grid_rules_line: str = (
        "- An answer grid must be 2-D, at most {max_side} on each side, and hold whole numbers in "
        "0..{max_last}. Floats are fine if they are whole; anything else comes back to you to fix.\n"
    )
    stopping_line: str = (
        "- When you are satisfied with what is in `test_out`, stop: reply without calling the tool. Your "
        "answer is read out of the kernel after every call, so there is no final step to remember and "
        "nothing is lost if you are cut off.\n"
    )
    mandatory_stopping_line: str = (
        "- A passing `check(fn)` submits immediately and ends the task. There is no confirmation phase or "
        "separate submission step. Until a check passes, keep working in Python.\n"
    )

    # Upstream's tool-limit block, the session line reduced to the one
    # lifetime there is here.
    compact_tool_session_head: str = (
        "\n\nTool limits and session lifetime:\n"
        "- The `python` session keeps your definitions and variables for the whole puzzle, so helpers you "
        "write stay available and results you compute can be reused instead of recomputed. Build up the "
        "tools you want rather than rewriting them each time.\n"
        "- You can call the `python` tool as many times as you want. Investigate until your code has a "
        "clear probe or plan.\n"
    )
    compact_tool_session_tail: str = (
        "- Each `python` tool call has a hard time limit of {python_timeout_s} seconds.\n"
        "- Tool responses are capped to about {tool_output_tokens} tokens. If a response is cut off, the tool "
        "result will tell you that.\n"
        "- Keep code snippets short and purpose-built rather than dumping large frameworks into one call.\n"
    )
    context_summary_line: str = (
        "- Your context is finite. When it fills, you will be asked to write a compact summary that "
        "replaces older turns, so investing in clear intermediate conclusions pays off.\n"
    )

    # Included once in the system prompt under brevity_line. Upstream's
    # wording with its game nouns swapped: asks for proportionate reasoning
    # rather than a fixed number, because a token budget the model cannot
    # measure invites it to guess, and the turns that genuinely need long
    # reasoning are the ones worth keeping.
    user_prompt_brevity_line: str = (
        "Keep your reasoning proportionate to the decision: think as long as the step genuinely needs, "
        "then act. Long deliberation is worth it for a rule you have not cracked; it is waste when you "
        "already know what to try. Every turn you spend re-deriving what you established earlier is a "
        "turn you are not solving."
    )

    # The `python` tool's own description and parameter help, which have to
    # agree with how the tool is actually configured.
    python_tool_description_text: str = (
        "Run one Python snippet in a game-scoped kernel against the preloaded puzzle. Available globals: "
        "`train_in`, `train_out`, `test_in`, `test_out`, `np`, and `ascii_grid(arr)`. Definitions and "
        "variables persist across calls. Use `print(...)` for compact output."
    )
    check_python_tool_description_text: str = (
        "Run one Python snippet in a game-scoped kernel against the preloaded puzzle. Available globals: "
        "`train_in`, `train_out`, `test_in`, `test_out`, `np`, `ascii_grid(arr)`, and `check(fn)`. "
        "Definitions and variables persist across calls. Use `print(...)` for compact output."
    )
    check_and_submit_tool_clause: str = (
        " A passing `check(fn)` automatically saves validated test predictions in `test_out[i][0]`, "
        "preserving other attempt slots and allowing further refinement."
    )
    show_tool_description_clause: str = " `show(grid, label='')` queues a grid for rendering in the next user message."
    code_param_persistent: str = (
        "Python code to run. Definitions and variables persist across tool calls in this session."
    )

    # -- per-turn prompts -------------------------------------------------

    palette_caption: str = "Colour legend:"
    task_header: str = "Here is the puzzle."
    train_intro: str = "{n} example pair{plural}, input on the left and output on the right:"
    kernel_train_intro: str = "{n} example pair{plural}, in train_in and train_out:"
    text_train_intro: str = "{n} example pair{plural}, shown as exact digit rows:"
    dual_train_intro: str = (
        "{n} example pair{plural}, input on the left and output on the right; exact digit rows follow each label:"
    )
    separate_train_intro: str = "{n} example pair{plural}, each input followed by its output:"
    separate_dual_train_intro: str = "{n} example pair{plural}, each grid shown as exact digit rows and a picture:"
    test_intro: str = "\n{n} test input{plural} to answer, in test_in:"
    text_test_intro: str = "\n{n} test input{plural} to answer, shown as exact digit rows in test_in:"
    dual_test_intro: str = "\n{n} test input{plural} to answer, as pictures and exact digit rows in test_in:"
    task_footer: str = "\nWork in the kernel. The arrays above are already there — you do not need to retype them."
    kernel_task_footer: str = (
        "\nWork in the kernel. The arrays named above are already there, and printing one is how you see it."
    )

    # The original's no-tool-call nudge. followup_nudge appends
    # tool_call_format_guidance at render time.
    answer_nudge_guidance: str = (
        "Call `python` to inspect `train_in`, `train_out` and `test_in`; derive the rule, test it against "
        "every example pair, then assign your answers into `test_out[i][0]` inside Python. "
    )
    check_answer_nudge_guidance: str = (
        "Call `python` to inspect `train_in`, `train_out` and `test_in`; derive the rule, verify it with "
        "`check(fn)`, then assign your answers into `test_out[i][0]` inside Python. "
    )
    check_and_submit_nudge_guidance: str = (
        "Call `python` to inspect `train_in`, `train_out` and `test_in`; derive the rule and verify it "
        "with `check(fn)`. A pass automatically saves its test predictions into `test_out[i][0]`. "
    )
    mandatory_answer_nudge_guidance: str = (
        "Call `python` to inspect `train_in`, `train_out` and `test_in`; derive the rule, implement it as "
        "`fn(grid) -> grid`, and submit it by passing `check(fn)`. "
    )

    #: Appended once to a tool result when the game budget runs low
    #: (``DuckAgent.budget_warning_tokens``). The harness decides when; this
    #: is only the wording.
    budget_warning_template: str = (
        "[budget] About {remaining:,} generated tokens remain for this puzzle. Make sure your best current "
        "candidate is in `test_out` now — for every test input — and refine only if room remains."
    )
    mandatory_budget_warning_template: str = (
        "[budget] About {remaining:,} generated tokens remain for this puzzle. Implement your best current "
        "candidate as `fn(grid) -> grid` and run `check(fn)`; only a passing check submits an answer."
    )

    answer_repair_intro: str = "Your answer cannot be read yet:"
    answer_repair_footer: str = "Fix that in the kernel and carry on."
    mandatory_answer_repair_text: str = (
        "You have not submitted an answer yet. Continue working in the kernel, then call `check(fn)`. "
        "Only a passing check submits; editing `test_out` has no effect."
    )

    # Appended to the CURRENT turn's user prompt when rounds of that turn
    # were pruned to fit the context window. The marker is what makes a note
    # findable, so it is prepended by rounds_pruned_note rather than written
    # into the template: an override of the wording cannot drop it, and
    # dropping it would stack notes instead of replacing them — exactly when
    # context pressure is highest.
    rounds_pruned_marker: str = "[Context note] "
    rounds_pruned_note_template: str = (
        "{n} earlier round(s) of this turn were dropped to fit the context window. "
        "Anything you established in them is gone from this conversation — the kernel still holds what you "
        "put in it, so re-read that rather than re-deriving it, and prefer acting on what you already know "
        "over re-investigating."
    )
    compaction_wrapper_prefix: str = (
        "[context compacted — the following is your own summary of your work so far; "
        "continue from where you left off]\n\n"
    )

    # -- the fragments the flags select -----------------------------------------

    budget_line_template: str = "[budget] Tokens left on this puzzle: about {left}k of {allowance}k."
    half_brevity_line: str = "Think as long as the step genuinely needs, then act."
    cost_model_lines: str = (
        "- What costs what: your reasoning and the code you write cost generated tokens. {budget} "
        "{answer} Running code does not spend generated tokens; it is subject to the "
        "Python time limit. Tool output occupies context but is not charged to the generated-token budget.\n"
    )
    cost_model_capped: str = (
        "This puzzle has a fixed budget of about {allowance:,} generated tokens; when it is spent, the puzzle ends."
    )
    cost_model_uncapped: str = "There is no per-puzzle generated-token cap."
    cost_model_answer: str = "Saved answers in `test_out` are scored when the puzzle ends."
    cost_model_mandatory_answer: str = "Nothing is submitted unless a `check(fn)` passes."

    #: ``text_grid_overview_line`` under ``no_hand_arithmetic_line``.
    text_grid_no_hand_arithmetic_line: str = (
        "- The task prompt also spells every grid out as digit rows so you can see exact values. Use them "
        "to orient yourself, not to count, index or diff cells by hand: the kernel does that work exactly, "
        "and hand-transcribing rows is where miscounts come from.\n"
    )
    #: The same caveat where the prompt spells no grid out, added after the
    #: ``ascii_grid`` runtime line under ``no_hand_arithmetic_line``. The
    #: render is described once, by that line; this one says what to do with
    #: what it prints, and repeats the other wording's second sentence so the
    #: two renders differ only in where the rows come from.
    printed_grid_no_hand_arithmetic_line: str = (
        "- Grids you print with `ascii_grid` are for orienting yourself, not for counting, indexing or "
        "diffing cells by hand: the kernel does that work exactly, and hand-transcribing rows is where "
        "miscounts come from.\n"
    )
    #: ``runtime_names_lines`` under ``test_out_structure_explicit``.
    explicit_test_out_runtime_names_lines: str = (
        "- Preloaded names: `train_in[i]` and `train_out[i]` are the example pairs as arrays; `test_in[i]` "
        "are the inputs you must answer. `test_out` has one entry per test input, and `test_out[i]` is the "
        "{attempt_slots} for `test_in[i]`: assign grids into those slots, and never "
        "append to, replace or reshape `test_out`.\n"
        "- Sizes are rows x cols throughout, and arrays index as `[row, col]`.\n"
    )
    #: ``second_attempt_line`` under ``second_attempt_mode="audit"``.
    audit_second_attempt_line: str = (
        "- `test_out[i][1]` is a free second guess, scored independently: the question counts if either "
        "attempt is right. Use it for the audit's strongest alternative — a rule that also reproduces every "
        "training pair and changes the test output — and for nothing else: never copy attempt 0 or a "
        "trivial variation into it.\n"
    )
    assign_early_text: str = (
        "- In the same call that {validation}, assign `fn(test_in[i])` to `test_out[i][0]` for every "
        "test input, then keep refining if you want. Only what is in `test_out` when the puzzle ends is "
        "scored; a passing candidate that lives in a variable scores nothing. Overwrite the slot whenever "
        "you find something better.\n"
    )
    multi_test_text: str = (
        "- With several test inputs, answer each as soon as you can: an unset question scores nothing, an "
        "answered one scores on its own.\n"
    )
    generalization_audit_text: str = (
        "- A rule that reproduces the training pairs is consistent with them, not identified by them. Once "
        "a candidate passes, do exactly one generalization audit before you stop: list what the test input "
        "has that no training input has (a repeated colour, an extra or missing object, a new shape or "
        "size, a tie, an object on the boundary, a different orientation), and ask whether a structurally "
        "different rule also fits every training pair but gives a different test output. If the audit "
        "changes your mind, fix your rule, {recording}.\n"
    )
    generalization_audit_recording: str = "re-check it, and re-assign `test_out[i][0]`"
    generalization_audit_auto_recording: str = "re-run `check(fn)` to update `test_out[i][0]` automatically"
    #: The ``candidate_first`` work policy: its own block after "How to answer".
    work_policy_header: str = "\n\nHow to work:\n"
    candidate_first_lines: str = (
        "- Get into the kernel early. Your first move is a `python` call that extracts the facts — shapes, "
        "colour counts, the changed-cell mask between each `train_in[i]` and `train_out[i]`, the objects — "
        "and your reasoning works from that printout, not from rows read by eye.\n"
        "- After at most two inspection-only calls, write your most concrete hypothesis as `fn(grid) -> grid` "
        "and run `check(fn)`, even if you expect it to fail. A failed check names the exact mismatched "
        "cells; more unchecked prose does not. After a failure, change one explicit assumption using the "
        "reported mismatches rather than adding per-example exceptions.\n"
        "- `fn` must depend only on its grid argument and general rules: no lookups into the training pairs, "
        "no constants that only make sense for one example.\n"
    )
    candidate_first_no_check_lines: str = (
        "- Get into the kernel early. Your first move is a `python` call that extracts facts from the "
        "training pairs; reason from that printout, not from rows read by eye.\n"
        "- After at most two inspection-only calls, implement your most concrete hypothesis as "
        "`fn(grid) -> grid`, apply it to every `train_in[i]`, and compare its output with `train_out[i]`. "
        "Print the mismatches and change one explicit assumption using those results.\n"
        "- `fn` must depend only on its grid argument and general rules: no lookups into the training pairs, "
        "no constants that only make sense for one example.\n"
    )
    #: ``compact_tool_session_head`` without ``investigation_lines``: the one
    #: fact about the session, and the ask for small code.
    terse_tool_session_head: str = (
        "\n\nTool limits and session lifetime:\n"
        "- The `python` session keeps your definitions and variables for the whole puzzle. Write small, "
        "task-specific code; do not build a general-purpose framework for one puzzle.\n"
    )
    #: The causal nudges: after a reply the cap cut, and after a finished
    #: reply that left the answer unreadable.
    cut_nudge_template: str = (
        "Your last reply was cut off at the {limit:,}-token reply limit, so nothing in it ran or was "
        "recorded. Do not restart the analysis and do not carry it on by hand. Take the most concrete "
        "hypothesis you had, write it as `fn(grid) -> grid`, run `check(fn)`, and read the reported "
        "mismatches. If you do not have one yet, compute the differences between each `train_in[i]` and "
        "`train_out[i]` in the kernel and reason from the printout."
    )
    cut_nudge_no_check_template: str = (
        "Your last reply was cut off at the {limit:,}-token reply limit, so nothing in it ran or was "
        "recorded. Do not restart the analysis and do not carry it on by hand. Take your most concrete "
        "hypothesis, implement it as `fn(grid) -> grid`, and compare `fn(train_in[i])` with `train_out[i]` "
        "in the kernel for every training pair. Print and inspect the mismatches before refining it."
    )
    diagnostic_nudge_intro: str = "Your reply made no `python` call and the answer is not complete:"
    diagnostic_nudge_footer: str = "Fix that in the kernel."
    #: The nudge after a reply the cap cut off while the answer stands, in
    #: either nudge mode. Where the agent's notice flag lets the cap be named,
    #: the causal cut wording replaces it and the legacy cutoff prefix leads
    #: it.
    answered_nudge: str = (
        "Your reply made no `python` call. The answer in `test_out` stands. Continue in the kernel, or "
        "reply without calling the tool if you are done."
    )
    #: The nudge after a reply refused as malformed on every retry and dropped
    #: (``DuckAgent.malformed_nudges``), which the model never sees. Its lead,
    #: then what stands of the answer.
    malformed_nudge_lead: str = (
        "Your last reply was discarded: it went in circles and ended with no answer and no tool call."
    )
    malformed_nudge_answered: str = (
        "The answer in `test_out` stands. Take a different approach: make your next step a `python` call, "
        "or reply without calling the tool if you are done."
    )
    malformed_nudge_open: str = "The answer is not complete:"
    malformed_nudge_footer: str = "Take a different approach, and make your next step a `python` call."

    # -- summaries that leave the history whole ----------------------------------
    # (``DuckAgent.intermediate_summary_prompt_lengths``, 2026-10-04.) The
    # request is the compaction request's items without what a compaction
    # implies: nothing is dropped or shown after it, so no size, no thorough
    # mode, no candidate source (it is in the context), and no listing.
    intermediate_summary_head: str = "Take stock: write a summary of your work so far."
    intermediate_summary_continue: str = "Continue working through the puzzle from where you left off."
    # The compaction request's clause once the history holds such a summary.
    compaction_request_earlier_summaries: str = (
        ", your earlier summaries included, so this summary must contain everything you still need"
    )
    #: ``stopping_line`` under ``audit_groundwork``: the final reply states
    #: the rule, and nothing is said about it carrying nothing.
    rule_stopping_line: str = (
        "- When you are satisfied with what is in `test_out`, stop: reply without calling the tool, and in "
        "that reply describe the rule you applied in full, every part of it, in enough detail that someone "
        "could apply it by hand to a new input. Your answer is read out of the kernel after every call, so "
        "nothing is lost if you are cut off.\n"
    )
    #: Ends the task prompt's footer under ``audit_groundwork``.
    rule_task_footer_text: str = " When you are done, describe the rule you applied in full in your final reply."
    #: The summary request's open-questions item, and its wording under
    #: ``audit_groundwork``.
    compaction_open_questions_line: str = "4. Open questions / what has been ruled out.\n"
    doubts_compaction_open_questions_line: str = (
        "4. Open questions, any doubts or inconsistencies you have noticed so far, and what has been ruled out.\n"
    )

    # -- the audit turn (``DuckAgent.audit_turn``) ------------------------------
    # The puzzle restated with the answer standing, the last reply quoted,
    # then the questions. After a summary it is the pinned prompt, so it says
    # everything on its own. A question set to "" is left out, and the rest
    # are numbered as they stand.

    audit_opening: str = "You have answered this puzzle. Here it is again, with your answer."
    audit_test_intro_template: str = "\nThe test input{plural} and your answers:"
    #: An answer slot shown as a grid is captioned like a test input.
    audit_slot_same: str = "{slot}: the same as {other}"
    audit_slot_empty: str = "{slot}: empty"

    # -- an answer changed by a call ------------------------------------------
    changed_answers_intro_template: str = "You changed {slots} to this:"
    audit_reply_intro: str = "\nYour last reply, which these answers rest on:"
    audit_no_reply: str = "(no text)"
    audit_instruction: str = (
        "\nAudit this answer. Do the work in the kernel first, revising `test_out` as you go, then answer "
        "each question in your final reply."
    )
    audit_question_support: str = (
        "Support. How much of the training data shows each part of your rule? One pair can show a part "
        "several times, on several objects. A part shown only once, or one you doubted while deriving it, is "
        "a doubt. Colours, counts, sizes and directions often do play a special role, but a rule that gives "
        "one such a role needs extra care."
    )
    audit_question_novelty: str = (
        "What is new. What do the test inputs have that no training input has? What do all training inputs "
        "share that the test does not? For each, does your rule decide it, or did you choose?"
    )
    audit_question_answer: str = (
        "Your answer. Leave `test_out` holding what you now think: your likeliest answer in `test_out[i][0]`. "
        "If a doubt above has another reading that also fits every training pair and changes a test output, "
        "put the strongest such output in `test_out[i][1]`."
    )
    #: The agent reads the number back with :meth:`read_audit_confidence`.
    audit_question_confidence: str = (
        "Confidence. How confident are you, from 0 to 100, that your answer is right? Put the number in \\boxed{}."
    )

    # -- the system prompt and its blocks --------------------------------

    def system_prompt(self, modes: PromptModes) -> str:
        """Select wording on a copy under the flags, then render it. Editing a
        copy is what keeps a render from retaining wording an earlier one, in
        another mode or under another arm, selected."""
        surface = replace(self)
        auto_submit = modes.check_and_submit and modes.check_mode is CheckMode.ENABLED
        if self.no_hand_arithmetic_line:
            if modes.text_grids:
                surface.text_grid_overview_line = self.text_grid_no_hand_arithmetic_line
                if not modes.images_enabled:
                    surface.text_only_puzzle_overview_addendum += self.text_grid_no_hand_arithmetic_line
            # No digit rows in the prompt (pictures alone, or kernel-only): the
            # rows the caveat is about are the ones the model prints, so it
            # follows whichever runtime line describes that render.
            elif modes.ascii_row_numbers:
                surface.ascii_row_numbers_runtime_line += self.printed_grid_no_hand_arithmetic_line
            else:
                surface.ascii_grid_runtime_line += self.printed_grid_no_hand_arithmetic_line
        if self.test_out_structure_explicit:
            slots = (
                "pair of attempt slots `[None, None]`" if modes.n_attempts == 2 else f"{modes.n_attempts} attempt slots"
            )
            surface.runtime_names_lines = self.explicit_test_out_runtime_names_lines.format(attempt_slots=slots)
        if self.second_attempt_mode != "optional":
            surface.second_attempt_line = ""
        additions = ""
        if self.assign_early_line and not auto_submit:
            validation = (
                "`check(fn)` passes" if modes.check_mode.enabled else "your candidate reproduces every training pair"
            )
            additions += self.assign_early_text.format(validation=validation)
        if self.multi_test_line:
            additions += self.multi_test_text
        if self.generalization_audit_line:
            recording = self.generalization_audit_auto_recording if auto_submit else self.generalization_audit_recording
            additions += self.generalization_audit_text.format(recording=recording)
        if self.second_attempt_mode == "audit" and modes.n_attempts > 1:
            additions += self.audit_second_attempt_line
        # The baseline omits both checking lines in mandatory mode, where a
        # pass ends the task before an assignment or a post-check audit.
        surface.checking_line += additions
        surface.check_checking_line += additions
        if not self.investigation_lines:
            surface.compact_tool_session_head = self.terse_tool_session_head
        if self.cost_model == "explicit":
            budget = (
                self.cost_model_uncapped
                if modes.game_token_allowance is None
                else self.cost_model_capped.format(allowance=modes.game_token_allowance)
            )
            answer = (
                self.cost_model_mandatory_answer if modes.check_mode is CheckMode.MANDATORY else self.cost_model_answer
            )
            surface.compact_tool_session_head += self.cost_model_lines.format(budget=budget, answer=answer)
        if self.work_policy == "candidate_first":
            policy = self.candidate_first_lines if modes.check_mode.enabled else self.candidate_first_no_check_lines
            surface.compact_tool_session_head = self.work_policy_header + policy + surface.compact_tool_session_head
        if self.brevity == "half":
            surface.user_prompt_brevity_line = self.half_brevity_line
        if self.audit_groundwork:
            surface.stopping_line = self.rule_stopping_line
        return surface._render_system_prompt(modes)

    def _render_system_prompt(self, modes: PromptModes) -> str:
        """Assemble the prompt from the fragments as they stand, with tool and
        runtime claims that match what the agent actually enforces."""
        text = self.system_prompt_opening
        if modes.images_enabled:
            text += self.puzzle_overview_addendum if modes.image_legend else self.unlegended_image_overview_addendum
            if modes.text_grids:
                text += self.text_grid_overview_line
                if modes.ascii_row_numbers:
                    text += self.text_grid_row_numbers_line
        elif modes.kernel_only:
            text += self.kernel_only_puzzle_overview_addendum
        else:
            text += self.text_only_puzzle_overview_addendum
            if modes.ascii_row_numbers:
                text += self.text_grid_row_numbers_line
        text += self.runtime_header
        text += self._runtime_tool_line(modes)
        text += self.runtime_import_line
        mandatory = modes.check_mode is CheckMode.MANDATORY
        auto_submit = modes.check_and_submit and modes.check_mode is CheckMode.ENABLED
        text += self.mandatory_runtime_names_lines if mandatory else self.runtime_names_lines
        text += self.ascii_row_numbers_runtime_line if modes.ascii_row_numbers else self.ascii_grid_runtime_line
        if modes.check_mode.enabled:
            if mandatory:
                text += self.mandatory_check_runtime_line
            else:
                text += self.check_and_submit_runtime_line if auto_submit else self.check_runtime_line
            if modes.check_images_enabled and not mandatory:
                text += self.check_images_runtime_line
        if modes.show_enabled:
            text += self.show_runtime_line.format(max_shows=modes.max_shows)
        if modes.show_changed_answers:
            text += self.changed_answers_runtime_line
        if modes.file_tools:
            text += self.runtime_file_lines
        text += self.answer_header
        if mandatory:
            text += self.mandatory_answer_line
        else:
            text += self.check_and_submit_answer_line if auto_submit else self.answer_line
        if modes.n_attempts > 1 and not mandatory:
            line = self.second_attempt_line
            if self.likeliest_first_line:
                # A bullet of its own where the second-attempt line is left out
                # (``second_attempt_mode``).
                line = f"{line.rstrip()} {self.likeliest_first_text}\n" if line else f"- {self.likeliest_first_text}\n"
            text += line
        if not mandatory:
            text += self.check_checking_line if modes.check_mode.enabled else self.checking_line
        if modes.file_tools:
            text += self._candidate_lines(modes)
        text += self.grid_rules_line.format(max_side=modes.max_side, max_last=modes.n_colors - 1)
        text += self.mandatory_stopping_line if mandatory else self.stopping_line
        text += self.compact_tool_session_head
        text += self.compact_tool_session_tail.format(
            tool_output_tokens=modes.tool_output_tokens,
            python_timeout_s=modes.python_timeout_s,
        )
        if modes.context_summary_enabled:
            text += self.context_summary_line
        if self.brevity_line:
            text += f"\n{self.user_prompt_brevity_line}"
        return text

    #: The total tool count spelled out: `python` plus three to five file tools.
    _TOOL_COUNT_WORDS: ClassVar[dict[int, str]] = {4: "four", 5: "five", 6: "six"}

    def _runtime_tool_line(self, modes: PromptModes) -> str:
        """The sentence naming the native tools, the file suite included or not."""
        if not modes.file_tools:
            return self.runtime_tool_line
        quoted = [f"`{name}`" for name in modes.file_tool_names]
        names = ", ".join(quoted[:-1]) + " and " + quoted[-1]
        return self.runtime_tool_line_with_files.format(
            count=self._TOOL_COUNT_WORDS[1 + len(modes.file_tool_names)], names=names
        )

    def _candidate_lines(self, modes: PromptModes) -> str:
        """Where the candidate lives, with the example its answer channel
        calls for: a passing check that submits or records the answer itself
        needs no assignment, and without the oracle there is no check."""
        if not modes.check_mode.enabled:
            example = self.candidate_example_no_check
        elif modes.check_mode is CheckMode.MANDATORY or modes.check_and_submit:
            example = self.candidate_example_check_answers
        else:
            example = self.candidate_example_check
        return self.candidate_file_line + example

    def python_tool_description(self, modes: PromptModes) -> str:
        """The `python` tool's description."""
        text = (
            self.check_python_tool_description_text if modes.check_mode.enabled else self.python_tool_description_text
        )
        if modes.check_and_submit and modes.check_mode is CheckMode.ENABLED:
            text += self.check_and_submit_tool_clause
        if modes.show_enabled:
            text += self.show_tool_description_clause
        return text

    def python_tool_code_param_description(self) -> str:
        """The `code` parameter's description."""
        return self.code_param_persistent

    # -- per-turn prompts -------------------------------------------------

    def task_intro(self, modes: PromptModes, *, n_train: int) -> str:
        """The line introducing example pairs in the configured media."""
        return f"{self.task_header}\n\n" + self._train_intro(modes, n_train=n_train)

    def _train_intro(self, modes: PromptModes, *, n_train: int) -> str:
        """The example pairs' introduction, without a header."""
        template = self.train_intro
        if modes.kernel_only:
            template = self.kernel_train_intro
        elif modes.images_enabled and modes.image_layout == "separate":
            template = self.separate_dual_train_intro if modes.text_grids else self.separate_train_intro
        elif modes.text_grids:
            template = self.dual_train_intro if modes.images_enabled else self.text_train_intro
        return template.format(n=n_train, plural="" if n_train == 1 else "s")

    def pair_label(self, index: int, *, in_shape: tuple[int, int], out_shape: tuple[int, int]) -> str:
        """The caption above one example pair's image.

        Naming the arrays and their sizes here is what lets the model connect
        a picture to a variable without measuring anything: the image says
        what the transformation looks like, the caption says which names hold
        it and how the size changed — which is often the transformation's
        most important single fact.
        """
        return f"train_in[{index}] -> train_out[{index}]  ({_shape(in_shape)} -> {_shape(out_shape)})"

    def train_input_label(self, index: int, *, shape: tuple[int, int]) -> str:
        """The caption above one separately rendered training input."""
        return f"train_in[{index}]  ({_shape(shape)})"

    def train_output_label(self, index: int, *, shape: tuple[int, int]) -> str:
        """The caption above one separately rendered training output."""
        return f"train_out[{index}]  ({_shape(shape)})"

    def question_label(self, index: int, *, shape: tuple[int, int]) -> str:
        """The caption above one test input's image."""
        return f"test_in[{index}]  ({_shape(shape)})"

    def test_intro_text(self, modes: PromptModes, *, n_test: int) -> str:
        """The line introducing test questions in the configured media."""
        template = self.test_intro
        if modes.text_grids:
            template = self.dual_test_intro if modes.images_enabled else self.text_test_intro
        return template.format(n=n_test, plural="" if n_test == 1 else "s")

    def task_footer_text(self, modes: PromptModes) -> str:
        """The line closing the task prompt, in the configured media."""
        footer = self.kernel_task_footer if modes.kernel_only else self.task_footer
        if self.audit_groundwork and modes.check_mode is not CheckMode.MANDATORY:
            footer += self.rule_task_footer_text
        return footer

    # -- the audit turn -----------------------------------------------------

    def audit_intro(self, modes: PromptModes, *, n_train: int) -> str:
        """The audit prompt's opening and the line introducing the example pairs."""
        return f"{self.audit_opening}\n\n" + self._train_intro(modes, n_train=n_train)

    def audit_test_intro(self, *, n_test: int) -> str:
        """The line introducing the test inputs and the answers standing."""
        return self.audit_test_intro_template.format(plural="" if n_test == 1 else "s")

    def answer_slot_label(self, index: int, attempt: int, *, shape: tuple[int, int]) -> str:
        """The caption above one answer slot shown as a grid, named as the
        kernel names it."""
        return f"test_out[{index}][{attempt}]  ({_shape(shape)})"

    def answer_slot_note(self, index: int, attempt: int, *, same_as: int | None = None, why: str = "") -> str:
        """One answer slot not shown as a grid. A copy is said to be one
        rather than drawn twice, so an unused second slot reads as unused."""
        slot = f"test_out[{index}][{attempt}]"
        if same_as is not None:
            return self.audit_slot_same.format(slot=slot, other=f"test_out[{index}][{same_as}]")
        if why:
            return f"{slot} {why}"
        return self.audit_slot_empty.format(slot=slot)

    def changed_answers_intro(self, slots: list[str]) -> str:
        """The line heading one test input's changed answers."""
        named = slots[0] if len(slots) == 1 else ", ".join(slots[:-1]) + " and " + slots[-1]
        return self.changed_answers_intro_template.format(slots=named)

    def audit_close(self, reply: str) -> str:
        """The last ``reply`` quoted, then the instruction and the questions
        left in, numbered as they stand."""
        quoted = "\n".join(f"> {line}".rstrip() for line in reply.strip().splitlines()) or self.audit_no_reply
        questions = [
            question
            for question in (
                self.audit_question_support,
                self.audit_question_novelty,
                self.audit_question_answer,
                self.audit_question_confidence if self.audit_confidence_line else "",
            )
            if question
        ]
        numbered = "\n".join(f"{number}. {question}" for number, question in enumerate(questions, start=1))
        return f"{self.audit_reply_intro}\n\n{quoted}\n{self.audit_instruction}\n\n{numbered}"

    def read_audit_confidence(self, text: str) -> int | None:
        """The last ``\\boxed{}`` number in ``text``, when it is a confidence
        from 0 to 100. The last, because the answer comes at the end. None
        unless ``audit_confidence_line`` asked for one: a boxed number in a
        reply to a question never put is not a confidence."""
        if not self.audit_confidence_line:
            return None
        found = _BOXED_NUMBER.findall(text)
        if not found:
            return None
        value = float(found[-1])
        return round(value) if 0 <= value <= 100 else None

    def followup_nudge(
        self, modes: PromptModes, completion_limit: int | None = None, problems: list[str] | None = None
    ) -> str:
        """The nudge after a reply with no tool call, under ``nudge_mode``.

        Legacy is the baseline's text with the cap named under the agent's
        notice flag. Causal says what happened: a reply the cap cut gets the
        cut wording with the limit when the notice flag permits it;
        otherwise it gets the answer's own problems, or ``answered_nudge``
        when there are none. The format guidance rides either under its own
        flag.
        """
        format_suffix = " " + self.tool_call_format_guidance if self.nudge_format_guidance else ""
        if self.nudge_mode == "causal":
            if modes.completion_cutoff_notice and completion_limit is not None:
                template = self.cut_nudge_template if modes.check_mode.enabled else self.cut_nudge_no_check_template
                return template.format(limit=completion_limit) + format_suffix
            if problems is not None and not problems:
                return self.answered_nudge + format_suffix
            if modes.check_mode is CheckMode.MANDATORY:
                return self.mandatory_answer_repair() + format_suffix
            bullets = "\n".join(f"  - {problem}" for problem in (problems or ["the answer is not complete"]))
            return f"{self.diagnostic_nudge_intro}\n{bullets}\n\n{self.diagnostic_nudge_footer}" + format_suffix
        surface = self if self.nudge_format_guidance else replace(self, tool_call_format_guidance="")
        return surface._legacy_followup_nudge(modes, completion_limit, problems)

    def _legacy_followup_nudge(
        self, modes: PromptModes, completion_limit: int | None = None, problems: list[str] | None = None
    ) -> str:
        """The legacy nudge, identifying a completion-limit cutoff when the
        agent's notice flag lets it. ``problems`` only says whether the answer
        stands: this nudge is one text whatever is wrong with the answer."""
        if not modes.completion_cutoff_notice:
            completion_limit = None
        if problems is not None and not problems:
            prefix = "" if completion_limit is None else self._completion_cutoff_prefix(completion_limit)
            suffix = " " + self.tool_call_format_guidance if self.tool_call_format_guidance else ""
            return prefix + self.answered_nudge + suffix
        if modes.check_mode is CheckMode.MANDATORY:
            nudge = self.mandatory_answer_nudge_guidance
        elif modes.check_and_submit and modes.check_mode is CheckMode.ENABLED:
            nudge = self.check_and_submit_nudge_guidance
        else:
            nudge = self.check_answer_nudge_guidance if modes.check_mode.enabled else self.answer_nudge_guidance
        guidance = nudge + self.tool_call_format_guidance
        if completion_limit is None:
            return "You have not answered yet. Investigate first. " + guidance
        # The cutoff-specific wording replaces ``Investigate first`` rather than
        # repeating two competing instructions at the start of the same nudge.
        return self._completion_cutoff_prefix(completion_limit) + "You have not answered yet. " + guidance

    def budget_warning(self, remaining: int, modes: PromptModes) -> str:
        """The one-off note that ``remaining`` generated tokens are left."""
        template = (
            self.mandatory_budget_warning_template
            if modes.check_mode is CheckMode.MANDATORY
            else self.budget_warning_template
        )
        return template.format(remaining=remaining)

    def budget_status(self, used: int, allowance: int | None, modes: PromptModes) -> str:
        """Show the remaining game allowance when enabled and bounded."""
        if not self.budget_line or allowance is None:
            return ""
        left = max(0, allowance - used)
        return self.budget_line_template.format(
            left="under 1" if 0 < left < 1000 else left // 1000, allowance=allowance // 1000
        )

    def _completion_cutoff_prefix(self, completion_limit: int) -> str:
        """Return the completion-cutoff explanation."""
        return (
            f"You hit the {completion_limit:,}-token limit, so your previous reasoning was cut off. "
            "Continue from where it stopped without restarting or altering it. "
        )

    def answer_repair(self, problems: list[str]) -> str:
        """What is wrong with the answer as it currently stands.

        One bullet per problem, in the model's own index vocabulary, so the
        fix is an assignment it can make without working out which question
        is meant.
        """
        bullets = "\n".join(f"  - {problem}" for problem in problems)
        return f"{self.answer_repair_intro}\n{bullets}\n\n{self.answer_repair_footer}"

    def mandatory_answer_repair(self) -> str:
        """The re-entry after a tool call that did not pass ``check(fn)``."""
        return self.mandatory_answer_repair_text

    def malformed_nudge(self, modes: PromptModes, problems: list[str] | None = None) -> str:
        """The nudge after a dropped malformed reply: the lead, then the answer
        as it stands, worded as the causal nudge words it. The format guidance
        rides under its own flag."""
        format_suffix = " " + self.tool_call_format_guidance if self.nudge_format_guidance else ""
        if problems is not None and not problems:
            return f"{self.malformed_nudge_lead} {self.malformed_nudge_answered}" + format_suffix
        if modes.check_mode is CheckMode.MANDATORY:
            return f"{self.malformed_nudge_lead} {self.mandatory_answer_repair()}" + format_suffix
        bullets = "\n".join(f"  - {problem}" for problem in (problems or ["the answer is not complete"]))
        return (
            f"{self.malformed_nudge_lead} {self.malformed_nudge_open}\n{bullets}\n\n{self.malformed_nudge_footer}"
            + format_suffix
        )

    def rounds_pruned_note(self, n: int) -> str:
        """The note that ``n`` rounds of the current turn were dropped."""
        return self.rounds_pruned_marker + self.rounds_pruned_note_template.format(n=n)

    def is_rounds_pruned_note(self, text: str) -> bool:
        """Whether ``text`` is a note this surface rendered."""
        return text.startswith(self.rounds_pruned_marker)

    compaction_hypotheses_check_line: str = (
        "1. Hypotheses tested and the outcome of each (include check failure/success tallies).\n"
    )
    compaction_hypotheses_line: str = "1. Hypotheses tested and the outcome of each.\n"
    compaction_candidate_check_line: str = "2. Your current best candidate fn — include its SOURCE CODE verbatim.\n"
    compaction_candidate_line: str = "2. Your current best transformation — include its SOURCE CODE verbatim.\n"
    # In place of the two above where the rebuilt conversation shows
    # ``candidate.py`` in full: the summary need not restate it, and what the
    # file cannot say about itself is where it is wrong or unfinished.
    compaction_candidate_shown_line: str = (
        "2. Your current best candidate fn: `candidate.py` will be shown in full after compaction, so do not "
        "restate it; say instead what it still gets wrong or leaves open.\n"
    )
    # The request's opening, composed by ``compaction_prompt``: ``{kind}`` is
    # what the summary is asked to be, and ``{kept}`` what survives besides
    # it, named where the rebuilt conversation says so.
    compaction_request_head: str = (
        "Your context window is nearly full. Write a {kind} summary of your work so far to carry "
        "forward; everything except this summary and the task presentation will be discarded{earlier}{kept}."
    )
    compaction_request_kind: str = "compact"
    compaction_request_kind_thorough: str = "thorough"
    compaction_request_kept_session: str = "; your `python` session (its variables and functions){and_files} will not"
    # Under ``thorough_summary``, after the head, in place of the size: the
    # target read as a ceiling, and the switch exists to lift the summary off
    # it. Not quite true where files are kept, and left so.
    compaction_request_thorough_line: str = (
        " Whatever you don't write in the summary is lost, so be thorough. Discuss everything you learned and "
        "the evidence behind it."
    )
    # Under ``thorough_summary``, once a previous summary is in the history.
    compaction_request_previous_summary_line: str = (
        " Make sure to also take over anything that is still relevant from the previous summary."
    )
    compaction_request_size: str = " At most ~{summary_tokens:,d} tokens."
    compaction_request_observations_line: str = (
        "3. Key observations about the grids (exact coordinates, colors, shapes) that informed your thinking.\n"
    )
    compaction_request_next_step_line: str = "5. Your immediate next step.\n"
    compaction_request_listing_line: str = (
        "A listing of your `python` session's objects{and_files} will be provided after compaction, so you "
        "need not repeat it, but do explain the most important ones.\n"
    )
    compaction_request_tail: str = (
        "Reply in plain text only — do NOT call any tools (any earlier instruction to make a "
        "tool call is superseded for this reply). Do not add anything else."
    )
    compaction_closing_line: str = (
        "[context compacted — your `python` session and files are intact; continue from here]"
    )
    compaction_closing_line_without_files: str = (
        "[context compacted — your `python` session is intact; continue from here]"
    )
    # The listing under the closing line. Each section shows its first
    # ``compaction_listing_lines`` entries one to a line, then the next
    # ``compaction_listing_names`` by name alone on one line, then a count of
    # the rest.
    compaction_listing_files_header: str = "Your files:"
    compaction_listing_objects_header: str = "Your `python` session, objects you defined:"
    compaction_listing_empty: str = "(none)"
    compaction_listing_more_names: str = "and more: {names}"
    compaction_listing_more: str = "and {n} more not named here."
    compaction_listing_lines: int = 20
    compaction_listing_names: int = 100
    compaction_candidate_header: str = "Your `candidate.py`:"

    def compaction_prompt(
        self,
        modes: PromptModes,
        summary_tokens: int,
        *,
        previous_summary: bool = False,
        candidate_shown: bool = False,
        earlier_summaries: bool = False,
    ) -> str:
        """Ask for the state that must survive wholesale history replacement.

        The first two items name what the model must carry forward, so they
        follow the arm: ``check`` and the ``fn`` it takes exist only where the
        oracle is bound, and asking an arm without it to tally check results
        is asking for something it has never had. The second asks for the
        candidate's source unless ``candidate_shown``.

        What the request claims survives follows the agent: the ``python``
        session where the rebuilt conversation says so (the file tools, or the
        listing), files only where the model has the file tools, and the
        listing only where the rebuilt conversation carries one. Under
        ``thorough_summary`` it asks for a thorough summary rather than a
        compact one, states no size, and asks for what still holds of the
        ``previous_summary`` where the history carries one. Under
        ``audit_groundwork`` the open questions take in doubts and
        inconsistencies noticed so far. With ``earlier_summaries`` it says the
        intermediate summaries in the history go too, so this one must hold
        everything.
        """
        and_files = " and your files" if modes.file_tools else ""
        kept = (
            self.compaction_request_kept_session.format(and_files=and_files)
            if modes.file_tools or modes.add_files_and_repl_to_summary
            else ""
        )
        kind = self.compaction_request_kind_thorough if self.thorough_summary else self.compaction_request_kind
        earlier = self.compaction_request_earlier_summaries if earlier_summaries else ""
        text = self.compaction_request_head.format(kind=kind, earlier=earlier, kept=kept)
        if self.thorough_summary:
            text += self.compaction_request_thorough_line
            if previous_summary:
                text += self.compaction_request_previous_summary_line
        else:
            text += self.compaction_request_size.format(summary_tokens=summary_tokens)
        text += " In this order:\n"
        text += self.compaction_hypotheses_check_line if modes.check_mode.enabled else self.compaction_hypotheses_line
        if candidate_shown:
            text += self.compaction_candidate_shown_line
        else:
            text += self.compaction_candidate_check_line if modes.check_mode.enabled else self.compaction_candidate_line
        text += self.compaction_request_observations_line
        text += (
            self.doubts_compaction_open_questions_line if self.audit_groundwork else self.compaction_open_questions_line
        )
        text += self.compaction_request_next_step_line
        if modes.add_files_and_repl_to_summary:
            text += self.compaction_request_listing_line.format(and_files=and_files)
        return text + self.compaction_request_tail

    def intermediate_summary_prompt(self, modes: PromptModes) -> str:
        """Ask for a summary that leaves the history whole: the compaction
        request's items, renumbered, less the candidate's (its source is in the
        context), with no size and no thorough mode."""
        items = [
            self.compaction_hypotheses_check_line if modes.check_mode.enabled else self.compaction_hypotheses_line,
            self.compaction_request_observations_line,
            self.doubts_compaction_open_questions_line
            if self.audit_groundwork
            else self.compaction_open_questions_line,
            self.compaction_request_next_step_line,
        ]
        numbered = "".join(
            f"{number}. {re.sub(r'^[0-9]+[.] ', '', item)}" for number, item in enumerate(items, start=1)
        )
        return f"{self.intermediate_summary_head} In this order:\n{numbered}{self.compaction_request_tail}"

    def compaction_wrapper(self, summary: str) -> str:
        """Carry one accepted summary into the durable conversation."""
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

    def compaction_candidate(self, source: str) -> str:
        """``candidate.py`` under its header, in a code fence longer than any
        run of backticks inside it, so the file cannot close its own fence."""
        longest = max((len(run) for run in re.findall(r"`+", source)), default=0)
        fence = "`" * max(3, longest + 1)
        return f"{self.compaction_candidate_header}\n{fence}python\n{source.rstrip()}\n{fence}"

    def compaction_closing(self, modes: PromptModes, listing: str, *, session_intact: bool = True) -> str:
        """Close the rebuilt conversation, over the listing."""
        line = self.compaction_closing_line if modes.file_tools else self.compaction_closing_line_without_files
        if not session_intact:
            line = "[context compacted — your `python` session was reset; continue from the summary]"
        return line + (f"\n\n{listing}" if listing else "")


def _format_bytes(size: int) -> str:
    """A file size as the listing shows it: bytes, then KB and MB to one decimal."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


#: The name the flags were reached through before they lived on ``DuckPrompts``,
#: kept for the callers and the pickled runs (sweeps 5-7's solvers) that use it.
PromptsConfigurable = DuckPrompts


#: The default nudge, for tests that pin the exact no-cutoff wording.
FOLLOWUP_NUDGE = DuckPrompts().followup_nudge(PromptModes())

# Defaults for a DuckPythonTool constructed on its own; DuckAgent overwrites
# both from its own configuration before the tool is ever used.
PYTHON_TOOL_DESCRIPTION = DuckPrompts().python_tool_description(PromptModes())
PYTHON_TOOL_CODE_PARAM_DESCRIPTION = DuckPrompts().python_tool_code_param_description()


__all__ = [
    "EMPTY_OUTPUT_NOTE",
    "FOLLOWUP_NUDGE",
    "PYTHON_TOOL_CODE_PARAM_DESCRIPTION",
    "PYTHON_TOOL_DESCRIPTION",
    "DuckPrompts",
    "PromptModes",
    "Prompts",
    "PromptsConfigurable",
    "truncation_note",
]
