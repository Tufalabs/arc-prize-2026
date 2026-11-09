"""DuckAgent: the duck, playing ARC-AGI-2.

How it plays. One task is one game and one call into this agent. The model
gets the puzzle as pictures and the same puzzle as numpy arrays in a
stateful kernel, then works there for as long as it wants. Ordinarily it
answers by assigning into ``test_out``; mandatory-check mode instead accepts
only a function that passes ``check(fn)``. There is no submit tool or separate
confirmation step: either answer channel is harvested from the Python call
and submitted immediately.

Reading order — the file follows one game through the agent:

1. configuration and per-game state, then the deployment
2. ``_do_move`` — one game, the whole spine on one screen
3. the analyzer loop — the LLM turns a game runs until the answer is complete
4. the answer — harvesting the configured answer channel and submitting it
5. prompts — the system prompt, the task prompt, and the nudge
6. context budget — eviction, exact token counts, retention
7. resources — the kernel's lifetime

What it leans on the harness for. Between them these are the reason there is
no plumbing in this file — no subprocess handling, no capture, no span
timing, no TAAF at all:

- ``ctx.submit(submission)`` answers, as often as it likes, and returns
  nothing. Scoring happens once the game ends, so submitting early is free
  and a run cut off mid-game still scores what stood.
- ``ctx.backend`` is the shared LLM client; ``resolve_sampling(ctx,
  profile)`` resolves a NAMED sampling profile and draws the per-call seed
  from the per-game RNG, so nothing here picks a seed.
- ``ctx.diag`` records spans; ``ctx.status`` publishes what this game is
  doing to the run's status line.
- platypus hosts the game in its own process, and TAAF scores it.

Ported from the ARC-AGI-3 duck. Everything downstream of a live game —
moves, levels, frames, actions, the journal, kernel persistence modes,
checkpointing — has no subject here and is gone; everything about running
a long conversation against a context window is kept as upstream has it.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import math
import os
import random
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Literal, NamedTuple, Self, cast

import numpy as np
import ullm.backend
import ullm.continuation
import ullm.conv
import ullm.manager
import ullm.tool

import platypus.diagnostics
import platypus.llm
import platypus.solver
import taaf.game
import taaf.grid
from duck import images, kernel_code, palette, presets
from duck import prompts as prompts_module
from duck.config import CheckMode
from duck.context_observer import ContextObserver
from duck.file_suite import file_tool_names as build_file_tool_names
from duck.file_suite import file_tools as build_file_tools
from duck.prompts import DuckPrompts, Prompts
from duck.python_tool import DuckPythonTool, SessionInspectionError
from duck.state import AnswerReading, reading_from_payload
from duck.workspace import Workspace
from platypus.contract import HANDICAP_HINT, GameOutcome, GiveUp, GiveUpError, RunConfig
from platypus.llm import LLMAgent, LLMMoveContext

#: What ``DuckAgent.build(kimi_student=True)`` sets: the model-visible
#: settings the Kimi K3.1 Pro teacher traces were played with that the
#: column has off, so a model trained on those traces plays the duck they
#: were recorded on. The summary written after the model's own reasoning (an
#: API takes no think-close injection). The teacher runs build from it too
#: (``tools/teachers/duck_smoke.py``), so the two cannot drift. What else the
#: teacher sets is its API's (``tools/teachers/README.md``) and stays the
#: column's here, the reasoning level included: the traces are rendered at
#: the column's in training (2026-10-29). An override passed to
#: ``build`` beside it still wins.
KIMI_STUDENT: dict[str, Any] = {
    "summary_thinking": True,
}
# The two fields a column composes rather than sets: replacing either through
# ``build`` would silently discard what the layers put inside it.
_COMPOSED_BY_THE_COLUMN = ("llm_config", "run_config")
# The model calls one game may make under ``summarize``, summaries included: a
# summary that never shrinks the context would otherwise loop. An ARC2 game is
# one move, so it bounds the whole game: a sweep 8 game still playing had made
# 512 calls 700k tokens in, and ten times that leaves a game at a million
# tokens room (2026-10-03).
_SUMMARY_REQUEST_BACKSTOP = 5_120
# What the summarize strategy injects after the ordinary generation prompt
# (``<think>\n``) so the summary is written without thinking, as the template
# would render it with ``enable_thinking=False`` (``<think>\n\n</think>\n\n``).
# The template's own switch cannot be used: it also drops the reasoning-effort
# line that opens the system block, and with it the whole cached prefix.
_SUMMARY_THINK_CLOSE = "\n</think>\n\n"
# The largest ``candidate.py`` a rebuilt conversation shows in full. The rebuilt
# history has to fit the window, or the summary fails the game; a file past this
# is listed by name, and the summary carries the candidate as it otherwise does.
CANDIDATE_SHOWN_MAX_CHARS = 16_000


class _SummaryCalledATool(Exception):
    """A summary reply carried a tool call: stop the turn before it runs."""


class _CallAllowanceExhausted(Exception):
    """Stop before an ordinary call when compaction spent its allowance."""

    def __init__(self, calls_made: int = 0) -> None:
        super().__init__("LLM-call allowance exhausted")
        self.calls_made = calls_made


class _IntermediateSummaryDue(Exception):
    """Stop before an ordinary call: the conversation passed one of
    ``intermediate_summary_prompt_lengths``, and a summary turn goes first."""


class _SummaryUnusable(RuntimeError):
    """The model returned no usable summary on either attempt."""


class _WrittenSummary(NamedTuple):
    """One accepted summary and how it was arrived at."""

    summary: str
    response: ullm.conv.LLMResponse
    attempts: int
    reasoning_retry: bool
    rerendered: bool
    cap: int


@dataclass
class PuzzleLayout:
    """How a game's puzzle is shown to the model: a dihedral orientation
    (``_dihedral``) and the original indices of the training pairs and test
    inputs, in the order shown."""

    transform: int
    train_order: tuple[int, ...]
    test_order: tuple[int, ...]


def _grid_image(
    grid: taaf.grid.Grid,
    *,
    scale: int,
    gridlines: bool,
    alt_text: str,
) -> ullm.conv.ImageContent:
    """One uncomposed task grid in the configured visual style."""
    return ullm.conv.ImageContent(
        data=images.render(grid, scale=scale, gridlines=gridlines),
        alt_text=alt_text,
    )


def _installed_vllm() -> ModuleType:
    """The installed vLLM, imported on use rather than at module scope.

    This module is imported in every game child process (the agent is
    unpickled there), and vllm is a per-process import cost paid by every
    game, for a package only the host's engine uses — the child reaches
    the engine through a proxy client. ``platypus.subproc`` puts the
    figure on that and asserts the child never gets it, so a top-level
    import here fails every subprocess run rather than quietly costing a
    run its memory.
    """
    import vllm

    return vllm


class _SpanBlocks:
    """A span's block store, shaped like ``platypus.diagnostics.BlockStore``.

    ``serialize_conversation_blocks`` wants a store object; the agent only
    holds a ``Span``, whose ``store_block`` is the same operation routed to
    the same store. On the null stream keys come back empty and nothing is
    written, which is exactly the inertness the replay references need."""

    def __init__(self, span: platypus.diagnostics.Span) -> None:
        """Wrap ``span`` as the store serialization writes through."""
        self._span = span

    def store(self, payload: Any, codec: str) -> str:
        """Store one payload; the block key (empty on the null stream)."""
        return self._span.store_block(payload, codec)


@dataclass
class _FirstReplyCappedResponseManager(ullm.manager.ResponseManager):
    """ullm's response manager with one extra rule: this manager's FIRST call
    may be capped below the profile's ``max_tokens``.

    The turn manager rewrites ``sampling_params.max_tokens`` before every
    call from a value it captured at turn start, so a cap set from a
    ``before_round`` hook is overwritten before the backend sees it. The call
    itself is the one place the ceiling can be lowered and put back. The
    agent hands the cap to the manager of the block that opens the game and
    ``None`` to every later block, so exactly the game's first reply is
    capped.
    """

    first_reply_max_tokens: int | None = None
    _calls: int = field(default=0, init=False, repr=False)

    async def _run(self) -> None:
        """One response; the first under the cap, if there is one."""
        cap = self.first_reply_max_tokens
        self._calls += 1
        params = cast(ullm.backend.GenericSamplingParameters, self.sampling_params)
        if cap is None or self._calls > 1:
            await super()._run()
            return
        original = params.max_tokens
        params.max_tokens = cap if original is None else min(original, cap)
        try:
            await super()._run()
        finally:
            params.max_tokens = original


@dataclass(kw_only=True)
class DuckAgent(LLMAgent):
    """The duck. Every model-facing choice is a dataclass field; the
    defaults are upstream's shipped configuration, set by measurement over
    ARC-AGI-3 games, and nothing here has re-measured them. Sampling knobs
    (temperature/top_p/max_tokens) live on the ``llm_config`` sampling
    profile named by ``profile``.

    Everything after the knobs is per-game state, and there is nothing else
    on the instance — no client, no diagnostics handle; those arrive per
    call on the context. That is what makes the template cheap to copy per
    (game, pass) and a pickle of a playing copy a complete description of
    where the game had got to."""

    # --- configuration ---
    # The window, compaction target and reply reserve are one coordinate
    # system; upstream's ``design.md`` ("The context budget") has the
    # arithmetic and its ``evidence.md`` the readings behind it.
    # A decode token is charged for every token of context it attends over,
    # and a narrower window also holds more games resident in the pool and
    # the tier, so the window is bought twice (`evidence.md`, "The narrowed
    # context window").
    #
    # 73,728 is the 48,640 prompt budget every sweep read arms at, plus the
    # 24,576 reply cap (``reply_cap_tokens``) and the 512 slack: when sweep 7
    # moved the cap up from 16,384 the window moved with it, so the budget
    # stayed where it was measured.
    context_window_tokens: int = 73728
    # The window the engine was built with, read off its arguments by the
    # build before this agent settles its own onto them: the checkpoint's
    # native context on our columns. What ``summarize`` runs under, since its
    # summary request goes past the agent's window by an amount the agent
    # cannot know (``_derived_engine_window``). ``None`` on a hand-built duck,
    # which states the engine window itself if it summarizes.
    engine_ceiling_tokens: int | None = None
    # Hysteretic compaction target. An over-budget prompt is compacted all
    # the way down to this many tokens, so the context runs as a sawtooth
    # between the budget and this target, and pays for a trim once per many
    # rounds. Must be positive and below the budget (raises otherwise).
    # Chosen as a RATIO of the prompt budget rather than as an absolute:
    # that is the coordinate the sweep draws, and it is what holds the
    # re-prefill a trim costs roughly flat as the window moves. 22000 of a
    # 48640 budget is 0.452, against the 0.463 it replaced.
    compact_to_tokens: int = 22000
    # Whole-round pruning is the class default: an agent built without a
    # reply cap has no real reserve, and a summary needs one. Every ARC2
    # column ships ``summarize`` (``presets.configure``, beside the cap):
    # sweep 6 read the live summary at +11.4 over pruning (evidence.md,
    # "Duck sweeps 5, 6 and 7"). ``summarize`` has the model write a summary
    # that replaces the history, on the live request itself:
    # the ordinary system prompt, tools and reasoning effort, thinking closed
    # by an injected ``</think>`` rather than by the template, and the whole
    # history -- so the request extends the served ids and its prefill is
    # cached. It never drops history to fit, so the engine window has to hold
    # it (see ``prepare_for_backend``), and the summary may run to twice its
    # target.
    compaction_strategy: Literal["prune", "summarize"] = "prune"
    compaction_ratio: int = 16
    # Whether the summary is written after the model's own reasoning. Off, the
    # summary request injects the end of thinking and caps the summary at twice
    # its target; on, it opens on the ordinary generation prompt, so the model
    # reasons first, and the cap is six times the target, since the reasoning
    # counts against it. Only the final text becomes the summary. Off, a reply
    # that comes back entirely empty, with nothing in the reasoning either, is
    # retried as if it were on (2026-10-02).
    summary_thinking: bool = False
    # What the rebuilt history keeps after the summary, beside the task and the
    # summary itself: ``full`` -- the latest response and the tool results that
    # follow it; ``without_reasoning`` -- the same, with that response's
    # reasoning left out; ``none`` -- nothing. A tail that does not fit the
    # window goes regardless. ``without_reasoning`` (2026-09-21):
    # sweep 7 read it +0.7 over ``full`` (p=0.37), the three within a point.
    summary_tail: Literal["full", "without_reasoning", "none"] = "without_reasoning"
    # Whether the conversation rebuilt around a summary carries a closing
    # message listing the kernel's working directory and the objects the
    # model defined in its ``python`` session -- with ``candidate.py`` shown
    # in full under it, where the file tools are on and the file exists --
    # and the summary request says it will. Inert under ``prune``. On: about
    # +0.5 in sweeps 8 and 9 alike, short of significance in each, and the
    # same throughput-adjusted (``evidence.md``, "Duck sweeps 8 and 9").
    add_files_and_repl_to_summary: bool = True
    # Optional model-specific token IDs banned only while writing a summary.
    # Empty keeps the gate provider-neutral; a deployment that knows its
    # tokenizer can categorically suppress tool-call markup here.
    compaction_forbidden_token_ids: tuple[int, ...] = ()
    max_llm_calls_per_move: int = 0  # 0 = unbounded; bounds one analyzer entry, not the game
    python_timeout_s: int = 90  # the kernel's whole-call wall clock
    # Cap on REPL output carried back into the conversation (x4 = chars).
    # Tighter than a full grid dump: the sweep favored terse tool results,
    # and the model is told when output was cut.
    tool_output_tokens: int = 2425
    # Training-pair oracle. Mandatory makes a passing checked function the
    # only answer channel; enabled preserves direct ``test_out`` answers.
    # Enabled rather than mandatory: forcing the channel costs score, while
    # merely offering it does not, so the model decides when to verify.
    check_mode: CheckMode = CheckMode.ENABLED
    # Task/show/check image modes. ``task+show`` compiles ``show()`` into the
    # kernel as well, so the model can put a grid of its own choosing in front
    # of itself; ``task`` leaves that function out entirely. ``off`` is far
    # behind either, and ``task+show+check`` needs the oracle.
    images: Literal["off", "task", "task+show", "task+show+check"] = "task+show"
    # Exact digit rows in the task prompt, independent of the image mode:
    # with both off, the prompt names the arrays and their shapes and the
    # kernel alone holds the cells. Off: with the throughput priced at its
    # Kaggle value, sweep 7 read no rows with the caveat +1.3 over the rows
    # with it (p=0.10), the rows' prompt tokens costing more than they buy
    # (``packages/duck/evidence.md``, "The calls at the Kaggle weight").
    text_grids: bool = False
    # Reflect grids left-right and shuffle training/test examples. Multi-take
    # wrappers give opted-in takes after the first distinct layouts, drawn
    # from all eight dihedral orientations from the third take on.
    # Submissions return to the original coordinates and test order.
    use_dihedral_and_order_permutation: bool = False
    # Pair composites are compact; separate images preserve each grid's own
    # text-image adjacency, which is worth its extra attachments alongside
    # the digit rows.
    image_layout: Literal["paired", "separate"] = "separate"
    # Whether image-enabled task prompts open with the shared palette image.
    # Off: sweep 7 read the legend at -1.3 (p=0.007), and the overview names
    # the standard ARC colours without it.
    image_legend: bool = False
    # Explicit show() grids allowed per model tool call.
    max_shows: int = 6
    # Whether a new continuation user message identifies a max-token cutoff.
    # Generated history is byte-identical in both modes.
    completion_cutoff_notice: bool = False
    # Whether a passing ``check`` prints its report and a reminder to record
    # the answer. Baked into the kernel prelude the way the row layout is; a
    # bare ``check(fn)`` statement is otherwise silent in exec mode. On
    # (2026-09-21): sweep 7 read nothing either way, and a pass that
    # says so is the design that makes sense.
    check_prints_on_pass: bool = True
    # In enabled check mode, save a passing candidate's validated predictions
    # into the first answer slots without ending the game.
    check_and_submit: bool = False
    # Whether the model gets the file tools (view/search/replace/create/undo)
    # over the kernel's working directory beside the python tool, and keeps
    # its candidate in ``candidate.py`` there. The directory is the kernel's
    # own, so a file the model writes is a module its snippets import: the
    # candidate survives a summary as a file rather than as text the summary
    # re-types, and a revision is an edit rather than a redefinition. Full:
    # sweep 9 read +1.3 (p<0.001) for 8.7k more effective tokens a game, a tie
    # throughput-adjusted, which the score breaks (``evidence.md``).
    file_tools: Literal["none", "full"] = "full"
    # Whether the file suite includes `search_files` and `undo_edit`. Off: the
    # model all but never reaches for either (``evidence.md``, the smokes),
    # while their schemas are spent on every request.
    expose_tool_search_files: bool = False
    expose_tool_undo_edit: bool = False
    # Once per game, where the game would stop with an answer standing, hand
    # the model the puzzle restated with that answer and its last reply, and
    # questions that audit it (``DuckPrompts``, "the audit turn"); the game
    # ends at the next stop. After a summary that restating is the pinned
    # prompt, rebuilt with the answer as it then stands. Nothing before the
    # turn differs, so a game read up to the ``audit`` event is the game
    # without it. On: +3.0 Kaggle points on one take and +2.0 on two, on
    # arc2-eval, and still ahead under the twelve-hour clock (``evidence.md``,
    # "Duck sweeps 8 and 9"). Mandatory check mode has no stop with an answer
    # standing to fire at, and is refused.
    audit_turn: bool = True
    # Tokens this game asks the slot ranking to put behind its spend once its
    # audit has reached its first compaction (2026-10-05): the rest of
    # the audit then waits for capacity no other game wants. Offered with the
    # slot at that compaction, where the prefix is gone anyway, and on every
    # offer after; ranked by the host's ``least_tokens`` policy
    # (``platypus.contract.HANDICAP_HINT``). 0 ranks the audit as the play
    # before it; ``MultiTakeDuck`` sets the second take's to 10M, behind every
    # game that has not deferred. Inert without the audit turn.
    audit_handicap_tokens: int = 0
    # Whether a ``python`` call that changes ``test_out`` is followed by each
    # changed answer as an image, beside its test input, in a user message of
    # its own after any ``show`` grids, and the system prompt says so
    # (2026-10-02). A slot counts as changed where it holds a usable grid other
    # than the one shown last. Needs the task images; refused under mandatory
    # check mode, where an assignment answers nothing. On, a judgment call:
    # throughput-adjusted a tie in both sweeps, and the model uses the images
    # as its last check (``evidence.md``).
    show_changed_answers: bool = True
    # Summaries that leave the history whole (2026-10-04): each time the
    # full conversation passes one of these lengths, in tokens as the request is
    # measured (no reply cap or margin added), a turn of its own asks for a
    # summary, which stays in the history, and the next message says to go on.
    # Nothing is compacted. A compaction re-arms every length above what it
    # leaves; where both fall due at once, the compaction runs alone, and
    # lengths passed together get one summary. Empty, none is asked. The
    # summarize arm only.
    intermediate_summary_prompt_lengths: tuple[int, ...] = ()
    # Remaining game budget (generated tokens) at which ONE warning is
    # appended to the next tool result, asking for the best current answer
    # to be in ``test_out``. ``None`` never warns. Only the first crossing
    # warns, so a late message cannot become a drumbeat.
    budget_warning_tokens: int | None = None
    # Ceiling on the game's FIRST reply alone, below the profile's
    # ``max_tokens``; every later reply keeps the profile's cap. ``None``
    # leaves the first reply as capped as the rest.
    first_response_max_tokens: int | None = None
    # Eviction chunk, and the round-pruning overshoot. 1, not upstream's 2:
    # the overshoot drops the NEWEST surviving rounds, i.e. the model's
    # freshest work — at 2 it was seen emptying a turn whose last round had
    # already fit under the target. One round of headroom still prevents
    # the re-prune-every-round thrash. Prune's alone; sweeps 5 and 6 resolved
    # nothing for it (evidence.md).
    evict_turns: int = 1
    # Keep the task prompt — the legend and the pictures, always element 0 —
    # through eviction. Eviction removes whole user-led blocks from the
    # oldest end, and every re-entry (a nudge, a repair note) opens a new
    # block, so a game that has re-entered and then compacts otherwise loses
    # the puzzle's pictures first and plays on from the kernel's arrays
    # alone. With this on, the first block's TURN is evicted like any other
    # but its prompt stays as the conversation's head; later evictions take
    # the block after it. Off, and prune's alone: the live summary pins the
    # task prompt by construction, and sweeps 5 and 6 resolved nothing for
    # it (evidence.md).
    keep_task_prompt: bool = False
    # Nearest-neighbour upscale on every grid image. Swept over 4, 8 and 16
    # with no separation between them, so this is the cheapest of three
    # equals rather than a measured optimum.
    image_scale: int = 4
    # One-pixel cell lattice, which gives every cell a countable border;
    # false uses the compact unlatticed renderer. Only bites from scale 4 up,
    # where a cell is large enough to carry one. Off: sweep 7 read the
    # lattice at -1.6 on the throughput-corrected score (p=0.001), 1.1 tasks
    # an hour of it.
    image_gridlines: bool = False
    # Whether every ASCII grid this agent renders — the task prompt's digit
    # rows, failed-check feedback, and the kernel's own ``ascii_grid`` —
    # prefixes each row with its index and a tab, the shape every other
    # numbered text a model reads has. A sweep axis, not a model choice: it
    # is never offered as a parameter, so one layout holds everywhere the
    # model is shown a grid. A render the model writes itself is its own.
    # On, for one indexing convention with the numpy arrays in the kernel:
    # rows count from 0 in both.
    ascii_row_numbers: bool = True
    # How the kernel reprs a NumPy scalar inside a container, which is what a
    # printed row or coordinate list is made of. ``np2`` is NumPy 2's own
    # ``np.int64(37)``; ``legacy`` restores the plain ``37`` of <=1.25. The
    # model reads this output, and the verbose form triples a coordinate
    # list's cost and reads as noise: measured upstream over a baseline run,
    # a third of the text in every result holding coordinates was the
    # constructor wrapper. Legacy is the shipped form here; the verbose one
    # is reachable so a sweep can price the difference.
    numpy_scalar_repr: Literal["np2", "legacy"] = "legacy"
    profile: str = "play"
    # The sampling profiles of the game's other phases, by name in
    # ``llm_config.sampling_profiles`` (2026-10-03): ``audit_profile``
    # samples the audit turn and every block after it, ``summary_profile``
    # every summary, in either phase. A summary keeps the live request's own
    # state (its seed, injection and recurrent-state keys) and takes the
    # profile's sampling: temperature, ``top_p``, ``after_reasoning`` and the
    # profile's ``extra_params`` over the live ones. None samples the phase on
    # ``profile``.
    audit_profile: str | None = None
    summary_profile: str | None = None
    # What bounds a game, in place of a wall clock (upstream's ``design.md``,
    # "Run budgets"): completion tokens, counted from the backend's per-game
    # meter so it includes recovered requests no longer present in the
    # conversation. Upstream budgets per LEVEL and leaves the game budget
    # off; a task has no levels, so the game budget is the whole of it.
    # ``None`` disables it, which leaves the analyzer loop unbounded — it
    # re-enters until the answer is complete, and nothing else stops it.
    #
    # Sized so an arm's MEAN spend can reach the budget a sweep reads arms
    # at (`sweeper.DEFAULT_BUDGET_TOKENS`, 45k charged tokens), with margin
    # for the draws past it; nothing past the cap is generated only to be
    # discarded at the reading. The measurement is in evidence.md
    # ("max_tokens_per_game"): scaled from a 40k reading, not re-measured.
    max_tokens_per_game: int | None = 65_000
    # Inert; accepted so existing constructions keep working. The
    # compaction decision runs on the engine's own billed ``usage`` plus
    # a provable bound for what was appended since (see "6. context
    # budget"), whatever this flag says — it selects nothing.
    exact_token_counts: bool = True
    # Retry and retention controls are independent of the caller's task.
    perturb_retry_prompt: bool = True
    preserve_failed_draws: bool = False
    classify_tool_calls: bool = False
    response_observer: Callable[[ullm.conv.LLMResponse], None] | None = field(default=None, repr=False)
    """Observe each processed ordinary or summary response before its tools run.

    The synchronous callback must be picklable with the per-game agent. It may
    persist response evidence; exceptions propagate and stop the turn.
    """
    # End the game on a retained reply that carries no text and no tool
    # call: the empty reply a training backend keeps as the model's turn
    # (``accept_malformed_output``). Off, the entry nudges and a policy can
    # spend its whole episode on one-token replies; on, the first such reply
    # is a give-up, so whatever stands is scored and the reply's malformed
    # count is charged once.
    terminate_on_empty_reply: bool = False
    # Re-samples allowed only when a backend rejects a malformed reply.
    # Inert when its accept_malformed_output policy retains the draw. Three,
    # before the nudge below: see there.
    malformed_retries: int = 3
    # How many times a game is nudged, rather than crashed, when a play reply
    # is refused on every one of those re-samples for ending with no text and
    # no tool call, or for degenerate repetition (2026-10-03). The
    # refused draws are dropped -- in practice they loop inside their
    # reasoning -- and the nudge says the reply was discarded and asks for a
    # different approach. 0 never nudges: the error crashes the game, as it
    # does past the count. Capped because refused draws
    # are not billed to the game, so its budget would never stop a model that
    # loops every time. Over sweeps 6-8 the next draw succeeded after 1, 2 and
    # 3 refusals 92%, 78% and 45% of the time, and never after 4, so 3
    # re-samples suit it. One: in sweep 9, 14 of 21,735 games crashed without
    # it and 1 of 22,365 with it, at no score or token cost (``evidence.md``).
    malformed_nudges: int = 1
    # Whether tools may repair malformed argument names, JSON or escapes.
    # Independent of retries: no re-sampling does not imply no execution repair.
    auto_correct_tool_arguments: bool = True
    # --- context accounting knobs (all conservative-side) ---
    # Subtracted from the window on top of the reply reserve when computing
    # the eviction budget: the slack that absorbs template re-render
    # wobble (retry newline perturbations, tool-call JSON re-encoding)
    # between a conservative prediction and the engine's own count.
    safety_margin_tokens: int = 512
    # Flat per-message add-on for chat-template framing tokens, which are
    # not part of any rendered text and so escape the character-count
    # bound: role header + end-of-message markers are ~4-6 tokens per
    # message on the Qwen chat template, plus tool-result framing.
    per_message_overhead_tokens: int = 8
    # Characters of the rendered request charged per text token. 1.0 is the
    # provable bound: a text token spans at least one character. Above it the
    # text term is a calibration for one tokenizer, set where the backend
    # cannot count (a hosted API) and the bound would compact far too early
    # (Mistral Large 4 runs 3.9-4.3 characters a token on ARC3 play); text
    # that tokenizes worse than calibrated can undercount, which the next
    # anchor heals (``usage_window_tokens``).
    bound_chars_per_token: float = 1.0
    # Fixed engine-side token cost charged per prompt image, overriding
    # the derivation in ``_image_bound_tokens``. None = derive from the
    # image's pixel dimensions (see that method for the arithmetic).
    image_tokens_override: int | None = None
    # The billed total the window invariant holds each response to. None =
    # the agent's own window. A hosted teacher held to a student's window by
    # the calibrated estimate above sets the API's window here: its bills
    # count in its own tokenizer, so one past the student's window is the
    # estimate missing (Kimi bills an image at roughly twice Flash-Next's),
    # which the next anchor heals, not an oversized sequence. Never tighter
    # than the agent's own (2026-10-27).
    usage_window_tokens: int | None = None
    # Preset-declared serving mode (the vLLM topologies set it): the backend
    # continues the served history by token ids and answers
    # ``Backend.token_count`` exactly — the id-ledger length on a continued
    # call, the server's own render otherwise. The decision path then
    # measures with that exact weight, squeezing all slack out of the
    # window: the trigger is ledger + reserve + margin, nothing else. Off
    # (the default), the engine-billed + provable-bound arithmetic governs —
    # the general contract for cloud backends, which must never be asked to
    # count (their recorded streams are pinned).
    token_native_context: bool = False
    # Keep sent history intact between requests; the next pre-call trim
    # includes the new status text and can prune the active turn as well.
    compact_before_request_only: bool = False
    # Consumers that need exact evidence must never silently use a bound.
    require_exact_context: bool = False
    context_observer: ContextObserver | None = field(default=None, repr=False)
    _context_compacting: bool = field(default=False, init=False, repr=False)
    # Ask the engine to zero the recurrent state at `zero_recurrent_at` on
    # every call: the model then decodes with verbatim (attention) recall
    # of the whole prompt but a running (recurrent) summary of only what
    # follows the position. What the blank buys is a change of REGISTER --
    # the model drops into a telegraphic form ("We need solve game.") and
    # stops ruminating -- and the register is where the points are
    # (evidence.md, "The reasoning register"); serving the register directly
    # (`inject_text_on_first_prompt`, below) reaches it on a stock engine,
    # and the blank stays as the one route to the register through the
    # STATE rather than the tokens. Off here: where the blank should land on
    # this prompt layout is unmeasured (the roadmap has the question), so
    # the treatment is reachable but not shipped. It requires the
    # arc-prize-2026-vllm fork this repo pins, which REJECTS the parameter on a
    # stock engine rather than ignoring it -- so a mismatched engine stops
    # the run instead of quietly scoring one that never blanked. Leave it
    # OFF for any backend that is not that fork (a cloud model, say).
    zero_recurrent: bool = False
    # Each call's generation cap is also the context budget's reply reserve,
    # so the window moves with it (``context_window_tokens``) and the prompt
    # budget stays at the 48,640 the sweeps read arms at. 24k (sweep 7,
    # 2026-09-21): over 12k-32k the score rose 0.13 points per 1k of cap,
    # nearly all of it below ~22k, while tasks an hour fell throughout, so
    # the cap sits just past where the evidence ends; below a knee near 11k
    # the score fell 2.5 points per 1k (packages/duck/evidence.md, "The
    # reply cap"). None leaves generation uncapped and reserves a nominal
    # 512 tokens.
    reply_cap_tokens: int | None = 24576
    # Settle the engine window from the agent's context window, at
    # ``prepare_for_backend``. The two are one contract — the agent may build
    # prompts up to its window, and the engine must accept exactly that — so
    # stating them separately is stating the same number twice, and a caller
    # who moves one is otherwise silently at odds with the other.
    #
    # Off is for a deployment whose engine length is not ours to choose: the
    # window then has to be set to match it by hand. An engine argument set
    # explicitly is honoured rather than overwritten either way — the derivation
    # refuses to run over a chosen value instead of quietly winning.
    derive_max_model_len: bool = True
    # Settle the engine's sequence cap from the harness concurrency, likewise.
    # The harness admits ``run_config.max_concurrent`` games and each holds at
    # most one generation, so that IS the engine's concurrent-sequence demand;
    # a separately stated cap is either slack the GDN state pays for or a
    # ceiling the harness quietly queues behind.
    derive_max_num_seqs: bool = True
    # Settle the image processor's pixel ceiling from what the agent can
    # render, likewise. The engine's start-up profile reserves activation for
    # one image of the processor's largest size, which for these checkpoints
    # is tens of megapixels; an agent that renders every image itself knows
    # its own largest, and the difference is KV pool. Past the ceiling the
    # processor SHRINKS an image rather than refusing it, so the ceiling has
    # to follow the render rules, not be stated beside them.
    # The same settle refuses video, which the duck never sends, so the
    # multimodal profile and caches size for images alone.
    derive_max_pixels: bool = True
    # Where the blank lands, in absolute prompt tokens: upstream's first
    # recurrent block boundary, which ITS shared system prompt spans, so what
    # the recurrent layers forget there is the game overview and tool
    # interface. Measured upstream as the position that reproduces the
    # accident, not tuned for score, and not re-derived for this prompt
    # layout. Any position inside the prompt is applied; one the prompt does
    # not reach is refused by the engine.
    zero_recurrent_at: int = 1568
    # Text served after the generation prompt of the game's FIRST model
    # call only, which the model then continues from (on a thinking
    # template it lands right inside the opened think block). Empty = off.
    #
    # The opening of the telegraphic register the model writes in when its
    # recurrent state is blanked, and serving it is the way INTO that
    # register: the sampler reaches it unprompted on some lanes, and serving
    # the register's own first words as the model's own tokens puts every
    # lane there, where asking for it in the prompt puts none. Three
    # tokens: one past the two-word threshold, and the form the A/B here
    # measured, worth 15.4 points at a matched budget on arc2-eval
    # (evidence.md, "The reasoning register, measured here").
    #
    # Served by the in-process vLLM backend alone; any other backend refuses
    # the call, so an agent on a cloud model, a scripted stub or an OpenAI
    # server must leave this "". In estimate-mode budgeting its tokens fall
    # outside the character-derived bound and are absorbed by
    # `safety_margin_tokens`, like a retry perturbation; token-native mode
    # weighs them exactly.
    inject_text_on_first_prompt: str = "We need solve"
    # Text served the same way on EVERY call after the first. Empty = off.
    # Exists because the first-call opening does not hold a game: it is an
    # example left in history, and compaction drops it. There is no
    # late-game analogue of the opening to serve -- concise and verbose
    # blocks open alike once the game is under way -- so what goes here has
    # to fit any moment: the opening's own first two tokens, content-free and
    # enough on their own to set the register. What sustains the register is
    # the schedule, not the words (evidence.md, "The reasoning register").
    inject_text_on_later_prompts: str = "We need"
    # "context" opens each declared history epoch with the first text.
    # Counting a prompt does not advance this schedule; only a prepared
    # model round does. The ordinary game-first/later schedule stays default.
    prompt_injection_schedule: Literal["game", "context"] = "game"
    # What the system prompt asks of the model's reasoning depth. The class
    # default is provider-neutral — "high" is a level every backend that
    # consumes the value accepts — because this field reaches whatever
    # backend the duck is paired with, not only the shipped presets. The
    # Qwen3.8 presets inject "xhigh" (the model's own default, the swept
    # configuration) at construction; a hand-built duck states its own.
    reasoning_effort: Literal["low", "medium", "high", "max", "xhigh"] | None = "high"
    # RLIMIT_AS for the kernel subprocess. The whale's number: at ullm's
    # default `import numpy` wedges rather than failing.
    kernel_memory_limit_mb: int = 8192
    # The prompt implementation: which WORDING the agent speaks. A swap seam —
    # a variant surface drops in as a different instance (the whale's
    # Modeler.prompts pattern). Wording is all it holds: the numbers a render
    # quotes reach it per call as a PromptModes (_prompt_modes), so there is
    # no second place to configure them and nothing here to set by mistake.
    prompts: Prompts = field(default_factory=DuckPrompts)
    # Infra rather than model-facing, and the one run-level default the duck
    # states for itself: the per-game wall net, ``presets.GAME_RUNTIME_CAP_S``.
    # A preset replaces this whole object rather than editing it, so each one
    # says what it wants — the served presets take the cap, Kaggle takes None.
    run_config: RunConfig = field(default_factory=presets.duck_run_config)

    # --- state: the kernel and the conversation (pickles; template
    #     deepcopies cheaply) ---
    repl: DuckPythonTool | None = None
    # Built with the first tool list that needs it, and kept: the undo log
    # lives on it. It reads its directory off the repl at call time, so a
    # restored agent's fresh tmpdir needs no re-pointing.
    workspace: Workspace | None = None
    messages: list[ullm.conv.ConversationElement] = field(default_factory=lambda: list[ullm.conv.ConversationElement]())
    _pending_tool_feedback: list[list[ullm.conv.Content]] = field(
        default_factory=lambda: list[list[ullm.conv.Content]](),
        repr=False,
    )

    # --- state: the answer as last read out of the kernel ---
    # Before the first kernel call nothing has been read, and the repair
    # prompt must say so rather than describe a namespace it never saw.
    reading: AnswerReading = field(
        default_factory=lambda: AnswerReading(bound=False, why="test_out has not been written to yet")
    )
    submissions_made: int = 0
    # How many of ``malformed_nudges`` this game has spent.
    _malformed_nudged: int = 0
    # Which of ``intermediate_summary_prompt_lengths`` this compaction cycle has
    # passed, and how many intermediate summaries its history holds.
    _intermediate_lengths_passed: tuple[int, ...] = ()
    _intermediate_summaries_in_cycle: int = 0
    _last_submitted: Any = None
    # The answer grids ``show_changed_answers`` last showed, per question and
    # attempt (``None`` for a slot without a usable grid): what a call's
    # reading is compared against. Reset with the kernel's task.
    _answers_shown: tuple[tuple[taaf.grid.Grid | None, ...], ...] = ()
    # How many questions this game asks; the nudge reads the answer's
    # problems against it. Set at the top of the game.
    _n_questions: int = 0
    # The layout this game's puzzle is shown in; ``None`` for the original.
    # Set at the top of the game.
    _layout: PuzzleLayout | None = None
    # The layout a multi-take wrapper planned for this take, used in place
    # of the duck's own draw.
    _take_layout: PuzzleLayout | None = None
    # Whether this game's budget warning has been sent.
    _budget_warned: bool = False
    # The reply the audit turn quoted and the puzzle it restated, kept to
    # rebuild the pinned prompt at each later summary; None until it fires.
    _audit_quote: str | None = None
    _audit_view: taaf.game.GameView | None = None
    # Whether a compaction has fired since the audit did: from then on the slot
    # offers carry ``audit_handicap_tokens``.
    _audit_compacted: bool = False
    # What ``answer_confidence`` states: the audit's final reply, read once the
    # analyzer is done.
    _confidence: int | None = None

    # --- state: context accounting ---
    counts_reconciled: int = 0  # billed usages checked against the window invariant (0 = the check never ran)
    context_summaries: int = 0
    _pruned_rounds: int = 0  # rounds dropped from the CURRENT turn, for the note
    _last_trim_fired: bool = False  # did the last trim fire? read by the diagnostics hook and the slot offer
    _count_drift_events: int = 0  # count-path HistoryDriftError degrades this run, carried on the span kv
    # The live Conversation for the current turn block, live only for the
    # duration of one game and cleared in its finally — the decision path
    # declares its history rewrites on it, and a live conversation must
    # never ride a snapshot.
    _live_conversation: Any = None
    # The block's backend and sampling params, for token-native measurement
    # only; cleared with the conversation so no backend rides a snapshot.
    _live_backend: Any = None
    _live_params: Any = None
    # Compaction spacing, for the per-trim diagnostics: the decision reading
    # left after the previous fired trim, and the LLM calls made since.
    _tokens_at_last_compaction: int | None = None
    # Every model call of the game, counted before the call is made; pickles
    # with the game, so a resumed game knows it has already opened.
    response_step: int = 0
    _rounds_since_compaction: int = 0
    _status_handle: Any = None  # ctx.status for the current block; publishes "compacting" during trims
    # One predicted next-prompt upper bound per before_round, paired 1:1
    # with the turn's responses for the post-turn window-invariant check.
    _pending_bounds: list[int] = field(default_factory=lambda: list[int]())
    # (bound, response) pairs whose rounds round-pruning removed from the
    # live turn mid-block. Every billed response faces the window invariant
    # exactly once, so leaving the live pairing they leave the check queue
    # too — drained by the block's _check_engine_usage.
    _deferred_usage_checks: list[tuple[int, ullm.conv.LLMResponse]] = field(
        default_factory=lambda: list[tuple[int, ullm.conv.LLMResponse]]()
    )
    # Every LLMResponse present when the last trim rewrote sent history.
    # Their ``usage`` describes the PRE-rewrite prompt, so the decision path
    # must not extrapolate from them; the first response generated after the
    # rewrite (absent from here) clears this. None = no rewrite pending
    # validation. Held as OBJECTS, not ``id()``s: the trim that fills this
    # list is the one that drops those responses, so their addresses are
    # free for the allocator to hand to a later response, which would then
    # read as stale. A reference cannot be recycled.
    _stale_responses: list[ullm.conv.LLMResponse] | None = None
    # Version counter of sent history — bumped once per trim that rewrote
    # it (a block evicted, a round pruned, the pruned-rounds note): the
    # conversation's ``history_epoch`` (ullm.continuation), kept here
    # because each analyzer entry hosts the persistent element list in a
    # fresh Conversation. A version, not a count of individual edits.
    history_epoch: int = 0
    _prompt_injection_epoch: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Move-boundary pickle: drop the staleness record -- a list of
        process-local objects that match nothing after a restore."""
        state = dict(self.__dict__)
        state["_stale_responses"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore, then mark every restored response stale: a pre-restore
        bill may describe a prompt a pre-boundary trim already rewrote, so
        nothing anchors the decision until the next fresh response
        ("trust after verify")."""
        vars(self).update(state)
        self._stale_responses = [
            sub
            for element in self.messages
            if isinstance(element, ullm.conv.ResponseTurn)
            for sub in element.elements
            if isinstance(sub, ullm.conv.LLMResponse)
        ]

    def __post_init__(self) -> None:
        """Reject what one field can be judged on alone.

        The checks that span fields run at ``prepare_for_backend`` instead, so
        they see a configuration that was assigned rather than constructed.
        These run in both places: cheap, and worth failing on immediately.
        """
        self._validate_fields()

    def _validate_fields(self) -> None:
        """Unknown modes and anti-conservative accounting knobs."""
        self._check_mode()
        if self.prompt_injection_schedule not in {"game", "context"}:
            raise ValueError(f"unknown prompt_injection_schedule {self.prompt_injection_schedule!r}")
        if self.safety_margin_tokens < 0:
            raise ValueError(f"safety_margin_tokens={self.safety_margin_tokens} must be >= 0")
        if self.per_message_overhead_tokens < 0:
            raise ValueError(f"per_message_overhead_tokens={self.per_message_overhead_tokens} must be >= 0")
        if not self.bound_chars_per_token >= 1.0:
            raise ValueError(f"bound_chars_per_token={self.bound_chars_per_token} must be >= 1.0")
        if self.image_tokens_override is not None and self.image_tokens_override < 0:
            raise ValueError(f"image_tokens_override={self.image_tokens_override} must be >= 0")
        if self.usage_window_tokens is not None and self.usage_window_tokens < self.context_window_tokens:
            raise ValueError(
                f"usage_window_tokens={self.usage_window_tokens} must be >= "
                f"context_window_tokens={self.context_window_tokens}"
            )
        if self.reply_cap_tokens is not None and (type(self.reply_cap_tokens) is not int or self.reply_cap_tokens <= 0):
            # A zero cap would read as "unset" downstream and quietly reserve 512.
            raise ValueError(f"reply_cap_tokens={self.reply_cap_tokens} must be positive, or None for no cap")
        if self.summary_tail not in {"full", "without_reasoning", "none"}:
            raise ValueError(f"unknown summary_tail {self.summary_tail!r}")
        if type(self.audit_handicap_tokens) is not int or self.audit_handicap_tokens < 0:
            raise ValueError(f"audit_handicap_tokens={self.audit_handicap_tokens!r} must be a whole number >= 0")
        if self.numpy_scalar_repr not in {"np2", "legacy"}:
            raise ValueError(f"unknown numpy_scalar_repr {self.numpy_scalar_repr!r}")
        profile = self.llm_config.sampling_profiles.get(self.profile)
        if isinstance(profile, ullm.backend.GenericSamplingParameters) and profile.max_tokens is not None:
            raise ValueError(
                f"sampling profile {self.profile!r} must leave max_tokens unset; set reply_cap_tokens instead"
            )
        self._reply_reserve_tokens()
        if not self._uses_context_summary():
            self._compact_target_tokens()
        self._check_images_enabled()

    def _validate_coherence(self) -> None:
        """Reject combinations no single field can be judged on.

        Each of these spans the agent, its config, the factory or the engine
        actually installed, so none of them can be settled at construction: a
        value assigned afterwards would never be seen. They run at
        ``prepare_for_backend``, which is after every override and before the
        engine starts, so a refusal still costs nothing.
        """
        if self.preserve_failed_draws and self.malformed_retries != 0:
            raise ValueError(
                "preserve_failed_draws=True requires malformed_retries=0: "
                "failed responses are retained and terminal, rather than rolled back for a retry"
            )
        if self.malformed_nudges < 0:
            raise ValueError(f"malformed_nudges must be 0 or more; got {self.malformed_nudges}")
        if self.preserve_failed_draws and self.malformed_nudges:
            raise ValueError(
                "preserve_failed_draws=True requires malformed_nudges=0: a retained draw is terminal, "
                "while the nudge drops it"
            )
        if self.audit_turn and self.check_mode is CheckMode.MANDATORY:
            raise ValueError(
                "audit_turn=True requires check_mode other than MANDATORY: a passing mandatory check ends "
                "the game, so no stop leaves an answer standing to audit"
            )
        self._check_phase_profile("audit_profile", self.audit_profile)
        if self.intermediate_summary_prompt_lengths:
            if any(type(n) is not int or n <= 0 for n in self.intermediate_summary_prompt_lengths):
                raise ValueError(
                    f"intermediate_summary_prompt_lengths={self.intermediate_summary_prompt_lengths} "
                    "must be positive token counts"
                )
            if not self._uses_context_summary():
                raise ValueError(
                    "intermediate_summary_prompt_lengths needs compaction_strategy='summarize': the lengths "
                    "are re-armed by each compaction, and the summaries are written as its own are"
                )
        self._check_phase_profile("summary_profile", self.summary_profile)
        engine_args = getattr(self.llm_config.backend_factory, "engine_args", None)
        engine_window = getattr(engine_args, "max_model_len", None)
        if engine_window is not None and self.context_window_tokens > engine_window:
            raise ValueError(
                f"context_window_tokens={self.context_window_tokens} exceeds the engine's "
                f"max_model_len={engine_window}: the agent would build prompts the engine rejects. "
                "Raise the engine's window, or leave derive_max_model_len on and it "
                "takes the agent's."
            )
        # A compaction target that cannot sit below the budget is the same
        # kind of incoherence, and the check is the computation.
        if not self._uses_context_summary():
            self._compact_target_tokens()
        if self.zero_recurrent:
            # Asked of the INSTALLED engine, because the pin that selects it
            # does not always apply. `override-dependencies` is read from the
            # uv workspace ROOT only, so a parent workspace consuming these
            # packages as members resolves vllm by its own rules and can bring
            # a stock build with our agent. Absent the field, the blank cannot
            # be asked for at all, so refuse here rather than at the first call.
            vllm = _installed_vllm()
            if "zero_recurrent_state" not in getattr(vllm.SamplingParams, "__struct_fields__", ()):
                raise ValueError(
                    f"zero_recurrent=True, but the installed vllm ({vllm.__version__}) has no "
                    "SamplingParams.zero_recurrent_state. It needs the arc-prize-2026-vllm fork this "
                    "repo pins (see vllm_fork.md); a workspace that resolved vllm by its own "
                    "rules will have a stock build. Install the pinned revision, or set "
                    "zero_recurrent=False to run without the blank."
                )
            if getattr(self.llm_config.backend_factory, "env_overrides", {}).get("VLLM_USE_V2_MODEL_RUNNER") == "0":
                # Only the V2 runner implements the blank, and the fork refuses a
                # blanked request on V1 at input validation -- so this combination
                # does not run and score unblanked, it dies at the first LLM call
                # of every game. Refused here for the reason the window check is:
                # the engine only says so once a game has paid for the hardware and
                # the model load. An absent key is left alone: a topology whose env
                # this config does not own may still be running V2.
                raise ValueError(
                    "zero_recurrent=True on an engine pinned to the V1 model runner: the blank is "
                    "implemented only by the V2 model runner, and the engine refuses every blanked "
                    "request without it (VLLM_USE_V2_MODEL_RUNNER=1). Drop the override, or set "
                    "zero_recurrent=False to run without the blank."
                )
            # A served engine is somebody else's build, and the blank rides
            # sampling fields only the fork declares.
            base_url = getattr(getattr(self.llm_config.backend_factory, "backend", None), "base_url", "")
            if base_url:
                self._require_matching_server_build(str(base_url))

    def _check_phase_profile(self, field_name: str, name: str | None) -> None:
        """Refuse a phase profile that is missing or sets what the agent owns:
        the cap is ``reply_cap_tokens`` (a summary's, its target) and the seed
        is drawn per call, so either set on the profile would be ignored."""
        if name is None:
            return
        profiles = self.llm_config.sampling_profiles
        if name not in profiles:
            raise ValueError(f"{field_name}={name!r} names no sampling profile; llm_config defines {sorted(profiles)}")
        profile = profiles[name]
        if not isinstance(profile, ullm.backend.GenericSamplingParameters):
            raise ValueError(f"{field_name}={name!r} is a {type(profile).__name__}, not GenericSamplingParameters")
        if profile.max_tokens is not None or profile.seed is not None:
            raise ValueError(
                f"{field_name}={name!r} must leave max_tokens and seed unset; the agent sets both per call"
            )

    def prepare_for_backend(self, *, settle_engine: bool = True) -> None:
        """Derive local engine limits from the agent's final configuration.

        Supplied engines require explicit opt-outs because their limits are
        owned by the caller. Replay has no factory and needs no derivations.
        ``settle_engine=False`` says the engine is another agent's (a later
        take, ``platypus.multi_take``): nothing is derived onto it, and only
        the agent's own checks, and those against that engine, apply. Under
        ``summarize`` the window derived is the engine's own ceiling
        (``_derived_engine_window``).
        """
        super().prepare_for_backend(settle_engine=settle_engine)
        self._validate_fields()
        factory = self.llm_config.backend_factory
        engine_args = getattr(factory, "engine_args", None)
        if settle_engine and factory is not None and engine_args is None:
            for flag in ("derive_max_model_len", "derive_max_num_seqs", "derive_max_pixels"):
                if getattr(self, flag):
                    raise ValueError(
                        f"{flag}=True, but the engine isn't built here: the supplied factory "
                        f"has no engine_args. Set {flag}=False and own that engine limit explicitly."
                    )
        if engine_args is not None and settle_engine:
            baseline = getattr(factory, "_derivation_baseline", {})
            if self.derive_max_model_len:
                self._settle_engine_arg(
                    engine_args,
                    baseline,
                    "max_model_len",
                    self._derived_engine_window(),
                    "derive_max_model_len",
                )
            if self.derive_max_num_seqs:
                self._settle_engine_arg(
                    engine_args, baseline, "max_num_seqs", self.run_config.max_concurrent, "derive_max_num_seqs"
                )
            if self.derive_max_pixels:
                ceiling = images.largest_image_pixels(scale=self._image_scale(), gridlines=self.image_gridlines)
                # The processor's other arguments are the engine's own (a
                # checkpoint's tokenizer quirk, say) and ride along untouched.
                current: dict[str, Any] = dict(getattr(engine_args, "mm_processor_kwargs", None) or {})
                self._settle_engine_arg(
                    engine_args,
                    baseline,
                    "mm_processor_kwargs",
                    {**current, "max_pixels": ceiling},
                    "derive_max_pixels",
                )
                limits: dict[str, Any] = dict(getattr(engine_args, "limit_mm_per_prompt", None) or {})
                self._settle_engine_arg(
                    engine_args, baseline, "limit_mm_per_prompt", {**limits, "video": 0}, "derive_max_pixels"
                )
        self._validate_coherence()

    def _derived_engine_window(self) -> int:
        """The engine window the agent derives.

        Under ``prune`` it IS ``context_window_tokens``: the agent never sends
        a prompt past its window. ``summarize`` sends its summary request on
        the whole live history, past the window by the round that overflowed
        it, the instruction and the summary itself -- and how far past is not
        the agent's to know: a tool result is capped in characters, not
        tokens, failed-check grids are sent uncapped, and how far the summary
        runs is the model's, within its cap. So the window is the ceiling the
        engine was built with -- the checkpoint's native window on our
        columns, which costs nothing in the pool -- and what the agent does
        own (the window, the reply reserve and the summary's cap) has to fit
        under it, or the derivation refuses rather than pick a number the run
        can exceed.

        ``engine_ceiling_tokens`` rather than the factory's own record of what
        it was built with: that record follows every settle, and the build
        settles once under whatever strategy the agent had then.
        """
        if not self._uses_context_summary():
            return self.context_window_tokens
        ceiling = self.engine_ceiling_tokens
        if ceiling is None:
            raise ValueError(
                "summarize needs an engine window past the agent's, and this agent has no engine_ceiling_tokens "
                "to take (the build sets it from the engine's own arguments): set engine_args.max_model_len and "
                "derive_max_model_len=False"
            )
        target = self._summary_target_tokens()
        # The summary's cap with reasoning whether or not it is on: a summary
        # written without it is retried with it when it comes back empty.
        owned = self.context_window_tokens + self._reply_reserve_tokens() + 6 * target
        if owned > ceiling:
            raise ValueError(
                f"the summary request reaches past the engine's ceiling of {ceiling} on what the agent owns alone: "
                f"context_window_tokens={self.context_window_tokens}, the reply reserve and the summary's cap come to "
                f"{owned}, before the round and the instruction the request also carries. Narrow the window, the "
                "reply cap or the summary (compaction_ratio), or state a wider engine window and turn "
                "derive_max_model_len off."
            )
        return ceiling

    def _validate_host_backend(self, backend: ullm.backend.Backend) -> None:
        """Reject a connected engine that cannot fit the agent's context window."""
        super()._validate_host_backend(backend)
        engine = getattr(backend, "engine", None)
        if engine is None:
            return
        window = getattr(getattr(engine, "model_config", None), "max_model_len", None)
        if type(window) is not int or window <= 0:
            raise ValueError("the connected engine must report a positive max_model_len")
        if window < self.context_window_tokens:
            raise ValueError(
                f"context_window_tokens={self.context_window_tokens} exceeds the connected engine's "
                f"max_model_len={window}. Raise the engine window or lower context_window_tokens."
            )

    @staticmethod
    def _settle_engine_arg(engine_args: Any, baseline: dict[str, Any], name: str, value: Any, flag: str) -> None:
        """Write a derived engine argument, unless someone chose it.

        "Chose it" is measured against the baseline: what the factory was built
        with, or what this settled onto it last. That is the only way to tell a
        caller's value from one nobody picked, since the field holds a number
        either way — and updating it on write is what lets the agent be
        settled, changed and settled again. A chosen value is never
        overwritten and never silently kept: it raises, because a run whose
        engine quietly disagrees with its agent is the failure this exists to
        prevent, and a caller who means it says so with the flag.
        """
        current = getattr(engine_args, name, None)
        if name in baseline and current != baseline[name]:
            raise ValueError(
                f"engine_args.{name}={current} was set explicitly, but {flag} is on and would "
                f"derive {value} from the agent. Both cannot hold. Either drop the explicit value and let "
                f"it follow the agent, or set {flag}=False and own the pairing."
            )
        setattr(engine_args, name, value)
        # A copy, because one of these values is a dict: a baseline holding
        # the engine's own object would follow a later in-place edit of it,
        # and the edit would then read as the value this settled rather than
        # as the caller's choice it is.
        baseline[name] = copy.deepcopy(value)

    # ------------------------------------------------------------------
    # 1. deployment
    #
    # A fully-configured agent instance describes a whole experiment, so
    # the deployment is a constructor rather than a solver factory. What it
    # configures lives in duck/presets.py.
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        engine: str = presets.DEFAULT_ENGINE,
        environment: str = presets.DEFAULT_ENVIRONMENT,
        serving: Literal["direct", "server"] = "direct",
        backend_factory: Any = None,
        kimi_student: bool = False,
        **overrides: Any,
    ) -> Self:
        """A duck configured for one engine in one environment.

        The two coordinates are the whole deployment: ``engine`` names a
        checkpoint (``ullm.backends.deployment.ENGINES``), ``environment`` the
        machine it runs on (``duck.presets.ENVIRONMENTS``). ``duck.presets.configure`` says what each
        contributes.

        ``backend_factory`` is the caller's, for a model this process does not
        start; ``ullm.backends.deployment`` supplies the engine facts one needs
        -- ``client_kwargs`` for a served model, ``backend_kwargs`` for an
        engine the caller already holds.

        ``serving`` picks the topology over one set of engine args — the
        in-process engine, or vLLM's HTTP server in this process.

        Everything else is a field: nothing on the returned agent is frozen, so
        a run that wants a different concurrency, window or tier sets it and the
        engine follows at ``prepare_for_backend``. ``llm_config`` and
        ``run_config`` are the exception — the column composed them, so they are
        assigned into rather than replaced.

        ``kimi_student`` sets :data:`KIMI_STUDENT` first, so a model trained on
        the Kimi K3.1 Pro teacher traces plays the duck those traces were
        recorded on; an override passed with it still wins. The checkpoint is
        the engine coordinate's, as for any other model.
        """
        if kimi_student:
            overrides = {**KIMI_STUDENT, **overrides}
        agent = cls()
        presets.configure(
            agent,
            engine=engine,
            environment=environment,
            serving=serving,
            backend_factory=backend_factory,
        )
        # Last, so an explicit value always beats the column's. Rejected by
        # name rather than assigned blindly: nothing freezes an agent's
        # attributes, so a typo would otherwise be a field nobody reads.
        known = {f.name for f in dataclasses.fields(agent)}
        for name, value in overrides.items():
            if name in _COMPOSED_BY_THE_COLUMN:
                raise TypeError(
                    f"{name} cannot be passed to build(): the column has already composed it, and a "
                    f"replacement drops what it put there — the backend factory, the deploy assets, "
                    f"the sampling profiles. Assign to the object instead: agent.{name}.<field> = ..."
                )
            if name not in known:
                raise TypeError(f"{cls.__name__} has no field {name!r}")
            setattr(agent, name, value)
        # Settle and check eagerly, so an interactive caller hears about a
        # broken configuration now rather than at run start. Idempotent, and
        # the solver runs it again over whatever was changed since.
        agent.prepare_for_backend()
        return agent

    @staticmethod
    def _require_matching_server_build(base_url: str) -> None:
        """Refuse an external server that is not this client's own vLLM build.

        The blank rides sampling parameters the fork declares; a server built
        from any other revision accepts the request and serves it unblanked,
        because the fields it does not declare are dropped as unknown extras.
        Nothing in the response says so, and the proxy the agent talks through
        deliberately hides what is behind it -- so the only moment this can be
        caught is here, before a run starts, by asking the server what it is.

        Equality of the full version string is the test rather than "has the
        feature": the local part carries the fork revision, and a different
        revision is a different engine whose blank may land elsewhere.
        """
        endpoint = base_url.rstrip("/").removesuffix("/v1") + "/version"
        try:
            with urllib.request.urlopen(endpoint, timeout=10) as response:
                served = json.load(response).get("version", "")
        # Narrow deliberately: URLError subclasses OSError and a bad body
        # raises ValueError, so this covers "could not ask" without also
        # swallowing a programming error and reporting it as a server fault.
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"zero_recurrent=True, but this agent could not ask {endpoint} which vLLM the "
                f"server runs ({type(exc).__name__}: {exc}). The blank is dropped silently by a "
                "server that is not the pinned fork, so the run is refused rather than scored "
                "unblanked. Set zero_recurrent=False to run against this server without it."
            ) from exc
        vllm = _installed_vllm()
        if served != vllm.__version__:
            raise ValueError(
                f"zero_recurrent=True, but {endpoint} reports vLLM {served!r} while this client "
                f"is built against {vllm.__version__!r}. A server on another revision accepts the "
                "sampling parameters as unknown extras and serves the request unblanked. Point at "
                "a server running this build (see vllm_fork.md), or set zero_recurrent=False."
            )

    def _uses_context_summary(self) -> bool:
        """Whether the validated model-written compaction arm is active."""
        if self.compaction_strategy not in {"prune", "summarize"}:
            raise ValueError(f"compaction_strategy must be 'prune' or 'summarize'; got {self.compaction_strategy!r}")
        if self.compaction_strategy == "prune":
            return False
        if self.compaction_ratio < 2:
            raise ValueError(f"compaction_ratio must be at least 2; got {self.compaction_ratio}")

        def invalid_token_id(token_id: object) -> bool:
            return isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0

        if any(invalid_token_id(token_id) for token_id in self.compaction_forbidden_token_ids):
            raise ValueError(
                "compaction_forbidden_token_ids must contain only non-negative integers; "
                f"got {self.compaction_forbidden_token_ids!r}"
            )
        return True

    def needs_images(self) -> bool:
        """Whether the duck's prompts carry pictures: every ``images`` mode but
        ``off``."""
        return self._images_enabled()

    def _images_enabled(self) -> bool:
        """Whether task images are enabled, validating the mode at its seam."""
        allowed = {"off", "task", "task+show", "task+show+check"}
        if self.images not in allowed:
            raise ValueError(f"images must be one of {sorted(allowed)}; got {self.images!r}")
        return self.images != "off"

    def _text_grids_enabled(self) -> bool:
        """Whether exact task digit rows are shown in the task prompt."""
        return self.text_grids

    def _image_layout(self) -> Literal["paired", "separate"]:
        """The validated task-image composition mode."""
        if self.image_layout not in {"paired", "separate"}:
            raise ValueError(f"image_layout must be 'paired' or 'separate'; got {self.image_layout!r}")
        return self.image_layout

    def _show_enabled(self) -> bool:
        """Whether ``show(grid)`` belongs to this arm's prompt and kernel."""
        self._images_enabled()
        return self.images in {"task+show", "task+show+check"}

    def _show_changed_answers(self) -> bool:
        """Whether a call that changes ``test_out`` is followed by the changed
        answers, validated against the modes it needs."""
        if not self.show_changed_answers:
            return False
        if not self._images_enabled():
            raise ValueError("show_changed_answers shows each answer as an image; it needs images on")
        if self.check_mode is CheckMode.MANDATORY:
            raise ValueError("show_changed_answers reads answers assigned to test_out; mandatory check mode takes none")
        return True

    def _check_images_enabled(self) -> bool:
        """Whether failed-check text is accompanied by grid images."""
        self._images_enabled()
        if self.images == "task+show+check" and not self.check_mode.enabled:
            raise ValueError("images='task+show+check' requires check_mode ENABLED or MANDATORY")
        return self.images == "task+show+check"

    def _check_mode(self) -> CheckMode:
        """The validated oracle/submission mode."""
        return CheckMode.validated(cast(object, self.check_mode))

    def _max_shows(self) -> int:
        """The validated per-tool-call ``show`` limit."""
        if self.max_shows < 0:
            raise ValueError(f"max_shows must be non-negative; got {self.max_shows}")
        return self.max_shows

    def _image_scale(self) -> int:
        """The validated nearest-neighbour scale for every task/tool image."""
        if self.image_scale < 1:
            raise ValueError(f"image_scale must be at least 1; got {self.image_scale}")
        return self.image_scale

    # ------------------------------------------------------------------
    # 2. the spine
    # ------------------------------------------------------------------

    async def _do_move(self, ctx: LLMMoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Play the whole task: set the kernel up with the puzzle in it, then
        run analyzer entries until the answer is complete.

        The only other exits are give-ups: on the token budget, or on an
        empty reply when ``terminate_on_empty_reply`` is set. Whatever was
        submitted along the way stands either way."""
        self._reply_reserve_tokens()  # validates static context config before any kernel work
        self._per_turn_cap(ctx)  # raises when the budget is spent, before any kernel work
        layout = None
        if self.use_dihedral_and_order_permutation:
            layout = self._take_layout
            if layout is None:
                # This tag seeds each game's layout; changing it changes every recorded seed's layout.
                rng = random.Random(f"lr_reflect_and_order_permutation:{ctx.rng.getstate()!r}")
                train_order = list(range(len(view.given)))
                test_order = list(range(len(view.questions)))
                rng.shuffle(train_order)
                rng.shuffle(test_order)
                layout = PuzzleLayout(
                    transform=LR_REFLECTION, train_order=tuple(train_order), test_order=tuple(test_order)
                )
            transform = layout.transform
            view = taaf.game.GameView(
                game_id=view.game_id,
                given=tuple(
                    (_dihedral(view.given[i][0], transform), _dihedral(view.given[i][1], transform))
                    for i in layout.train_order
                ),
                questions=tuple(_dihedral(view.questions[i], transform) for i in layout.test_order),
            )
        self._layout = layout
        try:
            repl = self._ensure_repl(ctx)
            await repl.setup_task(view)
            self._n_questions = len(view.questions)
            self._budget_warned = False
            repl.budget_note = lambda: self._budget_note(ctx)
            if self.check_mode is CheckMode.MANDATORY:
                repl.on_answer = None
                repl.on_checked_answer = lambda grids: self._absorb_checked_answer(ctx, view, grids)
            else:
                repl.on_checked_answer = None
                repl.on_answer = lambda payload: self._absorb_answer(ctx, view, payload)
            self._answers_shown = ()
            repl.answer_feedback = (
                (lambda: self._changed_answers_content(view)) if self._show_changed_answers() else None
            )
            with ctx.diag.span("duck/analyze") as analyze:
                if layout is not None:
                    analyze.kv(
                        reflection="lr" if layout.transform >= 4 else "none",
                        rotation_quarter_turns=layout.transform % 4,
                        train_order=json.dumps(layout.train_order),
                        test_order=json.dumps(layout.test_order),
                    )
                entries = nudges = total_calls = 0
                while (
                    not self._answered(view) or self._pending_tool_feedback or self._audit_due(ctx, view, total_calls)
                ):
                    # Between entries as well as before the first: a game that has
                    # not answered yet keeps re-entering. Queued grid feedback gets
                    # one following model response even if the call that produced
                    # it also completed the answer. A game that would stop here
                    # with its answer standing is audited once first.
                    self._per_turn_cap(ctx)  # raises when the budget is spent
                    remaining_game_calls = self._summary_request_backstop_remaining(total_calls)
                    opening = None
                    if self._answered(view) and not self._pending_tool_feedback:
                        opening = [self._fire_audit(ctx, view)]
                    entries += 1
                    calls_used, entry_nudges = await self._run_analyzer_entry(
                        ctx,
                        view,
                        remaining_game_calls=remaining_game_calls,
                        opening=opening,
                    )
                    total_calls += calls_used
                    nudges += entry_nudges
                if self._audit_quote is not None:
                    self._confidence = self._audit_confidence()
                    analyze.kv(audit_confidence=self._confidence)
                analyze.kv(
                    context_window=self.context_window_tokens,
                    entries=entries,
                    nudges=nudges,
                    llm_calls=total_calls,
                    # Cumulative, so a run whose counts were never verified is
                    # visible as a flat 0 rather than passing quietly.
                    counts_reconciled=self.counts_reconciled,
                    submissions=self.submissions_made,
                    placeholders=self.reading.n_placeholders(len(view.questions)),
                )
                analyze.set_outcome(f"{self.submissions_made} submission(s)")
            await self._persist_history()
            return None
        finally:
            self._live_conversation = None
            self._live_backend = None
            self._live_params = None
            self._status_handle = None

    # ------------------------------------------------------------------
    # 3. the analyzer loop
    #
    # The original's ToolAgent.analyze, re-entered until the answer is
    # complete. Three nested things, outermost first: analyzer ENTRIES
    # (each with a freshly built state prompt), BLOCKS within an entry (a
    # user prompt and the response turn answering it), and — inside ullm's
    # TurnManager — the LLM/tool ROUNDS of one block.
    # ------------------------------------------------------------------

    def _sampling(self, ctx: LLMMoveContext) -> ullm.backend.GenericSamplingParameters:
        """Profile + duck default: reasoning is sent back for ALL turns
        (the original ran vLLM with ``preserve_thinking: true``). A profile
        may still pin a different send_back_reasoning explicitly. Once the
        audit has fired, the block samples on ``audit_profile`` when set."""
        name = self.profile
        if self._audit_quote is not None and self.audit_profile is not None:
            name = self.audit_profile
        params = self.resolve_sampling(ctx, name)
        if params.max_tokens is not None:
            raise ValueError(f"sampling profile {name!r} must leave max_tokens unset; set reply_cap_tokens instead")
        params.max_tokens = self.reply_cap_tokens
        if params.send_back_reasoning is None:
            params.send_back_reasoning = "all"
        return params

    def _answered(self, view: taaf.game.GameView) -> bool:
        """Whether this game's decision has been made: every question has an
        answer the model actually produced. Placeholders never satisfy it."""
        return self.reading.is_complete(len(view.questions))

    # ------------------------------------------------------------------
    # the token budget
    #
    # A stop has to take hold at the next model call rather than at the
    # next entry: one analyzer entry can run many calls.
    # ------------------------------------------------------------------

    def _game_tokens_used(self, ctx: LLMMoveContext) -> int:
        """Completion tokens billed to this game's private backend."""
        return ctx.backend.total_usage.completion_tokens

    async def _offer_slot(self, ctx: LLMMoveContext) -> None:
        """Offer this game's concurrency slot back, at a round boundary.

        Offered on every round rather than at chosen moments, because the
        two things that decide whether an offer is worth taking up belong
        to different owners: whether a swap is free right now is the
        agent's fact (a trim just fired, so the prefix is gone anyway),
        and how long is too long to hold a slot is the run's — a clock,
        which agents do not read. So this reports both the swap point and
        the spend, and the host decides.

        The spend reported is live to the token: it is the backend's own
        meter, which every completed round has already been billed to, so
        a game mid-turn is priced at what it has generated rather than at
        its last turn's total.

        From the first compaction after the audit fired, the offer also asks
        to rank ``audit_handicap_tokens`` further back, where that is set:
        the compaction is the free point the rest of the audit is offered at.
        """
        if self._audit_quote is not None and self._last_trim_fired:
            self._audit_compacted = True
        hints: dict[str, Any] = {"tokens": self._game_tokens_used(ctx), "compacted": self._last_trim_fired}
        if self._audit_compacted and self.audit_handicap_tokens:
            hints[HANDICAP_HINT] = self.audit_handicap_tokens
        await ctx.yield_slot(hints)

    def _per_turn_cap(self, ctx: LLMMoveContext) -> int | None:
        """The generation ceiling for a turn about to run: what is left of the
        game allowance, or ``None`` when unbounded.

        **Gives up rather than ever returning zero**, which is the whole reason
        every ceiling is routed through here. A turn handed a ceiling of zero
        generates nothing: the manager appends its synthetic placeholder, the
        placeholder carries no tool call and no usage, so the entry nudges into
        another identical turn — and since nothing was generated, the spend
        never advances to the point that would end it. That loop cannot be
        broken from inside, so the only safe ceiling is a positive one.

        Called for the check alone wherever a turn is merely imminent (before
        the kernel is set up, and between entries): an attempt that cannot
        afford a call should not pay for arranging one either."""
        if self.max_tokens_per_game is None:
            return None
        game_used = self._game_tokens_used(ctx)
        game_remaining = self.max_tokens_per_game - game_used
        if game_remaining <= 0:
            raise GiveUpError(f"per-game token budget exhausted ({game_used}/{self.max_tokens_per_game})")
        return game_remaining

    def _charge_budget(self, ctx: LLMMoveContext, game_tokens_before: int) -> None:
        """Report one turn's generation against the game budget.

        Completion tokens only, as upstream charges them, so one extraction
        reads both agents. The budget uses the backend's per-game meter,
        including any billed round no longer retained in the conversation."""
        completion = self._game_tokens_used(ctx) - game_tokens_before
        if completion < 0:
            raise RuntimeError("backend completion-token meter moved backwards within a turn")
        values: dict[str, Any] = {"tokens_added": completion}
        if self.max_tokens_per_game is not None:
            values.update(
                tokens_used_this_game=self._game_tokens_used(ctx),
                max_tokens_per_game=self.max_tokens_per_game,
            )
        platypus.diagnostics.current_span().event("budget", **values)

    def _budget_note(self, ctx: LLMMoveContext) -> str | None:
        """Compose the standing status and independent one-shot warning.

        Both read the game's billed completion-token meter after the model
        call, so the tool result describes the allowance currently left.
        """
        used = self._game_tokens_used(ctx)
        surface = self._prompts()
        modes = self._prompt_modes()
        notes = [surface.budget_status(used, self.max_tokens_per_game, modes)]
        if self.budget_warning_tokens is not None and self.max_tokens_per_game is not None and not self._budget_warned:
            remaining = self.max_tokens_per_game - used
            if remaining <= self.budget_warning_tokens:
                self._budget_warned = True
                notes.append(surface.budget_warning(max(remaining, 0), modes))
        return "\n".join(note for note in notes if note) or None

    def _summary_request_backstop_remaining(self, calls_used: int) -> int | None:
        """Game-wide request allowance for the summarize arm."""
        if not self._uses_context_summary():
            return None
        remaining = _SUMMARY_REQUEST_BACKSTOP - calls_used
        if remaining <= 0:
            raise GiveUpError(f"request backstop exhausted ({calls_used}/{_SUMMARY_REQUEST_BACKSTOP})")
        return remaining

    async def _run_analyzer_entry(
        self,
        ctx: LLMMoveContext,
        view: taaf.game.GameView,
        *,
        remaining_game_calls: int | None = None,
        opening: list[ullm.conv.UserPrompt] | None = None,
    ) -> tuple[int, int]:
        """One analyzer entry: LLM turn blocks against one state prompt,
        until the answer is complete or ``max_llm_calls_per_move`` is spent.
        Returns ``(llm calls made, nudges sent)``.

        ``opening``, when given, replaces the state prompt and runs even with
        the answer complete: it is the audit turn, asked of a game that would
        otherwise have stopped.

        A response with no tool call gets the original's "you have not acted
        yet" nudge, which opens a new block within this same entry. A
        response that DID call tools but still hasn't answered means the
        rounds ran out or the turn truncated mid-tools; that ends the entry,
        and the caller re-enters with a freshly built state prompt.

        A response the reply cap cut off is not the model stopping, so it is
        nudged even when the answer is complete; only a response that ended
        on its own ends a game whose answer stands."""
        configured_cap = self.max_llm_calls_per_move if self.max_llm_calls_per_move > 0 else None
        cap = remaining_game_calls if configured_cap is None else configured_cap
        if remaining_game_calls is not None and configured_cap is not None:
            cap = min(remaining_game_calls, configured_cap)
        calls_used = nudges = 0
        deferred_feedback = self._pending_tool_feedback or None
        feedback_entry = deferred_feedback is not None
        next_prompts = self._build_state_prompts(view) if opening is None else opening
        must_run = feedback_entry or opening is not None
        while (must_run or not self._answered(view)) and (cap is None or calls_used < cap):
            must_run = False
            calls_used += await self._run_turn_block(
                ctx,
                next_prompts,
                remaining_calls=None if cap is None else cap - calls_used,
                deferred_feedback=deferred_feedback,
            )
            deferred_feedback = None
            if self.terminate_on_empty_reply and self._last_turn_was_empty_reply():
                raise GiveUpError("empty reply: the model ended its turn with no text and no tool call")
            if self._pending_tool_feedback or self._last_turn_had_tool_calls():
                break
            answered = self._answered(view)
            if answered and self._last_completion_limit() is None:
                break
            if cap is not None and calls_used >= cap:
                break
            nudges += 1
            next_prompts = [self._followup_prompt()]
            must_run = answered
        return calls_used, nudges

    def _turn_tools(self, ctx: LLMMoveContext | None = None) -> list[ullm.tool.Tool]:
        """Apply the agent's execution policy after all tools are assembled."""
        tools = self._build_turn_tools(ctx)
        for tool in tools:
            tool.auto_correct_arguments = self.auto_correct_tool_arguments
        return tools

    def _build_turn_tools(self, ctx: LLMMoveContext | None = None) -> list[ullm.tool.Tool]:
        """Build the configured tools shared by requests and token estimates.

        The order is prompt text. The python tool leads and the file tools
        follow it, because they act on the same working directory and are
        read together."""
        tools: list[ullm.tool.Tool] = [cast(ullm.tool.Tool, self._ensure_repl(ctx))]
        if self.file_tools == "full":
            tools.extend(
                build_file_tools(
                    self._ensure_workspace(ctx),
                    search_files=self.expose_tool_search_files,
                    undo_edit=self.expose_tool_undo_edit,
                )
            )
        return tools

    def _ensure_workspace(self, ctx: LLMMoveContext | None = None) -> Workspace:
        """The workspace the file tools act on, built once and kept."""
        if self.workspace is None:
            self.workspace = Workspace(repl=self._ensure_repl(ctx))
        return self.workspace

    def _file_tool_names(self) -> tuple[str, ...]:
        """The file tools this agent offers, for the prompt that names them.

        Same gates the builder reads, so the sentence and the tool list cannot
        describe different suites."""
        if self.file_tools == "none":
            return ()
        return build_file_tool_names(search_files=self.expose_tool_search_files, undo_edit=self.expose_tool_undo_edit)

    async def _run_turn_block(
        self,
        ctx: LLMMoveContext,
        user_prompts: list[ullm.conv.UserPrompt],
        *,
        remaining_calls: int | None,
        deferred_feedback: list[list[ullm.conv.Content]] | None = None,
    ) -> int:
        """One [UserPrompt, ResponseTurn] block, and the widest seam onto
        the LLM layer. Returns the number of LLM calls made.

        ullm owns the loop: ``ResponseManager`` runs a single call with the
        malformed-output retry and prompt-too-long defenses behind it, and
        ``TurnManager`` runs rounds of (call, execute the tool calls, feed
        the results back) until the model stops asking for tools. The duck
        supplies the pieces that are its own business — the conversation,
        the tool, and a per-round hook that trims to the context budget
        (``before_round``).

        The conversation is built fresh around ``self.messages`` each block
        rather than kept, because the system prompt and the tool schema
        follow configuration and the element list is edited in place by
        eviction; ``self.messages`` IS the live list the manager appends
        to."""
        tools = self._turn_tools(ctx)
        repl = cast(DuckPythonTool, tools[0])
        self.messages.extend(user_prompts)
        conversation = ullm.conv.Conversation(
            system_prompt=ullm.conv.SystemPrompt(
                content=ullm.conv.StringContent(text=self._system_prompt_text()),
                # Set explicitly so None ("no thinking") is honoured;
                # SystemPrompt's own default is "high".
                reasoning_effort=self.reasoning_effort,
                tools=tools,
            ),
            elements=self.messages,
            status="intermediate",
            history_epoch=self.history_epoch,
            conversation_id=self._conversation_id(),
        )
        # One sampling resolve (one rng seed draw) per block.
        sampling_params = self._sampling(ctx)
        summarizing = self._uses_context_summary()
        response_manager = _FirstReplyCappedResponseManager(
            backend=ctx.backend,
            conversation=conversation,
            sampling_params=sampling_params,
            max_retries=max(0, int(self.malformed_retries)),
            perturb_retry_prompt=self.perturb_retry_prompt,
            preserve_failed_draws=self.preserve_failed_draws,
            before_attempt=self._observe_request_attempt if self.context_observer is not None else None,
            # Only a block that opens the game gets the cap: once anything has
            # been generated the first reply is behind us, whatever block this is.
            first_reply_max_tokens=(self.first_response_max_tokens if self._game_tokens_used(ctx) == 0 else None),
        )
        # DiagnosticTurnManager: same TurnManager semantics, plus the
        # llm/turn span (per-response events, kernel/exec events drained
        # from the repl, and the FULL live conversation — pre-history
        # included — as a block-deduped artifact). Inert on the null
        # stream, so replay references are unaffected.
        # What is left of the game's allowance becomes this turn's ceiling,
        # so one turn cannot overshoot the budget by a whole turn's worth of
        # generation; ullm lowers each round's max_tokens against it, which
        # is what makes the stop land at the next call rather than the next
        # entry.
        per_turn_cap = self._per_turn_cap(ctx)
        game_tokens_before = self._game_tokens_used(ctx)
        ordinary_params = cast(ullm.backend.GenericSamplingParameters, response_manager.sampling_params)
        ordinary_max_tokens = ordinary_params.max_tokens
        compaction_calls = 0
        ordinary_calls = 0
        summaries_before = self.context_summaries

        def summary_call_limit() -> int | None:
            """Compaction attempts still available in this analyzer entry."""
            if remaining_calls is None:
                return None
            return max(0, remaining_calls - compaction_calls - ordinary_calls)

        def call_allowance_exhausted() -> bool:
            """Whether summary and analyzer responses spent this entry's cap."""
            if remaining_calls is None:
                return False
            return compaction_calls + ordinary_calls >= remaining_calls

        async def before_round() -> None:
            nonlocal compaction_calls, ordinary_calls
            # Counted and armed here, on both compaction arms alike: this hook
            # is the one place every ordinary call passes through, and the
            # prompt is measured after it on either arm, so a token-native
            # count weighs the injected ids too.
            self._prepare_context_observer()
            self.response_step += 1
            self._arm_prompt_injection()
            if summarizing:
                if call_allowance_exhausted():
                    raise _CallAllowanceExhausted
                summaries_seen = self.context_summaries
                try:
                    added, measured = await self._prepare_summarized_request(
                        ctx,
                        ordinary_max_tokens,
                        summary_call_limit=summary_call_limit(),
                    )
                except _CallAllowanceExhausted as exc:
                    compaction_calls += exc.calls_made
                    raise
                compaction_calls += added
                if call_allowance_exhausted():
                    raise _CallAllowanceExhausted
                if self.context_summaries > summaries_seen:
                    # A compaction runs alone, and re-arms the lengths above what it left.
                    self._rearm_intermediate_summaries(measured)
                elif self._intermediate_summary_due(measured):
                    raise _IntermediateSummaryDue
                self._pending_bounds.append(measured)
                self._arm_zero_recurrent(int(measured))
                self._record_prompt_injection_epoch()
            else:
                await self._trim_live_messages()
            await self._close_ended_thread(ctx, conversation.history_epoch)
            # After the trim on either arm, so the offer says whether the
            # prefix this round would have reused is already gone.
            await self._offer_slot(ctx)
            # Every call, not only the game's first: the request gap protects
            # a pool at its admission line, so it follows the run's slot tally
            # through a game that outlives the busy part of the run.
            platypus.solver.pace_while_busy(ctx, self.run_config, self.run_config.min_request_gap_s)
            ordinary_calls += 1

        def discard_unsent_empty_turn() -> bool:
            """Remove a call-less turn and prompts not retained by a summary."""
            nonlocal deferred_feedback
            tail = self.messages[-1] if self.messages else None
            if not isinstance(tail, ullm.conv.ResponseTurn) or tail.elements:
                return False
            self._mark_history_rewritten()
            self.messages.pop()
            installed = self.context_summaries > summaries_before
            pinned = self.messages[0] if installed and self.messages else None
            self.messages[:] = [
                element
                for element in self.messages
                if element is pinned or not any(element is prompt for prompt in user_prompts)
            ]
            self._pending_bounds.clear()
            if deferred_feedback is not None:
                self._pending_tool_feedback = [*deferred_feedback, *self._pending_tool_feedback]
                deferred_feedback = None
            return True

        turn_manager = platypus.llm.DiagnosticTurnManager(
            response_manager=response_manager,
            conversation=conversation,
            max_tool_rounds=remaining_calls if remaining_calls is not None else 10**9,
            before_round=before_round,
            stop_when=(
                self._stop_after_tool_feedback
                if self.check_mode.enabled or self._show_enabled() or self._show_changed_answers()
                else None
            ),
            # Summary calls happen inside ``before_round`` and therefore sit
            # outside ullm's local turn counter. The summarize hook instead
            # lowers each ordinary call against the live backend meter.
            max_generated_tokens_per_turn=None if summarizing else per_turn_cap,
            diag=ctx.diag,
            label="analyzer",
            classify_tool_calls=self.classify_tool_calls,
            after_response=self.response_observer,
        )
        # The trims declare rewrites on the live conversation; the
        # backend/params pair is what token-native measurement weighs the
        # next request with. Deliberately NOT cleared in a finally here: the
        # post-turn retention trim runs after the analyzer loop and must
        # measure the same way and declare on the same epoch counter.
        # _do_move clears it, still before the game ends, so nothing live
        # rides a snapshot.
        self._live_conversation = conversation
        self._live_backend = ctx.backend
        self._live_params = response_manager.sampling_params
        self._status_handle = ctx.status
        self._pending_bounds.clear()
        self._pruned_rounds = 0  # the note counts this turn's losses
        allowance_stopped_empty = False
        malformed_stopped_empty = False
        malformed_nudge: list[ullm.conv.Content] | None = None
        # The turn an intermediate summary stopped, None where it stopped none
        # or the turn had no round yet.
        intermediate_due = False
        stopped_turn: ullm.conv.ResponseTurn | None = None
        try:
            try:
                with ullm.conv.observe_history_rewrites(self._observe_history_rewrite):
                    await turn_manager.run()
            except ullm.backend.MalformedOutputError as error:
                if not self._nudges_past(error):
                    if not self.preserve_failed_draws:
                        discard_unsent_empty_turn()
                    raise
                # Every draw of the round was refused and rolled back, so the
                # round has no response: its pre-call bound goes too. The
                # prompts that opened the turn were sent and stay; an empty
                # turn is not a durable element, a nonempty one holds the
                # completed rounds before this one. Nothing of the refused
                # draws became a step, so the nudge extends what was served.
                self._malformed_nudged += 1
                if self._pending_bounds:
                    self._pending_bounds.pop()
                turn = self.messages[-1]
                assert isinstance(turn, ullm.conv.ResponseTurn)
                if turn.elements:
                    turn.status = "processed"
                else:
                    self.messages.pop()
                    malformed_stopped_empty = True
                text = self._prompts().malformed_nudge(self._prompt_modes(), self.reading.problems(self._n_questions))
                malformed_nudge = [ullm.conv.StringContent(text=text)]
                platypus.diagnostics.current_span().event(
                    "malformed_nudge", n=self._malformed_nudged, error=str(error)[:300]
                )
            except _IntermediateSummaryDue:
                # Raised before the round's call went out. The rounds the turn
                # completed are history; an empty turn is not, and the prompts
                # that opened it go out with the summary request instead.
                intermediate_due = True
                turn = self.messages[-1]
                assert isinstance(turn, ullm.conv.ResponseTurn)
                if turn.elements:
                    turn.status = "processed"
                    stopped_turn = turn
                else:
                    self.messages.pop()
            except _CallAllowanceExhausted:
                # before_round can spend the last call on compaction. An
                # empty turn is not a valid durable element and its ordinary
                # prompt was never sent; a nonempty turn contains earlier
                # completed rounds and remains valid history.
                allowance_stopped_empty = discard_unsent_empty_turn()
                if not allowance_stopped_empty:
                    turn = self.messages[-1]
                    assert isinstance(turn, ullm.conv.ResponseTurn)
                    turn.status = "processed"
            except Exception:
                if not self.preserve_failed_draws:
                    discard_unsent_empty_turn()
                raise
            feedback = repl.take_feedback()
            if not (self.check_mode is CheckMode.MANDATORY and repl.has_checked_answer):
                self._pending_tool_feedback.extend(feedback)
            if malformed_nudge is not None:
                # After the completed rounds' own feedback: it is the last thing
                # the model is told before it goes on.
                self._pending_tool_feedback.append(malformed_nudge)
            # Host-call failures are duck bugs, not model-visible tool errors —
            # fail loudly (TurnManager swallows tool exceptions into results).
            pending_host_exception = repl.pending_host_exception
            if pending_host_exception is not None:
                raise pending_host_exception
        finally:
            # Declarations the turn manager made on the live conversation
            # (retry rollbacks) outlive this entry's Conversation.
            self.history_epoch = max(self.history_epoch, conversation.history_epoch)
            # Summary-only failures have paid the backend too. Publish that
            # spend even when no ordinary analyzer response survives.
            self._charge_budget(ctx, game_tokens_before)
        if allowance_stopped_empty or malformed_stopped_empty:
            assert not self._pending_bounds
            return compaction_calls + ordinary_calls
        if intermediate_due:
            if stopped_turn is not None:
                self.counts_reconciled += self._check_engine_usage(stopped_turn)
            else:
                assert not self._pending_bounds
            spent = compaction_calls + ordinary_calls
            limit = None if remaining_calls is None else max(0, remaining_calls - spent)
            return spent + await self._write_intermediate_summary(ctx, call_limit=limit)
        turn = self.messages[-1]
        assert isinstance(turn, ullm.conv.ResponseTurn), "TurnManager must append its ResponseTurn to our history"
        self.counts_reconciled += self._check_engine_usage(turn)
        return compaction_calls + ordinary_calls

    def _intermediate_summary_due(self, measured: int) -> bool:
        """Whether the request about to go out, ``measured`` tokens, has passed a
        length of ``intermediate_summary_prompt_lengths`` this compaction cycle
        has not: then every length it passed counts as asked, for one summary."""
        passed = [n for n in self.intermediate_summary_prompt_lengths if measured > n]
        new = [n for n in passed if n not in self._intermediate_lengths_passed]
        if not new:
            return False
        self._intermediate_lengths_passed = tuple(sorted({*self._intermediate_lengths_passed, *passed}))
        return True

    def _rearm_intermediate_summaries(self, measured: int) -> None:
        """A compaction starts a cycle: every length is asked again, but those
        the rebuilt request, ``measured`` tokens, already passes count as asked,
        so no summary follows the compaction straight away."""
        self._intermediate_lengths_passed = tuple(n for n in self.intermediate_summary_prompt_lengths if measured > n)
        self._intermediate_summaries_in_cycle = 0

    async def _write_intermediate_summary(self, ctx: LLMMoveContext, *, call_limit: int | None) -> int:
        """The summary turn: the feedback still queued, the request, and the
        model's summary, all kept in the history as served, then the message to
        go on queued for the next block. Returns the calls it made.

        No usable summary on either attempt costs the game nothing more: the
        refused replies are already declared rewrites, and the game goes on."""
        groups = self._pending_tool_feedback
        self._pending_tool_feedback = []
        self.messages.extend(ullm.conv.UserPrompt(content=ullm.conv.ListContent(elements=group)) for group in groups)
        prompts = self._prompts()
        instruction = ullm.conv.UserPrompt(
            content=ullm.conv.StringContent(text=prompts.intermediate_summary_prompt(self._prompt_modes()))
        )
        tokens_before = self._game_tokens_used(ctx)
        passed = self._intermediate_lengths_passed
        try:
            written = await self._write_summary(
                ctx,
                instruction,
                # The play's own reply cap: the request names no size, and the
                # compaction ratio, which sizes a compaction's, does not apply.
                cap=self._reply_reserve_tokens(),
                reasoning_cap=self._reply_reserve_tokens(),
                call_limit=call_limit,
                label="intermediate_summary",
                occasion="an intermediate summary",
                failed_as="intermediate summary",
            )
        except _CallAllowanceExhausted as exc:
            return exc.calls_made
        except _SummaryUnusable as exc:
            # Both attempts were made and refused.
            platypus.diagnostics.current_span().event("intermediate_summary_failed", error=str(exc)[:300])
            return 2
        finally:
            self._charge_budget(ctx, tokens_before)
        self.messages.append(instruction)
        self.messages.append(ullm.conv.ResponseTurn(elements=[written.response], status="processed"))
        self._intermediate_summaries_in_cycle += 1
        self._pending_tool_feedback.append([ullm.conv.StringContent(text=prompts.intermediate_summary_continue)])
        usage = written.response.token_usage
        platypus.diagnostics.current_span().event(
            "intermediate_summary",
            # The highest length this cycle has passed: the one that asked.
            length=max(passed),
            n_in_cycle=self._intermediate_summaries_in_cycle,
            summary_tokens=usage.completion_tokens if usage is not None else None,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            attempts=written.attempts,
            reasoning_retry=written.reasoning_retry,
        )
        return written.attempts

    def _nudges_past(self, error: ullm.backend.MalformedOutputError) -> bool:
        """Whether a play reply refused on every re-sample is nudged rather than
        raised: under ``malformed_nudges``, with nudges left this game, for the
        refusals the nudge describes -- no text and no tool call (ullm's fixed
        message), or degenerate repetition. A parser failure is not one."""
        if self._malformed_nudged >= self.malformed_nudges:
            return False
        return isinstance(error, ullm.backend.DegenerateOutputError) or str(error).startswith(
            "Model produced no final content"
        )

    def _stop_after_tool_feedback(
        self,
        _response: ullm.conv.LLMResponse,
        result: ullm.conv.ToolResult,
    ) -> str | None:
        """End this turn after mandatory submission or queued grid feedback."""
        if result.tool_name == "python" and self.repl is not None:
            if self.repl.has_checked_answer:
                return "remaining calls skipped: passing check submitted the mandatory answer"
            if self.repl.has_feedback:
                return "remaining calls skipped: Python queued follow-up grid feedback"
        return None

    def _last_turn_had_tool_calls(self) -> bool:
        """Whether the newest response turn ended with tool calls (rounds
        exhausted / truncation) rather than a plain no-tool-call response
        (which gets the nudge)."""
        turn = self.messages[-1]
        if not isinstance(turn, ullm.conv.ResponseTurn):
            return False
        for element in reversed(turn.elements):
            if isinstance(element, ullm.conv.LLMResponse):
                return bool(element.final_content.tool_calls())
        return False

    def _last_turn_was_empty_reply(self) -> bool:
        """Whether the newest response turn ended with a reply the backend kept
        as malformed for carrying no text and no tool call. The manager's
        feedback prompt may follow it in the turn; the reply itself decides."""
        turn = self.messages[-1] if self.messages else None
        if not isinstance(turn, ullm.conv.ResponseTurn):
            return False
        for element in reversed(turn.elements):
            if isinstance(element, ullm.conv.LLMResponse):
                return element.stop_reason == "malformed" and ullm.backend._is_empty_content(element.final_content)
        return False

    def _last_completion_limit(self) -> int | None:
        """Generated-token limit when the latest response ended by hitting it."""
        turn = self.messages[-1] if self.messages else None
        if not isinstance(turn, ullm.conv.ResponseTurn):
            return None
        for element in reversed(turn.elements):
            if not isinstance(element, ullm.conv.LLMResponse):
                continue
            usage = element.token_usage
            if element.stop_reason == "max_tokens" and usage is not None and usage.completion_tokens > 0:
                return usage.completion_tokens
            return None
        return None

    def _followup_prompt(self) -> ullm.conv.UserPrompt:
        """Build a new continuation prompt without modifying any prior message.

        The agent gates disclosure of the reply cap; the surface composes
        the permitted facts and the answer reading's problems."""
        text = self._prompts().followup_nudge(
            self._prompt_modes(),
            self._last_completion_limit() if self.completion_cutoff_notice else None,
            self.reading.problems(self._n_questions),
        )
        return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))

    # ------------------------------------------------------------------
    # 4. the answer
    #
    # Harvested after every call the model makes, from test_out ordinarily
    # or a passing checked function in mandatory mode. Submitting returns
    # nothing, so this cannot reveal correctness to the model.
    # ------------------------------------------------------------------

    def _absorb_answer(self, ctx: LLMMoveContext, view: taaf.game.GameView, payload: dict[str, object] | None) -> None:
        """Take one answer reading and submit the best answer available.

        Not only complete ones: a question the model has not reached is
        filled with its own input grid, so partial work still scores. What
        ends the game is ``is_complete``, which placeholders never satisfy.

        Called from inside the tool, between the model's code running and its
        result being handed back — so the standing answer is up to date
        before the model's next call, and a run killed at a deadline finds
        the best answer the model had rather than nothing.

        Identical answers are not resubmitted: each submission is a record
        with a cost attached, and a model that leaves a correct ``test_out``
        alone across twenty calls should not read as twenty answers. Which
        grids the model wrote is part of the answer — the same grids with a
        placeholder slot now written by the model are a different answer to
        an agent composing from this one — so it goes with the submission and
        into the comparison.
        """
        self.reading = reading_from_payload(payload)
        submission = self.reading.submission(view.questions)
        if submission is None:
            return
        produced = self.reading.produced()
        # Readings, prompts and kernel feedback stay in the model's coordinates.
        layout = self._layout
        if layout is not None:
            original_order = sorted(range(len(layout.test_order)), key=layout.test_order.__getitem__)
            submission = taaf.game.Submission(
                attempts=tuple(
                    tuple(_dihedral(grid, layout.transform, inverse=True) for grid in submission.attempts[i])
                    for i in original_order
                )
            )
            produced = tuple(produced[i] for i in original_order)
        fingerprint = (submission.to_json_list(), produced)
        if fingerprint == self._last_submitted:
            return
        self._last_submitted = fingerprint
        self.submissions_made += 1
        ctx.submit(submission, produced=produced)
        # Placeholders are recorded on every answer, so a submission that is
        # mostly filler is visible as such in the trace rather than reading
        # as the model having answered everything.
        platypus.diagnostics.current_span().event(
            "answer",
            n=self.submissions_made,
            placeholders=self.reading.n_placeholders(len(view.questions)),
        )

    def _absorb_checked_answer(
        self,
        ctx: LLMMoveContext,
        view: taaf.game.GameView,
        grids: list[list[list[int]]],
    ) -> None:
        """Submit one mandatory-mode candidate produced by a passing check."""
        payload: dict[str, object] = {
            "bound": True,
            "rows": [[{"state": "ok", "grid": grid}] for grid in grids],
        }
        self._absorb_answer(ctx, view, payload)

    # ------------------------------------------------------------------
    # 5. prompts
    #
    # Assembly lives on prompts.DuckPrompts: the agent computes the facts
    # and the surface turns them into text. _prompt_modes() is the one site
    # where agent fields become render modes, so the two cannot be set apart.
    # ------------------------------------------------------------------

    def _prompts(self) -> Prompts:
        """The carried prompt implementation — wording, and nothing else."""
        return self.prompts

    def _mark_history_rewritten(self) -> None:
        """Declare a rewrite of sent history (``Conversation.mark_history_rewritten``),
        on the persistent counter and on the live conversation when one is up.
        The live conversation may be ahead (the turn manager declares its own
        rollbacks), so the counter moves past whichever is newer."""
        conversation = self._live_conversation
        if conversation is not None:
            conversation.history_epoch = max(self.history_epoch, conversation.history_epoch)
            with ullm.conv.observe_history_rewrites(self._observe_history_rewrite):
                conversation.mark_history_rewritten()
            self.history_epoch = conversation.history_epoch
        else:
            if self.context_observer is not None:
                self.context_observer.history_rewritten(epoch=self.history_epoch)
            self.history_epoch += 1
        if self.prompt_injection_schedule == "context":
            self._arm_prompt_injection()

    def _observe_history_rewrite(self, conversation: ullm.conv.Conversation) -> None:
        """Forward manager and agent declarations for this game's live history."""
        if conversation is self._live_conversation and self.context_observer is not None:
            self.context_observer.history_rewritten(epoch=conversation.history_epoch)

    def _prompt_modes(self) -> prompts_module.PromptModes:
        """The numbers a render may quote, taken from whatever enforces them.

        Read here rather than stored on the prompt surface, so a prompt
        cannot go on naming a timeout the kernel does not apply. The grid
        facts are TAAF's, derived rather than copied, so a prompt cannot keep
        telling the model old numbers if they ever moved."""
        return prompts_module.PromptModes(
            tool_output_tokens=self.tool_output_tokens,
            python_timeout_s=self._python_timeout_s(),
            n_attempts=taaf.game.N_ATTEMPTS,
            max_side=taaf.grid.MAX_SIDE,
            n_colors=taaf.grid.N_COLORS,
            check_mode=self.check_mode,
            images_enabled=self._images_enabled(),
            text_grids=self._text_grids_enabled(),
            image_layout=self._image_layout(),
            image_legend=self.image_legend,
            show_enabled=self._show_enabled(),
            max_shows=self._max_shows(),
            check_images_enabled=self._check_images_enabled(),
            context_summary_enabled=self._uses_context_summary(),
            ascii_row_numbers=self.ascii_row_numbers,
            completion_cutoff_notice=self.completion_cutoff_notice,
            check_and_submit=self.check_and_submit,
            game_token_allowance=self.max_tokens_per_game,
            file_tool_names=self._file_tool_names(),
            add_files_and_repl_to_summary=self._uses_context_summary() and self.add_files_and_repl_to_summary,
            show_changed_answers=self._show_changed_answers(),
        )

    def _system_prompt_text(self) -> str:
        """Assemble a prompt whose tool and runtime claims match every mode."""
        return self._prompts().system_prompt(self._prompt_modes())

    def _build_state_prompt(self, view: taaf.game.GameView) -> ullm.conv.UserPrompt:
        """The per-entry state prompt. The first entry is handed the puzzle
        itself; a re-entry is handed what is still wrong with the answer —
        the kernel holds the puzzle, so the pictures are not re-sent."""
        if self._pending_tool_feedback:
            content = ullm.conv.ListContent(elements=self._pending_tool_feedback.pop(0))
            return ullm.conv.UserPrompt(content=content)
        if not any(isinstance(element, ullm.conv.UserPrompt) for element in self.messages):
            return self._task_prompt(view)
        if self.check_mode is CheckMode.MANDATORY:
            return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=self._prompts().mandatory_answer_repair()))
        problems = self.reading.problems(len(view.questions))
        return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=self._prompts().answer_repair(problems)))

    def _build_state_prompts(self, view: taaf.game.GameView) -> list[ullm.conv.UserPrompt]:
        """Build one entry's consecutive user messages.

        ``show`` and failed-``check`` feedback are separate messages in that
        order, matching their distinct model-facing roles while still
        eliciting a single following response.
        """
        if not self._pending_tool_feedback:
            return [self._build_state_prompt(view)]
        groups = self._pending_tool_feedback
        self._pending_tool_feedback = []
        return [ullm.conv.UserPrompt(content=ullm.conv.ListContent(elements=group)) for group in groups]

    def _task_prompt(self, view: taaf.game.GameView) -> ullm.conv.UserPrompt:
        """The puzzle in the configured text/image presentation."""
        prompts = self._prompts()
        modes = self._prompt_modes()
        elements = self._puzzle_pairs(view, intro=prompts.task_intro(modes, n_train=len(view.given)))
        elements.append(ullm.conv.StringContent(text=prompts.test_intro_text(modes, n_test=len(view.questions))))
        for index, question in enumerate(view.questions):
            label = prompts.question_label(index, shape=question.shape)
            elements.extend(self._grid_block(label, question, name=f"test_in[{index}]"))
        elements.append(ullm.conv.StringContent(text=prompts.task_footer_text(modes)))
        return ullm.conv.UserPrompt(content=ullm.conv.ListContent(elements=elements))

    def _grid_rows(self, grid: taaf.grid.Grid) -> str:
        """One grid in this arm's text layout."""
        return kernel_code.grid_text(grid, row_numbers=self._prompt_modes().ascii_row_numbers)

    def _puzzle_pairs(self, view: taaf.game.GameView, *, intro: str) -> list[ullm.conv.Content]:
        """The legend, ``intro`` and the example pairs: the part of the
        puzzle the task prompt and the audit prompt show alike."""
        prompts = self._prompts()
        modes = self._prompt_modes()
        images_enabled = modes.images_enabled
        text_grids = modes.text_grids
        image_layout = modes.image_layout
        rows = self._grid_rows

        # The legend first, before anything task-specific. It is byte-identical
        # in every game, so leading with it extends the shared prefix — the
        # system prompt plus this image — and every game after the first gets
        # it from the KV cache instead of prefilling it again.
        elements: list[ullm.conv.Content] = []
        if images_enabled and self.image_legend:
            elements.extend(
                [
                    ullm.conv.StringContent(text=prompts.palette_caption),
                    ullm.conv.ImageContent(
                        data=palette.legend_array(),
                        alt_text="ARC colour legend: value and name",
                    ),
                ]
            )
        elements.append(ullm.conv.StringContent(text=intro))
        for index, (question, answer) in enumerate(view.given):
            if images_enabled and image_layout == "separate":
                input_label = prompts.train_input_label(index, shape=question.shape)
                output_label = prompts.train_output_label(index, shape=answer.shape)
                elements.extend(self._grid_block(input_label, question, name=f"train_in[{index}]"))
                elements.extend(self._grid_block(output_label, answer, name=f"train_out[{index}]"))
                continue
            label = prompts.pair_label(index, in_shape=question.shape, out_shape=answer.shape)
            elements.append(ullm.conv.StringContent(text=label))
            if text_grids:
                elements.append(
                    ullm.conv.StringContent(
                        text=(f"train_in[{index}]:\n{rows(question)}\n\ntrain_out[{index}]:\n{rows(answer)}")
                    )
                )
            if images_enabled:
                elements.append(
                    images.pair_image(
                        question,
                        answer,
                        scale=self._image_scale(),
                        alt_text=label,
                        gridlines=self.image_gridlines,
                    )
                )
        return elements

    def _grid_block(self, label: str, grid: taaf.grid.Grid, *, name: str) -> list[ullm.conv.Content]:
        """One grid shown on its own under ``label``, in the configured media:
        a test input, or an answer slot in the audit prompt. ``name`` heads
        its digit rows where the label does not."""
        modes = self._prompt_modes()
        if modes.images_enabled and modes.image_layout == "separate":
            return [
                ullm.conv.StringContent(text=(f"{label}\n{self._grid_rows(grid)}" if modes.text_grids else label)),
                _grid_image(grid, scale=self._image_scale(), gridlines=self.image_gridlines, alt_text=label),
            ]
        elements: list[ullm.conv.Content] = [ullm.conv.StringContent(text=label)]
        if modes.text_grids:
            elements.append(ullm.conv.StringContent(text=f"{name}:\n{self._grid_rows(grid)}"))
        if modes.images_enabled:
            elements.append(
                images.question_image(grid, scale=self._image_scale(), alt_text=label, gridlines=self.image_gridlines)
            )
        return elements

    def _changed_answers_content(self, view: taaf.game.GameView) -> list[ullm.conv.Content]:
        """The answers the last call changed, each test input shown before its
        changed slots, and the reading remembered as shown. A slot emptied or
        made unusable shows nothing: the repair prompt names it."""
        prompts = self._prompts()
        rows = self.reading.rows
        content: list[ullm.conv.Content] = []
        for index, row in enumerate(rows[: len(view.questions)]):
            before = self._answers_shown[index] if index < len(self._answers_shown) else ()
            changed = [
                (attempt, slot.grid)
                for attempt, slot in enumerate(row)
                if slot.state == "ok"
                and slot.grid is not None
                and (attempt >= len(before) or before[attempt] is None or before[attempt] != slot.grid)
            ]
            if not changed:
                continue
            question = view.questions[index]
            slots = [f"test_out[{index}][{attempt}]" for attempt, _ in changed]
            content.append(ullm.conv.StringContent(text=prompts.changed_answers_intro(slots)))
            content.extend(
                self._grid_block(
                    prompts.question_label(index, shape=question.shape), question, name=f"test_in[{index}]"
                )
            )
            for attempt, grid in changed:
                label = prompts.answer_slot_label(index, attempt, shape=grid.shape)
                content.extend(self._grid_block(label, grid, name=f"test_out[{index}][{attempt}]"))
        self._answers_shown = tuple(tuple(slot.grid if slot.state == "ok" else None for slot in row) for row in rows)
        return content

    # ------------------------------------------------------------------
    # the audit turn
    # ------------------------------------------------------------------

    def _audit_due(self, ctx: LLMMoveContext, view: taaf.game.GameView, total_calls: int) -> bool:
        """Whether a game about to stop with its answer standing is audited
        first: once a game, and only while the budget and the summarize arm's
        request backstop leave room for a turn, so the audit never turns a
        finished game into a give-up."""
        if not self.audit_turn or self._audit_quote is not None:
            return False
        if not self._answered(view) or self._pending_tool_feedback:
            return False
        if self.max_tokens_per_game is not None and self._game_tokens_used(ctx) >= self.max_tokens_per_game:
            return False
        return not (self._uses_context_summary() and total_calls >= _SUMMARY_REQUEST_BACKSTOP)

    def _fire_audit(self, ctx: LLMMoveContext, view: taaf.game.GameView) -> ullm.conv.UserPrompt:
        """The audit prompt for the answer standing, and the ``audit`` event
        marking the point: the answers submitted and the tokens spent up to
        it are where the game without the audit would have ended."""
        self._audit_quote = self._last_reply_text()
        self._audit_view = view
        platypus.diagnostics.current_span().event(
            "audit",
            submissions=self.submissions_made,
            tokens_used=self._game_tokens_used(ctx),
        )
        return self._audit_prompt(view, self._audit_quote)

    def _last_reply_text(self) -> str:
        """The text of the newest response, without its reasoning or any tool
        call: what the model last said to the user."""
        for element in reversed(self.messages):
            if not isinstance(element, ullm.conv.ResponseTurn):
                continue
            for sub in reversed(element.elements):
                if isinstance(sub, ullm.conv.LLMResponse):
                    return self._reply_text(sub.final_content)
        return ""

    @staticmethod
    def _reply_text(content: ullm.conv.Content) -> str:
        """The plain text of a response's final content, tool calls left out
        and the injected end of thinking stripped."""
        marker = _SUMMARY_THINK_CLOSE.strip()
        if isinstance(content, ullm.conv.ListContent):
            text = "".join(e.text for e in content.elements if isinstance(e, ullm.conv.StringContent))
        elif isinstance(content, ullm.conv.StringContent):
            text = content.text
        else:
            text = ""
        return text.strip().removeprefix(marker).strip()

    def _audit_prompt(self, view: taaf.game.GameView, quote: str) -> ullm.conv.UserPrompt:
        """The puzzle restated with the answer standing in ``test_out``, then
        ``quote`` and the questions. A slot that copies an earlier one, is
        empty or is unusable is said to be, not drawn."""
        prompts = self._prompts()
        modes = self._prompt_modes()
        elements = self._puzzle_pairs(view, intro=prompts.audit_intro(modes, n_train=len(view.given)))
        elements.append(ullm.conv.StringContent(text=prompts.audit_test_intro(n_test=len(view.questions))))
        rows = self.reading.rows
        for index, question in enumerate(view.questions):
            label = prompts.question_label(index, shape=question.shape)
            elements.extend(self._grid_block(label, question, name=f"test_in[{index}]"))
            slots = rows[index] if index < len(rows) else ()
            for attempt in range(modes.n_attempts):
                slot = slots[attempt] if attempt < len(slots) else None
                grid = slot.grid if slot is not None and slot.state == "ok" else None
                if grid is None:
                    why = slot.why if slot is not None and slot.state == "bad" else ""
                    note = prompts.answer_slot_note(index, attempt, why=why)
                    elements.append(ullm.conv.StringContent(text=note))
                    continue
                same_as = next((earlier for earlier in range(attempt) if slots[earlier].grid == grid), None)
                if same_as is not None:
                    note = prompts.answer_slot_note(index, attempt, same_as=same_as)
                    elements.append(ullm.conv.StringContent(text=note))
                    continue
                slot_label = prompts.answer_slot_label(index, attempt, shape=grid.shape)
                elements.extend(self._grid_block(slot_label, grid, name=f"test_out[{index}][{attempt}]"))
        elements.append(ullm.conv.StringContent(text=prompts.audit_close(quote)))
        return ullm.conv.UserPrompt(content=ullm.conv.ListContent(elements=elements))

    def _audit_confidence(self) -> int | None:
        """The confidence the game's last reply gave, read the way the
        prompt asked for it; None where it gave none."""
        return self._prompts().read_audit_confidence(self._last_reply_text())

    def answer_confidence(self) -> int | None:
        """The confidence the audit's final reply stated, read once the play is
        over; None without an audit, before it has answered, or where the
        reply gave no number."""
        return self._confidence

    # ------------------------------------------------------------------
    # 6. context budget
    #
    # The conversation grows without bound, the window does not. Three
    # layers, cheapest first: MEASURE the next prompt (the engine's own
    # billed usage for everything it has already processed, plus a
    # provable bound for what was appended since), EVICT whole user-led
    # blocks from the oldest end, and fall back to dropping the current
    # turn's own oldest rounds.
    #
    # It attaches at the TurnManager's before_round hook, so every LLM
    # call in a turn is measured and trimmed just before it is sent.
    #
    # Why engine truth and not a client-side prediction: no client-side
    # ruler is exact here. Re-tokenizing served text is unstable — a
    # sampled answer need not re-tokenize to its served ids — and no
    # text ruler can see engine-side costs at all: vision tokens for
    # prompt images carry no text to weigh, and the gap runs to
    # thousands of tokens on a multimodal window. ``usage.prompt_tokens``
    # / ``usage.completion_tokens`` ARE exact — the engine's own account
    # of the request, on every OpenAI-compatible response (non-streaming
    # always; streaming via stream_options.include_usage, which ullm's
    # client requests). Everything not yet billed carries a PROVABLE
    # bound: text is charged its raw character count (a text token
    # always spans at least one character, so a string's token count
    # never exceeds its length), and what no text carries —
    # per-message template framing, per-image vision cost — rides
    # derived constants. So the trigger can only fire early, the safe
    # direction: an early trim costs a deeper sawtooth, a late one hands
    # an over-window history downstream.
    # ------------------------------------------------------------------

    async def _prepare_summarized_request(
        self,
        ctx: LLMMoveContext,
        ordinary_max_tokens: int | None,
        *,
        summary_call_limit: int | None,
    ) -> tuple[int, int]:
        """Observe the complete summary compaction decision and its calls."""
        self._context_compacting = True
        try:
            return await self._prepare_summarized_context(
                ctx, ordinary_max_tokens, summary_call_limit=summary_call_limit
            )
        finally:
            self._context_compacting = False

    async def _prepare_summarized_context(
        self,
        ctx: LLMMoveContext,
        ordinary_max_tokens: int | None,
        *,
        summary_call_limit: int | None,
    ) -> tuple[int, int]:
        """Summarize on overflow, then return calls made and the ordinary
        request's decision reading."""
        if isinstance(ordinary_max_tokens, bool) or not isinstance(ordinary_max_tokens, int):
            raise ValueError("summarize compaction requires an explicit positive max_tokens")
        if ordinary_max_tokens <= 0:
            raise ValueError(f"summarize compaction requires positive max_tokens; got {ordinary_max_tokens}")
        if not isinstance(self._live_backend, ullm.backend.Backend) or not isinstance(
            self._live_conversation, ullm.conv.Conversation
        ):
            raise RuntimeError("summarize compaction requires a live backend and conversation")
        params = cast(ullm.backend.GenericSamplingParameters, self._live_params)
        game_cap = self._per_turn_cap(ctx)
        effective_reply_cap = ordinary_max_tokens if game_cap is None else min(ordinary_max_tokens, game_cap)
        params.max_tokens = effective_reply_cap
        request_tokens = await self._request_prompt_tokens(params, self._live_conversation)
        trigger = request_tokens + effective_reply_cap + self.safety_margin_tokens > self.context_window_tokens
        self._last_trim_fired = trigger
        if not trigger:
            return 0, request_tokens
        with self._compacting_scope():
            summary_calls = await self._summarize_live_history(
                ctx,
                ordinary_max_tokens,
                call_limit=summary_call_limit,
            )
        return summary_calls, await self._count_summarized_ordinary_request(ctx, ordinary_max_tokens)

    async def _count_summarized_ordinary_request(
        self,
        ctx: LLMMoveContext,
        ordinary_max_tokens: int,
    ) -> int:
        """Cap, validate, and measure the next ordinary summarize-mode request."""
        if not isinstance(self._live_conversation, ullm.conv.Conversation):
            raise RuntimeError("summarize compaction requires a live conversation")
        params = cast(ullm.backend.GenericSamplingParameters, self._live_params)
        game_cap = self._per_turn_cap(ctx)
        effective_reply_cap = ordinary_max_tokens if game_cap is None else min(ordinary_max_tokens, game_cap)
        params.max_tokens = effective_reply_cap
        request_tokens = await self._request_prompt_tokens(params, self._live_conversation)
        if request_tokens + effective_reply_cap + self.safety_margin_tokens > self.context_window_tokens:
            raise RuntimeError(
                "context overflow remained after model-written compaction: "
                f"{request_tokens} prompt + {effective_reply_cap} reply + {self.safety_margin_tokens} margin "
                f"exceeds window {self.context_window_tokens}"
            )
        return request_tokens

    def _summary_target_tokens(self) -> int:
        """Requested summary size under the validated compaction ratio."""
        self._uses_context_summary()
        target = self.context_window_tokens // self.compaction_ratio
        if target <= 0:
            raise ValueError(
                f"context_window_tokens={self.context_window_tokens} is too small for "
                f"compaction_ratio={self.compaction_ratio}"
            )
        return target

    @staticmethod
    def _detached_history(
        elements: list[ullm.conv.ConversationElement],
    ) -> list[ullm.conv.ConversationElement]:
        """Clone turn containers so the summary request cannot touch live state."""
        detached: list[ullm.conv.ConversationElement] = []
        for element in elements:
            if isinstance(element, ullm.conv.ResponseTurn):
                if element.elements:
                    detached.append(ullm.conv.ResponseTurn(elements=list(element.elements), status="processed"))
                continue
            detached.append(element)
        return detached

    def _summary_conversation(
        self,
        elements: list[ullm.conv.ConversationElement],
    ) -> ullm.conv.Conversation:
        """The summary request: the live request's own system prompt -- the
        same object, tools and reasoning effort included -- over the whole
        history, so it renders as the served history extended."""
        if not isinstance(self._live_conversation, ullm.conv.Conversation):
            raise RuntimeError("summarize compaction requires a live conversation")
        return ullm.conv.Conversation(
            system_prompt=self._live_conversation.system_prompt,
            elements=elements,
            status="intermediate",
            history_epoch=self.history_epoch,
            # It extends the served history: the same thread, whose next
            # request (a new epoch) ends it.
            conversation_id=self._conversation_id(),
        )

    def _conversation_id(self) -> str:
        """This agent's thread in the engine's prefix cache (ullm
        ``Conversation.conversation_id``), drawn at its first request, so an
        agent copied before play (a take) gets its own. Kept off the dataclass
        fields: it is play state, not configuration."""
        state = vars(self)
        conversation_id = state.get("_thread_conversation_id")
        if conversation_id is None:
            conversation_id = state["_thread_conversation_id"] = uuid.uuid4().hex
        return conversation_id

    async def _close_ended_thread(self, ctx: LLMMoveContext, epoch: int) -> None:
        """Before each request, ahead of the slot offer: if history was
        rewritten since the last request went out (a compaction, a retention
        trim, a discarded round), this conversation's thread in the engine is
        over, and the coming request, a new epoch, would end it anyway. Say so
        now, so its cached prefix leaves the host tier now rather than when
        that request comes, which a game giving up its slot here can make wait
        tens of minutes (ARC3's Kaggle D: 28 minutes at the median). ``epoch``
        is the coming request's."""
        state = vars(self)
        sent = state.get("_thread_epoch")
        state["_thread_epoch"] = epoch
        if sent is not None and sent != epoch:
            await self._close_thread(ctx)

    async def _close_thread(self, ctx: LLMMoveContext) -> None:
        """Tell the engine this agent's thread has ended
        (``Backend.close_conversation``). Nothing to close before the first
        request drew the thread; a failed close is only printed: it is a
        cache hint."""
        conversation_id = vars(self).get("_thread_conversation_id")
        if conversation_id is None:
            return
        try:
            await ctx.backend.close_conversation(conversation_id)
        except Exception as e:
            print(f"[duck] close_conversation({conversation_id}) failed: {e!r}", flush=True)

    async def _write_summary(
        self,
        ctx: LLMMoveContext,
        instruction: ullm.conv.UserPrompt,
        *,
        cap: int,
        reasoning_cap: int,
        call_limit: int | None,
        label: str,
        occasion: str,
        failed_as: str,
    ) -> _WrittenSummary:
        """Ask for one summary: the live request extended by ``instruction``,
        no tool reaching the kernel, ``cap`` tokens (``reasoning_cap`` where
        it reasons first), one reseeded retry, then the model's reply. Shared
        by compaction and the intermediate summaries; ``label`` names its turn
        span, ``occasion`` and ``failed_as`` its errors."""
        base_params = copy.deepcopy(cast(ullm.backend.GenericSamplingParameters, self._live_params))
        if self.summary_profile is not None:
            # The live request's state stays (its seed, and the recurrent-state
            # keys in its extra_params); how the summary is sampled is the
            # profile's. With no ``after_reasoning`` on the profile, the whole
            # summary samples at its temperature, the reasoning of a retry too.
            profile = cast(
                ullm.backend.GenericSamplingParameters, self.llm_config.sampling_profiles[self.summary_profile]
            )
            base_params.temperature = profile.temperature
            base_params.top_p = profile.top_p
            base_params.after_reasoning = copy.deepcopy(profile.after_reasoning)
            base_params.extra_params = {**base_params.extra_params, **copy.deepcopy(profile.extra_params)}
        # The live params carry this call's register injection. A summary is
        # not a reasoning block and the register was never measured on one, so
        # it goes. Without ``summary_thinking``, what the request opens on
        # instead is the end of thinking, so the summary is written without it
        # -- closed here rather than by the template, which would also drop the
        # effort line and with it the cached prefix -- and the cap is twice the
        # target (how often a summary runs past it is read off
        # ``summary_tokens``). With it, the request opens on the ordinary
        # generation prompt and the model reasons first, so the cap has to
        # cover the reasoning as well.
        if self.summary_thinking:
            base_params.response_injection = None
            summary_cap = reasoning_cap
        else:
            base_params.response_injection = _SUMMARY_THINK_CLOSE
            summary_cap = cap
        # A reply whose final text is empty is kept rather than rejected: a
        # model that thinks first can write its whole summary inside a thinking
        # block it never closes, and `_summary_response_text` reads it from
        # there. Rejected, a good summary would cost the game.
        base_params.keep_empty_reply = True
        if self.compaction_forbidden_token_ids:
            existing_bias = base_params.extra_params.get("logit_bias", {})
            if not isinstance(existing_bias, dict):
                raise ValueError("summary logit_bias must be a mapping when compaction token bans are configured")
            base_params.extra_params["logit_bias"] = {
                **existing_bias,
                **{token_id: -100.0 for token_id in self.compaction_forbidden_token_ids},
            }
        base_seed = base_params.seed if base_params.seed is not None else 0
        # The whole history, never reduced to fit: the engine window is sized
        # to hold the request instead (``prepare_for_backend``).
        request_elements = [*self._detached_history(self.messages), instruction]
        if min(summary_cap, self._per_turn_cap(ctx) or summary_cap) <= 0:
            raise GiveUpError(f"per-game token budget exhausted before {occasion}")

        invalid_tries = 0
        attempts = 0
        rerendered = False
        # Whether this attempt reasons first although ``summary_thinking`` is
        # off: the retry of a summary that came back entirely empty, with
        # nothing in the reasoning either (2026-10-02). A model can end
        # its reply on the first token after the injected end of thinking, and
        # does so again on a reseeded retry; reasoning first, it writes one.
        reasoning_retry = False
        while True:
            if call_limit is not None and attempts >= call_limit:
                raise _CallAllowanceExhausted(attempts)
            params = copy.deepcopy(base_params)
            if reasoning_retry:
                params.response_injection = None
                summary_cap = reasoning_cap
            remaining = self._per_turn_cap(ctx)
            completion_cap = summary_cap if remaining is None else min(summary_cap, remaining)
            if completion_cap <= 0:
                raise GiveUpError(f"per-game token budget exhausted during {occasion}")
            params.max_tokens = completion_cap
            params.seed = base_seed + invalid_tries
            attempt = self._summary_conversation(self._detached_history(request_elements))

            async def observe_summary_attempt(
                _attempt: int,
                params: ullm.backend.GenericSamplingParameters = params,
                attempt: ullm.conv.Conversation = attempt,
            ) -> None:
                if self.context_observer is None:
                    return
                self.context_observer.prepare_request(params)
                count = await self._request_prompt_tokens(params, attempt)
                self.context_observer.request_prepared(
                    params,
                    epoch=attempt.history_epoch,
                    prompt_tokens=count,
                    target=self._observer_compaction_target(),
                )

            def refuse_tool_calls(response: ullm.conv.LLMResponse) -> None:
                # The request carries the live tools so it renders as the
                # served history extended, but a summary is not a turn: a
                # call the model makes instead of writing one must not reach
                # the kernel, where it would run against the live namespace
                # and leave its effects behind when the reply is discarded.
                if self.response_observer is not None:
                    self.response_observer(response)
                if response.final_content.tool_calls():
                    raise _SummaryCalledATool()

            measured: int | None = None
            try:
                # The reading this attempt's usage is held against, taken before
                # it is sent, as before_round takes the live turn's.
                measured = await self._request_prompt_tokens(params, attempt)
                response_manager = ullm.manager.ResponseManager(
                    backend=ctx.backend,
                    conversation=attempt,
                    sampling_params=params,
                    max_retries=0,
                    preserve_failed_draws=self.preserve_failed_draws,
                    before_attempt=observe_summary_attempt if self.context_observer is not None else None,
                )
                manager = platypus.llm.DiagnosticTurnManager(
                    response_manager=response_manager,
                    conversation=attempt,
                    max_tool_rounds=1,
                    max_generated_tokens_per_turn=None,
                    diag=ctx.diag,
                    label=label,
                    after_response=refuse_tool_calls,
                )
                try:
                    await manager.run()
                except _SummaryCalledATool:
                    pass  # the reply is on the attempt; the check below refuses it
            except ullm.continuation.HistoryDriftError:
                if self.preserve_failed_draws or rerendered:
                    raise
                # The request did not extend the served ids after all. Declare
                # the rewrite and send it afresh -- an uncached prefill -- once,
                # and say so in the diagnostics.
                self._mark_history_rewritten()
                rerendered = True
                continue
            except ullm.backend.MalformedOutputError as exc:
                attempts += 1
                if self.preserve_failed_draws:
                    for element in attempt.last_turn.elements:
                        if isinstance(element, ullm.conv.LLMResponse):
                            assert measured is not None
                            self.counts_reconciled += self._check_one_usage(measured, element, enforce_window=False)
                    raise GiveUpError(
                        "context compaction failed: the model returned a malformed summary (retained draw, no retry)"
                    ) from exc
                if invalid_tries >= 1:
                    raise _SummaryUnusable(f"{failed_as} failed: model returned no usable summary twice") from exc
                invalid_tries += 1
                continue
            except ullm.backend.PromptTooLongError as exc:
                raise RuntimeError(
                    "context overflow: summarize never drops history to fit, and the summary request exceeded "
                    "the engine's max_model_len; size the engine window for it"
                ) from exc

            turn = attempt.last_turn
            attempts += 1
            responses = [element for element in turn.elements if isinstance(element, ullm.conv.LLMResponse)]
            if len(responses) != 1:
                raise RuntimeError(f"compaction produced {len(responses)} responses; expected exactly one")
            response = responses[0]
            # A summary request is billed like any other and faces the bound
            # check like any other, whether or not its answer is kept -- but
            # not the window invariant: it exceeds the agent's window by
            # design, and the engine's bounds it.
            self.counts_reconciled += self._check_one_usage(measured, response, enforce_window=False)
            if response.do_not_send_to_agent:
                raise RuntimeError(
                    "context overflow: summarize never drops history to fit, and the summary request exceeded "
                    "the engine's max_model_len; size the engine window for it"
                )

            summary = self._summary_response_text(response)
            invalid = not summary or "<tool_call>" in summary or bool(response.final_content.tool_calls())
            if invalid:
                if self.preserve_failed_draws:
                    raise GiveUpError(
                        "context compaction failed: the model returned no usable summary (retained draw, no retry)"
                    )
                if invalid_tries >= 1:
                    raise _SummaryUnusable(
                        f"{failed_as} failed: model returned an empty or tool-call summary twice "
                        + ("(retried with reasoning)" if reasoning_retry else "(reseeded retry)")
                    )
                reasoning_retry = not self.summary_thinking and not summary and not response.final_content.tool_calls()
                # The refused reply was served, so it is a step of the trace,
                # and the retry resends the request it answered: a prompt that
                # does not extend that reply's ids. Declared, the retry opens
                # a new epoch; undeclared, exact history stops the game. A
                # malformed reply (above) never became a step, so its retry
                # still extends the served history.
                self._mark_history_rewritten()
                invalid_tries += 1
                continue
            break
        return _WrittenSummary(summary, response, attempts, reasoning_retry, rerendered, summary_cap)

    async def _summarize_live_history(
        self,
        ctx: LLMMoveContext,
        ordinary_max_tokens: int,
        *,
        call_limit: int | None,
    ) -> int:
        """Replace old history with one summary and return generated attempts.

        The request is the live one extended by the instruction, so it is not a
        rewrite: it continues the served ids, which keeps its prefill cached.
        The rewrite is the rebuilt history, declared once the summary exists."""
        target = self._summary_target_tokens()
        # Read before the request, which says whether the file will be shown;
        # nothing can change it while the summary is written.
        candidate_source = self._summary_candidate_source()
        instruction = ullm.conv.UserPrompt(
            content=ullm.conv.StringContent(
                text=self._prompts().compaction_prompt(
                    self._prompt_modes(),
                    target,
                    previous_summary=self.context_summaries > 0,
                    candidate_shown=candidate_source is not None,
                    earlier_summaries=self._intermediate_summaries_in_cycle > 0,
                )
            )
        )
        written = await self._write_summary(
            ctx,
            instruction,
            cap=2 * target,
            reasoning_cap=6 * target,
            call_limit=call_limit,
            label="compaction",
            occasion="context compaction",
            failed_as="compaction",
        )
        summary, response, attempts = written.summary, written.response, written.attempts
        reasoning_retry, rerendered, summary_cap = written.reasoning_retry, written.rerendered, written.cap

        # Gathered before the rewrite, while the kernel is idle between rounds.
        closing: str | None = None
        if self.add_files_and_repl_to_summary:
            listing = await self._files_and_repl_listing()
            if candidate_source is not None:
                shown = self._prompts().compaction_candidate(candidate_source)
                listing = f"{listing}\n\n{shown}" if listing else shown
            closing = self._prompts().compaction_closing(
                self._prompt_modes(), listing, session_intact=self.repl is None or not self.repl.is_fresh
            )
        # A summary is two declared re-renders, both after its request. The
        # first: from here the history is being rewritten, and the install
        # counts the rebuilt candidates -- task, summary, tail -- which exact
        # history refuses to count on the epoch the summary request was served
        # on.
        self._mark_history_rewritten()
        self._prepare_context_observer()
        game_cap = self._per_turn_cap(ctx)
        effective_reply_cap = ordinary_max_tokens if game_cap is None else min(ordinary_max_tokens, game_cap)
        cast(ullm.backend.GenericSamplingParameters, self._live_params).max_tokens = effective_reply_cap
        tail_dropped, post_count = await self._install_summary(summary, closing, effective_reply_cap)
        # The second: the next ordinary call renders the rebuilt history
        # afresh, and the last thing served was the summary request. Under one
        # epoch a trace reads that as a prompt that fails to extend the
        # previous step's ids with no rewrite declared.
        self._mark_history_rewritten()
        self.context_summaries += 1
        usage = response.token_usage
        diagnostics: dict[str, Any] = {
            "compacted": True,
            "compaction_strategy": "summarize",
            "compaction_number": self.context_summaries,
            "compaction_target": target,
            "compaction_max_tokens": summary_cap,
            "summary_tokens": usage.completion_tokens if usage is not None else None,
            "summary_thinking": self.summary_thinking or reasoning_retry,
            "summary_reasoning_retry": reasoning_retry,
            "summary_tail": self.summary_tail,
            "summary_request_rerendered": rerendered,
            "recency_tail_dropped": tail_dropped,
            "context_after": post_count,
            "summary_closing_chars": None if closing is None else len(closing),
            "candidate_shown": candidate_source is not None,
        }
        platypus.diagnostics.current_span().kv(
            **diagnostics,
        )
        return attempts

    @staticmethod
    def _summary_response_text(response: ullm.conv.LLMResponse) -> str:
        """Plain final text, or historical reasoning when final text is empty --
        without the injected end of thinking. An engine's reasoning parser
        splits the reply there, but a backend that parses no reasoning leaves
        the marker at the head of the text, and a reply of the marker alone is
        an empty summary, not a valid one."""
        marker = _SUMMARY_THINK_CLOSE.strip()

        def text_of(content: ullm.conv.Content) -> str:
            return content.to_string().strip().removeprefix(marker).strip()

        return text_of(response.final_content) or text_of(response.reasoning_content)

    def _summary_candidate_source(self) -> str | None:
        """``candidate.py`` as the closing message of a rebuilt conversation
        shows it, or None where none is shown: without the file tools or the
        listing, before the file exists, and past
        :data:`CANDIDATE_SHOWN_MAX_CHARS`. Read host-side: the directory is the
        tool's."""
        if self.file_tools == "none" or not self.add_files_and_repl_to_summary:
            return None
        directory = self._ensure_repl()._tmpdir  # pyright: ignore[reportPrivateUsage] — the tool owns it, this only reads one file
        path = None if directory is None else os.path.join(directory, "candidate.py")
        if path is None or not os.path.isfile(path) or os.path.getsize(path) > CANDIDATE_SHOWN_MAX_CHARS:
            return None
        with open(path, encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        return source if len(source) <= CANDIDATE_SHOWN_MAX_CHARS else None

    async def _files_and_repl_listing(self) -> str:
        """The files in the kernel's working directory and the objects the
        model defined in its ``python`` session, for the message that closes
        a rebuilt conversation."""
        repl = self._ensure_repl()
        try:
            objects = await repl.session_objects()
        except SessionInspectionError as exc:
            platypus.diagnostics.current_span().event("repl/session_listing_failed", error=str(exc), output=exc.output)
            state = (
                "The Python kernel is unavailable; its next call starts a fresh session. "
                "Earlier variables may have been lost."
                if repl.is_fresh
                else "Your Python session was not cleared. Inspect its variables before relying on them."
            )
            diagnostic = f"Python session listing unavailable. {state}\n{exc}"
            return self._prompts().compaction_listing(
                self._prompt_modes(), repl.working_files(), [], objects_unavailable=diagnostic
            )
        return self._prompts().compaction_listing(self._prompt_modes(), repl.working_files(), objects)

    async def _install_summary(self, summary: str, closing: str | None, effective_reply_cap: int) -> tuple[bool, int]:
        """Atomically rebuild durable history around one accepted summary.
        Returns whether the recency tail had to go to fit (``summary_tail``
        says what it keeps when it fits), and the rebuilt request's decision
        reading.

        ``closing``, where there is one, follows the summary. The tail stays
        in the live turn, which the next request continues, so nothing can
        stand between it and that request."""
        if not self.messages or not isinstance(self.messages[-1], ullm.conv.ResponseTurn):
            raise RuntimeError("context compaction requires the live ResponseTurn")
        live_turn = self.messages[-1]
        task = next((element for element in self.messages if isinstance(element, ullm.conv.UserPrompt)), None)
        if task is None:
            raise RuntimeError("context compaction requires a pinned task prompt")
        if self._audit_quote is not None and self._audit_view is not None:
            # Once audited, the restated puzzle is what the game is pinned to,
            # showing the answer as it stands now rather than at the audit.
            task = self._audit_prompt(self._audit_view, self._audit_quote)
        wrapper = ullm.conv.UserPrompt(
            content=ullm.conv.StringContent(text=self._prompts().compaction_wrapper(summary))
        )
        head: list[ullm.conv.ConversationElement] = [task, wrapper]
        if closing is not None:
            head.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=closing)))
        opening_contents = self._opening_prompt_contents(self.messages)
        dropped_after_round = self._current_turn_rounds(self.messages)

        latest: tuple[int, int] | None = None
        for top_index in range(len(self.messages) - 1, -1, -1):
            element = self.messages[top_index]
            if not isinstance(element, ullm.conv.ResponseTurn):
                continue
            for sub_index in range(len(element.elements) - 1, -1, -1):
                if isinstance(element.elements[sub_index], ullm.conv.LLMResponse):
                    latest = (top_index, sub_index)
                    break
            if latest is not None:
                break

        # The tail as ``summary_tail`` keeps it. ``without_reasoning`` keeps a
        # COPY of the latest response without its reasoning, not an edit of the
        # served object: the copy carries no served witness and renders afresh,
        # which the rewrite declared around the install allows, and it takes
        # the original's place in the pending-bound pairing, which is positional.
        # It does carry the engine output (``_model_contents``, an init=False
        # field ``replace`` leaves behind): a backend renders a past response
        # from it -- the content through the reasoning parser, the reasoning
        # from ``reasoning_content``, which the copy has empty.
        stripped: ullm.conv.LLMResponse | None = None

        def tail_of(elements: list[ullm.conv.ConversationElement]) -> list[ullm.conv.ConversationElement]:
            nonlocal stripped
            if self.summary_tail != "without_reasoning" or not elements:
                return list(elements)
            head = elements[0]
            assert isinstance(head, ullm.conv.LLMResponse)
            stripped = dataclasses.replace(head, reasoning_content=ullm.conv.EmptyContent())
            stripped._model_contents = head._model_contents
            return [stripped, *elements[1:]]

        current_index = len(self.messages) - 1
        kept_current: list[ullm.conv.ConversationElement] = []
        dropped_current: list[ullm.conv.ConversationElement] = []
        if latest is None or self.summary_tail == "none":
            # No tail: nothing to keep, or ``none`` keeps nothing -- the live
            # turn's rounds are dropped and recorded like any compaction's.
            if latest is not None:
                dropped_current = list(live_turn.elements)
            candidate: list[ullm.conv.ConversationElement] = [*head, live_turn]
            candidate_for_count: list[ullm.conv.ConversationElement] = [
                *head,
                ullm.conv.ResponseTurn(elements=[], status="intermediate"),
            ]
        elif latest[0] == current_index:
            kept_current = tail_of(live_turn.elements[latest[1] :])
            dropped_current = list(live_turn.elements[: latest[1]])
            candidate = [*head, live_turn]
            candidate_for_count = [
                *head,
                ullm.conv.ResponseTurn(elements=list(kept_current), status="intermediate"),
            ]
        else:
            source_turn = cast(ullm.conv.ResponseTurn, self.messages[latest[0]])
            suffix = ullm.conv.ResponseTurn(elements=tail_of(source_turn.elements[latest[1] :]), status="processed")
            candidate = [*head, suffix, *self.messages[latest[0] + 1 :]]
            candidate_for_count = list(candidate)

        params = cast(ullm.backend.GenericSamplingParameters, self._live_params)
        count_conversation = self._ordinary_conversation(candidate_for_count)
        post_count = await self._request_prompt_tokens(params, count_conversation)
        tail_dropped = (
            latest is not None
            and self.summary_tail != "none"
            and post_count + effective_reply_cap + self.safety_margin_tokens > self.context_window_tokens
        )
        if tail_dropped:
            stripped = None
            dropped_current = list(live_turn.elements)
            kept_current = []
            candidate = [*head, live_turn]
            fallback: list[ullm.conv.ConversationElement] = [
                *head,
                ullm.conv.ResponseTurn(elements=[], status="intermediate"),
            ]
            post_count = await self._request_prompt_tokens(params, self._ordinary_conversation(fallback))
        if post_count + effective_reply_cap + self.safety_margin_tokens > self.context_window_tokens:
            raise RuntimeError("context overflow remained after compaction even without the recency tail")

        if dropped_current:
            self._record_compacted_rounds(dropped_current, opening_contents, dropped_after_round)
        # The summary rewrote every prompt the live turn's bills describe:
        # retire them as anchors, and move the bounds of the rounds it
        # replaced out of the live pairing — with the responses they were
        # recorded for — so every billed response still faces the window
        # invariant exactly once. The kept tail keeps its own bounds.
        self._mark_usage_stale(self.messages)
        if stripped is not None:
            # The copy carries the original's bill, which describes the same
            # rewritten prompt: retired as an anchor like the original.
            assert self._stale_responses is not None
            self._stale_responses.append(stripped)
        dropped_responses = [e for e in dropped_current if isinstance(e, ullm.conv.LLMResponse)]
        bounds = list(self._pending_bounds)
        self._deferred_usage_checks.extend(zip(bounds[: len(dropped_responses)], dropped_responses, strict=False))
        self._pending_bounds[:] = bounds[len(dropped_responses) :]
        live_turn.elements[:] = kept_current
        self.messages[:] = candidate
        return tail_dropped, post_count

    def _ordinary_conversation(
        self,
        elements: list[ullm.conv.ConversationElement],
    ) -> ullm.conv.Conversation:
        """One candidate durable history under the ordinary system and tools."""
        if isinstance(self._live_conversation, ullm.conv.Conversation):
            system = self._live_conversation.system_prompt
        else:
            system = ullm.conv.SystemPrompt(
                content=ullm.conv.StringContent(text=self._system_prompt_text()),
                reasoning_effort=self.reasoning_effort,
                tools=self._turn_tools(),
            )
        return ullm.conv.Conversation(
            system_prompt=system, elements=elements, status="intermediate", history_epoch=self.history_epoch
        )

    async def _measured_prompt_tokens(self, elements: list[ullm.conv.ConversationElement]) -> int:
        """The decision reading for the next request over ``elements``.

        Token-native mode asks the backend for the request's exact weight
        (``Backend.token_count``): on a continued call that is the length
        of the id ledger the request literally is, and after a declared
        rewrite it is the server's own render of the fresh prompt — exact
        either way, so the trigger squeezes every token out of the window.
        A backend that answers None (cloud, or no live block wired) falls
        back to the engine-billed + provable-bound arithmetic, as does any
        measurement of a list that is not the live conversation's."""
        if (
            self.token_native_context
            and self._live_backend is not None
            and self._live_conversation is not None
            and elements is self._live_conversation.elements
        ):
            try:
                exact = await self._live_backend.token_count(self._live_params, self._live_conversation)
            except ullm.continuation.HistoryDriftError as exc:
                if self.require_exact_context:
                    raise
                # Drift surfacing on the COUNT path must never kill an
                # episode: the measurement is advisory, so degrade this
                # reading to the bill+bound ruler below (which can only fire
                # early) and put the event on the span. The respond path
                # keeps its raises: serving a re-rendered history without
                # declaring it stays forbidden.
                self._count_drift_events += 1
                platypus.diagnostics.current_span().kv(
                    count_history_drift=str(exc)[:200],
                    count_drift_events=self._count_drift_events,
                )
                exact = None
            if exact is not None:
                if isinstance(exact, bool) or not isinstance(exact, int) or exact < 0:
                    raise RuntimeError("context measurement requires a non-negative token count")
                if self.context_observer is not None:
                    self.context_observer.prompt_measured(
                        exact,
                        epoch=self._live_conversation.history_epoch,
                        trigger=self._observer_compaction_trigger(),
                        compacting=self._context_compacting,
                    )
                return exact
        if self.require_exact_context:
            raise RuntimeError("exact context measurement requires a live token-native backend and token count")
        return self._decision_prompt_tokens(elements)

    def _decision_prompt_tokens(self, elements: list[ullm.conv.ConversationElement]) -> int:
        """The decision ruler: predicted upper bound on the next prompt.

        ``known + pending``: the newest engine-billed response pins
        everything up to and including its own generation at exactly
        ``usage.prompt_tokens + usage.completion_tokens`` (completion
        included because the duck re-sends reasoning for all turns, so the
        whole generation returns as input), and everything appended after
        it — tool results, the next user prompt, responses the backend
        returned without usage — is bounded by its character count plus
        the framing and image constants (``_bound_tokens``).

        Two states fall back to bounding the WHOLE conversation instead:
        cold start (nothing billed yet), and the rounds between a trim
        that rewrote sent history and the next response — a pre-rewrite
        usage describes the pre-rewrite prompt, and extrapolating from it
        would first overcount (blocking the sawtooth from ever settling)
        and, after a memory-snapshot refresh grew the system prompt,
        could undercount. The first post-rewrite response replaces the
        bound with engine truth again ("trust after verify")."""
        anchor, known = self._usage_anchor(elements)
        if anchor is None:
            return self._bounded_prompt_tokens(elements)
        # Engine truth newer than any rewrite: staleness (if any) is settled.
        self._stale_responses = None
        return known + _ANCHOR_RERENDER_OVERHEAD_TOKENS + self._pending_bound_tokens(elements, anchor)

    def _usage_anchor(self, elements: list[ullm.conv.ConversationElement]) -> tuple[ullm.conv.LLMResponse | None, int]:
        """The newest response whose engine usage may anchor the prediction,
        with its billed ``prompt + completion`` total; ``(None, 0)`` when no
        usable one exists.

        Skipped and searched past: synthetic placeholders (never sent) and
        responses the backend returned without usage (a known vLLM lapse) —
        an older billed response plus the bound over everything after it is
        still valid accounting. NOT searched past: a response served before
        the last history rewrite (``_stale_responses``); every older
        response is pre-rewrite too, so the whole search stops there."""
        stale = self._stale_responses
        for element in reversed(elements):
            if not isinstance(element, ullm.conv.ResponseTurn):
                continue
            for sub in reversed(element.elements):
                if not isinstance(sub, ullm.conv.LLMResponse) or sub.do_not_send_to_agent:
                    continue
                usage = sub.token_usage
                if usage is None or not usage.prompt_tokens:
                    continue
                if stale is not None and any(sub is known for known in stale):
                    return None, 0
                return sub, usage.prompt_tokens + usage.completion_tokens
        return None, 0

    def _pending_bound_tokens(
        self, elements: list[ullm.conv.ConversationElement], anchor: ullm.conv.LLMResponse
    ) -> int:
        """Upper bound on what the elements after ``anchor`` add to the next
        prompt: rendered like the diagnostics payload (so tool-call JSON and
        re-sent reasoning weigh in), text at its character count, images at
        their engine-side bound. The system prompt and tool schemas are
        already inside the anchor's ``prompt_tokens`` and are not re-added —
        any change to them comes from a trim, which retires the anchor."""
        rendered: list[dict[str, Any]] = []
        images: list[ullm.conv.ImageContent] = []
        seen_anchor = False
        for element in elements:
            if isinstance(element, ullm.conv.ResponseTurn) and any(sub is anchor for sub in element.elements):
                index = next(i for i, sub in enumerate(element.elements) if sub is anchor)
                tail = ullm.conv.ResponseTurn(elements=list(element.elements[index + 1 :]), status="intermediate")
                rendered.extend(self._element_request_dicts(tail))
                images.extend(_iter_images([tail]))
                seen_anchor = True
                continue
            if seen_anchor:
                rendered.extend(self._element_request_dicts(element))
                images.extend(_iter_images([element]))
        return self._bound_tokens(rendered, images)

    def _bounded_prompt_tokens(self, elements: list[ullm.conv.ConversationElement]) -> int:
        """Provable bound over the WHOLE next request (system prompt, tool
        schemas, every message): the ruler when nothing billed is
        applicable — cold start, and post-rewrite until the next response."""
        payload = self._bound_request_payload(elements)
        return self._bound_tokens(payload["messages"], list(_iter_images(elements)), tools=payload["tools"])

    def _bound_tokens(
        self,
        rendered: list[dict[str, Any]],
        images: list[ullm.conv.ImageContent],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """The bound arithmetic shared by the whole-prompt and pending
        rulers — no estimation anywhere in it at the default
        ``bound_chars_per_token``. Text is charged its RAW CHARACTER COUNT:
        a text token always spans at least one character, so a string's
        token count never exceeds its length, and the JSON rendering
        measured here contains every character of the text the request will
        carry (plus JSON syntax and ``\\uXXXX`` escapes, which only widen
        the bound). A ``bound_chars_per_token`` above 1 divides it, a
        tokenizer's calibration in place of the bound. What text cannot bound
        rides the constants: chat-template framing tokens per message
        (``per_message_overhead_tokens``) and the engine-side cost of each
        image (``_image_bound_tokens``)."""
        payload: dict[str, Any] = {"messages": rendered}
        if tools is not None:
            payload.update(tools=tools, tool_choice="auto")
        chars = math.ceil(_rendered_chars(payload) / self.bound_chars_per_token)
        overhead = len(rendered) * self.per_message_overhead_tokens
        image_cost = sum(self._image_bound_tokens(image) for image in images)
        return chars + overhead + image_cost

    def _image_bound_tokens(self, image: ullm.conv.ImageContent) -> int:
        """Engine-side token bound for one prompt image.

        Derivation (``image_tokens_override`` replaces it wholesale): the
        Qwen-VL family vision stack patchifies at 14 or 16 px and merges
        2x2 patches into one visual token, i.e. one token per 28x28 (Qwen2/
        2.5-VL) or 32x32 (Qwen3-VL) pixel cell after resizing to cell
        multiples — ``ceil(H/28) * ceil(W/28)`` tokens for the densest
        family. The duck's grid images are the board (<= 64x64 cells)
        nearest-upscaled by ``multimodal_upscale`` (default 4 -> <= 256x256
        px -> 100 cells). The x2 covers resize rounding, the processor's
        min-pixel upscaling of small grids, and per-row/begin/end vision
        delimiter tokens; the +32 covers the surrounding template framing.
        Deliberately a bound, not a calibration: a startup probe against
        the live engine would put an extra request into every captured/
        replayed stream, and at production scale (~10 retained images) the
        slack costs ~1.5k tokens of premature compaction on a 28k window —
        cheap next to any undercount, which breaches the window."""
        if self.image_tokens_override is not None:
            return self.image_tokens_override
        height, width = image.data.shape[:2]
        cells = math.ceil(height / 28) * math.ceil(width / 28)
        return 2 * cells + 32

    def _check_engine_usage(self, turn: ullm.conv.ResponseTurn) -> int:
        """Hold each response's billed usage against the window invariant,
        and return how many were checked.

        The invariant: ``prompt_tokens + completion_tokens`` never exceeds
        ``context_window_tokens`` — the bound every downstream consumer of
        a history (training above all) is entitled to assume. The decision
        path triggers compaction at window − reserve − margin with a ruler
        that can only overcount, so a billed total above the window is an
        accounting failure, and it raises: one game fails loudly at the
        moment of the overrun (RuntimeError, a duck bug by contract) rather
        than an oversized sequence travelling on.

        Covered pairs come from two queues, each response exactly once: the
        live ``_pending_bounds`` ↔ surviving-responses pairing, and
        ``_deferred_usage_checks`` for responses round-pruning removed from
        the turn mid-block.

        Below the invariant, a billed prompt above its recorded pre-call
        bound means the constants fell short of what no text carries —
        template framing beyond the per-message allowance, a retry's
        appended newlines, an image past its derived cost. The bound is
        load-bearing for one round — the next response re-anchors on engine
        truth — and the safety margin covers the gap meanwhile, so it is
        reported (diagnostics event + stdout), not raised."""
        deferred = list(self._deferred_usage_checks)
        self._deferred_usage_checks.clear()
        bounds = list(self._pending_bounds)
        self._pending_bounds.clear()
        responses = [e for e in turn.elements if isinstance(e, ullm.conv.LLMResponse)]
        assert len(responses) == len(bounds), (
            f"bound pairing broke: {len(bounds)} pre-call bound(s) but {len(responses)} response(s) "
            f"in the turn. before_round fires once per LLM call, so these must correspond; without "
            f"the pairing the billed usage cannot be held against what was predicted for it."
        )
        checked = 0
        for bound, response in [*deferred, *zip(bounds, responses, strict=True)]:
            checked += self._check_one_usage(bound, response)
        return checked

    def _check_one_usage(self, bound: int, response: ullm.conv.LLMResponse, *, enforce_window: bool = True) -> int:
        """One response against the window invariant: 1 when checked, 0 for
        the two skips — a synthetic placeholder carries no bill (the request
        was never sent; the overrun takes its normal course of synthetic
        response, no tool call, nudge, opening a new user-led block for
        eviction to drop), and a response without a usage block cannot be
        held against anything (a known vLLM lapse; the decision path already
        treats such a response as unbilled weight — bounded, not anchored) —
        logged so a run with many of them is visible.

        ``enforce_window=False`` keeps the bound check and drops only the
        window invariant: for a request that exceeds the agent's window by
        design (a summary request), bounded by the engine's. The window is
        ``usage_window_tokens`` where set, wider than the agent's."""
        if response.do_not_send_to_agent:
            return 0
        usage = response.token_usage
        if usage is None or not usage.prompt_tokens:
            print(
                f"[context-accounting] backend reported no usage for a call predicted at "
                f"<= {bound} input tokens; window check skipped for this call"
            )
            return 0
        billed = usage.prompt_tokens + usage.completion_tokens
        if usage.prompt_tokens > bound:
            # ``malformed_retries`` legitimately adds a few newline tokens
            # after the bound is recorded; anything beyond that is the bound
            # undercounting. Either way: report, do not kill — the margin
            # absorbs it and the next anchor heals it.
            platypus.diagnostics.current_span().event(
                "context_bound_undercount",
                predicted_max=bound,
                billed_prompt=usage.prompt_tokens,
                malformed_retries=response.malformed_retries,
            )
            print(
                f"[context-accounting] billed prompt of {usage.prompt_tokens} tokens exceeded "
                f"its predicted bound of {bound} (malformed retries: {response.malformed_retries}); "
                f"the safety margin absorbs this and the next response re-anchors on engine truth"
            )
        window = self.context_window_tokens
        if self.usage_window_tokens is not None:
            window = max(window, self.usage_window_tokens)
        if enforce_window and billed > window:
            raise RuntimeError(
                f"context window invariant violated: the engine billed {usage.prompt_tokens} prompt + "
                f"{usage.completion_tokens} completion = {billed} tokens against "
                f"a window of {window} (pre-call bound {bound}). "
                f"The compaction trigger fires before this can happen, so this is an accounting "
                f"bug — failing this game loudly here instead of handing an oversized sequence "
                f"downstream."
            )
        return 1

    def _bound_request_payload(self, elements: list[ullm.conv.ConversationElement]) -> dict[str, Any]:
        """The pseudo-request whose rendered characters the whole-prompt
        bound measures: system + rendered messages + tool schema, mirroring
        what is actually sent. Images render as nothing here — the bound
        weighs them separately at their engine-side cost."""
        rendered: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt_text()}]
        for element in elements:
            rendered.extend(self._element_request_dicts(element))
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._turn_tools()
        ]
        return {
            "messages": rendered,
            "tools": tools,
            "tool_choice": "auto",
        }

    def _element_request_dicts(self, element: ullm.conv.ConversationElement) -> list[dict[str, Any]]:
        """Render one conversation element into original-style message dicts for
        the bound (assistant entries carry tool_calls and re-sent reasoning)."""
        if isinstance(element, ullm.conv.UserPrompt):
            return [
                {
                    "role": "user",
                    "content": self._content_request_text(element.content),
                }
            ]
        if isinstance(element, ullm.conv.ResponseTurn):
            out: list[dict[str, Any]] = []
            for sub in element.elements:
                if isinstance(sub, ullm.conv.LLMResponse):
                    tool_calls = sub.final_content.tool_calls() if sub.final_content else []
                    entry: dict[str, Any] = {
                        "role": "assistant",
                        "content": sub.final_content.to_string() if sub.final_content else "",
                    }
                    # Reasoning is re-sent for all turns (send_back_reasoning
                    # "all", the original's preserve_thinking) — it must
                    # weigh in the bound, as re-sent input text does.
                    reasoning = sub.reasoning_content.to_string()
                    if reasoning:
                        entry["reasoning"] = reasoning
                    if tool_calls:
                        entry["tool_calls"] = [
                            {
                                "id": tc.call_id,
                                "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments)},
                            }
                            for tc in tool_calls
                        ]
                    out.append(entry)
                elif isinstance(sub, ullm.conv.ToolResult):
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": sub.call_id,
                            "content": self._content_request_text(sub.content),
                        }
                    )
            return out
        return []

    def _content_request_text(self, content: ullm.conv.Content) -> str:
        """Content -> the text a request will carry. Images render as
        nothing: they carry no text, and their engine-side cost is charged
        by ``_image_bound_tokens`` where the bound is assembled."""
        if isinstance(content, ullm.conv.ListContent):
            return "".join(self._content_request_text(sub) for sub in content.elements)
        if isinstance(content, ullm.conv.ImageContent):
            return ""
        return content.to_string()

    async def _backend_prompt_tokens(
        self,
        backend: ullm.backend.Backend,
        params: ullm.backend.SamplingParameters,
        conversation: ullm.conv.Conversation,
    ) -> int | None:
        """The backend's exact count for one request that is not the live
        turn's, or None where it cannot count."""
        count_conversation = conversation
        if not conversation.elements or not isinstance(conversation.elements[-1], ullm.conv.ResponseTurn):
            # ProxyBackend snapshots through ``conversation.last_turn``. A
            # detached one-shot request still ends in its UserPrompt until
            # TurnManager appends the empty turn; include that no-content
            # shape only for counting so direct and proxied requests render
            # the same prompt.
            count_conversation = ullm.conv.Conversation(
                system_prompt=conversation.system_prompt,
                elements=[
                    *conversation.elements,
                    ullm.conv.ResponseTurn(elements=[], status="intermediate"),
                ],
                status=conversation.status,
                history_epoch=conversation.history_epoch,
            )
        count = cast(object, await backend.token_count(params, count_conversation))
        if count is None:
            return None
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            print(f"[token-count] backend returned invalid count {count!r}; using the provable bound")
            return None
        return count

    async def _request_prompt_tokens(
        self,
        params: ullm.backend.SamplingParameters,
        conversation: ullm.conv.Conversation,
    ) -> int:
        """The decision reading for a request that is not the live turn's:
        the summarize arm's detached summary request, and the rebuilt
        history it installs. Token-native mode asks the backend for that
        conversation's exact weight — a declared rewrite precedes every one
        of these, so the answer is the server's own render. Elsewhere, and
        wherever the backend cannot count, it is the provable bound over the
        whole request, the same ruler the live turn's cold start uses."""
        if self.token_native_context and isinstance(self._live_backend, ullm.backend.Backend):
            exact = await self._backend_prompt_tokens(self._live_backend, params, conversation)
            if exact is not None:
                if self.context_observer is not None:
                    self.context_observer.prompt_measured(
                        exact,
                        epoch=conversation.history_epoch,
                        trigger=self._observer_compaction_trigger(),
                        compacting=self._context_compacting,
                    )
                return exact
        if self.require_exact_context:
            raise RuntimeError("exact context measurement requires a live token-native backend and token count")
        return self._bounded_conversation_tokens(conversation)

    def _bounded_conversation_tokens(self, conversation: ullm.conv.Conversation) -> int:
        """``_bounded_prompt_tokens`` for a conversation carrying its own
        system prompt and tools rather than the live turn's."""
        rendered: list[dict[str, Any]] = [{"role": "system", "content": conversation.system_prompt.content.to_string()}]
        for element in conversation.elements:
            rendered.extend(self._element_request_dicts(element))
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in conversation.system_prompt.tools
        ]
        return self._bound_tokens(rendered, list(_iter_images(conversation.elements)), tools=tools or None)

    def _mark_usage_stale(self, elements: list[ullm.conv.ConversationElement]) -> None:
        """Retire every billed usage currently in ``elements`` as a decision
        anchor: called at the moment a trim fires, before it mutates
        anything, because the rewrite makes each of those bills describe a
        prompt that will no longer be sent. A response generated after this
        call is not in the recorded id list, so the first post-rewrite
        usage re-anchors the decision path on engine truth (and clears the
        list)."""
        self._stale_responses = [
            sub
            for element in elements
            if isinstance(element, ullm.conv.ResponseTurn)
            for sub in element.elements
            if isinstance(sub, ullm.conv.LLMResponse)
        ]

    def _reply_reserve_tokens(self) -> int:
        """Reserve the same cap each request uses, or 512 when uncapped."""
        if self._uses_context_summary() and self.reply_cap_tokens is None:
            raise ValueError("the summarize compaction strategy requires a positive reply_cap_tokens")
        return self.reply_cap_tokens if self.reply_cap_tokens is not None else 512

    def context_budget_tokens(self) -> int:
        """Return the prompt budget after reply reserve and safety margin.

        This is the context budget, not a policy-specific early compaction
        trigger. It reflects the agent's current configuration.
        """
        return self._context_budget_tokens()

    def compact_target_tokens(self) -> int:
        """Return the validated compaction target for the active policy.

        Summary compaction reports the prompt budget as its reset ceiling.
        Prune compaction validates and reports ``compact_to_tokens``.
        """
        return self._observer_compaction_target()

    def _context_budget_tokens(self) -> int:
        """Context budget for eviction: window minus reply reserve minus the
        safety margin (floor 1024). The trigger comparison ``predicted >
        budget`` is therefore ``known + pending + reserve + margin >
        window`` — the invariant the whole decision path serves."""
        return max(1024, self.context_window_tokens - self._reply_reserve_tokens() - self.safety_margin_tokens)

    def _compact_target_tokens(self) -> int:
        """How far a trim goes once it fires: the compaction target.

        The target must be positive and below the budget. A shallow
        target — within a reply reserve of the budget — trims less headroom
        per compaction and a worst-case reply can re-cross within a round;
        but replies typically run far under their cap, so such a target
        still buys several rounds. That makes shallowness a preference (a
        slower configuration, deliberately reachable by the sweeps and the
        shipped defaults), not an error. Only a target at or above the
        budget itself has nothing left to choose and raises."""
        if self.compact_to_tokens <= 0:
            raise ValueError(
                f"compact_to_tokens={self.compact_to_tokens} must be positive: without a target below "
                "the budget a trim stops at the budget's edge, so the next round re-crosses it and "
                "re-trims, busting the KV prefix every round."
            )
        ceiling = self._context_budget_tokens()
        if self.compact_to_tokens >= ceiling:
            raise ValueError(
                f"compact_to_tokens={self.compact_to_tokens} must be below the context budget "
                f"({ceiling}): the trigger fires at the budget, so a prompt trimmed to at-or-above "
                "it is over budget again the moment the trim ends — no sawtooth at all."
            )
        return self.compact_to_tokens

    def _block_bounds(self, elements: list[ullm.conv.ConversationElement]) -> list[int]:
        """Start indices of user-led blocks.

        Consecutive user messages jointly open one block and must be evicted
        together before their one following response.
        """
        return [
            index
            for index, element in enumerate(elements)
            if isinstance(element, ullm.conv.UserPrompt)
            and (
                index == 0
                or not isinstance(elements[index - 1], ullm.conv.UserPrompt)
                # A kept task prompt whose turn has been evicted is a bare
                # prompt at the head; the prompt behind it opens the next
                # block rather than being a second message of the task's.
                or (index == 1 and self.keep_task_prompt)
            )
        ]

    def _block_prompts(
        self,
        elements: list[ullm.conv.ConversationElement],
        start: int,
    ) -> list[ullm.conv.UserPrompt]:
        """The consecutive user prompts beginning one block at ``start``."""
        prompts: list[ullm.conv.UserPrompt] = []
        for element in elements[start:]:
            if not isinstance(element, ullm.conv.UserPrompt):
                break
            prompts.append(element)
        return prompts

    async def _trim_live_messages(self) -> None:
        """The prune arm's share of ``before_round``: evict against the
        context budget just before each LLM call (in place — the live element
        list), then record the surviving prompt's predicted upper bound so
        the billed usage can be held against it post-turn
        (``_check_engine_usage``)."""
        self._rounds_since_compaction += 1
        await self._trim_messages_for_context(self.messages, enforce=True)
        # The reading the coming response's usage is held against, on the
        # same ruler as the thresholds (exact in token-native mode).
        measured = await self._measured_prompt_tokens(self.messages)
        self._pending_bounds.append(measured)
        self._arm_zero_recurrent(int(measured))
        self._record_prompt_injection_epoch()

    def _prepare_context_observer(self) -> None:
        """Retire metadata before this request's preparation measurements."""
        if self.context_observer is not None:
            assert self._live_params is not None
            self.context_observer.prepare_request(self._live_params)

    async def _observe_request_attempt(self, attempt: int) -> None:
        """Measure and annotate the actual call after retry and turn-cap adjustments."""
        if self.context_observer is None:
            return
        assert self._live_conversation is not None and self._live_params is not None
        if attempt:
            self._prepare_context_observer()
        measured = await self._measured_prompt_tokens(self._live_conversation.elements)
        if attempt and self._pending_bounds:
            self._pending_bounds[-1] = measured
        self.context_observer.request_prepared(
            self._live_params,
            epoch=self._live_conversation.history_epoch,
            prompt_tokens=measured,
            target=self._observer_compaction_target(),
        )

    def _observer_compaction_trigger(self) -> int:
        """Report the active compaction policy's trigger to observers."""
        return self._context_budget_tokens()

    def _observer_compaction_target(self) -> int:
        """Report the active compaction policy's reset ceiling to observers."""
        return self._context_budget_tokens() if self._uses_context_summary() else self._compact_target_tokens()

    def _arm_prompt_injection(self) -> None:
        """Select the coming round's injection without consuming its schedule.

        The default opens once per game; context mode opens once per declared
        history epoch. Rewrites re-arm before counting the replacement prompt,
        so exact context accounting includes its newly injected tokens.
        """
        params = self._live_params
        if params is None:
            return
        first = self.response_step == 1
        if self.prompt_injection_schedule == "context":
            first = self._prompt_injection_epoch != self._current_injection_epoch()
        text = self.inject_text_on_first_prompt if first else self.inject_text_on_later_prompts
        params.response_injection = text or None

    def _current_injection_epoch(self) -> int:
        """Read the live epoch, including rewrites declared by the turn manager."""
        conversation = self._live_conversation
        return max(self.history_epoch, conversation.history_epoch if conversation is not None else 0)

    def _record_prompt_injection_epoch(self) -> None:
        """Commit the prepared round's epoch after its final prompt measurement."""
        if self.prompt_injection_schedule == "context" and self._live_params is not None:
            self._prompt_injection_epoch = self._current_injection_epoch()

    def _arm_zero_recurrent(self, prompt_tokens: int) -> None:
        """Ask the engine to zero the recurrent state at ``zero_recurrent_at``.

        Asked on EVERY call, because the engine is the only side that can
        decide correctly: it declines whenever its cache already holds
        recurrent state past the position, which on a game-specific prefix
        can only be this conversation's own state from an earlier blanked
        call. Asking is therefore idempotent, and the fork's own comments
        specify this division of labour.

        Deciding here instead would mean guessing, and guessing wrong in a
        way that loses the effect. The blank propagates only once the
        sequence crosses the next recurrent block boundary and the state is
        written back; that crossing is not guaranteed -- measured upstream at
        a game start the margin is a median of 140 tokens, and 2 of 35
        openings missed it. A missed write-back leaves the next call resuming
        at or before the position, where asking again re-blanks it. A
        schedule that asked only at a game start and after history rewrites
        would leave those games with a single perturbed token and no way to
        know.

        The position is an absolute prompt position, and that is the only
        spelling the engine has: a lag counted back from the end resolves to
        a different absolute position on every call as the prompt grows
        while the cached state records where the blank actually landed, so
        no caller could keep the two consistent. The engine refuses a
        position the prompt does not reach rather than blanking off-position,
        which is why this method only asks once the measured prompt is past
        it: a request the engine would reject is not sent.
        """
        params = self._live_params
        if params is None:
            return
        for key in ("zero_recurrent_state", "zero_recurrent_at"):
            params.extra_params.pop(key, None)
        if not self.zero_recurrent:
            return
        if prompt_tokens <= self.zero_recurrent_at:
            return  # the prompt does not reach the position; nothing to blank
        params.extra_params["zero_recurrent_state"] = True
        params.extra_params["zero_recurrent_at"] = self.zero_recurrent_at
        platypus.diagnostics.current_span().kv(
            zero_recurrent_at=self.zero_recurrent_at,
            zero_recurrent_prompt_tokens=prompt_tokens,
        )

    def _compacting_scope(self) -> contextlib.AbstractContextManager[None]:
        """Publish "compacting" to the game's status line for the duration of
        a trim that fired — the eviction/prune loops re-count the full prompt
        per iteration, which otherwise reads as unexplained "other" time.
        A no-op when no live block wired ``ctx.status`` (snapshot trims)."""
        return contextlib.nullcontext() if self._status_handle is None else self._status_handle("compacting")

    async def _trim_messages_for_context(
        self,
        elements: list[ullm.conv.ConversationElement],
        *,
        enforce: bool = False,
    ) -> list[ullm.conv.ConversationElement]:
        """Apply the configured schedule around context eviction."""
        if self.compact_before_request_only and not enforce:
            return elements
        self._context_compacting = True
        try:
            return await self._compact_messages_for_context(elements, enforce=enforce)
        finally:
            self._context_compacting = False

    async def _compact_messages_for_context(
        self,
        elements: list[ullm.conv.ConversationElement],
        *,
        enforce: bool = False,
    ) -> list[ullm.conv.ConversationElement]:
        """The original's eviction: while over budget, drop whole user-led blocks
        from the oldest end (chunked by evict_turns, never the last block).
        Mutates ``elements`` IN PLACE — during a TurnManager run it is the
        live conversation element list — and returns it.

        The trigger is the BUDGET; how far the trim goes is the compaction
        TARGET (``compact_to_tokens``). Separating the two is what makes the
        context a sawtooth rather than a prompt pinned to the budget edge."""
        budget = self._context_budget_tokens()
        target = self._compact_target_tokens()
        blocks_evicted = 0
        measured = await self._measured_prompt_tokens(elements) if elements else 0
        fired = bool(elements) and measured > budget
        self._last_trim_fired = fired
        if fired:
            # From here every billed usage describes a prompt that no longer
            # exists: retire the anchors FIRST, so the eviction/prune loops
            # below (and every decision until the next response) measure the
            # rewritten conversation with the provable bound instead of
            # extrapolating from a pre-rewrite bill.
            self._mark_usage_stale(elements)
            # One declaration per trim, made BEFORE the first edit: from
            # here the conversation's sent history is being rewritten, and
            # the next request continues served ids that must already know
            # it.
            self._mark_history_rewritten()
            with self._compacting_scope():
                turns_evicted = 0
                while True:
                    over_budget = await self._measured_prompt_tokens(elements) > target
                    if not over_budget and turns_evicted >= max(1, self.evict_turns):
                        break
                    starts = self._block_bounds(elements)
                    # Never evict the most recent user-led block.
                    if len(starts) <= 1:
                        break
                    if self.keep_task_prompt:
                        # The task prompt stays. Its turn goes first (the
                        # block's tail, leaving the bare prompt at the head);
                        # after that, the block behind it is the oldest one
                        # that may leave — and it too must not be the last.
                        if starts[1] - starts[0] > 1:
                            del elements[starts[0] + 1 : starts[1]]  # the block stays, minus its turn
                        elif len(starts) >= 3:
                            del elements[starts[1] : starts[2]]
                            blocks_evicted += 1
                        else:
                            break
                    else:
                        del elements[starts[0] : starts[1]]
                        blocks_evicted += 1
                    turns_evicted += 1
        # Always start at a user-led block.
        starts = self._block_bounds(elements)
        if starts and starts[0] > 0:
            if not fired:
                self._mark_history_rewritten()  # the trim's one declaration, when only this ran
            del elements[: starts[0]]

        if fired and enforce:
            # Dropping whole blocks cannot touch the CURRENT turn, and a turn
            # grows from the inside: every round appends a response and its
            # tool results to one block, past reasoning re-sent each time. So
            # a long turn outruns block eviction. Fall back to that turn's own
            # oldest rounds.
            #
            # Gated on ``fired``, and pruning to the TARGET once it has: the
            # two halves of a trim have to share one trigger. Gate pruning on
            # the budget instead and it bails in the whole band between target
            # and budget -- which is exactly where eviction leaves a prompt,
            # since it stops at the last block rather than at a size -- and
            # the hysteresis is inert.
            #
            # Prune, note, repeat: the note is prompt text too, so adding it
            # can push back over the target pruning just reached. It is
            # rewritten in place rather than stacked, so this converges —
            # every pass either drops a round or stops.
            #
            # The opening prompt's content is taken by reference BEFORE the
            # notes rewrite it: the compaction record must show the prompt
            # as it read while the dropped rounds ran, note included if an
            # earlier trim had already left one.
            opening_contents = self._opening_prompt_contents(elements)
            dropped_after_round = self._pruned_rounds + self._current_turn_rounds(elements)
            dropped: list[ullm.conv.ConversationElement] = []
            with self._compacting_scope():
                while True:
                    pruned = await self._prune_current_turn(elements, target, target=target, dropped_into=dropped)
                    if not pruned:
                        break
                    self._note_pruned_rounds(elements, pruned)
            if dropped:
                self._record_compacted_rounds(dropped, opening_contents, dropped_after_round)
            prompt_tokens = await self._measured_prompt_tokens(elements)
            if self.require_exact_context and prompt_tokens > target:
                raise RuntimeError(
                    f"compaction cannot meet target {target}: irreducible prompt has {prompt_tokens} tokens"
                )
            if prompt_tokens > budget:
                raise RuntimeError(
                    f"context budget {budget} cannot be met: the user prompt alone weighs "
                    f"{prompt_tokens} tokens after evicting every "
                    f"earlier block and every round of this turn."
                )

        if fired:
            # Report every trim, and in particular the one that achieved
            # NOTHING: with a single user-led block there is no earlier block
            # to evict, and an unenforced trim then returns still over
            # budget, silently — against a backend too large to reject the
            # prompt, this kv is the only place that overrun is visible.
            # The spacing pair makes the sawtooth's period visible per trim:
            # how many decision-ruler tokens grew, over how many LLM calls,
            # since the previous fired trim left the prompt at its post-trim
            # size.
            after = await self._measured_prompt_tokens(elements)
            since = self._tokens_at_last_compaction
            platypus.diagnostics.current_span().kv(
                compacted=True,
                context_after=after,
                context_budget=budget,
                context_target=target,
                blocks_evicted=blocks_evicted,
                rounds_pruned=self._pruned_rounds,
                compaction_spacing_tokens=measured - (since if since is not None else measured),
                compaction_spacing_rounds=self._rounds_since_compaction,
            )
            self._tokens_at_last_compaction = after
            self._rounds_since_compaction = 0
        return elements

    async def _prune_current_turn(
        self,
        elements: list[ullm.conv.ConversationElement],
        trigger: int,
        *,
        target: int | None = None,
        dropped_into: list[ullm.conv.ConversationElement],
    ) -> int:
        """Drop the current turn's oldest rounds until it fits, keeping the
        user prompt that opened it. Returns how many rounds went; the
        dropped elements themselves are appended to ``dropped_into``, in
        order, because the caller records them (``_record_compacted_rounds``)
        and this is their last moment in any live structure.

        ``trigger`` is the size at which this starts and ``target`` (default:
        the trigger) is how far it then goes. Deciding whether a trim happens
        AT ALL belongs to the caller, which fires on the budget; by the time
        pruning is reached that decision is made, so both of these are the
        compaction target.

        A round is one assistant response plus the tool results answering it,
        and they leave together: a tool_use whose result is missing is
        malformed input on the next request. Emptying the turn is allowed —
        that is the state a turn starts in.

        Dropping a round moves its response out of the live pairing:
        ``_pending_bounds`` is 1:1 with the turn's existing responses at
        prune time (the current round's bound is appended only after this
        trim), oldest first — and this removes the oldest round — so the
        matching bound is the front one. The (bound, response) pair goes to
        ``_deferred_usage_checks`` rather than away: a pruned response was
        still billed, and every billed response faces the window invariant
        exactly once.

        Once under target, drops ``evict_turns`` more rounds for headroom.
        Dropping the bare minimum leaves the prompt at the target edge, so the
        next round re-crosses and prunes again — and every prune shifts the
        prefix right after the user turn, busting the whole turn body's KV
        cache. The slack means a re-prune once per ``evict_turns`` rounds
        rather than every round (block eviction chunks by the same knob)."""
        target = trigger if target is None else target
        if await self._measured_prompt_tokens(elements) <= trigger:
            return 0  # already small enough; no gratuitous headroom pruning

        def _drop_oldest_round() -> bool:
            turn = elements[-1] if elements else None
            if not isinstance(turn, ullm.conv.ResponseTurn) or not turn.elements:
                return False
            first = turn.elements[0]
            cut = 1
            while cut < len(turn.elements) and not isinstance(turn.elements[cut], ullm.conv.LLMResponse):
                cut += 1
            # Handed to the caller's record before the delete, or it is
            # unrecoverable: the live turn is written down at close, from
            # what is left of this list.
            dropped_into.extend(turn.elements[:cut])
            del turn.elements[:cut]
            if self._pending_bounds:
                # The response leaves the live pairing, not the invariant:
                # its bound moves with it to the deferred check queue.
                bound = self._pending_bounds.pop(0)
                if isinstance(first, ullm.conv.LLMResponse):
                    self._deferred_usage_checks.append((bound, first))
            return True

        pruned = 0
        while await self._measured_prompt_tokens(elements) > target:
            if not _drop_oldest_round():
                return pruned  # turn emptied and it still doesn't fit — caller raises
            pruned += 1
        for _ in range(max(1, self.evict_turns)):
            # Headroom is an optimization — the fit is already achieved — so
            # it never takes the LAST round: that is the model's freshest
            # work, and losing it costs more than the re-prune it saves.
            # Only the must-fit loop above may empty a turn.
            if self._current_turn_rounds(elements) <= 1:
                break
            if not _drop_oldest_round():
                break
            pruned += 1
        return pruned

    def _note_pruned_rounds(self, elements: list[ullm.conv.ConversationElement], pruned: int) -> None:
        """Tell the model its own earlier rounds are gone. Pruning is silent
        PARTIAL loss mid-turn, unlike the wholesale reset the nudge path
        produces, so without this it keeps reasoning from work it no longer
        has."""
        starts = self._block_bounds(elements)
        if not starts:
            return
        prompt = self._block_prompts(elements, starts[-1])[-1]
        self._pruned_rounds += pruned
        note = ullm.conv.StringContent(text=self._prompts().rounds_pruned_note(self._pruned_rounds))
        content = prompt.content
        if isinstance(content, ullm.conv.ListContent):
            surface = self._prompts()
            kept = [
                e
                for e in content.elements
                if not (isinstance(e, ullm.conv.StringContent) and surface.is_rounds_pruned_note(e.text))
            ]
            prompt.content = ullm.conv.ListContent(elements=[*kept, note])
        else:
            prompt.content = ullm.conv.ListContent(elements=[content, note])

    def _opening_prompt_contents(self, elements: list[ullm.conv.ConversationElement]) -> list[ullm.conv.Content]:
        """Contents of the user messages opening the current block.

        Each is retained by reference before the pruning note rewrites the
        last prompt, preserving both bytes and consecutive role boundaries.
        """
        starts = self._block_bounds(elements)
        if not starts:
            return []
        return [prompt.content for prompt in self._block_prompts(elements, starts[-1])]

    def _current_turn_rounds(self, elements: list[ullm.conv.ConversationElement]) -> int:
        """Responses currently in the live turn (rounds still in the prompt)."""
        turn = elements[-1] if elements else None
        if not isinstance(turn, ullm.conv.ResponseTurn):
            return 0
        return sum(1 for e in turn.elements if isinstance(e, ullm.conv.LLMResponse))

    def _record_compacted_rounds(
        self,
        dropped: list[ullm.conv.ConversationElement],
        opening_contents: list[ullm.conv.Content],
        dropped_after_round: int,
    ) -> None:
        """One compaction, logged as if it closed a turn.

        In code a compaction is mid-turn surgery: the turn keeps running and
        only its oldest rounds leave. But the record it deserves is a turn
        boundary — the opening prompt as it read then, plus the rounds now
        gone, is a complete conversation the model saw and no longer does,
        and the live turn continues without it. So each trim that pruned
        becomes a child span under the true ``llm/turn``, of the same kind
        with its own ``conversation`` artifact, so the viewer renders it
        with the machinery it already has. Its response events and token
        spend live here too; the parent's inclusive totals absorb them, so
        the turn's total is what the backend billed, while the parent's own
        events and ``conversation`` artifact stay exactly the live turn.

        ``dropped_after_round``: how many rounds the turn had run when these
        left. Pruning takes oldest-first and nothing returns, so this one
        number per segment is the entire availability history.

        Inert on the null stream (no span, no blocks), so replay references
        are unaffected."""
        segment_rounds = sum(1 for e in dropped if isinstance(e, ullm.conv.LLMResponse))
        with platypus.diagnostics.current_span().span("llm/turn", "compacted") as span:
            span.kv(rounds_dropped=segment_rounds, dropped_after_round=dropped_after_round)
            turn = ullm.conv.ResponseTurn(elements=dropped)
            platypus.llm.record_round_metrics(span, dropped)
            # The system prompt of the live conversation when available: the
            # segment renders as a full conversation, and it WAS this one's.
            system = (
                self._live_conversation.system_prompt
                if isinstance(self._live_conversation, ullm.conv.Conversation)
                else ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text=""))
            )
            elements: list[ullm.conv.ConversationElement] = [
                ullm.conv.UserPrompt(content=content) for content in opening_contents
            ]
            elements.append(turn)
            conversation = ullm.conv.Conversation(system_prompt=system, elements=elements)
            structure = platypus.llm.serialize_conversation_blocks(conversation, cast(Any, _SpanBlocks(span)))
            span.artifact("conversation", structure)

    async def _persist_history(self) -> None:
        """Post-turn retention: keep the trimmed conversation.

        Measures with the one decision ruler there is. This trim runs
        after the analyzer loop, so if it measured differently it would be
        the binding constraint and the decision path would never actually
        govern the conversation."""
        if not self._uses_context_summary():
            await self._trim_messages_for_context(self.messages)

    # ------------------------------------------------------------------
    # 7. resources
    #
    # The kernel is the only live resource the duck owns. One lifetime
    # here — it lives as long as the game — where upstream offers four.
    # ------------------------------------------------------------------

    def _ensure_repl(self, ctx: LLMMoveContext | None = None) -> DuckPythonTool:
        """The lazily-built DuckPythonTool (prelude + output rendering,
        per-call timeout); reused for the whole game.

        With a ``ctx``, the tool is (re-)wired to report queueing for a pool
        slot and running a snippet into the run's status line. Re-wired on
        every use rather than at construction because the hook is
        process-local: a deepcopied or unpickled agent arrives with none."""
        if self.repl is None:
            self.repl = DuckPythonTool(
                timeout=float(self._python_timeout_s()),
                tool_output_tokens=self.tool_output_tokens,
                memory_limit_mb=self.kernel_memory_limit_mb,
                check_mode=self.check_mode,
                show_enabled=self._show_enabled(),
                max_shows=self._max_shows(),
                check_images_enabled=self._check_images_enabled(),
                image_scale=self._image_scale(),
                image_gridlines=self.image_gridlines,
                ascii_row_numbers=self.ascii_row_numbers,
                check_prints_on_pass=self.check_prints_on_pass,
                check_and_submit=self.check_and_submit,
                file_tools=self.file_tools != "none",
                initial_code=self._prelude(),
            )
        elif (
            self.repl.check_mode,
            self.repl.show_enabled,
            self.repl.max_shows,
            self.repl.ascii_row_numbers,
            self.repl.check_prints_on_pass,
            self.repl.check_and_submit,
            self.repl.file_tools,
            self.repl.initial_code,
        ) != (
            self.check_mode,
            self._show_enabled(),
            self._max_shows(),
            self.ascii_row_numbers,
            self.check_prints_on_pass,
            self.check_and_submit,
            self.file_tools != "none",
            self._prelude(),
        ):
            if not self.repl.is_fresh:
                raise RuntimeError("kernel feature gates cannot change after the game kernel has started")
            self.repl.check_mode = self.check_mode
            self.repl.show_enabled = self._show_enabled()
            self.repl.max_shows = self._max_shows()
            self.repl.ascii_row_numbers = self.ascii_row_numbers
            self.repl.check_prints_on_pass = self.check_prints_on_pass
            self.repl.check_and_submit = self.check_and_submit
            self.repl.file_tools = self.file_tools != "none"
            self.repl.initial_code = self._prelude()
        self.repl.check_images_enabled = self._check_images_enabled()
        self.repl.image_scale = self._image_scale()
        self.repl.image_gridlines = self.image_gridlines
        self.repl.wire_feedback_hosts()
        surface = self._prompts()
        self.repl.description = surface.python_tool_description(self._prompt_modes())
        self.repl.parameters["properties"]["code"]["description"] = surface.python_tool_code_param_description()
        if ctx is not None:
            self.repl.status_hook = ctx.status
        return self.repl

    def _prelude(self) -> str:
        """The kernel's one-time setup under this agent's gates."""
        return kernel_code.prelude(
            include_check=self.check_mode.enabled,
            include_show=self._show_enabled(),
            row_numbers=self.ascii_row_numbers,
            check_prints_on_pass=self.check_prints_on_pass,
            numpy_scalar_repr=self.numpy_scalar_repr,
            file_tools=self.file_tools != "none",
        )

    def _python_timeout_s(self) -> int:
        """The tool's real per-call limit — what the model is told, and what
        the kernel enforces, come from here so they cannot disagree.

        Below 1 s is a config error, raised rather than quietly repaired, and
        there is no ceiling: how long a call may run is the author's decision.
        What else may or may not bound the run is another layer's business and
        not knowable here, so nothing about it is claimed."""
        if self.python_timeout_s < 1:
            raise ValueError(f"python_timeout_s must be at least 1 s; got {self.python_timeout_s}")
        return self.python_timeout_s

    async def _on_finish(  # pyright: ignore[reportIncompatibleMethodOverride] — same covariant ctx narrowing as LLMAgent._do_move
        self, ctx: LLMMoveContext, outcome: GameOutcome
    ) -> None:
        """Resource cleanup: final kernel/tmpdir teardown (no LLM calls).

        Awaited, not fired and forgotten: ``clear_state`` is a coroutine, and
        calling it without awaiting leaves the kernel subprocess alive for
        the rest of the run while raising nothing."""
        if self.repl is not None:
            self.repl.on_answer = None  # closes over a ctx that is about to go
            self.repl.on_checked_answer = None
            self.repl.answer_feedback = None
            await self.repl.clear_state()
            self.repl = None
        # A take's play is over here, and the game plays on: its thread would
        # otherwise stay in the engine's host tier to the game's end.
        await self._close_thread(ctx)
        self._live_conversation = None
        self._live_backend = None
        self._live_params = None
        self._status_handle = None


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------


LR_REFLECTION = 4


def _dihedral(grid: taaf.grid.Grid, transform: int, *, inverse: bool = False) -> taaf.grid.Grid:
    """Reflect left-right for ids 4–7, then rotate counterclockwise by id % 4.

    Inversion reverses that order, including for rectangular grids.
    """
    data = grid.data
    if inverse:
        data = np.rot90(data, -(transform % 4))
    if transform >= 4:
        data = np.fliplr(data)
    if not inverse:
        data = np.rot90(data, transform % 4)
    return taaf.grid.Grid(data)


# What re-sending the anchor response's own generation costs beyond its
# billed ``completion_tokens``: the generation's sampled ids (reasoning,
# text, tool-call JSON, end markers — all inside completion_tokens) come
# back as input nearly verbatim under the duck's send-back-all policy, and
# the chat template adds only its assistant framing around them plus small
# re-encoding wobble in the tool-call JSON. A flat, deliberately generous
# cover for those few tokens.
_ANCHOR_RERENDER_OVERHEAD_TOKENS = 32


def _iter_images(elements: list[ullm.conv.ConversationElement]) -> Iterator[ullm.conv.ImageContent]:
    """Every prompt image in ``elements``, in order: user-prompt content and
    tool-result content, walked through nested ListContent. LLM responses
    never carry images inbound."""

    def walk(content: ullm.conv.Content) -> Iterator[ullm.conv.ImageContent]:
        if isinstance(content, ullm.conv.ListContent):
            for sub in content.elements:
                yield from walk(sub)
        elif isinstance(content, ullm.conv.ImageContent):
            yield content

    for element in elements:
        if isinstance(element, ullm.conv.UserPrompt):
            yield from walk(element.content)
        elif isinstance(element, ullm.conv.ResponseTurn):
            for sub in element.elements:
                if isinstance(sub, ullm.conv.ToolResult):
                    yield from walk(sub.content)


def _rendered_chars(value: Any) -> int:
    """Character count of the canonical JSON rendering of ``value``. With
    ``ensure_ascii`` every non-ASCII character renders as a six-character
    escape, so multi-token exotic characters still count more characters
    than tokens."""
    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    return len(rendered)


__all__ = ["DuckAgent"]
