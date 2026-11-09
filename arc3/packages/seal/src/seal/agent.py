"""SealAgent: the seal, on the platypus agent contract.

How it plays. Each move the seal tells the model what its last actions did
and hands it the current board inside the kernel, as ASCII and segmented
objects; ``multimodal_context`` can additionally attach the grid to the
prompt as an image.
A frozen journal snapshot sits in the system prompt; new memories enter
history through a native tool, and compaction refreshes the snapshot. The
model answers with Python, which runs in a persistent kernel where
``submit_action([...])`` calls back into this agent and executes against the
live game — so one snippet can act, look at the result, and act again.
The move ends as soon as the model has acted; everything before that is
investigation, and how much of it happens is the model's choice.

Reading order — the file follows one move through the agent:

1. configuration and per-game state, then the deployment presets
2. ``_do_move`` — one move, the whole spine on one screen
3. the analyzer loop — the LLM turns a move runs until the model acts
4. acting — the host calls the model's Python reaches the game through
5. observation bookkeeping — what the model is told each action did
6. prompts — the system prompt and the per-turn user prompt
7. context budget — eviction, exact token counts, retention
8. resources — kernel lifetime, and the checkpoint cadence it permits

What it leans on the harness for. Each of these is the agent-facing half
of a contract requirement, and between them they are the reason there is
no plumbing in this file — no subprocess handling, no history
bookkeeping, no capture, no span timing, no TAAF at all:

- ``ctx.execute(action)`` executes mid-move and returns the real
  observations, which is what makes the model's in-snippet
  act-inspect-act possible (contract §2 in-move style, ledger #1).
- ``ctx.backend`` is the shared LLM client; ``resolve_sampling(ctx,
  profile)`` resolves a NAMED sampling profile and draws the per-call
  seed from ``ctx.rng``, which the seal never touches itself (A2/U8).
- every backend response carries the engine's own ``usage`` numbers
  (``LLMResponse.token_usage``), which is what lets eviction run on
  engine truth for everything already billed plus a provable bound
  for what was appended since (U1).
- ``ullm.manager.ResponseManager`` carries the malformed-output retry and
  prompt-too-long defenses; ``platypus.llm.DiagnosticTurnManager`` runs
  the tool-call rounds and spans them. The seal supplies two hooks:
  ``before_round`` (evict to budget) and ``stop_when`` (end the turn once
  the model has acted).
- ``GameREPL`` is agent STATE rather than a context service: it
  pickles with the agent, draws kernel slots from a machine-wide pool,
  and how long its kernel lives is what ``supported_checkpointing``
  reports (A1/A4/U9).
- ``ctx.diag`` spans nest under the move span infra already opened, with
  no wiring; ``ctx.status`` reaches into the kernel so a wait for a pool
  slot is reported by the process actually waiting (I7).

Configuration is dataclass fields on this class, or on the sampling
profile named by ``profile`` (A6). Nothing here reads a clock, the
environment, or any file outside the kernel's scratch directory (A2).
"""

from __future__ import annotations

import base64
import builtins
import contextlib
import copy
import dataclasses
import itertools
import json
import math
import operator
import os
import sys
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Awaitable, Callable, ClassVar, Literal, Self, cast

import numpy as np
import taaf.game
import ullm.backend
import ullm.continuation
import ullm.conv
import ullm.manager
import ullm.tool
import ullm.tools.python as python_tool

import platypus.diagnostics
import platypus.llm
from framediff import Decomposer
from framediff.animation import MAX_CHARS
from platypus import Action, GiveUp, GiveUpError, LLMAgent, LLMMoveContext, Observation, RunConfig
from platypus.attempts import Attempts
from platypus.contract import GameResult
from seal import presets, seeds
from seal.action_names import model_move, to_engine_action, to_model_action
from seal.board_image import AXIS_MIN_SCALE, AXIS_TICK_STEP, board_image
from seal.context_observer import ContextObserver
from seal.file_suite import file_tool_names as build_file_tool_names
from seal.file_suite import file_tools as build_file_tools
from seal.frame_delta import (
    ANIMATION_CUT_CHARS,
    MAX_BITS_UNDERSTOOD,
    AnimationDescription,
    default_decomposer,
    kept_animation,
)
from seal.harness import game as harness_game
from seal.memory_tool import SealMemoryTool
from seal.palette import ASCII_SYMBOLS, AsciiLayout, AsciiSymbols
from seal.prompts import (
    PLAN_ARGUMENT_MODES,
    SIDE_TURN_MODES,
    PlanArgument,
    PromptModes,
    Prompts,
    SealPrompts,
    SideTurnMode,
)
from seal.repl import GameREPL, SessionInspectionError, plan_only
from seal.state import FramePayload, MemoryEntry, MoveChange, PendingBatch, PendingLevelWin, StepSummary, TimelineEntry
from seal.workspace import Workspace

#: Engine outcome to the word a Turn carries. The whale's three-way
#: vocabulary, not the engine's four-way one: a level win and a game win are
#: both 'won' to a model reasoning about what its action did, and the
#: difference between them is the run's business rather than the board's.
_TURN_OUTCOMES = {"playing": "playing", "level_won": "won", "game_won": "won", "lost": "lost"}

#: What ``SealAgent.build(duck_like=True)`` sets: every seal setting at the
#: value that makes the seal play as the duck does, apart from what the seal
#: is by construction -- its kernel and per-level history
#: (2026-09-28). The world model and the frame delta off, the duck's kernel
#: timeout, eviction chunk and tool-output cap, its board as letters without
#: row numbers, its colour-name legend, and UNDO offered, as the duck ships
#: it; the duck's window, and none of the seal's side turns, its replay
#: reminder, its per-move changes or its chronological restart.
#: ``completion_cutoff_notice`` stays the seal's. An override passed
#: to ``build`` beside it still wins.
DUCK_LIKE: dict[str, Any] = {
    "use_world_model": "none",
    "settled_delta": False,
    "context_window_tokens": 158280,
    "level_reflection": "none",
    "strategy_audit": "none",
    "replay_reminder": False,
    "move_changes": False,
    "chronological_restart": False,
    "animation_description": "none",
    "animation_marks": False,
    "python_timeout_s": 78,
    "evict_turns": 2,
    "tool_output_tokens": 2425,
    "ascii_row_numbers": False,
    "ascii_symbols": "letters",
    "color_name_legend": True,
    "expose_undo": True,
}
#: What ``SealAgent.build(kimi_student=True)`` sets: the model-visible
#: settings the Kimi K3.1 Pro teacher traces were played with that the
#: column has off, so a model trained on those traces plays the seal they
#: were recorded on. The world-model default stated in the prompts, the
#: summary written after the model's own reasoning (an API takes no
#: think-close injection), and the colour-name legend, which the traces were
#: recorded with (2026-10-30). The teacher runs build from it too
#: (``tools/teachers/seal_smoke.py``), so the two cannot drift. What else the
#: teacher sets is its API's (``tools/teachers/README.md``) and stays the
#: column's here, the reasoning level included: the traces are rendered at
#: the column's in training (2026-10-29). An override passed to
#: ``build`` beside it still wins.
KIMI_STUDENT: dict[str, Any] = {
    "world_model_default": True,
    "summary_thinking": True,
    "color_name_legend": True,
}
# The two fields a column composes rather than sets: replacing either through
# ``build`` would silently discard what the layers put inside it.
_COMPOSED_BY_THE_COLUMN = ("llm_config", "run_config")
# The model calls one move may make under ``summarize``, summaries included:
# a summary that never shrinks the context would otherwise loop.
_SUMMARY_REQUEST_BACKSTOP = 512
# What the summarize strategy injects after the ordinary generation prompt
# (``<think>\n``) so the summary is written without thinking, as the template
# would render it with ``enable_thinking=False`` (``<think>\n\n</think>\n\n``).
# The template's own switch cannot be used: it also drops the reasoning-effort
# line that opens the system block, and with it the whole cached prefix.
_SUMMARY_THINK_CLOSE = "\n</think>\n\n"
#: What a summarize configuration must leave the summary under the engine
#: window, in the worst case it owns: a prompt at the full window plus a reply
#: reserve. Sized on the summaries sweep 12 wrote (the duck's ``evidence.md``,
#: "Sweeps 12 and 13"); a request past it is clamped to what the window has
#: left, never sent past it.
MIN_SUMMARY_ROOM_TOKENS = 8_192


class _SummaryCalledATool(Exception):
    """A summary reply carried a tool call: stop the turn before it runs."""


class _CallAllowanceExhausted(Exception):
    """Stop before an ordinary call when compaction spent its allowance."""

    def __init__(self, calls_made: int = 0) -> None:
        super().__init__("LLM-call allowance exhausted")
        self.calls_made = calls_made


class _SideTurnCut(_CallAllowanceExhausted):
    """Stop a side turn (the reflection, the audit) before its next call: its
    context reached the window it is held to, past which a summary is due."""


class _AuditDue(Exception):
    """Stop the turn before its next call: the level's spend passed a strategy
    audit threshold, and the audit runs first."""


# The name the harness's own cells import the kernel's store under. Not the
# prepend's ``_rs``: a model cell may have bound that name to anything, and a
# harness cell runs without the prepend that restores it.
_HARNESS_ALIAS = "_seal_harness_store"

# How long one of the harness's cells may run: the search chain's, which run
# the model's test and its path, and the one after every call, which runs an
# animation described on its own and the settled delta, two framediff
# searches each allowed ``frame_delta.TIME_LIMIT_S`` and able to overrun it
# (2026-09-29). Past it the game ends. Not the model's per-call
# limit: nothing of these cells counts against the model's time.
_HARNESS_CELL_TIMEOUT_S = 1200.0


@dataclass
class _SearchingPython(python_tool.BoundPythonTool):
    """The python tool as a turn offers it: the kernel's call, and after it
    the search, when the call earned one. The report joins the call's
    output."""

    after_call: Callable[[int, str], Awaitable[str | None]] = field(kw_only=True)
    actions_played: Callable[[], int] = field(kw_only=True)

    def _maybe_remap_single_arg(self) -> None:
        """ullm's lone-argument remap, minus a call that gives only its
        ``plan`` (:func:`seal.repl.plan_only`)."""
        if not plan_only(self.parameters, self.arguments):
            super()._maybe_remap_single_arg()

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Run the call, then the chain."""
        before = self.actions_played()
        content = await super()._execute(arguments)
        output = content.text if isinstance(content, ullm.conv.StringContent) else ""
        note = await self.after_call(before, output)
        if note is not None and isinstance(content, ullm.conv.StringContent):
            content = ullm.conv.StringContent(text=f"{content.text}\n\n{note}")
        return content


# The file suite the seal offers: the tools the builder makes and the names the
# prompt states both read this, so the two cannot describe different suites.
# `search_files` and `undo_edit` stay out, as they do on the shipped duck.
_FILE_SUITE: dict[str, bool] = {"search_files": False, "undo_edit": False}


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


#: The longest stop detail the kernel may hand the host: the text is
#: model-reachable (it quotes the model's own action and outcome values), and
#: it lands in the result ``submit_action`` returns.
_STOP_DETAIL_CAP = 2000

#: The stop reasons the kernel's world-model gate may end a stepped batch
#: with: the model mispredicted the move just played, or declined it.
_WORLD_MODEL_STOPS = {"world_model_mismatch": "mismatch", "world_model_unknown": "unknown"}


@dataclass
class _Batch:
    """One ``submit_action`` batch as it executes: what was staged, how far it
    got, and why it stopped. A value so a batch can pause between moves —
    parked on the agent while the kernel checks the new frame — and resume
    in the same loop that would have run it whole."""

    staged: list[Action]
    requested: int
    index: int = 0
    executed: list[Action] = field(default_factory=lambda: list[Action]())
    stopped_early: bool = False
    stop_reason: str | None = None
    stop_detail: str | None = None
    #: Set once a move ends the level or game: the reason when staged moves
    #: remain (they will not run), ``""`` when it was the batch's last move.
    terminal_reason: str | None = None
    #: How the kernel sent it: ``"checked"`` under a passing world model,
    #: ``"exploring"`` without one while exploration is open, ``"unchecked"``
    #: with no gate in the way.
    route: str = "unchecked"

    def stop(self, reason: str, detail: str) -> None:
        """Record an early stop."""
        self.stopped_early = True
        self.stop_reason = reason
        self.stop_detail = detail


@dataclass(kw_only=True)
class SealAgent(LLMAgent):
    """The seal. Every model-facing choice is a dataclass field; the
    defaults are the shipped configuration — set by measurement where there
    is one (``evidence.md``) and by judgement where there is not. Sampling
    knobs (temperature/top_p/max_tokens) live on the ``llm_config``
    sampling profile named by ``profile``.

    Everything after the knobs is per-game state, and there is nothing else
    on the instance — no client, no kernel pool, no diagnostics handle;
    those arrive per call on the context. That is what makes the template
    cheap to copy per (game, pass) and a pickle of a playing copy a
    complete description of where the game had got to (A1/A3)."""

    # --- configuration ---
    # The window, compaction target and reply reserve are one coordinate
    # system; the duck's ``design.md`` ("The context budget") has the
    # arithmetic and its ``evidence.md`` the readings behind it. A prompt
    # budget past the duck's, from seal sweep 2 (2026-10-12), as far
    # as the engine's native window allows: the summary after a side turn
    # needs this window, the side turn's extra, the reply reserve and the
    # summary's room under it (``_require_summary_fits``), and this window is
    # the largest that leaves them on every column. The budget plus the reply
    # cap and the safety margin. The longer contexts hold more cache, so the
    # columns run fewer slots than the duck's (``presets.WINDOW_SLOT_SHARE``).
    context_window_tokens: int = 181184
    # The window the engine was built with, read off its arguments by the
    # build before this agent settles its own onto them: the checkpoint's
    # native context on our columns. What ``summarize`` runs under, since its
    # summary request goes past the agent's window by an amount the agent
    # cannot know (``_derived_engine_window``). ``None`` on a hand-built seal;
    # a supplied engine that reports its window settles it at connection, and
    # cuts a stated one wider than itself (``_validate_host_backend``).
    engine_ceiling_tokens: int | None = None
    # Hysteretic compaction target. An over-budget prompt is compacted all
    # the way down to this many tokens, so the context runs as a sawtooth
    # between the budget and this target, and pays for a trim once per many
    # rounds. Must be positive and below the budget (raises otherwise). The
    # default sits within a reply reserve of the budget — a worst-case reply
    # can re-cross within a round, which trades sawtooth depth for retained
    # context, exactly as the best sweep arm ran it.
    compact_to_tokens: int = 33925
    # How a full context is compacted. ``prune`` is the journal-era eviction:
    # whole blocks from the oldest end, then the live turn's oldest rounds
    # (``compact_to_tokens`` and ``evict_turns`` are its knobs).
    # ``summarize`` has the model write a summary that replaces the history,
    # on the live request itself: the ordinary system prompt, tools and
    # reasoning effort, thinking closed by an injected ``</think>`` rather
    # than by the template, and the whole history -- so the request extends
    # the served ids and its prefill is cached. It never drops history to fit,
    # so the engine window has to hold it (see ``prepare_for_backend``), and
    # the summary may run to twice its target. The default summarizes
    # (2026-09-26; the duck's ``evidence.md``, "Sweeps 12 and 13"). A
    # summary needs a real reply reserve, so summarize refuses an agent built
    # without a reply cap (``_reply_reserve_tokens``); such an agent has to
    # prune.
    compaction_strategy: Literal["prune", "summarize"] = "summarize"
    # The summary's target size is the window over this.
    compaction_ratio: int = 16
    # Whether the summary is written after the model's own reasoning. Off, the
    # summary request injects the end of thinking and caps the summary at twice
    # its target; on, it opens on the ordinary generation prompt, so the model
    # reasons first, and the cap is six times the target, since the reasoning
    # counts against it. Only the final text becomes the summary. Either cap is
    # lowered further, per request, to what the engine window has left over
    # the request's prompt (``MIN_SUMMARY_ROOM_TOKENS``). Off, a reply that
    # comes back entirely empty, with nothing in the reasoning either, is
    # retried as if it were on.
    summary_thinking: bool = False
    # Whether the rebuilt history keeps the most recent tool call, with its
    # result, beside the status prompt and the summary: ``full`` -- that
    # response as served; ``without_reasoning`` -- the same, with its
    # reasoning left out; ``none`` -- no call. A call that does not fit the
    # window goes regardless.
    summary_tail: Literal["full", "without_reasoning", "none"] = "without_reasoning"
    # The order the rebuilt history tells its story in. Off, it opens on the
    # summary, then the status prompt and the kept call. On, it keeps the order
    # things happened in: the status prompt and the kept call (each where it
    # came), then the summary and the closing message, which the model wrote
    # and the harness took after both; a line ahead of them all says the game
    # is under way and that the summary follows. Off, a status prompt that opened the
    # turn with a level change or a session reset reads as news after the
    # summary, and the closing message then says the session is intact.
    # On from seal sweep 4 (2026-10-26; ``evidence.md``, "The
    # defaults of 2026-10-28").
    chronological_restart: bool = True
    # Optional model-specific token IDs banned only while writing a summary.
    # Empty keeps the gate provider-neutral; a deployment that knows its
    # tokenizer can categorically suppress tool-call markup here.
    compaction_forbidden_token_ids: tuple[int, ...] = ()
    # Whether the conversation rebuilt around a summary closes on a listing
    # of the kernel's working directory and of the objects the model defined
    # in its ``python`` session, and the summary request says it will. Inert
    # under ``prune``. Under ``persistence="round"`` the session is fresh at
    # every call, so the object half is always empty there. On: no evidence
    # either way, and it seems useful (2026-09-27; the duck's
    # ``evidence.md``, "Sweeps 12 and 13").
    add_files_and_repl_to_summary: bool = True
    # Whether the conversation rebuilt around a summary closes on the full
    # text of ``world_model.py`` as well, and the summary request says it
    # will. The model would otherwise read the file back first thing after
    # every summary; shown, the summary can spend itself on where the model is
    # wrong or open instead of restating it (2026-09-28). Only where a
    # world model is in use, and only once the file is the model's own: the
    # seeded example is a different game. Inert under ``prune``.
    add_world_model_to_summary: bool = True
    # Whether the seal keeps the journal: the ``memory`` tool the model
    # appends findings with, and their snapshot at the system prompt's tail.
    # Off, the tool is not offered and no sentence of any prompt speaks of it;
    # what then carries work across a compaction is the summary alone, since
    # eviction has nothing to carry (``prune`` without it is a seal with no
    # memory at all, legal and not something any column runs). Off: one place
    # for memory rather than two, on no evidence either way -- both of the
    # duck's sweeps leaned towards keeping it, neither significantly
    # (2026-09-27; the duck's ``evidence.md``, "Sweeps 12 and 13").
    journal: bool = False
    max_llm_calls_per_move: int = 0  # 0 = unbounded; bounds one analyzer entry, not the move (the duck's design.md)
    # The kernel's whole-call wall clock (the duck's design.md). A description
    # the model asks for stops at it, so it is long enough to give one most of
    # the time framediff's own searches get (2026-09-25).
    python_timeout_s: int = 180
    # Cap on REPL output carried back into the conversation (x4 = chars); the
    # model is told when output was cut. Seal sweep 4 read the score rising
    # with the cap across its whole range, and this is the range's top
    # (2026-10-26; ``evidence.md``, "The defaults of 2026-10-28").
    tool_output_tokens: int = 4000
    # How ascii_grid lays a board out; a sweep axis, not a model choice. Row
    # numbers because every other numbered text the model reads (view_file)
    # has them and the audit run's hand-rolled renderers added exactly this;
    # the column group is a tab every N cells, unset until swept.
    ascii_row_numbers: bool = True
    ascii_column_group: int | None = None
    # Which characters a board prints in: "hex", the colour's index as a hex
    # digit (0-9, a-f), or "letters", the duck's (W white, b blue, ...). The
    # board is indices either way (``.grid``, ``segment()``); this is how
    # ``ascii_grid`` shows it, what ``from_ascii_grid`` and framediff's code
    # blocks read and write, the prompt's legend and the index legend image.
    # Letters, which seal sweep 4 read ahead of hex (2026-10-26;
    # ``evidence.md``, "The defaults of 2026-10-28").
    ascii_symbols: AsciiSymbols = "letters"
    # Retained journal tail budget, in tokens: an exact ceiling of true
    # tokens in token-native mode (the backend tokenizes the rendered
    # journal), the long-standing nominal 3 chars/token calibration
    # elsewhere (see _journal_tokens).
    memory_context_tokens: int = 1000
    # Whether a new continuation user message identifies a max-token cutoff.
    # The previous generated response is byte-identical in both modes. On:
    # the reply cap is large enough that hitting it means a reply went wrong,
    # and a model that is not told it was cut off cannot tell.
    completion_cutoff_notice: bool = True
    # Eviction chunk, and the round-pruning overshoot. One rather than the
    # duck's two: the duck's sweep found it inert at its window, and the seal
    # starts from the smallest chunk until a sweep at its own window says
    # otherwise.
    evict_turns: int = 1
    # Attach the current grid to a user prompt as an image. On every prompt by
    # default, as on the duck (2026-09-26 and 2026-09-28; the duck's
    # ``evidence.md``, "Sweeps 12 and 13"). The board also reaches the model
    # where it can be worked on -- ``.grid``, ``ascii_grid`` and ``segment()``
    # inside the kernel -- and the image is a coarser second copy, paid for in
    # tokens and needing the checkpoint's vision tower
    # (``_validate_coherence``).
    #
    #   ""             off
    #   "current_grid" every user prompt
    #   "level_start"  the first user prompt of each level, restored after summarization
    #
    # `level_start` exists because the two costs differ in kind. An image on
    # every turn is paid every turn AND moves with the board, so each one
    # lands past the cached prefix; one per level is paid once and then sits
    # inside the prefix for the rest of the level. What it gives up is the
    # image tracking the board — the model sees where the level STARTED, not
    # where it is, which is why this is a third option rather than a cheaper
    # spelling of the second.
    multimodal_context: str = "current_grid"
    # Pixels a side per grid cell in the image; inert while the image is off.
    # 8 on weak evidence (2026-09-27; the duck's ``evidence.md``,
    # "Sweeps 12 and 13").
    multimodal_upscale: int = 8
    # Attach a colour-NAME legend to the first user prompt of each level,
    # wherever an image already travels, and restore it after summarization.
    # Off (2026-10-30): the index legend rides instead, the seal's
    # boards being indices.
    #
    # A different question from the index legend: that one explains what a
    # colour INDEX looks like; this one ties a swatch to the NAME the prompt
    # already uses, which an agent reading a picture of a board needs. The
    # whale sends its legend on exactly these terms. When on it REPLACES the
    # index legend rather than adding to it: two reference images per level
    # is a cost with no second question behind it.
    color_name_legend: bool = False
    # Whether the prompt after a move carries, after the board image, an image
    # of the cells the move changed, in their new colour and the rest navy:
    # where a move's effect is, at a glance. Compared with the frame framediff
    # described the move's frame from, where it did, so the image and the
    # block agree; otherwise with the world model's render when a checked move
    # missed it, and with the frame before the move when nothing checked it.
    # None for a winning move unless the level reflection follows, which reads
    # it, as the settled delta does; after a RESET or an UNDO back to an
    # earlier frame of the level; after a move predicted right (energy aside);
    # or when nothing changed (``harness._repl_state.diff_image_baseline``).
    # Only where the board image rides every prompt. Off until a sweep prices it
    # (2026-10-08; the competitor study's H60).
    diff_image: bool = False
    # Whether every board image (and the diff image) carries row and column
    # labels along its top and left edges, so a position on the image maps to
    # ``frame.grid[row, col]`` and a click's coordinates without counting
    # cells; the system prompt says so. The whale's labels (``seal.board_image``).
    # Off until a sweep prices it (2026-10-08).
    axis_labels: bool = False
    profile: str = "play"
    # The sampling profile every summary samples on, by name in
    # ``llm_config.sampling_profiles``. A summary keeps the live request's own
    # state (its seed, injection and recurrent-state keys) and takes the
    # profile's sampling: temperature, ``top_p``, ``after_reasoning`` and the
    # profile's ``extra_params`` over the live ones. None samples it on
    # ``profile``. A summary opens past the end of thinking unless
    # ``summary_thinking``, so an ``after_reasoning`` on the play profile would
    # otherwise sample the whole summary at the answer's settings.
    summary_profile: str | None = None
    max_moves: int | None = None  # give up after this many executed actions
    # What bounds a game, in place of a wall clock (the duck's ``design.md``,
    # "Run budgets"): completion tokens allowed per level, indexed by the level
    # the game is on, the last entry applying to every level past the end. So
    # ``(X,)`` allows X on every level and ``(X, Y)`` squeezes the first level
    # only. ``None`` disables the budget, which leaves the analyzer loop
    # unbounded — it re-enters until the model acts, and nothing else stops it.
    max_tokens_per_level: tuple[int, ...] | None = (90_000,)
    # Optional whole-game completion-token budget, counted from the backend's
    # per-game meter so it survives level changes and includes recovered
    # requests that are no longer present in the conversation. ``None`` keeps
    # the normal per-level policy as the only token budget. Unlike the
    # submission-only per-level lift below, this comparison control always
    # remains in force.
    max_tokens_per_game: int | None = None
    # Stop enforcing once no game is left waiting for a slot: giving up then
    # frees a slot with nothing to put in it, so the cap costs a game and buys
    # nobody anything. True where score under a deadline is the objective (the
    # Kaggle preset); false here, because a lifted cap makes what a game
    # achieves depend on where in the run it was scheduled, which is noise in
    # any comparison between runs.
    uncap_when_queue_drained: bool = False
    # Inert; accepted so existing constructions keep working. The
    # compaction decision runs on the engine's own billed ``usage`` plus
    # a provable bound for what was appended since (see "7. context
    # budget"), whatever this flag says — it selects nothing.
    exact_token_counts: bool = True
    # Re-samples allowed only when a backend rejects a malformed reply.
    # Inert when its accept_malformed_output policy retains the draw. Three,
    # before the nudge below, as ARC2's duck ships it (2026-10-06): over
    # its sweeps 6-8 the next draw succeeded after 1, 2 and 3 refusals 92%, 78%
    # and 45% of the time, and never after 4.
    malformed_retries: int = 3
    # How many times a game is nudged, rather than crashed, when a play reply
    # is refused on every one of those re-samples for ending with no text and
    # no tool call, or for degenerate repetition. The refused draws are
    # dropped -- in practice they loop inside their reasoning -- and the
    # nudge says the reply was discarded and asks for a different approach.
    # 0 never nudges: the error crashes the game, as it does past the count.
    # Capped because refused draws are not billed to the game, so its budget
    # would never stop a model that loops every time. On, as ARC2's duck
    # ships it: in its sweep 9 the nudge took crashed games from 14 of 21,735
    # to 1 of 22,365, at no score or token cost (that repo's ``evidence.md``).
    malformed_nudges: int = 1
    # Keep a failed draw and end on it rather than sample a replacement: a
    # trainer's ledger holds one draw per request, so a retry, or a summary
    # request re-sent after drift, is a second draw it cannot take. Requires
    # ``malformed_retries=0``; the contract is the duck's ``design.md``,
    # "Summarize".
    preserve_failed_draws: bool = False
    # Whether the game's UNDO is offered, where the game has it, as the duck
    # offers it: named among the game's actions and played like any move. Only
    # with ``use_world_model="none"``: games keep UNDO by different rules, and
    # a world model holding one of them in its state defeats the search's
    # dedup (2026-10-07). Off until a sweep prices it; where a game
    # has it, the duck's sweeps saw it used in most lanes, mostly in do/undo
    # loops, to no score.
    expose_undo: bool = False
    # Whether the python tool takes an optional ``plan``: one line on what the
    # call's actions should achieve and what would confirm it, kept on every
    # move the call plays as ``history[i].plan``. "first" asks for it ahead of
    # ``code``, so it is reasoning the code then follows; "second" after it.
    # Only the schema's order and the description differ: the schema takes
    # either order. Off until a sweep prices it (the competitor study's H42).
    plan_argument: PlanArgument = "none"
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
    # (Mistral Large 4: 3.9-4.3 characters a token on seal traffic); text
    # that tokenizes worse than calibrated can undercount, which the window
    # invariant (``_check_engine_usage``) catches on the billed totals.
    bound_chars_per_token: float = 1.0
    # Fixed engine-side token cost charged per prompt image, overriding
    # the derivation in ``_image_bound_tokens``. None = derive from the
    # image's pixel dimensions (see that method for the arithmetic).
    image_tokens_override: int | None = None
    # The billed total the window invariant holds each response to. None =
    # the agent's own window. A hosted teacher held to a student's window by
    # the calibrated estimate above sets the API's window here: its bills
    # count in its own tokenizer, so one past the student's window is the
    # estimate missing (Kimi bills an image at roughly twice the student's
    # 258), which the next anchor heals, not an oversized sequence. Never
    # tighter than the agent's own (2026-10-27: ball_bounce failed
    # 345 tokens over the student's window, its context 93 images deep).
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
    # Observes each processed ordinary or summary response before its tools
    # run. The synchronous callback must be picklable with the per-game agent.
    # It may persist response evidence; exceptions propagate and stop the turn.
    response_observer: Callable[[ullm.conv.LLMResponse], None] | None = field(default=None, repr=False)
    _context_compacting: bool = field(default=False, init=False, repr=False)
    # Ask the engine to zero the recurrent state at `zero_recurrent_at` on
    # every call: the model then decodes with verbatim (attention) recall
    # of the whole prompt but a running (recurrent) summary of only what
    # follows the position. What the blank buys is a change of REGISTER --
    # the model drops into a telegraphic form ("We need solve game.") and
    # stops ruminating -- and the register is where the points are
    # (the duck's evidence.md, "What the blank was actually worth"). OFF by default
    # because serving the register directly (`inject_text_on_first_prompt`,
    # below) reaches it on a stock engine; the blank stays as the one route
    # to the register through the STATE rather than the tokens. It requires
    # the arc-prize-2026-vllm fork this repo pins, which REJECTS the parameter on
    # a stock engine rather than ignoring it -- so a mismatched engine stops
    # the run instead of quietly scoring one that never blanked.
    zero_recurrent: bool = False
    # Where the blank lands, in absolute prompt tokens: the first recurrent
    # block boundary, which the shared system prompt spans, so what the
    # recurrent layers forget is the game overview and tool interface and
    # everything from the solving guidance on is read fresh. Measured as
    # the position that reproduces the effect, not tuned for score; the
    # position sweep is a roadmap item. Any position inside the prompt is
    # applied; one the prompt does not reach is refused by the engine.
    # Each call's generation cap is also the context budget's reply reserve.
    # None leaves generation uncapped and reserves a nominal 512 tokens.
    reply_cap_tokens: int | None = 32768
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
    # one image of the processor's largest size, 16,384 tokens on this
    # checkpoint; an agent that renders every image itself knows its own
    # largest, and the difference is KV pool. Past the ceiling the processor
    # SHRINKS an image rather than refusing it, so the ceiling has to follow
    # the render rules, not be stated beside them. The same settle refuses
    # video, which the seal never sends, so the multimodal profile and caches
    # size for images alone.
    derive_max_pixels: bool = True
    zero_recurrent_at: int = 1568
    # Text served after the generation prompt of the game's FIRST model
    # call only, which the model then continues from (on a thinking
    # template it lands right inside the opened think block). Empty = off.
    #
    # The shipped value is the opening of the telegraphic register the model
    # writes in when its recurrent state is blanked, and serving it is the
    # way INTO that register: the sampler reaches it unprompted on some
    # lanes, and serving the register's own first words as the model's own
    # tokens puts every lane there, where asking for it in the prompt puts
    # none (the duck's evidence.md, "Reaching it from the front instead").
    #
    # Three words, which is the shortest opening measured to induce the
    # register in every lane: at two the induction fails outright in about
    # one lane in thirty, and it fails in the configuration that ships, where
    # nothing blanks the state to catch it (the duck's evidence.md, "How
    # short the opening can be"). Shorter than the sentence it replaces because every
    # forced token is a continuation a trainer cannot improve on, so the
    # opening is cut to where reliability stops paying for it.
    #
    # Served by the in-process vLLM backend alone; any other backend refuses
    # the call, so an agent on a cloud model, a scripted stub or an OpenAI
    # server must set this to "". In estimate-mode budgeting its tokens fall
    # outside the character-derived bound and are absorbed by
    # `safety_margin_tokens`, like a retry perturbation; token-native mode
    # weighs them exactly.
    inject_text_on_first_prompt: str = "We need solve"
    # Text served the same way on EVERY call after the first. Empty = off.
    # Exists because the first-call opening does not hold a game: it is an
    # example left in history, and compaction drops it. There is no
    # late-game analogue of the opening to serve -- concise and verbose
    # blocks open alike once the game is under way -- so what goes here has
    # to fit any moment: the opening's own first token, content-free and
    # enough on its own. What sustains the register is the schedule, not the
    # words (the duck's evidence.md, "Repetition is what sustains it").
    #
    # One token, where the opening needs three. The jobs differ: the opening
    # induces the register from cold, this one re-induces it with the
    # register already in the context that compaction is eating away at, and
    # measured in the tail the shorter string holds it as well as the longer
    # (the duck's evidence.md, "How short the per-call reminder can be"). So the token
    # goes back to the model, whose continuation is then that much less
    # decided for it.
    inject_text_on_later_prompts: str = "We"
    # "context" opens each declared history epoch with the first text.
    # Counting a prompt does not advance this schedule; only a prepared
    # model round does. The ordinary game-first/later schedule stays default.
    prompt_injection_schedule: Literal["game", "context"] = "game"
    # How long the python kernel's state (module state, globals, tmpdir)
    # survives. "round" is a fresh process per call;
    # "move"/"level"/"always" let the model build up utilities and keep
    # results across a move/level/game instead of rewriting them every snippet.
    #
    # This is a SNAPSHOT constraint, not just a prompt one: a kernel that
    # outlives a move cannot be pickled at a move boundary (A4), so the
    # longer-lived modes narrow what `supported_checkpointing()` reports.
    #
    # "level" is the default. Levels of one game share their mechanics but
    # not their board, so a namespace built around the old level is likelier
    # to mislead than to help. What the model wants to keep goes in a file,
    # which crosses the boundary; what it inferred goes in the journal.
    persistence: Literal["round", "move", "level", "always"] = "level"
    # Whether the model must hold a world_model.py to act. "none" removes
    # the surface entirely: no test_world_model, no gate. "mandatory"
    # validates world_model.py silently at the first move of every cell; a
    # passing model is held to every new frame, and a mispredicted one stops
    # the batch at that move. Without a passing model only exploratory moves
    # run (below); an all-wildcard render passes. "optional" checks the same
    # way, but without a passing model a batch runs unchecked: there is no
    # exploration, and the prompt says when a model is worth writing.
    # Optional by default (2026-09-25), ahead of the sweeps that
    # price the three against each other.
    use_world_model: Literal["none", "optional", "mandatory"] = "optional"
    # With an optional world model, the prompt states attempting one as the
    # default, dropped only once the model is convinced it does not help on
    # the game: without it the Kimi teacher wrote no world model in its first
    # five games (2026-10-27). On in ``KIMI_STUDENT``, as the teacher
    # traces were played; off, no such line.
    world_model_default: bool = False
    # The exploratory moves a model without a passing world model may make:
    # one move per cell, while any are left, until its first passing model
    # ends exploration for the level. A game starts with more than a level
    # does, because the first level is where the game's mechanics are first
    # seen; each new level replaces what is left rather than adding to it.
    exploration_moves_new_game: int = 10
    exploration_moves_new_level: int = 5
    # What the model is given as blocks of numpy (seal.frame_delta), three
    # independent switches. The settled delta: a new frame the model did not
    # predict, or predicted wrongly, and a render a test_world_model() fails
    # on, explained from a frame it knew. The animation description: a move's
    # animation frames rebuilt the same way, on request ("available"), or on
    # request and for a call's longest animation on its own ("auto"). The
    # marks: which executed moves had animation frames, on the executed-actions
    # line. Reading a frame from pixels is where a model's turns go; the block
    # is the same fact in a few lines.
    settled_delta: bool = True
    animation_description: AnimationDescription = "available"
    animation_marks: bool = True
    # Whether the executed-actions report lists a batch's moves one per line,
    # each with how many cells it changed and their colours before and after.
    # The settled delta explains only the batch's last move, so a move inside
    # a batch that did something can read as one that did nothing (m0r0 in the
    # combined smoke: a dragged dot moved and came back unseen). On from seal
    # sweep 4 (2026-10-26; ``evidence.md``, "The defaults of
    # 2026-10-28").
    move_changes: bool = True
    # Whether the prompt says that ``history`` already holds what replaying a
    # sequence would show, and the turn after a RESET that closed an attempt
    # repeating an earlier one on the level, move for move from its start,
    # reminds the model it learned nothing and that the level is winnable.
    # Replaying and then trying something new is legitimate, so only the
    # RESET closing a repeat fires it. One switch for both; on since seal
    # sweep 2 (2026-10-12).
    replay_reminder: bool = True
    # How long an animation's block may run: the limit its stride is chosen
    # to fit, and where a block at a stride the model chose is cut off.
    animation_max_chars: int = MAX_CHARS
    animation_cut_chars: int = ANIMATION_CUT_CHARS
    # The framediff search behind every block: its budgets (``max_bits`` for
    # a frame, ``animation_max_bits`` for an animation's frame), its effort
    # and its time limit, framediff's defaults with the seal's time limit
    # (``frame_delta.default_decomposer``). And the budget a frame's block is
    # shown within once a world model has passed on the level, in place of
    # ``max_bits``. A sweep sets the decomposer's fields directly.
    decomposer: Decomposer = field(default_factory=default_decomposer)
    max_bits_understood: float = MAX_BITS_UNDERSTOOD
    # Whether a framediff search that reaches its time limit after a call (the
    # settled delta, an animation described on its own) ends the game. Off,
    # the best block it found by then is shown as any other, the timeout on
    # the span alone, and the game plays on (2026-09-29; the best
    # block 2026-10-08): the pilot's four such crashes were searches a node's memory
    # stalls had slowed, not runaways, and a lost game costs a sweep more than
    # an undescribed frame. On, a runaway search is found rather than
    # absorbed. A search inside the model's call stops before the call's
    # limit and never ends the game.
    crash_on_framediff_timeout: bool = False
    # Whether the model has ``segment()`` (with ``Segmentation`` and
    # ``Node``) and the prompt points it there as its first view. Off, the
    # kernel does not have them and the prompt names no starting point. A
    # switch for a quick test (2026-10-08): a game that plays worse
    # without segmentation says structure helps, which motivates scene
    # decomposition and framediff work even where today's methods do not.
    segmentation: bool = True
    # Whether the operating system confines the kernel's files
    # (``seal.landlock``): reads to the system, Python and its libraries
    # less our packages, the engine and the games; writes to the working
    # directory and the kernel's HOME. The source guard alone is patched
    # Python, which a shell-out or ``os.open`` walks around, and models in
    # the long runs did so to read the games' code; and nothing else stops a
    # model deleting what the run's user can. On, a machine without Landlock
    # cannot start a kernel, rather than play unconfined.
    confine_filesystem: bool = True
    # The search a passing test earns (``seal.search``): after a python call
    # whose last test_world_model() passed and which played no move, the
    # model is searched for a win or for a move it does not predict. How long
    # the search may run, and the memory its child process may add once its
    # imports are loaded — what bounds the searches' memory together, since a
    # machine runs at most ``search_process.SEARCH_SLOTS`` of them. Both bound
    # the child's load and test of the model and its replay of the level as
    # well as the search, so the time carries room for them
    # (2026-10-15; a sweep sizes it).
    search_seconds: float = 130.0
    search_memory_bytes: int = 2 * 2**30
    # Whether the search runs at all. Off, the rest of the world model is
    # unchanged — the test, the per-move check, the moves before a model
    # passes — so a run with it off is the same agent without the search.
    use_search: bool = True
    # How a world model's renders are held to the frames, in
    # test_world_model(), the check on every move and the host's test alike
    # (``seal.harness.test.MatchRules``): the starting values of the settings
    # the model changes in the REPL, which last the game. Strict energy holds a
    # frame to the model's own ``state['energy']``; off, a frame passes under
    # any energy value that makes it match, as the bar is the board's least
    # mechanical region. The pixel tolerance passes a frame the render misses
    # by at most that many cells, either way; whether the model is offered
    # ``set_pixel_tolerance`` at all is ``expose_pixel_tolerance``, and a
    # tolerance only starts above 0 where it is offered. Strict energy off
    # until a sweep prices it (2026-10-07); the setter offered, the
    # tolerance starting at none, from seal sweep 4, which drew both
    # (2026-10-26; ``evidence.md``, "The defaults of 2026-10-28").
    strict_energy: bool = False
    pixel_tolerance: int = 0
    expose_pixel_tolerance: bool = True
    # A turn after a move that wins a level (not the game), ahead of the next
    # level's first: the move reported as after any move, then a request to
    # reflect on the level and answer three questions, with Python and the
    # file tools and without submit_action. The kernel's history is still the
    # finished level's (``_kernel_level_entries``). No compaction runs inside
    # it: a summary due waits for the first call after it. Charged to the new
    # level. ``verbose`` asks for an extensive reflection with three
    # questions, ``concise`` for a brief one with none, ``none`` runs no turn.
    # Verbose from seal sweep 4: it spends less a reflection than concise,
    # whose turn goes to world-model edits, and is ahead of it by the Kaggle
    # deadline (2026-10-26; ``evidence.md``, "The defaults of
    # 2026-10-28").
    level_reflection: SideTurnMode = "verbose"
    # How far past ``context_window_tokens`` that turn's context may grow
    # before a summary is due: then the turn ends, the model is told, and the
    # summary comes with the next level's prompt. The window, this, the reply
    # reserve and the summary's room must fit the engine's window
    # (``_require_summary_fits``). A first guess (2026-10-07).
    reflection_extra_window_tokens: int = 40_000
    # The strategy audit: once a level's spend passes one of these thresholds
    # without the level cleared, a turn, not for acting, asks the model to
    # step back from it, with advice, the actions it has not tried, and two
    # questions; play then goes on. Checked before every model call, so it
    # interrupts a turn after a call that played no move, and follows the next
    # move's status prompt after one that did. The spend is the level's as its
    # budget counts it, less the reflection turn's; a threshold the audit's
    # own spend passes counts as fired. Ascending. One threshold, from seal
    # sweep 2 (2026-10-12): it fires only where a level may spend
    # past it, so not under a per-level cap below it.
    audit_level_tokens: tuple[int, ...] = (300_000,)
    # Whether the audit runs at those thresholds, and how it asks: ``verbose``
    # with two questions, ``concise`` for a brief reflection with none,
    # ``none`` not at all. Concise from seal sweep 2 (2026-10-12).
    strategy_audit: SideTurnMode = "concise"
    # How far past ``context_window_tokens`` an audit's context may grow
    # before a summary is due, as ``reflection_extra_window_tokens`` for the
    # reflection: then the audit ends, the model is told, and the summary
    # comes with the call after it. A first guess (2026-10-07).
    audit_extra_window_tokens: int = 40_000
    # Completion tokens without a move after which every tool result in a
    # playing turn ends with a note saying how many have passed and
    # suggesting experiments over more thinking; 0 is off. Counted from the
    # last move, summary, or reflection or audit end, on the game's own meter
    # (``_moveless_since``). Not in a side turn, which cannot move. Off until
    # a sweep prices it (2026-10-15).
    suggest_moves_after_tokens: int = 0
    # What the system prompt asks of the model's reasoning depth. The class
    # default is provider-neutral — "high" is a level every backend that
    # consumes the value accepts — because this field reaches whatever
    # backend the seal is paired with, not only the shipped presets. The
    # Qwen3.8 presets inject "xhigh" (the model's own default, the swept
    # configuration) at construction; a hand-built seal states its own.
    reasoning_effort: Literal["low", "medium", "high", "max", "xhigh"] | None = "high"
    # The prompt implementation: which WORDING the agent speaks. A swap seam —
    # a variant surface drops in as a different instance (the whale's
    # Modeler.prompts pattern). Only its wording matters: the mode fields on
    # whatever instance is carried are stamped from the agent's own fields at
    # every use (_prompts), so configuration cannot be set on two objects.
    prompts: Prompts = field(default_factory=SealPrompts)
    # Infra rather than model-facing, and the one run-level default the seal
    # states for itself: the per-game wall net, `presets.GAME_RUNTIME_CAP_S`.
    # A preset replaces this whole object rather than editing it, so each one
    # says what it wants — both served presets take the cap, Kaggle takes
    # None.
    run_config: RunConfig = field(default_factory=presets.seal_run_config)

    # --- state: the kernel and the conversation (pickles; template
    #     deepcopies cheaply) ---
    repl: GameREPL | None = None
    repl_reset_pending: str = ""  # reason for a kernel reset the model has not been told about yet
    # Built with the first tool list that needs it, and kept: the undo log
    # lives on it, so a workspace rebuilt per turn would have nothing to
    # take back. It reads its directory off the repl at call time, so a
    # restored agent's fresh tmpdir needs no re-pointing.
    workspace: Workspace | None = None
    messages: list[ullm.conv.ConversationElement] = field(default_factory=lambda: list[ullm.conv.ConversationElement]())
    memory_entries: list[MemoryEntry] = field(default_factory=lambda: list[MemoryEntry]())
    memory_entry_count: int = 0
    # Frozen at the system-prompt tail. New entries reach the model through
    # their immutable tool results; compaction refreshes this snapshot once,
    # instead of copying the whole journal into every state prompt.
    memory_prompt_snapshot: str | None = None
    response_step: int = 0

    # --- state: the game as the model is shown it ---
    # Whether the game's start line has gone to stdout (`_announce_start`).
    _start_announced: bool = False
    # Seed entry + one per action. Each entry carries the settled frame and
    # the complete animation that arrived there, ending in that settled frame.
    history_entries: list[TimelineEntry] = field(default_factory=lambda: list[TimelineEntry]())
    # The level's attempts, for the reminder after one repeated (``replay_reminder``).
    attempts: Attempts = field(default_factory=Attempts)
    current_valid_actions: list[str] = field(default_factory=lambda: list[str]())  # model-facing names
    valid_actions_observed: bool = False  # distinguishes known-empty from not observed yet
    # The game's action set, frozen at the first observation. Separate from
    # the per-turn list above because a prompt surface may name it in the
    # SYSTEM prompt, where the currently-legal subset would rewrite a stable
    # prefix on every refusal.
    game_action_names: list[str] = field(default_factory=lambda: list[str]())
    last_action_result: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    last_step_summary: StepSummary | None = None
    action_count: int = 0
    last_levels_completed: int = 0  # for the per-action reward delta
    # The level budget's counter, and the level it belongs to. Plain fields
    # rather than the whale's shared-counter object: nothing here rolls state
    # back, so there is no attempt whose spend has to survive being undone.
    level_tokens_used: int = 0
    budget_level: int = 0
    # The exploration budget and the level it belongs to. Over once a
    # world model has passed on the level, whatever is left.
    exploration_level: int | None = None
    exploration_moves_left: int = 0
    exploration_over: bool = False
    # Whether the example world_model.py has been written for this game.
    world_model_seeded: bool = False
    #: The newest frame's delta text, for the next user prompt, then cleared.
    frame_delta_text: str | None = None
    #: The animation block a call earned, and the turn it describes, for the
    #: next user prompt, then cleared.
    animation_text: str | None = None
    animation_described: int | None = None
    animation_concise: bool = False
    #: The diff image the last move earned, as the kernel sent it, for the
    #: next prompt that shows a board, then cleared.
    diff_image_pending: dict[str, Any] | None = None
    # The python call (``GameREPL.cells_started``) that made the last
    # exploratory move; one per call.
    explored_in_cell: int = -1
    # The matching settings as the model has set them this game; None until it
    # sets one, which reads as the starting value.
    matching_strict_energy: bool | None = None
    matching_pixel_tolerance: int | None = None

    # --- state: this move ---
    awaiting_action: Action | None = None  # the action we just returned, until its observation arrives
    pending_batch: PendingBatch | None = None  # drained-but-unreported executions
    pending_level_win: PendingLevelWin | None = None  # deferred to the new level's frame
    pending_level_reset: bool = False  # a level was won; the kernel reset it owes waits for the next decision move
    # The kernel's verdict on a winning move it checked: the world model passed
    # before the call and did not predict the win. Read by the reflection turn.
    win_missed: bool = False
    # The strategy audit's thresholds the level has passed, and the reflection
    # turn's spend on the level, which they do not count. Both start over with
    # a level.
    audits_fired: int = 0
    level_reflection_tokens: int = 0
    # A side turn (the reflection, the audit) while it runs: which, the
    # backend's completion meter when it began, and whether its context
    # reached its window and ended it.
    _side_turn: str | None = None
    _side_turn_started_at: int = 0
    _side_turn_cut: bool = False
    # The backend's completion meter at the last move, summary, or reflection
    # or audit end: what ``suggest_moves_after_tokens`` counts from; and the
    # action count it was taken at, by which a move is noticed
    # (``_suggest_moves``).
    _moveless_since: int = 0
    _moveless_moves: int = 0
    # A turn block stopped for the audit, which the entry runs next; and the
    # threshold it is for.
    _audit_due: bool = False
    _audit_threshold: int = 0
    game_over_active: bool = False  # newest observation is a loss (RESET-only until cleared)
    # In-snippet execution bookkeeping. ``last_obs_index``: highest
    # Observation.index already folded — in-move executed observations come
    # back through ctx.execute AND as the next move's delivered observation,
    # and this guard makes the second sighting a no-op. The rest is per-move
    # transient state (reset at each decision move).
    last_obs_index: int = -1
    executed_this_move: int = 0
    terminal_this_move: dict[str, Any] | None = None  # first terminal action result this move
    # A batch the kernel is stepping move by move (its world model is armed),
    # paused between moves while the kernel checks the new frame.
    paused_batch: _Batch | None = None
    live_available_actions: list[str] = field(
        default_factory=lambda: list[str]()
    )  # engine names, refreshed per execute

    # --- state: context accounting ---
    # The level the palette legend was last attached for, so it rides one
    # prompt per level rather than every turn. -1 = not yet sent.
    palette_level_sent: int = -1
    #: The level whose first prompt already carried the grid image, for
    #: ``multimodal_context="level_start"``. Summary compaction restores that
    #: image; pruning can evict it along with its user-led block.
    image_level_sent: int = -1
    # Share the prompt's original image so summaries preserve the level's
    # reference board even after it leaves the conversation history.
    _level_start_image: ullm.conv.ImageContent | None = field(default=None, repr=False)
    png_len_cache: dict[str, int] = field(default_factory=lambda: dict[str, int]())  # image estimate (data-URL chars)
    counts_reconciled: int = 0  # billed usages checked against the window invariant (0 = the check never ran)
    context_summaries: int = 0
    _pruned_rounds: int = 0  # rounds dropped from the CURRENT turn, for the note
    _omitted_tool_calls: int = 0  # tool calls of the CURRENT turn a summary replaced, for its note
    # The user prompt that started the current turn: the analyzer entry's
    # status prompt, which a summary keeps. A nudge opens a block, not a turn.
    _turn_prompt: ullm.conv.UserPrompt | None = field(default=None, repr=False)
    _last_trim_fired: bool = False  # did the last trim fire? read by the diagnostics hook below
    _count_drift_events: int = 0  # count-path HistoryDriftError degrades this run, carried on the span kv
    _malformed_nudged: int = 0  # how many of ``malformed_nudges`` this game has spent
    # A turn block dropped a reply refused on every re-sample, and the entry
    # sends the malformed nudge next.
    _malformed_nudge_due: bool = False
    # The live Conversation for the current turn block, cleared in the
    # block's finally — the decision path edits its system prompt (memory
    # snapshot refresh) and declares its history rewrites on it, and a live
    # conversation must never ride a snapshot (the agent pickles at move
    # boundaries).
    _live_conversation: Any = None
    # The block's backend and sampling params, for token-native measurement
    # only; cleared with the conversation so no backend rides a snapshot.
    _live_backend: Any = None
    _live_params: Any = None
    # Compaction spacing, for the per-trim diagnostics: the decision reading
    # left after the previous fired trim, and the LLM calls made since.
    _tokens_at_last_compaction: int | None = None
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
    # it (eviction, pruning, the refreshed memory snapshot in the system
    # prompt): the conversation's ``history_epoch`` (ullm.continuation),
    # kept here because each block hosts the persistent element list in a
    # fresh Conversation. A version, not a count of individual edits.
    history_epoch: int = field(default=0)
    _prompt_injection_epoch: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Move-boundary pickle: drop the staleness record -- a list of
        process-local objects that match nothing after a restore -- and the
        engine thread name. A copy is a player of its own, so it draws its own
        name at its next request (``_conversation_id``) rather than carrying
        the source's. Hygiene only: the engine's thread tag also carries the
        game's run id, which already keeps games branched from one checkpoint
        apart."""
        state = dict(self.__dict__)
        state["_stale_responses"] = None
        state.pop("_thread_conversation_id", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore, then mark every restored response stale: a pre-restore
        bill may describe a prompt a pre-boundary trim already rewrote, so
        nothing anchors the decision until the next fresh response
        ("trust after verify")."""
        vars(self).update(state)
        # The game's backend meter starts again at zero wherever the agent is
        # restored (``ullm.backend.Backend.__getstate__``), so the mark does
        # too: a restore is a level's start, past its winning move.
        self._moveless_since = 0
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
        profile = self.llm_config.sampling_profiles.get(self.profile)
        if isinstance(profile, ullm.backend.GenericSamplingParameters) and profile.max_tokens is not None:
            raise ValueError("Play profile max_tokens must be unset; set SealAgent.reply_cap_tokens instead")
        if self.ascii_symbols not in ASCII_SYMBOLS:
            raise ValueError(f"unknown ascii_symbols {self.ascii_symbols!r}; expected one of {sorted(ASCII_SYMBOLS)}")
        if self._multimodal_mode() not in self.MULTIMODAL_MODES:
            raise ValueError(
                f"unknown multimodal_context {self.multimodal_context!r}; "
                f"expected one of {sorted(self.MULTIMODAL_MODES)}"
            )
        if self.summary_tail not in {"full", "without_reasoning", "none"}:
            raise ValueError(f"unknown summary_tail {self.summary_tail!r}")
        if self.plan_argument not in PLAN_ARGUMENT_MODES:
            raise ValueError(f"unknown plan_argument {self.plan_argument!r}; expected one of {PLAN_ARGUMENT_MODES}")
        if self.malformed_nudges < 0:
            raise ValueError(f"malformed_nudges must be 0 or more; got {self.malformed_nudges}")
        if type(self.suggest_moves_after_tokens) is not int or self.suggest_moves_after_tokens < 0:
            raise ValueError(
                f"suggest_moves_after_tokens must be a whole number, 0 (off) or more; "
                f"got {self.suggest_moves_after_tokens!r}"
            )
        if type(self.pixel_tolerance) is not int or self.pixel_tolerance < 0:
            raise ValueError(f"pixel_tolerance must be a whole number, 0 or more; got {self.pixel_tolerance!r}")
        if any(type(t) is not int or t <= 0 for t in self.audit_level_tokens) or list(
            self.audit_level_tokens
        ) != sorted(set(self.audit_level_tokens)):
            raise ValueError(
                f"audit_level_tokens must be positive and ascending, without repeats; got {self.audit_level_tokens}"
            )
        for name in ("level_reflection", "strategy_audit"):
            if getattr(self, name) not in SIDE_TURN_MODES:
                raise ValueError(f"unknown {name} {getattr(self, name)!r}; expected one of {SIDE_TURN_MODES}")
        if self.audit_extra_window_tokens < 0:
            raise ValueError(f"audit_extra_window_tokens must be 0 or more; got {self.audit_extra_window_tokens}")
        if self.reflection_extra_window_tokens < 0:
            raise ValueError(
                f"reflection_extra_window_tokens must be 0 or more; got {self.reflection_extra_window_tokens}"
            )
        self._reply_reserve_tokens()

    def _validate_coherence(self) -> None:
        """Reject combinations no single field can be judged on.

        Each of these spans the agent, its config, the factory or the engine
        actually installed, so none of them can be settled at construction: a
        value assigned afterwards would never be seen. They run at
        ``prepare_for_backend``, which is after every override and before the
        engine starts, so a refusal still costs nothing.
        """
        if self.diff_image and self._multimodal_mode() != "current_grid":
            raise ValueError(
                f"diff_image=True requires multimodal_context='current_grid' (got {self.multimodal_context!r}): "
                "the image sits beside the board image, which rides every prompt only there"
            )
        if self.axis_labels and not self._multimodal_enabled():
            raise ValueError("axis_labels=True requires images (multimodal_context): the labels are drawn on them")
        if self.axis_labels and self.multimodal_upscale < AXIS_MIN_SCALE:
            raise ValueError(
                f"axis_labels=True requires multimodal_upscale >= {AXIS_MIN_SCALE} (got {self.multimodal_upscale}): "
                "a smaller cell leaves no room for legible labels"
            )
        if self.preserve_failed_draws and self.malformed_retries != 0:
            raise ValueError(
                "preserve_failed_draws=True requires malformed_retries=0: "
                "failed responses are retained and terminal, rather than rolled back for a retry"
            )
        if self.pixel_tolerance and not self.expose_pixel_tolerance:
            raise ValueError(
                f"pixel_tolerance={self.pixel_tolerance} requires expose_pixel_tolerance=True: a tolerance the model "
                "cannot see or change would pass frames it is never told about"
            )
        if self.expose_undo and self.use_world_model != "none":
            raise ValueError(
                f"expose_undo=True requires use_world_model='none' (got {self.use_world_model!r}): "
                "a world model has no contract for UNDO"
            )
        if self.preserve_failed_draws and self.malformed_nudges:
            raise ValueError(
                "preserve_failed_draws=True requires malformed_nudges=0: a retained draw is terminal, "
                "while the nudge drops it"
            )
        if self.level_reflection != "none" and self.journal:
            raise ValueError(
                f"level_reflection={self.level_reflection!r} requires journal=False: the journal asks for its own "
                "entry on a won level (SealPrompts.level_summary_request), and the reflection turn asks for the same"
            )
        if self.strategy_audit != "none" and not self.audit_level_tokens:
            raise ValueError(
                f"strategy_audit={self.strategy_audit!r} needs audit_level_tokens: with no threshold it never runs"
            )
        notes_guidance = getattr(self.prompts, "notes_guidance", "none")
        if notes_guidance != "none" and self.journal:
            raise ValueError(
                f"SealPrompts.notes_guidance={notes_guidance!r} requires journal=False: the journal's own guidance "
                "already says where findings go and what files are for"
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
        if self._multimodal_enabled() and getattr(engine_args, "language_model_only", False):
            # A column that drops the tower for a text-only agent, then an image
            # asked for after the build: the engine would refuse the first
            # image request, on the box where that is found out last.
            raise ValueError(
                f"multimodal_context={self.multimodal_context!r} sends the grid as an image, but the engine "
                "is built without its vision tower (language_model_only): keep the tower "
                "(ullm.backends.deployment.drop_vision_tower is what removes it), or set multimodal_context=''."
            )
        factory = self.llm_config.backend_factory
        if (
            self._multimodal_enabled()
            and self.llm_config.exact_history
            and not getattr(factory, "continues_images", True)
        ):
            # The server topology continues a history over /v1/completions,
            # which takes no image: the first request with one would be refused.
            raise ValueError(
                f"multimodal_context={self.multimodal_context!r} sends the grid as an image, but "
                f"{type(factory).__name__} cannot continue a history with an image by tokens under exact "
                "history: serve it directly (serving='direct'), or set multimodal_context=''."
            )
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

    def prepare_for_backend(self) -> None:
        """Derive local engine limits from the agent's final configuration.

        Supplied engines require explicit opt-outs because their limits are
        owned by the caller. Replay has no factory and needs no derivations.
        Under ``summarize`` the window derived is the engine's own ceiling
        (``_derived_engine_window``).
        """
        super().prepare_for_backend()
        self._validate_fields()
        factory = self.llm_config.backend_factory
        engine_args = getattr(factory, "engine_args", None)
        if factory is not None and engine_args is None:
            for flag in ("derive_max_model_len", "derive_max_num_seqs", "derive_max_pixels"):
                if getattr(self, flag):
                    raise ValueError(
                        f"{flag}=True, but the engine isn't built here: the supplied factory "
                        f"has no engine_args. Set {flag}=False and own that engine limit explicitly."
                    )
        if engine_args is not None:
            baseline = getattr(factory, "_derivation_baseline", {})
            if self.derive_max_model_len:
                self._settle_engine_arg(
                    engine_args, baseline, "max_model_len", self._derived_engine_window(), "derive_max_model_len"
                )
            if self.derive_max_num_seqs:
                self._settle_engine_arg(
                    engine_args, baseline, "max_num_seqs", self.run_config.max_concurrent, "derive_max_num_seqs"
                )
            if self.derive_max_pixels:
                # The processor's other arguments are the engine's own (a
                # checkpoint's tokenizer quirk, say) and ride along untouched.
                current: dict[str, Any] = dict(getattr(engine_args, "mm_processor_kwargs", None) or {})
                self._settle_engine_arg(
                    engine_args,
                    baseline,
                    "mm_processor_kwargs",
                    {
                        **current,
                        "max_pixels": largest_image_pixels(
                            upscale=self.multimodal_upscale, axis_labels=self.axis_labels
                        ),
                    },
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
        tokens. So the window is the ceiling the engine was built with -- the
        checkpoint's native window on our columns, which costs nothing in the
        pool -- and what the agent does own (the window and the reply reserve)
        has to fit under it with ``MIN_SUMMARY_ROOM_TOKENS`` to spare, or the
        derivation refuses rather than pick a number the run can exceed. The
        summary itself never passes the ceiling: each request's cap is lowered
        to what the ceiling leaves over its measured prompt.

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
        self._require_summary_fits(ceiling)
        return ceiling

    def _require_summary_fits(self, ceiling: int) -> None:
        """Refuse a summarize configuration whose own share of the summary
        request -- the window and the reply reserve -- leaves the engine
        window ``ceiling`` less than ``MIN_SUMMARY_ROOM_TOKENS`` for the
        summary.

        The summary's own cap (twice or six times its target) is not part of
        it: each request's cap is lowered to what the ceiling has left over
        the request's measured prompt (`_summary_completion_cap`), so the cap
        cannot pass the ceiling, and a configuration only has to leave the
        summary room in the worst case the agent owns (2026-09-23):
        summaries run far below their cap (the duck's ``evidence.md``, "Sweeps
        12 and 13")."""
        self._summary_target_tokens()  # validates the ratio against the window
        # A side turn runs past the window by its extra tokens, and the summary
        # after it has to fit as one after any turn does.
        extra = max(
            self.reflection_extra_window_tokens if self.level_reflection != "none" else 0,
            self.audit_extra_window_tokens if self.strategy_audit != "none" else 0,
        )
        owned = self.context_window_tokens + extra + self._reply_reserve_tokens() + MIN_SUMMARY_ROOM_TOKENS
        if owned > ceiling:
            raise ValueError(
                f"the summary request reaches past the engine's ceiling of {ceiling} on what the agent owns alone: "
                f"context_window_tokens={self.context_window_tokens}, the larger side turn's extra "
                f"(reflection_extra_window_tokens or audit_extra_window_tokens)={extra} and the reply reserve "
                "leave less than "
                f"{MIN_SUMMARY_ROOM_TOKENS} for the summary, before the round and the instruction the request "
                "also carries. Narrow the window or the reply cap, or state a wider engine window and turn "
                "derive_max_model_len off."
            )

    def _compaction_window(self) -> int:
        """The window a request is held to, by the compaction trigger and by
        the usage check: the agent's own, or in a side turn, which no summary
        interrupts, that and the turn's extra tokens
        (``reflection_extra_window_tokens``, ``audit_extra_window_tokens``).
        Under ``prune`` there is no room past the window to use."""
        if self._side_turn is None or not self._uses_context_summary():
            return self.context_window_tokens
        if self._side_turn == "reflection":
            return self.context_window_tokens + self.reflection_extra_window_tokens
        return self.context_window_tokens + self.audit_extra_window_tokens

    def _summary_completion_cap(self, cap: int, prompt_tokens: int) -> int:
        """``cap``, lowered to what the engine window has left over a summary
        request of ``prompt_tokens``. The prompt is the measured one -- exact
        where the backend counts it, a provable bound elsewhere -- so the
        request never asks the engine for more than its window holds. A
        prompt that already fills the window is an overflow, as the engine's
        own refusal would be."""
        ceiling = self.engine_ceiling_tokens
        if ceiling is None:
            return cap
        room = ceiling - prompt_tokens
        if room <= 0:
            raise RuntimeError(
                "context overflow: summarize never drops history to fit, and the summary request's prompt "
                f"({prompt_tokens} tokens) fills the engine's window of {ceiling}; size the engine window for it"
            )
        return min(cap, room)

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

    def _validate_host_backend(self, backend: ullm.backend.Backend) -> None:
        """Reject a connected engine that cannot fit the agent's context
        window, and settle the summary's ceiling onto the engine's window."""
        super()._validate_host_backend(backend)
        engine = getattr(backend, "engine", None)
        if engine is None:
            return
        window = getattr(getattr(engine, "model_config", None), "max_model_len", None)
        if type(window) is not int or window <= 0:
            raise ValueError("the connected engine must report a positive max_model_len")
        if self._uses_context_summary():
            # The summary request passes the agent's window by design; what
            # the engine must hold is the agent's share of that request.
            self._require_summary_fits(window)
            # Each summary's cap is clamped to what the ceiling leaves over its
            # prompt, so the ceiling cannot pass the engine actually connected:
            # a supplied engine arrives with none, and a stated one can be wider
            # than it. Settled on the template, before the games copy it.
            if self.engine_ceiling_tokens is None or self.engine_ceiling_tokens > window:
                self.engine_ceiling_tokens = window
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
        # A copy, because a derivable value can be a dict (the image
        # processor's arguments downstream): a baseline holding the engine's
        # own object would follow a later in-place edit of it, and the edit
        # would then read as the value this settled rather than as the
        # caller's choice it is.
        baseline[name] = copy.deepcopy(value)

    # ------------------------------------------------------------------
    # 1. deployment presets
    #
    # A fully-configured agent instance describes a whole experiment, so
    # the environment presets are constructors rather than solver
    # factories. What they configure lives in seal/presets.py.
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        engine: str = presets.DEFAULT_ENGINE,
        environment: str = presets.DEFAULT_ENVIRONMENT,
        serving: Literal["direct", "server"] = "direct",
        backend_factory: Any = None,
        checkpoint: str | None = None,
        duck_like: bool = False,
        kimi_student: bool = False,
        **overrides: Any,
    ) -> Self:
        """A seal configured for one engine in one environment.

        The two coordinates are the whole deployment: ``engine`` names a
        checkpoint (``ullm.backends.deployment.ENGINES``), ``environment`` the
        machine it runs on (``seal.presets.ENVIRONMENTS``). ``seal.presets.configure`` says what each
        contributes.

        ``backend_factory`` is the caller's, for a model this process does not
        start; ``ullm.backends.deployment`` supplies the engine facts one needs
        -- ``client_kwargs`` for a served model, ``backend_kwargs`` for an
        engine the caller already holds.

        ``serving`` picks the topology over one set of engine args — the
        in-process engine, or vLLM's HTTP server in this process.

        ``checkpoint`` plays that directory in place of the column's
        checkpoint, every other fact of the column kept, where ullm serves one
        (Flash-Next on the B200; ``seal.presets.configure``). The engine
        arguments are built around it, so an offline launcher needs no Hub
        cache.

        Everything else is a field: nothing on the returned agent is frozen, so
        a run that wants a different concurrency, window or tier sets it and the
        engine follows at ``prepare_for_backend``. ``llm_config`` and
        ``run_config`` are the exception — the column composed them, so they are
        assigned into rather than replaced.

        ``duck_like`` sets :data:`DUCK_LIKE` first, so the seal plays as the
        duck does; an override passed with it still wins.

        ``kimi_student`` sets :data:`KIMI_STUDENT` first, so a model trained on
        the Kimi K3.1 Pro teacher traces plays the seal those traces were
        recorded on; an override passed with it still wins. The checkpoint is
        the engine coordinate's, as for any other model.
        """
        if duck_like:
            overrides = {**DUCK_LIKE, **overrides}
        if kimi_student:
            overrides = {**KIMI_STUDENT, **overrides}
        agent = cls()
        presets.configure(
            agent,
            engine=engine,
            environment=environment,
            serving=serving,
            backend_factory=backend_factory,
            checkpoint=checkpoint,
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

    # ------------------------------------------------------------------
    # 2. one move
    #
    # The contract's only entry point; every section below is reached
    # from here. Delivery-only level-win calls cost no LLM at all.
    # ------------------------------------------------------------------

    def _announce_start(self, ctx: LLMMoveContext) -> None:
        """One stdout line as the game starts, with the per-level token cap it plays
        under as configured by then (a sweep sets it per arm before the first move):
        a wrong cap shows in the job's log within minutes, not at the game's end."""
        self._start_announced = True
        game = ctx.game
        print(
            f"[started] {game.game_id} pass={game.pass_index} max_tokens_per_level={self.max_tokens_per_level}",
            flush=True,
        )

    async def _do_move(self, ctx: LLMMoveContext, obs: Observation) -> Action | GiveUp | None:
        """One move (contract §2). Fold in the observation, then take the
        first branch that applies:

        - a level was just won — delivery-only, nothing to decide;
        - the action budget is spent — give up;
        - otherwise, run analyzer turns until the model acts.

        A lost game falls through to the LLM turns: the turn's current
        frame IS the lost board, and the prompt asks for the RESET. Only
        the last branch costs an LLM call, and it is the only one that can
        span minutes."""
        if not self._start_announced:
            self._announce_start(ctx)
        self._record_observation(ctx, obs)
        self._apply_pending_level_reset()

        if obs.outcome == "level_won":
            # Delivery-only call: the returned action is discarded. The
            # level split stays invisible to the model — the win rolls into
            # the pending-batch summary shown at the next LLM turn.
            platypus.diagnostics.current_span().kv(trivial=True)  # nothing decided here
            return Action("RESET")

        if self.max_moves is not None and self.action_count >= self.max_moves:
            return GiveUp(f"max_moves={self.max_moves} reached")

        # The model's submit_action(...) executes through ctx.execute while
        # the turns below run, so the decision is already carried out by the
        # time they return.
        self.executed_this_move = 0
        self.terminal_this_move = None
        self.paused_batch = None
        self.win_missed = False
        self.live_available_actions = list(obs.available_actions)

        await self._analyzer_turns_until_acted(ctx, obs)
        if self.level_reflection != "none" and self._won_level_this_move():
            await self._reflection_turn(ctx, obs)
        # The kernel resets owed at a move's end are paid now, while the
        # kernel is idle, rather than at the top of the next move: nothing
        # happens in between, and the boundary between the two moves is
        # where a checkpoint pickles the agent (A4), which a live kernel
        # refuses. "move" gives every decision move a fresh kernel (its
        # rounds share scratch work, moves never do); the notice is read at
        # the next move's first prompt, so it names that move. "level" pays
        # the reset a level won in this move owes.
        if self.persistence == "move":
            self._reset_repl("this move started")
        self._apply_pending_level_reset()

        return None  # at least one action already executed in-move

    # ------------------------------------------------------------------
    # 3. the analyzer loop
    #
    # The original's ToolAgent.analyze, re-entered until the model acts.
    # Three nested things, outermost first: analyzer ENTRIES (each with a
    # freshly built state prompt), BLOCKS within an entry (a user prompt
    # and the response turn answering it), and — inside ullm's
    # TurnManager — the LLM/tool ROUNDS of one block.
    # ------------------------------------------------------------------

    def _sampling(self, ctx: LLMMoveContext) -> ullm.backend.GenericSamplingParameters:
        """Profile + seal default: reasoning is sent back for ALL turns
        (the original ran vLLM with ``preserve_thinking: true``). A profile
        may still pin a different send_back_reasoning explicitly."""
        params = self.resolve_sampling(ctx, self.profile)
        if params.max_tokens is not None:
            raise ValueError("Play profile max_tokens must be unset; set SealAgent.reply_cap_tokens instead")
        params.max_tokens = self.reply_cap_tokens
        if params.send_back_reasoning is None:
            params.send_back_reasoning = "all"
        return params

    def _acted(self) -> bool:
        """Whether this move's decision has been made: at least one action
        executed in-snippet."""
        return self.executed_this_move > 0

    # ------------------------------------------------------------------
    # the token budgets (the duck's ``design.md``, "Run budgets")
    #
    # Three depths, because a stop has to take hold at the next model call
    # rather than at the next move: one analyzer entry can run many calls,
    # and a move runs entries until the model acts.
    # ------------------------------------------------------------------

    def _current_level(self, ctx: LLMMoveContext) -> int:
        """The level the game is standing on, from the engine's own counter.

        ``ctx.history[-1].observation`` is the settled observation — what this
        move started from — so this is live rather than remembered. Read on
        every consultation because the budget is consulted from inside a turn,
        where the move's own observation is not in scope.

        Before the first action the history is empty, and the level is the one
        the seal last folded in (``budget_level``): the level the game started
        on — 0, or a ``start_at_level`` window's first level (a game started
        from a checkpoint inherits its source's history). Reading 0 there would make such a game's first turn look as if it
        crossed a level and leave it unbilled (``_charge_budget``)."""
        return ctx.history[-1].observation.level if ctx.history else self.budget_level

    def _allowance_for(self, level: int) -> int | None:
        """Completion tokens ``level`` may spend, or ``None`` for unbounded."""
        if self.max_tokens_per_level is None:
            return None
        return self.max_tokens_per_level[min(level, len(self.max_tokens_per_level) - 1)]

    def _level_allowance(self, ctx: LLMMoveContext) -> int | None:
        """Completion tokens the level the game stands on may spend."""
        return self._allowance_for(self._current_level(ctx))

    def _budget_in_force(self, ctx: LLMMoveContext) -> bool:
        """Whether the allowance is being enforced at all.

        ``ctx.slots.drained`` is false whenever the queue is unknown, so a run
        that publishes no tally enforces the cap exactly as it always did.
        Kept apart from :meth:`_level_allowance` so the diagnostics can report
        what the level was allowed AND whether that was applied — folded
        together, a lifted cap records as no cap, which reads as zero."""
        return not (self.uncap_when_queue_drained and ctx.slots.drained)

    def _budget_cap(self, ctx: LLMMoveContext) -> int | None:
        """The allowance actually binding this move, or ``None`` for none."""
        allowance = self._level_allowance(ctx)
        return allowance if allowance is not None and self._budget_in_force(ctx) else None

    def _game_tokens_used(self, ctx: LLMMoveContext) -> int:
        """Completion tokens billed to this game's private backend."""
        return ctx.backend.total_usage.completion_tokens

    def _per_turn_cap_lifted(self, ctx: LLMMoveContext) -> bool:
        """Whether a turn ceiling may retire after the game queue drains."""
        return self.max_tokens_per_game is None and not self._budget_in_force(ctx)

    def _per_turn_cap(self, ctx: LLMMoveContext) -> int | None:
        """The generation ceiling for a turn about to run: what is left of the
        enabled level/game allowances, or ``None`` when unbounded.

        **Gives up rather than ever returning zero**, which is the whole reason
        every ceiling is routed through here. A turn handed a ceiling of zero
        generates nothing: the manager appends its synthetic placeholder, the
        placeholder carries no tool call and no usage, so the entry nudges into
        another identical turn — and since nothing was generated, the level's
        spend never advances to the point that would end it. That loop cannot be
        broken from inside, so the only safe ceiling is a positive one.

        Called for the check alone wherever a turn is merely imminent (before the
        kernel is set up, and between entries): an attempt that cannot afford a
        call should not pay for arranging one either."""
        remaining: list[int] = []
        level_cap = self._budget_cap(ctx)
        if level_cap is not None:
            level_remaining = level_cap - self.level_tokens_used
            if level_remaining <= 0:
                raise GiveUpError(f"per-level token budget exhausted ({self.level_tokens_used}/{level_cap})")
            remaining.append(level_remaining)

        if self.max_tokens_per_game is not None:
            game_used = self._game_tokens_used(ctx)
            game_remaining = self.max_tokens_per_game - game_used
            if game_remaining <= 0:
                raise GiveUpError(f"per-game token budget exhausted ({game_used}/{self.max_tokens_per_game})")
            remaining.append(game_remaining)

        return min(remaining) if remaining else None

    def _audit_check(self, ctx: LLMMoveContext, started_on: int, started_used: int, game_tokens_before: int) -> None:
        """Raise ``_AuditDue`` when the level's spend has passed an audit
        threshold that has not fired. The spend is the level's as its budget
        counts it, live to the token, less the reflection turn's. Not in a side
        turn, nor once the turn has crossed into another level: the spend it
        carries is the old level's."""
        if self.strategy_audit == "none" or self._side_turn is not None or self._current_level(ctx) != started_on:
            return
        spent = started_used + self._game_tokens_used(ctx) - game_tokens_before - self.level_reflection_tokens
        passed = sum(1 for threshold in self.audit_level_tokens if threshold <= spent)
        if passed > self.audits_fired:
            self.audits_fired = passed
            self._audit_threshold = self.audit_level_tokens[passed - 1]
            raise _AuditDue

    def _live_turn_cap(self, ctx: LLMMoveContext, started_used: int, game_tokens_before: int) -> int | None:
        """``_per_turn_cap`` from inside a running turn, live to the token.

        The level's counter is charged only when a turn ends, so mid-turn the
        spend is the turn's start plus what the backend has billed since —
        the summaries included, which run outside the turn manager's own
        tally. A turn ends once the model acts, so it never runs on past a
        level it won and the level it started on is the one charged. Gives
        up rather than return zero, for ``_per_turn_cap``'s reason."""
        remaining: list[int] = []
        level_cap = self._budget_cap(ctx)
        if level_cap is not None:
            used = started_used + self._game_tokens_used(ctx) - game_tokens_before
            if level_cap - used <= 0:
                raise GiveUpError(f"per-level token budget exhausted ({used}/{level_cap})")
            remaining.append(level_cap - used)
        if self.max_tokens_per_game is not None:
            game_used = self._game_tokens_used(ctx)
            if self.max_tokens_per_game - game_used <= 0:
                raise GiveUpError(f"per-game token budget exhausted ({game_used}/{self.max_tokens_per_game})")
            remaining.append(self.max_tokens_per_game - game_used)
        return min(remaining) if remaining else None

    def _charge_budget(
        self,
        ctx: LLMMoveContext,
        started_on: int,
        started_used: int,
        game_tokens_before: int,
    ) -> None:
        """Charge one turn's generation to the level it ran on, and report it.

        Completion tokens only, as the whale charges them, so one extraction
        reads both agents. Both level and game budgets use the backend's
        per-game meter, including any billed round no longer retained in the
        conversation, so their remaining allowances are directly comparable.

        **A turn that won its level mid-flight is recorded against the level it
        ran under, and the level that follows is not billed.** In live mode
        ``submit_action(...)`` executes inside the turn, so the win lands before the
        turn returns and the counter has already been reset for the new level;
        billing there would start every level after the first pre-spent by its
        predecessor. The generation belongs to the level whose cap it ran under,
        so the event says so — level, counter and allowance all taken from the
        turn's start, which is what makes the record describe one level rather
        than two. The live counter keeps the reset's zero: the level it names is
        finished, and nothing consults it again."""
        completion = self._game_tokens_used(ctx) - game_tokens_before
        if completion < 0:
            raise RuntimeError("backend completion-token meter moved backwards within a turn")
        crossed = self._current_level(ctx) != started_on
        if not crossed:
            self.level_tokens_used += completion
        values: dict[str, Any] = {
            "tokens_added": completion,
            # The old level's true total when the turn crossed: the live counter
            # belongs to the new level by then and would report its zero.
            "tokens_used_this_level": started_used + completion if crossed else self.level_tokens_used,
            "max_tokens_per_level": self._allowance_for(started_on) or 0,
            "budget_enforced": self._budget_in_force(ctx),
            "level": started_on,
            "level_crossed": crossed,
        }
        if self.max_tokens_per_game is not None:
            values.update(
                tokens_used_this_game=self._game_tokens_used(ctx),
                max_tokens_per_game=self.max_tokens_per_game,
            )
        platypus.diagnostics.current_span().event("budget", **values)

    async def _analyzer_turns_until_acted(self, ctx: LLMMoveContext, obs: Observation) -> None:
        """Run analyzer turns until the model executes at least one action
        in-snippet.

        Two nested loops, both from the original. The outer one re-enters
        the analyzer with a freshly built state prompt, as the original's
        solver loop did; the inner one (``_run_analyzer_entry``) is one such
        entry, which may itself run several LLM calls."""
        self._per_turn_cap(ctx)  # raises when the level is spent, before any kernel work
        # One try from the kernel's setup to the history's save: a raise
        # anywhere in the analyzer would otherwise leave the live
        # backend, its params and the status handle on the agent, which holds
        # no live resource between moves.
        try:
            repl = self._ensure_repl(ctx)
            self._seed_world_model(repl)
            repl.host_functions = self._kernel_host_functions(ctx, obs)
            with ctx.diag.span("seal/analyze") as analyze:
                entries = nudges = total_calls = 0
                while not self._acted():
                    # Between entries as well as before the first: a move that has
                    # not acted yet keeps re-entering, and the budget is what ends
                    # that rather than a clock.
                    self._per_turn_cap(ctx)  # raises when the level is spent
                    remaining_move_calls = self._summary_request_backstop_remaining(total_calls)
                    # Fresh analyzer-turn entry (the original solver's re-entry):
                    # take the step summary and rebuild the state prompt.
                    self._take_step_summary()
                    entries += 1
                    calls_used, entry_nudges = await self._run_analyzer_entry(
                        ctx, obs, remaining_move_calls=remaining_move_calls
                    )
                    total_calls += calls_used
                    nudges += entry_nudges
                analyze.kv(
                    context_window=self.context_window_tokens,
                    entries=entries,
                    nudges=nudges,
                    llm_calls=total_calls,
                    # Cumulative, so a run whose counts were never verified is
                    # visible as a flat 0 rather than passing quietly.
                    counts_reconciled=self.counts_reconciled,
                )
                analyze.kv(executed=self.executed_this_move)
                analyze.set_outcome(f"executed {self.executed_this_move}")
            await self._persist_history()
        finally:
            self._live_conversation = None
            self._live_backend = None
            self._live_params = None
            self._status_handle = None

    def _summary_request_backstop_remaining(self, calls_used: int) -> int | None:
        """The move's remaining model-call allowance under ``summarize``."""
        if not self._uses_context_summary():
            return None
        remaining = _SUMMARY_REQUEST_BACKSTOP - calls_used
        if remaining <= 0:
            raise GiveUpError(f"request backstop exhausted ({calls_used}/{_SUMMARY_REQUEST_BACKSTOP})")
        return remaining

    async def _run_analyzer_entry(
        self, ctx: LLMMoveContext, obs: Observation, *, remaining_move_calls: int | None = None
    ) -> tuple[int, int]:
        """One analyzer entry: LLM turn blocks against one state prompt,
        until the model acts or ``max_llm_calls_per_move`` is spent. Returns
        ``(llm calls made, nudges sent)``.

        A response with no tool call gets the original's "you have not acted
        yet" nudge, which opens a new block within this same entry. A
        response that DID call tools but still hasn't acted means the rounds
        ran out or the turn truncated mid-tools; that ends the entry, and
        the caller re-enters with a freshly built state prompt. A reply
        dropped after every re-sample was refused (``malformed_nudges``) gets
        the malformed nudge instead, whatever the rounds before it called."""
        configured_cap = self.max_llm_calls_per_move if self.max_llm_calls_per_move > 0 else None
        cap = remaining_move_calls if configured_cap is None else configured_cap
        if remaining_move_calls is not None and configured_cap is not None:
            cap = min(remaining_move_calls, configured_cap)
        calls_used = nudges = 0
        next_prompt = self._build_user_prompt_element(obs)
        self._turn_prompt = next_prompt
        self._omitted_tool_calls = 0  # the note counts this turn's losses
        while not self._acted() and (cap is None or calls_used < cap):
            calls_used += await self._run_turn_block(
                ctx, next_prompt, remaining_calls=None if cap is None else cap - calls_used
            )
            if self._acted():
                break
            if self._audit_due:
                self._audit_due = False
                text = await self._audit_turn(ctx)
                next_prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))
                continue
            if self._malformed_nudge_due:
                self._malformed_nudge_due = False
                nudges += 1
                text = self.prompts.malformed_nudge(self._prompt_modes())
                next_prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))
                continue
            if self._last_turn_had_tool_calls():
                break
            nudges += 1
            next_prompt = self._followup_prompt()
        # An entry the call cap ended before the nudge went out: the state
        # prompt that opens the next one takes its place.
        self._malformed_nudge_due = False
        return calls_used, nudges

    async def _reflection_turn(self, ctx: LLMMoveContext, obs: Observation) -> None:
        """The turn after a move that won a level (``level_reflection``).

        Its prompt reports the move as any turn's does, over the win screen,
        then asks for a reflection on the level, answered in a reply without a
        tool call, which ends the turn. The kernel still holds the finished
        level, and ``submit_action`` is refused, since the move has ended its
        level. No summary runs inside it: once its context passes the window
        and ``reflection_extra_window_tokens``, the turn ends, the model is
        told, and the summary comes with the next level's prompt. That prompt
        then opens at the level change, the report given here."""
        self._per_turn_cap(ctx)  # the new level's allowance, which the turn is charged to
        try:
            repl = self._ensure_repl(ctx)
            repl.host_functions = self._kernel_host_functions(ctx, obs)
            with ctx.diag.span("seal/reflection") as span:
                self._take_step_summary()
                prompt = self._build_reflection_prompt_element()
                self._turn_prompt = prompt
                self._omitted_tool_calls = 0
                calls, tokens, cut = await self._run_side_turn(ctx, "reflection", prompt)
                if cut and any(element is prompt for element in self.messages):
                    self.messages.append(
                        ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=self.prompts.reflection_cut_note))
                    )
                # Charged to the new level, and not counted toward its audits.
                self.level_reflection_tokens += tokens
                if self.last_step_summary is not None:
                    self.last_step_summary = dataclasses.replace(self.last_step_summary, reported=True)
                span.kv(llm_calls=calls, tokens=tokens, cut=cut, win_missed=self.win_missed)
            await self._persist_history()
        finally:
            self._live_conversation = None
            self._live_backend = None
            self._live_params = None
            self._status_handle = None

    async def _audit_turn(self, ctx: LLMMoveContext) -> str:
        """The strategy audit (``strategy_audit``), run by the entry ahead
        of the call a threshold stopped. Its prompt gives advice, the actions
        the level and the game have not tried, two questions, and the level's
        opening board and the current one; a reply without a tool call answers
        them and ends it. Returns what the interrupted turn goes on with:
        ``audit_continue_line``, or ``audit_cut_note`` when the audit reached
        its window. Thresholds its own spend passed count as fired, so two
        audits never run back to back."""
        with ctx.diag.span("seal/audit") as span:
            prompt = self._build_audit_prompt_element()
            calls, tokens, cut = await self._run_side_turn(ctx, "audit", prompt)
            spent = self.level_tokens_used - self.level_reflection_tokens
            passed = sum(1 for threshold in self.audit_level_tokens if threshold <= spent)
            self.audits_fired = max(self.audits_fired, passed)
            span.kv(threshold=self._audit_threshold, llm_calls=calls, tokens=tokens, cut=cut)
        if cut and any(element is prompt for element in self.messages):
            return self.prompts.audit_cut_note
        return self.prompts.audit_continue_line

    async def _run_side_turn(
        self, ctx: LLMMoveContext, kind: str, prompt: ullm.conv.UserPrompt
    ) -> tuple[int, int, bool]:
        """A turn that is not for acting, of ``kind`` (``reflection``,
        ``audit``): ``prompt``, then the model's rounds until it replies
        without a tool call. ``submit_action`` is refused in it, no search
        follows a call, and no summary runs inside it: past the window and the
        kind's extra tokens it ends before its next call. A reply dropped as
        malformed ends it too, since there is nothing to nudge the model into.
        Returns the calls made, the completion tokens spent, and whether it was
        cut."""
        self._side_turn, self._side_turn_cut = kind, False
        started = self._side_turn_started_at = self._game_tokens_used(ctx)
        try:
            calls = await self._run_turn_block(ctx, prompt, remaining_calls=None)
        finally:
            self._side_turn = None
            self._mark_moveless(ctx)
        self._malformed_nudge_due = False
        return calls, self._game_tokens_used(ctx) - started, self._side_turn_cut

    def _kernel_host_functions(self, ctx: LLMMoveContext, obs: Observation) -> dict[str, Any]:
        """The host calls the model's Python can reach: reading game state
        and acting. They run in the
        agent process while the kernel blocks, which is what lets
        ``submit_action(...)`` be a real call into
        the harness rather than a message the model writes and hopes for.

        A binding is MODEL-REACHABLE by contract (U9): whatever the kernel
        passes was written by the model, so the acting call validates its
        arguments here, host-side, before anything touches the game. Bound
        per use rather than kept on the tool — ``ctx`` and the observation
        they close over belong to this move only."""

        def get_state(*args: object, **kwargs: object) -> dict[str, Any]:
            if args or kwargs:
                return {"error": "get_state() takes no arguments."}
            return self._state_payload()

        async def execute_action(*args: object, **kwargs: object) -> dict[str, Any]:
            normalized, error = self._validate_host_action_arguments(args, kwargs)
            if error is not None:
                return {"error": error}
            assert normalized is not None
            refused = self._refused(ctx, normalized, route="unchecked")
            if refused is not None:
                return refused
            if self.use_world_model == "mandatory":
                # Moves run checked (stepped) or as exploration, never as an
                # unchecked batch.
                return {"error": "Actions run under the world-model gate in this game."}
            return await self._execute_actions_live(ctx, normalized)

        async def execute_action_stepped(*args: object, **kwargs: object) -> dict[str, Any]:
            normalized, error = self._validate_host_action_arguments(args, kwargs)
            if error is not None:
                return {"error": error}
            assert normalized is not None
            refused = self._refused(ctx, normalized, route="checked")
            if refused is not None:
                return refused
            # The kernel steps a batch only under a world model that passed,
            # which ends exploration for the level.
            self.exploration_over = True
            return await self._execute_actions_live(ctx, normalized, pause_each=True, route="checked")

        async def execute_action_exploring(*args: object, **kwargs: object) -> dict[str, Any]:
            normalized, error = self._validate_host_action_arguments(args, kwargs)
            if error is not None:
                return {"error": error}
            assert normalized is not None
            refused = self._refused(ctx, normalized, route="exploring")
            if refused is not None:
                return refused
            refusal = self._exploration_refusal(normalized)
            if refusal is not None:
                self._record_moves(ctx, route="exploring", requested=len(normalized), refused=str(refusal["error"]))
                return refusal
            before, level = self.action_count, self.exploration_level
            reply = await self._execute_actions_live(ctx, normalized, route="exploring")
            # Charged to the level it was played on. A move that won the level
            # has already opened the next one's allowance while it ran, and
            # that allowance stays whole.
            if self.action_count > before and self.exploration_level == level:
                self.explored_in_cell = self._cell_index()
                self.exploration_moves_left -= 1
            return reply

        async def continue_batch(*args: object, **kwargs: object) -> dict[str, Any]:
            if args or kwargs:
                return {"error": "continue takes no arguments."}
            return await self._continue_paused_batch(ctx)

        def stop_batch(*args: object, **kwargs: object) -> dict[str, Any]:
            if kwargs or len(args) != 2 or not all(isinstance(arg, str) for arg in args):
                return {"error": "stop takes a reason and a detail string."}
            return self._stop_paused_batch(ctx, str(args[0]), str(args[1]))

        def set_matching(*args: object, **kwargs: object) -> dict[str, Any]:
            if kwargs or len(args) != 1 or not isinstance(args[0], dict):
                return {"error": "set_matching takes one dict of settings."}
            return self._set_matching(cast("dict[str, Any]", args[0]))

        return {
            "get_state": get_state,
            "action": execute_action,
            "action_stepped": execute_action_stepped,
            "action_exploring": execute_action_exploring,
            "action_continue": continue_batch,
            "action_stop": stop_batch,
            "set_matching": set_matching,
        }

    def _exploration_refusal(self, normalized: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The reply refusing an exploratory batch, or None if it may run: one
        move, once per cell, while the level's exploration is open and
        budgeted. Once exploration has ended, for whichever reason, the refusal
        is flagged ``world_model_required``: the kernel then shows how the
        model failed, which the other refusals, being about exploration's own
        rules, do not."""
        if self.exploration_over or self.exploration_moves_left <= 0:
            return {"error": self.prompts.world_model_required, "world_model_required": True}
        if len(normalized) != 1:
            return {"error": self.prompts.exploration_single_action_refusal.format(count=len(normalized))}
        if self.explored_in_cell == self._cell_index():
            return {"error": self.prompts.exploration_one_per_call_refusal}
        return None

    def _seed_world_model(self, repl: GameREPL) -> None:
        """Write the example ``world_model.py`` into the working directory,
        once per game, before the model's first look: the one for a game with
        MOUSE, or without it. Nothing under ``use_world_model='none'``."""
        if self.use_world_model == "none" or self.world_model_seeded:
            return
        self.world_model_seeded = True
        directory = repl._tmpdir  # pyright: ignore[reportPrivateUsage] — the tool owns it, this only writes one file
        assert directory is not None
        source = seeds.seed_source(mouse="MOUSE" in self.game_action_names)
        with open(os.path.join(directory, "world_model.py"), "w", encoding="utf-8") as handle:
            handle.write(source)

    def _cell_index(self) -> int:
        """Which python call is running, as the tool counts them."""
        return self.repl.cells_started if self.repl is not None else 0

    def _validate_host_action_arguments(
        self, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Validate model-reachable parent callback arguments before dispatch."""
        if kwargs or len(args) != 1:
            return None, "submit_action() requires exactly one positional actions argument."
        raw_actions = args[0]
        if not isinstance(raw_actions, list):
            return None, "submit_action() received an invalid normalized action batch."
        normalized = cast(list[object], raw_actions)
        if any(not isinstance(item, dict) for item in normalized):
            return None, "submit_action() received an invalid normalized action batch."
        return cast(list[dict[str, Any]], normalized), None

    def _memory_entry_text(self, entry: MemoryEntry) -> str:
        """Render one host-timestamped journal entry for the model."""
        return f"{self.prompts.memory_entry_header(entry)}\n{entry.text}"

    def _render_memory(self, entries: list[MemoryEntry] | None = None) -> str:
        """Render the active memory journal oldest-to-newest."""
        selected = self.memory_entries if entries is None else entries
        return "\n".join(self._memory_entry_text(entry) for entry in selected)

    def _retained_memory_indices(self, elements: list[ullm.conv.ConversationElement]) -> set[int]:
        """Journal indices whose immutable memory results remain in history."""
        retained: set[int] = set()
        for element in elements:
            if not isinstance(element, ullm.conv.ResponseTurn):
                continue
            for sub in element.elements:
                if not isinstance(sub, ullm.conv.ToolResult) or sub.tool_name != "memory" or sub.is_error:
                    continue
                index = self.prompts.parse_memory_append_index(sub.content.to_string())
                if index is not None:
                    retained.add(index)
        return retained

    def _memory_snapshot_entries(self, elements: list[ullm.conv.ConversationElement] | None) -> list[MemoryEntry]:
        """Entries not already represented by retained immutable tool calls."""
        if elements is None:
            return self.memory_entries
        retained = self._retained_memory_indices(elements)
        if not retained:
            return self.memory_entries
        oldest_retained = min(retained)
        return [entry for entry in self.memory_entries if entry.index < oldest_retained]

    async def _append_memory(self, text: str) -> str:
        """Append a timestamped entry and evict oldest whole entries to the
        retention budget.

        The budget is a hard ceiling of ``memory_context_tokens`` true
        tokens on the retained journal: the rendered journal is measured
        by ``_journal_tokens`` -- exact where the backend tokenizes, the
        character upper bound where it cannot -- so the comparison never
        under-reads on any backend. Whatever is retained is additionally
        budget-covered downstream like all other prompt text."""
        if self.memory_context_tokens <= 0:
            raise ValueError(self.prompts.memory_budget_not_positive)
        if not text.strip():
            raise ValueError(self.prompts.memory_empty_text)
        frame = self.history_entries[-1].frame if self.history_entries else None
        last_entry = self.history_entries[-1] if len(self.history_entries) > 1 else None
        last_action = _format_action_display(last_entry.action) if last_entry and last_entry.action else None
        entry = MemoryEntry(
            index=self.memory_entry_count + 1,
            response=self.response_step,
            step=self.action_count,
            level=frame.level if frame is not None else 1,
            last_action=last_action,
            text=text,
        )
        entry_tokens = await self._journal_tokens(self._memory_entry_text(entry))
        if entry_tokens > self.memory_context_tokens:
            raise ValueError(
                self.prompts.memory_entry_too_large.format(entry_tokens=entry_tokens, budget=self.memory_context_tokens)
            )
        retained = [*self.memory_entries, entry]
        while (
            len(retained) > 1 and await self._journal_tokens(self._render_memory(retained)) > self.memory_context_tokens
        ):
            del retained[0]
        self.memory_entry_count += 1
        self.memory_entries = retained
        return self.prompts.memory_append_acknowledgement(entry)

    async def _journal_tokens(self, text: str) -> int:
        """The retention ruler. In token-native mode the backend tokenizes
        raw text exactly (vLLM ``/tokenize``) and its answer is REQUIRED:
        ``None`` is a legitimate answer only from a backend that genuinely
        cannot tokenize, which token-native configs never are, so a missing
        answer (a failed probe, no live backend) raises rather than
        silently changing ruler mid-run. Everywhere else no backend is
        asked and the journal's long-standing nominal 3 chars/token
        calibration stands -- a retention preference, never a budget
        measurement."""
        if self.token_native_context:
            backend = self._live_backend
            if backend is None:
                raise RuntimeError("journal count failed: token-native mode has no live backend to tokenize with")
            exact = await backend.text_token_count(text)
            if exact is None:
                raise RuntimeError(
                    "journal count failed: the token-native backend answered None for "
                    "text_token_count -- refusing to change ruler mid-run"
                )
            return exact
        return len(text) // 3

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

    def _refresh_memory_prompt_snapshot(
        self, retained_elements: list[ullm.conv.ConversationElement] | None = None
    ) -> None:
        """Replace the snapshot with entries absent from retained tool calls.

        Without the journal there is no snapshot, and the system prompt never
        moves."""
        if not self.journal:
            return
        self.memory_prompt_snapshot = self._render_memory(self._memory_snapshot_entries(retained_elements))
        conversation = self._live_conversation
        if conversation is not None:
            conversation.system_prompt.content = ullm.conv.StringContent(text=self._system_prompt_text())

    def _turn_tools(self, ctx: LLMMoveContext | None = None) -> list[ullm.tool.Tool]:
        """Apply the agent's execution policy after all tools are assembled."""
        tools = self._build_turn_tools(ctx)
        for tool in tools:
            tool.auto_correct_arguments = self.auto_correct_tool_arguments
        return tools

    def _build_turn_tools(self, ctx: LLMMoveContext | None = None) -> list[ullm.tool.Tool]:
        """Build the configured tools shared by requests and context measurement.

        The order is prompt text. The python tool leads and the file tools
        follow it, because they act on the same working directory and are
        read together; the journal is a different register and comes last.
        """
        repl = self._ensure_repl(ctx)
        python: ullm.tool.Tool = repl
        if ctx is not None and (self.use_world_model != "none" or self._delta_hook_needed()):
            python = _SearchingPython(
                target=repl,
                host_functions=repl.host_functions,
                after_call=self._after_call_hook(ctx, repl),
                actions_played=lambda: self.action_count,
            )
        tools: list[ullm.tool.Tool] = [python]
        tools.extend(build_file_tools(self._ensure_workspace(ctx), **_FILE_SUITE))
        if self.journal:
            tools.append(SealMemoryTool(appender=self._append_memory))
        return tools

    def _delta_hook_needed(self) -> bool:
        """Whether a python call is followed by the kernel's delta cell: for the settled delta, to describe
        the call's longest animation, to collect the records of those the model asked for, or for the diff
        image. The marks need no cell: the host counts the frames itself."""
        return self.settled_delta or self.animation_description != "none" or self.diff_image

    def _after_call_hook(self, ctx: LLMMoveContext, repl: GameREPL) -> Callable[[int, str], Awaitable[str | None]]:
        """What runs after each python call of a turn: the frame delta, then
        the search when the call earned one. Takes the action count from
        before the call, and the call's output. A call that ran into its time
        limit earns no search: whatever it left of the model's files is what
        timed out."""

        async def after_call(actions_before: int, output: str) -> str | None:
            if repl.pending_host_exception is not None:
                return None
            if repl.kernel_lost:
                # The call's own output says it was killed, not that the next
                # call starts on a fresh session.
                return self._restart_unusable_kernel(repl, self.prompts.kernel_stopped_responding_reason, output)
            if repl.is_fresh:
                return None
            # The model's call's, read before a harness cell sets them again.
            timed_out = repl.timed_out
            if repl.strays_killed:
                platypus.diagnostics.current_span().event("kernel_strays_killed", count=repl.strays_killed)
            restarted = await self._collect_after_call(ctx, repl)
            if restarted is not None:
                return restarted
            # A side turn plays no move, and the reflection's state is a won
            # one: nothing to search from.
            if timed_out or self.use_world_model == "none" or not self.use_search or self._side_turn is not None:
                return None
            if self.action_count != actions_before:
                return None
            return await self._search_after_call(ctx, repl, output)

        return after_call

    async def _collect_after_call(self, ctx: LLMMoveContext, repl: GameREPL) -> str | None:
        """Collect what the kernel recorded during the call: each explicit
        ``test_world_model()`` onto a ``seal/test`` span; the deltas it made,
        and the newest frame's if the call played one, each onto a span, the
        text for the next user prompt. A block that did not reach its frame
        ends the game (:class:`FrameDeltaError`), as does a kernel that could
        not answer for a reason of ours. One that could not answer for its own
        state -- it hung, or ran out of memory -- is restarted, the frame goes
        undescribed, and what the model is told comes back for the call's
        output."""
        from seal.frame_delta import FrameDeltaError

        output, failed = await repl.run_harness(
            f"import runtime._repl_state as {_HARNESS_ALIAS}\n"
            f"print({_HARNESS_ALIAS}.after_call())\n"
            f"del {_HARNESS_ALIAS}",
            timeout=_HARNESS_CELL_TIMEOUT_S,
        )
        if failed:
            reason = self._kernel_failure_reason(repl, output)
            if reason is None:
                raise FrameDeltaError(f"the kernel could not describe the frame:\n{output}")
            return self._restart_unusable_kernel(repl, reason, output)
        collected = json.loads(output.strip().splitlines()[-1])
        if collected["win_predicted"] is False:
            self.win_missed = True
        for test in collected["tests"]:
            with ctx.diag.span("seal/test") as span:
                span.kv(**test)
        payload = collected["frame_delta"]
        for record in payload["records"]:
            with ctx.diag.span("seal/frame_delta") as span:
                baselines = record.pop("baselines", None)
                grids = record.pop("grids", None)
                span.kv(**record)
                if baselines is not None:
                    span.artifact("baselines", baselines)
                # Packed rather than as png-grid, which keeps a cell's palette index only: a render's
                # wildcard would not survive it.
                if grids is not None:
                    span.artifact("grids", grids)
            if record.get("error"):
                raise FrameDeltaError(str(record["error"]))
        if payload["text"] is not None:
            self.frame_delta_text = str(payload["text"])
        if payload["described"] is not None:
            self.animation_text = payload["animation_text"]
            self.animation_described = int(payload["described"])
            self.animation_concise = bool(payload["concise"])
        if collected["diff_image"] is not None:
            self.diff_image_pending = collected["diff_image"]
            platypus.diagnostics.current_span().event(
                "diff_image",
                baseline=collected["diff_image"]["kind"],
                turn_index=collected["diff_image"]["turn_index"],
            )
        return None

    def _kernel_failure_reason(self, repl: GameREPL, output: str) -> str | None:
        """Why a harness cell failed, when it failed for the state the model
        left the kernel in rather than for a fault of ours: it ran into its
        time limit, it ran out of memory, or a library it loaded could not get
        the memory to load. Worded for the reset notice. None for any other
        failure, which is ours and ends the game."""
        if repl.timed_out or repl.kernel_lost:
            return self.prompts.kernel_stopped_responding_reason
        lines = output.strip().splitlines()
        last = lines[-1] if lines else ""
        if _names_memory_error(last) or "failed to map segment" in last or "Cannot allocate memory" in last:
            return self.prompts.kernel_out_of_memory_reason
        return None

    def _restart_unusable_kernel(self, repl: GameREPL, reason: str, output: str) -> str | None:
        """Restart a kernel that hung, ran out of memory or was lost in a cell,
        and return the notice for the call's output: the model is mid-turn,
        and its next call starts on a fresh session. None where every call
        starts on one anyway."""
        platypus.diagnostics.current_span().event("kernel_restarted", reason=reason, output=output[-500:])
        repl.reset()
        if self.persistence == "round":
            return None
        return self.prompts.repl_reset_notice_with_files.format(reason=reason)

    async def _search_after_call(self, ctx: LLMMoveContext, repl: GameREPL, call_output: str) -> str | None:
        """The chain after a call that played no move, outside its time limit.

        1. The kernel says whether the call's last ``test_world_model()``
           passed, and if so tests the model again: the model may have changed
           its files since. A failure now gets a short note and no search.
        2. The search, in a child process under its time and memory caps,
           which tests the same files first: a failure there ends the game,
           the two harnesses disagreeing (:class:`SearchHarnessError`).
        3. The kernel replays a found path under its own model and binds
           ``suggested_actions`` / ``suggested_history``; a path it does not
           reproduce ends the game the same way.
        4. The report, for the call's output — answering, when a path was
           bound, a call that looked it up before the search had run.
        """
        # Here rather than at module level: the search's machinery imports
        # scipy, which every game child would otherwise load on unpickling
        # the agent, searching or not.
        from seal.search import RETEST_FAILED_NOTE, SearchHarnessError, asked_early, search_level

        # The retest runs the model's code, so it gets the model's own limit:
        # past it there is no search. A kernel the limit killed is restarted
        # and the model told.
        verdict, failed = await repl.run_harness(
            f"import runtime._repl_state as {_HARNESS_ALIAS}\n"
            f"print({_HARNESS_ALIAS}.search_retest())\n"
            f"del {_HARNESS_ALIAS}",
            timeout=self.python_timeout_s,
        )
        if failed and repl.kernel_lost:
            return self._restart_unusable_kernel(repl, self.prompts.kernel_stopped_responding_reason, verdict)
        if failed and repl.timed_out:
            return None
        if failed:
            raise SearchHarnessError(f"the kernel could not say whether the call earned a search:\n{verdict}")
        verdict = verdict.strip().splitlines()[-1] if verdict.strip() else ""
        if verdict == "none":
            return None
        if verdict == "fail":
            return RETEST_FAILED_NOTE
        assert verdict == "pass", f"unexpected search verdict {verdict!r}"
        # The kernel's turns, which the REPL's replay of the path checks against.
        entries = self._kernel_level_entries()
        directory = repl._tmpdir  # pyright: ignore[reportPrivateUsage] — the tool owns it, this only reads it
        assert directory is not None
        with ctx.diag.span("seal/search") as span:
            report = await search_level(
                directory=directory,
                entries=entries,
                turns=[turn_payload(entry) for entry in entries],
                model_actions=list(self.game_action_names),
                game_actions=list(self.game_action_names),
                level=entries[-1].level,
                seconds=self.search_seconds,
                memory_bytes=self.search_memory_bytes,
                ascii_layout=AsciiLayout(
                    row_numbers=self.ascii_row_numbers, column_group=self.ascii_column_group, symbols=self.ascii_symbols
                ),
                matching=self._matching(),
                status=ctx.status,
            )
            payload = report.kernel_payload()
            call = (
                "install_suggestion(globals(), " + repr(payload) + ")"
                if payload is not None
                else "clear_suggestion(globals())"
            )
            output, failed = await repl.run_harness(
                f"import runtime._repl_state as {_HARNESS_ALIAS}\n{_HARNESS_ALIAS}.{call}\ndel {_HARNESS_ALIAS}",
                timeout=_HARNESS_CELL_TIMEOUT_S,
            )
            if failed:
                raise SearchHarnessError(f"the REPL did not reproduce the search's path:\n{output}")
            plan = report.plan
            diagnostics = report.result.value.diagnostics if report.result.value is not None else None
            early = asked_early(call_output)
            span.kv(
                # Nothing measured — no duration, node count or memory figure:
                # they vary from machine to machine, and a recorded search
                # carrying them would never replay.
                outcome=report.result.outcome,
                exit_reason=diagnostics.exit_reason if diagnostics is not None else None,
                target=plan.target if plan is not None else None,
                boundary_type=plan.diagnostics.boundary_type if plan is not None else None,
                path_length=len(plan.actions) if plan is not None else None,
                stuck_in=report.result.stuck_in,
                asked_early=early,
            )
        return report.text(asked_early=early)

    def _ensure_workspace(self, ctx: LLMMoveContext | None = None) -> Workspace:
        """The workspace the file tools act on, built once and kept."""
        if self.workspace is None:
            self.workspace = Workspace(repl=self._ensure_repl(ctx))
        return self.workspace

    async def _run_turn_block(
        self, ctx: LLMMoveContext, user_prompt: ullm.conv.UserPrompt, *, remaining_calls: int | None
    ) -> int:
        """One [UserPrompt, ResponseTurn] block, and the widest seam onto
        the LLM layer. Returns the number of LLM calls made.

        ullm owns the loop: ``ResponseManager`` runs a single call with the
        malformed-output retry and prompt-too-long defenses behind it, and
        ``TurnManager`` runs rounds of (call, execute the tool calls, feed
        the results back) until the model stops asking for tools. The seal
        supplies the pieces that are its own business — the conversation,
        the tool, a per-round hook that trims to the context budget
        (``before_round``), and a stop condition that ends the turn as soon
        as the model has acted (``stop_when``).

        The conversation is built fresh around ``self.messages`` each block
        rather than kept, because the system prompt and the tool schema
        follow configuration and the element list is edited in place by
        eviction; ``self.messages`` IS the live list the manager appends
        to."""
        tools = self._turn_tools(ctx)
        repl = self._ensure_repl(ctx)
        self.messages.append(user_prompt)
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
        response_manager = ullm.manager.ResponseManager(
            backend=ctx.backend,
            conversation=conversation,
            sampling_params=self._sampling(ctx),
            max_retries=max(0, int(self.malformed_retries)),
            preserve_failed_draws=self.preserve_failed_draws,
            before_attempt=self._observe_request_attempt if self.context_observer is not None else None,
        )
        # DiagnosticTurnManager: same TurnManager semantics, plus the
        # llm/turn span (per-response events, kernel/exec events drained
        # from the repl, the surviving live conversation — pre-history
        # included — and separately recorded rounds that compaction removed).
        # Inert on the null stream, so replay references are unaffected.
        # What is left of the level's allowance becomes this turn's ceiling, so
        # one turn cannot overshoot the level by a whole turn's worth of
        # generation; ullm lowers each round's max_tokens against it, which is
        # what makes the stop land at the next call rather than the next entry.
        per_turn_cap = self._per_turn_cap(ctx)
        started_on = self._current_level(ctx)
        started_used = self.level_tokens_used
        game_tokens_before = self._game_tokens_used(ctx)
        summarizing = self._uses_context_summary()
        turn_manager = platypus.llm.DiagnosticTurnManager(
            response_manager=response_manager,
            conversation=conversation,
            max_tool_rounds=remaining_calls if remaining_calls is not None else 10**9,
            # A side turn plays nothing: it ends when the model replies
            # without a tool call.
            stop_when=None if self._side_turn is not None else self._stop_when_acted,
            # Summary calls happen inside ``before_round`` and therefore sit
            # outside ullm's local turn counter. The summarize hook instead
            # lowers each ordinary call against the live backend meter.
            max_generated_tokens_per_turn=None if summarizing else per_turn_cap,
            # Asked per round: a turn can outlast the queue that justified
            # capping it, and the cap must stop binding the moment it does.
            per_turn_cap_lifted=lambda: self._per_turn_cap_lifted(ctx),
            diag=ctx.diag,
            label=self._side_turn or "analyzer",
            after_response=self.response_observer,
            after_tool_result=lambda result: self._suggest_moves(ctx, result),
        )

        async def trim_before_round() -> None:
            """Trim through this manager's active discarded-round channel,
            then offer this game's slot back to the run."""
            self._audit_check(ctx, started_on, started_used, game_tokens_before)
            await self._trim_live_messages(record_pruned_round=turn_manager.record_context_pruned_round)
            await self._close_ended_thread(ctx.backend, conversation.history_epoch)
            await self._offer_slot(ctx, started_on, started_used, game_tokens_before)

        ordinary_max_tokens = cast(ullm.backend.GenericSamplingParameters, response_manager.sampling_params).max_tokens
        compaction_calls = 0
        ordinary_calls = 0
        summaries_before = self.context_summaries

        def live_cap() -> int | None:
            """What is left of the level and game allowances now, mid-turn."""
            return self._live_turn_cap(ctx, started_used, game_tokens_before)

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

        async def summarize_before_round() -> None:
            """The summarize arm's share of ``before_round``: summarize on
            overflow, then measure the ordinary request, as the prune arm's
            trim does."""
            nonlocal compaction_calls, ordinary_calls
            self._audit_check(ctx, started_on, started_used, game_tokens_before)
            self._prepare_context_observer()
            self.response_step += 1
            self._rounds_since_compaction += 1
            self._arm_prompt_injection()
            if call_allowance_exhausted():
                raise _CallAllowanceExhausted
            try:
                added, measured = await self._prepare_summarized_request(
                    ctx,
                    ordinary_max_tokens,
                    live_cap,
                    summary_call_limit=summary_call_limit(),
                    record_pruned_round=turn_manager.record_context_pruned_round,
                )
            except _CallAllowanceExhausted as exc:
                compaction_calls += exc.calls_made
                raise
            compaction_calls += added
            if call_allowance_exhausted():
                raise _CallAllowanceExhausted
            self._pending_bounds.append(measured)
            self._arm_zero_recurrent(int(measured))
            self._record_prompt_injection_epoch()
            await self._close_ended_thread(ctx.backend, conversation.history_epoch)
            # After the summary, so the offer says whether the prefix this
            # round would have reused is already gone.
            await self._offer_slot(ctx, started_on, started_used, game_tokens_before)
            ordinary_calls += 1

        def discard_unsent_empty_turn(*, keep_prompt: bool = False, served: bool = True) -> bool:
            """Remove a call-less turn, and its prompt unless a summary kept it
            or ``keep_prompt`` asks for it to stay. ``served`` when a draw of
            the turn reached the model: removing it rewrites served history.
            A turn stopped before its first request went out removed nothing
            served, and the next prompt extends the thread the engine holds,
            which a declared rewrite would close ahead of it."""
            tail = self.messages[-1] if self.messages else None
            if not isinstance(tail, ullm.conv.ResponseTurn) or tail.elements:
                return False
            if served:
                self._mark_history_rewritten()
            self.messages.pop()
            if self.context_summaries == summaries_before and not keep_prompt:
                self.messages[:] = [element for element in self.messages if element is not user_prompt]
            self._pending_bounds.clear()
            return True

        turn_manager.before_round = summarize_before_round if summarizing else trim_before_round
        # The trims edit the live conversation's system prompt and declare
        # rewrites on it; the backend/params pair is what token-native
        # measurement weighs the next request with. Cleared below so nothing
        # live rides a snapshot.
        self._live_conversation = conversation
        self._live_backend = ctx.backend
        self._live_params = response_manager.sampling_params
        self._status_handle = ctx.status
        self._pending_bounds.clear()
        self._pruned_rounds = 0  # the note counts this turn's losses
        allowance_stopped_empty = False
        malformed_stopped_empty = False
        audit_stopped_empty = False
        try:
            try:
                with ullm.conv.observe_history_rewrites(self._observe_history_rewrite):
                    await turn_manager.run()
            except ullm.backend.MalformedOutputError as error:
                if not self._nudges_past(error):
                    if summarizing and not self.preserve_failed_draws:
                        discard_unsent_empty_turn()
                    raise
                # Every draw of the round was refused and rolled back, so the
                # round has no response: its pre-call bound goes too. The
                # prompt that opened the turn was sent and stays; an empty
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
                self._malformed_nudge_due = True
                platypus.diagnostics.current_span().event(
                    "malformed_nudge", n=self._malformed_nudged, error=str(error)[:300]
                )
            except _AuditDue:
                # The audit runs ahead of this round's call. Completed rounds
                # stay; an empty turn goes, and the prompt that opened it stays,
                # to be sent ahead of the audit's.
                audit_stopped_empty = discard_unsent_empty_turn(keep_prompt=True, served=False)
                if not audit_stopped_empty:
                    turn = self.messages[-1]
                    assert isinstance(turn, ullm.conv.ResponseTurn)
                    turn.status = "processed"
                self._audit_due = True
            except _CallAllowanceExhausted:
                # before_round can spend the last call on compaction. An
                # empty turn is not a valid durable element and its ordinary
                # prompt was never sent; a nonempty turn contains earlier
                # completed rounds and remains valid history.
                allowance_stopped_empty = discard_unsent_empty_turn(served=False)
                if not allowance_stopped_empty:
                    turn = self.messages[-1]
                    assert isinstance(turn, ullm.conv.ResponseTurn)
                    turn.status = "processed"
            except Exception:
                # A retained draw leaves history as served: the unsent turn
                # stays, and nothing is rewritten after the failure.
                if summarizing and not self.preserve_failed_draws:
                    discard_unsent_empty_turn()
                raise
        finally:
            # Declarations the turn manager made on the live conversation
            # (retry and cap-lift rollbacks) outlive this block's Conversation.
            self.history_epoch = max(self.history_epoch, conversation.history_epoch)
            # The live conversation is deliberately NOT cleared here: the
            # post-turn retention trim runs after this block and must refresh
            # the same system prompt and declare on the same epoch counter.
            # _analyzer_turns_until_acted clears it, still before the move
            # ends, so nothing live rides a snapshot (A4).
            #
            # Kill the kernel while idle so the agent stays picklable at move
            # boundaries (A4). Only in "round" mode is that also the reset
            # cadence; the longer-lived modes keep the kernel and give up the
            # move-boundary snapshot in exchange (supported_checkpointing
            # reports that).
            if self.persistence == "round" and repl.state != "running":
                repl.reset()
            if summarizing:
                # Summary-only failures have paid the backend too. Publish
                # that spend even when no ordinary analyzer response survives.
                self._charge_budget(ctx, started_on, started_used, game_tokens_before)
        # Host-call failures are seal bugs, not model-visible tool errors —
        # fail loudly (TurnManager swallows tool exceptions into results).
        pending_host_exception = repl.pending_host_exception
        if pending_host_exception is not None:
            raise pending_host_exception
        if allowance_stopped_empty or malformed_stopped_empty or audit_stopped_empty:
            assert not self._pending_bounds
            return compaction_calls + ordinary_calls
        turn = self.messages[-1]
        assert isinstance(turn, ullm.conv.ResponseTurn), "TurnManager must append its ResponseTurn to our history"
        if not summarizing:
            self._charge_budget(ctx, started_on, started_used, game_tokens_before)
        self.counts_reconciled += self._check_engine_usage(turn)
        if summarizing:
            return compaction_calls + ordinary_calls
        return sum(1 for element in turn.elements if isinstance(element, ullm.conv.LLMResponse))

    async def _offer_slot(
        self, ctx: LLMMoveContext, started_on: int, started_used: int, game_tokens_before: int
    ) -> None:
        """Offer this game's concurrency slot back, at a round boundary.

        Offered on every round rather than at chosen moments, because the
        two things that decide whether an offer is worth taking up belong
        to different owners: whether a swap is free right now is the
        agent's fact (a trim just fired, so the prefix is gone anyway),
        and how long is too long to hold a slot is the run's — a clock,
        which agents do not read (A2). So this reports both the swap
        point and the spend, and the host decides.

        The spend reported is live to the token: ``level_tokens_used`` is
        only charged when a turn ends, and the ranking is made of these
        numbers, so a game mid-turn would otherwise look a whole turn
        cheaper than it is. Charged the way ``_charge_budget`` charges —
        a turn that crossed a level belongs to the level it ran under,
        and the new level starts at the counter's own zero.
        """
        crossed = self._current_level(ctx) != started_on
        used = self.level_tokens_used if crossed else started_used + self._game_tokens_used(ctx) - game_tokens_before
        await ctx.yield_slot(
            {
                "level": self._current_level(ctx),
                "level_tokens": used,
                # The game's whole spend, live like the level's: what the
                # least_game_tokens policy ranks on.
                "game_tokens": self._game_tokens_used(ctx),
                # The trim that just ran, if it fired, invalidated the
                # prefix: a swap costs nothing the round has not lost.
                "compacted": self._last_trim_fired,
            }
        )

    def _stop_when_acted(self, response: ullm.conv.LLMResponse, result: ullm.conv.ToolResult) -> str | None:
        """TurnManager stop hook: end the turn once the model has acted
        in-snippet. Mirrors the original's ``if step_executed: break``: the
        analyze turn ends right after the first acting tool call; the next
        move re-enters with a fresh state prompt. The skip text (the
        original duck's JSON error shape) goes onto any not-yet-executed
        tool calls of the response."""
        del response, result
        if not self._acted():
            return None
        return json.dumps({"error": self.prompts.already_acted_skip.format(verb="executed")}, indent=2)

    def _mark_moveless(self, ctx: LLMMoveContext) -> None:
        """Start ``suggest_moves_after_tokens``'s count again from now."""
        self._moveless_since = self._game_tokens_used(ctx)
        self._moveless_moves = self.action_count

    def _suggest_moves(self, ctx: LLMMoveContext, result: ullm.conv.ToolResult) -> ullm.conv.ToolResult:
        """TurnManager result hook: ``result`` with the suggestion to move at
        its end once ``suggest_moves_after_tokens`` have passed without a move
        (``_moveless_since``), on every result from then on. Not in a side
        turn, where ``submit_action`` is refused.

        A move is noticed here, at the end of the call that made it, by the
        action count: moves are made only inside calls, and nothing is
        generated between a move and its call's result, so the meter read now
        is the meter at the move, and that call's own result goes without."""
        if not self.suggest_moves_after_tokens:
            return result
        if self.action_count != self._moveless_moves:
            self._mark_moveless(ctx)
        if self._side_turn is not None:
            return result
        idle = self._game_tokens_used(ctx) - self._moveless_since
        if idle < self.suggest_moves_after_tokens:
            return result
        note = self.prompts.suggest_moves_note.format(tokens=f"{idle:,}")
        content = result.content
        if isinstance(content, ullm.conv.StringContent):
            noted: ullm.conv.Content = ullm.conv.StringContent(text=f"{content.text}\n\n{note}")
        elif isinstance(content, ullm.conv.ListContent):
            noted = ullm.conv.ListContent(elements=[*content.elements, ullm.conv.StringContent(text=note)])
        else:
            noted = ullm.conv.ListContent(elements=[content, ullm.conv.StringContent(text=note)])
        return dataclasses.replace(result, content=noted)

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

    def _last_completion_limit(self) -> int | None:
        """Generated-token limit when the latest response ended by hitting it."""
        turn = self.messages[-1]
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
        """Build a new continuation prompt without modifying any prior message."""
        completion_limit = self._last_completion_limit() if self.completion_cutoff_notice else None
        text = self.prompts.followup_nudge(completion_limit, self._prompt_modes())
        return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))

    # ------------------------------------------------------------------
    # 4. acting
    #
    # What the kernel's get_state() and submit_action(...) actually
    # run in the agent process while the snippet blocks on them. Arguments
    # were written by the model, so everything here validates before it
    # acts.
    # ------------------------------------------------------------------

    def _state_payload(self) -> dict[str, Any]:
        """The runtime-state payload served to the kernel's ``get_state`` host call
        at the start of every snippet, and with every action reply."""
        turns = self._kernel_level_entries()
        payload: dict[str, Any] = {
            "level": turns[-1].level if turns else None,
            "turns": [turn_payload(entry) for entry in turns],
            "available_actions": list(self.current_valid_actions),
            "exploration": {
                "moves_left": self.exploration_moves_left,
                "over": self.exploration_over,
            },
            "matching": self._matching(),
        }
        if self.level_reflection != "none" and self._won_level_this_move():
            # A reflection turn follows the call that won: the kernel describes
            # the win for it, as it describes any other move.
            payload["reflection"] = True
        return payload

    def _matching(self) -> dict[str, Any]:
        """The matching settings in force (``seal.harness.test.MatchRules``):
        what the model set this game, else the starting values."""
        return {
            "strict_energy": self.strict_energy if self.matching_strict_energy is None else self.matching_strict_energy,
            "pixel_tolerance": (
                self.pixel_tolerance if self.matching_pixel_tolerance is None else self.matching_pixel_tolerance
            ),
            "tolerance_offered": self.expose_pixel_tolerance,
        }

    def _set_matching(self, changes: dict[str, Any]) -> dict[str, Any]:
        """The kernel's setters, validated host-side since the model can reach
        them: a pixel tolerance only where it is offered."""
        unknown = set(changes) - {"strict_energy", "pixel_tolerance"}
        if unknown:
            return {"error": f"unknown setting {sorted(unknown)[0]!r}"}
        if "strict_energy" in changes and not isinstance(changes["strict_energy"], bool):
            return {"error": "set_strict_energy takes True or False."}
        if "pixel_tolerance" in changes:
            value = changes["pixel_tolerance"]
            if not self.expose_pixel_tolerance:
                return {"error": "There is no pixel tolerance to set in this game."}
            if type(value) is not int or value < 0:
                return {"error": "set_pixel_tolerance takes a whole number, 0 or more."}
        # All checked before any applies: a refused call changes nothing.
        if "strict_energy" in changes:
            self.matching_strict_energy = changes["strict_energy"]
        if "pixel_tolerance" in changes:
            self.matching_pixel_tolerance = changes["pixel_tolerance"]
        return {"matching": self._matching()}

    def _current_level_entries(self) -> list[TimelineEntry]:
        """This level's turns, oldest first."""
        return _level_suffix(self.history_entries)

    def _kernel_level_entries(self) -> list[TimelineEntry]:
        """The turns the kernel's ``history`` holds: this level's, except in
        the rest of a move that won a level. There it is the level just won,
        ending on its winning turn, so the call that won can read the win.
        The new level replaces it when the move ends, where the kernel's
        level reset is paid too, and nothing in between acts on the old
        board: a later ``submit_action`` in the move is refused."""
        if not self._won_level_this_move():
            return self._current_level_entries()
        # The new level's opening turn is the one entry filed after the win.
        return _level_suffix(self.history_entries[:-1])

    def _won_level_this_move(self) -> bool:
        """Whether this move won a level, the game's last excepted."""
        return self.terminal_this_move is not None and bool(self.terminal_this_move.get("level_completed"))

    def _stage_actions(self, normalized: list[dict[str, Any]]) -> tuple[list[Action], str | None]:
        """Shape/name validation for a model-authored action batch:
        engine-map each item, clamp MOUSE coords. Returns
        ``(staged actions, error)`` — on error, staged is the empty prefix
        and nothing may execute."""
        staged: list[Action] = []
        note = self._valid_actions_note()
        for index, item in enumerate(normalized, start=1):
            name = str(item.get("action", "")).strip()
            engine_name = to_engine_action(name)
            if engine_name is None:
                return [], self.prompts.stage_error_unknown_action.format(index=index, name=name) + note
            if engine_name == "ACTION7" and not self.expose_undo:
                return [], self.prompts.stage_error_undo_disabled.format(index=index) + note
            if engine_name == "ACTION6":
                try:
                    row = int(item.get("row"))  # type: ignore[arg-type]
                    col = int(item.get("col"))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return [], self.prompts.stage_error_mouse_coords.format(index=index) + note
                row = max(0, min(63, row))
                col = max(0, min(63, col))
                action = Action("ACTION6", row=row, col=col)
            else:
                action = Action(engine_name)
            staged.append(action)
        return staged, None

    def _valid_actions_note(self) -> str:
        """What a refused batch ends with: the game's actions, by the names
        the model writes them with."""
        if not self.current_valid_actions:
            return ""
        return self.prompts.valid_actions_note.format(actions=", ".join(self.current_valid_actions))

    def _game_over_refusal(self, action: Action) -> str | None:
        """Model-visible refusal for acting past a loss. arcengine returns an
        empty frame for a non-RESET action after GAME_OVER and TAAF raises on
        that, so this refusal is what keeps the game alive — it is not merely
        advisory."""
        if not self.game_over_active or action.name == "RESET":
            return None
        return self.prompts.game_over_refusal.format(action=to_model_action(action.name))

    def _terminal_action_stop_detail(self, reason: str | None) -> str:
        """Explain a terminal batch stop."""
        surface = self.prompts
        if reason == "run_complete":
            return surface.stop_detail_run_complete
        if reason == "game_over":
            return surface.stop_detail_game_over
        if reason == "level_completed":
            return surface.stop_detail_level_completed
        if reason == "done":
            return surface.stop_detail_done
        return surface.stop_detail_terminal

    async def _execute_actions_live(
        self,
        ctx: LLMMoveContext,
        normalized: list[dict[str, Any]],
        *,
        pause_each: bool = False,
        route: str = "unchecked",
    ) -> dict[str, Any]:
        """The original's in-snippet ``submit_action()``: execute each action
        immediately via ``ctx.execute``, folding every resulting
        observation into the bookkeeping. Returns the original's reply
        shape — the compact result of the last executed action plus the
        refreshed runtime state (the kernel re-installs it, so the
        snippet's globals are live after the call). A batch stops early on
        a terminal/level event, an action that is no longer valid, or an
        exhausted ``max_moves`` cap; once a terminal result exists this
        move, further calls are no-ops carrying the original's
        stop_detail.

        ``pause_each`` is the kernel stepping the batch under an armed world
        model: every executed move returns a ``paused`` reply, and the kernel
        continues or stops the batch (``action_continue`` / ``action_stop``).
        ``route`` is how the kernel sent it, for the ``seal/moves`` span."""
        assert ctx.execute is not None, "in-snippet execution requires the move loop's ctx.execute"
        # A new batch supersedes one the kernel abandoned between moves (its
        # cell died or timed out); the moves that batch executed are already
        # folded into the bookkeeping.
        self.paused_batch = None
        refused = self._refused(ctx, normalized, route=route)
        if refused is not None:
            return refused

        staged, error = self._stage_actions(normalized)
        if error is not None:
            self._record_moves(ctx, route=route, requested=len(normalized), refused=error)
            return {"error": error}
        batch = _Batch(staged=staged, requested=len(normalized), route=route)
        return await self._run_batch(ctx, batch, pause_each=pause_each)

    async def _run_batch(self, ctx: LLMMoveContext, batch: _Batch, *, pause_each: bool) -> dict[str, Any]:
        """Execute ``batch`` from where it stands. With ``pause_each``, return
        after every executed move with the batch parked on the agent, so the
        kernel can check the new frame and continue or stop it; the early
        stops and the final result are this one loop's either way."""
        assert ctx.execute is not None, "in-snippet execution requires the move loop's ctx.execute"
        if batch.terminal_reason is not None:
            # Paused right after a terminal move: nothing more executes.
            return self._finish_batch(ctx, batch)
        while batch.index < len(batch.staged):
            action = batch.staged[batch.index]
            # The action cap must hold inside a batch too: this loop is the
            # only place actions execute, and _do_move's own check cannot see
            # mid-snippet growth.
            if self.max_moves is not None and self.action_count >= self.max_moves:
                batch.stop("max_moves", self.prompts.stop_detail_max_moves)
                break
            # Re-checked per action: a RESET earlier in the batch clears the
            # game-over state, so the rest of the batch runs normally.
            refusal = self._game_over_refusal(action)
            if refusal is not None:
                if not batch.executed:
                    self._record_moves(ctx, route=batch.route, requested=batch.requested, refused=refusal)
                    return {"error": refusal}
                batch.stop("game_over", refusal)
                break
            if action.name not in self.live_available_actions:
                model_name = to_model_action(action.name)
                if not batch.executed:
                    error = self.prompts.invalid_action_error.format(action=model_name) + self._valid_actions_note()
                    self._record_moves(ctx, route=batch.route, requested=batch.requested, refused=error)
                    return {"error": error}
                batch.stop("invalid_action", self.prompts.stop_detail_invalid_action.format(action=model_name))
                break
            self.awaiting_action = action
            observations = await ctx.execute(action)
            for element in observations:
                self._record_observation(ctx, element)
            batch.executed.append(action)
            batch.index += 1
            self.executed_this_move += 1
            self.live_available_actions = list(observations[-1].available_actions)
            if observations[0].outcome in ("level_won", "game_won", "lost"):
                self.terminal_this_move = dict(self.last_action_result)
                if batch.index < len(batch.staged):
                    reason = _terminal_action_reason(self.last_action_result) or "terminal_state"
                    batch.terminal_reason = reason
                else:
                    batch.terminal_reason = ""
            if pause_each:
                self.paused_batch = batch
                remaining = 0 if batch.terminal_reason is not None else len(batch.staged) - batch.index
                return {
                    "paused": True,
                    "remaining": remaining,
                    "result": dict(self.last_action_result),
                    "state": self._state_payload(),
                }
            if batch.terminal_reason is not None:
                break
        return self._finish_batch(ctx, batch)

    def _finish_batch(self, ctx: LLMMoveContext, batch: _Batch) -> dict[str, Any]:
        """The batch's one result: the last executed action's result with the
        batch fields on it, recorded as ``last_action_result``, and its
        ``seal/moves`` span."""
        if batch.terminal_reason:
            batch.stop(batch.terminal_reason, self._terminal_action_stop_detail(batch.terminal_reason))
        self._record_moves(
            ctx,
            route=batch.route,
            requested=batch.requested,
            executed=len(batch.executed),
            stop_reason=batch.stop_reason,
        )
        if batch.executed and self.pending_batch is not None:
            self.pending_batch.route = batch.route
        if batch.executed and self.workspace is not None:
            # The board has moved, so the file state that produced it is
            # history: undo_edit reaches back to the last executed action
            # and no further.
            self.workspace.forget_changes()
        result = dict(self.last_action_result)
        result["executed_actions"] = [model_move(action) for action in batch.executed]
        if batch.requested > 1 or batch.stopped_early:
            result["requested_count"] = batch.requested
            result["executed_count"] = len(batch.executed)
            result["stopped_early"] = batch.stopped_early
        if batch.stop_reason:
            result["stop_reason"] = batch.stop_reason
        if batch.stop_detail:
            result["stop_detail"] = batch.stop_detail
        self.last_action_result = dict(result)
        return {"result": result, "state": self._state_payload()}

    def _record_moves(
        self,
        ctx: LLMMoveContext,
        *,
        route: str,
        requested: int,
        executed: int = 0,
        stop_reason: str | None = None,
        refused: str | None = None,
    ) -> None:
        """One ``seal/moves`` span per acting call that reached the host: how
        it was sent, how much of it ran, and why it stopped or was refused —
        so a run can say how much of its play a world model checked."""
        with ctx.diag.span("seal/moves") as span:
            if self.plan_argument != "none":
                plan = self._call_plan()
                span.kv(
                    plan_chars=None if plan is None else len(plan),
                    plan_first=self.repl.call_plan_first if self.repl is not None else None,
                )
            span.kv(
                mode=self.use_world_model,
                route=route,
                requested=requested,
                executed=executed,
                stop_reason=stop_reason,
                world_model_stop=_WORLD_MODEL_STOPS.get(stop_reason or ""),
                refused=None if refused is None else refused[:_STOP_DETAIL_CAP],
            )

    async def _continue_paused_batch(self, ctx: LLMMoveContext) -> dict[str, Any]:
        """Resume the batch the kernel paused after its last checked move."""
        batch, self.paused_batch = self.paused_batch, None
        if batch is None:
            return {"error": "No action batch is waiting to continue."}
        return await self._run_batch(ctx, batch, pause_each=True)

    def _stop_paused_batch(self, ctx: LLMMoveContext, reason: str, detail: str) -> dict[str, Any]:
        """End the paused batch where it stands, for the kernel's world
        model: it mispredicted the move it just checked, or declined to
        predict it. ``detail`` says which move and how."""
        batch, self.paused_batch = self.paused_batch, None
        if batch is None:
            return {"error": "No action batch is waiting to stop."}
        if reason not in _WORLD_MODEL_STOPS:
            return {"error": f"Unknown stop reason {reason!r}."}
        batch.terminal_reason = None
        batch.stop(reason, detail[:_STOP_DETAIL_CAP])
        if self.pending_batch is not None:
            self.pending_batch.world_model_stop = _WORLD_MODEL_STOPS[reason]
        batch.stopped_early = batch.index < len(batch.staged)
        return self._finish_batch(ctx, batch)

    def _refused(self, ctx: LLMMoveContext, normalized: list[dict[str, Any]], *, route: str) -> dict[str, Any] | None:
        """The reply to a batch no move of may run, or None when it may.
        Answered before any route's own rule: an exploration budget or the
        gate's would otherwise be spent, or refuse, on a turn or level the
        batch has no business in.

        - In a turn that is not for acting (the reflection, the audit), an
          error: the prompt says ``submit_action`` is not available there.
        - After this move ended its level or game, the terminal reply."""
        if self._side_turn is not None:
            refusal = self.prompts.no_action_turn_refusal
            self._record_moves(ctx, route=route, requested=len(normalized), refused=refusal)
            return {"error": refusal}
        if self.terminal_this_move is None:
            return None
        self._record_moves(ctx, route=route, requested=len(normalized), refused="terminal")
        return self._terminal_no_op_reply(normalized)

    def _terminal_no_op_reply(self, normalized: list[dict[str, Any]]) -> dict[str, Any]:
        """The reply to a ``submit_action(...)`` issued after this move already
        reached a terminal state. Nothing executes — the move is over as far
        as the game is concerned — but the model is mid-snippet and gets a
        result in the shape it expects, restating the terminal event and
        saying plainly why the request was dropped."""
        terminal = self.terminal_this_move or {}
        reason = _terminal_action_reason(terminal) or "terminal_state"
        guard: dict[str, Any] = {
            "executed": False,
            "action_num": terminal.get("action_num"),
            "level": terminal.get("level"),
            "score": terminal.get("score"),
            "reward": 0.0,
            "state": terminal.get("state"),
            "valid_actions": [],
            "board_changed": False,
            "done": bool(terminal.get("done")),
            "level_completed": bool(terminal.get("level_completed")),
            "game_over": bool(terminal.get("game_over")),
            "run_complete": bool(terminal.get("run_complete")),
            "requested_count": len(normalized),
            "executed_count": 0,
            "stopped_early": True,
            "stop_reason": f"previous_{reason}",
            "stop_detail": self._terminal_action_stop_detail(reason),
        }
        self.last_action_result = dict(guard)
        return {"result": guard, "state": self._state_payload()}

    # ------------------------------------------------------------------
    # 5. observation bookkeeping
    #
    # Infra owns the canonical history (I6); this is the seal's own view
    # of it — the original solver's history list, its per-action result
    # dict, and its batch summary, rebuilt from real observations so the
    # payloads the model reads keep the shapes it was prompted about.
    # ------------------------------------------------------------------

    def _frame_payload(
        self,
        obs: Observation,
        *,
        frame: taaf.game.Frame,
        step: int,
        n_levels: int | None,
        level: int | None = None,
    ) -> FramePayload:
        """Serialize one animation or settled frame for the kernel."""
        raw_bytes, shape = _grid_bytes(frame)
        return FramePayload(
            grid_b64=base64.b64encode(raw_bytes).decode("ascii"),
            shape=[shape[0], shape[1]],
            step=step,
            level=self._level_number(obs, n_levels) if level is None else level,
        )

    def _animation_payloads(
        self,
        obs: Observation,
        *,
        step: int,
        n_levels: int | None,
        level: int | None = None,
    ) -> list[FramePayload]:
        """Serialize an observation's animation, ending exactly once in its settled frame."""
        payloads = [
            self._frame_payload(obs, frame=frame, step=step, n_levels=n_levels, level=level)
            for frame in obs.animation_frames
        ]
        settled = self._frame_payload(obs, frame=obs.frame, step=step, n_levels=n_levels, level=level)
        if not payloads or payloads[-1] != settled:
            payloads.append(settled)
        return payloads

    def _record_observation(self, ctx: LLMMoveContext, obs: Observation) -> None:
        """Consume the incoming observation: mirror the original's history list (seed
        entry, per-action entries, deferred level-win entry), update batch/result
        bookkeeping, refresh valid actions. Runs first in every ``_do_move``
        and once per observation returned by an in-move execute. Idempotent
        per observation: an index already folded (an in-move result arriving
        again as the next move's delivered observation) is a no-op."""
        if obs.index <= self.last_obs_index:
            return
        self.last_obs_index = obs.index
        # A new level gets a fresh allowance. Folded in here rather than at the
        # level-won branch because a level can also turn over mid-move, under an
        # in-move execute, and this is the one path every observation takes.
        if obs.level != self.budget_level:
            self.budget_level = obs.level
            self.level_tokens_used = 0
            self.audits_fired = 0
            self.level_reflection_tokens = 0
        if obs.level != self.exploration_level:
            first = self.exploration_level is None
            self.exploration_level = obs.level
            self.exploration_moves_left = self.exploration_moves_new_game if first else self.exploration_moves_new_level
            self.exploration_over = False
            self.explored_in_cell = -1
        n_levels = ctx.game.number_of_levels

        if obs.caused_by is None:
            if not self.history_entries:
                # Game-initial observation: seed the aligned timeline.
                animation = self._animation_payloads(obs, step=self.action_count, n_levels=n_levels)
                self.history_entries.append(
                    TimelineEntry(action=None, frame=animation[-1], animation=animation, level=animation[-1].level)
                )
            elif self.pending_level_win is not None:
                # A level-winning action arrives as two observations. Its
                # animation spans the old-level win frames and the new-level
                # start frames; its settled endpoint is the new-level board.
                step = self.pending_level_win.step
                new_level_animation = self._animation_payloads(obs, step=step, n_levels=n_levels)
                animation = [*self.pending_level_win.animation]
                for frame in new_level_animation:
                    if not animation or animation[-1] != frame:
                        animation.append(frame)
                payload = animation[-1]
                previous_grid = self.history_entries[-1].frame.grid_b64
                self.history_entries.append(
                    TimelineEntry(
                        action=self.pending_level_win.action,
                        frame=payload,
                        animation=animation,
                        outcome="won",
                        level=self.pending_level_win.level,
                        plan=self.pending_level_win.plan,
                    )
                )
                batch = self.pending_batch
                if batch is not None and batch.turns and batch.turns[-1][0] is None:
                    # The winning move's turn, filed now, at the end of its level.
                    batch.turns[-1] = (len(self.history_entries) - 1, batch.turns[-1][1])
                # Every level's history opens with a seed turn, so the board
                # the model acts from exists before it has acted. Without it
                # the new level would start with no turns at all, since the
                # winning action's own turn belongs to the level it ended.
                self.history_entries.append(
                    TimelineEntry(
                        action=None,
                        frame=payload,
                        animation=[payload],
                        level=payload.level,
                    )
                )
                # The winning action's board_changed compares its pre-action
                # board to this frame (the original compared pre vs post).
                board_changed = previous_grid != payload.grid_b64
                if self.pending_batch is not None:
                    self.pending_batch.board_changed = self.pending_batch.board_changed or board_changed
                if self.last_action_result.get("action_num") == self.pending_level_win.step:
                    self.last_action_result["board_changed"] = board_changed
                self.pending_level_win = None
        else:
            action = self.awaiting_action or obs.caused_by
            self.awaiting_action = None
            # Before the turn is filed: a click's colour is the board it clicked.
            label = _executed_label(action, self.history_entries[-1].frame if self.history_entries else None)
            # The original stamped step = len(history) AFTER the engine
            # appended the record: action N reads step N.
            self.action_count += 1
            board_changed: bool | None
            if obs.outcome == "level_won":
                # The settled endpoint belongs to the new level, so retain the
                # old-level animation until its split initial observation arrives.
                completed_level = max(1, int(obs.state.levels_completed))
                animation = self._animation_payloads(
                    obs,
                    step=self.action_count,
                    n_levels=n_levels,
                    level=completed_level,
                )
                level_entries = self._current_level_entries()
                # The next prompt opens the new level, whose history the winning turn is not in: no position.
                turn = (None, _kept_count(level_entries[-1].frame if level_entries else None, animation))
                self.pending_level_win = PendingLevelWin(
                    action=action,
                    step=self.action_count,
                    animation=animation,
                    level=completed_level,
                    plan=self._call_plan(),
                )
                board_changed = None  # resolved when the new-level frame arrives
                change = None  # its settled frame is the next level's
            else:
                animation = self._animation_payloads(obs, step=self.action_count, n_levels=n_levels)
                payload = animation[-1]
                previous = self.history_entries[-1].frame if self.history_entries else None
                previous_grid = previous.grid_b64 if previous is not None else None
                self.history_entries.append(
                    TimelineEntry(
                        action=action,
                        frame=payload,
                        animation=animation,
                        outcome=_TURN_OUTCOMES.get(obs.outcome, "playing"),
                        level=payload.level,
                        plan=self._call_plan(),
                    )
                )
                board_changed = previous_grid is not None and previous_grid != payload.grid_b64
                change = _move_change(previous, payload)
                level_entries = self._current_level_entries()
                turn = (
                    len(self.history_entries) - 1,
                    _kept_count(level_entries[-2].frame if len(level_entries) > 1 else None, animation),
                )
            self._update_pending_batch(action, label, obs, board_changed, n_levels, turn, change)
            if action.name == "RESET" and obs.outcome != "level_won" and self.use_world_model != "none":
                # A RESET that did not restore the level's first board: the
                # replay rebuilds every RESET from initial_state(), so the model
                # is told, and pointed at wildcards for the cells.
                differing = _cells_differing(self._current_level_entries()[0].frame, self.history_entries[-1].frame)
                if differing:
                    assert self.pending_batch is not None
                    self.pending_batch.irregular_reset = differing
                    platypus.diagnostics.current_span().event("seal/irregular_reset", cells=differing)
            if self.attempts.record(action, obs.level) > 1 and self.replay_reminder:
                assert self.pending_batch is not None
                self.pending_batch.repeated = [_executed_label(played, None) for played in self.attempts.closed]

        if obs.outcome == "level_won":
            self.pending_level_reset = True
        self.last_levels_completed = int(obs.state.levels_completed)
        self.game_over_active = obs.outcome == "lost"
        self.current_valid_actions = self._model_valid_actions(obs)
        self.valid_actions_observed = True
        if not self.game_action_names:
            self.game_action_names = self._game_action_names(obs)

    def _call_plan(self) -> str | None:
        """The running python call's ``plan``, for the turns it plays."""
        return self.repl.call_plan if self.repl is not None else None

    def _game_action_names(self, obs: Observation) -> list[str]:
        """The game's action set as the model is taught it — the engine's own
        list, minus UNDO unless ``expose_undo`` offers it."""
        return [
            to_model_action(name)
            for name in obs.available_actions
            if str(name).upper() != "ACTION7" or self.expose_undo
        ]

    def _model_valid_actions(self, obs: Observation) -> list[str]:
        """Model-facing valid actions. The engine keeps its list wide after a
        loss (it just refuses to advance), so the lost state advertises the
        one action that works."""
        if obs.outcome == "lost":
            return ["RESET"]
        return self._game_action_names(obs)

    def _level_number(self, obs: Observation, n_levels: int | None) -> int:
        """The original's 1-based level label: levels_completed+1, capped at the
        level count; the count itself on a game win."""
        completed = int(obs.state.levels_completed)
        if obs.outcome == "game_won":
            return max(1, int(n_levels)) if n_levels else max(1, completed)
        level = completed + 1
        if n_levels:
            level = min(int(n_levels), level)
        return max(1, level)

    def _update_pending_batch(
        self,
        action: Action,
        label: str,
        obs: Observation,
        board_changed: bool | None,
        n_levels: int | None,
        turn: tuple[int | None, int],
        change: MoveChange | None,
    ) -> None:
        """Fold one executed action into the drained-batch record and rebuild
        ``last_action_result`` (the original's compact result, reconstructed from
        the real observation). ``turn`` is where the move's entry sits in
        ``history_entries`` (None for a level win) and how many animation
        frames its turn keeps; ``change`` what it changed on the board."""
        if self.pending_batch is None:
            self.pending_batch = PendingBatch()
        batch = self.pending_batch
        batch.executed_actions.append(label)
        batch.turns.append(turn)
        batch.changes.append(change)
        batch.executed_count += 1
        batch.level = self._level_number(obs, n_levels)
        # board_changed is None on a level win — resolved retroactively when
        # the new-level frame arrives (see _record_observation).
        batch.board_changed = batch.board_changed or bool(board_changed)

        if obs.outcome == "level_won":
            batch.level_transition = True
            batch.stop_reason = "level_completed"
        elif obs.outcome == "game_won":
            batch.run_complete = True
            batch.stop_reason = "run_complete"
        elif obs.outcome == "lost":
            batch.game_over = True
            batch.stop_reason = "game_over"

        # The original's per-action reward: levels-completed delta over the
        # game's level count.
        completed = int(obs.state.levels_completed)
        reward = float(completed - self.last_levels_completed) / max(1.0, float(n_levels or 1))

        # Reconstruction of the original's persisted compact action result.
        self.last_action_result = {
            "executed": True,
            "action_num": self.action_count,
            "level": batch.level,
            "score": completed,
            "reward": reward,
            "state": obs.state.raw.state.name,
            "valid_actions": self._model_valid_actions(obs),
            "board_changed": bool(board_changed),
            "done": obs.outcome == "game_won",
            "level_completed": obs.outcome == "level_won",
            "game_over": obs.outcome == "lost",
            "run_complete": obs.outcome == "game_won",
            "action": model_move(action),
            "executed_actions": [model_move(action)],
        }

    def _take_step_summary(self) -> None:
        """Close the drained batch into the summary the next prompt reads."""
        batch = self.pending_batch
        self.pending_batch = None
        if batch is None or not batch.executed_count:
            # A no-action re-entry keeps the previous summary rather than
            # clearing it: the model is told what its last batch did, and
            # that is still the last batch there was.
            return
        # A move's history index, when its turn is in the history the kernel
        # now holds: the finished level's, in the reflection turn after a win.
        end = len(self.history_entries) - (1 if self._won_level_this_move() else 0)
        start = end - len(self._kernel_level_entries())
        animations: list[tuple[int, int | None] | None] = [
            None
            if not count
            else (count, None if position is None or not start <= position < end else position - start)
            for position, count in batch.turns
        ]
        self.last_step_summary = StepSummary(
            executed_count=batch.executed_count,
            executed_actions=list(batch.executed_actions),
            animations=animations if self.animation_marks else [],
            changes=list(batch.changes) if self.move_changes else [],
            level=batch.level,
            level_transition=batch.level_transition,
            run_complete=batch.run_complete,
            game_over=batch.game_over,
            world_model_stop=batch.world_model_stop,
            checked={"checked": True, "unchecked": False}.get(batch.route or ""),
            repeated=batch.repeated,
            irregular_reset=batch.irregular_reset,
        )

    # ------------------------------------------------------------------
    # 6. prompts
    #
    # Assembly lives on prompts.SealPrompts: the agent computes the facts
    # (state numbers, the journal snapshot) and the carried surface turns
    # them into text. _prompt_modes() is the one site where agent fields
    # become prompt input, and they travel as an argument rather than onto
    # the surface — so no copy of them can go stale, and no prompt
    # implementation can carry a mode that disagrees with the runtime.
    # ------------------------------------------------------------------

    def _prompt_modes(self) -> PromptModes:
        """What this agent's configuration means for a render, read fresh."""
        return PromptModes(
            persistence=self.persistence,
            file_tool_names=build_file_tool_names(**_FILE_SUITE),
            # The BOARD-image schedule, which is what the prompt describes.
            multimodal=self._multimodal_mode() if self._multimodal_enabled() else "",
            memory_context_tokens=self.memory_context_tokens,
            tool_output_tokens=self.tool_output_tokens,
            python_timeout_s=self._python_timeout_s(),
            action_names=tuple(self.game_action_names),
            ascii_row_numbers=self.ascii_row_numbers,
            ascii_column_group=self.ascii_column_group,
            ascii_symbols=self.ascii_symbols,
            world_model=self.use_world_model != "none",
            world_model_optional=self.use_world_model == "optional",
            world_model_default=self.world_model_default,
            search=self.use_world_model != "none" and self.use_search,
            settled_delta=self.settled_delta,
            animation_description=self.animation_description,
            animation_marks=self.animation_marks,
            journal=self.journal,
            context_summary_enabled=self._uses_context_summary(),
            add_files_and_repl_to_summary=self._uses_context_summary() and self.add_files_and_repl_to_summary,
            plan_argument=self.plan_argument,
            pixel_tolerance_offered=self.expose_pixel_tolerance,
            pixel_tolerance_start=self.pixel_tolerance,
            axis_label_step=AXIS_TICK_STEP if self.axis_labels else None,
            segmentation=self.segmentation,
            replay_reminder=self.replay_reminder,
        )

    def _system_prompt_text(self) -> str:
        """Assemble a prompt whose tools and runtime claims match every mode."""
        if self.memory_prompt_snapshot is None:
            self.memory_prompt_snapshot = self._render_memory(self._memory_snapshot_entries(self.messages))
        return self.prompts.system_prompt(self._prompt_modes(), self.memory_prompt_snapshot or "")

    #: The values ``multimodal_context`` accepts, lowercased.
    MULTIMODAL_MODES: ClassVar[frozenset[str]] = frozenset({"", "current_grid", "level_start"})

    def _multimodal_mode(self) -> str:
        """``multimodal_context``, normalised."""
        return self.multimodal_context.strip().lower()

    def _multimodal_enabled(self) -> bool:
        """True when the grid image rides SOME user prompt.

        Whether it rides THIS one is `_image_rides_this_turn`. The difference
        matters at the two call sites that ask this question: the prompt says
        images are part of the conversation, and the exact-history check asks
        whether an image can ever appear — both of which are about the mode,
        not about the turn.
        """
        return self._multimodal_mode() in {"current_grid", "level_start"}

    def _image_rides_this_turn(self, level: int) -> bool:
        """Whether THIS user prompt carries the grid image.

        Under ``level_start`` the image is reference material for the level
        rather than state, so it is sent once and then sits in the cached
        prefix — the same argument, and the same bookkeeping, as the palette
        legend.
        """
        mode = self._multimodal_mode()
        if mode == "current_grid":
            return True
        if mode == "level_start":
            return self.image_level_sent != level
        return False

    def _build_user_prompt_element(self, obs: Observation) -> ullm.conv.UserPrompt:
        """The per-turn UserPrompt: prompt text plus, in multimodal mode, the
        current frame as an ImageContent (``seal.board_image``)."""
        text = self._build_user_prompt_text()
        if not self._multimodal_enabled():
            return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))
        return self._user_prompt_element(text, self.history_entries[-1].frame)

    def _build_reflection_prompt_element(self) -> ullm.conv.UserPrompt:
        """The reflection turn's prompt (``level_reflection``): the winning
        move reported, and the reflection asked for, over the win screen."""
        won = self._kernel_level_entries()[-1]
        reset_reason = self.repl_reset_pending or None
        self.repl_reset_pending = ""
        frame_delta, self.frame_delta_text = self.frame_delta_text, None
        animation_text, self.animation_text = self.animation_text, None
        animation_described, self.animation_described = self.animation_described, None
        animation_concise, self.animation_concise = self.animation_concise, False
        text = self.prompts.reflection_prompt_text(
            self._prompt_modes(),
            concise=self.level_reflection == "concise",
            reset_reason=reset_reason,
            summary=self.last_step_summary,
            level=won.level,
            win_missed=self.win_missed,
            frame_delta=frame_delta,
            animation_text=animation_text,
            animation_described=animation_described,
            animation_concise=animation_concise,
        )
        return self._user_prompt_element(text, _played_frames(won)[-1])

    def _build_audit_prompt_element(self) -> ullm.conv.UserPrompt:
        """The strategy audit's prompt (``strategy_audit``), with the
        level's opening board and the current one where images ride."""
        level_entries = self._current_level_entries()
        level_actions, level_colors = self._untried(level_entries)
        game_actions, game_colors = self._untried(self.history_entries)
        text = self.prompts.audit_prompt_text(
            self._prompt_modes(),
            concise=self.strategy_audit == "concise",
            level=self.history_entries[-1].level,
            untried_actions=level_actions,
            untried_actions_game=game_actions,
            unclicked_colors=level_colors,
            unclicked_colors_game=game_colors,
        )
        if not self._multimodal_enabled():
            return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))

        def image(frame: FramePayload) -> ullm.conv.ImageContent:
            grid = np.frombuffer(base64.b64decode(frame.grid_b64), dtype=np.uint8).reshape(tuple(frame.shape))
            return board_image(grid.astype(np.int8), scale=self.multimodal_upscale, axis_labels=self.axis_labels)

        return ullm.conv.UserPrompt(
            content=ullm.conv.ListContent(
                elements=[
                    ullm.conv.StringContent(text=f"{text}\n\nLevel start image:"),
                    image(level_entries[0].frame),
                    ullm.conv.StringContent(text="Current grid image:"),
                    image(self.history_entries[-1].frame),
                ]
            )
        )

    def _untried(self, entries: list[TimelineEntry]) -> tuple[list[str], list[int]]:
        """The game's actions no move among ``entries`` played, RESET and UNDO
        apart (UNDO is in the action set under ``expose_undo``), and in a game
        with MOUSE the colours on the current board that no click among them
        hit, read off the board each click was made on."""
        names = [name for name in self.game_action_names if name not in ("RESET", "UNDO")]
        played = {to_model_action(entry.action.name) for entry in entries if entry.action is not None}
        actions = [name for name in names if name not in played]
        if "MOUSE" not in names:
            return actions, []
        hit: set[int] = set()
        for before, entry in itertools.pairwise(entries):
            action = entry.action
            if action is None or action.row is None or action.col is None:
                continue
            rows, cols = before.frame.shape
            if 0 <= action.row < rows and 0 <= action.col < cols:
                hit.add(base64.b64decode(before.frame.grid_b64)[action.row * cols + action.col])
        board = set(base64.b64decode(self.history_entries[-1].frame.grid_b64))
        return actions, sorted(board - hit)

    def _user_prompt_element(self, text: str, entry: FramePayload) -> ullm.conv.UserPrompt:
        """``text`` as a UserPrompt, with ``entry`` as its image when one rides
        this prompt, and the level's legend ahead of it once a level."""
        if not self._multimodal_enabled():
            return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))
        if not self._image_rides_this_turn(entry.level):
            return ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=text))
        self.image_level_sent = entry.level
        grid = np.frombuffer(base64.b64decode(entry.grid_b64), dtype=np.uint8).reshape(tuple(entry.shape))
        # The shared frame→pixels mapping (pixel-identical to the original
        # duck's ARC_COLOR_MAP — verified numerically at adoption).
        image = board_image(grid.astype(np.int8), scale=self.multimodal_upscale, axis_labels=self.axis_labels)
        if self._multimodal_mode() == "level_start":
            self._level_start_image = image
        elements: list[ullm.conv.Content] = [
            # Ahead of the turn's own text and board: the legend is the frame
            # of reference for everything after it, and at the front it sits in
            # the cached prefix for the rest of the level rather than shifting
            # the turn text on the one prompt that carries it.
            *self._palette_elements(entry.level),
            ullm.conv.StringContent(text=f"{text}\n\nCurrent grid image:"),
            image,
            *self._diff_image_elements(),
        ]
        return ullm.conv.UserPrompt(content=ullm.conv.ListContent(elements=elements))

    def _diff_image_elements(self) -> list[ullm.conv.Content]:
        """The last move's diff image and its caption, once, or nothing
        (``diff_image``)."""
        pending, self.diff_image_pending = self.diff_image_pending, None
        if pending is None or not self.diff_image:
            return []
        from seal.diff_image import render_diff_image
        from seal.frame_delta import PHRASES

        shape = tuple(pending["shape"])
        frame = np.frombuffer(base64.b64decode(pending["frame_b64"]), dtype=np.int8).reshape(shape)
        changed = np.frombuffer(base64.b64decode(pending["changed_b64"]), dtype=np.uint8).reshape(shape).astype(bool)
        template = self.prompts.diff_image_caption_won if pending["won"] else self.prompts.diff_image_caption
        caption = template.format(move=pending["move"], step=pending["step"], baseline=PHRASES[pending["kind"]])
        return [
            ullm.conv.StringContent(text=caption),
            render_diff_image(frame, changed, scale=self.multimodal_upscale, axis_labels=self.axis_labels),
        ]

    def _palette_elements(self, level: int) -> list[ullm.conv.Content]:
        """The palette legend, once per level, or nothing.

        Which legend depends on `color_name_legend`. On, the colour-NAME
        legend, because a name ties a swatch to wording the prompt already
        uses. Off, the index legend: the seal's boards are indices, which is
        what that one explains.
        Only where an image already travels: an agent with images off has
        chosen not to spend tokens on pixels. And once per level rather than
        per turn, because it is reference material, not state -- the level is
        the boundary the model re-grounds at anyway.

        Summary compaction restores the image on the retained status prompt.
        Under pruning, it does not survive eviction: block eviction
        drops whole user-led blocks oldest-first, so the block carrying the
        legend is the first of its level to go, and this returns nothing after
        that. The system prompt's own legend states index, glyph and name and
        is never evicted, so what is lost is the picture rather than the fact.
        """
        if self.palette_level_sent == level:
            return []
        if not self._palette_enabled():
            return []
        self.palette_level_sent = level
        return self._palette_image_elements()

    def _palette_enabled(self) -> bool:
        """Whether this configuration sends a visual palette legend: wherever
        an image travels, since either legend explains the seal's boards."""
        return self._multimodal_enabled()

    def _palette_image_elements(self) -> list[ullm.conv.Content]:
        """The configured legend and caption, without changing per-level state."""
        # Imported only once a legend is owed: the renderers pull in PIL and
        # the bundled font, and an arm with images off never draws one.
        from seal.palette_legend import render_color_name_legend, render_palette_legend

        if self.color_name_legend:
            return [
                ullm.conv.StringContent(text=self.prompts.color_name_legend_caption),
                ullm.conv.ImageContent(data=np.asarray(render_color_name_legend()), alt_text=""),
            ]
        return [
            ullm.conv.StringContent(text=self.prompts.palette_legend_caption),
            ullm.conv.ImageContent(
                data=np.asarray(render_palette_legend(ASCII_SYMBOLS[self.ascii_symbols])), alt_text=""
            ),
        ]

    def _summary_palette_content(self, content: ullm.conv.Content) -> ullm.conv.Content:
        """Restore the visual legend once in a summary's retained status prompt."""
        if not self._palette_enabled():
            return content
        elements = content.elements if isinstance(content, ullm.conv.ListContent) else [content]
        caption = (
            self.prompts.color_name_legend_caption if self.color_name_legend else self.prompts.palette_legend_caption
        )
        if any(
            isinstance(text, ullm.conv.StringContent)
            and text.text == caption
            and isinstance(image, ullm.conv.ImageContent)
            for text, image in itertools.pairwise(elements)
        ):
            return content
        return ullm.conv.ListContent(elements=[*self._palette_image_elements(), *elements])

    def _summary_reference_content(self, content: ullm.conv.Content) -> ullm.conv.Content:
        """Restore once-per-level images on the retained status, without duplicates."""
        content = self._summary_palette_content(content)
        image = self._level_start_image
        if self._multimodal_mode() != "level_start" or image is None:
            return content
        elements = content.elements if isinstance(content, ullm.conv.ListContent) else [content]
        if any(element is image for element in elements):
            return content
        return ullm.conv.ListContent(
            elements=[*elements, ullm.conv.StringContent(text="Level-start grid image:"), image]
        )

    def _build_user_prompt_text(self) -> str:
        """Compute the per-turn facts — where the game stands, what the last
        batch did, whether a kernel reset is owed — and hand them to the
        prompt surface, which owns the wording."""
        summary = self.last_step_summary
        latest_frame = self.history_entries[-1].frame if self.history_entries else None
        current_step = max(latest_frame.step if latest_frame else 0, max(0, self.action_count)) + 1
        current_level = latest_frame.level if latest_frame else 1
        if summary is not None and summary.level is not None:
            current_level = max(current_level, summary.level)
        observed_max_level = max(
            [current_level, *[entry.frame.level for entry in self.history_entries]],
            default=current_level,
        )
        reset_reason = self.repl_reset_pending or None
        self.repl_reset_pending = ""
        frame_delta, self.frame_delta_text = self.frame_delta_text, None
        animation_text, self.animation_text = self.animation_text, None
        animation_described, self.animation_described = self.animation_described, None
        animation_concise, self.animation_concise = self.animation_concise, False
        return self.prompts.user_prompt_text(
            self._prompt_modes(),
            reset_reason=reset_reason,
            summary=summary,
            any_actions_executed=self.action_count > 0,
            current_step=current_step,
            current_level=current_level,
            observed_max_level=observed_max_level,
            valid_actions=self.current_valid_actions,
            valid_actions_observed=self.valid_actions_observed,
            game_over_active=self.game_over_active,
            exploration_moves_left=self.exploration_moves_left,
            exploration_over=self.exploration_over,
            # The configured allowance, not the one binding this move: a
            # queue-drain lift can only widen it, and the prompt is built
            # without the context that knows whether the queue has drained.
            level_tokens_used=self.level_tokens_used,
            level_token_allowance=self._allowance_for(self.budget_level),
            frame_delta=frame_delta,
            animation_text=animation_text,
            animation_described=animation_described,
            animation_concise=animation_concise,
        )

    # ------------------------------------------------------------------
    # 7. context budget
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
    # text ruler can see engine-side costs at all. ``usage.prompt_tokens``
    # / ``usage.completion_tokens`` ARE exact — the engine's own account
    # of the request, on every OpenAI-compatible response (non-streaming
    # always; streaming via stream_options.include_usage, which ullm's
    # client requests). Everything not yet billed carries a PROVABLE
    # bound: text is charged its raw character count (a text token
    # always spans at least one character, so a string's token count
    # never exceeds its length), and what no text carries —
    # per-message template framing — rides a derived constant. So the trigger can only fire early, the safe
    # direction: an early trim costs a deeper sawtooth, a late one hands
    # an over-window history downstream.
    # ------------------------------------------------------------------

    async def _prepare_summarized_request(
        self,
        ctx: LLMMoveContext,
        ordinary_max_tokens: int | None,
        live_cap: Callable[[], int | None],
        *,
        summary_call_limit: int | None,
        record_pruned_round: Callable[[list[ullm.conv.ConversationElement]], None],
    ) -> tuple[int, int]:
        """Observe the complete summary compaction decision and its calls."""
        self._context_compacting = True
        try:
            return await self._prepare_summarized_context(
                ctx,
                ordinary_max_tokens,
                live_cap,
                summary_call_limit=summary_call_limit,
                record_pruned_round=record_pruned_round,
            )
        finally:
            self._context_compacting = False

    async def _prepare_summarized_context(
        self,
        ctx: LLMMoveContext,
        ordinary_max_tokens: int | None,
        live_cap: Callable[[], int | None],
        *,
        summary_call_limit: int | None,
        record_pruned_round: Callable[[list[ullm.conv.ConversationElement]], None],
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
        game_cap = live_cap()
        effective_reply_cap = ordinary_max_tokens if game_cap is None else min(ordinary_max_tokens, game_cap)
        params.max_tokens = effective_reply_cap
        request_tokens = await self._request_prompt_tokens(params, self._live_conversation)
        trigger = request_tokens + effective_reply_cap + self.safety_margin_tokens > self._compaction_window()
        if trigger and self._side_turn is not None:
            # No summary inside a side turn: it ends, and the summary waits for
            # the call after it.
            self._last_trim_fired = False
            self._side_turn_cut = True
            raise _SideTurnCut
        self._last_trim_fired = trigger
        if not trigger:
            return 0, request_tokens
        with self._compacting_scope():
            summary_calls = await self._summarize_live_history(
                ctx,
                ordinary_max_tokens,
                live_cap,
                call_limit=summary_call_limit,
                record_pruned_round=record_pruned_round,
            )
        return summary_calls, await self._count_summarized_ordinary_request(ordinary_max_tokens, live_cap)

    async def _count_summarized_ordinary_request(
        self,
        ordinary_max_tokens: int,
        live_cap: Callable[[], int | None],
    ) -> int:
        """Cap, validate, and measure the next ordinary summarize-mode request."""
        if not isinstance(self._live_conversation, ullm.conv.Conversation):
            raise RuntimeError("summarize compaction requires a live conversation")
        params = cast(ullm.backend.GenericSamplingParameters, self._live_params)
        game_cap = live_cap()
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

    async def _close_ended_thread(self, backend: ullm.backend.Backend, epoch: int) -> None:
        """Before each request, ahead of the slot offer: if history was
        rewritten since the last request went out (a compaction, a retention
        trim, a discarded round), this conversation's thread in the engine is
        over, and the coming request, a new epoch, would end it anyway. Say so
        now (``Backend.close_conversation``), so its cached prefix leaves the
        host tier now rather than when that request comes, which a game
        giving up its slot here can make wait tens of minutes (Kaggle D: 28
        minutes at the median). ``epoch`` is the coming request's. A failed
        close is only printed: it is a cache hint."""
        state = vars(self)
        sent = state.get("_thread_epoch")
        state["_thread_epoch"] = epoch
        conversation_id = state.get("_thread_conversation_id")
        if sent is None or sent == epoch or conversation_id is None:
            return
        try:
            await backend.close_conversation(conversation_id)
        except Exception as e:
            print(f"[seal] close_conversation({conversation_id}) failed: {e!r}", flush=True)

    async def _summarize_live_history(
        self,
        ctx: LLMMoveContext,
        ordinary_max_tokens: int,
        live_cap: Callable[[], int | None],
        *,
        call_limit: int | None,
        record_pruned_round: Callable[[list[ullm.conv.ConversationElement]], None],
    ) -> int:
        """Replace old history with one summary and return generated attempts.

        The request is the live one extended by the instruction, so it is not a
        rewrite: it continues the served ids, which keeps its prefill cached.
        The rewrite is the rebuilt history, declared once the summary exists."""
        target = self._summary_target_tokens()
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
            summary_cap = 6 * target
        else:
            base_params.response_injection = _SUMMARY_THINK_CLOSE
            summary_cap = 2 * target
        # The summary request is not blanked: the blank is a register for
        # play, and a summary is not a move.
        for key in ("zero_recurrent_state", "zero_recurrent_at"):
            base_params.extra_params.pop(key, None)
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
                **cast(dict[Any, Any], existing_bias),
                **{token_id: -100.0 for token_id in self.compaction_forbidden_token_ids},
            }
        base_seed = base_params.seed if base_params.seed is not None else 0
        # Read once, before the request says whether it will be shown: no tool
        # runs between the request and the install, so it is the same file.
        world_model_source = self._summary_world_model_source()
        instruction = ullm.conv.UserPrompt(
            content=ullm.conv.StringContent(
                text=self.prompts.compaction_prompt(
                    self._prompt_modes(),
                    target,
                    current_level=self._summary_level(),
                    # The previous summary sits in the rebuilt history until the
                    # next one replaces it: a level change does not rebuild it.
                    previous_summary=self.context_summaries > 0,
                    world_model_shown=world_model_source is not None,
                )
            )
        )
        # The whole history, never reduced to fit: the engine window is sized
        # to hold the request instead (``prepare_for_backend``).
        request_elements = [*self._detached_history(self.messages), instruction]
        if min(summary_cap, live_cap() or summary_cap) <= 0:
            raise GiveUpError("token budget exhausted before context compaction")

        invalid_tries = 0
        attempts = 0
        rerendered = False
        # Whether this attempt reasons first although ``summary_thinking`` is
        # off: the retry of a summary that came back entirely empty, with
        # nothing in the reasoning either. A model can end its reply on the
        # first token after the injected end of thinking, and does so again on
        # a reseeded retry; reasoning first, it writes one. The engine window
        # needs no room for it: each request's cap is clamped to what the
        # window has left (``_summary_completion_cap``).
        reasoning_retry = False
        while True:
            if call_limit is not None and attempts >= call_limit:
                raise _CallAllowanceExhausted(attempts)
            params = copy.deepcopy(base_params)
            if reasoning_retry:
                params.response_injection = None
                summary_cap = 6 * target
            remaining = live_cap()
            completion_cap = summary_cap if remaining is None else min(summary_cap, remaining)
            if completion_cap <= 0:
                raise GiveUpError("token budget exhausted during context compaction")
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
                params.max_tokens = self._summary_completion_cap(completion_cap, measured)
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
                    label="compaction",
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
                    raise RuntimeError("compaction failed: model returned no usable summary twice") from exc
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
                    raise RuntimeError(
                        "compaction failed: model returned an empty or tool-call summary twice "
                        + ("(retried with reasoning)" if reasoning_retry else "(reseeded retry)")
                    )
                # A tool call, or its markup in the text, is only reseeded.
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

        # Gathered before the rewrite, while the kernel is idle between rounds.
        listing = await self._files_and_repl_listing() if self.add_files_and_repl_to_summary else ""
        if world_model_source is not None:
            shown = self.prompts.compaction_world_model(world_model_source)
            listing = f"{listing}\n\n{shown}" if listing else shown
        # A summary is two declared re-renders, both after its request. The
        # first: from here the history is being rewritten, and the install
        # counts the rebuilt candidates -- summary, status, tail -- which exact
        # history refuses to count on the epoch the summary request was served
        # on.
        self._mark_history_rewritten()
        self._prepare_context_observer()
        game_cap = live_cap()
        effective_reply_cap = ordinary_max_tokens if game_cap is None else min(ordinary_max_tokens, game_cap)
        cast(ullm.backend.GenericSamplingParameters, self._live_params).max_tokens = effective_reply_cap
        tail_dropped, post_count = await self._install_summary(
            summary, listing, effective_reply_cap, record_pruned_round=record_pruned_round
        )
        # The second: the next ordinary call renders the rebuilt history
        # afresh, and the last thing served was the summary request. Under one
        # epoch a trace reads that as a prompt that fails to extend the
        # previous step's ids with no rewrite declared.
        self._mark_history_rewritten()
        self.context_summaries += 1
        self._mark_moveless(ctx)
        usage = response.token_usage
        platypus.diagnostics.current_span().kv(
            compacted=True,
            compaction_strategy="summarize",
            compaction_number=self.context_summaries,
            compaction_target=target,
            compaction_max_tokens=summary_cap,
            # What the last request was sent with: the cap, or less where the
            # engine window had less left over its prompt.
            compaction_sent_max_tokens=params.max_tokens,
            summary_tokens=usage.completion_tokens if usage is not None else None,
            # The summary text installed, in characters: `summary_tokens` counts
            # the request's whole generation, reasoning included under
            # ``summary_thinking``, so the text's own length is read here.
            summary_text_chars=len(summary),
            summary_thinking=self.summary_thinking or reasoning_retry,
            summary_reasoning_retry=reasoning_retry,
            summary_tail=self.summary_tail,
            chronological_restart=self.chronological_restart,
            summary_request_rerendered=rerendered,
            recency_tail_dropped=tail_dropped,
            context_after=post_count,
            compaction_system_prompt=self._system_prompt_text(),
        )
        return attempts

    def _summary_level(self) -> int:
        """The level the game stands on, as the status prompt reports it."""
        return self.history_entries[-1].frame.level if self.history_entries else 1

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

    def _summary_world_model_source(self) -> str | None:
        """``world_model.py`` as the closing message of a rebuilt conversation
        shows it, or None where none is shown: without a world model, with
        ``add_world_model_to_summary`` off, before the file exists, and while
        it is still the seeded example. Read host-side: the directory is the
        tool's."""
        if self.use_world_model == "none" or not self.add_world_model_to_summary:
            return None
        directory = self._ensure_repl()._tmpdir  # pyright: ignore[reportPrivateUsage] — the tool owns it, this only reads one file
        path = None if directory is None else os.path.join(directory, "world_model.py")
        if path is None or not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        if source == seeds.seed_source(mouse="MOUSE" in self.game_action_names):
            return None
        return source

    async def _files_and_repl_listing(self) -> str:
        """The files in the kernel's working directory and the objects the
        model defined in its ``python`` session, for the message that closes
        a rebuilt conversation. A listing that fails is said to be
        unavailable, with whether the kernel went with it, rather than ending
        the game: the inventory is optional."""
        repl = self._ensure_repl()
        try:
            objects = await repl.session_objects(timeout=_HARNESS_CELL_TIMEOUT_S)
        except SessionInspectionError as exc:
            platypus.diagnostics.current_span().event("repl/session_listing_failed", error=str(exc), output=exc.output)
            unavailable = self.prompts.compaction_objects_unavailable(kernel_lost=repl.is_fresh, error=str(exc))
            return self.prompts.compaction_listing(
                self._prompt_modes(), repl.working_files(), [], objects_unavailable=unavailable
            )
        return self.prompts.compaction_listing(self._prompt_modes(), repl.working_files(), objects)

    async def _install_summary(
        self,
        summary: str,
        listing: str,
        effective_reply_cap: int,
        *,
        record_pruned_round: Callable[[list[ullm.conv.ConversationElement]], None],
    ) -> tuple[bool, int]:
        """Atomically rebuild durable history around one accepted summary.
        Returns whether the kept tool call had to go to fit (``summary_tail``
        says what it keeps when it fits), and the rebuilt request's decision
        reading.

        The rebuilt history, in order: the summary; the user prompt that
        started the current turn, which is the current status (a turn ends as
        soon as the model acts; a nudge within it starts no turn), noting how
        many of the turn's tool calls the summary replaced; the most recent
        tool call anywhere in the history with its results, in its proper
        place -- after that prompt when this turn made it, before it when an
        earlier turn did; the closing message, over the listing when there is
        one; and the live turn, emptied, which the next request continues.
        Under ``chronological_restart`` the summary moves after the prompt
        and the kept call, beside the closing message, and a preamble opens
        the history, saying the game is under way and where the summary is.
        Everything else goes."""
        if not self.messages or not isinstance(self.messages[-1], ullm.conv.ResponseTurn):
            raise RuntimeError("context compaction requires the live ResponseTurn")
        live_turn = self.messages[-1]
        live_index = len(self.messages) - 1
        status_index = next(
            (index for index, element in enumerate(self.messages) if element is self._turn_prompt), None
        )
        if status_index is None:
            raise RuntimeError("context compaction requires the user prompt that started the current turn")
        status = cast(ullm.conv.UserPrompt, self.messages[status_index])
        # Include reference images in both candidate token counts and the installed
        # prompt. Keep the live status untouched until the rebuilt request fits.
        status_content = self._summary_reference_content(status.content)
        wrapper = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=self.prompts.compaction_wrapper(summary)))
        closing = ullm.conv.UserPrompt(
            content=ullm.conv.StringContent(
                text=self.prompts.compaction_closing(
                    self._prompt_modes(), listing, session_intact=self.repl is None or not self.repl.is_fresh
                )
            )
        )
        preamble = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=self.prompts.compaction_restart_preamble))

        # The most recent response that made a tool call, and where its round
        # ends: at the next response, or the end of its turn.
        latest: tuple[int, int, int] | None = None
        for top_index in range(live_index, -1, -1):
            element = self.messages[top_index]
            if not isinstance(element, ullm.conv.ResponseTurn):
                continue
            for sub_index in range(len(element.elements) - 1, -1, -1):
                sub = element.elements[sub_index]
                if (
                    isinstance(sub, ullm.conv.LLMResponse)
                    and not sub.do_not_send_to_agent
                    and sub.final_content.tool_calls()
                ):
                    end = sub_index + 1
                    while end < len(element.elements) and not isinstance(element.elements[end], ullm.conv.LLMResponse):
                        end += 1
                    latest = (top_index, sub_index, end)
                    break
            if latest is not None:
                break
        if self.summary_tail == "none":
            latest = None

        # The call as ``summary_tail`` keeps it. ``without_reasoning`` keeps a
        # COPY of the response without its reasoning, not an edit of the
        # served object: the copy carries no served witness and renders afresh,
        # which the rewrite declared around the install allows. It does carry
        # the engine output (``_model_contents``, an init=False field
        # ``replace`` leaves behind): a backend renders a past response from it
        # -- the content through the reasoning parser, the reasoning from
        # ``reasoning_content``, which the copy has empty.
        stripped: ullm.conv.LLMResponse | None = None
        kept: list[ullm.conv.ConversationElement] = []
        if latest is not None:
            source = cast(ullm.conv.ResponseTurn, self.messages[latest[0]])
            kept = list(source.elements[latest[1] : latest[2]])
            if self.summary_tail == "without_reasoning":
                head = cast(ullm.conv.LLMResponse, kept[0])
                stripped = dataclasses.replace(head, reasoning_content=ullm.conv.EmptyContent())
                stripped._model_contents = head._model_contents  # pyright: ignore[reportPrivateUsage]
                kept[0] = stripped
        kept_after_status = latest is not None and latest[0] > status_index
        kept_original = (
            cast(ullm.conv.ResponseTurn, self.messages[latest[0]]).elements[latest[1]] if latest is not None else None
        )

        def omitted_calls(keep: bool) -> int:
            """Tool calls this turn made after its prompt that the rebuilt
            history does not show: every block of the turn, the live one and
            any a nudge opened before it, and a call an earlier summary in
            this turn kept but this one does not."""
            return sum(
                len(sub.final_content.tool_calls())
                for element in self.messages[status_index + 1 :]
                if isinstance(element, ullm.conv.ResponseTurn)
                for sub in element.elements
                if isinstance(sub, ullm.conv.LLMResponse) and not (keep and sub is kept_original)
            )

        def noted_content(keep: bool) -> ullm.conv.Content:
            """The status prompt's content as the rebuilt history carries it:
            with the omitted-calls note, rewritten in place rather than
            stacked when an earlier summary in this turn left one."""
            count = self._omitted_tool_calls + omitted_calls(keep)
            if count == 0:
                return status_content
            note = ullm.conv.StringContent(text=self.prompts.tool_calls_omitted_note(count))
            content = status_content
            elements = content.elements if isinstance(content, ullm.conv.ListContent) else [content]
            kept_content = [
                e
                for e in elements
                if not (isinstance(e, ullm.conv.StringContent) and self.prompts.is_tool_calls_omitted_note(e.text))
            ]
            return ullm.conv.ListContent(elements=[*kept_content, note])

        def rebuilt(
            keep: bool, prompt: ullm.conv.UserPrompt, turn: ullm.conv.ResponseTurn
        ) -> list[ullm.conv.ConversationElement]:
            """The rebuilt history, with or without the kept call."""
            story: list[ullm.conv.ConversationElement] = [prompt]
            if keep:
                call = ullm.conv.ResponseTurn(elements=list(kept), status="processed")
                story = [prompt, call] if kept_after_status else [call, prompt]
            if self.chronological_restart:
                return [preamble, *story, wrapper, closing, turn]
            return [wrapper, *story, closing, turn]

        params = cast(ullm.backend.GenericSamplingParameters, self._live_params)

        async def count(keep: bool) -> int:
            """The rebuilt request's weight, over the snapshot it would carry."""
            candidate = rebuilt(
                keep,
                ullm.conv.UserPrompt(content=noted_content(keep)),
                ullm.conv.ResponseTurn(elements=[], status="intermediate"),
            )
            self._refresh_memory_prompt_snapshot(candidate)
            return await self._request_prompt_tokens(params, self._ordinary_conversation(candidate))

        keep = bool(kept)
        post_count = await count(keep)
        tail_dropped = (
            keep and post_count + effective_reply_cap + self.safety_margin_tokens > self.context_window_tokens
        )
        if tail_dropped:
            keep = False
            stripped = None
            post_count = await count(keep)
        if post_count + effective_reply_cap + self.safety_margin_tokens > self.context_window_tokens:
            raise RuntimeError("context overflow remained after compaction even without the kept tool call")

        # The note goes on the status prompt itself, as eviction's
        # pruned-rounds note does, so a second summary in this turn finds it.
        status.content = noted_content(keep)
        self._omitted_tool_calls += omitted_calls(keep)
        # The summary rewrote every prompt the live turn's bills describe:
        # retire them as anchors, and move the bounds of the rounds that leave
        # the live turn out of the live pairing -- with the responses they were
        # recorded for -- so every billed response still faces the window
        # invariant exactly once.
        self._mark_usage_stale(self.messages)
        if stripped is not None:
            # The copy carries the original's bill, which describes the same
            # rewritten prompt: retired as an anchor like the original.
            assert self._stale_responses is not None
            self._stale_responses.append(stripped)
        left = [e for e in live_turn.elements if isinstance(e, ullm.conv.LLMResponse)]
        bounds = list(self._pending_bounds)
        self._deferred_usage_checks.extend(zip(bounds[: len(left)], left, strict=False))
        self._pending_bounds[:] = bounds[len(left) :]
        # Every round leaves the live turn: those the summary replaced, and the
        # kept one, which moves into the history ahead of the closing message.
        # Each is recorded as eviction records a pruned round, oldest first and
        # while it is still in the turn, so the turn's record holds every
        # response it generated.
        while live_turn.elements:
            cut = 1
            while cut < len(live_turn.elements) and not isinstance(live_turn.elements[cut], ullm.conv.LLMResponse):
                cut += 1
            record_pruned_round(list(live_turn.elements[:cut]))
            del live_turn.elements[:cut]
        self.messages[:] = rebuilt(keep, status, live_turn)
        self._refresh_memory_prompt_snapshot(self.messages)
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
                # keeps its raises -- U5 still forbids silently serving a
                # re-rendered history.
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
        included because the seal re-sends reasoning for all turns, so the
        whole generation returns as input), and everything appended after
        it — tool results, the next user prompt, responses the backend
        returned without usage — is bounded by its character count plus
        the framing constant (``_bound_tokens``).

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
        a history (training above all) is entitled to assume — save in a side
        turn, which is held to the window and its extra tokens
        (``_compaction_window``), since no summary runs inside it. The decision
        path triggers compaction at window − reserve − margin with a ruler
        that can only overcount, so a billed total above the window is an
        accounting failure, and it raises: one game fails loudly at the
        moment of the overrun (RuntimeError, a seal bug by contract) rather
        than an oversized sequence travelling on.

        Covered pairs come from two queues, each response exactly once: the
        live ``_pending_bounds`` ↔ surviving-responses pairing, and
        ``_deferred_usage_checks`` for responses round-pruning removed from
        the turn mid-block.

        Below the invariant, a billed prompt above its recorded pre-call
        bound means the constants fell short of what no text carries —
        template framing beyond the per-message allowance, a retry's
        appended newlines. The bound is
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
        window = self._compaction_window()
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
        what is actually sent."""
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
        """Start indices of user-led blocks."""
        return [i for i, e in enumerate(elements) if isinstance(e, ullm.conv.UserPrompt)]

    async def _trim_live_messages(
        self,
        *,
        record_pruned_round: Callable[[list[ullm.conv.ConversationElement]], None] | None = None,
    ) -> None:
        """TurnManager before_round hook: evict against the context budget
        just before each LLM call (in place — the live element list), then
        record the surviving prompt's predicted upper bound so the billed
        usage can be held against it post-turn (``_check_engine_usage``).
        Removed current-turn rounds go through the active manager's recorder
        when supplied."""
        self._prepare_context_observer()
        self.response_step += 1
        self._rounds_since_compaction += 1
        self._arm_prompt_injection()
        await self._trim_messages_for_context(
            self.messages,
            enforce=True,
            record_pruned_round=record_pruned_round,
        )
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
        written back; that crossing is not guaranteed -- measured at a game
        start the margin is a median of 140 tokens, and 2 of 35 openings
        missed it. A missed write-back leaves the next call resuming at or
        before the position, where asking again re-blanks it. A schedule
        that asked only at a game start and after history rewrites would
        leave those games with a single perturbed token and no way to know.

        The position is an absolute prompt position, and that is the only
        spelling the engine has: a lag counted back from the end was
        removed, because it resolves to a different absolute position on
        every call as the prompt grows while the cached state records where
        the blank actually landed, so no caller could keep the two
        consistent. The engine refuses a position the prompt does not reach
        rather than blanking off-position, which is why this method only
        asks once the measured prompt is past it: a request the engine
        would reject is not sent.
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
        record_pruned_round: Callable[[list[ullm.conv.ConversationElement]], None] | None = None,
    ) -> list[ullm.conv.ConversationElement]:
        """Apply the configured compaction schedule around the eviction algorithm."""
        if self.compact_before_request_only and not enforce:
            return elements
        self._context_compacting = True
        try:
            return await self._compact_messages_for_context(
                elements, enforce=enforce, record_pruned_round=record_pruned_round
            )
        finally:
            self._context_compacting = False

    async def _compact_messages_for_context(
        self,
        elements: list[ullm.conv.ConversationElement],
        *,
        enforce: bool = False,
        record_pruned_round: Callable[[list[ullm.conv.ConversationElement]], None] | None = None,
    ) -> list[ullm.conv.ConversationElement]:
        """While over budget, drop whole user-led blocks
        from the oldest end (chunked by evict_turns, never the last block).
        Mutates ``elements`` IN PLACE — during a TurnManager run it is the
        live conversation element list — and returns it.

        The trigger is the BUDGET; how far the trim goes is the compaction
        TARGET (``compact_to_tokens``). Separating the two is what makes the
        context a sawtooth rather than a prompt pinned to the budget edge.
        ``record_pruned_round`` belongs only to the active pre-call trim;
        retention and snapshot trims leave it unset."""
        budget = self._context_budget_tokens()
        target = self._compact_target_tokens()
        blocks_evicted = 0
        pruned_before = self._pruned_rounds
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
            # This is the only normal mutation of the system prefix. Memory
            # written since the previous compaction is already present in
            # immutable tool-result history; refreshing it here carries the
            # retained journal across the blocks about to be evicted.
            self._refresh_memory_prompt_snapshot(elements)
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
                    del elements[starts[0] : starts[1]]
                    self._refresh_memory_prompt_snapshot(elements)
                    turns_evicted += 1
                    blocks_evicted += 1
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
            with self._compacting_scope():
                while True:
                    pruned = await self._prune_current_turn(
                        elements,
                        target,
                        target=target,
                        record_pruned_round=record_pruned_round,
                    )
                    if not pruned:
                        break
                    self._note_pruned_rounds(elements, pruned)
                    # Removed memory results move journal text into the
                    # system prefix; count that text before pruning ends.
                    self._refresh_memory_prompt_snapshot(elements)
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
            # The snapshot refresh comes first so context_after weighs the
            # prompt the next request actually starts from, and the spacing
            # pair makes the sawtooth's period visible per trim: how many
            # decision-ruler tokens grew, over how many LLM calls, since the
            # previous fired trim left the prompt at its post-trim size.
            self._refresh_memory_prompt_snapshot(elements)
            after = await self._measured_prompt_tokens(elements)
            since = self._tokens_at_last_compaction
            platypus.diagnostics.current_span().kv(
                compacted=True,
                context_before=measured,
                context_after=after,
                compaction_before_response=self.response_step + (0 if enforce else 1),
                compaction_system_prompt=self._system_prompt_text(),
                context_budget=budget,
                context_target=target,
                blocks_evicted=blocks_evicted,
                history_evicted=blocks_evicted > 0 or self._pruned_rounds > pruned_before,
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
        record_pruned_round: Callable[[list[ullm.conv.ConversationElement]], None] | None = None,
    ) -> int:
        """Drop the current turn's oldest rounds until it fits, keeping the
        user prompt that opened it. Returns how many rounds went.

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
        rather than every round (block eviction chunks by the same knob).
        Each removed round is handed to ``record_pruned_round`` before its
        elements leave the live list."""
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
            if record_pruned_round is not None:
                record_pruned_round(list(turn.elements[:cut]))
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
        prompt = elements[starts[-1]]
        assert isinstance(prompt, ullm.conv.UserPrompt)
        self._pruned_rounds += pruned
        note = ullm.conv.StringContent(text=self.prompts.rounds_pruned_note(self._pruned_rounds))
        content = prompt.content
        if isinstance(content, ullm.conv.ListContent):
            kept = [
                e
                for e in content.elements
                if not (isinstance(e, ullm.conv.StringContent) and self.prompts.is_rounds_pruned_note(e.text))
            ]
            prompt.content = ullm.conv.ListContent(elements=[*kept, note])
        else:
            prompt.content = ullm.conv.ListContent(elements=[content, note])

    async def _persist_history(self) -> None:
        """Post-turn retention: keep the trimmed conversation.

        Measures with the one decision ruler there is. This trim runs
        after every turn, so if it measured differently it would be the
        binding constraint and the decision path would never actually
        govern the conversation. A summarizing agent has nothing to do here:
        its summary fires at the next request that would overflow."""
        if not self._uses_context_summary():
            await self._trim_messages_for_context(self.messages)

    # ------------------------------------------------------------------
    # 8. resources
    #
    # The kernel is the only live resource the seal owns, and how long it
    # is allowed to live is a behavioral knob (persistence) with a
    # contract consequence (which checkpoint cadences it can support).
    # ------------------------------------------------------------------

    def supported_checkpointing(self) -> frozenset[str]:
        """A4: a kernel must not outlive a checkpoint boundary, so how long
        the seal keeps kernel state decides where it can be snapshotted.

        Resolved here rather than declared on the class because it follows
        ``persistence``, which is configuration."""
        if self.persistence in ("round", "move"):
            return frozenset({"move", "level", "off"})  # kernel dead at every move boundary
        if self.persistence == "level":
            return frozenset({"level", "off"})  # survives moves, not levels
        return frozenset({"off"})  # "always": nothing may snapshot it

    def _ensure_repl(self, ctx: LLMMoveContext | None = None) -> GameREPL:
        """The lazily-built GameREPL (runtime surface preloaded,
        per-call timeout); reused across moves. How often its kernel is reset follows
        ``persistence`` — per snippet only in "round" mode.

        With a ``ctx``, the tool is (re-)wired to report queueing for a pool
        slot and running a snippet into the run's status line. Re-wired on
        every use rather than at construction because the hook is
        process-local: a deepcopied or unpickled agent arrives with none."""
        if self.repl is None:
            self.repl = GameREPL(
                timeout=float(self._python_timeout_s()),
                tool_output_tokens=self.tool_output_tokens,
                reset_before_each_call=(self.persistence == "round"),
                use_world_model=self.use_world_model,
                settled_delta=self.settled_delta,
                animation_description=self.animation_description,
                animation_max_chars=self.animation_max_chars,
                animation_cut_chars=self.animation_cut_chars,
                decomposer=self.decomposer,
                max_bits_understood=self.max_bits_understood,
                crash_on_framediff_timeout=self.crash_on_framediff_timeout,
                segmentation=self.segmentation,
                confine_filesystem=self.confine_filesystem,
            )
        self.repl.reset_before_each_call = self.persistence == "round"
        # Before configure_ascii, which is what rebuilds the bootstrap text
        # these fields are baked into.
        self.repl.use_world_model = self.use_world_model
        self.repl.settled_delta = self.settled_delta
        self.repl.diff_image = self.diff_image
        self.repl.animation_description = self.animation_description
        self.repl.animation_max_chars = self.animation_max_chars
        self.repl.animation_cut_chars = self.animation_cut_chars
        self.repl.decomposer = self.decomposer
        self.repl.max_bits_understood = self.max_bits_understood
        self.repl.crash_on_framediff_timeout = self.crash_on_framediff_timeout
        self.repl.segmentation = self.segmentation
        self.repl.confine_filesystem = self.confine_filesystem
        self.repl.plan_argument = self.plan_argument
        self.repl.pixel_tolerance_offered = self.expose_pixel_tolerance
        self.repl.configure_ascii(
            row_numbers=self.ascii_row_numbers, column_group=self.ascii_column_group, symbols=self.ascii_symbols
        )
        modes = self._prompt_modes()
        self.repl.description = self.prompts.python_tool_description(modes)
        self.repl.parameters["properties"]["code"]["description"] = self.prompts.python_tool_code_param_description(
            modes
        )
        # Rebuilt whole on every call, so the order is this config's even on
        # a repl a previous config shaped.
        properties = {name: spec for name, spec in self.repl.parameters["properties"].items() if name != "plan"}
        if self.plan_argument != "none":
            plan = {"type": "string", "description": self.prompts.python_tool_plan_param_description(modes)}
            properties = {"plan": plan, **properties} if self.plan_argument == "first" else {**properties, "plan": plan}
        self.repl.parameters["properties"] = properties
        if ctx is not None:
            self.repl.status_hook = ctx.status
        return self.repl

    def _apply_pending_level_reset(self) -> None:
        """Pay the kernel reset a level win owes (``persistence="level"``):
        the new level is a new problem, and carrying the old one's scratch
        state into it is likelier to mislead than to help.

        Owed when the win is folded, paid where the kernel is idle: at the
        end of the move that won (so a level checkpoint between moves finds
        no kernel), and at the top of a do_move for a win delivered there.
        In live mode the win arrives mid-snippet through ``ctx.execute``,
        with the kernel blocked on the host call that delivered it, so there
        is nothing resettable at that moment."""
        if self.persistence == "level" and self.pending_level_reset:
            self.pending_level_reset = False
            self._reset_repl("a new level started")

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

    def _reset_repl(self, reason: str) -> None:
        """Drop kernel state and note it, so the next user prompt can tell the
        model its scratch work is gone (silent loss would have it referencing
        names that no longer exist).

        Nothing is announced before the first snippet has run: there is no
        scratch work to have lost, and opening the game with a notice about
        vanished definitions describes a session that never existed.

        What dies is kernel STATE — definitions, globals, imported modules.
        The working directory's contents are never cleared, at any reset
        cadence: a file the model wrote is its work, not scratch state, and
        a reset that deleted it would throw away the only durable thing the
        model has besides the journal."""
        # Called only between kernel operations (a move's end, a do_move's
        # top), so there is nothing in flight to guard against.
        had_session = self.repl is not None and self.repl.session_used
        if self.repl is not None:
            self.repl.reset()
        if had_session and self.persistence in ("move", "level"):
            self.repl_reset_pending = reason

    async def _on_finish(  # pyright: ignore[reportIncompatibleMethodOverride] — same covariant ctx narrowing as LLMAgent._do_move
        self, ctx: LLMMoveContext, result: GameResult
    ) -> None:
        """Resource cleanup: final kernel/tmpdir teardown (no LLM calls)."""
        if self.repl is not None:
            await self.repl.clear_state()
            self.repl = None


# ----------------------------------------------------------------------
# pure helpers
#
# No agent state, no I/O — the renderings and one heuristic the methods
# above lean on. They live at the end so the class reads first.
# ----------------------------------------------------------------------

# What re-sending the anchor response's own generation costs beyond its
# billed ``completion_tokens``: the generation's sampled ids (reasoning,
# text, tool-call JSON, end markers — all inside completion_tokens) come
# back as input nearly verbatim under the seal's send-back-all policy, and
# the chat template adds only its assistant framing around them plus small
# re-encoding wobble in the tool-call JSON. A flat, deliberately generous
# cover for those few tokens.
_ANCHOR_RERENDER_OVERHEAD_TOKENS = 32


#: ARC-AGI-3 frames are at most this many cells a side.
FRAME_MAX_SIDE = 64
# Qwen VL image processors round each side of an image up to a multiple of
# their patch size times merge size (32 on the checkpoints served here)
# before comparing the pixel count against their ceiling, so a ceiling taken
# from the raw render would shrink the very image it was sized for. Sides
# are rounded up to this instead: a multiple of every such factor up to it,
# and the margin costs a few hundred tokens of start-up profile, nothing
# the model sees.
PROCESSOR_SIDE_ROUNDING = 64


def _names_memory_error(line: str) -> bool:
    """Whether a traceback's last line names a ``MemoryError``, judged by the
    class it names rather than its spelling: a library's own subclass counts
    -- numpy's ``_ArrayMemoryError``, raised on an allocation the kernel's
    address-space cap refuses -- and a name that only looks like one does not.
    The class is looked up among the modules this process has already loaded:
    nothing a traceback names is imported."""
    name = line.split(":", 1)[0].strip()
    module_name, _, attribute = name.rpartition(".")
    module = sys.modules.get(module_name) if module_name else builtins
    cls = getattr(module, attribute, None) if module is not None and attribute else None
    return isinstance(cls, type) and issubclass(cls, MemoryError)


def _rounded_pixels(height: int, width: int) -> int:
    """The pixel count a processor compares against its ceiling: each side
    rounded up to :data:`PROCESSOR_SIDE_ROUNDING`."""
    rounding = PROCESSOR_SIDE_ROUNDING
    return -(-height // rounding) * rounding * (-(-width // rounding) * rounding)


def largest_image_pixels(*, upscale: int, axis_labels: bool) -> int:
    """The pixel ceiling an image processor needs so it never shrinks an
    image this agent sends.

    The candidates are every kind of image the seal sends: the largest frame
    at ``upscale``, labelled when ``axis_labels``, the colour-name legend, and the index legend in each
    alphabet a board can print in, whose sizes are fixed by their own
    layouts, so that at small upscales a legend decides. Rendered rather
    than computed from the layout constants, so no number can drift from
    what the agent actually sends.
    """
    # Inside the function: the legends draw with PIL and a bundled font,
    # which a game child has no reason to pay for on import.
    from seal.palette_legend import render_color_name_legend, render_palette_legend

    blank = np.zeros((FRAME_MAX_SIDE, FRAME_MAX_SIDE), dtype=np.int8)
    shapes = [
        np.asarray(board_image(blank, scale=upscale, axis_labels=axis_labels).data).shape,
        np.asarray(render_color_name_legend()).shape,
        *(np.asarray(render_palette_legend(chars)).shape for chars in ASCII_SYMBOLS.values()),
    ]
    return max(_rounded_pixels(shape[0], shape[1]) for shape in shapes)


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


def _level_suffix(entries: list[TimelineEntry]) -> list[TimelineEntry]:
    """The last level's turns in ``entries``, oldest first.

    A suffix rather than a filter: a level number can repeat across a
    RESET, and what the model is owed is the run of turns it has just
    played, not every turn ever played on a level with this number."""
    if not entries:
        return []
    level = entries[-1].level
    index = len(entries)
    while index > 0 and entries[index - 1].level == level:
        index -= 1
    return entries[index:]


def turn_payload(entry: TimelineEntry) -> dict[str, Any]:
    """One turn as the kernel builds ``history`` from it: the move the model
    knows (a click with its coordinates), the outcome, the frames of the
    level it was played on, and the plan of the call that played it."""
    payload: dict[str, Any] = {
        **(model_move(entry.action) if entry.action is not None else {"action": ""}),
        "outcome": entry.outcome,
        "animation": [frame.payload() for frame in _played_frames(entry)],
    }
    if entry.plan is not None:
        payload["plan"] = entry.plan
    return payload


def _played_frames(entry: TimelineEntry) -> list[FramePayload]:
    """A turn's frames on the level it was played on, its settled one last. A
    level's winning move animates on into the next level's opening board; its
    turn ends on the win screen, the last frame of the level it won."""
    frames = entry.animation or [entry.frame]
    if entry.outcome == "won":
        frames = [frame for frame in frames if frame.level == entry.level]
    return frames


def _executed_label(action: Action, before: FramePayload | None) -> str:
    """A move as the executed-actions line names it: the bare label, and for a
    click its coordinates and the colour of the cell it clicked on ``before``,
    the board the click was made on."""
    name = to_model_action(action.name)
    if action.row is None or action.col is None:
        return name
    clicked = ""
    if before is not None:
        rows, cols = before.shape
        if 0 <= action.row < rows and 0 <= action.col < cols:
            clicked = f" on color {base64.b64decode(before.grid_b64)[action.row * cols + action.col]}"
    return f"{name}(row={action.row}, col={action.col}){clicked}"


def _format_action_display(action: Action) -> str:
    """A move as the model reads it in prose — the executed-actions line, a
    journal entry: the kernel's own ``Action`` repr, so text and a printed
    value spell the same move the same way."""
    return repr(harness_game._move(model_move(action)))  # pyright: ignore[reportPrivateUsage]  # the kernel's one spelling of a move


def _cells_differing(first: FramePayload, other: FramePayload) -> int:
    """How many cells ``other``'s board differs from ``first``'s in; every cell
    of the larger where the two differ in size."""
    if tuple(first.shape) != tuple(other.shape):
        return max(int(np.prod(first.shape)), int(np.prod(other.shape)))
    a = np.frombuffer(base64.b64decode(first.grid_b64), dtype=np.uint8)
    b = np.frombuffer(base64.b64decode(other.grid_b64), dtype=np.uint8)
    return int((a != b).sum())


def _move_change(before: FramePayload | None, after: FramePayload) -> MoveChange | None:
    """What a move changed from the board ``before`` it to its settled ``after``:
    the cells that differ, and the colours they held on each side. None
    without a board before, or where the board changed size."""
    if before is None or tuple(before.shape) != tuple(after.shape):
        return None
    old = np.frombuffer(base64.b64decode(before.grid_b64), dtype=np.uint8)
    new = np.frombuffer(base64.b64decode(after.grid_b64), dtype=np.uint8)
    changed = old != new
    return (
        int(changed.sum()),
        tuple(int(color) for color in np.unique(old[changed])),
        tuple(int(color) for color in np.unique(new[changed])),
    )


def _kept_count(before: FramePayload | None, animation: list[FramePayload]) -> int:
    """How many animation frames the kernel's turn keeps from ``animation`` (its frames, the settled one
    last), after the board ``before`` it."""
    grids = [frame.grid_b64 for frame in animation]
    return len(kept_animation(None if before is None else before.grid_b64, grids[:-1], grids[-1], operator.eq))


def _grid_bytes(frame: Any) -> tuple[bytes, tuple[int, int]]:
    """Frame -> (raw uint8 bytes, shape) for the b64 grid payloads."""
    data = np.asarray(frame.data)
    clipped = np.clip(data, 0, 15).astype(np.uint8)
    rows, cols = clipped.shape
    return clipped.tobytes(), (int(rows), int(cols))


def _terminal_action_reason(result: dict[str, Any]) -> str | None:
    """Which terminal event an action result records, or None if it is an
    ordinary result. Ordered by precedence: the run ending outranks the
    game ending, which outranks a level completing."""
    if result.get("run_complete"):
        return "run_complete"
    if result.get("game_over"):
        return "game_over"
    if result.get("level_completed"):
        return "level_completed"
    if result.get("done"):
        return "done"
    return None


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
