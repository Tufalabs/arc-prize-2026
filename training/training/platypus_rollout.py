"""Assemble gateway-captured rollout windows into training trajectories.

The platypus harness is a black box behind the rollout gateway
(training/rollout_gateway.py): every model call it makes flows through the
gateway (which records engine-witnessed prompt/completion token ids and
sampled logprobs in memory), and at the end of each game run the harness
worker POSTs a ``RunResult`` — game outcome plus, per model step, the call
nonce, the ``action_index`` linking the step to the first action it
authorized, and an output checksum. No artifact files are read on the
training side; the platypus run dir remains a viewer/debug surface only.

This module joins the two in-memory sources into ``RolloutTrajectory``
objects carrying the fields the PPO pipeline consumes (training/dataset.py,
training/state_critic.py, training/rewards.py, training/episode_packing.py):

- per-step token ids + old logprobs (importance ratios / parity oracle),
  each capture verified against the harness-witnessed output (length +
  crc32) so a transport-retry race can never corrupt training data,
- per-step ARC score-delta rewards: each completed level's contribution to
  the run's incremental ARC score (the v0.9.8 formula TAAF mirrors) is
  assigned to the model step that authorized the completing action,
- ``segment_start`` verified against the declared history epoch and the
  captured token prefix: every call within an epoch must continue the
  previous prompt+completion; only declared rewrites open new segments,
- critic stage features (level, remaining budget) from action wallclocks.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

from training.contracts import RolloutTrajectory
from harness_support.arc3.scoring import (  # noqa: F401 - re-exported
    _action_index_ranges,
    _cumulative_arc_scores,
)
from training.rollout_gateway import (
    GatewayError,
    RunResult,
    StepCapture,
    WindowCapture,
    token_ids_crc32,
)

log = logging.getLogger(__name__)


def _prefix_extends(previous: StepCapture, current: StepCapture) -> bool:
    """True when ``current``'s prompt extends previous prompt+completion."""
    expected = previous.prompt_token_ids + previous.completion_token_ids
    prompt = current.prompt_token_ids
    from training.gateway_lineage import _media_extends

    return (
        len(prompt) >= len(expected)
        and prompt[: len(expected)] == expected
        and _media_extends(previous.media, current.media)
    )


def _verified_segment_starts(
    run: RunResult, captures: Sequence[StepCapture]
) -> list[bool]:
    """Verify every reported step, including windows with only two calls.

    The epoch declares a compaction, which the gateway lineage ledger verifies
    against the served or measured over-trigger prompt; tokens alone cannot
    establish why history changed. Within an epoch,
    the gateway's captured ids independently prove exact continuation.
    A false/missing continuation flag must never silently create a segment.
    Unreported deadline/rejected draws have no declaration and are handled
    separately as independent void segments.
    """
    segment_starts: list[bool] = []
    previous_epoch: int | None = None
    for position, (report, capture) in enumerate(zip(run.steps, captures, strict=True)):
        label = f"run {run.run_id} step {position}"
        continued = report.get("continued")
        epoch = report.get("history_epoch")
        if type(continued) is not bool:
            raise GatewayError(f"{label}: continued must be an explicit boolean")
        if type(epoch) is not int or epoch < 0:
            raise GatewayError(
                f"{label}: history_epoch must be an explicit nonnegative integer"
            )
        if report.get("exact_history") is not True:
            raise GatewayError(f"{label}: exact_history=True is required for training")
        if previous_epoch is not None and epoch < previous_epoch:
            raise GatewayError(
                f"{label}: history_epoch decreased from {previous_epoch} to {epoch}"
            )

        rewritten = previous_epoch is None or epoch != previous_epoch
        if continued:
            if position == 0 or not _prefix_extends(captures[position - 1], capture):
                raise GatewayError(
                    f"{label}: the harness continued the history by tokens but the "
                    "prompt does not extend the previous step's prompt+completion "
                    "token-for-token"
                )
            if rewritten:
                raise GatewayError(
                    f"{label}: continued=True contradicts the declared history rewrite "
                    f"from epoch {previous_epoch} to {epoch}"
                )
            if not capture.native_token_prompt:
                raise GatewayError(
                    f"{label}: continued=True requires a native token-id prompt "
                    "witnessed at the gateway; a freshly tokenized request is not "
                    "native continuation"
                )
        elif not rewritten:
            raise GatewayError(
                f"{label}: continued=False within history_epoch {epoch}; no history "
                "rewrite was declared, so token continuation is broken"
            )
        segment_starts.append(rewritten)
        previous_epoch = epoch
    return segment_starts


def _resolve_task_id(game_id: str, configured_games: Sequence[str] | None) -> str:
    """Map a TAAF run game_id back to the configured game name.

    TAAF suffixes instance/version onto the env name (``pathfinder`` ->
    ``pathfinder-0001``), but the trainer keys per-game return histories,
    seeds, and schedulers by the CONFIGURED name — an unresolved id makes
    every history lookup miss (avg_return pinned at 0). Exact matches win;
    otherwise the longest configured name that prefixes the id at a ``-``
    boundary; otherwise the raw id (caller-visible in metadata either way).
    """
    if not configured_games:
        return game_id
    names = {str(name).strip() for name in configured_games if str(name).strip()}
    if game_id in names:
        return game_id
    candidates = [name for name in names if game_id.startswith(name + "-")]
    if candidates:
        return max(candidates, key=len)
    return game_id


def _resolve_step_capture(
    window: WindowCapture, run: RunResult, step_report: dict[str, Any]
) -> StepCapture:
    """Look up and verify the gateway capture for one reported step."""
    nonce = str(step_report["nonce"])
    capture = window.captures.get((run.run_id, nonce))
    if capture is None:
        raise GatewayError(
            f"run {run.run_id}: no gateway capture for step nonce {nonce} "
            f"({len(window.captures)} captures in window)"
        )
    captured_time = capture.progress.get("data", {}).get("decision_elapsed_seconds")
    if (
        captured_time is not None
        and step_report.get("decision_elapsed_seconds") != captured_time
    ):
        raise GatewayError(
            f"run {run.run_id} nonce {nonce}: callback decision timestamp differs from capture"
        )
    expected_len = int(step_report["output_len"])
    expected_crc = int(step_report["output_crc32"])
    actual_len = len(capture.completion_token_ids)
    actual_crc = token_ids_crc32(capture.completion_token_ids)
    if actual_len != expected_len or actual_crc != expected_crc:
        raise GatewayError(
            f"run {run.run_id} step nonce {nonce}: gateway capture does not "
            f"match the harness-witnessed output (len {actual_len} vs "
            f"{expected_len}, crc {actual_crc} vs {expected_crc}) — likely a "
            "transport-retry race; refusing to train on unverified tokens"
        )
    return capture


def build_window_trajectories(
    window: WindowCapture,
    *,
    policy_iteration: int,
    max_runtime_seconds: float | None,
    game_ids: Sequence[str] | None = None,
    include_void_attempts: bool = False,
) -> list[RolloutTrajectory]:
    """Join gateway captures with harness run results into trajectories.

    ``include_void_attempts`` retains captured terminal draws which did not
    finish before the deadline. The gateway rejects omitted interior draws;
    an unreported final draw does not itself imply malformed output.
    """
    trajectories: list[RolloutTrajectory] = []
    for run_id in sorted(window.results):
        trajectories.append(
            _run_to_trajectory(
                window,
                window.results[run_id],
                policy_iteration=policy_iteration,
                max_runtime_seconds=max_runtime_seconds,
                game_ids=game_ids,
                include_void_attempts=include_void_attempts,
            )
        )
    return trajectories


def _apply_decision_time(
    stage: dict[str, Any],
    capture: StepCapture,
    run: RunResult,
    max_runtime_seconds: float | None,
) -> None:
    """Use the request's pre-generation clock, never its eventual completion time."""
    elapsed = capture.progress.get("data", {}).get("decision_elapsed_seconds")
    if elapsed is None:
        # Legacy recordings have only action clocks; live gateway requests stamp this field.
        return
    if (
        type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or elapsed < stage["run_elapsed_seconds"]
        or (
            run.final_wallclock_seconds is not None
            and elapsed > run.final_wallclock_seconds
        )
    ):
        raise GatewayError("captured draw has invalid decision timestamp")
    stage["run_elapsed_seconds"] = elapsed
    stage["time_remaining_seconds"] = (
        max(0.0, max_runtime_seconds - elapsed)
        if max_runtime_seconds is not None
        else None
    )


def _void_captures(window: WindowCapture, run: RunResult) -> list[StepCapture]:
    """Captured terminal draws not claimed by a reported step."""
    accepted = {str(step_report["nonce"]) for step_report in run.steps}
    return sorted(
        (
            capture
            for (run_id, nonce), capture in window.captures.items()
            if run_id == run.run_id and nonce not in accepted
        ),
        key=lambda capture: capture.order,
    )


def _run_to_trajectory(
    window: WindowCapture,
    run: RunResult,
    *,
    policy_iteration: int,
    max_runtime_seconds: float | None,
    game_ids: Sequence[str] | None = None,
    include_void_attempts: bool = False,
) -> RolloutTrajectory:
    """Build one trajectory from a run result + its verified captures.

    The gateway contract (captures, continuation, checksums) is verified by
    the shared helpers above; the selected harness adapter supplies only the
    reward and metadata translation.
    """
    from training.harness import get_trainer_support

    run.validate()
    return get_trainer_support(run.environment).run_to_trajectory(
        window,
        run,
        policy_iteration=policy_iteration,
        max_runtime_seconds=max_runtime_seconds,
        game_ids=game_ids,
        include_void_attempts=include_void_attempts,
    )


def _status_from_state(state: str) -> str:
    """Map TAAF GameRun states onto the trainer's status vocabulary.

    ``crashed`` is a game whose agent raised, contained by the harness
    (``on_agent_error="crash_game"``) and scored on what it had reached: a
    terminal of its own, ``CRASHED``. ``ERROR`` is left for a run the harness
    never finished, which fails the run (training/cli.py).
    """
    mapping = {
        "won": "COMPLETED_RUN",
        "gave_up": "TIMEOUT",
        "cancelled": "TIMEOUT",
        "crashed": "CRASHED",
        "playing": "ERROR",
        "not_started": "ERROR",
    }
    return mapping.get(state, "ERROR")


# How much of a crashed run's harness note its trajectory keeps, from each end:
# the note's head names where it failed, a traceback's tail what was raised.
CRASH_NOTE_CHARS = 300


def crash_notes(run: RunResult) -> dict[str, str | None]:
    """A crashed run's harness note for its trajectory metadata: ``crash_note``,
    the note's first ``CRASH_NOTE_CHARS`` characters, and ``crash_note_tail``,
    its last (the exception a traceback ends on). Both None for any other run,
    or a crashed one without a note."""
    note = run.solver_note if run.state == "crashed" else None
    if not note:
        return {"crash_note": None, "crash_note_tail": None}
    return {
        "crash_note": note[:CRASH_NOTE_CHARS],
        "crash_note_tail": note[-CRASH_NOTE_CHARS:],
    }
