"""ARC-2 translations for the shared trainer process.

Pure Python: this runs inside the trainer and imports no harness package.
The worker-side adapter (task construction, harness API calls, response
journaling) is ``harness_support/arc2/worker.py``; the scoring shared with
the worker callback is ``harness_support/arc2/results.py``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from harness_support.arc2.dataset import dataset_manifest
from harness_support.arc2.results import TERMINAL_STATES, charged_penalty
from training.config import DEFAULT_MALFORMED_CALL_PENALTY, resolve_context_contract
from training.contracts import RolloutStep, RolloutTrajectory
from training.rollout_gateway import (
    GatewayError,
    RunResult,
    WindowCapture,
)

# ARC-2 prompts carry grid images: the trainer pins the serving image stack.
MEDIA: dict[str, bool] = {"images": True}
# Each run must echo the immutable policy it was sampled from.
REQUIRES_BEHAVIOR_POLICY_SNAPSHOT = True
# A standing submission scored at the runtime limit is also a deadline stop.
DEADLINE_TERMINAL_STATES = frozenset({"gave_up", "cancelled", "won", "submitted"})
# A summary request carries the whole live history, which can run past the
# window: the served ceiling keeps this much room above window plus summary
# cap (training/FLASH_RL.md, "Shrinking the lengths").
SERVED_OVERFLOW_HEADROOM_TOKENS = 12288
# What the engine serves ahead of each reply, as the Duck's engine preset
# resolves it: every run reports these agent settings, and the checkpoint
# identity pins them at the first window (rollout_identity).
PROMPT_INJECTION_FIELDS = (
    "inject_text_on_first_prompt",
    "inject_text_on_later_prompts",
    "prompt_injection_schedule",
)


def duck_summary_ceiling(window: int, reply_cap: int | None) -> int:
    """The ARC-2 Duck's own floor for the served ceiling under summarize.

    Since arc2-platypus f380ffd the Duck's engine-window derivation
    (``DuckAgent._derived_engine_window``, off in training) budgets the window,
    the reply reserve (512 when uncapped) and six times its window/16 summary
    target, the cap of an empty summary's retry with reasoning. Training never
    takes that retry (a retained failed draw ends the game), but the ceiling
    still meets the bound. A copy of the Duck's arithmetic: the worker checks
    the served ceiling against the built Duck at window start, and
    tests/test_arc2_worker_construction.py compares this copy with it
    (docs/arc2_training.md, "Where the trainer can drift from the Duck").
    """
    reply = reply_cap if type(reply_cap) is int and reply_cap > 0 else 512
    return window + reply + 6 * (window // 16)


def _resized_limits(inference: Any, evaluation: Any) -> dict[str, int]:
    """Keep Duck's window/16 summary target and the configured overflow headroom."""
    window = evaluation.context_window_tokens
    if window is None or window == inference.agent_context_window:
        return {}
    context = resolve_context_contract(inference)
    summary_cap = 2 * (window // 16)
    overflow_headroom = (
        context.max_served_sequence_tokens
        - context.context_window_tokens
        - context.max_summary_completion_tokens
    )
    return {
        "agent_compact_to_tokens": max(
            1024, window - context.max_completion_tokens - context.safety_margin_tokens
        ),
        "agent_max_summary_tokens": summary_cap,
        "agent_max_served_sequence_tokens": window + summary_cap + overflow_headroom,
    }


def evaluation_context_overrides(inference: Any, evaluation: Any) -> dict[str, int]:
    """The resized limits, with a served ceiling that meets the ARC-2 Duck's bound."""
    overrides = _resized_limits(inference, evaluation)
    if overrides:
        overrides["agent_max_served_sequence_tokens"] = max(
            overrides["agent_max_served_sequence_tokens"],
            duck_summary_ceiling(
                evaluation.context_window_tokens,
                resolve_context_contract(inference).max_completion_tokens,
            ),
        )
    return overrides


def validate_config(config: Any) -> None:
    """ARC-2 recipes are critic-free and evaluate with training sampling."""
    if config.training.advantage_estimator not in {"grpo", "dr_grpo"}:
        raise ValueError(
            "ARC-2 training requires advantage_estimator='grpo' or 'dr_grpo'."
        )
    if config.inference.dataset is None:
        raise ValueError("inference.dataset is required for ARC-2.")
    # The ARC-2 Duck always summarises and caps a summary at twice its
    # window/16 target. The rollout worker refuses any other cap
    # (harness_support/arc2/worker.py), but only when the first window starts;
    # evaluation windows derive their cap by this rule
    # (evaluation_context_overrides). A missing window fails the full-game
    # contract at collection, as in every environment. The gateway refuses a
    # request past the served ceiling mid-run, and an unset ceiling (the
    # window) fails the summary cap when collection starts, so loading also
    # requires the overflow headroom above window plus summary cap, and the
    # Duck's own summary bound (duck_summary_ceiling).
    window = config.inference.agent_context_window
    if type(window) is int and window > 0:
        summary_cap = 2 * (window // 16)
        configured = config.inference.agent_max_summary_tokens
        if configured != summary_cap:
            raise ValueError(
                "ARC-2 summaries are capped at twice the window/16 summary target: "
                f"inference.agent_context_window={window} requires "
                f"inference.agent_max_summary_tokens={summary_cap}, got {configured}."
            )
        required = max(
            window + summary_cap + SERVED_OVERFLOW_HEADROOM_TOKENS,
            duck_summary_ceiling(window, config.inference.agent_max_completion_tokens),
        )
        served = config.inference.agent_max_served_sequence_tokens
        if served is None or served < required:
            raise ValueError(
                f"ARC-2 serves {SERVED_OVERFLOW_HEADROOM_TOKENS} tokens above window "
                "plus summary cap, and at least the Duck's window + reply cap + 6 x summary target: "
                f"inference.agent_context_window={window} requires "
                f"inference.agent_max_served_sequence_tokens>={required}, got {served}."
            )
    if config.official_eval.enabled and config.official_eval.dataset is None:
        raise ValueError("official_eval.dataset is required for ARC-2 evaluation.")
    if config.official_eval.enabled and (
        config.official_eval.temperature != 1.0
        or config.official_eval.top_p != 1.0
        or config.official_eval.top_k not in (0, -1)
    ):
        raise ValueError("ARC-2 evaluation requires unrestricted training sampling.")


def _selection_manifest(
    dataset: str | list[str], games: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """The puzzle sets a section names and the games it plays from their union.

    One set played whole keeps that set's manifest, the identity of every run
    before game lists; otherwise the manifest names each set and the games.
    """
    from harness_support.arc2.dataset import dataset_names

    names = dataset_names(dataset)
    manifests = [dataset_manifest(name) for name in names]
    union = [game for manifest in manifests for game in manifest["puzzle_ids"]]
    if len(set(union)) != len(union):
        raise ValueError(f"ARC-2 datasets {names} share puzzles; name disjoint sets")
    if len(set(games)) != len(games):
        raise ValueError("ARC-2 games must be listed once each")
    known = set(union)
    missing = [game for game in games if game not in known]
    if missing:
        raise ValueError(f"ARC-2 games {missing} are not in {' + '.join(names)}")
    if len(manifests) == 1 and games in ([], union):
        return manifests[0], union
    selected = list(games or union)
    return {"datasets": manifests, "games": selected}, selected


def resolve_datasets(config: Any) -> dict[str, Any]:
    """Resolve all named puzzles before scheduling, keeping evaluation separate."""
    train, config.inference.games = _selection_manifest(
        config.inference.dataset, config.inference.games
    )
    evaluation = None
    if config.official_eval.dataset:
        evaluation, config.official_eval.games = _selection_manifest(
            config.official_eval.dataset, config.official_eval.games
        )
    config.inference.game = config.inference.games[0]
    return {"training": train, "evaluation": evaluation}


def training_identity(
    config: Any, manifest: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Bind an ARC-2 checkpoint to its experiment.

    The prompt injection is the harness's, not the recipe's: the trainer
    pins what the first window's rollouts report (``rollout_identity``,
    training/rollout_identity.py). The shared trainer adds the pinned
    image-processing identity because ``MEDIA`` declares images
    (training/multimodal.py).
    """
    if manifest is None:
        return None
    identity = {
        "environment": config.inference.environment,
        "advantage_estimator": config.training.advantage_estimator,
        "dataset_manifest": manifest,
    }
    # Omission means the historical 0.05, preserving existing checkpoints.
    # A different coefficient changes the reward identity for both backends.
    if config.training.malformed_call_penalty != DEFAULT_MALFORMED_CALL_PENALTY:
        identity["malformed_call_penalty"] = config.training.malformed_call_penalty
    # Omission means no cap, as every checkpoint before the cap was trained.
    if config.training.max_episode_penalty is not None:
        identity["max_episode_penalty"] = config.training.max_episode_penalty
    return identity


_MISSING = object()


def validate_run_result(run: RunResult) -> None:
    """Require a terminal-only normalized score and explicit penalty counts.

    The gateway calls this independently of the worker's own scoring: the
    reported reward must be reproducible from the terminal evidence, with the
    trainer's own coefficient and cap (``run.malformed_call_penalty`` and
    ``run.max_episode_penalty``, never the worker's): correctness minus
    min(cap, coefficient x malformed calls). Each step reports its malformed
    calls and whether it was an empty reply, which is one of them. Every
    terminal state is scored alike, ``crashed`` included (its agent raised;
    the harness scored the answer standing then), and the outcome names it.
    """
    outcome = run.arc2_outcome
    if not isinstance(outcome, dict):
        raise ValueError("ARC-2 requires a terminal outcome")
    if run.state not in TERMINAL_STATES:
        raise ValueError(f"ARC-2 run did not reach a terminal state: {run.state!r}")
    if outcome.get("termination_category") != run.state:
        raise ValueError("ARC-2 termination category disagrees with the run state")
    success = outcome.get("per_output_success")
    malformed = outcome.get("malformed_call_count")
    empty_replies = outcome.get("empty_reply_count")
    if (
        not isinstance(success, list)
        or not success
        or any(type(v) is not bool for v in success)
    ):
        raise ValueError("ARC-2 requires binary correctness for every test output")
    if type(malformed) is not int or malformed < 0:
        raise ValueError("ARC-2 malformed count must be a nonnegative integer")
    if type(empty_replies) is not int or empty_replies < 0:
        raise ValueError("ARC-2 empty-reply count must be a nonnegative integer")
    counts = [step.get("malformed_call_count") for step in run.steps]
    if any(type(v) is not int or v < 0 for v in counts) or sum(counts) != malformed:
        raise ValueError("ARC-2 malformed counts disagree with processed draws")
    empties = [step.get("empty_reply") for step in run.steps]
    if any(type(v) is not bool for v in empties) or sum(empties) != empty_replies:
        raise ValueError("ARC-2 empty-reply counts disagree with processed draws")
    if any(empty and not count for empty, count in zip(empties, counts, strict=True)):
        raise ValueError("ARC-2 empty replies are malformed calls, and must be charged")
    expected = sum(success) / len(success)
    malformed_penalty = run.malformed_call_penalty * malformed
    capped, charged = charged_penalty(malformed_penalty, run.max_episode_penalty)
    for name, value in (
        ("correctness", expected),
        ("malformed_penalty", malformed_penalty),
    ):
        actual = outcome.get(name)
        if type(actual) not in (int, float) or not math.isclose(
            actual, value, abs_tol=1e-9
        ):
            raise ValueError(f"ARC-2 {name} disagrees with terminal evidence")
    cap = outcome.get("max_episode_penalty", _MISSING)
    if cap is _MISSING or (cap is None) != (run.max_episode_penalty is None) or (
        cap is not None
        and (
            type(cap) not in (int, float)
            or not math.isclose(cap, run.max_episode_penalty, abs_tol=1e-9)
        )
    ):
        raise ValueError(
            "ARC-2 max_episode_penalty disagrees with the trainer's: "
            f"{run.max_episode_penalty!r}, the worker scored with "
            f"{'nothing' if cap is _MISSING else repr(cap)}"
        )
    if outcome.get("penalty_capped") is not capped:
        raise ValueError("ARC-2 penalty_capped disagrees with terminal evidence")
    reward = expected - charged
    if run.episode_return is None or not math.isclose(
        run.episode_return, reward, abs_tol=1e-9
    ):
        raise ValueError("ARC-2 training reward disagrees with terminal evidence")
    if run.terminal_reward != run.episode_return or any(
        step["reward"] != 0 for step in run.steps
    ):
        raise ValueError("ARC-2 correctness and penalties are terminal rewards")
    if (
        not isinstance(run.rollout_slot_id, str)
        or not run.rollout_slot_id
        or type(run.rollout_seed) is not int
    ):
        raise ValueError("ARC-2 requires independent slot identity and seed")
    if (
        not isinstance(run.behavior_policy_snapshot, dict)
        or not run.behavior_policy_snapshot
    ):
        raise ValueError("ARC-2 requires behavior-policy snapshot identity")
    settings = run.agent_settings
    if (
        not isinstance(settings, dict)
        or set(settings) != set(PROMPT_INJECTION_FIELDS)
        or any(not isinstance(settings[name], str) for name in PROMPT_INJECTION_FIELDS)
        or settings["prompt_injection_schedule"] not in {"game", "context"}
    ):
        raise ValueError(
            "ARC-2 requires the agent's prompt injection as agent_settings "
            f"({', '.join(PROMPT_INJECTION_FIELDS)}), got {settings!r}"
        )


def rollout_identity(rollouts: Sequence[RolloutTrajectory]) -> dict[str, Any] | None:
    """The prompt injection every rollout of a window reports, for the trainer
    to pin in the checkpoint identity (training/rollout_identity.py)."""
    reported: dict[str, Any] | None = None
    for rollout in rollouts:
        settings = rollout.metadata.get("agent_settings")
        if not isinstance(settings, dict):
            raise ValueError(f"ARC-2 rollout {rollout.trajectory_id} reports no agent settings")
        injection = {name: settings.get(name) for name in PROMPT_INJECTION_FIELDS}
        if reported is None:
            reported = injection
            continue
        for name in PROMPT_INJECTION_FIELDS:
            if injection[name] != reported[name]:
                raise ValueError(
                    f"ARC-2 rollouts of one window report different {name}: "
                    f"{reported[name]!r} and {injection[name]!r}"
                )
    return reported


def run_to_trajectory(
    window: WindowCapture,
    run: RunResult,
    *,
    policy_iteration: int,
    max_runtime_seconds: float | None = None,
    game_ids: Sequence[str] | None = None,
    include_void_attempts: bool = False,
) -> RolloutTrajectory:
    """Preserve exact draws and the terminal ARC-2 result without level scoring."""
    from training.platypus_rollout import (
        _resolve_step_capture,
        _status_from_state,
        _verified_segment_starts,
        crash_notes,
    )

    del max_runtime_seconds, game_ids, include_void_attempts
    captures = [_resolve_step_capture(window, run, report) for report in run.steps]
    segments = _verified_segment_starts(run, captures)
    outcome = dict(run.arc2_outcome or {})
    if run.augmentation is not None:
        # The grids this outcome was scored in (harness_support/augmentation.py).
        outcome["augmentation"] = run.augmentation
    if any(capture.adapter != run.serving_adapter for capture in captures):
        raise GatewayError(
            f"run {run.run_id}: callback behavior policy differs from served adapter"
        )
    steps = [
        RolloutStep(
            step_index=i,
            prompt_messages=list(capture.request_messages),
            tool_schemas=[],
            assistant_message=dict(capture.assistant_message),
            reward=run.terminal_reward if i == len(captures) - 1 else 0.0,
            done=i == len(captures) - 1,
            prompt_token_ids=list(capture.prompt_token_ids),
            completion_token_ids=list(capture.completion_token_ids),
            completion_topk_ids=capture.completion_topk_ids,
            completion_topk_logprobs=capture.completion_topk_logprobs,
            completion_logprobs=[
                {"token_id": token, "logprob": probability}
                for token, probability in zip(
                    capture.completion_token_ids,
                    capture.completion_logprobs,
                    strict=True,
                )
            ],
            metadata={
                "segment_start": segment,
                "logit_bias": dict(capture.logit_bias),
                "summary_tool_call": capture.summary_request
                and bool(
                    report.get(
                        "has_tool_calls", report.get("stop_reason") == "tool_use"
                    )
                ),
                "finish_reason": capture.finish_reason,
                "malformed_call_count": report["malformed_call_count"],
                "empty_reply": report["empty_reply"],
                # Every penalized reply, for rollout/format_break_fraction and
                # the collapse watch.
                "format_break": bool(report["malformed_call_count"])
                or report["empty_reply"],
            },
            media=capture.media,
            media_evidence=capture.media_evidence,
        )
        for i, (capture, segment, report) in enumerate(
            zip(captures, segments, run.steps, strict=True)
        )
    ]
    # Answered and finished is a completed run; gave_up and cancelled are
    # TIMEOUT, crashed (its agent raised; scored on its standing answer like
    # those) CRASHED.
    status = (
        "COMPLETED_RUN"
        if run.state in {"won", "submitted"}
        else _status_from_state(run.state)
    )
    return RolloutTrajectory(
        trajectory_id=f"{run.run_id}-{run.rollout_slot_id}",
        task_id=run.game_id,
        steps=steps,
        done=True,
        trajectory_reward=float(run.episode_return),
        termination_reason=status,
        metadata={
            "environment": "arc2",
            # Every puzzle weighs the same in the window score, however many
            # rollouts it received (training/metrics.py).
            "score_aggregation": "per_task_mean",
            "game_id": run.game_id,
            "requested_game": run.game_id,
            "run_id": run.run_id,
            "status": status,
            "final_arc_score": outcome["correctness"],
            "arc2_outcome": outcome,
            "malformed_call_count": outcome["malformed_call_count"],
            "malformed_penalty": outcome["malformed_penalty"],
            # malformed_penalty is uncapped; whether max_episode_penalty was
            # charged instead (rollout/penalty_cap_fraction).
            "penalty_capped": outcome["penalty_capped"],
            # Among the malformed calls, counted only.
            "empty_reply_count": outcome["empty_reply_count"],
            "run_duration_seconds": run.final_wallclock_seconds,
            "run_total_steps": len(steps),
            "run_total_actions": len(run.action_wallclock_seconds),
            "policy_iteration": policy_iteration,
            "rollout_policy_iteration": policy_iteration,
            "rollout_slot_id": run.rollout_slot_id,
            "rollout_seed": run.rollout_seed,
            "behavior_policy_snapshot": run.behavior_policy_snapshot,
            "serving_adapter": run.serving_adapter,
            "augmentation": run.augmentation,
            # The prompt injection the run's agent served, as the harness
            # resolved it (rollout_identity).
            "agent_settings": dict(run.agent_settings),
            # A crashed run's harness note, head and tail (the traceback's
            # raise); None otherwise.
            **crash_notes(run),
        },
    )
