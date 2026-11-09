from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training.contracts import RolloutTrajectory


@dataclass(slots=True)
class RolloutMetrics:
    avg_return: float
    avg_levels_completed: float
    max_levels_completed: float
    return_std: float
    min_return: float
    max_return: float
    avg_rollout_length: float
    avg_actions_per_rollout: float
    avg_generation_rounds_per_rollout: float
    avg_tokens_per_generation: float
    avg_step_reward: float
    done_fraction: float
    # Logged independently of the configured reward: the mean fraction of a
    # game's levels completed (as a percentage) and the mean official ARC-AGI
    # score (0-100, action-efficiency based), regardless of reward_mode.
    avg_levels_completed_pct: float = 0.0
    avg_arc_score: float = 0.0
    # Number of distinct games in the window that earned a non-zero trajectory
    # return (the variance-sampling window's "solved at least once" count).
    games_nonzero_return: float = 0.0
    # Number of rollouts (environment instances) in the window that earned a
    # non-zero trajectory return.
    rollouts_nonzero_return: float = 0.0
    # Rejected attempts (malformed turns the harness rolled back and
    # retried) per rollout, and their share of all model turns: the
    # degeneration early-warning the collapsed runs lacked.
    avg_void_attempts_per_rollout: float = 0.0
    void_attempt_fraction: float = 0.0
    # ARC-2 malformed calls per rollout and their uncapped penalty
    # (harness_support/arc2/results.py); an empty reply is one malformed call,
    # and the empty replies among them are counted apart, for observation.
    avg_malformed_calls: float = 0.0
    avg_malformed_penalty: float = 0.0
    avg_empty_replies_per_rollout: float = 0.0
    # Share of episodes charged training.max_episode_penalty instead of their
    # penalty.
    penalty_cap_fraction: float = 0.0

    def as_scalars(self, *, scalar_prefix: str = "rollout") -> dict[str, float]:
        return {
            f"{scalar_prefix}/avg_return": self.avg_return,
            f"{scalar_prefix}/avg_levels_completed": self.avg_levels_completed,
            f"{scalar_prefix}/avg_levels_completed_pct": self.avg_levels_completed_pct,
            f"{scalar_prefix}/avg_arc_score": self.avg_arc_score,
            f"{scalar_prefix}/games_nonzero_return": self.games_nonzero_return,
            f"{scalar_prefix}/rollouts_nonzero_return": self.rollouts_nonzero_return,
            f"{scalar_prefix}/max_levels_completed": self.max_levels_completed,
            f"{scalar_prefix}/return_std": self.return_std,
            f"{scalar_prefix}/min_return": self.min_return,
            f"{scalar_prefix}/max_return": self.max_return,
            f"{scalar_prefix}/avg_length": self.avg_rollout_length,
            f"{scalar_prefix}/avg_actions_per_rollout": (self.avg_actions_per_rollout),
            f"{scalar_prefix}/avg_generation_rounds_per_rollout": (
                self.avg_generation_rounds_per_rollout
            ),
            f"{scalar_prefix}/avg_tokens_per_generation": (
                self.avg_tokens_per_generation
            ),
            f"{scalar_prefix}/avg_step_reward": self.avg_step_reward,
            f"{scalar_prefix}/done_fraction": self.done_fraction,
            f"{scalar_prefix}/avg_void_attempts_per_rollout": (
                self.avg_void_attempts_per_rollout
            ),
            f"{scalar_prefix}/void_attempt_fraction": self.void_attempt_fraction,
            f"{scalar_prefix}/avg_malformed_calls": self.avg_malformed_calls,
            f"{scalar_prefix}/avg_malformed_penalty": self.avg_malformed_penalty,
            f"{scalar_prefix}/avg_empty_replies_per_rollout": (
                self.avg_empty_replies_per_rollout
            ),
            f"{scalar_prefix}/penalty_cap_fraction": self.penalty_cap_fraction,
        }


def _return_std(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    return float(
        (sum((value - mean_return) ** 2 for value in returns) / len(returns)) ** 0.5
    )


def _step_token_count(step: Any, kind: str) -> int:
    """Prompt or generated tokens: captured ids, else server usage counts.

    Official-eval rollouts run with id/logprob capture off, so the usage
    counts recorded in the turn facts are the only generation-size signal.
    """
    token_ids = getattr(step, f"{kind}_token_ids")
    if token_ids:
        return len(token_ids)
    metadata = step.metadata if isinstance(step.metadata, dict) else {}
    usage = metadata.get("usage")
    if isinstance(usage, dict):
        try:
            return max(0, int(usage.get(f"{kind}_tokens") or 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _step_completion_tokens(step: Any) -> int:
    return _step_token_count(step, "completion")


def collection_scalars(
    rollouts: list[RolloutTrajectory],
    collection_seconds: float,
    *,
    scalar_prefix: str = "rollout",
) -> dict[str, float]:
    """Window throughput uses collection wall time, including environment waits.

    Segment starts after the first reported step are compactions: the harness
    flags them on ``history_epoch`` changes, and the gateway lineage ledger
    admits a reset only with compaction evidence (the ARC-3 duck never rewrites
    history at a level boundary). Void attempts are independent packing
    segments, not compactions. Counts are zero when the metadata is absent.
    """
    completion_tokens = sum(
        _step_completion_tokens(step) for rollout in rollouts for step in rollout.steps
    )
    compactions = sum(
        bool(step.metadata.get("segment_start"))
        for rollout in rollouts
        for step in [s for s in rollout.steps if not _is_void_attempt(s)][1:]
    )
    scalars = {
        f"{scalar_prefix}/tokens_per_s": (
            completion_tokens / collection_seconds if collection_seconds > 0 else 0.0
        ),
        f"{scalar_prefix}/compactions_per_rollout": (
            compactions / len(rollouts) if rollouts else 0.0
        ),
    }
    if scalar_prefix == "rollout":
        prompt_tokens = sum(
            _step_token_count(step, "prompt")
            for rollout in rollouts
            for step in rollout.steps
        )
        scalars["rollout/prompt_tokens_per_s"] = (
            prompt_tokens / collection_seconds if collection_seconds > 0 else 0.0
        )
        # Generated tokens of every turn of an episode, per raw rollout: the
        # iteration log's avg_tokens_per_rollout.
        scalars["rollout/completion_tokens_per_rollout"] = (
            completion_tokens / len(rollouts) if rollouts else 0.0
        )
    return scalars


# Expert load balance of the Flash learner's routing (training/expert_load.py),
# under the same names for every token role: train/ (every token of the
# trained sequences), rollout/ (the sampled completion tokens) and eval/ (the
# eval pass). Layer means first, then the worst layer of three of them.
EXPERT_LOAD_ROLES = ("train", "rollout", "eval")
EXPERT_LOAD_SCALARS = (
    "expert_load_cv",
    "expert_load_max_share",
    "expert_load_entropy",
    "experts_unused_fraction",
    "expert_load_cv_worst_layer",
    "expert_load_max_share_worst_layer",
    "expert_load_entropy_worst_layer",
)


def expert_load_scalars(summary: dict[str, dict[str, float]]) -> dict[str, float]:
    """TensorBoard scalars of one window's expert load summary, by tag.

    A role without counted tokens (eval when no eval episodes reached the
    learner) has no tags that window."""
    scalars = {
        f"{role}/{name}": float(summary[role][name])
        for role in EXPERT_LOAD_ROLES
        if summary.get(role)
        for name in EXPERT_LOAD_SCALARS
    }
    eval_pass = summary.get("eval_pass")
    if eval_pass:
        scalars["eval/expert_load_episodes"] = float(eval_pass["eval_episodes"])
        scalars["train/z_timing/expert_load_eval_seconds"] = float(
            eval_pass["eval_seconds"]
        )
    return scalars


def format_expert_load(summary: dict[str, dict[str, float]]) -> str:
    """One log-line rendering of a window's expert load summary."""
    parts = []
    for role in EXPERT_LOAD_ROLES:
        stats = summary.get(role)
        if not stats:
            continue
        parts.append(
            f"{role}[tokens={stats['tokens']:.0f} cv={stats['expert_load_cv']:.4f} "
            f"max_share={stats['expert_load_max_share']:.3f} "
            f"entropy={stats['expert_load_entropy']:.5f} "
            f"unused={stats['experts_unused_fraction']:.4f} "
            f"worst_layer={int(stats['worst_layer'])}"
            f"(cv={stats['expert_load_cv_worst_layer']:.4f} "
            f"max_share={stats['expert_load_max_share_worst_layer']:.3f} "
            f"entropy={stats['expert_load_entropy_worst_layer']:.5f})]"
        )
    eval_pass = summary.get("eval_pass")
    if eval_pass:
        parts.append(
            f"eval_pass[episodes={eval_pass['eval_episodes']:.0f} "
            f"seconds={eval_pass['eval_seconds']:.1f}]"
        )
    return " ".join(parts)


def _levels_completed(rollout: RolloutTrajectory) -> float:
    if not isinstance(rollout.metadata, dict):
        return 0.0
    try:
        return max(0.0, float(rollout.metadata.get("levels_completed", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _total_levels(rollout: RolloutTrajectory) -> float:
    if not isinstance(rollout.metadata, dict):
        return 1.0
    try:
        return max(1.0, float(rollout.metadata.get("game_total_levels", 1.0) or 1.0))
    except (TypeError, ValueError):
        return 1.0


def _arc_score(rollout: RolloutTrajectory) -> float:
    if not isinstance(rollout.metadata, dict):
        return 0.0
    try:
        return max(0.0, float(rollout.metadata.get("final_arc_score", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_void_attempt(step: Any) -> bool:
    metadata = step.metadata if isinstance(step.metadata, dict) else {}
    return bool(metadata.get("void_attempt"))


def _void_attempt_count(rollout: RolloutTrajectory) -> int:
    """Rejected attempts of the run: the assembly's count when it reports
    one, else the void steps present (a fixture without the count)."""
    metadata = rollout.metadata if isinstance(rollout.metadata, dict) else {}
    if "void_attempts" in metadata:
        try:
            return max(0, int(metadata.get("void_attempts") or 0))
        except (TypeError, ValueError):
            return 0
    return sum(1 for step in rollout.steps if _is_void_attempt(step))


def _run_total_actions(rollout: RolloutTrajectory) -> float:
    if not isinstance(rollout.metadata, dict):
        return 0.0
    try:
        return max(0.0, float(rollout.metadata.get("run_total_actions", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def summarize_rollouts(rollouts: list[RolloutTrajectory]) -> RolloutMetrics:
    if not rollouts:
        return RolloutMetrics(
            avg_return=0.0,
            avg_levels_completed=0.0,
            max_levels_completed=0.0,
            return_std=0.0,
            min_return=0.0,
            max_return=0.0,
            avg_rollout_length=0.0,
            avg_actions_per_rollout=0.0,
            avg_generation_rounds_per_rollout=0.0,
            avg_tokens_per_generation=0.0,
            avg_step_reward=0.0,
            done_fraction=0.0,
        )

    returns = [rollout.trajectory_reward for rollout in rollouts]
    levels_completed = [_levels_completed(rollout) for rollout in rollouts]
    levels_pct = [
        min(
            100.0, max(0.0, 100.0 * _levels_completed(rollout) / _total_levels(rollout))
        )
        for rollout in rollouts
    ]
    arc_scores = [_arc_score(rollout) for rollout in rollouts]
    scores_by_puzzle: dict[str, list[float]] = {}
    for rollout, score in zip(rollouts, arc_scores, strict=True):
        scores_by_puzzle.setdefault(rollout.task_id, []).append(score)
    equal_puzzle_score = sum(
        sum(values) / len(values) for values in scores_by_puzzle.values()
    ) / len(scores_by_puzzle)
    action_counts = [_run_total_actions(rollout) for rollout in rollouts]
    nonzero_return_games = {
        rollout.task_id
        for rollout in rollouts
        if rollout.task_id and float(rollout.trajectory_reward) != 0.0
    }
    nonzero_return_rollouts = sum(
        1 for rollout in rollouts if float(rollout.trajectory_reward) != 0.0
    )
    # Generation-round and token stats cover the REPORTED turns; rejected
    # attempts (void steps, when trained on) are summarized separately so
    # both series stay comparable across configurations.
    reported_steps = [
        [step for step in rollout.steps if not _is_void_attempt(step)]
        for rollout in rollouts
    ]
    lengths = [len(steps) for steps in reported_steps]
    total_generation_rounds = sum(lengths)
    total_generation_tokens = sum(
        _step_completion_tokens(step) for steps in reported_steps for step in steps
    )
    step_rewards = [step.reward for steps in reported_steps for step in steps]
    void_attempts = [_void_attempt_count(rollout) for rollout in rollouts]
    total_void_attempts = sum(void_attempts)
    done_flags = [1.0 if rollout.done else 0.0 for rollout in rollouts]
    avg_rollout_length = sum(lengths) / len(lengths)

    return RolloutMetrics(
        avg_return=sum(returns) / len(returns),
        avg_levels_completed=sum(levels_completed) / len(levels_completed),
        avg_levels_completed_pct=sum(levels_pct) / len(levels_pct),
        avg_arc_score=equal_puzzle_score
        if all(r.metadata.get("score_aggregation") == "per_task_mean" for r in rollouts)
        else sum(arc_scores) / len(arc_scores),
        avg_malformed_calls=sum(
            float(r.metadata.get("malformed_call_count", 0)) for r in rollouts
        )
        / len(rollouts),
        avg_malformed_penalty=sum(
            float(r.metadata.get("malformed_penalty", 0)) for r in rollouts
        )
        / len(rollouts),
        avg_empty_replies_per_rollout=sum(
            float(r.metadata.get("empty_reply_count", 0)) for r in rollouts
        )
        / len(rollouts),
        penalty_cap_fraction=sum(
            1.0 for r in rollouts if r.metadata.get("penalty_capped") is True
        )
        / len(rollouts),
        games_nonzero_return=float(len(nonzero_return_games)),
        rollouts_nonzero_return=float(nonzero_return_rollouts),
        max_levels_completed=max(levels_completed),
        return_std=_return_std(returns),
        min_return=min(returns),
        max_return=max(returns),
        avg_rollout_length=avg_rollout_length,
        avg_actions_per_rollout=sum(action_counts) / len(action_counts),
        avg_generation_rounds_per_rollout=avg_rollout_length,
        avg_tokens_per_generation=(
            total_generation_tokens / total_generation_rounds
            if total_generation_rounds > 0
            else 0.0
        ),
        avg_step_reward=(sum(step_rewards) / len(step_rewards))
        if step_rewards
        else 0.0,
        done_fraction=sum(done_flags) / len(done_flags),
        avg_void_attempts_per_rollout=total_void_attempts / len(rollouts),
        void_attempt_fraction=(
            total_void_attempts / (total_void_attempts + total_generation_rounds)
            if (total_void_attempts + total_generation_rounds) > 0
            else 0.0
        ),
    )


def episode_levels_completed(rollout: RolloutTrajectory) -> float:
    """Count the levels completed by a full-game rollout."""
    return max(0.0, float((rollout.metadata or {}).get("levels_completed", 0) or 0))
