"""Outcome-reward GRPO over complete puzzle groups from one behavior policy."""

from __future__ import annotations

import math

from training.baseline_batch import BaselineSignalBatch
from training.contracts import RolloutTrajectory
from training.rewards import AdvantageBatchStats, StepSignal


def compute_grpo_step_signals(
    rollouts: list[RolloutTrajectory],
    *,
    group_size: int,
    puzzles_per_window: int,
    normalize_std: bool = True,
) -> tuple[list[list[StepSignal]], AdvantageBatchStats]:
    """Validate the full collection window and assign one advantage per episode.

    With ``normalize_std=False`` (Dr. GRPO) the advantage is the episode reward
    minus the inclusive group mean, without the group standard-deviation
    division; everything else about the window contract is identical.
    """
    if group_size < 2 or puzzles_per_window < 1:
        raise ValueError(
            "GRPO requires at least two episodes per group and one puzzle."
        )
    if len(rollouts) != group_size * puzzles_per_window:
        raise ValueError(
            "Incomplete GRPO window: "
            f"expected {group_size * puzzles_per_window} episodes, got {len(rollouts)}."
        )
    groups: dict[str, list[int]] = {}
    slots: set[str] = set()
    trajectory_ids: set[str] = set()
    seeds: set[int] = set()
    snapshot = rollouts[0].metadata.get("behavior_policy_snapshot")
    if snapshot is None or snapshot == "" or snapshot == {}:
        raise ValueError("GRPO episodes require a behavior_policy_snapshot.")
    rewards: list[float] = []
    for index, rollout in enumerate(rollouts):
        if not rollout.task_id or not rollout.done:
            raise ValueError(
                "GRPO requires canonical puzzle identities and terminal episodes."
            )
        if rollout.metadata.get("behavior_policy_snapshot") != snapshot:
            raise ValueError("GRPO window mixes behavior-policy snapshots.")
        slot = rollout.metadata.get("rollout_slot_id")
        if not isinstance(slot, str) or not slot or slot in slots:
            raise ValueError("GRPO requires unique rollout_slot_id values.")
        if not rollout.trajectory_id or rollout.trajectory_id in trajectory_ids:
            raise ValueError("GRPO requires unique trajectory identities.")
        seed = rollout.metadata.get("rollout_seed")
        if type(seed) is not int or seed in seeds:
            raise ValueError("GRPO requires unique integer rollout_seed values.")
        reward = float(rollout.trajectory_reward)
        if not math.isfinite(reward):
            raise ValueError("GRPO requires finite terminal training rewards.")
        slots.add(slot)
        trajectory_ids.add(rollout.trajectory_id)
        seeds.add(seed)
        rewards.append(reward)
        groups.setdefault(rollout.task_id, []).append(index)
    if len(groups) != puzzles_per_window or any(
        len(indices) != group_size for indices in groups.values()
    ):
        raise ValueError(
            f"GRPO requires {puzzles_per_window} complete puzzle groups of {group_size}."
        )

    advantages = [0.0] * len(rollouts)
    for indices in groups.values():
        group_rewards = [rewards[index] for index in indices]
        if all(reward == group_rewards[0] for reward in group_rewards):
            continue
        mean = math.fsum(group_rewards) / group_size
        if normalize_std:
            variance = (
                math.fsum((reward - mean) ** 2 for reward in group_rewards) / group_size
            )
            denominator = math.sqrt(variance) + 1e-8
        else:
            denominator = 1.0
        for index in indices:
            advantages[index] = (rewards[index] - mean) / denominator

    signals = [
        [
            StepSignal(
                reward=reward if step_index == len(rollout.steps) - 1 else 0.0,
                return_to_go=reward,
                advantage=advantage,
            )
            for step_index in range(len(rollout.steps))
        ]
        for rollout, reward, advantage in zip(
            rollouts, rewards, advantages, strict=True
        )
    ]
    token_counts = [
        sum(len(step.completion_token_ids) for step in rollout.steps)
        for rollout in rollouts
    ]
    mass = sum(token_counts)
    mean_advantage = (
        math.fsum(
            count * advantage
            for count, advantage in zip(token_counts, advantages, strict=True)
        )
        / mass
        if mass
        else 0.0
    )
    mean_reward = math.fsum(rewards) / len(rewards)
    return signals, AdvantageBatchStats(
        rollout_count=len(rollouts),
        mean_reward=mean_reward,
        std_reward=math.sqrt(
            math.fsum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        ),
        min_reward=min(rewards),
        max_reward=max(rewards),
        mean_advantage=mean_advantage,
        std_advantage=(
            math.sqrt(
                math.fsum(
                    count * (advantage - mean_advantage) ** 2
                    for count, advantage in zip(token_counts, advantages, strict=True)
                )
                / mass
            )
            if mass
            else 0.0
        ),
    )


def _moments(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def compute_grpo_signal_batch(
    rollouts: list[RolloutTrajectory],
    *,
    group_size: int,
    puzzles_per_window: int,
    normalize_std: bool = True,
) -> BaselineSignalBatch:
    """Wrap the GRPO signals in the trainer's critic-free baseline batch.

    The group mean reward plays the value role: raw advantages are reward
    minus the group mean, before the group standard-deviation normalization
    that produces the training advantages. ``normalize_std=False`` (Dr. GRPO)
    skips that division, so raw and training advantages coincide.
    """
    signals, stats = compute_grpo_step_signals(
        rollouts,
        group_size=group_size,
        puzzles_per_window=puzzles_per_window,
        normalize_std=normalize_std,
    )
    rewards = [float(rollout.trajectory_reward) for rollout in rollouts]
    groups: dict[str, list[int]] = {}
    for index, rollout in enumerate(rollouts):
        groups.setdefault(rollout.task_id, []).append(index)
    baselines = [0.0] * len(rollouts)
    for indices in groups.values():
        mean = math.fsum(rewards[index] for index in indices) / len(indices)
        for index in indices:
            baselines[index] = mean
    raw_advantages = [
        reward - baseline for reward, baseline in zip(rewards, baselines, strict=True)
    ]
    advantages = [steps[0].advantage if steps else 0.0 for steps in signals]
    count = len(rollouts)
    mean_raw, std_raw = _moments(raw_advantages)
    return BaselineSignalBatch(
        per_rollout_signals=signals,
        stats=stats,
        rollouts=list(rollouts),
        mean_value=_moments(baselines)[0],
        mean_target=stats.mean_reward,
        mean_raw_advantage=mean_raw,
        std_raw_advantage=std_raw,
        explained_variance=(
            1.0 - (std_raw / stats.std_reward) ** 2 if stats.std_reward else 0.0
        ),
        diagnostics={
            "episode_count": float(count),
            "group_count": float(len(groups)),
            "group_size": float(group_size),
            "zero_advantage_episode_fraction": (
                sum(value == 0.0 for value in advantages) / count
            ),
            "nonzero_advantage_episode_count": float(
                sum(value != 0.0 for value in advantages)
            ),
            "raw_mean": mean_raw,
            "raw_std": std_raw,
            "mean_absolute_error": math.fsum(abs(value) for value in raw_advantages)
            / count,
            "advantage_mean": stats.mean_advantage,
            "advantage_std": stats.std_advantage,
        },
    )
