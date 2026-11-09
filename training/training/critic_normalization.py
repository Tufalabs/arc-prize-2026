"""Running critic target moments; actor rewards always remain in ARC-score units."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


def validate_target_normalization(
    mode: str, alpha: float, min_rollouts: int, min_std: float
) -> None:
    if mode not in ("fixed", "per_game"):
        raise ValueError(
            "state_critic_target_normalization must be 'fixed' or 'per_game'"
        )
    if not 0.0 < alpha <= 1.0:
        raise ValueError("state_critic_normalization_alpha must be in (0, 1]")
    if (
        isinstance(min_rollouts, bool)
        or not isinstance(min_rollouts, int)
        or min_rollouts < 2
    ):
        raise ValueError(
            "state_critic_normalization_min_rollouts must be an integer >= 2"
        )
    if not math.isfinite(min_std) or min_std <= 0.0:
        raise ValueError(
            "state_critic_normalization_min_std must be finite and positive"
        )


@dataclass
class TargetMoments:
    mean: float = 0.0
    variance: float = 0.0
    rollout_count: int = 0
    nonzero_seen: bool = False

    def update(self, mean: float, variance: float, count: int, *, alpha: float) -> None:
        if self.rollout_count:
            delta = mean - self.mean
            self.mean += alpha * delta
            self.variance = (
                (1.0 - alpha) * self.variance
                + alpha * variance
                + alpha * (1.0 - alpha) * delta**2
            )
        else:
            self.mean, self.variance = mean, variance
        self.rollout_count += count
        self.nonzero_seen |= mean != 0.0 or variance > 0.0
        if not math.isfinite(self.mean) or not math.isfinite(self.variance):
            raise RuntimeError("nonfinite critic target normalization statistics")

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TargetMoments:
        moments = cls(**state)
        if (
            not math.isfinite(moments.mean)
            or not math.isfinite(moments.variance)
            or moments.variance < 0.0
            or type(moments.rollout_count) is not int
            or moments.rollout_count < 0
            or type(moments.nonzero_seen) is not bool
            or (
                not moments.nonzero_seen
                and (moments.mean != 0.0 or moments.variance != 0.0)
            )
            or (moments.rollout_count == 0 and moments.nonzero_seen)
        ):
            raise ValueError("invalid critic target normalization moments")
        return moments


class PerGameTargetNormalizer:
    """One EMA update per window, with readiness counted in distinct rollouts.

    Game moments weight decisions equally, matching the critic MSE. Pooled
    fallback moments weight observed games equally. Repeated epochs and long
    conversations cannot count as extra rollouts. All-zero history retains the
    initial scale instead of amplifying the first sparse success by 100x.
    """

    def __init__(
        self,
        games: Iterable[str],
        *,
        initial_scale: float,
        alpha: float,
        min_rollouts: int,
        min_std: float,
    ) -> None:
        validate_target_normalization("per_game", alpha, min_rollouts, min_std)
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("critic initial target scale must be finite and positive")
        self.initial_scale = initial_scale
        self.alpha = alpha
        self.min_rollouts = min_rollouts
        self.min_std = min_std
        self.games = {game: TargetMoments() for game in sorted(games)}
        self.pooled = TargetMoments()

    def parameters(self, game: str) -> tuple[float, float, int]:
        """Return mean, std and source (0=initial, 1=pooled, 2=game)."""
        for source, moments in ((2, self.games.get(game)), (1, self.pooled)):
            if (
                moments is not None
                and moments.rollout_count >= self.min_rollouts
                and moments.nonzero_seen
            ):
                return (
                    moments.mean,
                    max(self.min_std, math.sqrt(moments.variance)),
                    source,
                )
        return 0.0, self.initial_scale, 0

    def update(self, rows: Iterable[tuple[str, int, float]]) -> None:
        grouped: dict[str, list[float]] = {}
        rollout_ids: dict[str, set[int]] = {}
        for game, rollout_index, target in rows:
            if not math.isfinite(target):
                raise RuntimeError("nonfinite critic target normalization input")
            grouped.setdefault(game, []).append(target)
            rollout_ids.setdefault(game, set()).add(rollout_index)
        if not grouped:
            return
        batches = {}
        for game, targets in grouped.items():
            mean = math.fsum(targets) / len(targets)
            variance = math.fsum((target - mean) ** 2 for target in targets) / len(
                targets
            )
            batches[game] = (mean, variance)
        pooled_mean = math.fsum(mean for mean, _ in batches.values()) / len(batches)
        pooled_variance = math.fsum(
            variance + (mean - pooled_mean) ** 2 for mean, variance in batches.values()
        ) / len(batches)
        self.pooled.update(
            pooled_mean,
            pooled_variance,
            sum(len(ids) for ids in rollout_ids.values()),
            alpha=self.alpha,
        )
        for game, (mean, variance) in batches.items():
            if game in self.games:
                self.games[game].update(
                    mean, variance, len(rollout_ids[game]), alpha=self.alpha
                )

    def state_dict(self) -> dict[str, Any]:
        return {
            "initial_scale": self.initial_scale,
            "alpha": self.alpha,
            "min_rollouts": self.min_rollouts,
            "min_std": self.min_std,
            "pooled": asdict(self.pooled),
            "games": {game: asdict(moments) for game, moments in self.games.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("initial_scale", "alpha", "min_rollouts", "min_std"):
            if state[key] != getattr(self, key):
                raise ValueError(
                    f"critic target normalization checkpoint {key} mismatch"
                )
        if state["games"].keys() != self.games.keys():
            raise ValueError(
                "critic target normalization checkpoint vocabulary mismatch"
            )
        pooled = TargetMoments.from_state(state["pooled"])
        games = {
            game: TargetMoments.from_state(value)
            for game, value in state["games"].items()
        }
        if sum(value.rollout_count for value in games.values()) > pooled.rollout_count:
            raise ValueError(
                "critic target normalization checkpoint rollout count mismatch"
            )
        self.pooled, self.games = pooled, games
