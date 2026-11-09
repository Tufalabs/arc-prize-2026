"""Opt-in decision-state critic for complete finite-horizon rollouts.

The policy's final prompt hidden state conditions a small detached value model.
GAE spans all model responses, including compaction summaries and a terminal
unreported deadline draw. Gamma is per generated token; lambda is per decision.
The pilot uses gamma=lambda=1, so targets are undiscounted reward-to-go.
This is not the token-level value model used by CompactionRL.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from training.contracts import RolloutTrajectory
from training.critic_normalization import (
    PerGameTargetNormalizer,
    validate_target_normalization,
)
from training.rewards import AdvantageBatchStats, StepSignal
from training.baseline_batch import _require_unique_rollouts


log = logging.getLogger(__name__)
STATE_CRITIC_ADVANTAGE_EPSILON = 1e-8
STATE_CRITIC_STATE_VERSION = 8
STATE_CRITIC_PRIOR_HEAD_WEIGHT_STD = 1e-2
DEFAULT_STATE_CRITIC_TARGET_SCALE = 100.0


@dataclass(slots=True)
class StateCriticObservation:
    rollout_index: int
    step_index: int
    game_id: str
    reward: float
    score_so_far: float
    return_to_go: float
    levels_remaining: float
    total_levels: float
    analysis_step: int
    max_steps: int
    discount_token_count: int = 1
    time_remaining_seconds: float = 0.0
    max_runtime_seconds: float = 1.0
    # ARC score (0-115) still achievable on the current level if it were completed
    # at the current action count -- decays as actions accumulate.
    potential_score_now: float = 0.0
    # actions spent on the current level so far, as a fraction of its baseline.
    action_efficiency_ratio: float = 0.0
    # fraction of the rollout context budget still unused at this step; the
    # context can reset at compaction; only the runtime budget ends the episode.
    context_remaining_fraction: float = 1.0
    # Detached last-layer hidden state of the POLICY at this step's final
    # prompt token (the position whose logits sample the first completion
    # token) — the model's own summary of the observed state. None when
    # feature-only unit fixtures omit it; the hidden-critic runtime requires it.
    policy_hidden: list[float] | None = None
    # A final captured response that the harness could not consume at its
    # deadline. It remains a terminal decision with zero immediate reward.
    void_attempt: bool = False
    # The level the decision is made on (0-based, absolute within the game):
    # a level episode starts anywhere in the game, and the value of a state
    # depends on which level it is.
    level_index: int = 0


@dataclass(slots=True)
class StateCriticSignalBatch:
    per_rollout_signals: list[list[StepSignal]]
    stats: AdvantageBatchStats
    observations: list[StateCriticObservation]
    mean_value: float
    mean_target: float
    mean_raw_advantage: float
    std_raw_advantage: float
    explained_variance: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class StateCriticUpdateStats:
    loss: float = 0.0
    update_steps: int = 0
    mean_prediction: float = 0.0
    mean_target: float = 0.0
    mean_absolute_error: float = 0.0
    explained_variance: float = 0.0


def explained_variance(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    epsilon: float = 1e-8,
) -> float:
    if len(predictions) != len(targets):
        raise ValueError(
            "explained variance requires equally sized predictions and targets"
        )
    if not targets:
        return 0.0
    mean_target = sum(float(value) for value in targets) / len(targets)
    target_variance = sum((float(value) - mean_target) ** 2 for value in targets) / len(
        targets
    )
    if target_variance <= epsilon:
        return 0.0
    residuals = [
        float(target) - float(prediction)
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    mean_residual = sum(residuals) / len(residuals)
    residual_variance = sum(
        (residual - mean_residual) ** 2 for residual in residuals
    ) / len(residuals)
    return 1.0 - residual_variance / target_variance


def stable_game_bucket(game_id: str, *, buckets: int) -> int:
    bucket_count = max(1, int(buckets))
    digest = hashlib.blake2b(game_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % bucket_count


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rollout_total_levels(rollout: RolloutTrajectory) -> float:
    metadata = _mapping(rollout.metadata)
    total_levels = _optional_float(metadata.get("game_total_levels"))
    if total_levels is not None and total_levels > 0.0:
        return total_levels
    observed_total = 0.0
    for step in rollout.steps:
        stage = _mapping(_mapping(step.metadata).get("stage"))
        level = _optional_float(stage.get("level"))
        levels_remaining = _optional_float(stage.get("levels_remaining"))
        if level is not None:
            observed_total = max(observed_total, level)
        if level is not None and levels_remaining is not None:
            observed_total = max(
                observed_total,
                level + max(0.0, levels_remaining) - 1.0,
            )
    return max(1.0, observed_total)


def _rollout_max_steps(rollout: RolloutTrajectory) -> int:
    metadata = _mapping(rollout.metadata)
    configured = _optional_float(metadata.get("max_steps"))
    observed = float(len(rollout.steps))
    for step in rollout.steps:
        step_metadata = _mapping(step.metadata)
        analysis_step = _optional_float(step_metadata.get("analysis_step"))
        if analysis_step is not None:
            observed = max(observed, analysis_step)
    if configured is not None and configured > 0.0:
        observed = max(observed, configured)
    return max(1, int(math.ceil(observed)))


def _rollout_max_runtime_seconds(rollout: RolloutTrajectory) -> float | None:
    metadata = _mapping(rollout.metadata)
    configured = _optional_float(metadata.get("max_runtime_seconds"))
    if configured is not None and configured > 0.0:
        return configured
    observed = 0.0
    for step in rollout.steps:
        stage = _mapping(_mapping(step.metadata).get("stage"))
        elapsed = _optional_float(stage.get("run_elapsed_seconds"))
        remaining = _optional_float(stage.get("time_remaining_seconds"))
        if elapsed is not None and remaining is not None:
            observed = max(observed, elapsed + remaining)
    return observed if observed > 0.0 else None


def _step_discount_token_count(step: Any) -> int:
    return len(getattr(step, "completion_token_ids", []) or [])


def extract_state_critic_observations(
    rollouts: list[RolloutTrajectory],
    *,
    context_window_tokens: int,
) -> list[StateCriticObservation]:
    context_window = max(1, int(context_window_tokens))
    observations: list[StateCriticObservation] = []
    for rollout_index, rollout in enumerate(rollouts):
        total_levels = _rollout_total_levels(rollout)
        max_steps = _rollout_max_steps(rollout)
        max_runtime_seconds = _rollout_max_runtime_seconds(rollout)
        score_so_far = 0.0
        trajectory_reward = float(rollout.trajectory_reward)
        for step in rollout.steps:
            metadata = _mapping(step.metadata)
            stage = _mapping(metadata.get("stage"))
            raw_analysis_step = _optional_float(metadata.get("analysis_step"))
            if raw_analysis_step is None:
                raw_analysis_step = float(step.step_index) + 1.0
            analysis_step = max(1, int(math.ceil(raw_analysis_step)))
            levels_completed = _optional_float(
                stage.get("levels_completed_before_decision")
            )
            if levels_completed is None:
                levels_completed = 0.0
            levels_remaining = _optional_float(stage.get("levels_remaining"))
            if levels_remaining is None:
                levels_remaining = max(0.0, total_levels - float(levels_completed))
            stage_level = _optional_float(stage.get("level"))
            level_index = (
                max(0, int(stage_level) - 1)
                if stage_level is not None
                else max(0, int(levels_completed))
            )
            time_remaining_seconds = _optional_float(
                stage.get("time_remaining_seconds")
            )
            observation_max_runtime_seconds = max_runtime_seconds
            if time_remaining_seconds is None:
                time_remaining_seconds = max(
                    0.0, float(max_steps) - float(analysis_step) + 1.0
                )
                if observation_max_runtime_seconds is None:
                    observation_max_runtime_seconds = float(max_steps)
            elif observation_max_runtime_seconds is None:
                elapsed_seconds = _optional_float(stage.get("run_elapsed_seconds"))
                observation_max_runtime_seconds = (
                    max(1.0, elapsed_seconds + time_remaining_seconds)
                    if elapsed_seconds is not None
                    else max(1.0, time_remaining_seconds)
                )

            observations.append(
                StateCriticObservation(
                    rollout_index=rollout_index,
                    step_index=int(step.step_index),
                    game_id=str(rollout.task_id),
                    reward=float(step.reward),
                    score_so_far=float(score_so_far),
                    return_to_go=trajectory_reward - score_so_far,
                    levels_remaining=max(0.0, float(levels_remaining)),
                    total_levels=max(1.0, float(total_levels)),
                    analysis_step=analysis_step,
                    max_steps=max(max_steps, analysis_step),
                    discount_token_count=_step_discount_token_count(step),
                    time_remaining_seconds=max(0.0, float(time_remaining_seconds)),
                    max_runtime_seconds=max(
                        1.0, float(observation_max_runtime_seconds or 1.0)
                    ),
                    potential_score_now=max(
                        0.0, _optional_float(stage.get("potential_score_now")) or 0.0
                    ),
                    action_efficiency_ratio=max(
                        0.0,
                        _optional_float(stage.get("action_efficiency_ratio")) or 0.0,
                    ),
                    void_attempt=bool(metadata.get("void_attempt")),
                    level_index=level_index,
                    context_remaining_fraction=min(
                        1.0,
                        max(
                            0.0,
                            1.0 - len(step.prompt_token_ids or []) / context_window,
                        ),
                    ),
                )
            )
            score_so_far += float(step.reward)
    return observations


class GameStateCritic(torch.nn.Module):
    def __init__(
        self,
        *,
        game_embedding_buckets: int,
        game_embedding_dim: int,
        hidden_dim: int,
        initial_value: float | None = None,
        policy_hidden_dim: int = 0,
        policy_hidden_projection_dim: int = 512,
        level_embedding_buckets: int = 32,
        level_embedding_dim: int = 8,
        value_head_count: int = 1,
    ) -> None:
        super().__init__()
        self.game_embedding_buckets = max(1, int(game_embedding_buckets))
        self.game_embedding = torch.nn.Embedding(
            self.game_embedding_buckets,
            max(1, int(game_embedding_dim)),
        )
        # Which level of the game the state is on (0-based; levels past the
        # last bucket share it). Learned alongside the game row so the
        # critic can tell a level-3 start from a level-1 start of one game.
        self.level_embedding_buckets = max(1, int(level_embedding_buckets))
        self.level_embedding = torch.nn.Embedding(
            self.level_embedding_buckets,
            max(1, int(level_embedding_dim)),
        )
        # Optional policy-hidden conditioning: the detached last-layer hidden
        # state of the policy at the step's final prompt token, normalized
        # (the representation drifts as LoRA updates land) and projected into
        # the critic's feature space. Gradients never reach the policy — the
        # input is detached at extraction; this projection trains with the
        # critic loss only.
        self.policy_hidden_dim = max(0, int(policy_hidden_dim))
        projection_dim = 0
        if self.policy_hidden_dim > 0:
            projection_dim = max(1, int(policy_hidden_projection_dim))
            self.policy_hidden_projection = torch.nn.Sequential(
                torch.nn.LayerNorm(self.policy_hidden_dim),
                torch.nn.Linear(self.policy_hidden_dim, projection_dim),
                torch.nn.SiLU(),
            )
        else:
            self.policy_hidden_projection = None
        # base scalar features: levels_remaining, time_remaining, potential_score,
        # action_efficiency, score_so_far, context_remaining
        feature_dim = (
            self.game_embedding.embedding_dim
            + self.level_embedding.embedding_dim
            + 6
            + projection_dim
        )
        hidden = max(1, int(hidden_dim))
        self.base = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, value_head_count),
        )
        if initial_value is not None:
            self._initialize_value_prior(initial_value)

    def _initialize_value_prior(self, initial_value: float) -> None:
        final_layer = self.base[-1]
        if not isinstance(final_layer, torch.nn.Linear):
            raise TypeError("state critic value head must be a Linear layer")
        with torch.no_grad():
            if initial_value == 0.0:
                # No random residual signal before the first observed reward:
                # batch normalization must not amplify initialization noise.
                torch.nn.init.zeros_(final_layer.weight)
            else:
                torch.nn.init.normal_(
                    final_layer.weight,
                    mean=0.0,
                    std=STATE_CRITIC_PRIOR_HEAD_WEIGHT_STD,
                )
            torch.nn.init.constant_(final_layer.bias, float(initial_value))

    def forward(
        self,
        *,
        game_buckets: torch.Tensor,
        levels_remaining_fraction: torch.Tensor,
        time_remaining_fraction: torch.Tensor,
        potential_score_fraction: torch.Tensor,
        action_efficiency: torch.Tensor,
        score_so_far_fraction: torch.Tensor,
        context_remaining_fraction: torch.Tensor,
        policy_hidden: torch.Tensor | None = None,
        level_buckets: torch.Tensor | None = None,
        value_heads: torch.Tensor | None = None,
    ) -> torch.Tensor:
        game_features = self.game_embedding(game_buckets)
        if level_buckets is None:
            level_buckets = torch.zeros_like(game_buckets)
        level_features = self.level_embedding(
            level_buckets.clamp(0, self.level_embedding_buckets - 1)
        )
        base_features = torch.stack(
            [
                levels_remaining_fraction,
                time_remaining_fraction,
                potential_score_fraction,
                action_efficiency,
                score_so_far_fraction,
                context_remaining_fraction,
            ],
            dim=-1,
        )
        features = [game_features, level_features, base_features]
        if self.policy_hidden_projection is not None:
            if policy_hidden is None:
                policy_hidden = torch.zeros(
                    (game_buckets.shape[0], self.policy_hidden_dim),
                    dtype=base_features.dtype,
                    device=base_features.device,
                )
            features.append(self.policy_hidden_projection(policy_hidden))
        values = self.base(torch.cat(features, dim=-1))
        if value_heads is not None:
            return values.gather(1, value_heads.unsqueeze(-1)).squeeze(-1)
        return values.squeeze(-1)


class StateCriticBaseline:
    def __init__(
        self,
        *,
        game_embedding_buckets: int = 4096,
        game_embedding_dim: int = 16,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | str = "cpu",
        seed: int = 0,
        initial_value: float | None = None,
        game_vocab: Sequence[str] | None = None,
        target_scale: float = DEFAULT_STATE_CRITIC_TARGET_SCALE,
        target_normalization: str = "fixed",
        normalization_alpha: float = 0.125,
        normalization_min_rollouts: int = 16,
        normalization_min_std: float = 1.0,
        policy_hidden_dim: int = 0,
        policy_hidden_projection_dim: int = 512,
    ) -> None:
        self.device = torch.device(device)
        self.target_scale = max(1e-8, float(target_scale))
        if not math.isfinite(self.target_scale):
            raise ValueError("state critic target_scale must be finite")
        validate_target_normalization(
            target_normalization,
            normalization_alpha,
            normalization_min_rollouts,
            normalization_min_std,
        )
        self.target_normalization = target_normalization
        self.game_embedding_buckets = max(1, int(game_embedding_buckets))
        # Known games get collision-free dedicated embedding rows (1..len);
        # row 0 plus the hashed tail region cover unexpected game ids.
        vocab_ids = sorted(
            {str(g).strip() for g in (game_vocab or ()) if str(g).strip()}
        )
        if len(vocab_ids) + 1 >= self.game_embedding_buckets:
            raise ValueError(
                f"state critic needs game_embedding_buckets > len(game_vocab) + 1; "
                f"got {self.game_embedding_buckets} buckets for {len(vocab_ids)} games."
            )
        self.game_vocab: dict[str, int] = {
            game_id: index + 1 for index, game_id in enumerate(vocab_ids)
        }
        if target_normalization == "per_game" and not self.game_vocab:
            raise ValueError(
                "per_game critic target normalization requires a game vocabulary"
            )
        self.target_normalizer = (
            PerGameTargetNormalizer(
                self.game_vocab,
                initial_scale=self.target_scale,
                alpha=normalization_alpha,
                min_rollouts=normalization_min_rollouts,
                min_std=normalization_min_std,
            )
            if target_normalization == "per_game"
            else None
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.network = GameStateCritic(
                game_embedding_buckets=self.game_embedding_buckets,
                game_embedding_dim=game_embedding_dim,
                hidden_dim=hidden_dim,
                initial_value=initial_value,
                policy_hidden_dim=policy_hidden_dim,
                policy_hidden_projection_dim=policy_hidden_projection_dim,
                value_head_count=len(self.game_vocab) + 1
                if self.target_normalizer
                else 1,
            )
            # Preserve the old raw-score initialization exactly while making
            # the network itself predict normalized units.
            final_layer = self.network.base[-1]
            if not isinstance(final_layer, torch.nn.Linear):
                raise TypeError("state critic value head must be a Linear layer")
            with torch.no_grad():
                final_layer.weight.div_(self.target_scale)
                final_layer.bias.div_(self.target_scale)
        self.network.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    def _game_bucket(self, game_id: str) -> int:
        known = self.game_vocab.get(str(game_id).strip())
        if known is not None:
            return known
        # Unknown ids hash into the region above the vocab so they can never
        # collide with a known game's row.
        floor = len(self.game_vocab) + 1
        span = max(1, self.game_embedding_buckets - floor)
        return floor + stable_game_bucket(game_id, buckets=span)

    def _tensorize(
        self, observations: list[StateCriticObservation]
    ) -> dict[str, torch.Tensor]:
        game_buckets = [
            self._game_bucket(observation.game_id) for observation in observations
        ]
        level_buckets = [
            min(
                max(0, int(observation.level_index)),
                self.network.level_embedding_buckets - 1,
            )
            for observation in observations
        ]
        levels_remaining_fraction = [
            min(
                1.0,
                max(
                    0.0,
                    observation.levels_remaining / max(1.0, observation.total_levels),
                ),
            )
            for observation in observations
        ]
        time_remaining_fraction = [
            min(
                1.0,
                max(
                    0.0,
                    float(observation.time_remaining_seconds)
                    / max(1.0, float(observation.max_runtime_seconds)),
                ),
            )
            for observation in observations
        ]
        potential_score_fraction = [
            min(1.5, max(0.0, float(observation.potential_score_now) / 100.0))
            for observation in observations
        ]
        action_efficiency = [
            min(1.0, max(0.0, float(observation.action_efficiency_ratio) / 4.0))
            for observation in observations
        ]
        score_so_far_fraction = [
            min(1.0, max(0.0, float(observation.score_so_far) / 100.0))
            for observation in observations
        ]
        context_remaining_fraction = [
            min(1.0, max(0.0, float(observation.context_remaining_fraction)))
            for observation in observations
        ]
        targets = [float(observation.return_to_go) for observation in observations]
        target_parameters = [
            self.target_parameters(observation.game_id) for observation in observations
        ]
        policy_hidden: torch.Tensor | None = None
        if self.network.policy_hidden_dim > 0:
            rows = []
            for observation in observations:
                hidden = observation.policy_hidden
                if (
                    hidden is None
                    or len(hidden) != self.network.policy_hidden_dim
                    or not all(math.isfinite(value) for value in hidden)
                ):
                    raise RuntimeError(
                        "hidden critic requires finite policy hidden features on every observation"
                    )
                rows.append([float(value) for value in hidden])
            policy_hidden = torch.tensor(rows, dtype=torch.float32, device=self.device)
        return {
            "game_buckets": torch.tensor(
                game_buckets, dtype=torch.long, device=self.device
            ),
            "level_buckets": torch.tensor(
                level_buckets, dtype=torch.long, device=self.device
            ),
            "levels_remaining_fraction": torch.tensor(
                levels_remaining_fraction, dtype=torch.float32, device=self.device
            ),
            "time_remaining_fraction": torch.tensor(
                time_remaining_fraction, dtype=torch.float32, device=self.device
            ),
            "potential_score_fraction": torch.tensor(
                potential_score_fraction, dtype=torch.float32, device=self.device
            ),
            "action_efficiency": torch.tensor(
                action_efficiency, dtype=torch.float32, device=self.device
            ),
            "score_so_far_fraction": torch.tensor(
                score_so_far_fraction, dtype=torch.float32, device=self.device
            ),
            "context_remaining_fraction": torch.tensor(
                context_remaining_fraction, dtype=torch.float32, device=self.device
            ),
            "targets": torch.tensor(targets, dtype=torch.float32, device=self.device),
            "normalized_targets": torch.tensor(
                [
                    (target - mean) / std
                    for target, (mean, std, _) in zip(
                        targets, target_parameters, strict=True
                    )
                ],
                dtype=torch.float32,
                device=self.device,
            ),
            "value_heads": (
                torch.tensor(
                    [
                        self.game_vocab.get(observation.game_id.strip(), 0)
                        for observation in observations
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                if self.target_normalizer
                else None
            ),
            "policy_hidden": policy_hidden,
        }

    def target_parameters(self, game_id: str) -> tuple[float, float, int]:
        if self.target_normalizer is None:
            return 0.0, self.target_scale, 0
        return self.target_normalizer.parameters(game_id.strip())

    def _update_target_normalization(
        self, observations: list[StateCriticObservation]
    ) -> None:
        if self.target_normalizer is None or not observations:
            return
        updated = copy.deepcopy(self.target_normalizer)
        updated.update(
            (
                observation.game_id.strip(),
                observation.rollout_index,
                observation.return_to_go,
            )
            for observation in observations
        )
        # Row 0 serves unknown games using the pooled fallback. Rescale every
        # head, including games absent from this window whose fallback changed.
        games = [""] + list(self.game_vocab)
        old = [self.target_normalizer.parameters(game) for game in games]
        new = [updated.parameters(game) for game in games]
        head = self.network.base[-1]
        ratio = head.weight.new_tensor(
            [a[1] / b[1] for a, b in zip(old, new, strict=True)]
        )
        shift = head.bias.new_tensor(
            [(a[0] - b[0]) / b[1] for a, b in zip(old, new, strict=True)]
        )
        if not bool(torch.isfinite(ratio).all() & torch.isfinite(shift).all()):
            raise RuntimeError("nonfinite critic normalization head transformation")
        with torch.no_grad():
            head.weight.mul_(ratio[:, None])
            head.bias.mul_(ratio).add_(shift)
            # Head gradients of normalized MSE change by this same scale.
            # Keep Adam's first/second moments in those new gradient units.
            for parameter, scale in ((head.weight, ratio[:, None]), (head.bias, ratio)):
                state = self.optimizer.state.get(parameter, {})
                if "exp_avg" in state:
                    state["exp_avg"].mul_(scale)
                    state["exp_avg_sq"].mul_(scale.square())
            for parameter in (head.weight, head.bias):
                values = [parameter, *self.optimizer.state.get(parameter, {}).values()]
                if any(
                    torch.is_tensor(value) and not bool(torch.isfinite(value).all())
                    for value in values
                ):
                    raise RuntimeError(
                        "nonfinite critic normalization head or optimizer state"
                    )
        self.target_normalizer = updated

    def _predict_tensor(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.network(
            game_buckets=tensors["game_buckets"],
            level_buckets=tensors["level_buckets"],
            levels_remaining_fraction=tensors["levels_remaining_fraction"],
            time_remaining_fraction=tensors["time_remaining_fraction"],
            potential_score_fraction=tensors["potential_score_fraction"],
            action_efficiency=tensors["action_efficiency"],
            score_so_far_fraction=tensors["score_so_far_fraction"],
            context_remaining_fraction=tensors["context_remaining_fraction"],
            policy_hidden=tensors.get("policy_hidden"),
            value_heads=tensors.get("value_heads"),
        )

    def predict(self, observations: list[StateCriticObservation]) -> list[float]:
        if not observations:
            return []
        self.network.eval()
        with torch.no_grad():
            predictions = self._predict_tensor(self._tensorize(observations))
        values = [
            self.target_parameters(observation.game_id)[0]
            + float(value) * self.target_parameters(observation.game_id)[1]
            for observation, value in zip(
                observations, predictions.detach().cpu().tolist(), strict=True
            )
        ]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("hidden critic produced a nonfinite value prediction")
        return values

    def update(
        self,
        observations: list[StateCriticObservation],
        *,
        epochs: int,
        batch_size: int,
        max_grad_norm: float,
    ) -> StateCriticUpdateStats:
        if not observations or epochs <= 0:
            predictions = self.predict(observations)
            targets = [observation.return_to_go for observation in observations]
            absolute_errors = [
                abs(float(target) - float(prediction))
                for prediction, target in zip(predictions, targets, strict=True)
            ]
            return StateCriticUpdateStats(
                loss=0.0,
                update_steps=0,
                mean_prediction=(sum(predictions) / len(predictions))
                if predictions
                else 0.0,
                mean_target=(sum(targets) / len(targets)) if targets else 0.0,
                mean_absolute_error=(
                    sum(absolute_errors) / len(absolute_errors)
                    if absolute_errors
                    else 0.0
                ),
                explained_variance=explained_variance(predictions, targets),
            )

        tensors = self._tensorize(observations)
        sample_count = len(observations)
        effective_batch_size = max(1, int(batch_size))
        update_steps = 0
        loss_sum = 0.0
        loss_weight = 0
        self.network.train()
        for _epoch in range(max(1, int(epochs))):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, effective_batch_size):
                indices = permutation[start : start + effective_batch_size]
                batch = {
                    key: value.index_select(0, indices)
                    for key, value in tensors.items()
                    if value is not None
                }
                predictions = self._predict_tensor(batch)
                loss = torch.nn.functional.mse_loss(
                    predictions,
                    batch["normalized_targets"],
                )
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(
                        "hidden critic produced a nonfinite training loss"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    float(max_grad_norm) if max_grad_norm > 0.0 else math.inf,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                batch_weight = int(indices.numel())
                loss_sum += float(loss.detach().cpu().item()) * batch_weight
                loss_weight += batch_weight
                update_steps += 1

        # Actor advantages and this fit use the previous window's statistics.
        # The new moments become available for the next window; this coordinate
        # change must preserve the just-fitted raw-score predictions.
        self._update_target_normalization(observations)
        final_predictions = self.predict(observations)
        final_targets = [
            float(observation.return_to_go) for observation in observations
        ]
        absolute_errors = [
            abs(float(target) - float(prediction))
            for prediction, target in zip(final_predictions, final_targets, strict=True)
        ]
        return StateCriticUpdateStats(
            loss=(loss_sum / loss_weight) if loss_weight > 0 else 0.0,
            update_steps=update_steps,
            mean_prediction=(sum(final_predictions) / len(final_predictions))
            if final_predictions
            else 0.0,
            mean_target=(sum(final_targets) / len(final_targets))
            if final_targets
            else 0.0,
            mean_absolute_error=(
                sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0
            ),
            explained_variance=explained_variance(final_predictions, final_targets),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_CRITIC_STATE_VERSION,
            "game_embedding_buckets": self.game_embedding_buckets,
            "game_vocab": dict(self.game_vocab),
            "target_scale": self.target_scale,
            "target_normalization": self.target_normalization,
            "target_normalizer": (
                self.target_normalizer.state_dict() if self.target_normalizer else None
            ),
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = int(state.get("version", 0))
        if version not in (7, STATE_CRITIC_STATE_VERSION):
            raise RuntimeError(
                f"Unsupported state critic checkpoint version {version}; "
                f"expected {STATE_CRITIC_STATE_VERSION}."
            )
        saved_buckets = int(state.get("game_embedding_buckets", 0))
        if saved_buckets != self.game_embedding_buckets:
            raise RuntimeError(
                "State critic checkpoint game_embedding_buckets mismatch "
                f"({saved_buckets} != {self.game_embedding_buckets})."
            )
        if state.get("game_vocab") != self.game_vocab:
            raise RuntimeError(
                "State critic checkpoint game vocabulary does not match current features"
            )
        saved_mode = "fixed" if version == 7 else state["target_normalization"]
        if saved_mode != self.target_normalization:
            raise RuntimeError("State critic checkpoint target_normalization mismatch")
        if saved_mode == "fixed" and state.get("target_normalizer") is not None:
            raise RuntimeError(
                "Fixed critic checkpoint contains unexpected normalization statistics"
            )
        normalizer = copy.deepcopy(self.target_normalizer)
        if normalizer is not None:
            normalizer.load_state_dict(state["target_normalizer"])
        saved_target_scale = float(state.get("target_scale", 0.0))
        if not math.isclose(
            saved_target_scale,
            self.target_scale,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "State critic checkpoint target_scale mismatch "
                f"({saved_target_scale} != {self.target_scale})."
            )
        self.network.load_state_dict(state["network_state_dict"])
        self.target_normalizer = normalizer
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in list(optimizer_state.items()):
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)


def _batch_stats(
    *,
    rollout_rewards: list[float],
    advantages: list[float],
) -> AdvantageBatchStats:
    if rollout_rewards:
        mean_reward = sum(rollout_rewards) / len(rollout_rewards)
        reward_variance = sum(
            (reward - mean_reward) ** 2 for reward in rollout_rewards
        ) / len(rollout_rewards)
        std_reward = math.sqrt(max(0.0, reward_variance))
        min_reward = min(rollout_rewards)
        max_reward = max(rollout_rewards)
    else:
        mean_reward = 0.0
        std_reward = 0.0
        min_reward = 0.0
        max_reward = 0.0

    if advantages:
        mean_advantage = sum(advantages) / len(advantages)
        advantage_variance = sum(
            (advantage - mean_advantage) ** 2 for advantage in advantages
        ) / len(advantages)
        std_advantage = math.sqrt(max(0.0, advantage_variance))
    else:
        mean_advantage = 0.0
        std_advantage = 0.0

    return AdvantageBatchStats(
        rollout_count=len(rollout_rewards),
        mean_reward=mean_reward,
        std_reward=std_reward,
        min_reward=min_reward,
        max_reward=max_reward,
        mean_advantage=mean_advantage,
        std_advantage=std_advantage,
    )


def _gae_returns_and_advantages(
    rollout_observations: list[StateCriticObservation],
    rollout_predictions: list[float],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    if not rollout_observations:
        return [], []
    gamma = min(1.0, max(0.0, float(gamma)))
    gae_lambda = min(1.0, max(0.0, float(gae_lambda)))
    advantages = [0.0] * len(rollout_observations)
    targets = [0.0] * len(rollout_observations)
    # Every captured decision belongs to the same finite-horizon chain.
    # History epochs and compaction/reset flags never introduce terminals.
    chain = list(range(len(rollout_observations)))
    next_advantage = 0.0
    for position in range(len(chain) - 1, -1, -1):
        index = chain[position]
        is_last = position == len(chain) - 1
        if is_last:
            # Every episode end is TERMINAL with V=0 -- game over/win AND the
            # configured rollout clock alike. The time budget is part of the evaluation
            # objective (a finite-horizon MDP with a time-aware critic), not a
            # collection artifact: at time_remaining=0 no further reward is
            # achievable, so the true value of the post-horizon state is 0.
            # Internal compaction boundaries never reach this branch -- GAE
            # spans the complete episode across compactions.
            next_value = 0.0
            nonterminal = 0.0
        else:
            next_value = float(rollout_predictions[chain[position + 1]])
            nonterminal = 1.0
        discount = gamma ** max(
            1, int(rollout_observations[index].discount_token_count)
        )
        delta = (
            float(rollout_observations[index].reward)
            + discount * next_value * nonterminal
            - float(rollout_predictions[index])
        )
        next_advantage = delta + discount * gae_lambda * nonterminal * next_advantage
        advantages[index] = next_advantage
        targets[index] = next_advantage + float(rollout_predictions[index])
    return targets, advantages


def _token_weights(observations: list[StateCriticObservation]) -> list[float]:
    """Generated-token mass over the complete window, before any sampling."""
    return [float(observation.discount_token_count) for observation in observations]


def _normalize_and_clip_advantages(
    raw_advantages: list[float],
    observations: list[StateCriticObservation],
    *,
    normalize_advantages: bool,
    advantage_clip: float,
    epsilon: float,
) -> tuple[list[float], float, float]:
    """Match global token-mean actor mass before packing or sharding."""
    if not all(math.isfinite(value) for value in raw_advantages):
        raise RuntimeError("hidden critic produced a nonfinite advantage")
    weights = _token_weights(observations)
    mass = math.fsum(weights)
    mean = (
        math.fsum(w * a for w, a in zip(weights, raw_advantages, strict=True)) / mass
        if mass
        else 0.0
    )
    variance = (
        math.fsum(
            w * (a - mean) ** 2 for w, a in zip(weights, raw_advantages, strict=True)
        )
        / mass
        if mass
        else 0.0
    )
    std = math.sqrt(max(0.0, variance))
    if normalize_advantages:
        advantages = (
            [(a - mean) / std for a in raw_advantages]
            if std > epsilon
            else [0.0] * len(raw_advantages)
        )
    else:
        advantages = list(raw_advantages)
    clip = max(0.0, float(advantage_clip))
    if clip:
        advantages = [max(-clip, min(clip, a)) for a in advantages]
    return advantages, mean, std


def _prediction_diagnostics(
    observations: list[StateCriticObservation], predictions: list[float]
) -> dict[str, float]:
    """Raw-score errors on the new window, before its critic fit or stats update."""
    groups: dict[str, list[tuple[StateCriticObservation, float]]] = {}
    for observation, prediction in zip(observations, predictions, strict=True):
        groups.setdefault("", []).append((observation, prediction))
        groups.setdefault(f"per_game/{observation.game_id}/", []).append(
            (observation, prediction)
        )
    diagnostics = {}
    for prefix, rows in groups.items():
        values = [prediction for _, prediction in rows]
        targets = [observation.return_to_go for observation, _ in rows]
        errors = [value - target for value, target in zip(values, targets, strict=True)]
        bias = math.fsum(errors) / len(errors)
        for name, value in {
            "bias": bias,
            "rmse": math.sqrt(math.fsum(error**2 for error in errors) / len(errors)),
            "mae": math.fsum(abs(error) for error in errors) / len(errors),
            "residual_std": math.sqrt(
                math.fsum((error - bias) ** 2 for error in errors) / len(errors)
            ),
            "explained_variance": explained_variance(values, targets),
            "mean_target": math.fsum(targets) / len(targets),
            "decisions": len(rows),
            "rollouts": len({observation.rollout_index for observation, _ in rows}),
        }.items():
            diagnostics[f"{prefix}pre_update/{name}"] = float(value)
    return diagnostics


def compute_state_critic_step_signals(
    rollouts: list[RolloutTrajectory],
    *,
    context_window_tokens: int,
    critic: StateCriticBaseline,
    normalize_advantages: bool = True,
    epsilon: float = STATE_CRITIC_ADVANTAGE_EPSILON,
    gae_gamma: float = 1.0,
    gae_lambda: float = 1.0,
    advantage_clip: float = 0.0,
    policy_hidden_by_step: dict[tuple[int, int], list[float]] | None = None,
) -> StateCriticSignalBatch:
    _require_unique_rollouts(rollouts)
    observations = extract_state_critic_observations(
        rollouts, context_window_tokens=context_window_tokens
    )
    if critic.network.policy_hidden_dim > 0:
        if policy_hidden_by_step is None and observations:
            raise RuntimeError(
                "hidden critic requires policy hidden states for every decision"
            )
        for observation in observations:
            hidden = (policy_hidden_by_step or {}).get(
                (observation.rollout_index, observation.step_index)
            )
            if (
                hidden is None
                or len(hidden) != critic.network.policy_hidden_dim
                or not all(math.isfinite(v) for v in hidden)
            ):
                raise RuntimeError(
                    "hidden critic requires finite, correctly sized policy hidden states for every decision"
                )
            observation.policy_hidden = hidden
    rollout_rewards = [float(rollout.trajectory_reward) for rollout in rollouts]
    predictions = critic.predict(observations)

    observations_by_rollout: list[list[StateCriticObservation]] = [
        [] for _rollout in rollouts
    ]
    predictions_by_rollout: list[list[float]] = [[] for _rollout in rollouts]
    for observation, prediction in zip(observations, predictions, strict=True):
        observations_by_rollout[observation.rollout_index].append(observation)
        predictions_by_rollout[observation.rollout_index].append(float(prediction))

    raw_advantages: list[float] = []
    for rollout_index, rollout_observations in enumerate(observations_by_rollout):
        targets, rollout_advantages = _gae_returns_and_advantages(
            rollout_observations,
            predictions_by_rollout[rollout_index],
            gamma=gae_gamma,
            gae_lambda=gae_lambda,
        )
        for observation, target, advantage in zip(
            rollout_observations, targets, rollout_advantages, strict=True
        ):
            observation.return_to_go = float(target)
            raw_advantages.append(float(advantage))

    advantages, mean_raw_advantage, std_raw_advantage = _normalize_and_clip_advantages(
        raw_advantages,
        observations,
        normalize_advantages=normalize_advantages,
        advantage_clip=advantage_clip,
        epsilon=epsilon,
    )
    advantages_by_observation = {
        (observation.rollout_index, observation.step_index): advantage
        for observation, advantage in zip(observations, advantages, strict=True)
    }
    signals: list[list[StepSignal]] = [[] for _rollout in rollouts]
    for observation in observations:
        key = (observation.rollout_index, observation.step_index)
        signals[observation.rollout_index].append(
            StepSignal(
                reward=float(observation.reward),
                return_to_go=float(observation.return_to_go),
                advantage=float(advantages_by_observation[key]),
            )
        )

    stats = _batch_stats(rollout_rewards=rollout_rewards, advantages=advantages)
    weights = _token_weights(observations)
    mass = math.fsum(weights)
    weighted_mean = (
        math.fsum(w * a for w, a in zip(weights, advantages, strict=True)) / mass
        if mass
        else 0.0
    )
    stats.mean_advantage = weighted_mean
    stats.std_advantage = (
        math.sqrt(
            math.fsum(
                w * (a - weighted_mean) ** 2
                for w, a in zip(weights, advantages, strict=True)
            )
            / mass
        )
        if mass
        else 0.0
    )
    diagnostics = _prediction_diagnostics(observations, predictions)
    if isinstance(critic, StateCriticBaseline):
        for game_id in sorted({observation.game_id for observation in observations}):
            mean, std, source = critic.target_parameters(game_id)
            prefix = f"per_game/{game_id}/normalization/"
            diagnostics.update(
                {
                    f"{prefix}mean": mean,
                    f"{prefix}std": std,
                    f"{prefix}source": float(source),
                }
            )
            if critic.target_normalizer is not None:
                moments = critic.target_normalizer.games.get(game_id.strip())
                diagnostics[f"{prefix}rollouts"] = float(
                    moments.rollout_count if moments else 0
                )
                diagnostics[f"{prefix}pooled_rollouts"] = float(
                    critic.target_normalizer.pooled.rollout_count
                )
    return StateCriticSignalBatch(
        per_rollout_signals=signals,
        stats=stats,
        observations=observations,
        mean_value=(sum(predictions) / len(predictions)) if predictions else 0.0,
        mean_target=(
            sum(observation.return_to_go for observation in observations)
            / len(observations)
        )
        if observations
        else 0.0,
        mean_raw_advantage=mean_raw_advantage,
        std_raw_advantage=std_raw_advantage,
        explained_variance=explained_variance(
            predictions,
            [observation.return_to_go for observation in observations],
        ),
        diagnostics=diagnostics,
    )
