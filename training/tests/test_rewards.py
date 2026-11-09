import math
from types import SimpleNamespace

import pytest

from training.contracts import RolloutStep, RolloutTrajectory
from training.state_critic import (
    StateCriticBaseline,
    compute_state_critic_step_signals,
    explained_variance,
    extract_state_critic_observations,
)


class _FixedCritic:
    def __init__(self, predictions: list[float]) -> None:
        self.predictions = predictions
        self.network = SimpleNamespace(policy_hidden_dim=0)

    def predict(self, observations):  # noqa: ANN001
        assert len(observations) == len(self.predictions)
        return list(self.predictions)


def test_compute_state_critic_step_signals_uses_return_to_go_residuals() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="traj-critic",
        task_id="pathfinder-0001",
        trajectory_reward=50.0,
        metadata={"game_total_levels": 3, "max_runtime_seconds": 300.0},
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=20.0,
                done=False,
                metadata={
                    "stage": {
                        "levels_completed_before_decision": 0,
                        "levels_remaining": 3,
                        "time_remaining_seconds": 250.0,
                    }
                },
            ),
            RolloutStep(
                step_index=1,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=30.0,
                done=True,
                metadata={
                    "stage": {
                        "levels_completed_before_decision": 1,
                        "levels_remaining": 2,
                        "time_remaining_seconds": 200.0,
                    }
                },
            ),
        ],
    )

    batch = compute_state_critic_step_signals(
        [rollout],
        critic=_FixedCritic([10.0, 25.0]),
        context_window_tokens=32768,
        normalize_advantages=False,
    )

    signals = batch.per_rollout_signals[0]
    assert [signal.reward for signal in signals] == pytest.approx([20.0, 30.0])
    assert [signal.return_to_go for signal in signals] == pytest.approx([50.0, 30.0])
    assert [signal.advantage for signal in signals] == pytest.approx([40.0, 5.0])
    assert batch.mean_value == pytest.approx(17.5)
    assert batch.mean_target == pytest.approx(40.0)


def test_compute_state_critic_step_signals_discounts_by_completion_tokens() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="traj-token-discount",
        task_id="pathfinder-0001",
        trajectory_reward=1.0,
        metadata={"game_total_levels": 1, "max_runtime_seconds": 300.0},
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=False,
                completion_token_ids=[101, 102, 103],
            ),
            RolloutStep(
                step_index=1,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=1.0,
                done=True,
                completion_token_ids=[104],
            ),
        ],
    )

    batch = compute_state_critic_step_signals(
        [rollout],
        critic=_FixedCritic([0.0, 0.0]),
        context_window_tokens=32768,
        normalize_advantages=False,
        gae_gamma=0.5,
        gae_lambda=1.0,
    )

    signals = batch.per_rollout_signals[0]
    assert [signal.return_to_go for signal in signals] == pytest.approx([0.125, 1.0])
    assert [signal.advantage for signal in signals] == pytest.approx([0.125, 1.0])


def test_state_critic_advantages_are_centered_before_scaling() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="traj-centered",
        task_id="pathfinder-0001",
        trajectory_reward=50.0,
        metadata={"game_total_levels": 3, "max_runtime_seconds": 300.0},
        steps=[
            RolloutStep(
                completion_token_ids=[1],
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=20.0,
                done=False,
            ),
            RolloutStep(
                completion_token_ids=[1],
                step_index=1,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=30.0,
                done=True,
            ),
        ],
    )

    batch = compute_state_critic_step_signals(
        [rollout],
        critic=_FixedCritic([10.0, 25.0]),
        context_window_tokens=32768,
        normalize_advantages=True,
    )

    signals = batch.per_rollout_signals[0]
    assert batch.mean_raw_advantage == pytest.approx(22.5)
    assert batch.std_raw_advantage == pytest.approx(17.5)
    assert [signal.advantage for signal in signals] == pytest.approx([1.0, -1.0])
    assert batch.stats.mean_advantage == pytest.approx(0.0)
    assert batch.stats.std_advantage == pytest.approx(1.0)


def test_state_critic_constant_residuals_normalize_to_zero_signal() -> None:
    rollouts = [
        RolloutTrajectory(
            trajectory_id=f"traj-constant-{index}",
            task_id="pathfinder-0001",
            trajectory_reward=10.0,
            metadata={"game_total_levels": 1, "max_runtime_seconds": 300.0},
            steps=[
                RolloutStep(
                    completion_token_ids=[1],
                    step_index=0,
                    prompt_messages=[],
                    tool_schemas=[],
                    assistant_message={},
                    reward=10.0,
                    done=True,
                )
            ],
        )
        for index in range(2)
    ]

    batch = compute_state_critic_step_signals(
        rollouts,
        critic=_FixedCritic([5.0, 5.0]),
        context_window_tokens=32768,
        normalize_advantages=True,
    )

    assert batch.mean_raw_advantage == pytest.approx(5.0)
    assert batch.std_raw_advantage == pytest.approx(0.0)
    assert [
        signal.advantage
        for rollout_signals in batch.per_rollout_signals
        for signal in rollout_signals
    ] == pytest.approx([0.0, 0.0])
    assert batch.stats.mean_advantage == pytest.approx(0.0)
    assert batch.stats.std_advantage == pytest.approx(0.0)


def test_zero_return_windows_retain_critic_residuals() -> None:
    rollouts = [
        RolloutTrajectory(
            trajectory_id=f"traj-zero-{index}",
            task_id="pathfinder-0001",
            trajectory_reward=0.0,
            metadata={"game_total_levels": 1, "max_runtime_seconds": 300.0},
            steps=[
                RolloutStep(
                    completion_token_ids=[1],
                    step_index=0,
                    prompt_messages=[],
                    tool_schemas=[],
                    assistant_message={},
                    reward=0.0,
                    done=True,
                )
            ],
        )
        for index in range(2)
    ]

    batch = compute_state_critic_step_signals(
        rollouts,
        critic=_FixedCritic([0.1, 10.0]),
        context_window_tokens=32768,
        normalize_advantages=True,
    )

    assert batch.mean_raw_advantage == pytest.approx(-5.05)
    assert batch.std_raw_advantage == pytest.approx(4.95)
    assert [
        signal.advantage
        for rollout_signals in batch.per_rollout_signals
        for signal in rollout_signals
    ] == pytest.approx([1.0, -1.0])
    assert batch.stats.mean_advantage == pytest.approx(0.0)
    assert batch.stats.std_advantage == pytest.approx(1.0)


def test_state_critic_starts_near_configured_value_prior() -> None:
    critic = StateCriticBaseline(
        game_embedding_buckets=16,
        game_embedding_dim=4,
        hidden_dim=8,
        learning_rate=0.0,
        seed=123,
        initial_value=1.0,
    )
    rollouts = [
        RolloutTrajectory(
            trajectory_id="traj-prior",
            task_id="pathfinder-0001",
            trajectory_reward=0.0,
            metadata={"game_total_levels": 2, "max_runtime_seconds": 300.0},
            steps=[
                RolloutStep(
                    step_index=0,
                    prompt_messages=[],
                    tool_schemas=[],
                    assistant_message={},
                    reward=0.0,
                    done=False,
                    metadata={
                        "stage": {
                            "levels_completed_before_decision": 0,
                            "levels_remaining": 2,
                            "time_remaining_seconds": 300.0,
                        }
                    },
                )
            ],
        )
    ]

    [prediction] = critic.predict(
        extract_state_critic_observations(rollouts, context_window_tokens=32768)
    )

    assert prediction == pytest.approx(1.0, abs=0.05)


def test_state_critic_uses_normalized_mse_and_reports_raw_error() -> None:
    critic = StateCriticBaseline(
        game_embedding_buckets=16,
        game_embedding_dim=4,
        hidden_dim=8,
        learning_rate=0.0,
        seed=123,
        initial_value=0.0,
        target_scale=100.0,
    )
    observation = extract_state_critic_observations(
        [
            RolloutTrajectory(
                trajectory_id="traj-normalized-target",
                task_id="pathfinder-0001",
                trajectory_reward=100.0,
                metadata={"game_total_levels": 1, "max_steps": 1},
                steps=[
                    RolloutStep(
                        step_index=0,
                        prompt_messages=[],
                        tool_schemas=[],
                        assistant_message={},
                        reward=100.0,
                        done=True,
                    )
                ],
            )
        ],
        context_window_tokens=32768,
    )

    stats = critic.update(
        observation,
        epochs=1,
        batch_size=1,
        max_grad_norm=10.0,
    )

    assert stats.loss == pytest.approx(1.0, abs=1e-3)
    assert stats.mean_prediction == pytest.approx(0.0, abs=0.05)
    assert stats.mean_target == pytest.approx(100.0)
    assert stats.mean_absolute_error == pytest.approx(100.0, abs=0.05)


def test_explained_variance_uses_raw_prediction_residuals() -> None:
    assert explained_variance([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]) == pytest.approx(0.75)
    assert explained_variance([2.0, 2.0], [2.0, 2.0]) == pytest.approx(0.0)


def test_state_critic_predictions_are_finite_with_step_budget_fallback() -> None:
    critic = StateCriticBaseline(
        game_embedding_buckets=16,
        game_embedding_dim=4,
        hidden_dim=8,
        learning_rate=0.0,
        seed=123,
    )
    rollouts = [
        RolloutTrajectory(
            trajectory_id=f"traj-time-{index}",
            task_id="pathfinder-0001",
            trajectory_reward=0.0,
            metadata={"game_total_levels": 2, "max_steps": 4},
            steps=[
                RolloutStep(
                    step_index=0,
                    prompt_messages=[],
                    tool_schemas=[],
                    assistant_message={},
                    reward=0.0,
                    done=False,
                    metadata={
                        "analysis_step": analysis_step,
                        "stage": {
                            "levels_completed_before_decision": 0,
                            "levels_remaining": 2,
                        },
                    },
                )
            ],
        )
        for index, analysis_step in enumerate((1, 4))
    ]
    extracted = extract_state_critic_observations(rollouts, context_window_tokens=32768)

    early_step, late_step = critic.predict(extracted)

    assert [observation.analysis_step for observation in extracted] == [1, 4]
    assert [observation.max_steps for observation in extracted] == [4, 4]
    assert math.isfinite(early_step)
    assert math.isfinite(late_step)


def test_state_critic_game_vocab_gives_collision_free_buckets() -> None:
    from training.state_critic import StateCriticBaseline

    games = [f"game-{index:04d}" for index in range(118)]
    critic = StateCriticBaseline(game_vocab=games, seed=3)

    buckets = [critic._game_bucket(game_id) for game_id in games]
    assert len(set(buckets)) == len(games)  # zero collisions among known games
    assert min(buckets) == 1  # row 0 reserved

    unknown = critic._game_bucket("never-seen-game")
    assert unknown > len(games)  # unknown ids land above the vocab region
    assert unknown < critic.game_embedding_buckets


def test_every_episode_end_is_terminal_with_zero_value() -> None:
    # The 30-minute budget is part of the evaluation objective, not a
    # collection artifact: at time_remaining=0 no further reward is possible,
    # so TIMEOUT is terminal exactly like game over -- the final GAE
    # transition uses V=0 and the critic is asked ONLY for the in-episode
    # observations (no post-horizon boundary prediction).
    class _RecordingCritic:
        def __init__(self, values_per_call):
            self.values_per_call = list(values_per_call)
            self.network = SimpleNamespace(policy_hidden_dim=0)
            self.calls = []

        def predict(self, observations):
            self.calls.append(list(observations))
            return self.values_per_call.pop(0)

    def rollout(
        trajectory_id: str, termination_reason: str, rewards: list[float]
    ) -> RolloutTrajectory:
        return RolloutTrajectory(
            trajectory_id=trajectory_id,
            task_id="pathfinder-0001",
            trajectory_reward=sum(rewards),
            termination_reason=termination_reason,
            metadata={"max_runtime_seconds": 1800.0},
            steps=[
                RolloutStep(
                    step_index=index,
                    prompt_messages=[],
                    tool_schemas=[],
                    assistant_message={},
                    reward=reward,
                    done=index == len(rewards) - 1,
                    metadata={
                        "stage": {
                            "levels_completed_before_decision": 1,
                            "levels_remaining": 4,
                        }
                    },
                )
                for index, reward in enumerate(rewards)
            ],
        )

    # One step, reward 20, V(s_0)=10: game over and the time horizon must
    # produce the IDENTICAL terminal target 20 = r + 0 and advantage 10,
    # from a single critic call.
    for termination_reason in ("COMPLETED_RUN", "TIMEOUT"):
        critic = _RecordingCritic([[10.0]])
        batch = compute_state_critic_step_signals(
            [rollout(f"traj-{termination_reason}", termination_reason, [20.0])],
            critic=critic,
            context_window_tokens=32768,
            normalize_advantages=False,
        )
        assert len(critic.calls) == 1  # no boundary prediction
        signal = batch.per_rollout_signals[0][0]
        assert signal.return_to_go == pytest.approx(20.0)
        assert signal.advantage == pytest.approx(10.0)

    # Multi-step propagation is unchanged: with gamma=1, lambda=1 and V=0
    # everywhere, targets are the reward-to-go from each step, terminating
    # at V=0 whether the episode timed out or finished.
    for termination_reason in ("COMPLETED_RUN", "TIMEOUT"):
        critic = _RecordingCritic([[0.0, 0.0, 0.0]])
        batch = compute_state_critic_step_signals(
            [
                rollout(
                    f"traj-multi-{termination_reason}",
                    termination_reason,
                    [0.0, 20.0, 5.0],
                )
            ],
            critic=critic,
            context_window_tokens=32768,
            normalize_advantages=False,
            gae_lambda=1.0,
        )
        assert len(critic.calls) == 1
        targets = [s.return_to_go for s in batch.per_rollout_signals[0]]
        assert targets == pytest.approx([25.0, 25.0, 5.0])
