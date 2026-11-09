import math


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
        from types import SimpleNamespace

        self.network = SimpleNamespace(policy_hidden_dim=0)
        self.predictions = predictions

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
        from types import SimpleNamespace

        network = SimpleNamespace(policy_hidden_dim=0)

        def __init__(self, values_per_call):
            self.values_per_call = list(values_per_call)
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


def test_state_critic_rejects_checkpoints_from_before_the_level_embedding() -> None:
    critic = StateCriticBaseline(
        game_embedding_buckets=16,
        game_embedding_dim=4,
        hidden_dim=8,
        learning_rate=0.001,
        seed=123,
        initial_value=5.0,
        target_scale=1.0,
    )
    stale = critic.state_dict()
    stale["version"] = 4
    with pytest.raises(
        RuntimeError, match="Unsupported state critic checkpoint version 4"
    ):
        critic.load_state_dict(stale)


def test_critic_context_feature_uses_only_the_pre_action_prompt() -> None:
    from dataclasses import replace

    step = RolloutStep(
        step_index=0,
        prompt_messages=[],
        tool_schemas=[],
        assistant_message={},
        prompt_token_ids=[1] * 20,
        completion_token_ids=[2],
        reward=0.0,
        done=True,
    )
    rollout = RolloutTrajectory(trajectory_id="short", task_id="game", steps=[step])
    longer = replace(
        rollout,
        trajectory_id="long",
        steps=[replace(step, completion_token_ids=[2] * 50)],
    )
    observations = extract_state_critic_observations(
        [rollout, longer], context_window_tokens=100
    )
    assert [o.context_remaining_fraction for o in observations] == pytest.approx(
        [0.8, 0.8]
    )


def test_critic_prediction_and_gradient_use_the_level_embedding() -> None:
    import torch
    from dataclasses import replace
    from training.state_critic import StateCriticObservation

    first = StateCriticObservation(
        rollout_index=0,
        step_index=0,
        game_id="game",
        reward=0.0,
        score_so_far=0.0,
        return_to_go=0.0,
        levels_remaining=1.0,
        total_levels=1.0,
        analysis_step=1,
        max_steps=10,
        level_index=0,
    )
    critic = StateCriticBaseline(seed=7, hidden_dim=8)
    tensors = critic._tensorize([first, replace(first, level_index=1)])
    predictions = critic._predict_tensor(tensors)
    assert not torch.isclose(predictions[0], predictions[1], atol=1e-9)
    predictions.sum().backward()
    gradients = critic.network.level_embedding.weight.grad
    assert gradients is not None
    assert gradients[0].abs().sum() > 0
    assert gradients[1].abs().sum() > 0
    assert gradients[2:].abs().sum() == 0


def _observation(index, *, rollout_index=0, reward=0.0, tokens=1000, hidden=None):
    from training.state_critic import StateCriticObservation

    return StateCriticObservation(
        rollout_index=rollout_index,
        step_index=index,
        game_id="game",
        reward=reward,
        score_so_far=0.0,
        return_to_go=reward,
        levels_remaining=1,
        total_levels=3,
        analysis_step=index + 1,
        max_steps=100,
        discount_token_count=tokens,
        policy_hidden=hidden,
    )


def test_gamma_lambda_one_preserve_late_rewards_across_long_summaries_and_deadline():
    from training.state_critic import _gae_returns_and_advantages

    observations = [
        _observation(0, reward=10, tokens=3000),
        _observation(1, tokens=1000),  # summary before reset
        _observation(2, reward=20, tokens=4000),
        _observation(3, tokens=1000),
    ]  # captured but unprocessed terminal draw
    observations[-1].void_attempt = True
    values = [2.0, 3.0, 4.0, 9.0]
    targets, advantages = _gae_returns_and_advantages(
        observations, values, gamma=1, gae_lambda=1
    )
    assert targets == pytest.approx([30, 20, 20, 0])
    assert advantages == pytest.approx([28, 17, 16, -9])
    # Lambda is applied once per MODEL RESPONSE, not once per token.
    short = [_observation(0, tokens=10000), _observation(1, reward=100)]
    _, advantages = _gae_returns_and_advantages(short, [0, 0], gamma=1, gae_lambda=0.95)
    assert advantages == pytest.approx([95, 100])


def test_global_token_normalization_is_unchanged_by_splitting_a_decision_span():
    from training.state_critic import _normalize_and_clip_advantages

    observations = [
        _observation(0, tokens=1),
        _observation(1, tokens=9),
        _observation(0, rollout_index=1, tokens=2),
    ]
    options = dict(normalize_advantages=True, advantage_clip=0, epsilon=1e-8)
    advantages, mean, std = _normalize_and_clip_advantages(
        [0, 10, 5], observations, **options
    )
    assert mean == pytest.approx(100 / 12)
    assert std == pytest.approx(math.sqrt(950 / 12 - (100 / 12) ** 2))
    split = [
        observations[0],
        _observation(1, tokens=3),
        _observation(2, tokens=6),
        observations[2],
    ]
    other, other_mean, other_std = _normalize_and_clip_advantages(
        [0, 10, 10, 5], split, **options
    )
    assert (other_mean, other_std) == pytest.approx((mean, std))
    assert other == pytest.approx(
        [advantages[0], advantages[1], advantages[1], advantages[2]]
    )


def test_hidden_critic_missing_features_are_fatal_and_update_uses_terminal_draw():
    import torch
    from training.state_critic import StateCriticBaseline

    critic = StateCriticBaseline(
        game_embedding_buckets=8,
        game_embedding_dim=2,
        hidden_dim=8,
        policy_hidden_dim=4,
        policy_hidden_projection_dim=4,
        learning_rate=0.01,
        initial_value=0,
        seed=3,
    )
    observations = [
        _observation(0, reward=100, hidden=[1, 2, 3, 4]),
        _observation(1, reward=0, hidden=[4, 3, 2, 1]),
    ]
    observations[-1].void_attempt = True
    before = {name: t.clone() for name, t in critic.network.state_dict().items()}
    result = critic.update(observations, epochs=2, batch_size=1, max_grad_norm=10)
    assert result.update_steps == 4  # final deadline target is fitted too
    assert any(
        not torch.equal(before[name], t)
        for name, t in critic.network.state_dict().items()
    )
    for hidden in (None, [1], [1, 2, float("nan"), 4]):
        observations[0].policy_hidden = hidden
        with pytest.raises(RuntimeError, match="hidden"):
            critic.predict(observations)


def test_critic_planner_verifies_complete_prefix_and_splits_only_declared_resets():
    from training.ppo import plan_policy_hidden_segments

    def step(i, prompt, output, reset=False):
        return RolloutStep(
            i, [], [], {}, 0, False, prompt, output, metadata={"segment_start": reset}
        )

    rollout = RolloutTrajectory(
        "r",
        "g",
        [
            step(0, [1, 2], [3]),
            step(1, [1, 2, 3, 4], [5, 6]),
            step(2, [7, 8], [9], True),
            step(3, [7, 8, 9, 10], [11]),
        ],
    )
    assert plan_policy_hidden_segments([rollout]) == [
        (0, [0, 1], [1, 3], [1, 2, 3, 4]),
        (0, [2, 3], [1, 3], [7, 8, 9, 10]),
    ]
    rollout.steps[1].prompt_token_ids[0] = 99
    with pytest.raises(RuntimeError, match="rewrite"):
        plan_policy_hidden_segments([rollout])


def test_hidden_features_are_exact_detached_last_prompt_states_across_compaction():
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch
    from training.ppo import PPOTrainer

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(max_sequence_tokens=64, max_micro_batch_tokens=128)
    trainer.device = torch.device("cpu")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._set_policy_mode_for_logprob_computation = lambda: None
    trainer._autocast_context = nullcontext
    calls = []

    class Decoder:
        def __call__(self, input_ids, attention_mask):
            calls.append(input_ids.tolist())
            # Causal toy decoder: each hidden feature is the prefix sum.
            states = input_ids.cumsum(-1).float().unsqueeze(-1)
            return SimpleNamespace(last_hidden_state=states.expand(-1, -1, 4))

    trainer._decoder_and_lm_head = lambda: (Decoder(), None)
    steps = [
        RolloutStep(0, [], [], {}, 0, False, [1, 2], [3]),
        RolloutStep(1, [], [], {}, 0, False, [1, 2, 3, 4], [5]),
        RolloutStep(
            2, [], [], {}, 1, True, [8, 9], [10], metadata={"segment_start": True}
        ),
    ]
    actual = trainer._extract_policy_hidden_states([RolloutTrajectory("r", "g", steps)])
    assert actual == {(0, 0): [3.0] * 4, (0, 1): [10.0] * 4, (0, 2): [17.0] * 4}
    assert calls == [[[1, 2, 3, 4], [8, 9, 0, 0]]]


@pytest.mark.parametrize("stop_at", ["before", "first", "last"])
def test_hidden_features_honor_shutdown_between_rank_zero_forwards(stop_at):
    from contextlib import nullcontext
    import signal
    from types import SimpleNamespace
    import torch
    from training.ppo import PPOTrainer
    from training.shutdown import TrainingShutdown, training_sigterm_handler

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(max_sequence_tokens=2, max_micro_batch_tokens=2)
    trainer.device = torch.device("cpu")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._set_policy_mode_for_logprob_computation = lambda: None
    trainer._autocast_context = nullcontext
    calls = []

    class Decoder:
        def __call__(self, input_ids, attention_mask):
            calls.append(input_ids.tolist())
            if (stop_at == "first" and len(calls) == 1) or (
                stop_at == "last" and len(calls) == 3
            ):
                signal.raise_signal(signal.SIGTERM)
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))

    trainer._decoder_and_lm_head = lambda: (Decoder(), None)
    rollouts = [
        RolloutTrajectory(
            str(i), "g", [RolloutStep(0, [], [], {}, 0, True, [1, 2], [3])]
        )
        for i in range(3)
    ]
    with training_sigterm_handler(), pytest.raises(TrainingShutdown):
        if stop_at == "before":
            signal.raise_signal(signal.SIGTERM)
        trainer._extract_policy_hidden_states(rollouts)
    assert len(calls) == {"before": 0, "first": 1, "last": 3}[stop_at]


@pytest.mark.parametrize("target_normalization", ["fixed", "per_game"])
def test_hidden_critic_checkpoint_restores_head_optimizer_and_rejects_ema(
    monkeypatch, tmp_path, target_normalization
):
    import torch
    from test_ppo import _make_runtime_export_test_trainer

    def trainer():
        value = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
        value.config.value_baseline = "hidden_critic"
        value.config.state_critic_target_normalization = target_normalization
        value.level_baseline = None
        value.state_critic = StateCriticBaseline(
            game_embedding_buckets=8,
            game_embedding_dim=2,
            hidden_dim=8,
            policy_hidden_dim=4,
            policy_hidden_projection_dim=4,
            initial_value=0,
            game_vocab=["game"],
            target_normalization=target_normalization,
            normalization_min_rollouts=2,
        )
        return value

    original = trainer()
    observations = [_observation(0, reward=100, hidden=[1, 2, 3, 4])]
    original.state_critic.update(observations, epochs=2, batch_size=1, max_grad_norm=10)
    payload = original._training_state_payload(
        iteration_index=2,
        adapter_dir=tmp_path / "a",
        runtime_adapter_dir=tmp_path / "a",
    )
    assert "level_baseline_state" not in payload
    assert payload["training_config"]["value_baseline"] == "hidden_critic"
    path = tmp_path / "state.pt"
    torch.save(payload, path)
    restored = trainer()
    restored.load_training_state(path)
    assert restored.state_critic.predict(observations) == pytest.approx(
        original.state_critic.predict(observations)
    )
    assert (
        restored.state_critic.optimizer.state_dict()["state"].keys()
        == original.state_critic.optimizer.state_dict()["state"].keys()
    )
    # Matching predictions alone would miss lost Adam moments: the next
    # update must also be identical after restoring the full critic state.
    rng = torch.get_rng_state()
    original.state_critic.update(observations, epochs=2, batch_size=1, max_grad_norm=10)
    torch.set_rng_state(rng)
    restored.state_critic.update(observations, epochs=2, batch_size=1, max_grad_norm=10)
    for key, expected in original.state_critic.network.state_dict().items():
        assert torch.equal(restored.state_critic.network.state_dict()[key], expected)
    del payload["state_critic_state"]
    payload["level_baseline_state"] = {}
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="fresh hidden critic"):
        restored.load_training_state(path)


def test_opt_in_trainer_uses_hidden_critic_signals_and_reports_actual_fit_time(
    monkeypatch,
):
    from test_ppo import _make_tiny_ppo_trainer
    from training.state_critic import StateCriticSignalBatch, StateCriticUpdateStats

    trainer = _make_tiny_ppo_trainer(advantage=1)
    trainer.config.value_baseline = "hidden_critic"
    trainer.level_baseline = None
    trainer.state_critic = StateCriticBaseline(
        game_embedding_buckets=8,
        game_embedding_dim=2,
        hidden_dim=8,
        policy_hidden_dim=4,
        policy_hidden_projection_dim=4,
        initial_value=0,
    )
    rollouts = [
        RolloutTrajectory(
            name,
            "game",
            [RolloutStep(0, [], [], {}, reward, True, [1, 2], [3, 4], [-0.1, -0.2])],
            trajectory_reward=reward,
        )
        for name, reward in (("failure", 0), ("success", 100))
    ]
    monkeypatch.setattr(
        trainer,
        "_extract_policy_hidden_states",
        lambda _: {(0, 0): [1, 2, 3, 4], (1, 0): [4, 3, 2, 1]},
    )
    signals, stats, batch = trainer._compute_step_signals(rollouts)
    assert isinstance(batch, StateCriticSignalBatch)
    assert [per_episode[0].return_to_go for per_episode in signals] == pytest.approx(
        [0, 100]
    )
    assert [per_episode[0].advantage for per_episode in signals] == pytest.approx(
        [-1, 1]
    )
    assert stats.mean_advantage == pytest.approx(0)
    assert trainer._baseline_feature_seconds >= 0
    update, seconds = trainer._update_baseline(batch)
    assert isinstance(update, StateCriticUpdateStats)
    assert update.update_steps == trainer.config.state_critic_epochs_per_window
    assert update.loss > 0 and seconds > 0


def test_zero_prior_has_no_initial_actor_noise_but_learns_hidden_features():
    import torch

    critic = StateCriticBaseline(
        game_embedding_buckets=8,
        game_embedding_dim=2,
        hidden_dim=8,
        policy_hidden_dim=4,
        policy_hidden_projection_dim=4,
        initial_value=0,
        learning_rate=0.01,
    )
    rollouts = [
        RolloutTrajectory(str(i), "g", [RolloutStep(0, [], [], {}, 0, True, [1], [2])])
        for i in range(2)
    ]
    hidden = {(0, 0): [1, 2, 3, 4], (1, 0): [4, 3, 2, 1]}
    batch = compute_state_critic_step_signals(
        rollouts, context_window_tokens=32, critic=critic, policy_hidden_by_step=hidden
    )
    assert batch.mean_value == batch.std_raw_advantage == 0
    assert [s[0].advantage for s in batch.per_rollout_signals] == [0, 0]
    before = {name: t.clone() for name, t in critic.network.state_dict().items()}
    batch.observations[1].return_to_go = 100
    critic.update(batch.observations, epochs=3, batch_size=2, max_grad_norm=10)
    assert any(
        not torch.equal(t, before[name])
        for name, t in critic.network.state_dict().items()
        if name.startswith("policy_hidden_projection.")
    )
    assert critic.predict(batch.observations) != [0, 0]


def test_zero_reward_window_retains_learned_failure_advantages():
    # A trained value estimate can distinguish failed states. Do not suppress
    # its residuals merely because no reward arrived in this particular window.
    rollouts = [
        RolloutTrajectory(str(i), "g", [RolloutStep(0, [], [], {}, 0, True, [1], [2])])
        for i in range(2)
    ]
    batch = compute_state_critic_step_signals(
        rollouts, context_window_tokens=32, critic=_FixedCritic([10, 20])
    )
    assert [s[0].advantage for s in batch.per_rollout_signals] == pytest.approx([1, -1])


@pytest.mark.parametrize("prediction", [math.nan, math.inf, -math.inf])
def test_nonfinite_critic_residual_is_fatal_instead_of_normalized_to_zero(prediction):
    rollout = RolloutTrajectory(
        "r", "g", [RolloutStep(0, [], [], {}, 0, True, [1], [2])]
    )
    with pytest.raises(RuntimeError, match="nonfinite advantage"):
        compute_state_critic_step_signals(
            [rollout], context_window_tokens=32, critic=_FixedCritic([prediction])
        )


def test_nonfinite_critic_loss_is_fatal_before_optimizer_mutation():
    import torch

    critic = StateCriticBaseline(
        hidden_dim=4, game_embedding_buckets=8, initial_value=0
    )
    before = {key: value.clone() for key, value in critic.network.state_dict().items()}
    # The score is finite, but squaring its float32 normalized target overflows.
    with pytest.raises(RuntimeError, match="nonfinite training loss"):
        critic.update(
            [_observation(0, reward=1e30)], epochs=1, batch_size=1, max_grad_norm=10
        )
    assert not critic.optimizer.state
    for key, value in critic.network.state_dict().items():
        assert torch.equal(value, before[key])


def test_nonfinite_critic_predictions_are_rejected_at_inference():
    import torch

    critic = StateCriticBaseline(
        hidden_dim=4, game_embedding_buckets=8, initial_value=0
    )
    with torch.no_grad():
        critic.network.base[-1].bias.fill_(math.nan)
    with pytest.raises(RuntimeError, match="nonfinite value prediction"):
        critic.predict([_observation(0)])
