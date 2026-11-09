import torch
import torch.nn.functional as F

from training.contracts import RolloutStep, RolloutTrajectory
from training.dataset import TokenizedSample
import torch.nn

from training.ppo import summarize_token_usage, token_scores_from_hidden_states
from training.metrics import collection_scalars, summarize_rollouts


def test_summarize_rollouts_tracks_avg_return_and_lengths() -> None:
    rollout_a = RolloutTrajectory(
        trajectory_id="a",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=1.0,
                done=False,
                completion_token_ids=[10, 11],
            ),
            RolloutStep(
                step_index=1,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=2.0,
                done=True,
                completion_token_ids=[12],
            ),
        ],
        done=True,
        trajectory_reward=3.0,
        metadata={"levels_completed": 2, "run_total_actions": 4},
    )
    rollout_b = RolloutTrajectory(
        trajectory_id="b",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=-1.0,
                done=True,
                completion_token_ids=[13, 14, 15],
            ),
        ],
        done=False,
        trajectory_reward=-1.0,
        metadata={"levels_completed": 0, "run_total_actions": 2},
    )

    metrics = summarize_rollouts([rollout_a, rollout_b])

    assert metrics.avg_return == 1.0
    assert metrics.avg_levels_completed == 1.0
    assert metrics.max_levels_completed == 2.0
    assert metrics.return_std == 2.0
    assert metrics.min_return == -1.0
    assert metrics.max_return == 3.0
    assert metrics.avg_rollout_length == 1.5
    assert metrics.avg_actions_per_rollout == 3.0
    assert metrics.avg_generation_rounds_per_rollout == 1.5
    assert metrics.avg_tokens_per_generation == 2.0
    assert metrics.avg_step_reward == 2.0 / 3.0
    assert metrics.done_fraction == 0.5


def test_summarize_rollouts_exports_window_scalars() -> None:
    rollout_a = RolloutTrajectory(
        trajectory_id="a",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=1.0,
                done=True,
                completion_token_ids=[1, 2],
            ),
        ],
        done=True,
        trajectory_reward=1.0,
        metadata={"levels_completed": 1, "run_total_actions": 5},
    )
    rollout_b = RolloutTrajectory(
        trajectory_id="b",
        task_id="taps-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=2.0,
                done=True,
                completion_token_ids=[3, 4, 5],
            ),
            RolloutStep(
                step_index=1,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=1.0,
                done=True,
                completion_token_ids=[6],
            ),
        ],
        done=True,
        trajectory_reward=3.0,
        metadata={"levels_completed": 3, "run_total_actions": 1},
    )

    metrics = summarize_rollouts([rollout_a, rollout_b])

    scalars = metrics.as_scalars()
    assert scalars["rollout/avg_return"] == 2.0
    assert scalars["rollout/avg_levels_completed"] == 2.0
    assert scalars["rollout/max_levels_completed"] == 3.0
    assert scalars["rollout/return_std"] == 1.0
    assert scalars["rollout/avg_actions_per_rollout"] == 3.0
    assert scalars["rollout/avg_generation_rounds_per_rollout"] == 1.5
    assert scalars["rollout/avg_tokens_per_generation"] == 2.0
    assert not any(name.startswith("rollout_by_game/") for name in scalars)

    eval_scalars = metrics.as_scalars(scalar_prefix="eval")
    assert eval_scalars["eval/avg_return"] == 2.0
    assert eval_scalars["eval/avg_levels_completed"] == 2.0
    assert eval_scalars["eval/max_levels_completed"] == 3.0
    assert eval_scalars["eval/return_std"] == 1.0
    assert eval_scalars["eval/avg_actions_per_rollout"] == 3.0
    assert eval_scalars["eval/avg_generation_rounds_per_rollout"] == 1.5
    assert eval_scalars["eval/avg_tokens_per_generation"] == 2.0
    assert not any(name.startswith("eval_by_game/") for name in eval_scalars)


def test_summarize_rollouts_reports_levels_pct_and_arc_score() -> None:
    def _rollout(tid, levels, total, arc):
        return RolloutTrajectory(
            trajectory_id=tid,
            task_id="g",
            steps=[
                RolloutStep(
                    step_index=0,
                    prompt_messages=[],
                    tool_schemas=[],
                    assistant_message={},
                    reward=0.0,
                    done=True,
                )
            ],
            done=True,
            trajectory_reward=0.0,
            metadata={
                "levels_completed": levels,
                "game_total_levels": total,
                "final_arc_score": arc,
            },
        )

    metrics = summarize_rollouts([_rollout("a", 1, 4, 80.0), _rollout("b", 2, 4, 20.0)])

    # levels pct: (25% + 50%) / 2 = 37.5 ; arc score: (80 + 20) / 2 = 50
    assert metrics.avg_levels_completed_pct == 37.5
    assert metrics.avg_arc_score == 50.0
    scalars = metrics.as_scalars(scalar_prefix="eval")
    assert scalars["eval/avg_levels_completed_pct"] == 37.5
    assert scalars["eval/avg_arc_score"] == 50.0


def test_summarize_rollouts_counts_distinct_games_with_nonzero_return() -> None:
    def _rollout(trajectory_id, task_id, reward):
        return RolloutTrajectory(
            trajectory_id=trajectory_id,
            task_id=task_id,
            steps=[
                RolloutStep(
                    step_index=0,
                    prompt_messages=[],
                    tool_schemas=[],
                    assistant_message={},
                    reward=0.0,
                    done=True,
                )
            ],
            done=True,
            trajectory_reward=reward,
        )

    metrics = summarize_rollouts(
        [
            _rollout("a1", "game-a", 12.0),
            _rollout("a2", "game-a", 0.0),
            _rollout("b1", "game-b", 0.0),
            _rollout("c1", "game-c", 5.0),
        ]
    )

    # game-a (one non-zero rollout) and game-c count once each; game-b is all-zero.
    assert metrics.games_nonzero_return == 2.0
    assert metrics.as_scalars()["rollout/games_nonzero_return"] == 2.0


def test_summarize_token_usage_tracks_rollout_and_env_step_averages() -> None:
    rollout_a = RolloutTrajectory(
        trajectory_id="a",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=1.0,
                done=False,
            ),
            RolloutStep(
                step_index=1,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=True,
            ),
        ],
        metadata={"run_total_actions": 4},
    )
    rollout_b = RolloutTrajectory(
        trajectory_id="b",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=True,
            ),
        ],
        metadata={"run_total_actions": 2},
    )
    samples = [
        TokenizedSample(
            trajectory_id="a",
            step_index=0,
            input_ids=[1, 2, 3],
            labels=[-100, 2, 3],
            attention_mask=[1, 1, 1],
            completion_mask=[0, 1, 1],
            reward=1.0,
            return_to_go=1.0,
            advantage=1.0,
        ),
        TokenizedSample(
            trajectory_id="a",
            step_index=1,
            input_ids=[1, 2],
            labels=[-100, 2],
            attention_mask=[1, 1],
            completion_mask=[0, 1],
            reward=0.0,
            return_to_go=0.0,
            advantage=0.0,
        ),
        TokenizedSample(
            trajectory_id="b",
            step_index=0,
            input_ids=[1, 2, 3, 4],
            labels=[-100, 2, 3, 4],
            attention_mask=[1, 1, 1, 1],
            completion_mask=[0, 1, 1, 1],
            reward=0.0,
            return_to_go=0.0,
            advantage=-1.0,
        ),
    ]

    for rollout in (rollout_a, rollout_b):
        for step in rollout.steps:
            sample = next(
                s
                for s in samples
                if s.trajectory_id == rollout.trajectory_id
                and s.step_index == step.step_index
            )
            step.prompt_token_ids = sample.input_ids[:1]
            step.completion_token_ids = sample.input_ids[1:]
    usage = summarize_token_usage([rollout_a, rollout_b], samples)

    assert usage.total_completion_tokens == 6
    assert usage.total_env_steps == 6
    # Completion tokens per rollout.
    assert usage.avg_tokens_per_rollout == 3.0
    assert usage.avg_tokens_per_step == 1.0
    assert usage.avg_tokens_per_model_turn == 2.0


def test_summarize_token_usage_honors_explicit_zero_action_counts() -> None:
    rollout_with_run_count = RolloutTrajectory(
        trajectory_id="run-count-zero",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=False,
            )
        ],
        metadata={"run_total_actions": 0},
    )
    rollout_with_step_count = RolloutTrajectory(
        trajectory_id="step-count-zero",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=False,
                metadata={"env": {"executed_count": 0}},
            )
        ],
    )
    samples = [
        TokenizedSample(
            trajectory_id="run-count-zero",
            step_index=0,
            input_ids=[1, 2, 3],
            labels=[-100, 2, 3],
            attention_mask=[1, 1, 1],
            completion_mask=[0, 1, 1],
            reward=0.0,
            return_to_go=0.0,
            advantage=0.0,
        )
    ]

    rollout_with_run_count.steps[0].prompt_token_ids = [1]
    rollout_with_run_count.steps[0].completion_token_ids = [2, 3]
    usage = summarize_token_usage(
        [rollout_with_run_count, rollout_with_step_count],
        samples,
    )

    assert usage.total_completion_tokens == 2
    assert usage.total_env_steps == 0
    assert usage.avg_tokens_per_rollout == 1.0
    assert usage.avg_tokens_per_step == 0.0
    assert usage.avg_tokens_per_model_turn == 1.0


def test_token_scores_from_hidden_states_matches_log_softmax_reference() -> None:
    logits = torch.tensor(
        [
            [
                [1.0, 0.0, -1.0],
                [0.2, 0.3, 0.5],
                [0.7, -0.4, 0.1],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    target_ids = torch.tensor([[1, 2, 0]], dtype=torch.long)
    target_mask = torch.tensor([[False, True, True]])

    actual, _entropies = token_scores_from_hidden_states(
        logits,
        lm_head=torch.nn.Identity(),
        target_ids=target_ids,
        target_mask=target_mask,
        compute_entropies=False,
    )
    actual = actual.sum(dim=-1)

    reference = (
        F.log_softmax(logits[:, :-1, :], dim=-1)
        .gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    reference = torch.where(target_mask, reference, torch.zeros_like(reference))
    reference = reference.sum(dim=-1)

    assert torch.allclose(actual, reference)


def test_collection_scalars_use_raw_tokens_usage_and_reported_compactions():
    def step(index, *, segment=False, void=False, captured=True):
        return RolloutStep(
            index,
            [],
            [],
            {},
            0.0,
            False,
            prompt_token_ids=[1, 2, 3] if captured else [],
            completion_token_ids=[4, 5] if captured else [],
            metadata={
                "segment_start": segment,
                "void_attempt": void,
                "usage": {"prompt_tokens": 30, "completion_tokens": 20},
            },
        )

    rollouts = [
        RolloutTrajectory(
            "a",
            "g",
            [
                step(0, segment=True),
                step(1),
                step(2, segment=True),
                step(3, segment=True),
                step(4, segment=True, void=True),
            ],
        ),
        RolloutTrajectory("b", "g", [step(0, segment=True, captured=False)]),
        RolloutTrajectory("empty", "g", []),
    ]
    scalars = collection_scalars(rollouts, 10.0)
    assert scalars == {
        # 10 captured completion tokens in "a", 20 from "b"'s usage counts and
        # none in "empty": 30 over three rollouts.
        "rollout/completion_tokens_per_rollout": 10.0,
        "rollout/tokens_per_s": 3.0,
        "rollout/prompt_tokens_per_s": 4.5,
        "rollout/compactions_per_rollout": 2 / 3,
    }
    assert collection_scalars(rollouts, 10.0, scalar_prefix="eval") == {
        "eval/tokens_per_s": 3.0,
        "eval/compactions_per_rollout": 2 / 3,
    }
    assert collection_scalars(rollouts, 0.0)["rollout/tokens_per_s"] == 0.0
    assert collection_scalars(rollouts, -1.0)["rollout/prompt_tokens_per_s"] == 0.0
    assert all(value == 0 for value in collection_scalars([], 1).values())
    assert (
        collection_scalars(
            [RolloutTrajectory("no-compaction", "g", [step(0), step(1)])], 1
        )["rollout/compactions_per_rollout"]
        == 0
    )
