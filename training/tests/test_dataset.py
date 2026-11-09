import pytest
import torch

from training.contracts import RolloutStep, RolloutTrajectory
from training.dataset import TokenizedSample, build_samples, collate_samples
from training.rewards import StepSignal


def _signals_for(
    rollout: RolloutTrajectory,
    *,
    reward: float = 1.0,
    return_to_go: float = 1.0,
    advantage: float = 0.0,
) -> list[list[StepSignal]]:
    return [
        [
            StepSignal(reward=reward, return_to_go=return_to_go, advantage=advantage)
            for _step in rollout.steps
        ]
    ]


def test_build_samples_uses_rollout_token_ids_and_logprobs() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="rollout-1",
        task_id="pathfinder-0001",
        trajectory_reward=1.0,
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "inspect"},
                ],
                tool_schemas=[],
                assistant_message={"role": "assistant", "content": "move"},
                reward=1.0,
                done=True,
                prompt_token_ids=[10, 11, 12],
                completion_token_ids=[13, 14],
                completion_logprobs=[
                    {"token_id": 13, "logprob": -0.25},
                    {"token_id": 14, "logprob": -0.75},
                ],
                metadata={"stage": {"levels_completed_before_decision": 2}},
            )
        ],
    )

    samples = build_samples([rollout], per_rollout_signals=_signals_for(rollout))

    assert samples == [
        TokenizedSample(
            trajectory_id="rollout-1",
            step_index=0,
            input_ids=[10, 11, 12, 13, 14],
            labels=[-100, -100, -100, 13, 14],
            attention_mask=[1, 1, 1, 1, 1],
            completion_mask=[0, 0, 0, 1, 1],
            reward=1.0,
            return_to_go=1.0,
            advantage=0.0,
            task_id="pathfinder-0001",
            level=3,
            prompt_message_count=2,
            prompt_token_count=3,
            old_logprob=-1.0,
            old_token_logprobs=[0.0, 0.0, -0.25, -0.75],
        )
    ]
    assert len(samples[0].input_ids) == len(samples[0].labels)
    assert len(samples[0].input_ids) == len(samples[0].attention_mask)
    assert len(samples[0].input_ids) == len(samples[0].completion_mask)
    assert len(samples[0].old_token_logprobs) == len(samples[0].input_ids) - 1


def test_build_samples_rejects_completion_tokens_without_prompt_tokens() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="rollout-1",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=True,
                prompt_token_ids=[],
                completion_token_ids=[21, 22, 23],
                completion_logprobs=[-0.1, -0.2, -0.3],
            )
        ],
    )

    with pytest.raises(ValueError, match="missing prompt token ids"):
        build_samples([rollout], per_rollout_signals=_signals_for(rollout))


def test_build_samples_skips_steps_without_completion_tokens() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="rollout-1",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=True,
                prompt_token_ids=[1, 2],
                completion_token_ids=[],
                completion_logprobs=[],
            )
        ],
    )

    assert (
        build_samples(
            [rollout],
            per_rollout_signals=_signals_for(
                rollout,
                reward=0.0,
                return_to_go=0.0,
                advantage=0.0,
            ),
        )
        == []
    )


def test_build_samples_rejects_rewarded_steps_without_completion_tokens() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="rollout-1",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=1.0,
                done=True,
                prompt_token_ids=[1, 2],
                completion_token_ids=[],
                completion_logprobs=[],
            )
        ],
    )

    with pytest.raises(ValueError, match="missing completion token ids"):
        build_samples(
            [rollout],
            per_rollout_signals=_signals_for(
                rollout,
                reward=1.0,
                return_to_go=1.0,
                advantage=1.0,
            ),
        )


def test_build_samples_rejects_action_steps_without_completion_tokens() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="rollout-1",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=True,
                prompt_token_ids=[1, 2],
                completion_token_ids=[],
                completion_logprobs=[],
                metadata={},
            )
        ],
    )

    # A step carrying training signal but no sampled tokens is corrupted
    # evidence, never a silently skipped sample.
    with pytest.raises(ValueError, match="missing completion token ids"):
        build_samples(
            [rollout],
            per_rollout_signals=_signals_for(
                rollout,
                reward=1.0,
                return_to_go=1.0,
                advantage=0.5,
            ),
        )


def test_build_samples_infers_level_from_levels_remaining() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="rollout-1",
        task_id="pathfinder-0001",
        metadata={"game_total_levels": 3},
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=True,
                prompt_token_ids=[1],
                completion_token_ids=[2],
                completion_logprobs=[-0.1],
                metadata={"stage": {"levels_remaining": 2}},
            )
        ],
    )

    assert (
        build_samples([rollout], per_rollout_signals=_signals_for(rollout))[0].level
        == 2
    )


def test_build_samples_requires_completion_logprobs_for_completion_tokens() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="rollout-1",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=True,
                prompt_token_ids=[1],
                completion_token_ids=[2],
                completion_logprobs=[],
            )
        ],
    )

    with pytest.raises(
        ValueError, match="completion logprobs must match completion token ids"
    ):
        build_samples([rollout], per_rollout_signals=_signals_for(rollout))


def test_build_samples_rejects_completion_logprob_token_id_mismatch() -> None:
    rollout = RolloutTrajectory(
        trajectory_id="rollout-1",
        task_id="pathfinder-0001",
        steps=[
            RolloutStep(
                step_index=0,
                prompt_messages=[],
                tool_schemas=[],
                assistant_message={},
                reward=0.0,
                done=True,
                prompt_token_ids=[1],
                completion_token_ids=[2, 3],
                completion_logprobs=[
                    {"token_id": 2, "logprob": -0.1},
                    {"token_id": 99, "logprob": -0.2},
                ],
            )
        ],
    )

    with pytest.raises(
        ValueError, match="logprob token ids must match completion token ids"
    ):
        build_samples([rollout], per_rollout_signals=_signals_for(rollout))


def test_collate_samples_preserves_rollout_logprobs() -> None:
    batch = collate_samples(
        [
            TokenizedSample(
                trajectory_id="rollout-1",
                step_index=0,
                input_ids=[1, 2, 3],
                labels=[-100, 2, 3],
                attention_mask=[1, 1, 1],
                completion_mask=[0, 1, 1],
                reward=1.0,
                return_to_go=1.0,
                advantage=0.25,
                old_logprob=-0.4,
                old_token_logprobs=[0.0, -0.1],
            ),
            TokenizedSample(
                trajectory_id="rollout-2",
                step_index=1,
                input_ids=[4, 5],
                labels=[-100, 5],
                attention_mask=[1, 1],
                completion_mask=[0, 1],
                reward=0.5,
                return_to_go=0.5,
                advantage=-0.25,
                old_logprob=-0.2,
                old_token_logprobs=[-0.2],
            ),
        ],
        pad_token_id=0,
    )

    assert torch.allclose(
        batch["old_logprobs"], torch.tensor([-0.4, -0.2], dtype=torch.float32)
    )
    assert torch.allclose(
        batch["old_token_logprobs"],
        torch.tensor([[0.0, -0.1], [-0.2, 0.0]], dtype=torch.float32),
    )


def test_collate_samples_pads_missing_token_logprobs_to_model_sequence_width() -> None:
    batch = collate_samples(
        [
            TokenizedSample(
                trajectory_id="rollout-1",
                step_index=0,
                input_ids=[1, 2, 3, 4],
                labels=[-100, 2, 3, 4],
                attention_mask=[1, 1, 1, 1],
                completion_mask=[0, 1, 1, 1],
                reward=1.0,
                return_to_go=1.0,
                advantage=0.5,
            ),
            TokenizedSample(
                trajectory_id="rollout-2",
                step_index=1,
                input_ids=[5, 6],
                labels=[-100, 6],
                attention_mask=[1, 1],
                completion_mask=[0, 1],
                reward=0.0,
                return_to_go=0.0,
                advantage=-0.5,
            ),
        ],
        pad_token_id=0,
    )

    assert batch["old_token_logprobs"].shape == (2, 3)
    assert torch.allclose(
        batch["old_token_logprobs"], torch.zeros((2, 3), dtype=torch.float32)
    )
