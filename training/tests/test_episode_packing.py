import pytest
import torch

from training.dataset import TokenizedSample, collate_samples
from training.episode_packing import assign_loss_weights, pack_episode_samples


def _step_sample(
    trajectory_id: str,
    step_index: int,
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    advantage: float,
    logprob_base: float = -1.0,
) -> TokenizedSample:
    input_ids = prompt_ids + completion_ids
    prompt_count = len(prompt_ids)
    target_len = len(input_ids) - 1
    old_token_logprobs = [0.0] * target_len
    for offset in range(len(completion_ids)):
        old_token_logprobs[prompt_count - 1 + offset] = logprob_base - offset
    return TokenizedSample(
        trajectory_id=trajectory_id,
        step_index=step_index,
        input_ids=input_ids,
        labels=([-100] * prompt_count) + completion_ids,
        attention_mask=[1] * len(input_ids),
        completion_mask=([0] * prompt_count) + ([1] * len(completion_ids)),
        reward=0.0,
        return_to_go=0.0,
        advantage=advantage,
        prompt_token_count=prompt_count,
        old_logprob=float(sum(old_token_logprobs)),
        old_token_logprobs=old_token_logprobs,
    )


def _nested_episode(trajectory_id: str = "t0") -> list[TokenizedSample]:
    # step 0: prompt [1,2,3], completion [4,5]
    s0 = _step_sample(trajectory_id, 0, [1, 2, 3], [4, 5], advantage=1.5)
    # env appends [6]; step 1 prompt = s0.input_ids + [6], completion [7,8,9]
    s1 = _step_sample(
        trajectory_id,
        1,
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9],
        advantage=-0.5,
        logprob_base=-2.0,
    )
    # env appends [10,11]; step 2 completion [12]
    s2 = _step_sample(
        trajectory_id,
        2,
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [12],
        advantage=3.0,
        logprob_base=-3.0,
    )
    return [s0, s1, s2]


@pytest.mark.parametrize("aggregation", ["token_mean", "span_mean"])
def test_unpacked_weights_follow_the_selected_objective(aggregation: str) -> None:
    long = _nested_episode("long")
    short = _step_sample("short", 0, [1], [2], advantage=1.0)
    assign_loss_weights([*long, short], loss_aggregation=aggregation)
    assert sum(sample.sequence_weight for sample in long) == pytest.approx(
        6.0 if aggregation == "token_mean" else 1.0
    )
    assert short.sequence_weight == 1.0
    expected = [2, 3, 1] if aggregation == "token_mean" else [1 / 3] * 3
    assert [sample.sequence_weight for sample in long] == pytest.approx(expected)


def test_packs_nested_steps_into_one_sequence_with_exact_alignment() -> None:
    samples = _nested_episode()
    packed, stats = pack_episode_samples(samples, max_sequence_tokens=64)

    assert stats.input_samples == 3
    assert stats.packed_sequences == 1
    assert stats.packed_spans == 3
    assert stats.passthrough_samples == 0
    assert len(packed) == 1

    p = packed[0]
    assert p.input_ids == samples[-1].input_ids
    assert p.span_count == 3
    # Span positions: completions at [3,5), [6,9), [11,12)
    assert [i for i, m in enumerate(p.completion_mask) if m] == [3, 4, 6, 7, 8, 11]
    # Target-aligned span ids at positions-1
    assert [p.span_ids[i] for i in (2, 3, 5, 6, 7, 10)] == [1, 1, 2, 2, 2, 3]
    assert all(
        p.span_ids[i] == 0
        for i in range(len(p.span_ids))
        if i not in (2, 3, 5, 6, 7, 10)
    )
    # Advantages per span
    assert [p.token_advantages[i] for i in (2, 3, 5, 6, 7, 10)] == [
        1.5,
        1.5,
        -0.5,
        -0.5,
        -0.5,
        3.0,
    ]
    # Old logprobs land at identical target positions as the originals
    for source in samples:
        start = source.prompt_token_count - 1
        end = len(source.input_ids) - 1
        assert p.old_token_logprobs[start:end] == source.old_token_logprobs[start:end]
    # Labels mark exactly the span tokens
    assert [i for i, label in enumerate(p.labels) if label != -100] == [
        3,
        4,
        6,
        7,
        8,
        11,
    ]


def test_nesting_break_raises() -> None:
    samples = _nested_episode()
    # Corrupt step 2's prompt so it no longer nests (history-rewrite analogue).
    # Token-continuation rollouts make nesting an invariant, so this is fatal.
    samples[2].input_ids[0] = 99
    with pytest.raises(RuntimeError, match="does not extend the previous sequence"):
        pack_episode_samples(samples, max_sequence_tokens=64)


def test_compaction_boundary_starts_new_segment() -> None:
    # One episode, one compaction: steps 0-1 nest (segment A); step 2 is flagged
    # segment_start with a FRESH compacted prefix that does NOT nest into step 1.
    s0 = _step_sample("t0", 0, [1, 2, 3], [4, 5], advantage=1.5)
    s1 = _step_sample(
        "t0", 1, [1, 2, 3, 4, 5, 6], [7, 8, 9], advantage=-0.5, logprob_base=-2.0
    )
    # Compaction: the runner reset the buffer to a compacted 16k-analogue prefix
    # [90,91,92,93] (unrelated to the pre-compaction tokens), then generated [94].
    s2 = _step_sample("t0", 2, [90, 91, 92, 93], [94], advantage=3.0, logprob_base=-3.0)
    s2.segment_start = True
    s3 = _step_sample(
        "t0", 3, [90, 91, 92, 93, 94, 95], [96, 97], advantage=2.0, logprob_base=-4.0
    )

    packed, stats = pack_episode_samples([s0, s1, s2, s3], max_sequence_tokens=64)

    assert stats.segment_boundaries == 1
    assert stats.nesting_breaks == 0
    assert len(packed) == 2  # two segments
    segA, segB = packed
    # Segment A packs steps 0-1 (2 spans); segment B packs steps 2-3 (2 spans).
    assert segA.span_count == 2 and segB.span_count == 2
    assert segA.input_ids == s1.input_ids  # last step of the segment
    assert segB.input_ids == s3.input_ids
    # Segment B masks the compacted prefix (its assistant tokens are prompt now):
    # completions are the NEW generations only — positions 4 ([94]) and 6,7 ([96,97]).
    assert [i for i, m in enumerate(segB.completion_mask) if m] == [4, 6, 7]
    # Episode-equal weighting sums to 1.0 across the two segments (2/4 + 2/4).
    assert segA.sequence_weight + segB.sequence_weight == pytest.approx(1.0)
    assert segA.sequence_weight == pytest.approx(0.5)


def test_unflagged_nesting_break_still_raises() -> None:
    # A non-nesting step WITHOUT segment_start is a real fork -> fatal.
    s0 = _step_sample("t0", 0, [1, 2, 3], [4, 5], advantage=1.0)
    s1 = _step_sample(
        "t0", 1, [90, 91, 92], [93], advantage=1.0
    )  # does not nest, not flagged
    with pytest.raises(RuntimeError, match="does not extend the previous sequence"):
        pack_episode_samples([s0, s1], max_sequence_tokens=64)


def test_length_budget_overflow_raises() -> None:
    samples = _nested_episode()
    with pytest.raises(RuntimeError, match="exceeds max_sequence_tokens"):
        pack_episode_samples(samples, max_sequence_tokens=9)


def test_different_trajectories_never_pack() -> None:
    a = _step_sample("a", 0, [1, 2], [3], advantage=1.0)
    b = _step_sample("b", 0, [1, 2, 3], [4], advantage=1.0)
    packed, stats = pack_episode_samples([a, b], max_sequence_tokens=64)
    assert len(packed) == 2 and stats.packed_sequences == 0


def test_collate_packed_matches_per_step_target_data() -> None:
    """The packed batch must expose, per target position of the shared
    sequence, exactly the (old_logprob, advantage, span membership) that the
    per-step samples carried — the structural core of exactness."""
    samples = _nested_episode()
    packed, _ = pack_episode_samples(samples, max_sequence_tokens=64)
    batch = collate_samples(packed, pad_token_id=0)

    assert int(batch["span_counts"][0].item()) == 3
    span_ids = batch["span_ids"][0]
    for source, expected_span in zip(samples, (1, 2, 3)):
        start = source.prompt_token_count - 1
        end = len(source.input_ids) - 1
        assert torch.all(span_ids[start:end] == expected_span)
        assert torch.allclose(
            batch["old_token_logprobs"][0][start:end],
            torch.tensor(source.old_token_logprobs[start:end]),
        )
        assert torch.allclose(
            batch["token_advantages"][0][start:end],
            torch.full((end - start,), source.advantage),
        )


def test_episode_equal_weights_are_one_for_whole_episodes() -> None:
    nested = _nested_episode("t0")
    single = [_step_sample("t1", 0, [1, 2], [3], advantage=1.0)]

    packed, stats = pack_episode_samples(nested + single, max_sequence_tokens=4096)

    by_traj: dict[str, list[TokenizedSample]] = {}
    for sample in packed:
        by_traj.setdefault(sample.trajectory_id, []).append(sample)

    # Every episode is exactly one sequence with weight 1.0; per-step weight
    # inside the sequence is sequence_weight / span_count.
    assert len(by_traj["t0"]) == 1
    assert by_traj["t0"][0].sequence_weight == 1.0
    assert by_traj["t0"][0].span_count == 3
    assert len(by_traj["t1"]) == 1
    assert by_traj["t1"][0].sequence_weight == 1.0
    assert by_traj["t1"][0].span_count == 1

    batch = collate_samples(packed, pad_token_id=0)
    assert torch.allclose(batch["sequence_weights"], torch.ones(2))


def test_packing_identity_is_one_sequence_per_episode_plus_boundaries() -> None:
    from training.episode_packing import PackingStats
    from training.ppo import _assert_packing_identity

    stats = PackingStats(input_samples=753, output_samples=33, segment_boundaries=1)
    _assert_packing_identity(stats, episodes=32)  # 32 episodes + 1 compaction
    stats.output_samples = 34
    with pytest.raises(RuntimeError, match="packing identity violated"):
        _assert_packing_identity(stats, episodes=32)
