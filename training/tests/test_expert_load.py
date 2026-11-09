"""Expert load-balance metrics: accumulator, statistics, token roles, pooling, tags."""

from datetime import timedelta
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gloo_ranks import run_rank
from training.expert_load import (
    EVAL_ROLES,
    ROLES,
    TRAINING_ROLES,
    ExpertLoadRecorder,
    install_expert_load_recorder,
    load_statistics,
    router_token_weights,
    summarize_counts,
    take_eval_handoff,
    write_eval_handoff,
)
from training.metrics import EXPERT_LOAD_SCALARS, expert_load_scalars, format_expert_load


def test_balanced_load_is_perfectly_balanced():
    stats = load_statistics(torch.full((3, 307), 40.0))
    assert stats["expert_load_cv"] == pytest.approx(0.0, abs=1e-12)
    assert stats["expert_load_max_share"] == pytest.approx(1.0)
    assert stats["expert_load_entropy"] == pytest.approx(1.0)
    assert stats["experts_unused_fraction"] == 0.0
    assert stats["expert_load_entropy_worst_layer"] == pytest.approx(1.0)


def test_one_hot_expert_takes_everything():
    experts = 8
    counts = torch.zeros(1, experts)
    counts[0, 3] = 100.0
    stats = load_statistics(counts)
    # mean = N/E, population std = N sqrt(E-1)/E: cv = sqrt(E-1), max share = E.
    assert stats["expert_load_cv"] == pytest.approx(math.sqrt(experts - 1))
    assert stats["expert_load_max_share"] == pytest.approx(experts)
    assert stats["expert_load_entropy"] == pytest.approx(0.0, abs=1e-12)
    assert stats["experts_unused_fraction"] == pytest.approx((experts - 1) / experts)


def test_layers_average_and_worst_layer_and_idle_layers():
    balanced = torch.full((4,), 10.0)
    skewed = torch.tensor([25.0, 5.0, 10.0, 0.0])
    idle = torch.zeros(4)
    stats = load_statistics(torch.stack([balanced, idle, skewed]))
    skew = load_statistics(skewed[None])
    # The idle layer routed nothing: it is skipped, not averaged in as zeros.
    assert stats["expert_load_cv"] == pytest.approx(skew["expert_load_cv"] / 2)
    assert stats["expert_load_max_share"] == pytest.approx((1.0 + 2.5) / 2)
    assert stats["experts_unused_fraction"] == pytest.approx(0.125)
    assert stats["expert_load_max_share_worst_layer"] == pytest.approx(2.5)
    assert stats["expert_load_cv_worst_layer"] == pytest.approx(skew["expert_load_cv"])
    assert stats["expert_load_entropy_worst_layer"] == pytest.approx(skew["expert_load_entropy"])
    assert stats["worst_layer"] == 2.0
    assert load_statistics(torch.zeros(2, 4)) is None


def test_recorder_counts_top_k_assignments_per_role():
    recorder = ExpertLoadRecorder(layers=2, experts=4, top_k=2, device="cpu")
    indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    weights = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]])
    recorder.record(0, indices, None)  # unarmed: diagnostic forwards add nothing
    with recorder.capture(TRAINING_ROLES, weights):
        recorder.record(1, indices, torch.rand(5, 2))
        with pytest.raises(RuntimeError, match="already armed"):
            with recorder.capture(EVAL_ROLES, torch.ones(1, 5)):
                pass
        with pytest.raises(ValueError, match="layout changed"):
            recorder.record(0, indices[:4], None)
    recorder.record(1, indices, None)  # checkpoint recompute after the capture
    assert recorder.counts[:, 0].sum() == 0
    torch.testing.assert_close(recorder.counts[0, 1], torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(recorder.counts[1, 1], torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64))
    assert recorder.counts[2].sum() == 0
    with recorder.capture(EVAL_ROLES, torch.ones(1, 5)):
        recorder.record(0, indices, None)
    assert recorder.counts[2, 0].tolist() == [3.0, 2.0, 3.0, 2.0]
    summary = summarize_counts(recorder.pooled_counts(), top_k=2)
    assert summary["train"]["tokens"] == 4 and summary["rollout"]["tokens"] == 2
    assert summary["eval"]["tokens"] == 5
    assert recorder.counts.sum() == 0  # pooled_counts resets for the next window


def sample(ids, completion, attention=None):
    return SimpleNamespace(
        input_ids=list(ids), completion_mask=list(completion),
        attention_mask=list(attention or [1] * len(ids)),
    )


def test_token_roles_follow_the_completion_mask_in_router_order():
    first = sample([5, 6, 7, 8, 9], [0, 0, 1, 1, 0])
    second = sample([5, 6, 7], [0, 1, 1])
    rows = router_token_weights([first, second], TRAINING_ROLES, packed_alignment=None)
    # A padded batch reaches the routers row by row: [B * T] tokens.
    assert rows.tolist() == [
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        # Position i decides token i + 1: the positions before each sampled token.
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    ]
    assert router_token_weights([second], EVAL_ROLES, packed_alignment=None).tolist() == [[1, 1, 0]]


def test_token_roles_follow_the_flex_qsa_packing():
    from training.flash.input_layout import pack_padded_batch

    first = sample([5, 6, 7, 8, 9], [0, 0, 1, 1, 0])
    second = sample([5, 6, 7], [0, 1, 1])
    rows = router_token_weights([first, second], TRAINING_ROLES, packed_alignment=4)
    # Each document is padded to a multiple of 4 with room for a separator.
    ids = torch.tensor([[5, 6, 7, 8, 9], [5, 6, 7, 0, 0]])
    layout = pack_padded_batch(ids, (ids > 0).long(), alignment=4)
    assert rows.shape == (2, layout.input_ids.shape[1]) == (2, 12)
    assert rows[0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    assert rows[1].tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    # The rows land exactly on the packed stream's real tokens.
    assert (layout.input_ids[0][rows[0].bool()] > 0).all()


def test_installs_on_every_router_once():
    class Router(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer_idx, self.num_experts, self.top_k = layer, 6, 2

    model = torch.nn.Sequential(Router(1), torch.nn.Linear(2, 2), Router(0))
    recorder = install_expert_load_recorder(model, router_type=Router, device="cpu")
    assert recorder.counts.shape == (len(ROLES), 2, 6) and recorder.top_k == 2
    assert [(m.route_recorder, m.route_layer) for m in (model[0], model[2])] == [(recorder, 1), (recorder, 0)]
    assert install_expert_load_recorder(torch.nn.Linear(2, 2), router_type=Router, device="cpu") is None
    model[2].layer_idx = 1
    with pytest.raises(ValueError, match="not 0..1"):
        install_expert_load_recorder(model, router_type=Router, device="cpu")


def _pool_rank(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        recorder = ExpertLoadRecorder(layers=1, experts=3, top_k=1, device="cpu")
        # Rank 0 routes both its tokens to expert 0, rank 1 one each to
        # experts 1 and 2: only the pooled counts see all four tokens.
        indices = torch.tensor([[0], [0]]) if rank == 0 else torch.tensor([[1], [2]])
        with recorder.capture(TRAINING_ROLES, torch.ones(2, 2)):
            recorder.record(0, indices, None)
        counts = recorder.pooled_counts()
        if rank == 0:
            Path(output).write_text(json.dumps(counts.tolist()))
    finally:
        dist.destroy_process_group()


def test_counts_pool_across_ranks(tmp_path):
    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=run_rank,
            args=(rank, _pool_rank, str(tmp_path / "rdzv"), str(tmp_path / "counts.json")),
        )
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=35)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
    counts = torch.tensor(json.loads((tmp_path / "counts.json").read_text()))
    assert counts[0, 0].tolist() == [2.0, 1.0, 1.0]
    assert counts[1, 0].tolist() == [2.0, 1.0, 1.0]
    assert counts[2].sum() == 0


def test_tags_are_identical_under_each_prefix():
    stats = load_statistics(torch.tensor([[3.0, 1.0, 0.0, 4.0]]))
    stats["tokens"] = 4.0
    summary = {"train": stats, "rollout": stats, "eval": stats,
               "eval_pass": {"eval_seconds": 12.5, "eval_episodes": 8.0}}
    scalars = expert_load_scalars(summary)
    for prefix in ("train", "rollout", "eval"):
        assert {f"{prefix}/{name}" for name in EXPERT_LOAD_SCALARS} <= set(scalars)
        assert scalars[f"{prefix}/experts_unused_fraction"] == 0.25
        assert scalars[f"{prefix}/expert_load_max_share"] == pytest.approx(2.0)
    assert set(scalars) - {f"{p}/{n}" for p in ("train", "rollout", "eval") for n in EXPERT_LOAD_SCALARS} == {
        "eval/expert_load_episodes", "train/z_timing/expert_load_eval_seconds"}
    assert scalars["eval/expert_load_episodes"] == 8.0
    assert scalars["train/z_timing/expert_load_eval_seconds"] == 12.5
    # No eval episodes reached the learner: no eval/ tags at all.
    assert not [name for name in expert_load_scalars({"train": stats}) if name.startswith("eval/")]
    line = format_expert_load({"train": stats, "rollout": stats})
    assert line.startswith("train[tokens=4 cv=") and " rollout[" in line and "eval" not in line


def test_cli_writes_rollout_tags_on_the_rollout_axis(caplog):
    from training.cli import _write_expert_load

    written = []
    writer = SimpleNamespace(add_scalar=lambda name, value, step: written.append((name, step)))
    stats = load_statistics(torch.tensor([[3.0, 1.0, 0.0, 4.0]]))
    stats["tokens"] = 4.0
    with caplog.at_level("INFO", logger="training.cli"):
        _write_expert_load(writer, summary={"train": stats, "rollout": stats}, iteration_index=7,
                           rollout_step=11, num_iterations=10)
    assert {step for name, step in written if name.startswith("rollout/")} == {11}
    assert {step for name, step in written if name.startswith("train/")} == {7}
    assert len(written) == 2 * len(EXPERT_LOAD_SCALARS)
    assert "expert load iteration 8/10 train[" in caplog.text
    written.clear()
    _write_expert_load(writer, summary={}, iteration_index=7, rollout_step=11, num_iterations=10)
    assert written == []


def _step(prompt, completion, index):
    return SimpleNamespace(
        step_index=index, prompt_token_ids=prompt, completion_token_ids=completion,
        completion_logprobs=[{"token_id": t, "logprob": -0.5} for t in completion],
        completion_topk_ids=None, completion_topk_logprobs=None, prompt_messages=[{}],
        metadata={}, media=None, media_evidence=None, reward=0.0,
    )


def _rollout(task, name, turns=2):
    steps, context = [], [1, 2, 3]
    for index in range(turns):
        completion = [10 + index, 11 + index]
        steps.append(_step(list(context), completion, index))
        context = context + completion + [4]
    return SimpleNamespace(trajectory_id=name, task_id=task, steps=steps, metadata={})


def test_eval_handoff_round_trip(tmp_path):
    rollouts = [_rollout("a", "a0"), _rollout("a", "a1"), _rollout("b", "b0"),
                _rollout("c", "c0", turns=0)]
    path = write_eval_handoff(tmp_path, cycle_index=3, policy_iteration=5, rollouts=rollouts,
                              episodes=3, max_sequence_tokens=64)
    assert path is not None and path.name == "cycle_00003.pt"
    older = write_eval_handoff(tmp_path, cycle_index=2, policy_iteration=4, rollouts=rollouts[:1],
                               episodes=3, max_sequence_tokens=64)
    payload = take_eval_handoff(tmp_path)
    assert not list(older.parent.glob("cycle_*.pt"))  # newest kept, all consumed
    assert (payload["cycle_index"], payload["policy_iteration"]) == (3, 5)
    # One episode per game in turn; the game without token ids has nothing to score.
    assert [s.trajectory_id for s in payload["samples"]] == ["a0", "b0", "a1"]
    assert payload["episodes"] == 3
    # Each episode packs into one nested sequence of all its turns.
    first = payload["samples"][0]
    assert first.input_ids == [1, 2, 3, 10, 11, 4, 11, 12]
    assert first.completion_mask == [0, 0, 0, 1, 1, 0, 1, 1]
    assert take_eval_handoff(tmp_path) is None
    assert write_eval_handoff(tmp_path, cycle_index=4, policy_iteration=6, rollouts=rollouts,
                              episodes=0, max_sequence_tokens=64) is None
    # An episode that does not fit the training budget is not handed over.
    assert write_eval_handoff(tmp_path, cycle_index=5, policy_iteration=6, rollouts=rollouts[:1],
                              episodes=1, max_sequence_tokens=4) is None
