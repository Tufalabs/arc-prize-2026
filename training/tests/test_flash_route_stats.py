"""Routing statistics for score-aware expert pruning."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from test_flash_model import _frozen_linear
from training.flash_route_stats import (
    RouteStats,
    aggregate_agreement,
    chains_with_logprobs,
    reply_targets,
    install_route_recorder,
    keep_plan,
    mass_scores,
    reap_scores,
    within_game_weights,
)
from training.frozen_nvfp4 import FrozenNVFP4Expert, FrozenNVFP4Experts


def test_trace_calls_read_their_token_ids_and_images(tmp_path):
    from training.flash_route_stats import iter_trace_calls_with_media

    path = tmp_path / "steps.jsonl"
    row = {"input_token_ids": [1, 2], "output_token_ids": [3], "messages": [], "media": [{"path": "media/a.png"}]}
    path.write_text("\n".join(json.dumps(row) for _ in range(2)))
    assert list(iter_trace_calls_with_media(path)) == [([1, 2], [3], [str(tmp_path / "media/a.png")])] * 2
    path.write_text(json.dumps({"input_token_ids": [], "output_token_ids": [3]}))
    with pytest.raises(ValueError, match="token ids"):
        list(iter_trace_calls_with_media(path))


def test_weights_standardise_scores_within_each_game():
    scores = {"a1": 10.0, "a2": 30.0, "b1": 5.0, "b2": 5.0}
    games = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    weights = within_game_weights(scores, games, beta=1.0)
    assert weights["a1"] == pytest.approx(math.exp(-1.0))
    assert weights["a2"] == pytest.approx(math.exp(1.0))
    assert weights["b1"] == weights["b2"] == 1.0  # no spread: no preference


def test_reap_scores_are_weighted_means_over_routed_tokens():
    shape = (1, 3)
    run = lambda gnorm, count: {"sum_gnorm": np.array([gnorm], float), "count": np.array([count], float)}  # noqa: E731
    per_run = {"good": run([8.0, 0.0, 2.0], [4, 0, 2]), "bad": run([0.0, 6.0, 4.0], [0, 2, 4])}
    assert per_run["good"]["sum_gnorm"].shape == shape
    plain = reap_scores(per_run)
    assert plain.tolist() == [[2.0, 3.0, 1.0]]
    weighted = reap_scores(per_run, {"good": 3.0, "bad": 1.0})
    # expert 2: (3*2 + 1*4) / (3*2 + 1*4) = 1; expert 1 only seen in the bad run.
    assert weighted.tolist() == [[2.0, 3.0, 1.0]]
    assert reap_scores({"x": run([0.0, 0.0, 0.0], [0, 0, 0])}).tolist() == [[0.0, 0.0, 0.0]]


def test_keep_plan_keeps_the_highest_scores_ascending():
    scores = np.array([[0.1, 0.9, 0.5, 0.7], [0.4, 0.3, 0.2, 0.1]])
    plan = keep_plan(scores, keep=2, source={"criterion": "test"})
    assert plan == {"num_experts": 4, "kept": {"0": [1, 3], "1": [0, 1]}, "criterion": "test"}
    with pytest.raises(ValueError):
        keep_plan(scores, keep=4, source={})


def _bank(hidden: int, inter: int, experts: int) -> FrozenNVFP4Experts:
    return FrozenNVFP4Experts(
        [
            FrozenNVFP4Expert(
                _frozen_linear(inter, hidden, 3 * e),
                _frozen_linear(inter, hidden, 3 * e + 1),
                _frozen_linear(hidden, inter, 3 * e + 2),
            )
            for e in range(experts)
        ]
    )


def test_the_recorder_sums_gate_times_output_norm_and_leaves_outputs_unchanged():
    torch.manual_seed(2)
    hidden, inter, experts, tokens, top_k = 32, 16, 4, 6, 2
    bank = _bank(hidden, inter, experts)
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=bank))]))
    x = torch.randn(tokens, hidden)
    index = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    weights = torch.rand(tokens, top_k)
    before = bank(x, index, weights)
    stats = RouteStats.zeros(1, experts, "cpu")
    assert install_route_recorder(model, stats) == 1
    after = bank(x, index, weights)
    assert torch.allclose(before, after, atol=1e-6)
    expected = torch.zeros(experts, 3, dtype=torch.float64)
    for t in range(tokens):
        for k in range(top_k):
            e = int(index[t, k])
            norm = torch.linalg.vector_norm(bank[e](x[t : t + 1])[0].float())
            expected[e] += torch.tensor([float(weights[t, k] * norm), float(weights[t, k]), 1.0], dtype=torch.float64)
    assert torch.allclose(stats.sum_gnorm[0], expected[:, 0], rtol=1e-5)
    assert torch.allclose(stats.sum_g[0], expected[:, 1], rtol=1e-6)
    assert torch.equal(stats.count[0], expected[:, 2])


def test_the_recorder_refuses_a_bank_it_cannot_read():
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(experts=nn.Linear(2, 2)))]))
    with pytest.raises(TypeError, match="frozen NVFP4"):
        install_route_recorder(model, RouteStats.zeros(1, 2, "cpu"))


def test_outcome_weights_move_mass_but_not_the_reap_mean():
    """An expert used only by a low-scoring rollout keeps its REAP mean but
    loses mass when that rollout is down-weighted."""
    def run(gnorm, count, tokens):
        return {"sum_gnorm": np.array([gnorm], float), "count": np.array([count], float), "tokens": tokens}

    per_run = {"good": run([4.0, 0.0], [2, 0], 10), "bad": run([0.0, 4.0], [0, 2], 10)}
    even = mass_scores(per_run)
    tilted = mass_scores(per_run, {"good": 3.0, "bad": 1.0})
    assert even.tolist() == [[0.2, 0.2]]
    assert tilted[0, 0] > tilted[0, 1]
    assert reap_scores(per_run, {"good": 3.0, "bad": 1.0}).tolist() == reap_scores(per_run).tolist()
    with pytest.raises(ValueError, match="replayed tokens"):
        mass_scores({"x": run([0.0, 0.0], [0, 0], 0)})


def test_reply_targets_cover_every_reply_token_once_across_chunks():
    replies, total = [(3, 5), (7, 10)], 10
    seen = []
    for start in range(0, total, 4):
        width = min(4, total - start)
        for row, position in reply_targets(start, width, total, replies):
            assert 0 <= row < width and position == start + row + 1
            seen.append(position)
    assert seen == [3, 4, 7, 8, 9]


def test_multimodal_replay_scatters_features_across_chunk_boundaries(monkeypatch):
    import sys
    import types

    from training import flash_route_stats

    IMAGE = 7
    ids = [1, IMAGE, IMAGE, IMAGE, 2, 3]  # the image's placeholders straddle chunks of 2
    calls = []

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 2)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])

    class Visual(nn.Module):
        def forward(self, pixel_values, grid_thw, return_dict):
            assert grid_thw.tolist() == [[1, 2, 6]]  # the processor's grid, passed through
            return SimpleNamespace(pooler_output=torch.tensor([[100.0, 100.0], [101.0, 101.0], [102.0, 102.0]]))

    text = Text()
    model = SimpleNamespace(model=text, config=None, parameters=text.parameters)
    decoder = SimpleNamespace(visual=Visual(), image_token_id=IMAGE)
    media = types.ModuleType("ullm.media")
    media.process_images = lambda processor, images, processing: {
        "pixel_values": torch.zeros(12, 4), "image_grid_thw": torch.tensor([[1, 2, 6]])}
    media.image_layout = lambda ids, grids, processing: [{"offset": 1, "length": 3, "grid_thw": [1, 2, 6]}]
    media.position_inputs = lambda length, layout, processing: {
        "position_ids": torch.arange(3 * length).reshape(3, length)}
    monkeypatch.setitem(sys.modules, "ullm.media", media)
    monkeypatch.setattr("transformers.DynamicCache", lambda config: object())
    flash_route_stats.replay_multimodal_chain(
        model, decoder, ids, ["img"], processor=None, processing={}, chunk_tokens=2)
    assert [call["inputs_embeds"].shape[1] for call in calls] == [2, 2, 2]
    assert calls[0]["inputs_embeds"][0, 1].tolist() == [100.0, 100.0]
    assert calls[1]["inputs_embeds"][0, :, 0].tolist() == [101.0, 102.0]
    assert torch.equal(calls[2]["ple_input_ids"][0], torch.tensor([2, 3]))
    assert calls[1]["position_ids"][:, 0].tolist() == [[2, 3], [8, 9], [14, 15]]


def test_chains_carry_the_engine_log_probs_of_their_replies(tmp_path):
    steps = tmp_path / "steps.jsonl"
    records = [
        {"input_token_ids": [1, 2], "output_token_ids": [3], "logprobs": [{"logprob": -0.5}]},
        {"input_token_ids": [1, 2, 3, 4], "output_token_ids": [5, 6],
         "logprobs": [{"logprob": -0.1}, {"logprob": -0.2}]},
        {"input_token_ids": [9], "output_token_ids": [7]},  # no log-probs recorded
    ]
    steps.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert chains_with_logprobs(steps) == [
        ([1, 2, 3, 4, 5, 6], {2: -0.5, 4: -0.1, 5: -0.2}),
        ([9, 7], {}),
    ]


def test_aggregate_agreement_reads_counts_and_keep_sets():
    rng = np.random.default_rng(0)
    count = rng.integers(1, 100, size=(2, 8)).astype(float)
    gnorm = rng.random((2, 8))
    same = aggregate_agreement({"count": count, "sum_gnorm": gnorm}, {"count": count, "sum_gnorm": gnorm}, keep=4)
    assert same["min_count_correlation"] == pytest.approx(1.0)
    assert same["min_keep_overlap"] == 1.0
    swapped = gnorm.copy()
    order = np.argsort(-gnorm[0], kind="stable")
    swapped[0, order[3]], swapped[0, order[4]] = gnorm[0, order[4]], gnorm[0, order[3]]
    moved = aggregate_agreement({"count": count, "sum_gnorm": gnorm}, {"count": count, "sum_gnorm": swapped}, keep=4)
    assert moved["min_keep_overlap"] == 0.75 and moved["mean_keep_overlap"] == 0.875


def test_placeholders_without_images_are_refused():
    from training.flash_route_stats import refuse_images_as_text

    image = 248056
    refuse_images_as_text([1, 2, 3], image, "run")
    with pytest.raises(ValueError, match="run: 2 image placeholder tokens and no images"):
        refuse_images_as_text([1, image, image, 3], image, "run")


def test_a_chain_replays_as_served_whichever_it_holds(monkeypatch, tmp_path):
    """One entry point for every replay: images go through the vision tower,
    a chain without any replays as text, and placeholders without their images
    never replay at all."""
    from PIL import Image

    from training import flash_route_stats

    image = 248056
    monkeypatch.setattr("training.flash_model.flash_checkpoint_config", lambda checkpoint: {"image_token_id": image})
    calls = []
    monkeypatch.setattr(flash_route_stats, "replay_chain", lambda model, ids, **kw: calls.append(("text", ids)))
    monkeypatch.setattr(
        flash_route_stats,
        "replay_multimodal_chain",
        lambda model, decoder, ids, images, **kw: calls.append(("images", ids, [im.size for im in images])),
    )
    replay = flash_route_stats.ServedReplay(model="model", checkpoint=tmp_path)
    monkeypatch.setattr(replay, "_load_vision", lambda: ("decoder", "processor", {}))
    path = tmp_path / "grid.png"
    Image.new("RGB", (4, 6)).save(path)

    replay([1, 2, 3], [], chunk_tokens=8)
    replay([1, image, image, 3], [str(path)], chunk_tokens=8)
    assert calls == [("text", [1, 2, 3]), ("images", [1, image, image, 3], [(4, 6)])]
    assert replay.images == 1
    with pytest.raises(ValueError, match="rollout-7: 2 image placeholder tokens and no images"):
        replay([1, image, image, 3], [], chunk_tokens=8, what="rollout-7")
    assert len(calls) == 2


def test_trace_chains_carry_replies_and_images(tmp_path):
    from training.flash_route_stats import trace_chains

    steps = tmp_path / "steps.jsonl"
    records = [
        {"input_token_ids": [1, 2], "output_token_ids": [3], "media": [{"path": "a.png"}]},
        # continues: its prompt starts with 1 2 3
        {"input_token_ids": [1, 2, 3, 4], "output_token_ids": [5, 6], "media": [{"path": "a.png"}, {"path": "b.png"}]},
        # a fresh render (after a summary): a new chain
        {"input_token_ids": [9], "output_token_ids": [8], "media": []},
    ]
    steps.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert trace_chains(steps) == [
        ([1, 2, 3, 4, 5, 6], [(2, 3), (4, 6)], [str(tmp_path / "a.png"), str(tmp_path / "b.png")]),
        ([9, 8], [(1, 2)], []),
    ]


def test_a_cut_chain_keeps_only_whole_images():
    from training.flash_route_stats import truncate_chain

    im = 248056
    ids = [1, im, im, im, 2, 3, im, im, 4, 5]
    replies = [(4, 6), (8, 10)]
    images = ["a.png", "b.png"]
    # A cut inside the second image moves back to before it.
    assert truncate_chain(ids, replies, images, 7, im) == ([1, im, im, im, 2, 3], [(4, 6)], ["a.png"])
    # A cut inside the first image leaves no image at all.
    assert truncate_chain(ids, replies, images, 2, im) == ([1], [], [])
    # A cut right after an image keeps it; a reply is cut where the chain is.
    assert truncate_chain(ids, replies, images, 9, im) == (ids[:9], [(4, 6), (8, 9)], images)
    assert truncate_chain(ids, replies, images, 99, im) == (ids, replies, images)
