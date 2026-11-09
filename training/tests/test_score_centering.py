"""Score centering: analytic gradients, evidence alignment and serving contract."""

import asyncio
import base64
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from aiohttp.test_utils import TestClient, TestServer

from training.config import ExperimentConfig, TrainingConfig
from training.contracts import RolloutStep, RolloutTrajectory
from training.dataset import build_samples, collate_samples
from training.episode_packing import pack_episode_samples
from training.native_vllm_server import (
    ContractError,
    NativeService,
    create_app,
    validate_generation,
)
from training.ppo import (
    _parity_prefix,
    _validate_policy_resume_recipe,
    token_scores_from_hidden_states,
)
from training.rewards import StepSignal
from training.score_centering import pack_head, unpack_head, score_centered_objective
from training.vllm_runtime import _validate_runtime_config
from test_native_vllm_server import Engine, request
from test_ppo import _make_tiny_ppo_trainer, _tiny_action_sample
import test_rollout_gateway as gateway_fixtures
from test_rollout_gateway import _start, _StubVllm

from engine_wire import generation

gateway = gateway_fixtures.gateway


@pytest.mark.parametrize(
    "cap,head", [(2.0, [0, 2]), (float("inf"), [0, 2]), (2.0, [0, 1, 2, 3])]
)
@pytest.mark.parametrize("y", [0, 1])
def test_gradient_matches_closed_form_with_detached_coefficients(cap, head, y):
    logits = torch.tensor(
        [1.0, -1.0, 0.2, -0.3], dtype=torch.float64, requires_grad=True
    )
    q = torch.tensor([0.12, 0.38, 0.3, 0.2], dtype=torch.float64)
    lp = logits.log_softmax(-1)
    p = lp.detach().exp()
    advantage = -1.7
    objective, diagnostics = score_centered_objective(
        lp[y], q[y].log(), lp[head], q[head].log(), advantage, cap=cap
    )
    (gradient,) = torch.autograd.grad(-objective, logits)
    w = torch.minimum(p / q, torch.tensor(cap))
    rho = (1 - q[head].sum()).clamp(min=1e-6) / (1 - p[head].sum()).clamp(min=1e-6)
    c = q[head] * w[head] - rho * min(1 / rho, cap) * p[head]
    score = torch.eye(4, dtype=torch.float64) - p.unsqueeze(0)
    expected = -advantage * (w[y] * score[y] - (c[:, None] * score[head]).sum(0))
    torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)
    if math.isinf(cap):
        assert diagnostics["relative_correction"] == pytest.approx(0.0, abs=1e-12)
        torch.testing.assert_close(c, torch.zeros_like(c), atol=1e-12, rtol=0)


def test_full_vocab_constant_advantage_has_zero_expected_gradient():
    logits = torch.tensor([2.0, -0.5, 0.3], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64)
    lp = logits.log_softmax(-1)
    objective, _ = score_centered_objective(
        lp, q.log(), lp.expand(3, 3), q.log().expand(3, 3), 1.3
    )
    (gradient,) = torch.autograd.grad(-(q * objective).sum(), logits)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


def test_matching_policies_zero_correction_and_tis_cap():
    lp = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64).log().requires_grad_()
    objective, diagnostics = score_centered_objective(
        lp[0], lp[0].detach(), lp[:2], lp[:2].detach(), 1.0
    )
    assert diagnostics["relative_correction"] == 0
    assert objective == lp[0]
    for ratio, cap in [(1.5, 2.0), (4.0, 2.0), (4.0, 3.0)]:
        value = torch.tensor(-0.4, requires_grad=True)
        result, _ = score_centered_objective(
            value,
            value.detach() - math.log(ratio),
            lp[:2],
            lp[:2].detach(),
            1.0,
            cap=cap,
        )
        assert torch.autograd.grad(result, value)[0] == pytest.approx(min(ratio, cap))


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"score_centering_top_k": 0},
        {"score_centering_top_k": True},
        {"score_centering_top_k": 1.5},
        {"score_centering_top_k": 2, "tis_ratio_cap": 1.0},
        {"score_centering_top_k": 2, "tis_ratio_cap": float("nan")},
        {"score_centering_top_k": 2, "tis_ratio_cap": float("inf")},
        {"score_centering_top_k": 2, "tis_ratio_cap": True},
    ],
)
def test_invalid_score_centering_config(fields):
    with pytest.raises(ValueError):
        TrainingConfig.from_dict({"policy_loss": "score_centered_tis", **fields})


@pytest.mark.parametrize(
    "fields", [{"score_centering_top_k": 2}, {"tis_ratio_cap": 2.0}]
)
def test_unused_knobs_rejected(fields):
    with pytest.raises(ValueError, match="require score_centered_tis"):
        TrainingConfig.from_dict(fields)


def test_config_dump_recipe_and_resume_identity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "configs/grpo_6gpu_arc2_overfit.json").read_text())
    expected = json.loads(json.dumps(base))
    expected["inference"]["agent_terminate_on_empty_reply"] = True
    expected["training"].update(
        policy_loss="score_centered_tis", score_centering_top_k=32, tis_ratio_cap=2.0
    )
    path = root / "configs/grpo_6gpu_arc2_overfit_score_centering.json"
    assert json.loads(path.read_text()) == expected
    config = ExperimentConfig.load(path)
    dump = tmp_path / "config.json"
    dump.write_text(json.dumps(config.to_dict()))
    assert ExperimentConfig.load(dump) == config
    for key, value in [
        ("score_centering_top_k", 16),
        ("tis_ratio_cap", 3.0),
        ("policy_loss", "ppo"),
    ]:
        changed = replace(config.training, **{key: value})
        with pytest.raises(RuntimeError, match="different training recipe"):
            _validate_policy_resume_recipe(asdict(config.training), changed)
        with pytest.raises(RuntimeError, match="different training recipe"):
            _validate_policy_resume_recipe(asdict(changed), config.training)
    default = TrainingConfig.from_dict({"output_dir": "runs", "model_name": "tiny"})
    assert default.score_centering_top_k is default.tis_ratio_cap is None
    assert TrainingConfig.from_dict(asdict(default)) == default
    assert (
        TrainingConfig.from_dict(
            {
                "output_dir": "runs",
                "model_name": "tiny",
                "policy_loss": "score_centered_tis",
                "score_centering_top_k": 2,
            }
        ).tis_ratio_cap
        == 2.0
    )


@pytest.mark.parametrize(
    "requested,configured,accepted",
    [
        (0, 0, True),
        (2, 0, False),
        (2, 2, True),
        (0, 2, True),
        (1, 2, False),
        (True, 2, False),
    ],
)
def test_native_logprobs_contract(requested, configured, accepted):
    kwargs = dict(max_model_len=32, vocab_size=100, score_centering_top_k=configured)
    if accepted:
        assert (
            validate_generation(request(logprobs=requested), **kwargs)["logprobs"]
            == requested
        )
    else:
        with pytest.raises(ContractError, match="logprobs"):
            validate_generation(request(logprobs=requested), **kwargs)


def test_runtime_capacity_refuses_inadequate_head():
    from test_vllm_runtime import config

    for fields in (
        {},
        {"max_logprobs": 20, "vocab_size": 100},
        {"max_logprobs": 32, "vocab_size": 20},
    ):
        with pytest.raises(RuntimeError, match="max_logprobs"):
            _validate_runtime_config(config(**fields), score_centering_top_k=32)
    _validate_runtime_config(
        config(max_logprobs=32, vocab_size=100), score_centering_top_k=32
    )


def _wire_head():
    return pack_head(
        [{1: NS(logprob=-0.5), 2: NS(logprob=-1.0), 9: NS(logprob=-4.0)}] * 2,
        k=2,
        vocab_size=10,
    )


@pytest.mark.parametrize(
    "mutation", ["missing", "k", "bytes", "nan", "positive", "vocab", "duplicate"]
)
def test_compact_wire_rejects_corrupted_evidence(mutation):
    payload = _wire_head()
    if mutation == "missing":
        payload.pop("ids_i32le")
    elif mutation == "k":
        payload["k"] = 3
    elif mutation == "bytes":
        payload["ids_i32le"] = payload["ids_i32le"][:-4]
    elif mutation in ("nan", "positive"):
        values = np.array(
            [[float("nan") if mutation == "nan" else 0.1, -1.0]] * 2, dtype="<f4"
        )
        payload["logprobs_f32le"] = base64.b64encode(values.tobytes()).decode()
    else:
        ids = np.array([[10 if mutation == "vocab" else 2, 2]] * 2, dtype="<i4")
        payload["ids_i32le"] = base64.b64encode(ids.tobytes()).decode()
    with pytest.raises(ValueError):
        unpack_head(payload, tokens=2, k=2, vocab_size=10)


def test_gateway_compact_roundtrip_strips_extension(gateway, monkeypatch):
    gateway.set_policy(replace(gateway._policy, score_centering_top_k=2, vocab_size=10))
    original = _StubVllm._respond

    def respond(server, body, status=200):
        original(server, {**body, "sampler_topk": _wire_head()}, status)

    monkeypatch.setattr(_StubVllm, "_respond", respond)
    response = _start(gateway, user="r#1")
    assert response.status_code == 200, response.text
    assert "sampler_topk" not in response.json()
    assert _StubVllm.seen[-1]["logprobs"] == 2
    capture = gateway.drain_window().captures[("r", "1")]
    np.testing.assert_array_equal(capture.completion_topk_ids, [[1, 2], [1, 2]])
    assert capture.completion_topk_ids.dtype == np.int32
    assert capture.completion_topk_logprobs.dtype == np.float32
    assert capture.completion_logprobs == [-0.1, -0.2]


def test_native_compact_http_response(monkeypatch):
    pytest.importorskip(
        "ullm.backends.engine_proxy", reason="requires the private harness transport"
    )
    import ullm.backends.engine_validation as validation
    import training.native_vllm_server as native

    monkeypatch.setattr(native, "tokenizer_fingerprint", lambda _: {})
    monkeypatch.setattr(native.importlib.metadata, "version", lambda _: "0.25.1")
    monkeypatch.setattr(validation, "validate_output_text", lambda *args: None)
    engine = Engine()
    service = NativeService(
        engine,
        sampling_factory=lambda **kw: NS(**kw),
        lora_factory=NS,
        score_centering_top_k=1,
    )

    async def run():
        async with TestClient(TestServer(create_app(service))) as client:
            response = await client.post(
                "/generate",
                json=generation(
                    request(logprobs=1), service.adapter("base").identity()
                ),
            )
            assert response.status == 200, await response.text()
            body = await response.json()
            ids, _ = unpack_head(body["sampler_topk"], tokens=2, k=1, vocab_size=100)
            np.testing.assert_array_equal(ids, [[99], [99]])
            assert list(body["outputs"][0]["logprobs"][0]) == ["3"]
            assert body["outputs"][0]["logprobs"][0]["3"]["logprob"] == -0.4
            assert engine.calls[0][1].logprobs == 1

    asyncio.run(run())


def test_samples_packing_prefix_and_passthrough_alignment():
    steps = []
    for i, (prompt, completion) in enumerate(
        [([1, 2], [3, 4]), ([1, 2, 3, 4, 8], [5])]
    ):
        ids = np.array([[token, 9] for token in completion], dtype=np.int32)
        logprobs = np.array(
            [[-0.3 - token / 10, -2.0] for token in completion], dtype=np.float32
        )
        steps.append(
            RolloutStep(
                i,
                [],
                [],
                {},
                0.0,
                False,
                prompt,
                completion,
                [-0.5] * len(completion),
                completion_topk_ids=ids,
                completion_topk_logprobs=logprobs,
            )
        )
    samples = build_samples(
        [RolloutTrajectory("r", "g", steps)],
        per_rollout_signals=[[StepSignal(reward=0, return_to_go=1, advantage=1)] * 2],
    )
    assert samples[0].completion_topk_ids is steps[0].completion_topk_ids
    packed, _ = pack_episode_samples(samples, max_sequence_tokens=20)
    assert len(packed) == 1
    for sample, expected in [
        (samples[0], [3, 4]),
        (packed[0], [3, 4, 5]),
        (_parity_prefix(packed[0], 2), [3, 4]),
    ]:
        batch = collate_samples([sample], pad_token_id=0)
        mask = batch["completion_mask"][:, 1:]
        assert batch["sampler_topk_ids"].dtype == torch.int64
        assert batch["sampler_topk_logprobs"].dtype == torch.float32
        assert batch["sampler_topk_ids"][mask][:, 0].tolist() == expected
        assert not batch["sampler_topk_ids"][~mask].any()
        assert not batch["sampler_topk_logprobs"][~mask].any()
        torch.testing.assert_close(
            batch["sampler_topk_logprobs"][mask][:, 0],
            torch.tensor([-0.3 - token / 10 for token in expected]),
        )


def test_chunked_scores_match_dense_biased_logits_and_gradients():
    torch.manual_seed(71)
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    head = torch.nn.Linear(4, 7)
    target = torch.tensor([[1, 2, 3, 4], [4, 2, 1, 0]])
    mask = torch.tensor([[False, True, True, False], [True, False, True, True]])
    ids = torch.tensor([0, 2, 5]).expand(2, 4, 3)
    bias_ids = torch.tensor([2, 4])
    biases = torch.randn(2, 4, 2)
    scores = token_scores_from_hidden_states(
        hidden,
        lm_head=head,
        target_ids=target,
        target_mask=mask,
        compute_entropies=True,
        chunk_size=2,
        bias_token_ids=bias_ids,
        bias_values=biases,
        head_ids=ids,
    )
    logits = head(hidden[:, :-1][mask]).float()
    logits.index_add_(-1, bias_ids, biases[mask])
    logp = logits - logits.logsumexp(-1, keepdim=True)
    expected = logp.gather(-1, ids[mask])
    torch.testing.assert_close(scores[2][mask], expected)
    actual_grad = torch.autograd.grad(
        scores[2].sum() + scores[0].sum(), (hidden, head.weight), retain_graph=True
    )
    dense_grad = torch.autograd.grad(
        expected.sum() + logp.gather(-1, target[mask, None]).sum(),
        (hidden, head.weight),
    )
    for actual, dense in zip(actual_grad, dense_grad, strict=True):
        torch.testing.assert_close(actual, dense)
    assert not scores[2][~mask].any()


def test_tiny_trainer_step_and_reference_forward_preserve_actor_heads():
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.config.policy_loss = "score_centered_tis"
    trainer.config.score_centering_top_k = 2
    trainer.config.tis_ratio_cap = 2.0
    sample = _tiny_action_sample(advantage=1.0)
    sample.completion_topk_ids = np.array([[1, 2]], dtype=np.int32)
    sample.completion_topk_logprobs = np.log(np.array([[0.5, 0.04]], dtype=np.float32))
    sample.old_token_logprobs = [math.log(0.04)]
    sample.old_logprob = math.log(0.04)
    batch = collate_samples([sample], pad_token_id=0)
    lp, entropy = trainer._token_logprobs_and_entropies(batch)
    actor_heads = batch["trainer_topk_logprobs"]
    with torch.no_grad():
        trainer._token_logprobs(batch)
    assert batch["trainer_topk_logprobs"] is actor_heads
    terms = trainer._span_loss_terms(batch, lp, entropy)
    assert terms["centering_statistics"]["score_centering/capped_sum"] == 1
    expected, _ = score_centered_objective(
        lp,
        batch["old_token_logprobs"],
        actor_heads,
        batch["sampler_topk_logprobs"],
        1.0,
    )
    torch.testing.assert_close(terms["policy_loss"], -expected.sum())
    terms["policy_loss"].backward()
    before = trainer.model.logits.detach().clone()
    trainer.optimizer.step()
    assert trainer.model.logits[2] > before[2]
    assert terms["centering_statistics"]["score_centering/tokens"] == 1
    assert trainer._compute_probe_entropy_forward([sample]) > 0


def test_arc2_trajectory_shares_compact_capture_arrays():
    from test_arc2_integration import result_payload
    from training.platypus_rollout import build_window_trajectories
    from training.rollout_gateway import RunResult, StepCapture, WindowCapture

    run = RunResult.from_payload(result_payload())
    ids, logprobs = unpack_head(_wire_head(), tokens=2, k=2, vocab_size=10)
    capture = StepCapture(
        run_id="abc",
        nonce="first",
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4],
        completion_logprobs=[-0.1, -0.2],
        finish_reason="stop",
        order=0,
        native_token_prompt=True,
        adapter=run.serving_adapter,
        completion_topk_ids=ids,
        completion_topk_logprobs=logprobs,
    )
    window = WindowCapture(captures={("abc", "first"): capture}, results={"abc": run})
    (trajectory,) = build_window_trajectories(
        window, policy_iteration=0, max_runtime_seconds=900.0
    )
    assert trajectory.steps[0].completion_topk_ids is ids
    assert trajectory.steps[0].completion_topk_logprobs is logprobs
    assert [v["logprob"] for v in trajectory.steps[0].completion_logprobs] == [
        -0.1,
        -0.2,
    ]


def test_diagnostics_pool_by_tokens_across_ranks():
    from training.policy_update import (
        merge_policy_update_statistics,
        summarize_policy_update_statistics,
    )

    accumulated = {}
    for tokens, value in [(2.0, 1.0), (8.0, 3.0)]:
        stats = {"score_centering/tokens": tokens}
        for key in [
            "relative_correction",
            "sampler_head_mass",
            "trainer_head_mass",
            "rho",
            "capped",
        ]:
            stats[f"score_centering/{key}_sum"] = tokens * value
        merge_policy_update_statistics(accumulated, stats)
    summary = summarize_policy_update_statistics(accumulated)
    assert summary["score_centering/mean_rho"] == pytest.approx(2.6)
    assert summary["score_centering/mean_relative_correction"] == pytest.approx(2.6)


def test_missing_sampler_head_fails_before_training():
    sample = _tiny_action_sample(advantage=1.0)
    trainer = _make_tiny_ppo_trainer(advantage=1.0)
    trainer.config.policy_loss = "score_centered_tis"
    trainer.config.score_centering_top_k = 2
    trainer.config.tis_ratio_cap = 2.0
    with pytest.raises(KeyError, match="sampler_topk_ids"):
        trainer._token_logprobs_and_entropies(collate_samples([sample], pad_token_id=0))
