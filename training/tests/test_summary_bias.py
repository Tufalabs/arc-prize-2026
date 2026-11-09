"""Admitted token bias follows exact targets through packing and scoring."""

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from test_context_contract import _inference
from training.config import resolve_context_contract
from training.dataset import TokenizedSample, collate_samples
from training.episode_packing import pack_episode_samples
from training.ppo import (
    PPOTrainer,
    _masked_token_scores,
    _parity_prefix,
    token_scores_from_hidden_states,
)
from training.rollout_gateway import GatewayError, SamplingPolicy


def policy_body():
    context = resolve_context_contract(
        _inference(
            agent_max_summary_tokens=4096,
            agent_max_served_sequence_tokens=49152,
            agent_summary_forbidden_token_ids=[2],
        )
    )
    policy = SamplingPolicy("model", 1.0, 1.0, -1, context=context)
    body = dict(
        model="model",
        user="run#nonce",
        prompt=[1] * (context.compaction_trigger_tokens + 1),
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        min_p=0.0,
        min_tokens=0,
    )
    return policy, body


@pytest.mark.parametrize("bias", [{2: -100}, {"2": -100.0}, {2: -100.0}])
def test_bias_admission_normalizes_only_the_exact_contract(bias):
    policy, body = policy_body()
    body["logit_bias"] = bias
    policy.validate(body)
    assert body["logit_bias"] == {2: -100.0}
    assert type(next(iter(body["logit_bias"]))) is int
    assert type(body["logit_bias"][2]) is float


@pytest.mark.parametrize(
    "bias",
    [
        None,
        {},
        {2: -99},
        {3: -100},
        {2: -100, 3: -100},
        {2: "-100"},
        {2: True},
        {True: -100},
        {2.0: -100},
        {"2.0": -100},
        {-2: -100},
        {2: float("nan")},
        {2: float("inf")},
        {2: 10**1000},
        {2: -100, "2": -100},
        [],
    ],
)
def test_gateway_refuses_missing_or_changed_bias(bias):
    policy, body = policy_body()
    if bias is not None:
        body["logit_bias"] = bias
    with pytest.raises(GatewayError, match="logit_bias"):
        policy.validate(body)


@pytest.mark.parametrize("bias", [{}, {2: -100}])
def test_ordinary_request_cannot_carry_any_bias(bias):
    policy, body = policy_body()
    body.update(prompt=[1], logit_bias=bias)
    with pytest.raises(GatewayError, match="only on over-trigger"):
        policy.validate(body)


def test_empty_bias_contract_allows_absent_or_empty_over_trigger_bias():
    policy, body = policy_body()
    policy = replace(
        policy, context=replace(policy.context, summary_forbidden_token_ids=())
    )
    policy.validate(body)
    policy.validate({**body, "logit_bias": {}})


@pytest.mark.parametrize("biased", [False, True])
def test_masked_scores_bias_matches_dense_logsoftmax_and_entropy(biased):
    torch.manual_seed(7)
    hidden = torch.randn(4, 3, requires_grad=True)
    head = torch.nn.Linear(3, 5)
    ids = torch.tensor([0, 2, 1, 4])
    columns = torch.tensor([2, 4])
    values = torch.tensor([[0.0, 0.0], [-100.0, -3.0], [0.0, 0.0], [-100.0, 0.0]])
    kwargs = dict(bias_token_ids=columns, bias_values=values) if biased else {}
    actual, entropy = _masked_token_scores(hidden, ids, head, True, **kwargs)
    logits = head(hidden).float()
    if biased:
        logits = logits + torch.zeros_like(logits).index_add(-1, columns, values)
    logprobs = torch.log_softmax(logits, -1)
    torch.testing.assert_close(actual, logprobs.gather(-1, ids[:, None]).squeeze(-1))
    torch.testing.assert_close(entropy, -(logprobs.exp() * logprobs).sum(-1))
    expected_grad = torch.autograd.grad(
        (
            logprobs.gather(-1, ids[:, None]).sum()
            + (-(logprobs.exp() * logprobs).sum(-1)).sum()
        ),
        hidden,
        retain_graph=True,
    )[0]
    actual_grad = torch.autograd.grad((actual + entropy).sum(), hidden)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def sample(index, prompt, output, bias=None):
    size = len(prompt) + len(output)
    return TokenizedSample(
        trajectory_id="r",
        step_index=index,
        input_ids=prompt + output,
        labels=[-100] * len(prompt) + output,
        attention_mask=[1] * size,
        completion_mask=[0] * len(prompt) + [1] * len(output),
        reward=0.0,
        return_to_go=1.0,
        advantage=1.0,
        prompt_token_count=len(prompt),
        old_token_logprobs=[0.0] * (size - 1),
        token_logit_bias={}
        if bias is None
        else {2: [0.0] * (len(prompt) - 1) + [bias] * len(output)},
    )


def test_packing_collation_and_parity_prefix_preserve_only_biased_targets():
    ordinary = sample(0, [0], [1])
    summary = sample(1, [0, 1, 3], [2, 4], -100)
    reset = replace(sample(2, [3], [4]), segment_start=True)
    packed, _ = pack_episode_samples([ordinary, summary, reset], max_sequence_tokens=10)
    assert len(packed) == 2
    assert packed[0].token_logit_bias == {2: [0.0, 0.0, -100.0, -100.0]}
    assert packed[1].token_logit_bias == {}
    batch = collate_samples(packed, pad_token_id=0)
    assert batch["logit_bias_token_ids"].tolist() == [2]
    assert batch["logit_bias_values"].tolist() == [
        [[0.0], [0.0], [-100.0], [-100.0]],
        [[0.0], [0.0], [0.0], [0.0]],
    ]
    prefix = _parity_prefix(packed[0], 2)
    assert prefix.token_logit_bias == {2: [0.0, 0.0, -100.0]}
    assert not any(
        key.startswith("logit_bias")
        for key in collate_samples([ordinary, reset], pad_token_id=0)
    )


@pytest.mark.parametrize("chunk_size", [1, 10])
def test_chunked_checkpointed_scores_apply_sparse_bias_to_masked_targets(chunk_size):
    torch.manual_seed(8)
    hidden = torch.randn(2, 5, 3, requires_grad=True)
    head = torch.nn.Linear(3, 5)
    targets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    mask = torch.tensor([[True, False, True, True], [False, True, True, False]])
    values = torch.zeros(2, 4, 1)
    values[0, 2:] = -100
    actual, entropy = token_scores_from_hidden_states(
        hidden,
        lm_head=head,
        target_ids=targets,
        target_mask=mask,
        compute_entropies=True,
        chunk_size=chunk_size,
        bias_token_ids=torch.tensor([2]),
        bias_values=values,
    )
    logits = head(hidden[:, :-1]).float()
    logits = logits + torch.zeros_like(logits).index_add(-1, torch.tensor([2]), values)
    lp = logits.log_softmax(-1)
    expected = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).masked_fill(~mask, 0)
    expected_entropy = (-(lp.exp() * lp).sum(-1)).masked_fill(~mask, 0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(entropy, expected_entropy)
    expected_grad = torch.autograd.grad(
        (expected + expected_entropy).sum(), hidden, retain_graph=True
    )[0]
    (actual + entropy).sum().backward()
    torch.testing.assert_close(hidden.grad, expected_grad)


def test_reference_trust_region_and_policy_share_biased_scores():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 3)
            self.head = torch.nn.Linear(3, 5)

        def disable_adapter(self):
            return nullcontext()

        def forward(self, input_ids, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.model = Model()
    trainer.device = torch.device("cpu")
    from types import SimpleNamespace

    trainer.config = SimpleNamespace(policy_loss="ppo")
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer._decoder_and_lm_head = lambda: (trainer.model, trainer.model.head)
    trainer._autocast_context = nullcontext
    samples = [sample(0, [0], [1, 2], -100)]
    batch = collate_samples(samples, pad_token_id=0)
    policy = trainer._token_logprobs(batch)
    torch.testing.assert_close(trainer._reference_token_logprobs(batch), policy)
    torch.testing.assert_close(
        trainer._trust_region_token_logprobs(samples), policy.flatten()
    )
    plain = trainer._token_logprobs(
        collate_samples([replace(samples[0], token_logit_bias={})], pad_token_id=0)
    )
    assert not torch.allclose(policy, plain)


def test_resume_migrates_slotted_probe_samples_without_recapturing(tmp_path):
    old = sample(0, [0], [1, 2])
    del old.token_logit_bias
    torch.save({"version": 1, "samples": [old]}, tmp_path / "probe_states.pt")
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer._ensure_probe_states([sample(1, [3], [4])], tmp_path)
    (loaded,) = trainer._probe_samples
    assert loaded.input_ids == [0, 1, 2]
    batch = collate_samples([loaded], pad_token_id=0)
    assert "logit_bias_values" not in batch
    torch.testing.assert_close(batch["input_ids"], torch.tensor([[0, 1, 2]]))


def test_budget_limited_ordinary_continuations_are_not_summary_proofs():
    from training.gateway_lineage import LineageLedger, LineageError

    policy, body = policy_body()
    ledger = LineageLedger()
    for index, size in enumerate([30000, 30002]):
        draw = {**body, "prompt": list(range(size)), "max_tokens": 1000 - index}
        policy.validate(draw)
        assert not policy.is_summary_request(draw)
        stamp = dict(
            version=1,
            epoch=0,
            kind="continue" if index else "start",
            compaction_prompt_token_ids=None,
        )
        ledger.admit(
            "run",
            str(index),
            draw["prompt"],
            stamp,
            policy.context.lineage_policy(max_completion_tokens=draw["max_tokens"]),
            native_token_prompt=True,
            summary_request=policy.is_summary_request(draw),
        )
        ledger.complete("run", str(index), draw["prompt"], [size])
    with pytest.raises(LineageError, match="served over-trigger proof"):
        ledger.admit(
            "run",
            "reset",
            [1],
            dict(version=1, epoch=2, kind="reset", compaction_prompt_token_ids=None),
            policy.context.lineage_policy(),
            native_token_prompt=True,
        )


def test_budget_limited_ordinary_request_cannot_exceed_its_window():
    policy, body = policy_body()
    draw = {**body, "prompt": [1] * 32768, "max_tokens": 1}
    with pytest.raises(GatewayError, match="logit_bias"):
        policy.validate(draw)


def test_gateway_forwards_and_captures_admitted_bias_without_sockets():
    import asyncio

    asyncio.run(_gateway_capture())


async def _gateway_capture():
    import json
    from types import SimpleNamespace

    from engine_wire import generation, output
    from training.rollout_gateway import CaptureGateway

    policy, body = policy_body()
    policy = replace(policy, model="stub-model")
    gateway = CaptureGateway(upstream_base_url="http://unused.invalid", port=0)
    gateway.set_policy(policy)
    forwarded = []

    async def upstream(method, url, **kwargs):
        wire = kwargs["json"]
        forwarded.append(wire)
        response = output(
            dict(
                request_id=wire["request_id"],
                adapter=dict(
                    name="stub-model", id=0, path=None, server_instance_id="instance"
                ),
                prompt_token_ids=wire["prompt"]["prompt_token_ids"],
                output_token_ids=[3, 4],
                output_logprobs=[-0.1, -0.2],
                finish_reason="stop",
                text="ok",
                num_cached_tokens=0,
            )
        )
        return 200, json.dumps(response).encode(), "application/json"

    gateway._upstream = upstream
    for index, prompt in enumerate(
        [[1, 2], [1, 2, 3, 4] + [5] * (policy.context.compaction_trigger_tokens - 3)]
    ):
        draw = {
            **body,
            "model": "stub-model",
            "user": f"run#draw{index}",
            "prompt": prompt,
            "arc3_lineage": dict(
                version=1,
                epoch=0,
                kind="continue" if index else "start",
                compaction_prompt_token_ids=None,
            ),
        }
        if index:
            draw["logit_bias"] = {"2": -100}

        async def request_json():
            return generation(draw)

        response = await gateway._handle_generate(
            SimpleNamespace(headers={}, json=request_json)
        )
        assert response.status == 200, response.text
    captures = list(gateway.drain_window().captures.values())
    assert "logit_bias" not in forwarded[0]["sampling_params"]
    assert forwarded[1]["sampling_params"]["logit_bias"] == {2: -100.0}
    assert [capture.logit_bias for capture in captures] == [{}, {2: -100.0}]
    assert [capture.summary_request for capture in captures] == [False, True]
