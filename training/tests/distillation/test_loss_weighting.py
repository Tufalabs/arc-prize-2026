import json
import tempfile
import unittest
from pathlib import Path

import torch

from sft.distillation.accumulation import (
    RouterGradientAccumulator,
    validate_resume_manifest,
)
from sft.distillation.config import load_config
from sft.distillation.gradient_noise import GradientNoise, local_moments
from sft.distillation.loss import objective_weights
from sft.distillation.train import worker_command
from sft.distillation.wandb_logging import aggregate_step


class LossWeightingTests(unittest.TestCase):
    def test_distributed_accumulation_matches_direct_objective_and_adam(self):
        # Unequal completion lengths, GA1/4/8, plus an actual three-micro tail.
        for micros in (1, 4, 8, 3):
            for policy in ("token", "example"):
                with self.subTest(micros=micros, policy=policy):
                    g = torch.Generator().manual_seed(91)
                    samples = [
                        torch.randn(1 + i * 2, 3, generator=g, dtype=torch.float64)
                        for i in range(4 * micros)
                    ]
                    initial = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
                    ref = torch.nn.Parameter(initial.clone())
                    losses = [((x @ ref) * 1.7).square().sum() for x in samples]
                    total = sum(map(len, samples))
                    objective = (
                        sum(losses) / total
                        if policy == "token"
                        else sum(v / len(x) for v, x in zip(losses, samples))
                        / len(samples)
                    )
                    objective.backward()
                    reduced = torch.zeros_like(ref)
                    moments = torch.zeros(2, dtype=torch.float64)
                    for rank in range(4):
                        p = torch.nn.Parameter(initial.clone())
                        routers = {"r": p}
                        acc = RouterGradientAccumulator()
                        for micro in range(micros):
                            x = samples[4 * micro + rank]
                            scale, weight = objective_weights(
                                policy, len(x), total, len(samples), 4
                            )
                            (((x @ p) * 1.7).square().sum() * scale).backward()
                            moments += local_moments(
                                routers, len(x), total, 4, sequence_weight=weight
                            )
                            acc.add(routers)
                        acc.assign(routers)
                        reduced += p.grad / 4
                    torch.testing.assert_close(
                        reduced, ref.grad, rtol=1e-12, atol=1e-12
                    )
                    p.grad = reduced
                    report = GradientNoise(routers).finish(routers, moments)
                    weights = [
                        len(x) / total if policy == "token" else 1 / len(samples)
                        for x in samples
                    ]
                    self.assertAlmostEqual(
                        report["effective_sequences"], 1 / sum(w * w for w in weights)
                    )
                    # Independently compute sequence-gradient variance, not using helper scales.
                    grads = [2 * 1.7**2 * x.T @ (x @ initial) / len(x) for x in samples]
                    mean = sum(w * grad for w, grad in zip(weights, grads))
                    second = sum(
                        w * grad.square().sum() for w, grad in zip(weights, grads)
                    )
                    q = sum(w * w for w in weights)
                    expected_noise = q * (second - mean.square().sum()) / (1 - q)
                    self.assertAlmostEqual(
                        report["scopes"]["global"]["raw"]["noise_power"],
                        expected_noise.item(),
                        places=10,
                    )
                    for param in (p, ref):
                        torch.nn.utils.clip_grad_norm_([param], 1.0)
                        torch.optim.AdamW([param], lr=1e-5, weight_decay=0).step()
                    torch.testing.assert_close(p, ref, rtol=1e-12, atol=1e-12)

    def test_equal_examples_invariant_to_repeated_identical_tokens(self):
        for policy, expected in [("example", 2.0), ("token", 2.6)]:
            p = torch.tensor(1.0, requires_grad=True)
            lengths = [1, 4]
            for n, value in zip(lengths, [1.0, 3.0]):
                scale, _ = objective_weights(policy, n, 5, 2, 1)
                (p * value * n * scale).backward()
            self.assertAlmostEqual(p.grad.item(), expected, places=6)

    def test_resume_legacy_token_and_objective_mismatch(self):
        validate_resume_manifest({}, {"loss_weighting": "token"})
        validate_resume_manifest(
            {"loss_weighting": "example"}, {"loss_weighting": "example"}
        )
        for old, new in [
            ({}, {"loss_weighting": "example"}),
            ({"loss_weighting": "example"}, {"loss_weighting": "token"}),
        ]:
            for allow in [False, True]:
                with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                    validate_resume_manifest(old, new, allow)

    def test_config_and_command(self):
        c = {
            "teacher": "/t",
            "student": "/s",
            "data": "/d",
            "data_sha256": "sha",
            "cache": "/c",
            "probe": {"cache": "/v"},
            "wandb": {"mode": "offline"},
            "steps": 2,
            "save_every": 1,
            "chunk": 64,
            "seed": 1,
            "lr": 1e-5,
            "temperature": 1.0,
            "ce_weight": 0.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            for policy in [None, "token", "example"]:
                cfg = c if policy is None else dict(c, loss_weighting=policy)
                p.write_text(json.dumps(cfg))
                loaded = load_config(p)
                self.assertEqual(loaded["loss_weighting"], policy or "example")
                cmd = worker_command(loaded, Path("/result"))
                direct = worker_command(cfg, Path("/result"))
                self.assertEqual(
                    direct[direct.index("--loss-weighting") + 1], policy or "example"
                )
                if policy is None:
                    with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                        validate_resume_manifest(
                            {}, {"loss_weighting": loaded["loss_weighting"]}
                        )
                self.assertEqual(
                    cmd[cmd.index("--loss-weighting") + 1], policy or "example"
                )
            p.write_text(json.dumps(dict(c, loss_weighting="invalid")))
            with self.assertRaisesRegex(ValueError, "loss_weighting"):
                load_config(p)

    def test_objective_logging_preserves_token_metric(self):
        rows = [
            {
                "step": 1,
                "loss_per_token": 0.2,
                "loss_per_example": 0.1,
                "objective_loss": 0.1,
                "loss_weighting": "example",
                "global_examples": 16,
                "router_grad_norm": 0.3,
                "seconds": 10,
                "global_completion_tokens": 100,
                "sequence_length": 1000,
                "peak_allocated_gib": 1,
                "peak_reserved_gib": 2,
            }
            for _ in range(4)
        ]
        m = aggregate_step(rows)
        self.assertEqual(m["train/objective_loss"], 0.1)
        self.assertEqual(m["train/loss_per_example"], 0.1)
        self.assertEqual(m["train/loss_per_completion_token"], 0.2)
        self.assertEqual(m["train/global_examples"], 16)
