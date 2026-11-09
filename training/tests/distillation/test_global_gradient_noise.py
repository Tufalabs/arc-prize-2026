import copy
import unittest

import torch

from sft.distillation.gradient_noise import GradientNoise, local_moments, wandb_metrics
from sft.distillation.wandb_logging import aggregate_step


class GlobalGradientNoiseTests(unittest.TestCase):
    def test_single_rank_multiple_accumulated_sequences_is_supported(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([0.5])
        actual = local_moments(
            {"router": parameter}, 3, 7, 1, sequence_weight=3 / 7
        )
        self.assertTrue(torch.isfinite(actual).all())

    def report(self, gradients, lengths):
        routers = {
            "a": torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)),
            "b": torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)),
        }
        total, world = sum(lengths), len(lengths)
        moments = []
        for g, n in zip(gradients, lengths):
            for p, value in zip(routers.values(), g):
                p.grad = torch.tensor([world * n / total * value], dtype=torch.float64)
            before = [p.grad.clone() for p in routers.values()]
            moments.append(local_moments(routers, n, total, world))
            for p, old in zip(routers.values(), before):
                torch.testing.assert_close(p.grad, old, rtol=0, atol=0)
        mean = sum(
            torch.tensor(g, dtype=torch.float64) * n / total
            for g, n in zip(gradients, lengths)
        )
        for p, value in zip(routers.values(), mean):
            p.grad = value.reshape(1).clone()
        estimator = GradientNoise(routers)
        reduced = torch.stack(moments).sum(0)
        self.assertEqual(reduced.numel(), 2)
        before = [p.grad.clone() for p in routers.values()]
        report = estimator.finish(routers, reduced)
        for p, old in zip(routers.values(), before):
            torch.testing.assert_close(p.grad, old, rtol=0, atol=0)
        return routers, estimator, reduced, report

    def test_known_global_snr(self):
        # Mean=(2,0), sample covariance trace=4/3, mean noise=1/3.
        _, _, _, report = self.report([(1, 0), (3, 0), (1, 0), (3, 0)], [1] * 4)
        self.assertEqual(set(report["scopes"]), {"global"})
        raw = report["scopes"]["global"]["raw"]
        self.assertAlmostEqual(raw["noise_power"], 1 / 3)
        self.assertAlmostEqual(raw["signal_power"], 11 / 3)
        self.assertAlmostEqual(raw["signal_to_noise"], 11)

    def test_unequal_weights_and_all_parameters(self):
        gradients, lengths = [(1, 2), (3, 4), (2, 1), (4, 3)], [1, 2, 3, 4]
        _, _, _, report = self.report(gradients, lengths)
        weights = [x / 10 for x in lengths]
        q = sum(w * w for w in weights)
        mean = [sum(w * g[j] for w, g in zip(weights, gradients)) for j in range(2)]
        m = sum(x * x for x in mean)
        a = sum(w * sum(x * x for x in g) for w, g in zip(weights, gradients))
        noise = q * (a - m) / (1 - q)
        raw = report["scopes"]["global"]["raw"]
        self.assertAlmostEqual(raw["noise_power"], noise)
        self.assertAlmostEqual(raw["signal_to_noise"], (m - noise) / noise)
        self.assertAlmostEqual(report["effective_sequences"], 1 / q)

    def test_unresolved_and_zero_noise(self):
        for gradients, signal_resolved in [
            ([(1, 0), (-1, 0)] * 2, False),
            ([(1, 2)] * 4, True),
        ]:
            _, _, _, report = self.report(gradients, [1] * 4)
            raw = report["scopes"]["global"]["raw"]
            self.assertEqual(raw["signal_resolved"], signal_resolved)
            self.assertIsNone(raw["signal_to_noise"])
            self.assertNotIn(
                "train/gradient_noise/global/raw/signal_to_noise", wandb_metrics(report)
            )

    def test_resume_ema_and_logging(self):
        routers, estimator, moments, report = self.report([(1, 0), (3, 0)] * 2, [1] * 4)
        restored = GradientNoise(routers)
        restored.restore(copy.deepcopy(estimator.state))
        self.assertEqual(
            estimator.finish(routers, moments), restored.finish(routers, moments)
        )
        row = {
            "step": 1,
            "seconds": 1,
            "loss_per_token": 0.1,
            "router_grad_norm": 2,
            "global_completion_tokens": 4,
            "sequence_length": 4,
            "peak_allocated_gib": 1,
            "peak_reserved_gib": 1,
            "global_gradient_noise": report,
        }
        logged = aggregate_step([row] * 4)
        self.assertAlmostEqual(
            logged["train/gradient_noise/global/raw/signal_to_noise"], 11
        )
        self.assertFalse(any("layer" in key for key in wandb_metrics(report)))
