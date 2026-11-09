import copy
import unittest

import torch
from sft.distillation.accumulation import (
    RouterGradientAccumulator,
    checkpoint_cursor,
    microbatch_count,
    remaining_updates,
    validate_resume_manifest,
)
from sft.distillation.common import batch_indices
from sft.distillation.gradient_noise import GradientNoise, local_moments


class AccumulationTests(unittest.TestCase):
    def test_token_weighted_eight_sequence_gradient_and_adam_equivalence(self):
        generator = torch.Generator().manual_seed(31)
        samples = [
            torch.randn(n, 3, dtype=torch.float64, generator=generator)
            for n in range(1, 9)
        ]
        initial = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
        reference = torch.nn.Parameter(initial.clone())
        total = sum(len(x) for x in samples)
        # Frozen body transform still propagates gradients to the router.
        frozen = torch.tensor(1.7, dtype=torch.float64)
        loss = sum(((x @ reference) * frozen).square().sum() for x in samples) / total
        loss.backward()
        reduced_grad = torch.zeros_like(reference)
        reduced_moments = torch.zeros(2, dtype=torch.float64)
        for rank in range(4):
            p = torch.nn.Parameter(initial.clone())
            routers = {"router": p}
            accumulation = RouterGradientAccumulator()
            for micro in range(2):
                x = samples[micro * 4 + rank]
                (((x @ p) * frozen).square().sum() * 4 / total).backward()
                reduced_moments += local_moments(routers, len(x), total, 4)
                accumulation.add(routers)
                self.assertIsNone(p.grad)
            accumulation.assign(routers)
            reduced_grad += p.grad / 4
        torch.testing.assert_close(reduced_grad, reference.grad, rtol=1e-13, atol=1e-13)
        p.grad = reduced_grad
        report = GradientNoise(routers).finish(routers, reduced_moments)
        weights = torch.tensor([len(x) / total for x in samples], dtype=torch.float64)
        self.assertAlmostEqual(
            report["effective_sequences"], 1 / weights.square().sum().item()
        )
        torch.nn.utils.clip_grad_norm_([reference], 1.0)
        torch.nn.utils.clip_grad_norm_([p], 1.0)
        a = torch.optim.AdamW([reference], lr=1e-5, weight_decay=0)
        b = torch.optim.AdamW([p], lr=1e-5, weight_decay=0)
        a.step()
        b.step()
        torch.testing.assert_close(p, reference, rtol=1e-13, atol=1e-13)

    def test_cursor_transition_covers_remaining_dataset_once(self):
        start, total, world = 50, 1481, 4
        self.assertEqual(remaining_updates(start, total, 2), 716)
        cursor, visited = start, []
        for _ in range(716):
            micros = microbatch_count(cursor, 2, total)
            for m in range(micros):
                visited.extend(batch_indices(5924, world, cursor + m, 20260916))
            cursor += micros
        previous = [
            idx
            for step in range(start)
            for idx in batch_indices(5924, world, step, 20260916)
        ]
        self.assertEqual(cursor, 1481)
        self.assertEqual(len(visited), 5724)
        self.assertEqual(len(set(previous + visited)), 5924)
        self.assertEqual(micros, 1)
        self.assertEqual(50 + 716, 766)

    def test_resume_only_explicit_accumulation_change(self):
        old = {"lr": 1e-5, "world_size": 4, "student": "fixed"}
        new = {**old, "gradient_accumulation": 2}
        with self.assertRaisesRegex(ValueError, "explicit"):
            validate_resume_manifest(old, new)
        self.assertTrue(validate_resume_manifest(old, new, True))
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_resume_manifest(old, {**new, "lr": 2e-5}, True)
        self.assertFalse(
            validate_resume_manifest(old, {**old, "gradient_accumulation": 1})
        )
        saved = {"step": 50, "manifest": old}
        self.assertEqual(checkpoint_cursor(saved), 50)
        saved["manifest"] = new
        with self.assertRaises(ValueError):
            checkpoint_cursor(saved)
        saved["data_microbatch_cursor"] = 100
        self.assertEqual(checkpoint_cursor(saved), 100)
        snapshot = copy.deepcopy(saved)
        self.assertEqual(checkpoint_cursor(snapshot), 100)

    def test_ga1_and_empty_or_nonfinite_rejected(self):
        p = torch.nn.Parameter(torch.tensor([2.0]))
        (p.square().sum()).backward()
        expected = p.grad.clone()
        accumulator = RouterGradientAccumulator()
        accumulator.add({"r": p})
        accumulator.assign({"r": p})
        torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            microbatch_count(1481, 2, 1481)
        p.grad.fill_(float("nan"))
        with self.assertRaises(RuntimeError):
            RouterGradientAccumulator().add({"r": p})
