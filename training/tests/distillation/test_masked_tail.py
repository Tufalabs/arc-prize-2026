import unittest

import torch
from sft.distillation.accumulation import validate_resume_manifest
from sft.distillation.batching import BatchPlanner, normalize_batching
from sft.distillation.loss import objective_weights


class MaskedTailTests(unittest.TestCase):
    def test_full_dataset_coverage_and_resume(self):
        for policy in ("random", "length_bucket"):
            cfg = {"policy": policy, "tail_policy": "masked"}
            p = BatchPlanner(list(range(1, 16167)), 4, 12, cfg)
            self.assertEqual(p.batches, 4042)
            for epoch in range(3):
                groups = p.update(epoch * p.batches, p.batches)
                flat = [i for group in groups for i in group]
                self.assertEqual(sorted(i for i in flat if i >= 0), list(range(16166)))
                self.assertEqual(flat.count(-1), 2)
            restored = BatchPlanner(p.lengths, 4, 12, p.config)
            self.assertEqual(restored.update(4039, 8), p.update(4039, 8))

    def test_legacy_order_and_resume_guard(self):
        for policy in ("random", "length_bucket"):
            old = BatchPlanner(list(range(1, 133)), 4, 12, {"policy": policy})
            new = BatchPlanner(
                old.lengths, 4, 12, {"policy": policy, "tail_policy": "masked"}
            )
            self.assertEqual(old.update(0, 99), new.update(0, 99))
            with self.assertRaises(ValueError):
                validate_resume_manifest(
                    {"batching": old.config}, {"batching": new.config}
                )
        with self.assertRaises(ValueError):
            BatchPlanner([1] * 6, 4, 12)
        with self.assertRaises(ValueError):
            normalize_batching(
                {"policy": "length_within_update", "tail_policy": "masked"}
            )

    def test_four_rank_masked_gradient_and_adam_match_unpadded_reference(self):
        for n in (5, 6, 7, 134):
            lengths = [1 + i % 9 for i in range(n)]
            samples = [
                torch.linspace(-1, 1, k, dtype=torch.float64) + i / 100
                for i, k in enumerate(lengths)
            ]
            planner = BatchPlanner(lengths, 4, 123, {"tail_policy": "masked"})
            for ga in (1, 4, 8):
                for cursor in range(0, planner.batches, ga):
                    groups = planner.update(cursor, min(ga, planner.batches - cursor))
                    real = [i for g in groups for i in g if i >= 0]
                    tokens = sum(lengths[i] for i in real)
                    for policy in ("token", "example"):
                        initial = torch.tensor([0.4, -0.2], dtype=torch.float64)
                        ref = initial.clone().requires_grad_()
                        reference = sum(
                            ((samples[i] * ref[0] + ref[1]) ** 2).sum()
                            / (tokens if policy == "token" else len(real) * lengths[i])
                            for i in real
                        )
                        reference.backward()
                        rank_grads = []
                        for rank in range(4):
                            p = initial.clone().requires_grad_()
                            for group in groups:
                                index = group[rank]
                                active = index >= 0
                                loaded = (
                                    index
                                    if active
                                    else min(
                                        (i for i in group if i >= 0),
                                        key=lengths.__getitem__,
                                    )
                                )
                                scale, _ = objective_weights(
                                    policy, lengths[loaded], tokens, len(real), 4
                                )
                                loss = ((samples[loaded] * p[0] + p[1]) ** 2).sum()
                                (loss * (scale if active else 0.0)).backward()
                            rank_grads.append(p.grad)
                        actual = torch.stack(rank_grads).mean(0)
                        torch.testing.assert_close(
                            actual, ref.grad, atol=1e-12, rtol=1e-12
                        )
                        q = initial.clone().requires_grad_()
                        q.grad = actual
                        opt1 = torch.optim.AdamW([q], lr=0.01)
                        opt2 = torch.optim.AdamW([ref], lr=0.01)
                        opt1.step()
                        opt2.step()
                        torch.testing.assert_close(q, ref, atol=1e-12, rtol=1e-12)
