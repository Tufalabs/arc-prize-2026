import copy
import json
import unittest

import torch
from sft.distillation.accumulation import validate_resume_manifest
from sft.distillation.batching import BatchPlanner, normalize_batching
from sft.distillation.common import batch_indices


class BatchingTests(unittest.TestCase):
    def test_default_matches_explicit_length_bucket(self):
        lengths = list(range(1, 133))
        cfg = {"policy": "length_bucket", "version": 1, "pool_size": 128}
        self.assertEqual(normalize_batching(), cfg)
        self.assertEqual(normalize_batching({}), cfg)
        self.assertEqual(
            BatchPlanner(lengths, 4, 123).update(0, 66),
            BatchPlanner(lengths, 4, 123, cfg).update(0, 66),
        )

    def test_random_legacy_exact_and_rng_untouched(self):
        lengths = list(range(1, 133))
        planner = BatchPlanner(lengths, 4, 123, {"policy": "random"})
        state = torch.random.get_rng_state()
        for cursor in [0, 1, 32, 33, 65, 3]:
            self.assertEqual(
                planner.update(cursor, 3),
                [batch_indices(132, 4, c, 123) for c in range(cursor, cursor + 3)],
            )
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_coverage_short_pool_epoch_resume_and_ties(self):
        lengths = [1 + i % 31 for i in range(276)]
        cfg = {"policy": "length_bucket", "pool_size": 128}
        planner = BatchPlanner(lengths, 4, 35, cfg)
        all_groups = planner.update(0, 138)
        for groups in [all_groups[:69], all_groups[69:]]:
            self.assertEqual(sorted(i for g in groups for i in g), list(range(276)))
        self.assertNotEqual(all_groups[:69], all_groups[69:])
        for cursor in [1, 32, 68, 69, 100]:
            restored = BatchPlanner(
                lengths, 4, 35, json.loads(json.dumps(planner.config))
            )
            self.assertEqual(
                restored.update(cursor, 5), all_groups[cursor : cursor + 5]
            )
        self.assertEqual(
            BatchPlanner([7] * 132, 4, 35, cfg).update(0, 33),
            BatchPlanner([7] * 132, 4, 35, cfg).update(0, 33),
        )

    def test_bucket_reduces_max_cost_sum(self):
        lengths = list(range(1, 129))
        random = BatchPlanner(lengths, 4, 12, {"policy": "random"}).update(0, 32)
        bucket = BatchPlanner(lengths, 4, 12, {"policy": "length_bucket"}).update(0, 32)

        def cost(groups):
            return sum(max(lengths[i] ** 2 for i in group) for group in groups)

        self.assertLess(cost(bucket), cost(random))
        self.assertTrue(
            all(
                max(lengths[i] for i in g) - min(lengths[i] for i in g) == 3
                for g in bucket
            )
        )

    def test_within_update_preserves_multiset_across_epoch_and_partial(self):
        lengths = list(range(1, 21))
        random = BatchPlanner(lengths, 4, 75, {"policy": "random"})
        grouped = BatchPlanner(lengths, 4, 75, {"policy": "length_within_update"})
        for cursor, n in [(0, 4), (4, 3), (8, 1), (10, 4)]:
            a, b = random.update(cursor, n), grouped.update(cursor, n)
            self.assertEqual(
                sorted(i for g in a for i in g), sorted(i for g in b for i in g)
            )
            self.assertEqual(
                b, BatchPlanner(lengths, 4, 75, grouped.config).update(cursor, n)
            )

    def test_token_weighted_gradient_update_invariant(self):
        lengths = list(range(1, 17))
        samples = [
            torch.linspace(-1, 1, n, dtype=torch.float64) + i / 10
            for i, n in enumerate(lengths)
        ]
        initial = torch.tensor([0.4, -0.2], dtype=torch.float64)
        results = []
        for policy in ["random", "length_within_update"]:
            p = initial.clone().requires_grad_()
            opt = torch.optim.AdamW([p], lr=0.01, weight_decay=0)
            groups = BatchPlanner(lengths, 4, 12, {"policy": policy}).update(0, 4)
            loss = 0.0
            for group in groups:
                for i in group:
                    term = ((samples[i] * p[0] + p[1]) ** 2).sum() / sum(lengths)
                    term.backward()
                    loss += term.detach()
            grad = p.grad.clone()
            opt.step()
            results.append((loss, grad, p.detach().clone()))
        for a, b in zip(*results, strict=True):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_manifest_legacy_and_reject_policy_change(self):
        old = {"gradient_accumulation": 1}
        new = dict(old, batching=normalize_batching({"policy": "random"}))
        with self.assertRaisesRegex(ValueError, "manifest"):
            validate_resume_manifest(old, dict(old, batching=normalize_batching()))
        self.assertFalse(validate_resume_manifest(old, new))
        for cfg in [{"policy": "length_bucket"}, {"policy": "length_within_update"}]:
            changed = dict(old, batching=normalize_batching(cfg))
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate_resume_manifest(old, changed)
        a = dict(
            old,
            batching=normalize_batching({"policy": "length_bucket", "pool_size": 128}),
        )
        b = copy.deepcopy(a)
        b["batching"]["pool_size"] = 256
        with self.assertRaises(ValueError):
            validate_resume_manifest(a, b)

    def test_validation(self):
        for cfg in [
            {"policy": "bad"},
            {"policy": "random", "pool_size": 128},
            {"policy": "length_bucket", "pool_size": 7},
            {"policy": "length_bucket", "pool_size": True},
            {"version": 2},
        ]:
            with self.assertRaises(ValueError):
                normalize_batching(cfg)
        for lengths in [[1] * 3, [1] * 5, [0] * 4]:
            with self.assertRaises(ValueError):
                BatchPlanner(lengths, 4, 1)
        planner = BatchPlanner([1] * 4, 4, 1)
        for cursor, n in [(-1, 1), (0, 0), (True, 1)]:
            with self.assertRaises(ValueError):
                planner.update(cursor, n)
