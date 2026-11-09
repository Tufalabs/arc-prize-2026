import unittest

import torch
from sft.distillation.update_ratio import router_update_ratio, snapshot_routers
from sft.distillation.wandb_logging import aggregate_step


class UpdateRatioTests(unittest.TestCase):
    def test_global_ratio_not_average_of_layer_ratios(self):
        routers = {
            "a": torch.nn.Parameter(torch.tensor([3.0])),
            "b": torch.nn.Parameter(torch.tensor([4.0])),
        }
        before = snapshot_routers(routers)
        with torch.no_grad():
            routers["a"].add_(0.3)
            routers["b"].add_(0.8)
        self.assertAlmostEqual(
            router_update_ratio(routers, before), (0.3**2 + 0.8**2) ** 0.5 / 5, places=6
        )

    def test_actual_adamw_after_clipping_and_no_training_change(self):
        routers = {
            "a": torch.nn.Parameter(torch.tensor([1.0, 2.0])),
            "b": torch.nn.Parameter(torch.tensor([3.0])),
        }
        reference = {
            k: torch.nn.Parameter(v.detach().clone()) for k, v in routers.items()
        }
        optimizer = torch.optim.AdamW(
            list(routers.values()), lr=0.01, weight_decay=0.1, foreach=False
        )
        reference_optimizer = torch.optim.AdamW(
            list(reference.values()), lr=0.01, weight_decay=0.1, foreach=False
        )
        for step in range(3):
            for k, p in routers.items():
                p.grad = torch.full_like(p, 10.0 + step)
                reference[k].grad = p.grad.clone()
            torch.nn.utils.clip_grad_norm_(list(routers.values()), 1.0)
            torch.nn.utils.clip_grad_norm_(list(reference.values()), 1.0)
            before = snapshot_routers(routers)
            vector_before = torch.cat(
                [p.detach().flatten().double() for p in routers.values()]
            )
            rng_before = torch.get_rng_state().clone()
            optimizer.step()
            reference_optimizer.step()
            grads = [p.grad.clone() for p in routers.values()]
            ratio = router_update_ratio(routers, before)
            vector_after = torch.cat(
                [p.detach().flatten().double() for p in routers.values()]
            )
            expected = torch.linalg.vector_norm(
                vector_after - vector_before
            ) / torch.linalg.vector_norm(vector_before)
            self.assertAlmostEqual(ratio, expected.item(), places=10)
            torch.testing.assert_close(
                torch.get_rng_state(), rng_before, rtol=0, atol=0
            )
            for (k, p), grad in zip(routers.items(), grads):
                torch.testing.assert_close(p, reference[k], rtol=0, atol=0)
                torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
                for field, value in optimizer.state[p].items():
                    torch.testing.assert_close(
                        value,
                        reference_optimizer.state[reference[k]][field],
                        rtol=0,
                        atol=0,
                    )

    def test_no_update_and_zero_denominator(self):
        routers = {"a": torch.nn.Parameter(torch.ones(2))}
        self.assertEqual(router_update_ratio(routers, snapshot_routers(routers)), 0.0)
        routers = {"a": torch.nn.Parameter(torch.zeros(2))}
        self.assertIsNone(router_update_ratio(routers, snapshot_routers(routers)))

    def test_logging_once_from_rank_zero_and_legacy_compatibility(self):
        row = {
            "step": 1,
            "seconds": 1,
            "loss_per_token": 0.1,
            "router_grad_norm": 0.3,
            "global_completion_tokens": 4,
            "sequence_length": 4,
            "peak_allocated_gib": 1,
            "peak_reserved_gib": 1,
        }
        rows = [dict(row) for _ in range(4)]
        key = "train/router_update_ratio"
        self.assertNotIn(key, aggregate_step(rows))
        rows[0]["router_update_ratio"] = 0.0123
        self.assertEqual(aggregate_step(rows)[key], 0.0123)
        rows[0]["router_update_ratio"] = None
        self.assertNotIn(key, aggregate_step(rows))
