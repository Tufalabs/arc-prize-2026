"""Run with torchrun --standalone --nproc_per_node=2 -m unittest this module."""

import copy
import os
import unittest

import torch
import torch.distributed as dist

from unittest.mock import Mock, patch

from training.flash.gradient_sync import ReplicatedGradientReducer


@unittest.skipUnless(torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1,
                     "Two CUDA ranks required")
class GradientSyncTests(unittest.TestCase):
    def test_bucketed_gradients_match_end_of_backward_average(self):
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
        torch.manual_seed(7)
        layers = torch.nn.ModuleList([torch.nn.Linear(16, 16).cuda() for _ in range(3)])
        reference = copy.deepcopy(layers)
        reducer = ReplicatedGradientReducer([list(layer.parameters()) for layer in reversed(layers)])
        try:
            for step in range(4):
                microbatches = 1 if step < 2 else step
                for micro in range(microbatches):
                    torch.manual_seed(11 + rank + step + micro * 17)
                    x = torch.randn(32, 16, device="cuda")
                    actual, expected = x, x.clone()
                    for a, b in zip(layers, reference):
                        actual, expected = a(actual).tanh(), b(expected).tanh()
                    if micro < microbatches - 1:
                        with reducer.no_sync():
                            actual.square().mean().backward()
                    else:
                        actual.square().mean().backward()
                    expected.square().mean().backward()
                reducer.finish()
                for a, b in zip(layers.parameters(), reference.parameters()):
                    dist.all_reduce(b.grad, op=dist.ReduceOp.AVG)
                    torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-7)
                layers.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
            # Reuse this group: destroying/reinitializing NCCL between unittest
            # cases can race the rendezvous store and fail a peer's connect.
            from tests.distillation.test_flash_ppo import run_two_rank_flash_step

            run_two_rank_flash_step()
            run_two_rank_flash_step(train_routers=True)
        finally:
            reducer.close()
            dist.destroy_process_group()


class GradientSyncGuardTests(unittest.TestCase):
    def test_no_sync_and_submitted_or_repeated_contributions(self):
        for device in ("cpu",):
            p = torch.nn.Parameter(torch.ones(2, device=device))
            work = Mock()
            with patch("training.flash.gradient_sync.dist.all_reduce", return_value=work) as reduce:
                reducer = ReplicatedGradientReducer([[p]])
                try:
                    with reducer.no_sync():
                        with reducer.no_sync():
                            (p * 2).sum().backward()
                        (p * 3).sum().backward()
                    reduce.assert_not_called()
                    (p * 4).sum().backward()
                    self.assertTrue(torch.equal(reduce.call_args.args[0], torch.full_like(p, 9)))
                    reducer.finish()
                    self.assertTrue(torch.equal(p.grad, torch.full_like(p, 9)))
                    p.grad = None
                    p.sum().backward()
                    with self.assertRaisesRegex(RuntimeError, "no_sync"):
                        p.sum().backward()
                    with self.assertRaisesRegex(RuntimeError, "invalidated"):
                        reducer.finish()
                finally:
                    reducer.close()

    def test_duplicate_readiness_before_bucket_submission_is_rejected(self):
        p, q = (torch.nn.Parameter(torch.ones(2)) for _ in range(2))
        reducer = ReplicatedGradientReducer([[p, q]])
        try:
            p.sum().backward()
            with self.assertRaisesRegex(RuntimeError, "contribution to bucket 0"):
                p.sum().backward()
        finally:
            reducer.close()
