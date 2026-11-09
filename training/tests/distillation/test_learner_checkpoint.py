import copy
import unittest

import torch

from tests.distillation.test_qsa_selection import ToySelector
from training.flash.offload import checkpoint_layer
from training.flash.policies import DistributionPolicy


class SelectiveCheckpointTests(unittest.TestCase):
    def test_small_gemm_replay(self):
        torch.manual_seed(3)
        eager = ToySelector()
        selective = copy.deepcopy(eager)
        wrapped = checkpoint_layer(selective, 0, 0, reuse_qsa_selection=False,
                                   checkpoint_policy="small_gemm")
        for n in (7, 13):
            x = torch.randn(1, n, 4, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            eager(x).backward()
            wrapped(y).backward()
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
            torch.testing.assert_close(eager.weight.grad, selective.weight.grad, rtol=0, atol=0)
        self.assertEqual((selective.calls, selective.reuses), (4, 0))

    def test_policy_validation(self):
        self.assertTrue(DistributionPolicy().reshard_after_forward)
        for kwargs in ({"forward_prefetch": -1}, {"backward_prefetch": True},
                       {"reshard_after_forward": 0}):
            with self.assertRaises(ValueError):
                DistributionPolicy(**kwargs)
