"""Two-rank FSDP placement must preserve frozen routing and adapter gradients."""

import copy
import os
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from training.flash.dense_lora import LoRALinear

import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = LoRALinear(nn.Linear(32, 32, bias=False, dtype=torch.bfloat16), 4, 8)
        self.router = nn.Linear(32, 12, bias=False).requires_grad_(False)
        with torch.no_grad():
            self.projection.lora_B.normal_(0, 0.05)

    def forward(self, x):
        hidden = self.projection(x)
        probabilities = F.linear(hidden, self.router.weight.to(hidden.dtype)).float().softmax(-1)
        return (hidden.float().square().sum(-1) * probabilities[:, 0]).mean()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FrozenRouterReplicationTests(unittest.TestCase):
    def test_two_rank_nested_checkpoint_fsdp(self):
        if int(os.environ.get("WORLD_SIZE", "1")) != 2:
            self.skipTest("Run with torchrun --nproc_per_node=2")

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        try:
            torch.manual_seed(193)
            original = ToyLayer().cuda()
            input_value = torch.randn(257, 32, device="cuda", dtype=torch.bfloat16)
            results = []
            for replicate in (False, True):
                layer = copy.deepcopy(original)
                trainable = [layer.projection.lora_A, layer.projection.lora_B]
                ignored = set(trainable)
                if replicate:
                    ignored.add(layer.router.weight)
                wrapped = checkpoint_wrapper(layer)
                fully_shard(wrapped, ignored_params=ignored)
                root = nn.Sequential(wrapped)
                fully_shard(root, ignored_params=ignored)
                self.assertEqual(isinstance(layer.router.weight, DTensor), not replicate)
                x = input_value.clone().requires_grad_()
                loss = root(x)
                loss.backward()
                self.assertIsNone(layer.router.weight.grad)
                results.append((loss.detach(), x.grad.clone(), *(p.grad.clone() for p in trainable)))
            for ordinary, replicated in zip(*results):
                torch.testing.assert_close(ordinary, replicated, rtol=0, atol=0)
            dist.barrier()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
