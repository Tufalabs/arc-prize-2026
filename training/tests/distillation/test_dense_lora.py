import copy
import types
import unittest

import torch
from torch import nn

from training.flash.dense_lora import ALL_TARGETS, LoRALinear, dense_lora_policy
from tests.distillation.numerical import check_op


class DenseLoRATests(unittest.TestCase):
    def test_zero_initialization_and_gradients(self):
        torch.manual_seed(4)
        base = nn.Linear(8, 12)
        x = torch.randn(3, 8, requires_grad=True)
        expected = base(x)
        wrapped = LoRALinear(base, 3, 6)
        torch.testing.assert_close(wrapped(x), expected, rtol=0, atol=0)
        wrapped(x).square().sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNone(base.bias.grad)
        self.assertGreater(wrapped.lora_B.grad.abs().sum().item(), 0)
        self.assertIsNotNone(x.grad)

    def test_target_mapping_and_validation(self):
        for parent, leaves in (
            ("self_attn", ALL_TARGETS[:4]),
            ("linear_attn", ALL_TARGETS[4:7]),
            ("shared_expert", ALL_TARGETS[7:]),
        ):
            layer = nn.Module()
            module = nn.Module()
            for leaf in leaves:
                module.add_module(leaf, nn.Linear(4, 4, bias=False))
            if parent == "shared_expert":
                layer.mlp = nn.Module()
                layer.mlp.shared_expert = module
                layer.mlp.experts = nn.Linear(4, 4).requires_grad_(False)
            else:
                layer.add_module(parent, module)
            settings = types.SimpleNamespace(r=2, alpha=4, target_modules=[leaves[0]])
            parameters = dense_lora_policy(
                settings.r, settings.alpha, settings.target_modules
            ).configure_layer(layer, 0)
            self.assertEqual(len(parameters), 2)
            # Leaf names are strings from the target table.
            self.assertIsInstance(getattr(module, leaves[0]), LoRALinear)
            self.assertIsInstance(getattr(module, leaves[1]), nn.Linear)
        for targets in ([], ["qkv_proj"], ["gate_up_proj"], ["q_proj", "typo"]):
            with self.assertRaises(ValueError):
                dense_lora_policy(target_modules=targets)
        with self.assertRaises(ValueError):
            dense_lora_policy(r=0)

    def test_group_ranks_and_routed_experts_are_opt_in(self):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.shared_expert = nn.Module()
        layer.mlp.shared_expert.gate_proj = nn.Linear(8, 4, bias=False)
        targets = ["q_proj", "gate_proj"]
        default = dense_lora_policy(2, 4, targets)
        self.assertIsNone(default.expert_lora)
        policy = dense_lora_policy(2, 4, targets, group_ranks={"mlp.shared_expert": (3, 6)},
                                   expert_lora=dict(rank=4, alpha=8, dropout=0))
        self.assertEqual(policy.mode, "custom")
        self.assertEqual(policy.expert_lora, dict(rank=4, alpha=8.0, dropout=0.0))
        parameters = policy.configure_layer(layer, 0)
        self.assertEqual(parameters["self_attn.q_proj.lora_A"].shape, (2, 8))
        self.assertEqual(parameters["mlp.shared_expert.gate_proj.lora_A"].shape, (3, 8))
        self.assertEqual(layer.mlp.shared_expert.gate_proj.scale, 2.0)
        for bad in ({"experts": (2, 4)}, {"self_attn": (0, 4)}, {"self_attn": 2}):
            with self.assertRaises(ValueError):
                dense_lora_policy(2, 4, targets, group_ranks=bad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_fused_epilogue_and_fp32_masters(self):
        torch.manual_seed(229)
        base = nn.Linear(2560, 2560, bias=False, device="cuda", dtype=torch.float16)
        eager = LoRALinear(base, 8, 16)
        with torch.no_grad():
            eager.lora_B.normal_(0, 0.02)
        fused = copy.deepcopy(eager)
        fused.fuse_epilogue = True
        high = copy.deepcopy(eager).float()
        x = torch.randn(1, 257, 2560, device="cuda", dtype=torch.float16)
        check_op(self, lambda x: (high if x.dtype == torch.float32 else eager)(x), fused,
                 [x], name="dense_lora_fp16")
        # Compare adapter-master adjoints too, using a shared upstream.
        y, ref, oracle = fused(x), eager(x), high(x.float())
        upstream = torch.randn_like(y)
        def grads(module, output, gradient):
            return torch.autograd.grad(output, (module.lora_A, module.lora_B), gradient)
        a, b, c = grads(fused, y, upstream), grads(eager, ref, upstream), grads(high, oracle, upstream.float())
        for actual, expected, fp32 in zip(a, b, c):
            self.assertLessEqual((actual - fp32).square().mean().sqrt(),
                                 1.5 * (expected - fp32).square().mean().sqrt())
            self.assertLessEqual((actual - fp32).abs().max(), 1.5 * (expected - fp32).abs().max())

    def test_projection_dtype_rejections(self):
        module = LoRALinear(nn.Linear(8, 12), 3, 6, fuse_epilogue=True)
        for dtype in (torch.float64, torch.float8_e4m3fn, torch.int32):
            with self.assertRaisesRegex(TypeError, "LoRALinear"):
                module(torch.ones(3, 8).to(dtype))
        with self.assertRaisesRegex(TypeError, "mixed projection"):
            module(torch.ones(3, 8, dtype=torch.float16))
        module.base.double()
        with self.assertRaisesRegex(TypeError, "LoRALinear"):
            module(torch.ones(3, 8))
