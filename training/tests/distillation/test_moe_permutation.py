import unittest
from types import SimpleNamespace

import torch

from tests.distillation.numerical import check_op
from training.flash.moe_permutation import combine_experts, gather_tokens
from training.flash.pointwise_optimizations import expert_forward_with_weights


class PermutationTests(unittest.TestCase):
    def check(self, device, dtype, n=19, k=3, d=64):
        torch.manual_seed(9)
        indices = torch.randint(0, 5, (n, k), device=device)
        _, permutation = indices.flatten().sort()
        inverse = permutation.argsort()
        x = torch.randn(n, d, device=device, dtype=dtype, requires_grad=True)
        weights = torch.randn(n, k, device=device, dtype=dtype, requires_grad=True)
        selected = gather_tokens(x, permutation, k)
        expected = x[permutation // k]
        upstream = torch.randn_like(selected)
        torch.testing.assert_close(selected, expected, atol=0, rtol=0)
        actual_gradient = torch.autograd.grad(selected, x, upstream)[0]
        expected_gradient = torch.autograd.grad(expected, x, upstream)[0]
        if dtype == torch.bfloat16:
            oracle = upstream[inverse].float().view(n, k, d).sum(1)
            torch.testing.assert_close(actual_gradient, oracle.to(dtype), atol=0, rtol=0)
            self.assertLessEqual((actual_gradient.float() - oracle).square().sum().item(),
                                 (expected_gradient.float() - oracle).square().sum().item())
        else:
            torch.testing.assert_close(actual_gradient, expected_gradient, atol=0.04, rtol=0.02)
        projected = torch.randn(n * k, d, device=device, dtype=dtype, requires_grad=True)
        actual = combine_experts(projected, weights, permutation)
        reference = (projected * weights.flatten()[permutation, None])[inverse]
        reference = reference.view(n, k, d).sum(1)
        torch.testing.assert_close(actual, reference, atol=0.04, rtol=0.02)
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (projected, weights), upstream)
        expected_grads = torch.autograd.grad(reference, (projected, weights), upstream)
        for a, b in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, b, atol=0.04, rtol=0.02)

    def test_cpu(self):
        self.check("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda(self):
        self.check("cuda", torch.bfloat16)
        self.check("cuda", torch.bfloat16, n=257, k=10, d=2560)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_production_fp32_oracle_and_packed_documents(self):
        torch.manual_seed(151)
        for lengths in ((4096,), (1, 73, 183)):
            n, k, d = sum(lengths), 10, 2560
            indices = torch.randint(12, (n, k), device="cuda")
            permutation = indices.flatten().sort().indices
            inverse = permutation.argsort()
            x = torch.randn(n, d, device="cuda", dtype=torch.bfloat16)
            projected = torch.randn(n * k, d, device="cuda", dtype=torch.bfloat16)
            weights = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            check_op(self, lambda x: x[permutation // k],
                     lambda x: gather_tokens(x, permutation, k), [x], name="gather")
            def eager(x, w):
                return (x * w.flatten()[permutation, None])[inverse].view(n, k, d).sum(1)
            check_op(self, eager, lambda x, w: combine_experts(x, w, permutation),
                     [projected, weights], name="combine")
            # A packed stream is just independent token rows for these operators.
            # Re-sort each document, then concatenate to check both adjoints too.
            def documents(x, w):
                outputs = []
                for part, weight, index in zip(x.split(lengths), w.split(lengths), indices.split(lengths)):
                    order = index.flatten().sort().indices
                    selected = gather_tokens(part, order, k)
                    outputs.append(combine_experts(selected, weight, order))
                return torch.cat(outputs)
            check_op(self, lambda x, w: (x[:, None] * w[..., None]).sum(1),
                     documents, [x, weights], name="packed_gather_combine")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp16_and_promoted_weights(self):
        torch.manual_seed(181)
        n, k, d = 257, 10, 2560
        permutation = torch.randperm(n * k, device="cuda")
        inverse = permutation.argsort()
        x = torch.randn(n, d * 2, device="cuda", dtype=torch.float16)[:, ::2]
        projected = torch.randn(n * k, d * 2, device="cuda", dtype=torch.float16)[:, ::2]
        check_op(self, lambda x: x[permutation // k],
                 lambda x: gather_tokens(x, permutation, k), [x], name="gather_fp16")
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            weights = torch.randn(n, k * 2, device="cuda", dtype=dtype)[:, ::2]
            check_op(self, lambda x, w: (x[inverse].view(n, k, d) * w[..., None]).sum(1),
                     lambda x, w: combine_experts(x, w, permutation),
                     [projected, weights], name="combine_promoted")

    def test_dtype_and_device_rejections(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            order = torch.arange(6, device=device)
            for dtype in (torch.float64, torch.float8_e4m3fn, torch.int32):
                x = torch.ones(2, 8, device=device).to(dtype)
                w = torch.ones(2, 3, device=device)
                with self.assertRaisesRegex(TypeError, "gather_tokens"):
                    gather_tokens(x, order, 3)
                with self.assertRaisesRegex(TypeError, "combine_experts"):
                    combine_experts(x.repeat_interleave(3, 0), w, order)
            if device == "cuda":
                with self.assertRaisesRegex(TypeError, "FP32"):
                    gather_tokens(torch.ones(2, 8, device=device), order, 3)
                with self.assertRaisesRegex(ValueError, "indices"):
                    gather_tokens(torch.ones(2, 8, device=device, dtype=torch.float16),
                                  order.cpu(), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fp32_combine_reference_fallback(self):
        torch.manual_seed(211)
        n, k, d = 19, 3, 64
        order = torch.randperm(n * k, device="cuda")
        inverse = order.argsort()
        x = torch.randn(n * k, d, device="cuda", requires_grad=True)
        w = torch.randn(n, k, device="cuda", requires_grad=True)
        expected = (x[inverse].view(n, k, d) * w[..., None]).sum(1)
        actual = combine_experts(x, w, order)
        self.assertTrue(torch.equal(actual, expected))
        upstream = torch.randn_like(actual)
        a = torch.autograd.grad(actual, (x, w), upstream)
        b = torch.autograd.grad(expected, (x, w), upstream)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))

    def test_expert_adapter_preserves_final_cast_with_mixed_routes(self):
        torch.manual_seed(233)
        module = SimpleNamespace(num_experts=3, _apply_gate=torch.nn.functional.silu)
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(19, 64, dtype=dtype, requires_grad=True)
            weights = torch.randn(19, 3, requires_grad=True)
            indices = torch.randint(3, weights.shape)
            def linear(x, weight, offsets):
                return x * weight
            args = (module, x, indices, weights, 1.5, 0.7)
            actual = expert_forward_with_weights(*args, grouped_linear=linear, optimization="fused_permutation")
            self.assertEqual(actual.dtype, dtype)
            check_op(self, lambda x, w: expert_forward_with_weights(
                module, x, indices, w, 1.5, 0.7, grouped_linear=linear, optimization="no_sentinel"),
                lambda x, w: expert_forward_with_weights(
                module, x, indices, w, 1.5, 0.7, grouped_linear=linear, optimization="fused_permutation"),
                [x, weights], name="expert_mixed_routes")
