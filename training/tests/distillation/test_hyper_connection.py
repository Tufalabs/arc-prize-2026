"""Production-width forward/backward error against a common FP32 oracle."""

import copy
import unittest

import torch
import torch.nn.functional as F

from tests.distillation.numerical import (
    check_op, check_packed, error, same_bits, reference_norm, reference_mix, reference_inject,
)

from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen
from training.flash.hyper_connection import (
    ManualGatedResidual,
    TritonGatedResidual,
    grouped_norm,
    residual_inject,
    residual_inject_norm,
    scaled_activation,
    sigmoid_mean,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class HyperConnectionTests(unittest.TestCase):
    check_op = check_op

    def test_all_primitives_packed_adjoint_boundaries(self):
        torch.manual_seed(157)
        x = torch.randn(1, 257, 10240, device="cuda", dtype=torch.bfloat16)
        gain = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.1
        h = torch.randn(1, 257, 2560, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(1, 257, 4, device="cuda", dtype=torch.bfloat16)
        for serving in (False, True):
            check_packed(self, lambda x: grouped_norm(x, gain, 4, 1e-6, serving), [x])
            check_packed(self, lambda x, u: sigmoid_mean(x, u, 4, serving), [x, torch.randn_like(x)])
            check_packed(self, lambda x, h, w: residual_inject(x, h, w, serving), [x, h, weights])
            check_packed(self, lambda x, h, w: residual_inject_norm(x, h, w, gain, 1e-6, serving),
                         [x, h, weights])
            for inject in (False, True):
                # HC projections produce contiguous activation inputs.
                low = weights if inject else x[..., :320].contiguous()
                check_packed(self, lambda x: scaled_activation(x, 4, inject, serving), [low])

    def test_primitives_production_shapes_and_packed_tokens(self):
        torch.manual_seed(71)
        torch.backends.cuda.matmul.allow_tf32 = False
        d, c = 2560, 4
        for n in (1, 257, 4096, 32768, 75776):
            with self.subTest(tokens=n):
                x = torch.randn(1, n, c * d, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(c * d, device="cuda", dtype=torch.bfloat16) * 0.1
                self.check_op(lambda x, weight=weight: reference_norm(x, weight, c),
                              lambda x, weight=weight: grouped_norm(x, weight, c, 1e-6), [x])
                up = torch.randn_like(x)
                self.check_op(lambda x, up: reference_mix(x, up, c),
                              lambda x, up: sigmoid_mean(x, up, c), [x, up])
                h = torch.randn(1, n, d, device="cuda", dtype=torch.bfloat16)
                w = torch.rand(1, n, c, device="cuda", dtype=torch.bfloat16) * 2
                self.check_op(reference_inject, residual_inject, [x, h, w])
                def pair(x, h, w, weight=weight):
                    y = reference_inject(x, h, w)
                    return y, reference_norm(y, weight, c)
                self.check_op(pair, lambda x, h, w, weight=weight: residual_inject_norm(
                    x, h, w, weight, 1e-6), [x, h, w])
                # Splitting a packed stream at arbitrary document boundaries must
                # preserve the token-local computation, including one-token docs.
                if n == 257:
                    x.requires_grad_()
                    actual = grouped_norm(x, weight, c, 1e-6)
                    pieces = [grouped_norm(t, weight, c, 1e-6) for t in x.split((1, 73, 183), dim=1)]
                    packed = torch.cat(pieces, dim=1)
                    self.assertTrue(torch.equal(actual, packed))
                    dy = torch.randn_like(actual)
                    self.assertTrue(torch.equal(torch.autograd.grad(actual, x, dy)[0],
                                                torch.autograd.grad(packed, x, dy)[0]))
        for n in (257, 32768):
            for width, inject in ((320, False), (4, True)):
                x = torch.randn(1, n, width, device="cuda", dtype=torch.bfloat16) * 3
                eager = (lambda x: 2 * (x / c).sigmoid()) if inject else (lambda x: F.silu(x / c))
                self.check_op(eager, lambda x, inject=inject: scaled_activation(x, c, inject), [x])

    def test_complete_gated_residual(self):
        torch.manual_seed(79)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = qwen.Qwen4ExpTextConfig(hidden_size=2560, hc_count=4, hc_lowrank=320)
        for combine in (False, True):
            eager = qwen.Qwen4ExpTextGatedResidual(config, use_combine=combine).cuda().bfloat16()
            eager.requires_grad_(False)
            high = copy.deepcopy(eager).float()
            fused = TritonGatedResidual(config, use_combine=combine).cuda().bfloat16().requires_grad_(False)
            fused.load_state_dict(eager.state_dict())
            x = torch.randn(1, 4096, 10240, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            xf = x.detach().float().requires_grad_()
            def outputs(module, value, combine=combine):
                result = module(value)
                return (result[0], result[2]) if combine else (result,)
            a, b, ref = outputs(fused, x), outputs(eager, x), outputs(high, xf)
            upstream = tuple(torch.randn_like(v) for v in a)
            ga = torch.autograd.grad(a, x, upstream)[0]
            gb = torch.autograd.grad(b, x, upstream)[0]
            gr = torch.autograd.grad(ref, xf, tuple(v.float() for v in upstream))[0]
            for av, bv, rv in zip((*a, ga), (*b, gb), (*ref, gr)):
                self.assertLessEqual(error(av, rv), 1.5 * error(bv, rv))
                self.assertLessEqual((av.float() - rv).abs().max().item(),
                                     1.5 * (bv.float() - rv).abs().max().item())
            again = outputs(fused, x)
            ga2 = torch.autograd.grad(again, x, upstream)[0]
            self.assertTrue(same_bits(ga, ga2))
            self.assertTrue(all(same_bits(v, w) for v, w in zip(a, again)))

    def test_manual_gated_residual_and_injected_input(self):
        torch.manual_seed(97)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = qwen.Qwen4ExpTextConfig(hidden_size=2560, hc_count=4, hc_lowrank=320)
        for combine in (False, True):
            eager = qwen.Qwen4ExpTextGatedResidual(config, use_combine=combine).cuda().bfloat16()
            eager.requires_grad_(False)
            high = copy.deepcopy(eager).float()
            fused = ManualGatedResidual(config, use_combine=combine).cuda().bfloat16().requires_grad_(False)
            fused.load_state_dict(eager.state_dict())
            x = torch.randn(1, 4096, 10240, device="cuda", dtype=torch.bfloat16)
            def baseline(x, eager=eager, high=high):
                return (high if x.dtype == torch.float32 else eager)(x)
            self.check_op(baseline, fused, [x])
            if combine:
                h = torch.randn(1, 4096, 2560, device="cuda", dtype=torch.bfloat16)
                w = torch.rand(1, 4096, 4, device="cuda", dtype=torch.bfloat16)
                def baseline_injected(x, h, w, eager=eager, high=high):
                    return (high if x.dtype == torch.float32 else eager)(reference_inject(x, h, w))
                self.check_op(baseline_injected, lambda x, h, w, fused=fused: fused(
                    x, injection=(h, w)), [x, h, w])

    def test_fp16_primitives_and_fp32_norm_weights(self):
        torch.manual_seed(173)
        x = torch.randn(1, 257, 10240, device="cuda", dtype=torch.float16)
        up = torch.randn_like(x)
        h = torch.randn(1, 257, 2560, device="cuda", dtype=torch.float16)
        w = torch.rand(1, 257, 4, device="cuda", dtype=torch.float16)
        for wdtype in (torch.float16, torch.float32):
            gain = torch.randn(10240, device="cuda", dtype=wdtype) * 0.1
            self.check_op(lambda x: reference_norm(x, gain, 4),
                          lambda x: grouped_norm(x, gain, 4, 1e-6), [x], name="hc_norm_fp16")
            def pair(x, h, w):
                y = reference_inject(x, h, w)
                return y, reference_norm(y, gain, 4)
            self.check_op(pair, lambda x, h, w: residual_inject_norm(x, h, w, gain, 1e-6),
                          [x, h, w], name="hc_inject_norm_fp16")
        self.check_op(lambda x, u: reference_mix(x, u, 4), lambda x, up: sigmoid_mean(x, up, 4), [x, up])
        self.check_op(reference_inject, residual_inject, [x, h, w])
        for inject in (False, True):
            low = x[..., :320] if not inject else w
            reference = (lambda x: 2 * (x / 4).sigmoid()) if inject else (lambda x: F.silu(x / 4))
            self.check_op(reference, lambda x: scaled_activation(x, 4, inject), [low])
        # Flat pointer kernels must accept strided token/feature views safely.
        check_packed(self, lambda x, u: sigmoid_mean(x, u, 4), [x[..., ::2], up[..., ::2]])
        check_packed(self, residual_inject, [x[..., ::2], h[..., ::2], w])

    def test_fp16_modules_and_manual_autocast(self):
        torch.manual_seed(179)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = qwen.Qwen4ExpTextConfig(hidden_size=2560, hc_count=4, hc_lowrank=320)
        eager = qwen.Qwen4ExpTextGatedResidual(config).cuda().half().requires_grad_(False)
        eager.hc_norm.float()
        high = copy.deepcopy(eager).float()
        x = torch.randn(1, 257, 10240, device="cuda", dtype=torch.float16)
        for cls in (TritonGatedResidual, ManualGatedResidual):
            fused = cls(config).cuda().half().requires_grad_(False)
            fused.hc_norm.float()
            fused.load_state_dict(eager.state_dict())
            self.check_op(lambda x: (high if x.dtype == torch.float32 else eager)(x), fused, [x])
        eager.float()
        # Autograd leaves the forward autocast scope before executing backward.
        x.requires_grad_()
        for cls in (TritonGatedResidual, ManualGatedResidual):
            fused = cls(config).cuda().requires_grad_(False)
            fused.load_state_dict(eager.state_dict())
            with torch.autocast("cuda", dtype=torch.float16):
                expected = eager(x)
                actual = fused(x)
            upstream = tuple(torch.randn_like(t) for t in expected)
            a = torch.autograd.grad(actual, x, upstream)[0]
            b = torch.autograd.grad(expected, x, upstream)[0]
            torch.testing.assert_close(a, b, atol=0.002, rtol=0.02)

    def test_dtype_and_device_rejections(self):
        for device in ("cpu", "cuda"):
            x = torch.ones(1, 2, 10240, device=device, dtype=torch.bfloat16)
            h = x[..., :2560]
            w = x[..., :4]
            gain = x[0, 0]
            calls = [lambda x: grouped_norm(x, gain, 4, 1e-6),
                     lambda x: scaled_activation(x, 4),
                     lambda x: sigmoid_mean(x, x, 4),
                     lambda x: residual_inject(x, h, w),
                     lambda x: residual_inject_norm(x, h, w, gain, 1e-6)]
            for call in calls:
                for dtype in (torch.float64, torch.float8_e4m3fn, torch.int32, torch.float32):
                    with self.assertRaisesRegex((TypeError, ValueError), "HC"):
                        call(x.to(dtype))
                if device == "cpu":
                    with self.assertRaisesRegex(ValueError, "CUDA"):
                        call(x)
            if device == "cuda":
                with self.assertRaisesRegex(TypeError, "dtype"):
                    sigmoid_mean(x, x.float(), 4)
                with self.assertRaisesRegex(TypeError, "dtype"):
                    residual_inject(x, h.float(), w)


    def test_cpu_module_reference_and_injection_rejection(self):
        config = qwen.Qwen4ExpTextConfig(hidden_size=8, hc_count=4, hc_lowrank=2)
        for cls in (TritonGatedResidual, ManualGatedResidual):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                module = cls(config).to(dtype).requires_grad_(False)
                x = torch.randn(1, 7, 32, dtype=dtype, requires_grad=True)
                actual = module(x)
                expected = qwen.Qwen4ExpTextGatedResidual.forward(module, x)
                upstream = tuple(torch.randn_like(t) for t in actual)
                self.assertTrue(all(same_bits(a, b) for a, b in zip(actual, expected)))
                self.assertTrue(same_bits(torch.autograd.grad(actual, x, upstream)[0],
                                          torch.autograd.grad(expected, x, upstream)[0]))
            with self.assertRaisesRegex(ValueError, "CUDA"):
                cls(config, serving=True)(x)
        with self.assertRaisesRegex(ValueError, "CPU injection"):
            module(x, injection=(x[..., :8], x[..., :4]))

    def test_projection_dtype_boundary_and_autocast_rejections(self):
        config = qwen.Qwen4ExpTextConfig(hidden_size=8, hc_count=4, hc_lowrank=2)
        for cls in (TritonGatedResidual, ManualGatedResidual):
            module = cls(config).cuda().half().requires_grad_(False)
            x = torch.ones(1, 7, 32, device="cuda", dtype=torch.float16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with self.assertRaisesRegex(TypeError, "autocast execution dtype"):
                    module(x)
            module.input_mix_weight_down.float()
            with self.assertRaisesRegex(TypeError, "mixed projection"):
                module(x)
            module.input_mix_weight_down.double()
            with self.assertRaisesRegex(TypeError, "dtype"):
                module(x)
