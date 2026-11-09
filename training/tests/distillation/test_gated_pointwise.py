"""FP32-oracle accuracy and deterministic gradients for remaining wide gates."""

import copy
import unittest

import torch
import torch.nn.functional as F
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen

from tests.distillation import numerical as hc_tests
from tests.distillation.numerical import reference_norm

from training.flash.gated_pointwise import (
    norm_gate,
    TritonGatedNorm,
    TritonProjectionNorm,
    TritonSharedMLP,
    pack_qkv,
    projection_norm,
    scaled_add,
    swiglu,
)


def reference_gate_norm(x, z, weight, sigmoid=False):
    normal = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return ((normal.to(x.dtype) * weight) * (z.float().sigmoid() if sigmoid else F.silu(z.float()))).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GatedPointwiseTests(unittest.TestCase):
    check_op = hc_tests.check_op

    def test_packed_adjoint_boundaries(self):
        torch.manual_seed(153)
        x = torch.randn(1, 257, 10240, device="cuda", dtype=torch.bfloat16)
        hc_tests.check_packed(self, lambda x: pack_qkv(x, 16, 48, 128), [x])
        hc_tests.check_packed(self, lambda x, y: scaled_add(x, y, 0.7), [x, torch.randn_like(x)])
        hc_tests.check_packed(self, swiglu, [x[..., :1280]])
        hc_tests.check_packed(self, swiglu, [x[..., :640], x[..., 640:1280]])
        gain = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
        core = torch.randn(1, 257, 48, 128, device="cuda", dtype=torch.bfloat16)
        for serving in (False, True):
            for sigmoid in (False, True):
                hc_tests.check_packed(self, lambda x, z: norm_gate(
                    x, z, gain, 1e-6, serving, sigmoid), [core, torch.randn_like(core)])
        gain = torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1
        queries = torch.randn(1, 257, 48, 256, device="cuda", dtype=torch.bfloat16)[:, :, :24]
        hc_tests.check_packed(self, lambda x: projection_norm(x, gain, 1e-6), [queries])

    def test_gdn_norm_and_gate_production_width(self):
        torch.manual_seed(103)
        weight = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
        for n in (1, 257, 4096, 32768, 75776):
            x = torch.randn(n * 48, 128, device="cuda", dtype=torch.bfloat16)
            z = torch.randn_like(x)
            for sigmoid in (False, True):
                self.check_op(lambda x, z, sigmoid=sigmoid: reference_gate_norm(x, z, weight, sigmoid),
                              lambda x, z, sigmoid=sigmoid: norm_gate(x, z, weight, 1e-6, sigmoid=sigmoid), [x, z],
                              name="gdn_sigmoid" if sigmoid else "gdn_silu")

    def test_fp16_and_mixed_norm_weights(self):
        torch.manual_seed(167)
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(257 * 48, 128, device="cuda", dtype=dtype)
            z = torch.randn_like(x)
            for wdtype in (torch.bfloat16, torch.float16, torch.float32):
                weight = torch.rand(128, device="cuda", dtype=wdtype) + 0.5
                for sigmoid in (False, True):
                    self.check_op(lambda x, z: reference_gate_norm(x, z, weight, sigmoid),
                                  lambda x, z: norm_gate(x, z, weight, 1e-6, sigmoid=sigmoid),
                                  [x, z], name="gdn_dtype")
            weight = torch.randn(256, device="cuda") * 0.1
            queries = torch.randn(2, 73, 48, 256, device="cuda", dtype=dtype)[:, :, :24]
            self.check_op(lambda x: reference_norm(x, weight, 1),
                          lambda x: projection_norm(x, weight, 1e-6), [queries], name="projection_dtype")
        base = torch.randn(1, 257, 10240, device="cuda", dtype=torch.float16)
        adapter = torch.randn_like(base)
        self.check_op(lambda x, a: x + a * 0.7, lambda x, a: scaled_add(x, a, 0.7),
                      [base.transpose(1, 2), adapter.transpose(1, 2)], name="scaled_add_fp16_strided")
        self.check_op(lambda x: (x[..., :2048].unflatten(-1, (16, 128)).repeat_interleave(3, -2),
                                 x[..., 2048:4096].unflatten(-1, (16, 128)).repeat_interleave(3, -2),
                                 x[..., 4096:].unflatten(-1, (48, 128))),
                      lambda x: pack_qkv(x, 16, 48, 128), [base], name="pack_fp16")
        gate = base[..., :1280]
        self.check_op(lambda x: F.silu(x[..., :640]) * x[..., 640:], swiglu, [gate], name="swiglu_fp16")
        self.check_op(lambda x, y: F.silu(x) * y, swiglu,
                      [gate[..., :640], gate[..., 640:]], name="swiglu_fp16_separate")

    def test_dtype_and_device_rejections(self):
        for device in ("cpu", "cuda"):
            x = torch.ones(1, 2, 2, 128, device=device, dtype=torch.bfloat16)
            weight = torch.ones(128, device=device, dtype=torch.bfloat16)
            cases = [("norm_gate", lambda x: norm_gate(x, x, weight, 1e-6)),
                     ("projection_norm", lambda x: projection_norm(x, weight, 1e-6)),
                     ("pack_qkv", lambda x: pack_qkv(x, 1, 2, 32)),
                     ("scaled_add", lambda x: scaled_add(x, x, 0.7))]
            for name, call in cases:
                for dtype in (torch.float64, torch.float8_e4m3fn, torch.int32):
                    with self.assertRaisesRegex((TypeError, ValueError), name):
                        call(x.to(dtype))
                if device == "cpu":
                    with self.assertRaisesRegex(ValueError, "CUDA"):
                        call(x)
            if device == "cuda":
                for call in (cases[0][1], cases[1][1], swiglu):
                    with self.assertRaisesRegex(TypeError, "FP32"):
                        call(x.float())
                with self.assertRaisesRegex(TypeError, "dtype"):
                    swiglu(x, x.float())
                with self.assertRaisesRegex(TypeError, "dtype"):
                    scaled_add(x, x.float(), 1.0)
            for dtype in (torch.float64, torch.float8_e4m3fn, torch.int32):
                with self.assertRaisesRegex(TypeError, "swiglu"):
                    swiglu(x.to(dtype))

    def test_shared_and_routed_swiglu(self):
        torch.manual_seed(107)
        for tokens in (257, 32768):
            for routed in (False, True):
                n = tokens * (10 if routed else 1)
                if routed:
                    x = torch.randn(n, 1280, device="cuda", dtype=torch.bfloat16)
                    self.check_op(lambda x: F.silu(x[..., :640]) * x[..., 640:], swiglu, [x], name="routed_swiglu")
                else:
                    x = torch.randn(n, 640, device="cuda", dtype=torch.bfloat16)
                    up = torch.randn_like(x)
                    self.check_op(lambda x, up: F.silu(x) * up, swiglu, [x, up], name="shared_swiglu")

    def test_qkv_pack_and_gradient_merge(self):
        torch.manual_seed(109)
        for n in (1, 257, 32768, 75776):
            for transposed in (False, True):
                x = torch.randn(1, n, 10240, device="cuda", dtype=torch.bfloat16)
                if transposed:
                    x = x.transpose(1, 2).contiguous().transpose(1, 2)
                def eager(x):
                    q, k, v = x.split((2048, 2048, 6144), -1)
                    return (q.unflatten(-1, (16, 128)).repeat_interleave(3, -2),
                            k.unflatten(-1, (16, 128)).repeat_interleave(3, -2),
                            v.unflatten(-1, (48, 128)).contiguous())
                self.check_op(eager, lambda x: pack_qkv(x, 16, 48, 128), [x], name="gdn_qkv_pack")

    def test_projection_norm_and_lora_epilogue(self):
        torch.manual_seed(139)
        weight = torch.randn(256, device="cuda", dtype=torch.bfloat16) * 0.1
        batched = torch.randn(2, 73, 48, 256, device="cuda", dtype=torch.bfloat16)[:, :, :24]
        self.check_op(lambda x: reference_norm(x, weight, 1),
                      lambda x: projection_norm(x, weight, 1e-6), [batched], name="qsa_projection_norm")
        for n in (1, 257, 32768, 75776):
            for heads in (2, 24):
                # QSA queries are a noncontiguous slice of a combined Q/gate projection.
                x = torch.randn(1, n, heads * 2, 256, device="cuda", dtype=torch.bfloat16)[:, :, :heads]
                self.check_op(lambda x: reference_norm(x, weight, 1),
                              lambda x: projection_norm(x, weight, 1e-6), [x], name="qsa_projection_norm")
                if n == 257:
                    combined = projection_norm(x, weight, 1e-6)
                    pieces = [projection_norm(p, weight, 1e-6) for p in x.split((1, 73, 183), 1)]
                    self.assertTrue(torch.equal(combined, torch.cat(pieces, 1)))
            base = torch.randn(1, n, 10240, device="cuda", dtype=torch.bfloat16)
            adapter = torch.randn_like(base)
            for scale in (2.0, 0.7):
                self.check_op(lambda x, a, scale=scale: x + a * scale,
                              lambda x, a, scale=scale: scaled_add(x, a, scale), [base, adapter],
                              name=f"lora_epilogue_{scale:g}")

    def test_fp32_copy_and_epilogue(self):
        torch.manual_seed(223)
        x = torch.randn(1, 257, 10240, device="cuda", requires_grad=True)
        q, k, v = x.split((2048, 2048, 6144), -1)
        expected = (q.unflatten(-1, (16, 128)).repeat_interleave(3, -2),
                    k.unflatten(-1, (16, 128)).repeat_interleave(3, -2),
                    v.unflatten(-1, (48, 128)))
        actual = pack_qkv(x, 16, 48, 128)
        upstream = tuple(torch.randn_like(t) for t in expected)
        for a, b in zip(actual, expected):
            self.assertTrue(hc_tests.same_bits(a, b))
        torch.testing.assert_close(torch.autograd.grad(actual, x, upstream)[0],
                                   torch.autograd.grad(expected, x, upstream)[0])
        adapter = torch.randn_like(x, requires_grad=True)
        expected = x + adapter * 0.7
        actual = scaled_add(x, adapter, 0.7)
        self.assertTrue(hc_tests.same_bits(actual, expected))
        upstream = torch.randn_like(x)
        self.assertTrue(all(hc_tests.same_bits(a, b) for a, b in zip(
            torch.autograd.grad(actual, (x, adapter), upstream),
            torch.autograd.grad(expected, (x, adapter), upstream))))

    def test_mixed_gate_and_unsupported_parameters(self):
        torch.manual_seed(227)
        x = torch.randn(257 * 48, 128, device="cuda", dtype=torch.float16)
        z = torch.randn(x.shape, device="cuda")
        w = torch.rand(128, device="cuda") + 0.5
        self.check_op(lambda x, z: reference_gate_norm(x, z, w),
                      lambda x, z: norm_gate(x, z, w, 1e-6), [x, z], name="gdn_mixed_gate")
        for dtype in (torch.float64, torch.float8_e4m3fn, torch.int32):
            with self.assertRaisesRegex(TypeError, "norm_gate"):
                norm_gate(x, z, w.to(dtype), 1e-6)
            with self.assertRaisesRegex(TypeError, "projection_norm"):
                projection_norm(x.view(1, 257, 48, 128), w.to(dtype), 1e-6)

    def test_cpu_module_and_swiglu_reference_fallbacks(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(1, 7, 2, 128, dtype=dtype, requires_grad=True)
            gate = torch.randn_like(x, requires_grad=True)
            norm = TritonGatedNorm(128)
            projection = TritonProjectionNorm(128)
            for actual, expected in ((norm(x, gate), qwen.Qwen4ExpTextRMSNormGated.forward(norm, x, gate)),
                                     (projection(x), qwen.Qwen4ExpTextRMSNorm.forward(projection, x)),
                                     (swiglu(x, gate), F.silu(x) * gate)):
                upstream = torch.randn_like(actual)
                self.assertTrue(hc_tests.same_bits(actual, expected))
                a = torch.autograd.grad(actual, x, upstream)[0]
                b = torch.autograd.grad(expected, x, upstream)[0]
                self.assertTrue(hc_tests.same_bits(a, b))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            TritonGatedNorm(128, serving=True)(x, gate)

    def test_shared_mlp_module_dtypes(self):
        torch.manual_seed(241)
        config = qwen.Qwen4ExpTextConfig(hidden_size=128, intermediate_size=640)
        for device in ("cpu", "cuda"):
            for dtype in (torch.bfloat16, torch.float16):
                module = TritonSharedMLP(config).to(device=device, dtype=dtype).requires_grad_(False)
                high = copy.deepcopy(module).float()
                x = torch.randn(1, 257, 128, device=device, dtype=dtype)
                self.check_op(lambda x: qwen.Qwen4ExpTextMLP.forward(
                    high if x.dtype == torch.float32 else module, x), module, [x], name="shared_mlp_dtype")
            # The projections are ordinary Linear modules, which reject mixed dtypes themselves.
            with self.assertRaises(RuntimeError):
                module(x.float())
            module.gate_proj.double()
            with self.assertRaises(RuntimeError):
                module(x)
