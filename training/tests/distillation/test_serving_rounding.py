"""Serving arithmetic, FP32-oracle gradients, and exact copied-kernel forwards."""

import copy
import json
import unittest

import torch
import torch.nn.functional as F

from tests.distillation import numerical as hc_tests
from tests.distillation.numerical import reference_norm
from tests.distillation.test_gated_pointwise import reference_gate_norm

from tests.distillation import serving_rounding_reference as serving
from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen
from training.flash.gated_pointwise import norm_gate
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
class ServingRoundingTests(unittest.TestCase):
    check_op = hc_tests.check_op

    def test_serving_inject_norm_small_widths(self):
        # vLLM pads the 512-wide tile count up to a power of two, including below 512.
        torch.manual_seed(127)
        for d in (128, 384, 600):
            x = torch.randn(1, 33, 4 * d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            h = torch.randn(1, 33, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            logits = torch.randn(1, 33, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            gain = torch.randn(4 * d, device="cuda", dtype=torch.bfloat16) * 0.1
            actual = residual_inject_norm(x, h, logits, gain, 1e-6, True)
            expected = serving.inject_norm(x.detach(), h.detach(), logits.detach(), gain)
            for new, ref in zip(actual, expected):
                self.assertTrue(hc_tests.same_bits(new, ref), f"inject_norm width {d}")
            sum(value.float().sum() for value in actual).backward()
            for value in (x, h, logits):
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_serving_forward_and_backward(self):
        torch.manual_seed(113)
        for n in (257, 32768, 75776):
            x = torch.randn(1, n, 10240, device="cuda", dtype=torch.bfloat16)
            up = torch.randn_like(x)
            h = torch.randn(1, n, 2560, device="cuda", dtype=torch.bfloat16)
            logits = torch.randn(1, n, 4, device="cuda", dtype=torch.bfloat16)
            gain = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.1
            low = torch.randn(1, n, 320, device="cuda", dtype=torch.bfloat16)
            def hf_inject(x, h, logits):
                return hc_tests.reference_inject(x, h, 2 * (logits / 4).sigmoid())
            def hf_pair(x, h, logits, gain=gain):
                y = hf_inject(x, h, logits)
                return y, reference_norm(y, gain, 4)
            cases = [
                ("norm", [x], lambda x, gain=gain: reference_norm(x, gain, 4),
                 lambda x, gain=gain: grouped_norm(x, gain, 4, 1e-6, True), lambda x, gain=gain: serving.norm(x, gain, 4)),
                ("silu", [low], lambda x: F.silu(x / 4),
                 lambda x: scaled_activation(x, 4, serving=True), lambda x: serving.silu(x, 4)),
                ("mix", [x, up], lambda x, up: hc_tests.reference_mix(x, up, 4),
                 lambda x, up: sigmoid_mean(x, up, 4, True), lambda x, up: serving.mix(x, up, 4)),
                ("inject", [x, h, logits], hf_inject,
                 lambda x, h, logits: residual_inject(x, h, logits, True), serving.inject),
                ("inject_norm", [x, h, logits], hf_pair,
                 lambda x, h, logits, gain=gain: residual_inject_norm(x, h, logits, gain, 1e-6, True),
                 lambda x, h, logits, gain=gain: serving.inject_norm(x, h, logits, gain)),
            ]
            gx = torch.randn(n * 48, 128, device="cuda", dtype=torch.bfloat16)
            gz = torch.randn_like(gx)
            gw = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
            for sigmoid in (False, True):
                cases.append(("gdn_sigmoid" if sigmoid else "gdn_silu", [gx, gz],
                              lambda x, z, gw=gw, sigmoid=sigmoid: reference_gate_norm(x, z, gw, sigmoid),
                              lambda x, z, gw=gw, sigmoid=sigmoid: norm_gate(x, z, gw, 1e-6, True, sigmoid),
                              lambda x, z, gw=gw, sigmoid=sigmoid: serving.gated_norm(x, z, gw, sigmoid)))
            hf_modes = {
                "norm": lambda x, gain=gain: grouped_norm(x, gain, 4, 1e-6),
                "silu": lambda x: scaled_activation(x, 4),
                "mix": lambda x, up: sigmoid_mean(x, up, 4),
                "inject": lambda x, h, logits: residual_inject(
                    x, h, scaled_activation(logits, 4, inject=True)),
                "inject_norm": lambda x, h, logits, gain=gain: residual_inject_norm(
                    x, h, scaled_activation(logits, 4, inject=True), gain, 1e-6),
                "gdn_silu": lambda x, z, gw=gw: norm_gate(x, z, gw, 1e-6),
                "gdn_sigmoid": lambda x, z, gw=gw: norm_gate(x, z, gw, 1e-6, sigmoid=True),
            }
            for name, inputs, hf, candidate, port in cases:
                with self.subTest(op=name, tokens=n):
                    hf_mode = hf_modes[name]
                    rng = torch.cuda.get_rng_state()
                    hf_errors = self.check_op(hf, hf_mode, inputs, name=name + "_hf")
                    torch.cuda.set_rng_state(rng)
                    serving_errors = self.check_op(hf, candidate, inputs, name=name + "_vllm")
                    native, actual, expected = hf_mode(*inputs), candidate(*inputs), port(*inputs)
                    outputs = [v if isinstance(v, tuple) else (v,) for v in (native, actual, expected)]
                    for i, (old, new, ref) in enumerate(zip(*outputs)):
                        print(json.dumps({"serving_op": name, "tokens": n, "output": i,
                                          "hf_vs_serving_rms": hc_tests.error(old, ref.float()),
                                          "hf_vs_serving_max": (old.float() - ref.float()).abs().max().item(),
                                          "candidate_vs_serving_rms": hc_tests.error(new, ref.float()),
                                          "candidate_vs_serving_max": (new.float() - ref.float()).abs().max().item(),
                                          "bitwise_equal": hc_tests.same_bits(new, ref),
                                          "hf_fp32_errors": hf_errors, "vllm_fp32_errors": serving_errors}), flush=True)
                        self.assertTrue(hc_tests.same_bits(new, ref), f"{name}: serving forward differs")

    def test_complete_serving_hyper_connection_with_packed_tokens(self):
        torch.manual_seed(127)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = qwen.Qwen4ExpTextConfig(hidden_size=2560, hc_count=4, hc_lowrank=320)
        hf = qwen.Qwen4ExpTextGatedResidual(config).cuda().bfloat16().requires_grad_(False)
        high = copy.deepcopy(hf).float()
        x = torch.randn(1, 4096, 10240, device="cuda", dtype=torch.bfloat16)
        h = torch.randn(1, 4096, 2560, device="cuda", dtype=torch.bfloat16)
        def baseline(x, h):
            mixed, residual, weights = (high if x.dtype == torch.float32 else hf)(x)
            return mixed, hc_tests.reference_inject(residual, h, weights)
        for cls in (TritonGatedResidual, ManualGatedResidual):
            fused = cls(config, serving=True).cuda().bfloat16().requires_grad_(False)
            fused.load_state_dict(hf.state_dict())
            def candidate(x, h, fused=fused):
                mixed, residual, logits = fused(x)
                return mixed, residual_inject(residual, h, logits, True)
            self.check_op(baseline, candidate, [x, h])
            # The serving primitives are token-local; packed boundaries have no
            # influence on GEMMs, reductions, or the deterministic adjoint.
            def packed_call(call, x, h):
                split = [call(a, b) for a, b in zip(x.split((1, 73, 183), 1), h.split((1, 73, 183), 1))]
                return tuple(torch.cat([value[i] for value in split], 1) for i in range(2))
            self.check_op(lambda x, h: packed_call(baseline, x, h),
                          lambda x, h, candidate=candidate: packed_call(candidate, x, h),
                          [x[:, :257], h[:, :257]])

    def test_manual_serving_injected_packed_input(self):
        torch.manual_seed(163)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = qwen.Qwen4ExpTextConfig(hidden_size=2560, hc_count=4, hc_lowrank=320)
        hf = qwen.Qwen4ExpTextGatedResidual(config).cuda().bfloat16().requires_grad_(False)
        high = copy.deepcopy(hf).float()
        fused = ManualGatedResidual(config, serving=True).cuda().bfloat16().requires_grad_(False)
        fused.load_state_dict(hf.state_dict())
        x = torch.randn(1, 257, 10240, device="cuda", dtype=torch.bfloat16)
        h = torch.randn(1, 257, 2560, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(1, 257, 4, device="cuda", dtype=torch.bfloat16)

        def baseline(x, h, logits):
            injected = hc_tests.reference_inject(x, h, 2 * (logits / 4).sigmoid())
            mixed, residual, weights = (high if x.dtype == torch.float32 else hf)(injected)
            return mixed, hc_tests.reference_inject(residual, h, weights)

        def candidate(x, h, logits):
            outputs = []
            for a, b, w in zip(x.split((1, 73, 183), 1), h.split((1, 73, 183), 1),
                               logits.split((1, 73, 183), 1)):
                mixed, residual, weights = fused(a, injection=(b, w))
                outputs.append((mixed, residual_inject(residual, b, weights, True)))
            return tuple(torch.cat(values, 1) for values in zip(*outputs))

        self.check_op(baseline, candidate, [x, h, logits], name="manual_serving_injected_packed")

    def test_fp16_and_mixed_affines_match_serving(self):
        torch.manual_seed(197)
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(1, 257, 10240, device="cuda", dtype=dtype)
            up = torch.randn_like(x)
            h = torch.randn(1, 257, 2560, device="cuda", dtype=dtype)
            logits = torch.randn(1, 257, 4, device="cuda", dtype=dtype)
            low = x[..., :320]
            for wdtype in (dtype, torch.float32):
                gain = torch.randn(10240, device="cuda", dtype=wdtype) * 0.1
                def pair(x, h, logits):
                    y = (x.float() + (h.float().unsqueeze(-2) *
                         (2 * (logits.float() / 4).sigmoid()).unsqueeze(-1)).flatten(-2)).to(x.dtype)
                    return y, reference_norm(y, gain, 4)
                cases = [(lambda x: reference_norm(x, gain, 4),
                          lambda x: grouped_norm(x, gain, 4, 1e-6, True),
                          lambda x: serving.norm(x, gain, 4), [x]),
                         (pair, lambda x, h, w: residual_inject_norm(x, h, w, gain, 1e-6, True),
                          lambda x, h, w: serving.inject_norm(x, h, w, gain), [x, h, logits])]
                gx = torch.randn(257 * 48, 128, device="cuda", dtype=dtype)
                gz = torch.randn_like(gx)
                gw = torch.rand(128, device="cuda", dtype=wdtype) + 0.5
                for sigmoid in (False, True):
                    def ref(x, z, sigmoid=sigmoid):
                        norm = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                        gate = z.float().sigmoid() if sigmoid else F.silu(z.float())
                        return (norm * gw.float() * gate).to(x.dtype)
                    cases.append((ref, lambda x, z, sigmoid=sigmoid: norm_gate(x, z, gw, 1e-6, True, sigmoid),
                                  lambda x, z, sigmoid=sigmoid: serving.gated_norm(x, z, gw, sigmoid), [gx, gz]))
                for ref, candidate, port, inputs in cases:
                    self.check_op(ref, candidate, inputs, name="serving_dtype")
                    actual, expected = candidate(*inputs), port(*inputs)
                    actual = actual if isinstance(actual, tuple) else (actual,)
                    expected = expected if isinstance(expected, tuple) else (expected,)
                    self.assertTrue(all(hc_tests.same_bits(a, b) for a, b in zip(actual, expected)))
            cases = [(lambda x: F.silu(x.float() / 4).to(x.dtype),
                      lambda x: scaled_activation(x, 4, serving=True), [low]),
                     (lambda x, u: (x.float() * u.float().sigmoid()).unflatten(-1, (4, 2560)).mean(-2).to(x.dtype),
                      lambda x, u: sigmoid_mean(x, u, 4, True), [x, up]),
                     (lambda x, h, w: (x.float() + (h.float().unsqueeze(-2) *
                       (2 * (w.float() / 4).sigmoid()).unsqueeze(-1)).flatten(-2)).to(x.dtype),
                      lambda x, h, w: residual_inject(x, h, w, True), [x, h, logits])]
            for ref, candidate, inputs in cases:
                self.check_op(ref, candidate, inputs, name="serving_fp16_primitives")
