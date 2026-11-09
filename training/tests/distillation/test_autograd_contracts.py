"""Gradient masks and higher-order contracts for every hand-written backward."""

import itertools
import unittest

import torch

from sft.distillation.loss import ChunkedDistillation
from training.flash import gated_pointwise as gp
from training.flash import hyper_connection as hc
from training.flash.expert_epilogue import WeightedPermutation
from training.flash.flex_qsa import selection_metadata
from training.flash.group_qsa import attention
from training.flash.head_loss import CachedHeadCE, ChunkedCE
from training.flash.indexed_qsa import hybrid_attention
from training.flash.moe_permutation import combine_experts, gather_tokens
from training.flash.norm_tail import NormTail
from training.flash.physical_qsa import hybrid_attention as physical_attention


def outputs(value):
    return (value,) if isinstance(value, torch.Tensor) else tuple(v for v in value if v is not None and v.requires_grad)


class AutogradContractTests(unittest.TestCase):
    def check_masks(self, call, inputs, *, double_rejected=True):
        active = [x.detach().clone().requires_grad_() for x in inputs]
        out = outputs(call(*active))
        upstream = [torch.randn_like(x) for x in out]
        expected = torch.autograd.grad(out, active, upstream)
        for mask in itertools.product((False, True), repeat=len(inputs)):
            if not any(mask):
                continue
            values = [x.detach().clone().requires_grad_(flag) for x, flag in zip(inputs, mask)]
            actual = torch.autograd.grad(outputs(call(*values)), [x for x in values if x.requires_grad], upstream)
            for a, b in zip(actual, [g for g, flag in zip(expected, mask) if flag]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        if double_rejected:
            values = [x.detach().clone().requires_grad_() for x in inputs]
            result = outputs(call(*values))
            # A differentiable upstream exposes once_differentiable's delayed error
            # even for linear kernels whose first derivative is constant.
            dy = [torch.randn_like(x, requires_grad=True) for x in result]
            with self.assertRaises(RuntimeError):
                first = torch.autograd.grad(result, values, dy, create_graph=True)
                sum(g.sum() for g in first).backward()

    def test_heads_masks_and_cached_double_backward(self):
        h, w = torch.randn(4, 8), torch.randn(13, 8)
        labels = torch.tensor([1, 3, -100, 2])
        self.check_masks(lambda h: CachedHeadCE.apply(h, w, labels, 2), [h])
        self.check_masks(lambda h: ChunkedCE.apply(h, w, labels, 2), [h])
        self.check_masks(lambda h: ChunkedDistillation.apply(h, w, h.detach(), w, labels, 1.3, 0.2, 2), [h])

    def test_differentiable_backwards_match_second_derivatives(self):
        x, r, gain = torch.randn(3, 4), torch.rand(3, 1), torch.rand(4)
        a = torch.autograd.functional.hessian(lambda x, r: NormTail.apply(x, r, gain, torch.float32).square().sum(), (x, r))
        b = torch.autograd.functional.hessian(lambda x, r: (x * r * gain).square().sum(), (x, r))
        for aa, bb in zip(a, b):
            for aaa, bbb in zip(aa, bb):
                torch.testing.assert_close(aaa, bbb)
        order = torch.tensor([2, 0, 1])
        undo = order.argsort()
        weights = torch.rand(3)
        cases = [(lambda x, w: WeightedPermutation.apply(x, w, order, undo),
                  lambda x, w: x[undo] * w[undo, None], (x, weights))]
        if torch.cuda.is_available():
            cases.append((lambda x, w: gp.scaled_add(x, w, 0.7),
                          lambda x, w: x + 0.7 * w, (x.cuda(), x.cuda())))
        for call, reference, args in cases:
            a = torch.autograd.functional.hessian(lambda *args: call(*args).square().sum(), args)
            b = torch.autograd.functional.hessian(lambda *args: reference(*args).square().sum(), args)
            for aa, bb in zip(a, b):
                for aaa, bbb in zip(aa, bb):
                    torch.testing.assert_close(aaa, bbb)

    def test_eager_adjoint_masks_and_frozen_rejections(self):
        x, r, gain = torch.randn(3, 4), torch.rand(3, 1), torch.rand(4)
        self.check_masks(lambda x, r: NormTail.apply(x, r, gain, torch.float32), [x, r], double_rejected=False)
        order = torch.tensor([2, 0, 1])
        self.check_masks(lambda x, w: WeightedPermutation.apply(x, w, order, order.argsort()),
                         [x, torch.rand(3)], double_rejected=False)
        with self.assertRaisesRegex(ValueError, "frozen"):
            NormTail.apply(x, r, gain.requires_grad_(), torch.float32)
        for index in (1, 2, 3):
            args = [torch.randn(3, 4), torch.randn(7, 4), torch.randn(3, 4), torch.randn(7, 4)]
            args[index].requires_grad_()
            with self.assertRaisesRegex(ValueError, "frozen"):
                ChunkedDistillation.apply(*args, torch.ones(3, dtype=torch.int64), 1, 0, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_kernel_adjoint_masks_and_double_backward(self):
        torch.manual_seed(311)
        def rand(*shape):
            return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        x, h, weights, gain = rand(1, 3, 64), rand(1, 3, 16), rand(1, 3, 4), rand(64)
        cases = [
            (lambda x: hc.grouped_norm(x, gain, 4, 1e-6), [x]),
            (lambda x: hc.scaled_activation(x, 4), [x]),
            (lambda x, up: hc.sigmoid_mean(x, up, 4), [x, rand(*x.shape)]),
            (lambda x, h, w: hc.residual_inject(x, h, w), [x, h, weights]),
            (lambda x, h, w: hc.residual_inject_norm(x, h, w, gain, 1e-6), [x, h, weights]),
            (lambda x, z: gp.norm_gate(x, z, rand_gain, 1e-6), [h, h]),
            (lambda a, b: gp.swiglu(a, b), [h, h]),
            (lambda x: gp.swiglu(x), [rand(1, 3, 32)]),
            (lambda x: gp.pack_qkv(x, 1, 2, 16), [rand(1, 3, 64)]),
            (lambda x: gp.projection_norm(x, projection_gain, 1e-6), [rand(1, 3, 2, 16)]),
            (lambda x: gather_tokens(x, order, 2), [rand(3, 16)]),
            (lambda x, w: combine_experts(x, w, order), [rand(6, 16), rand(3, 2)]),
        ]
        rand_gain, projection_gain, order = rand(16), rand(16), torch.randperm(6, device="cuda")
        for call, inputs in cases:
            with self.subTest(op=call.__code__.co_names):
                self.check_masks(call, inputs)
        down, up, block = rand(8, 64), rand(64, 8), rand(4, 64)
        self.check_masks(lambda x, h, w: hc._GatedResidual.apply(x, h, w, gain, down, up, block, 4, 1e-6, False),
                         [x, h, weights])
        for frozen in (gain, down, up, block):
            frozen.requires_grad_()
            with self.assertRaisesRegex(ValueError, "frozen"):
                hc._GatedResidual.apply(x, None, None, gain, down, up, block, 4, 1e-6, False)
            frozen.requires_grad_(False)
        for call, weight in ((lambda w: hc.grouped_norm(x, w, 4, 1e-6), gain),
                             (lambda w: hc.residual_inject_norm(x, h, weights, w, 1e-6), gain),
                             (lambda w: gp.norm_gate(h, h, w, 1e-6), rand_gain),
                             (lambda w: gp.projection_norm(h, w, 1e-6), projection_gain)):
            with self.assertRaisesRegex(ValueError, "frozen"):
                call(weight.detach().requires_grad_())
        self.check_masks(lambda a, b: gp.scaled_add(a, b, 0.7), [h, h], double_rejected=False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_attention_masks_and_double_backward(self):
        q = torch.randn(1, 24, 16, 256, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 2, 16, 256, device="cuda", dtype=torch.bfloat16)
        chosen = torch.full((16, 512), -1, device="cuda", dtype=torch.int32)
        mask, _, _ = selection_metadata(chosen, 4, 2048)
        for call in (lambda q, k, v: attention(q, k, v, chosen, 1 / 16),
                     lambda q, k, v: hybrid_attention(q, k, v, chosen, mask, 1 / 16),
                     lambda q, k, v: physical_attention(q, k, v, chosen, mask, 1 / 16)):
            self.check_masks(call, [q, k, k])
