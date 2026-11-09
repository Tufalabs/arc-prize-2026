"""Frozen-head activation retention must preserve loss and hidden gradients."""

import unittest

import torch
import torch.nn.functional as F

from tests.distillation import numerical as hc_tests
from training.flash.head_loss import CachedHeadCE, ChunkedCE


class HeadLogitCacheTests(unittest.TestCase):
    def test_bitwise_helper_distinguishes_signed_zero(self):
        self.assertFalse(hc_tests.same_bits(torch.tensor([0.0]), torch.tensor([-0.0])))

    def test_default_and_cached_loss_have_identical_gradients(self):
        torch.manual_seed(131)
        x = torch.randn(19, 8, requires_grad=True)
        weight = torch.randn(37, 8)
        target = torch.randint(37, (19,))
        old = ChunkedCE.apply(x, weight, target, 7)
        new = CachedHeadCE.apply(x, weight, target, 7)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(torch.autograd.grad(old, x)[0], torch.autograd.grad(new, x)[0]))

    def test_ignored_targets_have_zero_gradient(self):
        torch.manual_seed(139)
        x = torch.randn(19, 8, requires_grad=True)
        weight = torch.randn(37, 8)
        target = torch.randint(37, (19,))
        target[[0, 6, 7, 18]] = -100
        expected = F.cross_entropy(F.linear(x, weight), target, reduction="sum")
        expected_gradient = torch.autograd.grad(expected, x)[0]
        for head in (ChunkedCE, CachedHeadCE):
            actual = head.apply(x, weight, target, 7)
            torch.testing.assert_close(actual, expected)
            gradient = torch.autograd.grad(actual, x)[0]
            torch.testing.assert_close(gradient, expected_gradient)
            self.assertTrue(torch.equal(gradient[[0, 6, 7, 18]], torch.zeros(4, 8)))

    def test_autocast_and_native_backward_against_eager(self):
        torch.manual_seed(143)
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for autocast in (False, True):
                    x = torch.randn(19, 8, device=device, dtype=dtype, requires_grad=True)
                    weight = torch.randn(37, 8, device=device, dtype=dtype)
                    target = torch.randint(37, (19,), device=device)
                    target[[0, 6, 7, 18]] = -100
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
                        expected = F.cross_entropy(F.linear(x, weight).float(), target, reduction="sum")
                        actual = [head.apply(x, weight, target, 7) for head in (ChunkedCE, CachedHeadCE)]
                    upstream = expected.new_tensor(0.7)
                    gradient = torch.autograd.grad(expected, x, upstream)[0]
                    grads = [torch.autograd.grad(value, x, upstream)[0] for value in actual]
                    for value, grad in zip(actual, grads):
                        torch.testing.assert_close(value, expected)
                        torch.testing.assert_close(grad, gradient)
                    self.assertTrue(hc_tests.same_bits(*grads))

    def test_unsupported_dtype(self):
        for head in (ChunkedCE, CachedHeadCE):
            with self.assertRaisesRegex(TypeError, head.__name__):
                head.apply(torch.ones(2, 8, dtype=torch.float64), torch.ones(37, 8),
                           torch.zeros(2, dtype=torch.long), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_production_head_against_fp32(self):
        torch.manual_seed(137)
        torch.backends.cuda.matmul.allow_tf32 = False
        weight = torch.randn(248320, 2560, device="cuda", dtype=torch.bfloat16) / 2560**0.5
        high = weight.float()
        for n in (257, 4096):
            x = torch.randn(n, 2560, device="cuda", dtype=torch.bfloat16)
            target = torch.randint(weight.shape[0], (n,), device="cuda")
            for lengths in ((n,), (1, 73, n - 74)):
                def reference(x, target=target, lengths=lengths):
                    return sum(sum(F.cross_entropy(
                        F.linear(part, high if x.dtype == torch.float32 else weight).float(),
                        labels, reduction="sum")
                        for part, labels in zip(doc.split(256), targets.split(256)))
                        for doc, targets in zip(x.split(lengths), target.split(lengths)))
                def candidate(x, target=target, lengths=lengths):
                    return sum(CachedHeadCE.apply(doc, weight, targets, 256)
                               for doc, targets in zip(x.split(lengths), target.split(lengths)))
                hc_tests.check_op(self, reference, candidate, [x], name="cached_head_loss")
