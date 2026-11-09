import unittest

import torch
import torch.nn.functional as F

from training.flash.convolution import (
    compiled_ple_convolution,
    fla_convolution,
    ple_convolution,
)


def reference(x, weight, lengths, dilation):
    return torch.cat([
        F.silu(F.conv1d(
            F.pad(piece.transpose(1, 2), ((weight.shape[-1] - 1) * dilation, 0)),
            weight, groups=x.shape[-1], dilation=dilation,
        )).transpose(1, 2)
        for piece in x.split(lengths, dim=1)
    ], dim=1)


class ConvolutionTests(unittest.TestCase):
    def check_ple(self, device, compiled):
        torch.manual_seed(3)
        for lengths in ((37,), (1, 7, 29)):
            x = torch.randn(1, sum(lengths), 16, device=device, dtype=torch.bfloat16)
            x.requires_grad_()
            weight = torch.randn(16, 1, 4, device=device, dtype=x.dtype) / 4
            positions = torch.cat([torch.arange(n, device=device) for n in lengths])
            call = compiled_ple_convolution if compiled else ple_convolution
            actual = call(x, weight, 3, positions)
            expected = reference(x, weight, lengths, 3)
            upstream = torch.randn_like(x)
            ga = torch.autograd.grad(actual, x, upstream)[0]
            ge = torch.autograd.grad(expected, x, upstream)[0]
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(ga, ge, atol=0.03, rtol=0.03)

    def test_ple_packed_cpu(self):
        self.check_ple("cpu", False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_ple_compiled_cuda(self):
        self.check_ple("cuda", True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fla_packed_cuda(self):
        torch.manual_seed(7)
        for lengths in ((128,), (17, 1, 110)):
            x = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16)
            x.requires_grad_()
            weight = torch.randn(128, 1, 4, device="cuda", dtype=x.dtype) / 4
            cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                              device="cuda", dtype=torch.int32)
            actual = fla_convolution(x.transpose(1, 2), weight[:, 0], None, "silu", cu)
            actual = actual.transpose(1, 2)
            expected = reference(x, weight, lengths, 1)
            upstream = torch.randn_like(x)
            ga = torch.autograd.grad(actual, x, upstream)[0]
            ge = torch.autograd.grad(expected, x, upstream)[0]
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(ga, ge, atol=0.03, rtol=0.03)
