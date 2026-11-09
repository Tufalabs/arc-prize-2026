import types
import unittest

import torch

from training.flash.expert_epilogue import WeightedPermutation


class BackwardMemoryTests(unittest.TestCase):
    def check(self, device):
        torch.manual_seed(17)
        for dtype in (torch.float32, torch.bfloat16):
            for wtype in (dtype, torch.float32):
                n = 16391
                d = 2560 if device == "cuda" else 79
                x = torch.randn(n, d, device=device, dtype=dtype)
                w = torch.randn(n, device=device, dtype=wtype)
                p = torch.randperm(n, device=device)
                g = torch.randn_like(x)
                sorted_g = g[p]
                expected_x = sorted_g * w[:, None]
                expected_w = (sorted_g * x).sum(-1)
                for needs in ((True, True), (True, False), (False, True)):
                    ctx = types.SimpleNamespace(
                        saved_tensors=(x, w, p), needs_input_grad=needs
                    )
                    dx, dw, _, _ = WeightedPermutation.backward(ctx, g)
                    if needs[0]:
                        self.assertTrue(torch.equal(dx, expected_x))
                        self.assertEqual(dx.dtype, expected_x.dtype)
                    else:
                        self.assertIsNone(dx)
                    if needs[1]:
                        self.assertTrue(torch.equal(dw, expected_w))
                        self.assertEqual(dw.dtype, expected_w.dtype)
                    else:
                        self.assertIsNone(dw)

    def test_cpu_exact_across_chunk_boundaries_and_dtypes(self):
        self.check("cpu")

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA qualification runs this check"
    )
    def test_cuda_exact_production_width(self):
        self.check("cuda")


if __name__ == "__main__":
    unittest.main()
