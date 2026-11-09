"""Sparse execution policy and CUDA numerical regression tests."""

import unittest

import torch
from training.flash.flex_qsa import selection_metadata, sparse_attention
from training.flash.indexed_qsa import hybrid_attention
from training.flash.physical_qsa import hybrid_attention as physical
from training.flash.qsa_policy import validate_qsa_kernel


class SparsePolicyTests(unittest.TestCase):
    def test_default_and_scope(self):
        self.assertEqual(validate_qsa_kernel(), "physical_sparse")
        self.assertEqual(validate_qsa_kernel(None, "sdpa"), "flex")
        self.assertEqual(validate_qsa_kernel("flex", "sdpa", packed=True), "flex")
        for mode in ("indexed_backward", "physical_sparse"):
            self.assertEqual(validate_qsa_kernel(mode), mode)
            with self.assertRaises(ValueError):
                validate_qsa_kernel(mode, "sdpa")
            with self.assertRaises(ValueError):
                validate_qsa_kernel(mode, packed=True)
        self.assertEqual(validate_qsa_kernel("group_sparse", packed=True), "group_sparse")
        with self.assertRaises(ValueError):
            validate_qsa_kernel("group_sparse", "sdpa")
        with self.assertRaises(ValueError):
            validate_qsa_kernel("unknown")

    def test_checkpoint_cannot_silently_change_kernel(self):
        from sft.distillation.accumulation import validate_resume_manifest

        validate_resume_manifest({}, {"qsa_kernel": "flex"})
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_resume_manifest({}, {"qsa_kernel": "physical_sparse"})
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_resume_manifest({"qsa_kernel": "flex"}, {"qsa_kernel": "group_sparse"})


@unittest.skipUnless(
    torch.cuda.is_available(), "requires CUDA and pinned Flash runtime"
)
class SparseCudaTests(unittest.TestCase):
    def setUp(self):
        # Earlier tests in the same process compile Flex at many other shapes;
        # start from a fresh Dynamo cache so this test is order-independent.
        torch._dynamo.reset()

    def test_outputs_and_gradients_across_prefix_sparse_tail(self):
        generator = torch.Generator().manual_seed(20260929)
        for n in (33, 2053, 4099):
            chosen = torch.full((n, 512), -1, dtype=torch.int32)
            for row in range(2051, n):
                chosen[row] = torch.randperm((row + 1) // 4, generator=generator)[:512]
            chosen = chosen.cuda()
            mask, _, _ = selection_metadata(chosen, 4, 2048)
            torch.manual_seed(20260929 + n)
            q = torch.randn(
                1, 24, n, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            k = torch.randn(
                1, 2, n, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            v = torch.randn_like(k, requires_grad=True)
            upstream = torch.randn_like(q) * 1e-6
            reference = sparse_attention(q, k, v, mask, 1 / 16)
            expected = torch.autograd.grad(reference, (q, k, v), upstream)
            for name, function in (
                ("indexed_backward", hybrid_attention),
                ("physical_sparse", physical),
            ):
                with self.subTest(n=n, kernel=name):
                    actual = function(q, k, v, chosen, mask, 1 / 16)
                    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
                    errors = []
                    for a, b in zip(
                        (actual, *gradients), (reference, *expected), strict=True
                    ):
                        self.assertTrue(torch.isfinite(a).all().item())
                        relative = (
                            (a.float() - b.float()).norm()
                            / b.float().norm().clamp_min(1e-12)
                        ).item()
                        errors.append(relative)
                        self.assertTrue(torch.equal(a, b), f"relative L2 {relative}")
                    print(
                        {"n": n, "kernel": name, "output_dq_dk_dv_relative_l2": errors},
                        flush=True,
                    )
                    repeated = function(q, k, v, chosen, mask, 1 / 16)
                    replay = torch.autograd.grad(repeated, (q, k, v), upstream)
                    for a, b in zip(
                        (actual, *gradients), (repeated, *replay), strict=True
                    ):
                        self.assertTrue(torch.equal(a, b))

    def test_reject_wrong_geometry(self):
        q = torch.zeros(1, 4, 5, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            hybrid_attention(
                q,
                q,
                q,
                torch.zeros(5, 512, device="cuda", dtype=torch.int32),
                None,
                1 / 8,
            )


if __name__ == "__main__":
    unittest.main()
