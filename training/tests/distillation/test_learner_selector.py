import unittest

import torch

from training.flash.selector_scoring import bf16_scores, fp32_scores, compiled_fp32_scores


from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from training.flash.batched_qsa import BatchedQSAIndexer, _batched, batched_documents


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class LearnerSelectorTests(unittest.TestCase):
    def test_exact_policy_selections_and_packing(self):
        torch.manual_seed(11)
        torch.backends.cuda.matmul.allow_tf32 = False
        config = Qwen4ExpTextConfig(
            hidden_size=128, head_dim=128, indexer_n_heads=4, indexer_kv_heads=1,
            indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
        )
        module = BatchedQSAIndexer(config, 3).cuda().bfloat16()
        for n in (2052, 4096, 8192):
            x = torch.randn(1, n, 128, device="cuda", dtype=torch.bfloat16)
            pe = (torch.ones_like(x), torch.zeros_like(x))
            module.selector_policy = "batched_fp32_top513_v1"
            expected = _batched(module, x, pe, None, return_blocks=True)
            module.selector_policy = "compiled_fp32_top513_v1"
            actual = _batched(module, x, pe, None, return_blocks=True)
            self.assertTrue(torch.equal(actual, expected),
                            f"selection mismatch at {n}: "
                            f"ordered={(actual != expected).sum().item()}, "
                            f"membership={(actual.sort(-1).values != expected.sort(-1).values).sum().item()}")
        module.selector_policy = "bf16_dot_fp32_top513_v1"
        packed = batched_documents(module, x, pe, (4096, 4096))
        local = _batched(module, x[:, 4096:], tuple(t[:, 4096:] for t in pe),
                         None, return_blocks=True)
        torch.testing.assert_close(packed[4096:], torch.where(local >= 0, local + 1024, local))

    def test_bf16_dot_accumulates_to_fp32(self):
        torch.manual_seed(2)
        q = torch.randn(271, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(513, 128, device="cuda", dtype=torch.bfloat16)
        actual = bf16_scores(q, k)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, fp32_scores(q, k), rtol=1e-5, atol=1e-5)

    def test_fp16_fp32_mixed_and_strided_scoring(self):
        torch.manual_seed(193)
        torch.backends.cuda.matmul.allow_tf32 = False
        for device in ("cpu", "cuda"):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                query = torch.randn(271, 4, 256, device=device, dtype=dtype)[..., ::2]
                for kdtype in (dtype, torch.float32):
                    keys = torch.randn(513, 256, device=device, dtype=kdtype)[..., ::2]
                    actual = compiled_fp32_scores(query, keys)
                    torch.testing.assert_close(actual, fp32_scores(query, keys), rtol=1e-6, atol=1e-6)
                    if device == "cuda" and dtype == kdtype and dtype != torch.float32:
                        torch.testing.assert_close(bf16_scores(query, keys), fp32_scores(query, keys),
                                                   rtol=1e-5, atol=1e-5)

    def test_dtype_and_device_rejections(self):
        for device in ("cpu", "cuda"):
            q = torch.ones(2, 4, 128, device=device, dtype=torch.bfloat16)
            k = torch.ones(3, 128, device=device, dtype=torch.bfloat16)
            for call in (fp32_scores, compiled_fp32_scores, bf16_scores):
                for dtype in (torch.float64, torch.float8_e4m3fn, torch.int32):
                    with self.assertRaisesRegex((TypeError, ValueError), "dtype|BF16"):
                        call(q.to(dtype), k)
            for dtype in (torch.float32,):
                with self.assertRaisesRegex(ValueError, "BF16"):
                    bf16_scores(q.to(dtype), k.to(dtype))
            if device == "cpu":
                with self.assertRaisesRegex(ValueError, "CUDA"):
                    bf16_scores(q, k)
