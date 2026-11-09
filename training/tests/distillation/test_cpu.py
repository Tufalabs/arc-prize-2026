"""Small independent numerical tests; never load the complete checkpoint."""

import copy
import unittest

import torch

from sft.distillation.flash_model import (
    ChunkedCE,
    HostEmbedding,
    completion_loss,
    dequant_nvfp4,
)


class NumericalTests(unittest.TestCase):
    def test_attention_context_fresh_for_two_updates(self):
        from pathlib import Path

        from torch.nn.attention import SDPBackend, sdpa_kernel

        source = Path(
            __import__("sft.distillation.worker", fromlist=["__file__"]).__file__
        ).read_text()
        self.assertIn("with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):", source)
        self.assertNotIn("with backend:", source)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.SGD([parameter], lr=0.1)
        for _ in range(2):
            opt.zero_grad()
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                parameter.square().backward()
            opt.step()
        self.assertAlmostEqual(parameter.item(), 0.64, places=6)

    def test_ce_loss_and_gradient_matches_full(self):
        for dtype in [torch.float32, torch.bfloat16]:
            torch.manual_seed(7)
            h = torch.randn(17, 13, dtype=dtype, requires_grad=True)
            w = torch.randn(29, 13, dtype=dtype)
            y = torch.arange(17) % 29
            expected = torch.nn.functional.cross_entropy(
                torch.nn.functional.linear(h, w).float(), y, reduction="sum"
            )
            (g,) = torch.autograd.grad(expected, h)
            actual = ChunkedCE.apply(h, w, y, 4)
            (g2,) = torch.autograd.grad(actual, h)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
            torch.testing.assert_close(
                g2,
                g,
                rtol=0.01 if dtype == torch.bfloat16 else 1e-5,
                atol=0.04 if dtype == torch.bfloat16 else 2e-6,
            )

    def test_completion_shift_and_prompt_mask(self):
        h = torch.randn(1, 9, 5, requires_grad=True)
        head = torch.nn.Linear(5, 11, bias=False).requires_grad_(False)
        ids = torch.arange(9)[None]
        loss = completion_loss(h, head, ids, 6, 2)
        ref = torch.nn.functional.cross_entropy(
            head(h[0, 5:8]), ids[0, 6:], reduction="sum"
        )
        torch.testing.assert_close(loss, ref)
        (grad,) = torch.autograd.grad(loss, h)
        assert grad[:, :5].count_nonzero() == 0 and grad[:, 8:].count_nonzero() == 0

    def test_nvfp4_nibbles_and_scales(self):
        raw = torch.tensor(
            [[i | ((15 - i) << 4) for i in range(16)]], dtype=torch.uint8
        )
        scale = torch.tensor([[2.0, 3.0]])
        out = dequant_nvfp4(raw, scale, torch.tensor(0.5))
        lut = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        expected = [v for i in range(16) for v in [lut[i], lut[15 - i]]]
        expected = torch.tensor(expected)[None] * torch.tensor(
            [[1.0] * 16 + [1.5] * 16]
        )
        torch.testing.assert_close(out, expected.bfloat16(), rtol=0, atol=0)

    def test_batched_nvfp4_applies_per_expert_outer_scale(self):
        packed = torch.randint(0, 256, (3, 4, 8), dtype=torch.uint8)
        scale = torch.full((3, 4, 1), 0.125)
        outer = torch.linspace(0.5, 1.0, 3)
        expected = torch.stack(
            [dequant_nvfp4(packed[i], scale[i], outer[i]) for i in range(3)]
        )
        torch.testing.assert_close(
            dequant_nvfp4(packed, scale, outer), expected, rtol=0, atol=0
        )

    def test_host_embedding_shard_boundaries_and_scale(self):
        class FakeReader:
            def __init__(self):
                self.parts = [
                    (torch.arange(12).reshape(3, 4) % 7 + i % 3).to(torch.float8_e4m3fn)
                    for i in range(128)
                ]

            def tensor(self, name):
                if name.endswith("weight_scale"):
                    return torch.tensor([0.125], dtype=torch.bfloat16)
                return self.parts[int(name.split("shard_")[1].split(".")[0])]

        reader = FakeReader()
        embed = HostEmbedding(reader, "test")
        ids = torch.tensor([[0, 2, 3, 383, 0, 201]])
        expected = torch.cat([p.bfloat16() for p in reader.parts])[ids] * 0.125
        torch.testing.assert_close(embed(ids), expected, rtol=0, atol=0)
        assert list(embed.parameters()) == [] and list(embed.buffers()) == []
        with self.assertRaises(ValueError):
            embed(torch.tensor([384]))

    def test_router_fp32_master_preserves_forward(self):
        try:
            from transformers.models.qwen4_exp.configuration_qwen4_exp import (
                Qwen4ExpTextConfig,
            )
            from transformers.models.qwen4_exp.modeling_qwen4_exp import (
                Qwen4ExpTextTopKRouter,
            )

            from training.flash.adapters import FP32TopKRouter
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("transformers.models.qwen4_exp"):
                self.skipTest("Qwen4-Exp is available in the distillation environment")
            raise

        config = Qwen4ExpTextConfig(
            hidden_size=16, num_experts=12, num_experts_per_tok=3
        )
        original = Qwen4ExpTextTopKRouter(config).bfloat16()
        with torch.no_grad():
            original.weight.normal_()
        patched = FP32TopKRouter(config).float()
        patched.weight.data.copy_(original.weight.float())
        x = torch.randn(2, 5, 16, dtype=torch.bfloat16)
        a, b = original(x), patched(x)
        for u, v in zip(a, b):
            torch.testing.assert_close(u, v, rtol=0, atol=0)
        b[1].float().square().sum().backward()
        assert (
            patched.weight.grad.dtype == torch.float32
            and torch.isfinite(patched.weight.grad).all()
        )

    def test_router_optimizer_roundtrip(self):
        p = torch.nn.Parameter(torch.randn(4, 3))
        opt = torch.optim.AdamW([p], lr=1e-5, weight_decay=0)
        p.square().sum().backward()
        opt.step()
        opt.zero_grad()
        checkpoint = copy.deepcopy((p.detach(), opt.state_dict()))
        p.square().sum().backward()
        opt.step()
        q = torch.nn.Parameter(checkpoint[0].clone())
        other = torch.optim.AdamW([q], lr=1e-5, weight_decay=0)
        other.load_state_dict(checkpoint[1])
        q.square().sum().backward()
        other.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
