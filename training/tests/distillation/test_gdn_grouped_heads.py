"""Grouped value heads must preserve the existing repeated-Q/K accuracy."""

import json
import unittest

import torch

from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GroupedHeadsTests(unittest.TestCase):
    @unittest.skip(
        "gdn_grouped_heads is unqualified: FLA 0.5.1's grouped-value path exceeded the "
        "1.5x error gate (43x in job 2859065, 58x earlier) but passed in other runs; "
        "keep the option off until the cause is understood"
    )
    def test_grouped_against_fp32_with_packed_documents(self):
        # FLA ships only in the Linux flash-distillation extra.
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        torch.manual_seed(89)
        torch.backends.cuda.matmul.allow_tf32 = False
        for lengths in ((256,), (1, 73, 183)):
            n = sum(lengths)
            q = torch.randn(1, n, 16, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn(1, n, 48, 128, device="cuda", dtype=torch.bfloat16)
            beta = torch.rand(1, n, 48, device="cuda", dtype=torch.bfloat16)
            g = -torch.rand(1, n, 48, device="cuda", dtype=torch.float32)
            inputs = [x.requires_grad_() for x in (q, k, v, g, beta)]
            high = [x.detach().float().requires_grad_() for x in inputs]
            cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()],
                              device="cuda", dtype=torch.int32)
            def call(values, grouped, cu=cu):
                q, k, v, g, beta = values
                if not grouped:
                    q, k = q.repeat_interleave(3, 2), k.repeat_interleave(3, 2)
                return chunk_gated_delta_rule(q, k, v, g, beta,
                                             cu_seqlens=cu, use_qk_l2norm_in_kernel=True)[0]
            refs = []
            start = 0
            for length in lengths:
                qf, kf, vf, gf, bf = [x[:, start:start + length] for x in high]
                refs.append(qwen.torch_chunk_gated_delta_rule(
                    qf.repeat_interleave(3, 2), kf.repeat_interleave(3, 2), vf, gf, bf,
                    use_qk_l2norm_in_kernel=True)[0])
                start += length
            reference = torch.cat(refs, dim=1)
            eager, actual = call(inputs, False), call(inputs, True)
            dy = torch.randn_like(actual)
            ge = torch.autograd.grad(eager, inputs, dy)
            ga = torch.autograd.grad(actual, inputs, dy)
            gr = torch.autograd.grad(reference, high, dy.float())
            for name, a, b, r in zip(("output", "q", "k", "v", "g", "beta"),
                                     (actual, *ga), (eager, *ge), (reference, *gr)):
                ea = (a.float() - r).square().mean().sqrt().item()
                eb = (b.float() - r).square().mean().sqrt().item()
                ma, mb = (a.float() - r).abs().max().item(), (b.float() - r).abs().max().item()
                print(json.dumps({"gva": name, "lengths": lengths, "eager_rms": eb,
                                  "grouped_rms": ea, "eager_max": mb, "grouped_max": ma}), flush=True)
                self.assertLessEqual(ea, 1.5 * eb, name)
                self.assertLessEqual(ma, 1.5 * mb, name)
            again = call(inputs, True)
            gradients = torch.autograd.grad(again, inputs, dy)
            self.assertTrue(torch.equal(actual, again))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(ga, gradients)))
