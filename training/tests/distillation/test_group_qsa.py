"""Exact group QSA metadata and CUDA acceptance checks."""

import unittest

import torch

from training.flash.group_qsa_metadata import GROUP, group_layout, reference_union

from types import SimpleNamespace
from torch.utils.checkpoint import checkpoint
from training.flash.qsa_retention import retain_attention, retention_contexts
from training.flash.qsa_selection import select_blocks, selection_contexts
from tests.distillation.qsa_reference import dense_reference, errors
from training.flash.flex_qsa import selection_metadata, sparse_attention
from training.flash.group_qsa import attention, forward, prepare_groups
from training.flash.indexed_qsa import _raw_flex
from tests.distillation.numerical import check_op


def selections(lengths, pattern="local"):
    generator = torch.Generator().manual_seed(91)
    chosen = torch.full((sum(lengths), 512), -1, dtype=torch.int32)
    start, offset = 0, 0
    for n in lengths:
        for row in range(2051, n):
            complete = (row + 1) // 4
            if pattern == "random":
                blocks = torch.randperm(complete, generator=generator)[:512]
            elif pattern in ("identical", "disjoint"):
                blocks = torch.arange(512)
            else:
                blocks = torch.arange(complete - 512, complete)
            chosen[start + row] = blocks + offset
            if pattern == "empty":
                chosen[start + row] = -1
            if pattern == "padding":
                chosen[start + row, ::3] = -1
        if pattern == "disjoint":
            for lane in range(8):
                chosen[start + 16384 + lane] = torch.arange(lane * 512, (lane + 1) * 512) + offset
        start += n
        offset += (n + 3) // 4
    return chosen


class GroupLayoutTests(unittest.TestCase):
    def test_groups_stop_at_document_boundaries(self):
        layout = group_layout(30, "cpu", (3, 11, 16))
        self.assertEqual(layout.starts.tolist(), [0, 3, 11, 14, 22])
        self.assertEqual(layout.ends.tolist(), [3, 14, 14, 30, 30])
        self.assertEqual(layout.key_starts.tolist(), [0, 3, 7, 11, 14, 18, 22, 26])
        self.assertEqual(layout.block_offsets.tolist(), [0, 1, 1, 4, 4])
        for lengths in ((), (0, 30), (2, 27)):
            with self.assertRaises(ValueError):
                group_layout(30, "cpu", lengths)

    def test_prefix_and_tail_membership(self):
        chosen = selections((2061,), "identical")
        union = reference_union(chosen)
        self.assertEqual(union[0], [(0, 255), (1, 240)])
        group = dict(union[2056 // GROUP])
        self.assertEqual(group[0], 31)
        self.assertEqual(group[514], 7)  # row2059 has no partial tail
        self.assertEqual(group[515], 16)
        self.assertNotIn(512, group)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and pinned Flash runtime")
class GroupCudaTests(unittest.TestCase):
    def test_fp16_forward_backward_packed_and_range(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        # Isolate dtype/layout specializations from earlier BF16 oracle tests.
        code = _raw_flex.__wrapped__.__code__
        torch._dynamo.reset_code(code)
        self.addCleanup(torch._dynamo.reset_code, code)
        for lengths in ((257,), (3, 2053, 17)):
            n = sum(lengths)
            torch.manual_seed(191)
            chosen = selections(lengths, "random").cuda()
            q = torch.randn(1, 24, n * 2, 256, device="cuda", dtype=torch.float16)[:, :, ::2]
            k = torch.randn(1, 2, n * 2, 256, device="cuda", dtype=torch.float16)[:, :, ::2]
            v = torch.randn_like(k)
            def reference(q, k, v):
                outputs = []
                start, offset = 0, 0
                for length in lengths:
                    local = chosen[start:start + length]
                    local = torch.where(local >= 0, local - offset, local)
                    mask, _, _ = selection_metadata(local, 4, 2048)
                    a, b, c = (x[:, :, start:start + length] for x in (q, k, v))
                    if q.dtype == torch.float32:
                        allow = mask.mask_mod(0, 0, torch.arange(length, device=q.device)[:, None],
                                              torch.arange(length, device=q.device)[None])
                        b, c = b.repeat_interleave(12, 1), c.repeat_interleave(12, 1)
                        scores = (a @ b.transpose(-1, -2) / 16).masked_fill(~allow, -torch.inf)
                        outputs.append(scores.softmax(-1) @ c)
                    else:
                        outputs.append(_raw_flex(a, b, c, mask, 1 / 16)[0])
                    start += length
                    offset += (length + 3) // 4
                return torch.cat(outputs, 2)
            check_op(self, reference, lambda q, k, v: attention(q, k, v, chosen, 1 / 16, lengths),
                     [q, k, v], name="group_qsa_fp16")
            # Large finite upstreams exercise FP32 partials without FP16 overflow.
            inputs = [x.detach().requires_grad_() for x in (q, k, v)]
            actual = attention(*inputs, chosen, 1 / 16, lengths)
            upstream = torch.randn_like(actual) * 1000
            gradients = torch.autograd.grad(actual, inputs, upstream)
            self.assertTrue(all(torch.isfinite(x).all() for x in gradients))

    def test_dtype_and_device_rejections(self):
        for device in ("cpu", "cuda"):
            q = torch.ones(1, 24, 7, 256, device=device, dtype=torch.bfloat16)
            k = q[:, :2].contiguous()
            chosen = torch.full((7, 512), -1, device=device, dtype=torch.int32)
            for dtype in (torch.float32, torch.float64, torch.float8_e4m3fn, torch.int32):
                with self.assertRaisesRegex(ValueError, "group QSA"):
                    attention(q.to(dtype), k.to(dtype), k.to(dtype), chosen, 1 / 16)
            if device == "cpu":
                with self.assertRaisesRegex(ValueError, "CUDA"):
                    attention(q, k, k, chosen, 1 / 16)
                with self.assertRaisesRegex(ValueError, "CUDA"):
                    prepare_groups(chosen)
            else:
                with self.assertRaisesRegex(ValueError, "dtypes"):
                    attention(q, k.half(), k, chosen, 1 / 16)

    def test_union_matches_cpu_oracle(self):
        for lengths in ((1,), (2051,), (4099,), (3, 2061, 17)):
            for pattern in ("local", "random", "identical"):
                chosen = selections(lengths, pattern)
                expected = reference_union(chosen, lengths)
                _, ids, members, counts = prepare_groups(chosen.cuda(), lengths)
                ids, members, counts = ids.cpu(), members.cpu(), counts.cpu()
                for group, want in enumerate(expected):
                    count = counts[group].item()
                    self.assertEqual(
                        list(zip(ids[group, :count].tolist(), members[group, :count].tolist())),
                        want,
                    )

    def test_disjoint_union_and_padding(self):
        n = 16401
        chosen = selections((n,), "identical")
        start = 16384
        for lane in range(8):
            chosen[start + lane] = torch.arange(lane * 512, (lane + 1) * 512)
        chosen[start + 8:, ::3] = -1
        _, ids, members, counts = prepare_groups(chosen.cuda())
        group = start // GROUP
        count = counts[group].item()
        actual = dict(zip(ids[group, :count].cpu().tolist(), members[group, :count].cpu().tolist()))
        self.assertGreaterEqual(count, 4096)
        for block in range(4096):
            self.assertEqual(actual[block], 1 << (block // 512))
        for g in range(group + 1, len(counts)):
            self.assertGreater(counts[g].item(), 0)

    def test_checkpoint_selection_reuse_and_retention(self):
        torch.manual_seed(3)
        n = 137
        owner = SimpleNamespace(compress_ratio=4, token_budget=2048, block_topk=512)
        q = torch.randn(1, 24, n, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 2, n, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        expected = None
        for context in (None, selection_contexts, retention_contexts):
            calls = []

            def layer(q, k, v, calls=calls):
                def compute_selection():
                    calls.append("select")
                    return selections((n,)).cuda()

                chosen, _ = select_blocks(owner, q.transpose(1, 2), compute_selection)

                def core():
                    calls.append("core")
                    return attention(q, k, v, chosen, 1 / 16)

                result, _, _ = retain_attention(owner, (q, k, v), core)
                return result.sin().sum()

            options = {"context_fn": context} if context else {}
            output = checkpoint(layer, q, k, v, use_reentrant=False, **options)
            gradients = torch.autograd.grad(output, (q, k, v))
            if expected is None:
                expected = gradients
            else:
                for a, b in zip(gradients, expected, strict=True):
                    self.assertTrue(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
                self.assertEqual(calls.count("select"), 1)
            if context is retention_contexts:
                self.assertEqual(calls.count("core"), 1)

    def test_backward_dense_packed_extreme_and_repeatability(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        cases = [((7,), "random", 1), ((2053,), "random", 1),
                 ((4099,), "local", 1), ((2061,), "identical", 4),
                 ((3, 2053, 17), "random", 1), ((4099,), "padding", 1),
                 ((16384,), "random", 1), ((2060,), "empty", 1),
                 ((16401,), "disjoint", 1)]
        cases = [(*case, 1) for case in cases] + [((2053,), "random", 1, x) for x in (1e-12, 1e12)]
        for lengths, pattern, magnitude, grad_magnitude in cases:
            n = sum(lengths)
            torch.manual_seed(n)
            chosen = selections(lengths, pattern).cuda()
            q = (torch.randn(1, 24, n, 256, device="cuda", dtype=torch.bfloat16) * magnitude).requires_grad_()
            k = (torch.randn(1, 2, n, 256, device="cuda", dtype=torch.bfloat16) * magnitude).requires_grad_()
            v = torch.randn_like(k, requires_grad=True)
            grad = torch.randn_like(q) * grad_magnitude
            ref = dense_reference(q, k, v, grad, chosen, lengths)
            actual, lse = attention(q, k, v, chosen, 1 / 16, lengths, return_lse=True)
            deriv = torch.autograd.grad(actual, (q, k, v), grad)
            # Separate Flex calls allow unaligned document lengths; each call uses
            # exactly selection_metadata's per-document mask semantics.
            outputs, norms = [], []
            start, offset = 0, 0
            for length in lengths:
                local = chosen[start:start + length]
                local = torch.where(local >= 0, local - offset, local)
                mask, _, _ = selection_metadata(local, 4, 2048)
                outputs.append(sparse_attention(
                    q[:, :, start:start + length], k[:, :, start:start + length],
                    v[:, :, start:start + length], mask, 1 / 16,
                ))
                with torch.no_grad():
                    _, norm = _raw_flex(
                        q[:, :, start:start + length], k[:, :, start:start + length],
                        v[:, :, start:start + length], mask, 1 / 16,
                    )
                norms.append(norm[0])
                start += length
                offset += (length + 3) // 4
            flex = torch.cat(outputs, 2)
            flex_lse = torch.cat(norms, 1)
            flex_deriv = torch.autograd.grad(flex, (q, k, v), grad)
            report = {}
            for name, a, b, expected in zip(
                ("O", "LSE", "dQ", "dK", "dV"), (actual, lse, *deriv), (flex, flex_lse, *flex_deriv),
                ref, strict=True,
            ):
                error, baseline = errors(a, expected), errors(b, expected)
                report[name] = {"group": error, "flex": baseline}
                for metric in error:
                    self.assertLessEqual(error[metric], 1.5 * baseline[metric],
                                         (lengths, name, metric, report[name]))
            repeat, repeat_lse = attention(q, k, v, chosen, 1 / 16, lengths, return_lse=True)
            repeat_deriv = torch.autograd.grad(repeat, (q, k, v), grad)
            for a, b in zip((actual, lse, *deriv), (repeat, repeat_lse, *repeat_deriv), strict=True):
                self.assertTrue(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
            zero = attention(q, k, v, chosen, 1 / 16, lengths)
            for derivative in torch.autograd.grad(zero, (q, k, v), torch.zeros_like(grad)):
                self.assertEqual(torch.count_nonzero(derivative).item(), 0)
            print({"backward_dense_pass": lengths, "gradient_scale": grad_magnitude, "errors": report}, flush=True)

    def test_forward_against_dense(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        for n in (1, 7, 129, 2050, 2051, 2052, 2061, 4099):
            torch.manual_seed(n)
            chosen = selections((n,), "random").cuda()
            q = torch.randn(1, 24, n, 256, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, 2, n, 256, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            mask, _, _ = selection_metadata(chosen, 4, 2048)
            allow = mask.mask_mod(0, 0, torch.arange(n, device="cuda")[:, None],
                                  torch.arange(n, device="cuda")[None])
            actual, lse = forward(q, k, v, chosen, 1 / 16)
            flex, flex_lse = _raw_flex(q, k, v, mask, 1 / 16)
            # One head at a time bounds the FP32 dense oracle's workspace.
            expected_output = torch.empty_like(q, dtype=torch.float32)
            expected_norm = torch.empty_like(lse)
            for head in range(24):
                scores = q[0, head].float() @ k[0, head // 12].float().T / 16
                scores.masked_fill_(~allow, -torch.inf)
                expected_norm[head] = scores.logsumexp(-1) * 1.4426950408889634
                expected_output[0, head] = scores.softmax(-1) @ v[0, head // 12].float()
            report = {}
            for name, a, b, ref in (("O", actual, flex, expected_output),
                                    ("LSE", lse, flex_lse[0], expected_norm)):
                err, baseline = a.float() - ref, b.float() - ref
                report[name] = {"group_l2": err.norm().item(), "flex_l2": baseline.norm().item(),
                                "group_max_abs": err.abs().max().item(),
                                "flex_max_abs": baseline.abs().max().item()}
                self.assertLessEqual(err.norm().item(), 1.5 * baseline.norm().item(), (n, name, report))
                self.assertLessEqual(err.abs().max().item(), 1.5 * baseline.abs().max().item(), (n, name, report))
            repeated = forward(q, k, v, chosen, 1 / 16)
            self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8), repeated[0].contiguous().view(torch.uint8)))
            self.assertTrue(torch.equal(lse.contiguous().view(torch.uint8), repeated[1].contiguous().view(torch.uint8)))
            print({"forward_dense_pass": n, "errors": report}, flush=True)


if __name__ == "__main__":
    unittest.main()
