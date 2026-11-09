import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sft.distillation.common import Dataset, batch_indices, row_key
from sft.distillation.loss import (
    ChunkedDistillation,
    completion_distillation,
    fp32_logits,
)
from sft.distillation.routing import RouteRecorder


class DistillationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_fp32_logits_do_not_round_bf16_projection_output(self):
        hidden = torch.tensor([[1.003, -0.997]], dtype=torch.bfloat16)
        head = torch.tensor([[16.25, 15.75], [-17.0, 13.0]], dtype=torch.bfloat16)
        actual = fp32_logits(hidden, head)
        expected = F.linear(hidden.float(), head.float())
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_full_vocabulary_loss_and_gradient(self):
        for temperature in [0.7, 1.0, 2.3]:
            for ce in [0.0, 0.2]:
                h = torch.randn(7, 5, requires_grad=True)
                sh, th, ht = torch.randn(11, 5), torch.randn(11, 4), torch.randn(7, 4)
                targets = torch.randint(11, (7,))
                targets[2] = -100  # ignore_index must drop the CE term and its gradient
                actual = ChunkedDistillation.apply(
                    h, sh, ht, th, targets, temperature, ce, 3
                )
                s, t = F.linear(h, sh), F.linear(ht, th)
                lt, ls = (
                    (t / temperature).log_softmax(-1),
                    (s / temperature).log_softmax(-1),
                )
                reference = temperature**2 * (lt.exp() * (lt - ls)).sum()
                reference += ce * F.cross_entropy(s, targets, reduction="sum")
                torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
                ga = torch.autograd.grad(actual * 0.37, h, retain_graph=True)[0]
                gr = torch.autograd.grad(reference * 0.37, h)[0]
                torch.testing.assert_close(ga, gr, rtol=2e-5, atol=2e-6)

    def test_completion_shift_and_no_prompt_gradient(self):
        hidden = torch.randn(1, 9, 4, requires_grad=True)
        head, teacher = torch.randn(13, 4), torch.randn(5, 4)
        ids = torch.randint(13, (1, 9))
        loss = completion_distillation(hidden, head, teacher, head, ids, 4, chunk=2)
        loss.backward()
        self.assertEqual(hidden.grad[0, :3].abs().sum().item(), 0)
        self.assertEqual(hidden.grad[0, -1].abs().sum().item(), 0)
        self.assertGreater(hidden.grad[0, 3:-1].abs().sum().item(), 0)

    def test_identical_distributions_zero_kl(self):
        h, head = torch.randn(5, 4, requires_grad=True), torch.randn(11, 4)
        loss = ChunkedDistillation.apply(
            h,
            head,
            h.detach().clone(),
            head,
            torch.zeros(5, dtype=torch.long),
            1.0,
            0.0,
            2,
        )
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(h.grad, torch.zeros_like(h), atol=1e-6, rtol=0)

    def test_teacher_gradient_rejected(self):
        with self.assertRaises(ValueError):
            ChunkedDistillation.apply(
                torch.randn(2, 3, requires_grad=True),
                torch.randn(5, 3),
                torch.randn(2, 3, requires_grad=True),
                torch.randn(5, 3),
                torch.zeros(2, dtype=torch.long),
                1.0,
                0.0,
                1,
            )

    def test_usage_not_doubled_by_checkpoint_recomputation(self):
        recorder = RouteRecorder(1, 3, "cpu")
        x = torch.randn(6, 3, requires_grad=True)
        calls = []

        def layer(x):
            calls.append(1)
            probs = x.softmax(-1)
            weights, indices = probs.topk(2, -1)
            recorder.record(0, indices, weights)
            return weights.square().sum()

        recorder.begin(3)
        loss = checkpoint(layer, x, use_reentrant=False)
        recorder.finish()
        loss.backward()
        report = recorder.report(distributed=False)
        self.assertGreater(len(calls), 1)
        self.assertEqual(sum(report["all_context"]["selection_counts"][0]), 12)
        self.assertEqual(
            sum(report["completion_predictions"]["selection_counts"][0]), 6
        )

    def test_rows_and_resume_order(self):
        row = {"input_token_ids": [1, 2], "output_token_ids": [3, 4]}
        other = {"input_token_ids": [1, 2, 3], "output_token_ids": [4]}
        self.assertNotEqual(row_key(row), row_key(other))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "data.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
            data = Dataset(path)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[1], other)
        original = [batch_indices(11, 4, s, 42) for s in range(7)]
        self.assertEqual(
            original[3:], [batch_indices(11, 4, s, 42) for s in range(3, 7)]
        )
        self.assertEqual(len(set(original[0] + original[1])), 8)

    def test_token_weighted_distributed_gradient(self):
        p = torch.tensor(0.2, requires_grad=True)
        sums = [
            ((p - torch.tensor(values)) ** 2).sum()
            for values in [[1.0, 2.0], [3.0, 4.0, 5.0]]
        ]
        grads = [torch.autograd.grad(s * 2 / 5, p, retain_graph=True)[0] for s in sums]
        expected = torch.autograd.grad(sum(sums) / 5, p)[0]
        torch.testing.assert_close(sum(grads) / 2, expected)

    def test_router_only_update_and_optimizer_restore(self):
        from sft.distillation.diagnostics import assert_tree_exact, cpu_tree

        router = torch.nn.Parameter(torch.randn(3, 4) * 0.1)
        experts = torch.randn(3, 4, 4)
        head = torch.randn(9, 4)
        x, teacher = torch.randn(6, 4), torch.randn(6, 4)
        frozen = [experts.clone(), head.clone()]
        initial = router.detach().clone()
        optimizer = torch.optim.AdamW([router], lr=1e-3)

        def loss():
            weights = F.linear(x, router).softmax(-1)
            expert_outputs = torch.einsum("ni,eij->nej", x, experts)
            hidden = (weights.unsqueeze(-1) * expert_outputs).sum(1)
            return ChunkedDistillation.apply(
                hidden,
                head,
                teacher,
                head,
                torch.zeros(6, dtype=torch.long),
                1.0,
                0.0,
                2,
            )

        loss().backward()
        optimizer.step()
        self.assertFalse(torch.equal(initial, router))
        self.assertIsNone(experts.grad)
        self.assertIsNone(head.grad)
        torch.testing.assert_close(experts, frozen[0], rtol=0, atol=0)
        torch.testing.assert_close(head, frozen[1], rtol=0, atol=0)
        saved = cpu_tree(optimizer.state_dict())
        restored = torch.optim.AdamW([router], lr=1e-3)
        restored.load_state_dict(saved)
        assert_tree_exact(restored.state_dict(), saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
