import copy
import gc
import unittest
import weakref
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from sft.distillation.accumulation import validate_resume_manifest
from sft.distillation.train import worker_command
from training.flash.offload import checkpoint_layer
from training.flash.qsa_selection import select_blocks, selection_contexts


class ToySelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.indexer = SimpleNamespace(compress_ratio=4, token_budget=8, block_topk=2)
        self.calls = 0
        self.reuses = 0
        self.references = []

    def forward(self, x):
        def compute():
            self.calls += 1
            with torch.no_grad():
                ids = x[0].square().topk(2, dim=-1).indices.int()
            self.references.append(weakref.ref(ids))
            return ids

        ids, reused = select_blocks(self.indexer, x, compute)
        self.reuses += int(reused)
        return (x[0] @ self.weight).gather(1, ids.long()).sin().square().sum()


class SelectionReuseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_gradients_optimizer_and_fresh_selections_across_updates(self):
        base = ToySelector()
        cached = copy.deepcopy(base)
        wrapped = checkpoint_layer(cached, 0, 0, reuse_qsa_selection=True)
        opts = [torch.optim.AdamW(m.parameters(), lr=0.01) for m in (base, cached)]
        for _ in range(3):
            for opt in opts:
                opt.zero_grad()
            for n in (7, 11):  # sequential microbatch accumulation
                x = torch.randn(1, n, 4)
                a, b = (x.clone().requires_grad_() for _ in range(2))
                expected, actual = base(a), wrapped(b)
                expected.backward()
                actual.backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(b.grad, a.grad, rtol=0, atol=0)
                torch.testing.assert_close(
                    cached.weight.grad, base.weight.grad, rtol=0, atol=0
                )
            for opt in opts:
                opt.step()
            torch.testing.assert_close(cached.weight, base.weight, rtol=0, atol=0)
        self.assertEqual((cached.calls, cached.reuses), (6, 6))

    def test_multiple_outstanding_forwards_and_reverse_backward(self):
        base = ToySelector()
        cached = copy.deepcopy(base)
        xs = [torch.randn(1, n, 4, requires_grad=True) for n in (7, 7, 13)]
        ys = [x.detach().clone().requires_grad_() for x in xs]
        expected = [base(x) for x in xs]
        actual = [
            checkpoint(cached, y, use_reentrant=False, context_fn=selection_contexts)
            for y in ys
        ]
        for a, b in zip(reversed(expected), reversed(actual)):
            a.backward()
            b.backward()
        for x, y in zip(xs, ys):
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        torch.testing.assert_close(base.weight.grad, cached.weight.grad, rtol=0, atol=0)
        self.assertEqual((cached.calls, cached.reuses), (3, 3))

    def test_repeated_backward_retained_graph_and_cache_lifetime(self):
        cached = ToySelector()
        x = torch.randn(1, 7, 4, requires_grad=True)
        out = checkpoint(cached, x, use_reentrant=False, context_fn=selection_contexts)
        ref = cached.references[0]
        out.backward(retain_graph=True)
        grad = x.grad.clone()
        x.grad = None
        out.backward()
        torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
        self.assertEqual((cached.calls, cached.reuses), (1, 2))
        del out
        gc.collect()
        self.assertIsNone(ref())

    def test_no_grad_and_uncheckpointed_calls_do_not_share_cache(self):
        model = ToySelector()
        for _ in range(3):
            with torch.no_grad():
                checkpoint(
                    model,
                    torch.randn(1, 7, 4),
                    use_reentrant=False,
                    context_fn=selection_contexts,
                )
        self.assertEqual((model.calls, model.reuses), (3, 0))
        self.assertTrue(all(r() is None for r in model.references))

    def test_nested_contexts_and_exception_cleanup(self):
        model = ToySelector()
        x = torch.randn(1, 7, 4)
        outer, replay = selection_contexts()
        with outer:
            model(x)
            inner, _ = selection_contexts()
            with self.assertRaisesRegex(ValueError, "test"), inner:
                model(x)
                raise ValueError("test")
            model(x)
        with replay:
            model(x)
            model(x)
        model(x)
        self.assertEqual((model.calls, model.reuses), (4, 2))

    def test_reject_missing_mismatched_and_mutated_selection(self):
        model = ToySelector()
        x = torch.randn(1, 7, 4)
        _, replay = selection_contexts()
        with self.assertRaisesRegex(RuntimeError, "no forward selection"), replay:
            model(x)
        forward, replay = selection_contexts()
        with forward:
            model(x)
        with self.assertRaisesRegex(RuntimeError, "invocation mismatch"), replay:
            model(torch.randn(1, 8, 4))
        model.references[-1]().zero_()
        with self.assertRaisesRegex(RuntimeError, "modified in place"), replay:
            model(x)

    def test_resume_policy_and_worker_argument(self):
        validate_resume_manifest({}, {"reuse_qsa_selection": False})
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_resume_manifest({}, {"reuse_qsa_selection": True})
        c = {
            "teacher": "t",
            "student": "s",
            "data": "d",
            "cache": "c",
            "steps": 1,
            "lr": 1e-5,
            "temperature": 1,
            "ce_weight": 0,
            "chunk": 64,
            "save_every": 1,
            "seed": 1,
            "attention_backend": "flex_qsa",
            "reuse_qsa_selection": True,
        }
        self.assertIn("--reuse-qsa-selection", worker_command(c, "result"))


if __name__ == "__main__":
    unittest.main()
