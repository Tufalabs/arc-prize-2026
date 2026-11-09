import gc
import unittest
import weakref

import torch
from torch.utils.checkpoint import checkpoint

from training.flash.qsa_retention import retain_attention, retention_contexts


class RetentionTests(unittest.TestCase):
    def test_gradient_gate_and_outstanding_graphs(self):
        baseline = None
        for retain in (False, True):
            calls, refs = [], []
            owner = object()
            w = torch.randn(
                4,
                4,
                generator=torch.Generator().manual_seed(7),
                dtype=torch.float64,
                requires_grad=True,
            )
            x = torch.randn(
                3,
                4,
                generator=torch.Generator().manual_seed(8),
                dtype=torch.float64,
                requires_grad=True,
            )

            def layer(x, w=w, calls=calls, refs=refs, owner=owner):
                q = x @ w

                def core():
                    calls.append(1)
                    result = torch.softmax(q @ q.T, dim=-1) @ q
                    refs.append(weakref.ref(result))
                    return result

                y, _, _ = retain_attention(owner, (q,), core)
                return ((y * q.sigmoid()) @ w).square().sum()

            options = {"context_fn": retention_contexts} if retain else {}
            # Two outstanding graphs followed by repeated backward on both.
            a = checkpoint(layer, x, use_reentrant=False, **options)
            b = checkpoint(layer, x * 2, use_reentrant=False, **options)
            loss = a + b
            loss.backward(retain_graph=True)
            first = (x.grad.clone(), w.grad.clone(), loss.detach().clone())
            x.grad = w.grad = None
            loss.backward()
            torch.testing.assert_close(x.grad, first[0], rtol=0, atol=0)
            torch.testing.assert_close(w.grad, first[1], rtol=0, atol=0)
            if retain:
                self.assertEqual(len(calls), 2)
                for actual, expected in zip(first, baseline):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                baseline = first
                self.assertGreater(len(calls), 2)
            del loss, a, b
            gc.collect()
            self.assertTrue(all(r() is None for r in refs))

    def test_input_gradient_only_and_no_grad(self):
        owner, calls = object(), []

        def layer(x):
            def core():
                calls.append(1)
                return x.sin()

            y, _, _ = retain_attention(owner, (x,), core)
            return y.square().sum()

        x = torch.randn(7, requires_grad=True)
        result = checkpoint(
            layer, x, use_reentrant=False, context_fn=retention_contexts
        )
        result.backward()
        torch.testing.assert_close(x.grad, 2 * x.sin() * x.cos())
        self.assertEqual(len(calls), 1)
        with torch.no_grad():
            checkpoint(layer, x, use_reentrant=False, context_fn=retention_contexts)
        self.assertEqual(len(calls), 2)

    def test_saved_tensor_mutation_rejected(self):
        owner = object()
        x = torch.randn(4, requires_grad=True)

        def layer(x):
            y, _, _ = retain_attention(owner, (x,), lambda: x.square())
            return y.sum()

        result = checkpoint(
            layer, x, use_reentrant=False, context_fn=retention_contexts
        )
        with torch.no_grad():
            x.add_(1)
        with self.assertRaisesRegex(RuntimeError, "modified"):
            result.backward()

    def test_exception_does_not_leak_scope(self):
        forward, _ = retention_contexts()
        with self.assertRaisesRegex(ValueError, "test"), forward:
            raise ValueError("test")
        x = torch.randn(4, requires_grad=True)
        _, reused, nbytes = retain_attention(object(), (x,), lambda: x.sin())
        self.assertFalse(reused)
        self.assertEqual(nbytes, 0)


if __name__ == "__main__":
    unittest.main()
