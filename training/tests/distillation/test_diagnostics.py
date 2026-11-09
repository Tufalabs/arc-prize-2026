import copy
import unittest

import torch
from sft.distillation.diagnostics import (
    assert_tree_exact,
    compare_tensors,
    cpu_tree,
    tree_digest,
)


class DiagnosticTests(unittest.TestCase):
    def test_exact_restore_and_corruption_detection(self):
        p = torch.nn.Parameter(torch.randn(7, 4))
        opt = torch.optim.AdamW([p], lr=1e-5, weight_decay=0, foreach=False)
        p.square().sum().backward()
        opt.step()
        saved = cpu_tree(opt.state_dict())
        q = torch.nn.Parameter(p.detach().clone())
        restored = torch.optim.AdamW([q], lr=1e-5, weight_decay=0, foreach=False)
        restored.load_state_dict(copy.deepcopy(saved))
        assert_tree_exact(restored.state_dict(), saved)
        self.assertEqual(tree_digest(restored.state_dict()), tree_digest(saved))
        bad = copy.deepcopy(saved)
        bad["state"][0]["exp_avg"][0, 0] += 1e-4
        with self.assertRaises(AssertionError):
            assert_tree_exact(restored.state_dict(), bad)
        bad = copy.deepcopy(saved)
        bad["state"][0]["step"] += 1
        with self.assertRaises(AssertionError):
            assert_tree_exact(restored.state_dict(), bad)
        bad = copy.deepcopy(saved)
        bad["param_groups"][0]["lr"] = 2e-5
        with self.assertRaises(AssertionError):
            assert_tree_exact(restored.state_dict(), bad)

    def test_tensor_difference(self):
        a = {"w": torch.tensor([1.0, 2.0, 3.0])}
        b = {"w": torch.tensor([1.0, 2.0, 4.0])}
        d = compare_tensors(a, b)
        self.assertFalse(d["exact"])
        self.assertEqual(d["max_abs"], 1.0)
        self.assertEqual(d["per_tensor"]["w"]["changed"], 1)
        self.assertTrue(compare_tensors(a, a)["exact"])

    def test_snapshot_does_not_alias_live_optimizer(self):
        a = {"x": torch.ones(3)}
        b = cpu_tree(a)
        a["x"].zero_()
        self.assertEqual(b["x"].sum(), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
