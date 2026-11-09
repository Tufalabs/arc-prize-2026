import unittest
from unittest.mock import patch

import torch

from sft.distillation.accumulation import validate_resume_manifest
from training.flash.batched_qsa import SELECTOR_POLICY, _batched


def apply_rotary_pos_emb(x, **kwargs):
    return x


class UnguardedQSATest(unittest.TestCase):
    def test_tied_queries_stay_on_device_and_masks_are_causal(self):
        module = type("IndexerFixture", (), {})()
        for name, value in dict(
            selector_policy=SELECTOR_POLICY,
            compress_ratio=2,
            token_budget=4,
            block_topk=2,
            _qsa_tile=4,
            index_n_heads=1,
            index_kv_heads=1,
            index_head_dim=2,
            index_qk_proj=torch.nn.Linear(2, 4, bias=False).bfloat16(),
            q_layernorm=torch.nn.Identity(),
            k_layernorm=torch.nn.Identity(),
        ).items():
            setattr(module, name, value)
        module.index_qk_proj.weight.data.zero_()
        x = torch.zeros(1, 17, 2, dtype=torch.bfloat16)
        pe = (torch.ones_like(x), torch.zeros_like(x))
        causal = torch.ones(17, 17, dtype=torch.bool).tril()[None, None]
        # Exact ties exercised the former guard. No dynamic host query list is allowed.
        with (
            patch.object(
                torch.Tensor, "nonzero", side_effect=AssertionError("host guard")
            ),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("host list")
            ),
        ):
            ids = _batched(module, x, pe, None, return_blocks=True)
            actual = _batched(module, x, pe, causal)[0, 0]
            replay = _batched(module, x, pe, None, return_blocks=True)
        self.assertTrue(torch.equal(ids, replay))
        self.assertEqual(module._qsa_last_stats["selector_policy"], SELECTOR_POLICY)
        self.assertEqual(module._qsa_last_stats["guarded_rows"], 0)
        expected = causal[0, 0].clone()
        for q in range(5, 17):
            chosen = ids[q].tolist()
            self.assertEqual(len(set(chosen)), 2)
            self.assertTrue(all(0 <= i < (q + 1) // 2 for i in chosen))
            expected[q].zero_()
            for i in chosen:
                expected[q, 2 * i : 2 * i + 2] = True
            expected[q, ((q + 1) // 2) * 2 : q + 1] = True
        self.assertTrue(torch.equal(actual, expected))

    def test_resume_cannot_silently_change_selector(self):
        current = {"qsa_selector_policy": SELECTOR_POLICY}
        validate_resume_manifest(current, current)
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_resume_manifest({}, current)
