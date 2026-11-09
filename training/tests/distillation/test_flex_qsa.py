import unittest
from unittest.mock import patch

import torch

from sft.distillation.accumulation import validate_resume_manifest
from sft.distillation.train import worker_command
from training.flash.flex_qsa import selection_metadata


class FlexQSATest(unittest.TestCase):
    def test_telemetry_empty_and_checkpoint_partial_spans(self):
        from training.flash import flex_qsa

        class Event:
            def __init__(self, value):
                self.value = value

            def elapsed_time(self, other):
                return other.value - self.value

        flex_qsa.begin_metrics()
        self.assertEqual(flex_qsa.finish_metrics(), {})
        self.assertEqual(flex_qsa.finish_metrics(), {})
        flex_qsa._events = [
            {
                "stage": stage,
                "events": [Event(t) for t in (0, 2, 5, 12)],
                "live": torch.tensor(3),
                "total": 8,
                "causal": 4,
            }
            for stage in (0, 1, 2, 3)
        ]
        result = flex_qsa.finish_metrics()
        self.assertEqual(result["calls_including_recompute"], 4)
        self.assertEqual(result["completed_calls_including_recompute"], 1)
        self.assertEqual(result["selection_ms"], 6)
        self.assertEqual(result["mask_build_ms"], 6)
        self.assertEqual(result["completed_forward_after_mask_ms"], 7)
        self.assertEqual(result["tile_skip_fraction"], 0.625)
        self.assertEqual(result["qsa_extra_tile_skip_fraction"], 0.25)
        self.assertIsNone(flex_qsa._events)
        flex_qsa._events = [{"stage": 0, "events": []}]
        self.assertNotIn("tile_skip_fraction", flex_qsa.finish_metrics())

    def test_exact_edges_and_conservative_tiles(self):
        # Include partial final tiles, causal prefix, tail, and signed bit31.
        n, ratio, budget = 263, 4, 16
        chosen = torch.full((n, budget // ratio), -1, dtype=torch.int32)
        generator = torch.Generator().manual_seed(91)
        for q in range(budget + ratio - 1, n):
            chosen[q] = torch.randperm((q + 1) // ratio, generator=generator)[
                : budget // ratio
            ]
        chosen[131] = torch.tensor([0, 5, 12, 31])
        mask, live, total = selection_metadata(chosen, ratio, budget)
        qs, ks = torch.arange(n)[:, None], torch.arange(n)[None, :]
        actual = mask.mask_mod(torch.tensor(0), torch.tensor(0), qs, ks)
        expected = torch.zeros(n, n, dtype=torch.bool)
        for q in range(n):
            if q < budget + ratio - 1:
                expected[q, : q + 1] = True
            else:
                for block in chosen[q].tolist():
                    expected[q, block * ratio : (block + 1) * ratio] = True
                expected[q, ((q + 1) // ratio) * ratio : q + 1] = True
        self.assertTrue(torch.equal(actual, expected))
        for q in range(n):
            qi = q // 128
            count = mask.kv_num_blocks[0, 0, qi]
            scheduled = mask.kv_indices[0, 0, qi, :count].tolist()
            for k in expected[q].nonzero().flatten().tolist():
                self.assertIn(k // 128, scheduled)
        self.assertLessEqual(live, total)
        self.assertFalse(
            mask.mask_mod(
                torch.tensor(0), torch.tensor(0), torch.tensor(n), torch.tensor(n)
            ).item()
        )

    def test_short_dense_prefix(self):
        for n in (1, 3, 32, 129):
            ids = torch.full((n, 512), -1, dtype=torch.int32)
            mask, _, _ = selection_metadata(ids, 4, 2048)
            q, k = torch.arange(n)[:, None], torch.arange(n)[None, :]
            self.assertTrue(torch.equal(mask.mask_mod(0, 0, q, k), k <= q))

    def test_packed_documents_are_causal_and_isolated(self):
        n, ratio, budget = 256, 4, 16
        chosen = torch.full((n, budget // ratio), -1, dtype=torch.int32)
        mask, _, _ = selection_metadata(
            chosen, ratio, budget, packed_lengths=(128, 128)
        )
        q, k = torch.arange(n)[:, None], torch.arange(n)[None, :]
        actual = mask.mask_mod(0, 0, q, k)
        self.assertFalse(actual[:128, 128:].any())
        self.assertFalse(actual[128:, :128].any())
        self.assertTrue(torch.equal(actual[:19, :128], k[:, :128] <= q[:19]))
        self.assertTrue(torch.equal(actual[128:147, 128:], k[:, 128:] <= q[128:147]))

    def test_packed_document_block_ids_receive_global_offsets(self):
        from training.flash import batched_qsa

        module = type("Indexer", (), {"compress_ratio": 4, "block_topk": 2,
                                      "selector_policy": batched_qsa.SELECTOR_POLICY})()
        x = torch.zeros(1, 16, 3)
        positions = (torch.zeros(1, 16, 3), torch.zeros(1, 16, 3))
        local = torch.tensor(
            [
                [0, 1],
                [1, -1],
                [0, -1],
                [1, 0],
                [0, 1],
                [1, -1],
                [0, -1],
                [1, 0],
            ],
            dtype=torch.int32,
        )
        with patch.object(batched_qsa, "_batched", side_effect=[local, local]):
            actual = batched_qsa.batched_documents(module, x, positions, (8, 8))
        torch.testing.assert_close(actual[:8], local)
        torch.testing.assert_close(
            actual[8:], torch.where(local >= 0, local + 2, local)
        )

    def test_resume_backend_guard(self):
        validate_resume_manifest({}, {"attention_backend": "sdpa"})
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_resume_manifest({}, {"attention_backend": "flex_qsa"})

    def test_command_backend(self):
        config = {
            "teacher": "t",
            "student": "s",
            "data": "d",
            "cache": "c",
            "steps": 2,
            "lr": 1e-5,
            "temperature": 1,
            "ce_weight": 0,
            "chunk": 64,
            "save_every": 1,
            "seed": 1,
            "attention_backend": "flex_qsa",
        }
        command = worker_command(config, "result")
        self.assertEqual(command[command.index("--attention-backend") + 1], "flex_qsa")

    def test_reject_other_block_size(self):
        with self.assertRaises(ValueError):
            selection_metadata(torch.zeros(20, 4, dtype=torch.int32), 8, 32)


if __name__ == "__main__":
    unittest.main()
