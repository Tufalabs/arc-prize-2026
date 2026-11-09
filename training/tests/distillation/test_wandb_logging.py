import tempfile
import unittest
from pathlib import Path

from sft.distillation.wandb_logging import (
    Tail,
    WandbLogger,
    aggregate_step,
)


class WandbLoggingTests(unittest.TestCase):
    def test_online_entity_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "explicit W&B entity"):
                WandbLogger({"wandb": {"mode": "online", "entity": None}}, directory)

    def test_wait_for_complete_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            tail = Tail(path)
            self.assertEqual(tail.lines(), [])
            path.write_text("first\npart")
            self.assertEqual(tail.lines(), ["first\n"])
            with path.open("a") as f:
                f.write("ial\n")
            self.assertEqual(tail.lines(), ["partial\n"])
            self.assertEqual(tail.lines(), [])

    def test_global_metrics_are_not_multiplied_by_rank_count(self):
        rows = [
            {
                "step": 12,
                "loss_per_token": 0.2,
                "router_grad_norm": 0.3,
                "seconds": 10 + i,
                "global_completion_tokens": 1000,
                "sequence_length": 2000,
                "peak_allocated_gib": 40 + i,
                "peak_reserved_gib": 50 + i,
            }
            for i in range(4)
        ]
        metrics = aggregate_step(rows)
        self.assertEqual(metrics["train/loss_per_completion_token"], 0.2)
        self.assertEqual(metrics["train/global_completion_tokens"], 1000)
        self.assertEqual(metrics["train/completion_tokens_per_second"], 1000 / 13)
        self.assertEqual(metrics["train/input_tokens_per_second"], 8000 / 13)
        self.assertEqual(metrics["train/step"], 12)
        single = aggregate_step(rows[:1])
        self.assertEqual(single["train/input_tokens_per_second"], 2000 / 10)
        with self.assertRaises(ValueError):
            aggregate_step([])
        with self.assertRaises(ValueError):
            aggregate_step([rows[0], dict(rows[1], step=13)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
