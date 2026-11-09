import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from sft.distillation.expert_metrics import ExpertUsageWindow
from sft.distillation.wandb_logging import Tail, WandbLogger, probe_payload


def usage(counts, masses=None):
    return {
        "completion_predictions": {
            "selection_counts": counts,
            "routing_mass": masses or counts,
        }
    }


class ExpertMetricsTests(unittest.TestCase):
    def test_uniform_and_collapsed_layers(self):
        window = ExpertUsageWindow({"bottom_k": 1})
        window.add(usage([[1, 1, 1, 1], [4, 0, 0, 0]]))
        result = window.flush()
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result["experts/effective_expert_fraction"], 0.625)
        self.assertAlmostEqual(
            result["experts/worst_layer_effective_expert_fraction"], 0.25
        )
        self.assertAlmostEqual(result["experts/bottom_candidate_routing_mass"], 0.125)
        self.assertEqual(window.steps, 0)
        self.assertEqual(window.flush(), {})

    def test_pool_counts_not_step_metrics_and_rank_candidates_by_frequency(self):
        window = ExpertUsageWindow({"bottom_k": 1})
        window.add(usage([[9, 1, 0]], [[1, 8, 1]]))
        window.add(usage([[0, 0, 90]], [[0, 0, 90]]))
        result = window.flush()
        entropy = -sum(p * math.log(p) for p in (0.09, 0.01, 0.90))
        self.assertAlmostEqual(
            result["experts/effective_expert_fraction"], math.exp(entropy) / 3
        )
        # Least selected expert1 has 8% of mass; least-mass expert0 would be wrong.
        self.assertAlmostEqual(result["experts/bottom_candidate_routing_mass"], 0.08)

    def test_ties_and_empty_data(self):
        window = ExpertUsageWindow({"bottom_k": 1})
        window.add(usage([[1, 1, 3]], [[2, 1, 2]]))
        self.assertAlmostEqual(
            window.flush()["experts/bottom_candidate_routing_mass"], 0.4
        )
        window.add(usage([[0, 0, 0]]))
        self.assertEqual(window.flush(), {})
        with self.assertRaises(ValueError):
            window.add(usage([[float("nan"), 0, 0]]))

    def test_probe_average_no_initial_stability_or_layer_curves(self):
        report = {
            "step": 50,
            "baseline_kl": 0.2,
            "token_topk_overlap": [0.3, 0.4],
            "criteria": {
                "frequency": {
                    "layers": [{"overlap_previous": 0.5}, {"overlap_previous": 1.0}],
                    "min_overlap_previous": 0.5,
                    "ablation_delta_kl": 0.1,
                }
            },
        }
        result, rows = probe_payload(report)
        self.assertEqual(result["experts/pruning_candidate_stability"], 0.75)
        self.assertEqual(rows, [])
        self.assertEqual(result["probe/frequency/ablation_delta_kl"], 0.1)
        self.assertFalse(any("layer" in k or "min_overlap" in k for k in result))
        report["criteria"]["frequency"]["layers"][0]["overlap_previous"] = None
        self.assertNotIn(
            "experts/pruning_candidate_stability", probe_payload(report)[0]
        )

    def test_poll_windows_final_partial_and_no_duplicate_or_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student = root / "student"
            student.mkdir()
            logger = WandbLogger.__new__(WandbLogger)
            logger.result, logger.config = root, {"steps": 3, "cache": str(root)}
            logger.run, logger.wandb = Mock(), Mock()
            logger.interval = 2
            logger.expert_window = ExpertUsageWindow({"bottom_k": 1})
            logger.pending, logger.last_step, logger.probe_seen = {}, 0, set()
            logger.host_tail = logger.gpu_tail = Tail(root / "missing")
            logger.pinned_tails = logger.teacher_tails = []
            logger.tails = []
            for rank in range(4):
                path = student / f"metrics-rank{rank}.jsonl"
                rows = [
                    {
                        "step": step,
                        "seconds": 1,
                        "loss_per_token": 0.1,
                        "router_grad_norm": 0.3,
                        "global_completion_tokens": 4,
                        "sequence_length": 4,
                        "peak_allocated_gib": 1,
                        "peak_reserved_gib": 1,
                    }
                    for step in (1, 2, 3)
                ]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                logger.tails.append(Tail(path))
            for step in (1, 2, 3):
                (student / f"expert-usage-step{step}.json").write_text(
                    json.dumps(usage([[1, 1, 1]]))
                )
            (root / "teacher-expert-usage.json").write_text(
                json.dumps(usage([[1, 1, 1]]))
            )
            logger.poll()
            payloads = [call.args[0] for call in logger.run.log.call_args_list]
            expert_payloads = [
                p for p in payloads if any(k.startswith("experts/") for k in p)
            ]
            self.assertEqual([p["train/step"] for p in expert_payloads], [2, 3])
            self.assertTrue(all(len(p) == 4 for p in expert_payloads))
            logger.wandb.Table.assert_not_called()
            calls = logger.run.log.call_count
            logger.poll()
            self.assertEqual(logger.run.log.call_count, calls)
