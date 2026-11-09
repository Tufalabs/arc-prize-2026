import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from sft.distillation.common import dump, row_key, sha
from sft.distillation.flash_model import router_forward
from sft.distillation.pruning_metrics import (
    bottom,
    layer_metrics,
    route_overlap,
    spearman,
)
from sft.distillation.pruning_probe import (
    ProbeRecorder,
    PruningProbe,
    score,
    temporary_pruning,
    validate_probe,
)
from sft.distillation.routing import RouteRecorder, attach
from sft.distillation.wandb_logging import probe_payload


class PruningTests(unittest.TestCase):
    def test_overlap_persistence_and_boundary(self):
        report = layer_metrics([1, 0, 2, 4], [0, 2, 1, 4], [[0, 2, 1, 4]], 2)
        self.assertEqual(report["overlap_previous"], 0.5)
        self.assertEqual(report["candidate_persistence_min"], 0.5)
        self.assertEqual(report["boundary_gap"], 1)
        self.assertAlmostEqual(report["boundary_gap_normalized"], 1 / 7)

    def test_ties_not_false_rank_confidence(self):
        self.assertIsNone(spearman([0, 0, 0], [0, 0, 0]))
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertAlmostEqual(spearman([1, 1, 2], [2, 2, 1]), -1)
        self.assertTrue(layer_metrics([0, 0, 0], None, [], 1)["boundary_tied"])
        self.assertEqual(bottom([0, 0, 0], 2), [0, 1])
        with self.assertRaises(ValueError):
            bottom([0, 0], 2)

    def test_same_token_set_overlap(self):
        a = torch.tensor([[[1, 2], [3, 4]]])
        b = torch.tensor([[[2, 1], [1, 4]]])
        torch.testing.assert_close(route_overlap(a, b), torch.tensor([[1.0, 0.5]]))
        with self.assertRaises(ValueError):
            route_overlap(a, b[:, :1])

    def test_mask_reroutes_and_restores_after_exception(self):
        gate = SimpleNamespace(
            weight=torch.tensor([[4.0], [3.0], [2.0], [1.0]]),
            hidden_dim=1,
            top_k=2,
            norm_topk_prob=True,
        )
        weights = gate.weight.clone()
        self.assertEqual(router_forward(gate, torch.ones(1, 1))[2].tolist(), [[0, 1]])
        with (
            self.assertRaisesRegex(RuntimeError, "injected"),
            temporary_pruning([gate], [[0, 1]]),
        ):
            _, values, ids = router_forward(gate, torch.ones(1, 1))
            self.assertEqual(ids.tolist(), [[2, 3]])
            self.assertAlmostEqual(values.sum().item(), 1.0)
            raise RuntimeError("injected")
        self.assertIsNone(gate.probe_excluded_experts)
        torch.testing.assert_close(weights, gate.weight, rtol=0, atol=0)
        with self.assertRaises(ValueError), temporary_pruning([gate], [[0, 1, 2]]):
            pass
        self.assertIsNone(gate.probe_excluded_experts)

    def test_recorder_samples_completion_only_and_ignores_recompute(self):
        r = ProbeRecorder(1, 4, "cpu", 2)
        ids = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]])
        r.begin(2)
        r.record(0, ids, torch.ones_like(ids).float() / 2)
        r.finish()
        r.record(0, ids, torch.ones_like(ids).float() / 2)
        self.assertEqual(r.counts[1].sum().item(), 6)
        self.assertEqual(r.routes[0].tolist(), [[1, 2], [0, 3]])

    def test_chunked_heldout_loss_and_penalty(self):
        teacher = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
        head = torch.eye(2)
        target = torch.tensor([0, 1, 0])
        identical = score(teacher, head, teacher, head, target, 1.0, 1)
        bad = score(-teacher, head, teacher, head, target, 1.0, 2)
        self.assertAlmostEqual(identical[0].item(), 0.0)
        self.assertEqual(identical[2:].tolist(), [3.0, 3.0])
        self.assertGreater(bad[0], identical[0])
        self.assertGreater(bad[1], identical[1])
        torch.testing.assert_close(
            bad, score(-teacher, head, teacher, head, target, 1.0, 3)
        )

    def test_resume_identity_guard(self):
        p = PruningProbe.__new__(PruningProbe)
        p.identity = {"data": "original"}
        with self.assertRaises(ValueError):
            p.restore({"identity": {"data": "changed"}})
        state = {"identity": p.identity, "history": [], "routes": {}, "step": 5}
        p.restore(state)
        self.assertEqual(p.state, state)

    def test_holdout_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split, probe, train, manifest = [
                root / x
                for x in ["split.json", "probe.jsonl", "train.jsonl", "manifest.json"]
            ]
            dump(
                split,
                {
                    "summary": {
                        "train": {"games": ["train"]},
                        "eval": {"games": ["eval"]},
                    }
                },
            )
            row = {"game_id": "eval", "input_token_ids": [1], "output_token_ids": [2]}
            probe.write_text(json.dumps(row) + "\n")
            train.write_text(
                json.dumps(dict(row, game_id="train", output_token_ids=[3])) + "\n"
            )
            dump(
                manifest,
                {
                    "dataset_sha256": sha(probe),
                    "split": str(split),
                    "split_sha256": sha(split),
                },
            )
            c = {
                "data": str(train),
                "probe": {
                    "data": str(probe),
                    "manifest": str(manifest),
                    "data_sha256": sha(probe),
                    "manifest_sha256": sha(manifest),
                    "every": 1,
                    "window": 5,
                    "sample_tokens": 256,
                    "bottom_fraction": 0.1,
                    "min_persistence": 0.95,
                    "max_window_kl_range": 0.001,
                },
            }
            validate_probe(c)
            train.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_probe(c)
            probe.write_text(probe.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_probe(c)

    def test_end_to_end_probe_is_reversible_and_resume_keeps_history(self):
        class Gate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
                )
                self.hidden_dim, self.top_k, self.norm_topk_prob = 2, 1, True

            def forward(self, hidden):
                return router_forward(self, hidden)

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Module()
                self.mlp.gate = Gate()
                self.fail = False

            def forward(self, input_ids, use_cache=False):
                x = torch.nn.functional.one_hot(input_ids[0], 2).float()
                _, values, ids = self.mlp.gate(x)
                if self.fail:
                    raise RuntimeError("injected forward failure")
                experts = torch.tensor([[2.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
                hidden = (experts[ids] * values.unsqueeze(-1)).sum(1)
                return SimpleNamespace(last_hidden_state=hidden.unsqueeze(0))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {
                "input_token_ids": [0],
                "output_token_ids": [1, 0],
                "game_id": "eval",
            }
            data = root / "probe.jsonl"
            data.write_text(json.dumps(row) + "\n")
            target = root / "00000000.safetensors"
            save_file({"hidden": torch.tensor([[2.0, 0.0], [0.0, 2.0]])}, target)
            manifest = {
                "entries": {
                    "0": {"row_sha256": row_key(row), "file_sha256": sha(target)}
                }
            }
            dump(root / "manifest.json", manifest)
            config = {
                "data": str(data),
                "data_sha256": sha(data),
                "cache": str(root),
                "sample_tokens": 2,
                "bottom_k": 1,
                "bottom_fraction": 0.1,
                "window": 2,
                "max_window_kl_range": 0.001,
                "min_persistence": 0.95,
                "max_ablation_delta_kl": 0.0,
            }
            model = Toy()
            head = torch.nn.Linear(2, 2, bias=False)
            head.weight.data.copy_(torch.eye(2))
            recorder = RouteRecorder(1, 3, "cpu")
            attach(model, recorder)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with (
                patch(
                    "sft.distillation.pruning_probe.validate_cache",
                    return_value=manifest,
                ),
                patch("sft.distillation.pruning_probe.dist.get_rank", return_value=0),
                patch(
                    "sft.distillation.pruning_probe.dist.get_world_size", return_value=1
                ),
                patch("sft.distillation.pruning_probe.dist.all_reduce"),
            ):
                probe = PruningProbe(
                    config, root, model, head, head.weight, 1.0, 2, root, recorder
                )
                rng = torch.get_rng_state().clone()
                probe.evaluate(0)
                probe.evaluate(1)
                self.assertTrue(model.training)
                self.assertIs(model.mlp.gate.route_recorder, recorder)
                self.assertIsNone(model.mlp.gate.probe_excluded_experts)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                report = json.loads((root / "probe-step1.json").read_text())
                self.assertEqual(report["token_topk_overlap"], [1.0])
                self.assertEqual(report["completion_tokens"], 2)
                self.assertTrue(report["advisory_ready"])
                restored = PruningProbe(
                    config, root, model, head, head.weight, 1.0, 2, root, recorder
                )
                restored.restore(probe.state)
                restored.evaluate(1)
                self.assertEqual(len(restored.state["history"]), 2)
                cadence_state = copy.deepcopy(restored.state)
                restored.evaluate(2, record_history=False)
                self.assertEqual(restored.state["step"], cadence_state["step"])
                self.assertEqual(restored.state["history"], cadence_state["history"])
                self.assertEqual(restored.state["routes"].keys(), cadence_state["routes"].keys())
                for key in restored.state["routes"]:
                    torch.testing.assert_close(
                        restored.state["routes"][key], cadence_state["routes"][key]
                    )
                endpoint = json.loads((root / "probe-step2.json").read_text())
                self.assertFalse(endpoint["scheduled"])
                self.assertFalse(endpoint["advisory_ready"])
                model.fail = True
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    restored.evaluate(2)
                self.assertTrue(model.training)
                self.assertIs(model.mlp.gate.route_recorder, recorder)
                self.assertEqual(restored.state["step"], 1)

    def test_wandb_probe_axis_and_no_undefined_scalars(self):
        layer = layer_metrics([0, 1, 2], None, [], 1)
        report = {
            "step": 5,
            "baseline_kl": 0.2,
            "token_topk_overlap": [None],
            "frequency_mass_candidate_overlap": [1.0],
            "criteria": {"frequency": {"layers": [layer], "ablation_delta_kl": 0.01}},
        }
        payload, rows = probe_payload(report)
        self.assertEqual(payload["probe/step"], 5)
        self.assertNotIn("train/step", payload)
        self.assertNotIn("probe/frequency/layer0/spearman", payload)
        self.assertEqual(rows, [])
        self.assertFalse(any("layer" in key for key in payload))
        self.assertNotIn("experts/pruning_candidate_stability", payload)
        layer["overlap_previous"] = 0.5
        payload, _ = probe_payload(report)
        self.assertEqual(payload["experts/pruning_candidate_stability"], 0.5)


if __name__ == "__main__":
    unittest.main()
