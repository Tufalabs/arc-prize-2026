"""CPU tests of the Flash SFT trainer's row, schedule, loss-target and config logic (training/flash_sft.py)."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from training.flash_sft import (
    RENDER_FILES, RowIndex, RowInputs, check_render_files, completion_spans, config_sha256, load_config, lr_factor, model_reader_factory, summarize,
    wandb_metrics, wandb_run_id,
    target_positions, update_schedule,
)

RECIPE = Path(__file__).resolve().parents[1] / "configs" / "flash_sft" / "reap307-kimi-baseline.json"


def write_rows(directory, rows, name="rows.jsonl"):
    path = Path(directory) / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def row(prompt, output, packed=(), game="g", **fields):
    return {"input_token_ids": list(range(prompt)), "output_token_ids": list(range(output)),
            "packed_completions": [list(span) for span in packed], "game_id": game, **fields}


class SpanTests(unittest.TestCase):
    def test_spans_and_target_positions(self):
        r = row(10, 3, packed=[(2, 2), (6, 1)])
        self.assertEqual(completion_spans(r), [(2, 2), (6, 1), (10, 3)])
        # The hidden state at p - 1 predicts token p.
        self.assertEqual(target_positions(completion_spans(r)), [1, 2, 5, 9, 10, 11])
        self.assertEqual(completion_spans(row(4, 2)), [(4, 2)])

    def test_bad_spans_are_refused(self):
        for packed in ([(0, 2)], [(3, 2), (2, 1)], [(2, 0)], [(8, 4)], [(2, 3), (4, 1)]):
            with self.assertRaises(ValueError):
                completion_spans(row(10, 3, packed=packed))


class RowIndexTests(unittest.TestCase):
    def test_lazy_rows_lengths_tokens_and_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [row(5, 2, game="a", source_set="x"), row(9, 3, packed=[(2, 2)], game="b")]
            path, digest = write_rows(tmp, rows)
            index = RowIndex(path, digest, max_length=100, fields=("source_set",))
            self.assertEqual(len(index), 2)
            self.assertEqual(index.lengths, [7, 12])
            self.assertEqual(index.tokens, [2, 5])
            self.assertEqual(index.fields, [{"game_id": "a", "source_set": "x"}, {"game_id": "b", "source_set": ""}])
            self.assertEqual(index.row(1), rows[1])
            with self.assertRaises(ValueError):
                RowIndex(path, "0" * 64, max_length=100)
            with self.assertRaises(ValueError):
                RowIndex(path, digest, max_length=10)


class RowInputsTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("ullm"), "needs ullm (the trainer environment)")
    def test_processor_is_built_with_the_rows_kwargs(self):
        with mock.patch("transformers.AutoProcessor.from_pretrained") as build, \
                mock.patch("ullm.media.processing_identity"):
            RowInputs("/model", {"max_pixels": 262144}, 7)
        build.assert_called_once_with("/model", local_files_only=True, trust_remote_code=True, max_pixels=262144)


class RenderFilesTests(unittest.TestCase):
    def test_a_model_with_another_tokenizer_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in RENDER_FILES:
                (Path(tmp) / name).write_text(name)
            expected = {name: hashlib.sha256(name.encode()).hexdigest() for name in RENDER_FILES}
            check_render_files(tmp, expected)
            (Path(tmp) / "tokenizer.json").write_text("another tokenizer")
            with self.assertRaisesRegex(ValueError, "tokenizer.json"):
                check_render_files(tmp, expected)


class ScheduleTests(unittest.TestCase):
    def test_every_pass_covers_every_row_once(self):
        lengths = [((i * 37) % 101) + 1 for i in range(23)]
        for policy in ("length_bucket", "random"):
            updates = update_schedule(lengths, 4, 3, 7, policy, 8)
            self.assertEqual(len(updates), 3 * 6)
            for p in range(3):
                rows = [i for update in updates[p * 6:(p + 1) * 6] for i in update if i is not None]
                self.assertEqual(sorted(rows), list(range(23)))
            self.assertTrue(all(len(update) == 4 for update in updates))
            self.assertEqual(updates, update_schedule(lengths, 4, 3, 7, policy, 8))  # deterministic
            self.assertNotEqual(updates[:6], updates[6:12])  # passes reshuffle

    def test_length_buckets_group_similar_lengths(self):
        lengths = list(range(1, 129))
        updates = update_schedule(lengths, 4, 1, 3, "length_bucket", 128)
        for update in updates:
            spread = max(lengths[i] for i in update) - min(lengths[i] for i in update)
            self.assertEqual(spread, 3)
        with self.assertRaises(ValueError):
            update_schedule(lengths, 4, 1, 3, "length_bucket", 6)

    def test_lr_warmup_then_flat(self):
        self.assertEqual([lr_factor(s, 5) for s in range(7)], [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0])
        self.assertEqual(lr_factor(0, 0), 1.0)

    def test_lr_decay_then_hold(self):
        decay = {"decay_from": 5, "decay_until": 197, "final_fraction": 0.1}
        self.assertEqual([lr_factor(s, 5, decay) for s in range(6)], [0.2, 0.4, 0.6, 0.8, 1.0, 1.0])
        self.assertAlmostEqual(lr_factor(101, 5, decay), 0.55)
        self.assertAlmostEqual(lr_factor(197, 5, decay), 0.1)
        self.assertAlmostEqual(lr_factor(198, 5, decay), 0.1)
        self.assertAlmostEqual(lr_factor(395, 5, decay), 0.1)
        factors = [lr_factor(s, 5, decay) for s in range(5, 198)]
        self.assertEqual(factors, sorted(factors, reverse=True))


class SummaryTests(unittest.TestCase):
    def test_overall_and_grouped(self):
        values = [[-10.0, 10.0, 4.0, 5.0], [-30.0, 20.0, 5.0, 11.0], [-6.0, 3.0, 3.0, 1.5]]
        result = summarize(values, {"game": ["a", "a", "b"], "source_set": ["x", "y", "x"]})
        self.assertAlmostEqual(result["nll"], 46 / 33)
        self.assertAlmostEqual(result["top1"], 12 / 33)
        self.assertAlmostEqual(result["entropy_mean"], 17.5 / 33)
        self.assertEqual(result["by_game"]["b"], {"nll": 2.0, "top1": 1.0, "entropy_mean": 0.5, "rows": 1, "tokens": 3})
        self.assertAlmostEqual(result["by_source_set"]["x"]["nll"], 16 / 13)


class WandbMetricsTests(unittest.TestCase):
    def test_old_trainer_keys(self):
        train = {"update": 3, "pass": 1, "loss_per_token": 0.5, "tokens": 100, "rows": 2, "grad_norm": 2.0,
                 "lr": 5e-5, "lr_factor": 0.5, "loss_per_example": 0.6, "peak_allocated_gib": 9.0,
                 "max_sequence": 70, "seconds": 4.5,
                 "per_rank": {"sequence_length": [50, 30, 0], "seconds": [4.0, 2.0, 3.0],
                              "peak_allocated_gib": [9.0, 8.0, 7.0], "peak_reserved_gib": [10.0, 9.0, 8.0]}}
        validation = {"nll": 1.2, "top1": 0.7, "entropy_mean": 0.9, "rows": 4, "tokens": 400, "update": 3,
                      "nll_change": -0.1, "seconds": 7.0,
                      "by_game": {"g": {"nll": 1.1, "top1": 0.8, "entropy_mean": 0.8, "rows": 2, "tokens": 200}},
                      "by_source_set": {"s": {"nll": 1.3, "top1": 0.6, "entropy_mean": 1.0, "rows": 2, "tokens": 200}}}
        flat = wandb_metrics(3, train=train, validation=validation)
        self.assertEqual(flat["train/step"], 3)
        self.assertEqual(flat["train/loss_per_completion_token"], 0.5)
        self.assertEqual(flat["train/objective_loss"], 0.5)
        self.assertEqual(flat["train/trainable_grad_norm"], 2.0)
        self.assertEqual(flat["train/lr_factor"], 0.5)
        self.assertEqual(flat["train/step_seconds"], 4.0)
        self.assertEqual(flat["train/completion_tokens_per_second"], 25.0)
        self.assertEqual(flat["train/input_tokens_per_second"], 20.0)
        self.assertEqual(flat["train/rank2/sequence_length"], 0)
        self.assertEqual(flat["train/rank1/peak_reserved_gib"], 9.0)
        self.assertEqual(flat["train/sft_validation/nll"], 1.2)
        self.assertEqual(flat["train/sft_validation/step"], 3)
        self.assertEqual(flat["train/sft_validation/nll_change"], -0.1)
        self.assertEqual(flat["train/sft_validation/game/g/top1"], 0.8)
        self.assertEqual(flat["train/sft_validation/source_set/s/nll"], 1.3)
        self.assertFalse([k for k in flat if not k.startswith("train/")])
        self.assertEqual(wandb_metrics(0, validation=validation)["train/step"], 0)


class WandbIdTests(unittest.TestCase):
    def test_explicit_id_is_kept_and_null_follows_the_run_directory(self):
        self.assertEqual(wandb_run_id({"id": "explicit"}, "/data/a"), "explicit")
        derived = wandb_run_id({"id": None}, "/data/a")
        self.assertEqual(derived, wandb_run_id({"id": None}, "/data/b/../a"))
        self.assertNotEqual(derived, wandb_run_id({"id": None}, "/data/b"))
        self.assertRegex(derived, "^[0-9a-f]{16}$")


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(RECIPE.read_text())

    def load(self, config):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(config))
            return load_config(path)

    def test_sharding_needs_routed_lora(self):
        broken = copy.deepcopy(self.config)
        broken["expert_lora"], broken["expert_lora_sharded"], broken["expert_lora_inplace_delta"] = None, True, False
        with self.assertRaises(ValueError):
            self.load(broken)
        broken["expert_lora_sharded"], broken["expert_lora_inplace_delta"] = False, True
        with self.assertRaises(ValueError):
            self.load(broken)
        broken["expert_lora_inplace_delta"] = False
        self.load(broken)

    def test_cosine_recipe_differs_only_in_length_decay_and_name(self):
        cosine = json.loads((RECIPE.parent / "reap307-kimi-cosine.json").read_text())
        self.assertEqual(self.load(cosine), cosine)
        self.assertEqual(cosine["optimizer"].pop("decay"), {"decay_from": 5, "decay_until": 197, "final_fraction": 0.1})
        self.assertEqual(cosine.pop("passes"), 2)
        self.assertEqual(cosine["wandb"].pop("name"), "reap307-kimi-sft-cosine-main")
        baseline = copy.deepcopy(self.config)
        del baseline["passes"], baseline["wandb"]["name"]
        self.assertEqual(cosine, baseline)

    def test_decay_is_optional_and_validated(self):
        self.assertNotIn("decay", self.load(self.config)["optimizer"])
        good = {"decay_from": 5, "decay_until": 197, "final_fraction": 0.1}
        for change in ({"decay_from": 4}, {"decay_until": 5}, {"final_fraction": 1.0}, {"final_fraction": 0.0},
                       {"final_fraction": 1}, {"decay_until": None}, {"extra": 1}):
            broken = copy.deepcopy(self.config)
            broken["optimizer"]["decay"] = {**good, **change}
            with self.assertRaises(ValueError, msg=str(change)):
                self.load(broken)

    def test_recipe_is_complete_and_valid(self):
        self.assertEqual(self.load(self.config), self.config)
        self.assertEqual(config_sha256(self.config), config_sha256(copy.deepcopy(self.config)))

    def test_every_key_is_required_and_checked(self):
        for key in self.config:
            broken = {k: v for k, v in self.config.items() if k != key}
            with self.assertRaises(ValueError, msg=key):
                self.load(broken)
        for path, value in (
            (("optimizer", "lr"), 0), (("optimizer", "betas"), [0.9]), (("lora", "group_ranks"), {"experts": [2, 4]}),
            (("expert_storage",), "nvfp4"), (("batching", "policy"), "sorted"), (("passes",), 0),
            (("validation", "group_by"), ["game_id"]), (("kernel_mode",), "turbo"),
            (("offload_chunk_mib",), 3), (("offload_chunk_mib",), 0), (("retain_qsa_attention",), None),
            (("cuda_expandable_segments",), "yes"), (("optimizer", "placement"), "cpu"),
            (("preflight_longest_rows",), -1), (("preflight_longest_rows",), True),
            (("expert_lora_sharded",), "yes"), (("expert_lora_sharded",), None),
            (("expert_lora_inplace_delta",), "yes"), (("kernel_overrides",), {"checkpoint_policy": "partial"}),
            (("wandb", "id"), ""), (("wandb", "id"), 7),
            (("rows_render_files", "tokenizer.json"), "pending"), (("rows_render_files",), {}),
        ):
            broken = copy.deepcopy(self.config)
            target = broken
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError, msg=str(path)):
                self.load(broken)


class HostOptimizerTests(unittest.TestCase):
    def test_host_adamw_matches_adamw_and_round_trips_its_state(self):
        import torch

        from training.flash_sft import HostAdamW

        hyper = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        torch.manual_seed(3)
        values = [torch.randn(5, 3), torch.randn(7)]
        reference = [v.clone().requires_grad_(True) for v in values]
        device = [v.clone().requires_grad_(True) for v in values]
        plain = torch.optim.AdamW(reference, foreach=False, **hyper)
        host = HostAdamW(device, **hyper)
        for step in range(4):
            grads = [torch.randn_like(v) for v in values]
            for p, g in zip(reference, grads):
                p.grad = g.clone()
            for p, g in zip(device, grads):
                p.grad = g.clone()
            for group in host.param_groups:
                group["lr"] = 1e-2 * (step + 1)
            for group in plain.param_groups:
                group["lr"] = 1e-2 * (step + 1)
            plain.step()
            host.step()
            host.zero_grad(set_to_none=True)
            self.assertTrue(all(p.grad is None for p in device))
            for a, b in zip(reference, device):
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        state = host.state_dict()
        self.assertEqual(set(state["state"]), {0, 1})
        self.assertEqual(set(state["state"][0]), {"step", "exp_avg", "exp_avg_sq"})
        # Resume: the parameters are restored first, then the optimizer re-reads them as masters.
        fresh = [v.detach().clone().requires_grad_(True) for v in device]
        resumed = HostAdamW([torch.zeros_like(v).requires_grad_(True) for v in values], **hyper)
        with torch.no_grad():
            for p, q in zip(resumed.params, fresh):
                p.copy_(q)
        resumed.load_state_dict(state)
        for master, p in zip(resumed.masters, fresh):
            self.assertTrue(torch.equal(master, p.detach()))
        torch.testing.assert_close(resumed.state_dict()["state"][1]["exp_avg"], state["state"][1]["exp_avg"])


class PreflightTests(unittest.TestCase):
    def test_longest_rows_run_forward_backward_and_leave_rng_and_gradients_alone(self):
        import random
        import types
        from unittest import mock

        import torch
        import torch.distributed as dist

        from training.flash_sft import FlashSFT

        with tempfile.TemporaryDirectory() as tmp:
            index = RowIndex(*write_rows(tmp, [row(3, 2), row(9, 4), row(5, 1)]), max_length=100)
        weight = torch.nn.Parameter(torch.ones(3))
        seen = []
        trainer = FlashSFT.__new__(FlashSFT)
        trainer.config = {"preflight_longest_rows": 2}
        trainer.rows, trainer.validation, trainer.world, trainer.device = index, None, 1, torch.device("cpu")
        trainer.rows.row = lambda i: {"index": i}
        trainer.trainables = {"w": weight}
        trainer.optimizer = torch.optim.SGD([weight], lr=1.0)

        def row_loss(r):
            seen.append(r["index"])
            torch.rand(1)  # the preflight must hide RNG use from training
            return (weight * 2).sum()

        trainer._row_loss = row_loss
        trainer._reduce_gradients = lambda: None
        cuda = types.SimpleNamespace(
            reset_peak_memory_stats=lambda d: None, synchronize=lambda d: None, max_memory_allocated=lambda d: 3 * 2**30,
            max_memory_reserved=lambda d: 4 * 2**30, get_rng_state=lambda d: torch.zeros(1),
            set_rng_state=lambda state, d: None,
            get_device_properties=lambda d: types.SimpleNamespace(total_memory=8 * 2**30))
        torch.manual_seed(5)
        random.seed(5)
        expected = (torch.rand(1), random.random())
        torch.manual_seed(5)
        random.seed(5)
        with mock.patch.object(torch, "cuda", cuda), mock.patch.object(dist, "all_gather",
                                                                       lambda out, t: out[0].copy_(t)):
            report = trainer._preflight()
        self.assertEqual(seen, [1, 2])  # the two longest rows (13 and 6 tokens)
        self.assertEqual([r["tokens"] for r in report["rows"]], [13, 6])
        self.assertEqual(report["peak_allocated_gib"], [3.0])
        self.assertIsNone(weight.grad)
        torch.testing.assert_close(weight.detach(), torch.ones(3))
        self.assertEqual((torch.rand(1), random.random()), expected)
        trainer.config = {"preflight_longest_rows": 0}
        self.assertIsNone(trainer._preflight())


class AllocatorTests(unittest.TestCase):
    def test_expandable_segments_setting(self):
        from training.flash_sft import configure_cuda_allocator

        env = {}
        configure_cuda_allocator(True, env)
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        env = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"}
        configure_cuda_allocator(False, env)
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512,expandable_segments:False")
        env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        configure_cuda_allocator(True, env)
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        with self.assertRaises(ValueError):
            configure_cuda_allocator(False, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})


class StagedModelTests(unittest.TestCase):
    def test_staged_shards_are_read_from_the_copy(self):
        import torch
        from safetensors.torch import save_file

        from sft.distillation.staging import stage
        from training.flash.reader import Reader

        with tempfile.TemporaryDirectory() as tmp:
            source, local, other = Path(tmp) / "model", Path(tmp) / "local", Path(tmp) / "other"
            source.mkdir()
            save_file({"w": torch.arange(4.0)}, str(source / "a.safetensors"))
            (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "a.safetensors"}}))
            self.assertIs(model_reader_factory(source, staged=""), Reader)
            with self.assertRaises(FileNotFoundError):  # nothing staged there yet
                model_reader_factory(source, staged=str(local))
            stage(source, local, workers=1)
            reader = model_reader_factory(source, staged=str(local))(source)
            self.assertEqual(reader.shard_root, local)
            self.assertTrue(torch.equal(reader.tensor("w"), torch.arange(4.0)))
            with self.assertRaisesRegex(ValueError, "not the staged model"):
                model_reader_factory(source, staged=str(local))(other)
            other.mkdir()
            (other / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "a.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "provenance"):
                model_reader_factory(other, staged=str(local))


if __name__ == "__main__":
    unittest.main()
